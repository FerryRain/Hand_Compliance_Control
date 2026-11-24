"""Select and execute wrist poses using the palmar surface and the online tactile GPIS."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.spatial.transform import Rotation

from ..acitive_02.surface_scan import align,limit
from ..i04_geometric_traversal.physics import quaternion
from . import reachable_reentry as previous,tactile_field
from .multirate_wrist import contact_frame
from .palm_acquisition import PalmGuide
from . import palm_acquisition,palm_surface
from .palm_surface import OUT,normalize,on_plane,palm_sites
from .sweep import DT,PREVIEW_S,LOAD_LIMIT
from .ycb_fixtures import OBJECTS


class PalmFollower(previous.ReturnFollower):
    def __init__(self,base,sites):
        super().__init__(base)
        self.sites=sites
        self.last_patch=None

    def select(self,**state):
        result=super().select(**state)
        row=result['diagnostics']
        start=perf_counter()
        self.face_surface(state,row,result)
        row['palm_pose_ms']=1000*(perf_counter()-start)
        return result

    def face_surface(self,state,row,result):
        wrist,att=state['wrist_position'],state['wrist_rotation']
        sites=wrist+att.apply(self.sites)
        center=sites[0]
        outward=att.apply(OUT)
        surface=state['surface']
        row.update(palm_reference_valid=False,palm_center_m=center.tolist(),palm_outward=outward.tolist())
        if surface.tree is None or state['observed_phase']=='withdraw' or max(state['loads'])>=LOAD_LIMIT:
            return
        guide=self.guide
        model=self.global_model
        ids=np.flatnonzero(state['hits'])
        pads=state['pads']
        if len(ids):
            ids,pivot,n0=contact_frame(pads,state['normals'],ids,self.base.local_normal)
            local=pivot.copy()
            self.last_patch=(pivot.copy(),n0.copy())
            # Look one small physical step along the GPIS-selected approach.
            if row['mode']=='explore':
                local+=.009*result['direction']
        elif 'entry_point' in row:
            local,n0=np.asarray(row['entry_point']),np.asarray(row['entry_normal'])
            pivot=local
        elif self.last_patch is not None:
            local,n0=self.last_patch
            near,idx=surface.tree.query(pads)
            merit=np.where(surface.normals[idx]@n0>.30,near,np.inf)
            j=int(np.argmin(merit))
            if merit[j]<.041:
                local,n0=surface.points[idx[j]],surface.normals[idx[j]]
            elif np.linalg.norm(pads-local,axis=1).min()>.063:
                return
            pivot=local
        else:
            return
        anchor=guide.patches.query(model,surface,local,n0,.039)
        if not anchor.valid[0]:
            return
        point=anchor.points[0]
        axis=normalize(.50*n0+.50*anchor.normals[0])
        projection=center-np.dot(center-point,axis)*axis
        under=guide.patches.query(model,surface,projection,axis,.103,anchor.components)
        surface_point=under.points[0] if under.valid[0] else point
        # On a corner, the finger can be feeling a different face from the one
        # beneath the palm. Use a connected sheet and the whole plate footprint.
        patch_n=normalize(.34*axis+.66*under.normals[0]) if under.valid[0] else axis
        line=normalize(center-surface_point)
        blend=.34*np.clip(patch_n@line,0.,1.)
        desired_n=normalize((1.-blend)*patch_n+blend*line)
        yaw_ref=att
        goal=guide.palm_goal
        has_goal=goal is not None and goal.region is guide.region
        if has_goal:
            yaw_ref=goal.rotation
        target=align(yaw_ref.apply(OUT),desired_n)*yaw_ref
        height=float((center-surface_point)@patch_n)
        # Use the current hand's reach along the selected attitude to choose
        # height. A clenched and an extended hand have different standoffs.
        material=att.inv().apply(pivot-wrist)
        reach=float((target.apply(self.sites[0]-material))@patch_n)
        reach=float(np.clip(reach,.018,.18))
        correction=float(np.clip(.55*(reach-height),-.019,.019))
        if state['loads'].max()>3.5 or state['loads'][-1]>.35:
            correction=max(0.,correction)
        side_error=on_plane(surface_point-center,patch_n)
        drift=limit(.32*side_error,.014) if under.valid[0] else np.zeros(3)
        velocity=drift+correction*patch_n
        row.update(palm_reference_valid=True,palm_patch_m=surface_point.tolist(),palm_patch_outward=patch_n.tolist(),
            palm_patch_source=('gpis_mesh' if (under.mesh[0] if under.valid[0] else anchor.mesh[0]) else 'measured_tactile_patch'),
            palm_target_xyzw=target.as_quat().tolist(),palm_pivot_m=pivot.tolist(),palm_height_m=height,
            palm_reach_height_m=reach,palm_trim_velocity=velocity.tolist(),palm_facing_deg=float(np.degrees(np.arccos(np.clip(outward@patch_n,-1,1)))),
            palm_view_deg=float(np.degrees(np.arccos(np.clip(outward@line,-1,1)))),palm_goal_held=bool(has_goal))
        # Record the pose that the servo will approach, including pivot compensation.
        step=limit((target*att.inv()).as_rotvec(),.048)
        delta=Rotation.from_rotvec(step)
        result['rotation']=delta*att
        result['position']=result['position']+delta.apply(wrist-pivot)-(wrist-pivot)+.05*velocity


class PalmWrist(previous.ReturnWrist):
    def __init__(self,*args,**opts):
        super().__init__(*args,**opts)
        sites=palm_sites(self.hand)
        base=self.selector.base
        base.global_model=self.selector.global_model.base
        base.guide=PalmGuide(base.global_model,base.guide.history,opts['seed'],sites)
        self.selector=PalmFollower(base,sites)

    def command(self,touch):
        start,att=self.position.copy(),self.attitude
        cmd,preview=super().command(touch)
        row=self.transition
        if (row is None or not row.get('palm_reference_valid') or self.unloading or
                max(touch.loads)>=LOAD_LIMIT or touch.loads[-1]>.65):
            return cmd,preview
        # The underlying servo may follow a measured return corridor during a
        # transfer. Preserve it until an actual object patch is near the pads.
        ids=np.flatnonzero(touch.measured_heads)
        pivot=np.asarray(row['palm_pivot_m'])
        if len(ids):
            _,pivot,_=contact_frame(touch.points,touch.normals,ids,np.asarray(row['palm_patch_outward']))
        elif np.linalg.norm(pivot-self.measured_position)>.23:
            return cmd,preview
        target=Rotation.from_quat(row['palm_target_xyzw'])
        error=(target*att.inv()).as_rotvec()
        outward=target.apply(OUT)
        yaw=outward*(error@outward)
        omega=2.65*(error-yaw)+1.15*yaw
        turn=limit(DT*omega,.0125 if len(ids) else .0085)
        end=Rotation.from_rotvec(turn)*att
        change=end*self.attitude.inv()
        # The ordinary servo has already computed a translation and load
        # relief. Rotate that command about the measured support, and preserve
        # its normal force feedback while the plate changes attitude.
        offset=self.position-pivot
        step=self.position-start+change.apply(offset)-offset
        trim=np.asarray(row['palm_trim_velocity'])
        if max(touch.loads)>3.5:
            n=np.asarray(row['palm_patch_outward'])
            trim-=min(0.,trim@n)*n
        step=limit(step+DT*trim,.0019 if len(ids) else .0012)
        if self.planes is not None:
            v,p,n=self.planes
            after=end.apply(v)+start+step-p
            gap=np.sum(after*n,axis=1)
            before=np.sum((att.apply(v)+start-p)*n,axis=1)
            closing=(np.linalg.norm(after-gap[:,None]*n,axis=1)<.024)&(gap<.006)&(gap<before)
            if closing.any():
                j=np.argmax(np.where(closing,before-gap,-np.inf))
                step=limit(step+min(.0017,before[j]-gap[j]+.00018)*n[j],.002)
        self.position=self.workspace_center+limit(start+step-self.workspace_center,.45)
        self.attitude=end
        cmd=np.r_[self.position,quaternion(end)]
        horizon=PREVIEW_S/DT
        future=Rotation.from_rotvec(horizon*turn)*end
        return cmd,np.r_[self.position+horizon*(self.position-start),quaternion(future)]


def run(folder,obj,seconds=240,device='cuda',prior=None):
    return tactile_field.run(folder,obj,seconds,device,prior,wrist_cls=PalmWrist,worker_cls=previous.ReturnWorker,
        source_files=(__file__,palm_acquisition.__file__,palm_surface.__file__,previous.__file__,previous.ct.__file__,
                      previous.ct.entry_frontier.__file__,previous.ct.transition_compute.__file__,
                      previous.ct.delivery.__file__,previous.ct.tactile_reentry.__file__,previous.ct.tactile_stall.__file__),
        release='absolute-palmar-surface-pose')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--object',choices=OBJECTS,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--seconds',type=float,default=240)
    parser.add_argument('--prior',type=Path)
    args=parser.parse_args()
    print(run(args.out,args.object,args.seconds,prior=args.prior),flush=True)
