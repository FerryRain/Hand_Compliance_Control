"""Information-guided tactile atlas with a continuously levelled wrist and measured footprint."""
from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from ..acitive_02.aude import unit, tangent
from ..acitive_02.surface_scan import PadSupport, align, limit
from ..i04_geometric_traversal.physics import quaternion, rotation
from . import contact_velocity, face_wrist, informative_run, multirate_wrist, online_tactile_mesh, posterior_route, strip_information, tactile_atlas
from .contact_first import signed_gap
from .experiment import digest
from .informative_wrist import InformativeWrist
from .multirate_wrist import LOCAL_S, MultirateWorker, contact_frame
from .online_tactile_mesh import ContactBand, StreamingGPIS
from .sweep import DT, LOAD_LIMIT
from .ycb_fixtures import OBJECTS


class AtlasActions:
    def __init__(self,touch,orient,device):
        hits=touch.measured_heads
        self.ids=np.flatnonzero(hits)
        self.up=unit(touch.normals[hits].mean(0))
        self.row=unit(tangent(touch.points[self.ids[-1]]-touch.points[self.ids[0]],self.up))
        self.local_n,self.local_row=orient.inv().apply([self.up,self.row])
        self.global_model=StreamingGPIS(device,ContactBand(.95))
        self.guide=tactile_atlas.AtlasRoute(self.global_model,touch.points,self.up,orient)
        self.tilt=self.tip_forces=np.zeros(3)
        self.offset=0.
        self.last_n=self.up
        self.empty_time=0.

    def select(self,*,surface,hits,pads,normals,loads,wrist_rotation,wrist_position,tick,observed_phase,**_):
        real=self.last_n
        if hits.any():
            _,_,real=contact_frame(pads,normals,np.flatnonzero(hits),real)
            self.last_n=real
        else:
            self.empty_time+=LOCAL_S
        if hits.any():
            self.empty_time=0.
        began=perf_counter()
        event=self.guide.update(surface,tick,pads,hits,real,wrist_rotation,
                         hits.any() and loads.max()<LOAD_LIMIT and observed_phase!='withdraw')
        elapsed=1000*(perf_counter()-began) if event is not None else 0.
        info=dict(global_event=event,global_ms=elapsed,batch=self.guide.batch,local_period_s=LOCAL_S,
                  gpis_device=str(self.global_model.device),contact_iso=self.global_model.band.level,mode='recontact')
        result=dict(position=wrist_position,rotation=wrist_rotation,normal=real,direction=np.zeros(3),diagnostics=info)
        if self.guide.ray is None:
            return result
        quorum=hits.sum()
        angle=(.48 if quorum>=2 else .37 if quorum else .24)*LOCAL_S
        if observed_phase=='withdraw':
            angle=.07*LOCAL_S
        anchor,normal=self.guide.advance(angle)
        patch=surface.patch(anchor,preferred_normal=normal)
        if patch is not None:
            gap,gg=signed_gap(patch,anchor)
            weight=np.clip(1.-np.linalg.norm(anchor-patch.anchor)/.043,0.,1.)
            anchor=anchor-weight*np.clip(gap,-.009,.009)*gg
            normal=unit((1.-weight)*normal+weight*gg)
        # Keep an actual supporting surface authoritative over speculative radial depth.
        if hits.any():
            same=np.flatnonzero(hits & (normals@real>.72))
            if len(same):
                pivot=pads[same].mean(0)
                desired=pads[self.ids].mean(0)
                q=surface.patch(pivot,preferred_normal=real)
                if q is not None:
                    b=q.predict(pivot)[0][0]
                    # Shift the centroid's desired point into the real pad's frame.
                    gap,g=signed_gap(q,pivot+(anchor-desired),b)
                    agreement=np.clip((normal@real+.05)/.90,0.,1.)
                    weight=agreement*(.90 if len(same)>=2 else .65)
                    if np.linalg.norm(tangent(anchor-desired,real))<.12:
                        anchor-=weight*np.clip(gap,-.035,.035)*g
                        normal=unit((1.-.65*weight)*normal+.65*weight*real)
        local=wrist_rotation.inv().apply(pads-wrist_position)
        centered=local-local.mean(0)
        slope=np.cross(centered,self.local_n)
        tilt=np.linalg.solve(slope.T@slope+.025**2*np.eye(3),-slope.T@(centered@self.local_n))
        hand_n=Rotation.from_rotvec(-limit(tilt,.65)).apply(self.local_n)
        base=wrist_rotation*Rotation.from_rotvec(-self.tilt)
        adjust=align(base.apply(hand_n),normal).as_rotvec()
        desired=Rotation.from_rotvec(limit(adjust,1.35*.20))*base
        across=self.up if abs(normal@self.up)<.72 else self.row
        across=unit(tangent(across,normal))
        row=unit(tangent(desired.apply(self.local_row),normal))
        if across@row<0:
            across=-across
        yaw=np.arctan2(np.cross(row,across)@normal,row@across)
        desired=Rotation.from_rotvec(np.clip(yaw,-.12,.12)*normal)*desired
        pivot=local[self.ids].mean(0)
        direction=unit(tangent(self.guide.path_dir,normal))
        across=unit(np.cross(normal,direction))
        offsets=np.array([0.,-.012,.012,-.024,.024])
        sites=anchor+desired.apply(local-pivot)+offsets[:,None,None]*across
        scores=np.minimum(surface.tree.query(sites)[0],.018).mean(1)-.12*abs(offsets)-.09*abs(offsets-self.offset)
        self.offset+=np.clip(offsets[scores.argmax()]-self.offset,-.002,.002)
        anchor=anchor+self.offset*across
        info.update(mode='explore',goal_point=anchor.tolist(),goal_normal=normal.tolist(),desired_xyzw=desired.as_quat().tolist(),
                    local_heads=local.tolist(),pivot=pivot.tolist(),tangent_speed_m_s=float(angle/LOCAL_S*np.linalg.norm(anchor-self.guide.estimate.center)))
        result.update(position=anchor-desired.apply(pivot),rotation=desired,normal=normal,direction=direction)
        return result


class AtlasWrist(InformativeWrist):
    def __init__(self,hand,data,touch,*,device,guard,**options):
        super().__init__(hand,data,touch,**options)
        self.selector=AtlasActions(touch,self.attitude,device)
        self.vertices=PadSupport(hand).vertices
        self.position_ref=self.position.copy()
        self.contact_tilt=np.zeros(3)
        self.face_ready=True
        self.braking=False
        self.flow=None
        self.empty=0
        # The footprint executor has an optional radial workspace clamp. Its
        # radial bounds are disabled; enforce the robot's wrist workspace below.
        self.r_min,self.r_max=0.,np.inf
        self.pad_guard=contact_velocity.PadVelocity() if guard else None
        self.last_good_normal=self.selector.up.copy()
        self.last_shock=-100

    def observe(self,data,touch,tick):
        self.tick=tick
        if self.pad_guard is not None:
            self.pad_guard.observe(self.hand,data,touch)
        hit=touch.measured_heads
        peak,palm=touch.loads.max(),touch.loads[-1]
        moving=data.qpos[:3]-self.measured_position
        self.recent_motion=.85*self.recent_motion+.15*moving
        self.measured_position=data.qpos[:3].copy()
        self.empty=0 if hit.any() else self.empty+1
        if peak<LOAD_LIMIT:
            self.surface.add(touch,tick)
            self.last_normals[hit]=touch.normals[hit]
            if hit.any():
                self.last_good_normal=unit(touch.normals[hit].mean(0))
        orient=rotation(data.qpos[3:7])
        hand_n=orient.apply(self.selector.local_n)
        for j,g in enumerate(self.hand.pads):
            vertices=self.vertices[int(g)]
            h=vertices@(data.geom_xmat[g].reshape(3,3).T@hand_n)
            self.offsets[j]=vertices[h<=h.min()+.0004].mean(0)
        if peak>=LOAD_LIMIT or palm>1.8 or (palm>.70 and palm-self.last_palm_load>.65):
            region=int(np.argmax(touch.loads))
            pt=data.geom_xpos[self.hand.pads[min(3,region)]] if region<4 else self.measured_position
            radial=unit(pt-self.selector.guide.estimate.center)
            n=touch.normals[region] if region<4 and hit[region] else radial
            back=-unit(self.recent_motion) if np.linalg.norm(self.recent_motion)>.0001 else radial
            if not self.braking:
                self.escape_count+=1
                self.relief=unit(.5*n+.35*radial+.15*back)
            self.braking=True
            self.clear_ticks=0
            self.last_shock=tick
        elif self.braking:
            self.clear_ticks=self.clear_ticks+1 if peak<3. and palm<.35 else 0
            if self.clear_ticks>=7:
                self.braking=False
        self.last_palm_load=palm

    def phase(self):
        if self.braking:
            return 'withdraw'
        return 'approach' if self.empty>=18 else 'scan'

    def command(self,touch):
        start,orient=self.position_ref.copy(),self.attitude
        if self.braking or self.flow is None:
            v=(min(.13,.020+.006*touch.loads.max())*self.relief if self.braking else -.018*self.last_good_normal)
            step=DT*v
            cmd=np.r_[start+step,quaternion(orient)]
            ahead=cmd.copy()
        else:
            row=self.flow[0]
            self.goal_point,self.goal_normal=np.array(row['goal_point']),np.array(row['goal_normal'])
            self.goal_attitude=Rotation.from_quat(row['desired_xyzw'])
            self.local_heads,self.pivot=np.array(row['local_heads']),np.array(row['pivot'])
            cmd,ahead=face_wrist.FaceWrist.command(self,touch)
            if self.pad_guard is not None and touch.measured_heads.any() and touch.loads[-1]<.7:
                step,turn=self.pad_guard.filter(cmd[:3]-start,(self.attitude*orient.inv()).as_rotvec(),touch)
                self.attitude=Rotation.from_rotvec(turn)*orient
                cmd=np.r_[start+step,quaternion(self.attitude)]
                ahead=np.r_[cmd[:3]+12*step,quaternion(Rotation.from_rotvec(12*turn)*self.attitude)]
        if self.planes is not None and not self.braking:
            verts,anchors,normals=self.planes
            before=orient.apply(verts)+start-anchors
            delta=self.attitude.apply(verts)+cmd[:3]-anchors
            old=np.einsum('ij,ij->i',before,normals)
            gap=np.einsum('ij,ij->i',delta,normals)
            closing=(np.linalg.norm(delta-gap[:,None]*normals,axis=1)<.023)&(gap<.006)&(gap<old)
            if closing.any():
                i=int(np.argmin(np.where(closing,gap,np.inf)))
                shift=min(.0017,old[i]-gap[i]+.00016)*normals[i]
                cmd[:3]=start+limit(cmd[:3]+shift-start,.002)
                ahead[:3]+=shift
        self.position=self.workspace_center+limit(cmd[:3]-self.workspace_center,.45)
        self.position_ref=self.position.copy()
        cmd[:3]=self.position
        if self.pad_guard is not None:
            self.pad_guard.shift,self.pad_guard.turn=self.position-start,(self.attitude*orient.inv()).as_rotvec()
        return cmd,ahead


class AtlasWorker(MultirateWorker):
    def submit(self,data,touch,tick,lead):
        self.servo.selector.tilt=self.servo.contact_tilt.copy()
        return super().submit(data,touch,tick,lead)


def run(folder,obj,seconds,device,guard):
    modules=(face_wrist,contact_velocity,tactile_atlas,multirate_wrist,online_tactile_mesh,posterior_route,strip_information)
    sources=informative_run.source_hashes()
    for path in [Path(__file__),Path(__file__).with_name('contact_first.py'),*(Path(m.__file__) for m in modules)]:
        sources[path.name]=digest(path)
    with patch.multiple(informative_run,InformativeWrist=partial(AtlasWrist,device=device,guard=guard),
                        InformativeWorker=AtlasWorker,RELEASE=f'atlas-padguard{int(guard)}-{device}-global400-local5',
                        source_hashes=lambda:sources):
        return informative_run.run(folder,obj,seconds)


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    p.add_argument('--object',choices=OBJECTS,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=240.)
    p.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    p.add_argument('--guard',action='store_true')
    a=p.parse_args()
    print(run(a.out,a.object,a.seconds,a.device,a.guard),flush=True)
