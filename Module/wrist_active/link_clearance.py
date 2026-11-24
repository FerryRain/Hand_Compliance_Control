"""Preserve room for the finger links as a wrist-only stroke follows the tactile surface."""
import argparse
from functools import partial
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from ..acitive_02.aude import unit
from ..acitive_02.surface_scan import limit
from ..i04_geometric_traversal.physics import quaternion,rotation
from . import anchored_strokes, atlas_wrist, feasible_atlas, informative_run
from .experiment import digest
from .sweep import DT
from .ycb_fixtures import OBJECTS


class RobotLinkHull:
    def __init__(self,hand):
        axes=np.array([[x,y,z] for x in (-1,0,1) for y in (-1,0,1) for z in (-1,0,1) if x or y or z])
        self.ids,self.regions,self.vertices=[],[],[]
        model=hand.model
        for g in range(model.ngeom):
            if not model.geom_contype[g] or g in hand.pad_index or g==hand.palm_geom:
                continue
            mesh=model.geom_dataid[g]
            if mesh<0:
                continue
            start=model.mesh_vertadr[mesh]
            vertices=model.mesh_vert[start:start+model.mesh_vertnum[mesh]]
            v=vertices[np.unique(np.argmax(vertices@axes.T,axis=0))]
            self.ids.extend([g]*len(v))
            self.regions.extend([hand.load_region[g]]*len(v))
            self.vertices.extend(v)
        self.ids=np.array(self.ids)
        self.regions=np.array(self.regions)
        self.vertices=np.asarray(self.vertices)

    def position(self,data):
        return data.geom_xpos[self.ids]+np.einsum('vij,vj->vi',data.geom_xmat[self.ids].reshape(-1,3,3),self.vertices)

    def planes(self,surface,world):
        if surface.tree is None:
            return None
        d,idx=surface.tree.query(world,k=min(3,len(surface.points)))
        if d.ndim!=2:
            return None
        n=surface.normals[idx[:,0]]
        anchors=surface.points[idx[:,0]]
        delta=world-anchors
        gap=np.einsum('ij,ij->i',delta,n)
        lateral=np.linalg.norm(delta-gap[:,None]*n,axis=1)
        agree=np.min(np.einsum('vki,vi->vk',surface.normals[idx],n),axis=1)
        valid=(d[:,0]<.021)&(lateral<.014)&(gap<.011)&(gap>-.010)&(agree>.76)
        chosen=[]
        for region in range(4):
            ids=np.flatnonzero(valid&(self.regions==region))
            if len(ids):
                order=ids[np.argsort(gap[ids]+.16*lateral[ids])]
                chosen.append(order[0])
                other=np.flatnonzero(n[order]@n[order[0]]<.65)
                if len(other):
                    chosen.append(order[other[0]])
        if not chosen:
            return None
        ids=np.asarray(chosen)
        return dict(indices=ids.tolist(),normal=n[ids].tolist(),points=anchors[ids].tolist())


class LinkActions(anchored_strokes.StrokeActions):
    def __init__(self,touch,orient,device,hull):
        super().__init__(touch,orient,device)
        self.hull=hull

    def select(self,**p):
        result=super().select(**p)
        result['diagnostics']['link_planes']=self.hull.planes(p['surface'],self.link_pose)
        return result


class ClearanceWrist(anchored_strokes.StrokeWrist):
    def __init__(self,hand,data,touch,*,device,regulate,**options):
        super().__init__(hand,data,touch,device=device,**options)
        self.hull=RobotLinkHull(hand)
        self.selector=LinkActions(touch,self.attitude,device,self.hull)
        self.witnesses=None
        self.joint_shift=np.zeros((len(self.hull.ids),3))
        self.previous_links=None
        self.regulate=regulate
        self.avoided=self.loaded_avoided=0

    def observe(self,data,touch,tick):
        super().observe(data,touch,tick)
        points=self.hull.position(data)
        rot=rotation(data.qpos[3:7])
        if self.previous_links is not None:
            old,root,r=self.previous_links
            shifted=self.measured_position+(rot*r.inv()).apply(old-root)
            self.joint_shift=.55*self.joint_shift+.45*np.clip(points-shifted,-.001,.001)
        self.previous_links=(points,self.measured_position.copy(),rot)
        self.material_points=points

    def command(self,touch):
        origin,orient=self.position.copy(),self.attitude
        cmd,ahead=super().command(touch)
        if self.witnesses is None:
            return cmd,ahead
        ids,anchors,n=self.witnesses
        points=self.material_points[ids]
        gap=np.einsum('ij,ij->i',points-anchors,n)
        lateral=np.linalg.norm(points-anchors-gap[:,None]*n,axis=1)
        delta=cmd[:3]-origin
        turn=(self.attitude*orient.inv()).as_rotvec()
        hit=np.flatnonzero(touch.measured_heads)
        pivot=touch.points[hit].mean(0) if len(hit) else self.measured_position
        lever=pivot-self.measured_position
        v=delta+(Rotation.from_rotvec(turn).apply(lever)-lever)
        j=np.column_stack((n,np.cross(points-pivot,n)/.10))
        twist=np.r_[v,.10*turn]
        pred=j@twist+np.einsum('ij,ij->i',self.joint_shift[ids],n)
        regions=self.hull.regions[ids]
        extra=np.maximum(0.,touch.loads[regions]-touch.forces[regions])
        lower=np.clip(.20*(.0015-gap),-.0012,.00030)
        lower=np.maximum(lower,np.clip(.00007*(extra-1.2),-.0012,.00055))
        use=(lateral<.016)&(gap<.006)&(gap>-.010)&(pred<lower-.000020)
        if not use.any():
            return cmd,ahead
        self.avoided+=1
        self.loaded_avoided+=int(np.max(extra[use])>1.2)
        rows=list(j[use])
        correction=list(np.clip(lower[use]-pred[use],0.,.00095))
        # Fit the retreat of the loaded link about the live pads. Mild link
        # clearance gets a lower priority than preserving a light footprint.
        if len(hit) and touch.loads.max()<8.:
            pn=touch.normals[hit]
            pj=np.column_stack((pn,np.cross(touch.points[hit]-pivot,pn)/.10))
            weight=np.where(touch.forces[hit]<3.,1.15,.60)
            rows.extend(weight[:,None]*pj)
            correction.extend(np.zeros(len(hit)))
        j=np.asarray(rows)
        adjustment=np.linalg.solve(j.T@j+np.diag([.22,.22,.22,.35,.35,.35])**2,
                                   j.T@np.asarray(correction))
        twist+=adjustment
        turn=limit(twist[3:]/.10,.0125)
        shift=limit(twist[:3]-(Rotation.from_rotvec(turn).apply(lever)-lever),.002)
        new=self.workspace_center+limit(origin+shift-self.workspace_center,.45)
        actual_r=Rotation.from_rotvec(turn)*orient
        # Recheck the palm planes for the projected twist.
        if self.planes is not None:
            verts,loc,ns=self.planes
            before=orient.apply(verts)+origin-loc
            after=actual_r.apply(verts)+new-loc
            old=np.sum(before*ns,axis=1)
            h=np.sum(after*ns,axis=1)
            risk=(np.linalg.norm(after-h[:,None]*ns,axis=1)<.023)&(h<.006)&(h<old)
            if risk.any():
                k=int(np.argmin(np.where(risk,h,np.inf)))
                new=origin+limit(new-origin+min(.0017,old[k]-h[k]+.00016)*ns[k],.002)
                new=self.workspace_center+limit(new-self.workspace_center,.45)
        self.position_ref=self.position=new
        self.attitude=actual_r
        self.pad_guard.shift,self.pad_guard.turn=new-origin,turn
        cmd=np.r_[new,quaternion(actual_r)]
        if self.braking or self.retreat:
            return cmd,cmd.copy()
        future=Rotation.from_rotvec(12.*turn)*actual_r
        return cmd,np.r_[new+12.*(new-origin),quaternion(future)]


class LinkWorker(feasible_atlas.FeasibleWorker):
    def submit(self,data,touch,tick,lead):
        self.servo.selector.link_pose=self.servo.material_points.copy()
        return super().submit(data,touch,tick,lead)

    def install(self,tick,*,initial=False):
        row=super().install(tick,initial=initial)
        if row is not None:
            p=row['link_planes']
            self.servo.witnesses=(np.array(p['indices']),np.array(p['points']),np.array(p['normal'])) if p else None
            row['corrected_link_steps']=self.servo.avoided
            row['corrected_loaded_steps']=self.servo.loaded_avoided
        return row


def run(folder,obj,seconds,device):
    modules=(anchored_strokes,feasible_atlas,atlas_wrist,atlas_wrist.face_wrist,atlas_wrist.contact_velocity,
             atlas_wrist.tactile_atlas,atlas_wrist.multirate_wrist,atlas_wrist.online_tactile_mesh,
             atlas_wrist.posterior_route,atlas_wrist.strip_information)
    sources=informative_run.source_hashes()
    for path in [Path(__file__),Path(__file__).with_name('contact_first.py'),*(Path(m.__file__) for m in modules)]:
        sources[path.name]=digest(path)
    with patch.multiple(informative_run,InformativeWrist=partial(ClearanceWrist,device=device,regulate=True),
                        InformativeWorker=LinkWorker,RELEASE=f'physical-link-clearance-{device}-global400-local5',
                        source_hashes=lambda:sources):
        return informative_run.run(folder,obj,seconds)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--object',choices=OBJECTS,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=240.)
    p.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    a=p.parse_args()
    print(run(a.out,a.object,a.seconds,a.device),flush=True)
