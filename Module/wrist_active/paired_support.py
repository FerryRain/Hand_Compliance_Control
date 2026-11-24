"""Keep a measured pair on its axis while bringing a reachable third pad to the active frontier."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.spatial.transform import Rotation

from ..acitive_02.aude import unit
from ..i04_geometric_traversal.fingertip_hand import HEAD_FRACTION
from ..i04_geometric_traversal.physics import rotation
from . import contact_transition as transition, tactile_field
from .contact_first import palm_clear,signed_gap
from .history import normal_spread
from .multirate_wrist import LOCAL_S
from .ycb_fixtures import OBJECTS


def pad_caps(hand):
    caps=[]
    directions=np.array([[x,y,z] for x in (-1,0,1) for y in (-1,0,1) for z in (-1,0,1) if x or y or z])
    for f,g in enumerate(hand.pads):
        model=hand.model
        mesh=model.geom_dataid[g]
        start=model.mesh_vertadr[mesh]
        vertices=model.mesh_vert[start:start+model.mesh_vertnum[mesh]].astype(float)
        body=rotation(model.geom_quat[g]).apply(vertices)+model.geom_pos[g]
        height=(body-hand.head_origins[f])@hand.head_axes[f]/hand.head_spans[f]
        tip=vertices[height>=HEAD_FRACTION]
        caps.append(tip[np.unique(np.argmax(tip@directions.T,axis=0))])
    return caps


def fit_roll(state,result,cap,guide,forces):
    ids=np.flatnonzero(state['hits'])
    pads=state['pads']
    axis=pads[ids[1]]-pads[ids[0]]
    if np.linalg.norm(axis)<.018 or state['normals'][ids[0]]@state['normals'][ids[1]]<.55:
        return
    axis=unit(axis)
    pivot=pads[ids].mean(0)
    row=result['diagnostics']
    surface=state['surface']
    horizon=.20
    base_omega=np.array(row['angular_velocity'])
    forward=horizon*(row['tangent_speed_m_s']*result['direction']+row['height_speed_m_s']*result['normal'])
    orient=state['wrist_rotation']
    wrist=state['wrist_position']
    best=None
    for f in np.flatnonzero(~state['hits']):
        if state['loads'][f]>1.1:
            continue
        verts=cap[f]
        dist,idx=surface.tree.query(verts)
        normals=surface.normals[idx]
        gap=np.einsum('ij,ij->i',verts-surface.points[idx],normals)
        visible=(dist<.022)&(gap>-.001)&(gap<.011)&(normals@result['normal']>.62)
        if not visible.any():
            continue
        j=int(np.argmin(np.where(visible,dist+.32*np.maximum(gap,0),np.inf)))
        pt=verts[j]
        patch=surface.patch(pt,preferred_normal=normals[j])
        if patch is None:
            continue
        rotated=Rotation.from_rotvec(horizon*base_omega).apply(pt-pivot)
        trial=pivot+forward+rotated
        error,n=signed_gap(patch,trial)
        leverage=np.cross(rotated,n)@axis
        if not .0014<error<.012 or abs(leverage)<.012:
            continue
        angle=np.clip(-error*leverage/(leverage**2+.016**2),-.058,.058)
        final=pivot+forward+Rotation.from_rotvec(angle*axis).apply(rotated)
        remaining,_=signed_gap(patch,final)
        if remaining<-.0012 or remaining>error-.00075:
            continue
        turn=Rotation.from_rotvec(angle*axis)*Rotation.from_rotvec(horizon*base_omega)
        future=pivot+forward+turn.apply(wrist-pivot)
        if np.linalg.norm(future-state['workspace_center'])>.447:
            continue
        palm=state['palm_vertices']
        if not palm_clear(surface,palm,(pivot+forward+turn.apply(palm-pivot))[None,None])[0]:
            continue
        action=np.r_[orient.inv().apply(future-wrist),orient.inv().apply(turn.as_rotvec())][None]*(.5/horizon)
        _,light,weights=guide.history.predict(state['q'],state['hits'],normal_spread(state['normals'],state['hits']),action)
        credible=float(weights[0]@(light[0,:,0,ids[0]]*light[0,:,0,ids[1]]))
        benefit=(error-remaining)*(.35+.65*credible)
        if best is None or benefit>best[0]:
            best=benefit,f,angle,credible
    if best is None:
        return
    _,f,angle,credible=best
    old=base_omega.tolist()
    omega=base_omega+(angle/horizon)*axis
    h=Rotation.from_rotvec(LOCAL_S*(angle/horizon)*axis)
    result['position']=pivot+h.apply(result['position']-pivot)
    result['rotation']=h*result['rotation']
    row.update(angular_velocity=omega.tolist(),pair_anchor_ids=ids.tolist(),pair_recipient=int(f),
               pair_rotation_rad=float(angle),pair_prior=float(credible),pair_force=forces[ids].tolist(),
               pair_nominal=dict(tangent_speed_m_s=row['tangent_speed_m_s'],height_speed_m_s=row['height_speed_m_s'],
                                 angular_velocity=old,selected_direction=result['direction'].tolist()))


class PairFollower(transition.TransitionFollower):
    def select(self,**state):
        result=super().select(**state)
        row=result['diagnostics']
        ids=np.flatnonzero(state['hits'])
        if (self.settings==('both',True) and len(ids)==2 and min(self.dwell[ids])>=8 and
                self.tip_forces[ids].max()<3.5 and state['loads'].max()<4. and state['loads'][-1]<.35 and
                state['observed_phase']=='scan' and row['mode']=='explore'):
            began=perf_counter()
            fit_roll(state,result,self.caps,self.guide,self.tip_forces)
            row['pair_evaluation_ms']=1000*(perf_counter()-began)
        return result


class PairWrist(transition.TransitionWrist):
    def __init__(self,*args,**opts):
        super().__init__(*args,**opts)
        base=self.selector.base
        base.global_model=base.guide.model=self.selector.global_model.base
        self.selector=PairFollower(base)
        self.cap_material=pad_caps(self.hand)

    def command(self,touch):
        row=self.transition
        if row and 'pair_nominal' in row and self.flow is not None:
            same=np.array_equal(np.flatnonzero(touch.measured_heads),row['pair_anchor_ids'])
            if not same or touch.loads.max()>=4. or touch.loads[-1]>=.35:
                old,self.flow=self.flow,(row['pair_nominal'],self.flow[1])
                try:
                    return super().command(touch)
                finally:
                    self.flow=old
        return super().command(touch)


class PairWorker(transition.TransitionWorker):
    def submit(self,data,touch,tick,lead):
        s=self.servo
        if touch.measured_heads.sum()==2:
            s.selector.caps=[o+vs@r.T for o,vs,r in zip(s.origins,s.cap_material,s.matrices)]
        return super().submit(data,touch,tick,lead)


def run(folder,obj,seconds=240,device='cuda',prior=None):
    return tactile_field.run(folder,obj,seconds,device,prior,wrist_cls=PairWrist,worker_cls=PairWorker,
             source_files=(__file__,transition.__file__,transition.entry_frontier.__file__,transition.transition_compute.__file__,
                           transition.delivery.__file__,transition.tactile_reentry.__file__,transition.tactile_stall.__file__),
             release='measured-pair-axis-support')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--object',choices=OBJECTS,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=240)
    p.add_argument('--prior',type=Path)
    args=p.parse_args()
    print(run(args.out,args.object,args.seconds,prior=args.prior),flush=True)
