"""Limit outward pad velocity and load transients while following an information-selected wrist route."""
import argparse
from functools import partial
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from ..acitive_02.surface_scan import limit
from ..i04_geometric_traversal.physics import quaternion, rotation
from . import informative_run, multirate_wrist, online_tactile_mesh, posterior_route, route_pursuit, strip_information
from .experiment import digest
from .multirate_wrist import MultirateWorker
from .sweep import DT, LOAD_LIMIT, PREVIEW_S
from .ycb_fixtures import OBJECTS


LEVER = .10


class PadVelocity:
    """Measure finger articulation separately from wrist motion at each previously loaded material point."""
    def __init__(self):
        self.previous=None
        self.joint_drift=np.zeros(4)
        self.duration=np.zeros(4, dtype=int)
        self.last_loads=np.zeros(5)
        self.load_slope=np.zeros(5)
        self.shift=self.turn=np.zeros(3)
        self.corrected = 0

    def observe(self,hand,data,touch):
        sensor=rotation(data.qpos[3:7])
        root=data.qpos[:3].copy()
        origins=data.geom_xpos[hand.pads]
        matrices=data.geom_xmat[hand.pads].reshape(-1,3,3)
        if self.previous is not None:
            prev_root,prev_r,points,ns,hit,local=self.previous
            continued=hit & touch.measured_heads
            material=origins+np.einsum('fij,fj->fi',matrices,local)
            wrist_only=root+(sensor*prev_r.inv()).apply(points-prev_root)
            drift=np.einsum('ij,ij->i',material-wrist_only,ns)
            self.joint_drift=np.where(continued,.5*self.joint_drift+.5*np.clip(drift,-.0008,.0008),0.)
        offsets=np.einsum('fji,fj->fi',matrices,touch.points-origins)
        self.previous=(root,sensor,touch.points.copy(),touch.normals.copy(),touch.measured_heads.copy(),offsets)
        self.duration=np.where(touch.measured_heads & (max(touch.loads)<LOAD_LIMIT),self.duration+1,0)
        self.load_slope=.55*self.load_slope+.45*(touch.loads-self.last_loads)
        self.last_loads=touch.loads.copy()

    def filter(self,step,turn,touch):
        ids=np.flatnonzero(touch.measured_heads)
        # Limit acceleration at the contact centroid. A wrist rotation about a
        # stationary contact should not be mistaken for tangential pad motion.
        root=self.previous[0]
        points,normals=touch.points[ids],touch.normals[ids]
        pivot=points.mean(0)
        arm=pivot-root
        goal=step+np.cross(turn,arm)
        previous=self.shift+np.cross(self.turn,arm)
        forecast=touch.loads+2.*np.maximum(0.,self.load_slope)
        scale=np.clip((7.-max(forecast))/3.5,.18,1.)
        # Attenuate the tangential request before a large impulse triggers withdrawal.
        lead=normals[int(np.argmax(touch.forces[ids]))]
        target=scale*goal+(1.-scale)*(goal@lead)*lead
        slip=previous+limit(target-previous,.000065 if len(ids)>=2 else .000040)
        angular=self.turn+limit(scale*turn-self.turn,.00070)
        jac=np.column_stack((normals,np.cross(points-pivot,normals)/LEVER))
        proposal=np.r_[slip,angular*LEVER]
        predicted=jac@proposal+self.joint_drift[ids]
        # For a light pad retain the present normal gap. For increasing normal
        # load, permit that specific pad to retreat while retaining other support.
        forces=touch.forces[ids]
        lo=np.where(forces>3.,np.clip(.00009*(forces-2.),.00006,.00038),-.00020)
        hi=np.clip(.000075*(forces-1.15)+.000055,-.00007,.00038)
        hi=np.maximum(hi,lo+.00004)
        desired=np.clip(predicted,lo,hi)
        self.corrected+=int(np.any(abs(predicted-desired)>.000025))
        weights=np.where((forces>3.)|(self.duration[ids]>5),1.,.65)
        a=weights[:,None]*jac
        delta=np.linalg.solve(a@a.T+.06**2*np.eye(len(ids)),weights*(desired-predicted))
        proposal+=a.T@delta
        rot=limit(proposal[3:]/LEVER,.0125)
        delta_pos=proposal[:3]-(Rotation.from_rotvec(rot).apply(arm)-arm)
        return limit(delta_pos,.002),rot


class GuardWrist(route_pursuit.PursuitWrist):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.pad_guard=PadVelocity()

    def observe(self,data,touch,tick):
        super().observe(data,touch,tick)
        self.pad_guard.observe(self.hand,data,touch)

    def command(self,touch):
        start,att=self.position.copy(),self.attitude
        cmd,ahead=super().command(touch)
        step,turn=cmd[:3]-start,(self.attitude*att.inv()).as_rotvec()
        apply=(self.flow is not None and self.phase()=='scan' and touch.measured_heads.any()
               and touch.loads.max()<LOAD_LIMIT and touch.loads[-1]<.7)
        if not apply:
            self.pad_guard.shift,self.pad_guard.turn=step,turn
            return cmd,ahead
        step,turn=self.pad_guard.filter(step,turn,touch)
        # Recheck the servo's measured palm clearance after correcting the twist.
        next_r=Rotation.from_rotvec(turn)*att
        if self.planes is not None:
            verts,anchors,normals=self.planes
            before=att.apply(verts)+start-anchors
            delta=next_r.apply(verts)+start+step-anchors
            old=np.einsum('ij,ij->i',before,normals)
            gap=np.einsum('ij,ij->i',delta,normals)
            closes=(np.linalg.norm(delta-gap[:,None]*normals,axis=1)<.023)&(gap<.006)&(gap<old)
            if closes.any():
                i=int(np.argmin(np.where(closes,gap,np.inf)))
                step=limit(step+min(.0017,old[i]-gap[i]+.00016)*normals[i],.002)
        self.position=self.workspace_center+limit(start+step-self.workspace_center,.45)
        self.attitude=next_r
        self.pad_guard.shift,self.pad_guard.turn=self.position-start,turn
        cmd=np.r_[self.position,quaternion(next_r)]
        future=Rotation.from_rotvec(PREVIEW_S/DT*turn)*next_r
        return cmd,np.r_[self.position+PREVIEW_S/DT*(self.position-start),quaternion(future)]


def run(folder,obj,seconds,device):
    modules=(route_pursuit,multirate_wrist,online_tactile_mesh,posterior_route,strip_information)
    sources=informative_run.source_hashes()
    for path in [Path(__file__),Path(__file__).with_name('contact_first.py'),*(Path(m.__file__) for m in modules)]:
        sources[path.name]=digest(path)
    with patch.multiple(informative_run,InformativeWrist=partial(GuardWrist,device=device),
                        InformativeWorker=MultirateWorker,RELEASE=f'pad-velocity-{device}-global400-local5',
                        source_hashes=lambda:sources):
        return informative_run.run(folder,obj,seconds)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--object',choices=OBJECTS,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--seconds',type=float,default=240.)
    parser.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    args=parser.parse_args()
    print(run(args.out,args.object,args.seconds,args.device),flush=True)
