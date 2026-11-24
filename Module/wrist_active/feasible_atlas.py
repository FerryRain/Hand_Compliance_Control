"""Reserve space to unload the wrist and roll a missing fingertip toward an observed surface."""
import argparse
from collections import deque
from functools import partial
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from ..acitive_02.aude import tangent,unit
from ..acitive_02.surface_scan import limit
from ..i04_geometric_traversal.physics import quaternion,rotation
from . import atlas_wrist, informative_run, tactile_atlas
from .contact_first import signed_gap
from .experiment import digest
from .posterior_route import normalized
from .strip_information import batch_information
from .sweep import DT,LOAD_LIMIT
from .ycb_fixtures import OBJECTS


class FeasibleAtlas(tactile_atlas.AtlasRoute):
    def update(self,surface,tick,pads,hits,n,orient,light):
        event=super().update(surface,tick,pads,hits,n,orient,light)
        if not event or not event['direction_chosen']:
            return event
        pivot=pads.mean(0)
        # Transport the *measured* whole-hand wrist offset along proposed paths.
        # Leave space for a subsequent outward relief on each approach.
        r=self.ray
        best=unit(tangent(self.path_dir,r))
        angle=np.arange(24)*np.pi/12
        dirs=np.cos(angle[:,None])*best+np.sin(angle[:,None])*np.cross(r,best)
        arc=np.array([.12,.28,.52,.82,1.15,1.52])
        rays=np.cos(arc[None,:,None])*r+np.sin(arc[None,:,None])*dirs[:,None]
        pos,ns=self.envelope.predict(rays)
        live_n=unit(n)
        cross=np.cross(live_n,ns)
        def roll(arm):
            return arm+np.cross(cross,arm)+np.cross(cross,np.cross(cross,arm))/np.maximum(.035,1.+(ns@live_n)[...,None])
        predicted=pos+roll(self.wrist-pivot)
        radius=np.linalg.norm(predicted+.020*ns-self.workspace,axis=2)
        if radius[0].max()<.437 and np.linalg.norm(self.wrist-self.workspace)<.425:
            event['relief_reserve_m']=float(.45-radius[0].max())
            return event
        sites=np.stack([pos+roll(pad-pivot) for pad in pads],axis=2)
        actual,surface_n=self.envelope.predict(normalized(sites-self.estimate.center))
        distance,idx=surface.tree.query(actual)
        agreement=(surface.normals[idx]*surface_n).sum(-1)
        probability=(.24+.72*np.exp(-distance/.045)*np.maximum(0.,agreement)**2)*np.exp(-np.linalg.norm(sites-actual,axis=-1)/.025)
        novelty=np.clip(distance/.014,0.,1.)
        gain=batch_information(self.model,actual.reshape(len(dirs),-1,3),(probability*novelty**2).reshape(len(dirs),-1))
        maximum=radius.max(1)
        penalty=np.exp(-np.maximum(maximum-.425,0.)/.012)
        preference=gain*(.6+.4*np.maximum(0.,dirs@best))*penalty
        if np.linalg.norm(self.wrist-self.workspace)>.435:
            preference*=np.exp(-np.maximum(radius[:,:3].min(1)-.405,0.)/.012)
        pick=int(np.argmax(preference))
        self.path_dir=dirs[pick]
        event.update(source='workspace_feasible_strip',direction=self.path_dir.tolist(),target_m=pos[pick,-1].tolist(),
                     acquisition_nat=float(gain[pick]),relief_reserve_m=float(.45-maximum[pick]))
        return event


class FeasibleActions(atlas_wrist.AtlasActions):
    def __init__(self,touch,orient,device,recruit):
        super().__init__(touch,orient,device)
        self.guide=FeasibleAtlas(self.global_model,touch.points,self.up,orient)
        self.recruit=recruit
        self.stable=np.zeros(4,dtype=int)

    def select(self,**p):
        self.guide.wrist=p['wrist_position']
        self.guide.workspace=p['workspace_center']
        result=super().select(**p)
        info=result['diagnostics']
        ids=np.flatnonzero(p['hits'])
        if not self.recruit or info['mode']!='explore' or len(ids)!=2 or min(self.stable[ids])<10 or p['loads'].max()>4.:
            return result
        tips=p['pads']
        axis=unit(tips[ids[1]]-tips[ids[0]])
        pivot=tips[ids].mean(0)
        candidates=[]
        for i in np.flatnonzero(~p['hits']):
            patch=p['surface'].patch(tips[i],preferred_normal=result['normal'])
            if patch is None or p['surface'].tree.query(tips[i])[0]>.035:
                continue
            gap,n=signed_gap(patch,tips[i])
            jac=np.cross(axis,tips[i]-pivot)@n
            if abs(gap)>.024 or abs(jac)<.012:
                continue
            turn=float(np.clip(-gap*jac/(jac*jac+.010**2),-.22,.22))
            after=pivot+Rotation.from_rotvec(turn*axis).apply(tips[i]-pivot)
            score=abs(gap)-abs(signed_gap(patch,after)[0])
            if score>.0005:
                candidates.append((score,int(i),turn))
        if candidates:
            _,finger,angle=max(candidates)
            info.update(recruit_finger=finger,recruit_axis=axis.tolist(),recruit_angle=angle,
                        recruit_anchors=ids.tolist())
        return result


class FeasibleWrist(atlas_wrist.AtlasWrist):
    def __init__(self,hand,data,touch,*,device,recruit,**options):
        super().__init__(hand,data,touch,device=device,guard=recruit,**options)
        self.selector=FeasibleActions(touch,self.attitude,device,recruit)
        self.history=deque(maxlen=800)
        self.retreat=deque()
        self.stable=np.zeros(4,dtype=int)
        self.brake_age=self.retreat_age=self.no_sleep=0
        self.retry_after=0
        self.relocated=0
        self.safe_pose=(self.position.copy(),self.attitude)

    def observe(self,data,touch,tick):
        super().observe(data,touch,tick)
        pose=rotation(data.qpos[3:7])
        hits=touch.measured_heads
        self.stable=np.where(hits & (max(touch.loads)<5.) & (touch.loads[-1]<.50),self.stable+1,0)
        strong=np.sum(self.stable>=10)>=2
        if tick==0 or strong:
            self.safe_pose=(self.measured_position.copy(),pose)
        distance=np.linalg.norm(self.measured_position-self.workspace_center)
        self.brake_age=self.brake_age+1 if self.braking else max(0,self.brake_age-4)
        if not self.retreat:
            if not self.history or np.linalg.norm(self.history[-1][0]-self.measured_position)>.006 or (self.history[-1][1]*pose.inv()).magnitude()>.07:
                self.history.append((self.measured_position.copy(),pose,strong or tick==0))
            trigger=(distance>.432 and (self.brake_age>=12 or distance>.445)) or self.brake_age>46 or self.empty>110
            if tick>=self.retry_after and trigger and self.history:
                back=[]
                for point,orientation,ok in reversed(self.history):
                    back.append((point,orientation))
                    if ok and np.linalg.norm(point-self.workspace_center)<.415 and np.linalg.norm(point-self.measured_position)>.025:
                        break
                self.retreat=deque(back)
                self.retreat_age=0
                self.relocated+=1
        else:
            self.retreat_age+=1
            if self.retreat_age>18 and strong and distance<.418:
                self.finish(tick)
        self.last_region=int(np.argmax(touch.loads))
        idx=self.last_region
        self.loaded_point=(touch.points[idx] if idx<4 and hits[idx] else
                           data.geom_xpos[self.hand.pads[idx]] if idx<4 else data.geom_xpos[self.hand.palm_geom]).copy()

    def finish(self,tick):
        self.retreat.clear()
        self.retry_after=tick+60
        self.brake_age=0
        guide=self.selector.guide
        if guide.ray is not None:
            g=self.last_good_normal
            pivot=self.measured_position-self.attitude.apply(self.selector.local_n)*.13
            guide.ray=unit(pivot-guide.estimate.center)
            d=tangent(self.workspace_center-self.measured_position,guide.ray)
            if np.linalg.norm(d)>.004:
                guide.path_dir=unit(d)
            else:
                guide.path_dir=unit(tangent(guide.path_dir,guide.ray))
            guide.anchor=guide.envelope.predict(guide.ray)[0]

    def command(self,touch):
        if self.retreat:
            start=self.position.copy()
            orient=self.attitude
            while len(self.retreat)>1:
                p,r=self.retreat[0]
                if np.linalg.norm(start-p)>.007 or (r*orient.inv()).magnitude()>.080:
                    break
                self.retreat.popleft()
            goal,dest=self.retreat[0]
            step=limit(goal-start,.00125)
            turn=limit((dest*orient.inv()).as_rotvec(),.010)
            if self.braking:
                radial=unit(start-self.workspace_center)
                outward=self.relief
                if np.linalg.norm(start-self.workspace_center)>.424:
                    outward=outward-max(0.,outward@radial)*radial
                lever=self.loaded_point-self.measured_position
                jac=np.cross(lever,self.relief)
                # Rotate the loaded finger away when free translation is unavailable.
                if np.linalg.norm(jac)>.009:
                    turn=limit(turn+.00080*jac/(jac@jac+.020**2),.0125)
                step=limit(step+.00065*outward,.0015)
            self.position=self.workspace_center+limit(start+step-self.workspace_center,.45)
            self.position_ref=self.position.copy()
            self.attitude=Rotation.from_rotvec(turn)*orient
            if len(self.retreat)==1 and np.linalg.norm(self.position-goal)<.004 and (dest*self.attitude.inv()).magnitude()<.055:
                self.finish(self.tick)
            now=np.r_[self.position,quaternion(self.attitude)]
            return now,now.copy()
        if self.flow is not None and not self.braking:
            row=self.flow[0]
            if 'recruit_axis' in row and all(touch.measured_heads[row['recruit_anchors']]) and max(touch.loads)<4.:
                axis=np.asarray(row['recruit_axis'])
                adjust=Rotation.from_rotvec(np.clip(row['recruit_angle'],-.095,.095)*axis)
                desired=adjust*Rotation.from_quat(row['desired_xyzw'])
                p=touch.points[row['recruit_anchors']].mean(0)
                target=p+adjust.apply(np.asarray(row['goal_point'])-p)
                self.flow=(dict(row,desired_xyzw=desired.as_quat().tolist(),goal_point=target.tolist()),self.flow[1])
                try:
                    return super().command(touch)
                finally:
                    self.flow=(row,self.flow[1])
        return super().command(touch)


class FeasibleWorker(atlas_wrist.AtlasWorker):
    def submit(self,data,touch,tick,lead):
        self.servo.selector.stable=self.servo.stable.copy()
        return super().submit(data,touch,tick,lead)

    def install(self,tick,*,initial=False):
        row=super().install(tick,initial=initial)
        if row is not None:
            row.update(retreating=bool(self.servo.retreat),retreat_count=self.servo.relocated)
            if self.servo.retreat:
                row.update(mode='recontact',selected_direction=[0.,0.,0.])
        return row


def run(folder,obj,seconds,device,recruit):
    modules=(atlas_wrist,atlas_wrist.face_wrist,atlas_wrist.contact_velocity,tactile_atlas,atlas_wrist.multirate_wrist,
             atlas_wrist.online_tactile_mesh,atlas_wrist.posterior_route,atlas_wrist.strip_information)
    sources=informative_run.source_hashes()
    for path in [Path(__file__),Path(__file__).with_name('contact_first.py'),*(Path(m.__file__) for m in modules)]:
        sources[path.name]=digest(path)
    with patch.multiple(informative_run,InformativeWrist=partial(FeasibleWrist,device=device,recruit=recruit),
                        InformativeWorker=FeasibleWorker,RELEASE=f'reserve-recruit{int(recruit)}-{device}-global400-local5',
                        source_hashes=lambda:sources):
        return informative_run.run(folder,obj,seconds)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--object',choices=OBJECTS,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=240.)
    p.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    p.add_argument('--recruit',action='store_true')
    a=p.parse_args()
    print(run(a.out,a.object,a.seconds,a.device,a.recruit),flush=True)
