"""Choose measured surface strokes that admit a light three-pad wrist posture."""
from argparse import ArgumentParser
from functools import partial
from itertools import combinations
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from ..acitive_02.aude import tangent,unit
from ..acitive_02.surface_scan import limit
from . import atlas_wrist, feasible_atlas, informative_run
from .contact_first import signed_gap
from .experiment import digest
from .multirate_wrist import LOCAL_S
from .strip_information import batch_information
from .ycb_fixtures import OBJECTS


class StrokeGuide(feasible_atlas.FeasibleAtlas):
    def __init__(self,*args):
        super().__init__(*args)
        self.stroke=None
        self.distance=0.
        self.strokes=0
        self.failed_strokes=[]
        self.gain_start=0
        self.normal=None
        self.in_plane=0

    def update(self,surface,tick,pads,hits,n,orient,light):
        event=super().update(surface,tick,pads,hits,n,orient,light)
        if not event:
            return None
        previous=self.stroke
        self.stroke=None
        productive=self.model.count-self.gain_start
        self.gain_start=self.model.count
        if not event['direction_chosen'] or not light or hits.sum()<2:
            return event
        if previous is not None and productive<28:
            self.failed_strokes=(self.failed_strokes+[(previous[0],previous[1],tick)])[-12:]
            return event
        if self.in_plane>=3:
            self.in_plane=0
            return event
        dist,near=surface.tree.query(pads)
        ns=surface.normals[near]
        same=hits & (ns@n>.87)
        if same.sum()<2 or np.linalg.norm(pads[same]-pads[same].mean(0),axis=1).max()<.012:
            return event
        normal=unit(ns[same].mean(0))
        pivot=pads.mean(0)
        h=float(np.mean(pads[same]@normal))
        center=pivot-(pivot@normal-h)*normal
        axes=orient.as_matrix().T
        b=unit(max((tangent(x,normal) for x in axes),key=np.linalg.norm))
        a=np.arange(24)*np.pi/12
        dirs=np.cos(a[:,None])*b+np.sin(a[:,None])*np.cross(normal,b)
        lengths=np.array([.009,.021,.039,.063,.090,.12])
        # A single small wrist tilt can align three fingertips of a known live
        # hand shape. Larger deviations from the observed stance are discounted.
        arms=pads-pivot
        jac=np.column_stack((np.ones(4),np.cross(arms,normal)/.070))
        gaps=pads@normal-h
        fit_best=None
        for trip in combinations(range(4),3):
            ids=np.array(trip)
            j=jac[ids]
            fit=np.linalg.solve(j.T@j+np.diag([.18,.26,.26,.26])**2,-j.T@gaps[ids])
            fit[0]=np.clip(fit[0],-.005,.005)
            fit[1:]=.070*limit(fit[1:]/.070,.18)
            errors=gaps+jac@fit
            cost=np.mean(abs(errors[ids]))+.5*max(0.,-.002-min(errors))+.18*np.linalg.norm(fit)
            if fit_best is None or cost<fit_best[0]:
                fit_best=(cost,fit,errors)
        _,fit,errors=fit_best
        tilting=Rotation.from_rotvec(fit[1:]/.070)
        heads=pivot+fit[0]*normal+tilting.apply(arms)
        flatheads=heads-(heads@normal-h)[:,None]*normal
        sites=flatheads[None,:,None,:]+dirs[:,None,None,:]*lengths[None,None,:,None]
        d,idx=surface.tree.query(sites)
        agreement=surface.normals[idx]@normal
        lateral=abs((sites-surface.points[idx])@normal)
        valid=np.exp(-np.maximum(d-.025,0.)/.043)
        valid*=np.where((d<.045)&((agreement<.65)|(lateral>.010)),.07,1.)
        fit_reach=np.exp(-np.maximum(abs(errors)-.002,0.)/.009)
        prob=valid*fit_reach[None,:,None]
        new=np.clip(d/.014,0.,1.)
        gain=batch_information(self.model,sites.reshape(len(a),-1,3),(prob*new**2).reshape(len(a),-1))
        quorum=np.sort(prob,axis=1)[:,-3,:4].mean(1)
        cross_extent=np.sort((pads-pivot)@np.cross(normal,dirs).T,axis=0)
        width=.020+np.minimum(.020,np.diff(cross_extent,axis=0)).sum(0)
        score=(.20+.80*quorum)*gain*width/.080
        wrist=self.wrist+dirs*.11+.018*normal
        score*=np.exp(-np.maximum(np.linalg.norm(wrist-self.workspace,axis=1)-.423,0.)/.014)
        for pos,heading,at in self.failed_strokes:
            score*=1.-.90*np.exp(-np.linalg.norm(pos-center)/.045-(tick-at)*.01/30.)*np.maximum(0.,dirs@heading)**2
        pick=int(np.argmax(score))
        if score[pick]<1.1 or quorum[pick]<.29:
            return event
        self.stroke=(center.copy(),dirs[pick].copy(),normal.copy(),fit.copy())
        self.normal=normal.copy()
        self.distance=0.
        self.strokes+=1
        self.in_plane+=1
        event.update(source='attainable_three_pad_stroke',direction=dirs[pick].tolist(),normal=normal.tolist(),
                     target_m=(center+.120*dirs[pick]).tolist(),acquisition_nat=float(gain[pick]),
                     predicted_quorum=float(quorum[pick]),fitted_height_m=float(fit[0]),predicted_strip_m=.120)
        return event

    def advance(self,angle):
        if self.stroke is None:
            return super().advance(angle)
        start,d,n,_=self.stroke
        # Use the observed contact footprint to bound the lead of the sweep.
        achieved=(self.pads.mean(0)-start)@d
        pace=.055 if self.hits.sum()>=2 else .030
        self.distance=np.clip(min(achieved+.024,self.distance+pace*LOCAL_S),.0,.124)
        self.anchor=start+(self.distance+.007)*d
        self.ray=unit(self.anchor-self.estimate.center)
        dr=tangent(d,self.ray)
        self.path_dir=unit(dr) if np.linalg.norm(dr)>.15 else d
        return self.anchor,n


class StrokeActions(feasible_atlas.FeasibleActions):
    def __init__(self,touch,orient,device):
        super().__init__(touch,orient,device,True)
        self.guide=StrokeGuide(self.global_model,touch.points,self.up,orient)

    def select(self,**p):
        g=self.guide
        g.hits,g.pads=p['hits'],p['pads']
        result=super().select(**p)
        row=result['diagnostics']
        if g.stroke is None or row['mode']!='explore':
            return result
        anchor,heading,n,fit=g.stroke
        # Use the observed chart for the immediate wrist target. Reduce wrist yaw
        # about the footprint while it is delivering a broad row of contacts.
        base=p['wrist_rotation']
        desired=Rotation.from_quat(row['desired_xyzw'])
        delta=(desired*base.inv()).as_rotvec()
        delta=delta-.80*(delta@n)*n
        delta=limit(delta+.5*fit[1:]/.070,.10)
        desired=Rotation.from_rotvec(delta)*base
        point=g.anchor.copy()
        patch=p['surface'].patch(point,preferred_normal=n)
        if patch is not None:
            gap,ns=signed_gap(patch,point)
            point-=np.clip(gap,-.003,.003)*ns
        info_center=point+float(np.clip(fit[0],-.0015,.0015))*n
        row.update(goal_point=info_center.tolist(),goal_normal=n.tolist(),desired_xyzw=desired.as_quat().tolist(),
                   nominal_stroke=True,stroke_progress_m=float(g.distance))
        result.update(position=info_center-desired.apply(np.asarray(row['pivot'])),rotation=desired,
                      normal=n,direction=heading)
        return result


class StrokeWrist(feasible_atlas.FeasibleWrist):
    def __init__(self,hand,data,touch,*,device,**options):
        super().__init__(hand,data,touch,device=device,recruit=True,**options)
        self.selector=StrokeActions(touch,self.attitude,device)

    def observe(self,data,touch,tick):
        super().observe(data,touch,tick)
        # An interior backtrack ends when a real pad can resume following.
        # A changed finger shape may make the exact old wrist pose unattainable.
        if self.retreat and (self.retreat_age>=180 or
                (self.retreat_age>=38 and self.stable.max()>=15 and
                 np.linalg.norm(self.measured_position-self.workspace_center)<.422)):
            self.finish(tick)
            self.retry_after=tick+120


def run(folder,obj,seconds,device):
    modules=(feasible_atlas,atlas_wrist,atlas_wrist.face_wrist,atlas_wrist.contact_velocity,atlas_wrist.tactile_atlas,
             atlas_wrist.multirate_wrist,atlas_wrist.online_tactile_mesh,atlas_wrist.posterior_route,atlas_wrist.strip_information)
    sources=informative_run.source_hashes()
    for path in [Path(__file__),Path(__file__).with_name('contact_first.py'),*(Path(m.__file__) for m in modules)]:
        sources[path.name]=digest(path)
    with patch.multiple(informative_run,InformativeWrist=partial(StrokeWrist,device=device),
                        InformativeWorker=feasible_atlas.FeasibleWorker,RELEASE=f'anchored-strokes-{device}-global400-local5',
                        source_hashes=lambda:sources):
        return informative_run.run(folder,obj,seconds)


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    p.add_argument('--object',choices=OBJECTS,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=240.)
    p.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    a=p.parse_args()
    print(run(a.out,a.object,a.seconds,a.device),flush=True)
