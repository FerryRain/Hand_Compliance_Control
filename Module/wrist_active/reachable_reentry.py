"""Choose an informative contact with a reachable approach through measured wrist history."""

import argparse
from collections import deque
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np

from . import contact_transition as ct,tactile_field as field
from .entry_frontier import chart_options,verify_chart
from .tactile_reentry import Sample
from .ycb_fixtures import OBJECTS


@dataclass
class Return:
    path: list
    station: object
    finger: int
    landing: np.ndarray
    normal: np.ndarray
    next_region: object
    rate: float
    duration: float


def measured_path(trail,xyz,quat,station):
    points,qs,stamps=trail
    angle=2*np.arccos(np.clip(np.abs(qs@quat),0.,1.))
    dist=np.linalg.norm(points-xyz,axis=1)
    near=(dist<.043)&(angle<.55)
    if not near.any():
        return None
    # A short bridge between two recorded wrist poses must agree in both
    # position and attitude. Otherwise use consecutive low-load observations.
    end=int(np.argmin(abs(stamps-station.tick)))
    if np.linalg.norm(points[end]-station.position)>.017 or abs(qs[end]@station.attitude.as_quat())<.986:
        return None
    options=np.flatnonzero(near)
    start=int(options[np.argmin(dist[options]+.04*angle[options]+.000015*abs(options-end))])
    ids=[start]
    dt=float(dist[start]/.09+angle[start])
    while ids[-1]!=end:
        i=ids[-1]
        sign=1 if end>i else -1
        indices=np.arange(i+sign,end+sign,sign)
        d=np.linalg.norm(points[indices]-points[i],axis=1)
        a=2*np.arccos(np.clip(np.abs(qs[indices]@qs[i]),0.,1.))
        accepted=np.flatnonzero((d<=.024)&(a<=.23))
        if not len(accepted):
            return None
        k=accepted[-1]
        dt+=max(d[k]/.085,a[k]/1.)
        if dt>5.5:
            return None
        ids.append(int(indices[k]))
    return ids,dt


def destination(state,model,history,stations,failed):
    if len(history)<2 or model.mesh is None or not model.count:
        return None
    wrist,att=state['wrist_position'],state['wrist_rotation']
    xyz=np.stack([s.xyz for s in history])
    qs=np.stack([s.attitude.as_quat() for s in history])
    trail=(xyz,qs,np.array([s.tick for s in history]))
    options=[]
    for s in stations:
        displacement=np.linalg.norm(s.position-wrist)
        angle=(s.attitude*att.inv()).magnitude()
        if displacement<.022 and angle<.17 or displacement>.31:
            continue
        if len(s.fingers)<2:
            continue
        damping=min([1.]+[.06+.94*(1.-np.exp(-.5*(np.linalg.norm(s.position-at)/.037)**2))
                           for at,t in failed if state['tick']-t<2000])
        value=damping*(len(s.fingers)+min(s.dwell,35)/70.)/(.5+displacement/.085+angle)
        options.append((value,s,damping))
    best=None
    considered=[]
    mesh=model.mesh
    for _,s,damping in sorted(options,key=lambda a:-a[0])[:20]:
        if any(np.linalg.norm(s.position-p)<.019 for p in considered):
            continue
        route=measured_path(trail,wrist,att.as_quat(),s)
        if route is None:
            continue
        considered.append(s.position)
        ids,travel=route
        candidates=[]
        for j,(point,n) in enumerate(zip(s.points,s.normals)):
            if not len(mesh.nominees):
                break
            for goal in mesh.routes(point,n):
                k=int(np.flatnonzero(mesh.nominees==goal.face)[0])
                end=goal.center-model.band.half_width*goal.normal
                way=goal.waypoint-model.band.half_width*goal.waypoint_normal
                candidates.append((j,end,goal.normal,np.stack((way,end)),goal.distance,mesh.value[k]))
        for _,_,j,option in chart_options(state['surface'],model,[s],s.position,s.attitude,0):
            chart=verify_chart(state['surface'],s.points[j],s.normals[j],option)
            if chart is not None:
                candidates.append((j,*chart))
        for j,end,nn,path,length,gain in candidates:
            support=min(.96,.45+.13*len(s.fingers)+.003*min(s.dwell,40))
            rate=damping*support*gain/(.5+travel+length/.045)
            if best is None or rate>best.rate:
                region=field.Region(int(s.fingers[j]),end,nn,path,length,'history_supported_frontier')
                best=Return([history[k] for k in ids],s,int(s.fingers[j]),s.points[j],s.normals[j],region,float(rate),travel)
        if len(considered)>=4:
            break
    return best


class ReturnFollower(ct.TransitionFollower):
    def __init__(self,base):
        super().__init__(base)
        self.next_try=0
        self.request=None
        self.continuation=None

    def select(self,**state):
        self.request=None
        self.global_model.enabled=(self.settings==('both',True) and self.recovery_needed)
        tick=state['tick']
        elapsed=0.
        queried=False
        enabled=self.settings[1] and self.settings[0] in ('both','reacquire')
        if (enabled and self.recovery_needed and not self.in_transfer and not state['hits'].any() and
                self.empty>=15 and state['observed_phase']!='withdraw' and state['loads'][-1]<.45 and
                state['loads'].max()<6. and tick>=self.next_try):
            start=perf_counter()
            self.request=destination(state,self.global_model,self.trail,self.stations,self.failed_stations)
            self.next_try=tick+80
            elapsed=1000*(perf_counter()-start)
            queried=True
        if self.in_transfer or self.request is not None:
            self.entry=None
            self.base.tip_forces=self.tip_forces
            result=self.base.select(**state)
        else:
            if self.continuation is not None:
                region=self.continuation
                self.continuation=None
                if state['hits'][region.finger]:
                    self.guide.region,self.guide.cursor,self.guide.refresh=region,0,False
                    self.guide.last_tick,self.guide.owner_lost_since=tick,None
            result=super().select(**state)
        row=result['diagnostics']
        row['measured_stall']=self.recovery_needed
        if queried:
            row['global_ms']+=elapsed
            row['history_return_selection_ms']=elapsed
            if row['global_event'] is None:
                row['global_event']=dict(observed_tick=tick,direction_chosen=False,source='reachable_tactile_landing')
        if self.request:
            g=self.request
            row.update(history_return_finger=g.finger,history_landing_m=g.landing.tolist(),
                       history_continuation_m=g.next_region.point.tolist(),history_rate=g.rate,
                       measured_approach_nodes=len(g.path),estimated_approach_s=g.duration)
        return result


class ReturnWrist(ct.TransitionWrist):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        base=self.selector.base
        base.global_model=base.guide.model=self.selector.global_model.base
        self.selector=ReturnFollower(base)
        self.approach_trail=deque(maxlen=2200)
        self.drought=deque(maxlen=161)
        self.return_goal=None
        self.failed_stations=deque(maxlen=10)
        self.in_drought=False
        self.resume_cells=0

    def observe(self,data,touch,tick):
        super().observe(data,touch,tick)
        live=np.flatnonzero(touch.measured_heads)
        if touch.loads.max()<4.5 and touch.loads[-1]<.45:
            pos,att=self.measured_position,self.live_attitude
            prev=self.approach_trail[-1] if self.approach_trail else None
            if (prev is None or np.linalg.norm(pos-prev.xyz)>.007 or
                    (att*prev.attitude.inv()).magnitude()>.084):
                self.approach_trail.append(Sample(tick,pos.copy(),att,self.normal.copy(),len(live)))
        if tick%5==0:
            cells=len(self.surface.samples)
            self.drought.append((cells,len(live)))
            if self.in_drought:
                self.in_drought=not (cells-self.resume_cells>=9 and (np.max(self.dwell)>=12 or cells-self.resume_cells>=18))
            elif len(self.drought)==self.drought.maxlen and cells-self.drought[0][0]<=5:
                counts=np.array([x[1] for x in self.drought])
                self.in_drought=bool(np.mean(counts==0)>.30 and np.mean(counts<=1)>.75)
                if self.in_drought:
                    self.resume_cells=cells
        self.recovery_needed=self.in_drought
        if self.return_goal is not None:
            g,issued=self.return_goal
            expired=tick-issued>int(100*(1.4+1.6*g.duration))
            if self.anchor is None or expired:
                self.return_goal=None
                if expired and self.anchor is not None:
                    self.leave_return(tick)
                landed=live[(np.linalg.norm(touch.points[live]-g.landing,axis=1)<.025)&
                            (touch.normals[live]@g.normal>.5)] if len(live) else []
                if len(landed):
                    g.next_region.finger=int(landed[0])
                    self.selector.continuation=g.next_region
                elif not len(live) or len(self.surface.samples)-self.start_cells<4:
                    self.failed_stations.append((g.station.position,tick))
                self.selector.next_try=tick+45


class ReturnWorker(ct.TransitionWorker):
    def submit(self,data,touch,tick,lead):
        s=self.servo
        f=s.selector
        f.in_transfer=s.return_goal is not None
        f.trail=tuple(s.approach_trail) if s.in_drought and not touch.measured_heads.any() else ()
        f.failed_stations=tuple(s.failed_stations)
        return super().submit(data,touch,tick,lead)

    def install(self,tick,*,initial=False):
        row=super().install(tick,initial=initial)
        s=self.servo
        if row is None or 'measured_approach_nodes' not in row:
            return row
        g=s.selector.request
        if g is not None and not s.unloading and not s.touch.measured_heads.any():
            s.anchor=Sample(g.station.tick,g.station.position,g.station.attitude,g.normal,len(g.station.fingers))
            s.path=deque(g.path)
            s.at_station=s.began_return=s.last_progress=tick
            s.last_error=np.inf
            s.start_cells=len(s.surface.samples)
            s.rescue=s.flow=None
            s.return_goal=(g,tick)
            s.reentries+=1
            row['history_approach_installed']=True
        return row


def run(folder,obj,seconds=240,device='cuda',mode='both',prior=None):
    return field.run(folder,obj,seconds,device,prior,wrist_cls=partial(ReturnWrist,mode=mode),worker_cls=ReturnWorker,
       source_files=(__file__,ct.__file__,ct.entry_frontier.__file__,ct.transition_compute.__file__,
                     ct.delivery.__file__,ct.tactile_reentry.__file__,ct.tactile_stall.__file__),release='reachable-tactile-reentry-'+mode)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--object',choices=OBJECTS,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=240)
    p.add_argument('--prior',type=Path)
    args=p.parse_args()
    print(run(args.out,args.object,args.seconds,prior=args.prior),flush=True)
