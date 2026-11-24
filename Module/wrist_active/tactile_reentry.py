"""Leave an observed contact dead end along the measured wrist history."""

import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from ..acitive_02.aude import unit
from ..acitive_02.surface_scan import align, limit
from ..i04_geometric_traversal.physics import quaternion, rotation
from . import tactile_field as field
from .informative_wrist import InformativeWrist
from .multirate_wrist import MultirateWorker, MultirateWrist
from .sweep import DT, LOAD_LIMIT, PREVIEW_S
from .ycb_fixtures import OBJECTS


@dataclass
class Sample:
    tick: int
    xyz: np.ndarray
    attitude: Rotation
    normal: np.ndarray
    stable: int


class ReentryGuide(field.PosteriorGuide):
    def after_loss_weights(self, regions):
        failed, time = self.recovery_observation
        state = self.state
        age = (state['tick']-time)*DT
        strength = np.exp(-max(0.,age)/35.)
        ends = np.array([r.point for r in regions])
        owner = np.array([r.finger for r in regions])
        projected = np.array([r.point + align(state['normals'][r.finger],r.normal).apply(
            state['wrist_position']-state['pads'][r.finger]) for r in regions])
        radius = np.linalg.norm(projected-state['workspace_center'],axis=1)
        near = np.exp(-.5*(np.linalg.norm(ends-failed,axis=1)/.045)**2)
        return np.exp(-strength*(2.6*near+np.maximum(radius-.385,0.)/.014))


class ReentryWrist(MultirateWrist):
    def __init__(self,hand,data,touch,*,device,seed,**options):
        InformativeWrist.__init__(self,hand,data,touch,**options)
        self.selector = field.FieldFollower(device,seed)
        self.selector.guide = ReentryGuide(self.selector.global_model,self.selector.guide.history,seed)
        self.flow = None
        self.past = deque(maxlen=2048)
        self.path = deque()
        self.anchor = None
        self.dwell = np.zeros(4,dtype=int)
        self.last_pair = self.last_return = self.last_progress = 0
        self.last_contact_position = None
        self.failure = None
        self.last_error = np.inf
        self.at_station = 0
        self.reentries = self.confirmed = 0
        self.tried = set()

    def phase(self):
        if self.anchor is not None and not self.unloading:
            return 'approach'
        return super().phase()

    def remember(self, tick, xyz, attitude, normal, stable):
        if self.past:
            prev = self.past[-1]
            if np.linalg.norm(xyz-prev.xyz)<.008 and (attitude*prev.attitude.inv()).magnitude()<.095:
                if stable>prev.stable:
                    self.past[-1] = Sample(tick,xyz.copy(),attitude,normal.copy(),stable)
                return
        self.past.append(Sample(tick,xyz.copy(),attitude,normal.copy(),stable))

    def return_from_dead_end(self, tick, pos, attitude):
        if not self.past:
            return
        points = list(self.past)
        xyz = np.array([s.xyz for s in points])
        quats = np.array([s.attitude.as_quat() for s in points])
        angles = 2*np.arccos(np.clip(np.abs(quats@attitude.as_quat()),0.,1.))
        distance = np.linalg.norm(xyz-pos,axis=1)
        entry = len(points)-1-int(np.argmin((distance+.035*angles)[::-1]))
        diffs = np.linalg.norm(np.diff(xyz,axis=0),axis=1)
        travelled = np.r_[0.,np.cumsum(diffs)]
        length = abs(travelled-travelled[entry])+distance[entry]
        radial = np.linalg.norm(xyz-self.workspace_center,axis=1)
        here = np.linalg.norm(pos-self.workspace_center)
        stable = np.array([s.stable for s in points])
        eligible = (stable>=2)&(distance>.027)&(radial<.405)
        if not eligible.any():
            eligible = (stable>=2)&(distance>.018)
        possible = [i for i in np.flatnonzero(eligible) if points[i].tick not in self.tried]
        if not possible:
            self.tried.clear()
            possible = list(np.flatnonzero(eligible))
        if not possible:
            return
        # Prefer a nearby place with three actual pads, away from the failed
        # boundary. Travel is charged on the path that the wrist really took.
        cost = length+.55*abs(distance-.075)+1.6*np.maximum(radial-min(.35,here-.035),0.)-.032*np.minimum(stable,4)
        j = min(possible,key=lambda i:cost[i])
        self.anchor = points[j]
        self.tried.add(self.anchor.tick)
        stride = 1 if j>=entry else -1
        self.path = deque(points[k] for k in range(entry,j,stride))
        self.reentries += 1
        self.at_station = tick
        self.last_error,self.last_progress = np.inf,tick
        self.rescue = self.flow = None

    def observe(self,data,touch,tick):
        super().observe(data,touch,tick)
        pos,att = self.measured_position,rotation(data.qpos[3:7])
        active = touch.measured_heads & (touch.loads.max()<LOAD_LIMIT) & (touch.loads[-1]<.6)
        self.dwell = np.where(active,self.dwell+1,0)
        if active.any():
            self.last_contact_position = touch.points[active].mean(0)
        stable = np.count_nonzero(self.dwell>=8)
        if stable>=2:
            self.last_pair = tick
        if self.anchor is None:
            n = self.last_touch[2] if self.last_touch else self.normal
            self.remember(tick,pos,att,n,int(active.sum()) if tick==0 else stable)
            stamp = self.surface.last_observation_tick
            if stamp is None or self.last_contact_position is None or tick-self.last_return<250:
                return
            stalled = tick-stamp
            boundary = np.linalg.norm(pos-self.workspace_center)>.420
            if stalled >= (330 if boundary else 700) and tick-self.last_pair>=120:
                self.failure = (self.last_contact_position.copy(),tick)
                self.return_from_dead_end(tick,pos,att)
        else:
            moved = np.linalg.norm(pos-self.anchor.xyz)<.024
            if stable>=2 and (moved or not self.path) and not self.unloading:
                self.last_return = tick
                self.confirmed += 1
                self.anchor = self.rescue = None
                self.path.clear()
                self.tried.clear()
            elif not self.path and tick-self.at_station>220:
                self.return_from_dead_end(tick,pos,att)

    def command(self,touch):
        if self.anchor is None or self.unloading:
            return super().command(touch)
        start,orient = self.position,self.attitude
        while self.path:
            s = self.path[0]
            # A measured wrist pose was physically visited; adding a large
            # normal offset here can make a previously reachable point illegal.
            goal = self.workspace_center+limit(s.xyz+.003*s.normal-self.workspace_center,.447)
            error = np.linalg.norm(goal-start)+.030*(s.attitude*orient.inv()).magnitude()
            if error<self.last_error-.0006:
                self.last_error,self.last_progress = error,self.tick
            if error>.006 and self.tick-self.last_progress<40:
                break
            self.path.popleft()
            self.last_error,self.last_progress = np.inf,self.tick
            self.at_station = self.tick
        if not self.path:
            s=self.anchor
            depth=np.clip(.00013*(self.tick-self.at_station-10),-.002,.006)
            goal = s.xyz-depth*s.normal
        turn=limit((s.attitude*orient.inv()).as_rotvec(),.012)
        step=limit(goal-start,.0012)
        peak=float(max(touch.loads))
        if peak>3.5:
            outward=s.normal
            if touch.measured_heads.any():
                outward=unit(touch.normals[touch.measured_heads].mean(0))
            step-=min(0.,step@outward)*outward
            step+=min(.00055,.00013*(peak-3.5))*outward
        self.position = self.workspace_center+limit(start+step-self.workspace_center,.45)
        self.attitude=Rotation.from_rotvec(turn)*orient
        command=np.r_[self.position,quaternion(self.attitude)]
        return command,command.copy()


class ReentryWorker(MultirateWorker):
    def submit(self,data,touch,tick,lead):
        self.servo.selector.guide.recovery_observation = self.servo.failure
        return super().submit(data,touch,tick,lead)

    def install(self,tick,*,initial=False):
        row=super().install(tick,initial=initial)
        if row is not None:
            s=self.servo
            row.update(reentry_attempts=s.reentries,reentry_confirmed=s.confirmed,
                       reentry_anchor_tick=s.anchor.tick if s.anchor else None)
        return row


def run(folder,obj,seconds=240,device='cuda',prior=None):
    return field.run(folder,obj,seconds,device,prior,wrist_cls=ReentryWrist,worker_cls=ReentryWorker,
                     source_files=(__file__,),release='tactile-dead-end-reentry')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--object',required=True,choices=OBJECTS)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--seconds',default=240,type=int)
    parser.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    parser.add_argument('--prior',type=Path)
    args=parser.parse_args()
    print(run(args.out,args.object,args.seconds,args.device,args.prior),flush=True)
