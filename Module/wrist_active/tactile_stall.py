"""Recover from measured, spatially confined loss without interrupting productive contact."""

import argparse
from collections import deque
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from ..i04_geometric_traversal.physics import quaternion, rotation
from . import tactile_field as field, tactile_reentry as reentry
from .multirate_wrist import MultirateWrist
from .sweep import DT, LOAD_LIMIT, PREVIEW_S
from .ycb_fixtures import OBJECTS


class StallWrist(reentry.ReentryWrist):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.window = deque(maxlen=61)
        self.began_return = self.start_cells = self.partial_returns = 0
        self.initial_dwell = 0

    def leave_return(self,tick):
        self.anchor=self.rescue=self.flow=None
        self.path.clear()
        self.goal,self.goal_attitude=self.position.copy(),self.attitude
        self.last_return=tick

    def observe(self,data,touch,tick):
        MultirateWrist.observe(self,data,touch,tick)
        pos,att=self.measured_position,rotation(data.qpos[3:7])
        light=touch.measured_heads & (max(touch.loads)<LOAD_LIMIT) & (touch.loads[-1]<.7)
        self.dwell=np.where(light,self.dwell+1,0)
        stable=np.count_nonzero(self.dwell>=8)
        if stable>=2:
            self.last_pair=tick
        if light.any():
            self.last_contact_position=touch.points[light].mean(0)
        if tick%10==0:
            self.window.append((pos.copy(),int(light.sum())))
        if self.anchor is not None:
            age=tick-self.began_return
            new=len(self.surface.samples)-self.start_cells
            if stable>=2 and not self.unloading:
                self.confirmed+=1
                self.tried.clear()
                self.leave_return(tick)
            elif not self.unloading and age>=45 and new>=6 and self.dwell.max()>=14:
                # A stable pad on new surface can already drive a new posterior
                # step. Keep this distinct from a confirmed two-pad return.
                self.partial_returns+=1
                self.leave_return(tick)
            elif age>=500:
                self.leave_return(tick)
            return
        n=self.last_touch[2] if self.last_touch else self.normal
        self.remember(tick,pos,att,n,int(light.sum()) if tick==0 else stable)
        last=self.surface.last_observation_tick
        if (last is None or tick-last<850 or len(self.window)<61 or tick-self.last_return<700 or
                tick-self.last_pair<120 or self.last_contact_position is None or tick%10):
            return
        path,counts=zip(*self.window)
        span=np.linalg.norm(np.ptp(np.stack(path),axis=0))
        near_boundary=np.linalg.norm(pos-self.workspace_center)>.39
        poorly_supported=np.mean(np.asarray(counts)==0)>.45 and np.mean(np.asarray(counts)>=2)<.15
        if span<.029 and poorly_supported and (near_boundary or tick-last>=1800):
            self.return_from_dead_end(tick,pos,att)
            if self.anchor is not None:
                self.failure=(self.last_contact_position.copy(),tick)
                self.began_return,self.start_cells=tick,len(self.surface.samples)

    def command(self,touch):
        start,att=self.position,self.attitude
        engaged=self.anchor is not None and not self.unloading
        command,preview=super().command(touch)
        if not engaged:
            return command,preview
        # Published motion describes the same bounded return that is commanded.
        # Avoid presenting a stationary future to the moving finger policy.
        fraction=min(1.,PREVIEW_S/DT)
        if self.path:
            waypoint=self.path[0]
        else:
            waypoint=self.anchor
        step=self.position-start
        remaining=waypoint.xyz-self.position
        scale=min(PREVIEW_S/DT,np.linalg.norm(remaining)/max(np.linalg.norm(step),1.e-8))
        if step@remaining<=0:
            scale=fraction
        desired=(self.attitude*att.inv()).as_rotvec()
        rot=Rotation.from_rotvec(scale*desired)*self.attitude
        return command,np.r_[self.position+scale*step,quaternion(rot)]


class StallWorker(reentry.ReentryWorker):
    def install(self,tick,*,initial=False):
        row=super().install(tick,initial=initial)
        if row is not None:
            row['reentry_single_pad_on_new_surface']=self.servo.partial_returns
        return row


def run(folder,obj,seconds=240,device='cuda',prior=None):
    return field.run(folder,obj,seconds,device,prior,wrist_cls=StallWrist,worker_cls=StallWorker,
                     source_files=(__file__,reentry.__file__),release='spatially-confirmed-tactile-stall')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--object',required=True,choices=OBJECTS)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--seconds',default=240,type=int)
    parser.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    parser.add_argument('--prior',type=Path)
    args=parser.parse_args()
    print(run(args.out,args.object,args.seconds,args.device,args.prior),flush=True)
