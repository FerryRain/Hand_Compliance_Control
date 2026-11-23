"""Branch the known palm-load incident from an identical physical controller history."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from ..acitive_02.active_a import robot_state
from .experiment import CONTROL_STEPS,ForceFeedback,digest,save_json,setup
from .load_guard import GuardedWrist
from .planner import interpolate
from .sweep import SweepWrist
from .sweep_experiment import OUT
from .wrist_worker import WristWorker


START=23400
END=23750


class IncidentWrist(GuardedWrist):
    def observe(self,data,touch,tick):
        if tick<START:
            SweepWrist.observe(self,data,touch,tick)
        else:
            super().observe(data,touch,tick)


def replay(root):
    folder=root/'force-priority/brake-branch'
    folder.mkdir(parents=True,exist_ok=True)
    path=root/'runs/pear/active-02/sweep-dp'
    original=json.loads((path/'run.json').read_bytes())
    if original['trace_sha256']!=digest(path/'trace.npz'):
        raise ValueError('Source incident checksum changed')
    with np.load(path/'trace.npz') as recorded:
        raw={k:recorded[k][:END+1] for k in ('qpos','palm_command','finger_nominal','region_load_n')}
    plant,data,hand,sensed,touch,*_=setup('pear','active-02')
    wrist=IncidentWrist(hand,sensed,touch)
    feedback=ForceFeedback()
    previous=data.qpos[:7].copy()
    normals=touch.normals.copy()
    loads=[]
    ref=[]
    peak_error=0.
    with WristWorker(wrist,paced=False) as work:
        work.observe(sensed,touch,0)
        work.submit(sensed,touch,0,0)
        work.install(0,initial=True)
        for tick in range(END):
            if tick:
                work.update(sensed,touch,tick,END+30)
            cmd,_=wrist.command(touch)
            normals[touch.measured_heads]=touch.normals[touch.measured_heads]
            for sub in interpolate(previous,cmd,np.arange(1,CONTROL_STEPS+1)/CONTROL_STEPS):
                data.ctrl[:]=feedback.command(plant,data,raw['finger_nominal'][tick+1],normals,touch,True)
                data.mocap_pos[plant.driver],data.mocap_quat[plant.driver]=sub[:3],sub[3:]
                mujoco.mj_step(plant.model,data)
            previous=cmd
            mujoco.mj_forward(plant.model,data)
            touch=plant.measure(data)
            robot_state(plant,hand,data,sensed)
            if tick<START:
                peak_error=max(peak_error,float(np.max(np.abs(data.qpos-raw['qpos'][tick+1]))),
                               float(np.max(np.abs(cmd-raw['palm_command'][tick+1]))))
                if peak_error>1e-9:
                    raise ValueError(f'Incident replay diverged before the branch at {tick}')
            else:
                loads.append(touch.loads)
                ref.append(cmd)
    original_load=raw['region_load_n'][START+1:END+1]
    guarded=np.asarray(loads)
    np.savez_compressed(folder/'branch.npz',time_s=np.arange(START+1,END+1)*.01,original_load=original_load,
                        guarded_load=guarded,wrist=np.asarray(ref),original_wrist=raw['palm_command'][START+1:END+1])
    def stats(values):
        return dict(peak_region_n=float(values.max()),peak_palm_n=float(values[:,-1].max()),
                    over40_s=float((values.max(1)>40.).sum()*.01))
    result=dict(start_s=START*.01,end_s=END*.01,identical_prefix_frames=START,max_prefix_error=peak_error,
                original=stats(original_load),guarded=stats(guarded),source_trace_sha256=original['trace_sha256'],
                code=digest(Path(__file__)),guard=digest(Path(__file__).with_name('load_guard.py')),
                fixed='both branches execute identical pre-recorded nominal fingers and live finger MCC')
    save_json(folder/'result.json',result)
    print(result['original'],result['guarded'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=OUT)
    torch.set_num_threads(1)
    replay(p.parse_args().root)
