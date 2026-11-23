"""Reissue the complete selected wrist trajectory using only recorded sensor packets."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from ..i04_geometric_traversal.fingertip_hand import FingertipHand
from ..i04_geometric_traversal.physics import rotation
from .experiment import digest, save_json
from .history import ROBOT,read_model
from .sweep import SweepWrist
from .sweep_experiment import OUT
from .wrist_worker import WristWorker


def replay(root, case='pear/active-02', *, output=None):
    path=root/'runs'/case/'sweep-dp'
    receipt=json.loads((path/'run.json').read_bytes())
    if receipt['trace_sha256']!=digest(path/'trace.npz') or receipt['robot_sha256']!=digest(ROBOT):
        raise ValueError('Replay source mismatch')
    with np.load(path/'trace.npz') as f:
        trace={k:f[k] for k in ('qpos','pad_valid','pad_force_n','region_load_n','points_m',
                                'normals_outward','palm_command','published_preview','phase')}
    q=trace['qpos'][0]
    hand=FingertipHand(read_model(ROBOT),object_name='fixture',
                       grasp=(q[:3],rotation(q[3:7]),q[7:]),surface_queries=False)
    if hand.objects:
        raise ValueError('Object present in replay')
    data=mujoco.MjData(hand.model)
    def packet(t):
        data.qpos[:]=trace['qpos'][t]
        mujoco.mj_kinematics(hand.model,data)
        mujoco.mj_comPos(hand.model,data)
        return SimpleNamespace(measured_heads=trace['pad_valid'][t],forces=trace['pad_force_n'][t],
                               loads=trace['region_load_n'][t],points=trace['points_m'][t],
                               normals=trace['normals_outward'][t])
    initial=packet(0)
    plan=SweepWrist(hand,data,initial)
    n=len(trace['qpos'])-1
    logged=[json.loads(line) for line in (path/'plans.jsonl').read_text(encoding='utf-8').splitlines()]
    largest_wrist=largest_preview=0.
    i=0
    with WristWorker(plan,paced=False) as worker:
        worker.observe(data,initial,0)
        worker.submit(data,initial,0,0)
        first=worker.install(0,initial=True)
        for t in range(n):
            sense=packet(t) if t else initial
            row=worker.update(data,sense,t,n) if t else first
            if row is not None:
                keys=('tick','apply_tick','choice','cells','pass_index','escape_count','last_new_tick')
                if 'face_tick' in logged[i]:
                    keys+=('face_tick',)
                for key in keys:
                    if row[key]!=logged[i][key]:
                        raise ValueError(f'Wrist decision mismatch at tick {t}, field {key}')
                i+=1
            now,ahead=plan.command(sense)
            largest_wrist=max(largest_wrist,float(np.abs(now-trace['palm_command'][t+1]).max()))
            largest_preview=max(largest_preview,float(np.abs(ahead-trace['published_preview'][t+1]).max()))
            if max(largest_wrist,largest_preview)>1e-8:
                raise ValueError(f'Wrist/preview mismatch at tick {t}')
    if i!=len(logged):
        raise ValueError('Incomplete decision replay')
    save_json(output or root/'causal-replay.json',dict(physical_seconds=receipt['signature']['seconds'],packets=n,
              decisions=i,object_geometries=len(hand.objects),max_pose_component_error=largest_wrist,
              max_preview_component_error=largest_preview,trace_sha256=receipt['trace_sha256']))
    print(f'Reproduced {i} decisions, {n} wrist commands and previews',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=OUT)
    torch.set_num_threads(1)
    replay(parser.parse_args().root)
