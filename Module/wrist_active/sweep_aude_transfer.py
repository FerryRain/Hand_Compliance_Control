"""Apply the retained, complete Active-2-A Aude+MCC stack to matched I01-A fixtures."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from ..acitive_02 import active_a as aude
from .experiment import ACTIVE2, CASES, digest, save_json, seed_for
from .sweep_experiment import OUT, SMOOTH, fixture
from .sweep_score import score


def rollout(root,obj,start,seconds):
    folder=root/('baseline-check' if obj=='pear' else 'strong-baseline')/obj/start/'aude-mcc'
    folder.mkdir(parents=True,exist_ok=True)
    receipt=folder/'run.json'
    if receipt.exists():
        rec=json.loads(receipt.read_bytes())
        if (rec['code']!=digest(Path(__file__)) or rec['trace_sha256']!=digest(folder/'trace.npz')
                or rec['seconds']!=seconds):
            raise ValueError('Different baseline rollout exists')
        return folder
    (plant,data,hand,sensed,touch,initial,_),case,_=fixture(obj,start)
    live=aude.state_posture(hand,sensed)
    grasp=aude.GraspEnvelope(live.joints,hand.limits)
    ik=aude.JointIK(hand,limits=grasp.limits)
    explorer=aude.AudeExplorer(hand,ik,seed_for(obj,start),aude.DEFAULT_STEP_M)
    planner=aude.ActiveSurfacePlanner(ik,grasp,live,3,aude.CONTROL_DT)
    explorer.observe(sensed,touch)
    planner.observe(sensed,touch,0)
    wrist,fingers=aude.WristFeedback(palm_proximity=False),aude.GraspFeedback(grasp)
    raw=stabilized=np.r_[live.position,aude.quaternion(live.orientation)]
    normals=touch.normals.copy()
    records=[aude.record(data,plant,touch,raw,wrist,live.joints,live.joints,'initial',0.)]
    start_time=perf_counter()
    for tick in range(round(seconds/aude.CONTROL_DT)):
        if tick%aude.PLAN_TICKS==0:
            joint_proposal=explorer.plan(sensed,touch)
        phase=tick%aude.REPLAN_TICKS
        if phase==0:
            live=aude.state_posture(hand,sensed)
            goal,mode,normal=planner.plan(live,joint_proposal.posture,touch,tick*aude.CONTROL_DT)
            if planner.done:
                raise RuntimeError('Active-2 completed its three passes before the registered time budget')
            origin=aude.Posture(raw[:3],aude.rotation(raw[3:]),grasp.bound(live.joints))
            pos,quat,joints=aude.Leg(origin,goal,mode).samples(aude.REPLAN_TICKS)
            path=np.column_stack((pos,quat))
        raw=path[phase]
        supported=planner.stabilize(raw,sensed.qpos[hand.q],touch,normal,wrist,tick%aude.SERVO_TICKS==0)
        nominal=grasp.bound(joints[phase])
        segment=aude.blend_wrist(stabilized,supported)
        normals[touch.measured_heads]=touch.normals[touch.measured_heads]
        touch=aude.step_plant(plant,data,fingers,wrist,touch,segment,nominal,normals)
        stabilized=segment[-1]
        aude.robot_state(plant,hand,data,sensed)
        explorer.observe(sensed,touch)
        planner.observe(sensed,touch,tick+1)
        records.append(aude.record(data,plant,touch,stabilized,wrist,nominal,nominal,mode,
                                   perf_counter()-start_time))
    trace={k:np.asarray(v) for k,v in zip(aude.TRACE_KEYS,zip(*records))}
    if abs(data.time-seconds)>1e-6 or not np.isfinite(trace['qpos']).all():
        raise ValueError('Invalid end state')
    if np.any(trace['palm_proximity_drive_n']):
        raise ValueError('Unexpected native geometry proximity in Active-2')
    checked=[]
    if obj=='pear':
        with np.load(ACTIVE2/'aude-mcc/trace.npz') as archive:
            for key in ('qpos','finger_command','points_m','normals_outward','pad_valid','region_load_n'):
                np.testing.assert_array_equal(trace[key],archive[key][:len(records)])
                checked.append(key)
    np.savez_compressed(folder/'trace.npz',**trace)
    save_json(receipt,dict(obj=obj,start=start,seconds=seconds,code=digest(Path(__file__)),
                           active2_source=digest(Path(aude.__file__)),seed=seed_for(obj,start),
                           initial_sha256=sha256(initial.tobytes()).hexdigest(),model_sha256=digest(case/'model.mjb'),
                           trace_sha256=digest(folder/'trace.npz'),identical_archive_fields=checked))
    print(f'Active-2 Aude/MCC {obj} {start} {seconds}s; archive-matched fields {len(checked)}',flush=True)
    return folder


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=OUT)
    p.add_argument('--group',choices=('check','start-01','start-02',*SMOOTH),required=True)
    a=p.parse_args()
    torch.set_num_threads(1)
    if a.group=='check':
        rollout(a.root,'pear','active-02',20.)
        return
    check=a.root/'baseline-check/pear/active-02/aude-mcc/run.json'
    if len(json.loads(check.read_bytes())['identical_archive_fields'])!=6:
        raise ValueError('Complete the archived trace replay first')
    cases=[(o,s) for o,s in CASES if s==a.group] if a.group.startswith('start') else [(a.group,'start-03')]
    for obj,start in cases:
        folder=rollout(a.root,obj,start,240.)
        score(folder,folder,obj,start,[60.,120.,240.])


if __name__=='__main__':
    main()
