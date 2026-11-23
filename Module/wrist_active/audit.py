"""Recompute every chosen wrist pose from causal sensor packets in a hand-only scene."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import json

import mujoco
import numpy as np

from ..i04_geometric_traversal.fingertip_hand import FingertipHand
from .experiment import BUDGET, CASES, CONTROL_DT, PREVIEW, ROOT, digest, save_json
from .history import HandHistory, ROBOT, read_model
from .planner import HORIZON, PERIOD, WristPlanner, interpolate, rotation


def verify(root,obj,start):
    source=root/'runs'/obj/start/'history-dp'
    rows=[json.loads(x) for x in (source/'plans.jsonl').read_text(encoding='utf-8').splitlines()]
    with np.load(source/'trace.npz') as recorded:
        trace={key:recorded[key] for key in ('qpos','points_m','normals_outward','pad_force_n','region_load_n',
                                               'pad_valid','palm_command','published_preview')}
    q=trace['qpos'][0]
    hand=FingertipHand(read_model(ROBOT),object_name='fixture',
                       grasp=(q[:3],rotation(q[3:7]),q[7:]),surface_queries=False)
    assert not hand.objects
    data=mujoco.MjData(hand.model)
    planner=WristPlanner(hand,HandHistory(root/'hand-history.npz'),'history-dp')
    ticks=round(BUDGET/CONTROL_DT)
    n=round(PERIOD/CONTROL_DT)
    pending=np.tile(q[:7],(ticks+PREVIEW+n,1))
    end=PREVIEW;count=0
    largest_score_error=largest_pose_error=0.
    for t in range(ticks):
        touch=SimpleNamespace(forces=trace['pad_force_n'][t],loads=trace['region_load_n'][t],
            points=trace['points_m'][t],normals=trace['normals_outward'][t],measured_heads=trace['pad_valid'][t])
        planner.observe(touch,t)
        if t%n==0:
            row=rows[count]
            assert row['tick']==t
            data.qpos[:]=trace['qpos'][t]
            mujoco.mj_kinematics(hand.model,data)
            mujoco.mj_comPos(hand.model,data)
            base=pending[end-1]
            target,info=planner.plan(data,touch,base,t)
            if any(info[k]!=row[k] for k in ('choice','pair','mode','feasible')):
                raise ValueError(f'Wrist decision mismatch at {obj}/{start}/{t}')
            largest_pose_error=max(largest_pose_error,float(np.max(np.abs(target-row['goal']))))
            if info['candidates']:
                err=np.abs(np.asarray(info['candidates'])-np.asarray(row['candidates']))
                largest_score_error=max(largest_score_error,float(err.max()))
            if largest_pose_error>1e-7 or largest_score_error>1e-6:
                raise ValueError(f'Causal replay tolerance exceeded at {t}')
            pending[end:end+n]=interpolate(base,target,np.arange(1,n+1)*CONTROL_DT/HORIZON)
            end+=n;count+=1
    assert count==len(rows)
    np.testing.assert_allclose(pending[:ticks],trace['palm_command'][1:],rtol=0.,atol=1e-7)
    np.testing.assert_allclose(pending[PREVIEW-1:ticks+PREVIEW-1],trace['published_preview'][1:],rtol=0.,atol=1e-7)
    print(f'{obj}/{start}: {count} decisions and {ticks} wrist commands reproduced',flush=True)
    return dict(obj=obj,start=start,decisions=count,commands=ticks,planning_object_geoms=len(hand.objects),
                max_score_abs_error=largest_score_error,max_pose_component_abs_error=largest_pose_error,
                measured_trace_sha256=digest(source/'trace.npz'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT)
    root=p.parse_args().root
    out=[]
    for obj,start in [*CASES,('pear','active-02')]:
        out.append(verify(root,obj,start))
    save_json(root/'causality-audit.json',out)


if __name__=='__main__':
    main()
