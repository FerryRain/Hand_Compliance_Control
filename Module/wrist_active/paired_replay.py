"""Isolate the controller by replaying each executed GPIS/DP wrist tape under pure MCC."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from time import sleep

import mujoco
import numpy as np
import torch

from ..acitive_02.experiment import CONTROL_DT, CONTROL_STEPS, digest
from ..i04_geometric_traversal.fingertip_traversal import ForceFeedback
from .evaluate import movement, read_trace, reconstruct, reference
from .experiment import BUDGET, CASES, ROOT, save_json, setup
from .planner import LOAD_LIMIT, VOXEL_M, interpolate, rotation


def compare(root, obj, start):
    parent=root/'runs'/obj/start/'history-dp'
    target=root/'paired-replay'/obj/start
    target.mkdir(parents=True,exist_ok=True)
    result=target/'paired.json'
    if result.exists():
        return json.loads(result.read_bytes())
    receipt=json.loads((parent/'run.json').read_bytes())
    assert receipt['trace_sha256']==digest(parent/'trace.npz')
    dp=read_trace(parent/'trace.npz',round(BUDGET/CONTROL_DT))
    wrist=dp['palm_command']
    plant,data,_,_,touch,state,_=setup(obj,start)
    assert sha256(state.tobytes()).hexdigest()==receipt['integration_state_sha256']
    np.testing.assert_array_equal(data.qpos,dp['qpos'][0])
    feedback=ForceFeedback()
    nominal=data.qpos[7:].copy()
    normals=np.broadcast_to(rotation(wrist[0,3:]).apply([0.,0.,1.]),(4,3)).copy()
    q=[data.qpos.copy()]; hits=[touch.measured_heads]; loads=[touch.loads]
    applied=[wrist[0].copy()]
    for first,last in zip(wrist[:-1],wrist[1:]):
        normals[touch.measured_heads]=touch.normals[touch.measured_heads]
        for ref in interpolate(first,last,np.arange(1,CONTROL_STEPS+1)/CONTROL_STEPS):
            data.ctrl[:]=feedback.command(plant,data,nominal,normals,touch,True)
            data.mocap_pos[plant.driver]=ref[:3]
            data.mocap_quat[plant.driver]=ref[3:]
            mujoco.mj_step(plant.model,data)
        # Interpolation returns an equivalent unit quaternion; store the exact
        # command supplied to the same interpolation routine as the DP trial.
        applied.append(last.copy())
        mujoco.mj_forward(plant.model,data)
        touch=plant.measure(data)
        q.append(data.qpos.copy()); hits.append(touch.measured_heads); loads.append(touch.loads)
    applied=np.asarray(applied)
    np.testing.assert_array_equal(applied,wrist)
    q,hits,loads=map(np.asarray,(q,hits,loads))
    np.savez_compressed(target/'mcc-replay.npz',qpos=q,pad_valid=hits,region_load_n=loads,palm_command=applied)
    joint_motion=movement(q)&movement(dp['qpos'])
    joint_motion[0]=False
    exact=joint_motion & (hits.sum(1)==2) & (loads.max(1)<LOAD_LIMIT)
    clean=dp['pad_valid']&(dp['region_load_n'].max(1)<LOAD_LIMIT)[:,None]
    retained_pair=(~hits|clean).all(1)
    plus=exact&retained_pair&(clean.sum(1)>=3)
    ticks,fingers=np.nonzero(clean)
    extra=plus[ticks]&~hits[ticks,fingers]
    points=dp['points_m'][ticks,fingers]
    norm=dp['normals_outward'][ticks,fingers]
    all_cells=np.floor(points/VOXEL_M).astype(int)
    ref=reference(obj)
    areas,_=ref.area(points[~extra],norm[~extra],ticks[~extra],len(q)-1)
    recon=reconstruct(points[~extra],norm[~extra],ref,target/'masked-extra-signed.ply','signed')
    readout=dict(obj=obj,start=start,matched_motion_s=float(joint_motion.sum()*CONTROL_DT),
        pure_mcc_exactly_two_s=float(exact.sum()*CONTROL_DT),same_two_plus_s=float(plus.sum()*CONTROL_DT),
        same_two_plus_given_mcc_exactly_two=float(plus.sum()/exact.sum()) if exact.any() else None,
        dp_low3_whole=float((clean.sum(1)[1:]>=3).mean()),mcc_low3_whole=float(((hits.sum(1)>=3)&(loads.max(1)<8.))[1:].mean()),
        removed_rows=int(extra.sum()),exclusive_cells=int(len(np.unique(all_cells,axis=0))-len(np.unique(all_cells[~extra],axis=0))),
        masked_coverage10=float(areas[0,-1]),masked_signed=recon,initial_sha256=receipt['integration_state_sha256'],
        dp_trace_sha256=receipt['trace_sha256'],wrist_commands_sha256=sha256(applied.tobytes()).hexdigest())
    save_json(result,readout)
    print(f'{obj}/{start}: {exact.sum()*CONTROL_DT:.2f} s exact two, {plus.sum()*CONTROL_DT:.2f} s same pair plus extra',flush=True)
    return readout


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    args=parser.parse_args()
    torch.set_num_threads(1)
    save_json(args.root/'paired-replay-protocol.json',dict(controller='same frozen initial nominal q plus MCC',
        source='entire executed history-dp wrist tape',cases=CASES,seconds=BUDGET,tactile_dt_s=CONTROL_DT,
        conditioned_on='both wrists moving, MCC exactly two and all five regions <8 N',
        numerator='DP retains the same two physical fingers plus at least one more with all five regions <8 N',
        masking='only extra-finger samples from those matched instants; all other real DP samples kept',
        script_sha256=digest(Path(__file__))))
    for obj,start in CASES:
        while not (args.root/'runs'/obj/start/'history-dp/run.json').is_file():
            sleep(4)
        compare(args.root,obj,start)


if __name__=='__main__':
    main()
