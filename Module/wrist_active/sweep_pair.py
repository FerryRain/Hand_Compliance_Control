"""Replay precisely the selected wrist commands with fixed nominal fingers and MCC."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from .evaluate import Reference,movement,reconstruct
from .experiment import CONTROL_DT,CONTROL_STEPS,ForceFeedback,digest,save_json
from .planner import interpolate
from .sweep_experiment import OUT,fixture


def replay(root,obj,start):
    folder=root/'paired-wrist'/obj/start
    folder.mkdir(parents=True,exist_ok=True)
    source=root/'runs'/obj/start/'sweep-dp'
    result=folder/'result.json'
    receipt=json.loads((source/'run.json').read_bytes())
    if digest(source/'trace.npz')!=receipt['trace_sha256']:
        raise ValueError('Different source wrist tape')
    if result.exists() and json.loads(result.read_bytes())['code']==digest(Path(__file__)):
        return
    n=round(240./CONTROL_DT)
    with np.load(source/'trace.npz') as f:
        dp={k:f[k][:n+1] for k in ('palm_command','qpos','pad_valid','region_load_n','points_m','normals_outward')}
    (plant,data,_,_,touch,state,_),_,mesh=fixture(obj,start)
    if sha256(state.tobytes()).hexdigest()!=receipt['initial_sha256']:
        raise ValueError('Different initial state')
    np.testing.assert_array_equal(dp['qpos'][0],data.qpos)
    q0=data.qpos[plant.q].copy()
    normals=touch.normals.copy()
    control=ForceFeedback()
    tape=dp['palm_command']
    qpos=[data.qpos.copy()]
    valid=[touch.measured_heads]
    loads=[touch.loads]
    for a,b in zip(tape[:-1],tape[1:]):
        normals[touch.measured_heads]=touch.normals[touch.measured_heads]
        for ref in interpolate(a,b,np.arange(1,CONTROL_STEPS+1)/CONTROL_STEPS):
            data.ctrl[:]=control.command(plant,data,q0,normals,touch,True)
            data.mocap_pos[plant.driver],data.mocap_quat[plant.driver]=ref[:3],ref[3:]
            mujoco.mj_step(plant.model,data)
        mujoco.mj_forward(plant.model,data)
        touch=plant.measure(data)
        qpos.append(data.qpos.copy())
        valid.append(touch.measured_heads)
        loads.append(touch.loads)
    qpos,valid,loads=map(np.asarray,(qpos,valid,loads))
    np.savez_compressed(folder/'mcc-replay.npz',qpos=qpos,pad_valid=valid,region_load_n=loads)
    masks=dp['pad_valid']&(dp['region_load_n'].max(1)<8.)[:,None]
    moving=movement(qpos)&movement(dp['qpos'])
    moving[0]=False
    two=moving&(valid.sum(1)==2)&(loads.max(1)<8.)
    three=two&(masks.sum(1)>=3)&np.all(~valid|masks,axis=1)
    t,f=np.nonzero(masks)
    remove=three[t]&~valid[t,f]
    p,normals=dp['points_m'][t,f],dp['normals_outward'][t,f]
    ref=Reference(mesh)
    curves,_=ref.area(p[~remove],normals[~remove],t[~remove],n)
    masked=reconstruct(p[~remove],normals[~remove],ref,folder/'masked.ply','signed')
    save_json(result,dict(code=digest(Path(__file__)),seconds=240.,dp_trace_sha256=receipt['trace_sha256'],
            wrist_sha256=sha256(tape.tobytes()).hexdigest(),jointly_moving_s=moving.sum()*CONTROL_DT,
            mcc_exact2_s=two.sum()*CONTROL_DT,retained_plus_s=three.sum()*CONTROL_DT,
            conditional=float(three.sum()/two.sum()) if two.any() else None,
            dp_low3=float((masks[1:].sum(1)>=3).mean()),
            mcc_low3=float(((valid[1:].sum(1)>=3)&(loads[1:].max(1)<8.)).mean()),
            removed_rows=int(remove.sum()),masked_cover=float(curves[0,-1]),masked_signed=masked))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=OUT)
    p.add_argument('--object',choices=('pear','ovoid'),default='pear')
    a=p.parse_args()
    torch.set_num_threads(1)
    replay(a.root,a.object,'active-02' if a.object=='pear' else 'start-03')
