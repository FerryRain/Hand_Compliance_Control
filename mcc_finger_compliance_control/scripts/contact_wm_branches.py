"""Same-state structured action branches for the WM collector.

Capture the live CUDA plant, controller and driver, rather than resetting only
q/qvel. Root-group siblings share a split and never create cross-branch edges.
This contract does not yet include a diffusion policy or asynchronous planner.
"""
import copy
import dataclasses
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import warp as wp


def clone_state(value, cpu=False):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone() if cpu else value.detach().clone()
    if isinstance(value, np.ndarray):return value.copy()
    if isinstance(value, dict):return {k:clone_state(v,cpu) for k,v in value.items()}
    if isinstance(value, list):return [clone_state(v,cpu) for v in value]
    if isinstance(value, tuple):return tuple(clone_state(v,cpu) for v in value)
    return copy.deepcopy(value)


def simple_state(value):
    if value is None or isinstance(value,(torch.Tensor,np.ndarray,int,float,bool,str)):return True
    if isinstance(value,(list,tuple)):return all(simple_state(v) for v in value)
    if isinstance(value,dict):return all(isinstance(k,(str,int)) and simple_state(v) for k,v in value.items())
    return False


class CollectionSnapshot:
    def __init__(self,probe):
        torch.cuda.synchronize(probe.device);wp.synchronize()
        self.probe=probe;self.arrays=[]
        def walk(obj,path):
            for field in dataclasses.fields(obj):
                value=getattr(obj,field.name)
                if isinstance(value,wp.array):
                    self.arrays.append((path+'.'+field.name,value,wp.clone(value)))
                elif dataclasses.is_dataclass(value):walk(value,path+'.'+field.name)
        walk(probe.env.sim.wp_data,'data')
        objects=[('probe',probe),('mcc',probe.mcc),('driver',probe.driver),
                 ('observer',probe.observer),('env',probe.env),
                 ('action_manager',probe.env.action_manager)]
        objects.extend(('action.'+n,o) for n,o in probe.env.action_manager._terms.items())
        for name,entity in probe.env.scene.entities.items():
            objects.append((name+'.data',entity.data))
            objects.extend((name+f'.actuator{i}',a) for i,a in enumerate(entity._actuators))
        objects.extend(('sensor.'+n,o) for n,o in probe.env.scene.sensors.items())
        self.attributes=[]
        # Trace is output, never an input to physics. Keep branch traces separate.
        for name,obj in objects:
            entries={k:clone_state(v) for k,v in vars(obj).items()
                     if k not in ('trace','start') and simple_state(v)}
            self.attributes.append((name,obj,entries))
        self.rng=dict(torch=torch.get_rng_state().clone(),cuda=torch.cuda.get_rng_state_all(),
                      numpy=np.random.get_state(),python=random.getstate())
        wp.synchronize();torch.cuda.synchronize(probe.device)

    def restore(self):
        torch.cuda.synchronize(self.probe.device);wp.synchronize()
        for _,array,saved in self.arrays:wp.copy(array,saved)
        wp.synchronize()
        for _,obj,entries in self.attributes:
            for key,value in entries.items():
                current=getattr(obj,key,None)
                if isinstance(value,torch.Tensor) and isinstance(current,torch.Tensor) and current.shape==value.shape:
                    current.copy_(value)
                else:setattr(obj,key,clone_state(value))
        torch.set_rng_state(self.rng['torch']);torch.cuda.set_rng_state_all(self.rng['cuda'])
        np.random.set_state(self.rng['numpy']);random.setstate(self.rng['python'])
        # No forward/step here: restore all derived Warp state verbatim too.
        torch.cuda.synchronize(self.probe.device);wp.synchronize()
        mismatches=[]
        for name,array,saved in self.arrays:
            a=np.ascontiguousarray(array.numpy()).view(np.uint8)
            b=np.ascontiguousarray(saved.numpy()).view(np.uint8)
            if not np.array_equal(a,b):mismatches.append(name)
        for name,obj,entries in self.attributes:
            for key,value in entries.items():
                if isinstance(value,torch.Tensor) and not torch.equal(getattr(obj,key),value):
                    mismatches.append(name+'.'+key)
        if mismatches:raise RuntimeError('Snapshot restore mismatch: '+', '.join(mismatches))
        return dict(warp_arrays=len(self.arrays),exact=True,fields_checked='all Warp bytes and captured Torch tensors')

    def save(self,folder):
        folder.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(folder/'root_warp.npz',**{name:saved.numpy() for name,_,saved in self.arrays})
        torch.save(dict(attributes={name:clone_state(entries,cpu=True) for name,_,entries in self.attributes},
                        rng=clone_state(self.rng,cpu=True)),folder/'root_control.pt')
        return dict(warp_arrays=[dict(name=n,shape=list(a.shape),dtype=str(a.dtype)) for n,a,_ in self.arrays],
                    controller_fields={n:list(e) for n,_,e in self.attributes})


def pulse_scale(frame):
    """A 600ms pulse: 100ms rise, 300ms hold, 200ms withdrawal."""
    t=(frame+1)*.01
    u=np.clip(t/.1,0,1) if t<=.4 else np.clip((.6-t)/.2,0,1)
    return float(u*u*(3-2*u))


def compare_records(left,right):
    if len(left)!=len(right):return dict(passed=False,same_length=False)
    q=float(np.max(np.abs(np.stack([r['q'] for r in left])-np.stack([r['q'] for r in right]))))
    force=np.linalg.norm(np.stack([r['pad_net_force_world'] for r in left])-np.stack([r['pad_net_force_world'] for r in right]),axis=-1)
    mask=float(np.not_equal(np.stack([r['strict'] for r in left]),np.stack([r['strict'] for r in right])).mean())
    command=float(np.max(np.abs(np.stack([r['pre_mcc_q'] for r in left])-np.stack([r['pre_mcc_q'] for r in right]))))
    f95=float(np.quantile(force,.95))
    return dict(passed=bool(q<=1e-5 and f95<=.1 and mask<=.001 and command==0),same_length=True,
                q_max_rad=q,force_p95_n=f95,force_max_n=float(force.max()),strict_mask_disagreement=mask,pre_mcc_q_max_rad=command)


def collect_branches(probe,root,root_frame,main_records,meta,folder):
    from collect_contact_wm import export,save_json,job_fingers
    folder=Path(folder)/'branches';folder.mkdir(parents=True,exist_ok=True)
    group_id=hashlib.sha256(str(folder.resolve()).encode()).hexdigest()
    inventory=root.save(folder)
    save_json(folder/'root.json',dict(group_id=group_id,root_frame=root_frame,split=meta['split'],
        lane_root_ids=[group_id+f'/lane{i}' for i in range(probe.n)],
        history_file='../rollout.h5',history_indices=[max(0,root_frame-80),root_frame],
        state_inventory=inventory,scope='structured nominal only; no DP state in this collector',
        snapshot_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    labels=[('zero_repeat0',0.,False),('zero_repeat1',0.,False),('zero_repeat2',0.,False),
            ('local_minus_0.03',-.03,False),('local_plus_0.03',.03,False),
            ('local_minus_0.08',-.08,False),('local_plus_0.08',.08,False),
            ('all_minus_0.03',-.03,True),('all_plus_0.03',.03,True)]
    original_trace=probe.trace;original_measurements=copy.deepcopy(probe.measurement_audit)
    reports=[];zero=[];checks=[];effects=[];gate=None
    try:
        for label,amplitude,all_fingers in labels:
            checks.append(root.restore());probe.trace=[]
            branch=[probe.sample(2)];branch[0]['dt'][:]=0
            for key in ('candidate_requested_offset_q','candidate_clipped_offset_q'):
                branch[0][key]=np.zeros((probe.n,16),dtype=np.float32)
            vector=torch.zeros_like(probe.q0)
            for lane,job in enumerate(probe.jobs):
                selected=job_fingers(job) if all_fingers else job_fingers(job)[:1]
                for finger in selected:vector[lane,4*finger+2:4*finger+4]=amplitude
            for k in range(60):
                extra=vector*pulse_scale(k);nominal,pose=probe.desired(root_frame+k,extra)
                try:
                    record=probe.step(nominal,pose,2)
                    if not all(np.isfinite(v).all() for v in record.values()):
                        raise RuntimeError('nonfinite_physics_or_measurement: branch record')
                except RuntimeError as error:
                    if not str(error).startswith(('nonfinite_physics_or_measurement','physics_buffer_capacity_exhausted')):raise
                    probe.done_reason=str(error)
                    np.savez_compressed(folder/f'{label}_invalid_physics.npz',
                        **{key:getattr(probe.env.sim.data,key)[:].cpu().numpy() for key in ('qpos','qvel','qacc','ctrl')})
                    break
                record['candidate_requested_offset_q']=extra.cpu().numpy().copy()
                record['candidate_clipped_offset_q']=(nominal-probe.fixed_reference_q).cpu().numpy().copy()
                branch.append(record)
                if probe.done_reason:break
            bm={**meta,'root_group_id':group_id,'root_frame':root_frame,'candidate':label,
                'candidate_vector_rad':vector.cpu().tolist(),'pulse':'100ms rise / 300ms hold / 200ms fall',
                'terminal_reason':probe.done_reason,'branch_admission':'requires sibling zero-repeat verification',
                'diagnostic_only':getattr(probe.args,'branch_diagnostic_only',False),
                'training_admissible':False,
                'admission_policy':'Default quarantine; a separate approved dataset manifest must explicitly admit verified root groups.'}
            reports.append(export(folder/f'{label}.h5',branch,bm))
            if probe.trace:
                np.savez_compressed(folder/f'{label}_substeps.npz',
                    **{k:np.stack([r[k] for r in probe.trace]) for k in probe.trace[0]})
            if amplitude==0:zero.append(branch)
            if len(zero)==3 and gate is None:
                repeats=[compare_records(zero[i],zero[j]) for i,j in ((0,1),(0,2),(1,2))]
                gate=dict(repeat=repeats[0],all_repeats=repeats,
                          original_continuation=compare_records(main_records[root_frame:root_frame+61],zero[0]))
                gate['horizons']={str(n*.01):dict(repeats=[compare_records(zero[i][:n+1],zero[j][:n+1])
                    for i,j in ((0,1),(0,2),(1,2))],original_continuation=compare_records(
                    main_records[root_frame:root_frame+n+1],zero[0][:n+1])) for n in (10,20,40,60)}
                gate['passed']=all(r['passed'] for r in repeats) and gate['original_continuation']['passed']
                save_json(folder/'restore_gate.json',gate)
                if not gate['passed'] and not getattr(probe.args,'branch_diagnostic_only',False):break
            if amplitude:
                n=min(len(branch),len(zero[0]));baseline=zero[0][:n];compare=branch[:n]
                effects.append(dict(candidate=label,
                    actual_nominal_delta_max_rad=float(np.max(np.abs(np.stack([r['pre_mcc_q'] for r in compare])-np.stack([r['pre_mcc_q'] for r in baseline])))),
                    actuator_delta_max_rad=float(np.max(np.abs(np.stack([r['actuator_target_q'] for r in compare])-np.stack([r['actuator_target_q'] for r in baseline])))),
                    measured_q_delta_max_rad=float(np.max(np.abs(np.stack([r['q'] for r in compare])-np.stack([r['q'] for r in baseline]))))))
            print(f'[WM branch] {label}: {len(branch)-1} frames',flush=True)
        result=dict(branches=reports,snapshot_repeat=gate,restore_checks=checks,
                    action_effects=effects,branch_training_admissible=bool(gate and gate['passed']
                        and not any(r['terminal_reason'] for r in reports)
                        and not getattr(probe.args,'branch_diagnostic_only',False)),
                    diagnostic_only=getattr(probe.args,'branch_diagnostic_only',False))
        save_json(folder/'summary.json',result)
        return result
    finally:
        # The main NPZ must not concatenate reset boundaries or branch futures.
        probe.trace=original_trace;probe.measurement_audit=original_measurements
