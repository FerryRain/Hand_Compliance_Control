"""Full-length regional DP development evaluation alongside supervised training.

Workers have independent CUDA contexts and exit after a small batch. The same
batch composition, initial states, diffusion stream, and scheduler scale are
used for paired checkpoints. Force excursions are recorded without stopping.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PARTS = ('true_fingertip', 'connector', 'other_link', 'palm')


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def pad_episode(array, length):
    array = np.asarray(array)
    if len(array) > length or len(array) < 1:
        raise ValueError('Invalid episode padding length')
    return np.concatenate((array, np.repeat(array[-1:], length - len(array), axis=0)))


def contact_statistics(strict, found):
    four = strict.all(-1)
    longest = run = 0
    first = None
    for i, ok in enumerate(four):
        run = 0 if ok else run + 1
        longest = max(longest, run)
        if run == 6 and first is None:
            first = i - 5
    return dict(strict_healthy_four_rate=float(four.mean()),
                found_four_rate=float(found.all(-1).mean()),
                strict_healthy_per_finger=strict.mean(0).tolist(),
                found_per_finger=found.mean(0).tolist(),
                longest_four_loss_frames=longest, first_loss_over5_frame=first)


def _worker(request):
    import torch
    import mujoco
    import warp as wp
    import deploy_b_gpu_serial_parity as deployment
    from batched_b_control_chain import BatchedBControlChain
    from deploy_dp_inverse import _episode, load_episode
    from surface_mcc_finger import FullHandMCCFingerConfig
    from collection_force_regulation import CollectionForceObserver
    from diagnose_force_stop_parts import all_object_loads, region
    from held_relative_target import install_held_relative_targets

    task = json.loads(Path(request).read_text())
    output = Path(task['output'])
    rows = task['rows']
    batch = len(rows)
    frames = max(r['frames'] for r in rows)
    dataset = Path(task['dataset'])
    reference = [load_episode(dataset/'teachers_inverted.h5', r['episode_id'],
                              include_teacher_tactile=False) for r in rows]
    def prepare_reference(array):
        # Explicit diagnostic prefix only; full-rollout default still rejects
        # accidental truncation through pad_episode's length check.
        if task.get('reference_prefix_only',False):array=array[:frames]
        return pad_episode(array,frames)
    data = {key: np.stack([prepare_reference(ref[key]) for ref in reference], axis=1)
            for key in ('q_hand', 'palm_pose_object', 'palm_twist_object')}
    # Once a shorter episode ends its palm is held, and its padded tail is
    # excluded from every metric. Other lanes still complete their full paths.
    for lane, row in enumerate(rows):
        data['palm_twist_object'][row['frames']:, lane] = 0
    with h5py.File(dataset/'teachers_commanded_palm.h5') as h:
        commands = [prepare_reference(_episode(h, r['episode_id'], 'planner_palm_delta_pose_palm'))
                    for r in rows]
    data['planner_palm_delta_pose_palm_commanded'] = np.stack(commands, axis=2)
    initial_target = []
    for row in rows:
        with h5py.File(row['source']) as h:
            initial_target.append(h['q_cmd'][0, 0])
    device = 'cuda:0'
    tensors = dict(strict=torch.zeros((frames,batch,4), device=device, dtype=torch.bool),
                   found=torch.zeros((frames,batch,4), device=device, dtype=torch.bool),
                   q=torch.zeros((frames,batch,16),device=device),
                   nominal=torch.zeros((frames,batch,16),device=device),
                   command=torch.zeros((frames,batch,16),device=device),
                   target_error=torch.zeros((frames,batch),device=device),
                   force=torch.zeros((frames,batch,4),device=device),
                   observed=torch.zeros((frames,batch),device=device,dtype=torch.bool))
    if task.get('contact_substep_trace',False):
        tensors.update(substep_tip_force_world=torch.zeros((frames,10,batch,4,3),device=device),
                       substep_tip_found=torch.zeros((frames,10,batch,4),device=device,dtype=torch.bool),
                       substep_q=torch.zeros((frames,10,batch,16),device=device))
    substep_count = torch.zeros(frames, dtype=torch.long, device=device)
    noise = []
    initialization_trace=[]
    config = task['mcc_config']
    residual_mode=task.get('residual_mode','off')
    training=residual_mode=='train'
    residual_state=None
    agent=None
    dppo=None
    shadow_errors=dict(action=0.,nominal=0.,command=0.,observation=0.)
    if residual_mode!='off':
        if residual_mode not in ('zero','frozen','train'):
            raise ValueError(f'Unknown residual mode: {residual_mode}')
        from train_rl0 import RL0PPO,RL0Residual
        residual_state=torch.load(task['residual_checkpoint'],map_location='cpu',weights_only=False)
        regional=bool(residual_state['config'].get('regional_post_physics_reward'))
        if residual_state.get('actor_updates',0)<=0 and not regional:
            raise ValueError('Residual checkpoint has no learned actor updates')
        if residual_state['config']['actor_input_dim']!=528:
            raise ValueError('Expected the original RL0 observation layout')
        if regional:
            from regional_residual_training import RegionalResidual,restore_agent
            agent=restore_agent(residual_state,training)
        else:
            agent=RL0PPO(residual_state['config'],residual_state['bounds'])
            agent.actor.load_state_dict(residual_state['actor'],strict=True)
            agent.value.load_state_dict(residual_state['value'],strict=True)
            agent.actor.eval().requires_grad_(False)
            agent.value.eval().requires_grad_(False)
        tensors.update(residual_offset=torch.zeros((frames,batch,16),device=device),
                       residual_target=torch.zeros((frames,batch,16),device=device))
        if regional:
            # Completed lanes are physically parked only after their actual
            # reference ends; their padded tails are excluded from PPO.
            for lane,row in enumerate(rows):
                data['palm_pose_object'][row['frames']:,lane,2]+=2.

    class Observer(CollectionForceObserver):
        def configure(self, cfg):
            # Transfer the collection's robot contact materials before splitting
            # the distal geoms, as in the accepted long-palm transfer audit.
            from mjlab.tasks.leaphand.leaphand_mcc_finger_env_cfg import _load_mcc_leaphand_spec
            source = _load_mcc_leaphand_spec()
            properties = {g.name: {k: np.array(getattr(g,k),copy=True)
                           for k in ('friction','solref','solimp')}
                          | {k: getattr(g,k) for k in ('priority','solmix','margin','gap')}
                          for g in source.geoms}
            original = cfg.scene.entities['robot'].spec_fn
            def matched_spec():
                spec = original()
                for geom in spec.geoms:
                    for name, value in properties.get(geom.name, {}).items():
                        if name in ('friction','solref','solimp'):
                            getattr(geom,name)[:] = value
                        else:
                            setattr(geom,name,value)
                return spec
            cfg.scene.entities['robot'].spec_fn = matched_spec
            super().configure(cfg)

    observer = Observer(controller_scope='pad', capture_collision=True)
    if task.get('dppo'):
        if residual_mode!='off' or task['chunk_source']!='dp':
            raise ValueError('Regional DPPO requires DP chunks and zero residual')
        from regional_dppo import RegionalDPPO
        dppo=RegionalDPPO(task,observer)
    original_cfg = deployment.replay_env_cfg
    def env_cfg(**kwargs):
        kwargs.update(hand_stiffness=35., hand_damping=.2, hand_effort_limit=.8)
        return original_cfg(**kwargs)
    deployment.replay_env_cfg = env_cfg
    teacher = torch.as_tensor(data['q_hand'],device=device)
    aggregate = None

    class Chain(BatchedBControlChain):
        def __init__(self, *args, **kwargs):
            kwargs['mcc_config'] = FullHandMCCFingerConfig(**config)
            super().__init__(*args, **kwargs)
            controller = task.get('mcc_controller', 'legacy')
            if controller == 'finger_admittance_v1':
                from finger_admittance_mcc import BatchedFingerAdmittanceMCC, FingerAdmittanceConfig
                candidate = BatchedFingerAdmittanceMCC(self.batch,self.device,
                    FullHandMCCFingerConfig(**config),
                    FingerAdmittanceConfig(**task.get('admittance_config',{})))
                if task.get('admittance_after_bootstrap',False):
                    self.pending_admittance=candidate
                else:
                    self.mcc=candidate
            elif controller != 'legacy':
                raise ValueError(f'Unknown MCC controller: {controller}')
            self.runtime.policy.diffusion.num_inference_steps = 10
            self.scheduler.scale = torch.as_tensor(task['scheduler_scale'], device=device,
                                                   dtype=torch.float64).clamp_min(.02)
            if task['chunk_source'] == 'teacher':
                def provider(frame, q, history):
                    indices = (frame + torch.arange(1, self.runtime.pred_horizon+1,
                               device=device)*self.runtime.stride).clamp_max(frames-1)
                    return teacher[indices].permute(1,0,2).contiguous()
                self.chunk_provider = provider
            else:
                original = self.runtime.policy.diffusion.conditional_sample
                def sample(*args, **kwargs):
                    before = hashlib.sha256(kwargs['generator'].get_state().numpy().tobytes()).hexdigest()
                    result = original(*args, **kwargs)
                    after = hashlib.sha256(kwargs['generator'].get_state().numpy().tobytes()).hexdigest()
                    noise.append(dict(frame=self.control_frame, before=before, after=after))
                    return result
                self.runtime.policy.diffusion.conditional_sample = sample
            if dppo is not None:
                dppo.attach(self)
            self.shadow=None
            if dppo is not None and dppo.probe:
                self.shadow=BatchedBControlChain(*args,**kwargs)
                self.shadow.runtime.policy.diffusion.num_inference_steps=10
                self.shadow.scheduler.scale=self.scheduler.scale.clone()
                self.shadow.chunk_provider=lambda frame,q,history:dppo.reference_chunk
            if agent is not None:
                # Reuse the original actor preprocessing, history and limiter.
                # The stored MCC profile is intentionally NOT transplanted.
                residual_class=RegionalResidual if regional else RL0Residual
                extra=dict(lengths=[r['frames'] for r in rows]) if regional else {}
                self.residual_policy=residual_class(self,agent,observer,residual_state['bounds'],
                    residual_state['config'],zero=residual_mode=='zero' or task.get('zero_action',False),
                    evaluate=not training,**extra)
                if residual_mode=='zero':
                    self.shadow=BatchedBControlChain(*args,**kwargs)
                    self.shadow.runtime.policy.diffusion.num_inference_steps=10
                    self.shadow.scheduler.scale=self.scheduler.scale.clone()
                    self.shadow.chunk_provider=lambda frame,q,history:self.residual_policy.latest_chunk
        def prime(self,q,repeats=30):
            super().prime(q,repeats)
            if self.shadow is not None:self.shadow.prime(q,repeats)
        def begin_settle(self,q,steps):
            super().begin_settle(q,steps)
            if self.shadow is not None:self.shadow.begin_settle(q,steps)
        def set_training_context(self, **context):
            self.context = context
            if agent is not None:self.residual_policy.context=context
        def record_base_substep(self, env, target):
            self.env = env
        def record_substep_force(self):
            nonlocal aggregate
            if self.context['settling']:
                return
            env = self.env
            d = env.sim.wp_data
            m = env.sim.mj_model
            if aggregate is None:
                names = [mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g) or '' for g in range(m.ngeom)]
                groups = [[g for g,n in enumerate(names) if n.startswith('robot/') and region(n)==part]
                          for part in PARTS]
                with wp.ScopedDevice(env.sim.wp_device):
                    aggregate = dict(groups=groups,
                        objects=wp.array(np.asarray([n.startswith('target/') for n in names],dtype=np.int32),dtype=wp.int32),
                        hand=wp.array(np.asarray([n.startswith('robot/') for n in names],dtype=np.int32),dtype=wp.int32),
                        net=wp.zeros((batch,m.ngeom),dtype=wp.vec3),
                        point=wp.zeros((batch,m.ngeom),dtype=float))
            a = aggregate
            with wp.ScopedDevice(env.sim.wp_device):
                a['net'].zero_(); a['point'].zero_()
                wp.launch(all_object_loads,dim=d.naconmax,inputs=[env.sim.wp_model.opt.cone,
                    d.contact.frame,d.contact.friction,d.contact.dim,d.contact.efc_address,d.efc.force,
                    d.njmax,d.nacon,d.contact.geom,d.contact.worldid,a['objects'],a['hand']],
                    outputs=[a['net'],a['point']])
            stream = torch.cuda.ExternalStream(wp.get_stream(env.sim.wp_device).cuda_stream)
            with torch.cuda.stream(stream):
                net = wp.to_torch(a['net']).norm(dim=-1)
                point = wp.to_torch(a['point'])
                parts = torch.stack([torch.maximum(net[:,g].amax(-1),point[:,g].amax(-1))
                                      if g else net.new_zeros(batch) for g in a['groups']],-1)
            torch.cuda.current_stream().wait_stream(stream)
            frame = self.context['frame']
            tensors['force'][frame] = torch.maximum(tensors['force'][frame],parts)
            target_error=(env.scene['robot'].data.joint_pos_target[:, -16:]-self.expected_tick_target).abs().amax(-1)
            tensors['target_error'][frame]=torch.maximum(tensors['target_error'][frame],target_error)
            substep_count[frame] += 1
            # The final call in the tick overwrites earlier contact states.
            x = observer.latest
            if task.get('contact_substep_trace',False):
                tensors['substep_tip_force_world'][frame,self.tick_substeps]=x['pad_force']
                tensors['substep_tip_found'][frame,self.tick_substeps]=x['pad_found']
                tensors['substep_q'][frame,self.tick_substeps]=env.scene['robot'].data.joint_pos[:, -16:]
            tensors['strict'][frame] = (x['pad_found'] & torch.isfinite(x['pad_points']).all(-1)
                & torch.isfinite(x['pad_force']).all(-1) & (x['pad_force'].norm(dim=-1)>=.05))
            tensors['found'][frame] = x['pad_found'] | x['connector_found']
            tensors['q'][frame] = env.scene['robot'].data.joint_pos[:, -16:]
            tensors['observed'][frame] = True
            self.tick_substeps+=1
            if agent is not None and regional and self.tick_substeps==10:
                self.residual_policy.post_physics(frame,env.scene['robot'].data.joint_pos[:, -16:],
                    self.context['palm_pose'],tensors['force'][frame])
            if dppo is not None and self.tick_substeps==10:
                dppo.post_physics(self,frame,env.scene['robot'].data.joint_pos[:, -16:],
                    self.context['palm_pose'],tensors['force'][frame])
        def step(self, **kwargs):
            self.tick_substeps=0
            if (getattr(self,'pending_admittance',None) is not None
                    and not self.context['settling'] and self.context['frame']>=76):
                candidate=self.pending_admittance
                candidate.previous_command=self.mcc.previous_command.clone()
                candidate.handover_command=candidate.previous_command.clone()
                candidate.handover_nominal=self.previous_nominal.clone()
                self.mcc=candidate
                self.pending_admittance=None
            if dppo is not None:self.context['q']=kwargs['q']
            action, diagnostic = super().step(**kwargs)
            if task.get('initialization_audit') and (self.context['settling'] or self.context['frame']<=75):
                initialization_trace.append(dict(
                    phase=torch.tensor(-1 if self.context['settling'] else self.context['frame'],device=device),
                    q=kwargs['q'].detach().clone(),velocity=self.context['q_velocity'].detach().clone(),
                    palm=kwargs['palm_pose_world'].detach().clone(),
                    previous_target=self.context['previous_actuator_target'].detach().clone(),
                    contact_force=kwargs['contact_force_world'].detach().clone(),
                    contact_points=kwargs['contact_points_world'].detach().clone(),
                    contact_found=kwargs['contact_found'].detach().clone(),
                    nominal=diagnostic['nominal_q'].detach().clone(),command=diagnostic['command_q'].detach().clone(),
                    action=action.detach().clone(),recovery_offset=self.runtime_recovery_offset.detach().clone(),
                    force_offset=self.mcc.force_offset.detach().clone()))
            if self.shadow is not None:
                self.shadow.mesh_oracle=self.mesh_oracle
                for name,value in deployment.orchestration_config().items():setattr(self.shadow,name,value)
                other,diag=self.shadow.step(**kwargs)
                for name,a,b in [('action',action,other),('nominal',diagnostic['nominal_q'],diag['nominal_q']),
                                 ('command',diagnostic['command_q'],diag['command_q']),
                                 ('observation',self.observation.states,self.shadow.observation.states)]:
                    error=float((a-b).abs().max())
                    shadow_errors[name]=max(shadow_errors[name],error)
                    if error>1e-7:raise RuntimeError(f'Zero residual changed {name}: {error}')
            self.expected_tick_target=kwargs['q'].detach().clone()+.08*action
            if not self.context['settling']:
                frame = self.context['frame']
                if agent is not None:
                    tensors['residual_offset'][frame]=self.residual_policy.offset
                    tensors['residual_target'][frame]=self.residual_policy.target
                tensors['nominal'][frame] = diagnostic['nominal_q']
                tensors['command'][frame] = diagnostic['command_q']
                if task.get('contact_dynamics_trace',False):
                    for key,name in [('input_tip_force_world','contact_force_world'),
                                     ('input_tip_found','contact_found'),
                                     ('input_tip_points_world','contact_points_world'),
                                     ('input_palm_pose_world','palm_pose_world')]:
                        value=kwargs[name]
                        if key not in tensors:tensors[key]=torch.zeros((frames,*value.shape),device=device,dtype=value.dtype)
                        tensors[key][frame]=value
                for key,value in diagnostic.items():
                    if key.startswith('admittance_'):
                        if key not in tensors:
                            tensors[key]=torch.zeros((frames,*value.shape),device=device,dtype=value.dtype)
                        tensors[key][frame]=value
                if frame % 500 == 0:
                    print(f'[REGIONAL-EVAL] {task["arm"]} frame={frame}/{frames} batch={batch}',flush=True)
            return action, diagnostic
        def close(self):
            nonlocal trained_residual
            if agent is not None:trained_residual=self.residual_policy
            super().close()

    deployment.MODEL = task['checkpoint']
    trained_residual=None
    restore_targets = install_held_relative_targets()
    try:
        meta = deployment.run(batch, rows[0]['episode_id'], frames, device, task['seed'],
            settle_steps=100, max_safe_force_n=0., per_env_safety=False,
            episode_data=data, chain_factory=Chain, contact_observer=observer,
            initial_joint_target=np.stack(initial_target))
    finally:
        restore_targets()
    values = {k: v.cpu().numpy() for k,v in tensors.items()}
    if initialization_trace:
        np.savez_compressed(output.with_suffix('.initialization.npz'),
            **{k:torch.stack([r[k] for r in initialization_trace]).cpu().numpy() for k in initialization_trace[0]})
    counts = substep_count.cpu().numpy()
    lanes = []
    for lane, row in enumerate(rows):
        length = row['frames']
        target_error=float(values['target_error'][:length,lane].max())
        complete = bool(values['observed'][:length,lane].all() and np.all(counts[:length]==10)
                        and target_error<1e-6)
        report = dict(episode_id=row['episode_id'],branch=row['branch'],
                      parent=row['related_plan_group_id'],family=row['family'],frames=length,
                      complete=complete,held_target_max_error_rad=target_error)
        for start, scope in ((0,'full'),(76,'post_bootstrap')):
            strict = values['strict'][start:length,lane]
            found = values['found'][start:length,lane]
            peak = values['force'][start:length,lane].max(-1)
            report[scope] = dict(**contact_statistics(strict,found),
                peak_force_n=float(peak.max()), p95_force_n=float(np.quantile(peak,.95)),
                over100N_frames=int((peak>100).sum()),
                force_peak_by_part_n=dict(zip(PARTS, values['force'][start:length,lane].max(0).tolist())))
        lanes.append(report)
    trace = output.with_suffix('.npz') if task.get('persist_trace',True) else None
    if trace is not None:
        np.savez_compressed(trace, **values, substep_count=counts,
                            episode_ids=[r['episode_id'] for r in rows],
                            episode_lengths=[r['frames'] for r in rows])
    definition=observer.definition()
    definition['compliance']='Collection robot contact materials; Kp35,Kd0.2,effort0.8Nm; held absolute target for10physical substeps'
    result = dict(arm=task['arm'],chunk_source=task['chunk_source'],lanes=lanes,
                  deployment=meta,noise=noise,trace=str(trace) if trace else None,checkpoint=task['checkpoint'],
                  contact_definition=definition,source_config=task['mcc_config'],
                  actuator_contract='One absolute target held over all10physics substeps')
    result['mcc_controller']=task.get('mcc_controller','legacy')
    result['admittance_after_bootstrap']=task.get('admittance_after_bootstrap',False)
    if result['mcc_controller']=='finger_admittance_v1':
        from dataclasses import asdict
        from finger_admittance_mcc import FingerAdmittanceConfig
        result['admittance_config']=asdict(FingerAdmittanceConfig(**task.get('admittance_config',{})))
    if dppo is not None:
        result['dppo']=dppo.finish(shadow_errors)
    if residual_state is not None:
        result['residual']=dict(mode=residual_mode,checkpoint=task['residual_checkpoint'],
            checkpoint_sha256=sha(task['residual_checkpoint']),
            source_actor_updates=residual_state['actor_updates'],actor_updates_during_evaluation=agent.actor_updates,
            critic_updates_during_evaluation=agent.updates,config=residual_state['config'],
            bounds=residual_state['bounds'],stored_MCC_profile_applied=False,
            max_applied_offset_rad=float(np.abs(values['residual_offset']).max()),
            rms_applied_offset_rad=float(np.sqrt(np.square(values['residual_offset']).mean())),
            zero_shadow_errors=shadow_errors if residual_mode=='zero' else None)
        if not training and (agent.actor_updates or agent.updates or agent.rows):
            raise AssertionError('Evaluation changed the residual learner')
    if training:
        from regional_residual_training import save_agent
        expected=sum((r['frames']-76+4)//5 for r in rows)
        collected=sum(len(r['lane']) for r in agent.rows)
        if collected!=expected:
            raise ValueError(f'Variable-length PPO transitions differ: {collected} vs {expected}')
        if trained_residual.pending.any() or not trained_residual.finished.all():
            raise ValueError('Unclosed PPO lane at reference end')
        result['training']=dict(collected=collected,expected=expected,
            reward_components={k:float(v.cpu()) for k,v in trained_residual.sums.items()},
            episode_returns=trained_residual.return_per_lane.cpu().tolist(),
            valid_reward_frames=trained_residual.valid_reward_frames.cpu().tolist(),
            initial_actor_updates=agent.actor_updates,initial_critic_updates=agent.updates)
        result['training']['ppo']=agent.finish_rollout(value_only=task.get('critic_only',False))
        result['training'].update(actor_updates=agent.actor_updates,critic_updates=agent.updates,
            finite_parameters=all(torch.isfinite(p).all().item() for module in (agent.actor,agent.value) for p in module.parameters()))
        save_agent(task['learner_output'],agent,residual_state['config'],residual_state['bounds'],
                   residual_state.get('rollouts',0)+1)
    save(output,result)
    if meta['stop_reason'] is not None or not all(r['complete'] for r in lanes):
        raise RuntimeError(f'Incomplete deployment: {meta}')
    print('[REGIONAL-EVAL] complete', task['arm'],
          {k:float(np.mean([r['post_bootstrap'][k] for r in lanes]))
           for k in ('strict_healthy_four_rate','found_four_rate')},flush=True)


def worker(request):
    """Prevent duplicate CUDA rollouts when the two coordinators catch up."""
    import fcntl
    task=json.loads(Path(request).read_text())
    output=Path(task['output'])
    with output.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if output.exists():
            saved=json.loads(output.read_text())
            if saved['checkpoint']!=task['checkpoint'] or not all(r['complete'] for r in saved['lanes']):
                raise ValueError(f'Conflicting completed worker: {output}')
            return
        _worker(request)


def summarize(output, plan):
    tables = {}
    for arm in ('oracle','baseline19k','continued_best'):
        files = list(output.glob(f'batch_*_{arm}.json'))
        if not files:
            continue
        reports = [json.loads(f.read_text()) for f in sorted(files)]
        lanes = [r for report in reports for r in report['lanes']]
        tables[arm] = dict(lanes=lanes,completed_batches=len(files),
            strict_healthy_four_rate=float(np.mean([r['post_bootstrap']['strict_healthy_four_rate'] for r in lanes])),
            found_four_rate=float(np.mean([r['post_bootstrap']['found_four_rate'] for r in lanes])),
            over100N_rollouts=sum(r['post_bootstrap']['over100N_frames']>0 for r in lanes))
    gate = {}
    if 'oracle' in tables:
        for row in tables['oracle']['lanes']:
            scope = row['post_bootstrap']
            gate.setdefault(str(row['episode_id']),[]).append(bool(row['complete'] and
                scope['strict_healthy_four_rate']>=.95 and min(scope['strict_healthy_per_finger'])>=.95
                and scope['longest_four_loss_frames']<=10 and scope['over100N_frames']==0))
        gate = {ep:all(values) for ep,values in gate.items()}
    result = dict(arms=tables,oracle_fidelity_by_episode=gate,
        all_paths_reported=True,oracle_failed_paths_not_interpreted_as_model_failure=True,
        force_response='record_and_continue; over100N invalidates deployment success',
        source='All20 development episodes only; final test untouched')
    if 'baseline19k' in tables and 'continued_best' in tables:
        for batch in plan['batches']:
            baseline_file=output/f'batch_{batch["index"]:02d}_baseline19k.json'
            candidate_file=output/f'batch_{batch["index"]:02d}_continued_best.json'
            if baseline_file.exists() and candidate_file.exists():
                if json.loads(baseline_file.read_text())['noise'] != json.loads(candidate_file.read_text())['noise']:
                    raise ValueError('Paired diffusion-noise schedule differs')
        base={(r['episode_id'],r['branch']):r for r in tables['baseline19k']['lanes']}
        candidate={(r['episode_id'],r['branch']):r for r in tables['continued_best']['lanes']}
        pairs=[]
        for key in sorted(base.keys()&candidate.keys()):
            a,b=base[key]['post_bootstrap'],candidate[key]['post_bootstrap']
            pairs.append(dict(episode_id=key[0],branch=key[1],parent=base[key]['parent'],
                oracle_valid=gate.get(str(key[0]),False),
                strict_delta=b['strict_healthy_four_rate']-a['strict_healthy_four_rate'],
                found_delta=b['found_four_rate']-a['found_four_rate']))
        result['paired']=pairs
        for metric in ('strict_delta','found_delta'):
            grouped=defaultdict(lambda:defaultdict(list))
            for r in pairs:
                grouped[r['parent']][r['episode_id']].append(r[metric])
            rng=np.random.default_rng(42)
            groups=[list(episodes.values()) for episodes in grouped.values()];samples=[]
            for _ in range(3000):
                chosen=rng.integers(0,len(groups),len(groups))
                parent_means=[]
                for i in chosen:
                    episodes=groups[i]
                    selected_episodes=rng.integers(0,len(episodes),len(episodes))
                    parent_means.append(np.mean([rng.choice(episodes[j],len(episodes[j]),replace=True).mean()
                                                 for j in selected_episodes]))
                samples.append(np.mean(parent_means))
            result[metric]=dict(mean=float(np.mean([np.mean([np.mean(e) for e in g]) for g in groups])),
                episode_weighted_mean=float(np.mean([r[metric] for r in pairs])),
                bootstrap95=np.quantile(samples,[.025,.975]).tolist(),
                bootstrap_unit='parent geometry -> child episode -> diffusion branch')
    save(output/'summary.json',result)
    return result


def evaluate_ready_candidates(output):
    """Evaluate the finished continuation on baseline batches already ready.

    This schedules additional workers only; controller, inputs, sampling, and
    scoring are identical to the main coordinator. The main run later skips
    these atomically saved results and completes the remaining batches.
    """
    import shutil
    output=output.resolve()
    plan=json.loads((output/'plan.json').read_text())
    continuation=Path(plan['continuation'])
    if not (continuation/'dataset_info.json').is_file():
        raise ValueError('Continuation is not finished; cannot freeze its best model')
    candidate=output/'continued_best_frozen.pt'
    if not candidate.exists():
        temporary=candidate.with_suffix('.copying')
        shutil.copy2(continuation/'best.pt',temporary)
        temporary.replace(candidate)
    save(output/'candidate.json',dict(checkpoint=str(candidate),sha256=sha(candidate)))
    ready=[b for b in plan['batches'] if (output/f'batch_{b["index"]:02d}_baseline19k.json').exists()
           and (output/f'batch_{b["index"]:02d}_oracle.json').exists()]
    save(output/'candidate_overlap_protocol.json',dict(
        ready_batch_ids=[b['index'] for b in ready],checkpoint=str(candidate),
        change='Scheduling only: evaluate finished candidate alongside remaining Oracle/baseline workers',
        controls_and_metrics_unchanged=True,duplicate_protection='Exclusive per-result file lock'))
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
                                   MPLCONFIGDIR='/tmp/matplotlib',WARP_CACHE_PATH='/tmp/warp',PYTHONUNBUFFERED='1')
    for batch in ready:
        result=output/f'batch_{batch["index"]:02d}_continued_best.json'
        task=dict(batch,arm='continued_best',checkpoint=str(candidate),chunk_source='dp',
                  dataset=plan['dataset'],output=str(result),mcc_config=plan['mcc_config'],
                  scheduler_scale=plan['scheduler_scale'])
        request=output/f'request_{batch["index"]:02d}_continued_best.json';save(request,task)
        save(output/'candidate_progress.json',dict(state='running',batch=batch['index'],
             ready_batches=len(ready),updated_local=datetime.now().isoformat()))
        print(f'[CANDIDATE-OVERLAP] batch={batch["index"]} lanes={len(batch["rows"])}',flush=True)
        with result.with_suffix('.log').open('a') as log:
            code=subprocess.run([sys.executable,'-u',__file__,'--worker',str(request)],
                                 cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        if code:
            save(output/'candidate_progress.json',dict(state='failed',batch=batch['index'],returncode=code))
            raise RuntimeError(f'Candidate worker failed: {result.with_suffix(".log")}')
        summarize(output,plan)
    save(output/'candidate_progress.json',dict(state='completed_ready_batches',
         batches=[b['index'] for b in ready],updated_local=datetime.now().isoformat()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',type=Path)
    parser.add_argument('--ready-candidates',type=Path)
    parser.add_argument('--dataset',type=Path)
    parser.add_argument('--baseline',type=Path)
    parser.add_argument('--continuation',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--episodes-per-batch',type=int,default=2)
    parser.add_argument('--branches',type=int,default=2)
    args=parser.parse_args()
    if args.worker:
        worker(args.worker)
        return
    if args.ready_candidates:
        evaluate_ready_candidates(args.ready_candidates)
        return
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    split=json.loads((args.dataset/'episode_split.json').read_text())
    selected=sorted((r for r in split['episodes'] if r['split']=='val'),key=lambda r:(r['frames'],r['episode_id']))
    if not selected or args.episodes_per_batch<1 or args.branches<1:
        raise ValueError('Invalid development plan')
    import torch
    checkpoint=torch.load(args.baseline,map_location='cpu',weights_only=False,mmap=True)
    # Use the verified GPU deployment controller. Match the physical actuator
    # limits and force setpoints explicitly; do not claim the serial collection
    # QP or its newer recovery-handover implementation is the same controller.
    from deploy_b_gpu_serial_parity import mcc_config
    from dataclasses import asdict,replace
    config=asdict(replace(mcc_config(),desired_force_per_finger=(10.,)*4,
                         back_contact_x_limit_m=(1.,)*4))
    batches=[]
    for index,start in enumerate(range(0,len(selected),args.episodes_per_batch)):
        rows=[dict(row,branch=branch) for row in selected[start:start+args.episodes_per_batch]
              for branch in range(args.branches)]
        batches.append(dict(index=index,rows=rows,seed=20261022+index))
    plan=dict(dataset=str(args.dataset.resolve()),baseline=str(args.baseline.resolve()),
        baseline_sha256=sha(args.baseline),continuation=str(args.continuation.resolve()),
        split='val',episodes=[r['episode_id'] for r in selected],branches=args.branches,
        scheduler_scale=np.asarray(checkpoint['normalization']['state_std'])[:16].tolist(),
        mcc_config=config,physical=dict(kp=35.,kd=.2,effort_limit_nm=.8,force_targets_n=[10.]*4),
        execution='Existing batched GPU DP scheduler/MCC; free palm recentered each1ms; no residual; one absolute actuator target held for10substeps',
        collection_difference='Serial collection uses differential contact QP and recovery handover v2; deployment uses its previously accepted direct-servo GPU controller. Oracle fidelity is measured, not assumed.',
        force_response='record_and_continue; do not stop at100N',batches=batches)
    planfile=output/'plan.json'
    if planfile.exists() and json.loads(planfile.read_text())!=plan:
        raise ValueError('Immutable evaluation plan changed')
    save(planfile,plan)
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLCONFIGDIR='/tmp/matplotlib',WARP_CACHE_PATH='/tmp/warp',PYTHONUNBUFFERED='1')
    started=time.monotonic()
    def evaluate(arm,model,source,selected_batches=None):
        for batch in batches if selected_batches is None else selected_batches:
            result=output/f'batch_{batch["index"]:02d}_{arm}.json'
            if result.exists():
                saved=json.loads(result.read_text())
                if all(r['complete'] for r in saved['lanes']):continue
                raise RuntimeError(f'Incomplete previous worker: {result}')
            task=dict(batch,arm=arm,checkpoint=str(model),chunk_source=source,
                      dataset=plan['dataset'],output=str(result),mcc_config=config,
                      scheduler_scale=plan['scheduler_scale'])
            request=output/f'request_{batch["index"]:02d}_{arm}.json';save(request,task)
            save(output/'progress.json',dict(state='running',arm=arm,batch=batch['index'],
                total_batches=len(batches),elapsed_s=time.monotonic()-started,updated_local=datetime.now().isoformat()))
            print(f'[REGIONAL-EVAL] launching {arm}, batch={batch["index"]}, lanes={len(batch["rows"])}',flush=True)
            with result.with_suffix('.log').open('w') as log:
                process=subprocess.run([sys.executable,'-u',__file__,'--worker',str(request)],
                                        cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            if process.returncode:
                save(output/'progress.json',dict(state='failed',arm=arm,batch=batch['index'],
                    log=str(result.with_suffix('.log')),returncode=process.returncode))
                raise RuntimeError(f'GPU evaluation worker failed: {result.with_suffix(".log")}')
            summarize(output,plan)
    # Alternate the teacher and DP baseline by small batch, so model evidence
    # becomes available while the continuation is still collecting updates.
    for batch in batches:
        for arm,source in [('oracle','teacher'),('baseline19k','dp')]:
            evaluate(arm,args.baseline.resolve(),source,[batch])
    while not (args.continuation/'dataset_info.json').exists():
        save(output/'progress.json',dict(state='waiting_for_continuation',elapsed_s=time.monotonic()-started))
        pid=int((args.continuation/'training.pid').read_text())
        if not Path(f'/proc/{pid}').exists():
            raise RuntimeError('Continuation exited without completion metadata')
        time.sleep(10)
    candidate=output/'continued_best_frozen.pt'
    if not candidate.exists():
        import shutil
        shutil.copy2(args.continuation/'best.pt',candidate)
    save(output/'candidate.json',dict(checkpoint=str(candidate),sha256=sha(candidate)))
    evaluate('continued_best',candidate,'dp')
    summarize(output,plan)
    save(output/'progress.json',dict(state='completed',elapsed_s=time.monotonic()-started,
         updated_local=datetime.now().isoformat()))


if __name__=='__main__':
    main()
