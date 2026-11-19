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


def wall_noise_seed(seed, frame):
    """Evaluation-only common noise, independent of skipped DP calls."""
    key=f'{int(seed)}:{int(frame)}:dp_wall_noise_v1'.encode()
    return int.from_bytes(hashlib.blake2b(key,digest_size=8).digest(),'little')%(2**63-1)


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
    if task.get('palm_reset') is not None:
        learned_reset=(task.get('fast_feedback') and task.get('residual_mode')=='frozen'
                       and task['palm_reset'].get('posture_strategy')=='cached_measured'
                       and task['palm_reset'].get('dp_history_mode')=='freeze_rebuild')
        if task.get('load_branches') or task.get('scripted_load') or task.get('causal_trace') or (task.get('residual_mode','off')!='off' and not learned_reset):
            raise ValueError('Palm reset requires standalone DP or frozen feedback with cached measured recovery/history rebuild')
        frames += int(task['palm_reset'].get('extra_frames',1500))
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
    object_ids = set()
    for row in rows:
        with h5py.File(row['source']) as h:
            initial_target.append(h['q_cmd'][0, 0])
            object_ids.add(str(h.attrs['object_id']))
    if len(object_ids) != 1:
        raise ValueError('Batch must use one collision model; group evaluation by object_id')
    object_id = next(iter(object_ids))
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
    if task.get('pose_trace',False):
        tensors.update(measured_palm_pose_world=torch.zeros((frames,batch,7),device=device),
                       measured_object_pose_world=torch.zeros((frames,batch,7),device=device))
    substep_count = torch.zeros(frames, dtype=torch.long, device=device)
    noise = []
    mcc_snapshots = []
    initialization_trace=[]
    config = task['mcc_config']
    residual_mode=task.get('residual_mode','off')
    training=residual_mode=='train'
    residual_state=None
    agent=None
    dppo=None
    fast_feedback=task.get('fast_feedback',False)
    if task.get('self_contact_trace',False):
        tensors['self_force']=torch.zeros((frames,batch),device=device)
    regional=False
    shadow_errors=dict(action=0.,nominal=0.,command=0.,observation=0.)
    if fast_feedback:
        if task.get('dppo'):
            raise ValueError('Fast feedback freezes DP/MCC')
        if residual_mode not in ('zero','frozen','train'):raise ValueError('Feedback mode')
        from fast_history_feedback import FastHistoryFeedback,restore_feedback
        residual_state=torch.load(task['residual_checkpoint'],map_location='cpu',weights_only=False)
        agent=restore_feedback(residual_state,training)
        regional=True
        tensors.update(residual_offset=torch.zeros((frames,batch,16),device=device),
                       residual_target=torch.zeros((frames,batch,16),device=device),
                       feedback_load=torch.ones((frames,batch,4),device=device),
                       self_force=torch.zeros((frames,batch),device=device))
    elif residual_mode!='off':
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
    reset_report = None
    optimizer_teacher_report = None
    single_finger_report = None

    class Chain(BatchedBControlChain):
        def _orchestrate(self, **kwargs):
            if task.get('causal_trace', False) or task.get('recovery_search_branches'):
                self.pre_recovery_nominal = kwargs['nominal'].clone()
            result = super()._orchestrate(**kwargs)
            if hasattr(self, 'search_branch_experiment'):
                result = self.search_branch_experiment.filter_command(kwargs['nominal'], result)
            if hasattr(getattr(self, 'command_supervisor', None), 'filter_recovery'):
                result = self.command_supervisor.filter_recovery(kwargs['nominal'], result)
            return result
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
            elif controller == 'normal_constrained_hold_v1':
                from normal_constrained_hold import NormalConstrainedHoldMCC
                self.mcc=NormalConstrainedHoldMCC(self.batch,self.device,FullHandMCCFingerConfig(**config))
            elif controller == 'contact_intent_v1':
                from contact_intent_mcc import ContactIntentMCC, HandoverConfig
                self.mcc=ContactIntentMCC(self.batch,self.device,FullHandMCCFingerConfig(**config),
                    HandoverConfig(**task.get('handover_config',{})))
            elif controller in ('load_authority_v2','load_authority_v3','unified_command_v1'):
                from load_authority_mcc import LoadAuthorityMCC, LoadAuthorityConfig
                if controller=='unified_command_v1':
                    from unified_command_mcc import UnifiedCommandMCC, UnifiedCommandConfig
                    self.mcc=UnifiedCommandMCC(self.batch,self.device,FullHandMCCFingerConfig(**config),
                        LoadAuthorityConfig(**task.get('load_authority_config',{})),
                        unified=UnifiedCommandConfig(**task.get('unified_command_config',{})))
                else:
                    self.mcc=LoadAuthorityMCC(self.batch,self.device,FullHandMCCFingerConfig(**config),
                        LoadAuthorityConfig(**task.get('load_authority_config',{})))
                self.authority_unified_mode=controller=='unified_command_v1'
                self.authority_matched_mode=controller!='load_authority_v2'
                self.authority_guard_all=bool(task.get('matched_tip_guard_all',False))
                self.authority_palm_compensation=bool(task.get('palm_motion_compensation',False))
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
                    frame_seed=None
                    if task.get('dp_noise_clock')=='wall_frame_v1':
                        frame_seed=wall_noise_seed(task['seed'],self.control_frame)
                        kwargs['generator'].manual_seed(frame_seed)
                    before = hashlib.sha256(kwargs['generator'].get_state().numpy().tobytes()).hexdigest()
                    result = original(*args, **kwargs)
                    if fast_feedback:self.feedback_prediction=result.detach().clone()
                    after = hashlib.sha256(kwargs['generator'].get_state().numpy().tobytes()).hexdigest()
                    noise.append(dict(frame=self.control_frame, before=before, after=after,frame_seed=frame_seed))
                    return result
                self.runtime.policy.diffusion.conditional_sample = sample
            if task.get('single_finger_policy') is not None:
                if task['chunk_source'] != 'dp' or agent is not None or dppo is not None or task.get('optimizer_teacher') or (task.get('palm_reset') and task.get('local_recovery') is None):
                    raise ValueError('Single-finger pilot requires standalone DP chunks and frozen MCC')
                from single_finger_dp import SingleFingerRuntime
                if task['single_finger_policy'].get('dynamic_range', False):
                    from range_conditioned_finger_dp import RangeFingerRuntime
                    self.single_finger_runtime = RangeFingerRuntime(task['single_finger_policy'], device, task['seed'])
                elif task['single_finger_policy'].get('mode', 'all') == 'all':
                    self.single_finger_runtime = SingleFingerRuntime(task['single_finger_policy'], device, task['seed'])
                else:
                    from hybrid_finger_dp import HybridFingerRuntime, full_hand_chunk
                    self.single_finger_runtime = HybridFingerRuntime(
                        task['single_finger_policy'], device, task['seed'],
                        lambda frame, q, history: full_hand_chunk(self, frame, q, history))
                self.chunk_provider = self.single_finger_runtime
            if dppo is not None:
                dppo.attach(self)
            if task.get('optimizer_teacher') is not None:
                if agent is not None or dppo is not None or task.get('palm_reset') or task.get('snapshot_branches') or task['chunk_source']!='dp':
                    raise ValueError('Optimizer teacher is a standalone non-learning DP diagnostic')
                from optimizer_teacher_chunk import OptimizerTeacherChunk
                self.optimizer_teacher=OptimizerTeacherChunk(self,task['optimizer_teacher'],output)
                self.chunk_provider=self.optimizer_teacher
            self.shadow=None
            if dppo is not None and dppo.probe:
                self.shadow=BatchedBControlChain(*args,**kwargs)
                self.shadow.runtime.policy.diffusion.num_inference_steps=10
                self.shadow.scheduler.scale=self.scheduler.scale.clone()
                self.shadow.chunk_provider=lambda frame,q,history:dppo.reference_chunk
            if agent is not None:
                # Reuse the original actor preprocessing, history and limiter.
                # The stored MCC profile is intentionally NOT transplanted.
                residual_class=FastHistoryFeedback if fast_feedback else (RegionalResidual if regional else RL0Residual)
                extra=dict(lengths=[r['frames'] for r in rows]) if regional else {}
                if fast_feedback and task.get('palm_reset') is not None:
                    extra=dict(lengths=[frames]*batch)
                self.residual_policy=residual_class(self,agent,observer,residual_state['bounds'],
                    residual_state['config'],zero=residual_mode=='zero' or task.get('zero_action',False),
                    evaluate=not training,**extra)
                if residual_mode=='zero' and fast_feedback:
                    import copy
                    self.shadow=BatchedBControlChain(*args,**kwargs)
                    self.shadow.mcc=copy.deepcopy(self.mcc)
                    self.shadow.runtime.policy.diffusion.num_inference_steps=10
                    self.shadow.scheduler.scale=self.scheduler.scale.clone()
                    self.shadow.chunk_provider=lambda frame,q,history: self.feedback_prediction*self.action_std+self.action_mean+q[:,None]
                if residual_mode=='zero' and not fast_feedback:
                    self.shadow=BatchedBControlChain(*args,**kwargs)
                    self.shadow.runtime.policy.diffusion.num_inference_steps=10
                    self.shadow.scheduler.scale=self.scheduler.scale.clone()
                    self.shadow.chunk_provider=lambda frame,q,history:self.residual_policy.latest_chunk
            if task.get('scripted_intervention') is not None:
                if agent is not None or dppo is not None:
                    raise ValueError('Scripted diagnostics cannot be combined with learned residual/DPPO')
                from local_intervention_probe import ScriptedOffset
                self.residual_policy=ScriptedOffset(self,task['scripted_intervention'])
            if task.get('scripted_unload') is not None:
                if agent is not None or dppo is not None or task.get('scripted_intervention') or task.get('snapshot_branches'):
                    raise ValueError('Scripted unload is a standalone diagnostic')
                from scripted_contact_intent import ScriptedUnload
                self.residual_policy=ScriptedUnload(self,task['scripted_unload'])
            if task.get('scripted_load') is not None:
                if (agent is not None or dppo is not None or task.get('scripted_intervention')
                        or task.get('scripted_unload') or task.get('snapshot_branches')):
                    raise ValueError('Scripted load authority is a standalone diagnostic')
                from scripted_load_authority import ScriptedLoadAuthority
                self.residual_policy=ScriptedLoadAuthority(self,task['scripted_load'])
            if task.get('load_branches'):
                if (not task.get('scripted_load') or task.get('snapshot_branches')
                        or task.get('causal_trace') or agent is not None or dppo is not None):
                    raise ValueError('Load branching requires standalone scripted_load without DP trace')
                from load_authority_branches import LoadAuthorityBranches
                self.load_branch_experiment=LoadAuthorityBranches(self,task['load_branches'],output,tensors,substep_count)
                self.before_diagnostic_frame=self.load_branch_experiment.before
                self.after_diagnostic_frame=self.load_branch_experiment.after
            if task.get('palm_reset') is not None:
                if controller!='unified_command_v1':
                    raise ValueError('Palm reset requires unified final-command arbitration')
                from palm_posture_reset import PalmPostureReset
                if task.get('local_recovery') is not None:
                    from local_contact_recovery import LocalContactRecovery
                    if task.get('local_contact_search') is not None:
                        from local_contact_search import LocalContactSearch
                        self.command_supervisor=LocalContactSearch(self,[r['frames'] for r in rows],task['palm_reset'],task['local_recovery'],np.stack(initial_target),task['local_contact_search'])
                    else:
                        self.command_supervisor=LocalContactRecovery(self,[r['frames'] for r in rows],task['palm_reset'],task['local_recovery'],np.stack(initial_target))
                elif task.get('thumb_workspace') is not None:
                    if task.get('thumb_tangent_follow') is not None:
                        from thumb_tangent_follow import ThumbTangentFollow
                        self.command_supervisor=ThumbTangentFollow(self,[r['frames'] for r in rows],task['palm_reset'],task['thumb_workspace'],task['thumb_tangent_follow'])
                    else:
                        from thumb_workspace_supervisor import ThumbWorkspaceSupervisor
                        self.command_supervisor=ThumbWorkspaceSupervisor(self,[r['frames'] for r in rows],task['palm_reset'],task['thumb_workspace'])
                else:
                    self.command_supervisor=PalmPostureReset(self,[r['frames'] for r in rows],task['palm_reset'])
                self.select_execution_reference=self.command_supervisor.select_reference
                self.after_diagnostic_frame=self.command_supervisor.after_frame
            if task.get('mcc_snapshot_frames'):
                original_update=self.mcc.update
                def snapshot_update(**inputs):
                    frame=self.context['frame']
                    record=(not self.context['settling'] and frame in task['mcc_snapshot_frames'])
                    if record:
                        # Keep only lane0, before any in-place controller update.
                        def cpu(value):
                            return value[:1].detach().cpu().clone() if isinstance(value,torch.Tensor) and value.ndim and value.shape[0]==batch else value
                        state={k:cpu(v) for k,v in vars(self.mcc).items() if isinstance(v,torch.Tensor) or v is None}
                        sample=dict(frame=frame,state=state,inputs={k:cpu(v) for k,v in inputs.items()})
                    result=original_update(**inputs)
                    if record:
                        sample['command']=result[0][:1].detach().cpu().clone()
                        mcc_snapshots.append(sample)
                    return result
                self.mcc.update=snapshot_update
            if task.get('snapshot_branches'):
                if not task.get('scripted_intervention') or task.get('causal_trace') or agent is not None or dppo is not None:
                    raise ValueError('Snapshot branching is a scripted diagnostic only; disable full DP trace')
                from local_snapshot_branch import BranchExperiment
                self.branch_experiment=BranchExperiment(self,task['snapshot_branches'],output,tensors,substep_count)
                self.before_diagnostic_frame=self.branch_experiment.before
                self.after_diagnostic_frame=self.branch_experiment.after
            if task.get('recovery_search_branches'):
                if (task.get('causal_trace') or agent is not None or dppo is not None
                    or any(task.get(k) for k in ('load_branches', 'snapshot_branches',
                        'optimizer_teacher', 'palm_reset', 'fast_feedback'))
                    or residual_mode != 'off'):
                    raise ValueError('Search ablation requires frozen standalone DP/MCC, no full DP trace')
                from recovery_search_branches import RecoverySearchBranches
                self.search_branch_experiment = RecoverySearchBranches(
                    self, task['recovery_search_branches'], output, tensors, substep_count)
                self.before_diagnostic_frame = self.search_branch_experiment.before
                self.after_diagnostic_frame = self.search_branch_experiment.after
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
            if task.get('mobility_probe') and not getattr(self,'mobility_model_saved',False):
                mujoco.mj_saveModel(env.sim.mj_model,str(output.with_suffix('.physics.mjb')),None)
                self.mobility_model_saved=True
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
            if fast_feedback or task.get('self_contact_trace',False):
                if not hasattr(self,'feedback_self_diagnostic'):
                    from feedback_self_contact import SelfContactDiagnostic
                    self.feedback_self_diagnostic=SelfContactDiagnostic(env,a['hand'].numpy())
                own_force=self.feedback_self_diagnostic.read()
                tensors['self_force'][frame]=torch.maximum(tensors['self_force'][frame],own_force)
                if fast_feedback:
                    self.residual_policy.self_force=tensors['self_force'][frame]
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
            if task.get('pose_trace',False) and self.tick_substeps==9:
                # Offline geometry diagnostics only; never fed back into the
                # actor or actuator. Record actual poses at the final substep.
                tensors['measured_palm_pose_world'][frame]=env.scene['robot'].data.root_link_pose_w
                tensors['measured_object_pose_world'][frame]=env.scene['target'].data.root_link_pose_w
            self.tick_substeps+=1
            if agent is not None and regional and self.tick_substeps==10:
                self.residual_policy.post_physics(frame,env.scene['robot'].data.joint_pos[:, -16:],
                    self.context['palm_pose'],tensors['force'][frame])
            if dppo is not None and self.tick_substeps==10:
                dppo.post_physics(self,frame,env.scene['robot'].data.joint_pos[:, -16:],
                    self.context['palm_pose'],tensors['force'][frame])
        def step(self, **kwargs):
            self.tick_substeps=0
            if hasattr(self.mcc,'set_measured_contact'):
                self.mcc.trace_authority=bool(task.get('authority_trace',False))
                if hasattr(self.mcc,'unified_enabled'):
                    self.mcc.unified_enabled=(self.authority_unified_mode and not self.context['settling'] and self.context['frame']>=76)
                x=observer.latest
                self.mcc.use_matched_contact=self.authority_matched_mode
                self.mcc.matched_guard_all=(self.authority_guard_all and not self.context['settling'] and self.context['frame']>=76)
                self.mcc.compensate_palm_motion=(self.authority_palm_compensation and not self.context['settling'] and self.context['frame']>=76)
                self.mcc.set_measured_contact(x['physical_normal_found'],x['physical_contact_point'],x['physical_normal_inward'])
            if task.get('palm_reset') is not None and not self.context['settling'] and self.context['frame']>=76:
                if self.context['frame']==task.get('palm_reset_test_request_frame'):
                    self.command_supervisor.request_reset(torch.ones(self.batch,device=device,dtype=torch.bool))
                self.command_supervisor.observe(kwargs['q'],kwargs['palm_pose_world'],observer.latest)
            if task.get('scripted_unload') is not None or task.get('scripted_load') is not None:
                self.residual_policy.inputs=kwargs
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
                if fast_feedback:
                    for key in ['unified_enabled','use_matched_contact','matched_guard_all','compensate_palm_motion','trace_authority']:
                        setattr(self.shadow.mcc,key,getattr(self.mcc,key))
                    self.shadow.mcc.set_measured_contact(*self.mcc.measured_contact)
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
                    if fast_feedback:tensors['feedback_load'][frame]=self.residual_policy.load_request
                    if fast_feedback and task.get('feedback_debug_trace',False):
                        for key,value in dict(feedback_raw_action=self.residual_policy.last_raw_action,
                            feedback_executed_action=self.residual_policy.last_executed_action,
                            feedback_load_applied=self.mcc.regulation_load_scale,
                            feedback_actor_observation=self.residual_policy.latest_actor_observation).items():
                            if value is None:continue  # actor starts only after bootstrap
                            if key not in tensors:tensors[key]=torch.zeros((frames,*value.shape),device=device,dtype=value.dtype)
                            tensors[key][frame]=value
                tensors['nominal'][frame] = diagnostic['nominal_q']
                tensors['command'][frame] = diagnostic['command_q']
                if any(task.get(k) is not None for k in ('scripted_intervention','scripted_unload','scripted_load')):
                    for key,value in dict(scripted_offset=self.residual_policy.offset,
                                          scheduler_nominal=self.previous_nominal,
                                          mcc_hold=self.mcc.hold,
                                          mcc_force_offset=self.mcc.force_offset).items():
                        if key not in tensors:tensors[key]=torch.zeros((frames,*value.shape),device=device,dtype=value.dtype)
                        tensors[key][frame]=value
                    for key,value in dict(intent_recovery_offset=self.runtime_recovery_offset,
                                          intent_recovery_active=self.recovery_active).items():
                        if key not in tensors:tensors[key]=torch.zeros((frames,*value.shape),device=device,dtype=value.dtype)
                        tensors[key][frame]=value
                if task.get('causal_trace', False) or task.get('recovery_search_branches'):
                    extra = dict(pre_recovery_nominal=getattr(self, 'pre_recovery_nominal', diagnostic['nominal_q']),
                                 recovery_offset=self.runtime_recovery_offset,
                                 recovery_active=self.recovery_active,
                                 force_offset=self.mcc.force_offset)
                    for key, value in extra.items():
                        if key not in tensors:
                            tensors[key]=torch.zeros((frames,*value.shape),device=device,dtype=value.dtype)
                        tensors[key][frame]=value
                if task.get('contact_dynamics_trace',False):
                    for key,name in [('input_tip_force_world','contact_force_world'),
                                     ('input_tip_found','contact_found'),
                                     ('input_tip_points_world','contact_points_world'),
                                     ('input_palm_pose_world','palm_pose_world')]:
                        value=kwargs[name]
                        if key not in tensors:tensors[key]=torch.zeros((frames,*value.shape),device=device,dtype=value.dtype)
                        tensors[key][frame]=value
                    for key,value in dict(input_netforce_inward=kwargs['contact_normals_inward_world'],
                        physical_tip_inward=observer.latest['physical_normal_inward'],
                        physical_tip_points=observer.latest['physical_contact_point'],
                        input_q_live=kwargs['q'],input_palm_twist_world=kwargs['palm_twist_world'],
                        physical_tip_found=observer.latest['physical_normal_found']).items():
                        if key not in tensors:tensors[key]=torch.zeros((frames,*value.shape),device=device,dtype=value.dtype)
                        tensors[key][frame]=value
                for key,value in diagnostic.items():
                    if key.startswith(('admittance_','hold_','intent_')):
                        if key not in tensors:
                            tensors[key]=torch.zeros((frames,*value.shape),device=device,dtype=value.dtype)
                        tensors[key][frame]=value
                if task.get('palm_reset') is not None:
                    for key,value in self.command_supervisor.diagnostics().items():
                        if key not in tensors:
                            tensors[key]=torch.zeros((frames,*value.shape),device=device,dtype=value.dtype)
                        tensors[key][frame]=value
                if frame % 500 == 0:
                    print(f'[REGIONAL-EVAL] {task["arm"]} frame={frame}/{frames} batch={batch}',flush=True)
            return action, diagnostic
        def close(self):
            nonlocal trained_residual, reset_report, optimizer_teacher_report, single_finger_report
            if hasattr(self,'single_finger_runtime'):
                single_finger_report=self.single_finger_runtime.report()
                if hasattr(self.single_finger_runtime,'close'):
                    self.single_finger_runtime.close()
            if hasattr(self,'optimizer_teacher'):
                optimizer_teacher_report=self.optimizer_teacher.finish()
            if task.get('palm_reset') is not None:
                reset_report=self.command_supervisor.report()
            if agent is not None:trained_residual=self.residual_policy
            super().close()

    deployment.MODEL = task['checkpoint']
    trained_residual=None
    restore_targets = install_held_relative_targets()
    try:
        meta = deployment.run(batch, rows[0]['episode_id'], frames, device, task['seed'],
            settle_steps=100, max_safe_force_n=0., per_env_safety=False,
            episode_data=data, chain_factory=Chain, contact_observer=observer,
            initial_joint_target=np.stack(initial_target), object_id=object_id,
            dp_trace_h5=output.with_suffix('.dp_trace.h5') if task.get('causal_trace',False) else None)
    finally:
        restore_targets()
    values = {k: v.cpu().numpy() for k,v in tensors.items()}
    if mcc_snapshots:
        torch.save(dict(config=config,samples=mcc_snapshots),output.with_suffix('.mcc_snapshots.pt'))
    if initialization_trace:
        np.savez_compressed(output.with_suffix('.initialization.npz'),
            **{k:torch.stack([r[k] for r in initialization_trace]).cpu().numpy() for k in initialization_trace[0]})
    counts = substep_count.cpu().numpy()
    lanes = []
    for lane, row in enumerate(rows):
        length = row['frames']
        if task.get('palm_reset') is not None:
            done=np.flatnonzero((values['reset_reference_frame'][:,lane]==row['frames']-1)
                               & np.isin(values['reset_phase'][:,lane],[0,8]))
            observed_length=int(values['observed'][:,lane].sum())
            length=min(int(done[0]+1),observed_length) if len(done) else observed_length
        target_error=float(values['target_error'][:length,lane].max())
        complete = bool(values['observed'][:length,lane].all() and np.all(counts[:length]==10)
                        and target_error<1e-6)
        if task.get('palm_reset') is not None:
            complete=complete and reset_report['final_phase'][lane]=='done'
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
    if reset_report is not None:
        result['palm_reset']=reset_report
        result['reference_progress_semantics']='Rows specify native spatial reference length. Trace index is wall-control time including all resets; reset_reference_frame records progress. No contact loss is masked.'
    result['mcc_controller']=task.get('mcc_controller','legacy')
    if single_finger_report is not None:
        result['single_finger_policy']=single_finger_report
    if optimizer_teacher_report is not None:
        result['optimizer_teacher']=optimizer_teacher_report
    if task.get('scripted_intervention') is not None:
        result['scripted_intervention']=task['scripted_intervention']
    if task.get('scripted_unload') is not None:
        result['scripted_unload']=task['scripted_unload']
    if result['mcc_controller']=='contact_intent_v1':
        from dataclasses import asdict
        from contact_intent_mcc import HandoverConfig
        result['handover_config']=asdict(HandoverConfig(**task.get('handover_config',{})))
    if result['mcc_controller'] in ('load_authority_v2','load_authority_v3','unified_command_v1'):
        from dataclasses import asdict
        from load_authority_mcc import LoadAuthorityConfig
        result['load_authority_config']=asdict(LoadAuthorityConfig(**task.get('load_authority_config',{})))
        result['scripted_load']=task.get('scripted_load')
        result['matched_tip_guard_all']=task.get('matched_tip_guard_all',False)
        result['palm_motion_compensation']=task.get('palm_motion_compensation',False)
        if result['mcc_controller']=='unified_command_v1':
            from unified_command_mcc import UnifiedCommandConfig
            result['unified_command_config']=asdict(UnifiedCommandConfig(**task.get('unified_command_config',{})))
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
        if fast_feedback:
            result['residual']['kind']='fast_history_feedback_v1'
            freeze=trained_residual.freeze_status()
            if not all(freeze[k] for k in ['explicit_no_grad','no_parameter_mutation','no_dp_gradients']):
                raise AssertionError('Frozen DP was modified during feedback execution')
            result['residual']['freeze_status']=freeze
            result['residual']['reward_component_absolute_sums']={k:float(v.cpu()) for k,v in trained_residual.component_abs.items()}
            result['residual']['reward_component_sums']={k:float(v.cpu()) for k,v in trained_residual.sums.items()}
            result['residual']['episode_returns']=trained_residual.return_per_lane.cpu().tolist()
            result['residual']['history_lags_control_frames']=residual_state['config']['history_lags']
            result['residual']['self_force_peak_n']=float(values['self_force'].max())
        if not training and (agent.actor_updates or agent.updates or agent.rows):
            raise AssertionError('Evaluation changed the residual learner')
    if training:
        from regional_residual_training import save_agent
        period=residual_state['config']['decision_frames'] if fast_feedback else 5
        expected=sum((r['frames']-76+period-1)//period for r in rows)
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
        if fast_feedback:
            from fast_history_feedback import save_feedback
            save_agent=save_feedback
        save_agent(task['learner_output'],agent,residual_state['config'],residual_state['bounds'],
                   residual_state.get('rollouts',0)+1)
    save(output,result)
    # A reset timeout is an evaluated failure outcome, not a crashed worker.
    # The budget includes optional overhead, so completing the reference early
    # is also an expected supervisor termination. Never label it PASSED here.
    expected_supervisor_end=(reset_report is not None and meta['stop_reason']=='execution_supervisor_finished')
    if ((meta['stop_reason'] is not None and not expected_supervisor_end)
            or (reset_report is None and not all(r['complete'] for r in lanes))):
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
