"""Recorded-command fidelity and matched Oracle/DP tests on actual long palms.

Raw replay restores source joint position/velocity/target and starts at sample1.
Oracle and DP share initialization, timing, scheduler, MCC and physical safety.
Neither mode resets or switches hand trajectories during the task.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path
import h5py
import numpy as np
from invert_trajectories import _relative_pose, _backward_palm_twist
from long_palm_rollout import DP, PROFILE, write
from palm_planner_features import future_palm_delta_pose_palm

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'mcc_finger_compliance_control/data/long_palm_oracle_geometry_20261013'


def source_row(eid):
    report = json.loads((SOURCE / 'oracle_analysis.json').read_text())
    row = next(r for r in report['preferred_observations'] if r['trajectory_id'] == eid)
    if not row['whole_tip_internal_contact_gate']:
        raise ValueError('Use the corrected whole-tip Oracle only')
    return row


def load_reference(eid, frames):
    row = source_row(eid)
    if row['status'] != 'reference_completed':
        raise ValueError(f'{eid}: source did not complete; never pad its stopped tail')
    file = Path(row['probe']) / 'collected.h5'
    with h5py.File(file) as h:
        if not h.attrs['split_fingertip_collision'] or h.attrs['collection_safety_abort_json'] != 'null':
            raise ValueError('Expected complete split-tip physical collection')
        n = min(frames, len(h['q_hand']))
        q = np.asarray(h['q_hand'][:n, 0], dtype=np.float32)
        cmd = np.asarray(h['q_cmd'][:n, 0], dtype=np.float32)
        velocity = np.asarray(h['qvel'][:n, 0, -16:], dtype=np.float32)
        palm = _relative_pose(np.asarray(h['object_pose_world'][:n, 0]),
                              np.asarray(h['palm_pose_world'][:n, 0]))
        dt = float(h.attrs['control_dt'])
        strict = ((np.asarray(h['fingertip_collision_found'][:n, 0]) > .5)
                  & (np.asarray(h['true_fingertip_contact_valid'][:n, 0]) > .5)
                  & (np.linalg.norm(h['fingertip_force_world'][:n, 0], axis=-1) >= .05))
        found = np.asarray(h['fingertip_body_collision_found'][:n, 0]) > .5
        force = np.asarray(h['contact_safety_peak_force_n'][:n, 0])
    if abs(dt - .01) > 1e-9 or n != frames:
        raise ValueError('Expected the exact requested100Hz physical samples')
    episode = np.zeros((n, 1), np.int64)
    twist = _backward_palm_twist(palm[:, None], episode, dt)[:, 0]
    return dict(q=q, command=cmd, velocity=velocity, palm=palm, twist=twist,
                source_strict=strict, source_found=found, source_force=force,
                meta=dict(trajectory_id=eid, source=str(file), frames=n,
                          source_strict_four=float(strict.all(-1).mean()),
                          source_found_four=float(found.all(-1).mean()),
                          source_peak_force_n=float(force.max())))


def pack(rows, raw):
    hold = 0 if raw else 100
    def extend(value):
        if raw:
            return value[1:]
        return np.concatenate((np.repeat(value[:1], hold, axis=0), value))
    data = dict(q_hand=[], palm_pose_object=[], palm_twist_object=[],
                planner_palm_delta_pose_palm_commanded=[])
    teachers, commands = [], []
    for row in rows:
        pose = extend(row['palm'])
        twist = (row['twist'][1:] if raw else
                 np.concatenate((np.zeros((hold, 6), np.float32), row['twist'])))
        teacher = extend(row['q'])
        data['q_hand'].append(np.repeat(row['q'][:1], len(pose), axis=0))
        data['palm_pose_object'].append(pose)
        data['palm_twist_object'].append(twist)
        data['planner_palm_delta_pose_palm_commanded'].append(
            future_palm_delta_pose_palm(pose, np.zeros(len(pose), np.int64), step_frames=20))
        teachers.append(teacher); commands.append(extend(row['command']))
    for name in ('q_hand', 'palm_pose_object', 'palm_twist_object'):
        data[name] = np.stack(data[name], axis=1)
    data['planner_palm_delta_pose_palm_commanded'] = np.stack(
        data['planner_palm_delta_pose_palm_commanded'], axis=2)
    return data, np.stack(teachers, 1), np.stack(commands, 1), hold


def longest_loss(mask):
    longest = streak = 0
    first = None
    for i, ok in enumerate(mask):
        streak = 0 if ok else streak + 1
        longest = max(longest, streak)
        if streak == 6 and first is None:
            first = i - 5
    return longest, first


def run(args):
    import torch
    import deploy_b_gpu_serial_parity as deployment
    from batched_b_control_chain import BatchedBControlChain
    from true_fingertip_contact import ContactSlotObserver
    from mcc_force_profiles import configured_mcc, attach_force_profile
    from cuda_graph_hand_fk import CUDAGraphHandFK
    from cuda_graph_control_kernels import ControlGraphKernels
    from async_dp_planner import AsyncDPPlanner, DPPlan
    from mjlab.utils.lab_api.math import quat_apply, quat_error_magnitude
    raw = args.mode == 'raw'
    rows = [load_reference(eid, args.frames) for eid in args.ids]
    data, teacher, commands, hold = pack(rows, raw)
    if args.root_velocity == 'zero':
        data['palm_twist_object'] = np.zeros_like(data['palm_twist_object'])
    device = 'cuda:0'; batch = len(rows); total = len(teacher)
    class Observer(ContactSlotObserver):
        def configure(self, cfg):
            if args.materials == 'collection':
                from mjlab.tasks.leaphand.leaphand_mcc_finger_env_cfg import _load_mcc_leaphand_spec
                source_spec = _load_mcc_leaphand_spec()
                properties = {g.name: dict(friction=np.asarray(g.friction).copy(),
                    solref=np.asarray(g.solref).copy(), solimp=np.asarray(g.solimp).copy(),
                    priority=g.priority, solmix=g.solmix, margin=g.margin, gap=g.gap)
                    for g in source_spec.geoms}
                original = cfg.scene.entities['robot'].spec_fn
                def matched_spec():
                    spec = original()
                    for geom in spec.geoms:
                        if geom.name not in properties:
                            continue
                        values = properties[geom.name]
                        for name in ('friction', 'solref', 'solimp'):
                            getattr(geom, name)[:] = values[name]
                        for name in ('priority', 'solmix', 'margin', 'gap'):
                            setattr(geom, name, values[name])
                    return spec
                cfg.scene.entities['robot'].spec_fn = matched_spec
            super().configure(cfg)
    observer = Observer(controller_scope='pad', capture_collision=True)
    profile = json.loads(PROFILE.read_text())
    if args.projection_live_step is not None:
        profile['name'] += '_bounded_projection'
        profile['projection']['max_live_joint_step_rad'] = args.projection_live_step
    if args.command_error_limit is not None:
        profile['name'] += '_bounded_command_error'
        profile['projection']['max_live_joint_step_rad'] = args.command_error_limit
        profile['projection']['bound_all_commands'] = True
    teacher_t = torch.as_tensor(teacher, device=device)
    command_t = torch.as_tensor(commands, device=device)
    arrays = {name: torch.zeros((total, batch, *shape), dtype=dtype, device=device)
              for name, shape, dtype in [
                  ('q', (16,), torch.float32), ('q_pre', (16,), torch.float32),
                  ('nominal', (16,), torch.float32), ('command', (16,), torch.float32),
                  ('scheduler', (16,), torch.float32), ('recovery', (16,), torch.float32),
                  ('drop', (), torch.long),
                  ('projection_engaged', (4,), torch.bool),
                  ('projection_infeasible', (4,), torch.bool),
                  ('bounded_command_changed', (4,), torch.bool),
                  ('strict', (4,), torch.bool), ('found', (4,), torch.bool),
                  ('seen', (), torch.bool), ('force', (), torch.float32),
                  ('pre_force', (), torch.float32), ('substep_force', (), torch.float32),
                  ('pad_force', (4, 3), torch.float32), ('connector_force', (4, 3), torch.float32)]}
    record_base = bool(getattr(args, 'record_base', False))
    support_kind = getattr(args, 'palm_support', 'free')
    if support_kind == 'weld' and not raw:
        raise ValueError('Finite-weld actor/MCC measured-base feedback has not yet been accepted')
    if record_base:
        arrays.update({name: torch.zeros((total, batch, *shape), device=device)
                       for name, shape in [
                           ('base_substep_pose', (7,)), ('base_substep_velocity', (6,)),
                           ('base_error_m', ()), ('base_error_rad', ()),
                           ('base_linear_speed_max', ()), ('base_angular_speed_max', ())]})
    planner_report = None
    clipped = 0

    class Planner(AsyncDPPlanner):
        def _sample(self, snapshot):
            if args.mode == 'dp':
                return super()._sample(snapshot)
            before = hashlib.sha256(self.chain.runtime.generator.get_state().numpy().tobytes()).hexdigest()
            indices = (snapshot['frame'] + torch.arange(1, self.chain.runtime.pred_horizon+1,
                       device=device) * self.chain.runtime.stride).clamp_max(total-1)
            chunk = teacher_t[indices].permute(1, 0, 2).contiguous()
            prediction = (chunk - snapshot['q'][:, None] - self.chain.action_mean) / self.chain.action_std
            event = torch.cuda.Event(); event.record(); event.synchronize()
            return DPPlan(snapshot['frame'], snapshot['history'], snapshot['normalized'],
                          snapshot['q'], None, prediction, chunk, 0., None, before, before, event,
                          snapshot['frame'], self.chain.control_frame)

    class Chain(BatchedBControlChain):
        post_step_safety = True
        def __init__(self, *values, **kwargs):
            kwargs['mcc_config'] = configured_mcc(replace(kwargs['mcc_config'],
                back_contact_x_limit_m=(1.,)*4), profile)
            super().__init__(*values, **kwargs)
            self.runtime.policy.diffusion.num_inference_steps = 20
            self.substep_peak = arrays['substep_force']
            if not raw:
                graph = CUDAGraphHandFK(self.mcc.fk, batch, self.device, frame_cache=True)
                self.mcc.fk = graph; self.direction.fk = graph
                attach_force_profile(self, observer, profile)
                if args.delay_frames:
                    self.async_planner = Planner(self, minimum_latency_frames=args.delay_frames,
                        buffer_frames=30, max_age_frames=20)
                elif args.mode == 'oracle':
                    def provider(frame, q, history):
                        indices = (frame + torch.arange(1, self.runtime.pred_horizon+1,
                                   device=device)*self.runtime.stride).clamp_max(total-1)
                        return teacher_t[indices].permute(1, 0, 2).contiguous()
                    self.chunk_provider = provider
        def prime(self, *values, **kwargs):
            super().prime(*values, **kwargs)
            if not raw:
                self.control_graphs = ControlGraphKernels(self)
        def set_training_context(self, **ctx):
            self.context = ctx
        def record_substep_force(self):
            if self.context['settling']:
                return
            f = self.context['frame']
            self.substep_peak[f] = torch.maximum(self.substep_peak[f], observer.safety_peak_force_n)
        def record_base_substep(self, env, target):
            if not record_base or self.context['settling'] or target is None:
                return
            f = self.context['frame']
            robot = env.scene['robot']
            if robot.is_fixed_base:
                pose = robot.data.root_link_pose_w
                velocity = robot.data.root_link_vel_w
            else:
                # qpos/qvel are post-integration; cached xpos/cvel can still
                # describe the preceding forward pass at this sampling point.
                pose = env.sim.data.qpos[:, robot.data.indexing.free_joint_q_adr]
                vel = env.sim.data.qvel[:, robot.data.indexing.free_joint_v_adr]
                velocity = torch.cat((vel[:, :3], quat_apply(pose[:, 3:], vel[:, 3:])), -1)
            arrays['base_substep_pose'][f] = pose
            arrays['base_substep_velocity'][f] = velocity
            for name, value in [
                    ('base_error_m', (pose[:, :3]-target[:, :3]).norm(dim=-1)),
                    ('base_error_rad', quat_error_magnitude(pose[:, 3:], target[:, 3:7])),
                    ('base_linear_speed_max', velocity[:, :3].norm(dim=-1)),
                    ('base_angular_speed_max', velocity[:, 3:].norm(dim=-1))]:
                arrays[name][f] = torch.maximum(arrays[name][f], value)
        def record_post_physics(self, **ctx):
            f = ctx['frame']; x = observer.latest
            alive = ctx['active']; seen = alive | ctx['newly_stopped']
            strict = (x['pad_found'] & torch.isfinite(x['pad_points']).all(-1)
                      & torch.isfinite(x['pad_force']).all(-1) & (x['pad_force'].norm(dim=-1) >= .05))
            arrays['q'][f] = ctx['q']
            arrays['strict'][f] = strict & alive[:, None]
            arrays['found'][f] = (x['pad_found'] | x['connector_found']) & alive[:, None]
            arrays['seen'][f] = seen
            arrays['force'][f] = torch.where(seen, torch.maximum(self.substep_peak[f], observer.safety_peak_force_n), 0)
            arrays['pad_force'][f] = x['pad_force']; arrays['connector_force'][f] = x['connector_force']
        def step(self, **kwargs):
            nonlocal clipped
            ctx = self.context; f = ctx['frame']
            if not ctx['settling']:
                arrays['q_pre'][f] = kwargs['q']; arrays['pre_force'][f] = observer.safety_peak_force_n
                if f % 1000 == 0:
                    print(f'[FREE-PALM] {args.mode} frame={f}/{total} active={int(ctx["active"].sum())}/{batch}', flush=True)
            if raw:
                target = command_t[f]
                difference = (target - kwargs['q']) / .08
                clipped += int(((difference.abs() > 2) & ctx['active'][:, None]).sum())
                action = difference.clamp(-2, 2)
                diag = dict(nominal_q=target, command_q=kwargs['q'] + .08*action)
                self.control_frame += 1
            else:
                action, diag = super().step(**kwargs)
            if not ctx['settling']:
                arrays['nominal'][f] = diag['nominal_q']; arrays['command'][f] = diag['command_q']
                arrays['scheduler'][f] = target if raw else self.previous_nominal
                arrays['recovery'][f] = self.runtime_recovery_offset
                if self.scheduler.last_drop_index is not None:
                    arrays['drop'][f] = self.scheduler.last_drop_index
                for target,name in [('projection_engaged','paired_engaged'),
                                    ('projection_infeasible','paired_infeasible'),
                                    ('bounded_command_changed','bounded_command_changed')]:
                    if name in diag:arrays[target][f] = diag[name]
            display_callback = getattr(args, 'display_callback', None)
            if display_callback is not None:
                display_callback(self, kwargs, diag)
            return action, diag
        def close(self):
            nonlocal planner_report
            if self.async_planner is not None:
                planner_report = self.async_planner.report()
            super().close()

    initial_q = np.stack([r['q'][0] for r in rows])
    initial_vel = np.stack([r['velocity'][0] for r in rows])
    initial_cmd = np.stack([r['command'][0] for r in rows])
    deployment.MODEL = str(DP)
    driver = None
    if support_kind == 'kinematic':
        from palm_support import KinematicPalmSupport
        driver = KinematicPalmSupport()
    elif support_kind == 'weld':
        from palm_support import WeldPalmSupport
        driver = WeldPalmSupport(args.support_time_constant)
    meta = deployment.run(batch, args.ids[0], total, device, args.seed,
        settle_steps=0 if raw else 100, max_safe_force_n=100, per_env_safety=True,
        episode_data=data, chain_factory=Chain, contact_observer=observer,
        initial_joint_position=initial_q, initial_joint_velocity=initial_vel,
        initial_joint_target=initial_cmd, stop_when_all_inactive=True, palm_driver=driver)
    if meta['stop_reason'] is not None:
        raise RuntimeError(meta)
    values = {k: v.cpu().numpy() for k, v in arrays.items()}
    lanes = []
    for i, row in enumerate(rows):
        valid = values['strict'][hold:, i]; found = values['found'][hold:, i]
        seen = values['seen'][hold:, i]
        stop = meta['per_env_stop_frame'][i]
        longest, first = longest_loss(valid.all(-1))
        error = abs(values['q'][hold:, i] - teacher[hold:, i]).mean(-1)
        lane = dict(**row['meta'], status='force_stop' if stop >= 0 else 'completed',
            evaluated_frames=len(valid), observed_frames=int(seen.sum()),
            stop_task_frame=max(0, stop-hold)+(1 if raw else 0) if stop >= 0 else None,
            stopped_during_initialization=bool(not raw and 0 <= stop < hold),
            strict_healthy_four_rate=float(valid.all(-1).mean()), found_four_rate=float(found.all(-1).mean()),
            strict_healthy_per_finger=valid.mean(0).tolist(), found_per_finger=found.mean(0).tolist(),
            longest_four_loss_frames=longest, first_loss_over5_frame=first,
            peak_force_n=float(values['force'][:, i].max()),
            max_forward_only_force_n=float(values['pre_force'][:, i].max()),
            source_q_tracking_mae_rad=float(error[seen].mean()) if seen.any() else None,
            max_joint_error_rad=float(abs(values['q'][hold:, i]-teacher[hold:, i])[seen].max()) if seen.any() else None,
            source_q_mae_first100_rad=float(error[:100][seen[:100]].mean()) if seen[:100].any() else None)
        if record_base and seen.any():
            lane['base_substep'] = {
                name: dict(max=float(values[name][hold:, i][seen].max()),
                           p95=float(np.quantile(values[name][hold:, i][seen], .95)))
                for name in ('base_error_m', 'base_error_rad',
                             'base_linear_speed_max', 'base_angular_speed_max')}
        lanes.append(lane)
    name = f'{args.mode}_{args.frames}_delay{args.delay_frames}'
    if args.root_velocity != 'recorded':
        name += '_root' + args.root_velocity
    if args.materials != 'deployment':
        name += '_materials' + args.materials
    if args.projection_live_step is not None:
        name += '_live' + str(args.projection_live_step)
    if args.command_error_limit is not None:
        name += '_command' + str(args.command_error_limit)
    if support_kind != 'free' or record_base:
        name += '_support' + support_kind
    if support_kind == 'weld':
        name += '_tau' + str(args.support_time_constant)
    args.output.mkdir(parents=True, exist_ok=True)
    trace = args.output / (name + '.npz')
    np.savez_compressed(trace, **values, q_source=teacher,
                        palm_pose_object=data['palm_pose_object'], ids=args.ids)
    report = dict(mode=args.mode, ids=args.ids, requested_source_frames=args.frames,
        hold_frames=hold, recorded_index_start=1 if raw else 0, lanes=lanes,
        deployment=meta, trace=str(trace), mcc_profile=profile, planner=planner_report,
        delay_frames=args.delay_frames, actuator_clipped_joint_frames=clipped,
        physics_root_velocity=args.root_velocity,
        palm_support=(driver.report() if driver is not None else
                      dict(kind='free', support='legacy freejoint, recentered after each1ms solver step')),
        base_recording='Actual post-integration freejoint state before recenter, or native kinematic root; maxima across10substeps' if record_base else None,
        contact_materials=args.materials,
        checkpoint=str(DP), contact_definition=observer.definition(),
        initialization='Source q_hand[0],qvel[0],q_cmd[0]; raw starts at sample1; Oracle/DP share100settle+100stationary history frames',
        physics='Same free-palm/static-object deployment,10x1ms substeps; post-step and actual substep100N safety for all modes',
        source='Actual source palm_pose_world relative to actual object_pose_world, causal twist, fresh published200ms palm command',
        chunk='Oracle absolute future q_hand at t+5h or frozen DP; same scheduler/MCC; no RL residual',
        raw='Recorded actuator q_cmd only; scheduler/MCC bypassed deliberately for physical fidelity diagnosis')
    write(args.output / (name + '.json'), report)
    print(json.dumps(lanes, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('raw', 'oracle', 'dp'), required=True)
    p.add_argument('--ids', type=int, nargs='+', default=[36, 39, 41, 43, 45])
    p.add_argument('--frames', type=int, default=6000)
    p.add_argument('--delay-frames', type=int, default=0)
    p.add_argument('--root-velocity', choices=('recorded', 'zero'), default='recorded',
                   help='Raw-only diagnostic of the mocap collection velocity convention; not a deployment fix')
    p.add_argument('--materials', choices=('deployment', 'collection'), default='deployment')
    p.add_argument('--palm-support', choices=('free', 'kinematic', 'weld'), default='free',
                   help='Opt-in ideal prescribed support; freejoint replay remains the default')
    p.add_argument('--support-time-constant', type=float, default=.01,
                   help='Soft-weld time constant in seconds; diagnostic, not calibrated arm stiffness')
    p.add_argument('--record-base', action='store_true',
                   help='Record actual1ms base displacement before legacy recentering')
    p.add_argument('--projection-live-step',type=float,default=None,
                   help='Opt-in measured-q bound for the existing safety projection; baseline defaults preserved')
    p.add_argument('--command-error-limit',type=float,default=None,
                   help='Opt-in finalMCC position-target error bound on every finger; not a velocity limit')
    p.add_argument('--seed', type=int, default=20261013)
    p.add_argument('--output', type=Path, default=ROOT/'mcc_finger_compliance_control/data/long_free_palm_audit_20261013')
    args = p.parse_args()
    if args.mode == 'raw' and args.delay_frames:
        p.error('Raw command replay has no diffusion delivery delay')
    if args.mode != 'raw' and args.root_velocity != 'recorded':
        p.error('Root-velocity diagnostic is restricted to raw replay')
    if args.mode == 'raw' and args.projection_live_step is not None:
        p.error('Raw replay has no MCC projection')
    if args.command_error_limit is not None and (args.mode == 'raw' or args.projection_live_step is not None):
        p.error('Final-command candidate cannot be combined with raw replay or projection-only candidate')
    if args.palm_support == 'weld' and args.mode != 'raw':
        p.error('Finite-weld support is currently limited to command replay; actor base feedback is not yet accepted')
    run(args)


if __name__ == '__main__':
    main()
