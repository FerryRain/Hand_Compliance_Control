"""Run the collector with explicit control options and retain physical diagnostics."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=6000)
    parser.add_argument('--stable-initialization', action='store_true')
    parser.add_argument('--initialization-extra-steps', type=int, default=1200)
    parser.add_argument('--task-prefix-frames', type=int, default=0,
                        help='Diagnostic prefix only; retain the FULL reference timing, not a compressed path')
    parser.add_argument('--force-profile', choices=('legacy', 'regulated'), default='legacy')
    parser.add_argument('--force-audit', action='store_true')
    parser.add_argument('--audit-stride', type=int, default=20)
    parser.add_argument('--arm-stiffness-scale', type=float, default=1.)
    parser.add_argument('--finger-stiffness', type=float, default=35.)
    parser.add_argument('--finger-damping', type=float, default=2.5)
    parser.add_argument('--finger-effort-limit', type=float, default=35.)
    parser.add_argument('--mcc-authority', choices=('baseline','outward'), default='baseline')
    parser.add_argument('--seed', type=int, default=20261013)
    parser.add_argument('--execution-randomization', action='store_true',
                        help='Use the existing physical execution tube; never perturb labels directly')
    parser.add_argument('--execution-perturbation-min-rad', type=float, default=.003)
    parser.add_argument('--execution-perturbation-max-rad', type=float, default=.020)
    parser.add_argument('--execution-clean-env-fraction', type=float, default=0.,
                        help='For single-env jobs, choose clean jobs in the coordinator instead')
    parser.add_argument('--overforce-response',choices=('record','stop'),default='record',
                        help='Default: complete the reference, recording overforce rather than terminating')
    parser.add_argument('--substep-force-safety', action='store_true',
                        help='Observe the same 100 N stop at actual 1 ms solver steps')
    parser.add_argument('--force-parts', action='store_true',
                        help='Read-only actual solver loads for tip/connector/other finger/palm')
    args = parser.parse_args()
    if args.initialization_extra_steps < 0 or not 0 <= args.task_prefix_frames <= args.frames:
        parser.error('invalid initialization budget or task prefix')
    if args.finger_stiffness <= 0 or args.finger_damping < 0 or args.finger_effort_limit <= 0:
        parser.error('invalid finger actuator parameters')
    args.output.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.plan) as h:
        batch = h['palm_pose_object'].shape[1]
        object_id = h.attrs.get('object_id')
        if isinstance(object_id, bytes): object_id = object_id.decode('utf-8')
        if not isinstance(object_id, str) or not object_id:
            raise ValueError('Reference must explicitly identify its object_id')
    from object_catalog import load_object_config
    load_object_config(object_id)  # fail before allocating a mismatched simulation
    from mjlab.envs import ManagerBasedRlEnv
    from true_fingertip_contact import ContactSlotObserver
    import collect_trajectories as collector
    import collection_force_regulation as regulation
    original_force_config = regulation.collection_force_config
    if args.mcc_authority == 'outward':
        if args.force_profile != 'regulated':
            parser.error('outward authority requires regulated force profile')
        from collection_force_experiment import expanded_outward_authority
        def force_config(config, profile):
            value = original_force_config(config, profile)
            return expanded_outward_authority(value) if profile == 'regulated' else value
        regulation.collection_force_config = force_config
    auditor = None
    if args.force_audit:
        from independent_contact_force_audit import IndependentContactForceAudit
        auditor = IndependentContactForceAudit(args.audit_stride)
    original_step = ManagerBasedRlEnv.step
    original_read = ContactSlotObserver.read
    original_teacher = collector.FullHandMCCCollectionPolicy.__call__
    records = []
    physics_records = []
    inside_physics = False
    physical_substep = 0
    arm_intervention = None
    tick_force_peak = None
    part_state = None
    tick_part_net = tick_part_point = None
    peak_part_events = {}
    part_labels = ('true_fingertip', 'connector', 'other_link', 'palm')
    started = time.perf_counter()
    actual_motion_start = None
    last_initialization_debug = []
    initialization_overrides = []
    prefix_completed = False
    class PrefixComplete(Exception):
        pass
    controller_hash = hashlib.sha256(Path(__file__).with_name('collect_trajectories.py').read_bytes()).hexdigest()
    def teacher(policy, *values, **kwargs):
        nonlocal arm_intervention, actual_motion_start, last_initialization_debug, initialization_overrides
        motion = policy.motion_preview_controller
        if arm_intervention is None and motion is not None and bool(motion.motion_active.any()):
            actual_motion_start = int(motion.motion_start)
            from collection_force_experiment import scale_arm_support
            arm_intervention = scale_arm_support(policy.env, args.arm_stiffness_scale)
            import mujoco
            mujoco.mj_saveModel(policy.env.sim.mj_model,str(args.output/'scene.mjb'),None)
        result = original_teacher(policy, *values, **kwargs)
        initialization_overrides = list(policy.initialization_overrides)
        if motion is not None and not bool(motion.motion_active.any()):
            last_initialization_debug = [x.debug.copy() if x is not None else {}
                                         for x in policy.initializers]
            count = getattr(policy.env, '_long_probe_steps', 0)
            if args.stable_initialization and count >= 800 and count % 100 == 0:
                print('[INITIALIZATION-PROBE]', count, json.dumps(last_initialization_debug), flush=True)
        policy.env._long_probe_calibrated = policy.contact_calibrated.copy()
        policy.env._long_probe_force_state = dict(
            force_offset=np.asarray([c.force_servo_offset.copy() for c in policy.controllers]),
            force_filtered=np.asarray([c.force_servo_filtered.copy() for c in policy.controllers]),
            force_setpoint=np.asarray([c.force_setpoint.copy() for c in policy.controllers]),
            safety_active=np.asarray([x['active'] for x in policy.force_safety_debug]),
            safety_hard=np.asarray([x['hard_active'] for x in policy.force_safety_debug]),
            safety_correction=np.asarray([x['correction'] for x in policy.force_safety_debug]),
            safety_margin=np.asarray([x['min_constraint_margin_m'] for x in policy.force_safety_debug]))
        return result
    def step(env, *values, **kwargs):
        nonlocal inside_physics, tick_force_peak, tick_part_net, tick_part_point, physical_substep
        tick_force_peak = np.zeros(env.num_envs)
        tick_part_net = np.zeros((env.num_envs,4))
        tick_part_point = np.zeros((env.num_envs,4))
        inside_physics = True
        physical_substep = 0
        if auditor is not None: auditor.begin_tick(env)
        try:
            result = original_step(env, *values, **kwargs)
        finally:
            inside_physics = False
        env._long_probe_steps = getattr(env, '_long_probe_steps', 0) + 1
        if (args.task_prefix_frames and actual_motion_start is not None
                and env._long_probe_steps > actual_motion_start+args.task_prefix_frames):
            raise PrefixComplete()
        return result

    def sample_parts(env, control_step):
        nonlocal part_state, tick_part_net, tick_part_point
        import torch
        import warp as wp
        import mujoco
        from diagnose_force_stop_parts import all_object_loads, region
        d = env.sim.wp_data; m = env.sim.mj_model
        if part_state is None:
            names = [mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g) or '' for g in range(m.ngeom)]
            palm = next(b for b in range(m.nbody)
                        if (m.body(b).name or '').endswith('/palm_lower'))
            hand = np.zeros(m.ngeom,dtype=np.int32)
            for g in range(m.ngeom):
                b = int(m.geom_bodyid[g])
                while b > 0 and b != palm: b = int(m.body_parentid[b])
                hand[g] = int(b == palm)
            groups = [[g for g,n in enumerate(names) if hand[g] and region(n)==label]
                      for label in part_labels]
            with wp.ScopedDevice(env.sim.wp_device):
                part_state = dict(names=names,groups=groups,
                    objects=wp.array(np.asarray([n.startswith('target/') for n in names],dtype=np.int32),dtype=wp.int32),
                    hand=wp.array(hand,dtype=wp.int32),
                    net=wp.zeros((env.num_envs,m.ngeom),dtype=wp.vec3),
                    point=wp.zeros((env.num_envs,m.ngeom),dtype=float))
        a = part_state
        with wp.ScopedDevice(env.sim.wp_device):
            a['net'].zero_(); a['point'].zero_()
            wp.launch(all_object_loads,dim=d.naconmax,inputs=[env.sim.wp_model.opt.cone,
                d.contact.frame,d.contact.friction,d.contact.dim,d.contact.efc_address,d.efc.force,
                d.njmax,d.nacon,d.contact.geom,d.contact.worldid,a['objects'],a['hand']],
                outputs=[a['net'],a['point']])
        with torch.cuda.stream(torch.cuda.ExternalStream(wp.get_stream(env.sim.wp_device).cuda_stream)):
            net = wp.to_torch(a['net']).norm(dim=-1).cpu().numpy()
            point = wp.to_torch(a['point']).cpu().numpy()
        current_net=np.zeros((env.num_envs,4));current_point=current_net.copy()
        current_geom=np.full((env.num_envs,4),-1,dtype=np.int32)
        for j,group in enumerate(a['groups']):
            if not group: continue
            nets = net[:,group].max(-1); points = point[:,group].max(-1)
            current_net[:,j]=nets;current_point[:,j]=points
            current_geom[:,j]=np.asarray(group)[np.argmax(np.maximum(net[:,group],point[:,group]),axis=-1)]
            tick_part_net[:,j] = np.maximum(tick_part_net[:,j],nets)
            tick_part_point[:,j] = np.maximum(tick_part_point[:,j],points)
            for lane in range(env.num_envs):
                key = f'{lane}:{part_labels[j]}'
                value = max(nets[lane],points[lane])
                if value > peak_part_events.get(key,{}).get('force_n',0):
                    selected = int(np.argmax(np.maximum(net[lane,group],point[lane,group])))
                    g = group[selected]
                    peak_part_events[key] = dict(lane=lane,region=part_labels[j],
                        geom=a['names'][g],task_frame=int(control_step-(actual_motion_start or 1200)),
                        force_n=float(value),geom_net_force_n=float(net[lane,g]),
                        single_contact_force_n=float(point[lane,g]))
        return current_net,current_point,current_geom
    def read(observer, env, entries):
        nonlocal physical_substep
        result = original_read(observer, env, entries)
        count = getattr(env, '_long_probe_steps', 0)
        if inside_physics:
            physical_substep += 1
            tick_force_peak[:] = np.maximum(tick_force_peak, observer.safety_peak_force_n.detach().cpu().numpy())
            if auditor is not None: auditor.observe(env, observer, count, physical_substep)
            parts = sample_parts(env,count) if args.force_parts and count>=800 else None
            if args.overforce_response=='record' and count>=800:
                x=observer.latest
                physics_records.append(dict(step=count,substep=physical_substep,
                    qpos=env.sim.data.qpos.detach().cpu().numpy().copy(),
                    qvel=env.sim.data.qvel.detach().cpu().numpy().copy(),
                    ctrl=env.sim.data.ctrl.detach().cpu().numpy().copy(),
                    actuator_force=env.sim.data.actuator_force.detach().cpu().numpy().copy(),
                    qfrc_actuator=env.sim.data.qfrc_actuator.detach().cpu().numpy().copy(),
                    mocap_pos=env.sim.data.mocap_pos.detach().cpu().numpy().copy(),
                    mocap_quat=env.sim.data.mocap_quat.detach().cpu().numpy().copy(),
                    pad_force=x['pad_force'].detach().cpu().numpy().copy(),
                    connector_force=x['connector_force'].detach().cpu().numpy().copy(),
                    pad_found=x['pad_found'].detach().cpu().numpy().copy(),
                    connector_found=x['connector_found'].detach().cpu().numpy().copy(),
                    peak_force=observer.safety_peak_force_n.detach().cpu().numpy().copy(),
                    part_net=parts[0] if parts else np.zeros((batch,4)),
                    part_single=parts[1] if parts else np.zeros((batch,4)),
                    part_geom=parts[2] if parts else np.full((batch,4),-1,dtype=np.int32)))
        if records and records[-1]['step'] == count:
            return result
        from scipy.spatial.transform import Rotation
        robot = env.scene['robot']
        names = [b.name or '' for b in robot.data.indexing.bodies]
        index = next(i for i, name in enumerate(names) if name == 'palm_lower' or name.endswith('/palm_lower'))
        palm = robot.data.body_link_pose_w[:, index].detach().cpu().numpy()
        mid = int(env.scene['target'].data.indexing.mocap_id)
        position = env.sim.data.mocap_pos[:, mid].detach().cpu().numpy()
        quaternion = env.sim.data.mocap_quat[:, mid].detach().cpu().numpy()
        rotation = Rotation.from_quat(quaternion[:, [1, 2, 3, 0]])
        palm_rotation = Rotation.from_quat(palm[:, [4, 5, 6, 3]])
        rel_rotation = (rotation.inv() * palm_rotation).as_quat()[:, [3, 0, 1, 2]]
        rel_position = rotation.inv().apply(palm[:, :3] - position)
        x = observer.latest
        strict = (x['pad_found'] & x['pad_force'].isfinite().all(-1)
                  & x['pad_points'].isfinite().all(-1) & (x['pad_force'].norm(dim=-1) >= .05))
        records.append(dict(step=count, q=robot.data.joint_pos[:, -16:].detach().cpu().numpy(),
            palm=np.concatenate((rel_position, rel_rotation), axis=-1),
            strict=strict.detach().cpu().numpy(),
            found=(x['pad_found'] | x['connector_found']).detach().cpu().numpy(),
            teacher_calibrated=np.asarray(getattr(env, '_long_probe_calibrated', np.zeros(batch, bool))),
            pad_force=x['pad_force'].detach().cpu().numpy(),
            connector_force=x['connector_force'].detach().cpu().numpy(),
            force=(np.maximum(observer.safety_peak_force_n.detach().cpu().numpy(),tick_force_peak)
                   if (args.substep_force_safety or args.force_parts) and tick_force_peak is not None else
                   observer.safety_peak_force_n.detach().cpu().numpy()),
            part_net_force=(tick_part_net.copy() if tick_part_net is not None else np.zeros((batch,4))),
            part_single_force=(tick_part_point.copy() if tick_part_point is not None else np.zeros((batch,4)))))
        state = getattr(env, '_long_probe_force_state', None)
        for key in ('force_offset', 'force_filtered', 'force_setpoint', 'safety_active', 'safety_hard',
                    'safety_correction', 'safety_margin'):
            shape = (batch,) if key == 'safety_margin' else (batch, 16 if key == 'safety_correction' else 4)
            records[-1][key] = state[key].copy() if state is not None else np.zeros(shape)
        if count % 250 == 0:
            print(f'[ORACLE-PROBE] step={count} wall={time.perf_counter()-started:.1f}s', flush=True)
        return result
    ManagerBasedRlEnv.step = step
    ContactSlotObserver.read = read
    collector.FullHandMCCCollectionPolicy.__call__ = teacher
    name = str((args.output / 'collected').resolve()).removesuffix('.h5')
    command = ['collect_trajectories.py', '--viewer', 'headless', '--device', 'cuda:0',
        '--object-id', object_id, '--teacher-controller', 'fullhand_mcc',
        '--motion-mode', 'planner_inverse', '--planner-file', str(args.plan.resolve()),
        '--initial-orientation-mode', 'fixed', '--differential-contact-qp', '--manifold-qp-blend', '1.0',
        '--trajectory-length', str(1200 + args.frames), '--motion-start', '800',
        '--motion-length', str(args.frames), '--record-start-step', '800',
        '--max-prep-wait-steps', str(args.initialization_extra_steps if args.stable_initialization else 0),
        '--planner-settle-steps', '400', '--physics-substeps', '10',
        '--fixed-motion-start', '--no-contact-gate', '--num-envs', str(batch),
        '--max-trajectories', str(batch), '--max-attempts', '1', '--seed', str(args.seed),
        '--filename', name, '--split-fingertip-collision', '--max-safe-force-n', '100',
        '--overforce-response',args.overforce_response]
    command.extend(['--fullhand-force-profile', args.force_profile])
    command.extend(['--fullhand-finger-stiffness', str(args.finger_stiffness),
                    '--fullhand-finger-damping', str(args.finger_damping),
                    '--fullhand-finger-effort-limit', str(args.finger_effort_limit)])
    if args.stable_initialization:
        command.append('--stable-initialization')
    if args.execution_randomization:
        command.extend(['--execution-randomization',
            '--execution-perturbation-min-rad', str(args.execution_perturbation_min_rad),
            '--execution-perturbation-max-rad', str(args.execution_perturbation_max_rad),
            '--execution-clean-env-fraction', str(args.execution_clean_env_fraction),
            '--execution-perturbation-time-constant-s', '.35',
            '--execution-perturbation-ramp-steps', '50'])
    if args.substep_force_safety or args.force_parts:
        command.append('--substep-force-safety')
    error = None
    try:
        sys.argv = command
        collector.main()
    except PrefixComplete:
        prefix_completed = True
        print('[ORACLE-PROBE] diagnostic prefix completed at original reference speed', flush=True)
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        print('[ORACLE-PROBE]', error, flush=True)
    finally:
        ManagerBasedRlEnv.step = original_step
        ContactSlotObserver.read = original_read
        collector.FullHandMCCCollectionPolicy.__call__ = original_teacher
        regulation.collection_force_config = original_force_config
        arrays = {k: np.asarray([r[k] for r in records]) for k in (
            'step', 'q', 'palm', 'strict', 'found', 'force',
            'teacher_calibrated', 'pad_force', 'connector_force', 'part_net_force', 'part_single_force')}
        arrays.update({key: np.asarray([r[key] for r in records]) for key in
            ('force_offset', 'force_filtered', 'force_setpoint', 'safety_active', 'safety_hard',
             'safety_correction', 'safety_margin')})
        np.savez_compressed(args.output / 'trace.npz', **arrays)
        if physics_records:
            physical={k:np.asarray([r[k] for r in physics_records]) for k in physics_records[0]}
            physical['geom_names']=np.asarray(part_state['names'] if part_state else [],dtype='U160')
            np.savez_compressed(args.output/'physics_trace.npz',**physical)
        summary = dict(plan=str(args.plan.resolve()), command=command, error=error,
            elapsed_s=time.perf_counter()-started, last_control_step=records[-1]['step'] if records else 0,
            physics='Original planner_inverse collector: arm-held palm, mocap object; same relative palm reference, not a matched free-palm DP replay',
            changes=f'Force profile={args.force_profile}; overforce response={args.overforce_response}; 100N event threshold; motion, physics and valid-knot filtering retained',
            force_profile=args.force_profile,
            finger_actuator_config=dict(kp_nm_per_rad=args.finger_stiffness,
                kd_nm_s_per_rad=args.finger_damping,effort_limit_nm=args.finger_effort_limit,
                activation='from model creation, including initialization'),
            arm_intervention=arm_intervention, mcc_authority=args.mcc_authority,
            overforce_response=args.overforce_response,overforce_event_threshold_n=100.,
            physics_trace_samples=len(physics_records),
            physics_trace_semantics='1ms samples: constraint forces before integration; qpos/qvel after integration; ctrl is the actual actuator target used for that solve',
            collector_sha256=controller_hash,
            execution_randomization=args.execution_randomization,
            substep_force_safety=args.substep_force_safety or args.force_parts,
            force_part_labels=part_labels, peak_part_events=peak_part_events,
            force_parts_readonly=args.force_parts,
            requested_task_frames=args.frames,
            actual_motion_start_step=actual_motion_start,
            stable_initialization=args.stable_initialization,
            last_initialization_debug=last_initialization_debug,
            initialization_overrides=initialization_overrides,
            diagnostic_prefix_frames=args.task_prefix_frames,
            diagnostic_prefix_completed=prefix_completed,
            force_part_semantics='Maximum per collision geom net magnitude and single contact magnitude at actual solver substeps; not motor torque')
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        if auditor is not None: auditor.write(args.output/'independent_force_audit.json')
    if error:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
