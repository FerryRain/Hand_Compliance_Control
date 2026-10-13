"""New I04 MCC-only experiments on the actual nonconvex Bunny plant."""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import mujoco
import numpy as np
import trimesh

from Module.fr3_leap import ARM_HOME_Q, HAND_NATURAL_Q
from .contracts import ExperimentConfig, WholeHandTarget, WristTarget, array_digest
from .oracle import GTCoverageOracle, MeasuredCoverage
from .physics import (BasicPassiveMCC, contact_measurement, joint_state,
                      pose, set_control, set_joint_state, tip_positions, rotation)
from .planning import GenericTrajectoryExecutor, SharedGeometricTrajectoryPlanner


def prepare_initial_state(handles, graph, config):
    """One common privileged initial grasp, before either treatment starts."""
    h = handles
    data = mujoco.MjData(h.model)
    rng = np.random.default_rng(config.seed)
    q = np.r_[ARM_HOME_Q, HAND_NATURAL_Q]
    q += rng.normal(0, 0.0003, len(q))
    set_joint_state(h, data, q)
    set_control(h, data, q)
    tips = tip_positions(h, data)
    points, _, triangles = trimesh.proximity.closest_point(graph.mesh, tips - h.object_position_m)
    normals = graph.mesh.face_normals[triangles]
    from .physics import pad_support
    support = np.array([pad_support(h, data, i, normals[i]) for i in range(4)])
    targets = h.object_position_m + points + (support[:, None] - 0.0003) * normals
    initial_wrist = WristTarget(pose(h, data))
    target = WholeHandTarget(initial_wrist, targets)
    planner = SharedGeometricTrajectoryPlanner(h, config)
    goal, ep, er, ef = planner.solve(q, target, iterations=240)
    # Pad box support changes as the fingers curl. Calibrate the common
    # initial geometry before treatment instead of assuming its old radius.
    for _ in range(4):
        set_joint_state(h, planner.scratch, goal)
        support = np.array([pad_support(h, planner.scratch, i, normals[i]) for i in range(4)])
        targets = h.object_position_m + points + (support[:, None] - 0.0025) * normals
        target = WholeHandTarget(initial_wrist, targets)
        goal, ep, er, ef = planner.solve(goal, target, iterations=160)
    mcc = BasicPassiveMCC(h, config.dt_s)
    last_command = q.copy()
    acquisition_nominal = goal.copy()
    force_peak = 0.0
    unsafe_count = 0
    for _ in range(int(config.acquisition_s / config.dt_s)):
        forces, _, measured_normals, active, unsafe, _ = contact_measurement(h, data, config.contact_threshold_n)
        force_peak = max(force_peak, float(forces.max()))
        unsafe_count += bool(unsafe)
        # Privileged acquisition is shared and ends before either cell starts.
        # The treatment's passive MCC never receives these oracle normals or
        # this free-finger approach rule.
        from .physics import position_jacobian
        from Module.e05_physics.scene import FINGERS
        for finger in range(4):
            if not active[finger]:
                jp, _ = position_jacobian(h, data, h.tip_site_ids[finger])
                columns = np.array([int(name) for name in FINGERS[finger].joint_names]) + 7
                jf = jp[:, columns]
                delta = jf.T @ np.linalg.solve(jf @ jf.T + 1e-6 * np.eye(3),
                                                -normals[finger] * 0.006 * config.dt_s)
                acquisition_nominal[columns] += np.clip(delta, -0.002, 0.002)
                acquisition_nominal[columns] = np.clip(acquisition_nominal[columns],
                    goal[columns] - 0.6, goal[columns] + 0.6)
        command = mcc.command(data, acquisition_nominal, forces, measured_normals, active, config.desired_force_n)
        step = config.max_joint_command_speed_rad_s * config.dt_s
        last_command = np.clip(command, last_command - step, last_command + step)
        last_command = np.clip(last_command, planner.lower, planner.upper)
        set_control(h, data, last_command)
        mujoco.mj_step(h.model, data)
        mujoco.mj_forward(h.model, data)
    forces, _, _, active, unsafe, _ = contact_measurement(h, data, config.contact_threshold_n)
    state = {'qpos': data.qpos.copy(), 'qvel': data.qvel.copy(), 'ctrl': data.ctrl.copy(),
             'time': np.array(data.time)}
    audit = {'initial_contact_count': int(active.sum()), 'initial_forces_n': forces.tolist(),
             'acquisition_peak_n': force_peak, 'acquisition_non_tip_steps': int(unsafe_count),
             'initial_non_tip_contacts': unsafe, 'acquisition_ik': [ep, er, ef],
             'state_sha256': array_digest(np.r_[state['qpos'], state['qvel'], state['ctrl']])}
    return state, audit


def longest_gap(mask, dt):
    run = maximum = 0
    for value in mask:
        run = 0 if value else run + 1
        maximum = max(maximum, run)
    return maximum * dt


def run_cell(handles, graph, config, cell, initial_state, initial_audit, output):
    h = handles
    data = mujoco.MjData(h.model)
    data.qpos[:] = initial_state['qpos']
    data.qvel[:] = initial_state['qvel']
    data.ctrl[:] = initial_state['ctrl']
    data.time = float(initial_state['time'])
    mujoco.mj_forward(h.model, data)
    initial_nominal_q = joint_state(h, data)
    oracle = GTCoverageOracle(h, data, graph, config)
    coverage = MeasuredCoverage(graph, config)
    planner = SharedGeometricTrajectoryPlanner(h, config)
    executor = GenericTrajectoryExecutor(config)
    mcc = BasicPassiveMCC(h, config.dt_s)
    last_command = initial_nominal_q.copy()
    events = []
    logs = {key: [] for key in ('time_s', 'q_rad', 'command_rad', 'palm_pose_world',
            'nominal_palm_pose_world', 'tip_positions_world_m', 'forces_n', 'active',
            'contact_positions_world_m', 'coverage', 'covered_goals', 'goal_id',
            'max_penetration_m', 'controller_latency_s', 'wrist_tracking_m', 'wrist_tracking_rad')}
    steps = int((config.duration_s - config.acquisition_s) / config.dt_s)
    contact_counts = np.zeros(steps, dtype=int)
    peak = np.zeros(steps)
    per_finger_contact = np.zeros(4)
    force_square = 0.0
    force_samples = 0
    high_force_steps = 0
    high_force_run = 0
    non_tip_steps = 0
    goal_id = -1
    next_plan_time = 0.0
    planning_latencies = []
    oracle_latencies = []
    stop_reason = 'TIME_BUDGET_REACHED'
    wall_start = perf_counter()
    executed_steps = 0
    measured_samples = 0
    nominal_pose_data = mujoco.MjData(h.model)
    for step in range(steps):
        elapsed = step * config.dt_s
        measured_samples = step + 1
        forces, points, normals, active, unsafe, penetration = contact_measurement(h, data, config.contact_threshold_n)
        contact_counts[step] = int(active.sum())
        peak[step] = float(forces.max())
        per_finger_contact += active
        if np.any(active):
            force_square += float(np.sum((forces[active] - config.desired_force_n)**2))
            force_samples += int(active.sum())
        high_force_steps += bool(np.any(forces > config.hard_force_n))
        high_force_run = high_force_run + 1 if np.any(forces > config.hard_force_n) else 0
        non_tip_steps += bool(unsafe)
        # Hardware protection may stop dangerous execution. Loss of all
        # fingertip contacts is deliberately NOT a task-level stop or veto.
        hardware_reason = None
        if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
            hardware_reason = 'NONFINITE_STATE'
        elif unsafe:
            hardware_reason = ('SELF_COLLISION' if any(x.startswith('SELF_COLLISION:') for x in unsafe)
                               else 'NON_TIP_OBJECT_COLLISION')
        elif high_force_run * config.dt_s >= config.hard_force_duration_s:
            hardware_reason = 'SUSTAINED_DANGEROUS_FORCE'
        elif np.any(joint_state(h, data) < planner.lower - 0.02) or np.any(joint_state(h, data) > planner.upper + 0.02):
            hardware_reason = 'JOINT_LIMIT'
        if step % config.log_stride == 0:
            coverage.update(points - h.object_position_m, normals, active)
        if hardware_reason:
            events.append({'time_s': elapsed, 'kind': 'HARDWARE_STOP', 'reason': hardware_reason,
                           'forces_n': forces.tolist(), 'non_tip': unsafe})
            stop_reason = hardware_reason
            executed_steps = step
            break
        if executor.plan is None or executor.finished:
            if elapsed >= next_plan_time:
                oracle_start = perf_counter()
                candidates = oracle.ordered_candidates(data, coverage)
                if not candidates:
                    stop_reason = ('FULL_MEASURED_COVERAGE' if np.all(coverage.visited_vertices)
                                   else 'REQUIRED_SAMPLES_COVERED_AREA_INCOMPLETE')
                    executed_steps = step
                    break
                accepted = False
                for candidate in candidates[:8]:
                    oracle.attempts[candidate] += 1
                    latent = oracle.latent_target(candidate)
                    # The privileged Oracle checks terminal geometric
                    # feasibility for both adapters. Its joint witness is
                    # discarded; A's controller only receives WristTarget.
                    witness, ep, er, ef = planner.solve(joint_state(h, data), latent)
                    rejection = None
                    if ep > config.goal_position_tolerance_m or er > config.goal_rotation_tolerance_rad or ef > config.fingertip_tolerance_m:
                        rejection = 'ORACLE_GEOMETRIC_IK'
                    else:
                        rejection = planner.collision_reason(witness)
                    successor = None
                    if rejection is None:
                        successor = oracle.geometric_continuation(candidate, witness, planner)
                        if successor is None:
                            rejection = 'ORACLE_NO_GEOMETRIC_CONTINUATION'
                    if rejection:
                        events.append({'time_s': elapsed, 'kind': 'ORACLE_DEFER',
                                       'goal_id': int(candidate), 'reason': rejection,
                                       'position_m': ep, 'rotation_rad': er, 'tip_error_m': ef})
                        continue
                    exposed = oracle.expose(cell, latent)
                    plan, failure = planner.plan(data, exposed)
                    if plan is None:
                        events.append({'time_s': elapsed, 'kind': 'PLAN_REJECT',
                                       'goal_id': int(candidate), **failure})
                        continue
                    executor.commit(plan, joint_state(h, data), elapsed - config.dt_s)
                    planning_latencies.append(plan.latency_s)
                    goal_id = int(candidate)
                    events.append({'time_s': elapsed, 'kind': 'PLAN_COMMIT', 'goal_id': goal_id,
                                   'duration_s': plan.duration_s, 'digest': plan.digest,
                                   'wrist_target': latent.wrist.pose_world.tolist(),
                                   'privileged_continuation_goal_id': successor,
                                   'exposed_fields': ['wrist_pose'] if cell == 'A-MCC' else ['wrist_pose', 'tip_positions']})
                    accepted = True
                    break
                oracle_latencies.append(perf_counter() - oracle_start)
                if not accepted:
                    next_plan_time = elapsed + 0.50
        control_start = perf_counter()
        measured_q = joint_state(h, data)
        if executor.plan is not None:
            previous_barriers = executor.prefix_boundaries
            nominal = executor.step(elapsed, measured_q)
            if executor.prefix_boundaries > previous_barriers:
                events.append({'time_s': elapsed, 'kind': 'FRESH_PREFIX_FEEDBACK',
                               'measured_q_sha256': array_digest(measured_q),
                               'plan_digest': executor.plan.digest, 'phase_s': executor.elapsed})
        else:
            nominal = measured_q.copy()
        if cell == 'A-MCC':
            nominal[7:] = initial_nominal_q[7:]
        elif executor.plan is None:
            nominal[7:] = initial_nominal_q[7:]
        command = mcc.command(data, nominal, forces, normals, active, config.desired_force_n)
        limit = config.max_joint_command_speed_rad_s * config.dt_s
        command = np.clip(command, last_command - limit, last_command + limit)
        command = np.clip(command, planner.lower, planner.upper)
        last_command = command.copy()
        set_control(h, data, command)
        controller_latency = perf_counter() - control_start
        set_joint_state(h, nominal_pose_data, nominal)
        nominal_pose = pose(h, nominal_pose_data)
        measured_pose = pose(h, data)
        if step % config.log_stride == 0:
            values = {'time_s': elapsed, 'q_rad': measured_q, 'command_rad': command,
                'palm_pose_world': measured_pose, 'nominal_palm_pose_world': nominal_pose,
                'tip_positions_world_m': tip_positions(h, data), 'forces_n': forces.copy(),
                'active': active.copy(), 'contact_positions_world_m': points.copy(),
                'coverage': coverage.area_fraction, 'covered_goals': int(coverage.visited_goals.sum()),
                'goal_id': goal_id, 'max_penetration_m': penetration,
                'controller_latency_s': controller_latency,
                'wrist_tracking_m': float(np.linalg.norm(measured_pose[:3] - nominal_pose[:3])),
                'wrist_tracking_rad': float((rotation(measured_pose[3:]) * rotation(nominal_pose[3:]).inv()).magnitude())}
            for key, value in values.items():
                logs[key].append(value)
        mujoco.mj_step(h.model, data)
        mujoco.mj_forward(h.model, data)
        executed_steps = step + 1
        if step % 5000 == 0:
            print(f'{cell} seed={config.seed} t={elapsed:.1f}s contacts={active.sum()} '
                  f'coverage={coverage.area_fraction:.3%} plans={len(planning_latencies)}', flush=True)
    arrays = {key: np.asarray(value) for key, value in logs.items()}
    arrays['covered_vertices'] = coverage.visited_vertices
    arrays['required_goal_visited'] = coverage.visited_goals
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / 'trace.npz', **arrays)
    (output / 'events.json').write_text(json.dumps(events, indent=2), encoding='utf-8')
    counts = contact_counts[:measured_samples]
    peaks = peak[:measured_samples]
    simulated_duration = executed_steps * config.dt_s
    summary = {
        'cell': cell, 'seed': config.seed, 'status': 'EVALUATED', 'stop_reason': stop_reason,
        'config': asdict(config), 'initial_state': initial_audit,
        'requested_post_acquisition_s': config.duration_s - config.acquisition_s,
        'executed_post_acquisition_s': simulated_duration,
        'measurement_samples': measured_samples,
        'full_requested_budget': bool(executed_steps == steps),
        'contact_continuity': float(np.mean(counts >= 1)) if len(counts) else 0.0,
        'mean_contacts': float(np.mean(counts)) if len(counts) else 0.0,
        'four_contact_fraction': float(np.mean(counts == 4)) if len(counts) else 0.0,
        'at_least_three_fraction': float(np.mean(counts >= 3)) if len(counts) else 0.0,
        'per_finger_contact_fraction': (per_finger_contact / max(measured_samples, 1)).tolist(),
        'maximum_all_contact_gap_s': longest_gap(counts >= 1, config.dt_s),
        'final_covered_area_fraction': coverage.area_fraction,
        'visited_required_goals': int(coverage.visited_goals.sum()),
        'required_goal_count': graph.required_goal_count,
        'full_surface_complete': bool(np.all(coverage.visited_vertices)),
        'force_rmse_n': float(np.sqrt(force_square / max(force_samples, 1))),
        'peak_force_n': float(peaks.max()) if len(peaks) else 0.0,
        'dangerous_force_duration_s': high_force_steps * config.dt_s,
        'non_tip_contact_duration_s': non_tip_steps * config.dt_s,
        'plan_count': len(planning_latencies),
        'planning_latency_p95_s': float(np.percentile(planning_latencies, 95)) if planning_latencies else None,
        'oracle_latency_p95_s': float(np.percentile(oracle_latencies, 95)) if oracle_latencies else None,
        'prefix_boundaries': executor.prefix_boundaries,
        'task_contact_veto_count': executor.contact_task_vetoes,
        'normal_rejected_contact_samples': coverage.normal_rejected_contacts,
        'wall_duration_s': perf_counter() - wall_start,
        'trace_sha256': sha256((output / 'trace.npz').read_bytes()).hexdigest(),
        'terminal_observation': {'time_s': float(data.time - config.acquisition_s),
            'q_rad': joint_state(h, data).tolist(), 'forces_n': forces.tolist(),
            'contact_active': active.tolist()},
        'geometry_rejection_count': sum(e['kind'] == 'ORACLE_DEFER' for e in events),
        'plan_rejection_count': sum(e['kind'] == 'PLAN_REJECT' for e in events),
    }
    if len(arrays['palm_pose_world']) > 1:
        palm = arrays['palm_pose_world']
        summary['actual_wrist_translation_m'] = float(np.linalg.norm(np.diff(palm[:, :3], axis=0), axis=1).sum())
        summary['actual_wrist_rotation_rad'] = float(sum(
            (rotation(a[3:]) * rotation(b[3:]).inv()).magnitude() for a, b in zip(palm[1:], palm[:-1])))
        summary['wrist_tracking_p95_m'] = float(np.percentile(arrays['wrist_tracking_m'], 95))
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary
