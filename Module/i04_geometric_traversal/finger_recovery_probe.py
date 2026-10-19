"""Separate native probe: deliberately withdraw one pad, then recover it.

This perturbation is not natural whole-object traversal and earns no coverage.
The wrist reference stays fixed.  Only the selected finger's active-search
proposal is applied; other fingers receive measured-contact MCC throughout.
All three phases use one continuous MuJoCo trajectory after common acquisition.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import mujoco
import numpy as np

from .active_fingers import ActiveFingerSearch
from .contracts import ExperimentConfig
from .full_surface import build_scene, acquire, FullSurfacePlanner
from .physics import (BasicPassiveMCC, contact_measurement, joint_state, pose,
                      set_control, tip_positions)
from .recovery import SharedContactRecovery


def run(directory, *, seed=7, finger=0, budget_s=12., search_enabled=True, recovery_window_s=5.):
    if (finger not in range(4) or not np.isfinite(budget_s) or budget_s <= 0
            or not np.isfinite(recovery_window_s) or recovery_window_s <= .15):
        raise ValueError('finger must be 0..3 and budget must be positive')
    directory = Path(directory)
    if (directory / 'manifest.json').exists():
        raise FileExistsError('Use a fresh probe directory')
    root = Path(__file__).resolve().parents[2]
    names = ['Module/i04_geometric_traversal/' + name for name in
        ('finger_recovery_probe.py', 'active_fingers.py', 'full_surface.py', 'physics.py',
         'contracts.py', 'oracle.py', 'planning.py', 'recovery.py', 'fixtures.py')]
    names += ['Module/fr3_leap/model.py', 'Module/module_2_fingertip_mcc/controller.py']
    sources = {name: (root / name).read_bytes() for name in names}
    directory.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    h, graph, q, quality = build_scene(directory)
    data, initial = acquire(h, graph, q, seed=seed)
    cfg = ExperimentConfig(seed=seed)
    planner = FullSurfacePlanner(h, cfg)
    search = ActiveFingerSearch(h, graph)
    mcc = BasicPassiveMCC(h, .002)
    relief = SharedContactRecovery(cfg)
    columns = search.columns[finger] + 7
    geom = int(h.tip_geom_ids[finger])
    nominal = np.r_[data.ctrl[h.arm_actuator_ids], data.ctrl[h.hand_actuator_ids]]
    last = nominal.copy()
    held_arm = nominal[:7].copy()
    np.savez_compressed(directory / 'initial.npz', qpos=data.qpos, qvel=data.qvel, ctrl=data.ctrl)
    (directory / 'model.xml').write_text(h.xml, encoding='utf-8')
    for name, source in sources.items():
        destination = directory / 'source_snapshot' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source)
    manifest = dict(protocol='i04-single-finger-withdraw-recover.paired.v2', cell='B-MCC', seed=seed,
        object_position_m=h.object_position_m.tolist(), selected_finger=finger,
        controlled_perturbation=True, whole_traversal_evidence=False, coverage_authority='NONE',
        perturbation=dict(commanded_withdrawal_m=.003, maximum_withdrawal_m=.004,
                          speed_m_s=.003, pause_s=.3),
        recovery=dict(same_finger=True, active_min_n=.2, safe_max_n=8., stable_duration_s=.15,
                      observation_window_s=recovery_window_s, search_scope='selected_finger_only',
                      active_search_enabled=bool(search_enabled), stop_early_on_recontact=False),
        active_finger_search=asdict(search.config), budget_s=budget_s, initial_state=initial,
        geometry_quality=quality, source_hashes={name: sha256(value).hexdigest() for name, value in sources.items()})
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    phase, phase_started = 'BASELINE', 0.
    events, logs = [], {}
    outward = None
    withdrawal_start_center = None
    commanded_withdrawal = measured_withdrawal = 0.
    recovery_started = contact_lost_at = first_recontact_at = stable_since = stable_recovered_at = None
    peak = 0.
    high_run = 0
    success = False
    stop = 'TIME_BUDGET_REACHED'
    baseline_stable = None
    for step in range(int(budget_s / .002) + 1):
        t = step * .002
        observed_phase = phase
        forces, points, normals, active, unsafe, penetration = contact_measurement(h, data, .2)
        peak = max(peak, float(forces.max()))
        safe_selected = bool(active[finger] and forces[finger] <= 8.)
        if withdrawal_start_center is not None:
            measured_withdrawal = float((data.geom_xpos[geom] - withdrawal_start_center) @ outward)
            if not active[finger] and contact_lost_at is None:
                contact_lost_at = t
                events.append(dict(time_s=t, kind='ACTUAL_SELECTED_CONTACT_LOST', finger=finger))
        high_run = high_run + 1 if forces.max() > 12. else 0
        hardware = ('NON_TIP_OBJECT_COLLISION' if unsafe else
                    'SUSTAINED_DANGEROUS_FORCE' if high_run * .002 >= .020 else
                    'NONFINITE_STATE' if not np.all(np.isfinite(data.qpos)) else None)
        if hardware:
            stop = hardware
            events.append(dict(time_s=t, kind='HARDWARE_STOP', reason=hardware,
                               forces_n=forces.tolist(), non_tip=unsafe))
        if phase == 'BASELINE':
            baseline_stable = t if safe_selected and baseline_stable is None else baseline_stable
            if not safe_selected:
                baseline_stable = None
            if t >= .3:
                if baseline_stable is None or t - baseline_stable < .15:
                    stop = 'NO_STABLE_INITIAL_SELECTED_CONTACT'
                else:
                    phase, phase_started = 'WITHDRAW', t
                    nominal = last.copy()
                    mcc = BasicPassiveMCC(h, .002)
                    outward = normals[finger].copy()
                    withdrawal_start_center = data.geom_xpos[geom].copy()
                    events.append(dict(time_s=t, kind='CONTROLLED_WITHDRAW_BEGIN', finger=finger,
                                       measured_normal=outward.tolist()))
        if phase == 'WITHDRAW' and stop == 'TIME_BUDGET_REACHED':
            jac = search._finger_jacobian(data, finger)
            step_m = min(.003 * .002, .003 - commanded_withdrawal)
            dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-6 * np.eye(3), outward * max(step_m, 0.))
            dq *= min(1., .001 / max(float(np.max(abs(dq))), 1e-15))
            nominal[columns] = np.clip(last[columns] + dq, planner.lower[columns], planner.upper[columns])
            commanded_withdrawal += float(np.linalg.norm(jac @ (nominal[columns] - last[columns])))
            if commanded_withdrawal >= .00299 or t - phase_started >= 2.:
                phase, phase_started = 'WITHDRAW_HOLD', t
                events.append(dict(time_s=t, kind='CONTROLLED_WITHDRAW_END',
                    commanded_distance_m=commanded_withdrawal, actual_pad_displacement_m=measured_withdrawal))
        elif phase == 'WITHDRAW_HOLD' and t - phase_started >= .3:
            if active[finger] or contact_lost_at is None:
                stop = 'PERTURBATION_DID_NOT_LOSE_SELECTED_CONTACT'
            elif not .002 <= measured_withdrawal <= .004:
                stop = 'WITHDRAWAL_OUTSIDE_2_TO_4_MM'
            else:
                phase, phase_started = 'RECOVER', t
                recovery_started = t
                nominal = joint_state(h, data)
                nominal[:7] = held_arm
                search.rebase(nominal)
                mcc = BasicPassiveMCC(h, .002)
                events.append(dict(time_s=t, kind='RECOVERY_WINDOW_BEGIN', finger=finger,
                    active_search_enabled=bool(search_enabled), actual_pad_displacement_m=measured_withdrawal))
        elif phase == 'RECOVER':
            if safe_selected:
                if first_recontact_at is None:
                    first_recontact_at = t
                    events.append(dict(time_s=t, kind='ACTUAL_SELECTED_RECONTACT', finger=finger))
                stable_since = t if stable_since is None else stable_since
                if t - stable_since + 1e-12 >= .15 and not success:
                    success = True
                    stable_recovered_at = t
                    events.append(dict(time_s=t, kind='SAME_FINGER_STABLE_RECONTACT', finger=finger,
                                       stable_duration_s=t-stable_since))
            else:
                stable_since = None
            if t - phase_started >= recovery_window_s - 1e-12:
                stop = 'RECOVERY_WINDOW_COMPLETE'
        if hardware:
            # A simultaneous contact confirmation cannot override a physical
            # safety stop.  Preserve the terminal measurement in the ledger.
            success, stop = False, hardware
        in_relief = forces.max() >= 8 or (relief.mode == 'FORCE_RELEASE' and forces.max() >= 4.)
        relief.mode = 'FORCE_RELEASE' if in_relief else 'NORMAL'
        proposed, diag = search.update(data, nominal, forces, normals, active,
                                       enabled=search_enabled and phase == 'RECOVER' and not in_relief)
        finger_nominal = nominal.copy()
        if phase == 'RECOVER':
            # The recovery planner proposes four targets, but this controlled
            # probe applies only the deliberately withdrawn finger's proposal.
            finger_nominal[columns] = proposed[columns]
        control_active = active.copy()
        if phase in ('WITHDRAW', 'WITHDRAW_HOLD'):
            control_active[finger] = False
        command = mcc.command(data, finger_nominal, forces, normals, control_active, 2.)
        if in_relief:
            command = relief.relieve_force(h, data, command, forces, normals)
        command[:7] = held_arm
        command = np.clip(command, last - .0024, last + .0024)
        command = np.clip(command, planner.lower, planner.upper)
        if step % 5 == 0 and stop == 'TIME_BUDGET_REACHED':
            reason = planner.command_collision_reason(joint_state(h, data), command)
            if reason in ('NON_TIP_CLEARANCE', 'NON_TIP_OBJECT_COLLISION', 'SELF_COLLISION'):
                stop = 'COMMAND_GEOMETRY_STOP'
                events.append(dict(time_s=t, kind=stop, reason=reason))
        applied_search = np.zeros(4, dtype=bool)
        applied_search[finger] = diag['seeking'][finger] and phase == 'RECOVER' and not in_relief
        applied_bias, applied_delta = np.zeros(16), np.zeros(16)
        local_columns = search.columns[finger]
        if phase == 'RECOVER':
            applied_bias[local_columns] = np.array(diag['joint_bias_rad'])[local_columns]
            applied_delta[local_columns] = np.array(diag['search_delta_rad'])[local_columns]
        values = dict(time_s=t, q_rad=joint_state(h, data),
            qvel_rad_s=np.r_[data.qvel[h.arm_dof_adrs], data.qvel[h.hand_dof_adrs]],
            command_rad=command.copy(), sensor_command_rad=last.copy(),
            palm_pose_world=pose(h, data), tip_positions_world_m=tip_positions(h, data),
            contact_positions_world_m=points.copy(), normals_world=normals.copy(), forces_n=forces.copy(),
            active=active.copy(), phase=observed_phase, command_phase=phase,
            coverage=0., covered_goals=0, goal_id=-1, stroke_id=-1, stroke_distance_m=0., endpoint_offset_m=0.,
            finger_search_active=applied_search, finger_search_enabled=search_enabled and phase == 'RECOVER' and not in_relief,
            finger_search_bias_rad=applied_bias, finger_search_delta_rad=applied_delta,
            finger_search_distance_m=np.array(diag['search_distance_m']),
            finger_search_target_positions_world_m=np.array(diag['target_positions_world_m']),
            finger_search_target_normals_world=np.array(diag['target_normals_world']),
            finger_search_target_valid=np.array(diag['target_valid']),
            commanded_withdrawal_m=commanded_withdrawal, measured_withdrawal_m=measured_withdrawal,
            qacc_warmstart=data.qacc_warmstart.copy())
        for key, value in values.items():
            logs.setdefault(key, []).append(value)
        if stop != 'TIME_BUDGET_REACHED' or step == int(budget_s / .002):
            break
        last = command.copy()
        set_control(h, data, command)
        mujoco.mj_step(h.model, data)
        mujoco.mj_forward(h.model, data)
    arrays = {key: np.asarray(value) for key, value in logs.items()}
    np.savez_compressed(directory / 'trace.npz', **arrays)
    np.savez_compressed(directory / 'final_state.npz', qpos=data.qpos, qvel=data.qvel, ctrl=data.ctrl, time=data.time)
    (directory / 'events.json').write_text(json.dumps(events, indent=2), encoding='utf-8')
    summary = dict(protocol=manifest['protocol'], seed=seed, selected_finger=finger,
        success=success, stop_reason=stop, executed_s=t, wall_s=perf_counter() - started,
        peak_force_n=peak, commanded_withdrawal_m=commanded_withdrawal,
        withdrawal_m_at_search_start=next((e['actual_pad_displacement_m'] for e in events
            if e['kind'] == 'RECOVERY_WINDOW_BEGIN'), None),
        contact_lost_at_s=contact_lost_at, recovery_started_s=recovery_started,
        first_recontact_at_s=first_recontact_at,
        active_search_enabled=bool(search_enabled), recovery_observation_window_s=recovery_window_s,
        first_recontact_delay_s=first_recontact_at-recovery_started if first_recontact_at is not None else None,
        stable_recovered_at_s=stable_recovered_at,
        search_to_stable_confirmation_s=stable_recovered_at-recovery_started if success else None,
        observed_outcome='STABLE_RECONTACT_OBSERVED' if success else 'NO_STABLE_RECONTACT_WITHIN_WINDOW',
        interpretation='Bounded single-finger observation; not whole-traversal success or failure',
        final_selected_force_n=float(forces[finger]),
        actual_selected_joint_motion_rad=np.ptp(arrays['q_rad'][:, columns], axis=0).tolist(),
        maximum_arm_command_change_rad=float(np.max(abs(arrays['command_rad'][:, :7] - held_arm))),
        whole_traversal_evidence=False, coverage_counted=False,
        trace_sha256=sha256((directory / 'trace.npz').read_bytes()).hexdigest())
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('FINGER_RECOVERY_PROBE', json.dumps(summary), flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--finger', type=int, default=0, choices=range(4))
    parser.add_argument('--budget', type=float, default=12.)
    parser.add_argument('--search-off', action='store_true')
    parser.add_argument('--recovery-window', type=float, default=5.)
    args = parser.parse_args()
    run(args.output, seed=args.seed, finger=args.finger, budget_s=args.budget,
        search_enabled=not args.search_off, recovery_window_s=args.recovery_window)
