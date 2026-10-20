"""Local causal probe at the archived B failure boundary, not a new full trial."""
from __future__ import annotations

import argparse
from collections import deque
from hashlib import sha256
import json
from pathlib import Path

import mujoco
import numpy as np

from .matched_mcc import MatchedMCCTraversal
from .physics import set_control, set_joint_state
from .published_whole_hand import install_asset_adapter
from .whole_hand_surface_traversal import WholeHandTraversal


def run(case, output):
    install_asset_adapter()
    output.mkdir(parents=True, exist_ok=False)
    trace = case/'trace_0010.npz'
    with np.load(trace) as archive:
        z = {name: archive[name] for name in archive.files}
    k = int(np.argmin(abs(z['time_s']-212.210)))
    assert abs(z['time_s'][k]-212.210) < 1e-8
    variants = {}
    for mode in ('archived_reprojection', 'retain_previous_points', 'blend_points_0p5s'):
        runner = MatchedMCCTraversal(output/mode, 'B-MCC', 213.211, local_only=True)
        d = runner.data
        set_joint_state(runner.h, d, z['q_rad'][k])
        d.qvel[:] = z['qvel_rad_s'][k]
        set_control(runner.h, d, z['sensor_command_rad'][k])
        d.qacc_warmstart[:] = z['qacc_warmstart'][k]
        d.time = float(z['time_s'][k])
        mujoco.mj_forward(runner.h.model, d)
        runner.last = z['sensor_command_rad'][k].copy()
        runner.normal_filter = z['target_normals_world'][k-1].copy()
        runner.clearance = z['target_clearance_m'][k-1].copy()
        runner.bias = z['tangential_seam_bias_m'][k-1].copy()
        runner.offsets = z['mcc_offset_m'][k-1].copy()
        runner.window = deque(z['safe_belly'][k-150:k].copy(), maxlen=150)
        runner.anchor = int(z['leader_finger'][k])
        runner.goal = int(z['goal_vertex'][k])
        runner.public_wrist = z['public_wrist_pose_world'][k].copy()
        runner.reference_index = int(z['public_reference_index'][k])
        for forces in z['forces_n'][:k]:
            runner.release |= forces > 2.5
            runner.release &= forces >= 1.2
        velocities = np.clip((z['mcc_offset_m'][k-1]-z['mcc_offset_m'][k-2])/.002, -.00025, .00025)
        velocities[runner.release] = 0.
        for i, controller in enumerate(runner.controllers):
            controller.reset(offset_m=float(runner.offsets[i]), velocity_m_s=float(velocities[i]))
        delta = z['command_rad'][k-1]-z['sensor_command_rad'][k-1]
        limits = np.r_[np.full(7, .0004), np.full(16, .0008)]
        lo = np.maximum(-limits, runner.planner.lower-z['sensor_command_rad'][k-1])
        hi = np.minimum(limits, runner.planner.upper-z['sensor_command_rad'][k-1])
        active = np.zeros(23, int)
        active[np.isclose(delta, lo, atol=1e-12, rtol=0)] = -1
        active[np.isclose(delta, hi, atol=1e-12, rtol=0)] = 1
        active[lo == hi] = 2
        runner.step_solver.active = active
        runner.step_solver.previous_delta = delta.copy()
        nominal = runner.last.copy()
        projected, normals = runner.planner.project_pads(nominal)
        assert np.max(abs(projected-z['target_points_world'][k])) < 1e-12
        runner.method_normals = normals.copy()
        previous = z['target_points_world'][k-1].copy()
        initial_force_error = float(np.max(abs(runner.measure()['f']-z['forces_n'][k])))
        assert initial_force_error < 1e-8, initial_force_error
        np.savez_compressed(runner.out/'initial.npz', qpos=d.qpos, qvel=d.qvel, ctrl=d.ctrl,
                            qacc_warmstart=d.qacc_warmstart, time=d.time)
        np.savez_compressed(runner.out/'controller_initial.npz',
                            last=runner.last, normal_filter=runner.normal_filter,
                            clearance=runner.clearance, bias=runner.bias, offsets=runner.offsets,
                            mcc_velocity=velocities, release=runner.release, safe_window=np.asarray(runner.window),
                            solver_active=active, solver_previous_delta=delta,
                            projected_points=projected, previous_points=previous, planned_normals=normals, nominal=nominal)
        manifest = json.loads((runner.out/'manifest.json').read_text())
        manifest.update(protocol='i04.mcc.local-causal-probe.v1',
                        scope='LOCAL_STATE_RESTORED_DIAGNOSTIC_NOT_FULL_TRAVERSAL',
                        initial_state='Restored at 212.210s from the declared source trace, not a fresh t=0 trial',
                        source_case=str(case), source_trace_sha256=sha256(trace.read_bytes()).hexdigest(),
                        source_trace_index=k, coverage_ledger_is_diagnostic_only=True, mode=mode)
        (runner.out/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        stop = 'LOCAL_PROBE_1S_LIMIT'
        try:
            for step in range(500):
                if mode == 'archived_reprojection':
                    points = projected
                elif mode == 'retain_previous_points':
                    points = previous
                else:
                    alpha = min(1., step*.002/.5)
                    points = previous+alpha*(projected-previous)
                WholeHandTraversal.contact_step(runner, points, normals, nominal, 'LOCAL_CAUSAL_PROBE')
        except RuntimeError as error:
            stop = str(error)
        logs = {name: np.asarray(value) for name, value in runner.logs.items()}
        runner.flush()
        count = len(logs['time_s'])
        overlap = min(count, len(z['time_s'])-k)
        errors = {name: float(np.max(abs(logs[name][:overlap]-z[name][k:k+overlap])))
                  for name in ('q_rad', 'qvel_rad_s', 'command_rad', 'forces_n', 'mcc_offset_m', 'target_normals_world')}
        # Zero-coverage local branch; never counted as a continuation or a full result.
        result = dict(mode=mode, start_s=float(z['time_s'][k]), final_observation_s=float(logs['time_s'][-1]),
                      observations=count, stop_reason=stop, initial_force_error_n=initial_force_error,
                      peak_force_n=float(logs['forces_n'].max()), comparison_overlap_states=overlap,
                      common_106ms_peak_force_n=float(logs['forces_n'][:overlap].max()),
                      final_forces_n=logs['forces_n'][-1].tolist(),
                      observations_at_or_above_8n=int((logs['forces_n'].max(axis=1) >= 8.).sum()),
                      archived_overlap_max_errors=errors,
                      scope='Local state-restored diagnostic; not a full traversal or a paired A/B result',
                      only_changed_input='Per-finger nominal contact points; normals, nominal joints, MCC, guards and thresholds unchanged',
                      source_trace_sha256=sha256(trace.read_bytes()).hexdigest())
        (output/mode/'probe_result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        variants[mode] = result
        print(json.dumps(result), flush=True)
    result = dict(source_case=str(case), start_index_in_trace=k,
                  point_jump_mm=(np.linalg.norm(z['target_points_world'][k]-z['target_points_world'][k-1], axis=1)*1000).tolist(),
                  variants=variants)
    actual = variants['archived_reprojection']
    result['archived_reproduction_passed'] = bool(
        actual['stop_reason'] == 'SUSTAINED_DANGEROUS_FORCE'
        and abs(actual['final_observation_s']-z['time_s'][-1]) < 1e-8
        and actual['archived_overlap_max_errors']['forces_n'] < 1e-6
        and actual['archived_overlap_max_errors']['command_rad'] < 1e-8)
    (output/'probe.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    assert result['archived_reproduction_passed'], 'Local controller restoration did not reproduce the archived branch'
    print('REPRODUCTION_PASSED', result['archived_reproduction_passed'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.case, args.output)


if __name__ == '__main__':
    main()
