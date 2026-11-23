"""i01-a: matched finger-control trials on a wrist-only surface route."""
import argparse
from contextlib import nullcontext
from pathlib import Path
import subprocess
import sys
from time import perf_counter, perf_counter_ns

import mujoco
import numpy as np

from .fingertip_benchmark import digest, read, write_reconstruction
from .fingertip_dp_compare import CONTROL_STEPS, LOOKAHEAD_STEPS, policy_observation
from .fingertip_dp_worker import POLICY, ROOT
from .fingertip_hand import DT, FingertipHand, build_model, reference_grasp
from .fingertip_traversal import Traversal, latency_ms, surface_graph, write_json
from .i01_a_wrist import (ContactWristSchedule, STARTUP_S, UPDATE_STEPS, plan_wrist,
                         wrist_digest)
from .physics import rotation


EXPERIMENT = 'i01-a'
FIXED_BRANCHES = ('fixed-grasp', 'fixed-mcc', 'fixed-dp')
BRANCHES = (*FIXED_BRANCHES, 'adaptive-mcc', 'adaptive-dp')
ADAPTIVE_BUDGET_SCALE = 1.5
VIDEO_LABELS = {
    'fixed-grasp': 'i01-a | Fixed wrist | Fixed grasp (position servo)',
    'fixed-mcc': 'i01-a | Fixed wrist | MCC force feedback',
    'fixed-dp': 'i01-a | Fixed wrist | single_all + MCC',
    'adaptive-mcc': 'i01-a | Tactile wrist feedback | Fixed grasp + MCC',
    'adaptive-dp': 'i01-a | Tactile wrist feedback | single_all + MCC',
}


def calibrate(output):
    import torch

    policy_dir = ROOT / 'finger_policy'
    release = read(policy_dir / 'checkpoints/manifest.json')[POLICY]
    checkpoint = policy_dir / release['path']
    if digest(checkpoint) != release['sha256']:
        raise ValueError('Released checkpoint hash mismatch')
    bundle = torch.load(checkpoint, map_location='cpu', weights_only=False)
    norm = bundle['normalization']
    mean, std = (np.asarray(norm[key])[:, 33:] for key in ('mean', 'std'))
    if not (np.all(mean == mean[0]) and np.all(std == std[0]) and np.all(std > 0.)):
        raise ValueError('Expected the common twelve wrist-observation features')
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, dict(source='released_policy_training_normalization',
        policy=POLICY, checkpoint_sha256=release['sha256'],
        components=['wrist_linear_velocity', 'wrist_angular_velocity',
                    'wrist_delta_translation_200ms', 'wrist_delta_rotvec_200ms'],
        mean=mean[0].tolist(), std=std[0].tolist()))


def save_route(output, calibration, tape):
    info = dict(tape.geometry, calibration_sha256=digest(calibration),
                nominal_wrist_sha256=wrist_digest(tape.poses),
                nominal_wrist_steps=len(tape.poses), dt_s=DT)
    np.savez_compressed(output / 'nominal_wrist.npz', palm_command=tape.poses, arc_m=tape.arc_m)
    write_json(output / 'wrist_plan.json', info)
    return info


def check_plan(calibration, output):
    position, orientation, joints = reference_grasp()
    hand = FingertipHand(build_model())
    points = hand.measure(hand.initialize(position, orientation, joints)).points
    start = perf_counter()
    tape = plan_wrist(surface_graph().mesh, position, orientation, points, read(calibration))
    elapsed = perf_counter() - start
    output.mkdir(parents=True, exist_ok=False)
    info = save_route(output, calibration, tape)
    print(f'Nominal route {info}, planner time {elapsed:.3f} s', flush=True)


def run_trial(args):
    initial_setup = perf_counter()
    prepared = None
    common = {}
    if args.prepared is not None:
        from .i01_a_full_wrist import load_prepared

        prepared = load_prepared(args.prepared)
        if digest(args.calibration) != prepared.report['calibration_sha256']:
            raise ValueError('Full-surface wrist calibration has changed')
        common = dict(hand=prepared.hand, graph=prepared.graph, grasp=prepared.grasp,
                      surface=prepared.surface, object_name=prepared.report['object_name'])
    # Reuse the same native plant, measurement and MCC as i01-b.
    run = Traversal(args.output, args.seconds or 3600., planner_type=None,
                    collect_contacts=True, experiment=EXPERIMENT, **common)
    hand, data = run.h, run.data
    touch = hand.measure(data)
    initial_setup_s = perf_counter() - initial_setup

    planning_start = perf_counter()
    if prepared is None:
        position, orientation, q_initial = reference_grasp()
        tape = plan_wrist(run.graph.mesh, position, orientation, touch.points, read(args.calibration))
        wrist_planning_s = perf_counter() - planning_start
    else:
        q_initial = prepared.grasp[2]
        tape, wrist_planning_s = prepared.tape, 0.
    plan = save_route(args.output, args.calibration, tape)
    adaptive = args.branch.startswith('adaptive')
    budget = int(np.ceil(len(tape.poses) * ADAPTIVE_BUDGET_SCALE)) if adaptive else len(tape.poses)
    if args.seconds is not None:
        budget = min(budget, round(args.seconds / DT))
    schedule = ContactWristSchedule(tape, budget, LOOKAHEAD_STEPS) if adaptive else None
    dp = args.branch.endswith('-dp')
    mcc_enabled = args.branch != 'fixed-grasp'
    controller = None
    if dp:
        from .fingertip_local_policy import LocalFingerPolicy

        controller = LocalFingerPolicy(args.seed, (budget + CONTROL_STEPS - 1) // CONTROL_STEPS)
    manifest = read(args.output / 'manifest.json')
    label = VIDEO_LABELS[args.branch]
    if prepared is not None:
        label = label.replace('i01-a', 'i01-a ' + prepared.report['object_name'])
        manifest['source_grasp'] = prepared.report['source']
    manifest.update(branch=args.branch, video_label=label,
                    planning=dict(plan, mode='geometry_wrist_only', motion_variables=6,
                                  variables_optimized_per_knot=plan.get('variables_optimized_per_knot',
                                                                      3 if prepared is None else 1),
                                  wrist_feedback=adaptive, fixed_finger_target_trajectory=False),
                    requested_seconds=budget * DT,
                    wrist_feature_horizon_s=LOOKAHEAD_STEPS * DT,
                    sustained_force_behavior='record_and_continue_in_offline_benchmark')
    manifest['controller'].update(dp_inference=dp, mcc_feedback_enabled=mcc_enabled,
                                  nominal_finger_source=(POLICY if dp else
                                      'constant_initial_measured_joint_angles'))
    if not mcc_enabled:
        manifest['controller']['implementation'] = 'MuJoCo position servo'
    if controller is not None:
        manifest['policy'] = controller.info
    write_json(args.output / 'manifest.json', manifest)

    nominal = q_initial
    cursor = np.empty(budget)
    feedback_times, mcc_times = [], np.empty(budget if mcc_enabled else 0)
    first_high, longest_high = None, 0
    print(f'{args.branch}: {plan["path_length_m"]:.3f} m, '
          f'{plan["nominal_total_s"]:.3f} s nominal ({wrist_planning_s:.3f} s planning)', flush=True)
    motion_start = perf_counter()
    with controller if controller is not None else nullcontext():
        for step in range(budget):
            control_tick = step % CONTROL_STEPS == 0
            if schedule is not None and step % UPDATE_STEPS == 0:
                began = perf_counter()
                schedule.extend(step, touch)
                feedback_times.append((perf_counter() - began) * 1000.)
            poses = tape.poses if schedule is None else schedule.poses
            wrist = poses[step]
            cursor[step] = step if schedule is None else schedule.trace[step, 0]
            phase = ('prepare' if cursor[step] * DT < STARTUP_S else
                     'scan' if cursor[step] < tape.scan_end else 'settle')
            if control_tick:
                outward = rotation(wrist[3:]).apply(-hand.inward_in_palm)
                normals = np.broadcast_to(outward, (4, 3))
                if controller is not None:
                    future_step = min(step + LOOKAHEAD_STEPS - 1, budget - 1)
                    if schedule is not None and future_step >= schedule.published:
                        raise RuntimeError('Finger Policy was given an unpublished wrist command')
                    packet = policy_observation(hand, data, poses[future_step])
                    nominal = controller.command(step // CONTROL_STEPS, packet)
            ctrl = nominal
            if mcc_enabled:
                began = perf_counter()
                ctrl = run.feedback.command(hand, data, nominal, normals, touch, track=True)
                mcc_times[step] = (perf_counter() - began) * 1000.
            data.mocap_pos[hand.driver], data.mocap_quat[hand.driver], data.ctrl[:] = wrist[:3], wrist[3:], ctrl
            mujoco.mj_step(hand.model, data)
            mujoco.mj_forward(hand.model, data)
            if not np.isfinite(data.qpos).all() or abs(data.time - (step + 1) * DT) > 1.e-6:
                raise RuntimeError(f'Invalid native state at step {step + 1}')
            touch = hand.measure(data)
            run.record(touch, wrist, ctrl, phase)
            longest_high = max(longest_high, run.metrics.high_steps)
            if first_high is None and run.metrics.high_steps >= round(.2 / DT):
                first_high = dict(step=step + 1, time_s=(step + 1) * DT, region_load_n=touch.loads.tolist())
            if (step + 1) % 5000 == 0:
                print(f'{args.branch} {(step + 1) * DT:.1f} simulated s, '
                      f'{tape.arc_m[int(cursor[step])] / tape.arc_m[-1]:.0%} of route', flush=True)
            if cursor[step] >= len(tape.poses) - 1:
                break
    motion_wall_s = perf_counter() - motion_start
    run.flush()
    count = run.metrics.steps
    progress = np.interp(cursor[:count], np.arange(len(tape.poses)), tape.arc_m)
    signals = dict(nominal_cursor=cursor[:count], route_progress_m=progress,
                   mcc_all_steps_ms=mcc_times[:count], wrist_feedback_ms=feedback_times)
    if schedule is not None:
        signals.update(rate_and_correction=schedule.trace[:count, 1:])
    np.savez_compressed(args.output / 'wrist_progress.npz', **signals)
    coverage = run.metrics.coverage
    np.savez_compressed(args.output / 'coverage.npz', visited_vertices=coverage.visited_vertices,
                        visited_goals=coverage.visited_goals, required_vertices=coverage.graph.required_vertices)
    timing = dict(mcc=latency_ms(mcc_times[:count]), wrist_feedback=latency_ms(feedback_times))
    if controller is not None:
        timing.update(controller.report(args.output))
    completed = cursor[count - 1] >= len(tape.poses) - 1
    summary = dict(stop_reason='WRIST_TRAJECTORY_COMPLETE' if completed else 'TIME_BUDGET',
        longest_over_40n_s=longest_high * DT, first_sustained_overload_stop=first_high,
        live_head_observations=run.contact_tape.save(args.output), control_timing=timing,
        progress_fraction=float(progress[-1] / tape.arc_m[-1]), **run.metrics.summary())
    write_json(args.output / 'summary.json', summary)
    print(f'{args.branch}: {count * DT:.3f} simulated s, {summary["progress_fraction"]:.1%} of route, R-GPIS',
          flush=True)
    reconstruction_s = write_reconstruction(args.output) if prepared is None else write_reconstruction(
        args.output, prepared.surface)
    written_ns = perf_counter_ns()
    write_json(args.output / 'task_written.json', dict(experiment=EXPERIMENT, written_ns=written_ns,
        branch=args.branch, seed=args.seed, paced=False, states=count, executed_s=count * DT,
        wrist_generation_s=wrist_planning_s, wrist_feedback_s=sum(feedback_times) / 1000.,
        initial_setup_s=initial_setup_s, motion_wall_s=motion_wall_s, reconstruction_s=reconstruction_s,
        wrist_sha256=wrist_digest(poses[:count]), nominal_wrist_sha256=plan['nominal_wrist_sha256']))


def suite(args):
    if args.cpu_ids:
        import psutil

        psutil.Process().cpu_affinity(args.cpu_ids)
    args.root.mkdir(parents=True, exist_ok=True)
    if not args.calibration.exists():
        calibrate(args.calibration)
    for repeat in range(args.repeats):
        order = args.branches if repeat % 2 == 0 else args.branches[::-1]
        for branch in order:
            tag = f'{branch}_r{repeat + 1:02d}'
            case = args.root / tag
            if (case / 'stopwatch.json').exists():
                continue
            if case.exists():
                raise FileExistsError(f'Use a new directory for interrupted trial {case}')
            command = [sys.executable, '-u', '-m', __spec__.name, 'trial', '--branch', branch,
                       '--output', str(case), '--seed', str(args.seed + repeat),
                       '--calibration', str(args.calibration)]
            if args.seconds is not None:
                command.extend(('--seconds', str(args.seconds)))
            if args.prepared is not None:
                command.extend(('--prepared', str(args.prepared)))
            print(f'Start {tag}', flush=True)
            with (args.root / f'{tag}.log').open('wb') as log:
                started = perf_counter_ns()
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
                exited = perf_counter_ns()
            if result.returncode:
                raise RuntimeError(f'{tag} exited with {result.returncode}; read {tag}.log')
            stamp = read(case / 'task_written.json')
            if not started < stamp['written_ns'] < exited:
                raise ValueError(f'Invalid external stopwatch for {tag}')
            stamp.update(repeat=repeat + 1, seconds_to_written=(stamp['written_ns'] - started) * 1.e-9,
                         seconds_through_process_exit=(exited - started) * 1.e-9, cpu_affinity=args.cpu_ids)
            write_json(case / 'stopwatch.json', stamp)
            print(f'End {tag}: {stamp["seconds_to_written"]:.3f} s to written result', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    calibration = commands.add_parser('calibrate')
    calibration.add_argument('--output', type=Path, required=True)
    plan = commands.add_parser('plan')
    full = commands.add_parser('prepare-full')
    full.add_argument('--object', choices=('mustard', 'pear', 'cracker'), required=True)
    full.add_argument('--wrist-planner', choices=('single-anchor', 'multi-contact', 'policy-footprint'),
                      default='policy-footprint')
    single = commands.add_parser('trial')
    batch = commands.add_parser('suite')
    for action in (plan, full, single, batch):
        action.add_argument('--calibration', type=Path, required=True)
    for action in (plan, full, single):
        action.add_argument('--output', type=Path, required=True)
    for action in (single, batch):
        action.add_argument('--seconds', type=float)
        action.add_argument('--seed', type=int, default=20261109)
        action.add_argument('--prepared', type=Path, help='Use an independently prepared full-surface wrist')
    single.add_argument('--branch', choices=BRANCHES, required=True)
    batch.add_argument('--root', type=Path, required=True)
    batch.add_argument('--branches', nargs='+', choices=BRANCHES, default=FIXED_BRANCHES)
    batch.add_argument('--repeats', type=int, default=2)
    batch.add_argument('--cpu-ids', nargs='+', type=int)
    args = parser.parse_args()
    if getattr(args, 'prepared', None) is not None:
        selected = (args.branch,) if args.action == 'trial' else args.branches
        if not set(selected) <= set(FIXED_BRANCHES):
            parser.error('--prepared requires the three fixed-reference controls')
    if getattr(args, 'seconds', None) is not None and (not np.isfinite(args.seconds) or args.seconds < DT):
        parser.error('--seconds must specify at least one 2 ms step')
    if args.action == 'calibrate':
        calibrate(args.output)
    elif args.action == 'plan':
        check_plan(args.calibration, args.output)
    elif args.action == 'prepare-full':
        from .i01_a_full_wrist import prepare

        prepare(args.object, args.output, args.calibration, args.wrist_planner)
    elif args.action == 'trial':
        run_trial(args)
    else:
        suite(args)


if __name__ == '__main__':
    main()
