"""Run a new acitive-02 pair with full-object tactile coverage and native replay."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from hashlib import sha256
import json
from pathlib import Path
import shutil
from time import perf_counter

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .aude import DEFAULT_STEP_M, AudeExplorer, state_posture
from .experiment import (
    ARMS, CONTROL_DT, CONTROL_STEPS, DT, EXPERIMENT, HIGH_LOAD_N, HISTORY_TICKS,
    LOOKAHEAD_STEPS, PLAN_TICKS, START_SOURCE, STATE, digest, longest_run,
    pair_runtime, policy_observation, read_json, record, step_plant, write_json,
)
from .surface_scan import (GAP_M, ROW_SPACING_RAD, SURFACE_LOAD_N, SWEEP_SPEED_M_S,
                           SurfaceScanner, TactileSurface, sweep_directions)
from .wrist import WristFeedback
from ..i04_geometric_traversal.fingertip_traversal import ForceFeedback, Leg, latency_ms
from ..i04_geometric_traversal.physics import quaternion, rotation

BASE = Path('Module/generated/acitive-02_wrist_mcc_20261120')
PROFILE = 'full-surface'
TRACE_KEYS = (
    'qpos', 'wrist_command', 'wrist_nominal', 'wrist_offset', 'wrist_wrench_feedback',
    'palm_proximity_drive_n', 'finger_nominal', 'aude_joint_nominal', 'finger_command',
    'points_m', 'normals_outward', 'pad_force_n', 'region_load_n', 'pad_valid', 'mode', 'wall_s',
)


def blend_wrist(start, end):
    u = np.arange(1, CONTROL_STEPS + 1) / CONTROL_STEPS
    first = rotation(start[3:])
    relative = (first.inv() * rotation(end[3:])).as_rotvec()
    turns = (first * Rotation.from_rotvec(u[:, None] * relative)).as_quat()[:, [3, 0, 1, 2]]
    return np.column_stack((start[:3] + u[:, None] * (end[:3] - start[:3]), turns))


def aude_segment(explorer, data, touch, wrist_ref, second):
    started = perf_counter()
    plan = explorer.plan(data, touch)
    here = state_posture(explorer.hand, data)
    here.position = wrist_ref[:3]
    here.orientation = rotation(wrist_ref[3:])
    position, orient, joints = Leg(here, plan.posture, plan.mode).samples(PLAN_TICKS * CONTROL_STEPS)
    entry = dict(time_s=second, mode=plan.mode,
                 wrist_target=np.r_[plan.posture.position, quaternion(plan.posture.orientation)].tolist(),
                 aude_joint_target=plan.posture.joints.tolist(), virtual_contacts_m=plan.targets.tolist(),
                 live_pad_valid=plan.valid.tolist(), ik_error_mm=plan.error_mm,
                 gp_contacts=len(explorer.gp.surface), planning_ms=(perf_counter() - started) * 1000.)
    return np.column_stack((position, orient)), joints, entry


def new_scanner(hand, mode, surface, wrist):
    axes = ((0, 0., (.22, -.57, -.79), 2), (1, .026, (-.52, .39, -.76), 3),
            (2, -.019, (.20, -.71, -.68), 2))
    axis, angle, local, preferred = axes[mode]
    direction = sweep_directions(axis, angle)
    scanner = SurfaceScanner(hand, CONTROL_DT, direction, local, preferred)
    scanner.surface = surface
    scanner.start(wrist)
    return scanner


def run_arm(job):
    root, case, arm, pose, joints, timing = job
    case.mkdir(exist_ok=False)
    since = perf_counter()
    hand, ik = pair_runtime(root / 'model.mjb', pose, joints)
    with np.load(root / 'initial.npz', allow_pickle=False) as seed_data:
        state = seed_data['integration_state']
        history = seed_data['warm_policy_observations']
        normals = seed_data['outward_normals']
    data = mujoco.MjData(hand.model)
    mujoco.mj_setState(hand.model, data, state, STATE)
    mujoco.mj_forward(hand.model, data)
    explorer = AudeExplorer(hand, ik, timing['seed'], DEFAULT_STEP_M)
    feedback = ForceFeedback()
    wrist_mcc = WristFeedback()
    touch = hand.measure(data)
    explorer.observe(data, touch)
    surface = TactileSurface()
    surface.add(touch)
    here = state_posture(hand, data)
    nominal = np.r_[here.position, quaternion(here.orientation)]
    joint_nominal = here.joints.copy()
    rows, plans, passes = [], [], []
    local_ticks = int(round(timing['local_s'] / CONTROL_DT))
    max_ticks = int(round(timing['max_s'] / CONTROL_DT))
    policy = None
    if arm == ARMS[1]:
        from ..i04_geometric_traversal.fingertip_local_policy import LocalFingerPolicy

        policy = LocalFingerPolicy(timing['seed'], max_ticks + HISTORY_TICKS)
    scanner = None
    tick = n_passes = 0
    stopped = 'time_budget'
    grid = sweep_directions()
    with (policy if policy is not None else nullcontext()), (case / 'plans.jsonl').open('w', encoding='utf-8') as log:
        if policy is not None:
            for frame, packet in enumerate(history):
                policy.command(frame, packet)
        rows.append(record(data, hand, touch, nominal, wrist_mcc, joint_nominal, joint_nominal, 'initial', perf_counter() - since))
        while tick < max_ticks:
            if tick % PLAN_TICKS == 0 or tick == local_ticks:
                previous_plan_tick = tick
                path, q_curve, detail = aude_segment(explorer, data, touch, nominal, tick * CONTROL_DT)
                detail['wrist_source'] = 'tactile_surface_scan' if tick >= local_ticks else 'aude_joint_ik'
                plans.append(detail)
                log.write(json.dumps(detail, allow_nan=False) + '\n')
                log.flush()
            phase_tick = tick - previous_plan_tick
            step = phase_tick * CONTROL_STEPS
            aude_q = q_curve[min(step + CONTROL_STEPS - 1, len(q_curve) - 1)]
            if tick < local_ticks:
                mode = plans[-1]['mode']
                segment = path[step:step + CONTROL_STEPS]
                future = path[min(step + LOOKAHEAD_STEPS - 1, len(path) - 1)]
            else:
                if scanner is None or scanner.done:
                    if scanner is not None:
                        gaps = surface.uncovered(grid)
                        passes.append(dict(pass_number=n_passes, time_s=tick * CONTROL_DT,
                                           directions_total=len(scanner.directions), observed=scanner.sampled,
                                           unresolved_waypoints=scanner.skipped, remaining_tactile_rays=int(gaps.sum())))
                        print(f'{arm} pass {n_passes} at {tick * CONTROL_DT:.1f}s; {gaps.sum()} unresolved tactile rays', flush=True)
                        if not gaps.any() or n_passes >= timing['passes']:
                            stopped = 'full_tactile_grid' if not gaps.any() else 'scan_pass_budget'
                            break
                    scanner = new_scanner(hand, n_passes, surface, nominal)
                    n_passes += 1
                planned = scanner.command(data, touch, wrist_mcc.offset)
                mode = planned.phase
                segment = blend_wrist(nominal, planned.wrist)
                # The scan is generated online. Extrapolate the issued short
                # wrist motion for the unchanged policy lookahead interface.
                future = planned.wrist.copy()
                velocity = (segment[-1, :3] - nominal[:3]) / CONTROL_DT
                future[:3] += velocity * (LOOKAHEAD_STEPS * CONTROL_DT / CONTROL_STEPS)
            joint_nominal = aude_q
            if policy is not None:
                packet = policy_observation(hand, data, wrist_mcc.forecast(future))
                joint_nominal = policy.command(HISTORY_TICKS + tick, packet)
            normals[touch.measured_heads] = touch.normals[touch.measured_heads]
            touch = step_plant(hand, data, feedback, wrist_mcc, touch, segment, joint_nominal, normals)
            nominal = segment[-1]
            tick += 1
            explorer.observe(data, touch)
            surface.add(touch)
            rows.append(record(data, hand, touch, nominal, wrist_mcc, joint_nominal, aude_q,
                               mode, perf_counter() - since))
            if not np.isfinite(data.qpos).all() or abs(data.time - tick * CONTROL_DT) > 1e-6:
                raise RuntimeError(f'{arm} invalid native state at control tick {tick}')
            if tick % 2500 == 0:
                scan_now = f' pass {n_passes} {scanner.index}/{len(scanner.directions)}' if scanner else ''
                print(f'{arm} {tick * CONTROL_DT:.1f}s; {mode}{scan_now}; {len(surface.samples)} live contact cells',
                      flush=True)
    trace = {key: np.asarray(value) for key, value in zip(TRACE_KEYS, zip(*rows))}
    np.savez_compressed(case / 'trace.npz', **trace)
    path = trace['qpos'][:, :7]
    translation = np.linalg.norm(np.diff(path[:, :3], axis=0), axis=1)
    load = trace['region_load_n'][1:].max(axis=1)
    count = trace['pad_valid'][1:].sum(axis=1)
    budget = timing['max_s']
    elapsed = tick * CONTROL_DT
    unresolved = surface.uncovered(grid)
    result = dict(experiment=EXPERIMENT, profile=PROFILE, arm=arm, initial_state_sha256=sha256(state.tobytes()).hexdigest(),
                  start_frame=timing['start'], seconds=elapsed, budget_s=budget, stop=stopped,
                  wall_execution_s=perf_counter() - since, contact_voxels=len(surface.samples),
                  tactile_grid_unresolved=int(unresolved.sum()), tactile_grid_rays=len(unresolved),
                  scan_passes=passes, wall_at_last_sample_s=float(trace['wall_s'][-1]),
                  max_region_load_n=float(load.max(initial=0.)),
                  max_palm_load_n=float(trace['region_load_n'][1:, -1].max(initial=0.)),
                  longest_above_40n_s=longest_run(load > HIGH_LOAD_N) * CONTROL_DT,
                  longest_full_loss_s=longest_run(count == 0) * CONTROL_DT, wrist_translation_m=float(translation.sum()),
                  at_least_one_contact=float(np.mean(count > 0)), at_least_three_contacts=float(np.mean(count >= 3)),
                  four_contacts=float(np.mean(count == 4)), planning_timing=latency_ms([p['planning_ms'] for p in plans]),
                  aude_nominal_difference_max_rad=float(np.max(abs(trace['finger_nominal'] - trace['aude_joint_nominal']))))
    if policy is not None:
        result['policy'] = policy.info
        result['policy_timing'] = policy.report(case)
    write_json(case / 'run.json', result)
    print(f'{arm} finished {elapsed:.1f}s simulated, {result["wall_execution_s"]:.1f}s wall, {stopped}', flush=True)
    return result


def run_full(args):
    if (not np.isfinite([args.max_seconds, args.local_seconds]).all()
            or not .9 <= args.local_seconds < args.max_seconds or not 1 <= args.passes <= 3):
        raise ValueError('Expected a positive local stage inside the overall budget and one to three scan passes')
    # Quantize once so switching stages keeps the Aude segment boundary exact.
    local_s = round(args.local_seconds / (PLAN_TICKS * CONTROL_DT)) * PLAN_TICKS * CONTROL_DT
    if local_s >= args.max_seconds:
        raise ValueError('The budget must allow a scan after the local stage')
    timing = dict(seed=args.seed, start=args.start, local_s=local_s, max_s=args.max_seconds, passes=args.passes)
    output = args.output.resolve()
    base = args.base.resolve()
    initial = base / f'frame-{args.start:05d}' / 'initial.npz'
    with np.load(START_SOURCE, allow_pickle=False) as raw:
        pose = raw['input_palm_pose_world'][args.start, 0].astype(float)
        joints = raw['input_q_live'][args.start, 0].astype(float)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(base / 'model.mjb', output / 'model.mjb')
    shutil.copyfile(initial, output / 'initial.npz')
    manifest = dict(experiment=EXPERIMENT, profile=PROFILE, base_source=base.name,
                    base_protocol_sha256=digest(base / 'protocol.json'),
                    model_sha256=digest(output / 'model.mjb'), initial_sha256=digest(output / 'initial.npz'),
                    arms=list(ARMS), source_sha256={name: digest(Path(__file__).with_name(name))
                                                    for name in ('full.py', 'surface_scan.py', 'aude.py', 'wrist.py')},
                    simulation_dt_s=DT, control_dt_s=CONTROL_DT, **timing,
                    radial_gap_m=GAP_M, row_spacing_rad=ROW_SPACING_RAD, sweep_speed_m_s=SWEEP_SPEED_M_S,
                    object_mesh_input_to_scanner=False, shared_guard='one-step local MuJoCo collision preview',
                    qualified_surface_region_load_n=SURFACE_LOAD_N, execution_workers=args.workers)
    write_json(output / 'protocol.json', manifest)
    jobs = [(output, output / arm, arm, pose, joints, timing) for arm in ARMS
            if args.arm == 'both' or arm == args.arm]
    if args.workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(run_arm, jobs))
    else:
        results = [run_arm(job) for job in jobs]
    write_json(output / 'runs.json', results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    run = actions.add_parser('run')
    run.add_argument('--base', type=Path, default=BASE)
    run.add_argument('--output', type=Path, required=True)
    run.add_argument('--start', type=int, default=0)
    run.add_argument('--local-seconds', type=float, default=27.)
    run.add_argument('--max-seconds', type=float, default=1650.)
    run.add_argument('--passes', type=int, default=2)
    run.add_argument('--seed', type=int, default=20261120)
    run.add_argument('--workers', type=int, default=2)
    run.add_argument('--arm', choices=('both', *ARMS), default='both')
    for name in ('score', 'video'):
        sub = actions.add_parser(name)
        sub.add_argument('--output', type=Path, required=True)
        if name == 'score':
            sub.add_argument('--coverage-only', action='store_true')
    args = parser.parse_args()
    if args.action == 'run':
        run_full(args)
    elif args.action == 'score':
        from .full_report import score_full

        score_full(args.output.resolve(), args.coverage_only)
    else:
        from .full_report import render_full

        render_full(args.output.resolve())


if __name__ == '__main__':
    main()
