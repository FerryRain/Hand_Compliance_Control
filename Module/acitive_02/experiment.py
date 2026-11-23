"""Run the acitive-02 paired MuJoCo experiment from shared live contact states."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import nullcontext
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import mujoco
import numpy as np

from .aude import DEFAULT_STEP_M, TARGET_PRELOAD_M, UPSTREAM, AudeExplorer, JointIK, state_posture
from .wrist import (FILTER_ALPHA, FORCE_CAP_N, PAD_LOAD_FREE_N, PALM_CLEARANCE_M, PALM_LOOKAHEAD_S,
                    PALM_PROXIMITY_M, PROXIMITY_STIFFNESS_N_M, TORQUE_CAP_NM, WRIST_WELD_TIME_S,
                    WristFeedback, wrist_config)
from ..i04_geometric_traversal.fingertip_dp_compare import (
    CONTROL_STEPS, LOOKAHEAD_STEPS, policy_observation,
)
from ..i04_geometric_traversal.fingertip_hand import DT, ROOT, FingertipHand, build_model
from ..i04_geometric_traversal.fingertip_traversal import ForceFeedback, Leg, STATE, latency_ms, write_json
from ..i04_geometric_traversal.physics import quaternion, rotation

EXPERIMENT = 'acitive-02'
REVIEW = ROOT / 'mcc_finger_compliance_control/data/contact_wm_review_20261110'
PEAR = REVIEW / 'ycb_expansion_20261113/objects/ycb_016_pear/surface.obj'
PARTS = REVIEW / 'physics_contract_v1/controlled_policy_ablation_20261116/fruit_generalization/ycb_016_pear/collision'
START_SOURCE = (ROOT / 'finger_policy/data/evaluation_runs/single_all_optimized/evaluation'
                / 'single_all_seed20261116/ycb_016_pear_forward.npz')
START_FRAMES = (0, 618, 1237, 1855, 2474, 3719, 4337, 5599, 7461, 12375)
ARMS = ('aude-mcc', 'aude-single-all-mcc')
CONTROL_DT = CONTROL_STEPS * DT
PLAN_TICKS = 90
HISTORY_TICKS = 75
HIGH_LOAD_N = 40.
QUALITY = dict(fscore_5_mm=.40, coverage_10_mm=.70)


def digest(path):
    with path.open('rb') as stream:
        hasher = sha256()
        while block := stream.read(1024 * 1024):
            hasher.update(block)
    return hasher.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def step_plant(hand, data, feedback, wrist_feedback, touch, wrist, q_nominal, normals):
    for reference in wrist:
        command = wrist_feedback.command(hand, data, reference)
        data.mocap_pos[hand.driver] = command[:3]
        data.mocap_quat[hand.driver] = command[3:]
        data.ctrl[:] = feedback.command(hand, data, q_nominal, normals, touch, True)
        mujoco.mj_step(hand.model, data)
    mujoco.mj_forward(hand.model, data)
    return hand.measure(data)


def settle(hand, pose, joints):
    data = hand.initialize(pose[:3], rotation(pose[3:]), joints)
    feedback = ForceFeedback()
    wrist_feedback = WristFeedback()
    touch = hand.measure(data)
    normals = np.broadcast_to(rotation(pose[3:]).apply([0., 0., 1.]), (4, 3)).copy()
    hold = np.broadcast_to(pose, (CONTROL_STEPS, 7))
    history = []
    for tick in range(125):
        normals[touch.measured_heads] = touch.normals[touch.measured_heads]
        touch = step_plant(hand, data, feedback, wrist_feedback, touch, hold, joints, normals)
        if tick >= 125 - HISTORY_TICKS:
            history.append(policy_observation(hand, data, pose))
    if touch.measured_heads.sum() != 4 or touch.loads.max() >= 8.:
        raise RuntimeError(f'The shared start did not settle to four contacts: {touch.forces}, {touch.loads}')
    data.time = 0.
    state = np.empty(mujoco.mj_stateSize(hand.model, STATE))
    mujoco.mj_getState(hand.model, data, state, STATE)
    return state, np.array(history), normals


def record(data, hand, touch, wrist_nominal, wrist_feedback, q_nominal, aude_nominal, contact_mode, wall_s):
    return (data.qpos.copy(), np.r_[data.mocap_pos[hand.driver], data.mocap_quat[hand.driver]],
            wrist_nominal.copy(), wrist_feedback.offset.copy(), wrist_feedback.wrench.copy(),
            wrist_feedback.proximity_force_n,
            q_nominal.copy(), aude_nominal.copy(), data.ctrl.copy(), touch.points.copy(),
            touch.normals.copy(), touch.forces.copy(), touch.loads.copy(), touch.measured_heads,
            contact_mode, wall_s)


def longest_run(mask):
    boundaries = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)))
    return int(np.diff(boundaries)[::2].max(initial=0))


def run_arm(case, hand, ik, state, history, last_normals, budget_ticks, step_m, seed, arm):
    case.mkdir()
    started = perf_counter()
    data = mujoco.MjData(hand.model)
    mujoco.mj_setState(hand.model, data, state, STATE)
    mujoco.mj_forward(hand.model, data)
    initial = state_posture(hand, data)
    touch = hand.measure(data)
    explorer = AudeExplorer(hand, ik, seed, step_m)
    explorer.observe(data, touch)
    feedback = ForceFeedback()
    wrist_feedback = WristFeedback()
    current_reference = np.r_[initial.position, quaternion(initial.orientation)]
    initial_hash = sha256(state.tobytes()).hexdigest()
    is_policy = arm == ARMS[1]
    policy = None
    if is_policy:
        from ..i04_geometric_traversal.fingertip_local_policy import LocalFingerPolicy

        policy = LocalFingerPolicy(seed, budget_ticks + HISTORY_TICKS)
    with (policy if is_policy else nullcontext()):
        if is_policy:
            for tick, packet in enumerate(history):
                policy.command(tick, packet)
        rows = [record(data, hand, touch, current_reference, wrist_feedback, initial.joints, initial.joints,
                       'initial', perf_counter() - started)]
        plans = []
        tick = 0
        with (case / 'plans.jsonl').open('w', encoding='utf-8') as plan_log:
            while tick < budget_ticks:
                begin = perf_counter()
                plan = explorer.plan(data, touch)
                planning_ms = (perf_counter() - begin) * 1000
                here = state_posture(hand, data)
                # Continue the nominal wrist across replans. The measured wrist
                # already includes the admittance offset used for the next solve.
                here.position = current_reference[:3]
                here.orientation = rotation(current_reference[3:])
                position, orientation, joint = Leg(here, plan.posture, plan.mode).samples(PLAN_TICKS * CONTROL_STEPS)
                wrist = np.column_stack((position, orientation))
                plans.append(dict(time_s=tick * CONTROL_DT, mode=plan.mode,
                                  wrist_target=np.r_[plan.posture.position, quaternion(plan.posture.orientation)].tolist(),
                                  aude_joint_target=plan.posture.joints.tolist(), virtual_contacts_m=plan.targets.tolist(),
                                  direction=plan.direction.tolist(), live_pad_valid=plan.valid.tolist(),
                                  local_length_m=explorer.directions.tangent_length.tolist(),
                                  surface_observations=len(explorer.gp.surface), free_observations=len(explorer.gp.free),
                                  ik_error_mm=plan.error_mm, planned_min_self_gap_m=plan.self_gap_m,
                                  planning_ms=planning_ms))
                plan_log.write(json.dumps(plans[-1], allow_nan=False) + '\n')
                plan_log.flush()
                leg_ticks = min(PLAN_TICKS, budget_ticks - tick)
                for phase in range(leg_ticks):
                    index = phase * CONTROL_STEPS
                    aude_nominal = joint[index + CONTROL_STEPS - 1]
                    q_nominal = aude_nominal
                    if is_policy:
                        lookahead = wrist[min(index + LOOKAHEAD_STEPS - 1, len(wrist) - 1)]
                        packet = policy_observation(hand, data, wrist_feedback.forecast(lookahead))
                        q_nominal = policy.command(HISTORY_TICKS + tick, packet)
                    last_normals[touch.measured_heads] = touch.normals[touch.measured_heads]
                    segment = wrist[index:index + CONTROL_STEPS]
                    touch = step_plant(hand, data, feedback, wrist_feedback, touch, segment, q_nominal, last_normals)
                    current_reference = segment[-1]
                    tick += 1
                    explorer.observe(data, touch)
                    rows.append(record(data, hand, touch, current_reference, wrist_feedback, q_nominal, aude_nominal,
                                       plan.mode, perf_counter() - started))
                    if not np.isfinite(data.qpos).all() or abs(data.time - tick * CONTROL_DT) > 1e-7:
                        raise RuntimeError(f'Invalid native MuJoCo state at tick {tick}')
                if len(plans) % 25 == 0:
                    print(f'{arm} {case.parent.name} {tick * CONTROL_DT:.1f}s, {len(explorer.gp.surface)} contact voxels, '
                          f'{touch.measured_heads.sum()} live, {plan.mode}', flush=True)
    keys = ('qpos', 'wrist_command', 'wrist_nominal', 'wrist_offset', 'wrist_wrench_feedback', 'palm_proximity_drive_n',
            'finger_nominal', 'aude_joint_nominal', 'finger_command', 'points_m', 'normals_outward',
            'pad_force_n', 'region_load_n', 'pad_valid', 'mode', 'wall_s')
    trace = {key: np.asarray(value) for key, value in zip(keys, zip(*rows))}
    np.savez_compressed(case / 'trace.npz', **trace)
    gp_points, gp_labels = explorer.gp.training()
    np.savez_compressed(case / 'aude_gp_observations.npz', points_m=gp_points, label=gp_labels)
    count = trace['pad_valid'][1:].sum(axis=1)
    path = trace['qpos'][:, :7]
    travel = np.linalg.norm(np.diff(path[:, :3], axis=0), axis=1).sum()
    angles = 2. * np.arccos(np.clip(abs(np.einsum('ij,ij->i', path[1:, 3:], path[:-1, 3:])), 0., 1.))
    load = trace['region_load_n'][1:].max(axis=1)
    palm_load = trace['region_load_n'][1:, -1]
    wrist_offset = np.linalg.norm(trace['wrist_offset'][1:, :3], axis=1)
    report = dict(experiment=EXPERIMENT, arm=arm, initial_state_sha256=initial_hash, seed=seed,
                  seconds=tick * CONTROL_DT, wall_execution_s=perf_counter() - started,
                  budget_s=budget_ticks * CONTROL_DT, stop='budget_complete',
                  plans=len(plans), explore_plans=sum(p['mode'] == 'explore' for p in plans),
                  recovery_plans=sum(p['mode'] == 'recontact' for p in plans),
                  contact_voxels=len(explorer.gp.surface), free_voxels=len(explorer.gp.free),
                  at_least_one_contact=float(np.mean(count >= 1)), at_least_three_contacts=float(np.mean(count >= 3)),
                  four_contacts=float(np.mean(count == 4)), longest_full_loss_s=longest_run(count == 0) * CONTROL_DT,
                  max_region_load_n=float(load.max(initial=0.)), time_above_8n_s=float((load > 8.).sum() * CONTROL_DT),
                  longest_above_40n_s=longest_run(load > HIGH_LOAD_N) * CONTROL_DT,
                  max_palm_load_n=float(palm_load.max(initial=0.)),
                  p95_palm_contact_load_n=float(np.percentile(palm_load[palm_load > .2], 95)) if (palm_load > .2).any() else None,
                  wrist_mcc_translation_max_m=float(wrist_offset.max(initial=0.)),
                  wrist_mcc_translation_p95_m=float(np.percentile(wrist_offset, 95)),
                  wrist_mcc_yield_time_s=float((wrist_offset > .001).sum() * CONTROL_DT),
                  palm_proximity_yield_time_s=float((trace['palm_proximity_drive_n'][1:] > 0.).sum() * CONTROL_DT),
                  wrist_translation_m=float(travel), wrist_rotation_deg=float(np.rad2deg(angles.sum())),
                  finger_nominal_source='single_all' if is_policy else 'aude_joint_ik',
                  aude_nominal_difference_max_rad=float(np.max(abs(trace['finger_nominal'] - trace['aude_joint_nominal']))),
                  planning_timing=latency_ms([p['planning_ms'] for p in plans]))
    if is_policy:
        report['policy'] = policy.info
        report['policy_timing'] = policy.report(case)
    write_json(case / 'run.json', report)
    return report


def pair_runtime(model_path, pose, q):
    model = mujoco.MjModel.from_binary_path('pear.mjb', {'pear.mjb': model_path.read_bytes()})
    hand = FingertipHand(model, object_name='pear',
                        grasp=(np.asarray(pose[:3]), rotation(pose[3:]), np.asarray(q)), min_head_fraction=None)
    return hand, JointIK(hand)


def run_pair(job):
    output, index, frame, pose, q, tick_budget, step_m, seed = job
    hand, ik = pair_runtime(output / 'model.mjb', pose, q)
    pair = output / f'frame-{frame:05d}'
    pair.mkdir(exist_ok=True)
    state, history, normals = settle(hand, pose, q)
    initial = pair / 'initial.npz'
    if initial.is_file():
        with np.load(initial, allow_pickle=False) as prior:
            np.testing.assert_array_equal(prior['integration_state'], state)
    else:
        np.savez_compressed(initial, integration_state=state, warm_policy_observations=history, outward_normals=normals)
    results = []
    for arm in ARMS[::(-1 if index % 2 else 1)]:
        case = pair / arm
        run = case / 'run.json'
        if run.is_file() and (case / 'trace.npz').is_file():
            result = read_json(run)
            if result['initial_state_sha256'] != sha256(state.tobytes()).hexdigest():
                raise ValueError(f'Initial state differs in {run}')
        else:
            for attempt in range(1, 100):
                staging = pair / f'{arm}.attempt-{attempt:02d}'
                if not staging.exists():
                    break
            else:
                raise RuntimeError(f'No unused attempt path for {case}')
            result = run_arm(staging, hand, ik, state, history, normals.copy(), tick_budget, step_m, seed + index, arm)
            staging.rename(case)
        results.append(dict(start_frame=frame, **result))
        print(f'{EXPERIMENT} {frame=} {arm}: {result["stop"]}, {result["seconds"]:.2f}s simulated, '
              f'{result["wall_execution_s"]:.1f}s wall, loss {result["longest_full_loss_s"]:.2f}s', flush=True)
    return results


def suite(output, seconds, start_frames, step_m, seed, workers):
    if not np.isfinite(seconds) or seconds < CONTROL_DT or not np.isfinite(step_m) or step_m <= 0.:
        raise ValueError('Duration and target spacing must be positive and finite')
    if not start_frames or min(start_frames) < 0 or len(set(start_frames)) != len(start_frames):
        raise ValueError('Expected distinct nonnegative start frames')
    if workers < 1:
        raise ValueError('At least one execution worker is required')
    workers = min(workers, len(start_frames))
    output = output.resolve()
    tick_budget = int(round(seconds / CONTROL_DT))
    with np.load(START_SOURCE, allow_pickle=False) as data:
        poses = data['input_palm_pose_world'][list(start_frames), 0].astype(float)
        joints = data['input_q_live'][list(start_frames), 0].astype(float)
        if not data['observed'][list(start_frames), 0].all():
            raise ValueError('Start frames must be observed pear states')
    output.mkdir(parents=True, exist_ok=True)
    parts = sorted(PARTS.glob('collision_part_*.obj'))
    if len(parts) != 128:
        raise ValueError(f'Expected 128 pear collision components in {PARTS}')
    model = build_model(surface=PEAR, collision_parts=parts, object_name='pear', model_name=EXPERIMENT,
                        wrist_time_constant_s=WRIST_WELD_TIME_S, palm_proximity_m=PALM_PROXIMITY_M)
    physics = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, None, physics)
    manifest = dict(experiment=EXPERIMENT, upstream=UPSTREAM, backend=f'MuJoCo {mujoco.__version__}',
                    seconds=tick_budget * CONTROL_DT, arm_joints=0, simulation_dt_s=DT, control_dt_s=CONTROL_DT,
                    plan_period_s=PLAN_TICKS * CONTROL_DT, tried_start_frames=list(start_frames),
                    seed=seed, target_step_m=step_m, target_preload_m=TARGET_PRELOAD_M,
                    exploration_gp='GPyTorch ExactGP with fixed Aude squared exponential kernel',
                    wrist_weld_time_constant_s=WRIST_WELD_TIME_S,
                    object=PEAR.relative_to(ROOT).as_posix(), object_mesh_sha256=digest(PEAR),
                    model_sha256=sha256(physics.tobytes()).hexdigest(),
                    initialization_source=START_SOURCE.relative_to(ROOT).as_posix(), initialization_source_sha256=digest(START_SOURCE),
                    execution_source_sha256={path.name: digest(path) for path in sorted(Path(__file__).parent.glob('*.py'))
                                             if path.name != 'report.py'},
                    stiffness=asdict(ForceFeedback().fingers[0].config), desired_normal_force_n=1.,
                    wrist_mcc={key: value.tolist() if isinstance(value, np.ndarray) else value
                               for key, value in asdict(wrist_config()).items()},
                    wrist_wrench_filter_alpha=FILTER_ALPHA, finger_preload_excluded_n=PAD_LOAD_FREE_N,
                    wrist_input_force_cap_n=FORCE_CAP_N, wrist_input_torque_cap_nm=TORQUE_CAP_NM,
                    palm_proximity_radius_m=PALM_PROXIMITY_M, palm_prediction_s=PALM_LOOKAHEAD_S,
                    palm_clearance_target_m=PALM_CLEARANCE_M, palm_proximity_stiffness_n_m=PROXIMITY_STIFFNESS_N_M,
                    sensor='active MuJoCo full fingertip pads and raw collision normals',
                    tactile_map_uses_only_live_contacts=True, shared_geometry_scoring_offline=True,
                    high_load_reference_n=HIGH_LOAD_N, time_policy='continue during contact and recover after loss',
                    execution_workers=workers,
                    quality_thresholds=QUALITY, arms=list(ARMS))
    config = output / 'protocol.json'
    if config.is_file():
        if read_json(config) != manifest:
            raise ValueError(f'The existing experiment settings differ in {config}')
    else:
        (output / 'model.mjb').write_bytes(physics.tobytes())
        write_json(config, manifest)
    records = []
    jobs = [(output, index, frame, pose, q, tick_budget, step_m, seed)
            for index, (frame, pose, q) in enumerate(zip(start_frames, poses, joints))]

    def register(results):
        records.extend(results)
        temporary = output / 'runs.tmp.json'
        write_json(temporary, records)
        temporary.replace(output / 'runs.json')

    if workers == 1:
        for job in jobs:
            register(run_pair(job))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run_pair, job) for job in jobs]
            for future in as_completed(futures):
                register(future.result())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    cli = parser.add_subparsers(dest='action', required=True)
    run = cli.add_parser('run')
    run.add_argument('--output', type=Path, required=True)
    run.add_argument('--seconds', type=float, default=180.)
    run.add_argument('--starts', type=int, nargs='+', default=START_FRAMES)
    run.add_argument('--step-mm', type=float, default=DEFAULT_STEP_M * 1000)
    run.add_argument('--seed', type=int, default=20261120)
    run.add_argument('--workers', type=int, default=1)
    score = cli.add_parser('score')
    score.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'run':
        suite(args.output, args.seconds, args.starts, args.step_mm / 1000., args.seed, args.workers)
    else:
        from .report import score_suite

        score_suite(args.output)


if __name__ == '__main__':
    main()
