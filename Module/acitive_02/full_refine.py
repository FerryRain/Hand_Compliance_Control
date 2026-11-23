"""Continue a scored whole-object sweep with tactile-directed gap probes."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
import json
from pathlib import Path
import shutil
from time import perf_counter, sleep
from types import SimpleNamespace

import mujoco
import numpy as np
from scipy.spatial import cKDTree

from .aude import DEFAULT_STEP_M, AudeExplorer, state_posture, unit
from .experiment import (ARMS, CONTROL_DT, CONTROL_STEPS, HIGH_LOAD_N, HISTORY_TICKS,
                         LOOKAHEAD_STEPS, PLAN_TICKS, START_SOURCE, STATE, digest, longest_run,
                         pair_runtime, policy_observation, read_json, record, step_plant, write_json)
from .full import TRACE_KEYS, aude_segment, blend_wrist
from .surface_scan import SurfaceScanner, TactileSurface, sweep_directions
from .wrist import WristFeedback
from ..i04_geometric_traversal.fingertip_traversal import ForceFeedback, Leg, latency_ms
from ..i04_geometric_traversal.physics import rotation


GAP_TARGET_M = .004
REPLAY_TOLERANCE = 1.e-7


def gap_directions():
    return np.vstack([sweep_directions(axis, angle) for axis, angle in enumerate((.019, -.021, .016))])


def patch_directions(center):
    horizontal = unit(np.cross(center, np.eye(3)[np.argmin(np.abs(center))]))
    vertical = np.cross(center, horizontal)
    offsets = .031 * np.arange(-3, 4)
    raster = [center]
    for row, y in enumerate(offsets):
        columns = offsets[::-1] if row % 2 else offsets
        raster.extend(unit(center + x * horizontal + y * vertical) for x in columns)
    return np.repeat(np.asarray(raster), repeats=2, axis=0)


def select_gap(surface, directions, visits):
    surface.refresh(force=True)
    if surface.tree is None:
        raise ValueError('Gap-directed continuation requires live contact evidence')
    separation, nearest = surface.tree.query(directions)
    distance = separation * surface.radii[nearest]
    candidates = np.flatnonzero((distance >= GAP_TARGET_M) & (visits < 3))
    if not len(candidates):
        return None, 0.
    rank = distance[candidates] / (1. + .65 * visits[candidates])
    index = int(candidates[np.argmax(rank)])
    return index, float(distance[index])


def restore(hand, data, touch, explorer, feedback, wrist, surface, normals, old_trace, plans, local_ticks, history):
    reference = old_trace['wrist_nominal']
    expected = old_trace['qpos']
    nominal = old_trace['finger_nominal']
    peak = 0.
    for tick in range(len(reference) - 1):
        if tick < local_ticks:
            step = (tick % PLAN_TICKS) * CONTROL_STEPS
            if step == 0:
                plan = plans[tick // PLAN_TICKS]
                start, end = state_posture(hand, data), state_posture(hand, data)
                start.position, start.orientation = reference[tick, :3], rotation(reference[tick, 3:])
                end.position = np.asarray(plan['wrist_target'][:3])
                end.orientation = rotation(np.asarray(plan['wrist_target'][3:]))
                end.joints = np.asarray(plan['aude_joint_target'])
                position, orient, _ = Leg(start, end, plan['mode']).samples(PLAN_TICKS * CONTROL_STEPS)
                path = np.column_stack((position, orient))
            segment = path[step:step + CONTROL_STEPS]
            future = path[min(step + LOOKAHEAD_STEPS - 1, len(path) - 1)]
        else:
            segment = blend_wrist(reference[tick], reference[tick + 1])
            future = reference[tick + 1].copy()
            future[:3] += (reference[tick + 1, :3] - reference[tick, :3]) * (LOOKAHEAD_STEPS / CONTROL_STEPS)
        if tick % PLAN_TICKS == 0 or tick == local_ticks:
            explorer.plan(data, touch)
        if history is not None:
            history.push(policy_observation(hand, data, wrist.forecast(future)))
        normals[touch.measured_heads] = touch.normals[touch.measured_heads]
        touch = step_plant(hand, data, feedback, wrist, touch, segment, nominal[tick + 1], normals)
        explorer.observe(data, touch)
        surface.add(touch)
        error = float(np.max(np.abs(data.qpos - expected[tick + 1])))
        peak = max(peak, error)
        if not np.isfinite(error) or error > REPLAY_TOLERANCE:
            raise ValueError(f'Continuation replay departed from the saved state at {tick * CONTROL_DT:.2f}s ({error:g})')
    return touch, peak


def restore_logged_pose(hand, data, explorer, wrist, surface, normals, trace, history):
    """Reinitialize from logged coordinates when contact replay diverges."""
    qpos = trace['qpos']
    active = trace['pad_valid']
    for row in np.flatnonzero(active.any(axis=1)):
        contact = SimpleNamespace(points=trace['points_m'][row], normals=trace['normals_outward'][row],
                                  measured_heads=active[row], loads=trace['region_load_n'][row])
        data.qpos[:] = qpos[row]
        mujoco.mj_kinematics(hand.model, data)
        explorer.observe(data, contact)
        surface.add(contact)
        normals[contact.measured_heads] = contact.normals[contact.measured_heads]

    def seed_physical_sample(row):
        command = trace['wrist_command'][row]
        data.qpos[:] = qpos[row]
        mujoco.mj_differentiatePos(hand.model, data.qvel, CONTROL_DT, qpos[row - 1], qpos[row])
        data.mocap_pos[hand.driver] = command[:3]
        data.mocap_quat[hand.driver] = command[3:]
        data.ctrl[:] = trace['finger_command'][row]
        data.time = row * CONTROL_DT
        mujoco.mj_forward(hand.model, data)

    if history is not None:
        reference = trace['wrist_nominal']
        for row in range(max(1, len(qpos) - 251), len(qpos) - 1):
            seed_physical_sample(row)
            future = reference[row + 1].copy()
            future[:3] += (reference[row + 1, :3] - reference[row, :3]) * (LOOKAHEAD_STEPS / CONTROL_STEPS)
            wrist.offset = trace['wrist_offset'][row]
            history.push(policy_observation(hand, data, wrist.forecast(future)))
        wrist.offset = np.zeros(6)
    seed_physical_sample(len(qpos) - 1)
    np.testing.assert_array_equal(data.qpos, qpos[-1])
    return hand.measure(data)


def refine_arm(job):
    base, out, arm, budget, manifest, pose_resume = job
    original = base / arm
    deadline = perf_counter() + 3600
    while not (original / 'run.json').is_file():
        if perf_counter() >= deadline:
            raise TimeoutError(f'No completed scan to continue for {arm}')
        sleep(5.)
    case = out / arm
    case.mkdir()
    run = read_json(original / 'run.json')
    plans = [json.loads(line) for line in (original / 'plans.jsonl').read_text(encoding='utf-8').splitlines()]
    with np.load(original / 'trace.npz', allow_pickle=False) as archive:
        trace = {key: archive[key] for key in TRACE_KEYS}
    with np.load(START_SOURCE, allow_pickle=False) as raw:
        pose = raw['input_palm_pose_world'][manifest['start'], 0].astype(float)
        joints = raw['input_q_live'][manifest['start'], 0].astype(float)
    with np.load(base / 'initial.npz', allow_pickle=False) as seed:
        state = seed['integration_state']
        warm = seed['warm_policy_observations']
        normals = seed['outward_normals']
    hand, ik = pair_runtime(base / 'model.mjb', pose, joints)
    data = mujoco.MjData(hand.model)
    mujoco.mj_setState(hand.model, data, state, STATE)
    mujoco.mj_forward(hand.model, data)
    touch = hand.measure(data)
    explorer = AudeExplorer(hand, ik, manifest['seed'], DEFAULT_STEP_M)
    explorer.observe(data, touch)
    surface = TactileSurface()
    surface.add(touch)
    feedback, wrist = ForceFeedback(), WristFeedback()
    history = None
    if arm == ARMS[1]:
        from ..i04_geometric_traversal.fingertip_local_policy import LocalFingerPolicy, ObservationRing

        history = ObservationRing()
        for frame in warm:
            history.push(frame)
    base_ticks = len(trace['qpos']) - 1
    print(f'{arm}: restoring {"saved pose" if pose_resume else "native state"} at {run["seconds"]:.2f}s', flush=True)
    started = perf_counter()
    if pose_resume:
        touch = restore_logged_pose(hand, data, explorer, wrist, surface, normals, trace, history)
        reference = trace['wrist_command'][-1]
        error = None
    else:
        touch, error = restore(hand, data, touch, explorer, feedback, wrist, surface, normals, trace, plans,
                               round(manifest['local_s'] / CONTROL_DT), history)
        reference = trace['wrist_nominal'][-1]
    replay_s = perf_counter() - started
    print(f'{arm}: restored in {replay_s:.1f}s', flush=True)

    directions = gap_directions()
    neighbors = cKDTree(directions)
    visits = np.zeros(len(directions), dtype=np.int8)
    before = int(surface.uncovered(directions, GAP_TARGET_M).sum())
    axes = (((.22, -.57, -.79), 2), ((-.52, .39, -.76), 3), ((.20, -.71, -.68), 2),
            ((-.14, -.22, -.97), 1), ((.42, .41, -.82), 0))
    rows, probes = [], []
    policy = None
    tick, end_tick = base_ticks, base_ticks + round(budget / CONTROL_DT)
    since = perf_counter()
    if history is not None:
        policy = LocalFingerPolicy(manifest['seed'], end_tick + HISTORY_TICKS)
        policy.observation = history
        policy.initial = warm[0, :16].astype(np.float32)
        policy.nominal = trace['finger_nominal'][-1].astype(np.float32)
        for q in trace['finger_nominal'][-1 - policy.scheduler.history.maxlen:-1]:
            policy.scheduler.observe(q)
    shutil.copyfile(original / 'plans.jsonl', case / 'plans.jsonl')
    scanner = None
    stop = 'gap_probe_budget'
    with (policy if policy is not None else nullcontext()), (case / 'plans.jsonl').open('a', encoding='utf-8') as log:
        while tick < end_tick:
            if scanner is None or scanner.done:
                index, gap = select_gap(surface, directions, visits)
                if index is None:
                    stop = 'gap_probe_candidates_exhausted'
                    break
                center = directions[index]
                trial = int(visits[index])
                visits[neighbors.query_ball_point(center, .107)] += 1
                axis, finger = axes[(len(probes) + trial) % len(axes)]
                scanner = SurfaceScanner(hand, CONTROL_DT, patch_directions(center), axis, finger)
                scanner.surface = surface
                scanner.start(reference)
                probes.append(dict(time_s=tick * CONTROL_DT, center_direction=center.tolist(), gap_mm=gap * 1000.,
                                   previous_probes=trial, preferred_finger=finger))
            if (tick - base_ticks) % PLAN_TICKS == 0:
                _, q_curve, detail = aude_segment(explorer, data, touch, reference, tick * CONTROL_DT)
                detail['wrist_source'] = 'tactile_gap_probe'
                plans.append(detail)
                log.write(json.dumps(detail, allow_nan=False) + '\n')
                log.flush()
            step = (tick - base_ticks) % PLAN_TICKS * CONTROL_STEPS
            aude_q = q_curve[step + CONTROL_STEPS - 1]
            command = scanner.command(data, touch, wrist.offset)
            segment = blend_wrist(reference, command.wrist)
            nominal = aude_q
            if policy is not None:
                future = command.wrist.copy()
                future[:3] += (segment[-1, :3] - reference[:3]) * (LOOKAHEAD_STEPS / CONTROL_STEPS)
                nominal = policy.command(HISTORY_TICKS + tick, policy_observation(hand, data, wrist.forecast(future)))
            normals[touch.measured_heads] = touch.normals[touch.measured_heads]
            touch = step_plant(hand, data, feedback, wrist, touch, segment, nominal, normals)
            reference = segment[-1]
            tick += 1
            explorer.observe(data, touch)
            surface.add(touch)
            rows.append(record(data, hand, touch, reference, wrist, nominal, aude_q,
                               f'gap {command.phase}', run['wall_execution_s'] + perf_counter() - since))
            if abs(data.time - tick * CONTROL_DT) > 1e-6 or not np.isfinite(data.qpos).all():
                raise RuntimeError(f'{arm} invalid gap-probe state at control tick {tick}')
            if (tick - base_ticks) % 2000 == 0:
                print(f'{arm}: gap probes +{(tick - base_ticks) * CONTROL_DT:.1f}s, {len(probes)} attempted', flush=True)

    extension = {key: np.asarray(values) for key, values in zip(TRACE_KEYS, zip(*rows))}
    combined = {key: np.concatenate((value, extension[key])) if rows else value for key, value in trace.items()}
    np.savez_compressed(case / 'trace.npz', **combined)
    qpos = combined['qpos']
    load = combined['region_load_n'][1:].max(axis=1)
    count = combined['pad_valid'][1:].sum(axis=1)
    result = dict(run, seconds=tick * CONTROL_DT, budget_s=run['seconds'] + budget, stop=stop,
                  wall_execution_s=run['wall_execution_s'] + perf_counter() - since,
                  wall_at_last_sample_s=float(combined['wall_s'][-1]), contact_voxels=len(surface.samples),
                  tactile_grid_unresolved=int(surface.uncovered(sweep_directions()).sum()),
                  longest_above_40n_s=longest_run(load > HIGH_LOAD_N) * CONTROL_DT,
                  longest_full_loss_s=longest_run(count == 0) * CONTROL_DT,
                  max_region_load_n=float(load.max()), max_palm_load_n=float(combined['region_load_n'][1:, -1].max()),
                  wrist_translation_m=float(np.linalg.norm(np.diff(qpos[:, :3], axis=0), axis=1).sum()),
                  at_least_one_contact=float(np.mean(count > 0)), at_least_three_contacts=float(np.mean(count >= 3)),
                  four_contacts=float(np.mean(count == 4)), planning_timing=latency_ms([p['planning_ms'] for p in plans]),
                  aude_nominal_difference_max_rad=float(np.max(abs(combined['finger_nominal'] - combined['aude_joint_nominal']))),
                  gap_refinement=dict(prefix_wall_s=run['wall_execution_s'], prefix_sim_s=run['seconds'],
                                      extra_sim_s=(tick - base_ticks) * CONTROL_DT,
                                      replay_wall_s=replay_s, replay_max_state_abs_error=error,
                                      restart_from_saved_pose=pose_resume,
                                      prefix_trace_sha256=digest(original / 'trace.npz'), live_gap_rays_before=before,
                                      live_gap_rays_after=int(surface.uncovered(directions, GAP_TARGET_M).sum()),
                                      probes=probes))
    if policy is not None:
        result['gap_refinement']['policy_timing'] = policy.report(case)
    write_json(case / 'run.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--extra-seconds', type=float, default=175.)
    parser.add_argument('--resume-from-recorded-pose', action='store_true')
    args = parser.parse_args()
    if not np.isfinite(args.extra_seconds) or args.extra_seconds < 1.:
        raise ValueError('Give the gap probes a finite budget of one second or more')
    base, output = args.base.resolve(), args.output.resolve()
    manifest = read_json(base / 'protocol.json')
    if digest(base / 'model.mjb') != manifest['model_sha256'] or digest(base / 'initial.npz') != manifest['initial_sha256']:
        raise ValueError('The first sweep model or initial integration state has changed')
    output.mkdir(parents=True, exist_ok=False)
    for name in ('model.mjb', 'initial.npz'):
        shutil.copyfile(base / name, output / name)
    frozen = dict(manifest, refinement=dict(prefix=base.name, protocol_sha256=digest(base / 'protocol.json'),
                                            source_sha256=digest(Path(__file__)), sim_budget_s=args.extra_seconds,
                                            radial_gap_m=GAP_TARGET_M, restart_from_saved_pose=args.resume_from_recorded_pose,
                                            controller_state_replay_tol=(None if args.resume_from_recorded_pose else REPLAY_TOLERANCE),
                                            angular_candidates='three offset latitude grids', target_input='live touch'))
    write_json(output / 'protocol.json', frozen)
    with ProcessPoolExecutor(max_workers=2) as pool:
        jobs = [(base, output, arm, args.extra_seconds, manifest, args.resume_from_recorded_pose) for arm in ARMS]
        write_json(output / 'runs.json', list(pool.map(refine_arm, jobs)))


if __name__ == '__main__':
    main()
