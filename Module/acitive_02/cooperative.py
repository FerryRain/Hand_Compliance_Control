"""Whole-pear Aude exploration with a shared multi-fingertip contact constraint."""

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

from .aude import AudeExplorer, DEFAULT_STEP_M, state_posture, tangent, unit
from .experiment import (ARMS, CONTROL_DT, CONTROL_STEPS, DT, EXPERIMENT, HIGH_LOAD_N, HISTORY_TICKS,
                         LOOKAHEAD_STEPS, PLAN_TICKS, START_SOURCE, STATE, digest, longest_run, pair_runtime,
                         policy_observation, record, step_plant, write_json)
from .full import BASE, TRACE_KEYS, blend_wrist
from .surface_scan import SURFACE_LOAD_N, TactileSurface, align, limit, sweep_directions
from .wrist import WristFeedback
from ..i04_geometric_traversal.fingertip_plan import JointPatchPlanner, Posture
from ..i04_geometric_traversal.fingertip_traversal import ForceFeedback, Leg, latency_ms
from ..i04_geometric_traversal.physics import quaternion, rotation


REPLAN_TICKS = 18
SERVO_TICKS = 3
PROBE_GAP_M = -.00048
ROUTE_REACH_RAD = .050
MAX_ROUTE_STEP_RAD = .066
ROW_ANGLE = .235
COLUMN_ANGLE = .095


def cohort(touch):
    return int(touch.measured_heads.sum()) if touch.loads.max() <= SURFACE_LOAD_N else 0


def trajectory(axis, origin):
    direction = np.roll(unit(origin), -axis)
    mid = np.arcsin(np.clip(direction[0], -1., 1.))
    phase = np.arctan2(direction[2], direction[1])
    lowers = np.linspace(mid, -np.pi / 2, 1 + int(np.ceil((mid + np.pi / 2) / ROW_ANGLE)))
    uppers = np.linspace(mid, np.pi / 2, 1 + int(np.ceil((np.pi / 2 - mid) / ROW_ANGLE)))[1:]
    sequence = np.r_[lowers, uppers]
    previous = phase
    result = []
    for i, latitude in enumerate(sequence):
        radius = np.cos(latitude)
        count = max(1, int(np.ceil(2 * np.pi * radius / COLUMN_ANGLE)))
        theta = previous + np.linspace(0, (1 if i % 2 == 0 else -1) * 2 * np.pi, count + 1)
        result.extend(np.stack((np.repeat(np.sin(latitude), len(theta)), radius * np.cos(theta),
                               radius * np.sin(theta)), axis=1))
        previous = theta[-1]
    return np.roll(np.asarray(result), axis, axis=1)


class ContactProjection(JointPatchPlanner):
    """Local contact kinematics from I04-c with an Aude wrist-and-finger prior."""

    def __init__(self, hand):
        super().__init__(hand, graph=None)
        self.jacobian = np.empty((3, self.model.nv))

    def jac_position(self, point, body, width):
        mujoco.mj_jac(self.model, self.data, self.jacobian, None, point, body)
        return self.jacobian[:, :width]

    def solve(self, prior, start_q, *, joint, iterations, target_gap=PROBE_GAP_M):
        h, d, m = self.h, self.data, self.model
        width = 22 if joint else 6
        prior_weights = np.r_[[.54] * 3, [.030] * 3, [.010] * 16][:width]
        damping = .0017 * np.eye(width)
        self.set_pose(Posture(prior.position, prior.orientation, start_q))
        for _ in range(iterations):
            # Select three possible supports during wrist-only adaptation. The
            # coupled solve keeps a fourth normal task whenever it has a witness.
            probes = [self.nearest_part(f) for f in range(4)]
            order = np.argsort([g for g, _ in probes])
            rows = [prior_weights[:, None] * np.eye(width), damping]
            difference = np.r_[d.qpos[:3] - prior.position,
                                (prior.orientation.inv() * rotation(d.qpos[3:7])).as_rotvec(),
                                d.qpos[h.q] - prior.joints]
            residuals = [prior_weights * difference[:width], np.zeros(width)]
            for rank, finger in enumerate(order):
                gap, witness = probes[finger]
                if witness is None or gap > .055:
                    continue
                point, outward = witness
                weight = (2.9 if rank < 3 else .54) if joint else (2.4 if rank < 3 else .19)
                jac = outward @ self.jac_position(point, h.tip_bodies[finger], width)
                rows.append(weight * jac[None, :])
                residuals.append(np.array([weight * np.clip(gap - target_gap, -.009, .019)]))
            mujoco.mj_collision(m, d)
            rigid = {}
            for contact in d.contact:
                a, b = int(contact.geom[0]), int(contact.geom[1])
                if a in h.objects and b not in h.objects:
                    geom, normal = b, contact.frame[:3]
                elif b in h.objects and a not in h.objects:
                    geom, normal = a, -contact.frame[:3]
                else:
                    continue
                if geom in h.pad_index:
                    continue
                minimum = .005 if geom == h.palm_geom else .0013
                if contact.dist < minimum:
                    prior_gap = rigid.get(geom)
                    if prior_gap is None or contact.dist < prior_gap[0]:
                        rigid[geom] = (float(contact.dist), minimum, contact.pos.copy(), normal.copy())
            for geom, (gap, minimum, point, normal) in rigid.items():
                weight = 9. if geom == h.palm_geom else 5.
                row = normal @ self.jac_position(point, m.geom_bodyid[geom], width)
                rows.append(weight * row[None, :])
                residuals.append(np.array([weight * (gap - minimum)]))
            delta = np.linalg.lstsq(np.vstack(rows), -np.concatenate(residuals), rcond=None)[0]
            increment = np.zeros(m.nv)
            increment[:3] = limit(delta[:3], .006 if joint else .0035)
            increment[3:6] = limit(delta[3:6], .085 if joint else .048)
            if joint:
                increment[6:] = np.clip(delta[6:], -.105, .105)
            mujoco.mj_integratePos(m, d.qpos, increment, 1.)
            d.qpos[h.q] = np.clip(d.qpos[h.q], h.limits[:, 0] + .012, h.limits[:, 1] - .012)
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            if np.linalg.norm(delta[:3]) < .00012 and np.linalg.norm(delta[3:]) < .002:
                break
        return state_posture(h, d)


class CooperativeSweep:
    def __init__(self, hand, pose, surfaces=1):
        self.hand = hand
        self.surfaces = surfaces
        self.stage = 0
        self.route = trajectory(0, pose.position)
        self.index = 0
        self.missing = 0
        self.progress = 0
        self.best_error = np.inf
        self.passes = []
        self.projection = ContactProjection(hand)
        self.contact_surface = TactileSurface()
        self.last_servo = np.zeros(6)

    @property
    def done(self):
        return self.stage >= self.surfaces

    def waypoint(self, pose, second):
        position = unit(pose.position)
        while not self.done:
            while self.index < len(self.route) and position @ self.route[self.index] > np.cos(ROUTE_REACH_RAD):
                self.index += 1
                self.progress = 0
                self.best_error = np.inf
            if self.index < len(self.route):
                return self.route[self.index]
            self.passes.append(dict(pass_number=self.stage + 1, time_s=second, directions_total=len(self.route)))
            self.stage += 1
            self.route = trajectory(self.stage % 3, pose.position)
            self.index = 0
        return position

    def guide(self, live, aude, touch, second):
        next_direction = self.waypoint(live, second)
        radial = unit(live.position)
        delta = align(radial, next_direction).as_rotvec()
        distance = np.linalg.norm(delta)
        if distance < self.best_error - .010:
            self.best_error = distance
            self.progress = 0
        else:
            self.progress += 1
        healthy = cohort(touch) >= 2
        self.missing = 0 if healthy else self.missing + 1
        if self.progress > 28:
            self.index += 1
            self.best_error = np.inf
            self.progress = 0
        cap = MAX_ROUTE_STEP_RAD if healthy else (.035 if self.missing > 5 else .010)
        turn = Rotation.from_rotvec(limit(delta, cap))
        target = turn.apply(live.position)
        # Aude's active wrist displacement remains an input within that local
        # stroke. Transport dominates the large-scale coverage direction.
        gp_move = tangent(aude.position - live.position, radial)
        if healthy:
            target += .14 * limit(gp_move, .009)
        if self.missing and self.missing % 9 == 0:
            turn = Rotation.from_rotvec(radial * .060) * turn
        return Posture(target, turn * live.orientation, aude.joints), 'multi_slide' if healthy else 'multi_recontact'

    def stabilize(self, raw, data, touch, mcc, solve):
        orientation = rotation(raw[3:])
        if solve:
            # Use the current executed fingers to support the next wrist
            # increment, preserving the diffusion policy's command authority.
            positive = touch.loads.max() > SURFACE_LOAD_N
            if positive:
                target = np.r_[.0018 * unit(raw[:3]), np.zeros(3)]
            else:
                compliant = mcc.forecast(raw)
                prior = Posture(compliant[:3], rotation(compliant[3:]), data.qpos[self.hand.q])
                support = self.projection.solve(prior, prior.joints, joint=False, iterations=2)
                target = np.r_[limit(support.position - prior.position, .006),
                               limit((support.orientation * prior.orientation.inv()).as_rotvec(), .080)]
            self.last_servo = target
        wrist = raw.copy()
        wrist[:3] += self.last_servo[:3]
        wrist[3:] = quaternion(Rotation.from_rotvec(self.last_servo[3:]) * orientation)
        return wrist


def run_arm(job):
    root, arm, pose, q0, manifest = job
    case = root / arm
    case.mkdir()
    start = perf_counter()
    hand, ik = pair_runtime(root / 'model.mjb', pose, q0)
    with np.load(root / 'initial.npz', allow_pickle=False) as seed:
        initial, normals, warm = seed['integration_state'], seed['outward_normals'], seed['warm_policy_observations']
    data = mujoco.MjData(hand.model)
    mujoco.mj_setState(hand.model, data, initial, STATE)
    mujoco.mj_forward(hand.model, data)
    touch = hand.measure(data)
    local = state_posture(hand, data)
    planner = CooperativeSweep(hand, local, manifest['passes'])
    explorer = AudeExplorer(hand, ik, manifest['seed'], DEFAULT_STEP_M)
    explorer.observe(data, touch)
    planner.contact_surface.add(touch)
    wrist, fingers = WristFeedback(), ForceFeedback()
    raw = stabilized = np.r_[local.position, quaternion(local.orientation)]
    rows = [record(data, hand, touch, raw, wrist, local.joints, local.joints, 'initial', 0.)]
    budget = round(manifest['max_s'] / CONTROL_DT)
    policy = None
    if arm == ARMS[1]:
        from ..i04_geometric_traversal.fingertip_local_policy import LocalFingerPolicy

        policy = LocalFingerPolicy(manifest['seed'], budget + HISTORY_TICKS)
    tick = 0
    tracking = np.zeros(5, dtype=int)
    planning_ms = []
    termination = 'time_budget'
    with (policy if policy is not None else nullcontext()), (case / 'plans.jsonl').open('w', encoding='utf-8') as stream:
        if policy is not None:
            for i, packet in enumerate(warm):
                policy.command(i, packet)
        while tick < budget:
            if tick % PLAN_TICKS == 0:
                aude = explorer.plan(data, touch)
            phase = tick % REPLAN_TICKS
            if phase == 0:
                before = perf_counter()
                local = state_posture(hand, data)
                goal, mode = planner.guide(local, aude.posture, touch, tick * CONTROL_DT)
                if planner.done:
                    termination = 'scan_pass_budget'
                    break
                # Retain the previous joint solution near the policy-friendly
                # grasp. Aude's joint reference is warm-started at the live hand.
                guide = Posture(goal.position, goal.orientation, .68 * local.joints + .32 * aude.posture.joints)
                solution = planner.projection.solve(guide, guide.joints, joint=True, iterations=5)
                leg_start = Posture(raw[:3], rotation(raw[3:]), local.joints)
                pos, quat, commands = Leg(leg_start, solution, mode).samples(REPLAN_TICKS)
                path = np.column_stack((pos, quat))
                future = np.r_[solution.position, quaternion(solution.orientation)]
                planned_ms = (perf_counter() - before) * 1000
                planning_ms.append(planned_ms)
                details = dict(time_s=tick * CONTROL_DT, mode=mode, wrist_target=future.tolist(),
                               aude_joint_target=solution.joints.tolist(), original_aude_joint_target=aude.posture.joints.tolist(),
                               virtual_contacts_m=aude.targets.tolist(), route_pass=planner.stage + 1, route_index=planner.index,
                               contact_count=cohort(touch), planning_ms=planned_ms)
                stream.write(json.dumps(details, allow_nan=False) + '\n')
                stream.flush()
            raw = path[phase]
            supported = planner.stabilize(raw, data, touch, wrist, tick % SERVO_TICKS == 0)
            q_target = nominal = commands[phase]
            if policy is not None:
                ahead = path[min(phase + LOOKAHEAD_STEPS // CONTROL_STEPS, len(path) - 1)]
                supported_future = planner.stabilize(ahead, data, touch, wrist, False)
                packet = policy_observation(hand, data, wrist.forecast(supported_future))
                nominal = policy.command(HISTORY_TICKS + tick, packet)
            segment = blend_wrist(stabilized, supported)
            normals[touch.measured_heads] = touch.normals[touch.measured_heads]
            touch = step_plant(hand, data, fingers, wrist, touch, segment, nominal, normals)
            stabilized = segment[-1]
            tick += 1
            planner.contact_surface.add(touch)
            explorer.observe(data, touch)
            tracking[cohort(touch)] += 1
            rows.append(record(data, hand, touch, stabilized, wrist, nominal, q_target, mode, perf_counter() - start))
            if not np.isfinite(data.qpos).all() or abs(data.time - CONTROL_DT * tick) > 1.e-6:
                raise ValueError(f'{arm}: invalid physics tick {tick}')
            if tick % 1500 == 0:
                count = tracking[2:].sum() / tick
                print(f'{arm} {tick * CONTROL_DT:.0f}s: pass {planner.stage+1} {planner.index}/{len(planner.route)}; '
                      f'≥2 {count:.1%}; ≥3 {tracking[3:].sum()/tick:.1%}; {len(planner.contact_surface.samples)} cells',
                      flush=True)
    trace = {key: np.asarray(value) for key, value in zip(TRACE_KEYS, zip(*rows))}
    np.savez_compressed(case / 'trace.npz', **trace)
    valid = trace['pad_valid'][1:]
    load = trace['region_load_n'][1:].max(axis=1)
    count = valid.sum(axis=1)
    absent = count == 0
    motion = np.linalg.norm(np.diff(trace['qpos'][:, :3], axis=0), axis=1)
    moving = motion >= .00005
    rays = planner.contact_surface.uncovered(sweep_directions())
    report = dict(experiment=EXPERIMENT, profile=manifest['profile'], arm=arm,
                  initial_state_sha256=sha256(initial.tobytes()).hexdigest(), start_frame=manifest['start'],
                  seconds=tick * CONTROL_DT, budget_s=manifest['max_s'], stop=termination,
                  wall_execution_s=perf_counter() - start, wall_at_last_sample_s=float(trace['wall_s'][-1]),
                  contact_voxels=len(planner.contact_surface.samples), scan_passes=planner.passes,
                  final_route_pass=planner.stage + 1, final_route_index=planner.index, final_route_waypoints=len(planner.route),
                  tactile_grid_unresolved=int(rays.sum()), tactile_grid_rays=len(rays),
                  wrist_translation_m=float(motion.sum()), max_region_load_n=float(load.max()),
                  max_palm_load_n=float(trace['region_load_n'][1:, -1].max()),
                  longest_above_40n_s=longest_run(load > HIGH_LOAD_N) * CONTROL_DT,
                  longest_full_loss_s=longest_run(absent) * CONTROL_DT,
                  at_least_one_contact=float(np.mean(~absent)), at_least_two_contacts=float(np.mean(count >= 2)),
                  at_least_three_contacts=float(np.mean(count >= 3)), four_contacts=float(np.mean(count == 4)),
                  raw_contact_count_fractions=(np.bincount(count, minlength=5) / tick).tolist(),
                  qualified_contact_count_fractions=(tracking / tick).tolist(),
                  qualified_at_least_two_contacts=float(tracking[2:].sum() / tick),
                  qualified_at_least_three_contacts=float(tracking[3:].sum() / tick),
                  qualified_moving_at_least_two_contacts=(float(np.mean((count[moving] >= 2)
                                                                       & (load[moving] <= SURFACE_LOAD_N)))
                                                          if moving.any() else 0.),
                  planning_timing=latency_ms(planning_ms),
                  aude_nominal_difference_max_rad=float(np.max(np.abs(trace['finger_nominal'] - trace['aude_joint_nominal']))))
    if policy is not None:
        report['policy'] = policy.info
        report['policy_timing'] = policy.report(case)
    write_json(case / 'run.json', report)
    print(f'{arm}: {tick * CONTROL_DT:.2f}s complete; ≥2 {report["qualified_at_least_two_contacts"]:.2%}', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base', type=Path, default=BASE)
    parser.add_argument('--max-seconds', type=float, default=1300.)
    parser.add_argument('--passes', type=int, default=2)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--seed', type=int, default=20261120)
    parser.add_argument('--arm', choices=('both', *ARMS), default='both')
    args = parser.parse_args()
    if not np.isfinite(args.max_seconds) or args.max_seconds < .9 or not 1 <= args.passes <= 3:
        raise ValueError('Use a positive execution budget and one to three cooperative passes')
    root, base = args.output.resolve(), args.base.resolve()
    root.mkdir(parents=True, exist_ok=False)
    for src in (base / 'model.mjb', base / f'frame-{args.start:05d}' / 'initial.npz'):
        shutil.copyfile(src, root / src.name)
    with np.load(START_SOURCE, allow_pickle=False) as original:
        pose = original['input_palm_pose_world'][args.start, 0].astype(float)
        q0 = original['input_q_live'][args.start, 0].astype(float)
    manifest = dict(experiment=EXPERIMENT, profile='cooperative-surface', max_s=args.max_seconds, local_s=0.,
                    passes=args.passes, start=args.start, seed=args.seed, simulation_dt_s=DT, control_dt_s=CONTROL_DT,
                    model_sha256=digest(root / 'model.mjb'), initial_sha256=digest(root / 'initial.npz'),
                    base_protocol_sha256=digest(base / 'protocol.json'),
                    source_sha256={p: digest(Path(__file__).with_name(p)) for p in ('cooperative.py', 'aude.py', 'wrist.py')},
                    qualified_surface_region_load_n=SURFACE_LOAD_N, contact_goal=3,
                    majority_sustained_contact=dict(minimum_simultaneous_fingers=2, time_fraction_strictly_above=.50,
                                                    movement_speed_floor_m_s=.005),
                    whole_surface_check=dict(geodesic_radius_m=.025, required_area_fraction=1.),
                    row_spacing_rad=ROW_ANGLE, step_rad=MAX_ROUTE_STEP_RAD,
                    upper_loop_s=REPLAN_TICKS * CONTROL_DT, support_servo_s=SERVO_TICKS * CONTROL_DT,
                    near_field_input='native pad/rigid-link proximity at current and next commanded pose',
                    coverage_input='qualified physical fingertip contacts', object_mesh_input_to_global_route=False)
    write_json(root / 'protocol.json', manifest)
    jobs = [(root, arm, pose, q0, manifest) for arm in ARMS if args.arm in ('both', arm)]
    if len(jobs) == 2:
        with ProcessPoolExecutor(max_workers=2) as pool:
            runs = list(pool.map(run_arm, jobs))
    else:
        runs = [run_arm(jobs[0])]
    write_json(root / 'runs.json', runs)


if __name__ == '__main__':
    main()
