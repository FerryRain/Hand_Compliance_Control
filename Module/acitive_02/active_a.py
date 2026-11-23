"""Run active-02-a from a known start with no object geometry in either planner."""

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

from .active_gpis import NEIGHBORS, NOISE_VARIANCE, SIGN_OFFSET_M, TRUST_RADIUS_M, VOXEL_M
from .active_planner import ActiveSurfacePlanner, GraspEnvelope, LONG_DISTAL, REPLAN_TICKS, SERVO_TICKS
from .aude import AudeExplorer, DEFAULT_STEP_M, JointIK, state_posture
from .experiment import (ARMS, CONTROL_DT, CONTROL_STEPS, DT, HISTORY_TICKS, LOOKAHEAD_STEPS, PLAN_TICKS,
                         STATE, digest, longest_run, policy_observation, record, step_plant, write_json)
from .full import TRACE_KEYS, blend_wrist
from .surface_scan import SURFACE_LOAD_N
from .wrist import WRIST_WELD_TIME_S, WristFeedback
from ..i04_geometric_traversal.fingertip_hand import FingertipHand, build_model
from ..i04_geometric_traversal.fingertip_plan import Posture
from ..i04_geometric_traversal.fingertip_traversal import ForceFeedback, Leg, latency_ms
from ..i04_geometric_traversal.physics import quaternion, rotation


EXPERIMENT = 'active-02-a'
PROFILE = 'tactile-gpis-surface'
REFERENCE = Path('Module/generated/acitive-02_cooperative_20261120')
MAX_FINGER_RATE_RAD_S = 2.


class GraspFeedback(ForceFeedback):
    def __init__(self, grasp):
        super().__init__()
        self.grasp = grasp
        self.previous = grasp.reference.copy()

    def command(self, hand, data, nominal, normals, touch, track):
        target = self.grasp.bound(super().command(hand, data, nominal, normals, touch, track))
        step = MAX_FINGER_RATE_RAD_S * DT
        self.previous = self.grasp.bound(self.previous + np.clip(target - self.previous, -step, step))
        return self.previous


def save_model(model, path):
    data = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, None, data)
    path.write_bytes(data.tobytes())


def robot_state(plant, robot, source, target):
    target.qpos[:7] = source.qpos[:7]
    target.qpos[robot.q] = source.qpos[plant.q]
    mujoco.mj_kinematics(robot.model, target)
    mujoco.mj_comPos(robot.model, target)


def run_arm(job):
    root, arm, protocol = job
    case = root / arm
    case.mkdir()
    started = perf_counter()
    plant_model = mujoco.MjModel.from_binary_path('pear.mjb', {'pear.mjb': (root / 'model.mjb').read_bytes()})
    robot_model = mujoco.MjModel.from_binary_path('hand.mjb', {'hand.mjb': (root / 'planning-hand.mjb').read_bytes()})
    data, sensed = mujoco.MjData(plant_model), mujoco.MjData(robot_model)
    with np.load(root / 'initial.npz', allow_pickle=False) as start:
        initial, warm = start['integration_state'], start['warm_policy_observations']
    mujoco.mj_setState(plant_model, data, initial, STATE)
    mujoco.mj_forward(plant_model, data)
    seed = (data.qpos[:3].copy(), rotation(data.qpos[3:7]), data.qpos[7:].copy())
    plant = FingertipHand(plant_model, object_name='pear', grasp=seed, min_head_fraction=None, surface_queries=False)
    robot = FingertipHand(robot_model, object_name='fixture', grasp=seed, min_head_fraction=None, surface_queries=False)
    if robot.objects or robot_model.ngeom != protocol['planning_hand_geometry_count']:
        raise ValueError('Unexpected planning scene geometry')
    robot_state(plant, robot, data, sensed)
    np.testing.assert_allclose(data.geom_xpos[plant.pads], sensed.geom_xpos[robot.pads], rtol=0., atol=1.e-8)
    np.testing.assert_allclose(data.geom_xmat[plant.pads], sensed.geom_xmat[robot.pads], rtol=0., atol=1.e-8)
    touch = plant.measure(data)
    if not touch.measured_heads.all() or touch.loads.max() > SURFACE_LOAD_N:
        raise ValueError('Expected the shared measured four-contact start')
    normals = touch.normals.copy()
    live = state_posture(robot, sensed)
    grasp = GraspEnvelope(live.joints, robot.limits)
    ik = JointIK(robot, limits=grasp.limits)
    explorer = AudeExplorer(robot, ik, protocol['seed'], DEFAULT_STEP_M)
    planner = ActiveSurfacePlanner(ik, grasp, live, protocol['passes'], CONTROL_DT)
    explorer.observe(sensed, touch)
    planner.observe(sensed, touch, 0)
    wrist, fingers = WristFeedback(palm_proximity=False), GraspFeedback(grasp)
    raw = stabilized = np.r_[live.position, quaternion(live.orientation)]
    rows = [record(data, plant, touch, raw, wrist, live.joints, live.joints, 'initial', 0.)]
    budget = round(protocol['max_s'] / CONTROL_DT)
    policy = None
    if arm == ARMS[1]:
        from ..i04_geometric_traversal.fingertip_local_policy import LocalFingerPolicy

        policy = LocalFingerPolicy(protocol['seed'], budget + HISTORY_TICKS)
    tick = 0
    counts = np.zeros(5, dtype=int)
    planning_ms = []
    termination = 'time_budget'
    with (policy if policy is not None else nullcontext()), (case / 'plans.jsonl').open('w', encoding='utf-8') as stream:
        if policy is not None:
            for i, packet in enumerate(warm):
                policy.command(i, packet)
        while tick < budget:
            if tick % PLAN_TICKS == 0:
                aude = explorer.plan(sensed, touch)
            phase = tick % REPLAN_TICKS
            if phase == 0:
                since = perf_counter()
                live = state_posture(robot, sensed)
                goal, mode, normal = planner.plan(live, aude.posture, touch, tick * CONTROL_DT)
                if planner.done:
                    termination = 'scan_pass_budget'
                    break
                origin = Posture(raw[:3], rotation(raw[3:]), grasp.bound(live.joints))
                pos, quat, joint_track = Leg(origin, goal, mode).samples(REPLAN_TICKS)
                path = np.column_stack((pos, quat))
                elapsed = (perf_counter() - since) * 1000
                planning_ms.append(elapsed)
                plan = dict(tick=tick, time_s=tick * CONTROL_DT, mode=mode,
                            wrist_target=np.r_[goal.position, quaternion(goal.orientation)].tolist(),
                            aude_joint_target=goal.joints.tolist(), original_aude_joint_target=aude.posture.joints.tolist(),
                            virtual_contacts_m=aude.targets.tolist(), route_pass=planner.stage + 1,
                            route_index=planner.route_index, physical_contact_count=int(touch.measured_heads.sum()),
                            gpis_contact_voxels=len(planner.surface.samples),
                            last_surface_observation_tick=planner.surface.last_observation_tick,
                            chosen_direction_gpis_variance=planner.novelty, planning_ms=elapsed)
                stream.write(json.dumps(plan, allow_nan=False) + '\n')
            raw = path[phase]
            supported = planner.stabilize(raw, sensed.qpos[robot.q], touch, normal, wrist, tick % SERVO_TICKS == 0)
            aude_nominal = grasp.bound(joint_track[phase])
            nominal = aude_nominal
            if policy is not None:
                ahead = path[min(phase + LOOKAHEAD_STEPS // CONTROL_STEPS, len(path) - 1)]
                supported_future = planner.stabilize(ahead, sensed.qpos[robot.q], touch, normal, wrist, False)
                packet = policy_observation(plant, data, wrist.forecast(supported_future))
                nominal = policy.command(HISTORY_TICKS + tick, packet)
            segment = blend_wrist(stabilized, supported)
            normals[touch.measured_heads] = touch.normals[touch.measured_heads]
            touch = step_plant(plant, data, fingers, wrist, touch, segment, nominal, normals)
            stabilized = segment[-1]
            tick += 1
            robot_state(plant, robot, data, sensed)
            explorer.observe(sensed, touch)
            planner.observe(sensed, touch, tick)
            count = int(touch.measured_heads.sum()) if touch.loads.max() <= SURFACE_LOAD_N else 0
            counts[count] += 1
            rows.append(record(data, plant, touch, stabilized, wrist, nominal, aude_nominal, mode, perf_counter() - started))
            if not np.isfinite(data.qpos).all() or abs(data.time - CONTROL_DT * tick) > 1.e-6:
                raise ValueError(f'{arm}: invalid physics at control tick {tick}')
            if tick % 1500 == 0:
                stream.flush()
                print(f'{arm} {tick * CONTROL_DT:.0f}s: pass {planner.stage+1} {planner.route_index}/{len(planner.route)}; '
                      f'≥2 {counts[2:].sum()/tick:.1%}; ≥3 {counts[3:].sum()/tick:.1%}; '
                      f'{len(planner.surface.samples)} measured GPIS cells', flush=True)
    trace = {key: np.asarray(value) for key, value in zip(TRACE_KEYS, zip(*rows))}
    np.savez_compressed(case / 'trace.npz', **trace)
    planner.surface.refresh()
    np.savez_compressed(case / 'online-gpis-observations.npz', points_m=planner.surface.points,
                        normals_outward=planner.surface.normals)
    valid = trace['pad_valid'][1:]
    load = trace['region_load_n'][1:].max(axis=1)
    qualified = valid & (load <= SURFACE_LOAD_N)[:, None]
    moving = np.linalg.norm(np.diff(trace['qpos'][:, :3], axis=0), axis=1) >= .005 * CONTROL_DT
    distal = trace['qpos'][1:, plant.q][:, LONG_DISTAL]
    joints = trace['qpos'][1:, plant.q]
    issued = trace['finger_command'][1:]
    projection_error = np.max(abs(joints - grasp.bound(joints)), axis=1)
    np.testing.assert_allclose(grasp.bound(issued), issued, atol=1.e-9, rtol=0.)
    if np.any(trace['palm_proximity_drive_n']):
        raise ValueError('Object proximity entered the tactile-only wrist loop')
    result = dict(experiment=EXPERIMENT, profile=PROFILE, arm=arm, seconds=tick * CONTROL_DT,
                  stop=termination, budget_s=protocol['max_s'], wall_execution_s=perf_counter() - started,
                  initial_state_sha256=sha256(initial.tobytes()).hexdigest(), start_frame=protocol['start'],
                  scan_passes=planner.finished, final_route_pass=planner.stage + 1, final_route_index=planner.route_index,
                  contact_voxels=len(planner.surface.samples), max_region_load_n=float(load.max()),
                  max_palm_load_n=float(trace['region_load_n'][1:, -1].max()),
                  longest_above_40n_s=longest_run(load > 40.) * CONTROL_DT,
                  longest_full_loss_s=longest_run(~valid.any(axis=1)) * CONTROL_DT,
                  qualified_contact_count_fractions=(counts / tick).tolist(),
                  qualified_at_least_two_contacts=float(counts[2:].sum() / tick),
                  qualified_at_least_three_contacts=float(counts[3:].sum() / tick),
                  qualified_moving_at_least_two_contacts=(float(np.mean(qualified[moving].sum(1) >= 2))
                                                           if moving.any() else 0.),
                  qualified_per_finger=qualified.mean(axis=0).tolist(),
                  planning_timing=latency_ms(planning_ms),
                  aude_nominal_difference_max_rad=float(np.max(abs(trace['finger_nominal'] - trace['aude_joint_nominal']))),
                  posture=dict(bounds_rad=grasp.limits.tolist(), measured_minimum_long_distal_deg=np.rad2deg(distal.min(0)).tolist(),
                               any_reverse_long_distal_fraction=float(np.any(distal < 0., axis=1).mean()),
                               three_long_distal_below_negative_10deg_fraction=float(np.all(distal < np.deg2rad(-10.), axis=1).mean()),
                               issued_command_outside_envelope_fraction=float(np.any(abs(issued - grasp.bound(issued)) > 1.e-9, axis=1).mean()),
                               measured_outside_envelope_by_more_than_2deg_fraction=float(np.mean(projection_error > np.deg2rad(2.))),
                               nominal_required_envelope_projection_fraction=float(np.any(
                                   abs(trace['finger_nominal'][1:] - grasp.bound(trace['finger_nominal'][1:])) > 1.e-9, axis=1).mean())))
    if policy is not None:
        result['policy'] = policy.info
        result['policy_timing'] = policy.report(case)
    write_json(case / 'run.json', result)
    print(f'{arm}: {tick * CONTROL_DT:.2f}s complete, {len(planner.finished)} passes, '
          f'≥2 {result["qualified_at_least_two_contacts"]:.1%}', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path, default=REFERENCE)
    parser.add_argument('--max-seconds', type=float, default=1300.)
    parser.add_argument('--passes', type=int, choices=(1, 2, 3), default=3)
    parser.add_argument('--seed', type=int, default=20261120)
    parser.add_argument('--arm', choices=('both', *ARMS), default='both')
    args = parser.parse_args()
    if not np.isfinite(args.max_seconds) or args.max_seconds < CONTROL_DT * REPLAN_TICKS:
        parser.error('A finite budget of at least one planning interval is required')
    root, reference = args.output.resolve(), args.reference.resolve()
    root.mkdir(parents=True, exist_ok=False)
    for name in ('model.mjb', 'initial.npz'):
        shutil.copyfile(reference / name, root / name)
    robot = build_model(surface=None, collision_parts=(), object_name='fixture', model_name='active-02-a-hand',
                        wrist_time_constant_s=WRIST_WELD_TIME_S)
    save_model(robot, root / 'planning-hand.mjb')
    baseline = json.loads((reference / 'protocol.json').read_text(encoding='utf-8'))
    sources = [Path(__file__), *(Path(__file__).with_name(name) for name in ('active_gpis.py', 'active_planner.py',
                                                                          'aude.py', 'wrist.py'))]
    protocol = dict(experiment=EXPERIMENT, profile=PROFILE, max_s=args.max_seconds, passes=args.passes,
                    seed=args.seed, start=baseline['start'], simulation_dt_s=DT, control_dt_s=CONTROL_DT,
                    reference_experiment='active-02-b', reference_protocol_sha256=digest(reference / 'protocol.json'),
                    model_sha256=digest(root / 'model.mjb'), initial_sha256=digest(root / 'initial.npz'),
                    planning_hand_sha256=digest(root / 'planning-hand.mjb'), planning_hand_geometry_count=robot.ngeom,
                    source_sha256={source.name: digest(source) for source in sources},
                    initial_geometry_input='known fixture origin and matched initial hand state',
                    initial_surface_samples='empty before the first renewed physical touch measurement',
                    object_mesh_input_to_planner=False, candidate_native_collision_queries=False,
                    wrist_native_proximity=False, planning_wrist_proximity_input='online tactile GPIS',
                    signed_gpis=dict(kernel='GPyTorch Matérn 3/2', voxel_m=VOXEL_M, sign_offset_m=SIGN_OFFSET_M,
                                     neighbors=NEIGHBORS, trust_radius_m=TRUST_RADIUS_M, noise_variance=NOISE_VARIANCE),
                    qualified_surface_region_load_n=SURFACE_LOAD_N, maximum_finger_rate_rad_s=MAX_FINGER_RATE_RAD_S,
                    hand_envelope='measured initial grasp, positive long-finger flexion, thumb opposition sector',
                    majority_sustained_contact=dict(minimum_simultaneous_fingers=2,
                                                    time_fraction_strictly_above=.5, movement_speed_floor_m_s=.005),
                    whole_surface_check=dict(geodesic_radius_m=.025, required_area_fraction=1.))
    write_json(root / 'protocol.json', protocol)
    jobs = [(root, arm, protocol) for arm in ARMS if args.arm in ('both', arm)]
    if len(jobs) == 2:
        with ProcessPoolExecutor(max_workers=2) as pool:
            runs = list(pool.map(run_arm, jobs))
    else:
        runs = [run_arm(jobs[0])]
    write_json(root / 'runs.json', runs)


if __name__ == '__main__':
    main()
