"""Run I04-c from a finger-policy grasp with joint wrist/fingertip planning."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from dataclasses import asdict, dataclass
from hashlib import sha256
from itertools import islice
import json
from pathlib import Path
from time import monotonic

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

from Module.module_2_fingertip_mcc import FingertipMCC, MCCConfig
from .contracts import ExperimentConfig
from .fingertip_hand import (DT, FINGERS, HEAD_FRACTION, MAX_HEAD_ANGLE, MAX_JOINT_DEVIATION,
                             MAX_POSTURE_RMS, SEED, SURFACE, FingertipHand,
                             build_model, reference_grasp)
from .fingertip_plan import JointPatchPlanner, Posture
from .fixtures import SurfaceGraph
from .oracle import MeasuredCoverage
from .physics import quaternion

PROTOCOL = 'i04.c.finger-dp-grasp.incidental-contact.mcc.v1'
STATE = mujoco.mjtState.mjSTATE_INTEGRATION


def write_json(path, content):
    path.write_text(json.dumps(content, ensure_ascii=False, allow_nan=False, indent=2),
                    encoding='utf-8')


def latency_ms(values):
    values = np.asarray(values)
    if not len(values):
        return dict(calls=0)
    p50, p95, p99 = np.percentile(values, [50, 95, 99])
    return dict(calls=len(values), total_s=float(values.sum() / 1000),
                mean_ms=float(values.mean()), p50_ms=float(p50), p95_ms=float(p95),
                p99_ms=float(p99), max_ms=float(values.max()))


def surface_graph():
    mesh = trimesh.load(SURFACE, force='mesh', process=False)
    graph = SurfaceGraph(mesh, coverage_radius_m=.025)
    if len(graph.vertices_m) != 8192 or graph.required_goal_count != 388:
        raise ValueError('I04-b task surface or goal denominator changed')
    return graph


class Measurements:
    def __init__(self, graph):
        self.coverage = MeasuredCoverage(graph, ExperimentConfig())
        self.steps = self.illegal_postures = self.high_steps = self.dwell = 0
        self.incidental_steps = self.mixed_steps = 0
        self.last_parts = set()
        self.part_steps, self.part_episodes = Counter(), Counter()
        self.peak_load = np.zeros(5)
        self.max_posture = np.zeros(3)
        self.min_palm = np.inf
        self.min_fraction = np.ones(4) * np.inf
        self.peak = np.zeros(4)
        self.contact_steps = np.zeros(4, dtype=int)
        self.quorum = np.zeros(5, dtype=int)

    def update(self, touch):
        safe = touch.eligible
        self.coverage.update(touch.points, touch.normals, safe)
        self.peak = np.maximum(self.peak, touch.forces)
        self.min_fraction = np.minimum(self.min_fraction, touch.min_fraction)
        self.contact_steps += safe
        self.quorum[safe.sum()] += 1
        self.steps += 1
        self.incidental_steps += bool(touch.incidental)
        self.mixed_steps += bool(touch.incidental) and safe.any()
        self.part_steps.update(touch.incidental)
        self.part_episodes.update(touch.incidental - self.last_parts)
        self.last_parts = touch.incidental
        self.peak_load = np.maximum(self.peak_load, touch.loads)
        self.illegal_postures += not touch.legal_posture
        self.max_posture = np.maximum(self.max_posture, touch.posture_metrics[:3])
        self.min_palm = min(self.min_palm, touch.posture_metrics[3])
        self.high_steps = self.high_steps + 1 if touch.loads.max() > 40. else 0
        self.dwell = self.dwell + 1 if self.coverage.visited_vertices.all() and safe.any() else 0

    def summary(self):
        cov = self.coverage
        return dict(states=self.steps, executed_s=self.steps * DT,
                    area_fraction=min(1., cov.area_fraction),
                    covered_vertices=int(cov.visited_vertices.sum()), vertices=len(cov.visited_vertices),
                    covered_goals=int(cov.visited_goals.sum()), goals=len(cov.visited_goals),
                    full_surface_complete=self.dwell >= 75 and self.illegal_postures == 0,
                    final_dwell_s=self.dwell * DT, incidental_contact_steps=self.incidental_steps,
                    incidental_with_qualified_head_steps=int(self.mixed_steps),
                    incidental_steps_by_part=dict(self.part_steps), incidental_episodes_by_part=dict(self.part_episodes),
                    peak_load_n_fingers_then_palm=self.peak_load.tolist(),
                    illegal_grasp_states=self.illegal_postures,
                    max_joint_deviation_rad=self.max_posture[0], max_posture_rms_rad=self.max_posture[1],
                    max_fingertip_axis_angle_deg=self.max_posture[2],
                    min_palm_gap_lower_bound_m=self.min_palm if self.steps else None,
                    peak_normal_force_n=self.peak.tolist(),
                    minimum_head_fraction=[float(x) if np.isfinite(x) else None for x in self.min_fraction],
                    qualified_steps_per_finger=self.contact_steps.tolist(),
                    simultaneous_qualified_steps_0_to_4=self.quorum.tolist())


class ContactTape:
    """Store the live head witness used by offline geometry scoring and R-GPIS."""
    def __init__(self):
        self.block = np.empty((5000, 4, 8), dtype=np.float32)
        self.blocks = []
        self.index = 0

    def append(self, touch):
        row = self.block[self.index]
        row[:, :3] = touch.points
        row[:, 3:6] = touch.normals
        row[:, 6] = touch.forces
        row[:, 7] = touch.loads.max()
        self.index += 1
        if self.index == len(self.block):
            self.blocks.append(self.block)
            self.block = np.empty_like(self.block)
            self.index = 0

    def save(self, output):
        rows = np.concatenate([*self.blocks, self.block[:self.index]])
        loaded = (rows[:, :, 6] >= .2) & (np.linalg.norm(rows[:, :, 3:6], axis=2) > .99)
        step, finger = np.nonzero(loaded)
        rows = rows[loaded]
        path = output / 'live_head_contacts.npz'
        np.savez_compressed(path, step=step.astype(np.int32) + 1, finger=finger.astype(np.uint8),
                            point_m=rows[:, :3], normal=rows[:, 3:6], normal_force_n=rows[:, 6],
                            max_region_load_n=rows[:, 7])
        return len(step)


@dataclass
class Leg:
    start: Posture
    end: Posture
    phase: str
    tracks_surface = False

    def distance(self):
        return (np.linalg.norm(self.end.position - self.start.position),
                (self.start.orientation.inv() * self.end.orientation).magnitude(),
                abs(self.end.joints - self.start.joints).max())

    def samples(self, count):
        u = np.arange(1, count + 1, dtype=float) / count
        blend = (u * u * (3. - 2. * u))[:, None]
        delta = (self.start.orientation.inv() * self.end.orientation).as_rotvec()
        rotation_path = self.start.orientation * Rotation.from_rotvec(blend * delta)
        return (self.start.position + blend * (self.end.position - self.start.position),
                rotation_path.as_quat()[:, [3, 0, 1, 2]],
                self.start.joints + blend * (self.end.joints - self.start.joints))

    def ticks(self):
        if self.phase == 'hold':
            return 90
        length, angle, joint = self.distance()
        speed = {'slide': .075, 'approach': .07, 'withdraw': .11}.get(self.phase, .22)
        duration = max(.06, length / speed, angle / 1.35, joint / 1.35)
        return int(np.ceil(duration / DT))

    def sample_normals(self, count, default):
        return np.broadcast_to(default, (count, 4, 3))


class ForceFeedback:
    def __init__(self):
        self.fingers = [FingertipMCC(MCCConfig(dt_s=5 * DT, virtual_mass=.08, damping=40.,
            stiffness=80., max_offset_m=.0015, max_velocity_m_s=.006,
            max_acceleration_m_s2=.2)) for _ in FINGERS]
        self.bias = np.zeros(16)
        self.tick = 0

    def command(self, hand, data, nominal, normals, touch, track):
        if self.tick % 5 == 0:
            for f, controller in enumerate(self.fingers):
                sl = slice(4 * f, 4 * f + 4)
                if not track:
                    controller.reset()
                    self.bias[sl] *= .5
                    continue
                active = touch.forces[f] > .05
                inward = -(touch.normals[f] if active else normals[f])
                point = touch.points[f] if active else data.geom_xpos[hand.pads[f]]
                cmd = controller.step(point, inward, 1., float(touch.forces[f]))
                jp = np.empty((3, hand.model.nv))
                mujoco.mj_jac(hand.model, data, jp, None, point, hand.tip_bodies[f])
                row = inward @ jp[:, hand.dof[sl]]
                self.bias[sl] = np.clip(row * cmd.offset_m / (row @ row + .0002), -.04, .04)
        self.tick += 1
        return np.clip(nominal + self.bias, hand.limits[:, 0] + .01, hand.limits[:, 1] - .01)


class Traversal:
    def __init__(self, output, seconds, *, planner_type=JointPatchPlanner, collect_contacts=False,
                 hand=None, graph=None, experiment='I04-c'):
        if not np.isfinite(seconds) or seconds < DT:
            raise ValueError('The simulation budget must be at least one finite 2 ms step')
        self.out = Path(output).resolve()
        self.out.mkdir(parents=True, exist_ok=False)
        self.budget = int(np.floor(seconds / DT + 1e-9))
        self.h = hand if hand is not None else FingertipHand(build_model())
        self.graph = graph if graph is not None else surface_graph()
        self.planner = planner_type(self.h, self.graph) if planner_type is not None else None
        initial_grasp = reference_grasp()
        self.current = self.planner.seed if self.planner is not None else Posture(*initial_grasp)
        self.data = self.h.initialize(*initial_grasp)
        self.preview = mujoco.MjData(self.h.model) if self.planner is not None else None
        self.feedback = ForceFeedback()
        self.metrics = Measurements(self.graph)
        self.contact_tape = ContactTape() if collect_contacts else None
        self.capture_preview_contacts = False
        self.attempts = np.zeros(len(self.graph.vertices_m), int)
        self.nearby_sites = {}
        self.rejected = 0
        self.next_progress = monotonic() + 15.
        self.events, self.chunks, self.rows = [], [], []
        self.last_normals = self.h.measure(self.data).normals
        initial = self.h.measure(self.data)
        if not initial.eligible.all():
            raise RuntimeError('ep71 must begin with four qualified physical fingertip heads')
        model_bytes = np.empty(mujoco.mj_sizeModel(self.h.model), np.uint8)
        mujoco.mj_saveModel(self.h.model, None, model_bytes)
        (self.out / 'model.mjb').write_bytes(model_bytes.tobytes())
        state = np.empty(mujoco.mj_stateSize(self.h.model, STATE))
        mujoco.mj_getState(self.h.model, self.data, state, STATE)
        np.savez(self.out / 'initial.npz', integration_state=state)
        write_json(self.out / 'trace_chunks.json', [])
        write_json(self.out / 'manifest.json', dict(protocol=PROTOCOL, experiment=experiment,
            mujoco=mujoco.__version__, dt_s=DT, requested_seconds=seconds, arm_joints=0,
            controller=dict(implementation='Module.module_2_fingertip_mcc.controller.FingertipMCC',
                            desired_normal_force_n=1., parameters=asdict(self.feedback.fingers[0].config),
                            dp_inference=False),
            wrist='6-DoF free palm with a mocap weld', actuated_finger_joints=16,
            planning_variables=22 if self.planner is not None else 6,
            fingers=FINGERS, source_grasp=json.loads(SEED.read_bytes()),
            minimum_distal_fraction=HEAD_FRACTION, minimum_normal_force_n=.2,
            max_joint_deviation_rad=MAX_JOINT_DEVIATION, max_posture_rms_rad=MAX_POSTURE_RMS,
            max_fingertip_axis_angle_deg=MAX_HEAD_ANGLE, incidental_hand_contact='allowed_and_logged',
            coverage_exclusion_n=8., sustained_stop_n=40., sustained_stop_s=.2,
            coverage_radius_m=.025, coverage_inherited=False,
            surface_sha256=sha256(SURFACE.read_bytes()).hexdigest(),
            model_sha256=sha256(model_bytes).hexdigest()))

    def flush(self):
        if not self.rows:
            return
        keys = ('qpos', 'palm_command', 'finger_command', 'normal_force_n', 'min_head_fraction',
                'region_load_n', 'incidental_contact', 'phase')
        name = f'trace_{len(self.chunks):04d}.npz'
        np.savez_compressed(self.out / name, **dict(zip(keys, map(np.asarray, zip(*self.rows)))))
        self.chunks.append(dict(file=name, states=len(self.rows),
                                sha256=sha256((self.out / name).read_bytes()).hexdigest()))
        self.rows.clear()
        write_json(self.out / 'trace_chunks.json', self.chunks)
        write_json(self.out / 'progress.json', self.metrics.summary())

    def goal(self):
        cov = self.metrics.coverage
        available = ~cov.visited_vertices & (self.attempts < 5)
        if not available.any():
            # A concave crease can be inaccessible along its vertex normal.
            # Route to nearby sites on the original manifold; the actual
            # contact ledger retains the same 25 mm radius and denominator.
            available = ~cov.visited_vertices & (self.attempts < 25)
        ids = self.graph.required_vertices
        remaining = ids[available[ids]]
        if not len(remaining):
            remaining = np.flatnonzero(available)
        if not len(remaining):
            return None
        tips = self.data.geom_xpos[self.h.pads]
        offsets = self.graph.vertices_m[remaining, None] - tips
        distance = np.linalg.norm(offsets, axis=2).min(axis=1)
        up = self.current.orientation.apply(self.planner.seed.orientation.inv().apply([0., 0., 1.]))
        score = distance + .045 * (1. - self.graph.normals[remaining] @ up) + .12 * self.attempts[remaining]
        return int(remaining[np.argmin(score)])

    def alternate_site(self, vertex, attempt):
        if vertex not in self.nearby_sites:
            distances = self.graph.distances_from(vertex)
            near = np.flatnonzero((distances > 0) & (distances < .020))
            # Sample several sides of a crease instead of retrying the same
            # normal. The five-millimetre reserve allows for tracking error.
            sites = np.r_[vertex, near]
            features = np.column_stack((self.graph.vertices_m[sites],
                                        .015 * self.graph.normals[sites]))
            closest = np.full(len(sites), np.inf)
            order = []
            current = 0
            for _ in range(min(20, len(near))):
                closest = np.minimum(closest, np.linalg.norm(features - features[current], axis=1))
                current = int(closest.argmax())
                order.append(int(sites[current]))
            self.nearby_sites[vertex] = order or [vertex]
        order = self.nearby_sites[vertex]
        return order[(attempt - 5) % len(order)]

    def routes(self, patch):
        end = patch.posture
        start = self.current
        yield [Leg(start, end, 'slide')]
        palm_up = self.planner.seed.orientation.inv().apply([0., 0., 1.])
        n0, n1 = start.orientation.apply(palm_up), end.orientation.apply(palm_up)
        for distance in (.045, .14):
            a, b = start.lift(n0, distance), end.lift(n1, distance)
            yield [Leg(start, a, 'withdraw'), Leg(a, b, 'transfer'), Leg(b, end, 'approach')]
        # A large turn can intersect the bottle even between two local hover
        # poses. Connect the hovers on a sphere outside the entire assembly.
        center = self.graph.mesh.bounds.mean(axis=0)
        a, b = start.lift(n0, .12), end.lift(n1, .12)
        u, v = a.position - center, b.position - center
        from .oracle import align_vector
        arc = align_vector(u, v).as_rotvec()
        turns = (a.orientation.inv() * b.orientation).as_rotvec()
        poses = [a]
        for t in np.linspace(0., 1., 7):
            position = center + .58 * Rotation.from_rotvec(t * arc).apply(u / np.linalg.norm(u))
            poses.append(Posture(position, a.orientation * Rotation.from_rotvec(t * turns),
                                 a.joints + t * (b.joints - a.joints)))
        poses.append(b)
        yield [Leg(start, a, 'withdraw'), *[Leg(x, y, 'transfer') for x, y in zip(poses, poses[1:])],
               Leg(b, end, 'approach')]

    def geometry_clear(self, route):
        planner, d = self.planner, self.planner.data
        for leg in route:
            length, angle, joints = leg.distance()
            count = max(2, int(np.ceil(max(length / .004, angle / .045, joints / .06))))
            for p, quat, q in zip(*leg.samples(count)):
                d.qpos[:3], d.qpos[3:7], d.qpos[self.h.q] = p, quat, q
                mujoco.mj_kinematics(self.h.model, d)
                if not planner.buffered(maximum_depth=.004):
                    return False
        return True

    def forecast(self, route, normals):
        """Test a continuous candidate from the measured root before issuing any command."""
        h, d = self.h, self.preview
        mujoco.mj_copyData(d, h.model, self.data)
        feedback = copy.deepcopy(self.feedback)
        touch = h.measure(d)
        commands = []
        if self.capture_preview_contacts:
            self.preview_contacts = []
        remaining = self.budget - self.metrics.steps
        high_steps = self.metrics.high_steps
        for leg in route:
            count = leg.ticks()
            samples = list(zip(*leg.samples(count), leg.sample_normals(count, normals)))
            last = (leg.start.position, quaternion(leg.start.orientation), leg.start.joints, samples[0][3])
            index = loss = 0
            while index < count:
                # Require a measured qualifying tip before advancing after a
                # contact gap. Brief losses get an in-place MCC recovery.
                paused = leg.tracks_surface and loss >= 8
                if not paused:
                    last = samples[index]
                    index += 1
                p, quat, nominal, local_normals = last
                ctrl = feedback.command(h, d, nominal, local_normals, touch,
                                        leg.phase in ('slide', 'hold', 'approach'))
                d.mocap_pos[h.driver], d.mocap_quat[h.driver], d.ctrl[:] = p, quat, ctrl
                mujoco.mj_step(h.model, d)
                mujoco.mj_forward(h.model, d)
                touch = h.measure(d)
                high_steps = high_steps + 1 if touch.loads.max() > 40. else 0
                if leg.tracks_surface:
                    loss = 0 if touch.eligible.any() else loss + 1
                    if loss >= 120:
                        return None
                if (not touch.legal_posture or high_steps >= 100
                        or not np.isfinite(d.qpos).all()):
                    return None
                commands.append((p, quat, ctrl, 'recontact' if paused else leg.phase))
                if self.capture_preview_contacts:
                    self.preview_contacts.append(touch)
                if len(commands) >= remaining:
                    return commands, feedback
        # A completed patch must hand off in contact, with force reserve for
        # the next movement. A budget-limited prefix has no such handoff.
        if not touch.eligible.any() or touch.loads.max() >= 7.5:
            return None
        return commands, feedback

    def execute(self, forecast):
        h, d = self.h, self.data
        commands, self.feedback = forecast
        for p, quat, ctrl, phase in commands:
            d.mocap_pos[h.driver], d.mocap_quat[h.driver], d.ctrl[:] = p, quat, ctrl
            mujoco.mj_step(h.model, d)
            mujoco.mj_forward(h.model, d)
            touch = h.measure(d)
            self.record(touch, np.r_[p, quat], ctrl, phase)
            if not touch.legal_posture:
                return 'ILLEGAL_FINGER_DP_GRASP'
            if self.metrics.high_steps >= 100:
                return 'SUSTAINED_HIGH_FORCE'
            if self.metrics.dwell >= 75:
                return 'FULL_MEASURED_HEAD_SURFACE'
        if not np.allclose(d.qpos, self.preview.qpos, rtol=0., atol=1e-10):
            raise RuntimeError('Physical state diverged from the accepted native lookahead')
        return None

    def record(self, touch, palm, ctrl, phase):
        self.metrics.update(touch)
        if self.contact_tape is not None:
            self.contact_tape.append(touch)
        self.rows.append((self.data.qpos.copy(), palm, ctrl, touch.forces.astype('f4'),
                          touch.min_fraction.astype('f4'), touch.loads.astype('f4'),
                          bool(touch.incidental), phase))
        if len(self.rows) >= 5000:
            self.flush()

    def attempt(self, vertex):
        count = int(self.attempts[vertex])
        self.attempts[vertex] += 1
        site = vertex if count < 5 else self.alternate_site(vertex, count)
        for patch in islice(self.planner.candidates(site, self.current, count), 6):
            for route in self.routes(patch):
                if not self.geometry_clear(route):
                    self.rejected += 1
                    continue
                route.append(Leg(patch.posture, patch.posture, 'hold'))
                prediction = self.forecast(route, patch.normals)
                if prediction is None:
                    self.rejected += 1
                    continue
                self.events.append(dict(start_step=self.metrics.steps, vertex=vertex,
                    target_site=site, anchor=patch.anchor,
                    phases=[leg.phase for leg in route], wrist_xyz=patch.posture.position.tolist(),
                    wrist_wxyz=quaternion(patch.posture.orientation).tolist(),
                    finger_targets_rad=patch.posture.joints.tolist(), surface_targets_m=patch.points.tolist(),
                    surface_normals=patch.normals.tolist(), planned_gaps_m=patch.gaps.tolist()))
                reason = self.execute(prediction)
                self.events[-1]['end_step'] = self.metrics.steps
                self.current, self.last_normals = patch.posture, patch.normals
                return reason
        return None

    def run(self):
        reason = None
        hold = Leg(self.current, self.current, 'hold')
        while self.metrics.steps < self.budget and reason is None:
            if self.metrics.steps == 0 or self.metrics.coverage.visited_vertices.all():
                prediction = self.forecast([hold], self.last_normals)
                if prediction is None:
                    reason = 'NO_SAFE_CONTACT_HOLD'
                else:
                    reason = self.execute(prediction)
            else:
                vertex = self.goal()
                if vertex is None:
                    reason = 'PATCH_SEARCH_EXHAUSTED'
                else:
                    reason = self.attempt(vertex)
                hold = Leg(self.current, self.current, 'hold')
                if monotonic() >= self.next_progress:
                    print(f'{self.metrics.steps * DT:.2f} s, {self.metrics.coverage.area_fraction:.2%}, '
                          f'{len(self.events)} patches, {self.rejected} rejected routes', flush=True)
                    self.next_progress = monotonic() + 15.
        self.flush()
        write_json(self.out / 'planned_patches.json', self.events)
        cov = self.metrics.coverage
        np.savez_compressed(self.out / 'coverage.npz', visited_vertices=cov.visited_vertices,
                            visited_goals=cov.visited_goals, required_vertices=self.graph.required_vertices)
        summary = dict(experiment='I04-c', stop_reason=reason or 'SIMULATION_BUDGET',
                       planned_patches=len(self.events), rejected_routes=self.rejected,
                       joint_ik_steps=self.planner.solver_steps, **self.metrics.summary())
        if self.contact_tape is not None:
            summary['live_head_observations'] = self.contact_tape.save(self.out)
        write_json(self.out / 'summary.json', summary)
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    run = commands.add_parser('run')
    run.add_argument('--output', type=Path, required=True)
    run.add_argument('--seconds', type=float, default=9000.)
    run.add_argument('--planner', choices=('continuous', 'endpoint'), default='continuous')
    audit = commands.add_parser('audit')
    audit.add_argument('--case', type=Path, required=True)
    render = commands.add_parser('render')
    render.add_argument('--case', type=Path, required=True)
    render.add_argument('--output', type=Path, required=True)
    render.add_argument('--speed', type=float, default=8.)
    render.add_argument('--surface-evidence', type=Path, help='Measured contact overlay from compare-surface')
    comparison = commands.add_parser('compare-dp')
    from .fingertip_dp_compare import add_arguments
    add_arguments(comparison)
    report = commands.add_parser('compare-report', help='historical posture-gated endpoint diagnostics')
    report.add_argument('--case', type=Path, required=True)
    report.add_argument('--dp-case', type=Path, required=True)
    surface = commands.add_parser('compare-surface', help='compare contact area, R-GPIS and computation time')
    from .fingertip_surface_report import add_arguments as add_surface_arguments
    add_surface_arguments(surface)
    args = parser.parse_args()
    if args.action == 'run':
        if args.planner == 'continuous':
            from .fingertip_continuous import ContinuousTraversal
            result = ContinuousTraversal(args.output, args.seconds).run()
        else:
            result = Traversal(args.output, args.seconds).run()
    elif args.action == 'compare-dp':
        from .fingertip_dp_compare import run_comparison
        result = run_comparison(args.case, args.output, args.policy_python,
                                args.device, args.seed, args.seconds, args.schedule, args.sampler)
    elif args.action == 'compare-report':
        from .fingertip_dp_report import run_report
        result = run_report(args.case, args.dp_case)
    elif args.action == 'compare-surface':
        from .fingertip_surface_report import compare
        result = compare(args.case, args.dp_case, args.planning_case or args.case, args.output)
    else:
        from .fingertip_replay import audit_case, render_case
        result = (audit_case(args.case) if args.action == 'audit'
                  else render_case(args.case, args.output, args.speed, args.surface_evidence))
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
