"""Contact-first mustard scanning: surface strokes, bounded recovery, explicit lifts.

Geometry produces Cartesian poses. Only native MuJoCo contacts earn coverage.
The old whole-surface experiment and its frozen controller are not modified.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from time import perf_counter

import mujoco
import numpy as np
from scipy.sparse.csgraph import dijkstra
import trimesh

from .contracts import ExperimentConfig, WristTarget, WholeHandTarget
from .physics import (pose, rotation, quaternion, joint_state, set_control,
                      tip_positions, contact_measurement, BasicPassiveMCC, set_joint_state)
from .full_surface import (build_scene, acquire, FullSurfacePlanner,
    FullSurfaceOracle, pad_support_vector, endpoint_arm_offset, shifted_target, complete_surface)
from .oracle import MeasuredCoverage, align_vector
from .recovery import SharedContactRecovery
from .sliding_metrics import analyze_trace
from .active_fingers import ActiveFingerSearch
from .planning import SharedGeometricTrajectoryPlanner
from .sliding_safety import SlidingSafetyPlanner, arm_self_contacts


def unit(x):
    return np.asarray(x) / max(float(np.linalg.norm(x)), 1e-12)


def resample_surface(graph, points, spacing=.002):
    """Smooth mesh-edge corners, project back to the unchanged exterior."""
    points = np.asarray(points).copy()
    if len(points) > 2:
        for _ in range(3):
            points[1:-1] = .25*points[:-2] + .5*points[1:-1] + .25*points[2:]
    length = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    keep = np.r_[True, np.diff(length) > 1e-9]
    points, length = points[keep], length[keep]
    if length[-1] < .005:
        raise ValueError('Stroke is too short')
    arclength = np.linspace(0., length[-1], int(np.ceil(length[-1]/spacing))+1)
    points = np.column_stack([np.interp(arclength, length, points[:, j]) for j in range(3)])
    points, _, _ = trimesh.proximity.closest_point(graph.mesh, points)
    normals = []
    for point in points:
        distance, indices = graph._tree.query(point, k=12)
        normal = unit(np.sum(graph.normals[indices]/np.maximum(distance[:, None], .002), axis=0))
        normals.append(normal)
    return points, np.asarray(normals)


@dataclass
class SurfaceStroke:
    q: np.ndarray
    positions: np.ndarray
    normals: np.ndarray
    wrist: np.ndarray
    distances: np.ndarray
    finger: int
    goal: int
    normal_in_wrist: np.ndarray

    @property
    def length(self):
        return float(self.distances[-1])

    def sample(self, distance):
        i = min(int(np.searchsorted(self.distances, distance, side='right')), len(self.q)-1)
        i = max(i, 1)
        u = float(np.clip((distance-self.distances[i-1]) /
                         max(self.distances[i]-self.distances[i-1], 1e-9), 0., 1.))
        q = (1-u)*self.q[i-1] + u*self.q[i]
        # Control normal comes from exposed wrist orientation and a causal
        # measured normal at stroke start, never a GT contact label.
        r0, r1 = rotation(self.wrist[i-1, 3:]), rotation(self.wrist[i, 3:])
        r = rotation(self.wrist[i-1, 3:])
        delta = (r1*r0.inv()).as_rotvec()
        from scipy.spatial.transform import Rotation
        r = Rotation.from_rotvec(u*delta)*r
        return q, r.apply(self.normal_in_wrist)


class SurfaceStrokePlanner:
    def __init__(self, h, graph, config, cell):
        self.h, self.graph, self.cell = h, graph, cell
        self.planner = SlidingSafetyPlanner(h, config)
        self.attempts = np.zeros(len(graph.vertices_m))
        self.heading = None

    def plan(self, data, coverage, forces, contacts, measured_normals, active):
        """Try adjacent surface continuations before permitting any air move."""
        if not np.any(active):
            return None, [{'reason': 'NO_MEASURED_CONTACT'}]
        failures = []
        self.short_prefixes = []
        # A free finger can reach first while the previous anchor is near a
        # joint limit. Let each measured support propose the next stroke.
        for finger in np.argsort(-np.where(active, forces, -1.)):
            if not active[finger]:
                continue
            stroke, reasons = self._plan_finger(data, coverage, contacts, measured_normals, int(finger))
            failures.extend(dict(reason=r.get('reason'), finger=int(finger), **{
                k:v for k,v in r.items() if k != 'reason'}) for r in reasons)
            if stroke is not None:
                return stroke, failures
        # Use a short, already checked prefix only after every measured
        # anchor failed to provide a normal >=15 mm continuation.
        if self.short_prefixes:
            stroke = max(self.short_prefixes, key=lambda candidate: candidate.length)
            self.heading = unit(stroke.positions[-1]-stroke.positions[0])
            failures.append(dict(reason='SHORT_CONTACT_BRIDGE', finger=stroke.finger,
                                 feasible_prefix_m=stroke.length))
            return stroke, failures
        return None, failures

    def _plan_finger(self, data, coverage, contacts, measured_normals, finger):
        h, graph, planner = self.h, self.graph, self.planner
        contact = contacts[finger]-h.object_position_m
        starts, _ = graph.oriented_nearest_vertices(contact[None], measured_normals[finger][None])
        start = int(starts[0])
        distances, predecessors = dijkstra(graph.adjacency, directed=False, indices=start,
                                            return_predecessors=True)
        remaining = np.flatnonzero(~coverage.visited_vertices)
        candidates = remaining[(distances[remaining] >= .035) & (distances[remaining] <= .18)]
        if not len(candidates):
            # Covered patches are valid contact-preserving bridges. Follow
            # the shortest surface route toward a still-unmeasured frontier.
            candidates = remaining[np.isfinite(distances[remaining]) & (distances[remaining] >= .015)]
            if not len(candidates):
                return None, [{'reason': 'NO_SURFACE_FRONTIER'}]
            candidates = candidates[np.argsort(distances[candidates]+.06*self.attempts[candidates])[:64]]
        direction = graph.vertices_m[candidates]-contact
        direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-9)
        # Continuation cost: prefer useful 8–14 cm strokes and compatible
        # normals/heading. Failed geometry is deferred, never removed.
        score = (abs(distances[candidates]-.11) + .06*self.attempts[candidates]
                 + .04*(1-graph.normals[candidates]@graph.normals[start]))
        if self.heading is not None:
            score += .035*(1-direction@self.heading)
        ordered = candidates[np.argsort(score)]
        choices = []
        for vertex in ordered:
            if all(np.linalg.norm(graph.vertices_m[vertex]-graph.vertices_m[v]) > .025 for v in choices):
                choices.append(int(vertex))
            if len(choices) == 8:
                break
        failures = []
        for goal in choices:
            self.attempts[goal] += 1
            path = [goal]
            while path[-1] != start and predecessors[path[-1]] >= 0:
                path.append(int(predecessors[path[-1]]))
            if path[-1] != start:
                continue
            points = np.vstack([contact, graph.vertices_m[path[::-1]]])
            points, normals = resample_surface(graph, points)
            arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
            keep = max(2, int(np.searchsorted(arc, .18, side='right')))
            points, normals = points[:keep], normals[:keep]
            current = pose(h, data)
            r0 = rotation(current[3:])
            local_tips = r0.inv().apply(tip_positions(h, data)-current[:3])
            geom = int(h.tip_geom_ids[finger])
            local_center = r0.inv().apply(data.geom_xpos[geom]-current[:3])
            local_frame = r0.inv().as_matrix() @ data.geom_xmat[geom].reshape(3, 3)
            support = pad_support_vector(h.model, geom, r0.as_matrix()@local_frame, normals[0])
            predicted = h.object_position_m+points[0]+support-r0.apply(local_center)
            # Keep the real acquired preload and exact initial wrist pose.
            calibration = r0.inv().apply(current[:3]-predicted)
            q = joint_state(h, data)
            qs, wrists = [q.copy()], [current.copy()]
            r = r0
            failure = None
            for k in range(1, len(points)):
                if normals[k-1]@normals[k] < np.cos(.50):
                    failure = 'NORMAL_CURVATURE'; break
                transported = align_vector(normals[k-1], normals[k])*r
                from scipy.spatial.transform import Rotation
                feasible = False
                # Small pad-anchored roll can clear a neighboring link while
                # retaining surface position, before asking for any lift.
                for roll in (0., .04, -.04, .08, -.08):
                    trial_r = Rotation.from_rotvec(normals[k]*roll)*transported
                    support = pad_support_vector(h.model, geom, trial_r.as_matrix()@local_frame, normals[k])
                    p = h.object_position_m+points[k]+support-trial_r.apply(local_center)+trial_r.apply(calibration)
                    wrist = WristTarget(np.r_[p, quaternion(trial_r)])
                    target = wrist if self.cell == 'A-MCC' else WholeHandTarget(wrist, p+trial_r.apply(local_tips))
                    # For a 2 mm continuation prefer the damped local update.
                    # Unregularized bounded least squares can jump along the
                    # arm's redundant direction despite a tiny pose change.
                    next_q, ep, er, ef = SharedGeometricTrajectoryPlanner.solve(
                        planner, q, target, iterations=45)
                    if (ep > .0007 or er > .015 or ef > .001 or np.max(abs(next_q-q)) > .09
                            or not planner.edge_free(q, next_q)):
                        next_q, ep, er, ef = planner.solve(q, target, iterations=45)
                    if ep > .0007 or er > .015 or ef > .001:
                        failure = 'IK_CONTINUATION'; continue
                    if np.max(abs(next_q-q)) > .09:
                        failure = 'JOINT_CONTINUITY'; continue
                    if not planner.edge_free(q, next_q):
                        failure = 'COLLISION_OR_CLEARANCE'; continue
                    r = trial_r; feasible = True; break
                if not feasible:
                    break
                qs.append(next_q.copy()); wrists.append(wrist.pose_world.copy()); q = next_q
            # A safe prefix is useful even when the far end is blocked.
            n = len(qs)
            distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points[:n], axis=0), axis=1))]
            if distance[-1] >= .005:
                candidate = SurfaceStroke(np.asarray(qs), points[:n], normals[:n], np.asarray(wrists),
                    distance, finger, goal, r0.inv().apply(measured_normals[finger]))
                if distance[-1] >= .015:
                    self.heading = unit(points[n-1]-points[0])
                    return candidate, failures
                if hasattr(self, 'short_prefixes'):
                    self.short_prefixes.append(candidate)
            failures.append({'goal_vertex': goal, 'reason': failure or 'SHORT_PREFIX',
                             'feasible_prefix_m': float(distance[-1])})
        return None, failures


class GraceMCC(BasicPassiveMCC):
    """Retain measured normal and compliance for <=100 ms brief dropouts."""
    def __init__(self, h, dt):
        super().__init__(h, dt)
        self.dt = dt
        self.age = np.full(4, np.inf)
        self.normal_history = np.zeros((4, 3))

    def command(self, data, nominal_q, forces, normals, active, desired_force):
        self.age += self.dt
        self.age[active] = 0.
        self.normal_history[active] = normals[active]
        # This is bounded control continuation, not a fabricated sensor:
        # logs/coverage/safety always receive the original physical active.
        control_active = active | (self.age <= .1)
        return super().command(data, nominal_q, forces, self.normal_history,
                               control_active, desired_force)


def nearby_recontact_target(h, graph, data, vertex, finger, outward, roll=0.):
    """Keep current hand posture/roll while turning toward a nearby patch.

    This avoids gratuitously returning to a pose synthesized from the original
    acquisition frame each time a short surface stroke needs a reposition.
    """
    current = pose(h, data)
    r0 = rotation(current[3:])
    normal = graph.normals[vertex]
    r = align_vector(outward, normal)*r0
    if roll:
        from scipy.spatial.transform import Rotation
        r = Rotation.from_rotvec(normal*roll)*r
    geom = int(h.tip_geom_ids[finger])
    local_center = r0.inv().apply(data.geom_xpos[geom]-current[:3])
    local_frame = r0.inv().as_matrix()@data.geom_xmat[geom].reshape(3, 3)
    local_tips = r0.inv().apply(tip_positions(h, data)-current[:3])
    support = pad_support_vector(h.model, geom, r.as_matrix()@local_frame, normal)
    p = h.object_position_m+graph.vertices_m[vertex]+support+.002*normal-r.apply(local_center)
    return WholeHandTarget(WristTarget(np.r_[p, quaternion(r)]), p+r.apply(local_tips))


def arm_force_release(h, data, previous_command, forces, normals, speed=.030):
    """Integrate a measured-normal wrist withdrawal from the applied reference.

    Freezing the arm reference still presses into a contact made during an
    air transfer. A finger-only measured-q increment can stall a position servo.
    """
    loaded = forces >= 4.
    if not np.any(loaded):
        return previous_command[:7].copy()
    outward = unit(np.sum(normals[loaded]*forces[loaded, None], axis=0))
    return previous_command[:7] + endpoint_arm_offset(h, data, outward, -speed*.002)


def plan_surface_transition(sp, oracle, data, coverage, attempts, outward, witnesses=None):
    """Evaluate whole checked routes; prefer short posture-preserving bridges.

    Every synthetic state below belongs to planning scratch, never the plant.
    A geometric bridge can still lose contact: only the measured trace decides.
    """
    h, graph, planner, cell = sp.h, sp.graph, sp.planner, sp.cell
    current, root = pose(h, data), joint_state(h, data)
    forces, points, normals, active, _, _ = contact_measurement(h, data, .2)
    anchors = np.flatnonzero(active & (forces <= 8.))
    if not len(anchors):
        anchors = np.array([0])
    local_tips = tip_positions(h, data)-h.object_position_m
    required = graph.required_vertices[~coverage.visited_goals]
    remaining = required if len(required) else np.flatnonzero(~coverage.visited_vertices)
    if not len(remaining):
        return None
    _, nearest = graph._tree.query(local_tips[anchors])
    geodesic = np.min(dijkstra(graph.adjacency, directed=False, indices=np.atleast_1d(nearest)), axis=0)
    score = geodesic[remaining]+.08*attempts[remaining]
    cache = {} if witnesses is None else {int(v):i for i,v in enumerate(witnesses['vertex'])}
    # A slightly farther patch with a compatible hand orientation can avoid
    # a very long wrist flip to the geometrically nearest patch.
    for index, vertex in enumerate(remaining):
        if int(vertex) in cache:
            cached_pose = witnesses['pose'][cache[int(vertex)]]
            score[index] += (1.5*np.linalg.norm(cached_pose[:3]-current[:3]) +
                .06*(rotation(cached_pose[3:])*rotation(current[3:]).inv()).magnitude())
    ordered = remaining[np.argsort(score)]
    choices = []
    for vertex in ordered:
        if all(np.linalg.norm(graph.vertices_m[vertex]-graph.vertices_m[v]) > .012 for v in choices):
            choices.append(int(vertex))
        if len(choices) >= 16:
            break
    def expose(full):
        return full.wrist if cell == 'A-MCC' else full
    def scratch_at(q):
        scratch = mujoco.MjData(h.model)
        set_joint_state(h, scratch, q)
        return scratch
    # Try all local anchors before committing to a withdrawal.
    for vertex in choices[:4]:
        if geodesic[vertex] > .085:
            continue
        for finger in anchors:
            full = nearby_recontact_target(h, graph, data, vertex, int(finger), outward)
            if np.linalg.norm(full.wrist.pose_world[:3]-current[:3]) > .075:
                continue
            path, _ = planner.plan(data, expose(full))
            if path is not None and path.duration_s <= 12.:
                attempts[vertex] += 1
                n = (rotation(full.wrist.pose_world[3:])*rotation(current[3:]).inv()).apply(outward)
                return [('NEAR_TRANSFER', path)], n, dict(vertex=vertex,
                    pose_policy='CONTACT_NEAR_BRIDGE', duration_s=path.duration_s,
                    retreat_offset_m=None)
    retreats = []
    for distance in (.004, .008, .012, .025, .04):
        for direction in [outward]+[np.eye(3)[i]*sign for i in (2, 1, 0) for sign in (1., -1.)]:
            delta = direction*distance
            retreat = WholeHandTarget(WristTarget(np.r_[current[:3]+delta, current[3:]]),
                                     tip_positions(h, data)+delta)
            path, _ = planner.plan(data, expose(retreat))
            if path is not None:
                retreats.append((path, delta))
                break
    feasible = []
    for vertex in choices:
        variant = int(attempts[vertex]); attempts[vertex] += 1
        full = nearby_recontact_target(h, graph, data, vertex, int(anchors[0]), outward)
        solved, ep, er, ef = planner.solve(root, expose(full), iterations=65)
        policy = 'CURRENT_POSTURE_MINIMUM_NORMAL_ROTATION'
        current_feasible = ep <= .001 and er <= .02 and ef <= .002 and not planner.collision_reason(solved)
        if not current_feasible:
            for alternate_finger in np.roll(np.arange(4), -int(anchors[0])):
                for roll in (0., .35, -.35, .7, -.7):
                    trial = nearby_recontact_target(h, graph, data, vertex, int(alternate_finger), outward, roll)
                    trial_q, ep, er, ef = planner.solve(root, expose(trial), iterations=45)
                    if ep <= .001 and er <= .02 and ef <= .002 and not planner.collision_reason(trial_q):
                        full, solved = trial, trial_q
                        current_feasible = True
                        policy = 'CURRENT_POSTURE_LOCAL_ROLL_AND_FINGER'; break
                if policy == 'CURRENT_POSTURE_LOCAL_ROLL_AND_FINGER': break
        if not current_feasible:
            policy = 'GLOBAL_GEOMETRIC_POSE_FALLBACK'
            cached_safe = (vertex in cache and variant < 2 and 'q' in witnesses and
                           not planner.collision_reason(witnesses['q'][cache[vertex]]))
            if cached_safe:
                i = cache[vertex]
                full = WholeHandTarget(WristTarget(witnesses['pose'][i]), witnesses['tips'][i])
            else:
                witness, _ = oracle.find_pose(vertex, root, extended=variant >= 2, variant=variant)
                if witness is None:
                    # Residual fill can touch a neighboring anchor but never
                    # erases the originally uncovered vertex from the ledger.
                    nearby = dijkstra(graph.adjacency, directed=False, indices=vertex, limit=.020)
                    neighbors = np.flatnonzero(np.isfinite(nearby))
                    for neighbor in neighbors[::max(1, len(neighbors)//5)][:5]:
                        witness, _ = oracle.find_pose(int(neighbor), root, extended=True, variant=variant)
                        if witness is not None: break
                if witness is None: continue
                full = WholeHandTarget(WristTarget(witness['pose']), witness['tips'])
        n = ((rotation(full.wrist.pose_world[3:])*oracle.initial_rotation.inv()).apply(oracle.initial_normal)
             if policy.startswith('GLOBAL') else
             (rotation(full.wrist.pose_world[3:])*rotation(current[3:]).inv()).apply(outward))
        base = expose(full)
        # Retain both the shorter and the slightly larger escape until the
        # complete transfer is checked. A locally feasible lift may be trapped.
        options = retreats or [(None, None)]
        for retreat, delta in options:
            scratch = data if retreat is None else scratch_at(retreat.waypoints[-1])
            transfer, _ = planner.plan(scratch, shifted_target(base, n*.004), allow_rrt=True)
            approach = None
            if transfer is not None:
                approach, _ = planner.plan(scratch_at(transfer.waypoints[-1]), base)
            if transfer is None or approach is None:
                transfer, _ = planner.plan(scratch, base, allow_rrt=True)
                approach = None
            if transfer is None: continue
            segments = ([] if retreat is None else [('LIFT', retreat)]) + [('TRANSFER', transfer)]
            if approach is not None: segments.append(('APPROACH', approach))
            duration = sum(p.duration_s for _,p in segments)
            feasible.append((duration, segments, n, dict(vertex=vertex, duration_s=duration,
                pose_policy=policy, retreat_offset_m=None if delta is None else delta.tolist())))
            if duration <= 25.:
                break
        if len(feasible) >= 2 and min(item[0] for item in feasible) <= 25.:
            break
    if not feasible: return None
    _, segments, n, meta = min(feasible, key=lambda item:item[0])
    return segments, n, meta


def reanchor_transition(planner, data, path, cell, allow_rrt=False):
    """Recheck the remaining segment from the measured state at its boundary."""
    h = planner.h
    scratch = mujoco.MjData(h.model)
    set_joint_state(h, scratch, path.waypoints[-1])
    wrist = WristTarget(pose(h, scratch))
    target = wrist if cell == 'A-MCC' else WholeHandTarget(wrist, tip_positions(h, scratch))
    return planner.plan(data, target, allow_rrt=allow_rrt)


def tracking_progress_scale(measured, applied, nominal):
    """Do not wind the arm reference up against a stuck physical contact."""
    lag = max(float(np.max(abs(np.asarray(measured)[:7]-np.asarray(applied)[:7]))),
              float(np.max(abs(np.asarray(nominal)[:7]-np.asarray(applied)[:7]))))
    return float(np.clip((.010-lag)/.006, 0., 1.)), lag


class LocalProgressBudget:
    """Bound unsuccessful local retries using measured coverage only."""
    def __init__(self, maximum_empty_strokes=6, maximum_local_s=45.):
        self.maximum_empty_strokes = maximum_empty_strokes
        self.maximum_local_s = maximum_local_s
        self.covered_count = 0
        self.empty_strokes = 0
        self.local_s = 0.
        self.open_stroke = None

    def observe(self, covered_count, dt_s, *, local):
        if covered_count > self.covered_count:
            self.empty_strokes = 0
            self.local_s = 0.
        elif local:
            self.local_s += dt_s
        self.covered_count = covered_count

    def begin(self, stroke_id, covered_count):
        if self.open_stroke is not None:
            raise RuntimeError('Previous local stroke was not closed')
        self.open_stroke = (stroke_id, covered_count)

    def finish(self, covered_count, reason):
        if self.open_stroke is None:
            return None
        stroke_id, initial_count = self.open_stroke
        self.open_stroke = None
        gain = int(covered_count-initial_count)
        self.empty_strokes = self.empty_strokes+1 if gain <= 0 else 0
        return dict(stroke_id=stroke_id, reason=reason, new_vertices=gain,
                    empty_strokes=self.empty_strokes, local_without_gain_s=self.local_s)

    @property
    def exhausted(self):
        return self.empty_strokes >= self.maximum_empty_strokes or self.local_s >= self.maximum_local_s

    def transition_committed(self):
        # A checked route to a different unmeasured target gets a fresh local
        # budget. Merely rebasing fingers or selecting another stroke does not.
        self.empty_strokes = 0
        self.local_s = 0.


def run(directory, *, cell='B-MCC', seed=7, budget_s=120., speed=.006, expected_runtime_sha256=None,
        witness_path=None):
    if not np.isfinite(budget_s) or budget_s <= 0 or not 0 < speed <= .012:
        raise ValueError('Positive duration and surface speed in (0, 0.012] m/s required')
    if cell != 'B-MCC':
        raise ValueError('Active surface-target finger planning is B-MCC only; old A-MCC remains frozen')
    source_root = Path(__file__).resolve().parents[2]
    sources = ['Module/i04_geometric_traversal/'+name for name in
        ('surface_slide.py', 'full_surface.py', 'physics.py', 'contracts.py', 'oracle.py',
         'planning.py', 'recovery.py', 'fixtures.py', 'sliding_metrics.py', 'active_fingers.py', 'sliding_safety.py')]
    sources += ['Module/fr3_leap/model.py', 'Module/module_2_fingertip_mcc/controller.py']
    source_bytes = {name: (source_root/name).read_bytes() for name in sources}
    hashes = {name: sha256(content).hexdigest() for name, content in source_bytes.items()}
    if (expected_runtime_sha256 is not None and
            hashes['Module/i04_geometric_traversal/surface_slide.py'] != expected_runtime_sha256):
        raise RuntimeError('Controller changed before experiment startup')
    directory = Path(directory)
    if (directory/'manifest.json').exists():
        raise FileExistsError('Use a fresh output directory')
    directory.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    h, graph, q, quality = build_scene(directory)
    data, initial = acquire(h, graph, q, seed)
    config = ExperimentConfig(seed=seed, duration_s=budget_s+5., wrist_speed_m_s=.025,
                              wrist_rotation_speed_rad_s=.35, trajectory_joint_speed_rad_s=.8)
    sp = SurfaceStrokePlanner(h, graph, config, cell)
    planner = sp.planner
    oracle = FullSurfaceOracle(h, graph, data)
    oracle.planner = SlidingSafetyPlanner(h, config)
    coverage = MeasuredCoverage(graph, config)
    mcc = GraceMCC(h, .002)
    relief = SharedContactRecovery(config)
    fingers = ActiveFingerSearch(h, graph, dt=.002)
    if witness_path is None:
        witness_path = source_root/'Module/generated/mustard_true_ellipsoid_geometry07/pose_witnesses.npz'
    else:
        witness_path = Path(witness_path).resolve()
        if not witness_path.is_file():
            raise FileNotFoundError(f'Explicit witness cache does not exist: {witness_path}')
    witness_bytes = witness_path.read_bytes() if witness_path.exists() else None
    witnesses = None
    if witness_bytes is not None:
        with np.load(BytesIO(witness_bytes)) as saved_witnesses:
            witnesses = {key: saved_witnesses[key].copy() for key in saved_witnesses.files}
        (directory/'geometric_pose_witnesses.npz').write_bytes(witness_bytes)
    np.savez_compressed(directory/'initial.npz', qpos=data.qpos, qvel=data.qvel, ctrl=data.ctrl)
    (directory/'model.xml').write_text(h.xml, encoding='utf-8')
    np.savez_compressed(directory/'surface_graph.npz', vertices=graph.vertices_m, faces=graph.faces,
                        normals=graph.normals, required_vertices=graph.required_vertices)
    for name, content in source_bytes.items():
        dest = directory/'source_snapshot'/name
        dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(content)
    manifest = dict(protocol='i04-mustard-active-finger-sliding.development.v2.2', cell=cell, seed=seed,
        object_position_m=h.object_position_m.tolist(), budget_s=budget_s,
        speed_m_s=speed, source_hashes=hashes, initial_state=initial, geometry_quality=quality,
        dp='DEFERRED_BY_USER', full_surface_claim=False,
        active_fingers=True, completion_policy='All original vertices and required goals measured, then 160ms safe hold',
        active_finger_search=asdict(fingers.config),
        witness_path=str(witness_path.resolve()) if witnesses is not None else None,
        witness_snapshot='geometric_pose_witnesses.npz' if witness_bytes is not None else None,
        witness_sha256=sha256(witness_bytes).hexdigest() if witness_bytes is not None else None,
        plan=dict(surface_spacing_m=.002, minimum_stroke_m=.015, fallback_stroke_m=.005, preferred_stroke_m=.11,
                  local_no_gain_stroke_limit=6, local_no_gain_time_limit_s=45.,
                  contact_loss_grace_s=.1, tangent_pause_loss_s=.06, recovery_timeout_s=2.,
                  recovery_bound_m=.006, geometry_lift_clearance_m=[.004,.008,.012,.025,.04],
                  transfer_speed_m_s=.025, transfer_rotation_rad_s=.35),
        tracking=dict(full_speed_error_rad=.004, pause_error_rad=.010,
                      replan_after_pause_s=.30),
        arm_self_collision=dict(planning_clearance_m=.001, physical_stop_force_n=.5,
                                excludes='same body, directly adjacent or same welded body'),
        coverage=dict(radius_m=.025, residual_limit_m=.005, normal_cosine=.5, force_threshold_n=.2))
    if witness_bytes is None:
        manifest.pop('witness_snapshot')
    (directory/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    logs, events = {}, []
    last = np.r_[data.ctrl[h.arm_actuator_ids], data.ctrl[h.hand_actuator_ids]]
    nominal = joint_state(h, data)
    phase, stroke, stroke_id, s, offset = 'SELECT', None, -1, 0., 0.
    normal = oracle.initial_normal.copy()
    normal_in_wrist = rotation(pose(h, data)[3:]).inv().apply(normal)
    loss_start = None
    stable_since = None
    failed_starts = 0
    transfer = None
    transfer_elapsed = 0.
    transfer_queue = []
    lift_attempts = np.zeros(len(graph.vertices_m))
    high_run = 0
    peak = 0.
    peak_arm_self_force = 0.
    stop = 'TIME_BUDGET_REACHED'
    phase_started = 0.
    local_progress = LocalProgressBudget()
    pending_stroke = None
    full_since = None
    completion_nominal = None
    reacquire_nominal = None
    tracking_pause_since = None
    checkpoint_index = chunk_number = 0
    def checkpoint(t):
        nonlocal checkpoint_index, chunk_number
        chunks = directory/'trace_chunks'; chunks.mkdir(exist_ok=True)
        arrays = {key: np.asarray(value[checkpoint_index:]) for key,value in logs.items()}
        if len(arrays.get('time_s', [])):
            np.savez_compressed(chunks/f'{chunk_number:04d}.npz', **arrays)
            chunk_number += 1; checkpoint_index = len(logs['time_s'])
        np.savez_compressed(directory/'coverage_checkpoint.tmp.npz',
            covered_vertices=coverage.visited_vertices, required_goal_visited=coverage.visited_goals,
            qpos=data.qpos, qvel=data.qvel, ctrl=data.ctrl, time=data.time, elapsed=t)
        (directory/'coverage_checkpoint.tmp.npz').replace(directory/'coverage_checkpoint.npz')
    for step in range(int(budget_s/.002)+1):
        t = step*.002
        observed_phase = phase
        forces, points, normals, active, unsafe, penetration = contact_measurement(h, data, .2)
        self_contacts = arm_self_contacts(h, data, force_threshold_n=0.)
        self_force = max([0.] + [item['force_n'] for item in self_contacts])
        peak_arm_self_force = max(peak_arm_self_force, self_force)
        safe = active & (forces <= 8.)
        if forces.max() > max(8., peak):
            events.append(dict(time_s=t, kind='FORCE_PEAK_STATE', phase=phase,
                forces_n=forces.tolist(), q_rad=joint_state(h, data).tolist(),
                qvel_rad_s=np.r_[data.qvel[h.arm_dof_adrs], data.qvel[h.hand_dof_adrs]].tolist(),
                sensor_command_rad=last.tolist(), qacc_warmstart=data.qacc_warmstart.tolist()))
        peak = max(peak, float(forces.max()))
        high_run = high_run+1 if forces.max() > 12. else 0
        hard = ('SHALLOW_ARM_SELF_COLLISION' if self_force > .5 else
                'NON_TIP_OBJECT_COLLISION' if unsafe else 'SUSTAINED_DANGEROUS_FORCE'
                if high_run*.002 >= .02 else 'NONFINITE_STATE' if not np.all(np.isfinite(data.qpos))
                else 'JOINT_LIMIT' if np.any(joint_state(h, data) < planner.lower-.02)
                or np.any(joint_state(h, data) > planner.upper+.02) else None)
        if hard:
            stop = hard
            events.append(dict(time_s=t, kind='HARDWARE_STOP', reason=hard, non_tip=unsafe,
                arm_self_contacts=self_contacts, forces_n=forces.tolist(),
                q_rad=joint_state(h, data).tolist(),
                qvel_rad_s=np.r_[data.qvel[h.arm_dof_adrs], data.qvel[h.hand_dof_adrs]].tolist(),
                sensor_command_rad=last.tolist(), qacc_warmstart=data.qacc_warmstart.tolist()))
            break
        if step % 5 == 0:
            coverage.update(points-h.object_position_m, normals, active)
        if complete_surface(coverage):
            if completion_nominal is None:
                completion_nominal = joint_state(h, data)
                mcc = GraceMCC(h, .002)
                fingers.rebase(completion_nominal)
                events.append(dict(time_s=t, kind='ALL_SURFACE_MEASURED_PENDING_STABILITY'))
            phase = 'COMPLETION_HOLD'
            nominal = completion_nominal.copy()
            if np.any(safe) and forces.max() <= 8.:
                full_since = t if full_since is None else full_since
            else:
                full_since = None
        measured_count = int(coverage.visited_vertices.sum())
        local_progress.observe(measured_count, .002, local=phase in ('SLIDE', 'SELECT', 'LOCAL_SEARCH'))
        if local_progress.exhausted and phase in ('SLIDE', 'SELECT', 'LOCAL_SEARCH'):
            ended = local_progress.finish(measured_count, 'LOCAL_PROGRESS_BUDGET')
            events.append(dict(time_s=t, kind='LOCAL_PROGRESS_BUDGET_EXHAUSTED',
                ended_stroke=ended, empty_strokes=local_progress.empty_strokes,
                local_without_gain_s=local_progress.local_s))
            phase = 'LIFT_PLAN'  # The transition planner still tries contact-near bridges first.
            pending_stroke = None
            nominal = joint_state(h, data)
            fingers.rebase(nominal); mcc = GraceMCC(h, .002)
        # Run this before SELECT so a found stroke is installed from the same
        # measured state, without a second mutating planner call or mj_step.
        if phase == 'LOCAL_SEARCH' and t-phase_started >= 2.:
            candidate, failures = sp.plan(data, coverage, forces, points, normals, safe)
            if candidate is not None:
                pending_stroke = (candidate, failures)
                phase = 'SELECT'
                events.append(dict(time_s=t, kind='LOCAL_SEARCH_SUPPORT_FOUND'))
            else:
                phase = 'LIFT_PLAN'
                events.append(dict(time_s=t, kind='LIFT_REQUIRED', reasons=failures))
        if phase == 'SELECT':
            if pending_stroke is not None:
                stroke, failures = pending_stroke
                pending_stroke = None
            else:
                stroke, failures = sp.plan(data, coverage, forces, points, normals, safe)
            if stroke is not None:
                stroke_id += 1; s = 0.; offset = 0.; loss_start = None
                tracking_pause_since = None
                phase = 'SLIDE'; phase_started = t
                local_progress.begin(stroke_id, measured_count)
                nominal = stroke.q[0].copy()
                fingers.rebase(nominal)
                mcc = GraceMCC(h, .002)
                events.append(dict(time_s=t, kind='SURFACE_STROKE', stroke_id=stroke_id,
                    length_m=stroke.length, points=len(stroke.q), start=stroke.positions[0].tolist(),
                    end=stroke.positions[-1].tolist(), finger=int(stroke.finger), failures=failures,
                    exposed_fields=['wrist_pose'] if cell == 'A-MCC' else ['wrist_pose', 'tip_positions']))
                np.savez_compressed(directory/f'stroke_{stroke_id:04d}.npz', q=stroke.q,
                    wrist=stroke.wrist, surface_positions_local=stroke.positions,
                    normals=stroke.normals, distance=stroke.distances)
            else:
                phase = 'LOCAL_SEARCH'; phase_started = t
                nominal = joint_state(h, data)
                fingers.rebase(nominal)
                mcc = GraceMCC(h, .002)
                events.append(dict(time_s=t, kind='LOCAL_SEARCH_BEGIN', reasons=failures))
        if phase == 'LIFT_PLAN':
            outward = unit(normals[safe].mean(axis=0)) if np.any(safe) else normal
            result = plan_surface_transition(sp, oracle, data, coverage, lift_attempts, outward, witnesses)
            if result is None:
                failed_starts += 1
                if failed_starts >= 4:
                    stop = 'NO_SAFE_REPOSITION_AFTER_EXPANDED_SEARCH'; break
                events.append(dict(time_s=t, kind='REPOSITION_SEARCH_EXPANDED', attempt=failed_starts))
                phase = 'RETRY_REPOSITION'; phase_started = t
                nominal = joint_state(h, data)
                mcc = GraceMCC(h, .002); fingers.rebase(nominal)
            else:
                failed_starts = 0
                local_progress.transition_committed()
                segments, normal, transition_meta = result
                phase, transfer = segments[0]
                transfer_queue = segments[1:]
                transfer_elapsed = 0.; offset = 0.; phase_started = t
                mcc = GraceMCC(h, .002)
                fingers.rebase(joint_state(h, data))
                events.append(dict(time_s=t, kind='NEAR_BRIDGE_COMMIT' if phase == 'NEAR_TRANSFER' else 'LIFT_COMMIT',
                    **transition_meta, exposed_fields=['wrist_pose', 'tip_positions']))
        was_in_relief = relief.mode == 'FORCE_RELEASE'
        in_relief = forces.max() >= 8 or (was_in_relief and forces.max() >= 4)
        relief.mode = 'FORCE_RELEASE' if in_relief else 'NORMAL'
        if in_relief and not was_in_relief:
            events.append(dict(time_s=t, kind='FORCE_RELEASE_BEGIN', phase=phase,
                               forces_n=forces.tolist()))
            mcc = GraceMCC(h, .002)
        elif was_in_relief and not in_relief:
            events.append(dict(time_s=t, kind='FORCE_RELEASE_REPLAN', phase=phase))
            # The old path caused the overload. Re-root at the measured
            # released state rather than returning to the same compression.
            phase = 'SELECT' if np.any(safe) else 'LIFT_PLAN'
            nominal = joint_state(h, data)
            fingers.rebase(nominal)
            mcc = GraceMCC(h, .002)
            if complete_surface(coverage):
                completion_nominal = nominal.copy()
            offset = 0.
        if phase == 'SLIDE':
            finger = stroke.finger
            good = bool(safe[finger])
            if good:
                if loss_start is not None:
                    events.append(dict(time_s=t, kind='CONTACT_RECOVERED', duration_s=t-loss_start,
                                       stroke_id=stroke_id, same_arclength_resume_m=s))
                loss_start = None
            elif loss_start is None:
                loss_start = t
                events.append(dict(time_s=t, kind='CONTACT_GAP', stroke_id=stroke_id, arclength_m=s))
            previous_nominal, _ = stroke.sample(s)
            progress_scale, tracking_lag = tracking_progress_scale(joint_state(h, data), last, previous_nominal)
            stalled = progress_scale < .02
            if stalled and tracking_pause_since is None:
                tracking_pause_since = t
                events.append(dict(time_s=t, kind='ARM_TRACKING_PAUSE', error_rad=tracking_lag,
                                   forces_n=forces.tolist(), stroke_id=stroke_id))
            elif not stalled:
                tracking_pause_since = None
            advance = not in_relief and forces.max() < 4. and (good or t-loss_start < .06)
            if advance:
                # Constant arc-speed with a short starting ramp, no stop at
                # every mesh point. Bound the corresponding joint velocity.
                a, _ = stroke.sample(s)
                b, _ = stroke.sample(min(stroke.length, s+speed*.002))
                ratio = min(1., .8*.002/max(float(np.max(abs(b-a))), 1e-12))
                s = min(stroke.length, s+speed*.002*ratio*progress_scale*min(1., (t-phase_started+.002)/.3))
            nominal, normal = stroke.sample(s)
            # Slow common normal trim handles geometry mismatch; fingertip
            # MCC handles faster local force error. No tangential search.
            velocity = float(np.clip(.0005*(2.-forces[finger]), -.002, .0015))
            velocity = min(velocity, float(np.clip(.0005*(3.-forces.max()), -.002, .0015)))
            offset = float(np.clip(offset+velocity*.002, -.004, .006))
            nominal[:7] += endpoint_arm_offset(h, data, normal, offset)
            if tracking_pause_since is not None and t-tracking_pause_since >= .30:
                events.append(dict(time_s=t, kind='ARM_TRACKING_REPLAN', error_rad=tracking_lag,
                                   stroke_id=stroke_id))
                phase = 'SELECT'; nominal = joint_state(h, data)
                fingers.rebase(nominal); mcc = GraceMCC(h, .002)
                offset = 0.; tracking_pause_since = None
            elif loss_start is not None and t-loss_start >= 2.:
                events.append(dict(time_s=t, kind='LOCAL_RECOVERY_TIMEOUT', stroke_id=stroke_id))
                # Another finger may already support a continuation.
                phase = 'SELECT' if np.any(safe) else 'LOCAL_SEARCH'
                phase_started = t
                nominal = joint_state(h, data)
                fingers.rebase(nominal)
                mcc = GraceMCC(h, .002)
            elif s >= stroke.length:
                events.append(dict(time_s=t, kind='STROKE_ENDPOINT', stroke_id=stroke_id, safe_contact=good))
                phase = 'SELECT'
        elif phase == 'RETRY_REPOSITION':
            if t-phase_started >= .2:
                phase = 'SELECT' if np.any(safe) else 'LIFT_PLAN'
        elif phase in ('LIFT', 'TRANSFER', 'APPROACH', 'NEAR_TRANSFER'):
            if not in_relief:
                transfer_elapsed += .002
            nominal = transfer.sample(transfer_elapsed)
            if transfer_elapsed >= transfer.duration_s:
                if transfer_queue:
                    phase, transfer = transfer_queue.pop(0)
                    transfer_elapsed = 0.
                    measured = joint_state(h, data)
                    if (np.max(abs(measured-transfer.root_q)) > .02 or
                            not planner.edge_free(measured, transfer.root_q)):
                        revised, failure = reanchor_transition(planner, data, transfer, cell,
                                                              allow_rrt=phase == 'TRANSFER')
                        events.append(dict(time_s=t, kind='TRANSITION_REANCHOR', phase=phase,
                                           success=revised is not None, failure=failure))
                        if revised is None:
                            phase = 'SELECT' if np.any(safe) else 'LIFT_PLAN'
                            nominal = measured
                            fingers.rebase(nominal)
                            mcc = GraceMCC(h, .002)
                        else:
                            transfer = revised
                else:
                    phase = 'REACQUIRE'; phase_started = t; stable_since = None; offset = 0.
                    nominal = transfer.waypoints[-1].copy()
                    nominal[7:] = joint_state(h, data)[7:]
                    reacquire_nominal = nominal.copy()
                    fingers.rebase(nominal)
                    mcc = GraceMCC(h, .002)
        elif phase == 'REACQUIRE':
            nominal = reacquire_nominal.copy()
            if np.any(safe):
                stable_since = t if stable_since is None else stable_since
            else:
                stable_since = None
                if not in_relief:
                    offset = min(.006, offset+.0015*.002)
            nominal[:7] += endpoint_arm_offset(h, data, normal, offset)
            if stable_since is not None and t-stable_since >= .15:
                events.append(dict(time_s=t, kind='LIFT_RECONTACT', approach_offset_m=offset))
                phase = 'SELECT'
            elif t-phase_started > 5.:
                events.append(dict(time_s=t, kind='REACQUIRE_TIMEOUT'))
                phase = 'LIFT_PLAN'
        # Intentional withdrawal must not fight an inward force controller.
        search_enabled = phase in ('SLIDE', 'LOCAL_SEARCH', 'NEAR_TRANSFER', 'APPROACH', 'REACQUIRE') and not in_relief
        finger_nominal, finger_diag = fingers.update(data, nominal, forces, normals, active, enabled=search_enabled)
        command = (finger_nominal.copy() if phase in ('LIFT', 'TRANSFER', 'LIFT_PLAN') else
                   mcc.command(data, finger_nominal, forces, normals, active, 2.))
        if in_relief:
            command = relief.relieve_force(h, data, command, forces, normals)
            command[:7] = arm_force_release(h, data, last, forces, normals)
        command = np.clip(command, last-.0024, last+.0024)
        command = np.clip(command, planner.lower, planner.upper)
        if step % 5 == 0:
            reason = planner.command_collision_reason(joint_state(h, data), command)
            if reason in ('NON_TIP_CLEARANCE', 'NON_TIP_OBJECT_COLLISION', 'SELF_COLLISION',
                          'ARM_SELF_CLEARANCE', 'SHALLOW_ARM_SELF_COLLISION'):
                events.append(dict(time_s=t, kind='COMMAND_GEOMETRY_DEFER', reason=reason, phase=phase))
                command = np.clip(joint_state(h, data), last-.0024, last+.0024)
                command = np.clip(command, planner.lower, planner.upper)
                phase = ('SELECT' if np.any(safe) else 'LIFT_PLAN') if reason in (
                    'ARM_SELF_CLEARANCE', 'SHALLOW_ARM_SELF_COLLISION') else 'LIFT_PLAN'
                nominal = joint_state(h, data)
                fingers.rebase(nominal); mcc = GraceMCC(h, .002)
            values = dict(time_s=t, q_rad=joint_state(h, data),
                qvel_rad_s=np.r_[data.qvel[h.arm_dof_adrs], data.qvel[h.hand_dof_adrs]],
                command_rad=command.copy(), sensor_command_rad=last.copy(),
                palm_pose_world=pose(h, data), tip_positions_world_m=tip_positions(h, data),
                contact_positions_world_m=points.copy(), normals_world=normals.copy(), forces_n=forces.copy(),
                active=active.copy(), coverage=coverage.area_fraction,
                covered_goals=int(coverage.visited_goals.sum()), goal_id=-1,
                phase='FORCE_RELIEF' if in_relief else observed_phase,
                command_phase=phase, endpoint_offset_m=offset,
                stroke_id=stroke_id, stroke_distance_m=s)
            values['arm_tracking_error_rad'] = float(np.max(abs(joint_state(h, data)[:7]-last[:7])))
            values['arm_self_force_n'] = self_force
            values['finger_search_active'] = np.asarray(finger_diag['seeking'], dtype=bool)
            values['finger_search_enabled'] = bool(search_enabled)
            values['finger_search_bias_rad'] = np.asarray(finger_diag['joint_bias_rad']).copy()
            values['finger_search_distance_m'] = np.asarray(finger_diag['search_distance_m']).copy()
            values['finger_search_delta_rad'] = np.asarray(finger_diag['search_delta_rad']).copy()
            values['finger_search_target_positions_world_m'] = np.asarray(finger_diag['target_positions_world_m']).copy()
            values['finger_search_target_normals_world'] = np.asarray(finger_diag['target_normals_world']).copy()
            values['finger_search_target_valid'] = np.asarray(finger_diag['target_valid'], dtype=bool)
            values['finger_search_state'] = np.asarray(finger_diag['state'])
            values['finger_search_loss_age_s'] = np.asarray(finger_diag['loss_age_s']).copy()
            values['finger_search_measured_active'] = active.copy()
            values['nominal_path_distance_m'] = s
            values['qacc_warmstart'] = data.qacc_warmstart.copy()
            for key, value in values.items():
                logs.setdefault(key, []).append(value)
        if local_progress.open_stroke is not None and phase != 'SLIDE':
            ended = local_progress.finish(measured_count, phase)
            events.append(dict(time_s=t, kind='LOCAL_STROKE_RESULT', **ended))
        last = command.copy(); set_control(h, data, command)
        if step % 5000 == 0:
            progress = dict(time_s=t, phase=phase, stroke_id=stroke_id, coverage=coverage.area_fraction,
                            forces_n=forces.tolist(), peak_force_n=peak, wall_s=perf_counter()-started)
            print('SLIDE', json.dumps(progress), flush=True)
            (directory/'progress.json').write_text(json.dumps(progress, indent=2), encoding='utf-8')
            (directory/'events.json').write_text(json.dumps(events, indent=2), encoding='utf-8')
        if step % 30000 == 0:
            checkpoint(t)
        if full_since is not None and t-full_since >= .16 and step % 5 == 0:
            stop = 'FULL_MEASURED_SURFACE'; break
        if step % 500 == 0 and (directory/'stop_requested.json').exists():
            stop = 'DEVELOPMENT_STOP_REQUESTED'
            events.append(dict(time_s=t, kind='DEVELOPMENT_STOP_REQUESTED'))
            break
        if step == int(budget_s/.002):
            break
        mujoco.mj_step(h.model, data); mujoco.mj_forward(h.model, data)
    arrays = {key: np.asarray(value) for key, value in logs.items()}
    arrays['covered_vertices'] = coverage.visited_vertices
    arrays['required_goal_visited'] = coverage.visited_goals
    np.savez_compressed(directory/'trace.npz', **arrays)
    np.savez_compressed(directory/'final_state.npz', qpos=data.qpos, qvel=data.qvel, ctrl=data.ctrl, time=data.time)
    (directory/'events.json').write_text(json.dumps(events, indent=2), encoding='utf-8')
    summary = dict(cell=cell, seed=seed, stop_reason=stop, executed_s=t, peak_force_n=peak,
        peak_arm_self_force_n=peak_arm_self_force,
        wall_s=perf_counter()-started, final_covered_area_fraction=coverage.area_fraction,
        covered_vertex_count=int(coverage.visited_vertices.sum()), vertex_count=len(graph.vertices_m),
        visited_required_goals=int(coverage.visited_goals.sum()), required_goal_count=graph.required_goal_count,
        full_surface_complete=complete_surface(coverage) and stop == 'FULL_MEASURED_SURFACE',
        completion_stability_s=.15, all_surface_measured=complete_surface(coverage), stroke_count=stroke_id+1,
        lift_count=sum(e['kind'] == 'LIFT_COMMIT' for e in events),
        trace_sha256=sha256((directory/'trace.npz').read_bytes()).hexdigest())
    summary['sliding_metrics'] = analyze_trace(arrays)
    (directory/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('SLIDE_RESULT', json.dumps({k:v for k,v in summary.items() if k != 'sliding_metrics'}), flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cell', choices=['A-MCC', 'B-MCC'], default='B-MCC')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--budget', type=float, default=120.)
    parser.add_argument('--speed', type=float, default=.006)
    parser.add_argument('--expected-runtime-sha256')
    parser.add_argument('--witness-cache', type=Path)
    args = parser.parse_args()
    run(args.output, cell=args.cell, seed=args.seed, budget_s=args.budget, speed=args.speed,
        expected_runtime_sha256=args.expected_runtime_sha256, witness_path=args.witness_cache)
