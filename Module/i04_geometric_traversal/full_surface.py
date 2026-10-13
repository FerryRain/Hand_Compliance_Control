"""Whole-object geometry, reachability, and physical coverage traversal.

The original local 60 s evaluation is retained unchanged. This path first
audits the entire exterior, then executes and measures a whole-object sweep.
"""
from __future__ import annotations

import argparse
from dataclasses import replace, dataclass, asdict
from hashlib import sha256
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from time import perf_counter
import subprocess

import mujoco
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation, Slerp
from scipy.optimize import least_squares
from scipy.sparse.csgraph import dijkstra

from Module.fr3_leap import ARM_HOME_Q, HAND_NATURAL_Q
from Module.fr3_leap.model import compile_model_xml
from Module.e05_physics.scene import FINGERS
from .contracts import ExperimentConfig, WristTarget, WholeHandTarget, array_digest
from .fixtures import load_fixture, SurfaceGraph, ObjectPlant
from .oracle import align_vector, MeasuredCoverage
from .physics import (pose, rotation, quaternion, joint_state, set_joint_state,
                      set_control, tip_positions, position_jacobian,
                      contact_measurement, BasicPassiveMCC)
from .planning import SharedGeometricTrajectoryPlanner
from .recovery import SharedContactRecovery, ContactEpisodes


@dataclass
class JointPath:
    waypoints: np.ndarray
    segment_durations: np.ndarray
    root_pose: np.ndarray
    goal_pose: np.ndarray
    latency_s: float

    @property
    def root_q(self): return self.waypoints[0]
    @property
    def duration_s(self): return float(self.segment_durations.sum())
    @property
    def digest(self): return array_digest(self.waypoints)
    def sample(self, elapsed):
        ends = np.cumsum(self.segment_durations)
        i = min(int(np.searchsorted(ends, elapsed, side='right')), len(ends)-1)
        before = ends[i-1] if i else 0.
        u = np.clip((elapsed-before)/self.segment_durations[i], 0., 1.)
        s = u**3*(10-15*u+6*u*u)
        return self.waypoints[i] + s*(self.waypoints[i+1]-self.waypoints[i])


class FullSurfacePlanner(SharedGeometricTrajectoryPlanner):
    """Bounded geometric IK avoids the clipped unconstrained IK's dead ends."""
    def solve(self, q_root, target, *, iterations=100):
        wrist = target.wrist if isinstance(target, WholeHandTarget) else target
        desired_rotation = rotation(wrist.pose_world[3:])
        fixed = np.clip(np.asarray(q_root).copy(), self.lower, self.upper)
        cached = None
        def update_arm(arm):
            nonlocal cached
            if cached is None or not np.array_equal(cached, arm):
                q = fixed.copy(); q[:7] = arm
                set_joint_state(self.h, self.scratch, q)
                cached = arm.copy()
        def arm_error(arm):
            update_arm(arm)
            current = pose(self.h, self.scratch)
            return np.r_[current[:3]-wrist.pose_world[:3],
                .10*(rotation(current[3:])*desired_rotation.inv()).as_rotvec()]
        def arm_jacobian(arm):
            update_arm(arm)
            current = pose(self.h, self.scratch)
            e = (rotation(current[3:])*desired_rotation.inv()).as_rotvec()
            angle = np.linalg.norm(e)
            skew = np.array([[0., -e[2], e[1]], [e[2], 0., -e[0]], [-e[1], e[0], 0.]])
            coefficient = 1/12 if angle < 1e-4 else (1 - .5*angle/np.tan(.5*angle))/angle**2
            left_inverse = np.eye(3) - .5*skew + coefficient*skew@skew
            jp, jr = position_jacobian(self.h, self.scratch, self.h.palm_site_id)
            return np.vstack((jp[:, :7], .10*left_inverse@jr[:, :7]))
        best = least_squares(arm_error, fixed[:7], bounds=(self.lower[:7], self.upper[:7]),
            jac=arm_jacobian, max_nfev=iterations, ftol=1e-9, xtol=1e-9, gtol=1e-9)
        q = fixed.copy(); q[:7] = best.x
        if isinstance(target, WholeHandTarget):
            for i in range(4):
                columns = np.array([int(n) for n in FINGERS[i].joint_names]) + 7
                def finger_error(finger_q):
                    trial = q.copy(); trial[columns] = finger_q
                    set_joint_state(self.h, self.scratch, trial)
                    return self.scratch.site_xpos[self.h.tip_site_ids[i]] - target.fingertip_positions_world_m[i]
                def finger_jacobian(finger_q):
                    trial = q.copy(); trial[columns] = finger_q
                    set_joint_state(self.h, self.scratch, trial)
                    jp, _ = position_jacobian(self.h, self.scratch, self.h.tip_site_ids[i])
                    return jp[:, columns]
                finger = least_squares(finger_error, q[columns],
                    jac=finger_jacobian, bounds=(self.lower[columns], self.upper[columns]), max_nfev=40,
                    ftol=1e-9, xtol=1e-9, gtol=1e-9)
                q[columns] = finger.x
        set_joint_state(self.h, self.scratch, q)
        current = pose(self.h, self.scratch)
        ep = float(np.linalg.norm(current[:3]-wrist.pose_world[:3]))
        er = float((rotation(current[3:])*desired_rotation.inv()).magnitude())
        ef = (float(np.max(np.linalg.norm(tip_positions(self.h, self.scratch)
               - target.fingertip_positions_world_m, axis=1))) if isinstance(target, WholeHandTarget) else 0.)
        return q, ep, er, ef

    def collision_reason(self, q):
        # Planning gets 3 mm of non-pad clearance for servo tracking error.
        # Only the scratch contact query uses an expanded margin; the physical
        # model's contact margins are restored before any mj_step.
        margins = self.h.model.geom_margin.copy()
        object_ids = np.asarray(getattr(self.h, 'object_geom_ids', [self.h.object_geom_id]))
        self.h.model.geom_margin[object_ids] += .004
        try:
            reason = super().collision_reason(q)
        finally:
            self.h.model.geom_margin[:] = margins
        if reason:
            return reason
        objects, tips = set(map(int, object_ids)), set(map(int, self.h.tip_geom_ids))
        for c in self.scratch.contact:
            pair = (int(c.geom1), int(c.geom2))
            if objects.intersection(pair):
                if not tips.intersection(pair) and c.dist < .003:
                    return 'NON_TIP_CLEARANCE'
                if tips.intersection(pair) and c.dist < -.002:
                    return 'DEEP_PAD_PENETRATION'
        return None

    def edge_free(self, a, b):
        steps = max(2, int(np.ceil(np.max(abs(b-a))/.025))+1)
        return all(self.collision_reason(a+u*(b-a)) is None for u in np.linspace(0, 1, steps))

    def _rrt(self, root, goal, seed=7):
        rng = np.random.default_rng(seed)
        started = perf_counter()
        hold = root[7:].copy()
        initial_prefix = []
        arm_goal = goal.copy(); arm_goal[7:] = hold
        if self.collision_reason(arm_goal):
            # B's geometric finger objective may require a different posture.
            # Change that posture in the outside configuration before arm
            # transfer instead of freezing an incompatible hand throughout.
            prepared = root.copy(); prepared[7:] = goal[7:]
            if self.collision_reason(prepared) or not self.edge_free(root, prepared):
                return None
            initial_prefix = [root.copy()]
            root = prepared; hold = prepared[7:].copy()
            arm_goal = goal.copy()
        trees = [([root.copy()], [-1]), ([arm_goal.copy()], [-1])]
        def extend(tree, target):
            nodes, parents = tree
            distances = [np.linalg.norm(q[:7]-target[:7]) for q in nodes]
            parent = int(np.argmin(distances)); current = nodes[parent]
            delta = target[:7]-current[:7]; length = np.linalg.norm(delta)
            candidate = current.copy(); candidate[:7] += delta*min(1., .18/max(length, 1e-12))
            if not self.edge_free(current, candidate): return None, False
            nodes.append(candidate); parents.append(parent)
            return len(nodes)-1, length <= .18
        def ancestry(tree, index):
            nodes, parents = tree; path = []
            while index >= 0:
                path.append(nodes[index]); index = parents[index]
            return path[::-1]
        for iteration in range(1500):
            if perf_counter()-started > 2.0: return None
            which = iteration % 2
            sample = root.copy()
            sample[:7] = (trees[1-which][0][0][:7] if rng.random() < .18
                          else rng.uniform(self.lower[:7], self.upper[:7]))
            index, _ = extend(trees[which], sample)
            if index is None: continue
            target = trees[which][0][index]
            for _ in range(80):
                other_index, connected = extend(trees[1-which], target)
                if other_index is None: break
                if connected:
                    first = ancestry(trees[which], index)
                    second = ancestry(trees[1-which], other_index)
                    if which: first, second = second, first
                    path = first + second[-2::-1]
                    if not self.edge_free(path[-1], goal): return None
                    path.append(goal)
                    # Geometry-only shortcutting, independent of contacts.
                    simplified = [path[0]]; index = 0
                    while index < len(path)-1:
                        next_index = len(path)-1
                        while next_index > index+1 and not self.edge_free(path[index], path[next_index]):
                            next_index -= 1
                        simplified.append(path[next_index]); index = next_index
                    return np.array(initial_prefix+simplified)
        return None

    def plan(self, measured_data, target, *, allow_rrt=False):
        started = perf_counter()
        root = joint_state(self.h, measured_data)
        wrist = target.wrist if isinstance(target, WholeHandTarget) else target
        root_pose = pose(self.h, measured_data)
        initial_rotation, final_rotation = rotation(root_pose[3:]), rotation(wrist.pose_world[3:])
        angle = (final_rotation*initial_rotation.inv()).magnitude()
        distance = np.linalg.norm(wrist.pose_world[:3]-root_pose[:3])
        count = max(1, int(np.ceil(distance/.015)), int(np.ceil(angle/.10)))
        orient = Slerp([0., 1.], Rotation.concatenate([initial_rotation, final_rotation]))
        root_local = initial_rotation.inv().apply(tip_positions(self.h, measured_data)-root_pose[:3])
        goal_local = (final_rotation.inv().apply(target.fingertip_positions_world_m-wrist.pose_world[:3])
                      if isinstance(target, WholeHandTarget) else None)
        cartesian_path = [root.copy()]
        for fraction in np.linspace(0., 1., count+1)[1:]:
            p = (1-fraction)*root_pose[:3]+fraction*wrist.pose_world[:3]
            r = orient(float(fraction))
            intermediate = WristTarget(np.r_[p, quaternion(r)])
            if isinstance(target, WholeHandTarget):
                intermediate = WholeHandTarget(intermediate, p+r.apply(
                    (1-fraction)*root_local+fraction*goal_local))
            q, ep, er, ef = self.solve(cartesian_path[-1], intermediate, iterations=70)
            if ep > .002 or er > .04 or ef > .003 or not self.edge_free(cartesian_path[-1], q):
                cartesian_path = None; break
            cartesian_path.append(q)
        candidates = [np.array(cartesian_path)] if cartesian_path is not None else []
        if not candidates and not allow_rrt:
            return None, {'reason': 'CARTESIAN_PATH_BLOCKED'}
        seeds = [root.copy()]
        if hasattr(self, 'initial_measured_arm'):
            initial = root.copy(); initial[:7] = self.initial_measured_arm; seeds.append(initial)
        home = root.copy(); home[:7] = ARM_HOME_Q; seeds.append(home)
        for offset in (-.8, .8):
            trial = root.copy(); trial[2] += offset; trial[4] -= offset
            seeds.append(np.clip(trial, self.lower, self.upper))
        last_reason = 'IK'
        if not candidates:
            for seed in seeds:
                goal, ep, er, ef = self.solve(seed, target, iterations=100)
                if ep > .002 or er > .04 or ef > .003:
                    last_reason = 'IK'; continue
                reason = self.collision_reason(goal)
                if reason:
                    last_reason = reason; continue
                path = (np.array([root, goal]) if self.edge_free(root, goal)
                        else self._rrt(root, goal))
                if path is not None:
                    candidates.append(path); break
                last_reason = 'NO_GEOMETRIC_PATH'
        for path in candidates:
            durations = []
            for a, b in zip(path[:-1], path[1:]):
                delta = b-a
                translation, angular = 0., 0.
                for u in np.linspace(0, 1, 31):
                    s = u**3*(10-15*u+6*u*u); velocity = 30*u*u*(1-u)*(1-u)
                    set_joint_state(self.h, self.scratch, a+s*delta)
                    jp, jr = position_jacobian(self.h, self.scratch, self.h.palm_site_id)
                    translation = max(translation, velocity*np.linalg.norm(jp@delta))
                    angular = max(angular, velocity*np.linalg.norm(jr@delta))
                durations.append(max(.1, translation/self.config.wrist_speed_m_s,
                    angular/self.config.wrist_rotation_speed_rad_s,
                    1.875*np.max(abs(delta))/self.config.trajectory_joint_speed_rad_s))
            return JointPath(path, np.array(durations), pose(self.h, measured_data),
                             wrist.pose_world.copy(), perf_counter()-started), None
        return None, {'reason': last_reason}


def build_scene(directory, position=(.30, 0., .40)):
    directory = Path(directory).resolve()
    fixture = load_fixture('mustard', directory)
    components = fixture.mesh.split(only_watertight=False)
    exterior = max(components, key=lambda mesh: abs(mesh.volume))
    if not exterior.is_watertight or exterior.volume <= 0:
        raise ValueError('Whole-object traversal needs the closed physical exterior')
    removed = [mesh for mesh in components if mesh is not exterior]
    if (any(abs(mesh.volume) > 1e-12 for mesh in removed)
            or sum(mesh.area for mesh in removed)/fixture.mesh.area > 1e-7
            or sum(len(mesh.faces) for mesh in removed) > 4):
        raise ValueError('Cannot silently discard another physical surface')
    quality = {'raw_surface_area_m2': float(fixture.mesh.area),
        'exterior_area_m2': float(exterior.area), 'exterior_volume_m3': float(exterior.volume),
        'exterior_vertices': len(exterior.vertices), 'exterior_faces': len(exterior.faces),
        'excluded_zero_volume_components': [dict(vertices=len(mesh.vertices), faces=len(mesh.faces),
             area_m2=float(mesh.area), volume_m3=float(mesh.volume),
             bounds=mesh.bounds.tolist()) for mesh in removed],
        'removed_area_fraction': float(1 - exterior.area / fixture.mesh.area),
        'repair_basis': 'Two zero-volume duplicate/degenerate triangle components; no physical region is removed'}
    (directory / 'mesh_quality.json').write_text(json.dumps(quality, indent=2), encoding='utf-8')
    exterior.export(directory / 'canonical_task_surface.obj')
    graph = SurfaceGraph(exterior, coverage_radius_m=.025)
    original = fixture.make_plant(.002)
    delta = np.asarray(position) - original.object_position_m
    root = ET.fromstring(original.xml)
    root.find(".//body[@name='fr3_e05_object']").set('pos', ' '.join(map(str, position)))
    xml = ET.tostring(root, encoding='unicode')
    model = compile_model_xml(xml)
    base = replace(original.base, model=model, xml=xml, object_position_m=np.asarray(position))
    h = ObjectPlant(base, original.object_geom_ids, fixture)
    data = mujoco.MjData(model)
    q = np.r_[ARM_HOME_Q, HAND_NATURAL_Q]
    set_joint_state(h, data, q)
    initial_pose = pose(h, data)
    initial_pose[:3] += delta
    planner = SharedGeometricTrajectoryPlanner(h, ExperimentConfig())
    q, ep, er, _ = planner.solve(q, WristTarget(initial_pose), iterations=200)
    if ep > .001 or er > .02:
        raise RuntimeError('Common initial pose is unreachable after fixture placement')
    return h, graph, q, quality


def acquire(h, graph, q, seed=7):
    config = ExperimentConfig(seed=seed)
    data = mujoco.MjData(h.model)
    q = q + np.random.default_rng(seed).normal(0, .0003, 23)
    set_joint_state(h, data, q)
    set_control(h, data, q)
    points, _, triangles = trimesh.proximity.closest_point(graph.mesh,
        tip_positions(h, data) - h.object_position_m)
    inward_normals = graph.mesh.face_normals[triangles]
    planner = SharedGeometricTrajectoryPlanner(h, config)
    mcc = BasicPassiveMCC(h, config.dt_s)
    nominal, previous = q.copy(), q.copy()
    relief = SharedContactRecovery(config)
    peak, unsafe_count = 0., 0
    for _ in range(int(config.acquisition_s / config.dt_s)):
        forces, _, normals, active, unsafe, _ = contact_measurement(h, data, .2)
        peak = max(peak, float(forces.max()))
        unsafe_count += bool(unsafe)
        for i in range(4):
            if not active[i]:
                jp, _ = position_jacobian(h, data, h.tip_site_ids[i])
                columns = np.array([int(n) for n in FINGERS[i].joint_names]) + 7
                jf = jp[:, columns]
                dq = jf.T @ np.linalg.solve(jf @ jf.T + 1e-6*np.eye(3),
                    -inward_normals[i] * .006 * config.dt_s)
                nominal[columns] = np.clip(nominal[columns] + np.clip(dq, -.002, .002),
                    q[columns] - .6, q[columns] + .6)
        command = mcc.command(data, nominal, forces, normals, active, 2.)
        if forces.max() >= 8:
            relief.mode = 'FORCE_RELEASE'
            command = relief.relieve_force(h, data, joint_state(h, data), forces, normals)
            for i in range(4):
                if forces[i] >= 8:
                    columns = np.array([int(n) for n in FINGERS[i].joint_names]) + 7
                    nominal[columns] = command[columns]
                    mcc.controllers[i].reset()
        previous = np.clip(command, previous - .0024, previous + .0024)
        previous = np.clip(previous, planner.lower, planner.upper)
        set_control(h, data, previous)
        mujoco.mj_step(h.model, data)
        mujoco.mj_forward(h.model, data)
    f, _, normals, active, unsafe, _ = contact_measurement(h, data, .2)
    audit = dict(seed=seed, contacts=int(active.sum()), forces_n=f.tolist(), peak_n=peak,
                 non_tip_steps=int(unsafe_count), final_non_tip=unsafe,
                 state_sha256=array_digest(np.r_[data.qpos, data.qvel, data.ctrl]))
    if audit['contacts'] < 1 or unsafe_count or peak > 12:
        raise RuntimeError(f'Invalid common acquisition: {audit}')
    return data, audit


class FullSurfaceOracle:
    """Geometry-only SE(3) synthesis across all directions of the bottle."""
    def __init__(self, h, graph, data):
        self.h, self.graph = h, graph
        self.template_q = joint_state(h, data)
        self.initial_pose = pose(h, data)
        self.initial_rotation = rotation(self.initial_pose[3:])
        self.tip_local = self.initial_rotation.inv().apply(tip_positions(h, data) - self.initial_pose[:3])
        _, _, normals, active, _, _ = contact_measurement(h, data, .2)
        if not np.any(active):
            raise ValueError('Oracle template needs an acquired physical contact normal')
        self.initial_normal = normals[active].mean(axis=0)
        self.initial_normal /= np.linalg.norm(self.initial_normal)
        self.planner = FullSurfacePlanner(h, ExperimentConfig())
        self.pose_cache = {}
        self.pad_frames = [self.initial_rotation.inv().as_matrix() @
            data.geom_xmat[int(g)].reshape(3, 3) for g in h.tip_geom_ids]

    def target(self, vertex, roll, finger=0, clearance=.002, tilt=0.):
        normal = self.graph.normals[vertex]
        oriented = Rotation.from_rotvec(normal * roll) * align_vector(self.initial_normal, normal)
        # Tilt is a geometric pose choice at shoulders/neck, not a desired
        # contact-state label. Contact normals are still measured in physics.
        tangent = oriented.apply(np.eye(3)[np.argmin(abs(self.initial_normal))])
        tangent -= normal * (tangent @ normal)
        tangent /= np.linalg.norm(tangent)
        r = Rotation.from_rotvec(tangent * tilt) * oriented * self.initial_rotation
        frame = r.as_matrix() @ self.pad_frames[finger]
        support_vector = pad_support_vector(self.h.model, int(self.h.tip_geom_ids[finger]), frame, normal)
        point = self.h.object_position_m + self.graph.vertices_m[vertex]
        palm = point + support_vector + normal*clearance - r.apply(self.tip_local[finger])
        wrist = WristTarget(np.r_[palm, quaternion(r)])
        return WholeHandTarget(wrist, palm + r.apply(self.tip_local))

    def correct_cached_target(self, vertex, pose_value, tip_values, finger):
        normal = self.graph.normals[vertex]
        r = rotation(pose_value[3:])
        frame = r.as_matrix() @ self.pad_frames[finger]
        size = self.h.model.geom_size[int(self.h.tip_geom_ids[finger])]
        old_box_radius = float(abs(frame.T@normal)@size)
        vector = pad_support_vector(self.h.model, int(self.h.tip_geom_ids[finger]), frame, normal)
        delta = vector-normal*old_box_radius
        moved = pose_value.copy(); moved[:3] += delta
        return WholeHandTarget(WristTarget(moved), tip_values+delta)

    def find_pose(self, vertex, root_q=None, *, extended=False, variant=0):
        cache_key = (int(vertex), bool(extended), int(variant % 8))
        if cache_key in self.pose_cache:
            return self.pose_cache[cache_key].copy(), {}
        roots = [self.template_q] if root_q is None else [root_q, self.template_q]
        failures = {}
        # Preferred rotation first; all roll sectors are considered before
        # deciding a surface goal has no fixed-posture geometric witness.
        fingers = np.roll(np.arange(4), -(variant % 4))
        rolls = np.roll(np.array((0., np.pi/2, -np.pi/2, np.pi,
                         np.pi/4, -np.pi/4, 3*np.pi/4, -3*np.pi/4)), -(variant % 8))
        for finger in fingers:
            for roll, tilt in [(r, t) for t in ((0., np.pi/6, -np.pi/6) if extended else (0.,))
                    for r in rolls]:
                target = self.target(vertex, roll, finger, tilt=tilt)
                for root_q in roots:
                    root_q = root_q.copy()
                    root_q[7:] = self.template_q[7:]
                    q, ep, er, _ = self.planner.solve(root_q, target.wrist, iterations=75)
                    if ep > .001 or er > .015:
                        failures['IK'] = failures.get('IK', 0) + 1
                        continue
                    reason = self.planner.collision_reason(q)
                    if reason:
                        failures[reason] = failures.get(reason, 0) + 1
                        continue
                    object_geoms = set(map(int, self.h.object_geom_ids))
                    tips = set(map(int, self.h.tip_geom_ids))
                    penetration = max([0.] + [-c.dist for c in self.planner.scratch.contact
                        if object_geoms.intersection((int(c.geom1), int(c.geom2)))
                        and tips.intersection((int(c.geom1), int(c.geom2)))])
                    if penetration > .002:
                        failures['PAD_PENETRATION'] = failures.get('PAD_PENETRATION', 0) + 1
                        continue
                    witness = {'vertex': int(vertex), 'q': q, 'pose': target.wrist.pose_world.copy(),
                            'tips': target.fingertip_positions_world_m.copy(),
                            'roll_rad': float(roll), 'tilt_rad': float(tilt), 'finger_witness': finger,
                            'position_error_m': ep, 'rotation_error_rad': er}
                    self.pose_cache[cache_key] = witness
                    return witness.copy(), failures
        return None, failures


def geometry_audit(directory, position=(.30, 0., .40)):
    directory = Path(directory)
    h, graph, initial_q, quality = build_scene(directory, position)
    data, audit = acquire(h, graph, initial_q)
    print('INITIAL', json.dumps(audit), flush=True)
    oracle = FullSurfaceOracle(h, graph, data)
    witnesses, rejected = [], []
    previous = oracle.template_q
    for i, vertex in enumerate(graph.required_vertices):
        witness, failures = oracle.find_pose(int(vertex), previous)
        if witness is None:
            witness, failures = oracle.find_pose(int(vertex), previous, extended=True)
        if witness is None:
            rejected.append({'goal': i, 'vertex': int(vertex), 'normal': graph.normals[vertex].tolist(),
                'position': graph.vertices_m[vertex].tolist(), 'failures': failures})
        else:
            previous = witness['q']
            witness['goal'] = i
            witnesses.append(witness)
        if i % 20 == 0:
            print(f'GEOMETRY {i+1}/{graph.required_goal_count} feasible={len(witnesses)} rejected={len(rejected)}', flush=True)
            (directory / 'geometry_progress.json').write_text(json.dumps(
                dict(processed=i+1, required=graph.required_goal_count, feasible=len(witnesses), rejected=rejected), indent=2), encoding='utf-8')
    np.savez_compressed(directory / 'pose_witnesses.npz', **{key: np.array([w[key] for w in witnesses])
        for key in ('vertex', 'goal', 'q', 'pose', 'tips', 'roll_rad', 'tilt_rad', 'finger_witness')})
    (directory / 'geometry_audit.json').write_text(json.dumps(dict(initial=audit, quality=quality,
        support_model='actual_ellipsoid_vector.v1',
        required=graph.required_goal_count, feasible=len(witnesses), rejected=rejected), indent=2), encoding='utf-8')
    (directory / 'model.xml').write_text(h.xml, encoding='utf-8')
    np.savez_compressed(directory / 'initial.npz', qpos=data.qpos, qvel=data.qvel, ctrl=data.ctrl)
    print(f'GEOMETRY_DONE feasible={len(witnesses)}/{graph.required_goal_count}', flush=True)


def shifted_target(target, offset):
    wrist = target.wrist if isinstance(target, WholeHandTarget) else target
    value = wrist.pose_world.copy(); value[:3] += offset
    moved = WristTarget(value)
    return (WholeHandTarget(moved, target.fingertip_positions_world_m + offset)
            if isinstance(target, WholeHandTarget) else moved)


def pad_support_vector(model, geom, frame, normal):
    """Center-to-support vector, using the actual ellipsoid collision shape."""
    local = np.asarray(frame).T@np.asarray(normal)
    size = model.geom_size[geom]
    if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        radius = np.linalg.norm(size*local)
        return np.asarray(frame)@(size*size*local/max(radius, 1e-12))
    if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_SPHERE:
        return np.asarray(normal)*size[0]
    if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX:
        return np.asarray(frame)@(np.sign(local)*size)
    raise ValueError('Unsupported pad collision geometry')


def endpoint_arm_offset(h, data, outward, distance):
    """Translate the wrist without rotating away the finger's approach."""
    jp, jr = position_jacobian(h, data, h.palm_site_id)
    jacobian = np.vstack((jp[:, :7], .10*jr[:, :7]))
    motion = np.r_[-np.asarray(outward)*distance, np.zeros(3)]
    return jacobian.T @ np.linalg.solve(jacobian@jacobian.T+1e-6*np.eye(6), motion)


def complete_surface(coverage):
    return bool(np.all(coverage.visited_vertices) and np.all(coverage.visited_goals))


def cached_target(oracle, witnesses, index, support_model):
    if support_model == 'actual_ellipsoid_vector.v1':
        return WholeHandTarget(WristTarget(witnesses['pose'][index]), witnesses['tips'][index])
    if support_model == 'legacy_box_radius.v0':
        return oracle.correct_cached_target(int(witnesses['vertex'][index]), witnesses['pose'][index],
            witnesses['tips'][index], int(witnesses['finger_witness'][index]))
    raise ValueError(f'Unknown cached pad support model: {support_model}')


def run_sweep(h, graph, data, witnesses, directory, *, cell='B-MCC', seed=7,
              budget_s=1800., max_targets=0, resume_directory=None,
              support_model='actual_ellipsoid_vector.v1'):
    """Only mj_step advances the physical robot after common acquisition."""
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    config = ExperimentConfig(seed=seed, duration_s=budget_s+5,
        wrist_speed_m_s=.025, wrist_rotation_speed_rad_s=.35,
        trajectory_joint_speed_rad_s=.8, recovery_enabled=True)
    planner = FullSurfacePlanner(h, config)
    template_data = data
    if resume_directory is not None:
        template_data = mujoco.MjData(h.model)
        source = Path(resume_directory)
        source = source/('template_initial.npz' if (source/'template_initial.npz').exists() else 'initial.npz')
        with np.load(source) as initial_snapshot:
            template_data.qpos[:] = initial_snapshot['qpos']
            template_data.qvel[:] = initial_snapshot['qvel']
            template_data.ctrl[:] = initial_snapshot['ctrl']
        mujoco.mj_forward(h.model, template_data)
    oracle = FullSurfaceOracle(h, graph, template_data)
    np.savez_compressed(directory/'template_initial.npz', qpos=template_data.qpos,
        qvel=template_data.qvel, ctrl=template_data.ctrl)
    coverage = MeasuredCoverage(graph, config)
    if resume_directory is not None:
        with np.load(Path(resume_directory)/'trace.npz') as previous_trace:
            coverage.visited_vertices[:] = previous_trace['covered_vertices']
            coverage.visited_goals[:] = previous_trace['required_goal_visited']
        planner.initial_measured_arm = joint_state(h, template_data)[:7].copy()
    else:
        planner.initial_measured_arm = joint_state(h, data)[:7].copy()
    mcc = BasicPassiveMCC(h, .002)
    relief = SharedContactRecovery(config)
    gaps = ContactEpisodes(.15)
    last_command = np.r_[data.ctrl[h.arm_actuator_ids], data.ctrl[h.hand_actuator_ids]]
    nominal_hand = last_command[7:].copy()
    normal_in_hand = oracle.initial_rotation.inv().apply(oracle.initial_normal)
    available = {int(goal): index for index, goal in enumerate(witnesses['goal'])}
    attempted = np.zeros(graph.required_goal_count, dtype=int)
    residual_attempted = np.zeros(len(graph.vertices_m), dtype=int)
    events, logs = [], {key: [] for key in ('time_s', 'q_rad', 'qvel_rad_s',
        'command_rad', 'sensor_command_rad', 'palm_pose_world',
        'tip_positions_world_m', 'contact_positions_world_m', 'normals_world', 'forces_n',
        'active', 'coverage', 'covered_goals', 'goal_id', 'phase', 'endpoint_offset_m')}
    phase, plan, plan_elapsed = 'SELECT', None, 0.
    goal_id, goal_vertex, target, base_target = -1, -1, None, None
    queue = []
    transferred = False
    next_plan_time = 0.
    endpoint_started = stable_since = None
    full_since = None
    completion_nominal = None
    nominal_previous = last_command.copy()
    endpoint_offset = 0.
    control_normal = oracle.initial_normal.copy()
    plan_count = completed_targets = local_recoveries = geometric_refinements = 0
    peak = high_run = force_release_count = 0
    contact_sum = contact_ticks = force_square = force_samples = 0
    stop_reason = 'TIME_BUDGET_REACHED'
    wall_start = perf_counter()
    checkpoint_index = 0
    chunk_number = 0
    def record(elapsed, command, sensor_command, logged_phase):
        values = dict(time_s=elapsed, q_rad=joint_state(h, data),
            qvel_rad_s=np.r_[data.qvel[h.arm_dof_adrs], data.qvel[h.hand_dof_adrs]],
            command_rad=command.copy(), sensor_command_rad=sensor_command.copy(),
            palm_pose_world=pose(h, data), tip_positions_world_m=tip_positions(h, data),
            contact_positions_world_m=points.copy(), normals_world=normals.copy(), forces_n=forces.copy(),
            active=active.copy(), coverage=coverage.area_fraction,
            covered_goals=int(coverage.visited_goals.sum()), goal_id=goal_id,
            phase=logged_phase, endpoint_offset_m=endpoint_offset)
        for key, value in values.items(): logs[key].append(value)
    def checkpoint(elapsed):
        nonlocal checkpoint_index, chunk_number
        chunks = directory/'trace_chunks'; chunks.mkdir(exist_ok=True)
        arrays = {key: np.asarray(value[checkpoint_index:]) for key, value in logs.items()}
        if len(arrays['time_s']):
            np.savez_compressed(chunks/f'{chunk_number:04d}.npz', **arrays)
            chunk_number += 1; checkpoint_index = len(logs['time_s'])
        np.savez_compressed(directory/'coverage_checkpoint.tmp.npz',
            covered_vertices=coverage.visited_vertices, required_goal_visited=coverage.visited_goals,
            qpos=data.qpos, qvel=data.qvel, ctrl=data.ctrl, time=data.time, elapsed=elapsed)
        (directory/'coverage_checkpoint.tmp.npz').replace(directory/'coverage_checkpoint.npz')
        (directory/'events.json').write_text(json.dumps(events, indent=2), encoding='utf-8')
    def expose(full): return full.wrist if cell == 'A-MCC' else full
    def pick_target(elapsed):
        nonlocal goal_id, goal_vertex, base_target, control_normal, queue, target, transferred
        remaining = np.flatnonzero(~coverage.visited_goals)
        roots = coverage.last_vertices[active & (coverage.last_vertices >= 0)]
        if not len(roots):
            roots = coverage.last_vertices[coverage.last_vertices >= 0]
        if len(roots):
            distances = dijkstra(graph.adjacency, directed=False, indices=int(roots[0]))
        else:
            distances = np.linalg.norm(graph.vertices_m - (tip_positions(h, data)[0]-h.object_position_m), axis=1)
        candidates = [int(g) for g in remaining if int(g) in available]
        if candidates:
            measured_pose = pose(h, data)
            score = [distances[int(graph.required_vertices[g])] + .08*attempted[g]
                + 2*np.linalg.norm(witnesses['pose'][available[g], :3]-measured_pose[:3])
                + .04*(rotation(witnesses['pose'][available[g], 3:])*rotation(measured_pose[3:]).inv()).magnitude()
                for g in candidates]
            goal_id = candidates[int(np.argmin(score))]; index = available[goal_id]
            goal_vertex = int(graph.required_vertices[goal_id])
            full = cached_target(oracle, witnesses, index, support_model)
            if attempted[goal_id] >= 2:
                alternative, _ = oracle.find_pose(goal_vertex, joint_state(h, data), extended=True,
                    variant=int(attempted[goal_id]-1))
                if alternative is not None:
                    full = WholeHandTarget(WristTarget(alternative['pose']), alternative['tips'])
                    events.append({'time_s': elapsed, 'kind': 'ALTERNATIVE_GEOMETRIC_POSE',
                        'goal_id': goal_id, 'attempt': int(attempted[goal_id]+1),
                        'roll_rad': alternative['roll_rad'], 'tilt_rad': alternative['tilt_rad'],
                        'finger_witness_private': int(alternative['finger_witness'])})
        else:
            # Completion fill keeps the complete vertex denominator. A
            # nearby attainable touch may cover a geometrically blocked
            # point at the unchanged 25 mm geodesic scan resolution.
            residual = np.flatnonzero(~coverage.visited_vertices)
            if not len(residual): return False
            vertex = int(residual[np.argmin(distances[residual]+.05*residual_attempted[residual])])
            variant = int(residual_attempted[vertex])
            residual_attempted[vertex] += 1
            nearby = dijkstra(graph.adjacency, directed=False, indices=vertex, limit=.020)
            choices = np.flatnonzero(np.isfinite(nearby))
            choices = choices[np.argsort(nearby[choices])]
            witness = None
            sampled = choices[::max(1, len(choices)//12)][:12]
            for candidate in np.roll(sampled, -(variant % max(len(sampled), 1))):
                witness, _ = oracle.find_pose(int(candidate), joint_state(h, data), extended=True,
                    variant=variant)
                if witness is not None: break
            if witness is None:
                events.append({'time_s': elapsed, 'kind': 'UNREACHED_SURFACE', 'vertex': vertex})
                return False
            # A nearby attainable anchor is a pose proposal. Completion is
            # still judged at the originally uncovered residual vertex.
            goal_id, goal_vertex = -1, vertex
            full = WholeHandTarget(WristTarget(witness['pose']), witness['tips'])
        if goal_id >= 0: attempted[goal_id] += 1
        base_target = expose(full); target = base_target
        control_normal = rotation(full.wrist.pose_world[3:]).apply(normal_in_hand)
        queue = [('SURFACE', base_target)]
        transferred = False
        events.append({'time_s': elapsed, 'kind': 'SURFACE_TARGET', 'goal_id': goal_id,
            'vertex': goal_vertex, 'wrist_target': full.wrist.pose_world.tolist(),
            'exposed_fields': ['wrist_pose'] if cell == 'A-MCC' else ['wrist_pose', 'tip_positions']})
        return True
    steps = int(budget_s/.002)
    for step in range(steps):
        elapsed = step*.002
        forces, points, normals, active, unsafe, penetration = contact_measurement(h, data, .2)
        peak = max(peak, float(forces.max())); count = int(active.sum())
        contact_sum += count; contact_ticks += count > 0
        if count:
            force_square += float(np.sum((forces[active]-2.)**2)); force_samples += count
        safe_count = int(np.sum(active & (forces <= 8.)))
        gaps.observe(elapsed, safe_count)
        high_run = high_run+1 if forces.max() > 12 else 0
        hardware = ('NON_TIP_OBJECT_COLLISION' if unsafe else
                    'SUSTAINED_DANGEROUS_FORCE' if high_run*.002 >= .020 else
                    'NONFINITE_STATE' if not np.all(np.isfinite(data.qpos)) else
                    'JOINT_LIMIT' if np.any(joint_state(h, data) < planner.lower-.02)
                        or np.any(joint_state(h, data) > planner.upper+.02) else None)
        if hardware:
            events.append({'time_s': elapsed, 'kind': 'HARDWARE_STOP', 'reason': hardware,
                'forces_n': forces.tolist(), 'non_tip': unsafe, 'phase': phase})
            stop_reason = hardware; break
        if step % 5 == 0:
            coverage.update(points-h.object_position_m, normals, active)
        if complete_surface(coverage):
            if completion_nominal is None:
                completion_nominal = nominal_previous.copy()
                events.append({'time_s': elapsed, 'kind': 'ALL_SURFACE_MEASURED_PENDING_STABILITY'})
            phase = 'COMPLETION_HOLD'
            if safe_count and forces.max() <= 8.:
                full_since = elapsed if full_since is None else full_since
            else:
                full_since = None
            if full_since is not None and elapsed-full_since >= .15:
                record(elapsed, last_command, last_command, phase)
                stop_reason = 'FULL_MEASURED_SURFACE'; break
        # Intentional geometric transit is allowed to lose contact. Local
        # recovery begins at the new endpoint and never rewinds old coverage.
        if phase == 'SELECT' and elapsed >= next_plan_time:
            if max_targets and completed_targets >= max_targets:
                stop_reason = 'PILOT_TARGET_LIMIT'; break
            if not pick_target(elapsed):
                stop_reason = 'NO_GEOMETRIC_TARGET_FOR_REMAINING_SURFACE'; break
            phase = 'PLAN'
        if phase == 'PLAN':
            segment_phase, segment_target = queue[0]
            plan, failure = planner.plan(data, segment_target,
                allow_rrt=(segment_phase == 'OUTSIDE' or transferred))
            if plan is None and segment_phase == 'SURFACE' and not transferred:
                current = pose(h, data)
                if np.any(active):
                    outward = normals[active].mean(axis=0); outward /= np.linalg.norm(outward)
                else:
                    outward = rotation(current[3:]).apply(normal_in_hand)
                retreat = current.copy(); retreat[:3] += outward*.035
                retreat_target = WristTarget(retreat)
                if cell == 'B-MCC':
                    retreat_target = WholeHandTarget(retreat_target, tip_positions(h, data)+outward*.035)
                queue = [('RETREAT', retreat_target),
                         ('OUTSIDE', shifted_target(base_target, control_normal*.035)),
                         ('SURFACE', base_target)]
                plan, failure = planner.plan(data, queue[0][1])
                segment_phase = 'RETREAT'
                if plan is None:
                    # Moving the palm out along a pad normal can move an
                    # upstream arm link into the bottle. Find a small clear
                    # geometric escape first, keeping the measured hand pose.
                    directions = [np.eye(3)[i]*sign for i in (2, 1, 0) for sign in (1., -1.)]
                    escaped = False
                    for distance in (.010, .020, .035):
                        for direction in directions:
                            offset = direction*distance
                            alternative_pose = current.copy(); alternative_pose[:3] += offset
                            alternative_target = WristTarget(alternative_pose)
                            if cell == 'B-MCC':
                                alternative_target = WholeHandTarget(alternative_target,
                                    tip_positions(h, data)+offset)
                            plan, failure = planner.plan(data, alternative_target)
                            if plan is not None:
                                queue[0] = ('RETREAT', alternative_target)
                                events.append({'time_s': elapsed, 'kind': 'GEOMETRIC_RETREAT_ALTERNATIVE',
                                    'goal_id': goal_id, 'offset_m': offset.tolist()})
                                escaped = True; break
                        if escaped: break
                    if plan is None:
                        # Joint-space transfer need not begin with a straight
                        # Cartesian withdrawal if no such withdrawal is clear.
                        plan, failure = planner.plan(data, queue[1][1], allow_rrt=True)
                        if plan is not None:
                            queue.pop(0); segment_phase = 'OUTSIDE'
            if plan is None:
                events.append({'time_s': elapsed, 'kind': 'PATH_DEFER', 'goal_id': goal_id,
                               'segment': segment_phase, **failure})
                phase = 'SELECT'
                next_plan_time = elapsed+.50
                # No physical time is replaced with an IK witness.
            else:
                plan_elapsed = 0.; phase = segment_phase; target = queue[0][1]
                plan_count += 1
                events.append({'time_s': elapsed, 'kind': 'PLAN_COMMIT', 'goal_id': goal_id,
                    'phase': phase, 'duration_s': plan.duration_s, 'nodes': len(plan.waypoints),
                    'digest': plan.digest, 'latency_s': plan.latency_s})
        in_relief = forces.max() >= 8 or (relief.mode == 'FORCE_RELEASE' and forces.max() >= 4)
        if in_relief:
            if relief.mode != 'FORCE_RELEASE': force_release_count += 1
            relief.mode = 'FORCE_RELEASE'
        else:
            relief.mode = 'NORMAL'
        if phase in ('RETREAT', 'OUTSIDE', 'SURFACE') and plan is not None:
            if not in_relief: plan_elapsed += .002
            nominal = plan.sample(plan_elapsed)
            if plan_elapsed >= plan.duration_s:
                queue.pop(0)
                if phase == 'OUTSIDE': transferred = True
                if queue:
                    phase = 'PLAN'
                else:
                    phase = 'ENDPOINT'; endpoint_started = elapsed
                    endpoint_offset = 0.; stable_since = None; target = base_target
        elif phase == 'ENDPOINT':
            nominal = plan.waypoints[-1].copy()
            if safe_count:
                stable_since = elapsed if stable_since is None else stable_since
            else:
                stable_since = None
            reached = (coverage.visited_goals[goal_id] if goal_id >= 0 else coverage.visited_vertices[goal_vertex])
            if reached and stable_since is not None and elapsed-stable_since >= .15:
                completed_targets += 1
                events.append({'time_s': elapsed, 'kind': 'MEASURED_TARGET_VISIT', 'goal_id': goal_id,
                    'vertex': goal_vertex, 'contact_count': safe_count, 'covered_area_fraction': coverage.area_fraction})
                phase = 'SELECT'
            elif elapsed-endpoint_started > 7.:
                events.append({'time_s': elapsed, 'kind': 'CONTACT_ENDPOINT_TIMEOUT_CONTINUE',
                    'goal_id': goal_id, 'contact_count': count, 'target_covered': bool(reached)})
                completed_targets += 1; phase = 'SELECT'
            elif (not reached or not safe_count) and elapsed-endpoint_started >= .15 and not in_relief:
                if endpoint_offset == 0:
                    if safe_count:
                        geometric_refinements += 1
                        events.append({'time_s': elapsed, 'kind': 'ORACLE_GEOMETRIC_REFINEMENT',
                            'goal_id': goal_id, 'basis': 'Requested region has not received measured coverage',
                            'exposed_fields': ['wrist_pose'] if cell == 'A-MCC' else ['wrist_pose', 'tip_positions']})
                    else:
                        local_recoveries += 1
                        events.append({'time_s': elapsed, 'kind': 'ENDPOINT_LOCAL_RECOVERY', 'goal_id': goal_id})
                endpoint_offset = min(.008, endpoint_offset + .0015*.002)
            if endpoint_offset > 0 and phase == 'ENDPOINT':
                nominal[:7] += endpoint_arm_offset(h, data, control_normal, endpoint_offset)
        elif phase == 'COMPLETION_HOLD':
            nominal = completion_nominal.copy()
        else:
            nominal = last_command.copy()
        if cell == 'A-MCC': nominal[7:] = nominal_hand
        nominal_previous = nominal.copy()
        command = mcc.command(data, nominal, forces, normals, active, 2.)
        if in_relief:
            command[:7] = last_command[:7]
            command = relief.relieve_force(h, data, command, forces, normals)
        command = np.clip(command, last_command-.0024, last_command+.0024)
        command = np.clip(command, planner.lower, planner.upper)
        if step % 5 == 0 and phase != 'SELECT':
            reason = planner.collision_reason(command)
            if reason in ('NON_TIP_CLEARANCE', 'NON_TIP_OBJECT_COLLISION', 'SELF_COLLISION'):
                events.append({'time_s': elapsed, 'kind': 'COMMAND_GEOMETRY_DEFER',
                    'goal_id': goal_id, 'phase': phase, 'reason': reason})
                command = np.clip(joint_state(h, data), last_command-.0024, last_command+.0024)
                command = np.clip(command, planner.lower, planner.upper)
                phase = 'SELECT'; next_plan_time = elapsed+.50
        sensor_command = last_command.copy()
        last_command = command.copy(); set_control(h, data, command)
        if step % 5 == 0:
            record(elapsed, command, sensor_command, 'FORCE_RELIEF' if in_relief else phase)
        mujoco.mj_step(h.model, data); mujoco.mj_forward(h.model, data)
        if step % 10000 == 0:
            print(f'SWEEP {cell} seed={seed} t={elapsed:.1f}s coverage={coverage.area_fraction:.2%} '
                f'goals={coverage.visited_goals.sum()}/{graph.required_goal_count} phase={phase} '
                f'contacts={count} targets={completed_targets}', flush=True)
            (directory/'progress.json').write_text(json.dumps(dict(time_s=elapsed, phase=phase,
                coverage=coverage.area_fraction, goals=int(coverage.visited_goals.sum()),
                required=graph.required_goal_count, contacts=count, peak_force_n=peak,
                goal_id=goal_id, endpoint_offset_m=endpoint_offset, forces_n=forces.tolist(),
                q_rad=joint_state(h, data).tolist(), palm=pose(h, data).tolist(),
                target_pose=None if base_target is None else
                    (base_target.wrist if isinstance(base_target, WholeHandTarget) else base_target).pose_world.tolist()), indent=2), encoding='utf-8')
        if step % 60000 == 0:
            checkpoint(elapsed)
    checkpoint(step*.002)
    arrays = {key: np.asarray(value) for key, value in logs.items()}
    arrays['covered_vertices'] = coverage.visited_vertices
    arrays['required_goal_visited'] = coverage.visited_goals
    np.savez_compressed(directory/'final_state.npz', qpos=data.qpos, qvel=data.qvel, ctrl=data.ctrl, time=data.time)
    np.savez_compressed(directory/'trace.npz', **arrays)
    (directory/'events.json').write_text(json.dumps(events, indent=2), encoding='utf-8')
    elapsed = min(budget_s, (step+1)*.002) if stop_reason == 'TIME_BUDGET_REACHED' else step*.002
    summary = dict(cell=cell, seed=seed, config=asdict(config), stop_reason=stop_reason,
        full_surface_complete=complete_surface(coverage) and stop_reason == 'FULL_MEASURED_SURFACE',
        all_surface_measured=complete_surface(coverage), completion_stability_s=.15,
        final_covered_area_fraction=coverage.area_fraction,
        visited_required_goals=int(coverage.visited_goals.sum()), required_goal_count=graph.required_goal_count,
        covered_vertex_count=int(coverage.visited_vertices.sum()), vertex_count=len(graph.vertices_m),
        executed_s=elapsed, wall_s=perf_counter()-wall_start, plan_count=plan_count,
        contact_continuity=contact_ticks/max(step+1, 1), mean_contacts=contact_sum/max(step+1, 1),
        peak_force_n=peak, force_rmse_n=float(np.sqrt(force_square/max(force_samples, 1))),
        force_relief_count=force_release_count, endpoint_local_recovery_count=local_recoveries,
        oracle_geometric_refinement_count=geometric_refinements,
        contact_loss_episodes=gaps.result(elapsed), contact_loss_is_failure=False,
        normal_rejected_contacts=coverage.normal_rejected_contacts,
        trace_sha256=sha256((directory/'trace.npz').read_bytes()).hexdigest())
    (directory/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('SWEEP_RESULT', json.dumps({key: summary[key] for key in ('cell', 'seed', 'stop_reason',
        'full_surface_complete', 'final_covered_area_fraction', 'visited_required_goals',
        'covered_vertex_count', 'executed_s', 'wall_s', 'plan_count', 'peak_force_n')}), flush=True)
    return summary


def physical_run(geometry_directory, directory, cell, budget, max_targets=0, seed=7,
                 resume_directory=None, formal=False):
    geometry_directory, directory = Path(geometry_directory), Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory/'manifest.json').exists(): raise FileExistsError('Choose a fresh experiment directory')
    if formal and (resume_directory is not None or max_targets or budget != 9000.):
        raise ValueError('Frozen formal runs start from acquisition and use the 9000 s maximum budget')
    h, graph, q, quality = build_scene(directory)
    np.savez_compressed(directory/'surface_graph.npz', vertices=graph.vertices_m,
        faces=graph.faces, normals=graph.normals, required_vertices=graph.required_vertices)
    data, initial = acquire(h, graph, q, seed)
    if resume_directory is not None:
        resume_directory = Path(resume_directory)
        if (resume_directory/'final_state.npz').exists():
            with np.load(resume_directory/'final_state.npz') as state:
                data.qpos[:] = state['qpos']; data.qvel[:] = state['qvel']; data.ctrl[:] = state['ctrl']
                data.time = float(state['time'])
            restart_type = 'EXACT_RECORDED_STATE_REVISED_CONTROLLER_DEVELOPMENT'
        else:
            with np.load(resume_directory/'trace.npz') as trace:
                set_joint_state(h, data, trace['q_rad'][-1])
                data.qvel[:] = 0.
                set_control(h, data, trace['command_rad'][-1])
            restart_type = 'RECORDED_POSE_ZERO_VELOCITY_RESTART_DEVELOPMENT'
        mujoco.mj_forward(h.model, data)
    else:
        restart_type = None
    witnesses = np.load(geometry_directory/'pose_witnesses.npz')
    geometry_meta = json.loads((geometry_directory/'geometry_audit.json').read_text(encoding='utf-8'))
    geometry_xml = ET.fromstring((geometry_directory/'model.xml').read_text(encoding='utf-8'))
    geometry_position = np.fromstring(geometry_xml.find(".//body[@name='fr3_e05_object']").get('pos'), sep=' ')
    if not np.allclose(geometry_position, h.object_position_m, atol=1e-12, rtol=0):
        raise ValueError('Cached geometry was audited at a different fixture position')
    if geometry_meta['required'] != graph.required_goal_count:
        raise ValueError('Cached geometry and complete required surface differ')
    np.testing.assert_array_equal(witnesses['vertex'], graph.required_vertices[witnesses['goal']])
    support_model = geometry_meta.get('support_model', 'legacy_box_radius.v0')
    validation_path = geometry_directory/'support_validation.json'
    if validation_path.exists():
        validation = json.loads(validation_path.read_text(encoding='utf-8'))
        if validation['pose_witnesses_sha256'] != sha256((geometry_directory/'pose_witnesses.npz').read_bytes()).hexdigest():
            raise ValueError('Geometry support validation does not match the cached witnesses')
        support_model = validation['support_model']
    if formal and support_model != 'actual_ellipsoid_vector.v1':
        raise ValueError('Formal whole-surface runs use validated actual ellipsoid support targets')
    np.savez_compressed(directory/'initial.npz', qpos=data.qpos, qvel=data.qvel, ctrl=data.ctrl)
    (directory/'model.xml').write_text(h.xml, encoding='utf-8')
    root = Path(__file__).resolve().parents[2]
    snapshot = directory/'source_snapshot'; snapshot.mkdir(exist_ok=True)
    (snapshot/'full_surface.py').write_bytes(Path(__file__).read_bytes())
    manifest = dict(protocol='i04-mustard-full-surface.v1' if formal else 'i04-mustard-full-surface.development',
        object_position_m=h.object_position_m.tolist(),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        cell=cell, seed=seed, budget_s=budget, pilot_max_targets=max_targets, initial_state=initial,
        resumed_development_run=None if resume_directory is None else str(resume_directory.resolve()),
        restart_type=restart_type,
        geometry_quality=quality, geometry_source=str(geometry_directory.resolve()), dp='DEFERRED_BY_USER',
        fixture=h.fixture.metadata,
        model_xml_sha256=sha256(h.xml.encode('utf-8')).hexdigest(),
        pose_witnesses_sha256=sha256((geometry_directory/'pose_witnesses.npz').read_bytes()).hexdigest(),
        recovery_policy=dict(name='CAUSAL_ENDPOINT_LOCAL_SEARCH.v1', global_old_region_rewind=False,
            approach_delay_s=.15, stable_confirmation_s=.15, local_timeout_s=7.,
            approach_speed_m_s=.0015, approach_bound_m=.008, planned_transit_contact_loss_allowed=True),
        completion_policy=dict(all_vertices=True, all_required_goals=True,
            final_safe_contact_confirmation_s=.15, maximum_final_force_n=8.),
        cached_target_support_model=support_model,
        cached_target_support_correction=support_model == 'legacy_box_radius.v0',
        coverage=dict(radius_m=.025, residual_limit_m=.005, normal_cosine=.5, force_threshold_n=.2,
                      source='actual MuJoCo pad contacts only', required_goals=graph.required_goal_count,
                      vertices=len(graph.vertices_m), area_m2=float(graph.mesh.area)),
        code_hashes={name:sha256((root/name).read_bytes()).hexdigest() for name in (
            'Module/i04_geometric_traversal/full_surface.py', 'Module/i04_geometric_traversal/physics.py',
            'Module/i04_geometric_traversal/contracts.py', 'Module/i04_geometric_traversal/planning.py',
            'Module/i04_geometric_traversal/recovery.py', 'Module/i04_geometric_traversal/fixtures.py',
            'Module/i04_geometric_traversal/oracle.py', 'Module/fr3_leap/model.py')})
    (directory/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    run_sweep(h, graph, data, witnesses, directory, cell=cell, seed=seed,
              budget_s=budget, max_targets=max_targets, resume_directory=resume_directory,
              support_model=support_model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--position', type=float, nargs=3, default=(.30, 0., .40))
    parser.add_argument('--geometry', type=Path, help='Use a pre-audited complete geometry witness set for physical traversal')
    parser.add_argument('--cell', choices=('A-MCC','B-MCC'), default='B-MCC')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--budget', type=float, default=1800.)
    parser.add_argument('--max-targets', type=int, default=0)
    parser.add_argument('--resume', type=Path, help='Continue a development physical checkpoint; not a frozen formal run')
    parser.add_argument('--formal', action='store_true', help='Frozen fresh-acquisition evaluation; no development resume')
    args = parser.parse_args()
    if args.geometry:
        physical_run(args.geometry, args.output, args.cell, args.budget, args.max_targets, args.seed, args.resume, args.formal)
    else:
        geometry_audit(args.output, args.position)


if __name__ == '__main__':
    main()
