"""Fresh I04 A/B MCC trials with a wrist-only A information boundary.

The published development controller is deliberately left byte-for-byte intact.
Both cells call its SAME measurement, MCC, guard, native step and logging code.
Only reference construction differs. The private geometric task service can
inspect the object; WristOnlyReference receives only WristTarget instances.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import mujoco
import numpy as np

from .belly_contact import registered_belly_frame
from .contracts import WristTarget, ExperimentConfig
from .full_surface import JointPath
from .natural_belly_traversal import NaturalSafetyPlanner, set_kinematics, unit
from .physics import pose, rotation
from .published_whole_hand import CASE, GEOMETRY, SEED, ROOT, install_asset_adapter
from .whole_hand_adaptive_completion import AdaptiveCompletionPlanner
from .whole_hand_belly_probe import snapshot_sources
from .whole_hand_completion import CompletionWholeHandTraversal
from .whole_hand_geometry_cache import CommandGeometryCache
from .whole_hand_recovery import RecoveryWholeHandTraversal
from .whole_hand_surface_traversal import WholeHandTraversal


@dataclass(frozen=True)
class CommonSettings:
    protocol: str = 'i04.ab.whole_hand.v2.mcc.1'
    seed: int = 7
    dt_s: float = .002
    duration_s: float = 7000.
    measured_quorum: int = 2
    window_samples: int = 150
    window_fraction: float = .8
    recontact_timeout_s: float = 6.
    approach_minimum_s: float = 12.
    transfer_force_pause_budget_s: float = 40.
    sliding_speed_m_s: float = .004
    sliding_rotation_speed_rad_s: float = .12
    desired_force_n: float = 1.
    mcc_mass: float = .08
    mcc_damping: float = 70.
    mcc_stiffness: float = 20.
    mcc_offset_m: float = .004
    mcc_velocity_m_s: float = .00025
    mcc_acceleration_m_s2: float = .02
    release_n: float = 2.5
    resume_n: float = 1.2
    excluded_force_n: float = 8.
    hard_force_n: float = 12.
    hard_force_s: float = .020
    coverage_radius_m: float = .025
    final_safe_hold_s: float = .15
    nominal_pad_clearance_m: float = .0002


def json_hash(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def state_hash(data):
    digest = sha256()
    for name in ('qpos', 'qvel', 'ctrl', 'qacc_warmstart', 'time'):
        digest.update(name.encode())
        digest.update(np.asarray(getattr(data, name), dtype='<f8').tobytes())
    return digest.hexdigest()


def measured_ready(history, settings):
    """No oracle predictions, selected anchor, target normals or GT inputs."""
    safe = np.asarray(history, dtype=bool)
    return bool(safe.shape == (settings.window_samples, 4)
                and safe[-1].sum() >= settings.measured_quorum
                and np.mean(safe.sum(axis=1) >= settings.measured_quorum)
                >= settings.window_fraction)


class WristOnlyReference:
    """A's planner: arm IK, fixed public nominal fingers, robot FK only.

    Collision geometry is encapsulated in the safety planner. It is never used
    to project a pad onto the object or infer a contact target/normal. No graph,
    B joint witness, B finger targets or cooperative seed database is accepted.
    """
    def __init__(self, handles, initial_q, lower, upper, settings):
        self.h = handles
        self.settings = settings
        self.safety = NaturalSafetyPlanner(handles, ExperimentConfig(seed=settings.seed))
        self.safety.lower = np.asarray(lower).copy()
        self.safety.upper = np.asarray(upper).copy()
        self.scratch = mujoco.MjData(handles.model)
        self.nominal_hand = np.asarray(initial_q[7:]).copy()
        self.previous = np.asarray(initial_q).copy()

    def solve(self, target, *, root=None, fixed_hand=None):
        if type(target) is not WristTarget:
            raise TypeError('I04A accepts only a WristTarget')
        q = self.previous.copy() if root is None else np.asarray(root).copy()
        q[7:] = self.nominal_hand if fixed_hand is None else fixed_hand
        result, ep, er, _ = self.safety.solve(q, target, iterations=100)
        if ep > .002 or er > .04:
            raise RuntimeError(f'A_WRIST_IK_UNREACHABLE:{ep:.6g}m,{er:.6g}rad')
        np.testing.assert_array_equal(result[7:], q[7:])
        self.previous = result.copy()
        return result

    def pad_reference(self, q):
        """Nominal pad poses on the ROBOT, not surface points on the object."""
        set_kinematics(self.h, self.scratch, q)
        points, normals = [], []
        for geom in self.h.tip_geom_ids:
            frame = registered_belly_frame(self.h.model, self.scratch, int(geom))
            normal = -frame[:, 2]
            support = self.scratch.geom_xpos[geom] + self.h.model.geom_size[geom, 2] * frame[:, 2]
            points.append(support - self.settings.nominal_pad_clearance_m * normal)
            normals.append(normal)
        return np.asarray(points), np.asarray(normals)


class MatchedMCCTraversal(CompletionWholeHandTraversal):
    def __init__(self, output, cell, seconds, *, initial_file=None, local_only=False):
        if cell not in ('A-MCC', 'B-MCC'):
            raise ValueError(cell)
        self.cell = cell
        self.settings = CommonSettings(duration_s=seconds)
        self.local_only = local_only
        self.wall_start = perf_counter()
        RecoveryWholeHandTraversal.__init__(self, output, GEOMETRY, SEED, seconds, resume=None)
        if initial_file is not None:
            with np.load(initial_file) as z:
                for name in ('qpos', 'qvel', 'ctrl', 'qacc_warmstart'):
                    getattr(self.data, name)[:] = z[name]
                self.data.time = float(z['time'])
            mujoco.mj_forward(self.h.model, self.data)
            from .physics import joint_state
            self.last = joint_state(self.h, self.data)
            self.initial_q = self.last.copy()
            self.points, self.normals = self.planner.project_pads(self.last)
            self.normal_filter = self.normals.copy()
            np.savez_compressed(self.out/'initial.npz', qpos=self.data.qpos, qvel=self.data.qvel,
                                ctrl=self.data.ctrl, qacc_warmstart=self.data.qacc_warmstart, time=self.data.time)
        self.planner = AdaptiveCompletionPlanner(self.h, self.graph, SEED)
        if not local_only:
            print('Loading immutable geometry-only oracle seeds', flush=True)
            self.planner.load_cooperative_seeds(CASE)
        self.command_cache = CommandGeometryCache(self.planner.safety)
        self.planner.safety.command_collision_reason = self.command_cache.command_reason
        self.nominal = self.last.copy()
        self.attempts = np.zeros(len(self.graph.vertices_m), int)
        self.a_reference = WristOnlyReference(self.h, self.initial_q, self.planner.lower,
                                             self.planner.upper, self.settings)
        self.oracle_scratch = mujoco.MjData(self.h.model)
        self.reference_index = -1
        self.public_wrist = self._wrist(self.initial_q)
        self.reference_file = self.out/'nominal_wrist_reference.jsonl'
        self.reference_file.write_text('', encoding='utf-8')
        self._last_private_key = None
        self._last_adapted = None
        self.input_normal_source = 'A_ROBOT_NOMINAL' if cell == 'A-MCC' else 'B_SURFACE_PLANNING'
        manifest = json.loads((self.out/'manifest.json').read_text())
        sources = snapshot_sources(self.out)
        manifest.update(
            protocol=self.settings.protocol, method=cell, cell=cell,
            shared_config=asdict(self.settings), shared_config_sha256=json_hash(asdict(self.settings)),
            shared_source_hashes=sources, source_hashes=sources,
            initial_state_sha256=state_hash(self.data), initial_state_copied_from=str(initial_file),
            target_access='WRIST_POSE_ONLY' if cell == 'A-MCC' else 'WRIST_AND_PER_FINGER_SURFACE',
            nominal_hand_rad=self.a_reference.nominal_hand.tolist(),
            nominal_hand_source='Identical declared physical initialization; fixed thereafter in A nominal planning',
            nominal_wrist_reference_sha256=None, fresh_single_version=True, coverage_inherited=False, resume=None,
            shared_nominal_wrist_route=False, shared_target_selection_policy=True,
            nominal_wrist_reference_scope='Own causal oracle requests, logged before method-side adaptation; not a common A/B realized path',
            initial_handoff='300ms: measured >=2 contacts for80%, current>=2; no expected_contacts or anchor gate',
            subsequent_handoff='Same measured gate as initial handoff',
            all_four_contact_tasks_at_every_surface_tick=True,
            reference_sources='A robot-FK nominal pad poses; B GT finite surface points; both same sensor-filter/MCC/guard',
            history_recovery='Retained measured-normal filter and bounded MCC offsets; measured normals used when available; method reference only as fallback',
            seed_use='Private oracle geometry-only seeds; never passed to A method; no physical state or coverage inherited',
            cooperative_seed_source=str(CASE), local_validation_only=local_only,
            comparison_scope='One seed implementation trial; historical run10 is descriptive context, not a matched counterpart',
        )
        (self.out/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    def _wrist(self, q):
        set_kinematics(self.h, self.oracle_scratch if hasattr(self, 'oracle_scratch') else self.scratch, q)
        return pose(self.h, self.oracle_scratch if hasattr(self, 'oracle_scratch') else self.scratch).copy()

    def _publish_wrist(self, poses, phase, **schedule):
        poses = np.atleast_2d(poses)
        self.reference_index += 1
        entry = dict(index=self.reference_index, requested_at_s=float(self.data.time), phase=phase,
                     poses_world=poses.tolist(), **schedule)
        with self.reference_file.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(entry, separators=(',', ':'))+'\n')
        return [WristTarget(item) for item in poses]

    def _adapt_patch(self, points, normals, q, phase):
        # The private service extracts ONLY the palm pose before calling A.
        wrist = self._wrist(q)
        target = self._publish_wrist(wrist, phase)[0]
        if self.cell == 'A-MCC':
            nominal = self.a_reference.solve(target, root=self.last)
            own_points, own_normals = self.a_reference.pad_reference(nominal)
            return own_points, own_normals, nominal, wrist
        return np.asarray(points).copy(), np.asarray(normals).copy(), np.asarray(q).copy(), wrist

    def _execute(self, points, normals, nominal, wrist, phase):
        self.public_wrist = np.asarray(wrist).copy()
        self.method_normals = np.asarray(normals).copy()
        return WholeHandTraversal.contact_step(self, points, normals, nominal, phase)

    def contact_step(self, points, normals, nominal, phase):
        # Calls from the private completion service (hold/recontact) also pass
        # through the same boundary; they cannot bypass the A adapter.
        key = (self._wrist(nominal).tobytes() if self.cell == 'A-MCC' else
               (np.asarray(nominal).tobytes(), np.asarray(points).tobytes(), np.asarray(normals).tobytes()))
        if key != self._last_private_key:
            self._last_adapted = self._adapt_patch(points, normals, nominal, phase)
            self._last_private_key = key
        return self._execute(*self._last_adapted, phase)

    def tick(self, command, phase, obs=None):
        # Additional audit fields contain only public input and own controller state.
        obs = self.measure() if obs is None else obs
        normal_source = np.full(4, self.input_normal_source, dtype='U32')
        if hasattr(self, 'method_normals'):
            raw_ok = obs['a'] & (np.sum(obs['n']*self.method_normals, axis=1) > .7)
            belly_ok = obs['ba'] & (np.sum(obs['bn']*self.method_normals, axis=1) > .7)
            normal_source[raw_ok] = 'MEASURED_RAW'
            normal_source[belly_ok] = 'MEASURED_BELLY'
        extra = dict(public_wrist_pose_world=self.public_wrist.copy(),
                     public_reference_index=self.reference_index,
                     reference_is_robot_nominal=self.cell == 'A-MCC', input_normal_source=normal_source)
        for key, value in extra.items():
            self.logs.setdefault(key, []).append(value)
        before = self.states
        try:
            return WholeHandTraversal.tick(self, command, phase, obs)
        except Exception:
            if self.states == before:
                for key in extra:
                    self.logs[key].pop()
            raise

    def approach(self, patch, clearance=.006, strict_four=False):
        points, normals, goal, wrist = self._adapt_patch(patch['points'], patch['normals'], patch['q'], 'APPROACH')
        start = self.data.time
        root = self.last.copy()
        self.window.clear(); self.bias[:] = 0.; self.offsets[:] = 0.
        self.clearance[:] = clearance; self.normal_filter = normals.copy()
        for controller in self.controllers:
            controller.reset()
        budget = max(self.settings.approach_minimum_s, clearance/.003+8.)
        while self.data.time < self.duration and self.data.time-start < budget:
            fraction = min(1., (self.data.time-start)/max(2., clearance/.003))
            nominal = (1-fraction)*root+fraction*goal
            self._execute(points, normals, nominal, wrist, 'COOPERATIVE_APPROACH')
            if measured_ready(self.window, self.settings):
                return True
            if self.rejections > 300:
                return False
        return False

    def slide(self, route, phase='COOPERATIVE_SLIDE'):
        poses = np.asarray([self._wrist(q) for q in route['q']])
        targets = self._publish_wrist(poses, 'SLIDE', path_parameter_m=route['distance'].tolist())
        if self.cell == 'A-MCC':
            qs, points, normals = [], [], []
            previous = self.last.copy()
            for target in targets:
                q = self.a_reference.solve(target, root=previous)
                pp, nn = self.a_reference.pad_reference(q)
                qs.append(q); points.append(pp); normals.append(nn); previous = q
            qs, points, normals = np.asarray(qs), np.asarray(points), np.asarray(normals)
        else:
            qs, points, normals = route['q'], route['points'], route['normals']
        distances = route['distance']; length = distances[-1]
        self.goal = int(route.get('goal', -1)); self.routes += 1
        start = self.data.time; progress = 0.; gap = 0.
        while self.data.time < self.duration and self.data.time-start < max(10., length/.0015+5.):
            k = min(max(0, np.searchsorted(distances, progress)-1), len(distances)-2)
            fraction = (progress-distances[k])/max(1e-9, distances[k+1]-distances[k])
            pp = (1-fraction)*points[k]+fraction*points[k+1]
            nn = np.array([unit(n) for n in (1-fraction)*normals[k]+fraction*normals[k+1]])
            q = (1-fraction)*qs[k]+fraction*qs[k+1]
            # FK of own nominal q records the executed nominal palm reference.
            self.route_progress = progress
            obs = self._execute(pp, nn, q, self._wrist(q), phase)
            enough = obs['safe'].sum() >= self.settings.measured_quorum
            gap = 0. if enough else gap+self.settings.dt_s
            if enough and not self.release.any() and obs['rejected'].max() < .2 and self.rejections < 3:
                # Public wrist rotation, never a private per-finger GT normal,
                # determines the shared speed cap in BOTH cells.
                angle = (rotation(poses[k+1, 3:])*rotation(poses[k, 3:]).inv()).magnitude()
                speed = min(self.settings.sliding_speed_m_s,
                            self.settings.sliding_rotation_speed_rad_s*(distances[k+1]-distances[k])/max(angle, 1e-9))
                progress = min(length, progress+speed*self.settings.dt_s)
            if self.done_hold >= self.settings.final_safe_hold_s:
                return True, 'FULL_MEASURED_BELLY_SURFACE'
            if progress >= length-1e-10 and enough:
                return True, 'ROUTE_FINISHED'
            if gap > self.settings.recontact_timeout_s:
                return False, 'MEASURED_RECONTACT_REPLAN'
            if self.rejections > 250:
                return False, 'GEOMETRY_REPLAN'
        return False, 'ROUTE_TIME_BUDGET'

    def _a_transfer(self, path, root):
        poses = np.asarray([self._wrist(q) for q in path.waypoints])
        targets = self._publish_wrist(poses, 'TRANSFER', segment_durations_s=path.segment_durations.tolist())
        hand = root[7:].copy(); waypoints = [root.copy()]; durations = []
        for target in targets[1:]:
            q = self.a_reference.solve(target, root=waypoints[-1], fixed_hand=hand)
            # Safety RRT samples the arm only and interpolates the fixed hand.
            segment = self.planner.path_between(waypoints[-1], q)
            if segment is None:
                return None
            if not np.allclose(segment.waypoints[:, 7:], hand, rtol=0., atol=1e-12):
                raise RuntimeError('A_TRANSFER_CHANGED_NOMINAL_FINGERS')
            waypoints.extend(segment.waypoints[1:])
            durations.extend(segment.segment_durations)
        waypoints = np.asarray(waypoints)
        own_poses = [self._wrist(q) for q in waypoints]
        return JointPath(waypoints, np.asarray(durations), own_poses[0], own_poses[-1], 0.)

    def prepare_transfer(self, target):
        # Geometric proposal stays inside the private oracle. Method A receives
        # its palm poses and builds its own arm-only path from its own state.
        result = RecoveryWholeHandTraversal.prepare_transfer(self, target)
        if result is None or self.cell == 'B-MCC':
            return result
        first, middle, hover, clearance = result
        try:
            own_first = self._a_transfer(first, self.last)
            if own_first is None:
                return None
            own_middle = self._a_transfer(middle, own_first.waypoints[-1])
        except RuntimeError as error:
            self.events.append(dict(kind='A_WRIST_TRANSFER_REJECTED', time_s=float(self.data.time), reason=str(error)))
            return None
        if own_middle is None:
            return None
        return own_first, own_middle, hover, clearance

    def air_move(self, path, phase):
        self._publish_wrist(np.asarray([self._wrist(q) for q in path.waypoints]), phase)
        return RecoveryWholeHandTraversal.air_move(self, path, phase)

    def ranked_targets(self):
        # Private oracle gets the actual state of this cell. FK-only A control
        # references are never mistaken for points on the physical mesh.
        self.points, self.normals = self.planner.project_pads(self.last)
        return CompletionWholeHandTraversal.ranked_targets(self)

    def refresh_surface(self):
        self.points, self.normals = self.planner.project_pads(self.last)
        if self.cell == 'A-MCC':
            _, own_normals = self.a_reference.pad_reference(self.last)
            self.normal_filter = own_normals
        else:
            self.normal_filter = self.normals.copy()
        self.bias[:] = 0.; self.offsets[:] = 0.; self.clearance[:] = .006
        self.release[:] = False; self.window.clear()
        for controller in self.controllers:
            controller.reset()

    def run(self):
        # Explicit shared initial contact acquisition, unlike development v8.
        q, _ = self.planner.fit_targets(self.last, self.points, self.normals)
        patch = dict(q=q, points=self.points.copy(), normals=self.normals.copy(), anchor=1)
        try:
            ok = self.approach(patch)
            self.events.append(dict(kind='INITIAL_MATCHED_APPROACH', time_s=float(self.data.time), success=ok))
            if self.local_only:
                while self.data.time < self.duration:
                    self.contact_step(patch['points'], patch['normals'], patch['q'], 'LOCAL_CONTACT_VALIDATION')
        except RuntimeError as error:
            self.stop = str(error)
            self.duration = self.data.time
        result = CompletionWholeHandTraversal.run(self)
        result.update(coverage_inherited=False, fresh_single_version=True, resume=None, method=self.cell,
                      wall_elapsed_s=perf_counter()-self.wall_start,
                      force_releases_scope='Entire fresh trajectory; no continuation')
        (self.out/'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        manifest = json.loads((self.out/'manifest.json').read_text())
        manifest['nominal_wrist_reference_sha256'] = sha256(self.reference_file.read_bytes()).hexdigest()
        (self.out/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        print('MATCHED_RESULT', json.dumps({k: v for k, v in result.items() if k != 'trace_chunks'}), flush=True)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cell', choices=('A-MCC', 'B-MCC'), required=True)
    parser.add_argument('--seconds', type=float, default=7000.)
    parser.add_argument('--initial', type=Path)
    parser.add_argument('--local-only', action='store_true')
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error('--seconds must be positive')
    install_asset_adapter()
    MatchedMCCTraversal(args.output, args.cell, args.seconds,
                        initial_file=args.initial, local_only=args.local_only).run()


if __name__ == '__main__':
    main()
