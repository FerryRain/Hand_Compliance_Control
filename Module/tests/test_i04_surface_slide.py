"""Contact-first sliding regressions using sampled paths and native robot state.

The planning test never executes its IK states on the physical plant.  It checks
that both information boundaries produce an adjacent, usable surface path while
the acquired robot state and measured coverage remain unchanged.
"""
from pathlib import Path
from hashlib import sha256
import json
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.sparse import diags
import trimesh

from Module.fr3_leap import (
    ARM_HOME_Q, HAND_NATURAL_Q, FullRobotModelConfig, build_full_robot,
)
from Module.i04_geometric_traversal.contracts import (
    ExperimentConfig, WholeHandTarget, WristTarget,
)
from Module.i04_geometric_traversal.fixtures import MUSTARD
from Module.i04_geometric_traversal.full_surface import acquire, build_scene
from Module.i04_geometric_traversal.oracle import GTCoverageOracle, MeasuredCoverage
from Module.i04_geometric_traversal.planning import SharedGeometricTrajectoryPlanner
from Module.i04_geometric_traversal.physics import (
    contact_measurement, joint_state, pose, position_jacobian, quaternion,
    rotation, set_joint_state, tip_positions,
)
from Module.i04_geometric_traversal.surface_slide import (
    GraceMCC, LocalProgressBudget, SurfaceStroke, SurfaceStrokePlanner, arm_force_release,
    nearby_recontact_target, reanchor_transition, run,
)


class SurfaceStrokeSamplingTests(unittest.TestCase):
    def setUp(self):
        self.r0 = Rotation.from_euler('xyz', [.2, -.4, .3])
        self.world_normal = np.array([1., 2., 3.]) / np.sqrt(14.)
        rotations = [Rotation.from_rotvec([0., 0., angle]) * self.r0
                     for angle in (0., .7, 1.4)]
        self.q = np.vstack((np.zeros(23), np.linspace(-.1, .2, 23),
                            np.linspace(.3, -.1, 23)))
        self.stroke = SurfaceStroke(
            q=self.q.copy(), positions=np.array([[0., 0., 0.], [.02, 0., 0.], [.05, 0., 0.]]),
            # Deliberately unrelated private geometry normals: these must not
            # become the runtime force-control direction.
            normals=np.tile([0., 1., 0.], (3, 1)),
            wrist=np.array([np.r_[position, quaternion(r)] for position, r in
                            zip(([0., 0., 0.], [.02, 0., 0.], [.05, 0., 0.]), rotations)]),
            distances=np.array([0., .02, .05]), finger=0, goal=42,
            normal_in_wrist=self.r0.inv().apply(self.world_normal),
        )

    def test_nonuniform_arclength_interpolation_is_continuous_at_knots(self):
        np.testing.assert_allclose(self.stroke.sample(.01)[0], .5 * (self.q[0] + self.q[1]))
        np.testing.assert_allclose(self.stroke.sample(.035)[0], .5 * (self.q[1] + self.q[2]))
        for distance, expected in zip(self.stroke.distances, self.q):
            np.testing.assert_allclose(self.stroke.sample(distance)[0], expected, atol=1e-14)
        left, normal_left = self.stroke.sample(.02 - 1e-9)
        right, normal_right = self.stroke.sample(.02 + 1e-9)
        self.assertLess(np.linalg.norm(right - left), 1e-6)
        self.assertLess(np.linalg.norm(normal_right - normal_left), 1e-6)
        np.testing.assert_array_equal(self.stroke.sample(-1.)[0], self.q[0])
        np.testing.assert_array_equal(self.stroke.sample(1.)[0], self.q[-1])

    def test_normal_is_causal_measured_normal_rotated_by_exposed_wrist(self):
        for distance, angle in ((0., 0.), (.01, .35), (.02, .7), (.035, 1.05), (.05, 1.4)):
            _, normal = self.stroke.sample(distance)
            expected = Rotation.from_rotvec([0., 0., angle]).apply(self.world_normal)
            np.testing.assert_allclose(normal, expected, atol=1e-12)
            self.assertAlmostEqual(np.linalg.norm(normal), 1.)
        self.stroke.normals[:] = np.nan
        self.stroke.positions[:] = 100.
        _, normal = self.stroke.sample(.01)
        np.testing.assert_allclose(normal, Rotation.from_rotvec([0., 0., .35]).apply(self.world_normal), atol=1e-12)

    def test_sampled_commands_do_not_alias_saved_trajectory(self):
        saved = self.stroke.q.copy()
        q, _ = self.stroke.sample(.01)
        q[:7] += .1  # Normal trim must not progressively deform the path.
        np.testing.assert_array_equal(self.stroke.q, saved)


class ContinuationPolicyTests(unittest.TestCase):
    def test_weaker_real_contact_can_continue_after_strongest_anchor_is_blocked(self):
        planner = object.__new__(SurfaceStrokePlanner)
        expected = SimpleNamespace(finger=2)
        forces = np.array([3., 100., 1., 50.])
        active = np.array([True, False, True, False])
        with patch.object(planner, '_plan_finger', side_effect=[
                (None, [{'reason': 'COLLISION_OR_CLEARANCE'}]), (expected, [])]) as propose:
            result, failures = planner.plan(None, None, forces, None, None, active)
        self.assertIs(result, expected)
        self.assertEqual([call.args[-1] for call in propose.call_args_list], [0, 2])
        self.assertEqual(failures, [{'reason': 'COLLISION_OR_CLEARANCE', 'finger': 0}])

    def test_short_bridge_waits_until_all_anchors_have_failed_long_continuation(self):
        planner = object.__new__(SurfaceStrokePlanner)
        short = SimpleNamespace(finger=0, length=.012,
            positions=np.array([[0., 0., 0.], [.012, 0., 0.]]))
        normal = SimpleNamespace(finger=2, length=.016)
        def offer(_data, _coverage, _contacts, _normals, finger):
            if finger == 0:
                planner.short_prefixes.append(short)
                return None, [{'reason': 'SHORT_PREFIX'}]
            return normal, []
        with patch.object(planner, '_plan_finger', side_effect=offer) as propose:
            result, _ = planner.plan(None, None, np.array([2., 0., 1., 0.]), None, None,
                                     np.array([True, False, True, False]))
        self.assertIs(result, normal)
        self.assertEqual([call.args[-1] for call in propose.call_args_list], [0, 2])

    def test_only_after_all_long_routes_fail_is_longest_checked_short_bridge_selected(self):
        planner = object.__new__(SurfaceStrokePlanner)
        candidates = [SimpleNamespace(finger=i, length=length,
            positions=np.array([[0., 0., 0.], [length, 0., 0.]])) for i, length in ((0, .006), (2, .012))]
        def offer(_data, _coverage, _contacts, _normals, finger):
            planner.short_prefixes.extend(c for c in candidates if c.finger == finger)
            return None, [{'reason': 'COLLISION_OR_CLEARANCE'}]
        with patch.object(planner, '_plan_finger', side_effect=offer) as propose:
            result, reasons = planner.plan(None, None, np.array([2., 0., 1., 0.]), None, None,
                                          np.array([True, False, True, False]))
        self.assertIs(result, candidates[1])
        self.assertEqual(propose.call_count, 2)
        self.assertEqual(reasons[-1]['reason'], 'SHORT_CONTACT_BRIDGE')
        np.testing.assert_array_equal(planner.heading, [1., 0., 0.])


class LocalProgressBudgetTests(unittest.TestCase):
    def test_aborted_and_completed_zero_gain_strokes_share_one_bounded_budget(self):
        budget = LocalProgressBudget()
        budget.observe(100, .002, local=True)
        for index, reason in enumerate(('SELECT', 'LIFT_PLAN', 'LOCAL_SEARCH', 'SELECT', 'SELECT', 'SELECT')):
            budget.begin(index, 100)
            result = budget.finish(100, reason)
            self.assertEqual(result['new_vertices'], 0)
            self.assertEqual(result['empty_strokes'], index+1)
            self.assertEqual(budget.exhausted, index == 5)
            self.assertIsNone(budget.finish(100, reason))  # End each attempt once.

    def test_actual_new_vertex_resets_budget_but_restarting_a_stroke_does_not(self):
        budget = LocalProgressBudget()
        budget.observe(100, .002, local=True)
        for index in range(5):
            budget.begin(index, 100); budget.finish(100, 'SELECT')
        budget.begin(5, 100)
        self.assertEqual(budget.empty_strokes, 5)
        budget.observe(101, .002, local=True)
        result = budget.finish(101, 'TRACKING_REPLAN')
        self.assertEqual(result['new_vertices'], 1)
        self.assertEqual(budget.empty_strokes, 0)
        self.assertEqual(budget.local_s, 0.)

    def test_airtime_does_not_consume_local_search_budget_and_commit_gives_new_region_a_chance(self):
        budget = LocalProgressBudget()
        budget.observe(100, .002, local=True)
        budget.observe(100, 100., local=False)
        self.assertFalse(budget.exhausted)
        budget.observe(100, 45., local=True)
        self.assertTrue(budget.exhausted)
        budget.transition_committed()
        self.assertFalse(budget.exhausted)
        self.assertEqual(budget.covered_count, 100)


class LocalSearchRunnerSchedulingTests(unittest.TestCase):
    def test_found_stroke_is_committed_once_from_same_state_and_cache_bytes_are_frozen(self):
        # Scheduling isolation: use native FK but hold the plant still and
        # supply stable contacts. This is a state-machine test, not physics
        # evidence and never contributes to an experiment's coverage ledger.
        h = build_full_robot(FullRobotModelConfig(surface='plane', gravity_m_s2=0.))
        data = mujoco.MjData(h.model)
        q = np.r_[ARM_HOME_Q, HAND_NATURAL_Q]
        set_joint_state(h, data, q)
        data.ctrl[h.arm_actuator_ids] = q[:7]
        data.ctrl[h.hand_actuator_ids] = q[7:]
        graph = SimpleNamespace(vertices_m=np.array([[0., 0., 0.], [.02, 0., 0.], [0., .02, 0.]]),
            normals=np.tile([0., 0., 1.], (3, 1)), faces=np.array([[0, 1, 2]]),
            required_vertices=np.array([0]), required_goal_count=1)
        coverage = SimpleNamespace(visited_vertices=np.zeros(3, dtype=bool),
            visited_goals=np.zeros(1, dtype=bool), area_fraction=0., update=lambda *args: None)
        oracle = SimpleNamespace(initial_normal=np.array([0., 0., 1.]))
        normals = np.tile([0., 0., 1.], (4, 1))
        measurements = (np.full(4, 2.), tip_positions(h, data), normals, np.ones(4, dtype=bool), [], 0.)
        candidate = SurfaceStroke(q=np.vstack((q, q)),
            positions=np.array([[0., 0., 0.], [.02, 0., 0.]]), normals=normals[:2],
            wrist=np.tile(pose(h, data), (2, 1)), distances=np.array([0., .02]),
            finger=0, goal=1, normal_in_wrist=rotation(pose(h, data)[3:]).inv().apply(normals[0]))
        scratch_root = Path(__file__).resolve().parents[2] / '.runtime_tmp'; scratch_root.mkdir(exist_ok=True)
        with TemporaryDirectory(prefix='schedule_test_', dir=scratch_root) as directory:
            self.assertEqual(Path(directory).resolve().parent, scratch_root.resolve())
            cache = Path(directory) / 'custom_cache.npz'
            np.savez(cache, vertex=np.array([0]), q=q[None], pose=pose(h, data)[None],
                     tips=tip_positions(h, data)[None])
            expected_cache = cache.read_bytes()
            output = Path(directory) / 'run'
            prefix = 'Module.i04_geometric_traversal.surface_slide.'
            with patch(prefix+'build_scene', return_value=(h, graph, q, {})), \
                 patch(prefix+'acquire', return_value=(data, {})), \
                 patch(prefix+'FullSurfaceOracle', return_value=oracle), \
                 patch(prefix+'MeasuredCoverage', return_value=coverage), \
                 patch(prefix+'contact_measurement', return_value=measurements), \
                 patch(prefix+'arm_self_contacts', return_value=[]), \
                 patch(prefix+'mujoco.mj_step'), \
                 patch(prefix+'SlidingSafetyPlanner.command_collision_reason', return_value=None), \
                 patch.object(SurfaceStrokePlanner, 'plan', side_effect=[
                     (None, [{'reason': 'LOCAL_BLOCKED'}]), (candidate, [])]) as plan:
                summary = run(output, budget_s=2.004, witness_path=cache)
            self.assertEqual(plan.call_count, 2)
            self.assertEqual(summary['stroke_count'], 1)
            events = json.loads((output/'events.json').read_text(encoding='utf-8'))
            found = next(e for e in events if e['kind'] == 'LOCAL_SEARCH_SUPPORT_FOUND')
            started = next(e for e in events if e['kind'] == 'SURFACE_STROKE')
            self.assertEqual(found['time_s'], started['time_s'])
            with np.load(output/'stroke_0000.npz') as saved:
                np.testing.assert_array_equal(saved['q'], candidate.q)
            manifest = json.loads((output/'manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(manifest['protocol'], 'i04-mustard-active-finger-sliding.development.v2.2')
            self.assertEqual(manifest['witness_sha256'], sha256(expected_cache).hexdigest())
            self.assertEqual((output/manifest['witness_snapshot']).read_bytes(), expected_cache)
            self.assertEqual(summary['covered_vertex_count'], 0)
            # An absent optional default cache must not declare a snapshot
            # that an independent auditor is then required to find.
            default_cache = Path(__file__).resolve().parents[2] / 'Module/generated/mustard_true_ellipsoid_geometry07/pose_witnesses.npz'
            original_exists = Path.exists
            def exists(path):
                return False if path == default_cache else original_exists(path)
            absent_output = Path(directory) / 'no_default_cache'
            with patch(prefix+'build_scene', return_value=(h, graph, q, {})), \
                 patch(prefix+'acquire', return_value=(data, {})), \
                 patch(prefix+'FullSurfaceOracle', return_value=oracle), \
                 patch(prefix+'MeasuredCoverage', return_value=coverage), \
                 patch(prefix+'contact_measurement', return_value=measurements), \
                 patch(prefix+'arm_self_contacts', return_value=[]), \
                 patch(prefix+'mujoco.mj_step'), \
                 patch(prefix+'SlidingSafetyPlanner.command_collision_reason', return_value=None), \
                 patch.object(SurfaceStrokePlanner, 'plan', return_value=(None, [])), \
                 patch.object(Path, 'exists', exists):
                run(absent_output, budget_s=.01)
            missing_manifest = json.loads((absent_output/'manifest.json').read_text(encoding='utf-8'))
            self.assertNotIn('witness_snapshot', missing_manifest)
            self.assertIsNone(missing_manifest['witness_sha256'])


class GraceMCCNativeJacobianTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.h = build_full_robot(FullRobotModelConfig(surface='plane', gravity_m_s2=0.))
        cls.q = np.r_[ARM_HOME_Q, HAND_NATURAL_Q]

    def setUp(self):
        self.data = mujoco.MjData(self.h.model)
        set_joint_state(self.h, self.data, self.q)
        self.mcc = GraceMCC(self.h, .002)
        self.normal = np.zeros((4, 3)); self.normal[0] = [0., 0., 1.]
        self.active = np.array([True, False, False, False])
        self.forces = np.array([.5, 0., 0., 0.])

    def contact(self):
        return self.mcc.command(self.data, self.q, self.forces, self.normal, self.active, 2.)

    def absent(self):
        forces, normals, active = np.zeros(4), np.zeros((4, 3)), np.zeros(4, dtype=bool)
        command = self.mcc.command(self.data, self.q, forces, normals, active, 2.)
        # Continuing control must not mutate the actual sensor evidence.
        np.testing.assert_array_equal(forces, np.zeros(4))
        np.testing.assert_array_equal(normals, np.zeros((4, 3)))
        self.assertFalse(active.any())
        return command

    def test_short_dropout_retains_preload_without_inventing_sensor_contact(self):
        self.contact()
        preload = self.mcc.controllers[0].state.offset_m
        for _ in range(45):  # 90 ms: inside the bounded grace interval.
            command = self.absent()
        self.assertGreater(self.mcc.controllers[0].state.offset_m, preload)
        jp, _ = position_jacobian(self.h, self.data, self.h.tip_site_ids[0])
        self.assertLess(float(self.normal[0] @ jp @ (command - self.q)), 0.)
        np.testing.assert_array_equal(self.mcc.normal_history[0], self.normal[0])
        self.assertFalse(self.mcc.was_active[1:].any())
        np.testing.assert_array_equal(command[:7], self.q[:7])

    def test_long_dropout_expires_and_recontact_uses_only_new_normal(self):
        for _ in range(10):
            self.contact()
        for _ in range(51):  # 102 ms: beyond grace; old preload must expire.
            command = self.absent()
        np.testing.assert_array_equal(command, self.q)
        self.assertEqual(self.mcc.controllers[0].state.offset_m, 0.)
        self.assertFalse(self.mcc.was_active.any())
        self.normal[0] = [1., 0., 0.]
        resumed = self.contact()
        fresh = GraceMCC(self.h, .002).command(
            self.data, self.q, self.forces, self.normal, self.active, 2.)
        np.testing.assert_allclose(resumed, fresh, atol=1e-14)

    def test_never_contacted_fingers_do_not_search_and_excess_force_releases(self):
        for _ in range(60):
            np.testing.assert_array_equal(self.absent(), self.q)
        self.forces[0] = 10.
        command = self.contact()
        jp, _ = position_jacobian(self.h, self.data, self.h.tip_site_ids[0])
        self.assertGreater(float(self.normal[0] @ jp @ (command - self.q)), 0.)


class ArmForceReleaseNativeJacobianTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.h = build_full_robot(FullRobotModelConfig(surface='plane', gravity_m_s2=0.))
        cls.q = np.r_[ARM_HOME_Q, HAND_NATURAL_Q]

    def setUp(self):
        self.data = mujoco.MjData(self.h.model)
        set_joint_state(self.h, self.data, self.q)
        self.previous = self.q.copy()
        self.previous[:7] += np.linspace(-.005, .005, 7)
        self.forces = np.array([9., 5., 0., 0.])
        self.normals = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 0.], [0., 0., 0.]])

    def test_loaded_wrist_withdraws_along_measured_normal_without_rotating(self):
        arm = arm_force_release(self.h, self.data, self.previous, self.forces, self.normals)
        delta = arm - self.previous[:7]
        jp, jr = position_jacobian(self.h, self.data, self.h.palm_site_id)
        expected_normal = np.array([9., 5., 0.]) / np.sqrt(106.)
        np.testing.assert_allclose(jp[:, :7] @ delta, expected_normal * .030 * .002, atol=2e-7)
        self.assertLess(np.linalg.norm(jr[:, :7] @ delta), 1e-6)
        self.assertGreater(float(expected_normal @ jp[:, :7] @ delta), 0.)
        np.testing.assert_array_equal(joint_state(self.h, self.data), self.q)

    def test_repeated_release_accumulates_applied_reference_despite_measured_lag(self):
        before = self.previous.copy()
        command = self.previous.copy()
        first = arm_force_release(self.h, self.data, command, self.forces, self.normals)
        increment = first - command[:7]
        for _ in range(30):
            command[:7] = arm_force_release(self.h, self.data, command, self.forces, self.normals)
        # The plant is deliberately held still to reproduce servo lag. A
        # measured-q-rooted correction would stop advancing after one call.
        np.testing.assert_allclose(command[:7], before[:7] + 30 * increment, atol=1e-13)
        np.testing.assert_array_equal(command[7:], before[7:])
        np.testing.assert_array_equal(self.previous, before)
        np.testing.assert_array_equal(joint_state(self.h, self.data), self.q)

    def test_no_loaded_fingers_preserve_reference_and_return_independent_array(self):
        forces = np.array([3.99, 0., 1., 0.])
        original = self.previous.copy()
        arm = arm_force_release(self.h, self.data, self.previous, forces, self.normals)
        np.testing.assert_array_equal(arm, original[:7])
        arm[:] = 0.
        np.testing.assert_array_equal(self.previous, original)
        np.testing.assert_array_equal(joint_state(self.h, self.data), self.q)


class NearbyRecontactNativeGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.h = build_full_robot(FullRobotModelConfig(surface='plane', gravity_m_s2=0.))

    def setUp(self):
        self.data = mujoco.MjData(self.h.model)
        # A changed physical posture catches accidental use of acquisition or
        # natural-hand offsets instead of the present measured pad geometry.
        self.q = np.r_[ARM_HOME_Q, HAND_NATURAL_Q].copy()
        self.q[:7] += [.05, -.03, .07, -.04, .06, .02, -.05]
        self.q[7:] += np.linspace(-.025, .035, 16)
        set_joint_state(self.h, self.data, self.q)
        self.current = pose(self.h, self.data)
        self.r0 = rotation(self.current[3:])
        self.finger = 1
        self.geom = int(self.h.tip_geom_ids[self.finger])
        self.frame = self.data.geom_xmat[self.geom].reshape(3, 3).copy()
        self.outward = self.frame @ (np.array([1., 2., 3.]) / np.sqrt(14.))
        tangent = np.cross(self.outward, np.eye(3)[np.argmin(abs(self.outward))])
        self.tangent = tangent / np.linalg.norm(tangent)
        self.normal = Rotation.from_rotvec(.35 * self.tangent).apply(self.outward)
        self.graph = SimpleNamespace(vertices_m=np.array([[.025, -.01, .08]]),
                                     normals=self.normal[None])

    def target(self):
        return nearby_recontact_target(self.h, self.graph, self.data, 0,
                                       self.finger, self.outward)

    def test_reorientation_uses_shortest_normal_turn_and_retains_current_hand_posture(self):
        full = self.target()
        final_r = rotation(full.wrist.pose_world[3:])
        turn = final_r * self.r0.inv()
        self.assertAlmostEqual(turn.magnitude(), .35, places=12)
        np.testing.assert_allclose(turn.apply(self.outward), self.normal, atol=1e-12)
        # The minimum rotation leaves its tangent axis fixed; an added wrist
        # roll could align the normal yet fail this condition.
        np.testing.assert_allclose(turn.apply(self.tangent), self.tangent, atol=1e-12)
        before_local = self.r0.inv().apply(tip_positions(self.h, self.data) - self.current[:3])
        after_local = final_r.inv().apply(full.fingertip_positions_world_m - full.wrist.pose_world[:3])
        np.testing.assert_allclose(after_local, before_local, atol=1e-12)
        np.testing.assert_array_equal(joint_state(self.h, self.data), self.q)

    def test_oblique_ellipsoid_support_lands_at_requested_surface_clearance(self):
        full = self.target()
        final_r = rotation(full.wrist.pose_world[3:])
        local_center = self.r0.inv().apply(self.data.geom_xpos[self.geom] - self.current[:3])
        center = full.wrist.pose_world[:3] + final_r.apply(local_center)
        final_frame = (final_r * self.r0.inv()).as_matrix() @ self.frame
        size = self.h.model.geom_size[self.geom]
        self.assertEqual(self.h.model.geom_type[self.geom], mujoco.mjtGeom.mjGEOM_ELLIPSOID)
        local_n = final_frame.T @ self.normal
        # Analytic ellipsoid support, independent of the runtime helper.
        surface_support = final_frame @ (size ** 2 * local_n / np.linalg.norm(size * local_n))
        requested_point = self.h.object_position_m + self.graph.vertices_m[0]
        np.testing.assert_allclose(center - surface_support - requested_point,
                                   .002 * self.normal, atol=1e-12)
        self.assertGreater(np.linalg.norm(surface_support - self.normal * np.dot(surface_support, self.normal)), 1e-4)

    def test_equal_normal_keeps_roll_and_exposure_contains_only_cartesian_geometry(self):
        self.graph.normals = self.outward[None]
        full = self.target()
        np.testing.assert_allclose(rotation(full.wrist.pose_world[3:]).as_matrix(),
                                   self.r0.as_matrix(), atol=1e-12)
        a = GTCoverageOracle.expose('A-MCC', full)
        b = GTCoverageOracle.expose('B-MCC', full)
        self.assertEqual(set(a.__dataclass_fields__), {'pose_world'})
        self.assertEqual(set(b.__dataclass_fields__), {'wrist', 'fingertip_positions_world_m'})
        np.testing.assert_array_equal(a.pose_world, b.wrist.pose_world)

    def test_reanchor_uses_live_measured_root_and_preserves_old_cartesian_destination(self):
        goal = self.q.copy(); goal[:7] += [.02, -.01, .01, -.02, .01, 0., -.01]
        goal[7:] += .01
        scratch = mujoco.MjData(self.h.model)
        set_joint_state(self.h, scratch, goal)
        expected_pose, expected_tips = pose(self.h, scratch), tip_positions(self.h, scratch)
        physical_qpos, physical_ctrl = self.data.qpos.copy(), self.data.ctrl.copy()
        old_path = SimpleNamespace(waypoints=np.vstack((self.q - .03, goal)))
        for cell, kind in (('A-MCC', WristTarget), ('B-MCC', WholeHandTarget)):
            planner = SimpleNamespace(h=self.h, plan=lambda *args, **kwargs: None)
            with patch.object(planner, 'plan', return_value=('fresh-plan', None)) as plan:
                result = reanchor_transition(planner, self.data, old_path, cell, allow_rrt=True)
            self.assertEqual(result, ('fresh-plan', None))
            self.assertIs(plan.call_args.args[0], self.data)
            self.assertTrue(plan.call_args.kwargs['allow_rrt'])
            target = plan.call_args.args[1]
            self.assertIsInstance(target, kind)
            target_pose = target.pose_world if cell == 'A-MCC' else target.wrist.pose_world
            np.testing.assert_allclose(target_pose, expected_pose, atol=1e-13)
            if cell == 'B-MCC':
                np.testing.assert_allclose(target.fingertip_positions_world_m, expected_tips, atol=1e-13)
            np.testing.assert_array_equal(self.data.qpos, physical_qpos)
            np.testing.assert_array_equal(self.data.ctrl, physical_ctrl)

    def test_covered_region_can_bridge_toward_distant_unmeasured_frontier(self):
        # Isolate route selection from IK: native FK supplies the starting
        # hand, while a deliberately simple line graph has only far goals.
        count = 81
        vertices = np.column_stack((np.arange(count) * .005, np.zeros(count), np.zeros(count)))
        vertices += self.data.geom_xpos[self.geom] - self.h.object_position_m
        graph = SimpleNamespace(vertices_m=vertices, normals=np.tile([0., 0., 1.], (count, 1)),
            adjacency=diags([np.full(count - 1, .005), np.full(count - 1, .005)], [-1, 1], format='csr'),
            oriented_nearest_vertices=lambda *args: (np.array([0]), np.array([0.])))
        coverage = SimpleNamespace(visited_vertices=np.arange(count) < 51)
        saved = coverage.visited_vertices.copy()
        planner = SurfaceStrokePlanner(self.h, graph, ExperimentConfig(), 'B-MCC')
        points = np.tile(vertices[0] + self.h.object_position_m, (4, 1))
        normals = np.tile([0., 0., 1.], (4, 1))
        def sampled(_graph, path):
            distance = np.linalg.norm(path[-1] - path[0])
            positions = np.linspace(path[0], path[-1], int(np.ceil(distance / .002)) + 1)
            return positions, np.tile([0., 0., 1.], (len(positions), 1))
        def solved(root, target, **kwargs):
            return root.copy(), 0., 0., 0.
        with patch('Module.i04_geometric_traversal.surface_slide.resample_surface', side_effect=sampled), \
                patch.object(SharedGeometricTrajectoryPlanner, 'solve',
                    side_effect=lambda _planner, root, target, **kwargs: solved(root, target, **kwargs)), \
                patch.object(planner.planner, 'solve', side_effect=solved), \
                patch.object(planner.planner, 'edge_free', return_value=True):
            stroke, failures = planner._plan_finger(self.data, coverage, points, normals, self.finger)
        self.assertIsNotNone(stroke, failures)
        self.assertGreaterEqual(stroke.goal, 51)
        self.assertGreater(stroke.length, .15)
        self.assertLessEqual(stroke.length, .18 + 1e-12)
        np.testing.assert_array_equal(coverage.visited_vertices, saved)

    def test_short_fallback_keeps_only_checked_prefix_and_never_admits_sub_five_mm(self):
        # Native initial wrist/hand geometry, with a controlled IK/edge
        # boundary: later mesh points are deliberately blocked. This exercises
        # prefix retention itself; native collision rules have separate tests.
        count = 81
        vertices = np.column_stack((np.arange(count)*.005, np.zeros(count), np.zeros(count)))
        vertices += self.data.geom_xpos[self.geom]-self.h.object_position_m
        graph = SimpleNamespace(vertices_m=vertices, normals=np.tile([0., 0., 1.], (count, 1)),
            adjacency=diags([np.full(count-1, .005), np.full(count-1, .005)], [-1, 1], format='csr'),
            oriented_nearest_vertices=lambda *args: (np.array([0]), np.array([0.])))
        coverage = SimpleNamespace(visited_vertices=np.zeros(count, dtype=bool))
        contacts = np.tile(vertices[0]+self.h.object_position_m, (4, 1))
        normals = np.tile([0., 0., 1.], (4, 1))
        active = np.zeros(4, dtype=bool); active[self.finger] = True
        forces = active.astype(float)
        before = self.data.qpos.copy()
        def sampled(_graph, path):
            distance = np.linalg.norm(path[-1]-path[0])
            points = np.linspace(path[0], path[-1], int(np.ceil(distance/.002))+1)
            return points, np.tile([0., 0., 1.], (len(points), 1))
        def solved(root, _target, **kwargs):
            q = root.copy(); q[0] += .002
            return q, 0., 0., 0.
        for boundary, expected in ((.012, True), (.004, False)):
            with self.subTest(checked_joint_prefix=boundary):
                planner = SurfaceStrokePlanner(self.h, graph, ExperimentConfig(), 'B-MCC')
                def edge(_a, b):
                    return b[0]-self.q[0] <= boundary+1e-12
                with patch('Module.i04_geometric_traversal.surface_slide.resample_surface', side_effect=sampled), \
                     patch.object(SharedGeometricTrajectoryPlanner, 'solve',
                         side_effect=lambda _planner, root, target, **kwargs: solved(root, target, **kwargs)), \
                     patch.object(planner.planner, 'solve', side_effect=solved), \
                     patch.object(planner.planner, 'edge_free', side_effect=edge):
                    stroke, failures = planner.plan(self.data, coverage, forces, contacts, normals, active)
                self.assertEqual(stroke is not None, expected, failures)
                if expected:
                    self.assertGreaterEqual(stroke.length, .005)
                    self.assertLess(stroke.length, .015)
                    np.testing.assert_array_equal(stroke.q[0], self.q)
                    self.assertLessEqual(float(np.max(stroke.q[:, 0]-self.q[0])), boundary+1e-12)
                np.testing.assert_array_equal(self.data.qpos, before)
                self.assertFalse(coverage.visited_vertices.any())


@unittest.skipUnless((MUSTARD / 'visual_scaled.obj').exists(), 'Mustard physical asset is required')
class MustardSurfaceContinuationTests(unittest.TestCase):
    def test_dense_a_b_plans_preserve_native_state_coverage_and_contact_model(self):
        scratch_root = Path(__file__).resolve().parents[2] / '.runtime_tmp'
        scratch_root.mkdir(exist_ok=True)
        with TemporaryDirectory(prefix='i04_sliding_test_', dir=scratch_root) as directory:
            # Verify the scope before TemporaryDirectory's recursive cleanup.
            self.assertEqual(Path(directory).resolve().parent, scratch_root.resolve())
            h, graph, initial_q, _ = build_scene(directory)
            data, _ = acquire(h, graph, initial_q, seed=7)
            config = ExperimentConfig(seed=7)
            forces, points, normals, active, unsafe, _ = contact_measurement(h, data, .2)
            self.assertFalse(unsafe)
            self.assertTrue(active.any())
            coverage = MeasuredCoverage(graph, config)
            coverage.update(points - h.object_position_m, normals, active)
            expected_q, expected_wrist = joint_state(h, data), pose(h, data)
            physical = [data.qpos.copy(), data.qvel.copy(), data.ctrl.copy(), data.time]
            margins = h.model.geom_margin.copy()
            covered = coverage.visited_vertices.copy()
            for cell, target_class in (('A-MCC', WristTarget), ('B-MCC', WholeHandTarget)):
                with self.subTest(cell=cell):
                    sp = SurfaceStrokePlanner(h, graph, config, cell)
                    # Both local DLS and bounded least-squares fallback use
                    # the same geometric contract. Observe both while leaving
                    # their real native kinematic calculations intact.
                    with patch.object(SharedGeometricTrajectoryPlanner, 'solve',
                                      wraps=SharedGeometricTrajectoryPlanner.solve) as local_solve, \
                            patch.object(sp.planner, 'solve', wraps=sp.planner.solve) as bounded_solve:
                        stroke, failures = sp.plan(data, coverage, forces, points, normals,
                                                   active & (forces <= 8.))
                    self.assertIsNotNone(stroke, failures)
                    targets = ([call.args[2] for call in local_solve.call_args_list]
                               + [call.args[1] for call in bounded_solve.call_args_list])
                    self.assertTrue(targets)
                    self.assertTrue(all(type(target) is target_class for target in targets))
                    np.testing.assert_array_equal(stroke.q[0], expected_q)
                    np.testing.assert_array_equal(stroke.wrist[0], expected_wrist)
                    self.assertGreaterEqual(stroke.length, .025)
                    self.assertGreater(len(stroke.q), 10)
                    self.assertTrue(np.all(np.diff(stroke.distances) > 0.))
                    self.assertLessEqual(float(np.max(np.linalg.norm(np.diff(stroke.positions, axis=0), axis=1))), .003)
                    _, residual, _ = trimesh.proximity.closest_point(graph.mesh, stroke.positions)
                    self.assertLess(float(np.max(residual)), 1e-9)
                    # Independently revisit representative continuation edges
                    # and actual FK rather than merely trusting saved targets.
                    indices = np.unique(np.linspace(1, len(stroke.q) - 1, 8).astype(int))
                    for index in indices:
                        self.assertTrue(sp.planner.edge_free(stroke.q[index - 1], stroke.q[index]))
                        set_joint_state(h, sp.planner.scratch, stroke.q[index])
                        self.assertLess(np.linalg.norm(pose(h, sp.planner.scratch)[:3] - stroke.wrist[index, :3]), .0008)
                    np.testing.assert_array_equal(data.qpos, physical[0])
                    np.testing.assert_array_equal(data.qvel, physical[1])
                    np.testing.assert_array_equal(data.ctrl, physical[2])
                    self.assertEqual(data.time, physical[3])
                    np.testing.assert_array_equal(h.model.geom_margin, margins)
                    np.testing.assert_array_equal(coverage.visited_vertices, covered)
                    measured = contact_measurement(h, data, .2)
                    np.testing.assert_array_equal(measured[0], forces)
                    np.testing.assert_array_equal(measured[3], active)


if __name__ == '__main__':
    unittest.main()
