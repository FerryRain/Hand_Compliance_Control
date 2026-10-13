"""Regressions for actual completion and orientation-preserving approach."""
import unittest
from types import SimpleNamespace
import mujoco
import numpy as np
from Module.fr3_leap import build_full_robot, FullRobotModelConfig, ARM_HOME_Q, HAND_NATURAL_Q
from Module.i04_geometric_traversal.contracts import ExperimentConfig, WristTarget, WholeHandTarget
from Module.i04_geometric_traversal.physics import set_joint_state, pose, position_jacobian
from Module.i04_geometric_traversal.full_surface import (
    FullSurfacePlanner, endpoint_arm_offset, complete_surface, shifted_target, pad_support_vector,
    cached_target)


class FullSurfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.h = build_full_robot(FullRobotModelConfig(surface='plane', gravity_m_s2=0.))
        cls.data = mujoco.MjData(cls.h.model)
        cls.q = np.r_[ARM_HOME_Q, HAND_NATURAL_Q]

    def test_bounded_ik_reproduces_known_reachable_rotated_pose(self):
        desired_q = self.q.copy()
        desired_q[:7] += [.15, -.12, .14, -.16, .22, .10, -.20]
        set_joint_state(self.h, self.data, desired_q)
        target = WristTarget(pose(self.h, self.data))
        planner = FullSurfacePlanner(self.h, ExperimentConfig())
        solved, ep, er, _ = planner.solve(self.q, target)
        self.assertLess(ep, 1e-4)
        self.assertLess(er, 1e-3)
        self.assertTrue(np.all(solved >= planner.lower))
        self.assertTrue(np.all(solved <= planner.upper))

    def test_local_approach_preserves_wrist_orientation(self):
        set_joint_state(self.h, self.data, self.q)
        outward = np.array([0., 0., 1.])
        dq = endpoint_arm_offset(self.h, self.data, outward, .004)
        jp, jr = position_jacobian(self.h, self.data, self.h.palm_site_id)
        np.testing.assert_allclose(jp[:, :7]@dq, -outward*.004, atol=1e-5)
        self.assertLess(np.linalg.norm(jr[:, :7]@dq), 1e-4)

    def test_required_points_alone_do_not_mean_whole_surface_complete(self):
        coverage = SimpleNamespace(visited_vertices=np.array([True, False, True]),
                                   visited_goals=np.array([True, True]))
        self.assertFalse(complete_surface(coverage))
        coverage.visited_vertices[1] = True
        self.assertTrue(complete_surface(coverage))

    def test_geometric_refinement_preserves_a_b_information_boundary(self):
        target = WristTarget(np.array([.1, .2, .3, 1., 0., 0., 0.]))
        offset = np.array([0., 0., -.003])
        a = shifted_target(target, offset)
        b = shifted_target(WholeHandTarget(target, np.ones((4, 3))), offset)
        self.assertEqual(set(a.__dataclass_fields__), {'pose_world'})
        np.testing.assert_array_equal(a.pose_world, b.wrist.pose_world)
        np.testing.assert_array_equal(b.fingertip_positions_world_m, np.ones((4, 3))+offset)

    def test_oblique_pad_support_matches_actual_ellipsoid_not_box(self):
        geom = int(self.h.tip_geom_ids[0])
        axes = self.h.model.geom_size[geom]
        normal = np.array([1., 1., 1.])/np.sqrt(3)
        support = pad_support_vector(self.h.model, geom, np.eye(3), normal)
        radius = np.linalg.norm(axes*normal)
        self.assertAlmostEqual(float(support@normal), radius)
        self.assertAlmostEqual(float(np.sum((support/axes)**2)), 1.)
        self.assertLess(radius, float(axes@abs(normal)))

    def test_actual_ellipsoid_cache_is_not_corrected_twice(self):
        wrist = np.array([.1, .2, .3, 1., 0., 0., 0.])
        witnesses = dict(pose=wrist[None], tips=np.ones((1, 4, 3)), vertex=[0], finger_witness=[0])
        oracle = SimpleNamespace(correct_cached_target=lambda *args: self.fail('Double correction'))
        target = cached_target(oracle, witnesses, 0, 'actual_ellipsoid_vector.v1')
        np.testing.assert_array_equal(target.wrist.pose_world, wrist)
        np.testing.assert_array_equal(target.fingertip_positions_world_m, witnesses['tips'][0])
        with self.assertRaises(ValueError): cached_target(oracle, witnesses, 0, 'unknown')

    def test_planning_clearance_does_not_modify_physical_contact_margins(self):
        margins = self.h.model.geom_margin.copy()
        planner = FullSurfacePlanner(self.h, ExperimentConfig())
        planner.collision_reason(self.q)
        np.testing.assert_array_equal(self.h.model.geom_margin, margins)


if __name__ == '__main__': unittest.main()
