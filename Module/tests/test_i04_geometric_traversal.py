"""Method boundaries, physical coverage authority, and fresh-root semantics."""
import unittest
import numpy as np
import trimesh

from Module.i04_geometric_traversal.contracts import ExperimentConfig, WristTarget, WholeHandTarget
from Module.i04_geometric_traversal.oracle import GTCoverageOracle, MeasuredCoverage
from Module.i04_geometric_traversal.planning import GeometricPlan, GenericTrajectoryExecutor
from Module.i04_oracle_next_point.surface_graph import BunnySurfaceGraph


class NewI04ContractTests(unittest.TestCase):
    def setUp(self):
        self.pose = np.array([0.1, 0.2, 0.3, 1, 0, 0, 0])
        self.latent = WholeHandTarget(WristTarget(self.pose), np.zeros((4, 3)))

    def test_a_strips_finger_geometry_and_has_no_role_answers(self):
        a = GTCoverageOracle.expose('A-MCC', self.latent)
        self.assertIsInstance(a, WristTarget)
        self.assertEqual(set(a.__dataclass_fields__), {'pose_world'})
        self.assertFalse(a.pose_world.flags.writeable)
        np.testing.assert_array_equal(a.pose_world, self.latent.wrist.pose_world)

    def test_b_adds_geometry_only(self):
        b = GTCoverageOracle.expose('B-MCC', self.latent)
        self.assertEqual(set(b.__dataclass_fields__), {'wrist', 'fingertip_positions_world_m'})
        with self.assertRaises(ValueError):
            GTCoverageOracle.expose('A-DPRef', self.latent)

    def test_bad_pose_rejected(self):
        with self.assertRaises(ValueError):
            WristTarget([0, 0, 0, 2, 0, 0, 0])
        with self.assertRaises(ValueError):
            WholeHandTarget(WristTarget(self.pose), np.full((4, 3), np.nan))

    def test_executor_refuses_stale_root_but_does_not_veto_zero_contact(self):
        config = ExperimentConfig()
        q = np.zeros(23)
        plan = GeometricPlan(q, q + 0.01, 1.0, self.pose, self.pose,
                             0, 0, 0, 0, 'unit-plan')
        executor = GenericTrajectoryExecutor(config)
        with self.assertRaisesRegex(ValueError, 'STALE_ROOT'):
            executor.commit(plan, q + 0.001, 0.0)
        executor.commit(plan, q, 0.0)
        command = executor.step(0.002, q)
        self.assertTrue(np.any(command > 0))
        self.assertEqual(executor.contact_task_vetoes, 0)
        with self.assertRaises(ValueError):
            executor.step(0.002, q)


class PhysicalCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mesh = trimesh.creation.icosphere(subdivisions=1, radius=0.03)
        cls.graph = BunnySurfaceGraph(mesh, coverage_radius_m=0.015)

    def test_planned_points_or_absent_contact_cannot_add_coverage(self):
        coverage = MeasuredCoverage(self.graph, ExperimentConfig())
        point = self.graph.vertices_m[0]
        normal = self.graph.normals[0]
        coverage.update(np.tile(point, (4, 1)), np.tile(normal, (4, 1)), np.zeros(4, dtype=bool))
        self.assertEqual(coverage.area_fraction, 0)
        self.assertFalse(np.any(coverage.visited_goals))

    def test_real_oriented_contact_adds_geodesic_area_once(self):
        coverage = MeasuredCoverage(self.graph, ExperimentConfig(coverage_radius_m=0.01))
        points = np.tile(self.graph.vertices_m[0], (4, 1))
        normals = np.tile(self.graph.normals[0], (4, 1))
        mask = np.array([True, False, False, False])
        coverage.update(points, normals, mask)
        first = coverage.area_fraction
        self.assertGreater(first, 0)
        self.assertLess(first, 1)
        coverage.update(points, normals, mask)
        self.assertEqual(coverage.area_fraction, first)

    def test_reverse_normal_cannot_count_opposite_sheet(self):
        coverage = MeasuredCoverage(self.graph, ExperimentConfig())
        points = np.tile(self.graph.vertices_m[0], (4, 1))
        normals = np.tile(-self.graph.normals[0], (4, 1))
        coverage.update(points, normals, np.array([True, False, False, False]))
        self.assertEqual(coverage.area_fraction, 0)


class PassiveMCCBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import mujoco
        from Module.fr3_leap import build_full_robot, FullRobotModelConfig, ARM_HOME_Q, HAND_NATURAL_Q
        from Module.i04_geometric_traversal.physics import set_joint_state
        cls.h = build_full_robot(FullRobotModelConfig(surface='plane', gravity_m_s2=0.0))
        cls.data = mujoco.MjData(cls.h.model)
        cls.q = np.r_[ARM_HOME_Q, HAND_NATURAL_Q]
        set_joint_state(cls.h, cls.data, cls.q)

    def test_free_fingers_hold_without_oracle_search(self):
        from Module.i04_geometric_traversal.physics import BasicPassiveMCC
        controller = BasicPassiveMCC(self.h, 0.002)
        command = controller.command(self.data, self.q, np.zeros(4), np.zeros((4, 3)),
                                     np.zeros(4, dtype=bool), 2.0)
        np.testing.assert_array_equal(command, self.q)

    def test_excess_force_moves_pad_outward_on_actual_robot_jacobian(self):
        from Module.i04_geometric_traversal.physics import BasicPassiveMCC, position_jacobian
        controller = BasicPassiveMCC(self.h, 0.002)
        normal = np.tile([0., 0., 1.], (4, 1))
        command = controller.command(self.data, self.q, np.full(4, 3.0), normal,
                                     np.ones(4, dtype=bool), 2.0)
        for i, site in enumerate(self.h.tip_site_ids):
            jp, _ = position_jacobian(self.h, self.data, site)
            self.assertGreater(float(normal[i] @ jp @ (command - self.q)), 0)


if __name__ == '__main__':
    unittest.main()
