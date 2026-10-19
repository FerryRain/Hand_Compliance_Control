"""Bounded target diversity and shortest accepted-path selection."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from Module.i04_geometric_traversal.multi_target_finger_planner import (
    MultiTargetFingerConfig, MultiTargetFingerPlanner, dispersed_candidate_indices,
)
from Module.i04_geometric_traversal.finger_reconfiguration_planner import FingerReconfigurationConfig


class MultiTargetFingerTests(unittest.TestCase):
    def test_sampling_is_bounded_spaced_and_covers_directions_beyond_linear_preference(self):
        angle = np.linspace(0, 2*np.pi, 64, endpoint=False)
        points = np.c_[.03*np.cos(angle), .03*np.sin(angle), np.zeros(64)]
        scores = .03-points[:, 0]  # Local linear fit favors only +X.
        chosen = dispersed_candidate_indices(points, np.zeros(3), scores, 12, .004, .5)
        self.assertEqual(len(chosen), 12)
        self.assertEqual(len(set(chosen)), 12)
        selected = points[chosen]
        self.assertGreater(selected[:, 0].max(), .025)
        self.assertLess(selected[:, 0].min(), -.025)
        self.assertGreater(selected[:, 1].max(), .025)
        self.assertLess(selected[:, 1].min(), -.025)
        pairwise = np.linalg.norm(selected[:, None]-selected[None], axis=2)
        pairwise[np.diag_indices(len(chosen))] = np.inf
        self.assertGreaterEqual(pairwise.min(), .004-1e-12)
        self.assertEqual(chosen, dispersed_candidate_indices(points, np.zeros(3), scores, 12, .004, .5))

    def test_dense_duplicates_do_not_exhaust_budget_with_identical_targets(self):
        points = np.array([[0, 0, 0], [.01, 0, 0], [.01, 0, 0], [.0101, 0, 0], [-.02, 0, 0]])
        chosen = dispersed_candidate_indices(points, np.zeros(3), np.zeros(5), 20, .004)
        self.assertEqual(len(chosen), 2)
        self.assertEqual(set(chosen), {1, 4})
        self.assertEqual(dispersed_candidate_indices(points, np.zeros(3), np.zeros(5), 0, .004), [])

    def make_planner(self, nearest):
        planner = MultiTargetFingerPlanner.__new__(MultiTargetFingerPlanner)
        planner.config = MultiTargetFingerConfig(maximum_targets=4)
        planner.finger_config = FingerReconfigurationConfig()
        planner.h, planner.graph = object(), object()
        planner.nearest_planner = Mock()
        planner.nearest_planner.plan.return_value = nearest
        return planner

    def call(self, planner):
        return planner.plan(object(), 0, np.array([False, True, False, False]),
                            np.array([0., 1., 0., 0.]), np.zeros((4, 3)))

    def test_original_nearest_success_remains_fast_without_sampling_neighbors(self):
        path = SimpleNamespace(duration_s=1.)
        planner = self.make_planner((path, {'reason': 'GEOMETRIC_RECONFIGURATION_SELECTED'}))
        planner.candidate_targets = Mock(side_effect=AssertionError('Unnecessary neighborhood work'))
        selected, diag = self.call(planner)
        self.assertIs(selected, path)
        self.assertEqual(diag['attempted_target_count'], 1)
        self.assertEqual(diag['reason'], 'ORIGINAL_NEAREST_SAFE_PATH')
        self.assertFalse(diag['physical_recontact_tested'])

    def test_observed_force_rejection_does_not_trigger_alternative_geometry_search(self):
        planner = self.make_planner((None, {'reason': 'MEASURED_FORCE_OR_SELF_CONTACT'}))
        planner.candidate_targets = Mock(side_effect=AssertionError('Unsafe state must not be replanned'))
        selected, diag = self.call(planner)
        self.assertIsNone(selected)
        self.assertEqual(diag['reason'], 'MEASURED_FORCE_OR_SELF_CONTACT')

    def test_shortest_accepted_neighbor_selected_and_failed_reason_retained(self):
        planner = self.make_planner((None, {'reason': 'NO_SAFE_FIXED_WRIST_RECONFIGURATION'}))
        targets = [dict(face_id=i, point_local_m=[.005*(i+1), 0., 0.],
                        distance_from_original_nearest_m=.005*(i+1), linear_ordering_residual_m=0.)
                   for i in range(3)]
        planner.candidate_targets = Mock(return_value=(targets, {'selected_neighbor_count': 3}))
        slow, fast = SimpleNamespace(duration_s=4.), SimpleNamespace(duration_s=2.)
        adapters = [Mock(), Mock(), Mock()]
        for adapter, result in zip(adapters, ((slow, {'reason': 'SAFE'}),
                (None, {'reason': 'STRICT_COLLISION_EDGE'}), (fast, {'reason': 'SAFE'}))):
            adapter.plan.return_value = result
        with patch('Module.i04_geometric_traversal.multi_target_finger_planner.ExplicitSurfaceFingerPlanner',
                   side_effect=adapters):
            selected, diag = self.call(planner)
        self.assertIs(selected, fast)
        self.assertEqual(diag['safe_neighbor_count'], 2)
        self.assertEqual(diag['attempted_target_count'], 4)
        self.assertEqual(diag['selected']['face_id'], 2)
        self.assertEqual(diag['neighbor_attempts'][1]['planner']['reason'], 'STRICT_COLLISION_EDGE')
        self.assertTrue(diag['no_global_unreachability_or_optimality_claim'])
        for adapter in adapters:
            adapter.plan.assert_called_once()

    def test_no_path_is_only_a_bounded_search_result(self):
        planner = self.make_planner((None, {'reason': 'NO_SAFE_FIXED_WRIST_RECONFIGURATION'}))
        planner.candidate_targets = Mock(return_value=([], {'selected_neighbor_count': 0}))
        selected, diag = self.call(planner)
        self.assertIsNone(selected)
        self.assertEqual(diag['reason'], 'NO_SAFE_PATH_AMONG_BOUNDED_SURFACE_TARGETS')
        self.assertTrue(diag['no_global_unreachability_or_optimality_claim'])

    def test_invalid_radius_or_unbounded_candidate_budget_rejected(self):
        for kwargs in ({'maximum_surface_offset_m': .041}, {'maximum_targets': 129},
                       {'maximum_targets': 0}, {'minimum_candidate_spacing_m': 0.},
                       {'linear_priority_fraction': 1.1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                MultiTargetFingerConfig(**kwargs)


if __name__ == '__main__':
    unittest.main()
