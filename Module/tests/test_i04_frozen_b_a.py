"""Regression boundaries for the user-selected frozen-B supervisor ablation."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from Module.i04_geometric_traversal.a_frozen_b_mcc import FrozenBAMCC
from Module.i04_geometric_traversal.b_full_mcc import FullBTraversal
from Module.i04_geometric_traversal.contracts import WristTarget
from Module.i04_geometric_traversal.whole_hand_completion import CompletionWholeHandTraversal
from Module.i04_geometric_traversal.whole_hand_recovery import RecoveryWholeHandTraversal
from Module.i04_geometric_traversal.whole_hand_surface_traversal import WholeHandTraversal


class FrozenABoundaryTests(unittest.TestCase):
    def test_scheduler_and_safety_inheritance_does_not_use_failed_matched_overrides(self):
        self.assertIs(FrozenBAMCC.ranked_targets, CompletionWholeHandTraversal.ranked_targets)
        self.assertIs(FrozenBAMCC.air_move, RecoveryWholeHandTraversal.air_move)
        self.assertIs(FrozenBAMCC.pivot_route, FullBTraversal.pivot_route)
        self.assertIs(FrozenBAMCC.measure, FullBTraversal.measure)

    def test_private_pad_targets_never_become_A_control_targets(self):
        runner = object.__new__(FrozenBAMCC)
        wrist = np.array([.1, .2, .3, 1., 0., 0., 0.])
        own_q = np.linspace(.1, .4, 23)
        own_points = np.full((4, 3), .12)
        own_normals = np.tile([0., 0., 1.], (4, 1))
        runner._active_reference = None
        runner._cached_key = None
        runner.last = own_q.copy()
        runner._wrist = lambda q: wrist.copy()
        runner._publish_wrist = lambda value, phase: [WristTarget(value)]
        received = []
        runner.a_reference = SimpleNamespace(
            solve=lambda target, root: own_q.copy(),
            pad_reference=lambda q: (own_points.copy(), own_normals.copy()))
        def control(instance, points, normals, nominal, phase):
            received.append((points.copy(), normals.copy(), nominal.copy()))
            return {'safe': np.ones(4, bool)}
        with patch.object(WholeHandTraversal, 'contact_step', control):
            for value in (1., 20.):
                private_points = np.full((4, 3), value)
                private_normals = np.full((4, 3), -value)
                private_q = np.full(23, value)
                runner.contact_step(private_points, private_normals, private_q, 'TEST')
                np.testing.assert_array_equal(runner.points, private_points)
                np.testing.assert_array_equal(runner.nominal, private_q)
        for points, normals, nominal in received:
            np.testing.assert_array_equal(points, own_points)
            np.testing.assert_array_equal(normals, own_normals)
            np.testing.assert_array_equal(nominal, own_q)

    def test_A_approach_preserves_B_gate_anchor_clearance_and_strict_initial_gate(self):
        runner = object.__new__(FrozenBAMCC)
        runner._active_reference = None
        class Recorder:
            def call(self, name):
                from contextlib import nullcontext
                return nullcontext()
        runner.recorder = Recorder()
        own = (np.zeros(23), np.zeros((4, 3)), np.tile([0., 0., 1.], (4, 1)))
        runner.adapt_nominal = lambda q, phase: own
        private = dict(q=np.ones(23), points=np.ones((4, 3)), normals=np.ones((4, 3)),
                       anchor=3, expected_contacts=np.array([False, True, True, True]))
        def approach(instance, candidate, clearance, strict_four):
            self.assertIs(instance._active_reference, private)
            self.assertEqual(candidate['anchor'], 3)
            np.testing.assert_array_equal(candidate['expected_contacts'], private['expected_contacts'])
            np.testing.assert_array_equal(candidate['q'], own[0])
            self.assertEqual(clearance, .015)
            self.assertTrue(strict_four)
            return True
        with patch.object(WholeHandTraversal, 'approach', approach):
            self.assertTrue(runner.approach(private, .015, True))
        self.assertIsNone(runner._active_reference)


if __name__ == '__main__':
    unittest.main()
