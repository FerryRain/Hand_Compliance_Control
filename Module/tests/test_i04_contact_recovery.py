"""Recovery semantics: tolerated gaps, confirmed reacquisition, no task abort."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import trimesh

from Module.i04_geometric_traversal.contracts import ExperimentConfig
from Module.i04_geometric_traversal.fixtures import SurfaceGraph
from Module.i04_geometric_traversal.recovery import SharedContactRecovery, ContactEpisodes


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.config = ExperimentConfig()
        self.q = np.zeros(23)
        self.normals = np.tile([0., 0., 1.], (4, 1))
        self.forces = np.array([2., 2., 0., 0.])
        self.active = self.forces > .2
        self.policy = SharedContactRecovery(self.config)
        self.policy.observe(0, self.q, self.forces, self.normals, self.active, self.q + .01)
        self.policy.observe(.2, self.q, self.forces, self.normals, self.active, self.q + .01)

    def test_short_zero_contact_does_not_abort_or_start_recovery(self):
        self.policy.observe(.3, self.q, np.zeros(4), self.normals, np.zeros(4, dtype=bool))
        self.policy.observe(.35, self.q, self.forces, self.normals, self.active)
        self.assertEqual(self.policy.mode, 'NORMAL')
        self.assertEqual(self.policy.recovery_attempts, 0)

    def test_persistent_loss_recovers_only_after_stable_real_contact(self):
        self.policy.observe(.3, self.q, np.zeros(4), self.normals, np.zeros(4, dtype=bool))
        self.policy.observe(.41, self.q, np.zeros(4), self.normals, np.zeros(4, dtype=bool))
        self.assertEqual(self.policy.mode, 'RECOVER')
        np.testing.assert_array_equal(self.policy.last_good_q, self.q + .01)
        one = np.array([2., 0., 0., 0.])
        self.policy.observe(.5, self.q, one, self.normals, one > .2)
        self.assertEqual(self.policy.mode, 'RECOVER')
        self.policy.observe(.66, self.q, one, self.normals, one > .2)
        self.assertEqual(self.policy.mode, 'NORMAL')
        self.assertEqual(self.policy.recovery_successes, 1)

    def test_recovery_timeout_records_and_continues(self):
        self.policy.observe(.3, self.q, np.zeros(4), self.normals, np.zeros(4, dtype=bool))
        self.policy.observe(.41, self.q, np.zeros(4), self.normals, np.zeros(4, dtype=bool))
        self.policy.observe(.42 + self.config.recovery_timeout_s, self.q, np.zeros(4), self.normals, np.zeros(4, dtype=bool))
        self.assertEqual(self.policy.mode, 'NORMAL')
        self.assertEqual(self.policy.recovery_timeouts, 1)
        self.assertTrue(any(e['kind'] == 'CONTACT_RECOVERY_TIMEOUT_CONTINUE' for e in self.policy.pending_events))

    def test_return_reference_advances_despite_measured_servo_lag(self):
        self.policy.observe(.3, self.q, np.zeros(4), self.normals, np.zeros(4, dtype=bool), self.q)
        self.policy.observe(.41, self.q, np.zeros(4), self.normals, np.zeros(4, dtype=bool), self.q)
        # No physical joint motion: the second command must still advance.
        jacobian = np.zeros((3, 23))
        handles = SimpleNamespace(palm_site_id=0, tip_site_ids=np.arange(4))
        with patch('Module.i04_geometric_traversal.recovery.joint_state', return_value=self.q), \
             patch('Module.i04_geometric_traversal.recovery.position_jacobian', return_value=(jacobian, jacobian)):
            first = self.policy.nominal(handles, None, self.q)
            self.policy.observe(.412, self.q, np.zeros(4), self.normals, np.zeros(4, dtype=bool), first)
            second = self.policy.nominal(handles, None, self.q)
        self.assertTrue(np.all(second[:7] > first[:7]))

    def test_force_relief_preserves_an_open_recovery_attempt(self):
        zero = np.zeros(4)
        free = np.zeros(4, dtype=bool)
        self.policy.observe(.3, self.q, zero, self.normals, free)
        self.policy.observe(.41, self.q, zero, self.normals, free)
        high = np.array([9., 0., 0., 0.])
        self.policy.observe(.5, self.q, high, self.normals, high > .2)
        self.assertEqual(self.policy.mode, 'FORCE_RELEASE')
        self.policy.observe(.52, self.q, self.forces, self.normals, self.active)
        self.assertEqual(self.policy.mode, 'RECOVER')
        self.policy.observe(.68, self.q, self.forces, self.normals, self.active)
        self.assertEqual(self.policy.recovery_attempts, 1)
        self.assertEqual(self.policy.recovery_successes, 1)

    def test_transient_touch_does_not_close_measured_gap(self):
        gaps = ContactEpisodes(.15)
        gaps.observe(0, 0)
        gaps.observe(.1, 1)
        gaps.observe(.12, 0)
        self.assertEqual(len(gaps.episodes), 0)
        gaps.observe(.3, 1)
        gaps.observe(.46, 1)
        self.assertTrue(gaps.episodes[0]['recovered'])
        self.assertAlmostEqual(gaps.episodes[0]['recovery_time_s'], .46)

    def test_disconnected_object_components_all_remain_in_coverage(self):
        first = trimesh.creation.icosphere(subdivisions=0, radius=.01)
        second = first.copy()
        second.apply_translation([.1, 0, 0])
        graph = SurfaceGraph(trimesh.util.concatenate([first, second]), coverage_radius_m=.02)
        self.assertTrue(np.all(graph.vertex_goal_owner >= 0))
        self.assertGreaterEqual(graph.required_goal_count, 2)


if __name__ == '__main__':
    unittest.main()
