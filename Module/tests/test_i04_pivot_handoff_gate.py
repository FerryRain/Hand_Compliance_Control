"""Actual progress, tracking continuity and one-request recovery boundaries."""
import unittest
import numpy as np

from Module.i04_geometric_traversal.pivot_handoff_gate import PivotHandoffGate, PivotHandoffGateConfig


def observe(gate, step, elapsed=None, *, arm_error=0., finger_error=0., active=False,
            force=0., status='RUNNING'):
    q, applied = np.zeros(23), np.zeros(23)
    applied[:7] = arm_error
    applied[7:] = finger_error
    actual = np.array([active, False, False, True])
    forces = np.array([.5 if active else 0., force, 0., 1.])
    return gate.update(step*.002, min(step*.002, gate.path_duration_s) if elapsed is None else elapsed,
                       q, applied, actual, forces, previous_stage_status=status)


class PivotHandoffGateTests(unittest.TestCase):
    def gate(self, duration=.1):
        return PivotHandoffGate(duration, 0, started_s=0.)

    def test_endpoint_requires_full_150_ms_and_emits_only_one_request(self):
        gate = self.gate()
        for step in range(125):
            result = observe(gate, step)
            self.assertFalse(result['needs_finger_replan'])
        result = observe(gate, 125)
        self.assertTrue(result['ready'])
        self.assertTrue(result['needs_finger_replan'])
        self.assertTrue(result['requested'])
        self.assertAlmostEqual(result['endpoint_seen_s'], .1)
        self.assertAlmostEqual(result['endpoint_age_s'], .150)
        self.assertAlmostEqual(result['stable_tracking_age_s'], .150)
        self.assertFalse(result['budget_reset'])
        self.assertFalse(result['measured_recovery_success'])
        for step in range(126, 500):
            result = observe(gate, step)
            self.assertFalse(result['needs_finger_replan'])
            self.assertTrue(result['requested'])

    def test_long_wall_time_with_paused_path_is_not_an_endpoint(self):
        gate = self.gate()
        for step in range(500):
            result = observe(gate, step, elapsed=.08)
        self.assertFalse(result['ready'])
        self.assertIsNone(result['endpoint_seen_s'])
        self.assertEqual(result['endpoint_age_s'], 0.)
        self.assertEqual(result['stable_tracking_age_s'], 0.)

    def test_stable_tracking_before_endpoint_cannot_be_banked_after_a_gap(self):
        gate = self.gate()
        for step in range(50):
            observe(gate, step)
        # End reached t=.100 but arm remains unsettled until t=.202.
        for step in range(50, 101):
            result = observe(gate, step, arm_error=.004001)
        for step in range(101, 151):
            result = observe(gate, step, arm_error=.004)
            self.assertFalse(result['needs_finger_replan'])
        result = observe(gate, 151, arm_error=.004)
        self.assertTrue(result['needs_finger_replan'])
        self.assertAlmostEqual(result['stable_tracking_age_s'], .100)
        self.assertGreater(result['endpoint_age_s'], .150)

    def test_single_native_tracking_spike_restarts_continuous_window(self):
        gate = self.gate()
        for step in range(124):
            observe(gate, step)
        result = observe(gate, 124, arm_error=.00401)
        self.assertEqual(result['stable_tracking_age_s'], 0.)
        for step in range(125, 175):
            result = observe(gate, step)
            self.assertFalse(result['ready'])
        result = observe(gate, 175)
        self.assertTrue(result['needs_finger_replan'])
        self.assertAlmostEqual(result['stable_tracking_age_s'], .100)

    def test_arm_tracking_uses_last_applied_reference_and_excludes_finger_compliance(self):
        gate = self.gate()
        for step in range(126):
            result = observe(gate, step, finger_error=.25)
        self.assertTrue(result['needs_finger_replan'])
        self.assertEqual(result['arm_tracking_error_rad'], 0.)

    def test_actual_selected_contact_permanently_cancels_even_if_brief(self):
        for at in (0, 124, 125):
            with self.subTest(contact_tick=at):
                gate = self.gate()
                for step in range(250):
                    result = observe(gate, step, active=step == at)
                    self.assertFalse(result['needs_finger_replan'])
                self.assertTrue(result['cancelled'])
                self.assertFalse(result['requested'])
                self.assertEqual(result['reason'], 'SELECTED_FINGER_ALREADY_MEASURED_IN_CONTACT')

    def test_force_at_8_n_permanently_cancels_even_after_unloading(self):
        gate = self.gate()
        for step in range(250):
            result = observe(gate, step, force=8. if step == 100 else 0.)
            self.assertFalse(result['needs_finger_replan'])
        self.assertEqual(result['reason'], 'MEASURED_FORCE_RELEASE_REQUIRED')

    def test_aborted_or_recovered_previous_stage_cannot_request_next_stage(self):
        for status in ('ABORTED', 'RECOVERED'):
            with self.subTest(status=status):
                gate = self.gate()
                for step in range(250):
                    result = observe(gate, step, status=status if step == 75 else 'RUNNING')
                    self.assertFalse(result['needs_finger_replan'])
                self.assertTrue(result['cancelled'])

    def test_request_is_not_reissued_after_postrequest_tracking_or_contact_change(self):
        gate = self.gate()
        for step in range(126):
            result = observe(gate, step)
        self.assertTrue(result['needs_finger_replan'])
        for step in range(126, 300):
            result = observe(gate, step, arm_error=.1 if step < 140 else 0.)
            self.assertFalse(result['needs_finger_replan'])
        result = observe(gate, 300, active=True)
        self.assertTrue(result['requested'])
        self.assertTrue(result['cancelled'])
        self.assertFalse(result['ready'])

    def test_no_input_arrays_are_changed_and_no_target_pose_is_accepted(self):
        gate = self.gate()
        q = np.arange(23)*.01; applied = q.copy()
        active = np.array([False, True, False, False]); forces = np.array([0., 1., 0., 0.])
        original = [x.copy() for x in (q, applied, active, forces)]
        result = gate.update(0., 0., q, applied, active, forces)
        for actual, expected in zip((q, applied, active, forces), original):
            np.testing.assert_array_equal(actual, expected)
        self.assertFalse(result['cached_target_or_endpoint_pose_used'])
        self.assertFalse(result['coverage_credit'])
        with self.assertRaises(TypeError):
            gate.update(.002, .002, q, applied, active, forces, cached_endpoint=q)

    def test_skipped_duplicate_half_ticks_and_regressed_path_time_rejected(self):
        for invalid_step in (0, .5, 2):
            gate = self.gate(); observe(gate, 0)
            with self.subTest(invalid_step=invalid_step), self.assertRaises(ValueError):
                observe(gate, invalid_step)
        gate = self.gate(); observe(gate, 0, elapsed=.05)
        with self.assertRaises(ValueError):
            observe(gate, 1, elapsed=.04)
        with self.assertRaises(ValueError):
            observe(self.gate(), 0, elapsed=.2)

    def test_safety_requirements_cannot_be_relaxed_by_configuration(self):
        for values in ({'dt_s': .01}, {'minimum_endpoint_age_s': .149},
                       {'minimum_stable_tracking_s': .099}, {'maximum_arm_tracking_error_rad': .005},
                       {'force_pause_n': 9.}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                PivotHandoffGateConfig(**values)
        with self.assertRaises(ValueError):
            PivotHandoffGate(0., 0, started_s=0.)

    def constrained_observation(self, gate, step, *, measured_arm=0., applied_arm=0.,
                                anchor_active=True, anchor_force=1.):
        measured, applied = np.zeros(23), np.zeros(23)
        measured[:7] = measured_arm; applied[:7] = applied_arm
        return gate.update(step*.002, min(step*.002, .1), measured, applied,
            [False, False, False, anchor_active], [0., 0., 0., anchor_force])

    def constrained_gate(self, endpoint=None):
        return PivotHandoffGate(.1, 0, started_s=0.,
            endpoint_arm_q=np.zeros(7) if endpoint is None else endpoint, anchor_finger=3)

    def test_following_frozen_old_command_does_not_count_as_endpoint_settling(self):
        gate = self.constrained_gate(endpoint=np.full(7, .02))
        for step in range(250):
            result = self.constrained_observation(gate, step)
        self.assertEqual(result['arm_tracking_error_rad'], 0.)
        self.assertEqual(result['measured_endpoint_error_rad'], .02)
        self.assertEqual(result['applied_endpoint_error_rad'], .02)
        self.assertEqual(result['stable_tracking_age_s'], 0.)
        self.assertFalse(result['ready'])
        self.assertTrue(result['endpoint_check_enabled'])
        self.assertTrue(result['anchor_check_enabled'])

    def test_each_endpoint_error_and_mutual_tracking_error_are_required(self):
        for measured, applied in ((.0041, .003), (.003, .0041), (-.003, .003)):
            with self.subTest(measured=measured, applied=applied):
                gate = self.constrained_gate()
                for step in range(250):
                    result = self.constrained_observation(gate, step,
                        measured_arm=measured, applied_arm=applied)
                self.assertFalse(result['ready'])
                self.assertGreater(result['maximum_handoff_error_rad'], .004)
        gate = self.constrained_gate()
        for step in range(126):
            result = self.constrained_observation(gate, step, measured_arm=.004, applied_arm=.004)
        self.assertTrue(result['needs_finger_replan'])

    def test_single_native_anchor_gap_restarts_combined_stability_window(self):
        gate = self.constrained_gate()
        for step in range(124):
            self.constrained_observation(gate, step)
        result = self.constrained_observation(gate, 124, anchor_active=False, anchor_force=0.)
        self.assertFalse(result['anchor_safe_contact'])
        self.assertEqual(result['stable_tracking_age_s'], 0.)
        self.assertFalse(result['cancelled'])  # Execution owns the support timeout.
        for step in range(125, 175):
            result = self.constrained_observation(gate, step)
            self.assertFalse(result['needs_finger_replan'])
        result = self.constrained_observation(gate, 175)
        self.assertTrue(result['needs_finger_replan'])
        self.assertAlmostEqual(result['stable_tracking_age_s'], .100)

    def test_anchor_requires_actual_contact_and_threshold_force(self):
        for active, force in ((False, 1.), (True, .1999)):
            with self.subTest(active=active, force=force):
                gate = self.constrained_gate()
                for step in range(250):
                    result = self.constrained_observation(gate, step,
                        anchor_active=active, anchor_force=force)
                self.assertFalse(result['ready'])
                self.assertEqual(result['reason'], 'WAITING_FOR_ACTUAL_SAFE_ANCHOR')
        gate = self.constrained_gate()
        for step in range(126):
            result = self.constrained_observation(gate, step, anchor_force=.2)
        self.assertTrue(result['needs_finger_replan'])

    def test_endpoint_reference_is_copied_and_invalid_anchor_is_rejected(self):
        endpoint = np.zeros(7)
        gate = self.constrained_gate(endpoint)
        endpoint[:] = .5
        np.testing.assert_array_equal(gate.endpoint_arm_q, np.zeros(7))
        for endpoint in (np.zeros(6), np.full(7, np.nan)):
            with self.assertRaises(ValueError):
                self.constrained_gate(endpoint)
        for anchor in (0, -1, 4):
            with self.assertRaises(ValueError):
                PivotHandoffGate(.1, 0, started_s=0., endpoint_arm_q=np.zeros(7), anchor_finger=anchor)


if __name__ == '__main__':
    unittest.main()
