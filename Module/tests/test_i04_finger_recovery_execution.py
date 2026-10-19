"""Measured recovery behavior, independent of MuJoCo integration and planners."""
import unittest
import numpy as np

from Module.i04_geometric_traversal.finger_recovery_execution import (
    FingerRecoveryExecution, FingerRecoveryExecutionConfig,
)


class ReferencePath:
    """A deliberately moving 23-joint reference to expose failure to freeze."""
    selected_finger = 1
    anchor_finger = 3

    def __init__(self, duration=1.):
        self.duration_s = duration
        self.calls = []

    def sample(self, t):
        self.calls.append(t)
        return np.arange(23, dtype=float) * .01 + t * .1


def execution(duration=1., kind='FINGER_RECONFIGURE', **config):
    return FingerRecoveryExecution(ReferencePath(duration), kind=kind, started_s=0.,
                                   config=FingerRecoveryExecutionConfig(**config))


def tick(machine, step, *, selected=False, anchor=True, selected_force=1.,
         anchor_force=1., tracking=0., q=None, other_force=0.):
    q = np.arange(23, dtype=float) * .003 if q is None else q
    active = np.array([False, selected, False, anchor], bool)
    forces = np.array([other_force, selected_force if selected else 0., 0.,
                       anchor_force if anchor else 0.])
    return machine.update(step * .002, q, active, forces, tracking)


class FingerRecoveryExecutionTests(unittest.TestCase):
    def test_first_actual_contact_freezes_all_reference_joints_and_rebases_once(self):
        machine = execution(duration=10.)
        for step in range(10):
            result = tick(machine, step)
        elapsed = result['path_elapsed_s']
        measured = np.arange(23, dtype=float) * -.017
        result = tick(machine, 10, selected=True, q=measured)
        np.testing.assert_array_equal(result['nominal_q'], measured)
        self.assertTrue(result['first_contact'])
        self.assertTrue(result['rebase_required'])
        self.assertTrue(result['search_enabled'])
        self.assertFalse(result['measured_recovery_success'])
        self.assertEqual(result['path_elapsed_s'], elapsed)
        self.assertEqual(result['phase'], 'FINGER_STABILIZE')
        # Neither mutation of the caller's array nor later measured movement can
        # silently resume the old path or change its stored contact reference.
        frozen = measured.copy()
        measured[:] = 9.
        result = tick(machine, 11, selected=True, q=np.ones(23))
        np.testing.assert_array_equal(result['nominal_q'], frozen)
        self.assertFalse(result['first_contact'])
        self.assertFalse(result['rebase_required'])
        self.assertEqual(result['path_elapsed_s'], elapsed)

    def test_exactly_150_ms_continuous_safe_contact_is_required(self):
        machine = execution()
        for step in range(75):  # First contact t=0; last sample t=.148.
            result = tick(machine, step, selected=True)
            self.assertEqual(result['status'], 'RUNNING')
            self.assertFalse(result['measured_recovery_success'])
        result = tick(machine, 75, selected=True)
        self.assertEqual(result['status'], 'RECOVERED')
        self.assertEqual(result['reason'], 'MEASURED_SAFE_CONTACT_STABLE')
        self.assertTrue(result['measured_recovery_success'])
        self.assertFalse(result['coverage_credit'])
        self.assertFalse(result['search_enabled'])

    def test_short_contact_loss_resets_stability_but_preserves_measured_hold(self):
        machine = execution(duration=10.)
        frozen = np.arange(23, dtype=float) * .1
        for step in range(50):
            tick(machine, step, selected=True, q=frozen)
        result = tick(machine, 50, selected=False, q=frozen + .02)
        self.assertIsNone(result['stable_since_s'])
        self.assertTrue(result['search_enabled'])
        self.assertEqual(result['path_elapsed_s'], 0.)
        np.testing.assert_array_equal(result['nominal_q'], frozen)
        for step in range(51, 126):
            result = tick(machine, step, selected=True, q=frozen + .03)
            self.assertEqual(result['status'], 'RUNNING')
            self.assertEqual(result['path_elapsed_s'], 0.)
            np.testing.assert_array_equal(result['nominal_q'], frozen)
        result = tick(machine, 126, selected=True)
        self.assertEqual(result['status'], 'RECOVERED')
        self.assertAlmostEqual(result['first_contact_s'], 0.)
        self.assertAlmostEqual(result['stable_since_s'], .102)

    def test_active_flag_without_threshold_force_is_not_recovery(self):
        machine = execution(duration=10.)
        for step in range(100):
            result = tick(machine, step, selected=True, selected_force=.199999)
        self.assertEqual(result['status'], 'RUNNING')
        self.assertIsNone(result['first_contact_s'])
        self.assertGreater(result['path_elapsed_s'], 0.)
        # Equality at the active threshold is accepted.
        result = tick(machine, 100, selected=True, selected_force=.2)
        self.assertTrue(result['first_contact'])

    def test_force_without_active_sensor_flag_is_not_recovery(self):
        machine = execution(duration=10.)
        result = machine.update(0., np.zeros(23), [False, False, False, True],
                                [0., 1., 0., 1.], 0.)
        self.assertIsNone(result['first_contact_s'])
        self.assertFalse(result['measured_recovery_success'])

    def test_support_loss_pauses_path_immediately_and_aborts_at_60_ms(self):
        machine = execution(duration=10., kind='WRIST_PIVOT')
        tick(machine, 0)
        result = tick(machine, 1)
        frozen_elapsed = result['path_elapsed_s']
        for step in range(2, 32):  # Loss begins .004; last sample .062.
            result = tick(machine, step, anchor=False)
            self.assertEqual(result['status'], 'RUNNING')
            self.assertEqual(result['path_elapsed_s'], frozen_elapsed)
        measured = np.ones(23) * -.7
        result = tick(machine, 32, anchor=False, q=measured)
        self.assertEqual(result['reason'], 'MEASURED_ANCHOR_LOST')
        self.assertEqual(result['status'], 'ABORTED')
        self.assertFalse(result['search_enabled'])
        np.testing.assert_array_equal(result['nominal_q'], measured)

    def test_selected_finger_can_take_over_support_at_grace_boundary(self):
        machine = execution(duration=10., kind='WRIST_PIVOT')
        for step in range(30):
            result = tick(machine, step, anchor=False)
        result = tick(machine, 30, anchor=False, selected=True)
        self.assertEqual(result['status'], 'RUNNING')
        self.assertTrue(result['first_contact'])
        for step in range(31, 106):
            result = tick(machine, step, anchor=False, selected=True)
        self.assertEqual(result['status'], 'RECOVERED')

    def test_short_support_gap_does_not_accumulate_across_recontact(self):
        machine = execution(duration=10., kind='WRIST_PIVOT')
        for step in range(25):
            tick(machine, step, anchor=False)
        tick(machine, 25, anchor=True)
        for step in range(26, 56):
            result = tick(machine, step, anchor=False)
            self.assertEqual(result['status'], 'RUNNING')
        result = tick(machine, 56, anchor=False)
        self.assertEqual(result['reason'], 'MEASURED_ANCHOR_LOST')

    def test_fixed_wrist_reconfiguration_continues_and_records_anchor_loss(self):
        machine = execution(duration=10.)
        for step in range(101):
            result = tick(machine, step, anchor=False)
            self.assertEqual(result['status'], 'RUNNING')
        self.assertFalse(result['wrist_support_required'])
        self.assertFalse(result['anchor_measured_active'])
        self.assertFalse(result['anchor_safe_contact'])
        self.assertAlmostEqual(result['anchor_safe_loss_age_s'], .2)
        self.assertAlmostEqual(result['maximum_anchor_safe_loss_s'], .2)
        self.assertAlmostEqual(result['path_elapsed_s'], .2)
        result = tick(machine, 101, anchor=True)
        self.assertEqual(result['anchor_safe_loss_age_s'], 0.)
        self.assertAlmostEqual(result['maximum_anchor_safe_loss_s'], .2)
        self.assertGreater(result['path_elapsed_s'], .2)

    def test_fixed_wrist_still_freezes_on_selected_contact_without_anchor(self):
        machine = execution(duration=10.)
        for step in range(51):
            result = tick(machine, step, anchor=False)
        elapsed = result['path_elapsed_s']
        result = tick(machine, 51, selected=True, anchor=False)
        self.assertTrue(result['first_contact'])
        self.assertEqual(result['path_elapsed_s'], elapsed)
        for step in range(52, 127):
            result = tick(machine, step, selected=True, anchor=False)
        self.assertEqual(result['status'], 'RECOVERED')
        self.assertFalse(result['anchor_safe_contact'])
        self.assertGreater(result['anchor_safe_loss_age_s'], .25)

    def test_fixed_wrist_anchor_exception_never_bypasses_force_abort(self):
        machine = execution(duration=10.)
        for step in range(50):
            tick(machine, step, anchor=False)
        result = tick(machine, 50, anchor=False, other_force=8.)
        self.assertEqual(result['status'], 'ABORTED')
        self.assertEqual(result['reason'], 'FORCE_RELEASE_REQUIRED')

    def test_any_finger_force_at_limit_requests_external_release(self):
        for source in ('selected', 'anchor', 'other'):
            with self.subTest(source=source):
                machine = execution()
                kwargs = dict(selected=True)
                kwargs[dict(selected='selected_force', anchor='anchor_force',
                            other='other_force')[source]] = 8.
                q = np.arange(23) * .02
                result = tick(machine, 0, q=q, **kwargs)
                self.assertEqual(result['status'], 'ABORTED')
                self.assertEqual(result['reason'], 'FORCE_RELEASE_REQUIRED')
                self.assertIsNone(result['first_contact_s'])
                self.assertFalse(result['measured_recovery_success'])
                np.testing.assert_array_equal(result['nominal_q'], q)
                # State machine returns a measured reference; it does not
                # pretend to execute unloading or change the physical plant.
                np.testing.assert_array_equal(q, np.arange(23) * .02)

    def test_tracking_scales_then_pauses_and_times_out_at_300_ms(self):
        machine = execution(duration=10.)
        tick(machine, 0)
        result = tick(machine, 1, tracking=.007)
        self.assertAlmostEqual(result['path_elapsed_s'], .001)
        for step in range(2, 152):
            result = tick(machine, step, tracking=.010)
            self.assertEqual(result['status'], 'RUNNING')
            self.assertAlmostEqual(result['path_elapsed_s'], .001)
        result = tick(machine, 152, tracking=.010)
        self.assertEqual(result['reason'], 'ARM_TRACKING_TIMEOUT')
        self.assertFalse(result['search_enabled'])

    def test_tracking_recovery_resets_consecutive_pause_timer(self):
        machine = execution(duration=10.)
        for step in range(100):
            tick(machine, step, tracking=.010)
        tick(machine, 100, tracking=.004)
        for step in range(101, 251):
            result = tick(machine, step, tracking=.010)
            self.assertEqual(result['status'], 'RUNNING')
        result = tick(machine, 251, tracking=.010)
        self.assertEqual(result['reason'], 'ARM_TRACKING_TIMEOUT')

    def test_finger_path_search_starts_at_endpoint_and_is_bounded_to_two_seconds(self):
        machine = execution(duration=.010)
        for step in range(5):
            result = tick(machine, step)
            self.assertFalse(result['search_enabled'])
        result = tick(machine, 5)
        self.assertTrue(result['search_enabled'])
        self.assertAlmostEqual(result['path_elapsed_s'], .010)
        for step in range(6, 1005):
            result = tick(machine, step)
            self.assertEqual(result['status'], 'RUNNING')
        result = tick(machine, 1005)
        self.assertEqual(result['reason'], 'NO_STABLE_SELECTED_CONTACT')
        self.assertFalse(result['search_enabled'])

    def test_wrist_path_may_search_while_path_moves(self):
        machine = execution(kind='WRIST_PIVOT')
        result = tick(machine, 0)
        self.assertTrue(result['search_enabled'])
        self.assertFalse(result['measured_recovery_success'])

    def test_flickering_contact_cannot_extend_stabilization_forever(self):
        machine = execution(duration=10.)
        for step in range(1000):
            result = tick(machine, step, selected=(step % 50 != 49))
            self.assertEqual(result['status'], 'RUNNING')
        result = tick(machine, 1000, selected=True)
        self.assertEqual(result['status'], 'ABORTED')
        self.assertEqual(result['reason'], 'NO_STABLE_SELECTED_CONTACT')
        self.assertFalse(result['measured_recovery_success'])

    def test_slow_tracking_cannot_extend_total_execution_without_bound(self):
        machine = execution(duration=.1)
        # Just below the pause threshold makes progress but never gets the
        # consecutive tracking-pause timeout; the total deadline still applies.
        for step in range(1200):
            result = tick(machine, step, tracking=.00999)
            self.assertEqual(result['status'], 'RUNNING')
        result = tick(machine, 1200, tracking=.00999)
        self.assertEqual(result['reason'], 'RECOVERY_WALL_TIME_LIMIT')
        self.assertLess(result['path_elapsed_s'], .1)

    def test_terminal_state_cannot_resume_path_or_claim_late_contact_success(self):
        machine = execution()
        tick(machine, 0, other_force=8.)
        q = np.ones(23) * .7
        for step in range(1, 101):
            result = tick(machine, step, selected=True, q=q)
            self.assertEqual(result['status'], 'ABORTED')
            self.assertEqual(result['reason'], 'FORCE_RELEASE_REQUIRED')
            self.assertFalse(result['measured_recovery_success'])
            self.assertFalse(result['search_enabled'])
            np.testing.assert_array_equal(result['nominal_q'], q)

    def test_invalid_and_skipped_measurements_are_rejected(self):
        for bad in (np.full(23, np.nan), np.zeros(22)):
            with self.subTest(shape=bad.shape):
                with self.assertRaises(ValueError):
                    tick(execution(), 0, q=bad)
        for tracking in (-.001, np.nan):
            with self.assertRaises(ValueError):
                tick(execution(), 0, tracking=tracking)
        machine = execution()
        tick(machine, 0)
        with self.assertRaises(ValueError):
            tick(machine, 0)
        with self.assertRaises(ValueError):
            tick(machine, 2)  # A missed native frame breaks temporal evidence.
        with self.assertRaises(ValueError):
            FingerRecoveryExecutionConfig(contact_min_n=8.)
        with self.assertRaises(ValueError):
            FingerRecoveryExecutionConfig(tracking_full_rad=.011)
        with self.assertRaises(ValueError):
            execution(kind='GEOMETRIC_SUCCESS')


if __name__ == '__main__':
    unittest.main()
