"""Measured sliding must distinguish motion, stationary holds, and reacquisition."""
import unittest

import numpy as np

from Module.i04_geometric_traversal.sliding_metrics import analyze_trace


def synthetic(duration=5., dt=.01, speed=.01):
    times = np.arange(0., duration + dt / 2, dt)
    active = np.zeros((len(times), 4), dtype=bool)
    active[:, 0] = True
    points = np.zeros((len(times), 4, 3))
    points[:, 0, 0] = times * speed
    return dict(time_s=times, active=active, forces_n=active.astype(float) * 2.,
                contact_positions_world_m=points, tip_positions_world_m=points.copy(),
                phase=np.full(len(times), 'SLIDE'))


class SlidingMetricsTests(unittest.TestCase):
    def test_stationary_contact_and_small_jitter_are_not_sliding(self):
        trace = synthetic(speed=0.)
        trace['contact_positions_world_m'][:, 0, 0] = np.sin(trace['time_s'] * 73.) * .0001
        result = analyze_trace(trace)
        self.assertEqual(result['contact_continuity_fraction'], 1.)
        self.assertEqual(result['safe_contact_continuity_fraction'], 1.)
        self.assertEqual(result['moving_contact_fraction'], 0.)
        self.assertEqual(result['sliding_bout_count'], 0)
        self.assertEqual(result['measured_moving_path_sum_over_fingers_m'], 0.)

    def test_translating_contact_counts_true_motion_after_window(self):
        result = analyze_trace(synthetic())
        self.assertAlmostEqual(result['moving_contact_fraction'], .9)
        self.assertEqual(result['sliding_bout_count'], 1)
        bout = result['sliding_bouts'][0]
        self.assertAlmostEqual(bout['duration_s'], 4.5)
        self.assertAlmostEqual(bout['net_displacement_m'], .045)
        self.assertAlmostEqual(bout['path_length_m'], .045)
        self.assertAlmostEqual(result['phase_summary']['SLIDE']['moving_contact_fraction'], .9)

    def test_switching_fingers_does_not_join_contact_or_motion(self):
        trace = synthetic(duration=3., speed=.02)
        second = trace['time_s'] >= 1.5
        trace['active'][second, 0] = False
        trace['active'][second, 1] = True
        trace['forces_n'] = trace['active'].astype(float) * 2.
        trace['contact_positions_world_m'][:, 1, 0] = trace['time_s'] * .02 + .1
        trace['tip_positions_world_m'] = trace['contact_positions_world_m'].copy()
        result = analyze_trace(trace)
        self.assertEqual(result['contact_continuity_fraction'], 1.)
        self.assertLessEqual(result['longest_strict_same_finger_contact_s'], 1.5)
        self.assertEqual(result['sliding_bout_count'], 0)
        self.assertLess(result['moving_contact_fraction'], .67)

    def test_short_gap_is_recovery_only_and_restarts_motion_window(self):
        trace = synthetic()
        missing = (trace['time_s'] >= 2.5) & (trace['time_s'] < 2.55)
        trace['active'][missing, 0] = False
        trace['forces_n'][missing, 0] = 0.
        result = analyze_trace(trace)
        finger_runs = [run for run in result['strict_same_finger_runs'] if run['finger'] == 0]
        self.assertEqual(len(finger_runs), 2)
        debounced = [run for run in result['debounced_same_finger_runs'] if run['finger'] == 0]
        self.assertEqual(len(debounced), 1)
        self.assertEqual(debounced[0]['bridged_gap_count'], 1)
        self.assertAlmostEqual(debounced[0]['bridged_gap_duration_s'], .05)
        self.assertEqual(result['sliding_bout_count'], 0)
        self.assertEqual(len(result['moving_runs']), 2)
        self.assertGreaterEqual(result['moving_runs'][1]['start_s'], 3.05 - 1e-12)

    def test_force_safety_is_distinct_from_contact(self):
        trace = synthetic()
        trace['forces_n'][:, 0] = 9.
        result = analyze_trace(trace)
        self.assertEqual(result['contact_continuity_fraction'], 1.)
        self.assertEqual(result['safe_contact_continuity_fraction'], 0.)
        self.assertGreater(result['moving_contact_fraction'], .8)
        self.assertEqual(result['safe_moving_contact_fraction'], 0.)

    def test_moving_contact_point_with_stationary_tip_needs_corroboration(self):
        trace = synthetic()
        trace['tip_positions_world_m'][:] = 0.
        result = analyze_trace(trace)
        self.assertGreater(result['contact_migration_fraction'], .8)
        self.assertEqual(result['moving_contact_fraction'], 0.)
        self.assertEqual(result['sliding_bout_count'], 0)
        del trace['tip_positions_world_m']
        result = analyze_trace(trace)
        self.assertFalse(result['fingertip_motion_corroborated'])
        self.assertGreater(result['moving_contact_fraction'], .8)

    def test_pure_normal_pressing_is_not_surface_sliding(self):
        trace = synthetic()
        normals = np.zeros_like(trace['contact_positions_world_m'])
        normals[:, :, 0] = 2.  # Magnitude must not affect projection.
        trace['normals_world'] = normals
        result = analyze_trace(trace)
        self.assertTrue(result['tangent_projection_applied'])
        self.assertEqual(result['contact_continuity_fraction'], 1.)
        self.assertEqual(result['contact_migration_fraction'], 0.)
        self.assertEqual(result['moving_contact_fraction'], 0.)
        self.assertEqual(result['sliding_bout_count'], 0)

    def test_tangent_motion_counts_and_normal_drift_does_not_inflate_distance(self):
        trace = synthetic()
        trace['contact_positions_world_m'][:, 0, 2] = trace['time_s'] * .1
        trace['tip_positions_world_m'] = trace['contact_positions_world_m'].copy()
        normals = np.zeros_like(trace['contact_positions_world_m'])
        normals[:, :, 2] = 1.
        trace['normals_world'] = normals
        result = analyze_trace(trace)
        self.assertAlmostEqual(result['moving_contact_fraction'], .9)
        self.assertEqual(result['sliding_bout_count'], 1)
        self.assertAlmostEqual(result['sliding_bouts'][0]['path_length_m'], .045)
        self.assertAlmostEqual(result['sliding_bouts'][0]['span_from_start_m'], .045)

    def test_tip_motion_also_must_be_tangential(self):
        trace = synthetic()
        trace['tip_positions_world_m'][:] = 0.
        trace['tip_positions_world_m'][:, 0, 2] = trace['time_s'] * .01
        normals = np.zeros_like(trace['contact_positions_world_m'])
        normals[:, :, 2] = 1.
        trace['normals_world'] = normals
        result = analyze_trace(trace)
        self.assertGreater(result['contact_migration_fraction'], .8)
        self.assertEqual(result['moving_contact_fraction'], 0.)

    def test_invalid_surface_normal_breaks_motion_evidence(self):
        trace = synthetic(duration=3.)
        normals = np.zeros_like(trace['contact_positions_world_m'])
        normals[:, :, 2] = 1.
        normals[150, 0] = 0.
        trace['normals_world'] = normals
        result = analyze_trace(trace)
        self.assertEqual(result['contact_continuity_fraction'], 1.)
        self.assertEqual(result['sliding_bout_count'], 0)
        self.assertEqual(len(result['moving_runs']), 2)

    def test_invalid_positions_break_motion_evidence(self):
        trace = synthetic(duration=3.)
        trace['contact_positions_world_m'][150, 0] = np.nan
        result = analyze_trace(trace)
        self.assertEqual(result['contact_continuity_fraction'], 1.)
        self.assertEqual(result['sliding_bout_count'], 0)
        self.assertEqual(len(result['moving_runs']), 2)

    def test_irregular_samples_use_elapsed_time(self):
        trace = synthetic()
        chosen = np.r_[np.arange(0, 100), np.arange(100, len(trace['time_s']), 10)]
        trace = {name: value[chosen] for name, value in trace.items()}
        result = analyze_trace(trace)
        self.assertAlmostEqual(result['duration_s'], 5.)
        self.assertAlmostEqual(result['moving_contact_fraction'], .9)
        self.assertEqual(result['sliding_bout_count'], 1)

    def test_reversed_time_rejected(self):
        trace = synthetic()
        trace['time_s'][50] = trace['time_s'][49]
        with self.assertRaises(ValueError):
            analyze_trace(trace)

    def test_actual_joint_mapping_excludes_arm_and_ignores_commands(self):
        trace = synthetic(duration=2.)
        trace['q_rad'] = np.zeros((len(trace['time_s']), 23))
        trace['q_rad'][:, :7] = trace['time_s'][:, None]
        trace['command_rad'] = np.tile(trace['time_s'][:, None], (1, 23)) * 2.
        result = analyze_trace(trace)
        for finger in result['per_finger']:
            self.assertEqual(finger['measured_joint_motion']['observed_motion_duration_s'], 0.)
            self.assertEqual(finger['measured_joint_motion']['range_rad_by_joint'], [0.] * 4)
            self.assertFalse(finger['measured_joint_motion']['command_arrays_used'])
        self.assertEqual(result['measured_finger_joint_mapping']['finger_joint_columns'][0], [7, 8, 9, 10])

    def test_each_finger_reports_physical_participation_and_off_contact_motion(self):
        trace = synthetic(duration=2.)
        trace['finger_q_rad'] = np.zeros((len(trace['time_s']), 4, 4))
        trace['finger_q_rad'][:, 1, 0] = trace['time_s'] * .02
        result = analyze_trace(trace)
        index, middle = result['per_finger'][:2]
        self.assertEqual(index['contact_fraction'], 1.)
        self.assertEqual(middle['contact_fraction'], 0.)
        self.assertEqual(middle['observed_off_contact_duration_s'], 2.)
        self.assertAlmostEqual(middle['measured_joint_motion']['observed_motion_while_off_contact_s'], 1.5)
        self.assertEqual(result['finger_search_recovery']['evidence_status'], 'NOT_RECORDED')
        self.assertEqual(result['finger_search_recovery']['attempt_count'], 0)

    def test_declared_search_requires_same_finger_stable_physical_contact(self):
        trace = synthetic(duration=2.)
        times = trace['time_s']
        trace['active'][:] = False
        trace['active'][times >= 1., 0] = True
        trace['forces_n'] = trace['active'].astype(float) * 2.
        trace['finger_q_rad'] = np.zeros((len(times), 4, 4))
        trace['finger_q_rad'][:, 0, 0] = np.minimum(times, 1.) * .02
        trace['finger_search_active'] = np.zeros_like(trace['active'])
        trace['finger_search_active'][(times >= .2) & (times < 1.), 0] = True
        result = analyze_trace(trace)
        search = result['finger_search_recovery']
        self.assertEqual(search['attempt_count'], 1)
        self.assertEqual(search['measured_stable_recovery_count'], 1)
        self.assertEqual(search['measured_recovery_with_actual_joint_motion_count'], 1)
        attempt = search['attempts'][0]
        self.assertAlmostEqual(attempt['measured_recovery_latency_s'], .8)
        self.assertAlmostEqual(attempt['stable_safe_contact_confirmed_s'], 1.15)
        self.assertFalse(attempt['recovery_inferred_from_command'])

    def test_commanded_search_and_done_event_cannot_establish_recovery(self):
        trace = synthetic(duration=2.)
        times = trace['time_s']
        trace['active'][:] = False
        trace['forces_n'][:] = 0.
        trace['q_rad'] = np.zeros((len(times), 23))
        trace['command_rad'] = np.tile(times[:, None], (1, 23))
        events = [dict(kind='FINGER_SEARCH_BEGIN', finger=0, time_s=.2),
                  dict(kind='FINGER_SEARCH_DONE', finger=0, time_s=1.)]
        result = analyze_trace(trace, events=events)
        search = result['finger_search_recovery']
        self.assertEqual(search['attempt_count'], 1)
        self.assertEqual(search['measured_stable_recovery_count'], 0)
        self.assertFalse(search['attempts'][0]['actual_joint_motion_observed'])

    def test_other_finger_contact_is_not_search_success(self):
        trace = synthetic(duration=2.)
        trace['active'][:] = False
        trace['active'][trace['time_s'] >= 1., 1] = True
        trace['forces_n'] = trace['active'].astype(float) * 2.
        events = [dict(kind='FINGER_SEARCH_BEGIN', finger=0, time_s=.2),
                  dict(kind='FINGER_SEARCH_END', finger=0, time_s=1.1)]
        search = analyze_trace(trace, events=events)['finger_search_recovery']
        self.assertEqual(search['measured_stable_recovery_count'], 0)

    def test_brief_contact_return_does_not_pass_search_recovery(self):
        trace = synthetic(duration=2.)
        trace['active'][:] = False
        trace['active'][(trace['time_s'] >= 1.) & (trace['time_s'] < 1.1), 0] = True
        trace['forces_n'] = trace['active'].astype(float) * 2.
        events = [dict(kind='FINGER_SEARCH_BEGIN', finger=0, time_s=.2),
                  dict(kind='FINGER_SEARCH_END', finger=0, time_s=1.1)]
        search = analyze_trace(trace, events=events)['finger_search_recovery']
        self.assertEqual(search['measured_stable_recovery_count'], 0)
        self.assertEqual(search['attempts'][0]['first_measured_contact_s'], 1.)

    def test_contact_return_without_actual_finger_motion_is_reported_separately(self):
        trace = synthetic(duration=2.)
        trace['active'][:] = False
        trace['active'][trace['time_s'] >= 1., 0] = True
        trace['forces_n'] = trace['active'].astype(float) * 2.
        trace['q_rad'] = np.zeros((len(trace['time_s']), 23))
        events = [dict(kind='FINGER_SEARCH_BEGIN', finger=0, time_s=.2),
                  dict(kind='FINGER_SEARCH_END', finger=0, time_s=1.1)]
        search = analyze_trace(trace, events=events)['finger_search_recovery']
        self.assertEqual(search['measured_stable_recovery_count'], 1)
        self.assertEqual(search['measured_recovery_with_actual_joint_motion_count'], 0)

    def test_explicit_joint_mapping_is_validated(self):
        trace = synthetic(duration=2.)
        trace['q_rad'] = np.zeros((len(trace['time_s']), 8))
        result = analyze_trace(trace, finger_joint_columns=[[0, 1], [2, 3], [4, 5], [6, 7]])
        self.assertEqual(result['per_finger'][0]['measured_joint_motion']['range_rad_by_joint'], [0., 0.])
        with self.assertRaises(ValueError):
            analyze_trace(trace, finger_joint_columns=[[0, 1], [1, 3], [4, 5], [6, 7]])


if __name__ == '__main__':
    unittest.main()
