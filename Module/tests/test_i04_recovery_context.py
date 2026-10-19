"""Real loss continuity, native support and finite geometric retry contexts."""
import json
import unittest
from dataclasses import asdict, replace

import numpy as np

from Module.i04_geometric_traversal.recovery_context import RecoveryContextConfig, RecoveryContextObserver


def pose(x=0., degrees=0.):
    angle = np.radians(degrees)/2
    return np.array([x, 0., 0., np.cos(angle), 0., 0., np.sin(angle)])


class NativeObservations:
    def __init__(self):
        self.observer = RecoveryContextObserver()
        self.tick = -1
        self.result = None

    def until(self, end_s, *, palm=None, active=None, forces=None, phase='SLIDE'):
        if palm is None:
            palm = pose()
        if active is None:
            active = [True, False, False, False]
        if forces is None:
            forces = np.asarray(active, float)
        for tick in range(self.tick+1, round(end_s/.002)+1):
            self.result = self.observer.update(tick*.002, palm, active, forces, phase)
            self.tick = tick
        return self.result

    def grant(self, finger=3, **kwargs):
        return self.observer.grant_new_context(finger, self.tick*.002, **kwargs)

    def request(self, finger=3):
        return self.observer.note_request(finger, self.tick*.002)


class RecoveryContextTests(unittest.TestCase):
    def ready(self):
        h = NativeObservations()
        h.until(0.)
        h.until(20., palm=pose(.040))
        return h

    def test_one_grant_requires_explicit_exhaustion_and_keeps_real_loss(self):
        h = self.ready()
        diag = h.result['per_finger'][3]
        self.assertTrue(diag['eligible'])
        self.assertEqual(diag['unresolved_loss_age'], 20.)
        self.assertFalse(h.grant()['accepted'])
        grant = h.grant(budget_exhausted=True)
        self.assertTrue(grant['accepted'])
        self.assertEqual(grant['kind'], 'MEASURED_RECOVERY_CONTEXT_GRANTED')
        self.assertEqual(grant['context_id'], 1)
        self.assertEqual(grant['loss_age_s'], 20.)
        self.assertFalse(grant['clear_loss_history'])
        self.assertFalse(grant['path_authorized'])
        self.assertFalse(grant['contact_inferred'])
        self.assertFalse(grant['coverage_credit'])
        self.assertFalse(h.grant(budget_exhausted=True)['accepted'])
        # Every tick needed to independently confirm native 150 ms support is saved.
        window = grant['evidence']['native_support_window']
        self.assertEqual(len(window), 76)
        self.assertAlmostEqual(window[-1]['time_s']-window[0]['time_s'], .150)
        self.assertTrue(all(x['measured_active'][grant['support_finger']] for x in window))
        json.dumps(grant, allow_nan=False)
        grant['consumed_poses_before'][0]['palm_pose_world'][0] = 900.
        journal = h.observer.events
        self.assertEqual(journal[-1]['consumed_poses_before'][0]['palm_pose_world'][0], 0.)
        journal.clear()
        self.assertTrue(h.observer.events)

    def test_rotation_and_quaternion_sign_use_same_physical_pose(self):
        h = NativeObservations()
        h.until(0.)
        h.until(20., palm=pose(degrees=20.))
        self.assertTrue(h.grant(budget_exhausted=True)['accepted'])
        other_sign = pose(degrees=20.)
        other_sign[3:] *= -1
        result = h.until(40., palm=other_sign)
        self.assertFalse(result['per_finger'][3]['eligible'])
        self.assertEqual(h.grant(budget_exhausted=True)['reason'], 'PREVIOUSLY_CONSUMED_GEOMETRY')

    def test_returning_to_any_earlier_consumed_pose_cannot_refresh_quota(self):
        h = NativeObservations()
        h.until(0.)
        h.until(3., palm=pose(.010))
        self.assertTrue(h.request()['accepted'])
        h.until(23., palm=pose(.060))
        grant = h.grant(budget_exhausted=True)
        self.assertTrue(grant['accepted'])
        self.assertEqual(len(grant['consumed_poses_before']), 2)
        # A -> B -> A is far from B, but was already attempted at A.
        result = h.until(43., palm=pose(.010))
        self.assertAlmostEqual(result['per_finger'][3]['reference_translation_m'], .050)
        self.assertFalse(result['per_finger'][3]['eligible'])
        self.assertFalse(h.grant(budget_exhausted=True)['accepted'])
        h.until(43.002, palm=pose(.110))
        next_grant = h.grant(budget_exhausted=True)
        self.assertTrue(next_grant['accepted'])
        self.assertEqual(next_grant['context_id'], 2)
        self.assertEqual(next_grant['requests_this_episode'], 1)
        self.assertEqual(next_grant['unresolved_loss_age'], 43.002)

    def test_each_historical_pose_must_individually_pass_translation_or_rotation(self):
        h = NativeObservations()
        h.until(0.)
        h.until(20., palm=pose(.050, 25.))
        self.assertTrue(h.grant(budget_exhausted=True)['accepted'])
        # Translation is novel vs A, rotation novel vs B, hence both comparisons pass.
        h.until(40., palm=pose(.050, 0.))
        self.assertTrue(h.grant(budget_exhausted=True)['accepted'])
        # max distance and max rotation across history cannot make a reused pose novel.
        h.until(60., palm=pose(.050, 25.))
        self.assertFalse(h.grant(budget_exhausted=True)['accepted'])

    def test_stationary_strokes_and_subthreshold_jitter_do_not_make_contexts(self):
        h = NativeObservations()
        h.until(0.)
        h.until(10., palm=pose(.039, 19.), phase='LOCAL_SEARCH')
        h.until(20., palm=pose(.039, 19.), phase='REACQUIRE')
        self.assertFalse(h.grant(budget_exhausted=True)['accepted'])
        self.assertEqual(h.result['per_finger'][3]['consumed_pose_count'], 1)
        self.assertEqual(h.result['per_finger'][3]['unresolved_loss_age'], 20.)
        self.assertFalse(hasattr(h.observer, 'rebase'))

    def test_two_ms_support_gap_restarts_full_150_ms_window(self):
        h = self.ready()
        h.until(20.002, palm=pose(.04), active=[False]*4)
        self.assertFalse(h.grant(budget_exhausted=True)['accepted'])
        h.until(20.152, palm=pose(.04))  # support resumed at 20.004, only .148 s
        self.assertFalse(h.grant(budget_exhausted=True)['accepted'])
        h.until(20.154, palm=pose(.04))
        self.assertTrue(h.grant(budget_exhausted=True)['accepted'])

    def test_switching_support_fingers_cannot_pool_their_time(self):
        h = NativeObservations()
        h.until(0.)
        h.until(19.898, palm=pose(.04))
        h.until(20.048, palm=pose(.04), active=[False, True, False, False])
        self.assertFalse(h.grant(budget_exhausted=True)['accepted'])
        h.until(20.050, palm=pose(.04), active=[False, True, False, False])
        grant = h.grant(budget_exhausted=True)
        self.assertTrue(grant['accepted'])
        self.assertEqual(grant['support_finger'], 1)
        self.assertAlmostEqual(grant['same_finger_safe_support_s'], .15)

    def test_air_relief_and_force_spike_clear_support_without_clearing_loss(self):
        h = self.ready()
        for phase in ('LIFT', 'TRANSFER', 'FORCE_RELIEF', 'FINGER_RECOVERY'):
            t = (h.tick+1)*.002
            h.until(t, palm=pose(.04), phase=phase)
            self.assertFalse(h.grant(budget_exhausted=True)['accepted'])
            h.until(t+.150, palm=pose(.04))
            self.assertFalse(h.grant(budget_exhausted=True)['accepted'])
            h.until(t+.152, palm=pose(.04))
            self.assertTrue(h.result['per_finger'][3]['eligible'])
        t = (h.tick+1)*.002
        h.until(t, palm=pose(.04), forces=[1., 8., 0., 0.])
        self.assertFalse(h.grant(budget_exhausted=True)['accepted'])
        h.until(t+.150, palm=pose(.04))
        self.assertFalse(h.grant(budget_exhausted=True)['accepted'])
        h.until(t+.152, palm=pose(.04))
        self.assertTrue(h.grant(budget_exhausted=True)['accepted'])
        self.assertAlmostEqual(h.result['per_finger'][3]['unresolved_loss_age'], (h.tick)*.002)

    def test_sensor_flag_without_measured_safe_force_is_not_support(self):
        h = NativeObservations()
        h.until(0.)
        h.until(20., palm=pose(.04), forces=[.199, 0., 0., 0.])
        self.assertFalse(h.grant(budget_exhausted=True)['accepted'])
        h.until(20.150, palm=pose(.04), forces=[.2, 0., 0., 0.])
        self.assertFalse(h.grant(budget_exhausted=True)['accepted'])
        h.until(20.152, palm=pose(.04), forces=[.2, 0., 0., 0.])
        self.assertTrue(h.grant(budget_exhausted=True)['accepted'])

    def test_three_requests_per_context_and_request_cooldowns_persist(self):
        h = NativeObservations()
        h.until(3.)
        first = h.request()
        self.assertTrue(first['accepted'])
        self.assertFalse(h.request()['accepted'])
        self.assertEqual(h.request(2)['reason'], 'GLOBAL_REQUEST_COOLDOWN')
        h.until(23.)
        self.assertTrue(h.request()['accepted'])
        h.until(43.)
        self.assertTrue(h.request()['accepted'])
        h.until(63., palm=pose(.04))
        self.assertEqual(h.request()['reason'], 'CONTEXT_REQUEST_QUOTA_EXHAUSTED')
        grant = h.grant(budget_exhausted=True)
        self.assertTrue(grant['accepted'])
        self.assertEqual(grant['previous_context_requests'], 3)
        self.assertEqual(grant['requests_this_episode'], 3)
        fourth = h.request()
        self.assertTrue(fourth['accepted'])
        self.assertEqual(fourth['requests_this_context'], 1)
        self.assertEqual(fourth['requests_this_episode'], 4)
        self.assertEqual(fourth['lifetime_requests'], 4)
        self.assertEqual(fourth['unresolved_loss_age'], 63.)

    def test_only_150_ms_real_same_finger_safe_contact_resolves_history(self):
        h = NativeObservations()
        h.until(3.)
        h.request()
        h.until(23., palm=pose(.04))
        h.grant(budget_exhausted=True)
        h.request()
        h.until(23.150, palm=pose(.04), active=[True, False, False, True])
        diag = h.result['per_finger'][3]
        self.assertGreater(diag['unresolved_loss_age'], 23.)
        self.assertEqual(diag['requests_this_episode'], 2)
        self.assertFalse(h.grant(budget_exhausted=True)['accepted'])
        h.until(23.152, palm=pose(.04), active=[True, False, False, True])
        diag = h.result['per_finger'][3]
        self.assertEqual(diag['unresolved_loss_age'], 0.)
        self.assertEqual(diag['consumed_pose_count'], 0)
        self.assertEqual(diag['requests_this_episode'], 0)
        self.assertEqual(diag['lifetime_requests'], 2)
        self.assertEqual(diag['context_id'], 1)
        h.until(23.154, palm=pose(.04))
        self.assertEqual(h.result['per_finger'][3]['episode_id'], 2)
        h.until(43.154, palm=pose())
        grant = h.grant(budget_exhausted=True)
        self.assertTrue(grant['accepted'])
        self.assertEqual(grant['context_id'], 2)
        self.assertEqual(grant['lifetime_requests'], 2)
        self.assertEqual(grant['requests_this_episode'], 0)
        resolved = [x for x in h.observer.events if x['kind']=='MEASURED_LOSS_EPISODE_RESOLVED' and x['finger']==3]
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0]['requests_this_episode'], 2)

    def test_brief_recontact_keeps_episode_age_and_consumed_pose_history(self):
        h = self.ready()
        h.grant(budget_exhausted=True)
        h.until(20.100, palm=pose(.04), active=[True, False, False, True])
        h.until(40., palm=pose())
        diag = h.result['per_finger'][3]
        self.assertAlmostEqual(diag['unresolved_loss_age'], 40.)
        self.assertAlmostEqual(diag['loss_age_s'], 19.898)
        self.assertEqual(diag['consumed_pose_count'], 2)
        self.assertFalse(h.grant(budget_exhausted=True)['accepted'])

    def test_stale_proposal_cannot_be_consumed_after_actual_contact_changes(self):
        h = self.ready()
        old_time = h.tick*.002
        h.until(20.002, palm=pose(.04), active=[True, False, False, True])
        rejected = h.observer.grant_new_context(3, old_time, budget_exhausted=True)
        self.assertEqual(rejected['reason'], 'STALE_OR_MISSING_NATIVE_OBSERVATION')
        self.assertEqual(h.grant(budget_exhausted=True)['reason'], 'FINGER_CURRENTLY_IN_CONTACT')
        self.assertEqual(h.result['per_finger'][3]['context_id'], 0)

    def test_no_clock_gaps_no_input_mutation_and_configuration_cannot_relax_limits(self):
        observer = RecoveryContextObserver()
        actual_pose, active, forces = pose(), np.array([True, False, False, False]), np.array([1., 0., 0., 0.])
        original = [x.copy() for x in (actual_pose, active, forces)]
        observer.update(100., actual_pose, active, forces, 'SLIDE')
        for value, before in zip((actual_pose, active, forces), original):
            np.testing.assert_array_equal(value, before)
        with self.assertRaises(ValueError):
            observer.update(100.004, actual_pose, active, forces, 'SLIDE')
        observer.update(100.002, actual_pose, active, forces, 'SLIDE')
        for change in (dict(dt_s=.01), dict(minimum_unresolved_loss_s=2.9),
                       dict(minimum_translation_m=.039), dict(minimum_rotation_deg=19.),
                       dict(minimum_stable_support_s=.149), dict(context_cooldown_s=19.),
                       dict(max_requests_per_context=4), dict(surface_phases=('LIFT',))):
            with self.assertRaises(ValueError):
                replace(RecoveryContextConfig(), **change)
        json.dumps(asdict(RecoveryContextConfig()), allow_nan=False)


if __name__ == '__main__':
    unittest.main()
