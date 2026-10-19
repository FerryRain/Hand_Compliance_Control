"""Deferred recovery never interrupts SLIDE or revives stale diagnostic work."""
import json
import unittest
from dataclasses import replace
import numpy as np

from Module.i04_geometric_traversal.recovery_queue import RecoveryRequestQueue, RecoveryQueueConfig


class NativeQueue:
    def __init__(self):
        self.queue = RecoveryRequestQueue()
        self.tick = -1
        self.result = None

    def until(self, time_s, *, phase='SLIDE', active=None, forces=None, episodes=None, contexts=None):
        active = [True, False, False, False] if active is None else active
        forces = np.asarray(active, float) if forces is None else forces
        episodes = [1]*4 if episodes is None else episodes
        contexts = [0]*4 if contexts is None else contexts
        for tick in range(self.tick+1, round(time_s/.002)+1):
            self.result = self.queue.update(tick*.002, active, forces, episodes, contexts, phase)
            self.tick = tick
        return self.result

    def enqueue(self, finger=3, request_id=1, episode=1, context=0):
        return self.queue.enqueue(dict(kind='FINGER_POSTURE_RECOVERY_REQUEST', finger=finger,
                                      request_id=request_id, time_s=self.tick*.002,
                                      reasons=['LOW_MEASURED_TARGET_PROGRESS']),
                                  episode_id=episode, context_id=context)

    def take(self, **kwargs):
        return self.queue.take_boundary(self.tick*.002, **kwargs)


class RecoveryQueueTests(unittest.TestCase):
    def test_slide_only_queues_and_true_endpoint_returns_short_fixed_wrist_policy(self):
        h = NativeQueue(); h.until(0.)
        queued = h.enqueue()
        self.assertTrue(queued['accepted'])
        self.assertEqual(h.queue.pending_mask, [False, False, False, True])
        self.assertFalse(h.take(natural_stroke_endpoint=True)['accepted'])
        h.until(.002, phase='SELECT')
        self.assertEqual(h.take()['reason'], 'NO_NATURAL_STROKE_ENDPOINT')
        taken = h.take(natural_stroke_endpoint=True)
        self.assertTrue(taken['accepted'])
        self.assertEqual(taken['mode'], 'SHORT_FIXED_WRIST')
        self.assertEqual(taken['maximum_path_duration_s'], 8.)
        self.assertTrue(taken['fixed_wrist_required'])
        self.assertTrue(taken['selected_finger_only'])
        self.assertTrue(taken['plan_from_current_measured_state'])
        self.assertFalse(taken['path_authorized'])
        self.assertFalse(taken['budget_refunded'])
        self.assertEqual(h.queue.pending_mask, [False]*4)

    def test_only_one_take_per_select_visit_and_next_real_boundary_is_separate(self):
        h = NativeQueue(); h.until(0.)
        h.enqueue(3, 1)
        h.until(.002); h.enqueue(2, 2)
        h.until(.004, phase='SELECT')
        self.assertEqual(h.take(natural_stroke_endpoint=True)['finger'], 3)
        self.assertEqual(h.take(natural_stroke_endpoint=True)['reason'], 'ALREADY_TAKEN_THIS_NATIVE_TICK')
        h.until(.006, phase='SELECT')
        self.assertEqual(h.take(natural_stroke_endpoint=True)['reason'], 'ENDPOINT_ALREADY_USED')
        h.until(.008, phase='SLIDE')
        self.assertFalse(h.take(natural_stroke_endpoint=True)['accepted'])
        h.until(.010, phase='SELECT')
        self.assertEqual(h.take(natural_stroke_endpoint=True)['finger'], 2)

    def test_local_search_returns_existing_fallback_without_new_motion_budget(self):
        h = NativeQueue(); h.until(0.); h.enqueue()
        h.until(.002, phase='LOCAL_SEARCH')
        event = h.take()
        self.assertTrue(event['accepted'])
        self.assertEqual(event['mode'], 'FULL_FALLBACK')
        self.assertIsNone(event['maximum_path_duration_s'])
        self.assertTrue(event['external_safety_and_attempt_limits_unchanged'])
        self.assertFalse(event['path_authorized'])
        self.assertFalse(event['budget_refunded'])

    def test_brief_actual_contact_defers_take_but_does_not_discard_request(self):
        h = NativeQueue(); h.until(0.); h.enqueue()
        h.until(.100, phase='LOCAL_SEARCH', active=[True, False, False, True])
        self.assertEqual(h.take()['reason'], 'NO_CURRENTLY_LOST_PENDING_FINGER')
        self.assertTrue(h.queue.pending_mask[3])
        h.until(.102, phase='LOCAL_SEARCH')
        taken = h.take()
        self.assertTrue(taken['accepted'])
        self.assertEqual(taken['request_time_s'], 0.)
        self.assertEqual(taken['request_id'], 1)

    def test_native_two_ms_gap_restarts_stability_and_other_finger_cannot_resolve(self):
        h = NativeQueue(); h.until(0.); h.enqueue()
        h.until(.150, active=[True, False, False, True])  # selected has only .148 s
        self.assertTrue(h.queue.pending_mask[3])
        h.until(.152, active=[True, True, False, False])
        h.until(.302, active=[True, True, False, True])
        self.assertTrue(h.queue.pending_mask[3])
        result = h.until(.304, active=[True, True, False, True])
        self.assertFalse(h.queue.pending_mask[3])
        self.assertEqual(result['drop_events'][0]['reason'], 'ACTUAL_STABLE_SAME_FINGER_RECONTACT')
        self.assertAlmostEqual(result['drop_events'][0]['stable_safe_contact_s'], .15)
        self.assertFalse(result['drop_events'][0]['budget_refunded'])

    def test_force_exactly_eight_is_not_safe_recovery_and_blocks_take(self):
        h = NativeQueue(); h.until(0.); h.enqueue()
        h.until(.200, phase='LOCAL_SEARCH', active=[True, False, False, True], forces=[1., 0., 0., 8.])
        self.assertTrue(h.queue.pending_mask[3])
        self.assertEqual(h.take()['reason'], 'MEASURED_FORCE_RELEASE_REQUIRED')
        h.until(.350, active=[True, False, False, True], forces=[1., 0., 0., 1.])
        self.assertTrue(h.queue.pending_mask[3])
        h.until(.352, active=[True, False, False, True], forces=[1., 0., 0., 1.])
        self.assertFalse(h.queue.pending_mask[3])

    def test_real_context_change_discards_pending_without_refund(self):
        h = NativeQueue(); h.until(0.); h.enqueue(context=0)
        result = h.until(.002, contexts=[0, 0, 0, 1])
        event = result['drop_events'][0]
        self.assertEqual(event['reason'], 'MEASURED_GEOMETRY_CONTEXT_CHANGED')
        self.assertEqual(event['request_context_id'], 0)
        self.assertEqual(event['current_context_id'], 1)
        self.assertFalse(event['budget_refunded'])
        self.assertFalse(any(h.queue.pending_mask))

    def test_new_physical_loss_episode_discards_old_request(self):
        h = NativeQueue(); h.until(0.); h.enqueue()
        result = h.until(.002, episodes=[1, 1, 1, 2])
        self.assertEqual(result['drop_events'][0]['reason'], 'PHYSICAL_LOSS_EPISODE_CHANGED')
        self.assertEqual(result['drop_events'][0]['request_episode_id'], 1)
        self.assertEqual(result['drop_events'][0]['current_episode_id'], 2)

    def test_default_ttl_is_exactly_40_seconds_and_never_refreshes_while_waiting(self):
        h = NativeQueue(); h.until(0.); h.enqueue()
        h.until(39.998)
        self.assertTrue(h.queue.pending_mask[3])
        result = h.until(40.)
        self.assertFalse(h.queue.pending_mask[3])
        event = result['drop_events'][0]
        self.assertEqual(event['reason'], 'REQUEST_TTL_EXPIRED')
        self.assertEqual(event['request_age_s'], 40.)
        self.assertFalse(event['budget_refunded'])

    def test_duplicate_pending_does_not_replace_original_or_refund_emitted_request(self):
        h = NativeQueue(); h.until(0.); h.enqueue()
        h.until(.002)
        rejected = h.enqueue(3, 2)
        self.assertFalse(rejected['accepted'])
        self.assertEqual(rejected['reason'], 'FINGER_ALREADY_PENDING')
        self.assertFalse(rejected['budget_refunded'])
        self.assertEqual(h.queue.events[-1]['kind'], 'RECOVERY_REQUEST_ENQUEUE_REJECTED')
        h.until(.004, phase='SELECT')
        event = h.take(natural_stroke_endpoint=True)
        self.assertEqual(event['request_id'], 1)
        self.assertEqual(event['request_time_s'], 0.)
        h.until(.006)
        self.assertEqual(h.enqueue(3, 1)['reason'], 'REQUEST_ALREADY_SEEN')
        self.assertEqual(h.enqueue(3, 2)['reason'], 'REQUEST_ALREADY_SEEN')

    def test_oldest_currently_missing_request_wins_without_bridging_contact(self):
        h = NativeQueue(); h.until(0.); h.enqueue(3, 1)
        h.until(.002); h.enqueue(2, 2)
        h.until(.004, phase='SELECT', active=[True, False, False, True])
        event = h.take(natural_stroke_endpoint=True)
        self.assertEqual(event['finger'], 2)
        self.assertTrue(h.queue.pending_mask[3])

    def test_stale_observation_cannot_take_or_enqueue_and_air_never_takes(self):
        h = NativeQueue(); h.until(0.); h.enqueue()
        for phase in ('LIFT', 'TRANSFER', 'FINGER_RECOVERY', 'FORCE_RELIEF'):
            h.until((h.tick+1)*.002, phase=phase)
            self.assertFalse(h.take(natural_stroke_endpoint=True)['accepted'])
        self.assertEqual(h.queue.take_boundary(0., natural_stroke_endpoint=True)['reason'],
                         'STALE_OR_MISSING_NATIVE_OBSERVATION')
        stale = dict(kind='FINGER_POSTURE_RECOVERY_REQUEST', finger=1, request_id=2, time_s=0.)
        self.assertFalse(h.queue.enqueue(stale, episode_id=1, context_id=0)['accepted'])
        self.assertFalse(h.queue.pending_mask[1])

    def test_input_and_returned_event_copies_are_independent_and_json_safe(self):
        q = RecoveryRequestQueue()
        active, force, episodes, contexts = np.array([True, False, False, False]), np.array([1., 0., 0., 0.]), np.ones(4, int), np.zeros(4, int)
        originals = [x.copy() for x in (active, force, episodes, contexts)]
        q.update(0., active, force, episodes, contexts, 'SLIDE')
        request = dict(kind='FINGER_POSTURE_RECOVERY_REQUEST', finger=3, request_id=1, time_s=0., reasons=['LOSS'])
        q.enqueue(request, episode_id=1, context_id=0)
        request['reasons'].clear()
        mask = q.pending_mask; mask[3] = False
        self.assertTrue(q.pending_mask[3])
        q.update(.002, active, force, episodes, contexts, 'LOCAL_SEARCH')
        event = q.take_boundary(.002)
        self.assertEqual(event['request']['reasons'], ['LOSS'])
        event['request']['reasons'].clear()
        self.assertEqual(q.events[-1]['request']['reasons'], ['LOSS'])
        json.dumps(q.events, allow_nan=False)
        for original, actual in zip(originals, (active, force, episodes, contexts)):
            np.testing.assert_array_equal(actual, original)

    def test_native_clock_and_monotonic_physical_ids(self):
        h = NativeQueue(); h.until(0., contexts=[1]*4)
        with self.assertRaises(ValueError):
            h.queue.update(.004, [True]*4, [1.]*4, [1]*4, [1]*4, 'SLIDE')
        with self.assertRaises(ValueError):
            h.queue.update(.002, [True]*4, [1.]*4, [1]*4, [0]*4, 'SLIDE')
        h.until(.002, contexts=[1]*4)
        for change in (dict(dt_s=.01), dict(stable_recontact_s=.149), dict(maximum_endpoint_path_duration_s=8.001)):
            with self.assertRaises(ValueError):
                replace(RecoveryQueueConfig(), **change)


if __name__ == '__main__':
    unittest.main()
