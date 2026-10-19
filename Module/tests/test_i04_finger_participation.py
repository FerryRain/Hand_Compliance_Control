"""Behavioral evidence boundaries for coordinated finger recovery requests."""
import numpy as np
import unittest

from Module.i04_geometric_traversal.finger_participation import (
    FingerParticipationConfig, FingerParticipationMonitor,
)


def monitor(**kwargs):
    return FingerParticipationMonitor(np.tile([-2., 2.], (16, 1)),
                                      config=FingerParticipationConfig(**kwargs))


def observe(watcher, t, *, active=None, phase='SLIDE', q=None, tips=None, targets=None,
            states=None, forces=None, target_valid=None, target_available=None):
    if active is None:
        active = [False, True, True, True]
    if q is None:
        q = np.zeros(16)
    if tips is None:
        tips = np.zeros((4, 3))
    if targets is None:
        targets = np.tile([.05, 0., 0.], (4, 1))
    if states is None:
        states = ['SEEK', 'CONTACT_HOLD', 'CONTACT_HOLD', 'CONTACT_HOLD']
    delta = np.zeros(16)
    delta[0] = .001  # A command proposal must not masquerade as actual motion.
    return watcher.update(t, q, tips, active, phase, forces_n=forces,
                          planned_targets_world_m=targets, search_state=states,
                          search_delta_rad=delta, target_valid=target_valid,
                          target_available=target_available)


class FingerParticipationTests(unittest.TestCase):
    def test_target_distance_uses_pad_center_reference_not_offset_tip_site(self):
        watcher = monitor()
        centers = np.tile([.020, 0., 0.], (4, 1))
        targets = np.tile([.021, 0., 0.], (4, 1))
        record = watcher.update(0., np.zeros(16), pad_centers_world_m=centers,
                                measured_active=[False, True, True, True], phase='SLIDE',
                                planned_targets_world_m=targets)
        assert np.isclose(record['per_finger'][0]['target_distance_m'], .001)
        assert 'PAD_GEOM_CENTER' in record['geometry_reference']
        # The old ambiguous keyword must not silently accept an offset site.
        with self.assertRaises(TypeError):
            watcher.update(.1, np.zeros(16), tip_positions_world_m=np.zeros((4, 3)),
                           measured_active=[False] * 4, phase='SLIDE')

    def test_stationary_seek_requests_posture_recovery_without_claiming_contact(self):
        watcher = monitor()
        q = np.zeros(16); q[1] = 1.984
        original = q.copy()
        records = [observe(watcher, round(t, 3), q=q) for t in np.arange(0, 4.1, .1)]
        requests = [record['request'] for record in records if record['request']]
        assert len(requests) == 1
        request = requests[0]
        assert 3. <= request['time_s'] <= 3.1
        assert request['finger'] == 0
        assert 'JOINT_BOUNDARY_NEARBY' in request['reasons']
        assert 'SEEK_WITHOUT_MEASURED_PROGRESS' in request['reasons']
        assert request['evidence']['measured_joint_displacement_rad'] == 0.
        assert not request['recovery_success'] and not request['coverage_credit']
        assert not request['unreachable_proven']
        np.testing.assert_array_equal(q, original)


    def test_trajectory_selection_does_not_reset_physical_loss_clock(self):
        watcher = monitor()
        records = []
        for t in np.arange(0, 3.6, .1):
            # Repeated trajectory selection/rebase must not hide a missing finger.
            phase = 'SELECT' if int(round(t * 10)) in (9, 19, 29) else 'SLIDE'
            records.append(observe(watcher, round(t, 3), phase=phase))
        assert np.isclose(records[-1]['per_finger'][0]['loss_age_s'], 3.5)
        assert any(record['request'] for record in records)


    def test_air_moves_keep_true_loss_age_but_require_fresh_surface_evidence(self):
        watcher = monitor()
        records = []
        for t in np.arange(0, 10.1, .1):
            phase = 'SLIDE' if t < 2 or t >= 8 else 'LIFT' if t < 5 else 'TRANSFER'
            records.append(observe(watcher, round(t, 3), phase=phase))
        assert all(record['request'] is None for record in records if record['time_s'] < 9.)
        assert all(record['request'] is None for record in records if record['intentional_air_phase'])
        assert np.isclose(records[80]['per_finger'][0]['loss_age_s'], 8.)
        assert any(record['request'] for record in records if record['time_s'] > 9.)


    def test_actual_approach_is_progress_without_pretending_recovery(self):
        watcher = monitor()
        for t in np.arange(0, 8.1, .1):
            q = np.zeros(16); q[0] = .02 * t
            tips = np.zeros((4, 3)); tips[0, 0] = .004 * t
            record = observe(watcher, round(t, 3), q=q, tips=tips)
            assert record['request'] is None
            assert not record['per_finger'][0]['measured_contact']
        assert np.isclose(record['per_finger'][0]['measured_progress_toward_same_target_m_s'], .004)


    def test_moving_target_does_not_fake_progress_of_stationary_finger(self):
        watcher = monitor()
        emitted = []
        for t in np.arange(0, 4.1, .1):
            targets = np.tile([.05 - .004 * t, 0., 0.], (4, 1))
            record = observe(watcher, round(t, 3), targets=targets)
            if record['request']:
                emitted.append(record['request'])
        assert len(emitted) == 1
        assert emitted[0]['evidence']['measured_progress_toward_same_target_m_s'] == 0.


    def test_joint_motion_in_wrong_direction_does_not_hide_stagnation(self):
        watcher = monitor()
        requests = []
        for t in np.arange(0, 4.1, .1):
            q = np.zeros(16); q[0] = .08 * t
            tips = np.zeros((4, 3)); tips[0, 1] = .002 * t  # Tangential to target direction.
            record = observe(watcher, round(t, 3), q=q, tips=tips)
            if record['request']:
                requests.append(record['request'])
        assert requests and requests[0]['evidence']['measured_joint_displacement_rad'] > .002
        assert requests[0]['evidence']['measured_progress_toward_same_target_m_s'] < 0.


    def test_distance_gate_prompts_coordination_without_unreachable_claim(self):
        watcher = monitor()
        requests = []
        targets = np.tile([.12, 0., 0.], (4, 1))
        for t in np.arange(0, 3.6, .1):
            record = observe(watcher, round(t, 3), targets=targets,
                             states=['OUT_OF_REACH', 'CONTACT_HOLD', 'CONTACT_HOLD', 'CONTACT_HOLD'],
                             target_valid=[False, True, True, True])
            if record['request']:
                requests.append(record['request'])
        assert len(requests) == 1
        assert 'TARGET_DISTANCE_GATE_OR_OUT_OF_REACH_STATE' in requests[0]['reasons']
        assert np.isclose(requests[0]['evidence']['target_distance_m'], .12)
        assert not requests[0]['unreachable_proven']


    def test_missing_zero_filled_target_is_not_used_as_geometry(self):
        watcher = monitor()
        for t in np.arange(0, 4.1, .1):
            record = observe(watcher, round(t, 3), targets=np.zeros((4, 3)),
                             target_available=[False] * 4, target_valid=[False] * 4)
        finger = record['per_finger'][0]
        assert finger['target_distance_m'] is None
        assert finger['measured_progress_toward_same_target_m_s'] is None
        assert 'NO_MEASURED_FINGER_MOTION' in finger['reasons']


    def test_cooldown_budget_and_stable_contact_reset_not_brief_touch(self):
        watcher = monitor(global_request_cooldown_s=.5, finger_request_cooldown_s=.5,
                          max_requests_per_loss_episode=2)
        requests = []
        for t in np.arange(0, 12.1, .05):
            # One short real touch must not erase the request budget.
            contact = 4.0 <= t < 4.1 or 7.0 <= t < 7.3
            record = observe(watcher, round(t, 4), active=[contact, True, True, True])
            if record['request']:
                requests.append(record['request'])
            if 6.9 < t < 7.0:
                assert record['per_finger'][0]['request_budget_remaining'] == 0
        assert len([r for r in requests if r['time_s'] < 7.]) == 2
        assert len([r for r in requests if r['time_s'] > 7.3]) == 2
        assert [r['request_number_this_loss_episode'] for r in requests] == [1, 2, 1, 2]


    def test_global_throttle_and_fairness_for_multiple_lost_fingers(self):
        watcher = monitor(global_request_cooldown_s=1., finger_request_cooldown_s=1.,
                          max_requests_per_loss_episode=2)
        requests = []
        for t in np.arange(0, 8.1, .1):
            record = observe(watcher, round(t, 3), active=[False, False, True, True])
            if record['request']:
                requests.append(record['request'])
        assert [r['finger'] for r in requests] == [0, 1, 0, 1]
        assert min(np.diff([r['time_s'] for r in requests])) >= 1. - 1e-10


    def test_measured_force_threshold_and_force_pause_override_proposals(self):
        watcher = monitor()
        for t in np.arange(0, 4.1, .1):
            record = observe(watcher, round(t, 3), active=[True] * 4, forces=[.1, 9., 2., 2.])
            assert record['request'] is None
        assert not record['per_finger'][0]['measured_contact']
        assert np.isclose(record['per_finger'][0]['loss_age_s'], 4.)
        assert record['force_pause']


    def test_rejects_bad_observations_without_mutating_clock(self):
        watcher = monitor()
        observe(watcher, 0.)
        with self.assertRaisesRegex(ValueError, 'strictly increasing'):
            observe(watcher, 0.)
        with self.assertRaisesRegex(ValueError, 'q_rad'):
            observe(watcher, .1, q=np.zeros(7))
        record = observe(watcher, .1)
        assert np.isclose(record['per_finger'][0]['loss_age_s'], .1)

    def test_new_context_preserves_loss_history_cooldown_and_episode_request_total(self):
        watcher = monitor(global_request_cooldown_s=1., finger_request_cooldown_s=1.,
                          max_requests_per_loss_episode=2)
        requests = []
        for t in np.arange(0, 4.1, .1):
            record = observe(watcher, round(t, 3))
            if record['request']: requests.append(record['request'])
        assert len(requests) == 2
        grant = dict(kind='MEASURED_RECOVERY_CONTEXT_GRANTED', accepted=True,
                     finger=0, context_id=1)
        assert watcher.apply_recovery_context(grant)['accepted']
        record = observe(watcher, 4.1)
        assert record['request'] is None  # Still under the original cooldown.
        assert record['per_finger'][0]['unresolved_loss_episode_age_s'] == 4.1
        assert record['per_finger'][0]['requests_issued_this_loss_episode'] == 2
        assert record['per_finger'][0]['request_budget_remaining'] == 2
        record = observe(watcher, 5.)
        assert record['request']['request_number_this_loss_episode'] == 3
        assert record['request']['request_number_this_context'] == 1
        assert not watcher.apply_recovery_context(grant)['accepted']
        assert watcher._context_request_count[0] == 1

    def test_context_rejects_active_finger_and_bad_token(self):
        watcher = monitor()
        observe(watcher, 0., active=[True]*4)
        grant = dict(kind='MEASURED_RECOVERY_CONTEXT_GRANTED', accepted=True,
                     finger=0, context_id=1)
        assert not watcher.apply_recovery_context(grant)['accepted']
        with self.assertRaises(ValueError): watcher.apply_recovery_context({})

    def test_pending_mask_preserves_loss_and_budget_without_duplicate_requests(self):
        watcher = monitor()
        for t in np.arange(0, 5.1, .1):
            record = watcher.update(round(t, 3), np.zeros(16), np.zeros((4, 3)),
                [False, True, True, True], 'SLIDE',
                planned_targets_world_m=np.tile([.05, 0., 0.], (4, 1)),
                request_allowed_mask=[False, True, True, True])
            assert record['request'] is None
        assert record['per_finger'][0]['unresolved_loss_episode_age_s'] == 5.
        assert record['per_finger'][0]['request_budget_remaining'] == 3
        record = observe(watcher, 5.1)
        assert record['request']['finger'] == 0
        assert record['request']['request_number_this_loss_episode'] == 1

    def test_exact_force_pause_threshold_cannot_resolve_loss_episode(self):
        watcher = monitor()
        for t in np.arange(0, 3.2, .1): record = observe(watcher, round(t, 3))
        assert record['per_finger'][0]['requests_issued_this_loss_episode'] == 1
        for t in np.arange(3.2, 3.6, .1):
            record = observe(watcher, round(t, 3), active=[True]*4, forces=[8., 2., 2., 2.])
        assert not record['per_finger'][0]['measured_safe_contact']
        assert record['per_finger'][0]['unresolved_loss_episode_age_s'] == 3.5
        assert record['per_finger'][0]['requests_issued_this_loss_episode'] == 1
