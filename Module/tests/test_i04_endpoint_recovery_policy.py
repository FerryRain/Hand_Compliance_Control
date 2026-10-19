"""Finite endpoint permission uses continuous measured history, never a reset."""
from copy import deepcopy
from dataclasses import replace
import unittest
import numpy as np

from Module.i04_geometric_traversal.endpoint_recovery_policy import EndpointRecoveryPolicy, EndpointRecoveryConfig

POSE = [0., 0., 0., 1., 0., 0., 0.]
REASON = 'NO_SAFE_PATH_AMONG_BOUNDED_SURFACE_TARGETS'


class History:
    def __init__(self):
        self.p = EndpointRecoveryPolicy(); self.tick = -1
        self.active = [False, True, False, False]; self.forces = [0., 1., 0., 0.]
        self.episodes = [1, 0, 1, 1]; self.contexts = [0]*4; self.next_id = 1

    def until(self, end, phase='SLIDE'):
        for tick in range(self.tick+1, round(end/.002)+1):
            self.p.update(tick*.002, self.active, self.forces, self.episodes, self.contexts, phase, POSE)
            self.tick = tick

    def dispatch(self):
        self.until((self.tick+1)*.002, 'SELECT')
        event = dict(kind='RECOVERY_REQUEST_TAKEN_AT_BOUNDARY', accepted=True,
            mode='SHORT_FIXED_WRIST', natural_stroke_endpoint=True, phase='SELECT',
            request_id=self.next_id, finger=0, time_s=self.tick*.002,
            episode_id=self.episodes[0], context_id=self.contexts[0])
        self.next_id += 1
        assert self.p.note_boundary_dispatch(event)['accepted']
        return event

    def consider(self, dispatch, reason=REASON, **overrides):
        args=dict(measured_active=self.active, forces_n=self.forces, palm_pose_world=POSE,
            phase='SELECT', episode_id=self.episodes[0], context_id=self.contexts[0])
        args.update(overrides)
        return self.p.consider_fixed_failure((self.tick+1)*.002, 0, dispatch['request_id'], reason, **args)

    def fail(self, reason=REASON, **overrides):
        d=self.dispatch(); e=self.consider(d, reason, **overrides)
        self.until((self.tick+1)*.002, 'SELECT'); return e


class EndpointRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        h=History(); h.until(10.); first=h.fail()
        assert first['reason']=='UNRESOLVED_LOSS_TOO_SHORT'
        h.contexts[0]=1; h.until(31.)
        cls.ready=h

    def fresh(self):return deepcopy(self.ready)

    def test_real_distinct_failed_contexts_grant_with_same_support_and_no_budget_renewal(self):
        h=self.fresh(); e=h.fail()
        self.assertTrue(e['accepted']); self.assertEqual(e['stable_support_fingers'], [1])
        self.assertEqual(e['distinct_failed_context_ids'], [0,1])
        self.assertEqual(len(e['native_support_window']), 76)
        self.assertAlmostEqual(e['native_support_window'][-1]['time_s']-e['native_support_window'][0]['time_s'], .15)
        for key in ('search_budget_renewed','request_budget_renewed','clear_loss_history','contact_inferred','coverage_credit'):
            self.assertIs(e[key],False)
        e['failed_attempts'].clear(); self.assertEqual(len(h.p.events[-1]['failed_attempts']), 2)

    def test_repeated_same_context_does_not_satisfy_two_distinct_failures(self):
        h=History(); h.until(31.); h.fail(); e=h.fail()
        self.assertEqual(e['reason'],'INSUFFICIENT_DISTINCT_FAILED_CONTEXTS')

    def test_duration_preload_and_other_failures_do_not_become_geometric_evidence(self):
        for reason in ('FIXED_WRIST_PATH_EXCEEDS_ENDPOINT_TIME_LIMIT','DEEP_PAD_PENETRATION','STALE'):
            h=self.fresh(); e=h.fail(reason)
            self.assertEqual(e['reason'],'FIXED_FAILURE_NOT_GEOMETRIC_INFEASIBILITY')
            self.assertEqual(len(e['failed_attempts']),1)

    def test_current_measurement_revalidated_at_plan_tick(self):
        for active,forces,expected in (([True,True,False,False],[1.,1.,0.,0.],'FINGER_CURRENTLY_IN_CONTACT'),
                ([False,True,False,False],[0.,8.,0.,0.],'CURRENT_FORCE_RELEASE_REQUIRED')):
            h=self.fresh(); d=h.dispatch(); e=h.consider(d,measured_active=active,forces_n=forces)
            self.assertEqual(e['reason'],expected); self.assertFalse(e['accepted'])

    def test_support_cannot_be_spliced_between_fingers_or_across_air(self):
        for mode in ('switch','air','one_tick_gap'):
            h=self.fresh()
            if mode=='switch':h.active=[False,False,True,False];h.forces=[0.,0.,1.,0.]
            if mode=='one_tick_gap':h.active=[False]*4;h.forces=[0.]*4
            h.until(31.002,'TRANSFER' if mode=='air' else 'SLIDE')
            h.active=[False,True,False,False];h.forces=[0.,1.,0.,0.]
            e=h.fail(); self.assertEqual(e['reason'],'NO_STABLE_ACTUAL_SUPPORT')

    def test_cooldown_and_one_escalation_per_context_survive_requests(self):
        h=self.fresh(); self.assertTrue(h.fail()['accepted'])
        self.assertEqual(h.fail()['reason'],'CONTEXT_ALREADY_ESCALATED')
        h.contexts[0]=2; h.until(32.)
        self.assertEqual(h.fail()['reason'],'PER_FINGER_ESCALATION_COOLDOWN')
        h.until(92.); self.assertTrue(h.fail()['accepted'])

    def test_short_recontact_preserves_failures_but_150ms_resolves(self):
        h=self.fresh(); h.active[0]=True;h.forces[0]=1.
        h.until(31.100);h.active[0]=False;h.forces[0]=0.;h.until(31.102)
        self.assertTrue(h.fail()['accepted'])
        h.active[0]=True;h.forces[0]=1.;h.until(31.300)
        self.assertEqual(h.p._failures[0],[])
        self.assertIsNone(h.p._episode_start[0])
        h.active[0]=False;h.forces[0]=0.;h.episodes[0]=2;h.until(31.302)
        self.assertEqual(h.fail()['reason'],'UNRESOLVED_LOSS_TOO_SHORT')
        self.assertTrue(np.isfinite(h.p._last_escalation[0]))

    def test_8N_contact_does_not_clear_physical_episode(self):
        h=self.fresh();h.active[0]=True;h.forces[0]=8.;h.until(31.200)
        self.assertIsNotNone(h.p._episode_start[0]);self.assertEqual(len(h.p._failures[0]),1)

    def test_stale_duplicate_context_changed_and_nonendpoint_tokens_reject(self):
        h=self.fresh(); d=h.dispatch(); bad=deepcopy(d);bad['natural_stroke_endpoint']=False
        self.assertFalse(h.p.note_boundary_dispatch(bad)['accepted'])
        self.assertEqual(h.consider(d,context_id=9)['reason'],'CURRENT_CONTEXT_CHANGED')
        self.assertTrue(h.consider(d)['accepted'])
        self.assertEqual(h.consider(d)['reason'],'UNKNOWN_OR_USED_ENDPOINT_DISPATCH')
        h=self.fresh();d=h.dispatch();h.until((h.tick+1)*.002)
        self.assertEqual(h.consider(d)['reason'],'DISPATCH_NOT_PREVIOUS_TICK')

    def test_native_gap_external_history_reset_and_weakened_config_are_rejected(self):
        h=self.fresh()
        with self.assertRaises(ValueError):h.p.update(32.,h.active,h.forces,h.episodes,h.contexts,'SLIDE',POSE)
        h.episodes[0]=2
        with self.assertRaises(ValueError):h.until(31.002)
        for change in ({'minimum_unresolved_loss_s':29.},{'minimum_failed_contexts':1},
                       {'stable_support_s':.148},{'escalation_cooldown_s':59.},{'safe_force_max_n':8.1}):
            with self.assertRaises(ValueError):replace(EndpointRecoveryConfig(),**change)


if __name__=='__main__':unittest.main()
