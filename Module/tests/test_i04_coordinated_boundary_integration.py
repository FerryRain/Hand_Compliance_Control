"""Native runner integration for deferred recovery at natural stroke ends.

The short geometric proposals and request timing are controlled; the real
runner, queue, native 2 ms dynamics, actual sensors, guards, preload, search
and execution remain in place. The synthetic hold stroke tests scheduling,
not measured surface traversal or the separately tested 3 s eligibility.
"""
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
import json
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from Module.i04_geometric_traversal import coordinated_surface_slide as runner
from Module.i04_geometric_traversal.finger_reconfiguration_planner import FingerReconfigurationPath
from Module.i04_geometric_traversal.contact_recovery_planner import AnchoredRecoveryPath
from Module.i04_geometric_traversal.physics import joint_state, pose, rotation
from Module.i04_geometric_traversal.surface_slide import SurfaceStroke
from Module.tests import test_i04_coordinated_combo_integration as fixture


class CoordinatedBoundaryIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.CoordinatedComboRunnerTests.setUpClass()
        cls.fixture = fixture.CoordinatedComboRunnerTests

    @classmethod
    def tearDownClass(cls):
        fixture.CoordinatedComboRunnerTests.tearDownClass()

    def case(self, name, *, path_duration=.05, fixed_path_missing=False,
             endpoint_escalation=False, unqualified_anchor=False):
        f = self.fixture
        h, graph = f.h, f.graph
        data = mujoco.MjData(h.model)
        for field in ('qpos', 'qvel', 'ctrl', 'qacc_warmstart'):
            getattr(data, field)[:] = getattr(f.source, field)
        data.time = f.source.time
        mujoco.mj_forward(h.model, data)
        issued = False
        request_time = request_q = None
        calls, masks = [], []
        original_monitor = runner.FingerParticipationMonitor.update

        def observe(monitor, *args, **kwargs):
            nonlocal issued, request_time, request_q
            result = original_monitor(monitor, *args, **kwargs)
            now, phase = float(args[0]), args[4]
            masks.append((now, np.asarray(kwargs['request_allowed_mask']).copy()))
            if not issued and phase == 'SLIDE' and now >= .020:
                issued = True
                request_time = now
                request_q = np.asarray(args[1]).copy()
                result['request'] = dict(kind='FINGER_POSTURE_RECOVERY_REQUEST', request_id=1,
                                         finger=f.selected, time_s=now, recovery_context_id=0)
            else:
                result['request'] = None
            return result

        def stroke_plan(planner, native_data, coverage, forces, points, normals, active):
            # Keep a real support's measured normal and let native MCC evolve.
            # A very short held-reference proposal reaches its genuine arc end
            # after the production speed ramp, rather than faking an endpoint.
            q, palm = joint_state(h, native_data), pose(h, native_data)
            anchor = int(np.argmax(np.where(active, forces, -1.)))
            normal = normals[anchor]
            position = points[anchor]-h.object_position_m
            stroke = SurfaceStroke(np.stack([q, q]), np.stack([position, position]),
                np.stack([normal, normal]), np.stack([palm, palm]), np.array([0., .00045]),
                anchor, 0, rotation(palm[3:]).inv().apply(normal))
            return stroke, []

        def fixed_plan(planner, native_data, selected, active, forces, normals):
            q, palm = joint_state(h, native_data), pose(h, native_data)
            calls.append(dict(native_time_s=float(native_data.time), q=q.copy(), selected=selected))
            if fixed_path_missing:
                return None, dict(reason=('NO_SAFE_PATH_AMONG_BOUNDED_SURFACE_TARGETS'
                    if endpoint_escalation else 'INJECTED_NO_SAFE_FIXED_WRIST_PATH'))
            path = FingerReconfigurationPath(np.stack([q, q]), np.array([path_duration]),
                palm.copy(), palm.copy(), 0., f.anchor, selected)
            return path, dict(reason='INJECTED_FIXED_WRIST_HOLD_PATH')

        def anchored_plan(native_data, selected, active, forces, normals):
            q, palm = joint_state(h, native_data), pose(h, native_data)
            anchor = (f.anchor+1)%4 if unqualified_anchor else f.anchor
            path = AnchoredRecoveryPath(np.stack([q, q]), np.array([path_duration]),
                palm.copy(), palm.copy(), 0., anchor, selected)
            return path, dict(reason='INJECTED_ANCHORED_HOLD_PATH')

        directory = f.root/name
        with ExitStack() as stack:
            stack.enter_context(patch.object(runner, 'build_scene', return_value=(h, graph, f.q, f.quality)))
            stack.enter_context(patch.object(runner, 'acquire', return_value=(data, deepcopy(f.initial))))
            stack.enter_context(patch.object(runner.SurfaceStrokePlanner, 'plan', stroke_plan))
            stack.enter_context(patch.object(runner.FingerParticipationMonitor, 'update', observe))
            stack.enter_context(patch.object(runner.RecoveryContextObserver, 'note_request',
                side_effect=lambda finger, time_s: dict(accepted=True,
                    kind='INTEGRATION_TEST_REQUEST_ACKNOWLEDGED', finger=finger, time_s=time_s,
                    coverage_credit=False, contact_inferred=False)))
            stack.enter_context(patch.object(runner.MultiTargetFingerPlanner, 'plan', fixed_plan))
            wrist = stack.enter_context(patch.object(runner.ContactRecoveryPlanner, 'plan',
                side_effect=anchored_plan if endpoint_escalation else
                    AssertionError('Unqualified endpoint must not invoke a wrist fallback')))
            if endpoint_escalation:
                # Qualification history has separate native-cadence unit tests.
                # Inject only its permission to exercise production coupling,
                # fresh plant snapshots, anchor gate and original executor.
                stack.enter_context(patch.object(runner.EndpointRecoveryPolicy, 'consider_fixed_failure',
                    side_effect=lambda t, finger, request_id, reason, **kw: dict(
                        kind='ENDPOINT_RECOVERY_ESCALATION_GRANTED', accepted=True, event_id=1,
                        time_s=t, finger=finger, request_id=request_id, stable_support_fingers=[f.anchor],
                        search_budget_renewed=False, request_budget_renewed=False,
                        coverage_credit=False, contact_inferred=False,
                        test_scope='INJECTED_QUALIFICATION_ONLY_NOT_REAL_30S_LOSS')))
            stack.enter_context(patch.object(runner, 'plan_surface_transition_with_prepared',
                side_effect=lambda sp, *args: setattr(sp, 'last_prepared_transition_diagnostics',
                    dict(fallback_invoked=False))))
            stack.enter_context(redirect_stdout(io.StringIO()))
            runner.run(directory, budget_s=.45)
        if endpoint_escalation:
            self.assertEqual(wrist.call_count, 1)
        else:
            wrist.assert_not_called()
        events = json.loads((directory/'events.json').read_text(encoding='utf-8'))
        with np.load(directory/'trace.npz') as saved:
            trace = {key: saved[key] for key in saved.files}
        return directory, events, trace, calls, request_time, request_q, masks

    def assert_deferred_until_endpoint(self, events, trace, request_time, masks):
        queued = [e for e in events if e['kind']=='RECOVERY_REQUEST_QUEUED']
        taken = [e for e in events if e['kind']=='RECOVERY_REQUEST_TAKEN_AT_BOUNDARY']
        endpoint = [e for e in events if e['kind']=='STROKE_ENDPOINT'][0]
        self.assertEqual(len(queued), 1)
        self.assertEqual(len(taken), 1)
        self.assertEqual(queued[0]['time_s'], request_time)
        self.assertGreater(endpoint['time_s']-request_time, .1)
        self.assertEqual(taken[0]['time_s'], endpoint['time_s'])
        self.assertEqual(taken[0]['mode'], 'SHORT_FIXED_WRIST')
        before = (trace['time_s'] >= request_time) & (trace['time_s'] < endpoint['time_s'])
        self.assertTrue(np.any(before))
        self.assertTrue(np.all(trace['phase'][before] == 'SLIDE'))
        self.assertTrue(np.all(trace['coordinated_recovery_finger'][before] == -1))
        self.assertGreater(trace['nominal_path_distance_m'][before][-1], trace['nominal_path_distance_m'][before][0])
        selected = self.fixture.selected
        waiting_masks = [mask for t, mask in masks if request_time < t < endpoint['time_s']]
        self.assertTrue(waiting_masks)
        self.assertTrue(all(not mask[selected] for mask in waiting_masks))
        return endpoint['time_s']

    def test_real_advancing_stroke_finishes_before_native_short_fixed_wrist_attempt(self):
        directory, events, trace, calls, requested, request_q, masks = self.case('boundary_short')
        end = self.assert_deferred_until_endpoint(events, trace, requested, masks)
        starts = [e for e in events if e['kind']=='COORDINATED_RECOVERY_BEGIN']
        self.assertEqual(len(starts), 1)
        self.assertEqual(starts[0]['path_kind'], 'FINGER_RECONFIGURE')
        self.assertEqual(starts[0]['dispatch_mode'], 'SHORT_FIXED_WRIST')
        self.assertTrue(starts[0]['endpoint_motion_pause'])
        self.assertLessEqual(starts[0]['duration_s'], 8.)
        self.assertAlmostEqual(starts[0]['time_s']-end, .002)
        self.assertEqual(len(calls), 1)
        self.assertGreater(float(np.max(abs(calls[0]['q']-request_q))), 1e-8)
        with np.load(directory/'finger_recovery_0001.npz') as path:
            np.testing.assert_array_equal(path['q'][:, :7], np.repeat(path['actual_root_q'][None, :7], len(path['q']), axis=0))
            selected = self.fixture.selected
            other = np.r_[np.arange(7+4*selected), np.arange(11+4*selected, 23)]
            np.testing.assert_array_equal(path['q'][:, other], np.repeat(path['actual_root_q'][None, other], len(path['q']), axis=0))
        self.assertFalse(any(e['kind']=='COORDINATED_RECOVERY_ACTUAL_ENDPOINT_REPLAN' for e in events))

    def test_path_longer_than_eight_seconds_is_rejected_without_wrist_fallback(self):
        _, events, trace, calls, requested, _, masks = self.case('boundary_long', path_duration=8.002)
        self.assert_deferred_until_endpoint(events, trace, requested, masks)
        plans = [e for e in events if e['kind']=='COORDINATED_RECOVERY_PLAN']
        self.assertEqual(len(plans), 1)
        self.assertFalse(plans[0]['accepted'])
        self.assertEqual(plans[0]['finger_plan']['boundary_rejection'], 'FIXED_WRIST_PATH_EXCEEDS_ENDPOINT_TIME_LIMIT')
        self.assertIsNone(plans[0]['wrist_plan'])
        self.assertEqual(len(calls), 1)
        self.assertFalse(any(e['kind']=='COORDINATED_RECOVERY_BEGIN' for e in events))

    def test_no_fixed_wrist_solution_at_endpoint_does_not_escalate_to_pivot(self):
        _, events, trace, calls, requested, _, masks = self.case('boundary_no_solution', fixed_path_missing=True)
        self.assert_deferred_until_endpoint(events, trace, requested, masks)
        plans = [e for e in events if e['kind']=='COORDINATED_RECOVERY_PLAN']
        self.assertEqual(len(plans), 1)
        self.assertFalse(plans[0]['accepted'])
        self.assertIsNone(plans[0]['wrist_plan'])
        self.assertEqual(len(calls), 1)
        self.assertFalse(any(e['kind']=='COORDINATED_RECOVERY_BEGIN' for e in events))

    def test_explicit_endpoint_permission_uses_actual_root_and_native_support_snapshot(self):
        directory, events, trace, calls, requested, _, masks = self.case(
            'boundary_anchor_grant', fixed_path_missing=True, endpoint_escalation=True)
        end = self.assert_deferred_until_endpoint(events, trace, requested, masks)
        begins = [e for e in events if e['kind']=='COORDINATED_RECOVERY_BEGIN']
        self.assertEqual(len(begins), 1)
        begin = begins[0]
        self.assertAlmostEqual(begin['time_s'], end+.002)
        self.assertEqual(begin['dispatch_mode'], 'ANCHORED_ENDPOINT_FALLBACK')
        self.assertEqual(begin['path_kind'], 'WRIST_PIVOT')
        self.assertEqual(begin['endpoint_escalation_grant_id'], 1)
        self.assertTrue(begin['endpoint_motion_pause'])
        grant = next(e for e in events if e['kind']=='ENDPOINT_RECOVERY_ESCALATION_GRANTED')
        with np.load(directory/grant['native_support_window_file']) as window, \
             np.load(directory/begin['path_snapshot']) as path:
            self.assertEqual(len(window['time_s']), 76)
            np.testing.assert_allclose(np.diff(window['time_s']), .002, atol=1e-12)
            self.assertAlmostEqual(window['time_s'][-1], begin['time_s'])
            np.testing.assert_array_equal(window['q_rad'][-1], path['actual_root_q'])
            self.assertTrue(np.all(window['active'][:, self.fixture.anchor]))
            self.assertTrue(np.all(window['forces_n'][:, self.fixture.anchor] < 8.))
        self.assertEqual(int(trace['finger_posture_attempt_id'][-1, self.fixture.selected]), 1)
        running = trace['coordinated_recovery_finger'] == self.fixture.selected
        self.assertTrue(running.any())
        self.assertTrue(np.all(trace['coordinated_total_attempt_elapsed_s'][running] < 60.))

    def test_endpoint_pivot_anchor_without_qualified_support_is_rejected(self):
        _, events, trace, calls, requested, _, masks = self.case(
            'boundary_bad_anchor', fixed_path_missing=True, endpoint_escalation=True,
            unqualified_anchor=True)
        self.assert_deferred_until_endpoint(events, trace, requested, masks)
        plan = next(e for e in events if e['kind']=='COORDINATED_RECOVERY_PLAN')
        self.assertFalse(plan['accepted'])
        self.assertEqual(plan['wrist_plan']['endpoint_rejection'], 'PIVOT_ANCHOR_LACKS_150MS_MEASURED_SUPPORT')
        self.assertFalse(any(e['kind']=='COORDINATED_RECOVERY_BEGIN' for e in events))
        self.assertEqual(int(trace['finger_posture_attempt_id'][-1, self.fixture.selected]), 0)


if __name__ == '__main__':
    unittest.main()
