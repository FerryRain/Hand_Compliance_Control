"""Native runner boundary tests for the prepared-hand terminal fallback.

Only the already-tested expensive transition proposal and exhausted scheduling
budget are injected. Dynamics, measured contacts, final guards, search, preload,
and coverage accounting remain the real runner. These held-reference proposals
test control boundaries; they are not evidence of physical traversal completion.
"""
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
import json
import unittest
from unittest.mock import PropertyMock, patch

import mujoco
import numpy as np

from Module.i04_geometric_traversal import coordinated_surface_slide as runner
from Module.i04_geometric_traversal.active_fingers_v3 import ActiveFingerSearchV3
from Module.i04_geometric_traversal.contracts import ExperimentConfig
from Module.i04_geometric_traversal.full_surface import JointPath
from Module.i04_geometric_traversal.oracle import MeasuredCoverage
from Module.i04_geometric_traversal.physics import joint_state, pose
from Module.tests import test_i04_coordinated_combo_integration as fixture


class CoordinatedPreparedIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.CoordinatedComboRunnerTests.setUpClass()
        cls.fixture = fixture.CoordinatedComboRunnerTests

    @classmethod
    def tearDownClass(cls):
        fixture.CoordinatedComboRunnerTests.tearDownClass()

    def case(self, name, *, fail=False):
        f = self.fixture
        h, graph = f.h, f.graph
        data = mujoco.MjData(h.model)
        for field in ('qpos', 'qvel', 'ctrl', 'qacc_warmstart'):
            getattr(data, field)[:] = getattr(f.source, field)
        data.time = f.source.time
        mujoco.mj_forward(h.model, data)
        starting_q = joint_state(h, data)
        force, points, normals, active, _, _ = runner.contact_measurement(h, data, .2)
        self.assertTrue(np.any(active))
        point = points[np.flatnonzero(active)[0]] - h.object_position_m
        claimed_goal = int(np.argmax(np.linalg.norm(graph.vertices_m-point, axis=1)))
        proposals, search_calls, measured_updates = [], [], []
        original_search = ActiveFingerSearchV3.update
        original_coverage = runner.MeasuredCoverage.update

        def transition(sp, oracle, native, coverage, attempts, outward, witnesses):
            before = coverage.visited_vertices.copy()
            current = joint_state(h, native)
            sp.last_prepared_transition_diagnostics = dict(
                fallback_invoked=True, finite_candidate_search=True,
                accepted=not fail, attempted_goals=[claimed_goal],
                failures=[dict(vertex=claimed_goal, reason='INJECTED_FINITE_NO_SAFE_ROUTE')] if fail else [])
            proposals.append(dict(q=current.copy(), covered=before))
            if fail:
                return None
            path = JointPath(np.stack([current, current]), np.array([.4]),
                             pose(h, native), np.zeros(7), 0.)
            # The single transfer represents all internal lift/hand/transfer
            # stages; it must not activate inward contact control midway.
            return [('TRANSFER', path)], outward, dict(
                vertex=claimed_goal, duration_s=.4,
                pose_policy='PREPARED_HAND_GEOMETRIC_FALLBACK',
                internal_segment_phases=['LIFT', 'HAND_PREPARATION', 'TRANSFER'],
                physical_contact_or_coverage_claim=False)

        def observe_search(search, native, nominal, forces, normals, active, **kwargs):
            result = original_search(search, native, nominal, forces, normals, active, **kwargs)
            search_calls.append(dict(enabled=kwargs['enabled'], intentional_air=kwargs.get('intentional_air'),
                                     nominal=nominal.copy(), result=result[0].copy(),
                                     diagnostics=deepcopy(result[1])))
            return result

        def observe_coverage(coverage, points, normals, active):
            measured_updates.append((points.copy(), normals.copy(), active.copy()))
            return original_coverage(coverage, points, normals, active)

        directory = f.root/name
        with ExitStack() as stack:
            stack.enter_context(patch.object(runner, 'build_scene', return_value=(h, graph, f.q, f.quality)))
            stack.enter_context(patch.object(runner, 'acquire', return_value=(data, deepcopy(f.initial))))
            stack.enter_context(patch.object(runner.LocalProgressBudget, 'exhausted',
                                              new_callable=PropertyMock, return_value=True))
            stack.enter_context(patch.object(runner, 'plan_surface_transition_with_prepared', transition))
            stack.enter_context(patch.object(ActiveFingerSearchV3, 'update', observe_search))
            stack.enter_context(patch.object(runner.MeasuredCoverage, 'update', observe_coverage))
            if not fail:
                stack.enter_context(patch.object(runner.ContactPreservingMCC, 'command',
                    side_effect=AssertionError('Intentional prepared TRANSFER must bypass inward MCC')))
            stack.enter_context(redirect_stdout(io.StringIO()))
            summary = runner.run(directory, budget_s=.66 if fail else .12)
        events = json.loads((directory/'events.json').read_text(encoding='utf-8'))
        with np.load(directory/'trace.npz') as z:
            trace = {k:z[k] for k in z.files}
        independent = MeasuredCoverage(graph, ExperimentConfig())
        for measured in measured_updates:
            independent.update(*measured)
        return directory, summary, events, trace, proposals, search_calls, independent, starting_q, claimed_goal

    def test_prepared_composite_transfer_has_no_inward_search_or_mcc_and_no_planned_coverage(self):
        _, _, events, trace, proposals, calls, independent, starting_q, claimed = self.case('prepared_air')
        self.assertEqual(len(proposals), 1)
        self.assertTrue(calls)
        self.assertTrue(all(not c['enabled'] and c['intentional_air'] for c in calls))
        self.assertTrue(np.all(trace['command_phase'] == 'TRANSFER'))
        self.assertFalse(np.any(trace['finger_search_active']))
        self.assertFalse(np.any(trace['finger_search_enabled']))
        for c in calls:
            np.testing.assert_array_equal(c['nominal'], starting_q)
            np.testing.assert_array_equal(c['result'], starting_q)
            self.assertFalse(np.any(c['diagnostics']['seeking']))
        # The real plant evolves; the test did not replace dynamics with the
        # proposed held waypoint or manufacture a goal contact.
        self.assertGreater(np.max(abs(trace['q_rad'][-1]-trace['q_rad'][0])), 1e-8)
        np.testing.assert_array_equal(trace['covered_vertices'], independent.visited_vertices)
        self.assertFalse(trace['covered_vertices'][claimed])
        self.assertGreater(independent.visited_vertices.sum(), 0)
        search = [e for e in events if e['kind']=='PREPARED_SURFACE_FALLBACK_SEARCH']
        self.assertEqual(len(search), 1)
        self.assertTrue(search[0]['accepted'])
        self.assertFalse(search[0]['physical_contact_or_coverage_claim'])
        self.assertFalse(any(e.get('measured_recovery_success', False) for e in events))

    def test_four_finite_failures_are_logged_and_never_credit_the_failed_goal(self):
        _, summary, events, trace, proposals, _, independent, _, claimed = self.case('prepared_failures', fail=True)
        attempts = [e for e in events if e['kind']=='PREPARED_SURFACE_FALLBACK_SEARCH']
        self.assertEqual(len(attempts), 4)
        self.assertEqual(len(proposals), 4)
        self.assertEqual(summary['stop_reason'], 'NO_SAFE_REPOSITION_AFTER_EXPANDED_SEARCH')
        for attempt in attempts:
            self.assertFalse(attempt['accepted'])
            self.assertFalse(attempt['physical_contact_or_coverage_claim'])
            self.assertTrue(attempt['diagnostics']['finite_candidate_search'])
            self.assertEqual(attempt['diagnostics']['failures'][0]['reason'], 'INJECTED_FINITE_NO_SAFE_ROUTE')
        self.assertFalse(any(e['kind']=='LIFT_COMMIT' for e in events))
        np.testing.assert_array_equal(trace['covered_vertices'], independent.visited_vertices)
        self.assertFalse(trace['covered_vertices'][claimed])


if __name__ == '__main__':
    unittest.main()
