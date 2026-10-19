"""Run the real native runner through short, injected geometric stage plans.

Only the scheduler request (including its context journal acknowledgment) and
expensive geometric proposal are controlled. These sub-second cases exercise
stage execution, not the independently tested 3-second request eligibility.
The runner, 2-ms physics, sensors, execution, gate, V3 search, preload handoff,
final guards and attempt grant are the production implementations.
"""
from copy import deepcopy
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from Module.i04_geometric_traversal import coordinated_surface_slide as runner
from Module.i04_geometric_traversal.active_fingers_v3 import ActiveFingerSearchV3
from Module.i04_geometric_traversal.contact_handoff import ContactPreservingMCC
from Module.i04_geometric_traversal.contact_recovery_planner import AnchoredRecoveryPath
from Module.i04_geometric_traversal.finger_reconfiguration_planner import FingerReconfigurationPath
from Module.i04_geometric_traversal.finger_recovery_execution import FingerRecoveryExecution
from Module.i04_geometric_traversal.physics import joint_state, pose
from Module.i04_geometric_traversal.surface_slide import GraceMCC
from Module.tests import test_i04_contact_handoff as native_fixture


class CoordinatedComboRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        workspace_tmp = Path(__file__).resolve().parents[2]/'.runtime_tmp'
        workspace_tmp.mkdir(exist_ok=True)
        cls.tmp = tempfile.TemporaryDirectory(prefix='i04_combo_integration_', dir=workspace_tmp)
        cls.root = Path(cls.tmp.name)
        cls.h, cls.graph, cls.q, cls.quality = runner.build_scene(cls.root/'fixture')
        cls.source, cls.initial = runner.acquire(cls.h, cls.graph, cls.q, 7)
        force, _, _, active, _, _ = runner.contact_measurement(cls.h, cls.source, .2)
        cls.selected = int(np.flatnonzero(~active)[0])
        cls.anchor = int(np.argmax(np.where(active, force, -1.)))

    @classmethod
    def tearDownClass(cls):
        assert cls.root.resolve().parent == (Path(__file__).resolve().parents[2]/'.runtime_tmp').resolve()
        cls.tmp.cleanup()

    def case(self, label, *, geometry_abort=False):
        h, graph = self.h, self.graph
        source = self.source
        data = mujoco.MjData(h.model)
        for field in ('qpos', 'qvel', 'ctrl', 'qacc_warmstart'):
            getattr(data, field)[:] = getattr(source, field)
        data.time = source.time
        mujoco.mj_forward(h.model, data)
        calls, grants, roots = [], [], []
        issued = False
        original_monitor = runner.FingerParticipationMonitor.update
        original_grant = ActiveFingerSearchV3.begin_posture_attempt
        original_guard = runner.SlidingSafetyPlanner.command_collision_reason
        guard_calls = 0

        def observer(monitor, *args, **kwargs):
            nonlocal issued
            result = original_monitor(monitor, *args, **kwargs)
            if not issued and args[4] == 'LOCAL_SEARCH':
                issued = True
                result['request'] = dict(kind='FINGER_POSTURE_RECOVERY_REQUEST', request_id=1,
                                         finger=self.selected, time_s=float(args[0]))
            else:
                result['request'] = None
            return result

        def grant(search, finger):
            result = original_grant(search, finger)
            grants.append(deepcopy(result))
            return result

        def path_for(native_data, finger, cls, duration):
            q = joint_state(h, native_data); p = pose(h, native_data)
            return cls(np.stack([q, q]), np.array([duration]), p.copy(), p.copy(), 0.,
                       self.anchor, finger)

        def finger_plan(planner, native_data, finger, active, force, normals):
            calls.append(float(native_data.time))
            roots.append(joint_state(h, native_data).copy())
            if len(calls) == 1:
                return None, dict(reason='INJECTED_NO_FIXED_WRIST_PLAN')
            return path_for(native_data, finger, FingerReconfigurationPath, .20), dict(reason='INJECTED_SAFE_HOLD_PATH')

        def wrist_plan(planner, native_data, finger, active, force, normals):
            return path_for(native_data, finger, AnchoredRecoveryPath, .020), dict(reason='INJECTED_SAFE_HOLD_PIVOT')

        def collision_guard(planner, current, command):
            nonlocal guard_calls
            guard_calls += 1
            if geometry_abort and guard_calls == 5:
                return 'NON_TIP_CLEARANCE'
            return original_guard(planner, current, command)

        directory = self.root/label
        with ExitStack() as stack:
            stack.enter_context(patch.object(runner, 'build_scene', return_value=(h, graph, self.q, self.quality)))
            stack.enter_context(patch.object(runner, 'acquire', return_value=(data, deepcopy(self.initial))))
            stack.enter_context(patch.object(runner.SurfaceStrokePlanner, 'plan', return_value=(None, [])))
            stack.enter_context(patch.object(runner.FingerParticipationMonitor, 'update', observer))
            stack.enter_context(patch.object(runner.RecoveryContextObserver, 'note_request',
                side_effect=lambda finger, time_s: dict(accepted=True,
                    kind='INTEGRATION_TEST_REQUEST_ACKNOWLEDGED', finger=finger, time_s=time_s,
                    coverage_credit=False, contact_inferred=False)))
            stack.enter_context(patch.object(ActiveFingerSearchV3, 'begin_posture_attempt', grant))
            stack.enter_context(patch.object(runner.MultiTargetFingerPlanner, 'plan', finger_plan))
            stack.enter_context(patch.object(runner.ContactRecoveryPlanner, 'plan', wrist_plan))
            stack.enter_context(patch.object(runner.SlidingSafetyPlanner, 'command_collision_reason', collision_guard))
            stack.enter_context(patch.object(runner, 'plan_surface_transition_with_prepared',
                side_effect=lambda sp, *args: setattr(sp, 'last_prepared_transition_diagnostics',
                    dict(fallback_invoked=False))))
            stack.enter_context(redirect_stdout(io.StringIO()))
            runner.run(directory, budget_s=.4)
        events = json.loads((directory/'events.json').read_text(encoding='utf-8'))
        with np.load(directory/'trace.npz') as saved:
            trace = {key:saved[key] for key in saved.files}
        return directory, events, trace, calls, grants, roots

    def test_native_pivot_then_measured_replan_has_one_grant_and_persistent_loss(self):
        directory, events, tr, calls, grants, roots = self.case('same_attempt')
        replan = [e for e in events if e['kind']=='COORDINATED_RECOVERY_ACTUAL_ENDPOINT_REPLAN']
        self.assertEqual(len(replan), 1)
        self.assertTrue(replan[0]['accepted'])
        self.assertFalse(replan[0]['new_search_grant'])
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(grants), 1)
        self.assertTrue(grants[0]['accepted'])
        self.assertGreaterEqual(calls[1]-calls[0], .170-1e-9)
        # The second root comes from the physically evolved plant, not the
        # static, deliberately no-op pivot endpoint supplied by the test.
        self.assertGreater(float(np.max(abs(roots[1]-roots[0]))), 1e-8)
        with np.load(directory/'finger_recovery_0001_stage1_root.npz') as state:
            np.testing.assert_array_equal(state['actual_root_q'], roots[1])
        selected = self.selected
        recovery = tr['coordinated_recovery_finger']==selected
        self.assertTrue(np.any(tr['coordinated_recovery_stage'][recovery]==1))
        self.assertTrue(np.all(tr['finger_posture_attempt_id'][recovery,selected]==1))
        self.assertGreater(tr['finger_search_loss_age_s'][-1,selected], .3)
        self.assertTrue(np.all(np.diff(tr['finger_search_loss_age_s'][:,selected])>=-1e-12))
        self.assertFalse(any(e.get('measured_recovery_success',False) for e in events))

    def test_command_geometry_abort_cannot_reactivate_old_pivot_or_stage_two(self):
        _, events, tr, calls, grants, _ = self.case('geometry_abort', geometry_abort=True)
        rejects = [e for e in events if e['kind']=='COMMAND_GEOMETRY_DEFER']
        self.assertEqual(len(rejects), 1)
        self.assertEqual(len(grants), 1)
        self.assertEqual(len(calls), 1)
        self.assertFalse(any(e['kind']=='COORDINATED_RECOVERY_ACTUAL_ENDPOINT_REPLAN' for e in events))
        after = tr['time_s']>rejects[0]['time_s']+.010
        self.assertTrue(np.any(after))
        self.assertTrue(np.all(tr['coordinated_recovery_finger'][after]==-1))
        self.assertFalse(np.any(tr['coordinated_recovery_stage'][after]==1))


class FirstContactControllerIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        native_fixture.ContactHandoffNativeTests.setUpClass()

    def test_real_contact_clears_absorbed_search_bias_and_preserves_applied_preload_once(self):
        fixture = native_fixture.ContactHandoffNativeTests(); fixture.setUp()
        h, data, selected = fixture.h, fixture.data, fixture.contact
        q = joint_state(h,data); p = pose(h,data)
        other = (selected+1)%4
        path = FingerReconfigurationPath(np.stack([q,q+.01]), np.array([1.]),p,p,0.,other,selected)
        execution = FingerRecoveryExecution(path,kind='FINGER_RECONFIGURE',started_s=0.)
        result = execution.update(0.,q,fixture.active,fixture.force,0.)
        self.assertTrue(result['first_contact'])
        self.assertEqual(result['path_elapsed_s'],0.)
        search = ActiveFingerSearchV3(h,SimpleNamespace(mesh=None))
        search.rebase(q)
        search.bias[4*selected:4*selected+4] = [.02,-.01,.03,.01]
        # This is the actual runner's first-contact order. The already applied
        # command contains any prior search/compliance, so clear it before the
        # one new held-reference seed; do not add it a second time.
        nominal = result['nominal_q']
        search.rebase(nominal)
        mcc = ContactPreservingMCC(h,.002,nominal,fixture.previous,fixture.active,fixture.force)
        held = np.r_[np.zeros(7),mcc.held_bias_rad]
        finger, diag = search.update(data,nominal+held,fixture.force,fixture.normals,fixture.active,
                                   enabled=True,search_mask=np.zeros(4,bool))
        command = mcc.command(data,finger-held,fixture.force,fixture.normals,fixture.active,2.)
        baseline = GraceMCC(h,.002).command(data,nominal,fixture.force,fixture.normals,fixture.active,2.)
        columns = fixture.columns
        np.testing.assert_allclose(command[columns],fixture.previous[columns]+baseline[columns]-nominal[columns],atol=3e-16)
        np.testing.assert_array_equal(search.bias,np.zeros(16))
        self.assertFalse(any(diag['seeking']))
        self.assertFalse(result['measured_recovery_success'])


if __name__=='__main__': unittest.main()
