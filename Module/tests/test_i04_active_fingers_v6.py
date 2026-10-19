"""Gradient magnitude, unchanged bounds, and finite native FK progress checks."""
import copy
from dataclasses import fields, replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from Module.tests import test_i04_active_fingers as native_fixture
from Module.tests import test_i04_active_fingers_v5 as lifecycle_fixture
from Module.i04_geometric_traversal.active_fingers import ActiveFingerSearchConfig
from Module.i04_geometric_traversal.active_fingers_v4 import support_point_jacobians
from Module.i04_geometric_traversal.active_fingers_v5 import closest_finger_pair
from Module.i04_geometric_traversal.active_fingers_v6 import ActiveFingerSearchV6, bounded_damped_normal_step
from Module.i04_geometric_traversal.full_surface import pad_support_vector
from Module.i04_geometric_traversal.physics import set_joint_state


class V6LifecycleTests(lifecycle_fixture.V5LifecycleTests):
    """Separate the persistent lifecycle from the finite-FK tests below."""
    def configure(self, config=None):
        self.search = ActiveFingerSearchV6(self.h, self.graph, config=config)
        strategy = patch.object(self.search, '_normal_search_step', side_effect=
            lambda data, finger, *args, **kwargs: bounded_damped_normal_step(*args, **kwargs))
        strategy.start(); self.addCleanup(strategy.stop)

    def setUp(self):
        super().setUp()
        self.configure()

    def test_flicker_and_stroke_rebase_cannot_refresh_exhausted_search(self):
        self.configure(replace(ActiveFingerSearchConfig(), maximum_search_time_s=.12))
        result, _ = self.prime(count=55)
        for _ in range(8): _, diag = self.tick()
        self.assertEqual(diag['state'], ['SEARCH_LIMIT']*4)
        self.active[:] = True; self.forces[:] = 1.; self.normals[:] = [0., 0., 1.]
        for _ in range(10): self.tick()
        self.active[:] = False; self.forces[:] = 0.
        self.search.rebase(result)
        _, diag = self.tick(nominal=result)
        self.assertEqual(diag['state'], ['SEARCH_LIMIT']*4)


class DampedNormalBoundsTests(unittest.TestCase):
    def solve(self, center, support, *, lower=None, bias=None):
        return bounded_damped_normal_step(center, support, [1., 0., 0.], np.zeros(4),
            np.zeros(4) if bias is None else bias,
            np.full(4, -1.) if lower is None else lower, np.ones(4))

    def test_weak_gradient_reduces_step_instead_of_being_renormalized(self):
        deltas = []
        for gradient in (1e-4, 1e-5, 1e-6):
            jac = np.zeros((3, 4)); jac[0, 0] = gradient
            delta, diag = self.solve(jac, jac)
            self.assertTrue(diag['feasible'])
            expected = -8e-6*gradient/(gradient**2+.001**2)
            self.assertAlmostEqual(delta[0], expected, delta=1e-12)
            deltas.append(abs(delta[0]))
        self.assertGreater(deltas[0]/deltas[-1], 90.)
        self.assertLess(deltas[-1], .00001)

    def test_center_and_support_bounds_are_both_checked(self):
        center = np.eye(3, 4)*.1; support = np.eye(3, 4)*.001
        delta, diag = self.solve(center, support)
        self.assertTrue(diag['feasible'])
        self.assertLessEqual(np.linalg.norm(center@delta), 8e-6+1e-12)
        self.assertLessEqual(np.linalg.norm(support@delta), 8e-6+1e-12)
        self.assertEqual(diag['charged_step_m'], max(np.linalg.norm(center@delta), np.linalg.norm(support@delta)))

    def test_impossible_nonzero_repair_is_not_converted_to_feasible_hold(self):
        delta, diag = self.solve(np.eye(3, 4), np.eye(3, 4)*.001,
                                lower=np.array([.0005, -1., -1., -1.]))
        self.assertFalse(diag['feasible'])
        np.testing.assert_array_equal(delta, np.zeros(4))

    def test_weak_or_blocked_direction_has_explicit_zero_charge(self):
        jac = np.zeros((3, 4)); jac[0, 0] = 1e-12
        delta, diag = self.solve(jac, jac)
        self.assertTrue(diag['feasible'])
        np.testing.assert_array_equal(delta, np.zeros(4))
        self.assertEqual(diag['charged_step_m'], 0.)

    def test_nonfinite_input_returns_infeasible_hold(self):
        delta, diag = self.solve(np.full((3, 4), np.nan), np.eye(3, 4))
        self.assertFalse(diag['feasible']); np.testing.assert_array_equal(delta, np.zeros(4))


class FiniteProgressTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): native_fixture.ActiveFingerNativeGeometryTests.setUpClass()

    def setUp(self):
        fixture = native_fixture.ActiveFingerNativeGeometryTests(); fixture.setUp()
        # A private native model lets this finite-geometry fixture move the
        # plane without changing any shared model or any experiment source.
        values = {field.name:getattr(fixture.h, field.name) for field in fields(fixture.h)}
        values['model'] = copy.copy(fixture.h.model)
        values['object_geom_ids'] = np.array([fixture.h.object_geom_id])
        self.h = SimpleNamespace(**values); self.q = fixture.q.copy()
        self.data = mujoco.MjData(self.h.model); set_joint_state(self.h, self.data, self.q)
        geom = int(self.h.tip_geom_ids[0]); n = np.array([0., 0., 1.])
        support = pad_support_vector(self.h.model, geom, self.data.geom_xmat[geom].reshape(3, 3), n)
        self.h.model.geom_pos[self.h.object_geom_id, 2] = self.data.geom_xpos[geom, 2]-support[2]-.006
        mujoco.mj_forward(self.h.model, self.data)
        self.search = ActiveFingerSearchV6(self.h, fixture.graph)
        self.columns = self.search.columns[0]

    def step(self):
        pair = closest_finger_pair(self.h, self.data, 0)
        center, support = support_point_jacobians(self.h, self.data, 0, self.columns, pair['normal_world'])
        return self.search._normal_search_step(self.data, 0, center, support, pair['normal_world'],
            self.q[self.columns+7], np.zeros(4), self.search.lower[self.columns], self.search.upper[self.columns],
            joint_step=.001, cartesian_step=8e-6, maximum_bias=.5, damping=.001)

    def test_finite_check_improves_both_geometry_references_without_changing_plant(self):
        initial = (self.data.qpos.copy(), self.data.qvel.copy(), self.data.ctrl.copy(), self.data.time)
        delta, diag = self.step()
        self.assertTrue(diag['feasible'])
        self.assertGreater(np.linalg.norm(delta), 0.)
        self.assertEqual(diag['reason'], 'DAMPED_NORMAL_FINITE_PROGRESS')
        last = diag['backtracking_trials'][-1]
        self.assertGreaterEqual(last['actual_posture_gap_gain_m'], 2e-8)
        self.assertGreaterEqual(last['previous_reference_gap_gain_m'], 2e-8)
        for actual, original in zip((self.data.qpos, self.data.qvel, self.data.ctrl, self.data.time), initial):
            np.testing.assert_array_equal(actual, original)

    def test_failed_finite_progress_holds_without_charging_search_distance(self):
        actual = closest_finger_pair(self.h, self.data, 0)
        with patch('Module.i04_geometric_traversal.active_fingers_v6.closest_finger_pair', return_value=actual):
            delta, diag = self.step()
        self.assertTrue(diag['feasible'])
        self.assertEqual(diag['reason'], 'NO_FINITE_LOCAL_PROGRESS_HOLD')
        self.assertEqual(len(diag['backtracking_trials']), 7)
        self.assertEqual(diag['charged_step_m'], 0.)
        np.testing.assert_array_equal(delta, np.zeros(4))

    def test_negative_reference_progress_cannot_hide_behind_actual_pose_progress(self):
        actual = closest_finger_pair(self.h, self.data, 0)
        calls = [dict(actual), dict(actual)]
        for _ in range(7):
            calls.extend((dict(actual, gap_m=actual['gap_m']-1e-6), dict(actual, gap_m=actual['gap_m']+1e-6)))
        with patch('Module.i04_geometric_traversal.active_fingers_v6.closest_finger_pair', side_effect=calls):
            delta, diag = self.step()
        self.assertEqual(diag['reason'], 'NO_FINITE_LOCAL_PROGRESS_HOLD')
        np.testing.assert_array_equal(delta, np.zeros(4))


if __name__ == '__main__': unittest.main()
