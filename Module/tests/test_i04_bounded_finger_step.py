"""Constraint and useful-motion regressions for the independent IK proposal."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from Module.i04_geometric_traversal.bounded_finger_step import bounded_finger_step


class BoundedFingerStepTests(unittest.TestCase):
    def call(self, jac, motion, nominal=None, bias=None, lower=None, upper=None, **kwargs):
        return bounded_finger_step(jac, motion,
            np.zeros(4) if nominal is None else nominal,
            np.zeros(4) if bias is None else bias,
            np.full(4, -1.) if lower is None else lower,
            np.full(4, 1.) if upper is None else upper, **kwargs)

    def test_available_joint_recovers_motion_lost_by_clipping(self):
        jac = np.array([[.02, .01, 0., 0.], [0., .001, 0., 0.], [0., 0., .02, .01]])
        motion = np.array([8e-6, 0., 0.])
        upper = np.array([0., 1., 1., 1.])
        old = jac.T@np.linalg.solve(jac@jac.T+1e-6*np.eye(3), motion)
        old = np.minimum(old, upper)
        step, diag = self.call(jac, motion, upper=upper)
        self.assertTrue(diag['feasible'])
        self.assertGreater((jac@step)[0], 3.*(jac@old)[0])
        self.assertLess(np.linalg.norm(jac@step-motion), np.linalg.norm(jac@old-motion))
        self.assertLessEqual(np.linalg.norm(jac@step), 8e-6+1e-12)

    def test_nonzero_repair_is_not_scaled_outside_its_joint_box(self):
        jac = np.array([[.01, .01, 0., 0.], [0., 0., .01, 0.], [0., 0., 0., .01]])
        lower = np.array([.0006, -.001, -.001, -.001])
        # Required q0 repair contributes 6 um, but q1 can counteract it.
        step, diag = self.call(jac, [8e-6, 0., 0.], lower=lower, cartesian_step=4e-6)
        self.assertTrue(diag['feasible'])
        self.assertTrue(diag['limit_reconciliation_required'])
        self.assertGreaterEqual(step[0], .0006-1e-12)
        self.assertLessEqual(np.linalg.norm(jac@step), 4e-6+1e-12)
        self.assertLessEqual(np.max(abs(step)), .001+1e-12)

    def test_feasible_optimizer_fallback_is_not_reported_as_optimal(self):
        jac = np.array([[.01, .01, 0., 0.], [0., 0., .01, 0.], [0., 0., 0., .01]])
        def unfinished(fun, x0, **kwargs):
            return SimpleNamespace(x=x0.copy(), success=False, status=9, nit=100, message='Iteration limit')
        with patch('Module.i04_geometric_traversal.bounded_finger_step.minimize', side_effect=unfinished):
            step, diag = self.call(jac, [8e-6, 0., 0.], lower=[.0006, -.001, -.001, -.001], cartesian_step=4e-6)
        self.assertTrue(diag['feasible'])
        self.assertFalse(diag['optimality_confirmed'])
        self.assertFalse(diag['constrained_solver']['success'])
        self.assertLessEqual(np.linalg.norm(jac@step), 4e-6+1e-12)

    def test_impossible_cartesian_repair_reports_hold_not_false_feasibility(self):
        jac = np.zeros((3, 4)); jac[0, 0] = .01
        step, diag = self.call(jac, [8e-6, 0., 0.], lower=[.001, -1., -1., -1.])
        np.testing.assert_array_equal(step, np.zeros(4))
        self.assertFalse(diag['feasible'])
        self.assertEqual(diag['reason'], 'CARTESIAN_LIMIT_REPAIR_INFEASIBLE')
        self.assertAlmostEqual(diag['minimum_cartesian_repair_m'], 1e-5)

    def test_nominal_reconciliation_cannot_jump_over_per_tick_bound(self):
        step, diag = self.call(np.eye(3, 4)*.02, [8e-6, 0., 0.], nominal=[-1.01, 0., 0., 0.])
        np.testing.assert_array_equal(step, np.zeros(4))
        self.assertFalse(diag['feasible'])
        self.assertEqual(diag['reason'], 'EMPTY_JOINT_STEP_BOX')

    def test_bias_bounds_and_hand_limits_both_apply(self):
        step, diag = self.call(np.eye(3, 4)*.02, [8e-6, 0., 0.], bias=[.5, 0., 0., 0.])
        self.assertTrue(diag['feasible'])
        self.assertLessEqual(step[0], 1e-12)

    def test_random_box_solutions_satisfy_independent_convex_optimality_conditions(self):
        rng = np.random.default_rng(714)
        for _ in range(30):
            jac = rng.normal(size=(3, 4))*.02
            motion = rng.normal(size=3)*.0002
            lower, upper = -rng.uniform(.001, .008, 4), rng.uniform(.001, .008, 4)
            step, diag = self.call(jac, motion, lower=lower, upper=upper,
                                   joint_step=.01, cartesian_step=.01)
            self.assertTrue(diag['feasible'])
            gradient = jac.T@(jac@step-motion)+1e-6*step
            at_lower, at_upper = abs(step-lower) < 1e-9, abs(step-upper) < 1e-9
            self.assertTrue(np.all(gradient[at_lower] >= -1e-11))
            self.assertTrue(np.all(gradient[at_upper] <= 1e-11))
            self.assertTrue(np.all(abs(gradient[~(at_lower | at_upper)]) < 1e-11))

    def test_invalid_or_degenerate_inputs_never_return_nan(self):
        for jac in (np.full((3, 4), np.nan), np.zeros((4, 3))):
            step, diag = self.call(jac, [8e-6, 0., 0.])
            np.testing.assert_array_equal(step, np.zeros(4))
            self.assertFalse(diag['feasible'])
        step, diag = self.call(np.zeros((3, 4)), [8e-6, 0., 0.], lower=[.0005, -1., -1., -1.])
        self.assertTrue(diag['feasible'])
        self.assertAlmostEqual(step[0], .0005)
        self.assertEqual(diag['cartesian_step_m'], 0.)


if __name__ == '__main__': unittest.main()
