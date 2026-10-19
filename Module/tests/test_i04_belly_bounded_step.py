"""Shared bounded IK reallocation, with original joint and slew constraints."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from Module.i04_geometric_traversal.belly_bounded_step import bounded_belly_step


class BellyBoundedStepTests(unittest.TestCase):
    def test_blocked_joint_does_not_freeze_available_compensating_joint(self):
        J, error = np.array([[1., 1.]]), np.array([-.01])
        prior = np.array([1., 0.])
        delta, diag = bounded_belly_step(J, error, prior, [-1., -1.], [1., 1.])
        self.assertTrue(diag['feasible'])
        self.assertLessEqual(delta[0], 0.)
        self.assertGreaterEqual(delta[0], -16*np.finfo(float).eps)
        self.assertGreater(delta[1], .0023)
        self.assertLess(np.linalg.norm(error+J@delta), np.linalg.norm(error))
        # The former global-alpha strategy would be exactly zero here.
        old = -J.T@np.linalg.solve(J@J.T+.001**2*np.eye(1), .45*error)
        self.assertGreater(old[0], 0.)
        old_alpha = min(1., (1.-prior[0])/old[0])
        self.assertEqual(old_alpha, 0.)

    def test_no_constraint_matches_original_damped_shared_solution(self):
        rng = np.random.default_rng(19)
        J = rng.normal(size=(18, 19))*.02
        error = rng.normal(size=18)*1e-6
        expected = -J.T@np.linalg.solve(J@J.T+.001**2*np.eye(18), .45*error)
        actual, diag = bounded_belly_step(J, error, np.zeros(19), -np.ones(19), np.ones(19))
        self.assertTrue(diag['optimality_confirmed'])
        np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=1e-9)

    def test_shared_orientation_row_is_fitted_by_another_joint_after_limit(self):
        # First row stands for position, second for orientation. Joint 0 is
        # blocked; joint 1 must replace it, while joint 2 cancels orientation.
        J = np.array([[1., 1., 0.], [1., -1., 1.]])
        error = np.array([-.001, 0.])
        delta, diag = bounded_belly_step(J, error, [1., 0., 0.], [-1.]*3, [1.]*3, gain=1.)
        self.assertTrue(diag['feasible'])
        self.assertLessEqual(delta[0], 0.)
        self.assertGreaterEqual(delta[0], -16*np.finfo(float).eps)
        self.assertGreater(delta[1], .00099)
        self.assertGreater(delta[2], .00099)
        self.assertLess(abs((J@delta)[1]), 2e-9)

    def test_full_18_by_19_problem_obeys_original_range_and_per_joint_slew(self):
        rng = np.random.default_rng(7)
        J, error = rng.normal(size=(18, 19)), rng.normal(size=18)
        prior = np.linspace(-.9, .9, 19)
        lower, upper = -np.ones(19), np.ones(19)
        lower[3] = prior[3]; upper[11] = prior[11]
        limits = np.r_[np.full(7, .0004), np.full(12, .0024)]
        delta, diag = bounded_belly_step(J, error, prior, lower, upper, maximum_joint_step=limits)
        self.assertTrue(diag['feasible'])
        self.assertTrue(np.all(prior+delta >= lower))
        self.assertTrue(np.all(prior+delta <= upper))
        self.assertTrue(np.all(abs(delta) <= limits))
        self.assertGreaterEqual(delta[3], 0.)
        self.assertLessEqual(delta[11], 0.)
        self.assertLess(diag['predicted_residual_after'], diag['predicted_residual_before'])

    def test_exactly_fixed_joint_is_eliminated_without_inventing_motion(self):
        delta, diag = bounded_belly_step([[1., 1.]], [-.001], [0., 0.], [0., -1.], [0., 1.])
        self.assertTrue(diag['feasible'])
        self.assertEqual(diag['fixed_joint_indices'], [0])
        self.assertEqual(delta[0], 0.)
        self.assertGreater(delta[1], 0.)
        delta, diag = bounded_belly_step([[1., 1.]], [-1.], [0., 0.], [0., 0.], [0., 0.])
        np.testing.assert_array_equal(delta, [0., 0.])
        self.assertTrue(diag['feasible'])
        self.assertEqual(diag['reason'], 'HOLD_NO_FEASIBLE_LINEAR_MOTION')

    def test_outside_prior_is_explicit_failure_not_hidden_range_repair(self):
        delta, diag = bounded_belly_step([[1.]], [-.01], [1.000001], [-1.], [1.])
        self.assertFalse(diag['feasible'])
        self.assertEqual(diag['reason'], 'PRIOR_REFERENCE_OUTSIDE_JOINT_BOX')
        np.testing.assert_array_equal(delta, [0.])

    def test_failed_solver_is_not_accepted_merely_because_it_returns_a_vector(self):
        failed = SimpleNamespace(success=False, x=np.array([.001]), status=0, nit=100,
                                 optimality=1., message='iteration limit')
        with patch('Module.i04_geometric_traversal.belly_bounded_step.lsq_linear', return_value=failed):
            delta, diag = bounded_belly_step([[1.]], [-1.], [0.], [-1.], [1.])
        self.assertFalse(diag['feasible'])
        np.testing.assert_array_equal(delta, [0.])

    def test_only_roundoff_is_corrected_by_one_shared_scalar_and_never_clipping(self):
        raw = np.array([.0024*(1.+8e-16), .001])
        fit = SimpleNamespace(success=True, x=raw.copy(), status=1, nit=1,
                              optimality=0., message='mock saturated solution with roundoff')
        with patch('Module.i04_geometric_traversal.belly_bounded_step.lsq_linear', return_value=fit):
            delta, diag = bounded_belly_step(np.eye(2), [-.01, -.01], [0., 0.], [-1., -1.], [1., 1.])
        self.assertTrue(diag['feasible'])
        self.assertGreater(diag['numerical_uniform_scale'], 1.-1e-10)
        self.assertLess(diag['numerical_uniform_scale'], 1.)
        np.testing.assert_array_equal(delta, raw*diag['numerical_uniform_scale'])
        self.assertTrue(np.all(delta <= .0024))
        fit.x = np.array([.0025, .001])
        with patch('Module.i04_geometric_traversal.belly_bounded_step.lsq_linear', return_value=fit):
            delta, diag = bounded_belly_step(np.eye(2), [-.01, -.01], [0., 0.], [-1., -1.], [1., 1.])
        self.assertFalse(diag['feasible'])
        np.testing.assert_array_equal(delta, [0., 0.])
        fit.x = np.array([1e-20, .001])
        with patch('Module.i04_geometric_traversal.belly_bounded_step.lsq_linear', return_value=fit):
            delta, diag = bounded_belly_step(np.eye(2), [-.01, -.01], [1., 0.], [-1., -1.], [1., 1.])
        self.assertFalse(diag['feasible'], 'A blocked-direction alpha=0 is not a roundoff repair')

    def test_invalid_input_and_nonmutation(self):
        J, error, prior, lo, hi = np.eye(2), np.ones(2), np.zeros(2), -np.ones(2), np.ones(2)
        originals = [x.copy() for x in (J, error, prior, lo, hi)]
        bounded_belly_step(J, error, prior, lo, hi)
        for actual, expected in zip((J, error, prior, lo, hi), originals):
            np.testing.assert_array_equal(actual, expected)
        for kwargs in ({'gain':1.1}, {'damping':0.}, {'maximum_joint_step':float('nan')}):
            delta, diag = bounded_belly_step(J, error, prior, lo, hi, **kwargs)
            self.assertFalse(diag['feasible'])
            np.testing.assert_array_equal(delta, np.zeros(2))

    def test_unilateral_zero_bound_roundoff_stays_inside_original_box(self):
        def with_edge_roundoff(a, b, *, bounds, **kwargs):
            lo, hi = bounds
            # Reproduce a tiny BVLS undershoot/overshoot at each active bound.
            self.assertGreater(lo[0], 0.)
            self.assertLess(hi[1], 0.)
            result = np.array([lo[0]-1e-19, hi[1]+1e-19])
            return SimpleNamespace(success=True, x=result, status=1, nit=2,
                                   optimality=0., message='edge roundoff')
        with patch('Module.i04_geometric_traversal.belly_bounded_step.lsq_linear', side_effect=with_edge_roundoff):
            delta, diag = bounded_belly_step(np.eye(2), [1., -1.], [-1., 1.], [-1., -1.], [1., 1.])
        self.assertTrue(diag['feasible'])
        self.assertGreater(delta[0], 0.)
        self.assertLess(delta[1], 0.)
        self.assertEqual(diag['numerical_uniform_scale'], 1.)
        self.assertLess(np.max(abs(delta)), 4e-15)

    def test_one_hundred_finite_shared_cases_preserve_original_bounds(self):
        rng = np.random.default_rng(20261017)
        for case in range(100):
            J = rng.normal(size=(18, 19))*rng.uniform(.001, 1.)
            error = rng.normal(size=18)*rng.uniform(1e-5, .1)
            lower, upper = -np.ones(19), np.ones(19)
            previous = rng.uniform(-1., 1., size=19)
            previous[rng.choice(19, 5, replace=False)] = -1.
            previous[rng.choice(19, 5, replace=False)] = 1.
            fixed = int(rng.integers(19)); lower[fixed] = upper[fixed] = previous[fixed]
            delta, diag = bounded_belly_step(J, error, previous, lower, upper)
            self.assertTrue(diag['feasible'], (case, diag))
            self.assertTrue(np.all(previous+delta >= lower))
            self.assertTrue(np.all(previous+delta <= upper))
            self.assertTrue(np.all(abs(delta) <= .0024))
            self.assertEqual(delta[fixed], 0.)


if __name__ == '__main__': unittest.main()
