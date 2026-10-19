"""Box-constrained multi-finger belly IK increment, independent of the plant.

A saturated joint stays within its box while the other joints can compensate.
This solves the shared position/orientation least-squares problem; it does not
clip an unconstrained solution component by component or rescale it to zero
because one joint is blocked. Linear fit improvement is not a native force or
nonlinear belly-face guarantee. The caller retains FK, collision, force and
tracking validation, and must hold/abort explicitly if feasible is false.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import lsq_linear


def bounded_belly_step(jacobian, error, previous, joint_lower, joint_upper, *,
                       gain=.45, damping=.001, maximum_joint_step=.0024,
                       maximum_iterations=100):
    """Return ``delta[N], diag`` for min ||J delta + gain*error||²+λ²||delta||².

    Bounds are the intersection of the original joint range with the per-tick
    slew box about ``previous``. ``maximum_joint_step`` is a positive scalar or
    N-vector. A previous reference already outside the joint box is an explicit
    failure; no hidden repair, rebase, or changed budget is introduced here.
    Fixed variables are eliminated before BVLS, including a completely fixed
    box. Nonfixed solver bounds reserve at most 16 machine eps times the prior
    reference scale (and no more than a quarter of their width) inside each
    original edge. This prevents zero-edge roundoff from acquiring the wrong
    sign. Final checks still use the ORIGINAL box. The solver returns a new
    vector and never mutates any input.
    """
    prior = np.asarray(previous, dtype=float)
    zero = np.zeros(prior.size if prior.ndim == 1 else 0)
    diag = dict(feasible=False, optimality_confirmed=False, reason='INVALID_INPUT',
                authority='LINEAR_GEOMETRY_STEP_ONLY_NO_CONTACT_OR_COVERAGE')
    try:
        J, error = np.asarray(jacobian, float), np.asarray(error, float)
        lower, upper = np.asarray(joint_lower, float), np.asarray(joint_upper, float)
        n = prior.size
        step = np.broadcast_to(np.asarray(maximum_joint_step, float), (n,)).copy()
        if (prior.ndim != 1 or not n or lower.shape != (n,) or upper.shape != (n,)
                or J.ndim != 2 or J.shape[1] != n or not J.shape[0]
                or error.shape != (J.shape[0],)
                or not all(np.isfinite(x).all() for x in (J, error, prior, lower, upper, step))
                or np.any(lower > upper) or np.any(step <= 0.)
                or not np.isfinite(gain) or not 0. < gain <= 1.
                or not np.isfinite(damping) or damping <= 0.
                or not isinstance(maximum_iterations, (int, np.integer)) or maximum_iterations < 1):
            return zero, diag
        if np.any(prior < lower) or np.any(prior > upper):
            return zero, dict(diag, reason='PRIOR_REFERENCE_OUTSIDE_JOINT_BOX')
        lo = np.maximum(-step, lower-prior)
        hi = np.minimum(step, upper-prior)
        diag.update(lower_step_rad=lo.tolist(), upper_step_rad=hi.tolist())
        # Damping augmentation avoids squaring the condition number in J.T@J.
        a = np.vstack((J, damping*np.eye(n)))
        b = np.r_[-gain*error, np.zeros(n)]
        fixed = lo == hi
        free = ~fixed
        reserve = np.zeros(n)
        reserve[free] = np.minimum((hi[free]-lo[free])*.25,
            16.*np.finfo(float).eps*np.maximum(1., abs(prior[free])))
        solver_lo, solver_hi = lo+reserve, hi-reserve
        diag.update(numerical_bound_reserve_rad=reserve.tolist(),
                    solver_lower_step_rad=solver_lo.tolist(),
                    solver_upper_step_rad=solver_hi.tolist(),
                    optimality_domain='NUMERICALLY_INSET_ORIGINAL_JOINT_STEP_BOX')
        delta = lo.copy()
        if np.any(free):
            fit = lsq_linear(a[:, free], b-a[:, fixed]@delta[fixed],
                bounds=(solver_lo[free], solver_hi[free]), method='bvls', tol=1e-12,
                max_iter=int(maximum_iterations))
            diag['solver'] = dict(success=bool(fit.success), status=int(fit.status),
                iterations=int(fit.nit), optimality=float(fit.optimality), message=str(fit.message))
            if not fit.success or not np.isfinite(fit.x).all():
                return zero, dict(diag, reason='NO_CONVERGED_FINITE_BOX_SOLUTION')
            delta[free] = fit.x
        else:
            diag['solver'] = dict(success=True, status=0, iterations=0,
                                  optimality=0., message='Every joint step is fixed')
        # BVLS occasionally returns a saturated value a few ulps outside its
        # box. Correct only that roundoff by ONE common scalar. Because prior
        # is in the box, zero is feasible and any convex scaling toward zero
        # preserves all constraints of an otherwise feasible vector. This is
        # not the old global scaling of an unconstrained DLS solution: any
        # material correction or blocked-direction alpha=0 is still rejected.
        raw_delta = delta.copy()
        numerical_scale = 1.
        if np.isfinite(delta).all():
            for j, value in enumerate(delta):
                if value > 0.:
                    numerical_scale = min(numerical_scale, float(hi[j]/value))
                elif value < 0.:
                    numerical_scale = min(numerical_scale, float(lo[j]/value))
        if numerical_scale < 1.:
            if not 1.-1e-10 <= numerical_scale < 1.:
                return zero, dict(diag, reason='NUMERICAL_JOINT_BOX_FAILURE',
                                  required_uniform_scale=float(numerical_scale))
            numerical_scale = float(np.nextafter(numerical_scale, 0.))
            delta = raw_delta*numerical_scale
        # A final strict check keeps the numerical correction from becoming
        # an implicit tolerance on joint limits, slew, or the applied command.
        command = prior+delta
        if (not np.isfinite(delta).all() or np.any(delta < lo) or np.any(delta > hi)
                or np.any(command < lower) or np.any(command > upper)
                or np.any(abs(delta) > step)):
            return zero, dict(diag, reason='NUMERICAL_JOINT_BOX_FAILURE',
                              required_uniform_scale=float(numerical_scale))
        before = float(np.linalg.norm(error))
        after = float(np.linalg.norm(error+J@delta))
        if after > before+1e-12:
            return zero, dict(diag, reason='NO_LINEAR_RESIDUAL_IMPROVEMENT')
        return delta, dict(diag, feasible=True, optimality_confirmed=(numerical_scale == 1.),
            reason='BOUNDED_SHARED_DLS' if np.any(delta != 0.) else 'HOLD_NO_FEASIBLE_LINEAR_MOTION',
            numerical_uniform_scale=float(numerical_scale),
            predicted_residual_before=before, predicted_residual_after=after,
            scaled_fit_residual=float(np.linalg.norm(J@delta+gain*error)),
            maximum_joint_step_rad=float(np.max(abs(delta))),
            lower_bound_joint_indices=np.flatnonzero(delta == lo).tolist(),
            upper_bound_joint_indices=np.flatnonzero(delta == hi).tolist(),
            fixed_joint_indices=np.flatnonzero(fixed).tolist())
    except (ValueError, TypeError, FloatingPointError, np.linalg.LinAlgError):
        return zero, dict(diag, reason='INVALID_INPUT_OR_NUMERICAL_FAILURE')
