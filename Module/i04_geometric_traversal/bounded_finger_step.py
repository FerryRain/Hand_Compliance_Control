"""Standalone constrained finger increment; not used by the frozen controller.

The returned increment is usable only when diagnostics['feasible'] is true.
An infeasible limit repair returns zero and an explicit reason: zero is then a
hold, not a claim that the old reference meets the requested joint bounds.
"""
import numpy as np
from scipy.optimize import lsq_linear, minimize


def _box_step(jacobian, motion, lower, upper, damping):
    # Augmentation preserves the damped objective without forming J.T @ J.
    # BVLS handles this tiny dense problem directly, while exhaustive 81-set
    # enumeration is retained as an independent offline diagnostic oracle.
    a = np.vstack((jacobian, damping*np.eye(4)))
    b = np.r_[motion, np.zeros(4)]
    result = lower.copy()
    free = (upper-lower) > 1e-14
    if np.any(free):
        fit = lsq_linear(a[:, free], b-a[:, ~free]@result[~free],
                         bounds=(lower[free], upper[free]), method='bvls', tol=1e-14, max_iter=100)
        if not fit.success or not np.all(np.isfinite(fit.x)):
            return None
        result[free] = fit.x
    return np.clip(result, lower, upper)


def bounded_finger_step(jacobian, motion, nominal, previous_bias, joint_lower, joint_upper,
                        *, joint_step=.001, cartesian_step=.000008, maximum_bias=.50, damping=.001):
    """Return ``(delta[4], diagnostics)`` with joint, bias and Cartesian bounds.

    The small box-constrained damped least-squares solution reallocates motion
    to available joints. For a reference that requires a nonzero limit repair,
    the Cartesian constraint is solved inside that same box; scaling toward
    zero would violate the repair bound and is deliberately not used.
    """
    zero = np.zeros(4)
    diag = dict(feasible=False, reason='INVALID_INPUT', limit_reconciliation_required=False,
                optimality_confirmed=False)
    try:
        jacobian, motion = np.asarray(jacobian, dtype=float), np.asarray(motion, dtype=float)
        nominal, previous_bias, joint_lower, joint_upper = (
            np.asarray(x, dtype=float) for x in (nominal, previous_bias, joint_lower, joint_upper))
        values = (jacobian, motion, nominal, previous_bias, joint_lower, joint_upper)
        if (jacobian.shape != (3, 4) or motion.shape != (3,)
                or any(x.shape != (4,) for x in values[2:])
                or not all(np.all(np.isfinite(x)) for x in values)
                or any(not np.isfinite(x) or x <= 0 for x in
                       (joint_step, cartesian_step, maximum_bias, damping))
                or np.any(joint_lower > joint_upper)):
            return zero, diag
        lower = np.maximum.reduce((np.full(4, -joint_step), -maximum_bias-previous_bias,
                                   joint_lower-nominal-previous_bias))
        upper = np.minimum.reduce((np.full(4, joint_step), maximum_bias-previous_bias,
                                   joint_upper-nominal-previous_bias))
        diag.update(lower_step_rad=lower.tolist(), upper_step_rad=upper.tolist(),
                    limit_reconciliation_required=bool(np.any(lower > 0.) or np.any(upper < 0.)))
        if np.any(lower > upper):
            return zero, dict(diag, reason='EMPTY_JOINT_STEP_BOX')
        motion = motion*min(1., cartesian_step/max(float(np.linalg.norm(motion)), 1e-30))
        result = _box_step(jacobian, motion, lower, upper, damping)
        if result is None:
            return zero, dict(diag, reason='NO_FINITE_BOX_SOLUTION')
        reason = 'BOUNDED_DLS'
        if np.linalg.norm(jacobian@result) > cartesian_step:
            if not diag['limit_reconciliation_required']:
                result *= cartesian_step/np.linalg.norm(jacobian@result)
                reason = 'BOUNDED_DLS_CARTESIAN_SCALED'
            else:
                # Establish feasibility independently of the desired motion.
                # Scale variables/residuals so micron constraints are resolved
                # by the numerical optimizer, rather than its default tolerance.
                a = jacobian*joint_step/cartesian_step
                lo, hi = lower/joint_step, upper/joint_step
                fixed = (hi-lo) <= 1e-13
                free = ~fixed
                minimum = lo.copy()
                if np.any(free):
                    fit = lsq_linear(a[:, free], -a[:, fixed]@minimum[fixed],
                                     bounds=(lo[free], hi[free]), method='bvls', tol=1e-12)
                    minimum[free] = fit.x
                    diag['minimum_repair_solver'] = dict(success=bool(fit.success),
                        status=int(fit.status), optimality=float(fit.optimality), message=str(fit.message))
                else:
                    diag['minimum_repair_solver'] = dict(success=True, status=0,
                        optimality=0., message='All step bounds are fixed')
                if np.linalg.norm(a@minimum) > 1.+1e-9:
                    reason = ('CARTESIAN_LIMIT_REPAIR_INFEASIBLE' if diag['minimum_repair_solver']['success']
                              else 'CARTESIAN_REPAIR_FEASIBILITY_UNRESOLVED')
                    return zero, dict(diag, reason=reason,
                        minimum_cartesian_repair_m=float(np.linalg.norm(a@minimum)*cartesian_step))
                desired = motion/cartesian_step
                regularization = (damping*joint_step/cartesian_step)**2
                def objective(x):
                    residual = a@x-desired
                    return float(residual@residual+regularization*(x@x))
                def gradient(x):
                    return 2.*(a.T@(a@x-desired)+regularization*x)
                fit = minimize(objective, minimum, jac=gradient, method='SLSQP',
                    bounds=list(zip(lo, hi)), constraints=[dict(type='ineq',
                        fun=lambda x:1.-float((a@x)@(a@x)), jac=lambda x:-2.*a.T@(a@x))],
                    options=dict(ftol=1e-12, maxiter=100))
                diag['constrained_solver'] = dict(success=bool(fit.success), status=int(fit.status),
                    iterations=int(fit.nit), message=str(fit.message))
                proposed = np.clip(fit.x, lo, hi)
                if np.all(np.isfinite(proposed)) and np.linalg.norm(a@proposed) <= 1.+1e-9:
                    result = proposed*joint_step
                    reason = 'BOUNDED_DLS_WITH_LIMIT_REPAIR'
                else:
                    result = minimum*joint_step
                    reason = 'FEASIBLE_MINIMUM_CARTESIAN_REPAIR'
        if (not np.all(np.isfinite(result)) or np.any(result < lower-1e-12)
                or np.any(result > upper+1e-12)
                or np.linalg.norm(jacobian@result) > cartesian_step+1e-12):
            return zero, dict(diag, reason='NUMERICAL_CONSTRAINT_FAILURE')
        return result, dict(diag, feasible=True, reason=reason,
                            optimality_confirmed=bool(reason == 'BOUNDED_DLS' or
                                (reason == 'BOUNDED_DLS_WITH_LIMIT_REPAIR' and
                                 diag['constrained_solver']['success'])),
                            cartesian_step_m=float(np.linalg.norm(jacobian@result)),
                            maximum_joint_step_rad=float(np.max(abs(result))))
    except (ValueError, TypeError, FloatingPointError, np.linalg.LinAlgError):
        return zero, dict(diag, reason='INVALID_INPUT_OR_NUMERICAL_FAILURE')
