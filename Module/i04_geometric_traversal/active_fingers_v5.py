"""Independent normal-priority search candidate; V3/V4 and whole run frozen.

The objective is to reduce the actual collision pair's signed separation at
first order. Tangential motion is permitted within both original Cartesian
speed bounds but does not compete as a target error. A feasible inward linear
step is not a guarantee of actual nonlinear contact; native guards and measured
150 ms contact confirmation remain mandatory.
"""
from __future__ import annotations

import mujoco
import numpy as np
from scipy.optimize import minimize

from .active_fingers_v4 import ActiveFingerSearchV4
from .full_surface import pad_support_vector


def bounded_normal_finger_step(center_jacobian, support_jacobian, outward_normal,
                               nominal, previous_bias, joint_lower, joint_upper,
                               *, joint_step=.001, cartesian_step=.000008,
                               maximum_bias=.50, damping=.001):
    """Maximize inward linear gap progress subject to original finite bounds.

    ``damping`` remains an accepted API parameter but the objective deliberately
    differs from vector DLS. A tiny joint-norm tie-break avoids arbitrary null
    motion; feasibility and first-order inward motion are checked separately
    from the optimizer's success flag. No legal inward step yields an explicit
    zero hold if zero is in the box. A required nonzero repair cannot use that
    fallback and instead requests an outer measured rebase.
    """
    zero = np.zeros(4)
    diag = dict(feasible=False, reason='INVALID_INPUT', optimality_confirmed=False,
                cartesian_bound_metric='BOTH_CENTER_AND_SUPPORT')
    try:
        center, support, normal = (np.asarray(x, dtype=float) for x in
                                  (center_jacobian, support_jacobian, outward_normal))
        nominal, bias, joint_lower, joint_upper = (np.asarray(x, dtype=float) for x in
            (nominal, previous_bias, joint_lower, joint_upper))
        if (center.shape != (3, 4) or support.shape != (3, 4) or normal.shape != (3,)
                or any(x.shape != (4,) for x in (nominal, bias, joint_lower, joint_upper))
                or not all(np.isfinite(x).all() for x in
                    (center, support, normal, nominal, bias, joint_lower, joint_upper))
                or np.linalg.norm(normal) < .5 or np.any(joint_lower > joint_upper)
                or any(not np.isfinite(x) or x <= 0 for x in
                       (joint_step, cartesian_step, maximum_bias, damping))):
            return zero, diag
        normal = normal/np.linalg.norm(normal)
        lower = np.maximum.reduce((np.full(4, -joint_step), -maximum_bias-bias, joint_lower-nominal-bias))
        upper = np.minimum.reduce((np.full(4, joint_step), maximum_bias-bias, joint_upper-nominal-bias))
        zero_feasible = bool(np.all(lower <= 0.) and np.all(upper >= 0.))
        diag.update(lower_step_rad=lower.tolist(), upper_step_rad=upper.tolist(),
                    limit_reconciliation_required=not zero_feasible)
        if np.any(lower > upper):
            return zero, dict(diag, reason='EMPTY_JOINT_STEP_BOX')
        gradient = normal@support
        scale = float(np.linalg.norm(gradient))
        if scale <= 1e-15:
            if zero_feasible:
                return zero, dict(diag, feasible=True, reason='NO_LEGAL_INWARD_STEP',
                    center_step_m=0., support_step_m=0., charged_step_m=0., predicted_gap_change_m=0.)
            return zero, dict(diag, reason='NO_INWARD_REPAIR_DIRECTION')
        ac, ap = center*joint_step/cartesian_step, support*joint_step/cartesian_step
        lo, hi = lower/joint_step, upper/joint_step
        direction = gradient/scale
        regularization = 1e-6
        fit = minimize(lambda u: float(direction@u+regularization*(u@u)),
            np.clip(np.zeros(4), lo, hi), jac=lambda u: direction+2*regularization*u,
            bounds=list(zip(lo, hi)), method='SLSQP',
            constraints=[dict(type='ineq', fun=lambda u: 1.-float((ac@u)@(ac@u)),
                              jac=lambda u: -2.*ac.T@(ac@u)),
                         dict(type='ineq', fun=lambda u: 1.-float((ap@u)@(ap@u)),
                              jac=lambda u: -2.*ap.T@(ap@u))],
            options=dict(ftol=1e-12, maxiter=100))
        # SciPy's exact all-variables-fixed result omits status/nit.
        diag['constrained_solver'] = dict(success=bool(fit.success), status=getattr(fit, 'status', None),
            iterations=int(getattr(fit, 'nit', 0)), message=str(fit.message))
        delta = np.clip(np.asarray(fit.x)*joint_step, lower, upper)
        valid = bool(np.isfinite(delta).all())
        scaled = False
        if valid:
            traveled = max(float(np.linalg.norm(center@delta)), float(np.linalg.norm(support@delta)))
            if traveled > cartesian_step:
                candidate = delta*(cartesian_step/traveled)
                # This also handles tiny optimizer constraint errors without
                # silently invalidating a required nonzero joint repair.
                if np.any(candidate < lower-1e-12) or np.any(candidate > upper+1e-12): valid = False
                else: delta, scaled = candidate, True
            valid = valid and bool(np.all(delta >= lower-1e-12) and np.all(delta <= upper+1e-12)
                and max(np.linalg.norm(center@delta), np.linalg.norm(support@delta)) <= cartesian_step+1e-12)
        progress = float(gradient@delta) if valid else np.inf
        if not valid or progress >= -1e-12:
            if zero_feasible:
                return zero, dict(diag, feasible=True, reason='NO_LEGAL_INWARD_STEP_FOUND',
                    center_step_m=0., support_step_m=0., charged_step_m=0., predicted_gap_change_m=0.)
            return zero, dict(diag, reason='NO_FEASIBLE_INWARD_LIMIT_REPAIR')
        center_step, support_step = float(np.linalg.norm(center@delta)), float(np.linalg.norm(support@delta))
        return delta, dict(diag, feasible=True, reason='BOUNDED_CLOSEST_NORMAL_PRIORITY',
            optimality_confirmed=bool(fit.success and not scaled), scaled_for_cartesian_bound=scaled,
            center_step_m=center_step, support_step_m=support_step, cartesian_step_m=support_step,
            charged_step_m=max(center_step, support_step), predicted_gap_change_m=progress,
            maximum_joint_step_rad=float(np.max(abs(delta))))
    except (ValueError, TypeError, FloatingPointError, np.linalg.LinAlgError):
        return zero, dict(diag, reason='INVALID_INPUT_OR_NUMERICAL_FAILURE')


def closest_finger_pair(h, data, finger):
    """Read native closest geometry; returns no measured-contact authority."""
    geom = int(h.tip_geom_ids[finger])
    best = None
    for obj in h.object_geom_ids:
        obj = int(obj)
        if h.model.geom_contype[obj] == 0: continue
        ends = np.zeros(6)
        gap = float(mujoco.mj_geomDistance(h.model, data, geom, obj, .2, ends))
        if best is None or gap < best['gap_m']:
            normal = ends[:3]-ends[3:]
            length = float(np.linalg.norm(normal))
            if length < 1e-12 or not np.isfinite(ends).all(): continue
            normal /= length
            # Positive separation directly supplies the outward direction.
            # For shallow overlap, signed endpoints reverse; the pad center
            # remains outside the contacted object face and fixes that sign.
            if normal@(data.geom_xpos[geom]-ends[3:]) < 0.: normal = -normal
            best = dict(gap_m=gap, pad_point_world_m=ends[:3].copy(),
                        object_point_world_m=ends[3:].copy(), normal_world=normal,
                        object_geom_id=obj)
    return best


class ActiveFingerSearchV5(ActiveFingerSearchV4):
    """The V3 lifecycle/API with a primary physical-separation objective."""

    def reset(self):
        super().reset()
        self.closest_gaps = np.zeros(4)
        self.closest_object_geoms = np.full(4, -1, dtype=int)
        self._query_mask = np.zeros(4, dtype=bool)

    def _clear_bias_and_targets(self, finger_mask=None):
        super()._clear_bias_and_targets(finger_mask)
        mask = np.ones(4, dtype=bool) if finger_mask is None else np.asarray(finger_mask, dtype=bool)
        self.closest_gaps[mask] = 0.
        self.closest_object_geoms[mask] = -1

    def _refresh_targets(self, data):
        # Unlike V3/V4 there is no persistent 50 ms mesh-center target.
        # Native closest-pair queries are performed once in the tick below.
        self.target_age = 0.

    def _update_orientation_targets(self, data):
        for finger in np.flatnonzero(self._query_mask):
            pair = closest_finger_pair(self.h, data, finger)
            self.target_valid[finger] = False
            self._surface_target_valid[finger] = False
            if pair is None: continue
            geom = int(self.h.tip_geom_ids[finger]);normal = pair['normal_world']
            support = pad_support_vector(self.h.model, geom, data.geom_xmat[geom].reshape(3, 3), normal)
            point = pair['object_point_world_m']
            self.surface_points[finger] = point
            self.surface_face_ids[finger] = -1  # Collision geom, never an original mesh face claim.
            self.closest_gaps[finger] = pair['gap_m']
            self.closest_object_geoms[finger] = pair['object_geom_id']
            self.target_normals[finger] = normal
            self.targets[finger] = point+support+self.config.target_clearance_m*normal
            self.target_valid[finger] = bool(np.linalg.norm(self.targets[finger]-data.geom_xpos[geom])
                                            <= self.config.maximum_target_distance_m)
            self._surface_target_valid[finger] = self.target_valid[finger]
        if np.any(self._query_mask): self.refresh_count += 1

    def update(self, data, nominal_q, forces, normals, active, enabled=True,
               *, intentional_air=False, search_mask=None):
        nominal, force = np.asarray(nominal_q, dtype=float), np.asarray(forces, dtype=float)
        actual, sensor_normals = np.asarray(active, dtype=bool), np.asarray(normals, dtype=float)
        mask = np.ones(4, dtype=bool) if search_mask is None else np.asarray(search_mask, dtype=bool)
        if (nominal.shape != (23,) or force.shape != (4,) or actual.shape != (4,)
                or sensor_normals.shape != (4, 3) or mask.shape != (4,)
                or not all(np.all(np.isfinite(x)) for x in (nominal, force, sensor_normals))
                or np.any(force < 0.) or (intentional_air and enabled)):
            raise ValueError('Expected valid measured inputs; intentional air requires enabled=False')
        if not self.initialized:
            self.rebase(nominal)
        cfg = self.config
        result = nominal.copy()
        increments = np.zeros(16)
        seeking = np.zeros(4, dtype=bool)
        recontacted = np.zeros(4, dtype=bool)
        stable_recontacted = np.zeros(4, dtype=bool)
        requires_rebase = np.zeros(4, dtype=bool)
        step_feasible = np.ones(4, dtype=bool)
        solver = [None]*4
        state = ['DISABLED']*4
        force_pause = bool(np.any(force >= cfg.force_pause_n))
        self._query_mask = mask & ~actual & ~force_pause
        # Physical history remains truthful even while motion is disabled.
        for finger in range(4):
            safe_contact = bool(actual[finger] and force[finger] < cfg.force_pause_n)
            if safe_contact:
                recontacted[finger] = bool(not self.was_active[finger] and self.loss_age[finger] > 0.)
                self.contact_stable_age[finger] += cfg.dt_s
                if self.contact_stable_age[finger]+1e-12 >= self.stable_contact_s:
                    stable_recontacted[finger] = not self.stable_confirmed[finger]
                    self.stable_confirmed[finger] = True
                    self.loss_age[finger] = 0.
                    self.search_distance[finger] = 0.
                    self.search_elapsed[finger] = 0.
                    self.posture_attempts_this_episode[finger] = 0
                    self.posture_attempts_this_context[finger] = 0
                elif self.loss_age[finger] > 0.:
                    self.loss_age[finger] += cfg.dt_s
            else:
                self.contact_stable_age[finger] = 0.
                self.stable_confirmed[finger] = False
                self.loss_age[finger] += cfg.dt_s
            if enabled and mask[finger] and not actual[finger]:
                self.search_elapsed[finger] += cfg.dt_s
        if not enabled:
            self._clear_bias_and_targets()
            state = ['INTENTIONAL_AIR' if intentional_air else 'DISABLED']*4
        else:
            self.target_age += cfg.dt_s
            if np.any(~actual & mask) and not force_pause and self.target_age >= cfg.target_refresh_s:
                self._refresh_targets(data)
            self._update_orientation_targets(data)
            for finger, columns in enumerate(self.columns):
                if actual[finger]:
                    state[finger] = 'CONTACT_HOLD'
                elif not mask[finger]:
                    state[finger] = 'MASKED_HOLD'
                elif force_pause:
                    state[finger] = 'FORCE_PAUSE'
                elif self.loss_age[finger]+1e-12 < cfg.loss_debounce_s:
                    state[finger] = 'DEBOUNCE'
                elif not self.target_valid[finger]:
                    state[finger] = 'OUT_OF_REACH'
                elif (self.search_distance[finger] >= cfg.maximum_search_distance_m-1e-12
                      or self.search_elapsed[finger] >= cfg.maximum_search_time_s):
                    state[finger] = 'SEARCH_LIMIT'
                else:
                    error = self.targets[finger]-data.geom_xpos[int(self.h.tip_geom_ids[finger])]
                    distance = float(np.linalg.norm(error))
                    if distance <= .0001:
                        state[finger] = 'GEOMETRIC_TARGET_NO_CONTACT'
                    else:
                        distance_step = min(distance, cfg.speed_m_s*cfg.dt_s,
                            cfg.maximum_search_distance_m-self.search_distance[finger])
                        center_jac, support_jac = self._support_jacobians(data, finger)
                        delta, solve_diag = bounded_normal_finger_step(
                            center_jac, support_jac, self.target_normals[finger],
                            nominal[columns+7], self.bias[columns], self.lower[columns], self.upper[columns],
                            joint_step=cfg.maximum_search_joint_speed_rad_s*cfg.dt_s,
                            cartesian_step=distance_step, maximum_bias=cfg.maximum_joint_bias_rad,
                            damping=cfg.jacobian_damping_m)
                        solver[finger] = solve_diag
                        step_feasible[finger] = solve_diag['feasible']
                        if solve_diag['feasible']:
                            self.bias[columns] += delta
                            increments[columns] = delta
                            self.search_distance[finger] += solve_diag['charged_step_m']
                            seeking[finger] = np.max(abs(delta)) > 1e-12
                            state[finger] = 'SEEK' if seeking[finger] else 'NO_LEGAL_INWARD_STEP'
                        else:
                            # Preserve the actual previously applied reference;
                            # never disguise a failed repair with a larger clip.
                            held = data.ctrl[np.asarray(self.h.hand_actuator_ids)[columns]].copy()
                            self.bias[columns] = held-nominal[columns+7]
                            result[columns+7] = held
                            requires_rebase[finger] = True
                            state[finger] = 'INFEASIBLE_HOLD_REBASE_REQUIRED'
                            continue
                result[columns+7] = nominal[columns+7]+self.bias[columns]
                # Non-search nominal/MCC handoff retains the prior API's range
                # reconciliation. A feasible search step already obeys these
                # bounds, so this cannot enlarge its bounded inward increment.
                result[columns+7] = np.clip(result[columns+7], self.lower[columns], self.upper[columns])
                self.bias[columns] = result[columns+7]-nominal[columns+7]
        self.was_active[:] = actual
        diagnostics = dict(enabled=bool(enabled), intentional_air=bool(intentional_air),
            search_mask=mask.tolist(), force_pause=force_pause, state=state,
            seeking=seeking.tolist(), recontacted=recontacted.tolist(), stable_recontacted=stable_recontacted.tolist(),
            measured_active=actual.tolist(), target_valid=self.target_valid.tolist(),
            target_positions_world_m=self.targets.tolist(), target_normals_world=self.target_normals.tolist(),
            search_distance_m=self.search_distance.tolist(), search_elapsed_s=self.search_elapsed.tolist(),
            loss_age_s=self.loss_age.tolist(), contact_stable_age_s=self.contact_stable_age.tolist(),
            joint_bias_rad=self.bias.tolist(), search_delta_rad=increments.tolist(),
            step_feasible=step_feasible.tolist(), requires_rebase=requires_rebase.tolist(), solver=solver,
            posture_attempt_id=self.posture_attempt_id.tolist(),
            posture_attempts_this_episode=self.posture_attempts_this_episode.tolist(),
            posture_attempts_this_context=self.posture_attempts_this_context.tolist(),
            recovery_context_id=self.recovery_context_id.tolist(),
            surface_points_world_m=self.surface_points.tolist(),
            surface_face_ids=self.surface_face_ids.tolist(),
            search_geometry='CLOSEST_COLLISION_PAIR_NORMAL_PRIORITY',
            closest_collision_gap_m=self.closest_gaps.tolist(),
            closest_object_geom_ids=self.closest_object_geoms.tolist(),
            step_bound_metric='MAX_PREDICTED_CENTER_AND_SUPPORT_DISPLACEMENT',
            nonlinear_contact_progress_guaranteed=False,
            target_refresh_count=self.refresh_count, target_authority='COLLISION_GEOMETRY_ONLY_NO_CONTACT_OR_COVERAGE')
        return result, diagnostics
