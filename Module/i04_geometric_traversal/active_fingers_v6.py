"""Independent damped-normal candidate with conservative finite FK checks.

The V5 normalization is removed: scalar normal DLS retains gradient magnitude
and the original .001 m damping. A bounded step is accepted only if finite
geometry checks reduce the actual-posture and previous-reference gaps. This
is planning evidence, never a physical-contact claim or coverage update.
"""
from __future__ import annotations

import mujoco
import numpy as np

from .active_fingers_v5 import ActiveFingerSearchV5, closest_finger_pair
from .bounded_finger_step import bounded_finger_step
from .physics import joint_state, set_joint_state


def bounded_damped_normal_step(center_jacobian, support_jacobian, outward_normal,
                               nominal, previous_bias, joint_lower, joint_upper,
                               *, joint_step=.001, cartesian_step=.000008,
                               maximum_bias=.50, damping=.001):
    """Scalar normal DLS with unchanged joint/bias and both Cartesian bounds."""
    zero = np.zeros(4)
    try:
        center, support, normal = (np.asarray(x, dtype=float) for x in
                                  (center_jacobian, support_jacobian, outward_normal))
        if (center.shape != (3, 4) or support.shape != (3, 4) or normal.shape != (3,)
                or not all(np.isfinite(x).all() for x in (center, support, normal))
                or np.linalg.norm(normal) < .5):
            return zero, dict(feasible=False, reason='INVALID_NORMAL_JACOBIAN', optimality_confirmed=False)
        normal = normal/np.linalg.norm(normal)
        gradient = normal@support
        scalar = np.vstack((gradient, np.zeros((2, 4))))
        delta, diag = bounded_finger_step(scalar, [-cartesian_step, 0., 0.], nominal,
            previous_bias, joint_lower, joint_upper, joint_step=joint_step,
            cartesian_step=cartesian_step, maximum_bias=maximum_bias, damping=damping)
        diag = dict(diag, normal_gradient_norm_m_per_rad=float(np.linalg.norm(gradient)),
                    primary_objective='UNNORMALIZED_SCALAR_NORMAL_DLS', damping_m=damping,
                    cartesian_bound_metric='BOTH_CENTER_AND_SUPPORT')
        if not diag['feasible']: return zero, diag
        lower, upper = np.asarray(diag['lower_step_rad']), np.asarray(diag['upper_step_rad'])
        zero_ok = bool(np.all(lower <= 0.) and np.all(upper >= 0.))
        traveled = max(float(np.linalg.norm(center@delta)), float(np.linalg.norm(support@delta)))
        if traveled > cartesian_step:
            candidate = delta*(cartesian_step/traveled)
            if np.any(candidate < lower-1e-12) or np.any(candidate > upper+1e-12):
                return zero, dict(diag, feasible=False, reason='DUAL_BOUND_SCALE_VIOLATES_REQUIRED_REPAIR',
                                  optimality_confirmed=False)
            delta = candidate
            diag.update(optimality_confirmed=False, scaled_for_dual_bound=True)
        predicted = float(gradient@delta)
        if predicted >= -1e-12:
            if zero_ok:
                return zero, dict(diag, reason='NO_LOCAL_INWARD_PROGRESS', optimality_confirmed=False,
                    center_step_m=0., support_step_m=0., charged_step_m=0., predicted_gap_change_m=0.)
            return zero, dict(diag, feasible=False, reason='NO_INWARD_LIMIT_REPAIR', optimality_confirmed=False)
        center_step, support_step = float(np.linalg.norm(center@delta)), float(np.linalg.norm(support@delta))
        return delta, dict(diag, reason='BOUNDED_DAMPED_NORMAL_DLS', center_step_m=center_step,
            support_step_m=support_step, cartesian_step_m=support_step, charged_step_m=max(center_step, support_step),
            predicted_gap_change_m=predicted, maximum_joint_step_rad=float(np.max(abs(delta))))
    except (ValueError, TypeError, FloatingPointError, np.linalg.LinAlgError):
        return zero, dict(feasible=False, reason='INVALID_INPUT_OR_NUMERICAL_FAILURE', optimality_confirmed=False)


class ActiveFingerSearchV6(ActiveFingerSearchV5):
    """Unchanged episode rules with a damped, finite-progress search strategy."""

    def __init__(self, *args, minimum_gap_improvement_m=2e-8, backtracking_steps=7, **kwargs):
        if (not np.isfinite(minimum_gap_improvement_m) or minimum_gap_improvement_m <= 0.
                or type(backtracking_steps) is not int or not 1 <= backtracking_steps <= 12):
            raise ValueError('Require a positive finite progress threshold and 1..12 finite trials')
        self.minimum_gap_improvement_m = float(minimum_gap_improvement_m)
        self.backtracking_steps = backtracking_steps
        super().__init__(*args, **kwargs)
        self._gap_scratch = mujoco.MjData(self.h.model)

    def _normal_search_step(self, data, finger, center, support, normal,
                            nominal, previous_bias, lower, upper, **bounds):
        delta, diag = bounded_damped_normal_step(center, support, normal,
            nominal, previous_bias, lower, upper, **bounds)
        if not diag['feasible'] or not np.any(delta): return delta, diag
        q = joint_state(self.h, data)
        columns = self.columns[finger]+7
        previous_reference = q.copy(); previous_reference[columns] = nominal+previous_bias
        actual_pair = closest_finger_pair(self.h, data, finger)
        set_joint_state(self.h, self._gap_scratch, previous_reference)
        reference_pair = closest_finger_pair(self.h, self._gap_scratch, finger)
        lo, hi = np.asarray(diag['lower_step_rad']), np.asarray(diag['upper_step_rad'])
        zero_ok = bool(np.all(lo <= 0.) and np.all(hi >= 0.))
        trials = []
        if actual_pair is not None and reference_pair is not None:
            for index in range(self.backtracking_steps):
                scale = .5**index; candidate = delta*scale
                if np.any(candidate < lo-1e-12) or np.any(candidate > hi+1e-12):
                    trials.append(dict(scale=scale, reason='REQUIRED_REPAIR_WOULD_BE_BROKEN'))
                    continue
                proposed = q.copy(); proposed[columns] += candidate
                set_joint_state(self.h, self._gap_scratch, proposed)
                actual_next = closest_finger_pair(self.h, self._gap_scratch, finger)
                proposed = previous_reference.copy(); proposed[columns] += candidate
                set_joint_state(self.h, self._gap_scratch, proposed)
                reference_next = closest_finger_pair(self.h, self._gap_scratch, finger)
                if actual_next is None or reference_next is None: continue
                actual_gain = actual_pair['gap_m']-actual_next['gap_m']
                reference_gain = reference_pair['gap_m']-reference_next['gap_m']
                trials.append(dict(scale=scale, actual_posture_gap_gain_m=actual_gain,
                                   previous_reference_gap_gain_m=reference_gain))
                # Both finite geometry checks must show meaningful progress.
                # Sensor contacts and force guards still decide all success.
                if min(actual_gain, reference_gain) >= self.minimum_gap_improvement_m:
                    center_step, support_step = float(np.linalg.norm(center@candidate)), float(np.linalg.norm(support@candidate))
                    return candidate, dict(diag, reason='DAMPED_NORMAL_FINITE_PROGRESS',
                        optimality_confirmed=False, backtracking_trials=trials, accepted_scale=scale,
                        minimum_gap_improvement_m=self.minimum_gap_improvement_m,
                        center_step_m=center_step, support_step_m=support_step, cartesian_step_m=support_step,
                        charged_step_m=max(center_step, support_step),
                        predicted_gap_change_m=float(normal@(support@candidate)),
                        maximum_joint_step_rad=float(np.max(abs(candidate))))
        if zero_ok:
            return np.zeros(4), dict(diag, reason='NO_FINITE_LOCAL_PROGRESS_HOLD', optimality_confirmed=False,
                backtracking_trials=trials, minimum_gap_improvement_m=self.minimum_gap_improvement_m,
                center_step_m=0., support_step_m=0., charged_step_m=0., predicted_gap_change_m=0.)
        return np.zeros(4), dict(diag, feasible=False, reason='NO_FINITE_PROGRESS_LIMIT_REPAIR',
                                optimality_confirmed=False, backtracking_trials=trials)

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
                        delta, solve_diag = self._normal_search_step(
                            data, finger, center_jac, support_jac, self.target_normals[finger],
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
            search_geometry='DAMPED_NORMAL_WITH_FINITE_GEOMETRY_BACKTRACKING',
            closest_collision_gap_m=self.closest_gaps.tolist(),
            closest_object_geom_ids=self.closest_object_geoms.tolist(),
            step_bound_metric='MAX_PREDICTED_CENTER_AND_SUPPORT_DISPLACEMENT',
            nonlinear_contact_progress_guaranteed=False,
            target_refresh_count=self.refresh_count, target_authority='COLLISION_GEOMETRY_ONLY_NO_CONTACT_OR_COVERAGE')
        return result, diagnostics
