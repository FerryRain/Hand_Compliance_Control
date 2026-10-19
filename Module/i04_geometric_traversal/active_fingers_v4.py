"""Independent support-aware candidate; frozen V3 and its experiment stay intact.

Only search geometry differs from V3: the current ellipsoid support point has
Jacobian Jcenter - dSupport, and its orientation-dependent center target is
updated each tick against a stored real mesh point/normal. Both the original
center speed bound and the support speed bound apply. These are linearized
bounds, not a guarantee of nonlinear gap reduction or actual recontact.

Episode, context, explicit-attempt and rebase methods are inherited verbatim.
The update lifecycle below retains V3 history and handoff behavior; its only
changed control section is target/Jacobian evaluation and bounded step charge.
"""
from __future__ import annotations

import mujoco
import numpy as np
import trimesh

from .active_fingers_v3 import ActiveFingerSearchV3
from .bounded_finger_step import bounded_finger_step
from .finger_reconfiguration_planner import (
    ellipsoid_support_derivative as analytic_ellipsoid_support_derivative,
)
from .full_surface import pad_support_vector


def bounded_support_finger_step(center_jacobian, support_jacobian, motion,
                                nominal, previous_bias, joint_lower, joint_upper,
                                *, joint_step=.001, cartesian_step=.000008,
                                maximum_bias=.50, damping=.001):
    """Bound both predicted center/support increments within the original box.

    A second Cartesian bound can conservatively scale a feasible support step.
    Scaling is accepted only if every original joint/bias/repair bound still
    holds. Otherwise zero means an explicit hold requiring rebase, not a claim
    that the old reference is in bounds or that no joint-space solution exists.
    """
    zero = np.zeros(4)
    try:
        center = np.asarray(center_jacobian, dtype=float)
        support = np.asarray(support_jacobian, dtype=float)
        if (center.shape != (3, 4) or support.shape != (3, 4)
                or not np.all(np.isfinite(center)) or not np.all(np.isfinite(support))):
            return zero, dict(feasible=False, reason='INVALID_CENTER_OR_SUPPORT_JACOBIAN',
                              optimality_confirmed=False)
        delta, diag = bounded_finger_step(support, motion, nominal, previous_bias,
            joint_lower, joint_upper, joint_step=joint_step, cartesian_step=cartesian_step,
            maximum_bias=maximum_bias, damping=damping)
        diag = dict(diag, support_solver_reason=diag['reason'],
                    cartesian_bound_metric='BOTH_CENTER_AND_SUPPORT')
        if not diag['feasible']:
            return zero, diag
        center_step = float(np.linalg.norm(center@delta))
        if center_step > cartesian_step:
            scaled = delta*(cartesian_step/center_step)
            lower, upper = np.asarray(diag['lower_step_rad']), np.asarray(diag['upper_step_rad'])
            if np.any(scaled < lower-1e-12) or np.any(scaled > upper+1e-12):
                return zero, dict(diag, feasible=False,
                    reason='CENTER_BOUND_SCALE_VIOLATES_REQUIRED_REPAIR',
                    optimality_confirmed=False, unscaled_center_step_m=center_step)
            delta = scaled
            diag.update(reason='SUPPORT_DLS_CENTER_SCALED', optimality_confirmed=False)
        center_step = float(np.linalg.norm(center@delta))
        support_step = float(np.linalg.norm(support@delta))
        if (not np.all(np.isfinite(delta)) or max(center_step, support_step) > cartesian_step+1e-12):
            return zero, dict(diag, feasible=False, reason='DUAL_CARTESIAN_NUMERICAL_FAILURE',
                              optimality_confirmed=False)
        return delta, dict(diag, center_step_m=center_step, support_step_m=support_step,
                           cartesian_step_m=support_step,
                           charged_step_m=max(center_step, support_step),
                           maximum_joint_step_rad=float(np.max(abs(delta))))
    except (ValueError, TypeError, FloatingPointError, np.linalg.LinAlgError):
        return zero, dict(feasible=False, reason='INVALID_INPUT_OR_NUMERICAL_FAILURE',
                          optimality_confirmed=False)


def support_point_jacobians(h, data, finger, columns, normal):
    """Return native pad-center and near-side support Jacobians, fixed normal.

    The near-side support point is center minus the outward support vector.
    Only the selected finger's four physical DoFs are returned; no wrist
    column enters this search layer. The caller owns normal/point selection.
    """
    normal = np.asarray(normal, dtype=float)
    if normal.shape != (3,) or not np.all(np.isfinite(normal)) or np.linalg.norm(normal) < .5:
        raise ValueError('A finite surface normal is required')
    normal = normal/np.linalg.norm(normal)
    geom = int(h.tip_geom_ids[finger])
    jp, jr = np.zeros((3, h.model.nv)), np.zeros((3, h.model.nv))
    mujoco.mj_jacGeom(h.model, data, jp, jr, geom)
    dofs = np.asarray(h.hand_dof_adrs)[np.asarray(columns)]
    center, angular = jp[:, dofs], jr[:, dofs]
    kind = h.model.geom_type[geom]
    if kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        derivative = analytic_ellipsoid_support_derivative(
            data.geom_xmat[geom].reshape(3, 3), h.model.geom_size[geom], normal, angular)
    elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        derivative = np.zeros_like(center)
    else:
        raise ValueError('Support search requires an ellipsoid or sphere fingertip')
    return center, center-derivative


class ActiveFingerSearchV4(ActiveFingerSearchV3):
    """V3 API and finite recovery budgets with support-aware search geometry."""

    def reset(self):
        super().reset()
        self.surface_points = np.zeros((4, 3))
        self.surface_face_ids = np.full(4, -1, dtype=int)
        self._surface_target_valid = np.zeros(4, dtype=bool)

    def _clear_bias_and_targets(self, finger_mask=None):
        super()._clear_bias_and_targets(finger_mask)
        mask = np.ones(4, dtype=bool) if finger_mask is None else np.asarray(finger_mask, dtype=bool)
        self.surface_points[mask] = 0.
        self.surface_face_ids[mask] = -1
        self._surface_target_valid[mask] = False

    def _refresh_targets(self, data):
        centers = data.geom_xpos[self.h.tip_geom_ids].copy()
        points, _, faces = trimesh.proximity.closest_point(
            self.graph.mesh, centers-self.h.object_position_m)
        surface_normals = np.asarray(self.graph.mesh.face_normals)[faces]
        for finger in range(4):
            normal = surface_normals[finger]
            length = float(np.linalg.norm(normal))
            valid = bool(np.all(np.isfinite(points[finger]))
                         and np.all(np.isfinite(normal)) and length >= .5)
            self._surface_target_valid[finger] = valid
            if not valid:
                self.target_valid[finger] = False
                self.surface_face_ids[finger] = -1
                continue
            self.surface_points[finger] = self.h.object_position_m+points[finger]
            self.surface_face_ids[finger] = int(faces[finger])
            self.target_normals[finger] = normal/length
        self._update_orientation_targets(data)
        self.target_age = 0.
        self.refresh_count += 1

    def _update_orientation_targets(self, data):
        for finger in np.flatnonzero(self._surface_target_valid):
            geom = int(self.h.tip_geom_ids[finger])
            normal = self.target_normals[finger]
            support = pad_support_vector(self.h.model, geom,
                data.geom_xmat[geom].reshape(3, 3), normal)
            target = self.surface_points[finger]+support+self.config.target_clearance_m*normal
            self.targets[finger] = target
            self.target_valid[finger] = bool(np.all(np.isfinite(target)) and
                np.linalg.norm(target-data.geom_xpos[geom]) <= self.config.maximum_target_distance_m)

    def _support_jacobians(self, data, finger):
        return support_point_jacobians(self.h, data, finger, self.columns[finger],
                                      self.target_normals[finger])

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
                        delta, solve_diag = bounded_support_finger_step(
                            center_jac, support_jac, error*(distance_step/distance),
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
                            state[finger] = 'SEEK' if seeking[finger] else 'JOINT_LIMIT'
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
            search_geometry='ORIENTATION_CURRENT_ELLIPSOID_SUPPORT_POINT',
            step_bound_metric='MAX_PREDICTED_CENTER_AND_SUPPORT_DISPLACEMENT',
            nonlinear_contact_progress_guaranteed=False,
            target_refresh_count=self.refresh_count, target_authority='GT_GEOMETRY_ONLY_NO_CONTACT_OR_COVERAGE')
        return result, diagnostics
