"""Solver-isolation adapter: retain v2 episode policy, replace only its step.

This independent module is for paired local probes. It deliberately retains
v2 rebase/disable/debounce/contact budget semantics, so a comparison isolates
bounded IK rather than also introducing the v3 persistent episode policy.
"""
import numpy as np

from .active_fingers import ActiveFingerSearch
from .bounded_finger_step import bounded_finger_step


class BoundedOnlyFingerSearch(ActiveFingerSearch):
    def update(self, data, nominal_q, forces, normals, active, enabled=True):
        nominal = np.asarray(nominal_q, dtype=float)
        force = np.asarray(forces, dtype=float)
        actual = np.asarray(active, dtype=bool)
        sensor_normals = np.asarray(normals, dtype=float)
        if (nominal.shape != (23,) or force.shape != (4,) or actual.shape != (4,)
                or sensor_normals.shape != (4, 3) or not np.all(np.isfinite(nominal))
                or not np.all(np.isfinite(force)) or np.any(force < 0)):
            raise ValueError('Expected finite nominal[23], nonnegative forces[4], normals[4,3], active[4]')
        cfg = self.config
        result = nominal.copy()
        seeking, recontacted = np.zeros(4, dtype=bool), np.zeros(4, dtype=bool)
        increments = np.zeros(16)
        state = ['DISABLED']*4
        force_pause = bool(np.any(force >= cfg.force_pause_n))
        solver, hold_overrides = [None]*4, {}
        requires_rebase = np.zeros(4, dtype=bool)
        if not enabled:
            self.reset()
        else:
            if not self.initialized:
                self.rebase(nominal)
            self.target_age += cfg.dt_s
            if np.any(~actual) and not force_pause and self.target_age >= cfg.target_refresh_s:
                self._refresh_targets(data)
            for finger, columns in enumerate(self.columns):
                if actual[finger]:
                    recontacted[finger] = not self.was_active[finger] and self.search_distance[finger] > 0.
                    self.loss_age[finger] = 0.
                    state[finger] = 'CONTACT_HOLD'
                else:
                    if self.was_active[finger]:
                        self.loss_age[finger] = 0.
                        self.search_distance[finger] = 0.
                    self.loss_age[finger] += cfg.dt_s
                    if force_pause:
                        state[finger] = 'FORCE_PAUSE'
                    elif self.loss_age[finger]+1e-12 < cfg.loss_debounce_s:
                        state[finger] = 'DEBOUNCE'
                    elif not self.target_valid[finger]:
                        state[finger] = 'OUT_OF_REACH'
                    elif (self.search_distance[finger] >= cfg.maximum_search_distance_m-1e-12
                          or self.loss_age[finger] >= cfg.maximum_search_time_s):
                        state[finger] = 'SEARCH_LIMIT'
                    else:
                        error = self.targets[finger]-data.geom_xpos[int(self.h.tip_geom_ids[finger])]
                        distance = np.linalg.norm(error)
                        if distance <= .0001:
                            state[finger] = 'GEOMETRIC_TARGET_NO_CONTACT'
                        else:
                            distance_step = min(distance, cfg.speed_m_s*cfg.dt_s,
                                cfg.maximum_search_distance_m-self.search_distance[finger])
                            motion = error*(distance_step/distance)
                            jac = self._finger_jacobian(data, finger)
                            delta, solver[finger] = bounded_finger_step(jac, motion,
                                nominal[columns+7], self.bias[columns], self.lower[columns], self.upper[columns],
                                joint_step=cfg.maximum_search_joint_speed_rad_s*cfg.dt_s,
                                cartesian_step=distance_step, maximum_bias=cfg.maximum_joint_bias_rad,
                                damping=cfg.jacobian_damping_m)
                            if solver[finger]['feasible']:
                                self.bias[columns] += delta
                                self.search_distance[finger] += float(np.linalg.norm(jac@delta))
                                increments[columns] = delta
                                seeking[finger] = bool(np.max(abs(delta)) > 1e-12)
                                state[finger] = 'SEEK' if seeking[finger] else 'JOINT_LIMIT'
                            else:
                                hold_overrides[finger] = data.ctrl[np.asarray(self.h.hand_actuator_ids)[columns]].copy()
                                state[finger] = 'INFEASIBLE_HOLD_REBASE_REQUIRED'
                                requires_rebase[finger] = True
                self.was_active[finger] = actual[finger]
            result[7:] = np.clip(nominal[7:]+self.bias, self.lower, self.upper)
            for finger, held in hold_overrides.items():
                result[self.columns[finger]+7] = held
            self.bias[:] = result[7:]-nominal[7:]
        diagnostics = dict(enabled=bool(enabled), force_pause=force_pause, state=state,
            seeking=seeking.tolist(), recontacted=recontacted.tolist(), measured_active=actual.tolist(),
            target_valid=self.target_valid.tolist(), target_positions_world_m=self.targets.tolist(),
            target_normals_world=self.target_normals.tolist(), search_distance_m=self.search_distance.tolist(),
            loss_age_s=self.loss_age.tolist(), joint_bias_rad=self.bias.tolist(), search_delta_rad=increments.tolist(),
            target_refresh_count=self.refresh_count, target_authority='GT_GEOMETRY_ONLY_NO_CONTACT_OR_COVERAGE',
            bounded_solver=solver, requires_rebase=requires_rebase.tolist(),
            episode_policy='UNCHANGED_V2_FOR_SOLVER_ISOLATION')
        return result, diagnostics
