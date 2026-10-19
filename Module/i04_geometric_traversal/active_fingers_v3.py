"""Independent bounded-search candidate with persistent per-finger episodes.

The frozen v2 controller is unchanged. A trajectory rebase clears only the
geometric bias already incorporated in its new nominal. Actual loss history
and per-attempt search budgets survive rebase and intentional air travel.
Explicit measured-geometry contexts may renew finite budgets without ending
the unresolved physical loss episode.
"""
from __future__ import annotations

import numpy as np

from .active_fingers import ActiveFingerSearch
from .bounded_finger_step import bounded_finger_step


class ActiveFingerSearchV3(ActiveFingerSearch):
    def __init__(self, h, graph, dt=.002, *, config=None, stable_contact_s=.150,
                 maximum_posture_attempts_per_loss_episode=3):
        if not np.isfinite(stable_contact_s) or stable_contact_s < .150:
            raise ValueError('A stable contact needs at least 150 ms')
        if maximum_posture_attempts_per_loss_episode not in range(1, 4):
            raise ValueError('At most three explicit posture attempts per loss episode')
        self.stable_contact_s = float(stable_contact_s)
        self.maximum_posture_attempts_per_loss_episode = maximum_posture_attempts_per_loss_episode
        super().__init__(h, graph, dt, config=config)

    def reset(self):
        """Explicitly start a new experiment, never a routine stroke rebase."""
        super().reset()
        self.search_elapsed = np.zeros(4)
        self.contact_stable_age = np.zeros(4)
        self.stable_confirmed = np.zeros(4, dtype=bool)
        self.posture_attempt_id = np.zeros(4, dtype=int)
        self.posture_attempts_this_episode = np.zeros(4, dtype=int)
        self.posture_attempts_this_context = np.zeros(4, dtype=int)
        self.recovery_context_id = np.zeros(4, dtype=int)

    def apply_recovery_context(self, grant):
        """Consume an explicit measured-geometry grant without clearing loss.

        The observer owns the pose novelty, support and cooldown checks. This
        consumer independently rejects duplicate tokens and active fingers.
        A grant renews finite search/attempt quotas; it changes no reference,
        plant state, target, physical-contact flag or episode history.
        """
        finger, context = grant.get('finger'), grant.get('context_id')
        if (grant.get('kind') != 'MEASURED_RECOVERY_CONTEXT_GRANTED'
                or grant.get('accepted') is not True
                or type(finger) is not int or finger not in range(4)
                or type(context) is not int or context <= 0):
            raise ValueError('Expected an explicit measured recovery context grant')
        if context <= self.recovery_context_id[finger]:
            return dict(accepted=False, reason='CONTEXT_ALREADY_CONSUMED')
        if self.was_active[finger]:
            return dict(accepted=False, reason='FINGER_CURRENTLY_ACTIVE')
        self.recovery_context_id[finger] = context
        self.search_elapsed[finger] = 0.
        self.search_distance[finger] = 0.
        self.posture_attempts_this_context[finger] = 0
        return dict(accepted=True, reason='NEW_MEASURED_GEOMETRY_CONTEXT',
                    finger=finger, context_id=context,
                    loss_age_s=float(self.loss_age[finger]),
                    posture_attempts_this_episode=int(self.posture_attempts_this_episode[finger]))

    def _clear_bias_and_targets(self, finger_mask=None):
        mask = np.ones(4, dtype=bool) if finger_mask is None else np.asarray(finger_mask, dtype=bool)
        for finger in np.flatnonzero(mask): self.bias[self.columns[finger]] = 0.
        self.target_valid[mask] = False
        self.targets[mask] = 0.
        self.target_normals[mask] = 0.
        self.target_age = np.inf

    def rebase(self, nominal_q, *, finger_mask=None):
        """New measured nominal already contains the attained finger shape."""
        nominal = np.asarray(nominal_q, dtype=float)
        if nominal.shape != (23,) or not np.all(np.isfinite(nominal)):
            raise ValueError('A finite 23-joint nominal reference is required')
        mask = np.ones(4, dtype=bool) if finger_mask is None else np.asarray(finger_mask, dtype=bool)
        if mask.shape != (4,):
            raise ValueError('finger_mask must identify four physical fingers')
        self._clear_bias_and_targets(mask)
        self.initialized = True

    def begin_posture_attempt(self, finger):
        """Grant one explicitly scheduled safe-path attempt, not a new episode.

        Call only after the participation monitor grants a request and a safe
        wrist/finger path has actually been found. The outer scheduler owns its
        cooldown; this layer independently caps each measured context at three
        grants. Without an explicit new context this is the entire episode.
        """
        if finger not in range(4):
            raise ValueError('Expected a physical finger index')
        if self.was_active[finger]:
            return dict(accepted=False, reason='FINGER_CURRENTLY_ACTIVE')
        if self.posture_attempts_this_context[finger] >= self.maximum_posture_attempts_per_loss_episode:
            return dict(accepted=False, reason='POSTURE_ATTEMPT_LIMIT')
        self.posture_attempt_id[finger] += 1
        self.posture_attempts_this_episode[finger] += 1
        self.posture_attempts_this_context[finger] += 1
        self.search_elapsed[finger] = 0.
        self.search_distance[finger] = 0.
        self.target_age = np.inf
        return dict(accepted=True, reason='EXPLICIT_SAFE_POSTURE_ATTEMPT',
                    attempt_id=int(self.posture_attempt_id[finger]),
                    loss_age_s=float(self.loss_age[finger]))

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
                        jac = self._finger_jacobian(data, finger)
                        delta, solve_diag = bounded_finger_step(jac, error*(distance_step/distance),
                            nominal[columns+7], self.bias[columns], self.lower[columns], self.upper[columns],
                            joint_step=cfg.maximum_search_joint_speed_rad_s*cfg.dt_s,
                            cartesian_step=distance_step, maximum_bias=cfg.maximum_joint_bias_rad,
                            damping=cfg.jacobian_damping_m)
                        solver[finger] = solve_diag
                        step_feasible[finger] = solve_diag['feasible']
                        if solve_diag['feasible']:
                            self.bias[columns] += delta
                            increments[columns] = delta
                            self.search_distance[finger] += float(np.linalg.norm(jac@delta))
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
            target_refresh_count=self.refresh_count, target_authority='GT_GEOMETRY_ONLY_NO_CONTACT_OR_COVERAGE')
        return result, diagnostics
