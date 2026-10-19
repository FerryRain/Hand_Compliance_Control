"""Measured finger participation and bounded requests for posture recovery.

This observer owns its contact-loss clocks independently of trajectory/search
rebases. It never writes joint state, drives a controller, adds coverage, or
declares a geometric target physically contacted. The caller may consider a
request for a coordinated wrist/posture recovery while preserving a support.

Target progress uses *actual pad geometry centers* at two times against the
same current planned target. Merely moving a target toward a stationary pad center
does not count. Joint motion and SEEK deltas are reported separately; neither
proves effective progress or contact. Distance gates and nearby joint bounds
are diagnostic reasons, not proofs that any surface contact is unreachable.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class FingerParticipationConfig:
    minimum_loss_s: float = 3.
    current_loss_debounce_s: float = .06
    progress_window_s: float = .5
    low_progress_duration_s: float = 1.
    out_of_reach_duration_s: float = 1.
    minimum_target_progress_m_s: float = .0005
    minimum_actual_joint_displacement_rad: float = .002
    stationary_pad_center_displacement_m: float = .0002
    maximum_target_distance_m: float = .060
    joint_margin_rad: float = .015
    joint_boundary_distance_rad: float = .020
    global_request_cooldown_s: float = 8.
    finger_request_cooldown_s: float = 20.
    max_requests_per_loss_episode: int = 3
    stable_recontact_s: float = .15
    active_force_min_n: float = .2
    safe_force_max_n: float = 8.
    surface_phases: tuple[str, ...] = ('SLIDE', 'LOCAL_SEARCH', 'NEAR_TRANSFER', 'REACQUIRE')
    intentional_air_phases: tuple[str, ...] = ('LIFT', 'TRANSFER', 'APPROACH', 'LIFT_PLAN')

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if name.endswith('_phases'):
                if not value or not all(isinstance(phase, str) and phase for phase in value):
                    raise ValueError(f'{name} requires nonempty phase names')
            elif not np.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if not isinstance(self.max_requests_per_loss_episode, int):
            raise ValueError('max_requests_per_loss_episode must be an integer')
        if set(self.surface_phases) & set(self.intentional_air_phases):
            raise ValueError('Surface and intentional-air phases must be disjoint')
        if self.current_loss_debounce_s > self.minimum_loss_s:
            raise ValueError('Loss debounce must not exceed minimum loss')
        if self.active_force_min_n >= self.safe_force_max_n:
            raise ValueError('Safe force interval must be nonempty')


class FingerParticipationMonitor:
    """Independent observer; construct once per physical run, never per stroke.

    ``update`` accepts canonical actual q[23], hand q[16], or finger q[4,4].
    ``pad_centers_world_m`` MUST be data.geom_xpos[tip_geom_ids], matching
    ActiveFingerSearch target centers. Do not supply pad site positions: a site
    offset would corrupt distance/progress and the 60 mm gate diagnosis.
    Ranges are physical hand ranges [16,2] or [4,4,2]. Pass force measurements
    when available; otherwise ``measured_active`` is trusted as the sensor
    threshold decision. A request budget is spent on *emission*, not on a
    claimed successful or executed maneuver. Only 150 ms (configurable) of
    actual safe same-finger contact resets that finger's episode history.
    An explicit measured-geometry grant can renew the finite context quota,
    preserving the episode request total and both request cooldowns.

    ``target_valid`` reports the planner's gate, not existence of a target.
    Supply NaNs for unavailable targets, or use ``target_available`` to mask
    unused zero-filled rows. A finite OUT_OF_REACH target may still be useful
    for diagnosis. Intentional air phases preserve true loss age but clear
    surface-stagnation evidence; SELECT and other neutral phases pause it.
    """
    names = ('index', 'middle', 'ring', 'thumb')

    def __init__(self, hand_joint_ranges_rad, *, config=None):
        self.config = config or FingerParticipationConfig()
        ranges = np.asarray(hand_joint_ranges_rad, dtype=float)
        if ranges.shape == (16, 2):
            ranges = ranges.reshape(4, 4, 2)
        if ranges.shape != (4, 4, 2) or not np.isfinite(ranges).all():
            raise ValueError('Physical hand joint ranges must be finite [16,2] or [4,4,2]')
        self.lower = ranges[:, :, 0] + self.config.joint_margin_rad
        self.upper = ranges[:, :, 1] - self.config.joint_margin_rad
        if np.any(self.lower >= self.upper):
            raise ValueError('Joint margins leave no usable range')
        self._history = deque()
        self._last_time = None
        self._last_surface = False
        self._loss_start = [None] * 4
        self._episode_start = [None] * 4
        self._safe_start = [None] * 4
        self._low_progress_s = np.zeros(4)
        self._outside_s = np.zeros(4)
        self._request_count = np.zeros(4, dtype=int)
        self._context_request_count = np.zeros(4, dtype=int)
        self._recovery_context_id = np.zeros(4, dtype=int)
        self._last_finger_request = np.full(4, -np.inf)
        self._last_global_request = -np.inf
        self._sequence = 0

    def apply_recovery_context(self, grant):
        """Renew only the request quota after an explicit measured-pose grant.

        True loss age, total requests in this episode, evidence and cooldowns
        survive. Ordinary trajectory rebases never call this method.
        """
        finger, context = grant.get('finger'), grant.get('context_id')
        if (grant.get('kind') != 'MEASURED_RECOVERY_CONTEXT_GRANTED'
                or grant.get('accepted') is not True
                or type(finger) is not int or finger not in range(4)
                or type(context) is not int or context <= 0):
            raise ValueError('Expected an explicit measured recovery context grant')
        if context <= self._recovery_context_id[finger]:
            return dict(accepted=False, reason='CONTEXT_ALREADY_CONSUMED')
        if self._loss_start[finger] is None:
            return dict(accepted=False, reason='FINGER_CURRENTLY_ACTIVE')
        self._recovery_context_id[finger] = context
        self._context_request_count[finger] = 0
        return dict(accepted=True, reason='NEW_MEASURED_GEOMETRY_CONTEXT',
                    finger=finger, context_id=context,
                    requests_issued_this_loss_episode=int(self._request_count[finger]))

    @staticmethod
    def _hand_q(value, name):
        value = np.asarray(value, dtype=float)
        if value.shape == (23,):
            value = value[7:]
        if value.shape == (16,):
            value = value.reshape(4, 4)
        if value.shape != (4, 4) or not np.isfinite(value).all():
            raise ValueError(f'{name} must be finite actual hand[16], canonical[23], or [4,4]')
        return value

    def update(self, time_s, q_rad, pad_centers_world_m, measured_active, phase, *,
               forces_n=None, planned_targets_world_m=None, target_valid=None,
               target_available=None, search_state=None, search_delta_rad=None,
               request_allowed_mask=None):
        cfg = self.config
        now = float(time_s)
        if not np.isfinite(now) or (self._last_time is not None and now <= self._last_time):
            raise ValueError('Observation times must be finite and strictly increasing')
        q = self._hand_q(q_rad, 'q_rad')
        tips = np.asarray(pad_centers_world_m, dtype=float)
        active = np.asarray(measured_active, dtype=bool)
        if tips.shape != (4, 3) or not np.isfinite(tips).all() or active.shape != (4,):
            raise ValueError('Expected finite measured pad geometry centers[4,3] and measured_active[4]')
        force = None if forces_n is None else np.asarray(forces_n, dtype=float)
        if force is not None and (force.shape != (4,) or not np.isfinite(force).all() or np.any(force < 0)):
            raise ValueError('forces_n must be finite nonnegative [4]')
        request_allowed = (np.ones(4, dtype=bool) if request_allowed_mask is None
                           else np.asarray(request_allowed_mask, dtype=bool))
        if request_allowed.shape != (4,):
            raise ValueError('request_allowed_mask must identify four physical fingers')
        actual = active if force is None else active & (force >= cfg.active_force_min_n)
        safe = actual if force is None else actual & (force < cfg.safe_force_max_n)
        force_pause = bool(force is not None and np.any(force >= cfg.safe_force_max_n))
        targets = (np.full((4, 3), np.nan) if planned_targets_world_m is None
                   else np.asarray(planned_targets_world_m, dtype=float))
        if targets.shape != (4, 3):
            raise ValueError('planned_targets_world_m must have shape [4,3]')
        available = np.isfinite(targets).all(axis=1)
        if target_available is not None:
            supplied = np.asarray(target_available, dtype=bool)
            if supplied.shape != (4,):
                raise ValueError('target_available must have shape [4]')
            available &= supplied
        valid = available.copy() if target_valid is None else np.asarray(target_valid, dtype=bool)
        if valid.shape != (4,):
            raise ValueError('target_valid must have shape [4]')
        states = np.asarray(['UNRECORDED'] * 4 if search_state is None else search_state, dtype=str)
        if states.shape != (4,):
            raise ValueError('search_state must have shape [4]')
        increments = np.zeros((4, 4)) if search_delta_rad is None else self._hand_q(search_delta_rad, 'search_delta_rad')
        surface = str(phase) in cfg.surface_phases
        air = str(phase) in cfg.intentional_air_phases
        dt = 0. if self._last_time is None else now - self._last_time
        evidence_dt = dt if surface and self._last_surface and not force_pause else 0.
        if air:
            self._history.clear()
            self._low_progress_s[:] = 0.
            self._outside_s[:] = 0.
        if not air:
            self._history.append((now, q.copy(), tips.copy()))
        cutoff = now - cfg.progress_window_s
        while len(self._history) > 2 and self._history[1][0] <= cutoff:
            self._history.popleft()
        window_ready = surface and len(self._history) >= 2 and self._history[0][0] <= cutoff + 1e-12
        previous_q = previous_tips = None
        if window_ready:
            first, second = self._history[0], self._history[1]
            mix = float(np.clip((cutoff - first[0]) / (second[0] - first[0]), 0., 1.))
            previous_q = (1-mix) * first[1] + mix * second[1]
            previous_tips = (1-mix) * first[2] + mix * second[2]
        diagnostics, candidates = [], []
        for finger in range(4):
            if actual[finger]:
                self._loss_start[finger] = None
                self._low_progress_s[finger] = self._outside_s[finger] = 0.
            else:
                if self._loss_start[finger] is None:
                    self._loss_start[finger] = now
                if self._episode_start[finger] is None:
                    self._episode_start[finger] = now
            if safe[finger]:
                if self._safe_start[finger] is None:
                    self._safe_start[finger] = now
                if now - self._safe_start[finger] >= cfg.stable_recontact_s - 1e-12:
                    self._episode_start[finger] = None
                    self._request_count[finger] = 0
                    self._context_request_count[finger] = 0
            else:
                self._safe_start[finger] = None
            loss_age = 0. if self._loss_start[finger] is None else now - self._loss_start[finger]
            episode_age = 0. if self._episode_start[finger] is None else now - self._episode_start[finger]
            distance = float(np.linalg.norm(targets[finger] - tips[finger])) if available[finger] else None
            joint_motion = tip_motion = progress_rate = None
            low_progress = None
            if window_ready:
                joint_motion = float(np.linalg.norm(q[finger] - previous_q[finger]))
                tip_motion = float(np.linalg.norm(tips[finger] - previous_tips[finger]))
                if available[finger]:
                    progress_rate = (float(np.linalg.norm(targets[finger] - previous_tips[finger])) - distance) / cfg.progress_window_s
                    low_progress = progress_rate < cfg.minimum_target_progress_m_s
                elif joint_motion < cfg.minimum_actual_joint_displacement_rad and tip_motion < cfg.stationary_pad_center_displacement_m:
                    low_progress = True
            outside = states[finger] == 'OUT_OF_REACH' or (distance is not None and distance > cfg.maximum_target_distance_m)
            if surface and not actual[finger] and not force_pause:
                if low_progress is True:
                    self._low_progress_s[finger] += evidence_dt
                elif low_progress is False:
                    self._low_progress_s[finger] = 0.
                self._outside_s[finger] = self._outside_s[finger] + evidence_dt if outside else 0.
            lower_near = q[finger] - self.lower[finger] <= cfg.joint_boundary_distance_rad
            upper_near = self.upper[finger] - q[finger] <= cfg.joint_boundary_distance_rad
            proposed = bool(states[finger] == 'SEEK' or np.any(abs(increments[finger]) > 1e-12))
            reasons = []
            if self._low_progress_s[finger] >= cfg.low_progress_duration_s - 1e-12:
                reasons.append('LOW_MEASURED_TARGET_PROGRESS' if progress_rate is not None else 'NO_MEASURED_FINGER_MOTION')
            if self._outside_s[finger] >= cfg.out_of_reach_duration_s - 1e-12:
                reasons.append('TARGET_DISTANCE_GATE_OR_OUT_OF_REACH_STATE')
            if np.any(lower_near | upper_near):
                reasons.append('JOINT_BOUNDARY_NEARBY')
            if proposed and low_progress is True:
                reasons.append('SEEK_WITHOUT_MEASURED_PROGRESS')
            has_trigger = (self._low_progress_s[finger] >= cfg.low_progress_duration_s - 1e-12
                           or self._outside_s[finger] >= cfg.out_of_reach_duration_s - 1e-12)
            budget_left = int(cfg.max_requests_per_loss_episode - self._context_request_count[finger])
            cooldown_left = max(0., cfg.finger_request_cooldown_s - (now - self._last_finger_request[finger]))
            eligible = bool(surface and request_allowed[finger] and not force_pause and not actual[finger]
                            and loss_age >= cfg.current_loss_debounce_s - 1e-12
                            and episode_age >= cfg.minimum_loss_s - 1e-12 and has_trigger
                            and budget_left > 0 and cooldown_left <= 1e-12)
            item = dict(finger=finger, name=self.names[finger], measured_contact=bool(actual[finger]),
                        measured_safe_contact=bool(safe[finger]), loss_age_s=float(loss_age),
                        unresolved_loss_episode_age_s=float(episode_age),
                        stable_safe_contact_s=0. if self._safe_start[finger] is None else now-self._safe_start[finger],
                        low_progress_surface_time_s=float(self._low_progress_s[finger]),
                        outside_gate_surface_time_s=float(self._outside_s[finger]),
                        measured_joint_displacement_rad=joint_motion, measured_pad_center_displacement_m=tip_motion,
                        measured_progress_toward_same_target_m_s=progress_rate,
                        target_distance_m=distance, target_available=bool(available[finger]),
                        planner_target_valid=bool(valid[finger]), search_state=str(states[finger]),
                        search_proposal_active=proposed,
                        request_allowed=bool(request_allowed[finger]),
                        near_lower_joint_boundary=lower_near.tolist(), near_upper_joint_boundary=upper_near.tolist(),
                        reasons=reasons, eligible_before_global_cooldown=eligible,
                        requests_issued_this_loss_episode=int(self._request_count[finger]),
                        requests_issued_this_context=int(self._context_request_count[finger]),
                        recovery_context_id=int(self._recovery_context_id[finger]),
                        request_budget_remaining=budget_left, finger_cooldown_remaining_s=float(cooldown_left),
                        unreachable_proven=False, contact_inferred_from_target=False)
            diagnostics.append(item)
            if eligible:
                # Give each missing finger a chance before repeatedly selecting
                # the oldest lost finger. Persistent loss breaks equal-count ties.
                candidates.append((int(self._context_request_count[finger]), -episode_age, finger))
        request = None
        global_left = max(0., cfg.global_request_cooldown_s - (now - self._last_global_request))
        if candidates and global_left <= 1e-12:
            finger = min(candidates)[2]
            self._sequence += 1
            self._request_count[finger] += 1
            self._context_request_count[finger] += 1
            self._last_finger_request[finger] = self._last_global_request = now
            item = diagnostics[finger]
            item['requests_issued_this_loss_episode'] = int(self._request_count[finger])
            item['requests_issued_this_context'] = int(self._context_request_count[finger])
            item['request_budget_remaining'] -= 1
            item['finger_cooldown_remaining_s'] = cfg.finger_request_cooldown_s
            request = dict(kind='FINGER_POSTURE_RECOVERY_REQUEST', request_id=self._sequence,
                           time_s=now, finger=finger, name=self.names[finger], reasons=list(item['reasons']),
                           loss_age_s=item['loss_age_s'], unresolved_loss_episode_age_s=item['unresolved_loss_episode_age_s'],
                           request_number_this_loss_episode=int(self._request_count[finger]),
                           request_number_this_context=int(self._context_request_count[finger]),
                           recovery_context_id=int(self._recovery_context_id[finger]),
                           evidence=dict(target_distance_m=item['target_distance_m'],
                                         measured_progress_toward_same_target_m_s=item['measured_progress_toward_same_target_m_s'],
                                         measured_joint_displacement_rad=item['measured_joint_displacement_rad']),
                           action_authority='DIAGNOSTIC_REQUEST_ONLY', recovery_success=False,
                           coverage_credit=False, unreachable_proven=False)
            global_left = cfg.global_request_cooldown_s
        self._last_time, self._last_surface = now, surface
        return dict(schema='i04.finger_participation.v1', time_s=now, phase=str(phase),
                    surface_phase=surface, intentional_air_phase=air, force_pause=force_pause,
                    geometry_reference='ACTUAL_PAD_GEOM_CENTER_AND_PLANNED_PAD_CENTER_TARGET',
                    progress_window_ready=window_ready, per_finger=diagnostics, request=request,
                    global_cooldown_remaining_s=float(global_left), coverage_credit=False,
                    causal_scope='Observed progress only; wrist/finger contributions are not separated')
