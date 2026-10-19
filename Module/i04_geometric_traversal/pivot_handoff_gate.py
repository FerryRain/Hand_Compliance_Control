"""One measured-state wrist-pivot to finger-replan gate per recovery attempt.

The caller supplies every native 2 ms observation, actual executed path time,
the measured arm state and the last *applied* actuator reference. Replanning
can be requested only after the executed wrist path reaches its endpoint and
both endpoint settling and continuous measured tracking conditions hold.
No planned endpoint pose, cached finger solution, physics state or recovery
budget is owned or modified here. The caller replans from its current measured
state, handles all force/collision limits, and owns the attempt's total timeout.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PivotHandoffGateConfig:
    dt_s: float = .002
    minimum_endpoint_age_s: float = .150
    minimum_stable_tracking_s: float = .100
    maximum_arm_tracking_error_rad: float = .004
    force_pause_n: float = 8.

    def __post_init__(self):
        if any(not np.isfinite(value) or value <= 0. for value in vars(self).values()):
            raise ValueError('Gate limits must be finite and positive')
        if (abs(self.dt_s-.002) > 1e-12 or self.minimum_endpoint_age_s < .150
                or self.minimum_stable_tracking_s < .100
                or self.maximum_arm_tracking_error_rad > .004 or self.force_pause_n > 8.):
            raise ValueError('The native tick and measured handoff safety requirements may not be relaxed')


class PivotHandoffGate:
    """Emit at most one request for a current-state finger replan.

    Construct once for one WRIST_PIVOT stage. ``requested`` is sticky; the
    ``needs_finger_replan`` pulse is true only on its first eligible native
    observation. Any actual selected-finger contact, force >=8 N, or prior
    stage termination permanently cancels this gate. It has no reset method:
    a new gate does not itself grant a new attempt or reset any external budget.

    The coordinated runner must supply ``endpoint_arm_q`` and ``anchor_finger``.
    Then measured and last-applied arm poses must each be within 4 mrad of that
    endpoint, as well as each other, with actual safe anchor contact throughout
    the same 100 ms window. Anchor loss resets that window; the execution stage
    separately owns its 60 ms support-loss abort. Omitting the optional fields
    retains the original tracking-only observer API, explicitly flagged in the
    output, and is insufficient for a coordinated endpoint handoff.
    """
    def __init__(self, path_duration_s, selected_finger, *, started_s, config=None,
                 endpoint_arm_q=None, anchor_finger=None):
        if (not np.isfinite(path_duration_s) or path_duration_s <= 0.
                or selected_finger not in range(4) or not np.isfinite(started_s)):
            raise ValueError('Expected positive wrist path duration, physical finger and finite start time')
        self.path_duration_s = float(path_duration_s)
        self.selected_finger = int(selected_finger)
        self.started_s = float(started_s)
        self.config = config or PivotHandoffGateConfig()
        self.endpoint_arm_q = None
        if endpoint_arm_q is not None:
            endpoint = np.asarray(endpoint_arm_q, dtype=float)
            if endpoint.shape != (7,) or not np.isfinite(endpoint).all():
                raise ValueError('The planned wrist endpoint must have seven finite arm joints')
            self.endpoint_arm_q = endpoint.copy()
        if anchor_finger is not None and (anchor_finger not in range(4) or anchor_finger == selected_finger):
            raise ValueError('The measured anchor must be a different physical finger')
        self.anchor_finger = None if anchor_finger is None else int(anchor_finger)
        self.last_time_s = self.last_path_elapsed_s = None
        self.endpoint_seen_s = self.tracking_stable_since_s = None
        self.requested = False
        self.requested_at_s = None
        self.cancel_reason = None

    def update(self, time_s, path_elapsed_s, measured_q, last_applied_q, active, forces_n, *,
               previous_stage_status='RUNNING'):
        cfg = self.config
        now, elapsed = float(time_s), float(path_elapsed_s)
        measured, applied = np.asarray(measured_q, float), np.asarray(last_applied_q, float)
        actual, forces = np.asarray(active, bool), np.asarray(forces_n, float)
        status = str(previous_stage_status)
        if (measured.shape != (23,) or applied.shape != (23,) or actual.shape != (4,)
                or forces.shape != (4,) or not all(np.isfinite(x).all() for x in (measured, applied, forces))
                or np.any(forces < 0.) or not np.isfinite(now) or now < self.started_s
                or not np.isfinite(elapsed) or elapsed < 0. or elapsed > self.path_duration_s+1e-10
                or status not in ('RUNNING', 'ABORTED', 'RECOVERED')):
            raise ValueError('Expected finite native measured state, valid path progress and prior stage status')
        if self.last_time_s is not None:
            if abs(now-self.last_time_s-cfg.dt_s) > 1e-9:
                raise ValueError('A pivot handoff gate must observe every native 2 ms tick')
            if elapsed < self.last_path_elapsed_s-1e-12:
                raise ValueError('Executed path time cannot regress within one wrist stage')
        self.last_time_s, self.last_path_elapsed_s = now, elapsed
        if self.cancel_reason is None:
            if status == 'ABORTED':
                self.cancel_reason = 'PREVIOUS_STAGE_ABORTED'
            elif np.any(forces >= cfg.force_pause_n):
                self.cancel_reason = 'MEASURED_FORCE_RELEASE_REQUIRED'
            elif actual[self.selected_finger]:
                self.cancel_reason = 'SELECTED_FINGER_ALREADY_MEASURED_IN_CONTACT'
            elif status == 'RECOVERED':
                self.cancel_reason = 'PREVIOUS_STAGE_ALREADY_RECOVERED'

        arm_error = float(np.max(abs(measured[:7]-applied[:7])))
        measured_endpoint_error = (None if self.endpoint_arm_q is None else
            float(np.max(abs(measured[:7]-self.endpoint_arm_q))))
        applied_endpoint_error = (None if self.endpoint_arm_q is None else
            float(np.max(abs(applied[:7]-self.endpoint_arm_q))))
        maximum_error = max(arm_error, measured_endpoint_error or 0., applied_endpoint_error or 0.)
        anchor_safe = (None if self.anchor_finger is None else bool(actual[self.anchor_finger]
            and .2 <= forces[self.anchor_finger] < cfg.force_pause_n))
        # Wall-clock age alone cannot substitute for actual executed progress.
        if elapsed >= self.path_duration_s and self.endpoint_seen_s is None:
            self.endpoint_seen_s = now
        if (self.endpoint_seen_s is not None and maximum_error <= cfg.maximum_arm_tracking_error_rad
                and anchor_safe is not False):
            if self.tracking_stable_since_s is None:
                self.tracking_stable_since_s = now
        else:
            self.tracking_stable_since_s = None
        endpoint_age = 0. if self.endpoint_seen_s is None else now-self.endpoint_seen_s
        tracking_age = 0. if self.tracking_stable_since_s is None else now-self.tracking_stable_since_s
        ready = bool(self.cancel_reason is None and self.endpoint_seen_s is not None
            and endpoint_age >= cfg.minimum_endpoint_age_s-1e-12
            and tracking_age >= cfg.minimum_stable_tracking_s-1e-12)
        pulse = ready and not self.requested
        if pulse:
            self.requested = True
            self.requested_at_s = now
        reason = (self.cancel_reason if self.cancel_reason is not None else
                  'CURRENT_MEASURED_STATE_REPLAN_REQUEST' if pulse else
                  'REQUEST_ALREADY_EMITTED' if self.requested else
                  'PATH_NOT_AT_EXECUTED_ENDPOINT' if self.endpoint_seen_s is None else
                  'ENDPOINT_SETTLING' if endpoint_age < cfg.minimum_endpoint_age_s-1e-12 else
                  'WAITING_FOR_ACTUAL_SAFE_ANCHOR' if anchor_safe is False else
                  'WAITING_FOR_CONTINUOUS_ARM_TRACKING')
        return dict(needs_finger_replan=bool(pulse), ready=ready, requested=self.requested,
                    requested_at_s=self.requested_at_s, cancelled=self.cancel_reason is not None,
                    reason=reason, selected_finger=self.selected_finger,
                    endpoint_seen_s=self.endpoint_seen_s, endpoint_age_s=float(endpoint_age),
                    stable_tracking_age_s=float(tracking_age), arm_tracking_error_rad=arm_error,
                    measured_endpoint_error_rad=measured_endpoint_error,
                    applied_endpoint_error_rad=applied_endpoint_error,
                    maximum_handoff_error_rad=float(maximum_error),
                    anchor_finger=self.anchor_finger, anchor_safe_contact=anchor_safe,
                    endpoint_check_enabled=self.endpoint_arm_q is not None,
                    anchor_check_enabled=self.anchor_finger is not None,
                    executed_path_elapsed_s=elapsed, path_duration_s=self.path_duration_s,
                    planning_authority='CALLER_MUST_REPLAN_FROM_CURRENT_MEASURED_STATE',
                    cached_target_or_endpoint_pose_used=False, budget_reset=False,
                    measured_recovery_success=False, coverage_credit=False)
