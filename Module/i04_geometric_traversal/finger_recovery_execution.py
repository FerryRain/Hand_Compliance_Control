"""Measured-contact execution of a prechecked single-finger recovery path.

This small state machine returns references only. The caller owns native
integration, measured MCC, command geometry, overload relief and hard stops.
The first real selected-finger contact freezes the path at the measured pose;
only a subsequent 150 ms safe contact qualifies as stable recovery.
An anchored wrist pivot pauses on support loss and aborts after 60 ms without
safe anchor/selected support. A prechecked fixed-wrist single-finger path does
not require that anchor to rotate a wrist: it may continue, while recording
the same measured safe-support loss and retaining all other safety deadlines.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class FingerRecoveryExecutionConfig:
    dt_s: float = .002
    contact_min_n: float = .2
    force_pause_n: float = 8.
    stable_contact_s: float = .15
    support_loss_grace_s: float = .06
    endpoint_search_s: float = 2.
    tracking_full_rad: float = .004
    tracking_pause_rad: float = .010
    tracking_timeout_s: float = .30

    def __post_init__(self):
        if any(not np.isfinite(v) or v <= 0 for v in vars(self).values()):
            raise ValueError('Execution limits must be finite and positive')
        if self.contact_min_n >= self.force_pause_n or self.tracking_full_rad >= self.tracking_pause_rad:
            raise ValueError('Invalid force or tracking limits')


class FingerRecoveryExecution:
    def __init__(self, path, *, kind, started_s, config=None):
        if kind not in ('FINGER_RECONFIGURE', 'WRIST_PIVOT'):
            raise ValueError('Unknown recovery path kind')
        if (not np.isfinite(started_s) or path.selected_finger not in range(4)
                or path.anchor_finger not in range(4) or path.anchor_finger == path.selected_finger
                or not np.isfinite(path.duration_s) or path.duration_s <= 0):
            raise ValueError('Invalid selected finger, measured anchor, or path duration')
        self.path, self.kind, self.started_s = path, kind, float(started_s)
        self.config = config or FingerRecoveryExecutionConfig()
        self.elapsed_s = 0.
        self.contact_pose = None
        self.first_contact_s = self.stable_since = self.anchor_lost_since = None
        self.support_lost_since = None
        self.maximum_anchor_loss_s = 0.
        self.endpoint_since = self.tracking_pause_since = None
        self.status, self.reason = 'RUNNING', None
        self.last_time = None

    def update(self, time_s, measured_q, active, forces_n, tracking_error_rad):
        cfg = self.config
        now = float(time_s)
        q, active, force = np.asarray(measured_q, float), np.asarray(active, bool), np.asarray(forces_n, float)
        if (q.shape != (23,) or active.shape != (4,) or force.shape != (4,)
                or not np.isfinite(q).all() or not np.isfinite(force).all() or np.any(force < 0)
                or not np.isfinite(now) or now < self.started_s
                or self.last_time is not None and now <= self.last_time
                or not np.isfinite(tracking_error_rad) or tracking_error_rad < 0):
            raise ValueError('Expected increasing time and finite measured state/contact data')
        dt = 0. if self.last_time is None else now-self.last_time
        if dt > cfg.dt_s*1.01:
            raise ValueError('Recovery execution requires each native integration tick')
        self.last_time = now
        selected, anchor = self.path.selected_finger, self.path.anchor_finger
        safe = active & (force >= cfg.contact_min_n) & (force < cfg.force_pause_n)
        self.anchor_lost_since = (None if safe[anchor] else now
                                  if self.anchor_lost_since is None else self.anchor_lost_since)
        self.support_lost_since = (None if safe[anchor] or safe[selected] else now
                                   if self.support_lost_since is None else self.support_lost_since)
        anchor_loss = 0. if self.anchor_lost_since is None else now-self.anchor_lost_since
        self.maximum_anchor_loss_s = max(self.maximum_anchor_loss_s, anchor_loss)
        first_contact = False
        if self.status == 'RUNNING':
            if force.max() >= cfg.force_pause_n:
                self.status, self.reason = 'ABORTED', 'FORCE_RELEASE_REQUIRED'
            elif safe[selected]:
                if self.contact_pose is None:
                    self.contact_pose = q.copy()
                    self.first_contact_s = now
                    first_contact = True
                self.stable_since = now if self.stable_since is None else self.stable_since
                if now-self.stable_since >= cfg.stable_contact_s-1e-12:
                    self.status, self.reason = 'RECOVERED', 'MEASURED_SAFE_CONTACT_STABLE'
            else:
                self.stable_since = None
            if self.status == 'RUNNING':
                if (self.kind == 'WRIST_PIVOT' and self.support_lost_since is not None
                        and now-self.support_lost_since >= cfg.support_loss_grace_s-1e-12):
                    self.status, self.reason = 'ABORTED', 'MEASURED_ANCHOR_LOST'
            if self.status == 'RUNNING':
                if tracking_error_rad >= cfg.tracking_pause_rad:
                    self.tracking_pause_since = now if self.tracking_pause_since is None else self.tracking_pause_since
                    if now-self.tracking_pause_since >= cfg.tracking_timeout_s-1e-12:
                        self.status, self.reason = 'ABORTED', 'ARM_TRACKING_TIMEOUT'
                else:
                    self.tracking_pause_since = None
            if self.status == 'RUNNING' and self.contact_pose is None:
                scale = float(np.clip((cfg.tracking_pause_rad-tracking_error_rad)
                                      /(cfg.tracking_pause_rad-cfg.tracking_full_rad), 0., 1.))
                # The fixed-wrist path changes only the selected finger. Only
                # a wrist pivot requires the measured anchor to keep advancing.
                if self.kind == 'FINGER_RECONFIGURE' or safe[anchor]:
                    self.elapsed_s = min(self.path.duration_s, self.elapsed_s+dt*scale)
                if self.elapsed_s >= self.path.duration_s-1e-12:
                    self.endpoint_since = now if self.endpoint_since is None else self.endpoint_since
            if self.status == 'RUNNING':
                wait_start = self.first_contact_s if self.contact_pose is not None else self.endpoint_since
                if wait_start is not None and now-wait_start >= cfg.endpoint_search_s-1e-12:
                    self.status, self.reason = 'ABORTED', 'NO_STABLE_SELECTED_CONTACT'
                elif now-self.started_s >= self.path.duration_s+cfg.endpoint_search_s+cfg.tracking_timeout_s:
                    self.status, self.reason = 'ABORTED', 'RECOVERY_WALL_TIME_LIMIT'
        reference = (self.path.sample(self.elapsed_s) if self.contact_pose is None else self.contact_pose.copy())
        if self.status != 'RUNNING':
            reference = q.copy()
        seeking_allowed = self.status == 'RUNNING' and (
            self.kind == 'WRIST_PIVOT' or self.contact_pose is not None or self.endpoint_since is not None)
        return dict(nominal_q=reference, status=self.status, reason=self.reason,
                    selected_finger=selected, anchor_finger=anchor, path_kind=self.kind,
                    path_elapsed_s=self.elapsed_s, first_contact=first_contact,
                    first_contact_s=self.first_contact_s, stable_since_s=self.stable_since,
                    anchor_measured_active=bool(active[anchor]),
                    anchor_safe_contact=bool(safe[anchor]),
                    selected_safe_contact=bool(safe[selected]),
                    anchor_safe_loss_age_s=float(anchor_loss),
                    maximum_anchor_safe_loss_s=float(self.maximum_anchor_loss_s),
                    wrist_support_required=self.kind == 'WRIST_PIVOT',
                    search_enabled=seeking_allowed,
                    phase='FINGER_STABILIZE' if self.contact_pose is not None else 'FINGER_RECOVERY',
                    rebase_required=first_contact or self.status != 'RUNNING',
                    measured_recovery_success=self.status == 'RECOVERED', coverage_credit=False)
