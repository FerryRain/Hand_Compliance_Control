"""Deferred diagnostic recovery requests at measured trajectory boundaries.

The queue never plans a path or commands physics. A SLIDE request can wait
until a caller-confirmed natural stroke endpoint; taking it there constrains
the caller to a fixed-wrist, selected-finger path no longer than eight seconds.
LOCAL_SEARCH can return an existing full-fallback request, whose original
safety and attempt limits remain the caller's responsibility. A take is a
one-shot transfer of a diagnostic request, not authorization of an arbitrary
path. Plans must be computed from the current measured state.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import numpy as np


@dataclass(frozen=True)
class RecoveryQueueConfig:
    dt_s: float = .002
    request_ttl_s: float = 40.
    stable_recontact_s: float = .150
    maximum_endpoint_path_duration_s: float = 8.
    active_force_min_n: float = .2
    safe_force_max_n: float = 8.

    def __post_init__(self):
        if any(not np.isfinite(value) or value <= 0. for value in vars(self).values()):
            raise ValueError('Queue settings must be finite and positive')
        if (abs(self.dt_s-.002) > 1e-12 or self.stable_recontact_s < .150
                or self.maximum_endpoint_path_duration_s > 8.
                or self.active_force_min_n < .2 or self.safe_force_max_n > 8.
                or self.active_force_min_n >= self.safe_force_max_n):
            raise ValueError('Native contact and short endpoint-path limits may not be relaxed')


class RecoveryRequestQueue:
    """At most one pending diagnostic request per physical finger.

    Observe every native tick, including air and relief. ``episode_ids`` and
    ``context_ids`` are the current IDs from the measured context observer,
    never a planned future context. ``enqueue`` copies a monitor request that
    was actually emitted on the latest tick. It spends no new request quota;
    accepted, discarded and taken requests never refund the monitor's quota.

    A short measured contact temporarily prevents taking the request. Only
    same-finger safe contact continuously lasting 150 ms drops it as recovered.
    Episode/context changes and a 40 s TTL discard stale requests with explicit
    events. The caller should supply ``pending_mask`` to any future monitor
    mask API to avoid emitting redundant requests; this queue changes no monitor.
    """
    def __init__(self, *, config=None):
        self.config = config or RecoveryQueueConfig()
        self._time = self._phase = None
        self._actual = self._forces = None
        self._episodes = self._contexts = None
        self._safe_since = [None]*4
        self._pending = [None]*4
        self._seen_request_ids = set()
        self._journal = []
        self._select_generation = 0
        self._taken_select_generation = None
        self._last_take_time = None

    @property
    def events(self):
        return deepcopy(self._journal)

    @property
    def pending_mask(self):
        return [item is not None for item in self._pending]

    def _emit(self, kind, **fields):
        event = dict(kind=kind, event_id=len(self._journal)+1, time_s=self._time,
                     budget_refunded=False, contact_inferred=False, coverage_credit=False,
                     **fields)
        self._journal.append(deepcopy(event))
        return event

    def _drop(self, finger, reason):
        pending = self._pending[finger]
        self._pending[finger] = None
        return self._emit('RECOVERY_REQUEST_DROPPED', finger=finger, reason=reason,
                          request_id=pending['request_id'], request_time_s=pending['request_time_s'],
                          request_age_s=self._time-pending['request_time_s'],
                          request_episode_id=pending['episode_id'], request_context_id=pending['context_id'],
                          current_episode_id=int(self._episodes[finger]),
                          current_context_id=int(self._contexts[finger]),
                          stable_safe_contact_s=(0. if self._safe_since[finger] is None else
                                                 self._time-self._safe_since[finger]))

    @staticmethod
    def _ids(value, name):
        values = np.asarray(value)
        if values.shape != (4,) or values.dtype.kind not in ('i', 'u') or np.any(values < 0):
            raise ValueError(f'{name} must contain four nonnegative physical integer IDs')
        return values.copy()

    def update(self, time_s, measured_active, forces_n, episode_ids, context_ids, phase):
        now = float(time_s)
        active, forces = np.asarray(measured_active, bool), np.asarray(forces_n, float)
        episodes, contexts = self._ids(episode_ids, 'episode_ids'), self._ids(context_ids, 'context_ids')
        if (not np.isfinite(now) or active.shape != (4,) or forces.shape != (4,)
                or not np.isfinite(forces).all() or np.any(forces < 0.)
                or not isinstance(phase, str) or not phase):
            raise ValueError('Expected finite native actual contacts, nonnegative forces and phase')
        if self._time is not None:
            if abs(now-self._time-self.config.dt_s) > 1e-8:
                raise ValueError('Recovery request queues must observe every native 2 ms tick')
            if np.any(episodes < self._episodes) or np.any(contexts < self._contexts):
                raise ValueError('Physical episode and context IDs cannot regress')
        if phase == 'SELECT' and self._phase != 'SELECT':
            self._select_generation += 1
        self._time, self._phase = now, phase
        self._episodes, self._contexts, self._forces = episodes, contexts, forces.copy()
        self._actual = active & (forces >= self.config.active_force_min_n)
        safe = self._actual & (forces < self.config.safe_force_max_n)
        drop_events = []
        for finger in range(4):
            if safe[finger]:
                if self._safe_since[finger] is None:
                    self._safe_since[finger] = now
            else:
                self._safe_since[finger] = None
            pending = self._pending[finger]
            if pending is None:
                continue
            reason = ('ACTUAL_STABLE_SAME_FINGER_RECONTACT' if self._safe_since[finger] is not None
                      and now-self._safe_since[finger] >= self.config.stable_recontact_s-1e-12 else
                      'PHYSICAL_LOSS_EPISODE_CHANGED' if episodes[finger] != pending['episode_id'] else
                      'MEASURED_GEOMETRY_CONTEXT_CHANGED' if contexts[finger] != pending['context_id'] else
                      'REQUEST_TTL_EXPIRED' if now-pending['request_time_s'] >= self.config.request_ttl_s-1e-12 else None)
            if reason:
                drop_events.append(self._drop(finger, reason))
        return dict(schema='i04.recovery_request_queue.v1', time_s=now, phase=phase,
                    pending_mask=self.pending_mask, pending=deepcopy(self._pending),
                    drop_events=drop_events, budget_refunded=False, contact_inferred=False, coverage_credit=False)

    def _current(self, time_s):
        if not np.isfinite(time_s):
            raise ValueError('Request/boundary time must be finite')
        return self._time is not None and abs(float(time_s)-self._time) <= 1e-10

    def enqueue(self, request, *, episode_id, context_id):
        """Copy a newly emitted monitor request; duplicate pending work is rejected."""
        item = deepcopy(request)
        json.dumps(item, allow_nan=False)
        finger, request_id = item.get('finger'), item.get('request_id')
        if (item.get('kind') != 'FINGER_POSTURE_RECOVERY_REQUEST'
                or type(finger) is not int or finger not in range(4)
                or type(request_id) is not int or request_id <= 0
                or type(episode_id) is not int or episode_id <= 0
                or type(context_id) is not int or context_id < 0
                or 'time_s' not in item):
            raise ValueError('Expected a physical, identified monitor request and current episode/context IDs')
        reason = ('STALE_OR_MISSING_NATIVE_OBSERVATION' if not self._current(item['time_s']) else
                  'REQUEST_ALREADY_SEEN' if request_id in self._seen_request_ids else
                  'PHYSICAL_LOSS_EPISODE_CHANGED' if episode_id != self._episodes[finger] else
                  'MEASURED_GEOMETRY_CONTEXT_CHANGED' if context_id != self._contexts[finger] else
                  'SOURCE_REQUEST_CONTEXT_MISMATCH' if item.get('recovery_context_id', context_id) != context_id else
                  'FINGER_CURRENTLY_IN_CONTACT' if self._actual[finger] else
                  'FINGER_ALREADY_PENDING' if self._pending[finger] is not None else None)
        if reason:
            if self._current(item['time_s']):
                self._seen_request_ids.add(request_id)
            return self._emit('RECOVERY_REQUEST_ENQUEUE_REJECTED', accepted=False, reason=reason,
                              finger=finger, request_id=request_id, request_time_s=float(item['time_s']),
                              episode_id=episode_id, context_id=context_id)
        self._seen_request_ids.add(request_id)
        self._pending[finger] = dict(finger=finger, request_id=request_id,
                                    request_time_s=float(item['time_s']), episode_id=episode_id,
                                    context_id=context_id, request=item)
        return self._emit('RECOVERY_REQUEST_QUEUED', accepted=True, finger=finger,
                          request_id=request_id, request_time_s=float(item['time_s']),
                          episode_id=episode_id, context_id=context_id, pending_mask=self.pending_mask)

    def take_boundary(self, time_s, *, natural_stroke_endpoint=False):
        """Take oldest eligible request once; return policy, never a planned path.

        The endpoint flag must come from an actually completed stroke event,
        not merely entering SELECT for another reason. At most one take is
        allowed per SELECT visit. LOCAL_SEARCH remains the original fallback;
        no new time/attempt budget is conferred there.
        """
        reason = ('STALE_OR_MISSING_NATIVE_OBSERVATION' if not self._current(time_s) else
                  'ALREADY_TAKEN_THIS_NATIVE_TICK' if self._last_take_time == self._time else
                  'MEASURED_FORCE_RELEASE_REQUIRED' if np.any(self._forces >= self.config.safe_force_max_n) else
                  'NO_NATURAL_STROKE_ENDPOINT' if self._phase == 'SELECT' and natural_stroke_endpoint is not True else
                  'ENDPOINT_ALREADY_USED' if self._phase == 'SELECT' and self._taken_select_generation == self._select_generation else
                  'NOT_A_RECOVERY_BOUNDARY' if self._phase not in ('SELECT', 'LOCAL_SEARCH') else None)
        candidates = [item for finger, item in enumerate(self._pending)
                      if item is not None and not self._actual[finger]] if self._time is not None else []
        if reason is None and not candidates:
            reason = 'NO_CURRENTLY_LOST_PENDING_FINGER'
        if reason:
            return dict(accepted=False, reason=reason, time_s=self._time,
                        budget_refunded=False, contact_inferred=False, coverage_credit=False)
        pending = min(candidates, key=lambda p: (p['request_time_s'], p['request_id']))
        finger = pending['finger']
        self._pending[finger] = None
        self._last_take_time = self._time
        short = self._phase == 'SELECT'
        if short:
            self._taken_select_generation = self._select_generation
        return self._emit('RECOVERY_REQUEST_TAKEN_AT_BOUNDARY', accepted=True, finger=finger,
                          request_id=pending['request_id'], request_time_s=pending['request_time_s'],
                          request_age_s=self._time-pending['request_time_s'],
                          episode_id=pending['episode_id'], context_id=pending['context_id'],
                          request=deepcopy(pending['request']), phase=self._phase,
                          natural_stroke_endpoint=bool(short and natural_stroke_endpoint),
                          mode='SHORT_FIXED_WRIST' if short else 'FULL_FALLBACK',
                          maximum_path_duration_s=self.config.maximum_endpoint_path_duration_s if short else None,
                          selected_finger_only=short, fixed_wrist_required=short,
                          plan_from_current_measured_state=True, path_authorized=False,
                          external_safety_and_attempt_limits_unchanged=True)
