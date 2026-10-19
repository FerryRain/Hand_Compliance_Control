"""Measured geometric contexts for bounded retry budgets, without plant writes.

Construct once per physical run and observe every native 2 ms tick, including
intentional air and relief. ``palm_pose_world`` is the measured palm position
and quaternion [x,y,z,w,qx,qy,qz], not a planned wrist pose or a target point.
An explicit grant can reopen finite *search/request* quotas after real motion
to an unused geometric context. It does not grant a path, prove reachability,
infer contact, change a controller or add coverage.

Every initial-loss, request and granted-context pose is retained throughout
the unresolved episode. A new pose must differ from EACH of these poses by
40 mm translation OR 20 degrees rotation. Returning A -> B -> A cannot renew
the quota. Ordinary stroke rebases have no API here. Only 150 ms of measured
safe contact by that same lost finger resolves its episode; lifetime counters
and the serializable event journal survive that resolution.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass, asdict
import numpy as np


@dataclass(frozen=True)
class RecoveryContextConfig:
    dt_s: float = .002
    minimum_unresolved_loss_s: float = 3.
    minimum_translation_m: float = .040
    minimum_rotation_deg: float = 20.
    minimum_stable_support_s: float = .150
    stable_recontact_s: float = .150
    context_cooldown_s: float = 20.
    finger_request_cooldown_s: float = 20.
    global_request_cooldown_s: float = 8.
    max_requests_per_context: int = 3
    active_force_min_n: float = .2
    safe_force_max_n: float = 8.
    surface_phases: tuple[str, ...] = ('SLIDE', 'LOCAL_SEARCH', 'REACQUIRE')

    def __post_init__(self):
        for key, value in vars(self).items():
            if key == 'surface_phases':
                if not value or not all(isinstance(x, str) and x for x in value):
                    raise ValueError('surface_phases requires nonempty phase names')
            elif not np.isfinite(value) or value <= 0:
                raise ValueError(f'{key} must be finite and positive')
        if type(self.max_requests_per_context) is not int:
            raise ValueError('max_requests_per_context must be an integer')
        if not set(self.surface_phases) <= {'SLIDE', 'LOCAL_SEARCH', 'REACQUIRE'}:
            raise ValueError('Intentional air, recovery paths and relief cannot grant contexts')
        if (abs(self.dt_s-.002) > 1e-12 or self.minimum_unresolved_loss_s < 3.
                or self.minimum_translation_m < .040 or self.minimum_rotation_deg < 20.
                or self.minimum_stable_support_s < .150 or self.stable_recontact_s < .150
                or self.context_cooldown_s < 20. or self.finger_request_cooldown_s < 20.
                or self.global_request_cooldown_s < 8. or self.max_requests_per_context > 3
                or self.active_force_min_n < .2 or self.safe_force_max_n > 8.
                or self.active_force_min_n >= self.safe_force_max_n):
            raise ValueError('Native observation and bounded retry requirements may not be relaxed')


class RecoveryContextObserver:
    """Observe physical histories and issue explicit one-shot context tokens.

    ``update`` reports geometric eligibility; it cannot know whether the
    caller's search/request/attempt quota is exhausted. The caller must pass
    ``budget_exhausted=True`` to ``grant_new_context`` only after verifying
    that condition. It must also choose when to run a recovery, preserve all
    force/collision/rate limits, and atomically apply an accepted token to
    both quota consumers. Receiving this token never itself starts a path.

    ``note_request`` records a monitor request already emitted on the latest
    observation. It checks the finite context quota and request cooldowns,
    and updates the pose baseline even when planning subsequently fails.
    Its accepted event is evidence of a request, never a granted motion.
    All mutation methods require the exact latest observation time. Returned
    dictionaries are JSON-serializable copies, independent of internal state.
    """
    names = ('index', 'middle', 'ring', 'thumb')

    def __init__(self, *, config=None):
        self.config = config or RecoveryContextConfig()
        self._time = None
        self._pose = None
        self._actual = self._safe = None
        self._measured_active = self._forces = None
        self._phase = None
        self._force_pause = False
        self._loss_start = [None]*4
        self._episode_start = [None]*4
        self._safe_start = [None]*4
        self._support_start = [None]*4
        self._reference_pose = [None]*4
        self._reference_time = [None]*4
        self._pose_history = [[] for _ in range(4)]
        self._episode_id = [0]*4
        self._context_id = [0]*4
        self._context_requests = [0]*4
        self._episode_requests = [0]*4
        self._lifetime_requests = [0]*4
        self._episode_grants = [0]*4
        self._lifetime_grants = [0]*4
        self._last_request = [None]*4
        self._last_global_request = None
        self._journal = []
        self._support_samples = deque(maxlen=int(np.ceil(self.config.minimum_stable_support_s/self.config.dt_s))+1)

    @property
    def events(self):
        """Independent copies of lifetime loss/request/grant/resolution events."""
        return deepcopy(self._journal)

    def _event(self, kind, finger, **fields):
        event = dict(kind=kind, time_s=self._time, finger=int(finger),
                     event_id=len(self._journal)+1, episode_id=self._episode_id[finger],
                     context_id=self._context_id[finger], contact_inferred=False,
                     coverage_credit=False, **fields)
        self._journal.append(deepcopy(event))
        return event

    def _consume_pose(self, finger, kind):
        item = dict(time_s=self._time, kind=kind, palm_pose_world=self._pose.tolist())
        self._pose_history[finger].append(item)
        self._reference_pose[finger] = self._pose.copy()
        self._reference_time[finger] = self._time

    def _pose_difference(self, pose):
        translation = float(np.linalg.norm(self._pose[:3]-np.asarray(pose)[:3]))
        cosine = float(np.clip(abs(np.dot(self._pose[3:], np.asarray(pose)[3:])), 0., 1.))
        rotation = float(np.degrees(2.*np.arccos(cosine)))
        return translation, rotation

    def _cooldown_left(self, since, duration):
        return 0. if since is None else max(0., duration-(self._time-since))

    def _diagnostic(self, finger):
        cfg = self.config
        strict_age = 0. if self._loss_start[finger] is None else self._time-self._loss_start[finger]
        episode_age = 0. if self._episode_start[finger] is None else self._time-self._episode_start[finger]
        safe_age = 0. if self._safe_start[finger] is None else self._time-self._safe_start[finger]
        supports = [(self._time-start, other) for other, start in enumerate(self._support_start)
                    if other != finger and start is not None]
        support_age, support_finger = max(supports, default=(0., None))
        stable_support = support_finger is not None and support_age >= cfg.minimum_stable_support_s-1e-12
        differences = [self._pose_difference(item['palm_pose_world']) for item in self._pose_history[finger]]
        novel = bool(differences and all(d >= cfg.minimum_translation_m-1e-12
                       or r >= cfg.minimum_rotation_deg-1e-9 for d, r in differences))
        reference_difference = (None, None) if self._reference_pose[finger] is None else self._pose_difference(self._reference_pose[finger])
        cooldown = self._cooldown_left(self._reference_time[finger], cfg.context_cooldown_s)
        reason = ('FINGER_CURRENTLY_IN_CONTACT' if self._actual[finger] else
                  'NO_UNRESOLVED_LOSS_EPISODE' if self._episode_start[finger] is None else
                  'UNRESOLVED_LOSS_TOO_SHORT' if episode_age < cfg.minimum_unresolved_loss_s-1e-12 else
                  'PHASE_NOT_ELIGIBLE' if self._phase not in cfg.surface_phases else
                  'MEASURED_FORCE_RELEASE_REQUIRED' if self._force_pause else
                  'NO_SAME_FINGER_STABLE_SAFE_SUPPORT' if not stable_support else
                  'CONTEXT_COOLDOWN' if cooldown > 1e-12 else
                  'PREVIOUSLY_CONSUMED_GEOMETRY' if not novel else 'NEW_MEASURED_GEOMETRY_ELIGIBLE')
        return dict(finger=finger, name=self.names[finger], eligible=reason=='NEW_MEASURED_GEOMETRY_ELIGIBLE',
                    reason=reason, context_id=self._context_id[finger], episode_id=self._episode_id[finger],
                    measured_contact=bool(self._actual[finger]), measured_safe_contact=bool(self._safe[finger]),
                    loss_age_s=float(strict_age), unresolved_loss_age=float(episode_age),
                    unresolved_loss_age_s=float(episode_age), stable_safe_contact_s=float(safe_age),
                    support_finger=support_finger, same_finger_safe_support_s=float(support_age),
                    context_cooldown_remaining_s=float(cooldown),
                    reference_translation_m=reference_difference[0], reference_rotation_deg=reference_difference[1],
                    new_relative_to_every_consumed_pose=novel, consumed_pose_count=len(differences),
                    requests_this_context=self._context_requests[finger],
                    request_budget_remaining=cfg.max_requests_per_context-self._context_requests[finger],
                    requests_this_episode=self._episode_requests[finger],
                    lifetime_requests=self._lifetime_requests[finger],
                    context_grants_this_episode=self._episode_grants[finger],
                    lifetime_context_grants=self._lifetime_grants[finger],
                    contact_inferred=False, coverage_credit=False)

    def update(self, time_s, palm_pose_world, measured_active, forces_n, phase):
        cfg = self.config
        now = float(time_s)
        pose = np.asarray(palm_pose_world, float)
        active, force = np.asarray(measured_active, bool), np.asarray(forces_n, float)
        if (not np.isfinite(now) or pose.shape != (7,) or not np.isfinite(pose).all()
                or np.linalg.norm(pose[3:]) < 1e-12 or active.shape != (4,)
                or force.shape != (4,) or not np.isfinite(force).all() or np.any(force < 0.)
                or not isinstance(phase, str) or not phase):
            raise ValueError('Expected finite native palm xyz+wxyz, active[4], nonnegative force[4] and phase')
        if self._time is not None and abs(now-self._time-cfg.dt_s) > 1e-8:
            raise ValueError('Recovery contexts must observe every native 2 ms tick')
        pose = pose.copy()
        pose[3:] /= np.linalg.norm(pose[3:])
        self._time, self._pose, self._phase = now, pose, phase
        self._measured_active, self._forces = active.copy(), force.copy()
        self._actual = active & (force >= cfg.active_force_min_n)
        self._safe = self._actual & (force < cfg.safe_force_max_n)
        self._force_pause = bool(np.any(force >= cfg.safe_force_max_n))
        self._support_samples.append(dict(time_s=now, phase=phase,
                                          measured_active=active.tolist(), forces_n=force.tolist()))
        before = len(self._journal)
        for finger in range(4):
            if self._actual[finger]:
                self._loss_start[finger] = None
            else:
                if self._loss_start[finger] is None:
                    self._loss_start[finger] = now
                if self._episode_start[finger] is None:
                    self._episode_start[finger] = now
                    self._episode_id[finger] += 1
                    self._consume_pose(finger, 'INITIAL_MEASURED_LOSS')
                    self._event('MEASURED_LOSS_EPISODE_STARTED', finger,
                                palm_pose_world=pose.tolist())
            if self._safe[finger]:
                if self._safe_start[finger] is None:
                    self._safe_start[finger] = now
                if (self._episode_start[finger] is not None
                        and now-self._safe_start[finger] >= cfg.stable_recontact_s-1e-12):
                    self._event('MEASURED_LOSS_EPISODE_RESOLVED', finger,
                                unresolved_duration_s=now-self._episode_start[finger],
                                stable_safe_contact_s=now-self._safe_start[finger],
                                requests_this_episode=self._episode_requests[finger],
                                context_grants_this_episode=self._episode_grants[finger],
                                consumed_poses=deepcopy(self._pose_history[finger]),
                                lifetime_requests=self._lifetime_requests[finger],
                                lifetime_context_grants=self._lifetime_grants[finger])
                    self._episode_start[finger] = None
                    self._episode_requests[finger] = self._context_requests[finger] = 0
                    self._episode_grants[finger] = 0
                    self._pose_history[finger] = []
                    self._reference_pose[finger] = self._reference_time[finger] = None
            else:
                self._safe_start[finger] = None
            # One physical support owns each timer. A gap/air/relief cannot be
            # banked or bridged by a different finger entering contact.
            if self._safe[finger] and phase in cfg.surface_phases and not self._force_pause:
                if self._support_start[finger] is None:
                    self._support_start[finger] = now
            else:
                self._support_start[finger] = None
        return dict(schema='i04.recovery_context.v1', time_s=now, phase=phase,
                    per_finger=[self._diagnostic(i) for i in range(4)],
                    events=deepcopy(self._journal[before:]), contact_inferred=False, coverage_credit=False)

    def _check_current(self, finger, time_s):
        if type(finger) is not int or finger not in range(4):
            raise ValueError('finger must identify one physical finger 0..3')
        if not np.isfinite(time_s):
            raise ValueError('Request/grant time must be finite')
        if self._time is None or abs(float(time_s)-self._time) > 1e-10:
            return dict(accepted=False, reason='STALE_OR_MISSING_NATIVE_OBSERVATION',
                        finger=finger, time_s=float(time_s), contact_inferred=False, coverage_credit=False)
        return None

    def note_request(self, finger, time_s):
        """Record an actually emitted monitor request; never authorize a path."""
        invalid = self._check_current(finger, time_s)
        if invalid is not None:
            return invalid
        cfg = self.config
        diag = self._diagnostic(finger)
        reason = ('FINGER_CURRENTLY_IN_CONTACT' if self._actual[finger] else
                  'UNRESOLVED_LOSS_TOO_SHORT' if diag['unresolved_loss_age'] < cfg.minimum_unresolved_loss_s-1e-12 else
                  'PHASE_NOT_ELIGIBLE' if self._phase not in cfg.surface_phases else
                  'MEASURED_FORCE_RELEASE_REQUIRED' if self._force_pause else
                  'CONTEXT_REQUEST_QUOTA_EXHAUSTED' if self._context_requests[finger] >= cfg.max_requests_per_context else
                  'FINGER_REQUEST_COOLDOWN' if self._cooldown_left(self._last_request[finger], cfg.finger_request_cooldown_s) > 1e-12 else
                  'GLOBAL_REQUEST_COOLDOWN' if self._cooldown_left(self._last_global_request, cfg.global_request_cooldown_s) > 1e-12 else None)
        if reason:
            return dict(accepted=False, reason=reason, finger=finger, time_s=self._time,
                        contact_inferred=False, coverage_credit=False)
        self._context_requests[finger] += 1
        self._episode_requests[finger] += 1
        self._lifetime_requests[finger] += 1
        self._last_request[finger] = self._last_global_request = self._time
        self._consume_pose(finger, 'MONITOR_REQUEST')
        return self._event('MEASURED_RECOVERY_REQUEST_RECORDED', finger, accepted=True,
                           palm_pose_world=self._pose.tolist(),
                           requests_this_context=self._context_requests[finger],
                           requests_this_episode=self._episode_requests[finger],
                           lifetime_requests=self._lifetime_requests[finger],
                           unresolved_loss_age=diag['unresolved_loss_age'], path_authorized=False)

    def grant_new_context(self, finger, time_s, *, budget_exhausted=False):
        """Consume one current measured context; caller applies both consumers."""
        invalid = self._check_current(finger, time_s)
        if invalid is not None:
            return invalid
        diag = self._diagnostic(finger)
        reason = ('BUDGET_NOT_EXPLICITLY_EXHAUSTED' if budget_exhausted is not True else
                  diag['reason'] if not diag['eligible'] else None)
        if reason:
            return dict(accepted=False, reason=reason, finger=finger, time_s=self._time,
                        context_id=self._context_id[finger], contact_inferred=False, coverage_credit=False)
        previous = deepcopy(self._pose_history[finger])
        evidence = dict(phase=self._phase, measured_active=self._measured_active.tolist(),
                        forces_n=self._forces.tolist(), palm_pose_world=self._pose.tolist(),
                        reference_time_s=self._reference_time[finger],
                        cooldown_elapsed_s=self._time-self._reference_time[finger],
                        comparisons_to_consumed_poses=[dict(time_s=item['time_s'], kind=item['kind'],
                            translation_m=self._pose_difference(item['palm_pose_world'])[0],
                            rotation_deg=self._pose_difference(item['palm_pose_world'])[1]) for item in previous],
                        native_support_window=deepcopy(list(self._support_samples)), parameters=asdict(self.config))
        old_context_requests = self._context_requests[finger]
        self._context_id[finger] += 1
        self._context_requests[finger] = 0
        self._episode_grants[finger] += 1
        self._lifetime_grants[finger] += 1
        self._consume_pose(finger, 'GRANTED_RECOVERY_CONTEXT')
        return self._event('MEASURED_RECOVERY_CONTEXT_GRANTED', finger, accepted=True,
                           palm_pose_world=self._pose.tolist(), consumed_poses_before=previous,
                           previous_context_requests=old_context_requests,
                           requests_this_episode=self._episode_requests[finger],
                           lifetime_requests=self._lifetime_requests[finger],
                           context_grants_this_episode=self._episode_grants[finger],
                           lifetime_context_grants=self._lifetime_grants[finger],
                           unresolved_loss_age=diag['unresolved_loss_age'], loss_age_s=diag['loss_age_s'],
                           support_finger=diag['support_finger'],
                           same_finger_safe_support_s=diag['same_finger_safe_support_s'],
                           evidence=evidence,
                           budget_exhausted=True, reopen_local_search_budget=True,
                           clear_loss_history=False, path_authorized=False)
