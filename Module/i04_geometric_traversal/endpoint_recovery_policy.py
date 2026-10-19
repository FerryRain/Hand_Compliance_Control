"""Finite anchored recovery permission after repeated real endpoint failures.

This observer neither plans nor commands motion. Observe at the end of every
native tick, then register a natural-endpoint queue dispatch. On the following
tick a failed fixed-wrist plan may be considered using freshly read sensors.
The parent still owns collision/preload checks, actual-endpoint replanning,
force guards, its 60 s attempt deadline and its 45 s no-progress limit.
"""
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class EndpointRecoveryConfig:
    dt_s: float = .002
    minimum_unresolved_loss_s: float = 30.
    minimum_failed_contexts: int = 2
    stable_support_s: float = .150
    stable_recontact_s: float = .150
    escalation_cooldown_s: float = 60.
    active_force_min_n: float = .2
    safe_force_max_n: float = 8.
    support_phases: tuple = ('SLIDE', 'SELECT', 'LOCAL_SEARCH', 'REACQUIRE')
    qualifying_failure_reasons: tuple = ('NO_SAFE_PATH_AMONG_BOUNDED_SURFACE_TARGETS',)

    def __post_init__(self):
        for key,value in vars(self).items():
            if key in ('support_phases','qualifying_failure_reasons'):
                if not value or not all(isinstance(x,str) and x for x in value):raise ValueError(key)
            elif not np.isfinite(value) or value<=0:raise ValueError(key)
        if (abs(self.dt_s-.002)>1e-12 or self.minimum_unresolved_loss_s<30.
                or type(self.minimum_failed_contexts) is not int or self.minimum_failed_contexts<2
                or self.stable_support_s<.150 or self.stable_recontact_s<.150
                or self.escalation_cooldown_s<60. or self.active_force_min_n<.2
                or self.safe_force_max_n>8. or self.active_force_min_n>=self.safe_force_max_n):
            raise ValueError('Physical history, force and finite escalation limits may not be relaxed')
        if (not set(self.support_phases)<=set(('SLIDE','SELECT','LOCAL_SEARCH','REACQUIRE'))
                or not set(self.qualifying_failure_reasons)<=set(('NO_SAFE_PATH_AMONG_BOUNDED_SURFACE_TARGETS',))):
            raise ValueError('Only surface support and fixed-wrist geometric failure qualify')


class EndpointRecoveryPolicy:
    def __init__(self, *, config=None):
        self.config=config or EndpointRecoveryConfig()
        self._time=None
        self._episode_start=[None]*4
        self._safe_since=[None]*4
        self._failures=[[] for _ in range(4)]
        self._escalated_contexts=[set() for _ in range(4)]
        self._last_escalation=np.full(4,-np.inf)
        self._episode_ids=np.zeros(4,int);self._context_ids=np.zeros(4,int)
        self._dispatches={};self._seen_failures=set();self._events=[]
        self._samples=deque(maxlen=int(np.ceil(self.config.stable_support_s/.002))+1)

    @property
    def events(self):return deepcopy(self._events)

    @staticmethod
    def _measurement(active,forces,pose,phase):
        active=np.asarray(active,bool);forces=np.asarray(forces,float);pose=np.asarray(pose,float)
        if (active.shape!=(4,) or forces.shape!=(4,) or pose.shape!=(7,)
                or not np.isfinite(forces).all() or np.any(forces<0.) or not np.isfinite(pose).all()
                or np.linalg.norm(pose[3:])<1e-12 or not isinstance(phase,str) or not phase):
            raise ValueError('Expected measured active[4], force[4], palm xyz+wxyz and phase')
        pose=pose.copy();pose[3:]/=np.linalg.norm(pose[3:])
        return active,forces,pose

    def update(self,time_s,measured_active,forces_n,episode_ids,context_ids,phase,palm_pose_world):
        now=float(time_s);active,forces,pose=self._measurement(measured_active,forces_n,palm_pose_world,phase)
        episodes,contexts=np.asarray(episode_ids),np.asarray(context_ids)
        for ids,old in ((episodes,self._episode_ids),(contexts,self._context_ids)):
            if ids.shape!=(4,) or ids.dtype.kind not in ('i','u') or np.any(ids<old):raise ValueError('Monotonic physical IDs required')
        for finger in range(4):
            if episodes[finger]!=self._episode_ids[finger] and self._episode_ids[finger]>0 and self._failures[finger]:
                raise ValueError('External episode changed without observed stable recovery')
        if not np.isfinite(now) or (self._time is not None and abs(now-self._time-self.config.dt_s)>1e-8):
            raise ValueError('Observe exactly every native 2 ms tick')
        cfg=self.config;actual=active&(forces>=cfg.active_force_min_n);safe=actual&(forces<cfg.safe_force_max_n)
        resolved=[]
        for finger in range(4):
            if not actual[finger] and self._episode_start[finger] is None:self._episode_start[finger]=now
            if safe[finger]:
                if self._safe_since[finger] is None:self._safe_since[finger]=now
                if self._episode_start[finger] is not None and now-self._safe_since[finger]>=cfg.stable_recontact_s-1e-12:
                    self._episode_start[finger]=None;self._failures[finger]=[];self._escalated_contexts[finger].clear();resolved.append(finger)
            else:self._safe_since[finger]=None
        self._time=now;self._episode_ids=episodes.copy();self._context_ids=contexts.copy()
        self._samples.append(dict(time_s=now,active=active.tolist(),forces_n=forces.tolist(),phase=phase,palm_pose_world=pose.tolist()))
        return dict(time_s=now,resolved_fingers=resolved,
                    unresolved_loss_age_s=[0. if x is None else now-x for x in self._episode_start],
                    failed_context_count=[len({x['context_id'] for x in records}) for records in self._failures])

    def note_boundary_dispatch(self,event):
        """Remember one queue-owned natural endpoint, not an arbitrary SELECT."""
        request_id=event.get('request_id');finger=event.get('finger')
        if (event.get('kind')!='RECOVERY_REQUEST_TAKEN_AT_BOUNDARY' or event.get('accepted') is not True
                or event.get('mode')!='SHORT_FIXED_WRIST' or event.get('natural_stroke_endpoint') is not True
                or event.get('phase')!='SELECT' or type(request_id) is not int or type(finger) is not int
                or finger not in range(4) or self._time is None or abs(event['time_s']-self._time)>1e-10):
            return dict(accepted=False,reason='NOT_CURRENT_NATURAL_ENDPOINT_DISPATCH')
        if request_id in self._dispatches or request_id in self._seen_failures:
            return dict(accepted=False,reason='DISPATCH_ALREADY_CONSUMED')
        if event['episode_id']!=self._episode_ids[finger] or event['context_id']!=self._context_ids[finger]:
            return dict(accepted=False,reason='DISPATCH_CONTEXT_MISMATCH')
        measured=self._samples[-1]
        if measured['phase']!='SELECT' or measured['active'][finger] or max(measured['forces_n'])>=self.config.safe_force_max_n:
            return dict(accepted=False,reason='DISPATCH_CURRENT_MEASUREMENT_MISMATCH')
        self._dispatches[request_id]=deepcopy(event)
        return dict(accepted=True,request_id=request_id,finger=finger)

    def consider_fixed_failure(self,time_s,finger,request_id,reason,*,measured_active,forces_n,
                               palm_pose_world,phase,episode_id,context_id):
        """Consume a previous-tick dispatch using this tick's actual sensors.

        A qualifying failure records finite-search failure, never a proof of
        global unreachability. Granting permission does not renew any budget.
        The eventual wrist path MUST select one of stable_support_fingers.
        """
        if type(finger) is not int or finger not in range(4):raise ValueError('Physical finger required')
        now=float(time_s);active,forces,pose=self._measurement(measured_active,forces_n,palm_pose_world,phase)
        if not np.isfinite(now) or type(request_id) is not int or request_id<=0:raise ValueError('Finite native time and identified request required')
        cfg=self.config;dispatch=self._dispatches.get(request_id)
        invalid=('NOT_NEXT_NATIVE_TICK' if self._time is None or abs(now-self._time-cfg.dt_s)>1e-8 else
                 'UNKNOWN_OR_USED_ENDPOINT_DISPATCH' if dispatch is None or request_id in self._seen_failures else
                 'DISPATCH_NOT_PREVIOUS_TICK' if abs(dispatch['time_s']-self._time)>1e-10 else
                 'WRONG_DISPATCH_FINGER' if dispatch['finger']!=finger else
                 'CURRENT_CONTEXT_CHANGED' if episode_id!=dispatch['episode_id'] or context_id!=dispatch['context_id']
                    or episode_id!=self._episode_ids[finger] or context_id!=self._context_ids[finger] else
                 'NOT_CURRENT_NATURAL_ENDPOINT' if phase!='SELECT' else None)
        if invalid:return dict(accepted=False,reason=invalid,kind='ENDPOINT_RECOVERY_ESCALATION_REJECTED',time_s=now,finger=finger,request_id=request_id)
        self._seen_failures.add(request_id);self._dispatches.pop(request_id)
        actual=active&(forces>=cfg.active_force_min_n)
        failure=dict(time_s=now,request_id=request_id,finger=finger,episode_id=episode_id,context_id=context_id,
                     reason=reason,palm_pose_world=pose.tolist())
        qualifies=reason in cfg.qualifying_failure_reasons and not actual[finger] and forces.max()<cfg.safe_force_max_n
        if qualifies:self._failures[finger].append(failure)
        current=dict(time_s=now,active=active.tolist(),forces_n=forces.tolist(),phase=phase,palm_pose_world=pose.tolist())
        required=int(np.ceil(cfg.stable_support_s/cfg.dt_s))+1
        window=(list(self._samples)+[current])[-required:]
        supported=[]
        if len(window)==required and window[-1]['time_s']-window[0]['time_s']>=cfg.stable_support_s-1e-10:
            for other in range(4):
                if other!=finger and all(row['active'][other] and cfg.active_force_min_n<=row['forces_n'][other]<cfg.safe_force_max_n
                        and max(row['forces_n'])<cfg.safe_force_max_n and row['phase'] in cfg.support_phases for row in window):supported.append(other)
        age=0. if self._episode_start[finger] is None else now-self._episode_start[finger]
        contexts=sorted({item['context_id'] for item in self._failures[finger]})
        cooldown=max(0.,cfg.escalation_cooldown_s-(now-self._last_escalation[finger]))
        denial=('FIXED_FAILURE_NOT_GEOMETRIC_INFEASIBILITY' if reason not in cfg.qualifying_failure_reasons else
                'FINGER_CURRENTLY_IN_CONTACT' if actual[finger] else
                'CURRENT_FORCE_RELEASE_REQUIRED' if forces.max()>=cfg.safe_force_max_n else
                'UNRESOLVED_LOSS_TOO_SHORT' if age<cfg.minimum_unresolved_loss_s-1e-10 else
                'INSUFFICIENT_DISTINCT_FAILED_CONTEXTS' if len(contexts)<cfg.minimum_failed_contexts else
                'NO_STABLE_ACTUAL_SUPPORT' if not supported else
                'CONTEXT_ALREADY_ESCALATED' if context_id in self._escalated_contexts[finger] else
                'PER_FINGER_ESCALATION_COOLDOWN' if cooldown>1e-10 else None)
        accepted=denial is None
        event=dict(kind='ENDPOINT_RECOVERY_ESCALATION_GRANTED' if accepted else 'ENDPOINT_RECOVERY_ESCALATION_DENIED',
            event_id=len(self._events)+1,time_s=now,finger=finger,request_id=request_id,accepted=accepted,
            reason='REPEATED_FIXED_WRIST_FAILURE_AT_NATURAL_ENDPOINT' if accepted else denial,
            episode_id=episode_id,context_id=context_id,dispatch_time_s=dispatch['time_s'],
            failed_attempts=deepcopy(self._failures[finger]),current_fixed_failure=failure,
            distinct_failed_context_ids=contexts,unresolved_loss_age_s=age,
            stable_support_fingers=supported,native_support_window=deepcopy(window),
            previous_escalation_time_s=None if not np.isfinite(self._last_escalation[finger]) else float(self._last_escalation[finger]),
            escalation_cooldown_remaining_s=cooldown,search_budget_renewed=False,
            request_budget_renewed=False,clear_loss_history=False,contact_inferred=False,coverage_credit=False,
            authority='ANCHOR_PIVOT_CANDIDATE_PERMISSION_ONLY_ORIGINAL_GUARDS_AND_DEADLINES_REQUIRED')
        if accepted:self._last_escalation[finger]=now;self._escalated_contexts[finger].add(context_id)
        self._events.append(deepcopy(event));return event
