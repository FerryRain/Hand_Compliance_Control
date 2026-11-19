"""Outer regrasp v2: sustained low contact AND extreme posture; bounded retreat.

Frozen DP/MCC. Tactile and joint feedback only; no runtime object geometry.
"""
from dataclasses import dataclass
import torch
from smooth_regrasp_reference import regrasp_class as original_factory
from palm_posture_reset import TRACK, UNLOAD, RETREAT, REPOSTURE, APPROACH, RELOAD, DP_WARMUP, FAILED


@dataclass(frozen=True)
class AdaptiveConfig:
    low_contact_frames: int = 100
    edge_fraction: float = .02
    folded_fraction: float = .90
    max_retreat_m: float = .065
    retreat_increment_m: float = .005
    blocked_frames: int = 20
    stalled_frames: int = 50
    recovery_timeout_frames: int = 2500

    def __post_init__(self):
        if not (0 < self.edge_fraction < .1 and .8 <= self.folded_fraction < 1):
            raise ValueError('Invalid extreme-posture envelope')
        if not (.025 < self.max_retreat_m <= .08 and 0 < self.retreat_increment_m <= .01):
            raise ValueError('Invalid bounded retreat envelope')
        if min(self.low_contact_frames,self.blocked_frames,self.stalled_frames,self.recovery_timeout_frames)<1:
            raise ValueError('Positive frame counts required')


def extreme_posture(q,lower,upper,config):
    fraction=(q-lower)/(upper-lower)
    at_edge=((fraction<=config.edge_fraction)|(fraction>=1-config.edge_fraction)).any(-1)
    # PIP/DIP flexion coordinates occupy the last two entries of each block.
    folded=(fraction.reshape(-1,4,4)[...,2:]>=config.folded_fraction).all(-1).any(-1)
    return at_edge|folded


def update_trigger(q,strict,lower,upper,count,eligible,config):
    extreme=extreme_posture(q,lower,upper,config)
    low=strict.sum(-1)<=1
    count=torch.where(eligible&extreme&low,count+1,0)
    return count,count>=config.low_contact_frames,extreme,low


def adaptive_class(settings):
    config=AdaptiveConfig(**settings)
    Base=original_factory()
    class AdaptiveRegrasp(Base):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            if self.cfg.retreat_m>config.max_retreat_m:raise ValueError('Initial retreat exceeds envelope')
            self.trigger_streak=torch.zeros_like(self.count)
            self.extreme=torch.zeros_like(self.initial_ready);self.low_contact=self.extreme.clone()
            self.retreat_goal=torch.full_like(self.distance,self.cfg.retreat_m)
            self.extra_retreat_count=torch.zeros_like(self.count)
            self.blocked_count=torch.zeros_like(self.count);self.stalled_count=torch.zeros_like(self.count)
            self.last_reset_error=torch.full_like(self.distance,float('inf'))
            self.handoff_load_floor=torch.zeros(self.batch,4,device=self.device)
            self.handoff_reference=torch.zeros_like(self.reset_q)

        def observe(self,q,pose,tactile):
            strict=tactile['pad_found']&(tactile['pad_force'].norm(dim=-1)>=.05)&torch.isfinite(tactile['pad_points']).all(-1)
            eligible=(self.phase==TRACK)&self.initial_ready&(self.cooldown==0)
            self.trigger_streak,trigger,self.extreme,self.low_contact=update_trigger(
                q,strict,self.lower,self.upper,self.trigger_streak,eligible,config)
            # An explicit diagnostic reset remains possible; natural triggering
            # must never inherit the base class's reserve-only OR loss-only rule.
            explicit=self.external_reset_pending.clone()
            old_cooldown=self.cooldown.clone();old_count=self.count.clone()
            suppress=(self.phase==TRACK)&self.initial_ready&~(trigger|explicit)
            self.cooldown=torch.where(suppress,self.cooldown.clamp_min(1),self.cooldown)
            self.external_reset_pending |= trigger
            super().observe(q,pose,tactile)
            self.cooldown=old_cooldown
            new=self.count!=old_count
            if bool(new.any()):
                self.handoff_load_floor[new]=0
                self.retreat_goal=torch.where(new,self.cfg.retreat_m,self.retreat_goal)
                self.extra_retreat_count=torch.where(new,0,self.extra_retreat_count)
                for event in self.events:
                    if event['frame']==self.wall_frame and event['phase']=='unload':
                        lane=event['lane'];event['reason']=('explicit_interface_probe' if bool(explicit[lane]) else 'sustained_extreme_posture_AND_zero_or_one_contact')
            error=(q-self.reset_q).abs().amax(-1)
            reshaping=(self.phase==REPOSTURE)&(error>=.08)
            permission=self.chain.mcc.free_posture_reset.all(-1)
            self.blocked_count=torch.where(reshaping&~permission,self.blocked_count+1,0)
            progress=(self.last_reset_error-error)>.0005
            self.stalled_count=torch.where(reshaping&~progress,self.stalled_count+1,0)
            self.last_reset_error=error

        def apply_nominal(self,q,nominal,load):
            nominal,load=super().apply_nominal(q,nominal,load)
            # Do not unload a successfully re-established contact merely
            # because the next phase starts its ramp at age zero.
            load=torch.where((self.phase==RELOAD)[:,None],
                             torch.maximum(load,self.handoff_load_floor),load)
            return nominal,load

        def _transition(self,mask,phase,reason=None):
            if reason=='retreat_reached':mask &= self.distance>=self.retreat_goal-1e-7
            super()._transition(mask,phase,reason)

        def select_reference(self,frame,poses,twists,commands):
            before=None if self.previous_pose is None else self.previous_pose.clone()
            previous_distance=self.distance.clone()
            pose,twist,command=super().select_reference(frame,poses,twists,commands)
            distance=torch.minimum(previous_distance+self.cfg.palm_speed_m_s*.01,self.retreat_goal)
            self.distance=torch.where(self.phase==RETREAT,distance,self.distance)
            # The base FAILED branch caps at the initial retreat distance;
            # never allow that to move an extended retreat inward on failure.
            self.distance=torch.where(self.phase==FAILED,previous_distance,self.distance)
            pose[:,:3]=self.current_reference[:,:3]+self.offset+self.axis*self.distance[:,None]
            if before is not None:twist[:,:3]=(pose[:,:3]-before[:,:3])/.01
            self.previous_pose=pose.clone()
            return pose,twist,command

        def after_frame(self,env,frame):
            if not self.observed:return
            blocked=(self.blocked_count>=config.blocked_frames)|(self.stalled_count>=config.stalled_frames)
            extend=(self.phase==REPOSTURE)&blocked&(self.retreat_goal<config.max_retreat_m-1e-7)
            if bool(extend.any()):
                self.retreat_goal=torch.where(extend,(self.retreat_goal+config.retreat_increment_m).clamp_max(config.max_retreat_m),self.retreat_goal)
                self.open_q=torch.where(extend[:,None],env.scene['robot'].data.joint_pos[:,-16:],self.open_q)
                self.chain.mcc.free_posture_reset=self.chain.mcc.free_posture_reset & ~extend[:,None]
                self.extra_retreat_count+=extend.long()
                self._transition(extend,RETREAT,'more_space_before_reshaping')
                self.blocked_count=torch.where(extend,0,self.blocked_count)
                self.stalled_count=torch.where(extend,0,self.stalled_count)
            before=self.phase.clone()
            super().after_frame(env,frame)
            secured=(before==APPROACH)&(self.phase==RELOAD)
            if bool(secured.any()):
                # Transfer the already-issued PRE-MCC reference, including
                # bounded search, into the hold. This is not the actuator
                # target: transferring that would double-count compliance.
                issued=self.chain._reference_motion_filter.position
                self.reset_q=torch.where(secured[:,None],issued,self.reset_q)
                self.handoff_reference=torch.where(secured[:,None],issued,self.handoff_reference)
                self.handoff_load_floor=torch.where(secured[:,None],self.chain.mcc.load_request,self.handoff_load_floor)
                self.chain.runtime_recovery_offset[secured]=0
                self.chain.recovery_active[secured]=False
                self.chain.loss_streak[secured]=0
                self.chain.recovery_confirm_streak[secured]=0
            active=(self.phase!=TRACK)&(self.phase!=FAILED)&(self.phase!=8)
            expired=active&(self.wall_frame-self.reset_start_frame>=config.recovery_timeout_frames)
            self._transition(expired,FAILED,'bounded_recovery_time_exhausted')
            self.chain.execution_finished=bool(((self.phase==8)|(self.phase==FAILED)).all())

        def diagnostics(self):
            return dict(super().diagnostics(),adaptive_extreme_posture=self.extreme.clone(),
                adaptive_low_contact=self.low_contact.clone(),adaptive_trigger_streak=self.trigger_streak.clone(),
                adaptive_retreat_goal_m=self.retreat_goal.clone(),adaptive_extra_retreat_count=self.extra_retreat_count.clone(),
                adaptive_blocked_count=self.blocked_count.clone(),adaptive_stalled_count=self.stalled_count.clone(),
                adaptive_handoff_load_floor=self.handoff_load_floor.clone(),
                adaptive_handoff_reference=self.handoff_reference.clone())

        def report(self):
            return dict(super().report(),adaptive_regrasp=vars(config),
                additional_trigger='replaced: no reserve-only or contact-only midpath reset',
                natural_reset='extreme joint posture AND <=1 strict contact for100 consecutive frames; not low reserve alone',
                handoff='latch issued pre-MCC reference and load floor; suppress competing legacy search during reload/history rebuild',
                extra_retreat_steps=self.extra_retreat_count.cpu().tolist())
    return AdaptiveRegrasp
