"""Opt-in, bounded unload/retreat/reposture/recontact execution transaction.

Feedback: measured q, fingertip tactile force/point/normal only. No teacher q,
object mesh, connector/palm collision force or simulation object state enters
this supervisor. Joint-limit reserve is a proxy, not collision certification.
The phase clock is control time; cursor is spatial-reference progress.
"""
from dataclasses import dataclass, asdict
import time
import torch
from batched_fullhand_mcc import _quat_matrix

TRACK, UNLOAD, RETREAT, REPOSTURE, APPROACH, RELOAD, HANDBACK, FAILED, DONE = range(9)
DP_WARMUP=9
REPOSITION=10
PHASE_NAMES = ['track','unload','retreat','reposture','approach','reload','handback','failed','done','dp_history_warmup','return_to_healthy_reference']

@dataclass(frozen=True)
class PalmResetConfig:
    enabled: bool = True
    extra_frames: int = 1500
    reserve_trigger: float = .16
    reserve_target: float = .30
    reserve_confirm: int = 15
    loss_confirm: int = 30
    force_ratio_trigger: float = 4.
    force_confirm: int = 10
    max_resets: int = 3
    cooldown_frames: int = 100
    unload_frames: int = 20
    retreat_m: float = .020
    palm_speed_m_s: float = .015
    approach_speed_m_s: float = .008
    reposture_frames: int = 60
    reload_frames: int = 50
    handback_frames: int = 50
    recontact_timeout: int = 200
    low_force_n: float = .15
    healthy_confirm: int = 10
    ik_iterations: int = 30
    recontact_from_last_valid: bool = False
    continuous_reserve_guard: bool = False
    tactile_connector_clearance: bool = False
    connector_clearance_m: float = .003
    continuous_connector_guard: bool = False
    dp_history_mode: str = 'continuous'
    posture_strategy: str = 'local_ik'
    trigger_contact_loss: bool = True
    trigger_tip_force: bool = True
    thumb_approach_distance_m: float = 0.
    thumb_approach_confirm: int = 10
    recontact_reference: str = 'trigger'
    cache_confirm_frames: int = 1
    cache_thumb_distance_m: float = 0.
    cache_thumb_excursion_rad: float = 0.
    paired_posture_replay: bool = False
    recovery_thumb_slack_rad: float = 0.

    def __post_init__(self):
        if not 0 < self.reserve_trigger < self.reserve_target < 1:
            raise ValueError('Invalid reserve hysteresis')
        for k in ['extra_frames','reserve_confirm','loss_confirm','force_confirm','max_resets',
                  'cooldown_frames','unload_frames','reposture_frames','reload_frames',
                  'handback_frames','recontact_timeout','healthy_confirm','ik_iterations']:
            if not isinstance(getattr(self,k),int) or getattr(self,k)<1: raise ValueError(k)
        if not 0 < self.retreat_m <= .04 or min(self.palm_speed_m_s,self.approach_speed_m_s)<=0:
            raise ValueError('Invalid palm envelope')
        if not 0 < self.connector_clearance_m <= .01:
            raise ValueError('Invalid local connector clearance')
        if self.dp_history_mode not in ('continuous','freeze_rebuild'):
            raise ValueError('Unknown reset DP history mode')
        if self.posture_strategy not in ('local_ik','cached_measured'):
            raise ValueError('Unknown posture strategy')
        if not 0 <= self.thumb_approach_distance_m <= .1 or self.thumb_approach_confirm < 1:
            raise ValueError('Invalid thumb approach monitor')
        if self.posture_strategy=='cached_measured' and (self.tactile_connector_clearance or self.continuous_connector_guard):
            raise ValueError('Cached measured recovery excludes online posture optimization')
        if self.recontact_reference not in ('trigger','healthy_cache'):
            raise ValueError('Unknown recontact reference')
        if self.recontact_reference=='healthy_cache' and self.posture_strategy!='cached_measured':
            raise ValueError('Healthy reference must be paired with its measured posture')
        if self.cache_confirm_frames < 1 or self.cache_thumb_distance_m < 0 or self.cache_thumb_excursion_rad < 0 or not 0 <= self.recovery_thumb_slack_rad <= .1:
            raise ValueError('Invalid measured healthy cache envelope')
        if self.paired_posture_replay and self.recontact_reference!='healthy_cache':
            raise ValueError('Paired replay requires the measured healthy reference')


def joint_reserve(q, lower, upper):
    """Two signed directional margins and normalized worst-direction reserve."""
    minus=q-lower;plus=upper-q
    return minus,plus,2*torch.minimum(minus,plus)/(upper-lower)


class PalmPostureReset:
    def __init__(self, chain, lengths, config):
        self.chain=chain;self.cfg=PalmResetConfig(**config);self.device=chain.device;self.batch=chain.batch
        dev=self.device;b=self.batch
        self.lengths=torch.as_tensor(lengths,device=dev,dtype=torch.long)
        self.cursor=torch.zeros(b,device=dev,dtype=torch.long)
        self.phase=torch.zeros_like(self.cursor);self.age=torch.zeros_like(self.cursor)
        self.count=torch.zeros_like(self.cursor);self.cooldown=torch.zeros_like(self.cursor)
        self.reserve_count=torch.zeros_like(self.cursor);self.loss_count=torch.zeros_like(self.cursor)
        self.force_count=torch.zeros_like(self.cursor);self.good_count=torch.zeros_like(self.cursor)
        self.approach_count=torch.zeros_like(self.cursor)
        self.unloaded_count=torch.zeros_like(self.cursor)
        self.thumb_distance=torch.full((b,),float('inf'),device=dev)
        self.trigger_reason=torch.zeros_like(self.cursor)
        self.external_reset_pending=torch.zeros(b,device=dev,dtype=torch.bool)
        self.distance=torch.zeros(b,device=dev);self.axis=torch.zeros(b,3,device=dev)
        self.previous_pose=None
        self.lower=chain.mcc._clamp_q(chain.mcc.lower[None]).squeeze(0)
        self.upper=chain.mcc.upper
        self.healthy_q=torch.zeros(b,16,device=dev);self.healthy_valid=torch.zeros(b,device=dev,dtype=torch.bool)
        self.healthy_cursor=torch.zeros_like(self.cursor)
        self.cache_count=torch.zeros_like(self.cursor)
        self.initial_healthy_q=torch.zeros_like(self.healthy_q)
        self.initial_healthy_valid=torch.zeros_like(self.healthy_valid)
        self.posture_cursor=torch.zeros_like(self.cursor)
        self.entry_cursor=torch.zeros_like(self.cursor)
        self.measured_q_path=(torch.zeros(int(self.lengths.max()),b,16,device=dev)
                              if self.cfg.recontact_reference=='healthy_cache' else None)
        self.replay_posture_target=torch.zeros(b,16,device=dev)
        self.reset_q=torch.zeros_like(self.healthy_q);self.open_q=self.reset_q.clone();self.entry_q=self.reset_q.clone()
        self.last_points=torch.zeros(b,4,3,device=dev);self.point_valid=torch.zeros(b,4,device=dev,dtype=torch.bool)
        self.last_outward=torch.zeros_like(self.last_points)
        self.last_tactile_age=torch.full((b,4),10000,device=dev,dtype=torch.long)
        self.connector_geometry=None
        self.reset_plan_plane_clearance=torch.zeros(b,4,device=dev)
        self.reset_plan_tip_error=torch.zeros(b,4,device=dev)
        self.reset_plan_plane_valid=torch.zeros(b,4,device=dev,dtype=torch.bool)
        self.reset_planning_wall_s=torch.zeros(b,device=dev)
        self.reference_geometry_correction=torch.zeros(b,16,device=dev)
        self.current_pose=None
        self.last_contact_site_world=torch.zeros_like(self.last_points)
        self.last_contact_site_age=torch.full((b,4),10000,device=dev,dtype=torch.long)
        self.last_tip_force=torch.zeros(b,4,device=dev);self.last_strict=torch.zeros(b,4,device=dev,dtype=torch.bool)
        self.events=[];self.wall_frame=0;self.current_reference=None
        self._initialised=False
        chain.mcc.free_posture_reset=torch.zeros(b,4,device=dev,dtype=torch.bool)
        margin=.5*self.cfg.reserve_target*(self.upper-self.lower)
        chain.mcc.posture_reserve_lower=self.lower+margin
        chain.mcc.posture_reserve_upper=self.upper-margin
        chain.mcc.posture_reserve_enabled=torch.zeros(b,device=dev,dtype=torch.bool)
        self.reset_start_frame=torch.zeros_like(self.cursor)
        if self.cfg.dp_history_mode=='freeze_rebuild':
            from reset_policy_history import ResetPolicyHistory
            chain.policy_memory=ResetPolicyHistory(self,chain.observation)

    def select_reference(self, frame, poses, twists, commands):
        self.wall_frame=frame
        lane=torch.arange(self.batch,device=self.device)
        if self.cfg.paired_posture_replay:
            self.cursor=torch.where(self.phase==REPOSTURE,self.posture_cursor,self.cursor)
        rewind=(self.phase==REPOSITION)&(self.last_tip_force<self.cfg.low_force_n).all(-1)
        self.cursor=torch.where(rewind,torch.maximum(self.cursor-1,self.healthy_cursor),self.cursor)
        self.current_reference=poses[self.cursor,lane].clone()
        active=(self.phase!=TRACK)&(self.phase!=DONE)
        out=(self.phase==RETREAT)|(self.phase==FAILED)
        back=(self.phase==APPROACH)
        self.distance=torch.where(out,(self.distance+self.cfg.palm_speed_m_s*.01).clamp_max(self.cfg.retreat_m),self.distance)
        self.distance=torch.where(back,(self.distance-self.cfg.approach_speed_m_s*.01).clamp_min(0),self.distance)
        pose=self.current_reference.clone();pose[:,:3]+=self.axis*self.distance[:,None]
        twist=twists[self.cursor,lane].clone()
        # An override is a real executed displacement, so publish its actual
        # finite-difference velocity instead of the advancing path's velocity.
        if self.previous_pose is not None:
            velocity=(pose[:,:3]-self.previous_pose[:,:3])/.01
            delta=_quat_matrix(pose)@_quat_matrix(self.previous_pose).transpose(-1,-2)
            vee=torch.stack((delta[:,2,1]-delta[:,1,2],delta[:,0,2]-delta[:,2,0],delta[:,1,0]-delta[:,0,1]),-1)*.5
            angle=torch.acos(((delta.diagonal(dim1=-2,dim2=-1).sum(-1)-1)*.5).clamp(-1,1))
            factor=torch.where(angle<1e-4,torch.ones_like(angle),angle/torch.sin(angle).clamp_min(1e-6))
            omega=vee*factor[:,None]/.01
            twist=torch.where(active[:,None],torch.cat((velocity,omega),1),twist)
        twist=torch.where((self.phase==DONE)[:,None],0.,twist)
        self.previous_pose=pose.clone()
        command=commands[self.cursor,lane].clone()
        command=torch.where((active|(self.phase==DONE))[:,None],0.,command)
        return pose,twist,command

    def _transition(self, mask, phase, reason=None):
        if not bool(mask.any()):return
        for lane in torch.where(mask)[0].tolist():
            self.events.append(dict(frame=self.wall_frame,lane=lane,reference_frame=int(self.cursor[lane]),
                previous=PHASE_NAMES[int(self.phase[lane])],phase=PHASE_NAMES[phase],
                reason=reason,reset=int(self.count[lane]),retreat_m=float(self.distance[lane])))
            print(f'[PALM-RESET] tick={self.wall_frame} lane={lane} ref={int(self.cursor[lane])} '
                  f'{PHASE_NAMES[int(self.phase[lane])]}->{PHASE_NAMES[phase]} reason={reason}',flush=True)
        self.phase=torch.where(mask,phase,self.phase);self.age=torch.where(mask,0,self.age)
        self.good_count=torch.where(mask,0,self.good_count)

    def request_reset(self,mask):
        """Explicit higher-level reset request, distinct from health detection."""
        if mask.shape!=(self.batch,) or mask.dtype!=torch.bool or mask.device!=self.phase.device:
            raise ValueError('Reset request must be Boolean[B] on controller device')
        self.external_reset_pending|=mask

    @torch.no_grad()
    def _fit_grasp(self,q, preferred, target_sites):
        # A local redundant IK: retain the current fingertip layout while
        # providing a symmetric joint margin. Healthy measured posture is a
        # soft preference, never a commanded return to a fixed initial q0.
        margin=.5*self.cfg.reserve_target*(self.upper-self.lower)
        lo=self.lower+margin;hi=self.upper-margin
        x=torch.maximum(torch.minimum(q,hi),lo)
        pref=torch.maximum(torch.minimum(preferred,hi),lo)
        eye=torch.eye(4,device=q.device)
        for _ in range(self.cfg.ik_iterations):
            sites,_,_,_,jac=self.chain.mcc.fk.contact_kinematics(x)
            error=target_sites-sites
            jt=jac.transpose(-1,-2)
            # metres/radian Jacobian; 1e-5 m^2 regularization.
            a=jt@jac+1e-5*eye
            rhs=(jt@error[...,None]).squeeze(-1)+1e-5*(pref-x).reshape(-1,4,4)
            step=torch.linalg.solve(a,rhs[...,None]).squeeze(-1).reshape(-1,16).clamp(-.04,.04)
            x=torch.maximum(torch.minimum(x+step,hi),lo)
        return x

    def observe(self, q, pose, tactile):
        self.current_pose=pose
        if not self._initialised:
            self.healthy_q.copy_(q);self.reset_q.copy_(q);self.open_q.copy_(q);self.entry_q.copy_(q)
            self._initialised=True
        c=self.cfg
        force=tactile['pad_force'].norm(dim=-1)
        strict=tactile['pad_found']&(force>=.05)&torch.isfinite(tactile['pad_points']).all(-1)
        self.last_tip_force=force;self.last_strict=strict
        self.unloaded_count=torch.where((force<c.low_force_n).all(-1),self.unloaded_count+1,0)
        physical=tactile['physical_normal_found']&torch.isfinite(tactile['physical_contact_point']).all(-1)
        outward=-tactile['physical_normal_inward'];norm=outward.norm(dim=-1,keepdim=True)
        physical &= torch.isfinite(outward).all(-1)&(norm[...,0]>1e-6)
        self.last_points=torch.where(physical[...,None],tactile['physical_contact_point'],self.last_points)
        self.last_outward=torch.where(physical[...,None],outward/norm.clamp_min(1e-8),self.last_outward)
        self.point_valid|=physical
        self.last_tactile_age=torch.where(physical,0,(self.last_tactile_age+1).clamp_max(10000))
        sites=None
        if c.recontact_from_last_valid:
            sites=self.chain.mcc.fk(q)
            world_site=pose[:,None,:3]+torch.einsum('nij,nfj->nfi',_quat_matrix(pose),sites)
            valid=physical&strict
            self.last_contact_site_world=torch.where(valid[...,None],world_site,self.last_contact_site_world)
            self.last_contact_site_age=torch.where(valid,0,(self.last_contact_site_age+1).clamp_max(10000))
        reserve=joint_reserve(q,self.lower,self.upper)[2].amin(-1)
        track=self.phase==TRACK
        if self.measured_q_path is not None:
            ids=torch.where(track)[0]
            self.measured_q_path[self.cursor[ids],ids]=q[ids]
        ready=(self.wall_frame>=100)&(self.cooldown==0)&track&(self.cursor<self.lengths-1)
        approaching=torch.zeros_like(ready)
        if c.thumb_approach_distance_m>0 or c.cache_thumb_distance_m>0:
            if sites is None:sites=self.chain.mcc.fk(q)
            self.thumb_distance=(sites[:,3:4]-sites[:,:3]).norm(dim=-1).amin(-1)
            # A task-specific proximity warning, not a collision certificate.
            # A lost thumb converging on another finger is different from
            # brief loss alone; only proprioceptive FK and tactile validity enter.
            approaching=(self.thumb_distance<c.thumb_approach_distance_m)&~strict[:,3]
        self.approach_count=torch.where(ready&approaching,self.approach_count+1,0)
        healthy=(reserve>=c.reserve_target)&strict.all(-1)&(force<2*self.chain.mcc.force_setpoint).all(-1)
        # A single transient pad hit is not a recoverable grasp. Establish a
        # conservative envelope from the measured initial grasp, then admit
        # only sustained contact snapshots inside that envelope. This affects
        # recovery memory, not the nominal DP's joint targets.
        first=healthy&track&~self.initial_healthy_valid
        self.initial_healthy_q=torch.where(first[:,None],q,self.initial_healthy_q)
        self.initial_healthy_valid|=first
        if c.cache_thumb_distance_m>0:
            healthy &= self.thumb_distance>=c.cache_thumb_distance_m
        if c.cache_thumb_excursion_rad>0:
            healthy &= (q[:,12:]-self.initial_healthy_q[:,12:]).abs().amax(-1)<=c.cache_thumb_excursion_rad
        self.cache_count=torch.where(healthy&track,self.cache_count+1,0)
        cache=healthy&track&(self.cache_count>=c.cache_confirm_frames)
        self.healthy_q=torch.where(cache[:,None],q,self.healthy_q)
        self.healthy_cursor=torch.where(cache,self.cursor,self.healthy_cursor)
        self.healthy_valid|=cache
        self.reserve_count=torch.where(ready&(reserve<c.reserve_trigger),self.reserve_count+1,0)
        self.loss_count=torch.where(ready&~strict.all(-1),self.loss_count+1,0)
        self.force_count=torch.where(ready&(force>c.force_ratio_trigger*self.chain.mcc.force_setpoint).any(-1),self.force_count+1,0)
        reason=(self.reserve_count>=c.reserve_confirm).long()+2*(c.trigger_contact_loss&(self.loss_count>=c.loss_confirm)).long()+4*(c.trigger_tip_force&(self.force_count>=c.force_confirm)).long()+8*self.external_reset_pending.long()+16*(self.approach_count>=c.thumb_approach_confirm).long()
        trigger=(reason>0)&ready&c.enabled
        available=(self.point_valid&(self.last_tactile_age<=100)).any(-1)
        if c.posture_strategy=='cached_measured':available &= self.healthy_valid
        exhausted=trigger&((self.count>=c.max_resets)|~available)
        self._transition(exhausted,FAILED,'reset_budget_or_no_tactile_geometry')
        trigger &= ~exhausted
        self.external_reset_pending&=~(trigger|exhausted)
        if bool(trigger.any()):
            if self.device.type=='cuda':torch.cuda.synchronize(self.device)
            planning_start=time.perf_counter()
            self.count+=trigger.long();self.trigger_reason=torch.where(trigger,reason,self.trigger_reason)
            self.reset_start_frame=torch.where(trigger,self.wall_frame,self.reset_start_frame)
            self.posture_cursor=torch.where(trigger,self.cursor,self.posture_cursor)
            self.entry_cursor=torch.where(trigger,self.cursor,self.entry_cursor)
            self.entry_q=torch.where(trigger[:,None],q,self.entry_q)
            pref=torch.where(self.healthy_valid[:,None],self.healthy_q,q)
            if sites is None:sites=self.chain.mcc.fk(q)
            target_sites=sites
            if c.recontact_from_last_valid:
                cached=torch.einsum('nij,nfj->nfi',_quat_matrix(pose).transpose(-1,-2),
                    self.last_contact_site_world-pose[:,None,:3])
                use=(~strict)&(self.last_contact_site_age<=100)
                target_sites=torch.where(use[...,None],cached,sites)
            fit=(pref.clone() if c.posture_strategy=='cached_measured'
                 else self._fit_grasp(q,pref,target_sites))
            if c.tactile_connector_clearance:
                if self.connector_geometry is None:
                    from tactile_connector_clearance import TactileConnectorClearance
                    self.connector_geometry=TactileConnectorClearance(self.chain.env.sim.mj_model,self.device)
                rot=_quat_matrix(pose)
                plane_point=torch.einsum('bij,bfj->bfi',rot.transpose(-1,-2),self.last_points-pose[:,None,:3])
                plane_normal=torch.einsum('bij,bfj->bfi',rot.transpose(-1,-2),self.last_outward)
                valid=self.point_valid&(self.last_tactile_age<=100)
                margin=.5*c.reserve_target*(self.upper-self.lower)
                fit=self.connector_geometry.fit(self.chain.mcc.fk,fit,pref,target_sites,
                    self.lower+margin,self.upper-margin,plane_point,plane_normal,valid,
                    iterations=60,clearance=c.connector_clearance_m)
                clearance,_,available_plane=self.connector_geometry.minimum(self.chain.mcc.fk,fit,plane_point,plane_normal,valid)
                error=(self.chain.mcc.fk(fit)-target_sites).norm(dim=-1)
                self.reset_plan_plane_clearance=torch.where(trigger[:,None],clearance,self.reset_plan_plane_clearance)
                self.reset_plan_tip_error=torch.where(trigger[:,None],error,self.reset_plan_tip_error)
                self.reset_plan_plane_valid=torch.where(trigger[:,None],available_plane,self.reset_plan_plane_valid)
            # Move pads a few millimetres outward during retreat, independently
            # per finger. The palm's centroid direction alone cannot unload an
            # opposing thumb and three fingers simultaneously.
            rot=_quat_matrix(pose)
            local=torch.einsum('nij,nfj->nfi',rot.transpose(-1,-2),self.last_outward)
            # Cached mode never reshapes a loaded hand. MCC handles local
            # unloading; measured healthy posture becomes executable only
            # after retreat and a sustained low-force interval.
            opened=(q.clone() if c.posture_strategy=='cached_measured'
                    else self._fit_grasp(q,pref,sites+.004*local*self.point_valid[...,None]))
            if self.device.type=='cuda':torch.cuda.synchronize(self.device)
            self.reset_planning_wall_s=torch.where(trigger,time.perf_counter()-planning_start,self.reset_planning_wall_s)
            self.reset_q=torch.where(trigger[:,None],fit,self.reset_q)
            self.open_q=torch.where(trigger[:,None],opened,self.open_q)
            weights=self.point_valid.float();centroid=(self.last_points*weights[...,None]).sum(1)/weights.sum(1,keepdim=True).clamp_min(1)
            axis=pose[:,:3]-centroid;axis=axis/axis.norm(dim=-1,keepdim=True).clamp_min(1e-8)
            self.axis=torch.where(trigger[:,None],axis,self.axis)
            self._transition(trigger,UNLOAD,'reserve=1/loss=2/force=4/external=8/thumb_proximity=16')
        unloaded=(force<c.low_force_n).all(-1)
        if c.posture_strategy=='cached_measured':unloaded &= self.unloaded_count>=c.healthy_confirm
        self.chain.mcc.free_posture_reset=((self.phase==REPOSTURE)&unloaded)[:,None].expand(-1,4)
        self.replay_posture_target=self.reset_q.clone()
        if self.measured_q_path is not None:
            lane=torch.arange(self.batch,device=self.device)
            # Do not advance the measured path while the hand is blocked or
            # still catching up. Otherwise the replay silently becomes a jump.
            reached=(q-self.measured_q_path[self.posture_cursor,lane]).abs().amax(-1)<.03
            step=(self.phase==REPOSTURE)&unloaded&reached
            self.posture_cursor=torch.where(step,torch.maximum(self.posture_cursor-1,self.healthy_cursor),self.posture_cursor)
            self.replay_posture_target=torch.where((self.phase==REPOSTURE)[:,None],
                self.measured_q_path[self.posture_cursor,lane],self.reset_q)
        self.chain.mcc.reset_posture_target=self.replay_posture_target
        self.chain.mcc.posture_reserve_enabled=((self.phase==TRACK)|(self.phase==HANDBACK)) & c.enabled & c.continuous_reserve_guard

    def apply_nominal(self,q,nominal,load):
        p=self.phase;c=self.cfg
        override=(p!=TRACK)&(p!=DONE)
        load=torch.ones(self.batch,4,device=self.device) if load is None else load.clone()
        request=torch.zeros(self.batch,device=self.device)
        request=torch.where(p==APPROACH,((c.retreat_m-self.distance)/c.retreat_m).clamp(0,1)*.5,request)
        request=torch.where(p==RELOAD,(self.age.float()/c.reload_frames).clamp(0,1),request)
        request=torch.where(p==HANDBACK,1.,request)
        request=torch.where(p==DP_WARMUP,1.,request)
        load=torch.where(override[:,None],request[:,None],load)
        target=self.reset_q
        if self.measured_q_path is not None:
            target=torch.where((p==REPOSTURE)[:,None],self.replay_posture_target,target)
        target=torch.where(((p==UNLOAD)|(p==RETREAT))[:,None],self.open_q,target)
        if c.posture_strategy=='cached_measured':
            wait=(p==REPOSTURE)&~self.chain.mcc.free_posture_reset.all(-1)
            # An incidental contact during reshaping must not command the
            # original degenerate entry pose again. Hold measured q and let
            # the unchanged MCC unload before resuming the paused path.
            target=torch.where(wait[:,None],q,target)
        target=torch.where((p==FAILED)[:,None],q,target)
        blend=(self.age.float()/c.handback_frames).clamp(0,1)[:,None]
        target=torch.where((p==HANDBACK)[:,None],target+(nominal-target)*blend,target)
        nominal=torch.where(override[:,None],target,nominal)
        self.reference_geometry_correction.zero_()
        if c.continuous_connector_guard and self.connector_geometry is not None and self.current_pose is not None:
            # Correct the nominal physical-motion reference BEFORE MCC adds
            # its elastic/force preload. Projecting the absolute actuator
            # target as though it were the measured posture would remove
            # force-bearing tracking error and repeat the old interface bug.
            rotation=_quat_matrix(self.current_pose)
            point=torch.einsum('bij,bfj->bfi',rotation.transpose(-1,-2),self.last_points-self.current_pose[:,None,:3])
            normal=torch.einsum('bij,bfj->bfi',rotation.transpose(-1,-2),self.last_outward)
            valid=self.point_valid&(self.last_tactile_age<=20)
            # Geometry is shared, authority is lane-local: another hand's
            # reset must never arm this lane's post-reset feedback guard.
            allowed=(((p==TRACK)|(p==HANDBACK))&c.enabled&(self.count>0))[:,None]
            bounded=torch.maximum(torch.minimum(nominal,self.upper),self.lower)
            distance,_,has=self.connector_geometry.minimum(self.chain.mcc.fk,bounded,point,normal,valid)
            violated=allowed&has&(distance<c.connector_clearance_m)
            if bool(violated.any()):
                margin=.5*c.reserve_target*(self.upper-self.lower)
                fitted=self.connector_geometry.fit(self.chain.mcc.fk,bounded,bounded,self.chain.mcc.fk(bounded),
                    self.lower+margin,self.upper-margin,point,normal,violated,
                    iterations=3,clearance=c.connector_clearance_m)
                corrected=torch.where(violated.repeat_interleave(4,dim=-1),fitted,nominal)
                self.reference_geometry_correction=corrected-nominal
                nominal=corrected
        return nominal,load

    def after_frame(self,env,frame):
        c=self.cfg;self.age+=1
        # Increment only frames actually run under track authority. Spatial
        # cursor cannot jump across a reset; all extra ticks are overhead.
        track=self.phase==TRACK
        finished=track&(self.cursor>=self.lengths-1)
        self._transition(finished,DONE,'reference_completed')
        self.cursor=torch.minimum(self.cursor+((self.phase==TRACK).long()),self.lengths-1)
        self.cooldown=(self.cooldown-1).clamp_min(0)
        self._transition((self.phase==UNLOAD)&(self.age>=c.unload_frames),RETREAT,'load_ramp_complete')
        self._transition((self.phase==RETREAT)&(self.distance>=c.retreat_m-1e-7),REPOSTURE,'retreat_reached')
        # Reposture is checked from measured q, not requested command.
        q=env.scene['robot'].data.joint_pos[:,-16:]
        unloaded=(self.last_tip_force<c.low_force_n).all(-1)
        achieved=(q-self.reset_q).abs().amax(-1)<.08
        if self.measured_q_path is not None:achieved &= self.posture_cursor<=self.healthy_cursor
        next_phase=REPOSITION if c.recontact_reference=='healthy_cache' else APPROACH
        self._transition((self.phase==REPOSTURE)&(self.age>=c.reposture_frames)&achieved&unloaded,next_phase,'unloaded_posture_reached')
        posture_timeout=c.reposture_frames+150
        if self.measured_q_path is not None:
            posture_timeout=posture_timeout+(self.entry_cursor-self.healthy_cursor).clamp_min(0)
        self._transition((self.phase==REPOSTURE)&(self.age>=posture_timeout),FAILED,'posture_reset_timeout')
        self._transition((self.phase==REPOSITION)&(self.cursor<=self.healthy_cursor)&unloaded,APPROACH,'matched_healthy_reference_reached')
        self._transition((self.phase==REPOSITION)&(self.age>=c.recontact_timeout+500),FAILED,'healthy_reference_return_timeout')
        self._transition((self.phase==APPROACH)&(self.distance<=1e-7),RELOAD,'reference_palm_reached')
        good=self.last_strict.all(-1)&(self.last_tip_force<2*self.chain.mcc.force_setpoint).all(-1)
        if c.posture_strategy=='cached_measured':
            good &= joint_reserve(q,self.lower,self.upper)[2].amin(-1)>=c.reserve_trigger
            if c.cache_thumb_distance_m>0:
                good &= self.thumb_distance>=c.cache_thumb_distance_m
            if c.cache_thumb_excursion_rad>0:
                good &= (q[:,12:]-self.initial_healthy_q[:,12:]).abs().amax(-1)<=c.cache_thumb_excursion_rad+c.recovery_thumb_slack_rad
        self.good_count=torch.where(((self.phase==RELOAD)|(self.phase==DP_WARMUP)|(self.phase==HANDBACK))&good,self.good_count+1,0)
        release=(self.phase==RELOAD)&(self.age>=c.reload_frames)&(self.good_count>=c.healthy_confirm)
        if self.cfg.dp_history_mode=='freeze_rebuild':
            long=(self.wall_frame-self.reset_start_frame>=self.chain.policy_memory.capacity)|((q-self.entry_q).abs().amax(-1)>.05)
            self.chain.policy_memory.clear_lanes(release&long)
            self._transition(release,DP_WARMUP,'stable_contact_rebuild_real_history')
        else:
            self._transition(release,HANDBACK,'stable_recontact')
        self._transition((self.phase==RELOAD)&(self.age>=c.recontact_timeout),FAILED,'recontact_timeout')
        if self.cfg.dp_history_mode=='freeze_rebuild':
            # Align admission to the unchanged batch replan clock. A lane's
            # resume must not force an extra replan/noise draw on its peers.
            slot=(self.chain.control_frame-self.chain.bootstrap_end)%self.chain.scheduler.config.replan_interval==0
            ready=(self.phase==DP_WARMUP)&(self.chain.policy_memory.counts>=self.chain.policy_memory.capacity)&(self.good_count>=c.healthy_confirm)&slot
            self.chain.policy_memory.pending_restart|=ready
            self._transition(ready,HANDBACK,'fresh_history_ready_at_replan_slot')
            self._transition((self.phase==DP_WARMUP)&(self.age>=c.recontact_timeout),FAILED,'stable_history_warmup_timeout')
        resumed=(self.phase==HANDBACK)&(self.age>=c.handback_frames)
        if c.posture_strategy=='cached_measured':
            resumed &= self.good_count>=c.healthy_confirm
            self._transition((self.phase==HANDBACK)&(self.age>=c.recontact_timeout)&~resumed,FAILED,'handover_health_not_stable')
        self._transition(resumed,TRACK,'dp_authority_restored')
        self.cooldown=torch.where(resumed,c.cooldown_frames,self.cooldown)
        self.chain.execution_finished=bool(((self.phase==DONE)|(self.phase==FAILED)).all())
        return None

    def diagnostics(self):
        result=dict(reset_phase=self.phase.clone(),reset_age=self.age.clone(),
            reset_count=self.count.clone(),reset_reason=self.trigger_reason.clone(),
            reset_reference_frame=self.cursor.clone(),reset_retreat_m=self.distance.clone(),
            reset_healthy_reference_frame=self.healthy_cursor.clone(),
            reset_posture_replay_frame=self.posture_cursor.clone(),
            reset_axis_world=self.axis.clone(),reset_posture_target=self.reset_q.clone(),
            reset_plan_plane_clearance_m=self.reset_plan_plane_clearance.clone(),
            reset_plan_tip_error_m=self.reset_plan_tip_error.clone(),
            reset_plan_plane_valid=self.reset_plan_plane_valid.clone(),
            reset_planning_wall_s=self.reset_planning_wall_s.clone(),
            reset_reference_geometry_correction_q=self.reference_geometry_correction.clone(),
            reset_healthy_memory_valid=self.healthy_valid.clone(),reset_posture_permission=self.chain.mcc.free_posture_reset.clone())
        result.update(reset_thumb_distance_m=self.thumb_distance.clone(),reset_approach_count=self.approach_count.clone(),
                      reset_unloaded_count=self.unloaded_count.clone(),reset_cache_healthy_count=self.cache_count.clone(),
                      reset_good_count=self.good_count.clone())
        if self.cfg.dp_history_mode=='freeze_rebuild':
            memory=self.chain.policy_memory
            result.update(reset_policy_history_count=memory.counts.clone(),reset_policy_history_frozen_ticks=memory.frozen_ticks.clone(),
                reset_policy_history_clear_count=memory.clear_count.clone(),reset_policy_active=memory.active_mask())
        return result

    def report(self):
        return dict(config=asdict(self.cfg),phase_names=PHASE_NAMES,events=self.events,
            final_reference_frame=self.cursor.cpu().tolist(),lengths=self.lengths.cpu().tolist(),
            final_phase=[PHASE_NAMES[v] for v in self.phase.cpu().tolist()],resets=self.count.cpu().tolist(),
            feedback='q + joint limits + true-tip tactile force/point/normal only; no teacher or non-tip force feedback',
            reset_posture=('cached healthy measured posture; reverse measured q path when paired healthy reference is enabled; no online IK' if self.cfg.posture_strategy=='cached_measured'
                           else 'local bounded redundant IK retaining measured tip layout; preference for measured healthy grasp'),
            caveat='Joint-range proxy and tactile unloading do not certify non-tip/self collision clearance.')
