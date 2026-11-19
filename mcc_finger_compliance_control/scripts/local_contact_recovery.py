"""Opt-in bounded recovery; frozen DP/MCC, no runtime object mesh queries.

Planner must supply an outward direction for initialisation without tactile
history. This is an explicit command, not an inferred tactile observation.
"""
import torch
from reset_policy_history import ResetPolicyHistory


class LocalRecoveryHistory(ResetPolicyHistory):
    def active_mask(self):
        return super().active_mask() & (self.supervisor.initial_ready | (self.supervisor.phase==HANDBACK))

from palm_posture_reset import (PalmPostureReset, joint_reserve, TRACK, UNLOAD,
    RETREAT, REPOSTURE, APPROACH, RELOAD, HANDBACK, FAILED, DONE, DP_WARMUP)


class LocalContactRecovery(PalmPostureReset):
    def __init__(self, chain, lengths, config, local, pregrasp):
        config = dict(config)
        config.update(posture_strategy='cached_measured', dp_history_mode='freeze_rebuild',
                      trigger_contact_loss=False, trigger_tip_force=False,
                      continuous_reserve_guard=False, continuous_connector_guard=False,
                      tactile_connector_clearance=False, recontact_reference='trigger',
                      paired_posture_replay=False)
        super().__init__(chain, lengths, config)
        allowed = {'initial_outward_world', 'extra_approach_m', 'min_contacts',
                   'approach_timeout', 'max_total_offset_m'}
        if set(local)-allowed:
            raise ValueError(f'Unknown local recovery settings: {set(local)-allowed}')
        self.local = dict(local)
        self.extra_approach = float(local.get('extra_approach_m', .02))
        self.min_contacts = int(local.get('min_contacts', 3))
        self.approach_timeout = int(local.get('approach_timeout', 1200))
        self.max_total_offset = float(local.get('max_total_offset_m', .04))
        if not 0 <= self.extra_approach <= .03 or not 1 <= self.min_contacts <= 4 or self.approach_timeout < 1 or not 0 < self.max_total_offset <= .06:
            raise ValueError('Invalid bounded recovery envelope')
        axis = torch.as_tensor(local['initial_outward_world'], device=self.device, dtype=torch.float32)
        if axis.shape == (3,): axis = axis.expand(self.batch, -1).clone()
        if axis.shape != (self.batch,3) or not torch.isfinite(axis).all() or (axis.norm(dim=-1)<1e-6).any():
            raise ValueError('Planner must provide finite nonzero outward direction per lane')
        self.planner_axis = axis / axis.norm(dim=-1, keepdim=True)
        self.pregrasp = torch.as_tensor(pregrasp, device=self.device, dtype=torch.float32)
        if self.pregrasp.shape != (self.batch,16) or not torch.isfinite(self.pregrasp).all():
            raise ValueError('Finite pregrasp[B,16] required')
        margin = .5*self.cfg.reserve_target*(self.upper-self.lower)
        self.pregrasp = torch.maximum(torch.minimum(self.pregrasp, self.upper-margin), self.lower+margin)
        self.offset = torch.zeros(self.batch,3,device=self.device)
        self.initial_ready = torch.zeros(self.batch,dtype=torch.bool,device=self.device)
        self.observed = False
        self.current_q = self.pregrasp.clone()
        chain.policy_memory = LocalRecoveryHistory(self, chain.observation)

    def select_reference(self, frame, poses, twists, commands):
        # Use the parent's bounded retreat and velocity calculation, but extend
        # approach below its original reference plane. Recompute actual twist.
        before = self.previous_pose.clone() if self.previous_pose is not None else None
        old_distance = self.distance.clone()
        pose, twist, command = super().select_reference(frame, poses, twists, commands)
        approaching = self.phase == APPROACH
        next_distance = (old_distance-self.cfg.approach_speed_m_s*.01).clamp_min(-self.extra_approach)
        # Pause approach when contact is already sufficient for confirmation.
        next_distance = torch.where(self._good(), old_distance, next_distance)
        self.distance = torch.where(approaching, next_distance, self.distance)
        pose[:,:3] = self.current_reference[:,:3]+self.offset+self.axis*self.distance[:,None]
        if before is not None:
            twist[:,:3] = (pose[:,:3]-before[:,:3])/.01
        self.previous_pose = pose.clone()
        return pose, twist, command

    def _good(self):
        return ((self.last_strict.sum(-1)>=self.min_contacts)
                & (self.last_tip_force<2*self.chain.mcc.force_setpoint).all(-1)
                & (joint_reserve(self.current_q,self.lower,self.upper)[2].amin(-1)>=self.cfg.reserve_trigger))

    def observe(self, q, pose, tactile):
        self.observed = True
        self.current_q = q.clone(); self.current_pose = pose
        c = self.cfg
        force = tactile['pad_force'].norm(dim=-1)
        strict = tactile['pad_found'] & (force>=.05) & torch.isfinite(tactile['pad_points']).all(-1)
        self.last_tip_force = force; self.last_strict = strict
        self.unloaded_count = torch.where((force<c.low_force_n).all(-1),self.unloaded_count+1,0)
        outward = -tactile['physical_normal_inward']
        valid = tactile['physical_normal_found'] & torch.isfinite(outward).all(-1) & (outward.norm(dim=-1)>1e-6)
        valid &= torch.isfinite(tactile['physical_contact_point']).all(-1)
        self.last_points = torch.where(valid[...,None],tactile['physical_contact_point'],self.last_points)
        self.last_outward = torch.where(valid[...,None],outward/outward.norm(dim=-1,keepdim=True).clamp_min(1e-8),self.last_outward)
        self.point_valid |= valid
        self.last_tactile_age = torch.where(valid,0,(self.last_tactile_age+1).clamp_max(10000))
        reserve = joint_reserve(q,self.lower,self.upper)[2].amin(-1)
        track = self.phase==TRACK
        healthy = strict.all(-1) & (reserve>=c.reserve_target) & (force<2*self.chain.mcc.force_setpoint).all(-1)
        self.cache_count = torch.where(track&healthy,self.cache_count+1,0)
        cache = self.cache_count>=c.healthy_confirm
        self.healthy_q = torch.where(cache[:,None],q,self.healthy_q); self.healthy_valid |= cache
        self.initial_ready |= cache
        ready = track & (self.wall_frame>=100) & (self.cooldown==0)
        self.loss_count = torch.where(ready & ~strict.any(-1),self.loss_count+1,0)
        self.reserve_count = torch.where(ready & (reserve<c.reserve_trigger),self.reserve_count+1,0)
        trigger = ready & ((self.loss_count>=c.loss_confirm) | (self.reserve_count>=c.reserve_confirm) | self.external_reset_pending)
        # Startup partial contact is given a bounded interval to stabilise.
        trigger |= ready & ~self.initial_ready & (self.wall_frame>=200)
        exhausted = trigger & (self.count>=c.max_resets)
        self._transition(exhausted,FAILED,'recovery_budget_exhausted')
        trigger &= ~exhausted
        if bool(trigger.any()):
            recent = self.point_valid & (self.last_tactile_age<=100)
            centroid = (self.last_points*recent[...,None]).sum(1)/recent.sum(-1,keepdim=True).clamp_min(1)
            direction = pose[:,:3]-centroid
            valid_axis = recent.any(-1) & (direction.norm(dim=-1)>1e-6)
            direction = direction/direction.norm(dim=-1,keepdim=True).clamp_min(1e-8)
            # No stale-normal inference after tracking loss. Initialisation
            # alone may use its explicitly supplied planner direction.
            available = valid_axis | ~self.initial_ready
            self._transition(trigger&~available,FAILED,'request_planner_direction_no_recent_contact')
            trigger &= available
            direction = torch.where(valid_axis[:,None],direction,self.planner_axis)
            self.axis = torch.where(trigger[:,None],direction,self.axis)
            target = torch.where(self.healthy_valid[:,None],self.healthy_q,self.pregrasp)
            self.reset_q = torch.where(trigger[:,None],target,self.reset_q)
            self.open_q = torch.where(trigger[:,None],q,self.open_q)
            self.entry_q = torch.where(trigger[:,None],q,self.entry_q)
            self.count += trigger.long()
            self.reset_start_frame = torch.where(trigger,self.wall_frame,self.reset_start_frame)
            self._transition(trigger,UNLOAD,'complete_loss_or_low_joint_reserve_or_initialisation')
        self.external_reset_pending.zero_()
        unloaded = self.unloaded_count>=c.healthy_confirm
        self.chain.mcc.free_posture_reset = ((self.phase==REPOSTURE)&unloaded)[:,None].expand(-1,4)
        self.chain.mcc.reset_posture_target = self.reset_q

    def apply_nominal(self,q,nominal,load):
        nominal,load = super().apply_nominal(q,nominal,load)
        initial_hold = (self.phase==TRACK)&~self.initial_ready
        nominal = torch.where(initial_hold[:,None],self.pregrasp,nominal)
        # Early-contact fingers keep normal MCC compliance while the other
        # fingers approach. Final actuator command remains owned by MCC.
        load = torch.where((((self.phase==APPROACH)|(self.phase==RELOAD))[:,None]&self.last_strict),torch.ones_like(load),load)
        return nominal,load

    def after_frame(self,env,frame):
        if not self.observed:
            return
        c=self.cfg; self.age+=1
        track=self.phase==TRACK
        finished=track & self.initial_ready & (self.cursor>=self.lengths-1)
        self._transition(finished,DONE,'reference_completed')
        self.cursor=torch.minimum(self.cursor+(track & self.initial_ready).long(),self.lengths-1)
        self.cooldown=(self.cooldown-1).clamp_min(0)
        self._transition((self.phase==UNLOAD)&(self.age>=c.unload_frames),RETREAT,'load_removed')
        self._transition((self.phase==RETREAT)&(self.distance>=c.retreat_m-1e-7),REPOSTURE,'retreat_reached')
        q=env.scene['robot'].data.joint_pos[:,-16:]
        achieved=(q-self.reset_q).abs().amax(-1)<.08
        achieved &= joint_reserve(q,self.lower,self.upper)[2].amin(-1)>=c.reserve_trigger
        self._transition((self.phase==REPOSTURE)&(self.age>=c.reposture_frames)&achieved&(self.unloaded_count>=c.healthy_confirm),APPROACH,'measured_pregrasp_reached_unloaded')
        self._transition((self.phase==REPOSTURE)&(self.age>=c.reposture_frames+300),FAILED,'pregrasp_not_reached')
        good=self._good()
        self.good_count=torch.where(((self.phase==APPROACH)|(self.phase==RELOAD)|(self.phase==DP_WARMUP)|(self.phase==HANDBACK))&good,self.good_count+1,0)
        contact=(self.phase==APPROACH)&(self.good_count>=c.healthy_confirm)
        proposed=self.offset+self.axis*self.distance[:,None]
        self._transition(contact&(proposed.norm(dim=-1)>self.max_total_offset),FAILED,'total_local_shift_bound')
        contact &= self.phase==APPROACH
        self.offset=torch.where(contact[:,None],proposed,self.offset)
        self.distance=torch.where(contact,0.,self.distance)
        self._transition(contact,RELOAD,'stable_contact_hold_new_palm_position')
        self._transition((self.phase==APPROACH)&(self.age>=self.approach_timeout),FAILED,'bounded_approach_failed_request_upper_planner')
        release=(self.phase==RELOAD)&(self.age>=c.reload_frames)&(self.good_count>=c.healthy_confirm)
        self.chain.policy_memory.clear_lanes(release)
        self._transition(release,DP_WARMUP,'rebuild_measured_history')
        slot=(self.chain.control_frame-self.chain.bootstrap_end)%self.chain.scheduler.config.replan_interval==0
        warm=(self.phase==DP_WARMUP)&(self.chain.policy_memory.counts>=self.chain.policy_memory.capacity)&(self.good_count>=c.healthy_confirm)&slot
        self.chain.policy_memory.pending_restart |= warm
        self._transition(warm,HANDBACK,'fresh_chunk_authority')
        resumed=(self.phase==HANDBACK)&(self.age>=c.handback_frames)&(self.good_count>=c.healthy_confirm)
        self.initial_ready |= resumed
        self._transition(resumed,TRACK,'resume_remaining_reference')
        self.cooldown=torch.where(resumed,c.cooldown_frames,self.cooldown)
        timeout=((self.phase==RELOAD)|(self.phase==DP_WARMUP)|(self.phase==HANDBACK))&(self.age>=c.recontact_timeout)
        self._transition(timeout,FAILED,'contact_not_stable_for_handover')
        self.chain.execution_finished=bool(((self.phase==DONE)|(self.phase==FAILED)).all())

    def diagnostics(self):
        return dict(super().diagnostics(),local_recovery_offset=self.offset.clone(),
                    local_recovery_initial_ready=self.initial_ready.clone())

    def report(self):
        return dict(super().report(),local_recovery=self.local,
                    reset_posture='bounded initial pregrasp or confirmed measured healthy cache; no online IK',
                    feedback='q + true-tip tactile; explicit planner outward direction only for startup',
                    total_local_offset_m=self.offset.cpu().tolist())
