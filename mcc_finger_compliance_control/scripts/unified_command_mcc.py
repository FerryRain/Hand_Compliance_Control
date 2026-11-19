"""Opt-in final command arbitration with continuous unload/reload handover.

No non-tip simulation force is used. Bounds are never overridden to satisfy an
infeasible normal constraint. Palm feedforward is an independent switch.
"""
from dataclasses import dataclass
import torch
from batched_fullhand_mcc import BatchedFullHandMCC, _quat_matrix
from load_authority_mcc import LoadAuthorityMCC
from normal_constrained_hold import project_box_halfspace


@dataclass(frozen=True)
class UnifiedCommandConfig:
    release_force_ratio: float = 1.0
    release_confirm_frames: int = 3
    handback_step_rad: float = .002
    protection_is_unload: bool = True
    protection_pauses_recovery: bool = False
    force_limited_recovery: bool = False
    normal_reference: str = 'live'
    # Optional final force-to-normal-velocity gain; the inner legacy force
    # integrator retains its own gain. None preserves the frozen baseline.
    final_normal_force_gain: float | None = None
    normal_force_regulation: bool = False

    def __post_init__(self):
        if not 0 < self.release_force_ratio <= 1:
            raise ValueError('release_force_ratio must be in (0,1]')
        if not isinstance(self.release_confirm_frames,int) or self.release_confirm_frames<1:
            raise ValueError('release_confirm_frames must be positive integer')
        if not 0 < self.handback_step_rad <= .02:
            raise ValueError('Invalid handback step')
        if any(not isinstance(v,bool) for v in (self.protection_is_unload,self.protection_pauses_recovery,self.force_limited_recovery,self.normal_force_regulation)):
            raise ValueError('Protection switches must be Boolean')
        if self.normal_reference not in ('live','previous_target'):
            raise ValueError('Unknown normal-motion reference')
        if self.final_normal_force_gain is not None and not 0 < self.final_normal_force_gain <= .01:
            raise ValueError('Invalid final normal force gain')


def joint_slew_box(q, previous, joint_lower, joint_upper, rate, actuator_delta=None):
    """Joint limits intersect command slew; explicit bounded re-entry if empty.

    Unlike v3, no additional measured-q tracking box is silently introduced.
    A previous target outside the mechanical range cannot be repaired instantly
    while respecting slew: move toward the range and expose that conflict.
    """
    if actuator_delta is not None:
        hard_lo=torch.maximum(joint_lower,q-actuator_delta)
        hard_hi=torch.minimum(joint_upper,q+actuator_delta)
        impossible=hard_lo>hard_hi
        toward=q+(torch.maximum(torch.minimum(q,joint_upper),joint_lower)-q).clamp(-actuator_delta,actuator_delta)
        hard_lo=torch.where(impossible,toward,hard_lo);hard_hi=torch.where(impossible,toward,hard_hi)
        lo=torch.maximum(previous-rate,hard_lo);hi=torch.minimum(previous+rate,hard_hi)
        conflict=(lo>hi)|impossible
        # The downstream actuator can only accept q +/- actuator_delta. If
        # physics makes slew incompatible with that hard bound, expose it.
        closest=torch.maximum(torch.minimum(previous,hard_hi),hard_lo)
        lo=torch.where(conflict,closest,lo);hi=torch.where(conflict,closest,hi)
        return (lo-q).reshape(-1,4,4),(hi-q).reshape(-1,4,4),conflict.reshape(-1,4,4).any(-1)
    lo=torch.maximum(previous-rate,joint_lower)
    hi=torch.minimum(previous+rate,joint_upper)
    conflict=lo>hi
    reentry=previous+(torch.maximum(torch.minimum(previous,joint_upper),joint_lower)-previous).clamp(-rate,rate)
    lo=torch.where(conflict,reentry,lo);hi=torch.where(conflict,reentry,hi)
    return (lo-q).reshape(-1,4,4),(hi-q).reshape(-1,4,4),conflict.reshape(-1,4,4).any(-1)


class UnifiedCommandMCC(LoadAuthorityMCC):
    def __init__(self,*args,unified=None,**kwargs):
        self.unified_config=unified or UnifiedCommandConfig()
        super().__init__(*args,**kwargs)

    def reset(self):
        super().reset()
        self.unified_enabled=False
        self.protection_latched=torch.zeros_like(self.intent_unloading)
        self.release_count=torch.zeros_like(self.intent_age)
        self.handoff_offset=torch.zeros((self.batch,4,4),device=self.device)
        self.contact_anchor_local=torch.zeros((self.batch,4,3),device=self.device)
        self.measured_cache_valid=torch.zeros_like(self.intent_unloading)
        self.external_final_arbitration=False
        self.recovery_unloading=torch.zeros_like(self.intent_unloading)
        self.recovery_search_gain=torch.ones_like(self.load_request)

    def prepare_load_authority(self,request):
        super().prepare_load_authority(request)
        if self.unified_enabled and self.unified_config.protection_is_unload:
            # This affects inward force search and the outer recovery integrator,
            # not DP predictions. User unload remains persistent until revoked.
            recovering=self.handoff_offset.abs().amax(-1)>1e-7
            self.intent_unloading |= self.protection_latched
            gain=(1-self.handoff_offset.abs().amax(-1)/self.config.action_rate_limit).clamp(0,1)
            gain=torch.where(self.protection_latched,0.,torch.where(recovering,gain,1.))
            self.intent_inward_gain=torch.minimum(self.intent_inward_gain,gain)
        pause=(self.protection_latched if self.unified_enabled and self.unified_config.protection_pauses_recovery
               else torch.zeros_like(self.protection_latched))
        # Outer recovery and force-target authority are separate consumers.
        # Cancelling a stale search offset must not zero force preload/integral.
        self.recovery_unloading=self.intent_unloading|pause
        self.recovery_search_gain=torch.where(pause,0.,self.intent_inward_gain)

    def update(self,**inputs):
        if not self.unified_enabled:
            self.regulation_measurement=None
            return super().update(**inputs)
        if self.measured_contact is None:
            raise ValueError('Unified final command requires fingertip point/normal measurement')
        q=inputs['q_live'].to(self.load_request);pose=inputs['palm_pose_world'].to(q)
        previous=q.clone() if self.previous_command is None else self.previous_command.clone()
        user_unload=self.load_request==0
        measured,point,inward=self.measured_contact
        rot=_quat_matrix(pose)
        sites,_,origins,body_rot,_=self.fk.contact_kinematics(q)
        pp=torch.einsum('nij,nfj->nfi',rot.transpose(-1,-2),point-pose[:,None,:3])
        magnitude=inward.norm(dim=-1,keepdim=True)
        capture=(measured & torch.isfinite(point).all(-1) & torch.isfinite(inward).all(-1)
                 & (magnitude[...,0]>1e-6) & ((pp-sites).norm(dim=-1)<=self.config.contact_anchor_max_site_distance))
        outward=torch.nan_to_num(-inward/magnitude.clamp_min(1e-12))
        self.intent_normal_world=torch.where(capture[...,None],outward,self.intent_normal_world)
        self.intent_normal_age=torch.where(capture,0,(self.intent_normal_age+1).clamp_max(self.authority.normal_max_age_frames+1))
        local_anchor=torch.einsum('nfij,nfj->nfi',body_rot.transpose(-1,-2),torch.nan_to_num(pp)-origins)
        self.contact_anchor_local=torch.where(capture[...,None],local_anchor,self.contact_anchor_local)
        # The legacy branch may cache netforce's placeholder. A physical
        # constraint needs at least one real matched point/normal capture.
        self.measured_cache_valid |= capture
        valid=self.measured_cache_valid & (self.intent_normal_age<=self.authority.normal_max_age_frames)
        force=inputs['force_world'].to(q).norm(dim=-1)
        # Tactile normal force regulates normal compliance. Full resultant
        # remains the independent protection/hard-escape signal below.
        normal_force=(inputs['force_world'].to(q)*outward).sum(-1).abs()
        feedback_force=torch.where(capture,normal_force,force) if self.unified_config.normal_force_regulation else force
        self.regulation_measurement=feedback_force if self.unified_config.normal_force_regulation else None
        enter=capture & (force>=self.config.solver_contact_safety_force_ratio*self.force_setpoint)
        low=(~capture)|(force<=self.unified_config.release_force_ratio*self.force_setpoint)
        self.release_count=torch.where(low,self.release_count+1,0).clamp_max(self.unified_config.release_confirm_frames)
        self.protection_latched=(self.protection_latched & (self.release_count<self.unified_config.release_confirm_frames))|enter
        # Current force can suppress the inner search immediately. Outer search
        # used last tick's latch, but cannot bypass this final arbitration.
        self.intent_unloading=(user_unload|self.protection_latched if self.unified_config.protection_is_unload else user_unload)
        self.intent_inward_gain=torch.where(self.intent_unloading,0.,self.intent_inward_gain)
        integral_before=self.force_offset.clone()
        self.external_final_arbitration=True
        try:
            candidate,diagnostic=BatchedFullHandMCC.update(self,**inputs)
        finally:
            self.external_final_arbitration=False
        _,_,origins,body_rot,jac=self.fk.contact_kinematics(q,self.contact_anchor_local,valid)
        local_normal=torch.einsum('nij,nfj->nfi',rot.transpose(-1,-2),self.intent_normal_world)
        jn=torch.einsum('nfi,nfij->nfj',local_normal,jac)
        palm_step=torch.zeros_like(force)
        if self.compensate_palm_motion:
            if self.palm_twist_world is None:raise ValueError('Published palm twist required')
            contact_palm=origins+torch.einsum('nfij,nfj->nfi',body_rot,self.contact_anchor_local)
            lever=torch.einsum('nij,nfj->nfi',rot,contact_palm)
            velocity=self.palm_twist_world[:,None,:3]+torch.linalg.cross(self.palm_twist_world[:,None,3:].expand_as(lever),lever,dim=-1)
            palm_step=(velocity*self.intent_normal_world).sum(-1)*self.config.control_dt
            palm_step=torch.where(valid,palm_step,0.)
        rate=self.config.action_rate_limit
        joint_lower=self._clamp_q(self.lower.expand_as(q));joint_upper=self.upper.expand_as(q)
        # BatchedBControlChain maps (command-q)/.08 and clips to +/-2.
        # Include that existing .16-rad actuator envelope in the same solve.
        lo,hi,conflict=joint_slew_box(q,previous,joint_lower,joint_upper,rate,.16)
        reference=previous if self.unified_config.normal_reference=='previous_target' else q
        shift=(q-reference).reshape(-1,4,4)
        lo=lo+shift;hi=hi+shift
        active=((torch.ones_like(valid) if self.unified_config.force_limited_recovery
                 else (self.protection_latched|user_unload))&valid)
        unknown=user_unload&~valid
        # Explicit reset permission is granted only by the palm supervisor
        # after retreat and tactile unloading. Missing normals alone never
        # grant this permission. This prevents the normal-cache timeout from
        # locking a deliberately unloaded hand forever in its old posture.
        free_reset=getattr(self,'free_posture_reset',torch.zeros_like(valid))
        free_reset=free_reset & (force<self.config.contact_on_force)
        unknown=unknown & ~free_reset
        active=active & ~free_reset
        candidate_before_posture=candidate.clone()
        if hasattr(self,'posture_reserve_enabled'):
            healthy=torch.maximum(torch.minimum(candidate,self.posture_reserve_upper),self.posture_reserve_lower)
            # Soft posture preference, below final force/normal constraints.
            # Conflicting contact motion can override it; sustained measured
            # margin loss requests a palm reset rather than a hidden q jump.
            candidate=torch.where(self.posture_reserve_enabled[:,None],healthy,candidate)
        if hasattr(self,'reset_posture_target'):
            candidate=torch.where(free_reset.repeat_interleave(4,dim=-1),self.reset_posture_target,candidate)
            self.force_offset=torch.where(free_reset,0.,self.force_offset)
        hard_ratio=self._finger_constant('hard_ratio',self.config.overforce_hard_ratio,self.config.thumb_overforce_hard_ratio)
        hard=capture&(force>=hard_ratio*self.force_setpoint)
        normal_goal=torch.zeros_like(force)
        if self.unified_config.force_limited_recovery:
            # Continuous force-dependent normal-velocity bound, including
            # gentle re-approach at low force. Explicit unload targets zero,
            # so it never grants inward motion. Contact loss uses a recent
            # measured normal only, never a non-tip or mesh-derived normal.
            target=self.force_setpoint*self.regulation_load_scale
            target=torch.where(user_unload,0.,target)
            gain=(self.config.force_servo_integral_gain if self.unified_config.final_normal_force_gain is None
                  else self.unified_config.final_normal_force_gain)
            normal_goal=(gain*(feedback_force-target)*self.config.control_dt).clamp(
                -self.config.force_servo_max_step,self.config.force_servo_max_step)
        normal_goal=torch.where(hard,torch.maximum(normal_goal,torch.full_like(force,self.config.solver_contact_escape_step)),normal_goal)
        required=normal_goal-palm_step
        # Carry the prior correction relative to the moving nominal candidate;
        # do not hold a fixed posture or jump back to the uncorrected candidate.
        decay=self.handoff_offset.clamp(-self.unified_config.handback_step_rad,self.unified_config.handback_step_rad)
        carried=self.handoff_offset-decay
        desired=(candidate-reference).reshape(-1,4,4)+torch.where(active[...,None],0.,carried)
        desired=torch.where(unknown[...,None],(previous-reference).reshape(-1,4,4),desired)
        bounded=torch.maximum(torch.minimum(desired,hi),lo)
        projected,feasible=project_box_halfspace(desired,jn,lo,hi,required)
        delta=torch.where(active[...,None],projected,bounded)
        command=(reference.reshape(-1,4,4)+delta).reshape(-1,16)
        self.handoff_offset=torch.where(active[...,None],(command-candidate).reshape(-1,4,4),carried)
        self.previous_command=command.clone()
        self.intent_touched &= (self.load_request<1)|(self.regulation_load_scale<1)
        normal_step=(jn*delta).sum(-1)
        windup=torch.zeros_like(active)
        if self.unified_config.force_limited_recovery:
            requested_normal=(jn*(candidate-reference).reshape(-1,4,4)).sum(-1)
            # Conditional integration: preserve prior preload, but do not
            # integrate additional inward bias that the final command blocks.
            windup=active&(self.force_offset>integral_before)&(normal_step>requested_normal+1e-8)
            self.force_offset=torch.where(windup,integral_before,self.force_offset)
        diagnostic.update(
            intent_force_regulation_measurement=feedback_force.clone(),
            intent_force_resultant_n=force.clone(),
            intent_unified_enabled=torch.ones_like(active),
            intent_unified_latched=self.protection_latched.clone(),
            intent_unified_handoff_offset=self.handoff_offset.clone(),
            intent_unified_joint_conflict=conflict,
            intent_unified_recovery_block=self.recovery_unloading.clone(),
            intent_unified_normal_goal_m=normal_goal,
            intent_unified_windup_blocked=windup,
            intent_unified_normal_reference_q=reference.clone(),
            intent_unified_target_increment_normal_m=(jn*(command-previous).reshape(-1,4,4)).sum(-1),
            intent_unified_live_target_error_normal_m=(jn*(command-q).reshape(-1,4,4)).sum(-1),
            intent_unified_slew_violation=((command-previous).abs().reshape(-1,4,4).amax(-1)>rate+2e-6),
            intent_unified_candidate_q=candidate.clone(),
            intent_posture_candidate_correction_q=candidate-candidate_before_posture,
            intent_load_request=self.load_request.clone(),intent_load_applied=self.regulation_load_scale.clone(),
            intent_force_target=self.force_setpoint*self.regulation_load_scale,
            intent_inward_gain=self.intent_inward_gain.clone(),intent_unloading=self.intent_unloading.clone(),
            intent_normal_age=self.intent_normal_age.clone(),intent_missing_normal=unknown,
            intent_projection_active=active,intent_projection_infeasible=active&~feasible,
            intent_slew_conflict=conflict,
            intent_applied_normal_m=(jn*(command-q).reshape(-1,4,4)).sum(-1),
            intent_parent_normal_m=(jn*(candidate-q).reshape(-1,4,4)).sum(-1),
            intent_matched_geometry_active=capture,intent_matched_guard_active=active,
            intent_palm_normal_step_m=palm_step,intent_motion_guard_active=active,
            intent_motion_guard_infeasible=active&~feasible,intent_world_normal_step_m=palm_step+normal_step,
            intent_after_motion_guard_q=command.clone(),intent_normal_jacobian=jn,
            intent_cached_outward_world=self.intent_normal_world.clone(),intent_raw_tip_force_n=force,
            intent_lower_delta_q=lo,intent_upper_delta_q=hi,
        )
        return command,diagnostic
