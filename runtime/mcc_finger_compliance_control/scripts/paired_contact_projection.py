"""Experimental MCC projection using a matched point, normal and normal load.

The ordinary MCC still receives netforce/centroid/source-mesh feedback. Only
its final safety projection is replaced. The prescribed target object is static
in the inverse replay, so relative base motion is the palm's world twist.
"""
from dataclasses import replace
import math
import torch
from batched_fullhand_mcc import _quat_matrix


def project_halfspace_box(candidate,origin,jac,required,lower,upper):
    """Euclidean projection in four joint coordinates onto a box/halfspace.

At most four joints can become bound-active. Every remaining free coordinate
moves along its normal Jacobian; no per-environment loop is used.
"""
    value=candidate.clamp(min=lower,max=upper)
    for _ in range(4):
        deficit=(required-(jac*(value-origin)).sum(-1)).clamp_min(0)
        free=((jac>0)&(value<upper))|((jac<0)&(value>lower))
        direction=jac*free
        multiplier=deficit/direction.square().sum(-1).clamp_min(1e-12)
        value=(value+multiplier[...,None]*direction).clamp(min=lower,max=upper)
    margin=(jac*(value-origin)).sum(-1)-required
    return value,margin


def measured_step_bounds(lower,upper,q,joint_lower,joint_upper,limit):
    """Restrict a safety projection to a bounded offset from measured q.

If the old command band and measured band have no intersection, withdraw
toward the measured band. A stale actuator target must not authorize a large
new Cartesian escape. Mechanical joint limits continue to apply.
"""
    if limit is None:return lower,upper,None
    live_lower=torch.maximum(joint_lower,q-float(limit))
    live_upper=torch.minimum(joint_upper,q+float(limit))
    bounded_lower=torch.maximum(lower,live_lower)
    bounded_upper=torch.minimum(upper,live_upper)
    empty=bounded_lower>bounded_upper
    return (torch.where(empty,live_lower,bounded_lower),
            torch.where(empty,live_upper,bounded_upper),empty)


class PairedContactProjection:
    def __init__(self,chain,observer,include_palm_motion,max_live_joint_step_rad=None,bound_all_commands=False):
        self.chain,self.observer=chain,observer
        self.include_palm_motion=include_palm_motion
        if max_live_joint_step_rad is not None and (
                not math.isfinite(max_live_joint_step_rad) or not 0<max_live_joint_step_rad<=.16):
            raise ValueError('Invalid measured-q safety step bound')
        self.max_live_joint_step_rad=max_live_joint_step_rad
        if bound_all_commands and max_live_joint_step_rad is None:
            raise ValueError('All-command bound needs a measured-q error limit')
        self.bound_all_commands=bool(bound_all_commands)
        self.previous_update=chain.mcc.update
        self.disabled_config=replace(chain.mcc.config,solver_contact_safety=False)
        self.counts=torch.zeros(4,dtype=torch.long,device=chain.device)
        self.flexion_indices=torch.tensor((2,3),device=chain.device,dtype=torch.long)
        self._hard_ratio_values=None
        self._hard_ratios=None

    @torch.no_grad()
    def update(self,**values):
        mcc=self.chain.mcc;context=getattr(self.chain,'context',{})
        if context.get('settling',True):return self.previous_update(**values)
        previous=mcc.previous_command
        original=mcc.config;mcc.config=self.disabled_config
        try:command,diagnostic=self.previous_update(**values)
        finally:mcc.config=original
        command,extra=self.project(values['q_live'],command,values['palm_pose_world'],
                                   self.chain.projection_palm_twist,previous)
        mcc.previous_command=command.clone()
        diagnostic.update(extra)
        if context['frame']>=76:
            self.counts+=torch.stack((extra['paired_engaged'].sum(),extra['paired_changed'].sum(),
                extra['paired_infeasible'].sum(),(extra['paired_engaged']&(extra['palm_normal_step_m']<0)).sum()))
        return command,diagnostic

    @torch.no_grad()
    def project(self,q,command,pose,twist,previous=None):
        mcc=self.chain.mcc;cfg=mcc.config;x=self.observer.latest
        command=mcc._clamp_q(command)
        point=x['physical_contact_point'];inward=x['physical_normal_inward']
        normal_force=x['physical_normal_force'];found=x['physical_normal_found']
        rot=_quat_matrix(pose);length=inward.norm(dim=-1,keepdim=True)
        outward=-inward/length.clamp_min(1e-8)
        point_palm=torch.einsum('bij,bfj->bfi',rot.transpose(-1,-2),point-pose[:,None,:3])
        normal_palm=torch.einsum('bij,bfj->bfi',rot.transpose(-1,-2),outward)
        sites,_,origins,body_rot,_=mcc.fk.contact_kinematics(q)
        valid=found&torch.isfinite(point).all(-1)&torch.isfinite(inward).all(-1)&torch.isfinite(normal_force)
        valid&=(length[...,0]>1e-8)&((point_palm-sites).norm(dim=-1)<=cfg.contact_anchor_max_site_distance)
        anchor=torch.einsum('bfij,bfj->bfi',body_rot.transpose(-1,-2),torch.nan_to_num(point_palm)-origins)
        _,_,_,_,jac=mcc.fk.contact_kinematics(q,anchor,valid)
        jn=torch.nan_to_num(torch.einsum('bfi,bfij->bfj',normal_palm,jac))
        engaged=valid&(normal_force>=cfg.solver_contact_safety_force_ratio*mcc.force_setpoint)
        ratio_values=(cfg.overforce_hard_ratio,)*3+(
            cfg.overforce_hard_ratio if cfg.thumb_overforce_hard_ratio is None else cfg.thumb_overforce_hard_ratio,)
        if ratio_values!=self._hard_ratio_values:
            self._hard_ratios=q.new_tensor(ratio_values)
            self._hard_ratio_values=ratio_values
        ratios=self._hard_ratios
        hard=valid&(normal_force>=ratios*mcc.force_setpoint)
        palm_step=torch.zeros_like(normal_force)
        if self.include_palm_motion:
            velocity=twist[:,None,:3]+torch.linalg.cross(twist[:,None,3:].expand_as(point),point-pose[:,None,:3])
            palm_step=(outward*velocity).sum(-1)*cfg.control_dt
            palm_step=torch.where(valid,torch.nan_to_num(palm_step),0.)
        minimum=torch.where(hard,cfg.solver_contact_escape_step,0.)
        required=minimum-palm_step
        lower=mcc.lower.reshape(4,4).expand(len(q),-1,-1).clone()
        upper=mcc.upper.reshape(4,4).expand_as(lower).clone()
        if cfg.natural_flexion_floor is not None:
            lower.index_copy_(-1,self.flexion_indices,
                lower.index_select(-1,self.flexion_indices).clamp_min(cfg.natural_flexion_floor))
        prev=(q if previous is None else previous).reshape(-1,4,4)
        seed=command.reshape(-1,4,4)
        # Preserve an already-issued hard-safety seed outside the ordinary
        # rate box, while preventing the new projection from extending it.
        lower=torch.minimum(torch.maximum(lower,prev-cfg.action_rate_limit),seed)
        upper=torch.maximum(torch.minimum(upper,prev+cfg.action_rate_limit),seed)
        lower,upper,empty=measured_step_bounds(lower,upper,q.reshape(-1,4,4),
            mcc.lower.reshape(4,4),mcc.upper.reshape(4,4),self.max_live_joint_step_rad)
        projected,margin=project_halfspace_box(seed,q.reshape(-1,4,4),jn,required,lower,upper)
        ordinary=seed.clamp(min=lower,max=upper) if self.bound_all_commands else seed
        result=torch.where(engaged[...,None],projected,ordinary).reshape(-1,16)
        changed=((result-command).reshape(-1,4,4).abs().amax(-1)>1e-8)&engaged
        infeasible=engaged&(margin< -1e-6)
        extra=dict(paired_engaged=engaged,paired_changed=changed,paired_infeasible=infeasible,
            palm_normal_step_m=palm_step,paired_constraint_margin_m=margin,
            paired_normal_force_n=normal_force)
        if empty is not None:
            extra['paired_live_box_fallback']=empty.any(-1)&engaged
        if self.bound_all_commands:
            extra['bounded_command_changed']=(result-command).reshape(-1,4,4).abs().amax(-1)>1e-8
        return result,extra
