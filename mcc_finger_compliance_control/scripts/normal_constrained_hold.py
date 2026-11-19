"""Experimental MCC hold arbitration; ordinary force regulation is unchanged.

During hold, preserve feasible nominal motion instead of replacing the entire
finger target with live q. A local contact-normal inequality and bounded joint
steps are enforced. This is NOT a guarantee about other contacts or dynamics.
"""
import torch
from batched_fullhand_mcc import BatchedFullHandMCC, _quat_matrix


def project_box_halfspace(target, jacobian, lower, upper, minimum):
    """Nearest box point with J*dq >= minimum (one halfspace per finger).

If infeasible, achieve the maximal available normal displacement and expose
the infeasibility. No relaxation is hidden from the diagnostic caller.
"""
    bounded=torch.maximum(torch.minimum(target,upper),lower)
    maximum=(jacobian*torch.where(jacobian>=0,upper,lower)).sum(-1)
    feasible=maximum>=minimum-1e-9
    goal=torch.minimum(minimum,maximum)
    nonzero=jacobian.abs()>1e-12
    safe_j=torch.where(nonzero,jacobian,torch.ones_like(jacobian))
    edge=torch.where(jacobian>0,upper,lower)
    high=torch.where(nonzero,(edge-target)/safe_j,0).clamp_min(0).amax(-1)+1.
    low=torch.zeros_like(high)
    needs=(jacobian*bounded).sum(-1)<goal
    for _ in range(28):
        middle=(low+high)*.5
        value=torch.maximum(torch.minimum(target+middle[...,None]*jacobian,upper),lower)
        below=(jacobian*value).sum(-1)<goal
        low=torch.where(below,middle,low);high=torch.where(below,high,middle)
    projected=torch.maximum(torch.minimum(target+high[...,None]*jacobian,upper),lower)
    return torch.where(needs[...,None],projected,bounded),feasible


class NormalConstrainedHoldMCC(BatchedFullHandMCC):
    def update(self,**inputs):
        command,diagnostic=super().update(**inputs)
        safety=diagnostic['collision_safety_active']
        q=inputs['q_live'].to(device=self.device,dtype=torch.float32)
        nominal=inputs['nominal_posture_q'].to(q)
        pose=inputs['palm_pose_world'].to(q)
        outward=inputs.get('collision_normals_outward_world')
        if outward is None:outward=inputs['surface_normals_world']
        outward=outward.to(q)
        norm=outward.norm(dim=-1,keepdim=True)
        # No fabricated normal: unsupported directions retain the old guard.
        valid=norm[...,0]>1e-6
        outward=outward/norm.clamp_min(1e-12)
        local=torch.einsum('nij,nfj->nfi',_quat_matrix(pose).transpose(-1,-2),outward)
        _,_,_,_,jac=self.fk.contact_kinematics(q,self.anchor,self.anchor_valid)
        jn=torch.einsum('nfi,nfij->nfj',local,jac)
        rate=self.config.action_rate_limit
        # If physics has pushed q outside a joint bound, return gradually;
        # do not teleport the target outside the permitted step to clamp q.
        lower=(self._clamp_q(q-rate)-q).clamp(-rate,rate).reshape(-1,4,4)
        upper=(self._clamp_q(q+rate)-q).clamp(-rate,rate).reshape(-1,4,4)
        hard_ratio=self._finger_constant('hard_ratio',self.config.overforce_hard_ratio,self.config.thumb_overforce_hard_ratio)
        hard=inputs['found'].bool() & (diagnostic['raw_force_magnitude']>=hard_ratio*self.force_setpoint)
        minimum=torch.where(hard,self.config.solver_contact_escape_step,0.)
        desired=(nominal-q).reshape(-1,4,4)
        delta,feasible=project_box_halfspace(desired,jn,lower,upper,minimum)
        enabled=safety & valid
        command=torch.where(enabled[...,None],q.reshape(-1,4,4)+delta,command.reshape(-1,4,4)).reshape(-1,16)
        self.previous_command=command.clone()
        diagnostic.update(hold_projection_active=enabled,
                          hold_projection_infeasible=enabled & ~feasible,
                          hold_requested_normal_m=(jn*desired).sum(-1),
                          hold_applied_normal_m=(jn*(command-q).reshape(-1,4,4)).sum(-1),
                          hold_minimum_normal_m=minimum)
        return command,diagnostic
