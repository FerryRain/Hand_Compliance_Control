"""Batched normal constraints for split pad and connector collisions.

This optional MCC safety extension never counts connector contact as success.
It projects the final MCC target using actual collision normals, with the
known palm velocity accounted for. It does not change DP, its scheduler or
the prescribed palm path. The independent100N termination remains outside.
"""
import torch
from batched_fullhand_mcc import _quat_matrix


class TrueTipCollisionGuard:
    def __init__(self, mcc, *, connector_warn_n=10., pad_warn_n=30., escape_step_m=.0001):
        self.mcc=mcc
        self.connector_warn_n=connector_warn_n
        self.pad_warn_n=pad_warn_n
        self.escape_step_m=escape_step_m
        self.last=None

    @torch.no_grad()
    def __call__(self, q, command, pose, twist, contacts):
        rotation=_quat_matrix(pose)
        _,_,origins,body_rotation,_=self.mcc.fk.contact_kinematics(q)
        normals=[];jacobians=[];minimum=[];engaged=[]
        for name,warn in [('pad',self.pad_warn_n),('connector',self.connector_warn_n)]:
            points=contacts[name+'_points']
            inward=contacts[name+'_inward']
            force=contacts[name+'_force'].norm(dim=-1)
            finite=torch.isfinite(points).all(-1)&torch.isfinite(inward).all(-1)&torch.isfinite(force)
            magnitude=inward.norm(dim=-1)
            active=contacts[name+'_found']&finite&(magnitude>1e-6)&(force>=warn)
            outward=-torch.nan_to_num(inward)/magnitude.clamp_min(1e-9)[...,None]
            points_palm=torch.einsum('bij,bfj->bfi',rotation.transpose(-1,-2),points-pose[:,None,:3])
            local=torch.einsum('bfij,bfj->bfi',body_rotation.transpose(-1,-2),points_palm-origins)
            _,_,_,_,jac=self.mcc.fk.contact_kinematics(q,local,active)
            outward_palm=torch.einsum('bij,bfj->bfi',rotation.transpose(-1,-2),outward)
            jn=torch.einsum('bfi,bfij->bfj',outward_palm,jac)
            palm_velocity=twist[:,None,:3]+torch.cross(twist[:,None,3:].expand_as(points),points-pose[:,None,:3],dim=-1)
            palm_step=(outward*palm_velocity).sum(-1)*self.mcc.config.control_dt
            # At warning forbid additional inward movement. At twice warning
            # request the existing MCC-scale outward escape, not a force clamp.
            escape=torch.where(force>=2*warn,self.escape_step_m,0.)
            normals.append(jn);minimum.append(-palm_step+escape);engaged.append(active)
        normal=torch.stack(normals,2);bound=torch.stack(minimum,2);active=torch.stack(engaged,2)
        delta=(command-q).reshape(-1,4,4)
        lower=torch.maximum(self.mcc.lower-q,-torch.full_like(q,.02)).reshape(-1,4,4)
        upper=torch.minimum(self.mcc.upper-q,torch.full_like(q,.02)).reshape(-1,4,4)
        # Alternating halfspace/box projections. Tiny4D problems, every finger
        # and environment in one CUDA batch. No CPU per-finger solve.
        for _ in range(4):
            for contact in range(2):
                n=normal[:,:,contact]
                violation=(bound[:,:,contact]-(n*delta).sum(-1)).clamp_min(0)
                scale=torch.where(active[:,:,contact],violation/(n.square().sum(-1)+1e-8),0.)
                delta=delta+scale[...,None]*n
                delta=torch.maximum(torch.minimum(delta,upper),lower)
        safe=self.mcc._clamp_q(q+delta.reshape(-1,16))
        # Preserve the ordinary command bitwise on unconstrained fingers.
        safe=torch.where(active.any(-1).repeat_interleave(4,-1),safe,command)
        unresolved=torch.where(active,(bound-(normal*delta[:,:,None,:]).sum(-1)).clamp_min(0),0.)
        self.last=dict(active=active,unresolved_m=unresolved,correction=safe-command)
        self.mcc.previous_command=safe.clone()
        return safe
