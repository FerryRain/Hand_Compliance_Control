"""Explicit MCC candidates; the existing controller remains the default.

Connector loading is a safety measurement, never a valid-pad observation or
force setpoint. The optional safety branch acts inside MCC before actuator
conversion, only after settling and only above a declared force warning.
"""
from dataclasses import replace
import torch
from batched_fullhand_mcc import _quat_matrix


PROFILES={
    'baseline':dict(name='baseline',overrides={}),
    'preload1mm':dict(name='preload1mm',overrides={'nominal_surface_preload':.001}),
    'pad_early_release':dict(name='pad_early_release',overrides={
        'overforce_hard_ratio':2.5,'thumb_overforce_hard_ratio':2.5,'solver_contact_escape_step':.00020}),
    'connector30':dict(name='connector30',overrides={},connector=dict(warn_n=30.,release_ratio=.7,escape_m=.00020)),
    'connector20':dict(name='connector20',overrides={},connector=dict(warn_n=20.,release_ratio=.7,escape_m=.00020)),
}


def configured_mcc(config,profile):
    return replace(config,**profile.get('overrides',{})) if profile else config


class ConnectorForceSafety:
    def __init__(self,chain,observer,config):
        self.chain,self.observer,self.config=chain,observer,config
        if not 0<config['warn_n']<100 or not 0<config['release_ratio']<1 or not 0<config['escape_m']<=.001:
            raise ValueError('Invalid connector force safety configuration')
        self.active=torch.zeros(chain.batch,4,device=chain.device,dtype=torch.bool)
        self.previous_update=chain.mcc.update

    @torch.no_grad()
    def update(self,**values):
        mcc=self.chain.mcc
        command,diagnostic=self.previous_update(**values)
        context=getattr(self.chain,'context',{})
        if context.get('settling',True):return command,diagnostic
        x=self.observer.latest
        force=x['connector_force'].norm(dim=-1)
        found=x['connector_found']&torch.isfinite(force)
        threshold=torch.where(self.active,self.config['warn_n']*self.config['release_ratio'],self.config['warn_n'])
        self.active=found&(force>=threshold)
        diagnostic['connector_force_safety_active']=self.active.clone()
        if not bool(self.active.any()):return command,diagnostic
        q=values['q_live'].float();pose=values['palm_pose_world'].float();rotation=_quat_matrix(pose)
        points=torch.einsum('bij,bfj->bfi',rotation.transpose(-1,-2),x['connector_points']-pose[:,None,:3])
        outward=torch.einsum('bij,bfj->bfi',rotation.transpose(-1,-2),-x['connector_inward'])
        length=outward.norm(dim=-1,keepdim=True)
        valid=self.active&torch.isfinite(points).all(-1)&torch.isfinite(outward).all(-1)&(length[...,0]>1e-6)
        outward=torch.nan_to_num(outward/length.clamp_min(1e-6))
        _,_,origins,body_rotation,_=mcc.fk.contact_kinematics(q)
        anchor=torch.einsum('bfij,bfj->bfi',body_rotation.transpose(-1,-2),torch.nan_to_num(points)-origins)
        _,_,_,_,jac=mcc.fk.contact_kinematics(q,anchor,valid)
        jn=torch.einsum('bfi,bfij->bfj',outward,jac)
        delta=self.config['escape_m']*jn/(jn.square().sum(-1,keepdim=True)+mcc.config.contact_point_jacobian_regularization)
        delta=torch.where(valid[...,None],delta,0.).clamp(-mcc.config.action_rate_limit,mcc.config.action_rate_limit)
        safe=mcc._clamp_q(q+delta.reshape(-1,16))
        command=torch.where(self.active.repeat_interleave(4,dim=-1),safe,command)
        # Do not store a hidden inward recovery command while retreating.
        mcc.force_offset=torch.where(self.active,mcc.force_offset.clamp_max(0),mcc.force_offset)
        mcc.transient_offset=torch.where(self.active,0.,mcc.transient_offset)
        self.chain.runtime_recovery_offset=torch.where(self.active.repeat_interleave(4,dim=-1),0.,self.chain.runtime_recovery_offset)
        mcc.previous_command=command.clone()
        diagnostic['connector_escape_m']=(jn*(command-q).reshape(-1,4,4)).sum(-1)
        return command,diagnostic


def attach_force_profile(chain,observer,profile):
    if not profile:return
    if profile.get('projection'):
        if profile['projection'].get('mode')!='paired' or profile.get('connector'):
            raise ValueError('Only independently validated paired projection is supported')
        if not observer.capture_collision:
            raise ValueError('Paired projection requires matched physical collision sensors')
        from paired_contact_projection import PairedContactProjection
        projection=PairedContactProjection(chain,observer,profile['projection'].get('include_palm_motion',False),
            profile['projection'].get('max_live_joint_step_rad'),
            profile['projection'].get('bound_all_commands',False))
        chain.paired_contact_projection=projection
        chain.mcc.update=projection.update
    if profile.get('connector'):
        safety=ConnectorForceSafety(chain,observer,profile['connector'])
        chain.connector_force_safety=safety
        chain.mcc.update=safety.update
