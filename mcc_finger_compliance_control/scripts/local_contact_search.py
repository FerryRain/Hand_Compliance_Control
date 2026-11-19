"""Bounded per-finger search above unchanged MCC, only during recontact.

No object mesh or non-tip force feedback. A fresh tactile normal is preferred;
otherwise use a bounded opposition direction derived from hand FK. This is a
local search hypothesis, not a certified collision-free grasp planner.
"""
import torch
from batched_fullhand_mcc import _quat_matrix
from local_contact_recovery import LocalContactRecovery
from palm_posture_reset import APPROACH,RELOAD,HANDBACK,TRACK

class LocalContactSearch(LocalContactRecovery):
    def __init__(self,chain,lengths,config,local,pregrasp,search):
        super().__init__(chain,lengths,config,local,pregrasp)
        allowed={'speed_m_s','max_travel_m','max_joint_step_rad','require_thumb','normal_max_age'}
        if set(search)-allowed:raise ValueError('Unknown search option')
        self.search_cfg=dict(speed_m_s=.01,max_travel_m=.025,max_joint_step_rad=.005,require_thumb=True,normal_max_age=50,**{})
        self.search_cfg.update(search)
        c=self.search_cfg
        if not 0<c['speed_m_s']<=.02 or not 0<c['max_travel_m']<=.03 or not 0<c['max_joint_step_rad']<=.01 or not 0<=c['normal_max_age']<=100:raise ValueError('Search bounds')
        self.search_travel=torch.zeros(self.batch,4,device=self.device)
        self.search_active=torch.zeros_like(self.last_strict)
        self.search_direction=torch.zeros(self.batch,4,3,device=self.device)
        self.search_delta=torch.zeros(self.batch,16,device=self.device)
        self.search_reset_count=self.count.clone()

    def _good(self):
        good=super()._good()
        if self.search_cfg['require_thumb']:good &= self.last_strict[:,3]
        return good

    def observe(self,q,pose,tactile):
        super().observe(q,pose,tactile)
        new=self.count!=self.search_reset_count
        self.search_travel=torch.where(new[:,None],0.,self.search_travel)
        self.search_reset_count=self.count.clone()

    @torch.no_grad()
    def apply_nominal(self,q,nominal,load):
        c=self.search_cfg
        eligible=(self.phase==APPROACH)&(self.distance<=0)&self.last_strict.any(-1)
        active=eligible[:,None]&~self.last_strict&(self.search_travel<c['max_travel_m'])
        self.search_active=active;self.search_delta.zero_();self.search_direction.zero_()
        if bool(active.any()):
            sites,_,_,_,jac=self.chain.mcc.fk.contact_kinematics(self.reset_q)
            # Thumb aims toward the long-finger group; each long finger aims
            # toward the thumb. No global object center or surface lookup.
            goal=sites[:,3:4].expand(-1,4,-1).clone();goal[:,3]=sites[:,:3].mean(1)
            inward=goal-sites
            inward=inward/inward.norm(dim=-1,keepdim=True).clamp_min(1e-8)
            tactile=-torch.einsum('bij,bfj->bfi',_quat_matrix(self.current_pose).transpose(-1,-2),self.last_outward)
            recent=self.point_valid&(self.last_tactile_age<=c['normal_max_age'])
            direction=torch.where(recent[...,None],tactile,inward)
            step=torch.minimum(torch.full_like(self.search_travel,c['speed_m_s']*.01),(c['max_travel_m']-self.search_travel).clamp_min(0))
            desired=direction*step[...,None]*active[...,None]
            jt=jac.transpose(-1,-2)
            delta=(jt@torch.linalg.solve(jac@jt+1e-5*torch.eye(3,device=self.device),desired[...,None])).squeeze(-1)
            delta=delta.clamp(-c['max_joint_step_rad'],c['max_joint_step_rad']).reshape(self.batch,16)
            margin=.5*self.cfg.reserve_target*(self.upper-self.lower)
            target=torch.maximum(torch.minimum(self.reset_q+delta,self.upper-margin),self.lower+margin)
            target=torch.where(active.repeat_interleave(4,-1),target,self.reset_q)
            self.search_delta=target-self.reset_q
            self.reset_q=target
            self.search_travel+=step*active
            self.search_direction=direction*active[...,None]
        command,load=super().apply_nominal(q,nominal,load)
        # Search is a nominal target request. MCC keeps final compliance,
        # normal constraints and joint slew authority, including unloaded tips.
        return command,load

    def diagnostics(self):
        return dict(super().diagnostics(),contact_search_active=self.search_active.clone(),
                    contact_search_travel_m=self.search_travel.clone(),
                    contact_search_direction_palm=self.search_direction.clone(),
                    contact_search_delta_q=self.search_delta.clone())

    def report(self):
        return dict(super().report(),contact_search=self.search_cfg,
                    search_caveat='FK opposition fallback is bounded, not surface/collision certification; travel is requested path budget, not measured displacement')
