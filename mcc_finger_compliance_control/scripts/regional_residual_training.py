"""Post-physics PPO transitions on the regional DP deployment chain.

The actor and offset-rate interface are the original RL0 interface. Reward
uses actual completed physics ticks, including non-fingertip loads. Each lane
ends at its own reference length; padded batch tails never enter PPO.
"""
from __future__ import annotations

import torch

from train_rl0 import RL0Residual, RL0PPO, reward_components, state_potential
from batched_fullhand_mcc import _quat_matrix


def regional_rewards(strict, connector, force, seam, joint, task, offset,
                     previous, previous_phi, completed, unsafe_seen, parts, cfg):
    terminal=torch.zeros_like(completed)
    values,phi=reward_components(strict,connector,force,seam,joint,task,offset,previous,
        previous_phi,terminal,completed,cfg,parts[:,1:2].expand(-1,4))
    breach=force>100.
    first_breach=breach&~unsafe_seen
    unsafe_next=unsafe_seen|breach
    # Force excursions remain ordinary transitions: the simulation continues.
    values['unsafe_event']=-100.*first_breach.float()
    values['non_tip']=-.2/5*(parts[:,1:]>=.05).any(-1).float()
    values['terminal']=torch.where(completed&~unsafe_next,20.,0.)
    values['total']=sum(v for k,v in values.items() if k!='total')
    return values,phi,unsafe_next


class RegionalResidual(RL0Residual):
    def __init__(self,chain,agent,observer,bounds,cfg,*,lengths,zero=False,evaluate=False):
        super().__init__(chain,agent,observer,bounds,cfg,zero=zero,evaluate=evaluate)
        self.lengths=torch.as_tensor(lengths,device=chain.device,dtype=torch.long)
        self.finished=torch.zeros(chain.batch,device=chain.device,dtype=torch.bool)
        self.unsafe_seen=self.finished.clone()
        self.last_parts=torch.zeros(chain.batch,4,device=chain.device)
        self.last_task_nominal=None
        self.reward_sums=torch.zeros(chain.batch,device=chain.device)
        self.valid_reward_frames=torch.zeros(chain.batch,device=chain.device,dtype=torch.long)

    def geometry_costs(self,q,pose,x):
        strict=(x['pad_found']&torch.isfinite(x['pad_points']).all(-1)
                &torch.isfinite(x['pad_force']).all(-1)&(x['pad_force'].norm(dim=-1)>=.05))
        rotation=_quat_matrix(pose)
        points=torch.einsum('bij,bfj->bfi',rotation.transpose(-1,-2),x['pad_points']-pose[:,None,:3])
        local=self.geom.source_from_palm(q,points)
        distance=torch.stack([(local[:,i]*self.geom.parts[1 if i==3 else 0]['normal']).sum(-1)
                              -self.geom.parts[1 if i==3 else 0]['offset'] for i in range(4)],-1)
        margin=self.cfg['seam_warn_m']
        cost=((margin-distance)/margin).clamp(0,2).square()
        seam=torch.where(strict,cost,self.seam)
        travel=(self.chain.mcc.upper-self.chain.mcc.lower).clamp_min(.01)
        boundary=torch.minimum(q-self.chain.mcc.lower,self.chain.mcc.upper-q)/travel
        joint=((.05-boundary)/.05).clamp(0,2).square().mean(-1)
        return strict,seam,joint

    def observations(self,q,nominal,state,contacts,joint):
        actor,critic=super().observations(q,nominal,state,contacts,joint)
        critic[:,-1]=(self.context['frame']/self.lengths.float()).clamp(0,1)
        critic=torch.cat((critic,(self.last_parts/100).clamp(0,10)),-1)
        if critic.shape[-1]!=542:raise ValueError('Regional critic layout')
        return actor,critic

    @torch.no_grad()
    def forward(self,q,nominal,state):
        ctx=self.context;x=self.observer.latest
        _,self.seam,joint=self.geometry_costs(q,ctx['palm_pose'],x)
        if self.previous_phi is None:
            self.previous_phi=state_potential(self.seam,joint,x['connector_force'].norm(dim=-1))
        obs,critic=self.observations(q,nominal,state,x,joint)
        active=(ctx['frame']<self.lengths)&~self.finished
        if self.step_index%5==0:
            self.flush(self.pending,obs,critic,torch.zeros_like(active))
            action=self.agent.act(obs,zero=self.zero,evaluate=self.evaluate)
            self.target=(self.offset+action*self.slew).clamp(-self.amplitude,self.amplitude)
            self.previous_obs=obs.clone();self.previous_critic=critic.clone()
            self.previous_z=self.agent.last_z.clone();self.previous_logp=self.agent.last_logp.clone()
            self.pending.copy_(active)
        self.previous_offset=self.offset.clone()
        self.offset+=(self.target-self.offset).clamp(-self.slew/5,self.slew/5)
        self.offset=torch.where(active[:,None],self.offset.clamp(-self.amplitude,self.amplitude),0.)
        self.last_task_nominal=nominal.clone()
        self.step_index+=1
        return self.offset.clone()

    @torch.no_grad()
    def post_physics(self,frame,q,palm_pose,parts):
        # The residual is first applied after the original bootstrap at frame76.
        if self.previous_phi is None:return
        active=self.pending&~self.finished&(frame<self.lengths)
        x=self.observer.latest
        strict,seam,joint=self.geometry_costs(q,palm_pose,x)
        connector=x['connector_found']&(x['connector_force'].norm(dim=-1)>=.05)
        disruption=(self.chain.mcc.fk(self.last_task_nominal+self.offset)
                    -self.chain.mcc.fk(self.last_task_nominal)).norm(dim=-1)
        task=((disruption-.002)/.002).clamp(0,2).square().mean(-1)
        completed=(frame+1>=self.lengths)&active
        values,phi,unsafe=regional_rewards(strict,connector,parts.amax(-1),seam,joint,task,
            self.offset/self.amplitude,self.previous_offset/self.amplitude,self.previous_phi,
            completed,self.unsafe_seen,parts,self.cfg)
        self.accumulated+=torch.where(active,self.cfg['gamma_frame']**self.elapsed*values['total'],0.)
        self.elapsed+=active.long()
        self.return_per_lane+=torch.where(active,values['total'],0.)
        self.valid_reward_frames+=active.long()
        for name,value in values.items():
            self.sums[name]=self.sums.get(name,0.)+torch.where(active,value,0.).sum()
        self.previous_phi=torch.where(active,phi,self.previous_phi)
        self.unsafe_seen=torch.where(active,unsafe,self.unsafe_seen)
        self.last_parts=parts.clone();self.seam=seam
        # Complete lanes close their actual last transition immediately. The
        # terminal discount is zero, so no invented next observation is used.
        self.flush(completed,self.previous_obs,self.previous_critic,completed)
        self.finished|=completed


def restore_agent(state,training=False):
    agent=RL0PPO(state['config'],state['bounds'])
    agent.actor.load_state_dict(state['actor']);agent.value.load_state_dict(state['value'])
    if training:
        agent.actor_optimizer.load_state_dict(state['actor_optimizer'])
        agent.value_optimizer.load_state_dict(state['value_optimizer'])
        agent.generator.set_state(state['generator'].cpu())
        agent.transitions=state['transitions'];agent.updates=state['updates'];agent.actor_updates=state['actor_updates']
    agent.actor.eval().requires_grad_(training);agent.value.eval().requires_grad_(training)
    return agent


def save_agent(path,agent,cfg,bounds,rollouts):
    state=dict(actor=agent.actor.state_dict(),value=agent.value.state_dict(),
        actor_optimizer=agent.actor_optimizer.state_dict(),value_optimizer=agent.value_optimizer.state_dict(),
        config=cfg,bounds=bounds,transitions=agent.transitions,updates=agent.updates,
        actor_updates=agent.actor_updates,generator=agent.generator.get_state(),rollouts=rollouts)
    from pathlib import Path
    path=Path(path);temporary=path.with_suffix('.tmp')
    torch.save(state,temporary);temporary.replace(path)
