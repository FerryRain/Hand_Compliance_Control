"""DPPO transitions on the unchanged regional deployment physics.

No residual or domain randomization. Rewards use completed physics ticks;
overforce is penalized but never terminates a lane. Native path ends are
terminal, with padded tails excluded. Probe compares on identical live states.
"""
from pathlib import Path
import time
import torch

from train_dp0_dppo import DPPOAgent
from dppo_control_env import DecisionCollector
from regional_residual_training import RegionalResidual, regional_rewards
from train_rl0 import state_potential
from regional_dppo_protocol import training_reward_adjustments


class RegionalDPPO(DecisionCollector):
    def __init__(self, task, observer):
        self.task=task
        spec=task['dppo'];cfg=spec['config']
        self.training=spec['mode']=='train';self.probe=spec['mode']=='probe'
        if self.training:
            torch.backends.cudnn.allow_tf32=False
            torch.backends.cuda.matmul.allow_tf32=False
        agent=DPPOAgent(cfg,spec.get('resume'))
        super().__init__(agent,cfg,len(task['rows']),observer)
        self.geom=self.geometry
        self.lengths=torch.tensor([r['frames'] for r in task['rows']],device=agent.device)
        self.finished=torch.zeros_like(self.lengths,dtype=torch.bool)
        self.unsafe_seen=self.finished.clone()
        self.last_parts=torch.zeros(len(self.lengths),4,device=agent.device)
        self.valid_frames=torch.zeros_like(self.lengths)
        self.checks=dict(sample_max_abs=0.,absolute_chunk_max_abs=0.,rng_equal=True,plans=0)
        self.started=time.monotonic()

    def critic_input(self,condition,context):
        x=self.observer.latest
        private=torch.cat((x['pad_force'].norm(dim=-1)/100,self.last_parts/100,
            (context['frame']/self.lengths.float()).clamp(0,1)[:,None]),-1)
        return torch.nan_to_num(torch.cat((condition,private),-1),nan=0.,posinf=10.,neginf=-10.)

    def attach(self,chain):
        self.chain=chain
        original=chain.runtime.policy.diffusion.conditional_sample
        @torch.no_grad()
        def sample(batch_size,global_cond=None,generator=None,noise=None):
            if noise is not None:raise ValueError('Unexpected explicit diffusion noise')
            if self.probe:
                before=generator.get_state().clone()
                reference=original(batch_size,global_cond=global_cond,generator=generator)
                rng=generator.get_state().clone();generator.set_state(before)
            critic=self.critic_input(global_cond,chain.context)
            if self.training:self.close(self.pending,critic)
            result=self.agent.sampler.sample(batch_size,global_cond,generator,
                exploration=self.training,record=self.training)
            if self.training:
                self.current=self.agent.sampler.last
                self.critic=critic.clone();self.value=self.agent.critic(critic).squeeze(-1)
                self.pending.copy_((chain.context['frame']<self.lengths)&~self.finished)
            if self.probe:
                self.checks['sample_max_abs']=max(self.checks['sample_max_abs'],float((reference-result).abs().max()))
                self.checks['rng_equal'] &= bool(torch.equal(rng,generator.get_state()))
                self.reference_chunk=reference*chain.action_std+chain.action_mean+chain.context['q'][:,None]
                decoded=result*chain.action_std+chain.action_mean+chain.context['q'][:,None]
                self.checks['absolute_chunk_max_abs']=max(self.checks['absolute_chunk_max_abs'],float((decoded-self.reference_chunk).abs().max()))
            self.checks['plans']+=1
            return result
        chain.runtime.policy.diffusion.conditional_sample=sample

    @torch.no_grad()
    def post_physics(self,chain,frame,q,pose,parts):
        if not self.training:return
        active=self.pending&~self.finished&(frame<self.lengths)&(frame>chain.bootstrap_end)
        x=self.observer.latest
        strict,seam,joint=RegionalResidual.geometry_costs(self,q,pose,x)
        if self.previous_phi is None:
            self.previous_phi=state_potential(seam,joint,parts[:,1:2].expand(-1,4))
        connector=x['connector_found']&(x['connector_force'].norm(dim=-1)>=.05)
        completed=(frame+1>=self.lengths)&active
        zero=torch.zeros(len(q),16,device=q.device)
        values,phi,unsafe=regional_rewards(strict,connector,parts.amax(-1),seam,joint,
            torch.zeros(len(q),device=q.device),zero,zero,self.previous_phi,completed,
            self.unsafe_seen,parts,self.cfg)
        # DP has no residual action: omit its identically-zero regularizers.
        for name in ('task','effort','smooth'):values.pop(name)
        values=training_reward_adjustments(values,parts.amax(-1),joint,self.cfg)
        total=values['total']
        if not torch.isfinite(total[active]).all():raise RuntimeError('Non-finite DPPO reward')
        self.accumulated+=torch.where(active,self.cfg['gamma_frame']**self.elapsed*total,0.)
        self.returns+=torch.where(active,total,0.)
        self.elapsed+=active.long();self.valid_frames+=active.long()
        for name,value in values.items():
            self.components[name]=self.components.get(name,0.)+torch.where(active,value,0.).sum()
        self.previous_phi=torch.where(active,phi,self.previous_phi)
        self.unsafe_seen=torch.where(active,unsafe,self.unsafe_seen)
        self.last_parts=parts.clone();self.seam=seam
        self.close(completed,terminal=True)
        self.finished|=completed

    def finish(self,shadow_errors):
        report=dict(mode=self.task['dppo']['mode'],residual='zero',domain_randomization=False,
            probe={**self.checks,**shadow_errors},
            reward_components={k:float(v.cpu()) for k,v in self.components.items()},
            episode_returns=self.returns.cpu().tolist(),valid_reward_frames=self.valid_frames.cpu().tolist(),
            decisions=sum(len(r['lane']) for r in self.rows))
        if self.probe:
            passed=(self.checks['plans']>0 and self.checks['rng_equal']
                and max(self.checks['sample_max_abs'],self.checks['absolute_chunk_max_abs'],*shadow_errors.values())<=1e-7)
            report['probe']['passed']=passed
            if not passed:raise RuntimeError(f'DPPO native equivalence failed: {report}')
        if self.training:
            if self.pending.any() or not self.finished.all():raise RuntimeError('Unclosed DPPO episodes')
            if not torch.equal(self.valid_frames,self.lengths-76):raise RuntimeError('Wrong reward frame count')
            report['likelihood']=self.agent.likelihood_check(self.rows)
            if not report['likelihood']['passed']:raise RuntimeError(f'DPPO likelihood mismatch: {report}')
            report['update']=self.agent.update(self.rows,value_only=self.task['dppo'].get('value_only',False))
            self.agent.save(Path(self.task['dppo']['folder']))
        report['elapsed_s']=time.monotonic()-self.started
        report['cuda_peak_allocated_mib']=torch.cuda.max_memory_allocated()/2**20
        report['cuda_peak_reserved_mib']=torch.cuda.max_memory_reserved()/2**20
        return report
