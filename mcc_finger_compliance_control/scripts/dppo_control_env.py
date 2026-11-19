"""DP decisions around the accepted GPU scheduler/MCC/actuator chain.

Decisions are issued at frames75,85,...; the first chunk starts executing at76.
Rewards are attributed to the previous decision before a new chunk is sampled.
Each lane terminates independently. Planned trajectory completion is terminal;
an explicitly shorter collection horizon is truncated with value bootstrap.
"""
import hashlib
import torch
from dataclasses import replace

import deploy_b_gpu_serial_parity as deployment
from batched_b_control_chain import BatchedBControlChain,ZeroResidualPolicy
from evaluate_dp0_closed_loop import Metrics,pack
from mcc_force_profiles import configured_mcc,attach_force_profile
from true_fingertip_contact import ContactSlotObserver,TrueFingertipContact
from batched_fullhand_mcc import _quat_matrix
from dppo_diffusion import surrogate_logprob


class DecisionCollector:
    def __init__(self,agent,cfg,batch,observer):
        self.agent,self.cfg,self.observer=agent,cfg,observer
        device=agent.device
        self.pending=torch.zeros(batch,dtype=torch.bool,device=device)
        self.elapsed=torch.zeros(batch,dtype=torch.long,device=device)
        self.accumulated=torch.zeros(batch,device=device)
        self.returns=torch.zeros(batch,device=device)
        self.previous_phi=None;self.rows=[];self.components={}
        self.geometry=TrueFingertipContact(device)
        self.seam=torch.zeros(batch,4,device=device)
        self.current=None;self.critic=None;self.value=None
        self.sampling_time_s=0.

    def critic_input(self,condition,context):
        x=self.observer.latest
        private=torch.cat((x['pad_force'].norm(dim=-1)/100,
            x['connector_force'].norm(dim=-1)/100,
            condition.new_full((len(condition),1),context['frame']/self.cfg['trajectory_frames'])),-1)
        return torch.nan_to_num(torch.cat((condition,private),-1),nan=0.,posinf=10.,neginf=-10.)

    @torch.no_grad()
    def close(self,mask,next_critic=None,terminal=False):
        ids=(mask&self.pending).nonzero().flatten()
        if not len(ids):return
        if next_critic is None:next_value=torch.zeros(len(ids),device=self.agent.device)
        else:next_value=self.agent.critic(next_critic[ids]).squeeze(-1)
        discount=self.cfg['gamma_frame']**self.elapsed[ids]
        if terminal:discount=torch.zeros_like(discount)
        row={k:v[ids].detach().cpu().clone() for k,v in self.current.items()}
        row.update(lane=ids.cpu(),critic=self.critic[ids].cpu().clone(),
            value=self.value[ids].cpu().clone(),next_value=next_value.cpu(),
            reward=self.accumulated[ids].cpu(),discount=discount.cpu(),duration=self.elapsed[ids].cpu())
        self.rows.append(row)
        self.pending[ids]=False;self.elapsed[ids]=0;self.accumulated[ids]=0

    @torch.no_grad()
    def sample(self,condition,generator,chain,exploration,record):
        critic=self.critic_input(condition,chain.context)
        record=record and not chain.context['last_observation']
        if record:self.close(self.pending,critic)
        output=self.agent.sampler.sample(len(condition),condition,generator,
            exploration=exploration,record=record)
        if record:
            self.current=self.agent.sampler.last
            self.critic=critic.clone();self.value=self.agent.critic(critic).squeeze(-1)
            self.pending.copy_(chain.context['active'])
        return output

    @torch.no_grad()
    def observe(self,chain):
        ctx=chain.context
        if ctx['settling']:return
        x=self.observer.latest;q=ctx['q'];frame=ctx['frame']
        strict=x['pad_found']&torch.isfinite(x['pad_points']).all(-1)&torch.isfinite(x['pad_force']).all(-1)&(x['pad_force'].norm(dim=-1)>=.05)
        connector=x['connector_found']&(x['connector_force'].norm(dim=-1)>=.05)
        rotation=_quat_matrix(ctx['palm_pose'])
        points=torch.einsum('bij,bfj->bfi',rotation.transpose(-1,-2),x['pad_points']-ctx['palm_pose'][:,None,:3])
        source=self.geometry.source_from_palm(q,torch.nan_to_num(points))
        distance=torch.stack([(source[:,i]*self.geometry.parts[1 if i==3 else 0]['normal']).sum(-1)-self.geometry.parts[1 if i==3 else 0]['offset'] for i in range(4)],-1)
        margin=self.cfg['seam_warn_m']
        seam=((margin-distance)/margin).clamp(0,2).square()
        self.seam=torch.where(strict,seam,self.seam)
        travel=(chain.mcc.upper-chain.mcc.lower).clamp_min(.01)
        boundary=torch.minimum(q-chain.mcc.lower,chain.mcc.upper-q)/travel
        joint=((.05-boundary)/.05).clamp(0,2).square().mean(-1)
        phi=-.3*self.seam.mean(-1)-.1*joint-.5*(x['connector_force'].norm(dim=-1)/20).clamp(0,2).square().mean(-1)
        terminal=ctx['newly_stopped']|~torch.isfinite(q).all(-1)
        final=bool(ctx['last_observation'])
        completed=(ctx['active']&torch.full_like(terminal,final and frame==self.cfg['trajectory_frames']-1))
        next_phi=torch.where(terminal|completed,0.,phi)
        previous=phi if self.previous_phi is None else self.previous_phi
        risk=((self.observer.safety_peak_force_n-self.cfg['force_warn_n'])/(100-self.cfg['force_warn_n'])).clamp(0,2).square()
        terms=dict(contact=.2*(.8*strict.all(-1).float()+.2*strict.float().mean(-1))/5,
            loss=-.2*(~strict).float().mean(-1)/5,
            connector=-.2*connector.float().mean(-1)/5,force=-4*risk/5,
            posture=-.03*joint/5,potential=self.cfg['gamma_frame']*next_phi-previous,
            terminal=torch.where(terminal,-100.,torch.where(completed,20.,0.)))
        total=sum(terms.values())
        self.returns+=torch.where(self.pending,total,0.)
        self.accumulated+=torch.where(self.pending,self.cfg['gamma_frame']**self.elapsed*total,0.)
        for name,value in terms.items():self.components[name]=self.components.get(name,0.)+torch.where(self.pending,value,0.).sum()
        self.elapsed+=self.pending.long();self.previous_phi=phi
        self.close(terminal,terminal=True)
        if final:
            if frame==self.cfg['trajectory_frames']-1:self.close(self.pending,terminal=True)
            else:
                history=chain.observation.history()
                normalized=(history-chain.state_mean)/chain.state_std*chain.state_mask
                # Actor has no learned encoder: flattened96x16 is its conditioning.
                self.close(self.pending,self.critic_input(normalized.flatten(1),ctx))


def collect_rollout(agent,cfg,ids,seed,*,exploration=False,record=True,probe=False,deployment_checkpoint=None):
    observer=ContactSlotObserver(controller_scope='pad',capture_collision=True)
    metric=Metrics(observer,len(ids),cfg['frames'],agent.device)
    collector=DecisionCollector(agent,cfg,len(ids),observer)
    data=pack(ids,cfg['frames']);old=deployment.MODEL
    deployment.MODEL=str(deployment_checkpoint or cfg['checkpoint'])
    checks=dict(sample_max_abs=0.,action_max_abs=0.,nominal_max_abs=0.,command_max_abs=0.,
        observation_max_abs=0.,rng_equal=True,plans=0)
    noise_schedule=[]
    class Chain(BatchedBControlChain):
        def __init__(self,*args,**kwargs):
            kwargs['mcc_config']=configured_mcc(replace(kwargs['mcc_config'],back_contact_x_limit_m=(1.,1.,1.,1.)),cfg['mcc_profile'])
            super().__init__(*args,**kwargs)
            attach_force_profile(self,observer,cfg['mcc_profile'])
            self.residual_policy=ZeroResidualPolicy()
            self.scheduler.scale=torch.as_tensor(cfg['scheduler_scale'],device=self.device,dtype=torch.float64).clamp_min(.02)
            self.runtime.policy.diffusion.num_inference_steps=cfg['inference_steps']
            original=self.runtime.policy.diffusion.conditional_sample
            def sample(batch_size,global_cond=None,generator=None,noise=None):
                if noise is not None:raise ValueError('Explicit noise is not used by the deployment chain')
                before=generator.get_state().clone()
                if probe:
                    reference=original(batch_size,global_cond=global_cond,generator=generator)
                    reference_rng=generator.get_state().clone();generator.set_state(before)
                result=collector.sample(global_cond,generator,self,exploration,record)
                after=generator.get_state()
                if probe:
                    checks['sample_max_abs']=max(checks['sample_max_abs'],float((reference-result).abs().max()))
                    checks['rng_equal']&=bool(torch.equal(reference_rng,after))
                    self.reference_chunk=reference*self.action_std+self.action_mean+self.context['q'][:,None]
                noise_schedule.append(dict(frame=self.control_frame,before=hashlib.sha256(before.numpy().tobytes()).hexdigest(),
                    after=hashlib.sha256(after.numpy().tobytes()).hexdigest()))
                checks['plans']+=1
                return result
            self.runtime.policy.diffusion.conditional_sample=sample
            self.shadow=None
            if probe:
                self.shadow=BatchedBControlChain(*args,**kwargs)
                attach_force_profile(self.shadow,observer,cfg['mcc_profile'])
                self.shadow.scheduler.scale=self.scheduler.scale.clone()
                self.shadow.residual_policy=ZeroResidualPolicy()
                self.shadow.chunk_provider=lambda frame,q,history:self.reference_chunk

        def prime(self,q,repeats=30):
            super().prime(q,repeats)
            if self.shadow is not None:self.shadow.prime(q,repeats)

        def begin_settle(self,q,steps):
            super().begin_settle(q,steps)
            if self.shadow is not None:self.shadow.begin_settle(q,steps)

        def set_training_context(self,**ctx):
            self.context=ctx
            if self.shadow is not None:self.shadow.context=ctx

        def step(self,**kwargs):
            self.context['q']=kwargs['q'];metric.record(self.context,self)
            collector.observe(self)
            action,diagnostic=super().step(**kwargs)
            if not self.context['settling']:
                frame=self.context['frame'];metric.nominal[frame]=diagnostic['nominal_q'];metric.command[frame]=diagnostic['command_q']
                metric.drop[frame]=self.scheduler.last_drop_index
            if self.shadow is not None:
                self.shadow.mesh_oracle=self.mesh_oracle
                for name,value in deployment.orchestration_config().items():setattr(self.shadow,name,value)
                other,extra=self.shadow.step(**kwargs)
                for key,a,b in [('action',action,other),('nominal',diagnostic['nominal_q'],extra['nominal_q']),
                    ('command',diagnostic['command_q'],extra['command_q']),('observation',self.observation.states,self.shadow.observation.states)]:
                    checks[key+'_max_abs']=max(checks[key+'_max_abs'],float((a-b).abs().max()))
            return action,diagnostic
    try:
        meta=deployment.run(len(ids),ids[0],cfg['frames'],str(agent.device),seed,settle_steps=100,
            max_safe_force_n=100,per_env_safety=True,episode_data=data,chain_factory=Chain,contact_observer=observer)
        if meta['completed_frames']!=cfg['frames'] or meta['stop_reason'] is not None:
            raise RuntimeError(f'DPPO rollout chain failed: {meta}')
        if collector.pending.any():raise RuntimeError('Unclosed DP decision transitions')
        report=metric.report(ids,meta,data,noise_schedule)
        report.update(reward_components={k:float(v.cpu()) for k,v in collector.components.items()},
            episode_returns=collector.returns.cpu().tolist(),decisions=sum(len(x['lane']) for x in collector.rows),
            residual='zero',exploration=exploration,probe=checks)
        if probe:
            report['probe']['passed']=checks['rng_equal'] and checks['plans']>0 and all(
                checks[k]<=1e-8 for k in ('sample_max_abs','action_max_abs','nominal_max_abs','command_max_abs','observation_max_abs'))
        return report,collector.rows
    finally:
        deployment.MODEL=old
