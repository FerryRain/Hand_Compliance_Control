"""Fresh PPO residual for frozen DP0 and split whole-fingertip contacts.

No old posture reward or model state is loaded. Four50ms measurement histories,
time-aligned DP future at50/100/200ms, actual execution error, current offset.
The residual enters before MCC; an optional frozen collision guard follows MCC.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import deploy_b_gpu_serial_parity as deployment
from batched_b_control_chain import BatchedBControlChain
from batched_fullhand_mcc import _quat_matrix
from evaluate_dp0_closed_loop import JOB, DP0_MODEL, Metrics, pack, write
from residual_ppo import PPO, GaussianActor
from true_fingertip_contact import ContactSlotObserver, TrueFingertipContact
from true_tip_collision_guard import TrueTipCollisionGuard

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'mcc_finger_compliance_control/data/RL0'
FRAME_FIELDS=[('q',16),('execution_error',16),('DP_target_delta',16),('DP_future_delta_50_100_200ms',48),
              ('contact_point_palm',12),('contact_normal_palm',12),('pad_found',4),('pad_force',4)]


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def state_potential(seam,joint,connector_force):
    # Actual connector loading supplies a warning before100N. Split-plane
    # contact margins alone may stay healthy while the connector also loads.
    return -.3*seam.mean(-1)-.1*joint-.5*(connector_force/20).clamp(0,2).square().mean(-1)


def reward_components(strict, connector, force, seam, joint, task, offset, previous, previous_phi, terminal, completed, cfg,connector_force=None):
    """Per-control-frame costs; potential and terminal are transition terms."""
    contact=.2*(.8*strict.all(-1).float()+.2*strict.float().mean(-1))/5
    if connector_force is None:connector_force=torch.zeros_like(seam)
    phi=state_potential(seam,joint,connector_force)
    next_phi=torch.where(terminal|completed,0.,phi)
    risk=((force-cfg['force_warn_n'])/(100-cfg['force_warn_n'])).clamp(0,2).square()
    values=dict(contact=contact,loss=-.2*(~strict).float().mean(-1)/5,
        connector=-.2*connector.float().mean(-1)/5,force=-4*risk/5,
        posture=-.03*joint/5,task=-.05*task/5,
        potential=cfg['gamma_frame']*next_phi-previous_phi,
        effort=-.01*offset.square().mean(-1)/5,smooth=-.01*(offset-previous).square().mean(-1)/5,
        terminal=torch.where(terminal,-100.,torch.where(completed,20.,0.)))
    values['total']=sum(values.values())
    return values,phi


class RL0PPO(PPO):
    def __init__(self,cfg,bounds):
        super().__init__(cfg,bounds,'cuda:0')
        self.actor=GaussianActor(528,cfg['hidden'],torch.full((16,),.1,device='cuda:0')).to('cuda:0')
        self.actor_optimizer=torch.optim.Adam(self.actor.parameters(),lr=cfg['actor_lr'])

    def finish_rollout(self,value_only=False):
        errors=torch.cat([r['obs'][:,16:32] for r in self.rows])* .5
        result=super().finish_rollout(value_only)
        result['command_error_rms_rad']=float(errors.square().mean().sqrt())
        return result


class RL0Residual(nn.Module):
    def __init__(self,chain,agent,observer,bounds,cfg,*,zero=False,evaluate=False):
        super().__init__()
        self.chain,self.agent,self.observer,self.cfg=chain,agent,observer,cfg
        self.zero,self.evaluate=zero,evaluate
        self.amplitude=torch.tensor(bounds['amplitude_rad'],device=chain.device)
        self.slew=torch.tensor(bounds['decision_slew_rad'],device=chain.device)
        B=chain.batch;self.offset=torch.zeros(B,16,device=chain.device);self.target=self.offset.clone()
        self.previous_offset=self.offset.clone();self.history=None;self.step_index=0;self.latest_chunk=None
        self.geom=TrueFingertipContact(chain.device,fk=chain.mcc.fk)
        self.seam=torch.zeros(B,4,device=chain.device)
        self.pending=torch.zeros(B,dtype=torch.bool,device=chain.device)
        self.elapsed=torch.zeros(B,dtype=torch.long,device=chain.device);self.accumulated=torch.zeros(B,device=chain.device)
        self.previous_obs=None;self.previous_critic=None;self.previous_phi=None;self.sums={}
        self.max_offset=0.;self.return_per_lane=torch.zeros(B,device=chain.device)
        install=chain.scheduler.install
        def install_chunk(chunk,**kwargs):
            self.latest_chunk=chunk.clone()
            return install(chunk,**kwargs)
        chain.scheduler.install=install_chunk

    def observations(self,q,nominal,state,contacts,joint):
        if self.latest_chunk is None:
            future=nominal[:,None].expand(-1,3,-1)
        else:
            chunk=self.latest_chunk;phase=self.chain.scheduler.path_index
            coordinate=(q.new_tensor([5,10,20])+phase)/5-1
            floor=coordinate.floor().long();fraction=(coordinate-floor).clamp(0,1)
            index=(self.chain.scheduler.last_drop_index[:,None]+floor[None]).clamp(0,15)
            left=chunk.gather(1,index[:,:,None].expand(-1,-1,16))
            right=chunk.gather(1,(index+1).clamp_max(15)[:,:,None].expand(-1,-1,16))
            future=left+(right-left)*fraction[None,:,None]
        frame=torch.cat((q,(q-self.context['previous_actuator_target'])/.5,(nominal-q)/.5,
            (future-q[:,None]).flatten(1)/.5,state[:,16:28]*10,state[:,28:40],
            contacts['pad_found'].float(),contacts['pad_force'].norm(dim=-1)/self.cfg['force_warn_n']),-1)
        frame=torch.nan_to_num(frame,nan=0.,posinf=10.,neginf=-10.).clamp(-10,10)
        decision=self.step_index%5==0
        if self.history is None:self.history=frame[:,None].expand(-1,4,-1).clone()
        elif decision:self.history=torch.cat((frame[:,None],self.history[:,:3]),1)
        view=self.history if decision else torch.cat((frame[:,None],self.history[:,:3]),1)
        actor=torch.cat((view.flatten(1),self.offset/self.amplitude),-1)
        private=torch.cat((contacts['connector_force'].norm(dim=-1)/100,self.seam,
            joint[:,None],q.new_full((len(q),1),self.context['frame']/self.cfg['frames'])),-1)
        critic=torch.cat((actor,private),-1)
        if actor.shape[-1]!=528 or critic.shape[-1]!=538:raise AssertionError('RL0 input layout')
        return actor,torch.nan_to_num(critic,nan=0.,posinf=10.,neginf=-10.).clamp(-10,10)

    def flush(self,mask,obs,critic,terminal):
        ids=(mask&self.pending).nonzero().flatten()
        if not len(ids):return
        if not self.evaluate:
            elapsed=self.elapsed[ids]
            discount=self.cfg['gamma_frame']**elapsed
            discount=torch.where(terminal[ids],0.,discount)
            row=dict(lane=ids,obs=self.previous_obs[ids],critic=self.previous_critic[ids],
                z=self.previous_z[ids],logp=self.previous_logp[ids],reward=self.accumulated[ids],
                discount=discount,lam=self.cfg['gae_lambda']**(elapsed.float()/5),
                value=self.agent.value(self.previous_critic[ids]).squeeze(-1),
                next_value=self.agent.value(critic[ids]).squeeze(-1))
            self.agent.rows.append({k:v.detach().cpu().clone() for k,v in row.items()})
            self.agent.transitions+=len(ids)
        self.pending[ids]=False;self.elapsed[ids]=0;self.accumulated[ids]=0

    @torch.no_grad()
    def forward(self,q,nominal,state):
        ctx=self.context;x=self.observer.latest
        strict=(x['pad_found']&torch.isfinite(x['pad_points']).all(-1)&(x['pad_force'].norm(dim=-1)>=.05))
        connector=x['connector_found']&(x['connector_force'].norm(dim=-1)>=.05)
        rotation=_quat_matrix(ctx['palm_pose'])
        points=torch.einsum('bij,bfj->bfi',rotation.transpose(-1,-2),x['pad_points']-ctx['palm_pose'][:,None,:3])
        source=self.geom.source_from_palm(q,points)
        distance=torch.stack([(source[:,i]*self.geom.parts[1 if i==3 else 0]['normal']).sum(-1)-self.geom.parts[1 if i==3 else 0]['offset'] for i in range(4)],-1)
        margin=self.cfg['seam_warn_m']
        cost=((margin-distance)/margin).clamp(0,2).square()
        self.seam=torch.where(strict,cost,self.seam)
        travel=(self.chain.mcc.upper-self.chain.mcc.lower).clamp_min(.01)
        boundary=torch.minimum(q-self.chain.mcc.lower,self.chain.mcc.upper-q)/travel
        joint=((.05-boundary)/.05).clamp(0,2).square().mean(-1)
        fk=self.chain.mcc.fk
        disruption=(fk(nominal+self.offset)-fk(nominal)).norm(dim=-1)
        task=((disruption-.002)/.002).clamp(0,2).square().mean(-1)
        terminal=ctx['newly_stopped']|~torch.isfinite(q).all(-1)
        completed=torch.full_like(terminal,bool(ctx['last_observation']))&ctx['active']
        connector_force=x['connector_force'].norm(dim=-1)
        phi=state_potential(self.seam,joint,connector_force)
        if self.previous_phi is None:self.previous_phi=phi.clone()
        else:
            rewards,_=reward_components(strict,connector,self.observer.safety_peak_force_n,self.seam,joint,task,
                self.offset/self.amplitude,self.previous_offset/self.amplitude,self.previous_phi,terminal,completed,self.cfg,connector_force)
            alive=self.pending
            self.accumulated+=torch.where(alive,self.cfg['gamma_frame']**self.elapsed*rewards['total'],0.)
            self.elapsed+=alive.long();self.return_per_lane+=torch.where(alive,rewards['total'],0.)
            for name,value in rewards.items():self.sums[name]=self.sums.get(name,0.)+torch.where(alive,value,0.).sum()
        self.previous_phi=phi.clone()
        obs,critic=self.observations(q,nominal,state,x,joint)
        self.flush(terminal|completed,obs,critic,terminal|completed)
        if self.step_index%5==0 or bool(ctx['last_observation']):
            self.flush(self.pending,obs,critic,terminal|completed)
            if not bool(ctx['last_observation']):
                action=self.agent.act(obs,zero=self.zero,evaluate=self.evaluate)
                self.target=(self.offset+action*self.slew).clamp(-self.amplitude,self.amplitude)
                self.previous_obs=obs.clone();self.previous_critic=critic.clone()
                self.previous_z=self.agent.last_z.clone();self.previous_logp=self.agent.last_logp.clone()
                self.pending.copy_(ctx['active'])
        self.previous_offset=self.offset.clone()
        self.offset+=(self.target-self.offset).clamp(-self.slew/5,self.slew/5)
        self.offset=self.offset.clamp(-self.amplitude,self.amplitude)
        self.offset=torch.where(ctx['active'][:,None],self.offset,0.)
        self.step_index+=1
        return self.offset.clone()


def rollout(agent,cfg,bounds,ids,seed,*,zero=False,evaluate=False,attach=True,shadow_zero=False):
    observer=ContactSlotObserver(controller_scope='pad',capture_collision=bool(cfg.get('mcc_profile',{}).get('projection')))
    metric=Metrics(observer,len(ids),cfg['frames'],'cuda:0')
    data=pack(ids,cfg['frames']);old=deployment.MODEL;deployment.MODEL=str(DP0_MODEL)
    residual=[];noise=[];shadow_error=dict(action=0.,nominal=0.,command=0.,observation=0.)
    class Chain(BatchedBControlChain):
        def __init__(self,*args,**kwargs):
            from dataclasses import replace
            if cfg['contact_gate']!='legacy':kwargs['mcc_config']=replace(kwargs['mcc_config'],back_contact_x_limit_m=(1.,1.,1.,1.))
            if cfg.get('mcc_profile'):
                from mcc_force_profiles import configured_mcc
                kwargs['mcc_config']=configured_mcc(kwargs['mcc_config'],cfg['mcc_profile'])
            super().__init__(*args,**kwargs)
            if cfg.get('mcc_profile'):
                from mcc_force_profiles import attach_force_profile
                attach_force_profile(self,observer,cfg['mcc_profile'])
            self.runtime.policy.diffusion.num_inference_steps=cfg['inference_steps']
            sample=self.runtime.policy.diffusion.conditional_sample
            def recorded_sample(*args,**kwargs):
                generator=kwargs['generator']
                before=hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest()
                result=sample(*args,**kwargs)
                after=hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest()
                noise.append(dict(frame=self.control_frame,before=before,after=after))
                return result
            self.runtime.policy.diffusion.conditional_sample=recorded_sample
            self.guard=TrueTipCollisionGuard(self.mcc) if cfg['contact_gate']=='whole_tip_guard' else None
            if attach:
                self.residual_policy=RL0Residual(self,agent,observer,bounds,cfg,zero=zero,evaluate=evaluate)
                residual.append(self.residual_policy)
            self.shadow=None
            if shadow_zero:
                if not (attach and zero and evaluate):raise ValueError('Shadow test requires a zero evaluation')
                self.shadow=BatchedBControlChain(*args,**kwargs)
                if cfg.get('mcc_profile'):
                    attach_force_profile(self.shadow,observer,cfg['mcc_profile'])
                self.shadow.runtime.policy.diffusion.num_inference_steps=cfg['inference_steps']
                self.shadow.chunk_provider=lambda frame,q,history:self.residual_policy.latest_chunk
        def prime(self,q,repeats=30):
            super().prime(q,repeats)
            if self.shadow is not None:self.shadow.prime(q,repeats)
        def begin_settle(self,q,steps):
            super().begin_settle(q,steps)
            if self.shadow is not None:self.shadow.begin_settle(q,steps)
        def set_training_context(self,**ctx):
            self.context=ctx
            if self.shadow is not None:self.shadow.context=ctx
            if attach:self.residual_policy.context=ctx
        def step(self,**kwargs):
            self.context['q']=kwargs['q'];metric.record(self.context,self)
            action,diagnostic=super().step(**kwargs)
            if self.shadow is not None:
                from deploy_b_gpu_serial_parity import orchestration_config
                self.shadow.mesh_oracle=self.mesh_oracle
                for name,value in orchestration_config().items():setattr(self.shadow,name,value)
                shadow_action,shadow_diagnostic=self.shadow.step(**kwargs)
                for name,a,b in [('action',action,shadow_action),('nominal',diagnostic['nominal_q'],shadow_diagnostic['nominal_q']),
                                 ('command',diagnostic['command_q'],shadow_diagnostic['command_q']),
                                 ('observation',self.observation.states,self.shadow.observation.states)]:
                    difference=float((a-b).abs().max())
                    shadow_error[name]=max(shadow_error[name],difference)
                    if difference>1e-7:raise RuntimeError(f'Zero interface changed {name} at frame{self.control_frame-1}: {difference}')
            if self.guard is not None and not self.context['settling']:
                command=self.guard(kwargs['q'],diagnostic['command_q'],kwargs['palm_pose_world'],kwargs['palm_twist_world'],observer.latest)
                diagnostic['command_q']=command;action=((command-kwargs['q'])/.08).clamp(-2,2)
            if not self.context['settling']:
                frame=self.context['frame'];metric.command[frame]=diagnostic['command_q'];metric.nominal[frame]=diagnostic['nominal_q']
            return action,diagnostic
    try:
        meta=deployment.run(len(ids),ids[0],cfg['frames'],'cuda:0',seed,settle_steps=100,max_safe_force_n=100,
            per_env_safety=True,episode_data=data,chain_factory=Chain,contact_observer=observer)
        if meta['stop_reason'] is not None or meta['completed_frames']!=cfg['frames']:raise RuntimeError(meta)
        result=metric.report(ids,meta,data,noise)
        result['reward_components']={k:float(v.cpu()) for k,v in residual[0].sums.items()} if residual else {}
        result['episode_returns']=residual[0].return_per_lane.cpu().tolist() if residual else []
        result['mean_strict']=float(np.mean([r['post_bootstrap']['strict_healthy_four_rate'] for r in result['lanes']]))
        result['mean_found']=float(np.mean([r['post_bootstrap']['found_four_rate'] for r in result['lanes']]))
        result['safety_stops']=sum(r['safety_stop_frame']>=0 for r in result['lanes'])
        if shadow_zero:result['shadow_zero_equivalence']=dict(max_errors=shadow_error,passed=all(v<=1e-7 for v in shadow_error.values()),frames=cfg['frames'],settle_frames=100,
            scope='Same physical observations and decoded chunks; compare scheduler/MCC/action and full96-D observation history each frame')
        if attach and not evaluate:result['ppo']=agent.finish_rollout(value_only=zero)
        return result,metric
    finally:
        deployment.MODEL=old
        residual.clear()
        import gc
        gc.collect();torch.cuda.empty_cache()


def save(path,agent,cfg,bounds,rollouts,rng):
    state=dict(actor=agent.actor.state_dict(),value=agent.value.state_dict(),actor_optimizer=agent.actor_optimizer.state_dict(),
        value_optimizer=agent.value_optimizer.state_dict(),config=cfg,bounds=bounds,transitions=agent.transitions,
        updates=agent.updates,actor_updates=agent.actor_updates,generator=agent.generator.get_state(),
        episode_rng=rng.bit_generator.state,rollouts=rollouts)
    torch.save(state,path.with_suffix('.tmp'));path.with_suffix('.tmp').replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=OUT/'ppo_DP0')
    p.add_argument('--contact-gate',choices=['legacy','whole_tip','whole_tip_guard'],required=True)
    p.add_argument('--num-envs',type=int,default=64);p.add_argument('--transitions',type=int,default=500000)
    p.add_argument('--seed',type=int,default=20261010);p.add_argument('--integration-only',action='store_true')
    p.add_argument('--inference-steps',type=int,default=20)
    p.add_argument('--mcc-profile',type=Path,help='Frozen selected MCC profile JSON')
    p.add_argument('--resume',type=Path,help='Resume a complete rollout checkpoint in its existing experiment directory')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'manifest.json').exists() and a.resume is None:raise ValueError('Fresh output required; do not overwrite training')
    reference=json.loads((OUT/'legacy_reference.json').read_text());bounds=reference['bounds']
    calibration=json.loads((JOB/'RL0_calibration_source.json').read_text())
    split=json.loads((JOB/'DP0_split.json').read_text());train=[r['episode_id'] for r in split['episodes'] if r['split']=='train']
    dev=[r['episode_id'] for r in split['episodes'] if r['split']=='val'];test=[r['episode_id'] for r in split['episodes'] if r['split']=='test']
    cfg=dict(history=4,actor_frame_dim=128,actor_input_dim=528,critic_input_dim=538,hidden=256,
        actor_lr=2.5e-5,value_lr=3e-4,epochs=4,minibatch=4096,clip=.1,target_kl=.01,gae_lambda=.97,
        gamma_frame=.997**.2,seed=a.seed,initial_normalized_action_std=.1,frames=1500,
        force_warn_n=max(40.,calibration['force_quantiles'][2]),seam_warn_m=float(calibration['split_margin_quantiles_m'][0]),
        contact_gate=a.contact_gate,inference_steps=a.inference_steps,num_envs=a.num_envs,total_transitions=a.transitions)
    if a.mcc_profile is not None:cfg['mcc_profile']=json.loads(a.mcc_profile.read_text())
    manifest=dict(config=cfg,bounds=bounds,DP0=str(DP0_MODEL),DP0_sha256=digest(DP0_MODEL),
        split=dict(train=train,dev=dev,test=test),initialization='Fresh actor, critic and optimizers; zero actor mean head; no old weights',
        actor_fields=FRAME_FIELDS,history='current,50ms,100ms,150ms; current applied offset16 appended once',
        future='Linear interpolation of installed chunk at fixed50/100/200ms after the current scheduler phase',
        critic_extra='connector force4, held split-plane margin cost4, joint-limit cost1, palm phase1',
        reward='whole true-tip contact + immediate loss + connector contact + force risk + joint boundary + FK intent tube + split-margin/connector-load potential + offset effort/smoothness + terminal',
        timing='5frames/decision, offset ramp by decision_slew/5 every control frame, before MCC;100N stop unchanged',
        calibration=calibration,code_sha256={n:digest(Path(__file__).with_name(n)) for n in ['train_rl0.py','true_tip_collision_guard.py','batched_b_control_chain.py','true_fingertip_contact.py']})
    if a.resume is None:write(a.output/'manifest.json',manifest)
    import shutil
    snapshot=a.output/('sources' if a.resume is None else 'resume_sources');snapshot.mkdir(exist_ok=a.resume is not None)
    for name in ['train_rl0.py','residual_ppo.py','true_tip_collision_guard.py','batched_b_control_chain.py',
                 'batched_fullhand_mcc.py','true_fingertip_contact.py','deploy_b_gpu_serial_parity.py']:
        shutil.copy2(Path(__file__).with_name(name),snapshot/name)
    if cfg.get('mcc_profile'):
        for name in ['mcc_force_profiles.py','paired_contact_projection.py']:
            shutil.copy2(Path(__file__).with_name(name),snapshot/name)
    torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    agent=RL0PPO(cfg,bounds);rng=np.random.default_rng(a.seed);started=time.monotonic()
    if a.resume is not None:
        state=torch.load(a.resume,map_location='cpu',weights_only=False)
        if state['config']!=cfg or state['bounds']!=bounds:raise ValueError('Resume configuration/bounds changed')
        prior=json.loads((a.output/'manifest.json').read_text())
        if digest(DP0_MODEL)!=prior['DP0_sha256']:raise ValueError('Frozen DP0 checkpoint changed')
        agent.actor.load_state_dict(state['actor']);agent.value.load_state_dict(state['value'])
        agent.actor_optimizer.load_state_dict(state['actor_optimizer']);agent.value_optimizer.load_state_dict(state['value_optimizer'])
        agent.generator.set_state(state['generator'].cpu());rng.bit_generator.state=state['episode_rng']
        agent.transitions=state['transitions'];agent.updates=state['updates'];agent.actor_updates=state['actor_updates']
        rollouts=state['rollouts'];origin=torch.load(a.output/'initial.pt',map_location='cpu',weights_only=False)['transitions']
        baseline=json.loads((a.output/'dev_baseline.json').read_text())
        best_state=torch.load(a.output/'best.pt',map_location='cpu',weights_only=False)
        best_step=best_state['transitions']-origin;best=baseline['mean_strict'];bad=0
        for f in sorted(a.output.glob('dev_[0-9]*.json'),key=lambda p:int(p.stem.split('_')[1])):
            e=json.loads(f.read_text());harm=e['mean_strict']<baseline['mean_strict']-.03 or e['safety_stops']>baseline['safety_stops']+2 or bool(e.get('healthy_baseline_regressions'))
            bad=bad+1 if harm else 0
            if not harm:best=max(best,e['mean_strict'])
        next_eval=(int((agent.transitions-origin)/150000)+1)*150000
        dev_ids=[ep for ep in dev[:12] for _ in range(2)]
        write(a.output/'resume_record.json',dict(checkpoint=str(a.resume),transitions=agent.transitions-origin,rollouts=rollouts,
            scope='Full rollout boundary; actor/value/optimizers/action RNG/episode RNG restored; incomplete rollout discarded; no mid-trajectory state reset'))
        print('[RL0-RESUME]',agent.transitions-origin,flush=True)
    else:
        ids=[4,103,105,136]
        zero,_=rollout(agent,cfg,bounds,ids,a.seed+100,zero=True,evaluate=True,shadow_zero=True)
        eq=zero['shadow_zero_equivalence']
        eq['independent_repeat_caveat']='Independent long rollouts can diverge under MJWarp contact reductions; previous maximum-q gate failed and remains preserved in the first attempt folder'
        write(a.output/'zero_equivalence.json',eq)
        if not eq['passed']:raise RuntimeError('RL0 zero-residual equivalence failed; training not started')
        dev_ids=[ep for ep in dev[:12] for _ in range(2)]
        baseline,_=rollout(agent,cfg,bounds,dev_ids,a.seed+200,zero=True,evaluate=True)
        write(a.output/'dev_baseline.json',baseline)
        ids=rng.choice(train,a.num_envs,replace=True).tolist()
        warmup,_=rollout(agent,cfg,bounds,ids,a.seed+1000,zero=True);write(a.output/'value_warmup.json',warmup)
        origin=agent.transitions;best=baseline['mean_strict'];best_step=0;bad=0;next_eval=150000;rollouts=0
        save(a.output/'initial.pt',agent,cfg,bounds,rollouts,rng)
        save(a.output/'best.pt',agent,cfg,bounds,rollouts,rng)
    while agent.transitions-origin<a.transitions:
        ids=rng.choice(train,a.num_envs,replace=True).tolist()
        result,_=rollout(agent,cfg,bounds,ids,a.seed+2000+rollouts)
        rollouts+=1;new=agent.transitions-origin
        record=dict(rollout=rollouts,training_transitions=new,strict=result['mean_strict'],found=result['mean_found'],
            safety_stops=result['safety_stops'],episode_return_mean=float(np.mean(result['episode_returns'])),
            reward_components=result['reward_components'],losses=result['ppo'],elapsed_s=time.monotonic()-started)
        with (a.output/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        save(a.output/'last.pt',agent,cfg,bounds,rollouts,rng);write(a.output/'progress.json',record)
        print('[RL0]',json.dumps(record),flush=True)
        if a.integration_only:break
        if new>=next_eval or new>=a.transitions:
            evaluation,_=rollout(agent,cfg,bounds,dev_ids,a.seed+200,evaluate=True)
            if evaluation['noise_schedule']!=baseline['noise_schedule']:raise RuntimeError('Paired DP noise schedule mismatch')
            health_damage=[];differences={}
            for ep in dev[:12]:
                b=[r for r in baseline['lanes'] if r['episode_id']==ep]
                e=[r for r in evaluation['lanes'] if r['episode_id']==ep]
                change=float(np.mean([r['post_bootstrap']['strict_healthy_four_rate'] for r in e])-np.mean([r['post_bootstrap']['strict_healthy_four_rate'] for r in b]))
                differences[ep]=change
                healthy=all(r['safety_stop_frame']<0 and r['post_bootstrap']['strict_healthy_four_rate']>=.98 for r in b)
                if healthy and (change<-.03 or any(r['safety_stop_frame']>=0 for r in e)):health_damage.append(ep)
            evaluation['paired_strict_by_episode']=differences
            evaluation['healthy_baseline_regressions']=health_damage
            evaluation['common_diffusion_noise_verified']=True
            write(a.output/f'dev_{new}.json',evaluation)
            harm=evaluation['mean_strict']<baseline['mean_strict']-.03 or evaluation['safety_stops']>baseline['safety_stops']+2 or bool(health_damage)
            bad=bad+1 if harm else 0
            if not harm and evaluation['mean_strict']>best:
                best=evaluation['mean_strict'];best_step=new;save(a.output/'best.pt',agent,cfg,bounds,rollouts,rng)
            next_eval+=150000
            if bad>=2:break
    write(a.output/'summary.json',dict(training_transitions=agent.transitions-origin,rollouts=rollouts,
        actor_updates=agent.actor_updates,best_dev_strict=best,best_training_transitions=best_step,baseline_dev_strict=baseline['mean_strict'],
        stopped_for_regression=bad>=2,finite_parameters=all(torch.isfinite(p).all().item() for p in agent.actor.parameters()),
        checkpoint=str(a.output/'last.pt'),elapsed_s=time.monotonic()-started,
        scope='RL0 canary on train155/dev33; diagnostic test33 not used for RL updates or checkpoint selection'))


if __name__=='__main__':main()
