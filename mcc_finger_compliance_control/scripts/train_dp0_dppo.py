"""DPPO canary with original DP0, zero residual and accepted paired MCC.

Each rollout/update and development batch owns a separate CUDA process.
Default budget: one critic warm-up followed by two actor training rollouts.
Train155 / development33 / test33 stay unchanged. No mid-trajectory reset.
"""
import argparse
from copy import deepcopy
import hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
import evaluate_dp0_closed_loop as evaluation
from train_dp import build_policy
from dppo_diffusion import DPPOSampler,surrogate_logprob,variable_gae
from dppo_control_env import collect_rollout

ROOT=Path(__file__).resolve().parents[2]
PROFILE=ROOT/'mcc_finger_compliance_control/data/contact_projection_full_development_20261011/selected_mcc_profile.json'
BASELINE=ROOT/'mcc_finger_compliance_control/data/dp_geometry_transfer_20261012/evaluation/DP0_initial'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def cpu_tree(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
    return value


class DPPOAgent:
    def __init__(self,cfg,resume=None):
        self.cfg=cfg;self.device=torch.device(cfg['device'])
        torch.manual_seed(cfg['seed'])
        if self.device.type=='cuda':torch.cuda.manual_seed_all(cfg['seed'])
        self.origin=torch.load(cfg['checkpoint'],map_location='cpu',weights_only=False,mmap=True)
        if 'dppo' in self.origin:raise ValueError('Start this experiment from original supervised DP0')
        args=SimpleNamespace(**self.origin['config']);args.inference_steps=cfg['inference_steps']
        self.policy=build_policy(args,self.device);self.policy.load_state_dict(self.origin['model']);self.policy.eval()
        self.base_unet=deepcopy(self.policy.diffusion.unet)
        for p in self.base_unet.parameters():p.requires_grad_(False)
        self.critic=nn.Sequential(nn.Linear(1545,256),nn.Tanh(),nn.Linear(256,256),nn.Tanh(),nn.Linear(256,1)).to(self.device)
        self.actor_optimizer=torch.optim.AdamW(self.policy.diffusion.unet.parameters(),lr=cfg['actor_lr'],weight_decay=0.)
        self.value_optimizer=torch.optim.Adam(self.critic.parameters(),lr=cfg['value_lr'])
        self.sampler=DPPOSampler(self.policy.diffusion,self.base_unet,cfg['ft_steps'],cfg['min_std'])
        self.actor_updates=0;self.critic_updates=0
        self.update_rng=torch.Generator(device='cpu').manual_seed(cfg['seed']+700)
        if resume is not None:
            state=torch.load(resume,map_location='cpu',weights_only=False,mmap=True)
            if state['config']!=cfg:raise ValueError('Resume config changed')
            self.policy.load_state_dict(state['model']);self.critic.load_state_dict(state['critic'])
            self.actor_optimizer.load_state_dict(state['actor_optimizer']);self.value_optimizer.load_state_dict(state['value_optimizer'])
            self.actor_updates=state['actor_updates'];self.critic_updates=state['critic_updates']
            self.update_rng.set_state(state['update_rng'])

    def update(self,rows,value_only=False):
        if not rows:raise RuntimeError('No DPPO decisions collected')
        data={k:torch.cat([r[k] for r in rows]) for k in rows[0]}
        advantage,returns=variable_gae(data['reward'],data['value'],data['next_value'],
            data['discount'],data['duration'],data['lane'],self.cfg['gae_lambda'])
        advantage=(advantage-advantage.mean())/advantage.std(unbiased=False).clamp_min(1e-8)
        n=len(returns);losses=[];actor_rows=[];stopped=False
        for _ in range(self.cfg['epochs']):
            order=torch.randperm(n,generator=self.update_rng)
            for indices in order.split(256):
                predicted=self.critic(data['critic'][indices].to(self.device)).squeeze(-1)
                value_loss=.5*(predicted-returns[indices].to(self.device)).square().mean()
                self.value_optimizer.zero_grad(set_to_none=True);value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(),1.)
                self.value_optimizer.step();self.critic_updates+=1
                losses.append(float(value_loss.detach()))
        self.sampler.prepare(self.device)
        k=self.cfg['ft_steps'];total=n*k
        if not value_only:
            for _ in range(self.cfg['epochs']):
                order=torch.randperm(total,generator=self.update_rng)
                for indices in order.split(self.cfg['batch_size']):
                    self.actor_optimizer.zero_grad(set_to_none=True);stats=[]
                    for micro in indices.split(self.cfg['microbatch']):
                        decision=micro//k;step=micro%k
                        coordinates=self.sampler.logprob(data['condition'][decision].to(self.device),
                            data['before'][decision,step].to(self.device),data['after'][decision,step].to(self.device),
                            data['timesteps'][decision,step].to(self.device))
                        new=surrogate_logprob(coordinates,self.cfg['reward_horizon'])
                        old=surrogate_logprob(data['logprob'][decision,step].to(self.device),self.cfg['reward_horizon'])
                        logratio=new-old;ratio=logratio.exp()
                        adv=advantage[decision].to(self.device)*self.cfg['gamma_denoising']**(k-step.to(self.device)-1)
                        phase=step.to(self.device).float()/max(1,k-1)
                        clip=self.cfg['clip_base']+(self.cfg['clip']-self.cfg['clip_base'])*(torch.exp(3*phase)-1)/(np.exp(3)-1)
                        loss=torch.maximum(-adv*ratio,-adv*ratio.clamp(1-clip,1+clip)).mean()
                        (loss*len(micro)/len(indices)).backward()
                        with torch.no_grad():
                            stats.append([len(micro),float(loss),float(((ratio-1)-logratio).mean()),float(((ratio-1).abs()>clip).float().mean())])
                    if not all(torch.isfinite(p.grad).all() for p in self.policy.diffusion.unet.parameters() if p.grad is not None):
                        raise RuntimeError('Non-finite DPPO gradient')
                    norm=nn.utils.clip_grad_norm_(self.policy.diffusion.unet.parameters(),1.)
                    mean=np.average(np.array(stats)[:,1:],axis=0,weights=np.array(stats)[:,0])
                    if mean[1]>self.cfg['target_kl']:
                        self.actor_optimizer.zero_grad(set_to_none=True);stopped=True;break
                    self.actor_optimizer.step();self.actor_updates+=1
                    actor_rows.append([*mean.tolist(),float(norm)])
                if stopped:break
        changed=sum(not torch.equal(a,b) for a,b in zip(self.policy.diffusion.unet.parameters(),self.base_unet.parameters()))
        return dict(decisions=n,denoising_transitions=total,critic_loss=float(np.mean(losses)),
            actor_metrics_mean=np.mean(actor_rows,axis=0).tolist() if actor_rows else None,
            actor_metric_order=['policy_loss','approx_kl_coordinate_mean','clip_fraction','gradient_norm'],
            actor_updates=self.actor_updates,critic_updates=self.critic_updates,kl_early_stop=stopped,
            actor_tensors_changed=changed,finite_parameters=all(torch.isfinite(p).all().item() for p in self.policy.parameters()),
            discounted_interval_return_mean=float(data['reward'].mean()),
            decision_duration_min=int(data['duration'].min()),decision_duration_max=int(data['duration'].max()),
            advantage_std=float(advantage.std(unbiased=False)))

    @torch.no_grad()
    def likelihood_check(self,rows):
        if not rows:return dict(passed=False,reason='No exploration transitions')
        data={key:torch.cat([r[key] for r in rows])[:8] for key in ('condition','before','after','timesteps','logprob')}
        self.sampler.prepare(self.device);error=0.
        for j in range(self.cfg['ft_steps']):
            lp=self.sampler.logprob(data['condition'].to(self.device),data['before'][:,j].to(self.device),
                data['after'][:,j].to(self.device),data['timesteps'][:,j].to(self.device))
            error=max(error,float((lp-data['logprob'][:,j].to(self.device)).abs().max()))
        return dict(passed=error<=2e-3,max_coordinate_logprob_error=error,tolerance=.002,
            exact_law='Untruncated Gaussian; same sampling/logprob std; final unclipped latent recorded')

    def save(self,folder):
        model=cpu_tree(self.policy.state_dict())
        state=dict(model=model,critic=cpu_tree(self.critic.state_dict()),
            actor_optimizer=cpu_tree(self.actor_optimizer.state_dict()),value_optimizer=cpu_tree(self.value_optimizer.state_dict()),
            config=self.cfg,actor_updates=self.actor_updates,critic_updates=self.critic_updates,update_rng=self.update_rng.get_state())
        temp=folder/'learner.tmp';torch.save(state,temp);temp.replace(folder/'learner.pt')
        exported={k:v for k,v in self.origin.items() if k not in ('model','optimizer','optimizer_state_dict')}
        exported['model']=model
        exported['dppo']=dict(inference_steps=self.cfg['inference_steps'],ft_steps=self.cfg['ft_steps'],
            min_std=self.cfg['min_std'],sampling='native DDPM evaluation / Gaussian floor exploration',residual='zero')
        exported['dppo_base_unet']=cpu_tree(self.base_unet.state_dict())
        exported['config']=dict(exported['config'],inference_steps=self.cfg['inference_steps'])
        exported['dppo_training']=dict(actor_updates=self.actor_updates,critic_updates=self.critic_updates)
        temp=folder/'candidate.tmp';torch.save(exported,temp);temp.replace(folder/'candidate.pt')


def worker(request):
    x=json.loads(Path(request).read_text());cfg=x['config'];torch.set_num_threads(1)
    if x.get('frozen_sources'):
        expected=json.loads(Path(x['frozen_sources']).read_text())
        changed=[name for name,sha in expected.items() if digest(ROOT/name)!=sha]
        if changed:raise RuntimeError(f'Frozen DPPO execution sources changed: {changed}')
    agent=DPPOAgent(cfg,x.get('resume'));start=time.monotonic()
    mode=x['mode']
    if mode=='train':
        # cuDNN can select TF32 for large collection batches and FP32 for
        # small likelihood/gradient batches. The resulting means differ enough
        # to perturb low-variance DDPM densities. Use full FP32 consistently
        # throughout collection and updates; keep the native inference backend
        # unchanged for paired development evaluation.
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cuda.matmul.allow_tf32=False
    report,rows=collect_rollout(agent,cfg,x['ids'],x['seed'],
        exploration=mode=='train',record=mode=='train',probe=mode=='probe',
        deployment_checkpoint=x.get('deployment_checkpoint'))
    if x.get('resume'):
        actor_state=next(iter(agent.actor_optimizer.state.values()))
        report['resume_state']=dict(source=x['resume'],actor_updates_before=agent.actor_updates,
            critic_updates_before=agent.critic_updates,actor_optimizer_entries=len(agent.actor_optimizer.state),
            critic_optimizer_entries=len(agent.value_optimizer.state),
            first_actor_optimizer_step=float(actor_state['step']))
    if mode=='train':
        likelihood=agent.likelihood_check(rows);report['likelihood']=likelihood
        if not likelihood['passed']:
            torch.save(rows[0],Path(x['folder'])/'likelihood_failure.pt')
            raise RuntimeError(f'Old logprob check failed: {likelihood}')
        report['update']=agent.update(rows,value_only=x.get('value_only',False));agent.save(Path(x['folder']))
    if mode=='probe' and not report['probe']['passed']:raise RuntimeError(f'Native parity failed: {report["probe"]}')
    report['elapsed_s']=time.monotonic()-start
    report['cuda_peak_allocated_mib']=torch.cuda.max_memory_allocated()/2**20
    report['cuda_peak_reserved_mib']=torch.cuda.max_memory_reserved()/2**20
    report['dp_decisions_per_s']=report['decisions']/report['elapsed_s']
    report['training_math']='full FP32; TF32 disabled' if mode=='train' else 'native inference backend'
    evaluation.write(Path(x['output']),report)
    print('[DPPO-WORKER]',mode,'decisions',report['decisions'],'seconds',report['elapsed_s'],report.get('update'),flush=True)


def run_worker(folder,name,request):
    output=folder/f'{name}.json';request['output']=str(output)
    req=folder/f'{name}.request.json';evaluation.write(req,request)
    with (folder/f'{name}.log').open('w') as log:
        child=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',str(req)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    if child.returncode:raise RuntimeError(f'Worker failed; inspect {folder/name}.log')
    return json.loads(output.read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'mcc_finger_compliance_control/data/DPPO/DP0_zero_residual_canary_20261012')
    parser.add_argument('--num-envs',type=int,default=8);parser.add_argument('--iterations',type=int,default=3)
    parser.add_argument('--warmup-iterations',type=int,default=1)
    parser.add_argument('--microbatch',type=int,default=4);parser.add_argument('--batch-size',type=int,default=64)
    parser.add_argument('--seed',type=int,default=20261018);parser.add_argument('--probe-only',action='store_true')
    a=parser.parse_args();folder=a.output.resolve();folder.mkdir(parents=True,exist_ok=True)
    if (folder/'manifest.json').exists():raise ValueError('Fresh DPPO directory required')
    if a.iterations<=a.warmup_iterations and not a.probe_only:raise ValueError('Budget must include actor training')
    if not json.loads((PROFILE.parent/'acceptance.json').read_text())['passed']:raise ValueError('MCC not accepted')
    profile=json.loads(PROFILE.read_text());calibration=json.loads((evaluation.JOB/'RL0_calibration_source.json').read_text())
    split=json.loads((evaluation.JOB/'DP0_split.json').read_text());train=[r['episode_id'] for r in split['episodes'] if r['split']=='train']
    cfg=dict(checkpoint=str(evaluation.DP0_MODEL),device='cuda:0',frames=1500,trajectory_frames=1500,
        inference_steps=20,ft_steps=3,min_std=.02,reward_horizon=8,actor_lr=1e-6,value_lr=3e-4,
        epochs=2,microbatch=a.microbatch,batch_size=a.batch_size,clip=.02,clip_base=.001,target_kl=.005,
        gamma_frame=.997**.2,gae_lambda=.95,gamma_denoising=.99,
        force_warn_n=40.,seam_warn_m=float(calibration['split_margin_quantiles_m'][0]),
        mcc_profile=profile,scheduler_scale=evaluation.scheduler_scale().tolist(),seed=a.seed)
    manifest=dict(config=cfg,iterations=a.iterations,warmup_iterations=a.warmup_iterations,num_envs=a.num_envs,
        DP0_sha256=digest(evaluation.DP0_MODEL),profile_sha256=digest(PROFILE),split=split,
        baseline=str(BASELINE),reference='https://github.com/irom-princeton/dppo',
        actor_observation='Unchanged96x16; no residual or privileged actor features',
        critic='Normalized actor history1536 + pad netforce4/100 + connector netforce4/100 + palm phase1',
        reward='RL0 contact/loss/connector/force/joint/potential/terminal; residual effort/smooth/FK terms removed',
        reward_timing='10ms rewards accumulated into actual DP intervals; planned end terminal; short horizon bootstrap',
        likelihood='Raw Gaussian coordinate densities; official clipped coordinate-mean PPO surrogate on first8 waypoints',
        sampling='20 original DDPM steps; first17 frozen base, last3 trained; evaluation uses native std/RNG',
        limits='Canary checks interfaces/updates; full-development safety and health preservation required')
    evaluation.write(folder/'manifest.json',manifest)
    sources=folder/'sources';sources.mkdir()
    for name in ['dppo_diffusion.py','dppo_control_env.py','train_dp0_dppo.py','deploy_dp_inverse.py',
        'batched_b_control_chain.py','batched_fullhand_mcc.py','deploy_b_gpu_serial_parity.py',
        'paired_contact_projection.py','mcc_force_profiles.py','true_fingertip_contact.py']:
        shutil.copy2(Path(__file__).with_name(name),sources/name)
    def status(phase,**kw):
        evaluation.write(folder/'status.json',dict(phase=phase,pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw))
        print('[DPPO]',phase,kw,flush=True)
    try:
        status('native_equivalence')
        probe=run_worker(folder,'native_equivalence',dict(mode='probe',config=cfg,ids=[4,103],seed=a.seed+100))
        if a.probe_only:status('probe_complete',probe=probe['probe']);return
        rng=np.random.default_rng(a.seed);reports=[]
        for i in range(a.iterations):
            status('training',iteration=i,budget=a.iterations,critic_only=i<a.warmup_iterations)
            report=run_worker(folder,f'train_{i:02d}',dict(mode='train',config=cfg,
                ids=rng.choice(train,a.num_envs,replace=True).tolist(),seed=a.seed+1000+i,
                value_only=i<a.warmup_iterations,folder=str(folder),
                **({'resume':str(folder/'learner.pt')} if i else {})))
            reports.append(report)
            evaluation.write(folder/'training_progress.json',dict(iteration=i,update=report['update'],
                strict=float(np.mean([r['post_bootstrap']['strict_healthy_four_rate'] for r in report['lanes']])),
                found=float(np.mean([r['post_bootstrap']['found_four_rate'] for r in report['lanes']])),
                safety_stops=sum(r['safety_stop_frame']>=0 for r in report['lanes']),reward_components=report['reward_components'],
                cuda_peak_mib=report['cuda_peak_allocated_mib'],elapsed_s=report['elapsed_s']))
        status('development_evaluation')
        baseline=json.loads((BASELINE/'summary.json').read_text());plan=json.loads((BASELINE/'plan.json').read_text())
        candidate=folder/'evaluation';candidate.mkdir()
        for b in plan['batches']:
            status('development_evaluation',batch=b['index'],lanes=len(b['ids']))
            request=dict(checkpoint=str(folder/'candidate.pt'),ids=b['ids'],seed=b['seed'],
                output=str(candidate/f"batch_{b['index']:02d}_DP.json"),profile=profile,scheduler_scale=cfg['scheduler_scale'])
            req=candidate/f"batch_{b['index']:02d}.request.json";evaluation.write(req,request)
            with (candidate/f"batch_{b['index']:02d}.log").open('w') as log:
                child=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--eval-worker',str(req)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if child.returncode:raise RuntimeError(f'Development evaluation failed: batch{b["index"]}')
            old=json.loads((BASELINE/f"batch_{b['index']:02d}_DP.json").read_text());new=json.loads(Path(request['output']).read_text())
            if old['input_sha256']!=new['input_sha256'] or old['noise_schedule']!=new['noise_schedule']:
                raise RuntimeError('Paired inputs/diffusion-noise schedules changed')
        result=evaluation.summarize(candidate,plan)
        result['scope']='DPPO canary; full33 development paths; zero residual; not locked final test'
        evaluation.write(candidate/'summary.json',result)
        from run_mcc_then_residual import gate
        verdict=gate(baseline,result,False)
        verdict['passed']=verdict['passed'] and verdict['candidate']['strict']>verdict['baseline']['strict']
        evaluation.write(folder/'acceptance.json',verdict)
        summary=dict(native_equivalence=probe['probe'],iterations=len(reports),total_dp_decisions=sum(r['decisions'] for r in reports),
            last_update=reports[-1]['update'],baseline=baseline['aggregate']['post_bootstrap'],
            candidate=result['aggregate']['post_bootstrap'],acceptance=verdict,checkpoint=str(folder/'candidate.pt'),
            production_default_changed=False,next='Review development improvement/health gate before increasing budget')
        evaluation.write(folder/'summary.json',summary);status('complete',acceptance=verdict)
    except BaseException as error:status('failed',error=str(error));raise


def eval_worker(request):
    x=json.loads(Path(request).read_text())
    if x.get('frozen_sources'):
        expected=json.loads(Path(x['frozen_sources']).read_text())
        changed=[name for name,sha in expected.items() if digest(ROOT/name)!=sha]
        if changed:raise RuntimeError(f'Frozen DPPO evaluation sources changed: {changed}')
    evaluation.DP0_MODEL=Path(x['checkpoint'])
    report=evaluation.evaluate(x['ids'],1500,x['seed'],'cuda:0',contact_gate='whole_tip',
        inference_steps=20,mcc_profile=x['profile'],scheduler_scale_override=x['scheduler_scale'])
    evaluation.write(Path(x['output']),report)


if __name__=='__main__':
    if len(sys.argv)==3 and sys.argv[1]=='--worker':worker(sys.argv[2])
    elif len(sys.argv)==3 and sys.argv[1]=='--eval-worker':eval_worker(sys.argv[2])
    else:main()
