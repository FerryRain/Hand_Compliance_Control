"""DPPO adapter for the existing LeRobot DDPM U-Net.

Reference: https://github.com/irom-princeton/dppo (diffusion_vpg/ppo).
Keep the base network frozen at early denoising steps, update a copy at the
last few steps, and use the official clipped, coordinate-averaged PPO
surrogate. This average is not the joint Gaussian log density; retain raw
coordinate densities separately for likelihood verification.

Native mode calls the original scheduler.step unchanged. Exploration mode
uses an untruncated Gaussian with the same std for sampling and likelihood.
The final raw latent is recorded before the deterministic action clip.
"""
from copy import deepcopy
import math
import torch


def gaussian_logprob(value, mean, std):
    return -.5*((value-mean)/std).square()-std.log()-.5*math.log(2*math.pi)


def surrogate_logprob(coordinates, reward_horizon=8):
    return coordinates[:, :reward_horizon].clamp(-5, 2).mean((-1, -2))


def variable_gae(rewards, values, next_values, discounts, durations, lanes,
                 gae_lambda=.95, decision_frames=10):
    """Rows may be interleaved across lanes; terminal discounts cut recursion."""
    advantage=torch.zeros_like(rewards)
    tail={}
    for i in range(len(rewards)-1, -1, -1):
        lane=int(lanes[i]);following=tail.get(lane, rewards.new_tensor(0.))
        delta=rewards[i]+discounts[i]*next_values[i]-values[i]
        advantage[i]=delta+discounts[i]*(gae_lambda**(float(durations[i])/decision_frames))*following
        tail[lane]=advantage[i]
    return advantage, advantage+values


class DPPOSampler:
    def __init__(self, diffusion, base_unet=None, ft_steps=3, min_std=.02):
        self.diffusion=diffusion
        self.base_unet=diffusion.unet if base_unet is None else base_unet
        self.ft_steps=int(ft_steps);self.min_std=float(min_std)
        scheduler=diffusion.noise_scheduler
        if scheduler.__class__.__name__!='DDPMScheduler':raise ValueError('DPPO adapter requires original DDPM')
        if scheduler.config.prediction_type!='epsilon' or scheduler.config.variance_type!='fixed_small':
            raise ValueError('Only epsilon/fixed_small DDPM is verified')
        if scheduler.config.thresholding:raise ValueError('Dynamic thresholding is not supported')
        self.native_step=scheduler.step
        self.last=None

    def prepare(self, device):
        scheduler=self.diffusion.noise_scheduler
        scheduler.set_timesteps(self.diffusion.num_inference_steps)
        self.timesteps=[int(x) for x in scheduler.timesteps]
        if not 1<=self.ft_steps<=len(self.timesteps):raise ValueError('Invalid ft_steps')
        self.ft_timesteps=self.timesteps[-self.ft_steps:]
        self.alphas=scheduler.alphas_cumprod.to(device)
        self.previous=torch.full((len(self.alphas),), -1, device=device, dtype=torch.long)
        for t in self.timesteps:self.previous[t]=int(scheduler.previous_timestep(t))

    def moments(self, sample, model_output, timesteps, min_std=None):
        scheduler=self.diffusion.noise_scheduler
        t=timesteps.to(sample.device, torch.long);prev=self.previous[t]
        at=self.alphas[t][:,None,None]
        ap=torch.where(prev>=0,self.alphas[prev.clamp_min(0)],torch.ones_like(at[:,0,0]))[:,None,None]
        bt=1-at;bp=1-ap;alpha=at/ap;beta=1-alpha
        original=(sample-bt**.5*model_output)/at**.5
        if scheduler.config.clip_sample:
            original=original.clamp(-scheduler.config.clip_sample_range,scheduler.config.clip_sample_range)
        mean=(ap**.5*beta/bt)*original+(alpha**.5*bp/bt)*sample
        variance=(bp/bt*beta).clamp_min(1e-20)
        std=variance**.5
        floor=self.min_std if min_std is None else min_std
        return mean, std.clamp_min(floor)

    def logprob(self, condition, before, after, timesteps):
        output=self.diffusion.unet(before,timesteps,global_cond=condition)
        mean,std=self.moments(before,output,timesteps)
        return gaussian_logprob(after,mean,std)

    @torch.no_grad()
    def sample(self, batch_size, global_cond=None, generator=None, noise=None,
               *, exploration=False, record=False):
        unet=self.diffusion.unet
        device=next(unet.parameters()).device;dtype=next(unet.parameters()).dtype
        self.prepare(device)
        shape=(batch_size,self.diffusion.config.horizon,self.diffusion.config.action_feature.shape[0])
        sample=torch.randn(shape,device=device,dtype=dtype,generator=generator) if noise is None else noise
        records=[]
        for t in self.timesteps:
            selected=t in self.ft_timesteps
            actor=unet if selected else self.base_unet
            times=torch.full((batch_size,),t,device=device,dtype=torch.long)
            output=actor(sample,times,global_cond=global_cond)
            before=sample
            if exploration and selected:
                mean,std=self.moments(before,output,times)
                raw=mean+std*torch.randn(shape,device=device,dtype=dtype,generator=generator)
                sample=raw
                if t==0 and self.diffusion.noise_scheduler.config.clip_sample:
                    bound=self.diffusion.noise_scheduler.config.clip_sample_range
                    sample=sample.clamp(-bound,bound)
            else:
                # Exact original RNG consumption and scheduler arithmetic.
                sample=self.native_step(output,t,before,generator=generator).prev_sample
                raw=sample
                mean,std=self.moments(before,output,times)
            if selected and record:
                records.append(dict(before=before.detach().clone(),after=raw.detach().clone(),
                    timestep=times,logprob=gaussian_logprob(raw,mean,std)))
        self.last=None
        if record:
            self.last=dict(condition=global_cond.detach().clone(),
                before=torch.stack([x['before'] for x in records],1),
                after=torch.stack([x['after'] for x in records],1),
                timesteps=torch.stack([x['timestep'] for x in records],1),
                logprob=torch.stack([x['logprob'] for x in records],1))
        return sample


def restore_dppo_sampler(diffusion, metadata, base_state):
    """Optional deployment hook; ordinary DP checkpoints do not enter here."""
    if diffusion.num_inference_steps!=metadata['inference_steps']:
        raise ValueError('DPPO checkpoint inference schedule must remain frozen')
    base=deepcopy(diffusion.unet);base.load_state_dict(base_state);base.eval()
    for parameter in base.parameters():parameter.requires_grad_(False)
    sampler=DPPOSampler(diffusion,base,metadata['ft_steps'],metadata['min_std'])
    # Do not register frozen copies in the ordinary policy state_dict.
    object.__setattr__(diffusion,'_dppo_sampler',sampler)
    diffusion.conditional_sample=lambda batch_size,global_cond=None,generator=None,noise=None: sampler.sample(
        batch_size,global_cond,generator,noise,exploration=False)
    return sampler
