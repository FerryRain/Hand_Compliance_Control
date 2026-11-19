"""Opt-in denoised physical joint-error objective for anchored-delta DP.

No rollout states or teacher-reference attraction: this supervises only the
original valid windows. It does not by itself provide recovery labels.
"""
import numpy as np
import torch
import torch.nn.functional as F


def fit_error_scale(episodes,ids,obs_horizon,pred_horizon,quantile=.95,floor_rad=.001745329252):
    chunks=[]
    for eid in ids:
        state,q=episodes[eid];t=np.arange(obs_horizon-1,len(q)-pred_horizon)
        if len(t):chunks.append(q[t[:,None]+1+np.arange(pred_horizon)]-state[t,None,:16])
    if not chunks:raise ValueError('No training windows for physical error scale')
    return np.maximum(np.quantile(abs(np.concatenate(chunks)),quantile,axis=0),floor_rad).astype(np.float32)


def physical_clean_error(noisy,predicted_epsilon,target,alpha,action_scale,error_scale,valid,prefix=7):
    """Physical Huber error; FP32 reconstruction, no clipping of gradients."""
    clean=(noisy.float()-torch.sqrt(1-alpha)*predicted_epsilon.float())/torch.sqrt(alpha)
    error=(clean-target.float())*action_scale.float()/error_scale.float()
    terms=F.huber_loss(error,torch.zeros_like(error),reduction='none',delta=1.)
    weights=terms.new_ones(terms.shape[1]);weights[:min(prefix,len(weights))]=3.
    weights=valid.float()*weights[None]
    return (terms*weights[:,:,None]).sum()/(weights.sum()*terms.shape[-1]).clamp_min(1.)


def diffusion_with_near_q_loss(policy,batch,*,action_scale,error_scale,weight,max_timestep=24,prefix=7):
    diffusion=policy.diffusion
    if diffusion.config.prediction_type!='epsilon':raise ValueError('Requires epsilon prediction')
    trajectory=batch['action'];cond=diffusion._prepare_global_conditioning(batch)
    epsilon=torch.randn(trajectory.shape,device=trajectory.device)
    timestep=torch.randint(0,diffusion.noise_scheduler.config.num_train_timesteps,
        (len(trajectory),),device=trajectory.device).long()
    noisy=diffusion.noise_scheduler.add_noise(trajectory,epsilon,timestep)
    pred=diffusion.unet(noisy,timestep,global_cond=cond)
    terms=F.mse_loss(pred,epsilon,reduction='none')
    valid=~batch['action_is_pad']
    if diffusion.config.do_mask_loss_for_padding:terms=terms*valid[:,:,None]
    noise_loss=terms.mean();chosen=timestep<=max_timestep
    if weight==0 or not bool(chosen.any()):return noise_loss,noise_loss,noise_loss.new_zeros(())
    alpha=diffusion.noise_scheduler.alphas_cumprod.to(trajectory.device,dtype=torch.float32)[timestep[chosen]][:,None,None]
    aux=physical_clean_error(noisy[chosen],pred[chosen],trajectory[chosen],alpha,action_scale,error_scale,valid[chosen],prefix)
    return noise_loss+weight*aux,noise_loss,aux
