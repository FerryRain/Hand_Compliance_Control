"""DPPO likelihood, native sampler parity and decision termination checks."""
import sys
from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace
import torch
from torch import nn
from diffusers import DDPMScheduler

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'mcc_finger_compliance_control/scripts'))
from dppo_diffusion import DPPOSampler,gaussian_logprob,variable_gae,restore_dppo_sampler


class ToyUNet(nn.Module):
    def __init__(self):
        super().__init__();self.layer=nn.Linear(4,4)

    def forward(self,sample,times,global_cond):
        return self.layer(sample)+.001*times[:,None,None]+.01*global_cond[:,None,:1]


def diffusion():
    torch.manual_seed(8)
    return SimpleNamespace(unet=ToyUNet(),num_inference_steps=20,
        config=SimpleNamespace(horizon=8,action_feature=SimpleNamespace(shape=(4,))),
        noise_scheduler=DDPMScheduler(num_train_timesteps=100,beta_schedule='squaredcos_cap_v2',
            prediction_type='epsilon',clip_sample=True,variance_type='fixed_small'))


def original(model,condition,generator):
    x=torch.randn((len(condition),8,4),generator=generator)
    model.noise_scheduler.set_timesteps(20)
    for t in model.noise_scheduler.timesteps:
        output=model.unet(x,torch.full((len(condition),),int(t),dtype=torch.long),global_cond=condition)
        x=model.noise_scheduler.step(output,t,x,generator=generator).prev_sample
    return x


def test_native_samples_and_random_stream_unchanged():
    model=diffusion();condition=torch.randn(3,6)
    left=torch.Generator().manual_seed(47);right=torch.Generator().manual_seed(47)
    with torch.no_grad():expected=original(model,condition,left)
    actual=DPPOSampler(model).sample(3,condition,right,exploration=False)
    assert torch.equal(expected,actual)
    assert torch.equal(left.get_state(),right.get_state())


def test_raw_final_latent_and_likelihood_recompute():
    model=diffusion();sampler=DPPOSampler(model,min_std=.5);condition=torch.randn(3,6)
    output=sampler.sample(3,condition,torch.Generator().manual_seed(62),exploration=True,record=True)
    record=sampler.last
    assert (record['after'][:,-1].abs()>1).any()
    assert torch.equal(output,record['after'][:,-1].clamp(-1,1))
    for j in range(3):
        actual=sampler.logprob(condition,record['before'][:,j],record['after'][:,j],record['timesteps'][:,j])
        assert torch.allclose(actual,record['logprob'][:,j],atol=1e-5,rtol=1e-5)
    mean=torch.tensor([.2,-.1]);std=torch.tensor([.5,.7]);value=torch.tensor([.8,-1.])
    assert torch.allclose(gaussian_logprob(value,mean,std),torch.distributions.Normal(mean,std).log_prob(value))


def test_interleaved_terminal_and_truncation_bootstrap():
    # Lane0's second transition terminates. Lane1 is explicitly truncated:
    # next value contributes to its TD target, but no nonexistent row follows.
    rewards=torch.tensor([1.,3.,2.]);values=torch.tensor([.5,1.,.75])
    next_values=torch.tensor([.75,4.,0.]);discount=torch.tensor([.9,.8,0.])
    durations=torch.tensor([10,4,5]);lanes=torch.tensor([0,1,0])
    advantage,returns=variable_gae(rewards,values,next_values,discount,durations,lanes,gae_lambda=1.)
    assert torch.allclose(advantage,torch.tensor([2.3,5.2,1.25]))
    assert torch.allclose(returns,torch.tensor([2.8,6.2,2.]))


def test_frozen_early_network_and_checkpoint_sampling():
    model=diffusion();base=deepcopy(model.unet)
    for p in base.parameters():p.requires_grad_(False)
    sampler=DPPOSampler(model,base)
    condition=torch.randn(2,6)
    before=sampler.sample(2,condition,torch.Generator().manual_seed(9))
    with torch.no_grad():model.unet.layer.bias.add_(.01)
    restored=diffusion();restored.unet.load_state_dict(model.unet.state_dict())
    restored_sampler=restore_dppo_sampler(restored,dict(inference_steps=20,ft_steps=3,min_std=.02),base.state_dict())
    expected=sampler.sample(2,condition,torch.Generator().manual_seed(9))
    actual=restored.conditional_sample(2,condition,torch.Generator().manual_seed(9))
    assert torch.equal(actual,expected)
    assert not torch.equal(before,actual)
    assert all(not p.requires_grad for p in restored_sampler.base_unet.parameters())
