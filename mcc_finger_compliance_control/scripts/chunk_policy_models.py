"""State-history adaptations of ACT and Transformer Diffusion Policy.

References: https://tonyzhaozh.github.io/aloha/ (CVAE, L1+KL, z=0 inference)
https://diffusion-policy.cs.columbia.edu/ (conditional temporal diffusion transformer).
These are local low-dimensional adaptations, not exact official reproductions.
All observed history and palm-command channels are retained. No new execution
filter, FK loss, contact teacher, finger partition, or action clamp is introduced.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


def positional_encoding(length, width):
    position = torch.arange(length).float()[:, None]
    frequency = torch.exp(torch.arange(0, width, 2).float() * (-math.log(10000.) / width))
    result = torch.zeros(length, width)
    result[:, 0::2] = torch.sin(position * frequency)
    result[:, 1::2] = torch.cos(position * frequency)
    return result[None]


def encoder(width, heads, layers):
    # Disable nested tensors to keep padded/nonpadded semantics identical.
    return nn.TransformerEncoder(nn.TransformerEncoderLayer(
        width, heads, 4*width, dropout=0., activation='gelu',
        batch_first=True, norm_first=True), layers,
        norm=nn.LayerNorm(width), enable_nested_tensor=False)


def decoder(width, heads, layers):
    return nn.TransformerDecoder(nn.TransformerDecoderLayer(
        width, heads, 4*width, dropout=0., activation='gelu',
        batch_first=True, norm_first=True), layers, norm=nn.LayerNorm(width))


def initialize(module):
    # TransformerEncoder/Decoder clone a layer: explicitly break identical
    # initial weights across layers while preserving norm and bias defaults.
    for parameter in module.parameters():
        if parameter.ndim > 1: nn.init.xavier_uniform_(parameter)


class HistoryCondition:
    def _prepare_global_conditioning(self, batch):
        x = torch.cat((batch['observation.state'], batch['observation.environment_state']), -1)
        if x.shape[1:] != (self.config.n_obs_steps, self.state_dim):
            raise ValueError(f'Expected full history, got {x.shape}')
        return x


class ACTModel(nn.Module, HistoryCondition):
    def __init__(self, config, args):
        super().__init__(); self.config = config
        self.state_dim = config.robot_state_feature.shape[0] + config.env_state_feature.shape[0]
        self.action_dim = config.action_feature.shape[0]
        width = args.transformer_dim; heads = args.transformer_heads
        self.kl_weight = args.act_kl_weight
        self.latent_dim = args.act_latent_dim
        self.num_inference_steps = 1
        self.obs_projection = nn.Linear(self.state_dim, width)
        self.latent_projection = nn.Linear(self.latent_dim, width)
        self.context_encoder = encoder(width, heads, args.transformer_encoder_layers)
        self.action_decoder = decoder(width, heads, args.transformer_decoder_layers)
        self.output = nn.Linear(width, self.action_dim)
        self.posterior_encoder = encoder(width, heads, args.act_posterior_layers)
        self.posterior_q = nn.Linear(self.action_dim, width)
        self.posterior_action = nn.Linear(self.action_dim, width)
        self.posterior_cls = nn.Parameter(torch.zeros(1, 1, width))
        self.posterior_stats = nn.Linear(width, 2*self.latent_dim)
        self.register_buffer('history_pos', positional_encoding(config.n_obs_steps+1, width))
        self.register_buffer('action_pos', positional_encoding(config.horizon, width))
        self.register_buffer('posterior_pos', positional_encoding(config.horizon+2, width))
        initialize(self)
        self.last_loss_components = {}

    def predict(self, history, z):
        tokens = torch.cat((self.latent_projection(z)[:, None], self.obs_projection(history)), 1)
        memory = self.context_encoder(tokens + self.history_pos)
        queries = self.action_pos.expand(len(history), -1, -1)
        return self.output(self.action_decoder(queries, memory))

    def compute_loss(self, batch):
        history = self._prepare_global_conditioning(batch)
        target = batch['action']; b = len(target)
        pad = batch['action_is_pad'].bool()
        tokens = torch.cat((self.posterior_cls.expand(b, -1, -1),
                            self.posterior_q(history[:, -1, :self.action_dim])[:, None],
                            self.posterior_action(target)), 1)
        mask = torch.cat((torch.zeros(b, 2, device=target.device, dtype=torch.bool), pad), 1)
        encoded = self.posterior_encoder(tokens + self.posterior_pos, src_key_padding_mask=mask)
        mu, logvar = self.posterior_stats(encoded[:, 0]).float().chunk(2, -1)
        z = mu + torch.exp(.5*logvar) * torch.randn_like(mu)
        predicted = self.predict(history, z.to(history.dtype))
        valid = (~pad).unsqueeze(-1)
        l1 = ((predicted-target).abs()*valid).sum() / (valid.sum()*self.action_dim).clamp_min(1)
        kl = (-.5*(1 + logvar - mu.square() - logvar.exp()).sum(-1)).mean()
        self.last_loss_components = {'reconstruction_l1': float(l1.detach()), 'kl': float(kl.detach())}
        return l1 + self.kl_weight*kl

    def conditional_sample(self, batch_size, global_cond=None, generator=None, noise=None):
        if global_cond is None or len(global_cond) != batch_size: raise ValueError('Missing observation')
        # Standard ACT deployment: prior mean, no posterior/future labels.
        z = global_cond.new_zeros((batch_size, self.latent_dim))
        return self.predict(global_cond, z)


class ACTPolicy(nn.Module):
    def __init__(self, config, args):
        super().__init__(); self.config = config
        # Shared runtime API; this member is a CVAE, not a diffusion process.
        self.diffusion = ACTModel(config, args)


class TemporalDenoiser(nn.Module):
    def __init__(self, config, args):
        super().__init__()
        width = args.transformer_dim
        self.width = width
        state_dim = config.robot_state_feature.shape[0] + config.env_state_feature.shape[0]
        self.condition = nn.Linear(state_dim, width)
        self.action = nn.Linear(config.action_feature.shape[0], width)
        self.time = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.condition_encoder = encoder(width, args.transformer_heads, args.transformer_encoder_layers)
        self.action_decoder = decoder(width, args.transformer_heads, args.transformer_decoder_layers)
        self.output = nn.Linear(width, config.action_feature.shape[0])
        self.register_buffer('condition_pos', positional_encoding(config.n_obs_steps+1, width))
        self.register_buffer('action_pos', positional_encoding(config.horizon, width))
        self.register_buffer('causal_mask', torch.triu(torch.ones(config.horizon, config.horizon, dtype=torch.bool), 1))
        initialize(self)

    def forward(self, x, timestep, global_cond=None):
        half = self.width//2
        frequency = torch.exp(torch.arange(half, device=x.device).float()*(-math.log(10000.)/(half-1)))
        phase = timestep.float().reshape(-1, 1)*frequency[None]
        time = self.time(torch.cat((phase.sin(), phase.cos()), -1))
        context = torch.cat((time[:, None], self.condition(global_cond)), 1) + self.condition_pos
        memory = self.condition_encoder(context)
        hidden = self.action_decoder(self.action(x)+self.action_pos, memory, tgt_mask=self.causal_mask)
        return self.output(hidden)


def transformer_diffusion(original, args):
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionModel

    class HistoryTransformerDiffusion(HistoryCondition, DiffusionModel):
        def __init__(self):
            nn.Module.__init__(self)
            self.config = original.config
            self.state_dim = self.config.robot_state_feature.shape[0] + self.config.env_state_feature.shape[0]
            self.noise_scheduler = original.noise_scheduler
            self.num_inference_steps = original.num_inference_steps
            self.unet = TemporalDenoiser(self.config, args)
    return HistoryTransformerDiffusion()
