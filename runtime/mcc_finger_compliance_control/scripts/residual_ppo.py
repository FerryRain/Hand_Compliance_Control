"""On-policy residual PPO: raw Gaussian ratios, causal history, lane-wise GAE.

Observations use the existing fixed physical scaling (no moving normalization).
Updates occur only after a complete rollout. Critic-only reference features never
enter the actor. The bounded controller remains responsible for slew limiting.
"""
import torch
from torch import nn
from residual_training_adapter import TrainingResidual, OBS_DIM

PPO_EXTRA_FIELDS = [('actual_command_error', 16), ('limiter_target_gap', 16)]
PPO_OBS_DIM = OBS_DIM + sum(width for _, width in PPO_EXTRA_FIELDS)


def execution_feedback(q, actuator_target, requested_offset, applied_offset, amplitude):
    """Latest completed substep's actual target; never the upcoming command."""
    if actuator_target.shape != q.shape:
        raise ValueError('actuator target must match live hand joint order')
    return torch.nan_to_num(torch.cat(((q-actuator_target)/.5,
        (requested_offset-applied_offset)/amplitude), -1), nan=0., posinf=10., neginf=-10.).clamp(-10,10)


def mlp(dim, out, hidden):
    return nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, out))


class GaussianActor(nn.Module):
    def __init__(self, dim, hidden, initial_std):
        super().__init__()
        self.net = mlp(dim, 16, hidden)
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
        self.log_std = nn.Parameter(initial_std.log())

    def distribution(self, obs):
        return torch.distributions.Normal(self.net(obs), self.log_std.clamp(-9, 0).exp())

    def forward(self, obs):
        return self.net(obs).tanh()


class PPOResidual(TrainingResidual):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history = None
        self.history_step = -1

    def augment_frame_observation(self, obs, *args):
        return obs

    def observations(self, *args):
        obs, critic = super().observations(*args)
        obs = torch.cat((obs, execution_feedback(args[0], self.context['previous_actuator_target'],
            self.limiter.target, self.limiter.offset, self.normalization_amplitude)), -1)
        obs = self.augment_frame_observation(obs, *args)
        h = self.agent.cfg['history']
        if self.history is None:
            # Initial padding repeats the first available live observation.
            self.history = obs[:, None].expand(-1, h, -1).clone()
        step = self.limiter.control_steps
        if step % 5 == 0 and step != self.history_step:
            self.history = torch.cat((obs[:, None], self.history[:, :h-1]), 1)
            self.history_step = step
        # Failure between decisions: current observation plus older decision history.
        history = self.history if step % 5 == 0 else torch.cat((obs[:, None], self.history[:, :h-1]), 1)
        actor = history.flatten(1)
        return actor, torch.cat((actor, critic[:, OBS_DIM:]), -1)

    def flush(self, mask, obs, critic, terminal):
        mask = mask & self.pending
        if not bool(mask.any()): return
        ids = mask.nonzero().flatten()
        if not self.evaluate:
            elapsed = self.elapsed[ids]
            discount = self.reward.config.gamma_frame ** elapsed
            discount = torch.where(terminal[ids], 0., discount)
            with torch.no_grad():
                transition=dict(lane=ids.clone(), obs=self.previous_obs[ids].clone(),
                    critic=self.previous_critic[ids].clone(), z=self.agent.last_z[ids].clone(),
                    logp=self.agent.last_logp[ids].clone(), reward=self.accumulated[ids].clone(),
                    discount=discount, lam=self.agent.cfg['gae_lambda'] ** (elapsed.float()/5),
                    value=self.agent.value(self.previous_critic[ids]).squeeze(-1),
                    next_value=self.agent.value(critic[ids]).squeeze(-1))
                # Offload completed transitions so long P1 histories do not
                # compete with Warp graph allocations on the 8 GiB device.
                self.agent.rows.append({k:v.detach().cpu() for k,v in transition.items()})
            self.agent.transitions += len(ids)
        self.pending[mask] = False; self.elapsed[mask] = 0; self.accumulated[mask] = 0


def advantages(rows):
    """Reverse recurrence by lane; zero discount cuts true terminal boundaries."""
    lanes = max(int(row['lane'].max()) for row in rows) + 1
    carry = torch.zeros(lanes, device=rows[0]['reward'].device)
    output = []
    for row in reversed(rows):
        following = carry[row['lane']]
        delta = row['reward'] + row['discount'] * row['next_value'] - row['value']
        adv = delta + row['discount'] * row['lam'] * following
        carry[row['lane']] = adv
        output.append(adv)
    return torch.cat(list(reversed(output)))


class PPO:
    residual_class = PPOResidual

    def __init__(self, config, bounds, device):
        self.cfg = dict(config, updates_per_decision=0)
        self.device = torch.device(device)
        dim = config.get('actor_frame_dim', PPO_OBS_DIM) * config['history']
        amplitude = torch.tensor(bounds['amplitude_rad'], device=device)
        slew = torch.tensor(bounds['decision_slew_rad'], device=device)
        std = torch.minimum(.05 * amplitude, .25 * slew) / amplitude
        if 'initial_normalized_action_std' in config:
            scale=float(config['initial_normalized_action_std'])
            if not 0<scale<=1:
                raise ValueError('normalized action std must be in (0,1]')
            std=torch.full_like(std,scale)
        self.actor = GaussianActor(dim, config['hidden'], std).to(device)
        self.value = mlp(config.get('critic_input_dim', dim + 5), 1, config['hidden']).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config['actor_lr'])
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=config['value_lr'])
        self.generator = torch.Generator(device=device).manual_seed(config['seed']+19)
        self.rows = []; self.transitions = 0; self.updates = 0; self.actor_updates = 0; self.last_losses = {}

    @torch.no_grad()
    def act(self, obs, zero=False, evaluate=False):
        dist = self.actor.distribution(obs)
        if zero: z = torch.zeros_like(dist.loc)
        elif evaluate: z = dist.loc
        else: z = dist.loc + dist.scale * torch.randn(dist.loc.shape, device=self.device, generator=self.generator)
        self.last_z = z; self.last_logp = dist.log_prob(z).sum(-1)
        return z.tanh()

    def update(self):
        raise RuntimeError('PPO updates are forbidden inside the physical rollout')

    def finish_rollout(self, value_only=False):
        if not self.rows: raise ValueError('empty PPO rollout')
        if value_only:
            # Full baseline episodes provide Monte Carlo return targets, not
            # lambda-truncated targets dominated by an untrained value net.
            for row in self.rows:
                row['lam'] = torch.ones_like(row['lam'])
        adv = advantages(self.rows)
        data = {k: torch.cat([r[k] for r in self.rows]) for k in ('obs','critic','z','logp','value')}
        target = adv + data['value']
        adv = (adv-adv.mean()) / adv.std(unbiased=False).clamp_min(1e-8)
        n = len(adv); records = []; actor_enabled = not value_only
        actor_updates_before = self.actor_updates
        measured_kl = []
        with torch.enable_grad():
            for _ in range(self.cfg['epochs']):
                permutation = torch.randperm(n, device=self.device, generator=self.generator).cpu()
                for ids in permutation.split(self.cfg['minibatch']):
                    batch={k:v[ids].to(self.device) for k,v in data.items()}
                    batch_adv=adv[ids].to(self.device)
                    value_loss = .5*(self.value(batch['critic']).squeeze(-1)-target[ids].to(self.device)).square().mean()
                    self.value_optimizer.zero_grad(set_to_none=True); value_loss.backward()
                    nn.utils.clip_grad_norm_(self.value.parameters(), 1., error_if_nonfinite=True)
                    self.value_optimizer.step(); self.updates += 1
                    kl = torch.zeros((), device=self.device); actor_loss = torch.zeros_like(kl)
                    if actor_enabled:
                        dist = self.actor.distribution(batch['obs'])
                        logp = dist.log_prob(batch['z']).sum(-1)
                        logratio = logp-batch['logp']; ratio = logratio.exp()
                        kl = ((ratio-1)-logratio).mean().detach()
                        if not torch.isfinite(kl):
                            raise FloatingPointError('nonfinite PPO KL')
                        measured_kl.append(float(kl))
                        if float(kl) > self.cfg['target_kl']:
                            actor_enabled = False
                        else:
                            clip = self.cfg['clip']
                            actor_loss = -torch.minimum(ratio*batch_adv, ratio.clamp(1-clip,1+clip)*batch_adv).mean()
                            self.actor_optimizer.zero_grad(set_to_none=True); actor_loss.backward()
                            nn.utils.clip_grad_norm_(self.actor.parameters(), 1., error_if_nonfinite=True)
                            self.actor_optimizer.step(); self.actor_updates += 1
                    records.append([float(value_loss.detach()),float(actor_loss.detach()),float(kl)])
        stats = torch.tensor(records).mean(0).tolist()
        kl_stopped = not actor_enabled and not value_only
        if kl_stopped:
            for group in self.actor_optimizer.param_groups:
                group['lr'] = max(1e-6, group['lr']*.5)
        feedback = data['obs'][:, OBS_DIM:OBS_DIM+16]*.5
        self.last_losses = dict(value_loss=stats[0], actor_loss=stats[1],
            approx_kl=sum(measured_kl)/max(1,len(measured_kl)), max_kl=max(measured_kl,default=0.),
            actor_updates_this_rollout=self.actor_updates-actor_updates_before,
            next_actor_lr=self.actor_optimizer.param_groups[0]['lr'],
            command_error_rms_rad=float(feedback.square().mean().sqrt()),
            collected=n, value_only=value_only, kl_stopped=kl_stopped)
        self.rows.clear()
        return self.last_losses
