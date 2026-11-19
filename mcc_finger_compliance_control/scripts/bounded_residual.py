"""Five-frame residual decisions with per-control-frame bounded slew."""
from __future__ import annotations

import torch


class BoundedResidualPolicy(torch.nn.Module):
    def __init__(self, actor: torch.nn.Module, amplitude: torch.Tensor,
                 decision_slew: torch.Tensor, decision_frames: int = 5):
        super().__init__()
        amplitude = torch.as_tensor(amplitude, dtype=torch.float32)
        decision_slew = torch.as_tensor(decision_slew, dtype=torch.float32, device=amplitude.device)
        if amplitude.ndim != 2 or amplitude.shape[-1] != 16 or decision_slew.shape != amplitude.shape:
            raise ValueError("amplitude and slew must have shape [batch,16]")
        if (not torch.isfinite(amplitude).all() or not torch.isfinite(decision_slew).all()
                or (amplitude < 0).any() or (decision_slew < 0).any() or decision_frames < 1):
            raise ValueError("bounds must be finite and nonnegative, decision_frames positive")
        self.actor = actor
        self.register_buffer("amplitude", amplitude.clone())
        self.register_buffer("decision_slew", decision_slew.clone())
        self.register_buffer("offset", torch.zeros_like(amplitude))
        self.register_buffer("target", torch.zeros_like(amplitude))
        self.decision_frames = decision_frames
        self.control_steps = 0
        self.actor_decisions = 0

    @torch.no_grad()
    def forward(self, q_live, nominal_q, observation_state):
        if self.control_steps % self.decision_frames == 0:
            action = self.actor(q_live, nominal_q, observation_state)
            if action.shape != self.offset.shape or not torch.isfinite(action).all():
                raise ValueError("residual actor must return finite [batch,16] normalized actions")
            self.target.copy_(action.clamp(-1, 1) * self.amplitude)
            self.actor_decisions += 1
        frame_slew = self.decision_slew / self.decision_frames
        self.offset.add_((self.target - self.offset).clamp(-frame_slew, frame_slew))
        self.offset.clamp_(-self.amplitude, self.amplitude)
        self.control_steps += 1
        # Do not expose mutable limiter memory to downstream control code.
        return self.offset.clone()

    @torch.no_grad()
    def reset(self):
        self.offset.zero_()
        self.target.zero_()
        self.control_steps = self.actor_decisions = 0
        if hasattr(self.actor, "reset"):
            self.actor.reset()


class RecordedResidualActor(torch.nn.Module):
    """Deterministic normalized action table used only by calibration."""
    def __init__(self, actions):
        super().__init__()
        self.register_buffer("actions", torch.as_tensor(actions, dtype=torch.float32))
        self.index = 0

    def forward(self, q_live, nominal_q, observation_state):
        del q_live, nominal_q, observation_state
        if self.index >= len(self.actions):
            raise RuntimeError("residual calibration action table exhausted")
        action = self.actions[self.index]
        self.index += 1
        return action

    def reset(self):
        self.index = 0
