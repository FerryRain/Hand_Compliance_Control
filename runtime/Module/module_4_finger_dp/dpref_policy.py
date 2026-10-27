"""Two-head Finger DP reference generator.

The diffusion head predicts a continuous measured-q-anchored nominal command
chunk.  A separate categorical head proposes one contact role per finger.
Neither head owns low-level force regulation or executable contact authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as functional

from Module.module_4_finger_dp.contracts import (
  ACTION_HORIZON_STEPS,
  NUM_FINGERS,
  NUM_FINGER_JOINTS,
)
from Module.module_4_finger_dp.policy import (
  ConditionalActionDenoiser,
  SharedForceHistoryEncoder,
  TemporalVectorEncoder,
)
from Module.module_4_whole_hand_mcc.reference_interpreter import ContactRole


DPREF_INPUT_NAMES = (
  "force_history",
  "finger_state_geometry",
  "wrist_real_twist_history",
  "wrist_mcc_offset_history",
  "wrist_mcc_velocity_history",
  "future_wrist_plan_twist",
  "q_meas_rad",
  "previous_nominal_command_rad",
  "previous_mcc_correction_rad",
)


class DPRefConditionEncoder(nn.Module):
  """Shared causal encoder used by both trajectory and role heads."""

  def __init__(self, condition_dimension: int = 256) -> None:
    super().__init__()
    self.force_encoder = SharedForceHistoryEncoder(3, 32)
    self.finger_id = nn.Embedding(NUM_FINGERS, 8)
    self.state_encoder = nn.Sequential(
      nn.Linear(20 + 8, 64),
      nn.SiLU(),
      nn.Linear(64, 64),
      nn.SiLU(),
    )
    self.finger_fusion = nn.Sequential(
      nn.Linear((32 + 64) * NUM_FINGERS, 224),
      nn.SiLU(),
    )
    self.wrist_history_encoder = TemporalVectorEncoder(18, 64)
    self.wrist_plan_encoder = TemporalVectorEncoder(6, 64)
    self.command_state_encoder = nn.Sequential(
      nn.Linear(NUM_FINGER_JOINTS * 3, 64),
      nn.SiLU(),
      nn.Linear(64, 64),
      nn.SiLU(),
    )
    self.output = nn.Sequential(
      nn.Linear(224 + 64 + 64 + 64, condition_dimension),
      nn.SiLU(),
      nn.Linear(condition_dimension, condition_dimension),
    )
    self.condition_dimension = condition_dimension

  def forward(self, inputs: Mapping[str, Tensor]) -> Tensor:
    missing = set(DPREF_INPUT_NAMES).difference(inputs)
    if missing:
      raise ValueError(f"DPRef inputs missing {sorted(missing)}")
    state = inputs["finger_state_geometry"]
    if state.ndim != 3 or state.shape[1:] != (NUM_FINGERS, 20):
      raise ValueError("finger_state_geometry must have shape (B,4,20)")
    batch = state.shape[0]
    ids = torch.arange(NUM_FINGERS, device=state.device)[None].expand(batch, -1)
    state_latent = self.state_encoder(torch.cat((state, self.finger_id(ids)), dim=-1))
    force_latent = self.force_encoder(inputs["force_history"])
    finger_latent = self.finger_fusion(
      torch.cat((force_latent, state_latent), dim=-1).flatten(1)
    )
    wrist_history = torch.cat(
      (
        inputs["wrist_real_twist_history"],
        inputs["wrist_mcc_offset_history"],
        inputs["wrist_mcc_velocity_history"],
      ),
      dim=-1,
    )
    wrist_latent = self.wrist_history_encoder(wrist_history)
    plan_latent = self.wrist_plan_encoder(inputs["future_wrist_plan_twist"])
    command_latent = self.command_state_encoder(
      torch.cat(
        (
          inputs["q_meas_rad"],
          inputs["previous_nominal_command_rad"],
          inputs["previous_mcc_correction_rad"],
        ),
        dim=-1,
      )
    )
    return self.output(
      torch.cat((finger_latent, wrist_latent, plan_latent, command_latent), dim=-1)
    )


@dataclass(frozen=True, slots=True)
class DPRefPolicyConfig:
  action_horizon_steps: int = ACTION_HORIZON_STEPS
  diffusion_steps: int = 20
  beta_start: float = 1e-4
  beta_end: float = 0.20
  action_scale_rad: float = 0.10
  max_abs_action_offset_rad: float = 0.20
  role_count: int = len(ContactRole)
  role_loss_weight: float = 0.25

  def __post_init__(self) -> None:
    if self.action_horizon_steps != ACTION_HORIZON_STEPS:
      raise ValueError("DPRef v1 uses the frozen action horizon")
    if self.diffusion_steps < 2:
      raise ValueError("diffusion_steps must be at least two")
    if not 0.0 < self.beta_start < self.beta_end < 1.0:
      raise ValueError("invalid diffusion beta schedule")
    if self.action_scale_rad <= 0.0 or self.max_abs_action_offset_rad <= 0.0:
      raise ValueError("action scales must be positive")
    if self.role_count != len(ContactRole) or self.role_loss_weight <= 0.0:
      raise ValueError("invalid role-head configuration")


@dataclass(frozen=True, slots=True)
class DPRefLoss:
  total: Tensor
  diffusion: Tensor
  role: Tensor
  valid_role_count: int


class FingerDPRefPolicy(nn.Module):
  """Nominal trajectory generator plus categorical contact-intention head."""

  def __init__(self, config: DPRefPolicyConfig | None = None) -> None:
    super().__init__()
    self.config = config or DPRefPolicyConfig()
    self.condition_encoder = DPRefConditionEncoder(256)
    self.denoiser = ConditionalActionDenoiser(256)
    self.role_head = nn.Sequential(
      nn.Linear(256, 128),
      nn.SiLU(),
      nn.Linear(128, NUM_FINGERS * self.config.role_count),
    )
    betas = torch.linspace(
      self.config.beta_start,
      self.config.beta_end,
      self.config.diffusion_steps,
      dtype=torch.float32,
    )
    alphas = 1.0 - betas
    self.register_buffer("betas", betas)
    self.register_buffer("alphas", alphas)
    self.register_buffer("alpha_bar", torch.cumprod(alphas, dim=0))

  def encode(self, inputs: Mapping[str, Tensor]) -> Tensor:
    return self.condition_encoder(inputs)

  def role_logits_from_condition(self, condition: Tensor) -> Tensor:
    return self.role_head(condition).reshape(
      condition.shape[0], NUM_FINGERS, self.config.role_count
    )

  def loss(
    self,
    inputs: Mapping[str, Tensor],
    target_nominal_offsets: Tensor,
    target_role: Tensor,
    role_valid: Tensor,
    *,
    role_class_weights: Tensor | None = None,
  ) -> DPRefLoss:
    if target_nominal_offsets.shape[1:] != (
      self.config.action_horizon_steps,
      NUM_FINGER_JOINTS,
    ):
      raise ValueError("target_nominal_offsets has the wrong shape")
    if target_role.shape != role_valid.shape or target_role.shape[1:] != (NUM_FINGERS,):
      raise ValueError("role target/mask must have shape (B,4)")
    condition = self.encode(inputs)
    normalized = target_nominal_offsets / self.config.action_scale_rad
    batch = normalized.shape[0]
    timestep = torch.randint(
      0,
      self.config.diffusion_steps,
      (batch,),
      device=normalized.device,
    )
    noise = torch.randn_like(normalized)
    alpha_bar = self.alpha_bar[timestep][:, None, None]
    noisy = alpha_bar.sqrt() * normalized + (1.0 - alpha_bar).sqrt() * noise
    predicted_noise = self.denoiser(noisy, timestep, condition)
    diffusion_loss = functional.mse_loss(predicted_noise, noise)
    logits = self.role_logits_from_condition(condition)
    valid = role_valid.bool()
    valid_count = int(torch.count_nonzero(valid).item())
    if valid_count:
      role_loss = functional.cross_entropy(
        logits[valid],
        target_role.long()[valid],
        weight=role_class_weights,
      )
    else:
      role_loss = logits.sum() * 0.0
    total = diffusion_loss + self.config.role_loss_weight * role_loss
    return DPRefLoss(total, diffusion_loss, role_loss, valid_count)

  @torch.no_grad()
  def sample(
    self,
    inputs: Mapping[str, Tensor],
    *,
    initial_noise: Tensor | None = None,
    deterministic: bool = True,
  ) -> tuple[Tensor, Tensor, Tensor]:
    condition = self.encode(inputs)
    logits = self.role_logits_from_condition(condition)
    probabilities = functional.softmax(logits, dim=-1)
    batch = condition.shape[0]
    shape = (batch, self.config.action_horizon_steps, NUM_FINGER_JOINTS)
    value = (
      torch.randn(*shape, device=condition.device, dtype=condition.dtype)
      if initial_noise is None
      else initial_noise.to(device=condition.device, dtype=condition.dtype).clone()
    )
    if tuple(value.shape) != shape:
      raise ValueError(f"initial_noise must have shape {shape}")
    for step in reversed(range(self.config.diffusion_steps)):
      timestep = torch.full((batch,), step, device=condition.device, dtype=torch.long)
      predicted_noise = self.denoiser(value, timestep, condition)
      alpha = self.alphas[step]
      alpha_bar = self.alpha_bar[step]
      if deterministic:
        predicted_clean = (
          value - (1.0 - alpha_bar).sqrt() * predicted_noise
        ) / alpha_bar.sqrt()
        if step:
          previous_alpha_bar = self.alpha_bar[step - 1]
          value = (
            previous_alpha_bar.sqrt() * predicted_clean
            + (1.0 - previous_alpha_bar).sqrt() * predicted_noise
          )
        else:
          value = predicted_clean
      else:
        mean = (
          value - self.betas[step] * predicted_noise / (1.0 - alpha_bar).sqrt()
        ) / alpha.sqrt()
        value = (
          mean + self.betas[step].sqrt() * torch.randn_like(value)
          if step
          else mean
        )
    offsets = torch.clamp(
      value * self.config.action_scale_rad,
      -self.config.max_abs_action_offset_rad,
      self.config.max_abs_action_offset_rad,
    )
    return offsets, probabilities.argmax(dim=-1), probabilities


__all__ = [
  "DPREF_INPUT_NAMES",
  "DPRefConditionEncoder",
  "DPRefLoss",
  "DPRefPolicyConfig",
  "FingerDPRefPolicy",
]
