"""Minimal force-history-conditioned Finger Diffusion Policy v1."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as functional

from Module.module_4_finger_dp.contracts import (
  ACTION_HORIZON_STEPS,
  FingerDPObservation,
  NUM_FINGERS,
  NUM_FINGER_JOINTS,
)


class CausalConv1d(nn.Module):
  def __init__(
    self,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    dilation: int = 1,
  ) -> None:
    super().__init__()
    self.left_padding = dilation * (kernel_size - 1)
    self.convolution = nn.Conv1d(
      in_channels,
      out_channels,
      kernel_size,
      dilation=dilation,
    )

  def forward(self, value: Tensor) -> Tensor:
    return self.convolution(functional.pad(value, (self.left_padding, 0)))


class CausalResidualBlock(nn.Module):
  def __init__(self, channels: int, dilation: int) -> None:
    super().__init__()
    self.first = CausalConv1d(channels, channels, 3, dilation)
    self.second = CausalConv1d(channels, channels, 3, dilation)
    self.first_norm = nn.GroupNorm(1, channels)
    self.second_norm = nn.GroupNorm(1, channels)

  def forward(self, value: Tensor) -> Tensor:
    residual = value
    value = functional.silu(self.first_norm(self.first(value)))
    value = self.second_norm(self.second(value))
    return functional.silu(value + residual)


class SharedForceHistoryEncoder(nn.Module):
  """One causal TCN shared by all four fingertips."""

  def __init__(self, input_channels: int = 3, channels: int = 32) -> None:
    super().__init__()
    self.input_projection = nn.Conv1d(input_channels, channels, 1)
    self.blocks = nn.Sequential(
      CausalResidualBlock(channels, 1),
      CausalResidualBlock(channels, 2),
      CausalResidualBlock(channels, 4),
    )
    self.output_channels = channels

  def forward(self, force_history: Tensor) -> Tensor:
    if force_history.ndim != 4 or force_history.shape[1] != NUM_FINGERS or force_history.shape[-1] != 3:
      raise ValueError("force_history must have shape (B,4,L,3)")
    batch, fingers, steps, channels = force_history.shape
    value = force_history.reshape(batch * fingers, steps, channels).transpose(1, 2)
    value = self.blocks(self.input_projection(value))
    return value[:, :, -1].reshape(batch, fingers, self.output_channels)


class TemporalVectorEncoder(nn.Module):
  def __init__(self, input_channels: int, output_channels: int = 64) -> None:
    super().__init__()
    self.network = nn.Sequential(
      CausalConv1d(input_channels, output_channels, 3, 1),
      nn.SiLU(),
      CausalResidualBlock(output_channels, 2),
      CausalResidualBlock(output_channels, 4),
    )

  def forward(self, history: Tensor) -> Tensor:
    if history.ndim != 3:
      raise ValueError("temporal input must have shape (B,L,C)")
    return self.network(history.transpose(1, 2))[:, :, -1]


class FingerDPConditionEncoder(nn.Module):
  """Fuse force, state/geometry, actual wrist and future wrist-plan tokens."""

  per_finger_state_dimension = 20

  def __init__(self, condition_dimension: int = 256) -> None:
    super().__init__()
    self.force_encoder = SharedForceHistoryEncoder(3, 32)
    self.finger_id = nn.Embedding(NUM_FINGERS, 8)
    self.state_encoder = nn.Sequential(
      nn.Linear(self.per_finger_state_dimension + 8, 64),
      nn.SiLU(),
      nn.Linear(64, 64),
      nn.SiLU(),
    )
    self.finger_fusion = nn.Sequential(
      nn.Linear((32 + 64) * NUM_FINGERS, 256),
      nn.SiLU(),
    )
    self.wrist_history_encoder = TemporalVectorEncoder(18, 64)
    self.wrist_plan_encoder = TemporalVectorEncoder(6, 64)
    self.previous_command_encoder = nn.Sequential(
      nn.Linear(NUM_FINGER_JOINTS, 32),
      nn.SiLU(),
    )
    self.output = nn.Sequential(
      nn.Linear(256 + 64 + 64 + 32, condition_dimension),
      nn.SiLU(),
      nn.Linear(condition_dimension, condition_dimension),
    )
    self.condition_dimension = condition_dimension

  def forward(self, inputs: Mapping[str, Tensor]) -> Tensor:
    required = {
      "force_history",
      "finger_state_geometry",
      "wrist_real_twist_history",
      "wrist_mcc_offset_history",
      "wrist_mcc_velocity_history",
      "future_wrist_plan_twist",
      "previous_executed_command",
    }
    missing = required.difference(inputs)
    if missing:
      raise ValueError(f"condition inputs missing {sorted(missing)}")
    force = inputs["force_history"]
    state = inputs["finger_state_geometry"]
    if state.ndim != 3 or state.shape[1:] != (NUM_FINGERS, self.per_finger_state_dimension):
      raise ValueError("finger_state_geometry must have shape (B,4,20)")
    batch = state.shape[0]
    finger_ids = torch.arange(NUM_FINGERS, device=state.device)[None, :].expand(batch, -1)
    state_latent = self.state_encoder(torch.cat((state, self.finger_id(finger_ids)), dim=-1))
    force_latent = self.force_encoder(force)
    fingers = self.finger_fusion(torch.cat((force_latent, state_latent), dim=-1).flatten(1))
    wrist_history = torch.cat(
      (
        inputs["wrist_real_twist_history"],
        inputs["wrist_mcc_offset_history"],
        inputs["wrist_mcc_velocity_history"],
      ),
      dim=-1,
    )
    wrist = self.wrist_history_encoder(wrist_history)
    plan = self.wrist_plan_encoder(inputs["future_wrist_plan_twist"])
    previous = self.previous_command_encoder(inputs["previous_executed_command"])
    return self.output(torch.cat((fingers, wrist, plan, previous), dim=-1))


def observation_to_tensors(
  observation: FingerDPObservation,
  *,
  device: torch.device | str = "cpu",
) -> dict[str, Tensor]:
  """Convert one validated causal observation to a batch of size one."""

  def tensor(value: np.ndarray) -> Tensor:
    # Observation contracts intentionally expose read-only NumPy arrays.  Copy
    # before handing storage to Torch so no tensor operation can mutate them.
    return torch.as_tensor(
      np.array(value, dtype=np.float32, copy=True),
      dtype=torch.float32,
      device=device,
    ).unsqueeze(0)

  return {
    "force_history": tensor(observation.force_encoder_input()),
    "finger_state_geometry": tensor(observation.per_finger_state_geometry()),
    "wrist_real_twist_history": tensor(observation.wrist_real_twist_history),
    "wrist_mcc_offset_history": tensor(observation.wrist_mcc_offset_history),
    "wrist_mcc_velocity_history": tensor(observation.wrist_mcc_velocity_history),
    "future_wrist_plan_twist": tensor(observation.future_wrist_plan_twist),
    "previous_executed_command": tensor(observation.previous_executed_finger_command_rad),
  }


class DiffusionStepEmbedding(nn.Module):
  def __init__(self, dimension: int) -> None:
    super().__init__()
    self.dimension = dimension
    self.project = nn.Sequential(
      nn.Linear(dimension, dimension),
      nn.SiLU(),
      nn.Linear(dimension, dimension),
    )

  def forward(self, timestep: Tensor) -> Tensor:
    half = self.dimension // 2
    frequencies = torch.exp(
      -math.log(10000.0)
      * torch.arange(half, device=timestep.device, dtype=torch.float32)
      / max(half - 1, 1)
    )
    angles = timestep.float()[:, None] * frequencies[None, :]
    embedding = torch.cat((angles.sin(), angles.cos()), dim=1)
    if embedding.shape[1] < self.dimension:
      embedding = functional.pad(embedding, (0, self.dimension - embedding.shape[1]))
    return self.project(embedding)


class ConditionalActionBlock(nn.Module):
  def __init__(self, channels: int, condition_dimension: int, dilation: int) -> None:
    super().__init__()
    self.condition = nn.Linear(condition_dimension, channels)
    self.first = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
    self.second = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
    self.first_norm = nn.GroupNorm(1, channels)
    self.second_norm = nn.GroupNorm(1, channels)

  def forward(self, value: Tensor, condition: Tensor) -> Tensor:
    residual = value
    value = value + self.condition(condition)[:, :, None]
    value = functional.silu(self.first_norm(self.first(value)))
    value = self.second_norm(self.second(value))
    return functional.silu(value + residual)


class ConditionalActionDenoiser(nn.Module):
  def __init__(self, condition_dimension: int = 256, channels: int = 96) -> None:
    super().__init__()
    self.timestep = DiffusionStepEmbedding(condition_dimension)
    self.input = nn.Conv1d(NUM_FINGER_JOINTS, channels, 1)
    self.blocks = nn.ModuleList(
      ConditionalActionBlock(channels, condition_dimension, dilation)
      for dilation in (1, 2, 4, 8)
    )
    self.output = nn.Conv1d(channels, NUM_FINGER_JOINTS, 1)

  def forward(self, noisy_action: Tensor, timestep: Tensor, condition: Tensor) -> Tensor:
    if noisy_action.ndim != 3 or noisy_action.shape[-1] != NUM_FINGER_JOINTS:
      raise ValueError("noisy_action must have shape (B,H,16)")
    embedded = condition + self.timestep(timestep)
    value = self.input(noisy_action.transpose(1, 2))
    for block in self.blocks:
      value = block(value, embedded)
    return self.output(value).transpose(1, 2)


@dataclass(frozen=True, slots=True)
class DiffusionPolicyConfig:
  action_horizon_steps: int = ACTION_HORIZON_STEPS
  diffusion_steps: int = 50
  beta_start: float = 1e-4
  beta_end: float = 0.02
  max_abs_action_offset_rad: float = 0.30

  def __post_init__(self) -> None:
    if self.action_horizon_steps < 1 or self.diffusion_steps < 2:
      raise ValueError("action horizon and diffusion steps are too small")
    if not 0.0 < self.beta_start < self.beta_end < 1.0:
      raise ValueError("diffusion betas must satisfy 0 < start < end < 1")
    if self.max_abs_action_offset_rad <= 0.0:
      raise ValueError("max_abs_action_offset_rad must be positive")


class FingerDiffusionPolicy(nn.Module):
  """Trainable DP model; it contains no MCC or force-error fallback."""

  def __init__(self, config: DiffusionPolicyConfig | None = None) -> None:
    super().__init__()
    self.config = config or DiffusionPolicyConfig()
    self.condition_encoder = FingerDPConditionEncoder(256)
    self.denoiser = ConditionalActionDenoiser(256)
    betas = torch.linspace(
      self.config.beta_start,
      self.config.beta_end,
      self.config.diffusion_steps,
      dtype=torch.float32,
    )
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    self.register_buffer("betas", betas)
    self.register_buffer("alphas", alphas)
    self.register_buffer("alpha_bar", alpha_bar)

  def diffusion_loss(
    self,
    inputs: Mapping[str, Tensor],
    target_action_offsets: Tensor,
  ) -> Tensor:
    if target_action_offsets.ndim != 3 or target_action_offsets.shape[1:] != (
      self.config.action_horizon_steps,
      NUM_FINGER_JOINTS,
    ):
      raise ValueError("target_action_offsets has the wrong shape")
    condition = self.condition_encoder(inputs)
    batch = target_action_offsets.shape[0]
    timestep = torch.randint(
      0,
      self.config.diffusion_steps,
      (batch,),
      device=target_action_offsets.device,
    )
    noise = torch.randn_like(target_action_offsets)
    alpha_bar = self.alpha_bar[timestep][:, None, None]
    noisy = alpha_bar.sqrt() * target_action_offsets + (1.0 - alpha_bar).sqrt() * noise
    prediction = self.denoiser(noisy, timestep, condition)
    return functional.mse_loss(prediction, noise)

  @torch.no_grad()
  def sample(self, inputs: Mapping[str, Tensor]) -> Tensor:
    condition = self.condition_encoder(inputs)
    batch = condition.shape[0]
    value = torch.randn(
      batch,
      self.config.action_horizon_steps,
      NUM_FINGER_JOINTS,
      device=condition.device,
    )
    for step in reversed(range(self.config.diffusion_steps)):
      timestep = torch.full((batch,), step, device=condition.device, dtype=torch.long)
      predicted_noise = self.denoiser(value, timestep, condition)
      alpha = self.alphas[step]
      alpha_bar = self.alpha_bar[step]
      mean = (value - self.betas[step] * predicted_noise / (1.0 - alpha_bar).sqrt()) / alpha.sqrt()
      if step > 0:
        value = mean + self.betas[step].sqrt() * torch.randn_like(value)
      else:
        value = mean
    return torch.clamp(
      value,
      -self.config.max_abs_action_offset_rad,
      self.config.max_abs_action_offset_rad,
    )
