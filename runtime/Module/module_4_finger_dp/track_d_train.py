"""Intentional Dataset-D overfit used only to validate the DP pipeline."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Mapping

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from Module.module_4_finger_dp.policy import (
  DiffusionPolicyConfig,
  FingerDiffusionPolicy,
)
from Module.module_4_finger_dp.gpu_runtime import require_cuda, synchronize_cuda
from Module.module_4_finger_dp.track_d_dataset import (
  TRACK_D_INPUT_NAMES,
  TRACK_D_SAMPLE_SCHEMA_VERSION,
  TrackDSamples,
)


TRACK_D_CHECKPOINT_VERSION = "fr3-leap-track-d-checkpoint.v1"


@dataclass(frozen=True, slots=True)
class TrackDTrainingConfig:
  seed: int = 20260823
  updates: int = 4000
  batch_size: int = 64
  learning_rate: float = 5e-4
  weight_decay: float = 1e-6
  gradient_clip_norm: float = 1.0
  ema_decay: float = 0.99
  diffusion_steps: int = 20
  beta_start: float = 1e-4
  beta_end: float = 0.20
  action_scale_rad: float = 0.10
  maximum_action_offset_rad: float = 0.20
  log_period_updates: int = 20
  device: str = "cuda:0"

  def __post_init__(self) -> None:
    if self.updates < 1 or self.batch_size < 1 or self.log_period_updates < 1:
      raise ValueError("training counts must be positive")
    if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
      raise ValueError("optimizer values are invalid")
    if not 0.0 < self.ema_decay < 1.0:
      raise ValueError("ema_decay must be in (0,1)")
    if not self.device.startswith("cuda"):
      raise ValueError("Track-D training is CUDA-only")


@dataclass(frozen=True, slots=True)
class TrackDOpenLoopMetrics:
  full_chunk_rmse_rad: float
  full_chunk_mae_rad: float
  full_chunk_maximum_error_rad: float
  first_command_rmse_rad: float
  first_command_mae_rad: float
  first_command_maximum_error_rad: float
  predicted_first_seam_rmse_rad: float
  teacher_first_seam_rmse_rad: float
  inference_latency_mean_s: float
  inference_latency_p95_s: float
  sample_count: int


@dataclass(frozen=True, slots=True)
class TrackDTrainingResult:
  checkpoint_path: Path
  history_path: Path
  predictions_path: Path
  summary_path: Path
  metrics: TrackDOpenLoopMetrics
  final_training_loss: float
  minimum_training_loss: float
  elapsed_s: float


class _TrackDTorchDataset(Dataset[tuple[dict[str, Tensor], Tensor]]):
  def __init__(self, samples: TrackDSamples) -> None:
    self.inputs = {
      name: torch.from_numpy(np.array(samples.inputs[name], copy=True))
      for name in TRACK_D_INPUT_NAMES
    }
    self.targets = torch.from_numpy(
      np.array(samples.target_action_offsets_rad, copy=True)
    )

  def __len__(self) -> int:
    return len(self.targets)

  def __getitem__(self, index: int) -> tuple[dict[str, Tensor], Tensor]:
    return ({name: value[index] for name, value in self.inputs.items()}, self.targets[index])


def _model_config(config: TrackDTrainingConfig) -> DiffusionPolicyConfig:
  return DiffusionPolicyConfig(
    diffusion_steps=config.diffusion_steps,
    beta_start=config.beta_start,
    beta_end=config.beta_end,
    action_scale_rad=config.action_scale_rad,
    max_abs_action_offset_rad=config.maximum_action_offset_rad,
  )


def _move(inputs: Mapping[str, Tensor], device: torch.device) -> dict[str, Tensor]:
  return {
    name: value.to(
      device=device,
      dtype=torch.float32,
      non_blocking=True,
    )
    for name, value in inputs.items()
  }


def _ema_update(target: FingerDiffusionPolicy, source: FingerDiffusionPolicy, decay: float) -> None:
  with torch.no_grad():
    for target_value, source_value in zip(target.parameters(), source.parameters()):
      target_value.mul_(decay).add_(source_value, alpha=1.0 - decay)
    for target_buffer, source_buffer in zip(target.buffers(), source.buffers()):
      target_buffer.copy_(source_buffer)


@torch.no_grad()
def predict_track_d_samples(
  model: FingerDiffusionPolicy,
  samples: TrackDSamples,
  *,
  seed: int,
  batch_size: int = 32,
  device: torch.device | str = "cuda:0",
) -> tuple[NDArray[np.float32], NDArray[np.float64]]:
  model.eval()
  resolved, _ = require_cuda(str(device))
  generator = torch.Generator(device=resolved)
  generator.manual_seed(seed)
  predictions: list[NDArray[np.float32]] = []
  latency: list[float] = []
  for start in range(0, samples.count, batch_size):
    stop = min(start + batch_size, samples.count)
    inputs = {
      name: torch.from_numpy(
        np.array(samples.inputs[name][start:stop], copy=True)
      ).to(resolved)
      for name in TRACK_D_INPUT_NAMES
    }
    noise = torch.randn(
      stop - start,
      model.config.action_horizon_steps,
      16,
      generator=generator,
      dtype=torch.float32,
      device=resolved,
    )
    synchronize_cuda(resolved)
    begin = perf_counter()
    prediction = model.sample(
      inputs,
      initial_noise=noise,
      deterministic=True,
    )
    synchronize_cuda(resolved)
    elapsed = perf_counter() - begin
    predictions.append(prediction.cpu().numpy().astype(np.float32))
    latency.extend([elapsed / (stop - start)] * (stop - start))
  return np.concatenate(predictions, axis=0), np.asarray(latency, dtype=np.float64)


def open_loop_metrics(
  samples: TrackDSamples,
  predicted_offsets_rad: NDArray[np.float32],
  inference_latency_s: NDArray[np.float64],
) -> TrackDOpenLoopMetrics:
  prediction = np.asarray(predicted_offsets_rad, dtype=np.float64)
  target = np.asarray(samples.target_action_offsets_rad, dtype=np.float64)
  if prediction.shape != target.shape:
    raise ValueError("open-loop prediction has the wrong shape")
  error = prediction - target
  first_error = error[:, 0]
  previous = np.asarray(samples.inputs["previous_executed_command"], dtype=np.float64)
  predicted_first = np.asarray(samples.anchor_q_meas_rad) + prediction[:, 0]
  teacher_first = np.asarray(samples.future_teacher_command_rad)[:, 0]
  predicted_seam = predicted_first - previous
  teacher_seam = teacher_first - previous
  return TrackDOpenLoopMetrics(
    full_chunk_rmse_rad=float(np.sqrt(np.mean(error**2))),
    full_chunk_mae_rad=float(np.mean(np.abs(error))),
    full_chunk_maximum_error_rad=float(np.max(np.abs(error))),
    first_command_rmse_rad=float(np.sqrt(np.mean(first_error**2))),
    first_command_mae_rad=float(np.mean(np.abs(first_error))),
    first_command_maximum_error_rad=float(np.max(np.abs(first_error))),
    predicted_first_seam_rmse_rad=float(np.sqrt(np.mean(predicted_seam**2))),
    teacher_first_seam_rmse_rad=float(np.sqrt(np.mean(teacher_seam**2))),
    inference_latency_mean_s=float(np.mean(inference_latency_s)),
    inference_latency_p95_s=float(np.quantile(inference_latency_s, 0.95)),
    sample_count=samples.count,
  )


def train_track_d_policy(
  samples: TrackDSamples,
  output_directory: str | Path,
  config: TrackDTrainingConfig = TrackDTrainingConfig(),
) -> TrackDTrainingResult:
  """Overfit Dataset-D and emit auditable predictions/checkpoint artefacts."""

  if not samples.audit.passed:
    raise RuntimeError(f"causal audit failed: {samples.audit.reasons}")
  output = Path(output_directory)
  output.mkdir(parents=True, exist_ok=True)
  device, cuda_info = require_cuda(config.device)
  torch.manual_seed(config.seed)
  torch.cuda.manual_seed_all(config.seed)
  np.random.seed(config.seed)
  model = FingerDiffusionPolicy(_model_config(config)).to(device)
  ema_model = deepcopy(model).to(device)
  ema_model.eval()
  optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.learning_rate,
    weight_decay=config.weight_decay,
  )
  dataset = _TrackDTorchDataset(samples)
  loader_generator = torch.Generator().manual_seed(config.seed)
  loader = DataLoader(
    dataset,
    batch_size=min(config.batch_size, len(dataset)),
    shuffle=True,
    drop_last=False,
    generator=loader_generator,
    pin_memory=True,
  )
  iterator = iter(loader)
  update_log: list[int] = []
  loss_log: list[float] = []
  rolling: list[float] = []
  begin = perf_counter()
  model.train()
  for update in range(1, config.updates + 1):
    try:
      inputs, target = next(iterator)
    except StopIteration:
      iterator = iter(loader)
      inputs, target = next(iterator)
    inputs = _move(inputs, device)
    target = target.to(
      device=device,
      dtype=torch.float32,
      non_blocking=True,
    )
    loss = model.diffusion_loss(inputs, target)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
    optimizer.step()
    _ema_update(ema_model, model, config.ema_decay)
    rolling.append(float(loss.detach()))
    if update % config.log_period_updates == 0 or update == 1:
      update_log.append(update)
      loss_log.append(float(np.mean(rolling)))
      rolling.clear()
  synchronize_cuda(device)
  elapsed = perf_counter() - begin

  prediction, latency = predict_track_d_samples(
    ema_model,
    samples,
    seed=config.seed + 1,
    device=device,
  )
  metrics = open_loop_metrics(samples, prediction, latency)
  checkpoint_path = output / "track_d_overfit_checkpoint.pt"
  torch.save(
    {
      "checkpoint_version": TRACK_D_CHECKPOINT_VERSION,
      "dataset_schema_version": TRACK_D_SAMPLE_SCHEMA_VERSION,
      "dataset_class": "DATASET_D_DIAGNOSTIC",
      "training_authorization": "D_GATE_ONLY",
      "generalization_claim_allowed": False,
      "cuda_runtime": cuda_info.to_dict(),
      "model_config": asdict(ema_model.config),
      "training_config": asdict(config),
      "state_dict": ema_model.state_dict(),
    },
    checkpoint_path,
  )
  history_path = output / "training_history.npz"
  np.savez_compressed(
    history_path,
    update=np.asarray(update_log, dtype=np.int64),
    loss=np.asarray(loss_log, dtype=np.float64),
  )
  predictions_path = output / "open_loop_predictions.npz"
  np.savez_compressed(
    predictions_path,
    timestamp_s=samples.timestamp_s,
    source_raw_index=samples.source_raw_index,
    target_action_offsets_rad=samples.target_action_offsets_rad,
    predicted_action_offsets_rad=prediction,
    anchor_q_meas_rad=samples.anchor_q_meas_rad,
    previous_executed_command_rad=samples.inputs["previous_executed_command"],
    inference_latency_s=latency,
  )
  summary_path = output / "open_loop_summary.json"
  summary = {
    "stage": "D1_OPEN_LOOP_OVERFIT",
    "dataset_class": "DATASET_D_DIAGNOSTIC",
    "generalization_evaluated": False,
    "formal_dataset_i_training": False,
    "cuda_only": True,
    "cuda_runtime": cuda_info.to_dict(),
    "training_config": asdict(config),
    "model_config": asdict(ema_model.config),
    "causal_audit": asdict(samples.audit),
    "metrics": asdict(metrics),
    "final_training_loss": loss_log[-1],
    "minimum_training_loss": min(loss_log),
    "elapsed_s": elapsed,
  }
  summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
  return TrackDTrainingResult(
    checkpoint_path=checkpoint_path,
    history_path=history_path,
    predictions_path=predictions_path,
    summary_path=summary_path,
    metrics=metrics,
    final_training_loss=loss_log[-1],
    minimum_training_loss=min(loss_log),
    elapsed_s=elapsed,
  )


def load_track_d_policy(
  checkpoint_path: str | Path,
  *,
  device: torch.device | str = "cuda:0",
) -> FingerDiffusionPolicy:
  resolved, _ = require_cuda(str(device))
  checkpoint = torch.load(checkpoint_path, map_location=resolved, weights_only=False)
  if checkpoint.get("checkpoint_version") != TRACK_D_CHECKPOINT_VERSION:
    raise ValueError("unsupported Track-D checkpoint")
  if checkpoint.get("dataset_class") != "DATASET_D_DIAGNOSTIC":
    raise ValueError("Track-D loader refuses a checkpoint with ambiguous provenance")
  config = DiffusionPolicyConfig(**checkpoint["model_config"])
  model = FingerDiffusionPolicy(config).to(resolved)
  model.load_state_dict(checkpoint["state_dict"], strict=True)
  model.eval()
  return model
