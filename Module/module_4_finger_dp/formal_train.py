"""CUDA-only formal Finger DP training on RAW_VERIFIED Dataset-I."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader

from Module.module_4_finger_dp.dataset_i_pipeline import (
  DATASET_I_SAMPLE_VERSION,
  DatasetISampleBundle,
  load_dataset_i_samples,
)
from Module.module_4_finger_dp.gpu_runtime import require_cuda, synchronize_cuda
from Module.module_4_finger_dp.policy import DiffusionPolicyConfig, FingerDiffusionPolicy
from Module.module_4_finger_dp.track_d_train import (
  TrackDOpenLoopMetrics,
  _TrackDTorchDataset,
  _ema_update,
  _move,
  open_loop_metrics,
  predict_track_d_samples,
)


FORMAL_DP_CHECKPOINT_VERSION = "fr3-leap-finger-dp-dataset-i.v1"


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class FormalDPTrainingConfig:
  seed: int = 20260824
  updates: int = 10000
  batch_size: int = 128
  learning_rate: float = 3e-4
  weight_decay: float = 1e-6
  gradient_clip_norm: float = 1.0
  ema_decay: float = 0.995
  diffusion_steps: int = 20
  beta_start: float = 1e-4
  beta_end: float = 0.20
  action_scale_rad: float = 0.10
  maximum_action_offset_rad: float = 0.20
  log_period_updates: int = 25
  maximum_validation_first_command_rmse_rad: float = 0.020
  device: str = "cuda:0"

  def __post_init__(self) -> None:
    if self.updates < 1 or self.batch_size < 1 or self.log_period_updates < 1:
      raise ValueError("training counts must be positive")
    if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
      raise ValueError("optimizer settings are invalid")
    if not 0.0 < self.ema_decay < 1.0:
      raise ValueError("ema_decay must be in (0,1)")
    if not self.device.startswith("cuda"):
      raise ValueError("formal DP training is CUDA-only")


@dataclass(frozen=True, slots=True)
class FormalDPTrainingResult:
  checkpoint_path: Path
  summary_path: Path
  history_path: Path
  train_metrics: TrackDOpenLoopMetrics
  validation_metrics: TrackDOpenLoopMetrics
  training_gate: str
  blocking_reason: tuple[str, ...]


def _model_config(config: FormalDPTrainingConfig) -> DiffusionPolicyConfig:
  return DiffusionPolicyConfig(
    diffusion_steps=config.diffusion_steps,
    beta_start=config.beta_start,
    beta_end=config.beta_end,
    action_scale_rad=config.action_scale_rad,
    max_abs_action_offset_rad=config.maximum_action_offset_rad,
  )


def _validate_splits(
  train: DatasetISampleBundle,
  validation: DatasetISampleBundle,
) -> dict[str, object]:
  train_objects = set(np.unique(train.object_id).tolist())
  validation_objects = set(np.unique(validation.object_id).tolist())
  train_episodes = set(np.unique(train.episode_id).tolist())
  validation_episodes = set(np.unique(validation.episode_id).tolist())
  checks = {
    "train_split_label": bool(np.all(train.split == "train")),
    "validation_split_label": bool(np.all(validation.split == "validation")),
    "object_disjoint": not bool(train_objects & validation_objects),
    "episode_disjoint": not bool(train_episodes & validation_episodes),
    "train_causal_audit": train.samples.audit.passed,
    "validation_causal_audit": validation.samples.audit.passed,
    "train_nonempty": train.samples.count > 0,
    "validation_nonempty": validation.samples.count > 0,
  }
  failed = [name for name, value in checks.items() if not value]
  if failed:
    raise RuntimeError(f"formal Dataset-I split audit failed: {failed}")
  return {
    "checks": checks,
    "train_objects": sorted(train_objects),
    "validation_objects": sorted(validation_objects),
    "train_episode_count": len(train_episodes),
    "validation_episode_count": len(validation_episodes),
    "train_sample_count": train.samples.count,
    "validation_sample_count": validation.samples.count,
  }


def train_formal_dp(
  train_path: str | Path,
  validation_path: str | Path,
  output_directory: str | Path,
  config: FormalDPTrainingConfig = FormalDPTrainingConfig(),
) -> FormalDPTrainingResult:
  train_source = Path(train_path)
  validation_source = Path(validation_path)
  train = load_dataset_i_samples(train_source)
  validation = load_dataset_i_samples(validation_source)
  split_audit = _validate_splits(train, validation)
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
  dataset = _TrackDTorchDataset(train.samples)
  loader = DataLoader(
    dataset,
    batch_size=min(config.batch_size, len(dataset)),
    shuffle=True,
    drop_last=False,
    generator=torch.Generator().manual_seed(config.seed),
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
    target = target.to(device=device, dtype=torch.float32, non_blocking=True)
    loss = model.diffusion_loss(inputs, target)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
    optimizer.step()
    _ema_update(ema_model, model, config.ema_decay)
    rolling.append(float(loss.detach()))
    if update == 1 or update % config.log_period_updates == 0:
      update_log.append(update)
      loss_log.append(float(np.mean(rolling)))
      rolling.clear()
  synchronize_cuda(device)
  elapsed = perf_counter() - begin
  train_prediction, train_latency = predict_track_d_samples(
    ema_model,
    train.samples,
    seed=config.seed + 1,
    batch_size=64,
    device=device,
  )
  validation_prediction, validation_latency = predict_track_d_samples(
    ema_model,
    validation.samples,
    seed=config.seed + 2,
    batch_size=64,
    device=device,
  )
  train_metrics = open_loop_metrics(train.samples, train_prediction, train_latency)
  validation_metrics = open_loop_metrics(
    validation.samples,
    validation_prediction,
    validation_latency,
  )
  checks = {
    "dataset_i_provenance": True,
    "cuda_training": device.type == "cuda",
    "validation_first_command_rmse": (
      validation_metrics.first_command_rmse_rad
      <= config.maximum_validation_first_command_rmse_rad
    ),
  }
  failed = tuple(name for name, passed in checks.items() if not passed)
  training_gate = "PASS" if not failed else "FAIL"
  checkpoint_path = output / "formal_finger_dp_checkpoint.pt"
  torch.save(
    {
      "checkpoint_version": FORMAL_DP_CHECKPOINT_VERSION,
      "dataset_schema_version": DATASET_I_SAMPLE_VERSION,
      "dataset_class": "DATASET_I_RAW_VERIFIED",
      "replay_repair_policy": "NONE",
      "formal_e05_authorized": training_gate == "PASS",
      "cuda_runtime": cuda_info.to_dict(),
      "model_config": asdict(ema_model.config),
      "training_config": asdict(config),
      "train_dataset_sha256": _sha256(train_source),
      "validation_dataset_sha256": _sha256(validation_source),
      "split_audit": split_audit,
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
  np.savez_compressed(
    output / "validation_predictions.npz",
    timestamp_s=validation.samples.timestamp_s,
    episode_id=validation.episode_id,
    object_id=validation.object_id,
    target_action_offsets_rad=validation.samples.target_action_offsets_rad,
    predicted_action_offsets_rad=validation_prediction,
    inference_latency_s=validation_latency,
  )
  summary_path = output / "training_summary.json"
  summary_path.write_text(
    json.dumps(
      {
        "stage": "FORMAL_DATASET_I_TRAINING",
        "dataset_class": "DATASET_I_RAW_VERIFIED",
        "replay_repair_policy": "NONE",
        "cuda_only": True,
        "cuda_runtime": cuda_info.to_dict(),
        "training_config": asdict(config),
        "model_config": asdict(ema_model.config),
        "split_audit": split_audit,
        "train_metrics": asdict(train_metrics),
        "validation_metrics": asdict(validation_metrics),
        "final_training_loss": loss_log[-1],
        "minimum_training_loss": min(loss_log),
        "elapsed_s": elapsed,
        "training_gate": {
          "status": training_gate,
          "blocking_reason": ("NONE",) if not failed else failed,
          "checks": checks,
        },
      },
      indent=2,
      sort_keys=True,
    ),
    encoding="utf-8",
  )
  return FormalDPTrainingResult(
    checkpoint_path=checkpoint_path,
    summary_path=summary_path,
    history_path=history_path,
    train_metrics=train_metrics,
    validation_metrics=validation_metrics,
    training_gate=training_gate,
    blocking_reason=("NONE",) if not failed else failed,
  )


def load_formal_dp_policy(
  checkpoint_path: str | Path,
  *,
  device: torch.device | str = "cuda:0",
) -> FingerDiffusionPolicy:
  resolved, _ = require_cuda(str(device))
  checkpoint = torch.load(checkpoint_path, map_location=resolved, weights_only=False)
  if checkpoint.get("checkpoint_version") != FORMAL_DP_CHECKPOINT_VERSION:
    raise ValueError("unsupported formal Finger DP checkpoint")
  if checkpoint.get("dataset_class") != "DATASET_I_RAW_VERIFIED":
    raise ValueError("formal loader refuses ambiguous teacher provenance")
  if checkpoint.get("replay_repair_policy") != "NONE":
    raise ValueError("formal loader refuses replay-repaired training data")
  if not checkpoint.get("formal_e05_authorized", False):
    raise ValueError("checkpoint did not pass the formal training gate")
  model = FingerDiffusionPolicy(
    DiffusionPolicyConfig(**checkpoint["model_config"])
  ).to(resolved)
  model.load_state_dict(checkpoint["state_dict"], strict=True)
  model.eval()
  return model


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--train", type=Path, required=True)
  parser.add_argument("--validation", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--updates", type=int, default=10000)
  parser.add_argument("--batch-size", type=int, default=128)
  parser.add_argument("--device", default="cuda:0")
  args = parser.parse_args()
  result = train_formal_dp(
    args.train,
    args.validation,
    args.output,
    FormalDPTrainingConfig(
      updates=args.updates,
      batch_size=args.batch_size,
      device=args.device,
    ),
  )
  print(
    json.dumps(
      {
        "training_gate": result.training_gate,
        "blocking_reason": result.blocking_reason,
        "checkpoint": str(result.checkpoint_path),
        "summary": str(result.summary_path),
      },
      indent=2,
    )
  )


if __name__ == "__main__":
  main()
