"""CUDA-only training and open-loop audit for the two-head DPRef policy."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from Module.module_4_finger_dp.dpref_dataset import SCHEMA_VERSION as DATASET_SCHEMA
from Module.module_4_finger_dp.dpref_policy import (
  DPREF_INPUT_NAMES,
  DPRefPolicyConfig,
  FingerDPRefPolicy,
)
from Module.module_4_finger_dp.gpu_runtime import require_cuda, synchronize_cuda
from Module.module_4_whole_hand_mcc.reference_interpreter import ContactRole


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = REPO_ROOT / "Module/generated/dpref_v1/relabelled_dataset_i"
DEFAULT_OUTPUT = REPO_ROOT / "Module/generated/dpref_v1/training"
CHECKPOINT_VERSION = "fr3-leap-finger-dpref-role.v1"


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1 << 20), b""):
      digest.update(block)
  return digest.hexdigest()


def _metadata(path: Path) -> dict[str, Any]:
  value = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
  if value.get("schema_version") != DATASET_SCHEMA:
    raise ValueError("unsupported DPRef dataset schema")
  if value.get("source_dataset_class") != "DATASET_I_RAW_VERIFIED":
    raise ValueError("DPRef trainer refuses non-RAW_VERIFIED provenance")
  if value.get("replay_repair_policy") != "NONE":
    raise ValueError("DPRef trainer refuses replay-repaired data")
  return value


class DPRefTorchDataset(Dataset[tuple[dict[str, Tensor], Tensor, Tensor, Tensor]]):
  def __init__(self, path: str | Path) -> None:
    self.path = Path(path)
    self.metadata = _metadata(self.path)
    with np.load(self.path, allow_pickle=False) as archive:
      self.inputs = {
        name: torch.from_numpy(np.array(archive[name], dtype=np.float32, copy=True))
        for name in DPREF_INPUT_NAMES
      }
      self.target = torch.from_numpy(
        np.array(archive["target_nominal_offsets_rad"], dtype=np.float32, copy=True)
      )
      self.role = torch.from_numpy(
        np.array(archive["target_role"], dtype=np.int64, copy=True)
      )
      self.valid = torch.from_numpy(
        np.array(archive["role_label_valid"], dtype=np.bool_, copy=True)
      )
      self.episode_id = np.array(archive["episode_id"], copy=True)
      self.object_id = np.array(archive["object_id"], copy=True)
      self.split = np.array(archive["split"], copy=True)
      self.timestamp_s = np.array(archive["timestamp_s"], copy=True)
      self.q_meas_rad = np.array(archive["q_meas_rad"], copy=True)

  def __len__(self) -> int:
    return len(self.target)

  def __getitem__(self, index: int) -> tuple[dict[str, Tensor], Tensor, Tensor, Tensor]:
    return (
      {name: value[index] for name, value in self.inputs.items()},
      self.target[index],
      self.role[index],
      self.valid[index],
    )


def _to_device(value: Mapping[str, Tensor], device: torch.device) -> dict[str, Tensor]:
  return {
    name: tensor.to(device=device, dtype=torch.float32, non_blocking=True)
    for name, tensor in value.items()
  }


def _ema_update(target: torch.nn.Module, source: torch.nn.Module, decay: float) -> None:
  with torch.no_grad():
    for target_parameter, source_parameter in zip(target.parameters(), source.parameters()):
      target_parameter.mul_(decay).add_(source_parameter, alpha=1.0 - decay)
    for target_buffer, source_buffer in zip(target.buffers(), source.buffers()):
      target_buffer.copy_(source_buffer)


def _role_weights(dataset: DPRefTorchDataset) -> tuple[Tensor, dict[str, int]]:
  counts = torch.zeros(len(ContactRole), dtype=torch.float64)
  for role in ContactRole:
    counts[int(role)] = torch.count_nonzero(
      dataset.valid & (dataset.role == int(role))
    )
  if torch.any(counts == 0):
    missing = [ContactRole(i).name for i, count in enumerate(counts) if count == 0]
    raise RuntimeError(f"training split has no valid labels for {missing}")
  weights = torch.rsqrt(counts)
  weights /= weights.mean()
  weights = torch.clamp(weights, max=12.0).to(torch.float32)
  return weights, {ContactRole(i).name: int(value) for i, value in enumerate(counts)}


@dataclass(frozen=True, slots=True)
class DPRefTrainingConfig:
  seed: int = 20260824
  updates: int = 10000
  batch_size: int = 128
  learning_rate: float = 3e-4
  weight_decay: float = 1e-6
  gradient_clip_norm: float = 1.0
  ema_decay: float = 0.995
  log_period_updates: int = 50
  validation_batch_size: int = 128
  maximum_validation_first_command_rmse_rad: float = 0.025
  minimum_validation_observed_role_accuracy: float = 0.80
  device: str = "cuda:0"

  def __post_init__(self) -> None:
    if not self.device.startswith("cuda"):
      raise ValueError("formal DPRef training is CUDA-only")
    if self.updates < 1 or self.batch_size < 1 or self.log_period_updates < 1:
      raise ValueError("training counts must be positive")
    if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
      raise ValueError("invalid optimizer configuration")
    if not 0.0 < self.ema_decay < 1.0:
      raise ValueError("ema_decay must be in (0,1)")


@torch.no_grad()
def evaluate_open_loop(
  model: FingerDPRefPolicy,
  dataset: DPRefTorchDataset,
  *,
  device: torch.device,
  batch_size: int,
  seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
  loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
  generator = torch.Generator(device=device).manual_seed(seed)
  predictions: list[np.ndarray] = []
  roles: list[np.ndarray] = []
  probabilities: list[np.ndarray] = []
  latencies: list[float] = []
  model.eval()
  for inputs, _, _, _ in loader:
    batch_inputs = _to_device(inputs, device)
    noise = torch.randn(
      len(next(iter(batch_inputs.values()))),
      model.config.action_horizon_steps,
      16,
      generator=generator,
      device=device,
    )
    synchronize_cuda(device)
    start = perf_counter()
    offsets, role, probability = model.sample(
      batch_inputs,
      initial_noise=noise,
      deterministic=True,
    )
    synchronize_cuda(device)
    elapsed = perf_counter() - start
    latencies.extend([elapsed / len(offsets)] * len(offsets))
    predictions.append(offsets.cpu().numpy())
    roles.append(role.cpu().numpy())
    probabilities.append(probability.cpu().numpy())
  prediction = np.concatenate(predictions)
  role_prediction = np.concatenate(roles)
  probability = np.concatenate(probabilities)
  target = dataset.target.numpy()
  valid = dataset.valid.numpy()
  role_target = dataset.role.numpy()
  error = prediction - target
  role_accuracy = float(np.mean(role_prediction[valid] == role_target[valid]))
  per_class: dict[str, Any] = {}
  observed_accuracies: list[float] = []
  missing_classes: list[str] = []
  for role in ContactRole:
    mask = valid & (role_target == int(role))
    count = int(np.count_nonzero(mask))
    if count:
      accuracy = float(np.mean(role_prediction[mask] == int(role)))
      observed_accuracies.append(accuracy)
    else:
      accuracy = None
      missing_classes.append(role.name)
    per_class[role.name] = {"count": count, "accuracy": accuracy}
  metrics = {
    "sample_count": len(dataset),
    "first_command_rmse_rad": float(np.sqrt(np.mean(error[:, 0] ** 2))),
    "chunk_rmse_rad": float(np.sqrt(np.mean(error**2))),
    "first_command_mae_rad": float(np.mean(np.abs(error[:, 0]))),
    "role_accuracy": role_accuracy,
    "minimum_observed_role_accuracy": float(min(observed_accuracies)),
    "role_per_class": per_class,
    "missing_role_classes": missing_classes,
    "inference_latency_mean_s": float(np.mean(latencies)),
    "inference_latency_p95_s": float(np.percentile(latencies, 95.0)),
  }
  outputs = {
    "predicted_nominal_offsets_rad": prediction,
    "predicted_role": role_prediction,
    "predicted_role_probability": probability,
    "inference_latency_s": np.asarray(latencies),
  }
  return metrics, outputs


def train_dpref(
  train_path: str | Path,
  validation_path: str | Path,
  output_directory: str | Path,
  config: DPRefTrainingConfig = DPRefTrainingConfig(),
) -> dict[str, Any]:
  train_source = Path(train_path)
  validation_source = Path(validation_path)
  train = DPRefTorchDataset(train_source)
  validation = DPRefTorchDataset(validation_source)
  train_objects = set(train.object_id.tolist())
  validation_objects = set(validation.object_id.tolist())
  train_episodes = set(train.episode_id.tolist())
  validation_episodes = set(validation.episode_id.tolist())
  split_checks = {
    "train_labels": bool(np.all(train.split == "train")),
    "validation_labels": bool(np.all(validation.split == "validation")),
    "object_disjoint": not bool(train_objects & validation_objects),
    "episode_disjoint": not bool(train_episodes & validation_episodes),
  }
  if not all(split_checks.values()):
    raise RuntimeError(f"DPRef split audit failed: {split_checks}")
  device, cuda_info = require_cuda(config.device)
  torch.manual_seed(config.seed)
  torch.cuda.manual_seed_all(config.seed)
  np.random.seed(config.seed)
  model = FingerDPRefPolicy(DPRefPolicyConfig()).to(device)
  ema_model = deepcopy(model).to(device).eval()
  optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.learning_rate,
    weight_decay=config.weight_decay,
  )
  weights, train_role_counts = _role_weights(train)
  weights = weights.to(device)
  loader = DataLoader(
    train,
    batch_size=min(config.batch_size, len(train)),
    shuffle=True,
    pin_memory=True,
    generator=torch.Generator().manual_seed(config.seed),
  )
  iterator = iter(loader)
  history = {"update": [], "total": [], "diffusion": [], "role": []}
  rolling_total: list[float] = []
  rolling_diffusion: list[float] = []
  rolling_role: list[float] = []
  model.train()
  start = perf_counter()
  for update in range(1, config.updates + 1):
    try:
      inputs, target, role, valid = next(iterator)
    except StopIteration:
      iterator = iter(loader)
      inputs, target, role, valid = next(iterator)
    batch_inputs = _to_device(inputs, device)
    target = target.to(device=device, dtype=torch.float32, non_blocking=True)
    role = role.to(device=device, dtype=torch.long, non_blocking=True)
    valid = valid.to(device=device, dtype=torch.bool, non_blocking=True)
    loss = model.loss(
      batch_inputs,
      target,
      role,
      valid,
      role_class_weights=weights,
    )
    optimizer.zero_grad(set_to_none=True)
    loss.total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
    optimizer.step()
    _ema_update(ema_model, model, config.ema_decay)
    rolling_total.append(float(loss.total.detach()))
    rolling_diffusion.append(float(loss.diffusion.detach()))
    rolling_role.append(float(loss.role.detach()))
    if update == 1 or update % config.log_period_updates == 0:
      history["update"].append(update)
      history["total"].append(float(np.mean(rolling_total)))
      history["diffusion"].append(float(np.mean(rolling_diffusion)))
      history["role"].append(float(np.mean(rolling_role)))
      rolling_total.clear()
      rolling_diffusion.clear()
      rolling_role.clear()
  synchronize_cuda(device)
  elapsed = perf_counter() - start
  train_metrics, _ = evaluate_open_loop(
    ema_model,
    train,
    device=device,
    batch_size=config.validation_batch_size,
    seed=config.seed + 1,
  )
  validation_metrics, predictions = evaluate_open_loop(
    ema_model,
    validation,
    device=device,
    batch_size=config.validation_batch_size,
    seed=config.seed + 2,
  )
  checks = {
    "cuda_training": device.type == "cuda",
    "validation_first_command_rmse": (
      validation_metrics["first_command_rmse_rad"]
      <= config.maximum_validation_first_command_rmse_rad
    ),
    "validation_observed_role_accuracy": (
      validation_metrics["minimum_observed_role_accuracy"]
      >= config.minimum_validation_observed_role_accuracy
    ),
  }
  status = "PASS" if all(checks.values()) else "FAIL"
  role_coverage = {
    "all_roles_present_in_train": all(value > 0 for value in train_role_counts.values()),
    "all_roles_present_in_validation": not validation_metrics["missing_role_classes"],
    "missing_validation_roles": validation_metrics["missing_role_classes"],
    "handover_generalization_claim_allowed": not validation_metrics["missing_role_classes"],
  }
  output = Path(output_directory)
  output.mkdir(parents=True, exist_ok=True)
  checkpoint_path = output / "dpref_checkpoint.pt"
  torch.save(
    {
      "checkpoint_version": CHECKPOINT_VERSION,
      "dataset_schema_version": DATASET_SCHEMA,
      "dataset_class": "DATASET_I_RAW_VERIFIED_RELABELLED_DPREF",
      "replay_repair_policy": "NONE",
      "training_gate": status,
      "role_coverage": role_coverage,
      "cuda_runtime": cuda_info.to_dict(),
      "policy_config": asdict(ema_model.config),
      "training_config": asdict(config),
      "train_dataset_sha256": _sha256(train_source),
      "validation_dataset_sha256": _sha256(validation_source),
      "state_dict": ema_model.state_dict(),
    },
    checkpoint_path,
  )
  np.savez_compressed(
    output / "training_history.npz",
    **{name: np.asarray(values) for name, values in history.items()},
  )
  np.savez_compressed(
    output / "validation_predictions.npz",
    timestamp_s=validation.timestamp_s,
    episode_id=validation.episode_id,
    object_id=validation.object_id,
    target_nominal_offsets_rad=validation.target.numpy(),
    target_role=validation.role.numpy(),
    role_label_valid=validation.valid.numpy(),
    **predictions,
  )
  summary = {
    "stage": "DPREF_TWO_HEAD_TRAINING",
    "status": status,
    "blocking_reason": [name for name, passed in checks.items() if not passed] or ["NONE"],
    "checks": checks,
    "role_coverage": role_coverage,
    "cuda_runtime": cuda_info.to_dict(),
    "training_config": asdict(config),
    "policy_config": asdict(ema_model.config),
    "split_audit": {
      "checks": split_checks,
      "train_objects": sorted(train_objects),
      "validation_objects": sorted(validation_objects),
      "train_episodes": len(train_episodes),
      "validation_episodes": len(validation_episodes),
    },
    "train_role_counts": train_role_counts,
    "train_metrics": train_metrics,
    "validation_metrics": validation_metrics,
    "training_elapsed_s": elapsed,
    "final_losses": {name: values[-1] for name, values in history.items() if name != "update"},
  }
  (output / "training_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True),
    encoding="utf-8",
  )
  return summary


def load_dpref_policy(
  checkpoint_path: str | Path,
  *,
  device: str = "cuda:0",
  require_training_pass: bool = True,
) -> FingerDPRefPolicy:
  resolved, _ = require_cuda(device)
  checkpoint = torch.load(checkpoint_path, map_location=resolved, weights_only=False)
  if checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION:
    raise ValueError("unsupported DPRef checkpoint")
  if checkpoint.get("replay_repair_policy") != "NONE":
    raise ValueError("DPRef loader refuses replay-repaired provenance")
  if require_training_pass and checkpoint.get("training_gate") != "PASS":
    raise ValueError("DPRef checkpoint did not pass its training gate")
  model = FingerDPRefPolicy(DPRefPolicyConfig(**checkpoint["policy_config"])).to(resolved)
  model.load_state_dict(checkpoint["state_dict"], strict=True)
  model.eval()
  return model


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--train", type=Path, default=DEFAULT_DATA / "dpref_i100_train.npz")
  parser.add_argument(
    "--validation",
    type=Path,
    default=DEFAULT_DATA / "dpref_validation.npz",
  )
  parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
  parser.add_argument("--updates", type=int, default=10000)
  parser.add_argument("--batch-size", type=int, default=128)
  parser.add_argument("--device", default="cuda:0")
  args = parser.parse_args()
  result = train_dpref(
    args.train,
    args.validation,
    args.output,
    DPRefTrainingConfig(
      updates=args.updates,
      batch_size=args.batch_size,
      device=args.device,
    ),
  )
  print(json.dumps({"status": result["status"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
  main()
