"""
Diffusion Policy Full Training for Finger Compliance Control.

Loads headless dataset with quality filtering, trains DiffusionPolicy,
saves in LeRobot pretrained format for inference.

Usage:
  # Full training with default settings
  python finger_compliance_control/scripts/dp_train.py

  # Quick overfit test with small model
  python finger_compliance_control/scripts/dp_train.py --small --steps 500 --no-filter

  # Resume from checkpoint
  python finger_compliance_control/scripts/dp_train.py --resume <output_dir>

  # Custom window and filtering
  python finger_compliance_control/scripts/dp_train.py \\
      --n-obs-steps 16 --horizon 16 --n-action-steps 4 \\
      --delta-percentile 90 --filter-ratio 0.7
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors

# Shared data module (same directory)
from dp_dataset import (
    FilterConfig,
    FilteredWindowDataset,
    DataDict,
    compute_stats,
    load_data,
)

# ── Paths ────────────────────────────────────────────────────────────────────
H5_PATH = Path(
    "./finger_compliance_control/data/train_dp/"
    "collect_20260609_215735.h5"
)
OUTPUT_ROOT = Path(
    "./finger_compliance_control/data/models"
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Config ────────────────────────────────────────────────────────────────────
class TrainConfig:
    """Training hyperparameters — all overridable via CLI."""

    def __init__(
        self,
        # Data
        h5_path: str = str(H5_PATH),
        stride: int = 1,
        clip_fsr_pct: float = 99.9,
        val_envs: tuple[int, ...] = (30, 31),
        # Window
        n_obs_steps: int = 16,
        horizon: int = 24,               # must be >= n_obs_steps + n_action_steps - 1
        n_action_steps: int = 4,
        # Filter
        filter_enabled: bool = True,
        min_fingers_in_contact: int = 3,
        delta_percentile: float = 90.0,
        delta_threshold: float | None = None,
        filter_ratio: float = 0.7,
        min_action_norm: float = 0.01,
        min_action_std: float = 0.005,
        # Training
        training_steps: int = 100_000,
        batch_size: int = 256,
        lr: float = 3e-4,
        weight_decay: float = 1e-5,
        grad_clip_norm: float = 1.0,
        # UNet
        down_dims: tuple[int, ...] = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
        diffusion_step_embed_dim: int = 128,
        num_inference_steps: int = 100,
        noise_scheduler_type: str = "DDPM",
        prediction_type: str = "sample",
        # Logging
        log_interval: int = 100,
        val_interval: int = 2000,
        save_interval: int = 10_000,
        num_workers: int = 4,
        output_dir: str | None = None,
        state_dim: int = 69,
        action_dim: int = 22,
    ):
        self.h5_path = h5_path
        self.stride = stride
        self.clip_fsr_pct = clip_fsr_pct
        self.val_envs = val_envs
        self.n_obs_steps = n_obs_steps
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.filter_enabled = filter_enabled
        self.min_fingers_in_contact = min_fingers_in_contact
        self.delta_percentile = delta_percentile
        self.delta_threshold = delta_threshold
        self.filter_ratio = filter_ratio
        self.min_action_norm = min_action_norm
        self.min_action_std = min_action_std
        self.training_steps = training_steps
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.grad_clip_norm = grad_clip_norm
        self.down_dims = down_dims
        self.kernel_size = kernel_size
        self.n_groups = n_groups
        self.diffusion_step_embed_dim = diffusion_step_embed_dim
        self.num_inference_steps = num_inference_steps
        self.noise_scheduler_type = noise_scheduler_type
        self.prediction_type = prediction_type
        self.log_interval = log_interval
        self.val_interval = val_interval
        self.save_interval = save_interval
        self.num_workers = num_workers
        self.output_dir = output_dir
        self.state_dim = state_dim
        self.action_dim = action_dim


# ── Model ─────────────────────────────────────────────────────────────────────

def build_policy(cfg: TrainConfig) -> DiffusionPolicy:
    """Build DiffusionPolicy from config."""
    input_features = {
        "observation.state": PolicyFeature(
            type=FeatureType.STATE, shape=(cfg.state_dim,)
        ),
        "observation.environment_state": PolicyFeature(
            type=FeatureType.ENV, shape=(1,)
        ),
    }
    output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(cfg.action_dim,))
    }

    policy_cfg = DiffusionConfig(
        input_features=input_features,
        output_features=output_features,
        n_obs_steps=cfg.n_obs_steps,
        horizon=cfg.horizon,
        n_action_steps=cfg.n_action_steps,
        device=DEVICE,
        down_dims=cfg.down_dims,
        kernel_size=cfg.kernel_size,
        n_groups=cfg.n_groups,
        diffusion_step_embed_dim=cfg.diffusion_step_embed_dim,
        num_inference_steps=cfg.num_inference_steps,
        noise_scheduler_type=cfg.noise_scheduler_type,
        prediction_type=cfg.prediction_type,
    )
    return DiffusionPolicy(policy_cfg).to(DEVICE)


# ── Metrics ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_q_metrics(
    policy: DiffusionPolicy,
    dataloader: DataLoader,
    preprocessor,
    n_action_steps: int,
    max_batches: int = 50,
) -> dict[str, float]:
    """Compute action prediction R² and MAE on validation data.

    Bypasses select_action's queue logic (which is designed for online
    single-step inference). Instead calls diffusion.generate_actions()
    directly with observation-only keys, which is correct for offline
    evaluation with full windowed batches.
    """
    policy.eval()
    all_preds = []
    all_targets = []

    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        batch = preprocessor(batch)

        # Pass only observation keys — extra keys (action, next.reward, etc.)
        # from the preprocessor break generate_actions.
        obs_batch = {
            "observation.state": batch["observation.state"],
            "observation.environment_state": batch["observation.environment_state"],
        }
        pred = policy.diffusion.generate_actions(obs_batch)  # [B, n_action_steps, D_action]
        # NOTE: pred and target are BOTH in preprocessor-normalized [0,1] space.
        # Do NOT postprocess — that would compare denormalized preds against normalized targets.

        # Standard LeRobot alignment: action index (n_obs-1) = current timestep t.
        # Take n_action_steps starting from there as the ground-truth for execution.
        start = policy.config.n_obs_steps - 1
        target = batch["action"][:, start : start + n_action_steps, :]

        all_preds.append(pred.cpu())
        all_targets.append(target.cpu())

    act_dim = pred.shape[-1]  # e.g., 19 (Δpalm=3 + finger=16)
    preds = torch.cat(all_preds, dim=0).reshape(-1, act_dim)
    targets = torch.cat(all_targets, dim=0).reshape(-1, act_dim)

    # Reshape back to [N, n_action_steps, act_dim] for per-step analysis
    preds_3d = torch.cat(all_preds, dim=0)
    targets_3d = torch.cat(all_targets, dim=0)

    # Per-dimension R² (all steps)
    ss_res = ((targets - preds) ** 2).sum(dim=0)
    ss_tot = ((targets - targets.mean(dim=0, keepdim=True)) ** 2).sum(dim=0)
    r2_per_dim = 1 - ss_res / (ss_tot + 1e-8)
    mean_r2 = r2_per_dim.mean().item()

    # Hierarchical action is palm pose (pos3 + rot6D) + q_pre16 = 25D.
    # Keep the legacy pos3 + rotvec3 + finger16 = 22D interpretation too.
    rot_end = 9 if act_dim == 25 else 6
    r2_pos = r2_per_dim[:3].mean().item()
    r2_rot = r2_per_dim[3:rot_end].mean().item()
    r2_finger = r2_per_dim[rot_end:].mean().item()
    mae = (preds - targets).abs().mean().item()

    # Future-only R²: skip step 0
    future_preds = preds_3d[:, 1:, :].reshape(-1, act_dim)
    future_targets = targets_3d[:, 1:, :].reshape(-1, act_dim)
    ss_res_f = ((future_targets - future_preds) ** 2).sum(dim=0)
    ss_tot_f = ((future_targets - future_targets.mean(dim=0, keepdim=True)) ** 2).sum(dim=0)
    r2_future_per_dim = 1 - ss_res_f / (ss_tot_f + 1e-8)
    mean_r2_future = r2_future_per_dim.mean().item()
    r2_pos_future = r2_future_per_dim[:3].mean().item()
    r2_rot_future = r2_future_per_dim[3:rot_end].mean().item()
    r2_finger_future = r2_future_per_dim[rot_end:].mean().item()
    mae_future = (future_preds - future_targets).abs().mean().item()

    policy.train()
    return {
        "val_r2": float(mean_r2),
        "val_mae": float(mae),
        "val_r2_future": float(mean_r2_future),
        "val_mae_future": float(mae_future),
        "val_r2_pos": float(r2_pos),
        "val_r2_rot": float(r2_rot),
        "val_r2_finger": float(r2_finger),
        "val_r2_pos_future": float(r2_pos_future),
        "val_r2_rot_future": float(r2_rot_future),
        "val_r2_finger_future": float(r2_finger_future),
    }


# ── Checkpointing ─────────────────────────────────────────────────────────────

def save_pretrained_full(
    policy: DiffusionPolicy,
    preprocessor,
    postprocessor,
    save_dir: Path,
    stats: dict | None = None,
):
    """Save model in LeRobot pretrained format + preprocessor stats.

    Produces: config.json, model.safetensors, and standard LeRobot preprocessor files.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # LeRobot standard format
    policy.save_pretrained(save_dir)

    # Save preprocessor / postprocessor in LeRobot standard format
    # (creates stats files that from_pretrained / make_pre_post_processors can load)
    preprocessor.save_pretrained(save_dir)
    postprocessor.save_pretrained(save_dir)

    # Also save raw dataset_stats dict for reference / debugging
    if stats is not None:
        stats_path = save_dir / "dataset_stats.pt"
        torch.save(stats, stats_path)

    print(f"  Pretrained model saved → {save_dir}")


def save_training_checkpoint(
    policy: DiffusionPolicy,
    optimizer: torch.optim.Optimizer,
    scaler,
    step: int,
    stats: dict,
    cfg: TrainConfig,
    save_dir: Path,
    metrics: dict | None = None,
    is_best: bool = False,
):
    """Save full training checkpoint (resumable)."""
    save_dir.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "step": step,
        "model_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "stats": {k: {kk: vv.cpu() for kk, vv in v.items()} for k, v in stats.items()},
        "config": _serializable_config(cfg),
    }
    if metrics:
        ckpt["metrics"] = metrics

    path = save_dir / f"checkpoint_step{step:07d}.pt"
    torch.save(ckpt, path)
    shutil.copy(path, save_dir / "latest.pt")

    if is_best:
        shutil.copy(path, save_dir / "best.pt")


def _serializable_config(cfg: TrainConfig) -> dict:
    return {
        "n_obs_steps": cfg.n_obs_steps,
        "horizon": cfg.horizon,
        "n_action_steps": cfg.n_action_steps,
        "down_dims": list(cfg.down_dims),
        "kernel_size": cfg.kernel_size,
        "n_groups": cfg.n_groups,
        "diffusion_step_embed_dim": cfg.diffusion_step_embed_dim,
        "num_inference_steps": cfg.num_inference_steps,
        "noise_scheduler_type": cfg.noise_scheduler_type,
        "prediction_type": cfg.prediction_type,
        "clip_fsr_pct": cfg.clip_fsr_pct,
        "stride": cfg.stride,
        "filter_enabled": cfg.filter_enabled,
        "min_fingers_in_contact": cfg.min_fingers_in_contact,
        "delta_percentile": cfg.delta_percentile,
        "delta_threshold": cfg.delta_threshold,
        "filter_ratio": cfg.filter_ratio,
        "state_dim": cfg.state_dim,
        "action_dim": cfg.action_dim,
    }


def load_training_checkpoint(
    checkpoint_dir: Path, policy, optimizer, scaler
) -> tuple[int, dict]:
    """Resume from latest.pt."""
    ckpt_path = checkpoint_dir / "latest.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    policy.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scaler.load_state_dict(ckpt["scaler_state_dict"])
    step = ckpt["step"]
    print(f"Resumed from step {step}  ({ckpt_path})")
    return step


# ── Main Training ─────────────────────────────────────────────────────────────

def train(cfg: TrainConfig, resume_from: Path | None = None):
    print(f"\n{'='*70}")
    print(f"  Diffusion Policy Training — Finger Compliance Control")
    print(f"  Device: {DEVICE}  Steps: {cfg.training_steps}  Batch: {cfg.batch_size}")
    print(f"  Window: obs={cfg.n_obs_steps}  H={cfg.horizon}  act={cfg.n_action_steps}")
    print(
        f"  UNet: {cfg.down_dims}  LR: {cfg.lr}  "
        f"Diffusion: {cfg.noise_scheduler_type}/{cfg.prediction_type} "
        f"inference_steps={cfg.num_inference_steps}"
    )
    print(f"{'='*70}\n")

    # ── Validate horizon constraint ──
    min_horizon = cfg.n_obs_steps + cfg.n_action_steps - 1
    if cfg.horizon < min_horizon:
        raise ValueError(
            f"horizon ({cfg.horizon}) must be >= n_obs_steps + n_action_steps - 1 "
            f"({min_horizon}). LeRobot's generate_actions hardcodes slicing "
            f"action[n_obs-1 : n_obs-1+n_action_steps], which would go out of bounds."
        )

    # ── Data ──
    train_data, val_data = load_data(
        cfg.h5_path, cfg.stride, cfg.clip_fsr_pct, cfg.val_envs
    )
    cfg.state_dim = int(train_data.state.shape[-1])
    cfg.action_dim = int(train_data.action.shape[-1])
    if val_data.state.shape[-1] != cfg.state_dim or val_data.action.shape[-1] != cfg.action_dim:
        raise ValueError("Train/validation state or action dimensions differ")
    print(f"  Model IO inferred from H5: state={cfg.state_dim} action={cfg.action_dim}")
    stats = compute_stats(train_data.state, train_data.action)

    filter_cfg = None
    if cfg.filter_enabled:
        filter_cfg = FilterConfig(
            min_fingers_in_contact=cfg.min_fingers_in_contact,
            delta_threshold=cfg.delta_threshold,
            delta_percentile=cfg.delta_percentile,
            filter_ratio=cfg.filter_ratio,
        )

    # 缓存路径: 基于 H5 路径 + 窗口/过滤参数生成唯一文件名, 避免重复计算
    import hashlib, os
    _cache_key = f"{cfg.h5_path}_{cfg.stride}_{cfg.n_obs_steps}_{cfg.horizon}_{cfg.min_fingers_in_contact}_{cfg.filter_ratio}_{cfg.min_action_norm}_{cfg.min_action_std}"
    _cache_hash = hashlib.md5(_cache_key.encode()).hexdigest()[:12]
    _cache_dir = os.path.join(os.path.dirname(cfg.h5_path), ".window_cache")
    _train_cache = os.path.join(_cache_dir, f"train_{_cache_hash}.npz")
    _val_cache = os.path.join(_cache_dir, f"val_{_cache_hash}.npz")

    train_ds = FilteredWindowDataset(train_data, cfg.n_obs_steps, cfg.horizon, filter_cfg, cache_path=_train_cache)
    val_ds = FilteredWindowDataset(val_data, cfg.n_obs_steps, cfg.horizon, filter_cfg, cache_path=_val_cache)

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.num_workers,
        pin_memory=(DEVICE == "cuda"),
        persistent_workers=(cfg.num_workers > 0),
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=min(2, cfg.num_workers),
        pin_memory=(DEVICE == "cuda"),
    )
    print(f"  Train: {len(train_ds):,} windows  Val: {len(val_ds):,} windows")
    print(f"  Batches/epoch: {len(train_dl):,}  Val batches: {len(val_dl):,}")

    # ── Model ──
    policy = build_policy(cfg)
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"  Parameters: {n_params:,}")

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, dataset_stats=stats
    )

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    use_amp = DEVICE == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ── Output directory ──
    start_step = 0
    best_val_loss = float("inf")
    metrics_history: list[dict] = []

    if resume_from is not None:
        start_step = load_training_checkpoint(resume_from, policy, optimizer, scaler)
        save_dir = resume_from
    else:
        if cfg.output_dir:
            save_dir = Path(cfg.output_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = OUTPUT_ROOT / f"dp_train_{timestamp}"
    print(f"  Output: {save_dir}\n")

    if resume_from is not None and (save_dir / "metrics.json").exists():
        with open(save_dir / "metrics.json") as f:
            metrics_history = json.load(f)
        if metrics_history:
            best_val_loss = min(m.get("val_loss", float("inf")) for m in metrics_history)
            print(f"  Best val loss so far: {best_val_loss:.5f}")

    # Save config for reproducibility
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "train_config.json", "w") as f:
        json.dump(_serializable_config(cfg), f, indent=2)

    # ── Training loop ──
    policy.train()
    train_iter = iter(train_dl)
    train_losses: list[float] = []

    pbar = (
        tqdm(
            total=cfg.training_steps,
            initial=start_step, # type: ignore
            desc="Training",
            unit="step",
            dynamic_ncols=True,
        )
        if tqdm is not None
        else None
    )

    step = start_step
    try:
        while step < cfg.training_steps:# type: ignore
            # Infinite DataLoader — restart on epoch end
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dl)
                batch = next(train_iter)

            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            batch = preprocessor(batch)

            with torch.autocast(device_type=DEVICE, enabled=use_amp):
                loss, _ = policy.forward(batch)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=cfg.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            train_losses.append(float(loss.item()))
            step += 1# type: ignore

            if pbar is not None:
                pbar.update(1)

            # ── Logging ──
            if step % cfg.log_interval == 0:
                avg_loss = np.mean(train_losses[-cfg.log_interval :])
                lr_now = optimizer.param_groups[0]["lr"]
                if pbar is not None:
                    pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr_now:.2e}")

            # ── Validation + Save ──
            if step % cfg.val_interval == 0 or step == cfg.training_steps:
                val_loss = run_validation(policy, val_dl, preprocessor)

                # Also compute action metrics
                action_m = compute_q_metrics(
                    policy, val_dl, preprocessor,
                    n_action_steps=cfg.n_action_steps,
                )

                train_avg = float(np.mean(train_losses[-100:]))
                is_best = val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss

                entry = {
                    "step": step,
                    "train_loss": train_avg,
                    "val_loss": val_loss,
                    **action_m,
                }
                metrics_history.append(entry)

                status = "✅ BEST" if is_best else ""
                print(
                    f"\n  Step {step:7d} | "
                    f"train_loss={train_avg:.5f}  val_loss={val_loss:.5f}  "
                    f"R²(all)={action_m['val_r2']:.4f}  pos={action_m['val_r2_pos_future']:.4f}  "
                    f"rot={action_m['val_r2_rot_future']:.4f}  "
                    f"finger={action_m['val_r2_finger_future']:.4f}  MAE={action_m['val_mae']:.4f}  "
                    f"{status}"
                )

                save_training_checkpoint(
                    policy, optimizer, scaler, step, stats, cfg,
                    save_dir, metrics=entry, is_best=is_best,
                )

                # Save pretrained format on best
                if is_best:
                    save_pretrained_full(
                        policy, preprocessor, postprocessor,
                        save_dir / "pretrained", stats=stats,
                    )

            # ── Periodic save ──
            elif step % cfg.save_interval == 0:
                save_training_checkpoint(
                    policy, optimizer, scaler, step, stats, cfg, save_dir
                )

    except KeyboardInterrupt:
        print("\n  ⚠️  Interrupted — saving checkpoint...")
    finally:
        if pbar is not None:
            pbar.close()

    # ── Final save ──
    print("\n  Saving final model...")
    val_loss = run_validation(policy, val_dl, preprocessor)
    action_m = compute_q_metrics(
        policy, val_dl, preprocessor, n_action_steps=cfg.n_action_steps
    )
    is_best = val_loss < best_val_loss
    if is_best:
        best_val_loss = val_loss

    entry = {
        "step": step,
        "train_loss": float(np.mean(train_losses[-100:])),
        "val_loss": val_loss,
        **action_m,
    }
    metrics_history.append(entry)

    save_training_checkpoint(
        policy, optimizer, scaler, step, stats, cfg,# type: ignore
        save_dir, metrics=entry, is_best=is_best,
    )
    save_pretrained_full(policy, preprocessor, postprocessor, save_dir / "pretrained", stats=stats)

    # Save metrics history
    with open(save_dir / "metrics.json", "w") as f:
        json.dump(metrics_history, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  Training complete!")
    print(f"    Steps: {step}")
    print(f"    Best val loss: {best_val_loss:.5f}")
    print(f"    Best val R²: {max(m.get('val_r2', -99) for m in metrics_history):.4f}")
    print(f"    Model: {save_dir}")
    print(f"    Pretrained: {save_dir / 'pretrained'}")
    print(f"{'='*70}")


@torch.no_grad()
def run_validation(policy, val_dl, preprocessor) -> float:
    """Compute average diffusion loss on validation set."""
    policy.eval()
    total_loss = 0.0
    n_batches = 0
    for batch in val_dl:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        batch = preprocessor(batch)
        loss, _ = policy.forward(batch)
        total_loss += float(loss.item())
        n_batches += 1
    policy.train()
    return total_loss / max(n_batches, 1)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    global DEVICE

    parser = argparse.ArgumentParser(
        description="Train Diffusion Policy for hierarchical palm/finger compliance control"
    )
    # Data
    parser.add_argument("--h5-path", type=str, default=str(H5_PATH))
    parser.add_argument("--stride", type=int, default=3,
                       help="Temporal downsampling (1=100Hz, 3=~33Hz, 5=20Hz)")
    parser.add_argument("--clip-fsr-pct", type=float, default=99.9)
    parser.add_argument("--val-envs", type=int, nargs="+", default=[30, 31],
                       help="Environment indices for validation split")
    # Window
    parser.add_argument("--n-obs-steps", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=24,
                       help="Action horizon (must be >= n_obs_steps + n_action_steps - 1)")
    parser.add_argument("--n-action-steps", type=int, default=4)
    # Filter
    parser.add_argument("--no-filter", action="store_true",
                       help="Disable quality filtering")
    parser.add_argument("--min-fingers", type=int, default=3,
                       help="Min fingers in contact")
    parser.add_argument("--delta-percentile", type=float, default=90.0,
                       help="Percentile for auto delta_threshold")
    parser.add_argument("--delta-threshold", type=float, default=None,
                       help="Manual fsr_delta_norm threshold (overrides percentile)")
    parser.add_argument("--filter-ratio", type=float, default=0.7,
                       help="Fraction of window frames that must pass filter")
    parser.add_argument("--min-action-norm", type=float, default=0.01,
                       help="Min mean action L2-norm in window (remove static frames)")
    parser.add_argument("--min-action-std", type=float, default=0.005,
                       help="Min std of action L2-norm in window (remove static frames)")
    # Training
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    # Logging
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--val-interval", type=int, default=2000)
    parser.add_argument("--save-interval", type=int, default=10_000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default=None)
    # Model
    parser.add_argument("--small", action="store_true",
                       help="Use smaller UNet (64,128,256) for quick tests")
    parser.add_argument("--down-dims", type=int, nargs="+", default=None,
                       help="UNet down dims, e.g. '--down-dims 256 512 1024'")
    parser.add_argument("--num-inference-steps", type=int, default=100,
                       help="Reverse diffusion steps. Training uses 100 noise steps; use 100 for strict overfit tests")
    parser.add_argument("--noise-scheduler", choices=["DDPM", "DDIM"], default="DDPM")
    parser.add_argument("--prediction-type", choices=["epsilon", "sample"], default="sample",
                       help="UNet target: added noise (standard DP) or the clean action trajectory")
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint directory to resume from")
    args = parser.parse_args()

    DEVICE = args.device
    if DEVICE == "cuda":
        torch.backends.cudnn.benchmark = True

    if args.down_dims is not None:
        down_dims = tuple(args.down_dims)
    elif args.small:
        down_dims = (64, 128, 256)
    else:
        down_dims = (256, 512, 1024)  # TrainConfig default
    diffusion_step_embed_dim = 64 if args.small else 128

    cfg = TrainConfig(
        h5_path=args.h5_path,
        stride=args.stride,
        clip_fsr_pct=args.clip_fsr_pct,
        val_envs=tuple(args.val_envs),
        n_obs_steps=args.n_obs_steps,
        horizon=args.horizon,
        n_action_steps=args.n_action_steps,
        filter_enabled=not args.no_filter,
        min_fingers_in_contact=args.min_fingers,
        delta_percentile=args.delta_percentile,
        delta_threshold=args.delta_threshold,
        filter_ratio=args.filter_ratio,
        min_action_norm=args.min_action_norm,
        min_action_std=args.min_action_std,
        training_steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip,
        down_dims=down_dims,
        diffusion_step_embed_dim=diffusion_step_embed_dim,
        num_inference_steps=args.num_inference_steps,
        noise_scheduler_type=args.noise_scheduler,
        prediction_type=args.prediction_type,
        log_interval=args.log_interval,
        val_interval=args.val_interval,
        save_interval=args.save_interval,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
    )

    resume_from = Path(args.resume) if args.resume else None
    train(cfg, resume_from=resume_from)


if __name__ == "__main__":
    main()
