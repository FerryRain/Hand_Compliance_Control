#!/usr/bin/env python3
"""Train a goal-conditioned compliance policy from collected H5 trajectories.

This training script is designed for the headless collector outputs and supports
both old and new H5 schemas. It trains a GRU policy with:
- sequence input: windowed [FSR, q_hand, prev_action]
- goal conditioning: desired contact profile vector
- action objective: predict action or residual action
- auxiliary objective: predict contact-quality labels
"""

from __future__ import annotations

import argparse
import glob
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import cast

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


@dataclass(frozen=True)
class Normalizers:
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    q_mean: np.ndarray
    q_std: np.ndarray


class SequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        fsr: np.ndarray,
        q: np.ndarray,
        action: np.ndarray,
        quality: np.ndarray,
        indices: np.ndarray,
        window: int,
        normalizers: Normalizers,
        goal_vec: np.ndarray,
        predict_residual: bool,
        quality_target_step: int,
    ) -> None:
        self.fsr = fsr
        self.q = q
        self.action = action
        self.quality = quality
        self.indices = indices
        self.window = window
        self.norm = normalizers
        self.goal_vec = goal_vec.astype(np.float32)
        self.predict_residual = predict_residual
        self.quality_target_step = quality_target_step

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = int(self.indices[idx])
        num_envs = self.fsr.shape[1]
        t = flat // num_envs
        e = flat % num_envs

        t0 = t - self.window + 1

        fsr_seq = self.fsr[t0 : t + 1, e]
        q_seq = self.q[t0 : t + 1, e]

        prev_action_seq = np.zeros_like(self.action[t0 : t + 1, e], dtype=np.float32)
        prev_action_seq[1:] = self.action[t0:t, e]

        x_seq = np.concatenate([fsr_seq, q_seq, prev_action_seq], axis=-1).astype(np.float32)
        x_seq = (x_seq - self.norm.x_mean) / self.norm.x_std

        y_action = self.action[t, e].astype(np.float32)
        if self.predict_residual:
            prev_action = self.action[t - 1, e] if t > 0 else np.zeros_like(y_action)
            y_action = y_action - prev_action
        y_action = (y_action - self.norm.y_mean[0]) / self.norm.y_std[0]

        tq = min(t + self.quality_target_step, self.quality.shape[0] - 1)
        y_quality = self.quality[tq, e].astype(np.float32)
        y_quality = (y_quality - self.norm.q_mean[0]) / self.norm.q_std[0]

        return (
            torch.from_numpy(x_seq),
            torch.from_numpy(self.goal_vec),
            torch.from_numpy(y_action),
            torch.from_numpy(y_quality),
        )


class GoalConditionedGRUPolicy(nn.Module):
    def __init__(self, in_dim: int, goal_dim: int, action_dim: int, quality_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size=in_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        self.backbone = nn.Sequential(
            nn.Linear(hidden_dim + goal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.quality_head = nn.Linear(hidden_dim, quality_dim)

    def forward(self, x_seq: torch.Tensor, goal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out, _ = self.gru(x_seq)
        h = out[:, -1, :]
        h = torch.cat([h, goal], dim=-1)
        z = self.backbone(h)
        return self.action_head(z), self.quality_head(z)


def _find_input_file(explicit_path: str | None, pattern: str) -> str:
    if explicit_path is not None:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"Input H5 not found: {path}")
        return str(path)
    matches = sorted(glob.glob(pattern), key=lambda p: Path(p).stat().st_mtime)
    matches = [m for m in matches if not m.endswith("_inverted.h5")]
    if not matches:
        raise FileNotFoundError(f"No H5 files match: {pattern}")
    return matches[-1]


def _q_hand(q: np.ndarray) -> np.ndarray:
    if q.shape[-1] >= 16:
        return q[..., -16:]
    return q


def _compute_quality_fallback(fsr: np.ndarray, threshold: float) -> np.ndarray:
    finger_ids = ((4, 5, 6), (7, 8, 9), (10, 11, 12), (13, 14, 15))
    finger_force = np.stack([fsr[..., ids].mean(axis=-1) for ids in finger_ids], axis=-1)
    finger_contact = (finger_force >= threshold).astype(np.float32)
    full_contact_mask = np.asarray((finger_contact > 0.5).all(axis=-1), dtype=np.bool_)
    full_contact = np.expand_dims(full_contact_mask.astype(np.float32), axis=-1)

    stability = np.zeros_like(full_contact)
    run = np.zeros((fsr.shape[1],), dtype=np.float32)
    for t in range(fsr.shape[0]):
        full_t = np.asarray(full_contact_mask[t], dtype=np.bool_)
        run = np.where(full_t, run + 1.0, 0.0)
        stability[t, :, 0] = np.clip(run / 20.0, 0.0, 1.0)

    force_balance = finger_force.std(axis=-1, keepdims=True)
    fsr_delta = np.zeros((fsr.shape[0], fsr.shape[1], 1), dtype=np.float32)
    fsr_delta[1:, :, 0] = np.linalg.norm(fsr[1:] - fsr[:-1], axis=-1)

    return np.concatenate(
        [finger_force, finger_contact, full_contact, stability, force_balance, fsr_delta],
        axis=-1,
    ).astype(np.float32)


def _load_data(path: str, drop_palm_fsr: bool, max_steps: int | None, step_stride: int, threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as f:
        sl = slice(0, max_steps if max_steps is not None else None, step_stride)
        fsr_full = np.asarray(cast(h5py.Dataset, f["fsr"])[sl], dtype=np.float32)
        q = np.asarray(cast(h5py.Dataset, f["q"])[sl], dtype=np.float32)
        action = np.asarray(cast(h5py.Dataset, f["action"])[sl], dtype=np.float32)

        quality_names = (
            "finger_force",
            "finger_contact",
            "full_contact",
            "contact_stability",
            "force_balance",
            "fsr_delta_norm",
        )
        quality_parts: list[np.ndarray] = []
        has_quality = True
        for name in quality_names:
            if name not in f:
                has_quality = False
                break
            arr = np.asarray(cast(h5py.Dataset, f[name])[sl], dtype=np.float32)
            if arr.ndim == 2:
                arr = arr[..., None]
            quality_parts.append(arr)

    fsr = fsr_full[..., 4:] if drop_palm_fsr else fsr_full

    qh = _q_hand(q)
    if has_quality:
        quality = np.concatenate(quality_parts, axis=-1).astype(np.float32)
    else:
        quality = _compute_quality_fallback(fsr_full, threshold)

    return fsr, qh, action, quality


def _split_indices(num_steps: int, num_envs: int, train_ratio: float, val_ratio: float, window: int, seed: int) -> SplitIndices:
    train_end = max(window + 1, int(num_steps * train_ratio))
    val_end = max(train_end + 1, int(num_steps * (train_ratio + val_ratio)))
    val_end = min(val_end, num_steps - 1)

    rng = np.random.default_rng(seed)

    def flat_range(t0: int, t1: int) -> np.ndarray:
        t = np.arange(max(window - 1, t0), t1, dtype=np.int64)
        grid_t, grid_e = np.meshgrid(t, np.arange(num_envs, dtype=np.int64), indexing="ij")
        idx = (grid_t * num_envs + grid_e).reshape(-1)
        rng.shuffle(idx)
        return idx

    return SplitIndices(
        train=flat_range(0, train_end),
        val=flat_range(train_end, val_end),
        test=flat_range(val_end, num_steps),
    )


def _subsample(idx: np.ndarray, max_count: int | None) -> np.ndarray:
    if max_count is None or idx.shape[0] <= max_count:
        return idx
    return idx[:max_count]


def _fit_normalizers(
    fsr: np.ndarray,
    q: np.ndarray,
    action: np.ndarray,
    quality: np.ndarray,
    train_idx: np.ndarray,
    window: int,
    predict_residual: bool,
    quality_target_step: int,
) -> Normalizers:
    num_envs = fsr.shape[1]
    seqs = []
    y_actions = []
    y_quality = []

    for flat in train_idx:
        t = int(flat // num_envs)
        e = int(flat % num_envs)
        t0 = t - window + 1
        fsr_seq = fsr[t0 : t + 1, e]
        q_seq = q[t0 : t + 1, e]
        prev_action_seq = np.zeros_like(action[t0 : t + 1, e], dtype=np.float32)
        prev_action_seq[1:] = action[t0:t, e]
        x_seq = np.concatenate([fsr_seq, q_seq, prev_action_seq], axis=-1)
        seqs.append(x_seq)

        y = action[t, e]
        if predict_residual:
            y = y - (action[t - 1, e] if t > 0 else np.zeros_like(y))
        y_actions.append(y)
        tq = min(t + quality_target_step, quality.shape[0] - 1)
        y_quality.append(quality[tq, e])

    x_all = np.concatenate(seqs, axis=0).astype(np.float32)
    y_all = np.stack(y_actions).astype(np.float32)
    q_all = np.stack(y_quality).astype(np.float32)

    x_mean = x_all.mean(axis=0, keepdims=True)
    x_std = np.where(x_all.std(axis=0, keepdims=True) < 1e-6, 1.0, x_all.std(axis=0, keepdims=True))

    y_mean = y_all.mean(axis=0, keepdims=True)
    y_std = np.where(y_all.std(axis=0, keepdims=True) < 1e-6, 1.0, y_all.std(axis=0, keepdims=True))

    q_mean = q_all.mean(axis=0, keepdims=True)
    q_std = np.where(q_all.std(axis=0, keepdims=True) < 1e-6, 1.0, q_all.std(axis=0, keepdims=True))

    return Normalizers(x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std, q_mean=q_mean, q_std=q_std)


def _make_goal_vector(force_target: float, stability_target: float, balance_target: float) -> np.ndarray:
    return np.asarray([force_target, force_target, force_target, force_target, 1.0, stability_target, balance_target], dtype=np.float32)


def _quality_goal_from_labels(quality: torch.Tensor) -> torch.Tensor:
    if quality.ndim == 3 and quality.shape[1] == 1:
        quality = quality[:, 0, :]
    if quality.ndim != 2:
        raise ValueError(f"Expected quality tensor rank 2 or [B,1,D], got {tuple(quality.shape)}")
    if quality.shape[-1] < 11:
        raise ValueError(f"Expected quality feature dim >= 11, got {quality.shape[-1]}")
    # [finger_force(4), finger_contact(4), full_contact, stability, balance, fsr_delta]
    return torch.cat(
        [quality[:, :4], quality[:, 8:9], quality[:, 9:10], quality[:, 10:11]],
        dim=-1,
    )


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_mean = y_true.mean(axis=0, keepdims=True)
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - y_mean) ** 2, axis=0)
    valid = ss_tot > 1e-12
    r2 = np.zeros_like(ss_tot, dtype=np.float32)
    r2[valid] = 1.0 - ss_res[valid] / ss_tot[valid]
    return float(r2.mean())


def _evaluate(
    model: GoalConditionedGRUPolicy,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    device: torch.device,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    quality_w: float,
) -> tuple[float, float, float]:
    model.eval()
    huber = nn.SmoothL1Loss()
    mse = nn.MSELoss()

    losses = []
    y_true_all = []
    y_pred_all = []

    with torch.no_grad():
        for x, goal, y, yq in loader:
            x = x.to(device)
            goal = goal.to(device)
            y = y.to(device)
            yq = yq.to(device)

            yp, yq_pred = model(x, goal)
            loss_action = huber(yp, y)
            goal_q = _quality_goal_from_labels(yq)
            pred_q = _quality_goal_from_labels(yq_pred)
            loss_quality = mse(pred_q, goal_q)
            loss = loss_action + quality_w * loss_quality
            losses.append(float(loss.item()))

            y_true_all.append(y.cpu().numpy())
            y_pred_all.append(yp.cpu().numpy())

    y_true_z = np.concatenate(y_true_all, axis=0)
    y_pred_z = np.concatenate(y_pred_all, axis=0)

    y_true = y_true_z * y_std + y_mean
    y_pred = y_pred_z * y_std + y_mean

    mae = float(np.mean(np.abs(y_true - y_pred)))
    r2 = _r2_score(y_true, y_pred)
    return float(np.mean(losses)), mae, r2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train goal-conditioned compliance policy from H5 data.")
    p.add_argument("--h5", type=str, default=None)
    p.add_argument("--glob", type=str, default="./finger_compliance_control/data/headless/*.h5")
    p.add_argument("--device", type=str, choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--drop-palm-fsr", action="store_true")
    p.add_argument("--max-steps", type=int, default=200000)
    p.add_argument("--step-stride", type=int, default=1)
    p.add_argument("--window", type=int, default=20)
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--max-train-samples", type=int, default=250000)
    p.add_argument("--max-val-samples", type=int, default=80000)
    p.add_argument("--max-test-samples", type=int, default=80000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--quality-loss-weight", type=float, default=0.2)
    p.add_argument("--quality-target-step", type=int, default=1)
    p.add_argument("--predict-residual", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument(
        "--show-batch-progress",
        action="store_true",
        help="Show per-batch progress bar (disabled by default for cleaner logs).",
    )
    p.add_argument("--goal-force", type=float, default=1.0)
    p.add_argument("--goal-stability", type=float, default=1.0)
    p.add_argument("--goal-balance", type=float, default=0.1)
    p.add_argument("--contact-threshold", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-dir", type=str, default="./finger_compliance_control/data/models")
    p.add_argument("--name", type=str, default="goal_gru")
    p.add_argument(
        "--resume-checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint .pt file to continue training from.",
    )
    p.add_argument(
        "--resume-optimizer",
        action="store_true",
        help="Restore optimizer/scaler state when resuming if present in checkpoint.",
    )
    p.add_argument(
        "--log-dir",
        type=str,
        default="./finger_compliance_control/data/models/logs",
        help="Directory to save training summary logs.",
    )
    p.add_argument(
        "--log-name",
        type=str,
        default=None,
        help="Optional log filename (without directory). If omitted, use timestamped name.",
    )
    return p.parse_args()


def _save_train_log(
    args: argparse.Namespace,
    h5_path: str,
    fsr_shape: tuple[int, ...],
    q_shape: tuple[int, ...],
    action_shape: tuple[int, ...],
    quality_shape: tuple[int, ...],
    train_size: int,
    val_size: int,
    test_size: int,
    device: torch.device,
    best_val: float,
    test_loss: float,
    test_mae: float,
    test_r2: float,
    ckpt_path: Path,
    norm_path: Path,
    epoch_records: list[tuple[int, float, float, float, float]],
) -> Path:
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.log_name:
        filename = args.log_name
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"train_{args.name}_{ts}.log"

    log_path = log_dir / filename
    lines = [
        "=" * 72,
        "Goal-Conditioned Compliance Training Summary",
        "=" * 72,
        f"timestamp={datetime.now().isoformat(timespec='seconds')}",
        f"data={h5_path}",
        f"shape fsr={fsr_shape} q={q_shape} action={action_shape} quality={quality_shape}",
        (
            f"device={device.type} window={args.window} residual={args.predict_residual} "
            f"quality_target_step={args.quality_target_step}"
        ),
        f"train/val/test={train_size}/{val_size}/{test_size}",
        "",
        "[Config]",
        f"name={args.name}",
        f"seed={args.seed}",
        f"epochs={args.epochs}",
        f"batch_size={args.batch_size}",
        f"lr={args.lr}",
        f"weight_decay={args.weight_decay}",
        f"hidden_dim={args.hidden_dim}",
        f"quality_loss_weight={args.quality_loss_weight}",
        f"drop_palm_fsr={args.drop_palm_fsr}",
        f"step_stride={args.step_stride}",
        f"max_steps={args.max_steps}",
        f"max_train_samples={args.max_train_samples}",
        f"max_val_samples={args.max_val_samples}",
        f"max_test_samples={args.max_test_samples}",
        f"resume_checkpoint={args.resume_checkpoint}",
        f"resume_optimizer={args.resume_optimizer}",
        "",
        "[Epochs]",
    ]
    for ep, train_loss, val_loss, val_mae, val_r2 in epoch_records:
        lines.append(
            f"epoch={ep:03d} train_loss={train_loss:.5f} val_loss={val_loss:.5f} "
            f"val_mae={val_mae:.5f} val_r2={val_r2:+.3f}"
        )

    lines.extend(
        [
            "",
            "[Result]",
            f"best_val_loss={best_val:.5f}",
            f"test_loss={test_loss:.5f} test_mae={test_mae:.5f} test_r2={test_r2:+.3f}",
            f"saved_model={ckpt_path}",
            f"saved_norm={norm_path}",
        ]
    )

    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return log_path


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA not available")
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")

    h5_path = _find_input_file(args.h5, args.glob)
    fsr, q, action, quality = _load_data(
        h5_path,
        drop_palm_fsr=args.drop_palm_fsr,
        max_steps=args.max_steps,
        step_stride=args.step_stride,
        threshold=args.contact_threshold,
    )

    num_steps, num_envs, fsr_dim = fsr.shape
    action_dim = action.shape[-1]
    quality_dim = quality.shape[-1]

    splits = _split_indices(
        num_steps=num_steps,
        num_envs=num_envs,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        window=args.window,
        seed=args.seed,
    )

    train_idx = _subsample(splits.train, args.max_train_samples)
    val_idx = _subsample(splits.val, args.max_val_samples)
    test_idx = _subsample(splits.test, args.max_test_samples)

    norm = _fit_normalizers(
        fsr=fsr,
        q=q,
        action=action,
        quality=quality,
        train_idx=train_idx,
        window=args.window,
        predict_residual=args.predict_residual,
        quality_target_step=args.quality_target_step,
    )

    goal_vec = _make_goal_vector(
        force_target=args.goal_force,
        stability_target=args.goal_stability,
        balance_target=args.goal_balance,
    )

    train_ds = SequenceDataset(
        fsr=fsr,
        q=q,
        action=action,
        quality=quality,
        indices=train_idx,
        window=args.window,
        normalizers=norm,
        goal_vec=goal_vec,
        predict_residual=args.predict_residual,
        quality_target_step=args.quality_target_step,
    )
    val_ds = SequenceDataset(
        fsr=fsr,
        q=q,
        action=action,
        quality=quality,
        indices=val_idx,
        window=args.window,
        normalizers=norm,
        goal_vec=goal_vec,
        predict_residual=args.predict_residual,
        quality_target_step=args.quality_target_step,
    )
    test_ds = SequenceDataset(
        fsr=fsr,
        q=q,
        action=action,
        quality=quality,
        indices=test_idx,
        window=args.window,
        normalizers=norm,
        goal_vec=goal_vec,
        predict_residual=args.predict_residual,
        quality_target_step=args.quality_target_step,
    )

    pin_memory = dev.type == "cuda"
    num_workers = max(args.num_workers, 0)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "pin_memory": pin_memory,
        "num_workers": num_workers,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = max(args.prefetch_factor, 1)

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    in_dim = fsr_dim + q.shape[-1] + action_dim
    goal_dim = goal_vec.shape[-1]

    model = GoalConditionedGRUPolicy(
        in_dim=in_dim,
        goal_dim=goal_dim,
        action_dim=action_dim,
        quality_dim=quality_dim,
        hidden_dim=args.hidden_dim,
    ).to(dev)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    huber = nn.SmoothL1Loss()
    mse = nn.MSELoss()
    use_amp = dev.type == "cuda" and not args.no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True

    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    epoch_records: list[tuple[int, float, float, float, float]] = []
    start_epoch = 1

    if args.resume_checkpoint is not None:
        resume_path = Path(args.resume_checkpoint)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

        resume_ckpt = torch.load(resume_path, map_location=dev)
        model.load_state_dict(resume_ckpt["model_state_dict"])

        if "best_val_loss" in resume_ckpt:
            best_val = float(resume_ckpt["best_val_loss"])

        if "epoch" in resume_ckpt:
            start_epoch = int(resume_ckpt["epoch"]) + 1

        if args.resume_optimizer:
            if "optimizer_state_dict" in resume_ckpt:
                optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
            if use_amp and "scaler_state_dict" in resume_ckpt and resume_ckpt["scaler_state_dict"] is not None:
                scaler.load_state_dict(resume_ckpt["scaler_state_dict"])

    print("=" * 72)
    print("Goal-Conditioned Compliance Training")
    print("=" * 72)
    print(f"data={h5_path}")
    print(
        f"shape fsr={tuple(fsr.shape)} q={tuple(q.shape)} action={tuple(action.shape)} quality={tuple(quality.shape)}"
    )
    print(
        f"device={dev.type} window={args.window} residual={args.predict_residual} "
        f"quality_target_step={args.quality_target_step} amp={use_amp} num_workers={num_workers} "
        f"train/val/test={len(train_ds)}/{len(val_ds)}/{len(test_ds)}"
    )
    if args.resume_checkpoint is not None:
        print(
            f"resume_from={args.resume_checkpoint} start_epoch={start_epoch} "
            f"restore_optimizer={args.resume_optimizer}"
        )

    end_epoch = start_epoch + args.epochs - 1
    epoch_bar = tqdm(range(start_epoch, end_epoch + 1), desc="epochs", dynamic_ncols=True)
    for epoch in epoch_bar:
        model.train()
        epoch_losses = []
        if args.show_batch_progress:
            train_iter = tqdm(
                train_loader,
                desc=f"train {epoch:03d}",
                leave=False,
                dynamic_ncols=True,
                mininterval=0.75,
            )
        else:
            train_iter = train_loader

        for batch_idx, (x, goal, y, yq) in enumerate(train_iter, start=1):
            x = x.to(dev, non_blocking=pin_memory)
            goal = goal.to(dev, non_blocking=pin_memory)
            y = y.to(dev, non_blocking=pin_memory)
            yq = yq.to(dev, non_blocking=pin_memory)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, enabled=use_amp):
                yp, yq_pred = model(x, goal)
                loss_action = huber(yp, y)
                goal_q = _quality_goal_from_labels(yq)
                pred_q = _quality_goal_from_labels(yq_pred)
                loss_quality = mse(pred_q, goal_q)
                loss = loss_action + args.quality_loss_weight * loss_quality

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_losses.append(float(loss.item()))
            if args.show_batch_progress and isinstance(train_iter, tqdm):
                if batch_idx == 1 or batch_idx % 20 == 0:
                    train_iter.set_postfix(loss=f"{loss.item():.4f}")

        val_loss, val_mae, val_r2 = _evaluate(
            model,
            val_loader,
            dev,
            y_mean=norm.y_mean,
            y_std=norm.y_std,
            quality_w=args.quality_loss_weight,
        )

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        tqdm.write(
            f"epoch={epoch:03d} train_loss={train_loss:.5f} "
            f"val_loss={val_loss:.5f} val_mae={val_mae:.5f} val_r2={val_r2:+.3f}"
        )
        epoch_records.append((epoch, train_loss, val_loss, val_mae, val_r2))
        epoch_bar.set_postfix(train_loss=f"{train_loss:.5f}", val_loss=f"{val_loss:.5f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_mae, test_r2 = _evaluate(
        model,
        test_loader,
        dev,
        y_mean=norm.y_mean,
        y_std=norm.y_std,
        quality_w=args.quality_loss_weight,
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / f"{args.name}.pt"
    norm_path = save_dir / f"{args.name}_norm.npz"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if use_amp else None,
            "config": vars(args),
            "in_dim": in_dim,
            "goal_dim": goal_dim,
            "action_dim": action_dim,
            "quality_dim": quality_dim,
            "best_val_loss": best_val,
            "epoch": end_epoch,
        },
        ckpt_path,
    )

    np.savez(
        norm_path,
        x_mean=norm.x_mean,
        x_std=norm.x_std,
        y_mean=norm.y_mean,
        y_std=norm.y_std,
        q_mean=norm.q_mean,
        q_std=norm.q_std,
        goal_vec=goal_vec,
    )

    print("\n[Result]")
    print(f"best_val_loss={best_val:.5f}")
    print(f"test_loss={test_loss:.5f} test_mae={test_mae:.5f} test_r2={test_r2:+.3f}")
    print(f"saved_model={ckpt_path}")
    print(f"saved_norm={norm_path}")

    log_path = _save_train_log(
        args=args,
        h5_path=h5_path,
        fsr_shape=tuple(fsr.shape),
        q_shape=tuple(q.shape),
        action_shape=tuple(action.shape),
        quality_shape=tuple(quality.shape),
        train_size=len(train_ds),
        val_size=len(val_ds),
        test_size=len(test_ds),
        device=dev,
        best_val=best_val,
        test_loss=test_loss,
        test_mae=test_mae,
        test_r2=test_r2,
        ckpt_path=ckpt_path,
        norm_path=norm_path,
        epoch_records=epoch_records,
    )
    print(f"saved_log={log_path}")


if __name__ == "__main__":
    main()
