#!/usr/bin/env python3
"""Signal-strength diagnostics for learning action from FSR and joint state.

This script answers a practical question:
"Do my observations contain predictive signal for the recorded action?"

It runs four checks on H5 trajectory data:
1. Baselines with time-wise split (no leakage):
   - constant predictor
   - ridge regression on current features
   - ridge regression on windowed history
2. Shuffle control (train on shuffled targets) to detect spurious correlations.
3. Lag scan: corr(FSR_t, action_{t+lag}) for lag in [0, lag_max].
4. Automatic interpretation summary.

Usage example:
  uv run python finger_compliance_control/scripts/signal_strength_check.py \
    --glob "./finger_compliance_control/data/headless/*.h5" --window 20
"""

from __future__ import annotations

import argparse
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import h5py
import numpy as np
import torch


@dataclass(frozen=True)
class SplitData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


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


def _load_arrays(
    path: str,
    max_steps: int | None,
    step_stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    with h5py.File(path, "r") as f:
        if step_stride <= 0:
            raise ValueError("step_stride must be >= 1")
        end = max_steps if max_steps is not None else None
        sl = slice(0, end, step_stride)
        fsr_dset = cast(h5py.Dataset, f["fsr"])
        action_dset = cast(h5py.Dataset, f["action"])
        q_dset = cast(h5py.Dataset, f["q"])
        fsr = np.asarray(fsr_dset[sl], dtype=np.float32)
        action = np.asarray(action_dset[sl], dtype=np.float32)
        q = np.asarray(q_dset[sl], dtype=np.float32)

        quality_names = (
            "finger_force",
            "finger_contact",
            "full_contact",
            "contact_stability",
            "force_balance",
            "fsr_delta_norm",
        )
        quality_parts: list[np.ndarray] = []
        has_all_quality = True
        for name in quality_names:
            if name not in f:
                has_all_quality = False
                break
            dset = cast(h5py.Dataset, f[name])
            arr = np.asarray(dset[sl], dtype=np.float32)
            if arr.ndim == 2:
                arr = arr[..., None]
            quality_parts.append(arr)

    quality = None
    if has_all_quality and quality_parts:
        quality = np.concatenate(quality_parts, axis=-1)
    return fsr, q, action, quality


def _q_hand(q: np.ndarray) -> np.ndarray:
    if q.shape[-1] >= 16:
        return q[..., -16:]
    return q


def _select_fsr_features(fsr: np.ndarray, drop_palm_fsr: bool) -> np.ndarray:
    if not drop_palm_fsr:
        return fsr
    if fsr.shape[-1] <= 4:
        raise ValueError(
            "Cannot drop palm FSR channels: fsr dimension must be > 4."
        )
    # Palm channels are [0..3]; keep finger channels [4..].
    return fsr[..., 4:]


def _standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    z = (x - mean) / std
    return z, mean, std


def _standardize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def _ridge_fit_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    alpha: float,
) -> np.ndarray:
    ones_train = np.ones((x_train.shape[0], 1), dtype=np.float32)
    ones_eval = np.ones((x_eval.shape[0], 1), dtype=np.float32)
    x_train_aug = np.concatenate([x_train, ones_train], axis=1)
    x_eval_aug = np.concatenate([x_eval, ones_eval], axis=1)

    d = x_train_aug.shape[1]
    reg = np.eye(d, dtype=np.float32) * alpha
    reg[-1, -1] = 0.0

    xtx = x_train_aug.T @ x_train_aug
    xty = x_train_aug.T @ y_train
    w = np.linalg.solve(xtx + reg, xty)
    return x_eval_aug @ w


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    diff = y_true - y_pred
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))

    y_mean = y_true.mean(axis=0, keepdims=True)
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - y_mean) ** 2, axis=0)
    valid = ss_tot > 1e-12
    r2_dim = np.zeros_like(ss_tot, dtype=np.float32)
    r2_dim[valid] = 1.0 - (ss_res[valid] / ss_tot[valid])
    r2_mean = float(np.mean(r2_dim))

    return {
        "mse": mse,
        "mae": mae,
        "r2_mean": r2_mean,
    }


def _time_splits(num_steps: int, train_ratio: float, val_ratio: float) -> tuple[int, int]:
    train_end = int(num_steps * train_ratio)
    val_end = int(num_steps * (train_ratio + val_ratio))
    train_end = max(2, min(train_end, num_steps - 2))
    val_end = max(train_end + 1, min(val_end, num_steps - 1))
    return train_end, val_end


def _sample_range(
    fsr: np.ndarray,
    q_hand: np.ndarray,
    quality: np.ndarray | None,
    action: np.ndarray,
    t_start: int,
    t_stop: int,
    feature_mode: str,
    window: int,
    max_samples: int | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    num_steps, num_envs, fsr_dim = fsr.shape
    _ = num_steps
    action_dim = int(action.shape[-1])
    q_dim = int(q_hand.shape[-1])
    quality_dim = int(quality.shape[-1]) if quality is not None else 0

    if feature_mode == "fsr":
        feat_dim = fsr_dim
        valid_t0 = t_start
    elif feature_mode == "q":
        feat_dim = q_dim
        valid_t0 = t_start
    elif feature_mode == "fsr_q":
        feat_dim = fsr_dim + q_dim
        valid_t0 = t_start
    elif feature_mode == "fsr_q_quality":
        if quality is None:
            raise ValueError(
                "feature_mode='fsr_q_quality' requires quality labels in H5."
            )
        feat_dim = fsr_dim + q_dim + quality_dim
        valid_t0 = t_start
    elif feature_mode == "window_fsr_q":
        valid_t0 = max(t_start, t_start + window - 1)
        feat_dim = window * (fsr_dim + q_dim)
    else:
        raise ValueError(f"Unknown feature mode: {feature_mode}")

    valid_len = t_stop - valid_t0
    if valid_len <= 0:
        raise ValueError(
            "No valid time indices in split. Try smaller window or different split."
        )

    candidate_count = valid_len * num_envs
    sample_count = candidate_count if max_samples is None else min(max_samples, candidate_count)
    flat_idx = np.arange(candidate_count, dtype=np.int64)
    if sample_count < candidate_count:
        flat_idx = rng.choice(flat_idx, size=sample_count, replace=False)

    x = np.zeros((sample_count, feat_dim), dtype=np.float32)
    y = np.zeros((sample_count, action_dim), dtype=np.float32)

    for i, idx in enumerate(flat_idx):
        t = valid_t0 + int(idx // num_envs)
        e = int(idx % num_envs)
        if feature_mode == "fsr":
            x[i] = fsr[t, e]
        elif feature_mode == "q":
            x[i] = q_hand[t, e]
        elif feature_mode == "fsr_q":
            x[i] = np.concatenate([fsr[t, e], q_hand[t, e]], axis=0)
        elif feature_mode == "fsr_q_quality":
            assert quality is not None
            x[i] = np.concatenate([fsr[t, e], q_hand[t, e], quality[t, e]], axis=0)
        else:
            t0 = t - window + 1
            x[i] = np.concatenate(
                [
                    fsr[t0 : t + 1, e].reshape(-1),
                    q_hand[t0 : t + 1, e].reshape(-1),
                ],
                axis=0,
            )
        y[i] = action[t, e]

    return x, y


def _build_split(
    fsr: np.ndarray,
    q_hand: np.ndarray,
    quality: np.ndarray | None,
    action: np.ndarray,
    feature_mode: str,
    window: int,
    train_ratio: float,
    val_ratio: float,
    max_train_samples: int | None,
    max_eval_samples: int | None,
    seed: int,
) -> SplitData:
    train_end, val_end = _time_splits(fsr.shape[0], train_ratio, val_ratio)
    rng = np.random.default_rng(seed)

    x_train, y_train = _sample_range(
        fsr,
        q_hand,
        quality,
        action,
        0,
        train_end,
        feature_mode,
        window,
        max_train_samples,
        rng,
    )
    x_val, y_val = _sample_range(
        fsr,
        q_hand,
        quality,
        action,
        train_end,
        val_end,
        feature_mode,
        window,
        max_eval_samples,
        rng,
    )
    x_test, y_test = _sample_range(
        fsr,
        q_hand,
        quality,
        action,
        val_end,
        fsr.shape[0],
        feature_mode,
        window,
        max_eval_samples,
        rng,
    )

    return SplitData(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
    )


def _run_ridge(split: SplitData, alpha: float, shuffle_train_target: bool) -> dict[str, float]:
    rng = np.random.default_rng(0)
    y_train = split.y_train.copy()
    if shuffle_train_target:
        perm = rng.permutation(y_train.shape[0])
        y_train = y_train[perm]

    x_train_z, x_mean, x_std = _standardize_fit(split.x_train)
    y_train_z, y_mean, y_std = _standardize_fit(y_train)

    x_val_z = _standardize_apply(split.x_val, x_mean, x_std)
    x_test_z = _standardize_apply(split.x_test, x_mean, x_std)

    y_val_pred_z = _ridge_fit_predict(x_train_z, y_train_z, x_val_z, alpha=alpha)
    y_test_pred_z = _ridge_fit_predict(x_train_z, y_train_z, x_test_z, alpha=alpha)

    y_val_pred = y_val_pred_z * y_std + y_mean
    y_test_pred = y_test_pred_z * y_std + y_mean

    val_m = _metrics(split.y_val, y_val_pred)
    test_m = _metrics(split.y_test, y_test_pred)

    out = {
        "val_mse": val_m["mse"],
        "val_mae": val_m["mae"],
        "val_r2": val_m["r2_mean"],
        "test_mse": test_m["mse"],
        "test_mae": test_m["mae"],
        "test_r2": test_m["r2_mean"],
    }
    return out


def _run_ridge_torch(
    split: SplitData,
    alpha: float,
    shuffle_train_target: bool,
    device: str,
) -> dict[str, float]:
    rng = np.random.default_rng(0)
    y_train_np = split.y_train.copy()
    if shuffle_train_target:
        perm = rng.permutation(y_train_np.shape[0])
        y_train_np = y_train_np[perm]

    x_train_z, x_mean, x_std = _standardize_fit(split.x_train)
    y_train_z, y_mean, y_std = _standardize_fit(y_train_np)
    x_val_z = _standardize_apply(split.x_val, x_mean, x_std)
    x_test_z = _standardize_apply(split.x_test, x_mean, x_std)

    dev = torch.device(device)
    x_train = torch.from_numpy(x_train_z).to(dev)
    y_train = torch.from_numpy(y_train_z).to(dev)
    x_val = torch.from_numpy(x_val_z).to(dev)
    x_test = torch.from_numpy(x_test_z).to(dev)

    ones_train = torch.ones((x_train.shape[0], 1), device=dev, dtype=x_train.dtype)
    ones_val = torch.ones((x_val.shape[0], 1), device=dev, dtype=x_val.dtype)
    ones_test = torch.ones((x_test.shape[0], 1), device=dev, dtype=x_test.dtype)

    x_train_aug = torch.cat([x_train, ones_train], dim=1)
    x_val_aug = torch.cat([x_val, ones_val], dim=1)
    x_test_aug = torch.cat([x_test, ones_test], dim=1)

    d = x_train_aug.shape[1]
    reg = torch.eye(d, device=dev, dtype=x_train_aug.dtype) * float(alpha)
    reg[-1, -1] = 0.0

    xtx = x_train_aug.T @ x_train_aug
    xty = x_train_aug.T @ y_train
    w = torch.linalg.solve(xtx + reg, xty)

    y_val_pred_z = x_val_aug @ w
    y_test_pred_z = x_test_aug @ w

    y_mean_t = torch.from_numpy(y_mean).to(dev)
    y_std_t = torch.from_numpy(y_std).to(dev)
    y_val_pred = (y_val_pred_z * y_std_t + y_mean_t).cpu().numpy()
    y_test_pred = (y_test_pred_z * y_std_t + y_mean_t).cpu().numpy()

    val_m = _metrics(split.y_val, y_val_pred)
    test_m = _metrics(split.y_test, y_test_pred)
    return {
        "val_mse": val_m["mse"],
        "val_mae": val_m["mae"],
        "val_r2": val_m["r2_mean"],
        "test_mse": test_m["mse"],
        "test_mae": test_m["mae"],
        "test_r2": test_m["r2_mean"],
    }


def _run_constant_baseline(split: SplitData) -> dict[str, float]:
    mean_action = split.y_train.mean(axis=0, keepdims=True)
    y_val_pred = np.repeat(mean_action, split.y_val.shape[0], axis=0)
    y_test_pred = np.repeat(mean_action, split.y_test.shape[0], axis=0)

    val_m = _metrics(split.y_val, y_val_pred)
    test_m = _metrics(split.y_test, y_test_pred)
    return {
        "val_mse": val_m["mse"],
        "val_mae": val_m["mae"],
        "val_r2": val_m["r2_mean"],
        "test_mse": test_m["mse"],
        "test_mae": test_m["mae"],
        "test_r2": test_m["r2_mean"],
    }


def _lag_scan(
    fsr: np.ndarray,
    action: np.ndarray,
    lag_max: int,
    max_pairs: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    means = []
    maxes = []
    rng = np.random.default_rng(0)
    use_torch = device != "cpu"
    dev = torch.device(device) if use_torch else None

    for lag in range(lag_max + 1):
        if lag == 0:
            x = fsr.reshape(-1, fsr.shape[-1])
            y = action.reshape(-1, action.shape[-1])
        else:
            x = fsr[:-lag].reshape(-1, fsr.shape[-1])
            y = action[lag:].reshape(-1, action.shape[-1])

        n = x.shape[0]
        if max_pairs > 0 and n > max_pairs:
            idx = rng.choice(n, size=max_pairs, replace=False)
            x = x[idx]
            y = y[idx]

        if use_torch and dev is not None:
            xt = torch.from_numpy(x).to(dev)
            yt = torch.from_numpy(y).to(dev)
            x_c = xt - xt.mean(dim=0, keepdim=True)
            y_c = yt - yt.mean(dim=0, keepdim=True)
            x_s = x_c.std(dim=0, keepdim=True)
            y_s = y_c.std(dim=0, keepdim=True)
            x_s = torch.where(x_s < 1e-8, torch.ones_like(x_s), x_s)
            y_s = torch.where(y_s < 1e-8, torch.ones_like(y_s), y_s)
            corr = (x_c / x_s).T @ (y_c / y_s)
            corr = corr / max(1, x.shape[0] - 1)
            abs_corr = torch.abs(corr)
            means.append(float(abs_corr.mean().item()))
            maxes.append(float(abs_corr.max().item()))
        else:
            x_c = x - x.mean(axis=0, keepdims=True)
            y_c = y - y.mean(axis=0, keepdims=True)
            x_s = x_c.std(axis=0, keepdims=True)
            y_s = y_c.std(axis=0, keepdims=True)
            x_s = np.where(x_s < 1e-8, 1.0, x_s)
            y_s = np.where(y_s < 1e-8, 1.0, y_s)

            corr = (x_c / x_s).T @ (y_c / y_s)
            corr = corr / max(1, x.shape[0] - 1)
            abs_corr = np.abs(corr)
            means.append(float(abs_corr.mean()))
            maxes.append(float(abs_corr.max()))

    return np.asarray(means, dtype=np.float32), np.asarray(maxes, dtype=np.float32)


def _print_metrics(name: str, m: dict[str, float]) -> None:
    print(
        f"{name:>20} | "
        f"val_r2={m['val_r2']:+.3f} test_r2={m['test_r2']:+.3f} | "
        f"val_mse={m['val_mse']:.5f} test_mse={m['test_mse']:.5f}"
    )


def _interpret(
    const_m: dict[str, float],
    fsr_m: dict[str, float],
    fsr_q_m: dict[str, float],
    win_m: dict[str, float],
    shuffle_m: dict[str, float],
    lag_mean: np.ndarray,
    lag_max: np.ndarray,
) -> list[str]:
    notes: list[str] = []

    gap_vs_shuffle = fsr_q_m["test_r2"] - shuffle_m["test_r2"]
    gap_vs_const = fsr_q_m["test_r2"] - const_m["test_r2"]
    history_gain = win_m["test_r2"] - fsr_q_m["test_r2"]

    best_lag = int(np.argmax(lag_mean))
    best_lag_peak = int(np.argmax(lag_max))

    if fsr_q_m["test_r2"] < 0.10:
        notes.append("Overall predictive signal appears weak (test R2 < 0.10).")
    elif fsr_q_m["test_r2"] < 0.30:
        notes.append("Predictive signal is moderate (0.10 <= test R2 < 0.30).")
    else:
        notes.append("Predictive signal is strong (test R2 >= 0.30).")

    if gap_vs_shuffle > 0.15:
        notes.append(
            "Real-signal check passed: model beats shuffled-target control clearly."
        )
    else:
        notes.append(
            "Shuffle control gap is small; check for weak signal or feature mismatch."
        )

    if gap_vs_const > 0.15:
        notes.append("Model clearly improves over constant baseline.")
    else:
        notes.append("Improvement over constant baseline is limited.")

    if history_gain > 0.05:
        notes.append("History window helps; dynamics/time-dependence are important.")
    else:
        notes.append("Windowed history gives limited gain; mapping may be near-instant.")

    if best_lag > 0 or best_lag_peak > 0:
        notes.append(
            f"Lag scan peak is not at lag=0 (mean@lag={best_lag}, max@lag={best_lag_peak})."
        )
    else:
        notes.append("Lag scan peaks at lag=0; immediate mapping dominates.")

    return notes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether FSR/q contains predictive signal for action."
    )
    parser.add_argument(
        "--h5",
        type=str,
        default=None,
        help="Path to one H5 file. If omitted, latest file from --glob is used.",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="./finger_compliance_control/data/headless/*.h5",
        help="Fallback glob to find latest H5 when --h5 is not set.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Compute backend. 'cuda' uses GPU with torch if available.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=120000,
        help="Load at most this many timesteps from H5 (after stride).",
    )
    parser.add_argument(
        "--step-stride",
        type=int,
        default=1,
        help="Use every N-th timestep when loading H5.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=250000,
        help="Cap sampled train pairs per baseline to avoid OOM.",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=80000,
        help="Cap sampled val/test pairs per baseline to avoid OOM.",
    )
    parser.add_argument(
        "--lag-max-pairs",
        type=int,
        default=1200000,
        help="Cap sampled pair count per lag in lag scan.",
    )
    parser.add_argument(
        "--drop-palm-fsr",
        action="store_true",
        help="Ignore palm FSR channels [0..3] when building model features.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=20,
        help="History window for window_fsr_q baseline.",
    )
    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=1.0,
        help="L2 regularization for ridge baselines.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Train ratio in time-wise split.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation ratio in time-wise split.",
    )
    parser.add_argument(
        "--lag-max",
        type=int,
        default=15,
        help="Maximum lag for lag scan corr(FSR_t, action_{t+lag}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    h5_path = _find_input_file(args.h5, args.glob)

    if args.device == "auto":
        compute_device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available.")
        compute_device = "cuda"
    else:
        compute_device = "cpu"

    fsr_raw, q, action, quality = _load_arrays(
        h5_path,
        max_steps=args.max_steps,
        step_stride=args.step_stride,
    )
    fsr = _select_fsr_features(fsr_raw, drop_palm_fsr=args.drop_palm_fsr)
    qh = _q_hand(q)

    print("=" * 72)
    print("Signal-Strength Diagnostics")
    print("=" * 72)
    print(f"File: {h5_path}")
    print(
        "Shapes: "
        f"fsr_raw={tuple(fsr_raw.shape)}, fsr_feat={tuple(fsr.shape)}, "
        f"q={tuple(q.shape)}, action={tuple(action.shape)}"
    )
    if quality is None:
        print("Quality labels: not found in this H5 (running without quality features)")
    else:
        print(f"Quality labels: loaded with shape={tuple(quality.shape)}")
    print(
        "Split/params: "
        f"train={args.train_ratio:.2f}, val={args.val_ratio:.2f}, "
        f"test={1.0 - args.train_ratio - args.val_ratio:.2f}, "
        f"window={args.window}, lag_max={args.lag_max}, ridge_alpha={args.ridge_alpha}, "
        f"device={compute_device}, drop_palm_fsr={args.drop_palm_fsr}"
    )
    print(
        "Sampling caps: "
        f"max_steps={args.max_steps}, step_stride={args.step_stride}, "
        f"train={args.max_train_samples}, eval={args.max_eval_samples}, "
        f"lag_pairs={args.lag_max_pairs}"
    )

    split_fsr = _build_split(
        fsr,
        qh,
        quality,
        action,
        feature_mode="fsr",
        window=args.window,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        seed=1,
    )
    split_fsr_q = _build_split(
        fsr,
        qh,
        quality,
        action,
        feature_mode="fsr_q",
        window=args.window,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        seed=2,
    )
    split_q = _build_split(
        fsr,
        qh,
        quality,
        action,
        feature_mode="q",
        window=args.window,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        seed=4,
    )
    split_win = _build_split(
        fsr,
        qh,
        quality,
        action,
        feature_mode="window_fsr_q",
        window=args.window,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        seed=3,
    )
    split_fsr_q_quality: SplitData | None = None
    if quality is not None:
        split_fsr_q_quality = _build_split(
            fsr,
            qh,
            quality,
            action,
            feature_mode="fsr_q_quality",
            window=args.window,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            max_train_samples=args.max_train_samples,
            max_eval_samples=args.max_eval_samples,
            seed=5,
        )

    const_m = _run_constant_baseline(split_fsr_q)
    if compute_device == "cuda":
        fsr_m = _run_ridge_torch(split_fsr, args.ridge_alpha, False, compute_device)
        q_m = _run_ridge_torch(split_q, args.ridge_alpha, False, compute_device)
        fsr_q_m = _run_ridge_torch(
            split_fsr_q, args.ridge_alpha, False, compute_device
        )
        fsr_q_quality_m = None
        if split_fsr_q_quality is not None:
            fsr_q_quality_m = _run_ridge_torch(
                split_fsr_q_quality,
                args.ridge_alpha,
                False,
                compute_device,
            )
        win_m = _run_ridge_torch(split_win, args.ridge_alpha, False, compute_device)
        shuffle_m = _run_ridge_torch(
            split_fsr_q, args.ridge_alpha, True, compute_device
        )
    else:
        fsr_m = _run_ridge(split_fsr, args.ridge_alpha, False)
        q_m = _run_ridge(split_q, args.ridge_alpha, False)
        fsr_q_m = _run_ridge(split_fsr_q, args.ridge_alpha, False)
        fsr_q_quality_m = None
        if split_fsr_q_quality is not None:
            fsr_q_quality_m = _run_ridge(split_fsr_q_quality, args.ridge_alpha, False)
        win_m = _run_ridge(split_win, args.ridge_alpha, False)
        shuffle_m = _run_ridge(split_fsr_q, args.ridge_alpha, True)

    print("\n[Model comparison]")
    _print_metrics("constant", const_m)
    _print_metrics("ridge(fsr)", fsr_m)
    _print_metrics("ridge(q)", q_m)
    _print_metrics("ridge(fsr+q)", fsr_q_m)
    if fsr_q_quality_m is not None:
        _print_metrics("ridge(fsr+q+quality)", fsr_q_quality_m)
    _print_metrics("ridge(window)", win_m)
    _print_metrics("ridge(fsr+q,shuf)", shuffle_m)

    lag_mean, lag_peak = _lag_scan(
        fsr,
        action,
        lag_max=args.lag_max,
        max_pairs=args.lag_max_pairs,
        device=compute_device,
    )
    best_lag_mean = int(np.argmax(lag_mean))
    best_lag_peak = int(np.argmax(lag_peak))
    print("\n[Lag scan] corr(FSR_t, action_{t+lag})")
    print(
        "Best lag by mean|corr|: "
        f"{best_lag_mean} (score={lag_mean[best_lag_mean]:.4f})"
    )
    print(
        "Best lag by max|corr|: "
        f"{best_lag_peak} (score={lag_peak[best_lag_peak]:.4f})"
    )

    print("\n[Interpretation]")
    for line in _interpret(
        const_m,
        fsr_m,
        fsr_q_m,
        win_m,
        shuffle_m,
        lag_mean,
        lag_peak,
    ):
        print(f"- {line}")


if __name__ == "__main__":
    main()
