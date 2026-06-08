#!/usr/bin/env python3
"""Offline replay consistency test for LeRobot policy.

Feeds training-dataset states into the policy and compares postprocessed action
with ground-truth action labels from H5.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import h5py
import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors

DEFAULT_MODEL_DIR = (
    "/home/rimlab/Code/lerobot/outputs/robot_learning_tutorial/diffusion_h5_state_only_overfit_128"
)
DEFAULT_DATA_GLOB = "finger_compliance_control/data/headless/headless_train_20260414_175152.h5"


def _resolve_h5(path: str | None) -> Path:
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"H5 not found: {p}")
        return p

    candidates = [
        Path(p)
        for p in glob.glob(DEFAULT_DATA_GLOB)
        if not p.endswith("_inverted.h5")
    ]
    if not candidates:
        raise FileNotFoundError(
            "No H5 files found. Pass --h5-file explicitly or place data under "
            f"{DEFAULT_DATA_GLOB}."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _load_h5(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as f:
        q = np.asarray(f["q"], dtype=np.float32)
        fsr = np.asarray(f["fsr"], dtype=np.float32)
        action = np.asarray(f["action"], dtype=np.float32)

    if q.ndim != 3 or fsr.ndim != 3 or action.ndim != 3:
        raise ValueError(
            f"Expected [T, N, D] arrays, got q={q.shape}, fsr={fsr.shape}, action={action.shape}"
        )

    if q.shape[-1] < 22:
        raise ValueError(f"q dim should be >=22, got {q.shape[-1]}")
    if fsr.shape[-1] < 16:
        raise ValueError(f"fsr dim should be >=16, got {fsr.shape[-1]}")
    if action.shape[-1] < 16:
        raise ValueError(f"action dim should be >=16, got {action.shape[-1]}")

    q22 = q[..., :22]
    fsr16 = fsr[..., :16]
    a16 = action[..., :16]
    return q22, fsr16, a16


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_mean = y_true.mean(axis=0, keepdims=True)
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - y_mean) ** 2, axis=0)
    valid = ss_tot > 1e-12
    r2 = np.zeros_like(ss_tot, dtype=np.float32)
    r2[valid] = 1.0 - ss_res[valid] / ss_tot[valid]
    return float(r2.mean())


def _cosine_mean(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    dot = np.sum(y_true * y_pred, axis=-1)
    n1 = np.linalg.norm(y_true, axis=-1)
    n2 = np.linalg.norm(y_pred, axis=-1)
    denom = np.clip(n1 * n2, 1e-8, None)
    return float(np.mean(dot / denom))


def _sign_match(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 0.05) -> float:
    mask = np.abs(y_true) > eps
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.sign(y_true[mask]) == np.sign(y_pred[mask])))


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline replay consistency test.")
    parser.add_argument("--model-dir", type=str, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--h5-file", type=str, default=DEFAULT_DATA_GLOB)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--env-state-value", type=float, default=0.0)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    h5_path = _resolve_h5(args.h5_file)
    q, fsr, gt_action = _load_h5(h5_path)
    state = np.concatenate([q, fsr], axis=-1).astype(np.float32)

    flat_state = state.reshape(-1, state.shape[-1])
    flat_gt = gt_action.reshape(-1, gt_action.shape[-1])

    if args.max_samples > 0:
        flat_state = flat_state[: args.max_samples]
        flat_gt = flat_gt[: args.max_samples]

    model_dir = Path(args.model_dir)
    policy = DiffusionPolicy.from_pretrained(model_dir, local_files_only=True)
    if args.num_inference_steps > 0:
        policy.diffusion.num_inference_steps = int(args.num_inference_steps)
    policy.eval()

    cfg = PreTrainedConfig.from_pretrained(model_dir, local_files_only=True)
    preprocessor, postprocessor = make_pre_post_processors(
        cfg,
        pretrained_path=str(model_dir),
    )

    model_device = next(policy.parameters()).device
    run_device = torch.device(args.device) if args.device else model_device
    policy.to(run_device)

    env_state_dim = 0
    if "observation.environment_state" in policy.config.input_features:
        env_shape = policy.config.input_features["observation.environment_state"].shape
        env_state_dim = int(np.prod(env_shape))

    preds: list[np.ndarray] = []
    total = flat_state.shape[0]
    bs = max(1, int(args.batch_size))

    for i in range(0, total, bs):
        s = torch.from_numpy(flat_state[i : i + bs]).to(run_device)
        batch: dict[str, torch.Tensor] = {"observation.state": s}
        if env_state_dim > 0:
            batch["observation.environment_state"] = torch.full(
                (s.shape[0], env_state_dim),
                float(args.env_state_value),
                device=run_device,
                dtype=torch.float32,
            )

        with torch.no_grad():
            batch = preprocessor(batch)
            out = policy.select_action(batch)
            out = postprocessor(out)
        preds.append(out.detach().cpu().numpy())

    pred = np.concatenate(preds, axis=0)
    common = min(pred.shape[0], flat_gt.shape[0])
    pred = pred[:common]
    gt = flat_gt[:common]

    mae = float(np.mean(np.abs(pred - gt)))
    rmse = float(np.sqrt(np.mean((pred - gt) ** 2)))
    r2 = _r2_score(gt, pred)
    cos = _cosine_mean(gt, pred)
    sign_acc = _sign_match(gt, pred)

    print(f"[INFO] H5: {h5_path}")
    print(f"[INFO] samples={pred.shape[0]}, action_dim={pred.shape[1]}")
    print(
        "[INFO] config: "
        f"device={run_device}, num_inference_steps={policy.diffusion.num_inference_steps}, "
        f"env_state_value={args.env_state_value}"
    )
    print(
        "[RESULT] "
        f"MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}, Cosine={cos:.4f}, SignAcc@|gt|>0.05={sign_acc:.4f}"
    )
    print(
        "[RANGE] "
        f"pred(mean={float(pred.mean()):.4f}, abs_mean={float(np.abs(pred).mean()):.4f}, "
        f"min={float(pred.min()):.4f}, max={float(pred.max()):.4f}) | "
        f"gt(mean={float(gt.mean()):.4f}, abs_mean={float(np.abs(gt).mean()):.4f}, "
        f"min={float(gt.min()):.4f}, max={float(gt.max()):.4f})"
    )


if __name__ == "__main__":
    main()
