"""Train a LeRobot conditional 1-D U-Net diffusion policy for fingertip pose."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dp_dataset import (
    ACTION_DIM,
    ACTION_REPRESENTATIONS,
    ENV_STATE_DIM,
    ROBOT_STATE_DIM,
    ActionRepresentation,
    FingertipDiffusionDataset,
    compute_normalization,
    input_frame,
    load_episodes,
    split_episode_ids,
    state_dimensions,
    state_fields,
    state_schema,
)


def build_policy(args: argparse.Namespace, device: torch.device) -> DiffusionPolicy:
    """Build the official LeRobot DiffusionPolicy without image observations."""
    robot_state_dim = int(getattr(args, "robot_state_dim", ROBOT_STATE_DIM))
    environment_state_dim = int(
        getattr(args, "environment_state_dim", ENV_STATE_DIM)
    )
    config = DiffusionConfig(
        input_features={
            "observation.state": PolicyFeature(
                type=FeatureType.STATE, shape=(robot_state_dim,)
            ),
            "observation.environment_state": PolicyFeature(
                type=FeatureType.ENV, shape=(environment_state_dim,)
            ),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))
        },
        n_obs_steps=args.obs_horizon,
        horizon=args.pred_horizon,
        # We use conditional_sample() to consume the complete future horizon.
        n_action_steps=1,
        device=str(device),
        down_dims=tuple(args.down_dims),
        kernel_size=args.kernel_size,
        n_groups=args.n_groups,
        diffusion_step_embed_dim=args.diffusion_step_embed_dim,
        noise_scheduler_type=args.noise_scheduler,
        num_train_timesteps=args.diffusion_steps,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
        clip_sample=True,
        clip_sample_range=1.0,
        num_inference_steps=args.inference_steps,
    )
    return DiffusionPolicy(config).to(device)


def _to_device(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
    }


@torch.no_grad()
def validation_loss(
    policy: DiffusionPolicy,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 50,
) -> float:
    policy.eval()
    losses = []
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        losses.append(policy.diffusion.compute_loss(_to_device(batch, device)).item())
    policy.train()
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def overfit_metrics(
    policy: DiffusionPolicy,
    dataset: FingertipDiffusionDataset,
    device: torch.device,
    action_mean: np.ndarray,
    action_scale: np.ndarray,
    action_representation: ActionRepresentation,
    sample_count: int,
    seed: int,
) -> dict[str, float]:
    """Sample complete future trajectories and compare with teacher labels."""
    if sample_count <= 0 or len(dataset) == 0:
        return {}
    indices = np.linspace(
        0, len(dataset) - 1, min(sample_count, len(dataset)), dtype=np.int64
    )
    samples = [dataset[int(index)] for index in indices]
    batch = {
        key: torch.stack([sample[key] for sample in samples]).to(device)
        for key in samples[0]
    }
    observation = {
        "observation.state": batch["observation.state"],
        "observation.environment_state": batch["observation.environment_state"],
    }
    generator = torch.Generator(device=device).manual_seed(seed)
    global_condition = policy.diffusion._prepare_global_conditioning(observation)
    prediction = policy.diffusion.conditional_sample(
        len(indices), global_cond=global_condition, generator=generator
    )
    target = batch["action"]
    mean = torch.as_tensor(action_mean, device=device).view(1, 1, -1)
    scale = torch.as_tensor(action_scale, device=device).view(1, 1, -1)
    error = (prediction - target) * scale
    target_physical = target * scale + mean
    current_q = torch.as_tensor(
        np.stack([dataset.current_q(int(index)) for index in indices]),
        device=device,
        dtype=torch.float32,
    ).unsqueeze(1)
    if action_representation == "delta_q":
        hold_action = torch.zeros_like(current_q)
    else:
        hold_action = current_q
    baseline_error = hold_action - target_physical
    return {
        "sample_count": int(len(indices)),
        "sample_mae_rad": float(error.abs().mean()),
        "sample_rmse_rad": float(error.square().mean().sqrt()),
        "sample_final_mae_rad": float(error[:, -1].abs().mean()),
        "hold_current_q_baseline_mae_rad": float(baseline_error.abs().mean()),
        "hold_current_q_baseline_final_mae_rad": float(
            baseline_error[:, -1].abs().mean()
        ),
    }


def write_metrics(
    output: Path,
    records: list[dict[str, float | int]],
) -> None:
    """Persist machine-readable metrics and a final training-curve PNG."""
    (output / "metrics.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    if records:
        with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = [int(record["step"]) for record in records]
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].plot(
            steps,
            [float(record["train_noise_loss"]) for record in records],
            marker="o",
            label="train noise loss",
        )
        val_loss = [float(record["val_noise_loss"]) for record in records]
        if np.isfinite(val_loss).any():
            axes[0].plot(steps, val_loss, marker="o", label="val noise loss")
        axes[0].set_yscale("log")
        axes[0].set_xlabel("training step")
        axes[0].set_ylabel("MSE")
        axes[0].set_title("Diffusion noise prediction")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        for key, label in (
            ("train_sample_mae_rad", "train generated MAE"),
            ("val_sample_mae_rad", "val generated MAE"),
            ("train_hold_q_mae_rad", "train hold-q baseline"),
            ("val_hold_q_mae_rad", "val hold-q baseline"),
        ):
            values = [float(record[key]) for record in records]
            if np.isfinite(values).any():
                axes[1].plot(steps, values, marker="o", label=label)
        axes[1].set_yscale("log")
        axes[1].set_xlabel("training step")
        axes[1].set_ylabel("joint trajectory MAE [rad]")
        axes[1].set_title("Generated future trajectory")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        figure.tight_layout()
        figure.savefig(output / "training_curves.png", dpi=180)
        plt.close(figure)
    except ImportError:
        print("[WARNING] matplotlib unavailable; CSV/JSON metrics were saved without PNG")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--obs-horizon", type=int, default=16)
    parser.add_argument("--pred-horizon", type=int, default=32)
    parser.add_argument(
        "--action-representation",
        choices=ACTION_REPRESENTATIONS,
        default="delta_q",
        help=(
            "delta_q predicts q_future-q_current (legacy); absolute_q predicts "
            "future joint positions directly and avoids integrating DP bias."
        ),
    )
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--inference-steps", type=int, default=100)
    parser.add_argument("--noise-scheduler", choices=("DDPM", "DDIM"), default="DDPM")
    parser.add_argument("--down-dims", type=int, nargs="+", default=(256, 512, 1024))
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--n-groups", type=int, default=8)
    parser.add_argument("--diffusion-step-embed-dim", type=int, default=128)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--contact-dropout-probability",
        type=float,
        default=0.10,
        help=(
            "Geometry schema only: probability per finger/window of simulating "
            "a short tactile contact loss."
        ),
    )
    parser.add_argument(
        "--max-contact-dropout-steps",
        type=int,
        default=3,
        help="Maximum stride-rate samples held during synthetic contact loss.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=10_000)
    parser.add_argument(
        "--eval-every",
        type=int,
        default=1_000,
        help="Record train/validation metrics at this interval.",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=64,
        help="Training-window trajectories sampled after each checkpoint; 0 disables.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output = args.output or Path(
        f"mcc_finger_compliance_control/data/models/dp_unet_{datetime.now():%Y%m%d_%H%M%S}"
    )
    output.mkdir(parents=True, exist_ok=True)

    downsampling_factor = 2 ** len(args.down_dims)
    if args.pred_horizon % downsampling_factor:
        raise ValueError(
            f"pred-horizon={args.pred_horizon} must be divisible by "
            f"2**len(down_dims)={downsampling_factor}"
        )

    episodes = load_episodes(args.file, args.stride)
    dataset_input_frame = input_frame(args.file)
    dataset_state_schema = state_schema(args.file)
    dataset_state_fields = state_fields(args.file)
    robot_state_dim, environment_state_dim, state_dim = state_dimensions(args.file)
    args.robot_state_dim = robot_state_dim
    args.environment_state_dim = environment_state_dim
    args.state_dim = state_dim
    args.state_schema = dataset_state_schema
    train_ids, val_ids = split_episode_ids(list(episodes), args.val_ratio, args.seed)
    normalization = compute_normalization(
        episodes,
        train_ids,
        args.action_representation,
        dataset_state_schema,
    )
    train_set = FingertipDiffusionDataset(
        episodes,
        train_ids,
        normalization,
        args.obs_horizon,
        args.pred_horizon,
        robot_state_dim,
        dataset_state_schema,
        args.contact_dropout_probability,
        args.max_contact_dropout_steps,
        args.action_representation,
    )
    val_set = FingertipDiffusionDataset(
        episodes,
        val_ids,
        normalization,
        args.obs_horizon,
        args.pred_horizon,
        robot_state_dim,
        dataset_state_schema,
        0.0,
        args.max_contact_dropout_steps,
        args.action_representation,
    )
    if not train_set:
        raise ValueError("No training windows for the selected horizons")
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    policy = build_policy(args, device)
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=args.lr, weight_decay=1.0e-5
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    iterator = iter(train_loader)
    progress = tqdm(range(1, args.steps + 1), dynamic_ncols=True, desc="LeRobot DP")
    best_metric = float("inf")
    metric_records: list[dict[str, float | int]] = []

    def save(step: int, loss: float, metrics: dict[str, float], name: str) -> None:
        torch.save(
            {
                "step": step,
                "model": policy.state_dict(),
                "optimizer": optimizer.state_dict(),
                "normalization": asdict(normalization),
                "config": vars(args),
                "architecture": "lerobot_diffusion_conditional_unet1d",
                "state_dim": state_dim,
                "robot_state_dim": robot_state_dim,
                "environment_state_dim": environment_state_dim,
                "action_dim": ACTION_DIM,
                "action_representation": args.action_representation,
                "input_frame": dataset_input_frame,
                "state_schema": dataset_state_schema,
                "state_fields": dataset_state_fields,
                "train_episode_ids": train_ids,
                "val_episode_ids": val_ids,
                "noise_loss": loss,
                "trajectory_metrics": metrics,
            },
            output / name,
        )

    for step in progress:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        batch = _to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            loss = policy.diffusion.compute_loss(batch)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if step % 100 == 0:
            progress.set_postfix(loss=f"{loss.item():.5f}")
        should_evaluate = step % args.eval_every == 0 or step == args.steps
        should_save = step % args.save_every == 0 or step == args.steps
        if should_evaluate:
            val_noise_loss = (
                validation_loss(policy, val_loader, device)
                if val_ids
                else float("nan")
            )
            train_metrics = overfit_metrics(
                policy,
                train_set,
                device,
                normalization.action_mean,
                normalization.action_std,
                args.action_representation,
                args.eval_samples,
                args.seed,
            )
            val_metrics = overfit_metrics(
                policy,
                val_set,
                device,
                normalization.action_mean,
                normalization.action_std,
                args.action_representation,
                args.eval_samples,
                args.seed + 1,
            )
            record: dict[str, float | int] = {
                "step": step,
                "train_noise_loss": float(loss.item()),
                "val_noise_loss": val_noise_loss,
                "train_sample_mae_rad": train_metrics.get(
                    "sample_mae_rad", float("nan")
                ),
                "val_sample_mae_rad": val_metrics.get(
                    "sample_mae_rad", float("nan")
                ),
                "train_sample_final_mae_rad": train_metrics.get(
                    "sample_final_mae_rad", float("nan")
                ),
                "val_sample_final_mae_rad": val_metrics.get(
                    "sample_final_mae_rad", float("nan")
                ),
                "train_hold_q_mae_rad": train_metrics.get(
                    "hold_current_q_baseline_mae_rad", float("nan")
                ),
                "val_hold_q_mae_rad": val_metrics.get(
                    "hold_current_q_baseline_mae_rad", float("nan")
                ),
            }
            metric_records.append(record)
            write_metrics(output, metric_records)
            selection_metric = (
                float(record["val_sample_mae_rad"])
                if val_ids
                else float(record["train_sample_mae_rad"])
            )
            checkpoint_metrics = {
                "train": train_metrics,
                "validation": val_metrics,
                "record": record,
            }
            if should_save:
                save(
                    step,
                    val_noise_loss,
                    checkpoint_metrics,
                    f"checkpoint_{step:07d}.pt",
                )
                save(step, val_noise_loss, checkpoint_metrics, "latest.pt")
            if selection_metric < best_metric:
                best_metric = selection_metric
                save(step, val_noise_loss, checkpoint_metrics, "best.pt")
            print(
                f"[DP] step={step} train_noise={loss.item():.6f} "
                f"val_noise={val_noise_loss:.6f} "
                f"train_mae={record['train_sample_mae_rad']:.6f}rad "
                f"val_mae={record['val_sample_mae_rad']:.6f}rad"
            )

    metadata = {
        "source": str(args.file),
        "input_frame": dataset_input_frame,
        "state_schema": dataset_state_schema,
        "action_representation": args.action_representation,
        "architecture": "LeRobot DiffusionPolicy / conditional 1-D U-Net",
        "input": {
            "observation.state": {
                name: size for name, size in dataset_state_fields[:-1]
            },
            "observation.environment_state": {
                dataset_state_fields[-1][0]: dataset_state_fields[-1][1]
            },
            "total_dim": state_dim,
        },
        "output": {
            (
                "future_q_hand_delta"
                if args.action_representation == "delta_q"
                else "future_q_hand_absolute"
            ): [args.pred_horizon, ACTION_DIM]
        },
        "train_episodes": len(train_ids),
        "val_episodes": len(val_ids),
        "train_windows": len(train_set),
        "val_windows": len(val_set),
    }
    (output / "dataset_info.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"[SUCCESS] model directory: {output}")


if __name__ == "__main__":
    main()
