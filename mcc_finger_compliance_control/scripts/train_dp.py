"""Train a LeRobot conditional 1-D U-Net diffusion policy for fingertip pose."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm

from dp_dataset import (
    ACTION_DIM,
    ACTION_REPRESENTATIONS,
    ENV_STATE_DIM,
    ROBOT_STATE_DIM,
    ActionRepresentation,
    FingertipDiffusionDataset,
    Normalization,
    compute_normalization,
    control_dt,
    episode_domains,
    input_frame,
    load_episodes,
    motion_configuration,
    planner_configuration,
    split_episode_ids,
    state_dimensions,
    state_field_groups,
    state_fields,
    state_schema,
)


def build_policy(args: argparse.Namespace, device: torch.device) -> DiffusionPolicy:
    """Build the official LeRobot DiffusionPolicy without image observations."""
    robot_state_dim = int(getattr(args, "robot_state_dim", ROBOT_STATE_DIM))
    environment_state_dim = int(
        getattr(args, "environment_state_dim", ENV_STATE_DIM)
    )
    action_dim = int(getattr(args, "action_dim", ACTION_DIM))
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
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))
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
    teacher_future = torch.as_tensor(
        np.stack([dataset.future_q(int(index)) for index in indices]),
        device=device,
        dtype=torch.float32,
    )
    action_dim = teacher_future.shape[-1]
    # Joint-space baselines (hold-current-q, kinematic) are defined for 16D
    # joint actions; Cartesian intent channels (tip_delta, 12D) instead use the
    # zero-delta hold-tip baseline — predicting no displacement, whose error
    # equals the teacher displacement magnitude itself (see the else branch).
    if action_dim == ACTION_DIM:
        current_q = torch.as_tensor(
            np.stack([dataset.current_q(int(index)) for index in indices]),
            device=device,
            dtype=torch.float32,
        ).unsqueeze(1)
        baseline_error = current_q - teacher_future
        kinematic = torch.as_tensor(
            np.stack([dataset.kinematic_baseline(int(index)) for index in indices]),
            device=device,
            dtype=torch.float32,
        )
        kinematic_error = kinematic - teacher_future
        hold_mae = float(baseline_error.abs().mean())
        hold_first = float(baseline_error[:, 0].abs().mean())
        hold_first4 = float(baseline_error[:, :4].abs().mean())
        hold_final = float(baseline_error[:, -1].abs().mean())
        kinematic_mae = float(kinematic_error.abs().mean())
        kinematic_first4 = float(kinematic_error[:, :4].abs().mean())
    else:
        # 12D Cartesian intent (tip_delta, meters): the natural reference is the
        # zero-delta hold-tip baseline — predicting no displacement at all, whose
        # error equals the teacher displacement magnitude itself. The kinematic
        # baseline degenerates to the same zero-delta value for Cartesian intents
        # (dp_dataset.kinematic_baseline returns zeros), so both coincide.
        baseline_error = teacher_future
        hold_mae = float(baseline_error.abs().mean())
        hold_first = float(baseline_error[:, 0].abs().mean())
        hold_first4 = float(baseline_error[:, :4].abs().mean())
        hold_final = float(baseline_error[:, -1].abs().mean())
        kinematic_mae = hold_mae
        kinematic_first4 = hold_first4
    return {
        "sample_count": int(len(indices)),
        "action_unit": "rad" if action_dim == ACTION_DIM else "m",
        "sample_mae_rad": float(error.abs().mean()),
        "sample_first_step_mae_rad": float(error[:, 0].abs().mean()),
        "sample_first_4_mae_rad": float(error[:, :4].abs().mean()),
        "sample_rmse_rad": float(error.square().mean().sqrt()),
        "sample_final_mae_rad": float(error[:, -1].abs().mean()),
        "hold_current_q_baseline_mae_rad": hold_mae,
        "hold_current_q_baseline_first_step_mae_rad": hold_first,
        "hold_current_q_baseline_first_4_mae_rad": hold_first4,
        "hold_current_q_baseline_final_mae_rad": hold_final,
        "kinematic_baseline_mae_rad": kinematic_mae,
        "kinematic_baseline_first_4_mae_rad": kinematic_first4,
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

        unit = str(records[0].get("action_unit", "rad"))
        is_joint = unit == "rad"
        baseline_label = "hold-q" if is_joint else "hold-tip (zero-delta)"
        for key, label in (
            ("train_sample_mae_rad", "train generated MAE"),
            ("val_sample_mae_rad", "val generated MAE"),
            ("train_hold_q_mae_rad", f"train {baseline_label} baseline"),
            ("val_hold_q_mae_rad", f"val {baseline_label} baseline"),
            # Kinematic baseline exists only for joint actions; for Cartesian
            # intent channels it degenerates to the identical zero-delta value.
            ("train_kinematic_baseline_mae_rad", "train kinematic baseline"),
            ("val_kinematic_baseline_mae_rad", "val kinematic baseline"),
        ):
            if not is_joint and "kinematic" in key:
                continue
            values = [float(record[key]) for record in records]
            if np.isfinite(values).any():
                axes[1].plot(steps, values, marker="o", label=label)
        axes[1].set_yscale("log")
        axes[1].set_xlabel("training step")
        axes[1].set_ylabel(f"future trajectory MAE [{unit}]")
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
    parser.add_argument(
        "--exclude-episode-ids",
        type=int,
        nargs="*",
        default=(),
        help=(
            "Drop these episode ids from --file before the train/validation "
            "split; validation is drawn from the remaining episodes only."
        ),
    )
    parser.add_argument(
        "--dagger-file",
        type=Path,
        nargs="*",
        default=(),
        help=(
            "Optional closed-loop DAgger DP files. Their live observation "
            "episodes are training-only; validation always comes from --file."
        ),
    )
    parser.add_argument(
        "--dagger-sample-ratio",
        type=float,
        default=0.0,
        help="Target fraction of training batches sampled from DAgger windows.",
    )
    parser.add_argument(
        "--max-dagger-action-outside-fraction",
        type=float,
        default=0.10,
        help=(
            "Abort when more than this fraction of DAgger normalized action "
            "scalars exceed [-1,1]. Use a negative value only for diagnostics."
        ),
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Initialize policy weights from an existing compatible checkpoint.",
    )
    parser.add_argument(
        "--preserve-checkpoint-split",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "With --resume, retain its clean train/validation episode split "
            "and add every DAgger episode to training only."
        ),
    )
    parser.add_argument(
        "--reuse-checkpoint-normalization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "With --resume, keep the original state/action normalization so "
            "the initialized network and deployment action scale remain aligned."
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--obs-horizon", type=int, default=16)
    parser.add_argument("--pred-horizon", type=int, default=32)
    parser.add_argument(
        "--input-profile",
        choices=("B2", "B0"),
        default="B2",
        help=(
            "B2 uses every exported dual-track-v3 channel. B0 keeps only "
            "the 84-D execution-observation subset while retaining the same "
            "242-D network interface; disabled normalized channels are zero."
        ),
    )
    parser.add_argument(
        "--action-representation",
        choices=ACTION_REPRESENTATIONS,
        default="absolute_q",
        help=(
            "absolute_q (default) predicts future joint positions directly and "
            "avoids integrating DP bias; kinematic_residual_q predicts a local "
            "constant-velocity correction but reconstructs absolute q before "
            "execution; delta_q is retained only for legacy A/B."
        ),
    )
    parser.add_argument(
        "--kinematic-velocity-clip-rad-s",
        type=float,
        default=1.0,
        help="Per-joint velocity bound used by the local kinematic baseline.",
    )
    parser.add_argument(
        "--action-field",
        default=None,
        help="Label channel override for dual-track A/B: 'q_ref' (16D joint "
        "intent, Variant A) or 'tip_delta_tangent_palm' (12D tangential "
        "fingertip intent, Variant B). Defaults to the H5 attrs.",
    )
    parser.add_argument(
        "--action-dim",
        type=int,
        default=None,
        help="Label dimensionality override (16 for q_ref, 12 for "
        "tip_delta_tangent_palm). Defaults to the H5 attrs.",
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
    parser.add_argument(
        "--qhand-dropout-probability",
        type=float,
        default=0.0,
        help=(
            "Window-level probability of a synthetic q_hand encoder glitch: "
            "one finger's 4-joint block receives additive Gaussian noise over "
            "a short contiguous span (training only; validation is clean)."
        ),
    )
    parser.add_argument(
        "--qhand-dropout-max-steps",
        type=int,
        default=6,
        help="Maximum stride-rate samples held during a q_hand glitch.",
    )
    parser.add_argument(
        "--qhand-dropout-sigma",
        type=float,
        default=0.02,
        help="Per-joint standard deviation (rad) of a synthetic q_hand glitch.",
    )
    parser.add_argument(
        "--bus-contact-dropout-probability",
        type=float,
        default=0.0,
        help=(
            "Window-level probability of a bus-level contact loss: all four "
            "fingertips report no contact for the whole window with geometry "
            "frozen at the first history sample (training only)."
        ),
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

    if not 0.0 <= args.dagger_sample_ratio < 1.0:
        raise ValueError("--dagger-sample-ratio must be in [0, 1)")
    if args.dagger_sample_ratio > 0.0 and not args.dagger_file:
        raise ValueError("A positive --dagger-sample-ratio requires --dagger-file")
    resume_checkpoint = (
        torch.load(args.resume, map_location="cpu", weights_only=False)
        if args.resume is not None
        else None
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {args.device!r} was requested, but PyTorch cannot access "
            "CUDA. Use --device cpu explicitly for a CPU smoke test; formal DP "
            "training must not silently fall back to CPU."
        )
    if device.type == "cuda":
        torch.cuda.set_device(device)
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

    episodes = load_episodes(
        args.file, args.stride, args.action_field, args.action_dim
    )
    if args.exclude_episode_ids:
        unknown = sorted(set(args.exclude_episode_ids) - set(episodes))
        if unknown:
            raise ValueError(
                f"--exclude-episode-ids not present in --file: {unknown}"
            )
        episodes = {
            eid: episode
            for eid, episode in episodes.items()
            if eid not in args.exclude_episode_ids
        }
    clean_episode_ids = set(episodes)
    dagger_episode_ids: set[int] = set()
    bounded_dagger_episode_ids: set[int] = set()
    for dagger_path in args.dagger_file:
        if input_frame(dagger_path) != input_frame(args.file):
            raise ValueError(f"{dagger_path}: DAgger input frame differs from clean H5")
        if state_schema(dagger_path) != state_schema(args.file):
            raise ValueError(f"{dagger_path}: DAgger state schema differs from clean H5")
        with h5py.File(dagger_path, "r") as dagger_file:
            bound_normalized_action = bool(
                dagger_file.attrs.get("bound_normalized_action", False)
            )
        dagger_episodes = load_episodes(
            dagger_path, args.stride, args.action_field, args.action_dim
        )
        for episode_id, episode in dagger_episodes.items():
            remapped = int(episode_id)
            while remapped in episodes:
                remapped += 1
            episodes[remapped] = episode
            dagger_episode_ids.add(remapped)
            if bound_normalized_action:
                bounded_dagger_episode_ids.add(remapped)
    dataset_input_frame = input_frame(args.file)
    dataset_state_schema = state_schema(args.file)
    dataset_state_fields = state_fields(args.file)
    robot_state_fields, environment_state_fields = state_field_groups(args.file)
    dataset_planner_config = planner_configuration(args.file)
    dataset_motion_config = motion_configuration(args.file)
    dataset_control_dt = control_dt(args.file)
    with h5py.File(args.file, "r") as dataset_file:
        dataset_normal_polarity = str(
            dataset_file.attrs.get("contact_normal_polarity", "unspecified")
        )
        dataset_action_field = args.action_field or str(
            dataset_file.attrs.get("action_field", "q_hand")
        )
        dataset_action_dim = args.action_dim or int(
            dataset_file.attrs.get("action_dim", ACTION_DIM)
        )
    args.action_dim = dataset_action_dim
    args.action_field = dataset_action_field
    cartesian_action_fields = (
        "tip_delta_tangent_palm",
        "tip_motion_tangent_palm",
        "tip_motion_palm",
        "tip_target_palm",
    )
    if dataset_action_dim == 12 and dataset_action_field not in cartesian_action_fields:
        raise ValueError(
            "12D action_dim requires a supported Cartesian fingertip action; "
            f"got {dataset_action_field!r}"
        )
    if dataset_action_field in cartesian_action_fields and dataset_action_dim != 12:
        raise ValueError(
            f"action_field={dataset_action_field!r} requires action_dim=12; "
            f"got {dataset_action_dim}"
        )
    if dataset_planner_config is not None:
        action_horizon_frames = args.pred_horizon * args.stride
        if dataset_planner_config["horizon_frames"] > action_horizon_frames:
            raise ValueError(
                "Planner command extends beyond the DP action horizon: "
                f"H5={dataset_planner_config['horizon_frames']} raw frames, "
                f"pred_horizon*stride={action_horizon_frames}"
            )
        args.planner_waypoints = int(dataset_planner_config["waypoints"])
        args.planner_step_frames = int(dataset_planner_config["step_frames"])
        args.planner_horizon_frames = int(
            dataset_planner_config["horizon_frames"]
        )
    robot_state_dim, environment_state_dim, state_dim = state_dimensions(args.file)
    args.robot_state_dim = robot_state_dim
    args.environment_state_dim = environment_state_dim
    args.state_dim = state_dim
    args.state_schema = dataset_state_schema
    args.control_dt = dataset_control_dt
    args.action_waypoint_dt = args.stride * dataset_control_dt
    field_slices: dict[str, slice] = {}
    field_cursor = 0
    for field_name, field_size in dataset_state_fields:
        field_slices[field_name] = slice(field_cursor, field_cursor + field_size)
        field_cursor += field_size
    state_input_mask = np.ones(state_dim, dtype=np.float32)
    if args.input_profile == "B0":
        b0_fields = {
            "fingertip_contact_pos_palm",
            "fingertip_contact_normal_palm",
            "fingertip_contact_mask",
            "fingertip_contact_point_velocity_palm",
            "fingertip_contact_normal_angular_rate_palm",
            "q_hand",
            "q_live_velocity",
        }
        missing_b0 = b0_fields - set(field_slices)
        if missing_b0:
            raise ValueError(
                "B0 requires the dual-track-v3 execution fields; missing "
                f"{sorted(missing_b0)}"
            )
        state_input_mask.fill(0.0)
        for field_name in b0_fields:
            state_input_mask[field_slices[field_name]] = 1.0
        if int(state_input_mask.sum()) != 84:
            raise AssertionError(
                f"B0 contract must expose exactly 84 dimensions, got "
                f"{int(state_input_mask.sum())}"
            )
    active_input_fields = [
        name
        for name, field_slice in field_slices.items()
        if bool(np.all(state_input_mask[field_slice] > 0.5))
    ]
    args.active_input_fields = active_input_fields
    if args.action_representation == "kinematic_residual_q":
        if dataset_motion_config is None:
            raise ValueError(
                "kinematic_residual_q requires an explicit motion-schema H5"
            )
        if int(dataset_motion_config["step_frames"]) != args.stride:
            raise ValueError(
                "Motion feature lag must equal training stride for identical "
                f"train/live velocity semantics: H5={dataset_motion_config['step_frames']} "
                f"stride={args.stride}"
            )
        args.motion_feature_step_frames = int(dataset_motion_config["step_frames"])
        args.motion_feature_dt = float(dataset_motion_config["feature_dt"])
    if resume_checkpoint is not None:
        compatibility = {
            "state_dim": (int(resume_checkpoint["state_dim"]), state_dim),
            "robot_state_dim": (
                int(resume_checkpoint["robot_state_dim"]), robot_state_dim
            ),
            "environment_state_dim": (
                int(resume_checkpoint["environment_state_dim"]),
                environment_state_dim,
            ),
            "action_representation": (
                str(resume_checkpoint["action_representation"]),
                args.action_representation,
            ),
            "state_schema": (
                str(resume_checkpoint["state_schema"]), dataset_state_schema
            ),
        }
        mismatched = {
            name: values for name, values in compatibility.items() if values[0] != values[1]
        }
        if mismatched:
            raise ValueError(f"Resume checkpoint is incompatible: {mismatched}")
    episode_domain = episode_domains(args.file)
    with h5py.File(args.file, "r") as split_file:
        train_only_domains = set(
            json.loads(str(split_file.attrs.get("train_only_domains_json", "[]")))
        )
    if resume_checkpoint is not None and args.preserve_checkpoint_split:
        checkpoint_train = {
            int(value) for value in resume_checkpoint["train_episode_ids"]
        }
        checkpoint_val = {
            int(value) for value in resume_checkpoint["val_episode_ids"]
        }
        unknown = clean_episode_ids - checkpoint_train - checkpoint_val
        if unknown:
            raise ValueError(
                "The clean H5 contains episodes absent from the resume checkpoint "
                f"split: {sorted(unknown)[:10]}"
            )
        train_ids = sorted((clean_episode_ids & checkpoint_train) | dagger_episode_ids)
        val_ids = sorted(clean_episode_ids & checkpoint_val)
        split_strategy = "resume_checkpoint_clean_split_plus_dagger_train_only"
    elif episode_domain:
        # Split every object independently.  A global shuffle can move the
        # standalone Mustard validation trajectories into the combined
        # training set, making a single-vs-multi-object A/B comparison leak.
        train_ids, val_ids = [], []
        for domain in sorted(set(episode_domain.values())):
            domain_ids = sorted(
                eid for eid in episodes if episode_domain.get(eid) == domain
            )
            if domain in train_only_domains:
                train_ids.extend(domain_ids)
                continue
            domain_train, domain_val = split_episode_ids(
                domain_ids, args.val_ratio, args.seed
            )
            train_ids.extend(domain_train)
            val_ids.extend(domain_val)
        train_ids.sort()
        val_ids.sort()
        split_strategy = (
            "stratified_by_object_domain_with_train_only="
            + ",".join(sorted(train_only_domains))
        )
    else:
        train_ids, val_ids = split_episode_ids(
            list(episodes), args.val_ratio, args.seed
        )
        split_strategy = "random_episode"
    if resume_checkpoint is not None and args.reuse_checkpoint_normalization:
        checkpoint_normalization = resume_checkpoint["normalization"]
        normalization = Normalization(
            state_mean=np.asarray(
                checkpoint_normalization["state_mean"], dtype=np.float32
            ),
            state_std=np.asarray(
                checkpoint_normalization["state_std"], dtype=np.float32
            ),
            action_mean=np.asarray(
                checkpoint_normalization["action_mean"], dtype=np.float32
            ),
            action_std=np.asarray(
                checkpoint_normalization["action_std"], dtype=np.float32
            ),
        )
    else:
        normalization = compute_normalization(
            episodes,
            train_ids,
            args.action_representation,
            dataset_state_schema,
            args.obs_horizon,
            args.pred_horizon,
            args.action_waypoint_dt,
            args.kinematic_velocity_clip_rad_s,
            action_dim=dataset_action_dim,
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
        args.action_waypoint_dt,
        args.kinematic_velocity_clip_rad_s,
        bounded_dagger_episode_ids,
        action_dim=dataset_action_dim,
        state_input_mask=state_input_mask,
        qhand_dropout_probability=args.qhand_dropout_probability,
        qhand_dropout_max_steps=args.qhand_dropout_max_steps,
        qhand_dropout_sigma=args.qhand_dropout_sigma,
        bus_contact_dropout_probability=args.bus_contact_dropout_probability,
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
        args.action_waypoint_dt,
        args.kinematic_velocity_clip_rad_s,
        None,
        action_dim=dataset_action_dim,
        state_input_mask=state_input_mask,
    )
    validation_sets_by_domain = {
        domain: FingertipDiffusionDataset(
            episodes,
            [eid for eid in val_ids if episode_domain.get(eid) == domain],
            normalization,
            args.obs_horizon,
            args.pred_horizon,
            robot_state_dim,
            dataset_state_schema,
            0.0,
            args.max_contact_dropout_steps,
            args.action_representation,
            args.action_waypoint_dt,
            args.kinematic_velocity_clip_rad_s,
            None,
            action_dim=dataset_action_dim,
            state_input_mask=state_input_mask,
        )
        for domain in sorted(set(episode_domain.values()))
    }
    if not train_set:
        raise ValueError("No training windows for the selected horizons")

    dagger_window = np.asarray(
        [episode_id in dagger_episode_ids for episode_id, _ in train_set.windows],
        dtype=bool,
    )
    dagger_action_stats: dict[str, float | int] | None = None
    if np.any(dagger_window):
        absolute_actions = np.concatenate(
            [
                np.abs(train_set[index]["action"].numpy()).reshape(-1)
                for index in np.flatnonzero(dagger_window)
            ]
        )
        outside_fraction = float(np.mean(absolute_actions > 1.0))
        dagger_action_stats = {
            "scalars": int(len(absolute_actions)),
            "outside_fraction": outside_fraction,
            "abs_p95": float(np.percentile(absolute_actions, 95)),
            "abs_max": float(np.max(absolute_actions)),
        }
        print(
            "[DAGGER] normalized action "
            f"outside={100.0 * outside_fraction:.2f}% "
            f"p95={dagger_action_stats['abs_p95']:.3f} "
            f"max={dagger_action_stats['abs_max']:.3f}"
        )
        if (
            args.max_dagger_action_outside_fraction >= 0.0
            and outside_fraction > args.max_dagger_action_outside_fraction
        ):
            raise ValueError(
                "DAgger recovery labels are outside the deployed action range: "
                f"{100.0 * outside_fraction:.2f}% > "
                f"{100.0 * args.max_dagger_action_outside_fraction:.2f}%. "
                "Collect earlier/smaller nominal deviations instead of clipping "
                "a distant one-shot teacher return."
            )

    run_manifest = {
        "source": str(args.file),
        "output": str(output),
        "requested_device": args.device,
        "resolved_device": str(device),
        "cuda_device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "config": vars(args),
        "dataset": {
            "input_frame": dataset_input_frame,
            "state_schema": dataset_state_schema,
            "state_fields": dataset_state_fields,
            "state_dim": state_dim,
            "input_profile": args.input_profile,
            "active_input_fields": active_input_fields,
            "robot_state_dim": robot_state_dim,
            "environment_state_dim": environment_state_dim,
            "action_dim": dataset_action_dim,
            "action_field": dataset_action_field,
            "action_representation": args.action_representation,
            "episodes": len(episodes),
            "clean_episodes": len(clean_episode_ids),
            "dagger_episodes": len(dagger_episode_ids),
            "bounded_dagger_episodes": len(bounded_dagger_episode_ids),
            "dagger_files": [str(path) for path in args.dagger_file],
            "dagger_sample_ratio": args.dagger_sample_ratio,
            "dagger_action_stats": dagger_action_stats,
            "train_episode_ids": train_ids,
            "val_episode_ids": val_ids,
            "train_windows": len(train_set),
            "val_windows": len(val_set),
            "planner": dataset_planner_config,
            "motion": dataset_motion_config,
            "episode_domains": episode_domain,
            "train_only_domains": sorted(train_only_domains),
            "split_strategy": split_strategy,
            "contact_normal_polarity": dataset_normal_polarity,
        },
    }
    (output / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, default=str), encoding="utf-8"
    )
    print(
        f"[DP] device={device} "
        f"name={run_manifest['cuda_device_name'] or 'CPU'} "
        f"episodes={len(episodes)} train/val={len(train_ids)}/{len(val_ids)} "
        f"windows={len(train_set)}/{len(val_set)} state={state_dim}d"
    )
    train_sampler = None
    if dagger_episode_ids and args.dagger_sample_ratio > 0.0:
        clean_count = int(np.sum(~dagger_window))
        dagger_count = int(np.sum(dagger_window))
        if clean_count == 0 or dagger_count == 0:
            raise ValueError(
                f"Invalid clean/DAgger window counts: {clean_count}/{dagger_count}"
            )
        weights = np.empty(len(train_set), dtype=np.float64)
        weights[~dagger_window] = (
            1.0 - args.dagger_sample_ratio
        ) / clean_count
        weights[dagger_window] = args.dagger_sample_ratio / dagger_count
        train_sampler = WeightedRandomSampler(
            torch.from_numpy(weights),
            num_samples=len(train_set),
            replacement=True,
        )
        print(
            f"[DAGGER] windows clean/dagger={clean_count}/{dagger_count} "
            f"sample_ratio={args.dagger_sample_ratio:.3f}"
        )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
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
    if resume_checkpoint is not None:
        policy.load_state_dict(resume_checkpoint["model"], strict=True)
        print(f"[DP] initialized model from {args.resume}")
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
                "action_dim": dataset_action_dim,
                "action_field": dataset_action_field,
                "action_representation": args.action_representation,
                "input_frame": dataset_input_frame,
                "state_schema": dataset_state_schema,
                "state_fields": dataset_state_fields,
                "input_profile": args.input_profile,
                "state_input_mask": state_input_mask,
                "active_input_fields": active_input_fields,
                "contact_normal_polarity": dataset_normal_polarity,
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
                "train_sample_first_step_mae_rad": train_metrics.get(
                    "sample_first_step_mae_rad", float("nan")
                ),
                "val_sample_first_step_mae_rad": val_metrics.get(
                    "sample_first_step_mae_rad", float("nan")
                ),
                "train_sample_first_4_mae_rad": train_metrics.get(
                    "sample_first_4_mae_rad", float("nan")
                ),
                "val_sample_first_4_mae_rad": val_metrics.get(
                    "sample_first_4_mae_rad", float("nan")
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
                "train_hold_q_first_step_mae_rad": train_metrics.get(
                    "hold_current_q_baseline_first_step_mae_rad", float("nan")
                ),
                "val_hold_q_first_step_mae_rad": val_metrics.get(
                    "hold_current_q_baseline_first_step_mae_rad", float("nan")
                ),
                "train_hold_q_first_4_mae_rad": train_metrics.get(
                    "hold_current_q_baseline_first_4_mae_rad", float("nan")
                ),
                "val_hold_q_first_4_mae_rad": val_metrics.get(
                    "hold_current_q_baseline_first_4_mae_rad", float("nan")
                ),
                "train_kinematic_baseline_mae_rad": train_metrics.get(
                    "kinematic_baseline_mae_rad", float("nan")
                ),
                "val_kinematic_baseline_mae_rad": val_metrics.get(
                    "kinematic_baseline_mae_rad", float("nan")
                ),
                "train_kinematic_baseline_first_4_mae_rad": train_metrics.get(
                    "kinematic_baseline_first_4_mae_rad", float("nan")
                ),
                "val_kinematic_baseline_first_4_mae_rad": val_metrics.get(
                    "kinematic_baseline_first_4_mae_rad", float("nan")
                ),
                "action_unit": train_metrics.get("action_unit", "rad"),
            }
            for domain, domain_set in validation_sets_by_domain.items():
                domain_metrics = overfit_metrics(
                    policy,
                    domain_set,
                    device,
                    normalization.action_mean,
                    normalization.action_std,
                    args.action_representation,
                    args.eval_samples,
                    args.seed + 101 + len(record),
                )
                prefix = f"val_domain_{domain}"
                record[f"{prefix}_sample_mae_rad"] = domain_metrics.get(
                    "sample_mae_rad", float("nan")
                )
                record[f"{prefix}_sample_first_4_mae_rad"] = domain_metrics.get(
                    "sample_first_4_mae_rad", float("nan")
                )
                record[f"{prefix}_hold_q_mae_rad"] = domain_metrics.get(
                    "hold_current_q_baseline_mae_rad", float("nan")
                )
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
        "input_profile": args.input_profile,
        "active_input_fields": active_input_fields,
        "contact_normal_polarity": dataset_normal_polarity,
        "action_representation": args.action_representation,
        "architecture": "LeRobot DiffusionPolicy / conditional 1-D U-Net",
        "input": {
            "observation.state": {
                name: size for name, size in robot_state_fields
            },
            "observation.environment_state": {
                name: size for name, size in environment_state_fields
            },
            "total_dim": state_dim,
        },
        "output": {
            (
                "future_q_hand_delta"
                if args.action_representation == "delta_q"
                else (
                    "future_q_hand_kinematic_residual_reconstructed_absolute"
                    if args.action_representation == "kinematic_residual_q"
                    else dataset_action_field
                )
            ): [args.pred_horizon, dataset_action_dim]
        },
        "train_episodes": len(train_ids),
        "val_episodes": len(val_ids),
        "train_windows": len(train_set),
        "val_windows": len(val_set),
        "planner": dataset_planner_config,
        "motion": dataset_motion_config,
    }
    (output / "dataset_info.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"[SUCCESS] model directory: {output}")


if __name__ == "__main__":
    main()
