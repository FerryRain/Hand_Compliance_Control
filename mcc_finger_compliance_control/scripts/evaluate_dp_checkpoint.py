"""Evaluate a DP checkpoint on an explicit, fixed episode set."""

from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from dp_dataset import (
    ACTION_DIM,
    FingertipDiffusionDataset,
    Normalization,
    load_episodes,
    state_dimensions,
    state_fields,
    state_schema,
)
from train_dp import build_policy, overfit_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--episode-ids", type=int, nargs="+", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--sampling-seeds", type=int, default=5)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {device} is unavailable")
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    config = dict(checkpoint["config"])
    runtime_args = SimpleNamespace(**config)
    policy = build_policy(runtime_args, device)
    policy.load_state_dict(checkpoint["model"])
    policy.eval()

    checkpoint_fields = tuple(tuple(value) for value in checkpoint["state_fields"])
    if tuple(state_fields(args.file)) != checkpoint_fields:
        raise ValueError(
            f"state fields differ: data={state_fields(args.file)}, "
            f"checkpoint={checkpoint_fields}"
        )
    robot_dim, environment_dim, _ = state_dimensions(args.file)
    if robot_dim != int(checkpoint["robot_state_dim"]):
        raise ValueError("robot state dimensions differ")
    if environment_dim != int(checkpoint["environment_state_dim"]):
        raise ValueError("environment state dimensions differ")

    action_dim = int(
        checkpoint.get("action_dim", config.get("action_dim", ACTION_DIM))
    )
    action_field = str(
        checkpoint.get("action_field", config.get("action_field", "q_hand"))
    )
    episodes = load_episodes(args.file, int(config["stride"]), action_field, action_dim)
    missing = sorted(set(args.episode_ids) - set(episodes))
    if missing:
        raise ValueError(f"unknown episode IDs: {missing}")
    value = checkpoint["normalization"]
    normalization = Normalization(
        **{field.name: np.asarray(value[field.name]) for field in fields(Normalization)}
    )
    dataset = FingertipDiffusionDataset(
        episodes,
        args.episode_ids,
        normalization,
        int(config["obs_horizon"]),
        int(config["pred_horizon"]),
        robot_dim,
        state_schema(args.file),
        0.0,
        int(config["max_contact_dropout_steps"]),
        str(checkpoint["action_representation"]),
        float(config.get("action_waypoint_dt", int(config["stride"]) * 0.01)),
        float(config.get("kinematic_velocity_clip_rad_s", 1.0)),
        action_dim=action_dim,
    )
    records = [
        overfit_metrics(
            policy,
            dataset,
            device,
            normalization.action_mean,
            normalization.action_std,
            str(checkpoint["action_representation"]),
            args.samples,
            int(config["seed"]) + 1000 + seed,
        )
        for seed in range(args.sampling_seeds)
    ]
    keys = (
        "sample_mae_rad",
        "sample_first_step_mae_rad",
        "sample_first_4_mae_rad",
        "sample_final_mae_rad",
        "hold_current_q_baseline_mae_rad",
        "hold_current_q_baseline_first_4_mae_rad",
        "kinematic_baseline_mae_rad",
        "kinematic_baseline_first_4_mae_rad",
    )
    print(
        f"[EVAL] model={args.model} episodes={args.episode_ids} "
        f"windows={len(dataset)} draws={args.sampling_seeds}x{args.samples}"
    )
    for key in keys:
        values = np.asarray([record[key] for record in records])
        print(f"  {key}: {values.mean():.8f} +/- {values.std():.8f}")


if __name__ == "__main__":
    main()
