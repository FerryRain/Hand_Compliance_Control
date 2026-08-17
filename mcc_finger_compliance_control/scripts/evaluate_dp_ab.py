"""Evaluate DP checkpoints on identical held-out windows and diffusion seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from dp_dataset import FingertipDiffusionDataset, load_episodes
from train_dp import build_policy


@torch.no_grad()
def evaluate(checkpoint_path: Path, device: torch.device, samples: int, seeds: list[int], batch_size: int):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = SimpleNamespace(**dict(checkpoint["config"]))
    policy = build_policy(config, device)
    policy.load_state_dict(checkpoint["model"])
    policy.eval()
    episodes = load_episodes(config.file, config.stride)
    normalization = checkpoint["normalization"]
    from dp_dataset import Normalization
    norm = Normalization(**{key: np.asarray(value, dtype=np.float32) for key, value in normalization.items()})
    dataset = FingertipDiffusionDataset(
        episodes,
        list(checkpoint["val_episode_ids"]),
        norm,
        config.obs_horizon,
        config.pred_horizon,
        int(checkpoint["robot_state_dim"]),
        str(checkpoint["state_schema"]),
        0.0,
        int(config.max_contact_dropout_steps),
        str(checkpoint["action_representation"]),
    )
    indices = np.linspace(0, len(dataset) - 1, min(samples, len(dataset)), dtype=np.int64)
    loader = DataLoader(Subset(dataset, indices.tolist()), batch_size=batch_size, shuffle=False)
    seed_records = []
    for seed in seeds:
        generator = torch.Generator(device=device).manual_seed(seed)
        errors = []
        final_errors = []
        for batch in loader:
            state = batch["observation.state"].to(device)
            environment = batch["observation.environment_state"].to(device)
            target = batch["action"].to(device)
            condition = policy.diffusion._prepare_global_conditioning({
                "observation.state": state,
                "observation.environment_state": environment,
            })
            prediction = policy.diffusion.conditional_sample(
                len(state), global_cond=condition, generator=generator
            )
            scale = torch.as_tensor(norm.action_std, device=device).view(1, 1, -1)
            physical_error = (prediction - target) * scale
            errors.append(physical_error.abs().cpu().numpy().reshape(-1))
            final_errors.append(physical_error[:, -1].abs().cpu().numpy().reshape(-1))
        error = np.concatenate(errors)
        final = np.concatenate(final_errors)
        seed_records.append({
            "seed": seed,
            "mae_rad": float(error.mean()),
            "p95_rad": float(np.percentile(error, 95)),
            "final_mae_rad": float(final.mean()),
            "final_p95_rad": float(np.percentile(final, 95)),
        })
    aggregate = {
        key: float(np.mean([record[key] for record in seed_records]))
        for key in ("mae_rad", "p95_rad", "final_mae_rad", "final_p95_rad")
    }
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(checkpoint["step"]),
        "state_schema": str(checkpoint["state_schema"]),
        "state_dim": int(checkpoint["state_dim"]),
        "val_episode_ids": list(checkpoint["val_episode_ids"]),
        "sample_count": len(indices),
        "seeds": seed_records,
        "mean": aggregate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, nargs="+", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--seeds", type=int, nargs="+", default=(101, 202, 303))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = [
        evaluate(model, torch.device(args.device), args.samples, args.seeds, args.batch_size)
        for model in args.models
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
