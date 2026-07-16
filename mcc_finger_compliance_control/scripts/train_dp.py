"""Train a task-local conditional DDPM for future fingertip joint poses."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dp_dataset import (
    ACTION_DIM,
    STATE_DIM,
    FingertipDiffusionDataset,
    compute_normalization,
    load_episodes,
    split_episode_ids,
)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        half = self.dimension // 2
        scale = math.log(10_000) / max(half - 1, 1)
        frequency = torch.exp(
            torch.arange(half, device=timestep.device, dtype=torch.float32) * -scale
        )
        angle = timestep.float().unsqueeze(-1) * frequency.unsqueeze(0)
        return torch.cat((angle.sin(), angle.cos()), dim=-1)


class ConditionalPoseDiffusion(nn.Module):
    def __init__(
        self, obs_horizon: int, pred_horizon: int, hidden_dim: int = 1024
    ):
        super().__init__()
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        time_dim = 128
        self.time_embedding = SinusoidalTimeEmbedding(time_dim)
        input_dim = (
            obs_horizon * STATE_DIM + pred_horizon * ACTION_DIM + time_dim
        )
        output_dim = pred_horizon * ACTION_DIM
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self, noisy_action: torch.Tensor, timestep: torch.Tensor, observation: torch.Tensor
    ) -> torch.Tensor:
        batch = observation.shape[0]
        features = torch.cat(
            (
                observation.reshape(batch, -1),
                noisy_action.reshape(batch, -1),
                self.time_embedding(timestep),
            ),
            dim=-1,
        )
        return self.network(features).reshape_as(noisy_action)


def cosine_beta_schedule(steps: int) -> torch.Tensor:
    x = torch.linspace(0, steps, steps + 1)
    alpha_bar = torch.cos(((x / steps) + 0.008) / 1.008 * math.pi / 2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    return torch.clip(1 - alpha_bar[1:] / alpha_bar[:-1], 1.0e-4, 0.999)


@torch.no_grad()
def validation_loss(
    model: nn.Module,
    loader: DataLoader,
    alpha_bar: torch.Tensor,
    device: torch.device,
    max_batches: int = 50,
) -> float:
    model.eval()
    losses = []
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        obs = batch["observation"].to(device)
        action = batch["action"].to(device)
        timestep = torch.randint(0, len(alpha_bar), (len(obs),), device=device)
        noise = torch.randn_like(action)
        weight = alpha_bar[timestep].view(-1, 1, 1)
        noisy = weight.sqrt() * action + (1 - weight).sqrt() * noise
        losses.append(nn.functional.mse_loss(model(noisy, timestep, obs), noise).item())
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


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
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=10_000)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output = args.output or Path(
        f"mcc_finger_compliance_control/data/models/dp_{datetime.now():%Y%m%d_%H%M%S}"
    )
    output.mkdir(parents=True, exist_ok=True)

    episodes = load_episodes(args.file, args.stride)
    train_ids, val_ids = split_episode_ids(
        list(episodes), args.val_ratio, args.seed
    )
    normalization = compute_normalization(episodes, train_ids)
    train_set = FingertipDiffusionDataset(
        episodes,
        train_ids,
        normalization,
        args.obs_horizon,
        args.pred_horizon,
    )
    val_set = FingertipDiffusionDataset(
        episodes,
        val_ids,
        normalization,
        args.obs_horizon,
        args.pred_horizon,
    )
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
    if not train_set:
        raise ValueError("No training windows for the selected horizons")

    model = ConditionalPoseDiffusion(
        args.obs_horizon, args.pred_horizon, args.hidden_dim
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    beta = cosine_beta_schedule(args.diffusion_steps).to(device)
    alpha_bar = torch.cumprod(1 - beta, dim=0)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    iterator = iter(train_loader)
    progress = tqdm(range(1, args.steps + 1), dynamic_ncols=True, desc="DP training")
    best_val = float("inf")

    def save(step: int, val: float, name: str) -> None:
        torch.save(
            {
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "normalization": asdict(normalization),
                "config": vars(args),
                "state_dim": STATE_DIM,
                "action_dim": ACTION_DIM,
                "train_episode_ids": train_ids,
                "val_episode_ids": val_ids,
                "val_noise_loss": val,
            },
            output / name,
        )

    for step in progress:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        observation = batch["observation"].to(device, non_blocking=True)
        action = batch["action"].to(device, non_blocking=True)
        timestep = torch.randint(
            0, args.diffusion_steps, (len(observation),), device=device
        )
        noise = torch.randn_like(action)
        weight = alpha_bar[timestep].view(-1, 1, 1)
        noisy = weight.sqrt() * action + (1 - weight).sqrt() * noise
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            prediction = model(noisy, timestep, observation)
            loss = nn.functional.mse_loss(prediction, noise)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if step % 100 == 0:
            progress.set_postfix(loss=f"{loss.item():.5f}")
        if step % args.save_every == 0 or step == args.steps:
            # A one-episode overfit run intentionally has no validation split.
            # Use the checkpoint's train loss for model selection in that case.
            val = (
                validation_loss(model, val_loader, alpha_bar, device)
                if val_ids
                else float(loss.item())
            )
            save(step, val, f"checkpoint_{step:07d}.pt")
            save(step, val, "latest.pt")
            if val < best_val:
                best_val = val
                save(step, val, "best.pt")
            print(f"[DP] step={step} train={loss.item():.6f} val={val:.6f}")

    metadata = {
        "source": str(args.file),
        "train_episodes": len(train_ids),
        "val_episodes": len(val_ids),
        "train_windows": len(train_set),
        "val_windows": len(val_set),
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
    }
    (output / "dataset_info.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"[SUCCESS] model directory: {output}")


if __name__ == "__main__":
    main()
