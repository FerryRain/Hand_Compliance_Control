"""Pretrain a small PointNet to encode causal GP surface manifolds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from dp_dataset import split_episode_ids


class SurfaceDataset(Dataset):
    def __init__(self, arrays: dict[str, np.ndarray], indices: np.ndarray, mean: np.ndarray, std: np.ndarray):
        self.arrays = arrays
        self.indices = np.asarray(indices, dtype=np.int64)
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        index = int(self.indices[item])
        points = (self.arrays["gp_points"][index] - self.mean) / self.std
        return {
            "points": torch.from_numpy(points),
            "q": torch.from_numpy(self.arrays["q_hand"][index]),
            "planner": torch.from_numpy(self.arrays["planner_command"][index]),
            "delta": torch.from_numpy(self.arrays["future_contact_delta"][index]),
            "normal": torch.from_numpy(self.arrays["future_contact_normal"][index]),
            "mask": torch.from_numpy(self.arrays["future_contact_mask"][index]),
        }


class SurfacePointNet(nn.Module):
    def __init__(self, point_dim: int = 10, latent_dim: int = 32):
        super().__init__()
        self.point_dim = point_dim
        self.latent_dim = latent_dim
        self.point_mlp = nn.Sequential(
            nn.Linear(point_dim, 64), nn.SiLU(), nn.Linear(64, 128), nn.SiLU()
        )
        self.finger_embedding = nn.Parameter(torch.randn(4, 16) * 0.02)
        self.finger_mlp = nn.Sequential(nn.Linear(144, 64), nn.SiLU())
        self.geometry_head = nn.Sequential(
            nn.Linear(4 * 64, 128), nn.SiLU(), nn.Linear(128, latent_dim)
        )
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim + 16 + 6, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 4 * 7),
        )

    def encode(self, points: torch.Tensor) -> torch.Tensor:
        # [B, finger, query, feature] -> one geometry latent per hand.
        point_features = self.point_mlp(points)
        finger_features = point_features.amax(dim=2)
        identity = self.finger_embedding.unsqueeze(0).expand(len(points), -1, -1)
        finger_features = self.finger_mlp(torch.cat((finger_features, identity), dim=-1))
        return self.geometry_head(finger_features.flatten(start_dim=1))

    def forward(self, points: torch.Tensor, q: torch.Tensor, planner: torch.Tensor):
        latent = self.encode(points)
        prediction = self.predictor(torch.cat((latent, q, planner), dim=-1)).view(-1, 4, 7)
        return latent, prediction[..., :3], prediction[..., 3:6], prediction[..., 6]


def loss_and_metrics(model: SurfacePointNet, batch: dict[str, torch.Tensor], device: torch.device):
    batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
    _, delta, normal_raw, contact_logit = model(batch["points"], batch["q"], batch["planner"])
    mask = batch["mask"]
    denominator = mask.sum().clamp_min(1.0)
    delta_error = delta - batch["delta"]
    # A handful of contact-patch switches create centimetre-scale target
    # outliers.  Smooth-L1 keeps those frames from dominating the geometric
    # representation while remaining quadratic around normal millimetre motion.
    scaled_delta = F.smooth_l1_loss(
        delta / 0.01,
        batch["delta"] / 0.01,
        reduction="none",
        beta=0.1,
    ).sum(dim=-1)
    delta_loss = (scaled_delta * mask).sum() / denominator
    normal = F.normalize(normal_raw, dim=-1, eps=1e-6)
    target_normal = F.normalize(batch["normal"], dim=-1, eps=1e-6)
    cosine = (normal * target_normal).sum(dim=-1).clamp(-1.0, 1.0)
    normal_loss = ((1.0 - cosine) * mask).sum() / denominator
    contact_loss = F.binary_cross_entropy_with_logits(contact_logit, mask)
    loss = delta_loss + normal_loss + 0.1 * contact_loss
    with torch.no_grad():
        delta_mae_mm = ((delta_error.abs().mean(dim=-1) * mask).sum() / denominator) * 1000.0
        normal_angle_deg = (
            (torch.acos(cosine) * mask).sum() / denominator * (180.0 / torch.pi)
        )
        contact_accuracy = ((contact_logit > 0) == (mask > 0.5)).float().mean()
    return loss, {
        "loss": float(loss.detach()),
        "delta_mae_mm": float(delta_mae_mm),
        "normal_angle_deg": float(normal_angle_deg),
        "contact_accuracy": float(contact_accuracy),
    }


@torch.no_grad()
def evaluate(model, loader, device, max_batches=100):
    model.eval()
    records = []
    for index, batch in enumerate(loader):
        if index >= max_batches:
            break
        _, metrics = loss_and_metrics(model, batch, device)
        records.append(metrics)
    model.train()
    return {key: float(np.mean([record[key] for record in records])) for key in records[0]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    args.output.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.file, "r") as file:
        arrays = {name: np.asarray(file[name], dtype=np.float32) for name in (
            "gp_points", "q_hand", "planner_command", "future_contact_delta",
            "future_contact_normal", "future_contact_mask"
        )}
        episode_id = np.asarray(file["episode_id"], dtype=np.int64)
        gp_config = str(file.attrs["gp_config"])
    train_ids, val_ids = split_episode_ids(np.unique(episode_id).tolist(), args.val_ratio, args.seed)
    train_indices = np.flatnonzero(np.isin(episode_id, train_ids))
    val_indices = np.flatnonzero(np.isin(episode_id, val_ids))
    valid_points = arrays["gp_points"][train_indices]
    valid_mask = valid_points[..., 9] > 0.5
    flattened = valid_points[valid_mask]
    point_mean = flattened.mean(axis=0)
    point_std = np.maximum(flattened.std(axis=0), 1e-5)
    # Keep binary validity semantically exact after normalization.
    point_mean[9], point_std[9] = 0.0, 1.0
    val_mask = arrays["future_contact_mask"][val_indices] > 0.5
    val_denominator = max(int(val_mask.sum()), 1)
    zero_delta_mae_mm = float(
        (
            np.abs(arrays["future_contact_delta"][val_indices]).mean(axis=-1)
            * val_mask
        ).sum()
        / val_denominator
        * 1000.0
    )
    current_normal = arrays["gp_points"][val_indices, :, 0, 3:6]
    target_normal = arrays["future_contact_normal"][val_indices]
    cosine = np.sum(current_normal * target_normal, axis=-1) / np.maximum(
        np.linalg.norm(current_normal, axis=-1)
        * np.linalg.norm(target_normal, axis=-1),
        1e-8,
    )
    hold_normal_angle_deg = float(
        (np.arccos(np.clip(cosine, -1.0, 1.0)) * val_mask).sum()
        / val_denominator
        * (180.0 / np.pi)
    )
    baseline = {
        "zero_delta_mae_mm": zero_delta_mae_mm,
        "hold_current_normal_angle_deg": hold_normal_angle_deg,
    }
    (args.output / "baseline.json").write_text(
        json.dumps(baseline, indent=2), encoding="utf-8"
    )
    print(f"[PointNet baseline] {json.dumps(baseline)}")
    train_dataset = SurfaceDataset(arrays, train_indices, point_mean, point_std)
    val_dataset = SurfaceDataset(arrays, val_indices, point_mean, point_std)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, generator=generator,
                              persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True,
                            persistent_workers=args.num_workers > 0)
    model = SurfacePointNet(latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    iterator = iter(train_loader)
    records, best = [], float("inf")
    progress = tqdm(range(1, args.steps + 1), desc="Surface PointNet", unit="step")
    for step in progress:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        loss, train_metrics = loss_and_metrics(model, batch, device)
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            progress.set_postfix(loss=f"{train_metrics['loss']:.4f}")
        if step % args.eval_every == 0 or step == args.steps:
            val_metrics = evaluate(model, val_loader, device)
            record = {"step": step, **{f"train_{k}": v for k, v in train_metrics.items()},
                      **{f"val_{k}": v for k, v in val_metrics.items()}}
            records.append(record)
            print(json.dumps(record))
            checkpoint = {
                "step": step, "model": model.state_dict(), "config": vars(args),
                "point_mean": point_mean, "point_std": point_std,
                "train_episode_ids": train_ids, "val_episode_ids": val_ids,
                "gp_config": gp_config, "metrics": record,
            }
            torch.save(checkpoint, args.output / "last.pt")
            if val_metrics["loss"] < best:
                best = val_metrics["loss"]
                torch.save(checkpoint, args.output / "best.pt")
            (args.output / "metrics.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"[SUCCESS] best validation loss={best:.6f}; output={args.output}")


if __name__ == "__main__":
    main()
