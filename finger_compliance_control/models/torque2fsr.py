"""
Torque2FSR: 从关节力矩+状态非线性映射到 FSR 接触力估计。

用于 sim2real 桥接:
  仿真: 训练 MLP(q_finger, τ_finger, action_finger, dτ_finger) → FSR_finger
  实机: 推理 MLP(编码器, 电机电流, 指令) → 虚拟FSR → 柔顺控制器 (接口不变)

Architecture (v2 - per-finger):
  每根手指独立的小 MLP，输入仅限该手指的关节数据:
    Index:  [q[0,2,3], τ[0,2,3], action[0,2,3], dτ[0,2,3]] → FSR[4,5,6]   (12→3)
    Middle: [q[4,6,7], τ[4,6,7], action[4,6,7], dτ[4,6,7]] → FSR[7,8,9]   (12→3)
    Ring:   [q[8,10,11], τ[8,10,11], action[8,10,11], dτ[8,10,11]] → FSR[10,11,12] (12→3)
    Thumb:  [q[12,14,15], τ[12,14,15], action[12,14,15], dτ[12,14,15]] → FSR[13,14,15] (12→3)
  Palm FSR 不预测 (手指柔顺控制器不使用手掌 FSR)。

用法:
  python -m finger_compliance_control.models.torque2fsr --train
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ── config ──────────────────────────────────────────────────────────
DEFAULT_H5_PATH = Path(
    "/home/rimlab/Code/Hand_Compliance_Control/"
    "finger_compliance_control/data/headless/"
    "headless_train_20260414_175152.h5"
)
DEFAULT_MODEL_DIR = Path(
    "/home/rimlab/Code/Hand_Compliance_Control/"
    "finger_compliance_control/models/"
)

# 逐指配置: {name: (qfrc_indices, fsr_indices)}
# qfrc indices 对应 hand joints 6-21 (16 dims):
#   index:  [0,2,3]   = global[6,8,9]   = MCP,PIP,DIP (flexion)
#   middle: [4,6,7]   = global[10,12,13] = MCP,PIP,DIP
#   ring:   [8,10,11] = global[14,16,17] = MCP,PIP,DIP
#   thumb:  [12,14,15]= global[18,20,21] = CMC,IP,DIP
FINGER_SPECS = {
    "index":  {"qfrc": [0, 2, 3],    "fsr": [4, 5, 6]},
    "middle": {"qfrc": [4, 6, 7],    "fsr": [7, 8, 9]},
    "ring":   {"qfrc": [8, 10, 11],  "fsr": [10, 11, 12]},
    "thumb":  {"qfrc": [12, 14, 15], "fsr": [13, 14, 15]},
}

FSR_SHORT_NAMES = [
    "p0","p1","p2","p3",
    "i_p0","i_p1","i_d",
    "m_p0","m_p1","m_d",
    "r_p0","r_p1","r_d",
    "t_p0","t_p1","t_d",
]

HIDDEN_DIMS = [64, 32]
DROPOUT = 0.05
FEATURES_PER_JOINT = 4  # q, τ, action, dτ


# ── per-finger model ─────────────────────────────────────────────────
class FingerTorque2FSR(nn.Module):
    """Single-finger MLP: (q, τ, action, dτ) → FSR."""

    def __init__(
        self,
        n_joints: int = 3,
        n_fsr: int = 3,
        hidden_dims: list[int] | None = None,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = HIDDEN_DIMS
        in_dim = n_joints * FEATURES_PER_JOINT
        layers = []
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, n_fsr))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiFingerTorque2FSR(nn.Module):
    """Container holding 4 independent per-finger MLPs."""

    def __init__(self, hidden_dims=None, dropout=DROPOUT):
        super().__init__()
        self.fingers = nn.ModuleDict({
            name: FingerTorque2FSR(
                n_joints=len(spec["qfrc"]),
                n_fsr=len(spec["fsr"]),
                hidden_dims=hidden_dims,
                dropout=dropout,
            )
            for name, spec in FINGER_SPECS.items()
        })

    def forward(self, x_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {name: self.fingers[name](x) for name, x in x_dict.items()}


# ── data loading ─────────────────────────────────────────────────────
def _load_per_finger_data(
    h5_path: Path,
    max_steps: int = 200_000,
    max_envs: int = 8,
    fsr_threshold: float = 0.05,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load (X, Y) for each finger independently.

    Uses contiguous chunk reads (much faster than fancy-indexing HDF5).
    """
    with h5py.File(h5_path, "r") as f:
        total_steps = f["fsr"].shape[0]
        total_envs = f["fsr"].shape[1]
        n_steps = min(total_steps, max_steps)
        n_envs = min(total_envs, max_envs)

        # Contiguous read: take the first n_steps (fast path)
        stride = max(1, total_steps // n_steps)
        print(f"[DATA] Loading {n_steps} steps (stride={stride}) x {n_envs} envs "
              f"from {h5_path.name} ({total_steps} total)")

        # Read contiguous blocks with striding for better HDF5 performance
        fsr_raw = np.asarray(f["fsr"][:total_steps:stride, :n_envs, :], dtype=np.float32)
        q_raw = np.asarray(f["q"][:total_steps:stride, :n_envs, 6:22], dtype=np.float32)
        tau_raw = np.asarray(f["qfrc_actuator"][:total_steps:stride, :n_envs, :], dtype=np.float32)
        action_raw = np.asarray(f["action"][:total_steps:stride, :n_envs, :], dtype=np.float32)

    T, E, D = fsr_raw.shape

    # Flatten
    fsr = fsr_raw.reshape(-1, D)
    q = q_raw.reshape(-1, D)
    tau = tau_raw.reshape(-1, D)
    action = action_raw.reshape(-1, D)

    # dτ
    dtau_raw = np.diff(tau_raw, axis=0, prepend=tau_raw[:1])
    dtau = dtau_raw.reshape(-1, D)

    results = {}
    for name, spec in FINGER_SPECS.items():
        qfrc_ids = spec["qfrc"]
        fsr_ids = spec["fsr"]

        # Per-finger input features
        X = np.concatenate([
            q[:, qfrc_ids],
            tau[:, qfrc_ids],
            action[:, qfrc_ids],
            dtau[:, qfrc_ids],
        ], axis=1).astype(np.float32)

        Y = fsr[:, fsr_ids].astype(np.float32)

        # Filter: at least one of this finger's FSRs > threshold
        mask = (Y > fsr_threshold).any(axis=1)
        X_f, Y_f = X[mask], Y[mask]

        print(f"  [{name:<7s}] {X_f.shape[0]:7d} samples "
              f"({mask.mean():.1%} kept)  X={X_f.shape[1]}d → Y={Y_f.shape[1]}d")
        results[name] = (X_f, Y_f)

    return results


def _create_per_finger_dataloaders(
    finger_data: dict[str, tuple[np.ndarray, np.ndarray]],
    batch_size: int = 2048,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[dict, dict, dict]:
    """Create train/val dataloaders per finger."""
    train_loaders = {}
    val_loaders = {}
    norm_stats = {}

    for name, (X, Y) in finger_data.items():
        split = int(len(X) * (1 - val_ratio))
        rng = np.random.RandomState(seed)
        train_idx = rng.permutation(split)
        val_idx = np.arange(split, len(X))

        X_train, Y_train = X[train_idx], Y[train_idx]
        X_val, Y_val = X[val_idx], Y[val_idx]

        # Normalize per finger
        x_mean = X_train.mean(axis=0, keepdims=True)
        x_std = X_train.std(axis=0, keepdims=True) + 1e-8
        X_train = (X_train - x_mean) / x_std
        X_val = (X_val - x_mean) / x_std

        y_mean = Y_train.mean(axis=0, keepdims=True)
        y_std = Y_train.std(axis=0, keepdims=True) + 1e-8
        Y_train = (Y_train - y_mean) / y_std
        Y_val = (Y_val - y_mean) / y_std

        train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
        val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val))

        train_loaders[name] = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
        val_loaders[name] = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

        norm_stats[name] = {
            "x_mean": x_mean.astype(np.float32),
            "x_std": x_std.astype(np.float32),
            "y_mean": y_mean.astype(np.float32),
            "y_std": y_std.astype(np.float32),
            "qfrc_ids": FINGER_SPECS[name]["qfrc"],
            "fsr_ids": FINGER_SPECS[name]["fsr"],
        }

    return train_loaders, val_loaders, norm_stats


# ── training ─────────────────────────────────────────────────────────
def train(
    h5_path: Path = DEFAULT_H5_PATH,
    model_dir: Path = DEFAULT_MODEL_DIR,
    epochs: int = 60,
    batch_size: int = 2048,
    lr: float = 1e-3,
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
) -> tuple[MultiFingerTorque2FSR, dict]:
    """Train per-finger Torque2FSR models."""
    device = torch.device(device)
    print(f"[TRAIN] Device: {device}")

    finger_data = _load_per_finger_data(h5_path)
    train_loaders, val_loaders, norm_stats = _create_per_finger_dataloaders(
        finger_data, batch_size=batch_size
    )

    model = MultiFingerTorque2FSR().to(device)
    optimizers = {
        name: torch.optim.AdamW(model.fingers[name].parameters(), lr=lr, weight_decay=1e-4)
        for name in FINGER_SPECS
    }
    schedulers = {
        name: torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        for name, opt in optimizers.items()
    }
    criterion = nn.MSELoss()

    best_val_loss = {name: float("inf") for name in FINGER_SPECS}
    best_states = {}

    for epoch in range(epochs):
        # Train each finger independently
        for name in FINGER_SPECS:
            model.fingers[name].train()
        train_losses = {name: 0.0 for name in FINGER_SPECS}

        # Iterate over all fingers' data jointly
        n_batches = max(len(dl) for dl in train_loaders.values())
        train_iters = {name: iter(dl) for name, dl in train_loaders.items()}

        for _ in range(n_batches):
            for name in FINGER_SPECS:
                try:
                    xb, yb = next(train_iters[name])
                except StopIteration:
                    train_iters[name] = iter(train_loaders[name])
                    xb, yb = next(train_iters[name])
                xb, yb = xb.to(device), yb.to(device)
                optimizers[name].zero_grad()
                pred = model.fingers[name](xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizers[name].step()
                train_losses[name] += loss.item() * xb.size(0)

        # Validation
        for name in FINGER_SPECS:
            model.fingers[name].eval()
        val_losses = {name: 0.0 for name in FINGER_SPECS}
        with torch.no_grad():
            for name in FINGER_SPECS:
                for xb, yb in val_loaders[name]:
                    xb, yb = xb.to(device), yb.to(device)
                    pred = model.fingers[name](xb)
                    loss = criterion(pred, yb)
                    val_losses[name] += loss.item() * xb.size(0)

        for name in FINGER_SPECS:
            train_losses[name] /= len(train_loaders[name].dataset)
            val_losses[name] /= len(val_loaders[name].dataset)
            schedulers[name].step()
            if val_losses[name] < best_val_loss[name]:
                best_val_loss[name] = val_losses[name]
                best_states[name] = {k: v.cpu().clone() for k, v in model.fingers[name].state_dict().items()}

        if epoch % 15 == 0 or epoch == epochs - 1:
            parts = "  ".join(
                f"{n}: tr={train_losses[n]:.3e} val={best_val_loss[n]:.3e}"
                for n in FINGER_SPECS
            )
            print(f"  Epoch {epoch:3d}/{epochs}: {parts}")

    # Load best states
    for name in FINGER_SPECS:
        model.fingers[name].load_state_dict(best_states[name])

    # Per-channel R²
    print("\n[TRAIN] Per-finger R² on validation set:")
    all_r2 = {}
    for name in FINGER_SPECS:
        model.fingers[name].eval()
        yt, yp = [], []
        with torch.no_grad():
            for xb, yb in val_loaders[name]:
                xb = xb.to(device)
                yp.append(model.fingers[name](xb).cpu().numpy())
                yt.append(yb.numpy())
        y_true = np.concatenate(yt, axis=0)
        y_pred = np.concatenate(yp, axis=0)
        y_std = norm_stats[name]["y_std"]
        y_mean = norm_stats[name]["y_mean"]
        y_true_raw = y_true * y_std + y_mean
        y_pred_raw = y_pred * y_std + y_mean

        fsr_ids = FINGER_SPECS[name]["fsr"]
        r2_ch = []
        for ch in range(len(fsr_ids)):
            ss_res = ((y_true_raw[:, ch] - y_pred_raw[:, ch])**2).sum()
            ss_tot = ((y_true_raw[:, ch] - y_true_raw[:, ch].mean())**2).sum()
            r2 = 1 - ss_res / max(ss_tot, 1e-10)
            r2_ch.append(r2)
            bar = "█" * max(0, int(r2 * 40))
            print(f"  FSR[{fsr_ids[ch]:02d}] {FSR_SHORT_NAMES[fsr_ids[ch]]:6s}: R²={r2:.4f}  {bar}")
        all_r2[name] = r2_ch
        print(f"  [{name:<7s}] avg R²={np.mean(r2_ch):.4f}")

    # Save
    model_dir.mkdir(parents=True, exist_ok=True)
    save_dict = {
        "finger_states": best_states,
        "norm_stats": norm_stats,
        "hidden_dims": HIDDEN_DIMS,
        "dropout": DROPOUT,
        "finger_specs": FINGER_SPECS,
        "per_finger_r2": all_r2,
    }
    save_path = model_dir / "torque2fsr.pt"
    torch.save(save_dict, save_path)
    print(f"\n[TRAIN] Model saved to {save_path}")

    return model, norm_stats


# ── inference wrapper ────────────────────────────────────────────────
class Torque2FSRInference:
    """Per-finger Torque2FSR 推理封装。

    用法:
        estimator = Torque2FSRInference("torque2fsr.pt")
        fsr_estimated = estimator(q_hand, tau_hand, prev_action)
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        self.finger_specs = checkpoint["finger_specs"]
        self.norm_stats = checkpoint["norm_stats"]
        hidden_dims = checkpoint.get("hidden_dims", HIDDEN_DIMS)
        dropout = checkpoint.get("dropout", DROPOUT)

        self.models: dict[str, FingerTorque2FSR] = {}
        for name in self.finger_specs:
            n_joints = len(self.finger_specs[name]["qfrc"])
            n_fsr = len(self.finger_specs[name]["fsr"])
            m = FingerTorque2FSR(n_joints, n_fsr, hidden_dims, dropout).to(self.device)
            m.load_state_dict(checkpoint["finger_states"][name])
            m.eval()
            self.models[name] = m

        self._prev_tau: torch.Tensor | None = None

    def __call__(
        self,
        q_hand: torch.Tensor,       # [B, 16] hand joint positions
        tau_hand: torch.Tensor,     # [B, 16] actuator torque
        prev_action: torch.Tensor,  # [B, 16] previous action
    ) -> torch.Tensor:             # → [B, 16] estimated FSR
        """Per-finger inference → full 16-dim FSR vector."""
        B = q_hand.shape[0]
        fsr_out = torch.zeros((B, 16), device=self.device, dtype=torch.float32)

        # dτ
        if self._prev_tau is None or self._prev_tau.shape[0] != B:
            self._prev_tau = tau_hand.clone()
        dtau = tau_hand - self._prev_tau
        self._prev_tau = tau_hand.clone()

        for name, spec in self.finger_specs.items():
            qfrc_ids = spec["qfrc"]
            fsr_ids = spec["fsr"]
            stats = self.norm_stats[name]

            # Build per-finger input
            x = torch.cat([
                q_hand[:, qfrc_ids],
                tau_hand[:, qfrc_ids],
                prev_action[:, qfrc_ids],
                dtau[:, qfrc_ids],
            ], dim=-1)

            # Normalize
            x_mean = torch.as_tensor(stats["x_mean"], device=self.device, dtype=torch.float32)
            x_std = torch.as_tensor(stats["x_std"], device=self.device, dtype=torch.float32)
            y_mean = torch.as_tensor(stats["y_mean"], device=self.device, dtype=torch.float32)
            y_std = torch.as_tensor(stats["y_std"], device=self.device, dtype=torch.float32)
            x = (x - x_mean) / x_std

            with torch.no_grad():
                y = self.models[name](x)

            # Denormalize + clamp
            y = y * y_std + y_mean
            y = torch.clamp(y, min=0.0)

            # Scatter into full FSR vector
            fsr_out[:, fsr_ids] = y

        return fsr_out

    def to(self, device: str | torch.device):
        self.device = torch.device(device) if isinstance(device, str) else device
        for m in self.models.values():
            m.to(self.device)
        if self._prev_tau is not None:
            self._prev_tau = self._prev_tau.to(self.device)
        return self

    @property
    def total_params(self) -> int:
        return sum(sum(p.numel() for p in m.parameters()) for m in self.models.values())


# ── CLI ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Torque2FSR per-finger model training")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--h5-path", type=str, default=str(DEFAULT_H5_PATH))
    parser.add_argument("--model-dir", type=str, default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.train:
        train(
            h5_path=Path(args.h5_path),
            model_dir=Path(args.model_dir),
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
        )
    elif args.eval:
        model_path = Path(args.model_dir) / "torque2fsr.pt"
        if not model_path.exists():
            print(f"[ERROR] Model not found at {model_path}")
            return
        estimator = Torque2FSRInference(str(model_path), device=args.device)
        print(f"[EVAL] Model loaded, total params: {estimator.total_params:,}")
        q = torch.randn(4, 16, device=args.device)
        tau = torch.randn(4, 16, device=args.device) * 0.5 + 0.3
        action = torch.zeros(4, 16, device=args.device)
        fsr_est = estimator(q, tau, action)
        print(f"[EVAL] Output shape: {fsr_est.shape}, range: [{fsr_est.min():.3f}, {fsr_est.max():.3f}]")
        # Show which channels are active
        active = (fsr_est > 0.01).float().mean(dim=0)
        for ch in range(16):
            if active[ch] > 0:
                print(f"  FSR[{ch:02d}] active={active[ch]:.0%} mean={fsr_est[:,ch].mean():.3f}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
