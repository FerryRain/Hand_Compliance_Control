"""
Shared data utilities for Diffusion Policy training and evaluation.

Provides:
  - H5 metadata inspection
  - Data loading with environment-based train/val split
  - Contact quality computation (filtering labels)
  - FilteredWindowDataset: sliding windows with noise filtering
  - Normalization stats computation
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

# ── FSR finger groups (matches compliance controller) ──────────────────────
FINGER_FSR_IDS: tuple[tuple[int, ...], ...] = (
    (4, 5, 6),     # index:  proximal(4,5), distal(6)
    (7, 8, 9),     # middle: proximal(7,8), distal(9)
    (10, 11, 12),  # ring:   proximal(10,11), distal(12)
    (13, 14, 15),  # thumb:  proximal(13,14), distal(15)
)


# ── H5 Metadata ────────────────────────────────────────────────────────────

def load_h5_metadata(h5_path: str | Path) -> dict:
    """Inspect H5 file: list fields, shapes, dtypes, and data ranges."""
    info = {}
    with h5py.File(str(h5_path), "r") as f:
        for key in f.keys():
            dset = f[key]
            arr = np.asarray(dset[:100])  # first 100 frames for range # type: ignore
            info[key] = {
                "shape": dset.shape, # type: ignore
                "dtype": str(dset.dtype), # type: ignore
                "min": float(arr.min()),
                "max": float(arr.max()),
                "mean": float(arr.mean()),
                "std": float(arr.std()),
            }
    return info


# ── Data Loading ───────────────────────────────────────────────────────────

@dataclass
class DataDict:
    """Container for loaded data arrays."""
    state: np.ndarray       # [T, N, D_state]  e.g., [T, N, 38]
    action: np.ndarray      # [T, N, D_action] e.g., [T, N, 16]
    # Quality fields for filtering (may be None if unavailable in H5)
    finger_contact: np.ndarray | None     # [T, N, 4]
    contact_stability: np.ndarray | None  # [T, N, 1]
    fsr_delta_norm: np.ndarray | None     # [T, N, 1]
    fsr: np.ndarray | None                # [T, N, 16]  raw FSR for on-the-fly quality


def load_data(
    h5_path: str | Path,
    stride: int = 1,
    clip_fsr_pct: float = 99.9,
    val_envs: Sequence[int] = (14, 15),
) -> tuple[DataDict, DataDict]:
    """Load full dataset from H5, split by environment into train/val.

    Returns (train_data, val_data) as DataDict tuples.
    """
    h5_path = str(h5_path)
    with h5py.File(h5_path, "r") as f:
        n_total = f["q"].shape[0] # type: ignore
        all_envs = np.arange(f["q"].shape[1]) # type: ignore

        q = np.asarray(f["q"][::stride, :, :22], dtype=np.float32) # type: ignore
        fsr = np.asarray(f["fsr"][::stride, :, :16], dtype=np.float32) # type: ignore
        act = np.asarray(f["action"][::stride, :, :16], dtype=np.float32) # type: ignore

        # Optional quality fields
        def _try_load(name: str) -> np.ndarray | None:
            if name in f:
                return np.asarray(f[name][::stride], dtype=np.float32) # type: ignore
            return None

        finger_contact = _try_load("finger_contact")
        contact_stability = _try_load("contact_stability")
        fsr_delta_norm = _try_load("fsr_delta_norm")

    # Clip FSR outliers
    if clip_fsr_pct < 100:
        fsr = np.clip(fsr, 0, float(np.percentile(fsr, clip_fsr_pct)))

    # Concatenate state: [q(22), fsr(16)] = 38 dims
    state = np.concatenate([q, fsr], axis=-1).astype(np.float32)

    # Split by environment
    val_envs_set = set(val_envs)
    train_envs = [e for e in range(state.shape[1]) if e not in val_envs_set]
    val_envs_list = list(val_envs_set)

    def _split(arr: np.ndarray | None, envs: list[int]) -> np.ndarray | None:
        if arr is None:
            return None
        return arr[:, envs, :]

    train = DataDict(
        state=state[:, train_envs, :],
        action=act[:, train_envs, :],
        finger_contact=_split(finger_contact, train_envs),
        contact_stability=_split(contact_stability, train_envs),
        fsr_delta_norm=_split(fsr_delta_norm, train_envs),
        fsr=_split(fsr, train_envs),
    )
    val = DataDict(
        state=state[:, val_envs_list, :],
        action=act[:, val_envs_list, :],
        finger_contact=_split(finger_contact, val_envs_list),
        contact_stability=_split(contact_stability, val_envs_list),
        fsr_delta_norm=_split(fsr_delta_norm, val_envs_list),
        fsr=_split(fsr, val_envs_list),
    )

    s = state.shape
    print(f"Loaded H5: {n_total} total steps, stride={stride} → {s[0]} steps × {s[1]} envs")
    print(f"  state={s[-1]}d  action={act.shape[-1]}d"
          f"  fsr∈[{fsr.min():.1f}, {fsr.max():.1f}]")
    print(f"  Train: {train.state.shape[0]} steps × {len(train_envs)} envs")
    print(f"  Val:   {val.state.shape[0]} steps × {len(val_envs_list)} envs")
    if finger_contact is not None:
        print(f"  Quality fields available: finger_contact, contact_stability, fsr_delta_norm")
    else:
        print(f"  Quality fields NOT in H5 — computing on the fly from FSR")

    return train, val


# ── Contact Quality (on-the-fly computation) ───────────────────────────────

def compute_contact_quality(
    fsr_data: np.ndarray,          # [..., 16]
    prev_fsr_data: np.ndarray | None = None,  # [..., 16]
    contact_threshold: float = 0.20,
) -> dict[str, np.ndarray]:
    """Compute per-finger contact metrics from raw FSR.

    Returns dict with:
      - finger_force [..., 4]: mean FSR per finger
      - finger_contact [..., 4]: binary contact per finger
      - full_contact [..., 1]: all four fingers in contact
      - fsr_delta_norm [..., 1]: L2 norm of FSR change between consecutive frames
    """
    # Per-finger force: mean of proximal + distal FSR sensors
    finger_force = np.stack(
        [fsr_data[..., ids].mean(axis=-1) for ids in FINGER_FSR_IDS],
        axis=-1,
    )  # [..., 4]

    finger_contact = (finger_force >= contact_threshold).astype(np.float32)
    full_contact = np.all(finger_contact > 0.5, axis=-1, keepdims=True).astype(np.float32)

    if prev_fsr_data is not None:
        fsr_delta_norm = np.linalg.norm(
            fsr_data - prev_fsr_data, axis=-1, keepdims=True
        )
    else:
        fsr_delta_norm = np.zeros((*fsr_data.shape[:-1], 1), dtype=np.float32)

    return {
        "finger_force": finger_force,
        "finger_contact": finger_contact,
        "full_contact": full_contact,
        "fsr_delta_norm": fsr_delta_norm,
    }


# ── Stats ──────────────────────────────────────────────────────────────────

def compute_stats(state: np.ndarray, action: np.ndarray) -> dict:
    """Compute min-max normalization stats from training data."""
    return {
        "observation.state": {
            "min": torch.from_numpy(state.min(axis=(0, 1)).astype(np.float32)),
            "max": torch.from_numpy(state.max(axis=(0, 1)).astype(np.float32)),
        },
        "action": {
            "min": torch.from_numpy(action.min(axis=(0, 1)).astype(np.float32)),
            "max": torch.from_numpy(action.max(axis=(0, 1)).astype(np.float32)),
        },
    }


# ── Filtered Window Dataset ────────────────────────────────────────────────

@dataclass
class FilterConfig:
    """Parameters for window-level quality filtering.

    Thresholds can be set directly or auto-computed from data percentiles.
    Set delta_percentile to auto-compute delta_threshold from training data.
    """
    min_fingers_in_contact: int = 3       # at least this many fingers touching
    delta_threshold: float | None = None   # fsr_delta_norm <= this (auto if None)
    delta_percentile: float = 90.0         # auto-compute threshold at this percentile
    filter_ratio: float = 0.7              # fraction of window frames that must pass
    contact_threshold: float = 0.20        # threshold for finger_contact binarization


class FilteredWindowDataset(Dataset):
    """Sliding-window dataset with per-window quality filtering.

    Each sample: {
        "observation.state":           [n_obs_steps, D_state]
        "observation.environment_state": [n_obs_steps, 1]
        "action":                      [horizon, D_action]
        "action_is_pad":               [horizon]
    }

    Windows are filtered out if they contain unstable contact or FSR spikes.
    """

    def __init__(
        self,
        data: DataDict,
        n_obs_steps: int,
        horizon: int,
        filter_cfg: FilterConfig | None = None,
    ):
        self.state = data.state
        self.action = data.action
        self.n_obs = n_obs_steps
        self.H = horizon
        self.T = data.state.shape[0]
        self.N = data.state.shape[1]
        self.filter_cfg = filter_cfg

        # Window-valid time range: need n_obs_steps-1 frames before t,
        # and horizon-1 frames after t (action starts at t-n_obs+1)
        t0 = n_obs_steps - 1
        t1 = self.T - horizon
        if t1 <= t0:
            raise ValueError(
                f"No valid windows: T={self.T} n_obs={n_obs_steps} H={horizon}"
            )

        all_candidates = [(t, e) for t in range(t0, t1) for e in range(self.N)]

        # ── Quality filtering ──
        if filter_cfg is not None and self._has_quality(data):
            self.indices = self._filter_indices(all_candidates, data, filter_cfg)
            n_total = len(all_candidates)
            n_kept = len(self.indices)
            print(
                f"  Filtered windows: {n_kept:,}/{n_total:,} kept "
                f"({100*n_kept/max(1,n_total):.1f}%)"
            )
        else:
            self.indices = all_candidates
            if filter_cfg is not None:
                print("  [WARN] FilterConfig provided but no quality fields in H5 — "
                      "skipping filter")
            else:
                print(f"  Windows (no filter): {len(self.indices):,}")

    @staticmethod
    def _has_quality(data: DataDict) -> bool:
        return (
            data.finger_contact is not None
            and data.fsr_delta_norm is not None
        )

    def _filter_indices(
        self,
        candidates: list[tuple[int, int]],
        data: DataDict,
        cfg: FilterConfig,
    ) -> list[tuple[int, int]]:
        """Pre-compute indices that pass quality filter."""
        assert data.finger_contact is not None
        assert data.fsr_delta_norm is not None
        fc = data.finger_contact    # [T, N, 4]
        fdn = data.fsr_delta_norm   # [T, N, 1]

        # Auto-compute delta threshold from data percentile if not set
        delta_thr = cfg.delta_threshold
        if delta_thr is None:
            delta_thr = float(np.percentile(fdn, cfg.delta_percentile))
            print(f"  Auto delta_threshold = {delta_thr:.1f} (P{cfg.delta_percentile:.0f})")

        # Per-frame pass/fail masks
        frame_ok = np.ones((self.T, self.N), dtype=bool)

        # Condition 1: fsr_delta_norm <= threshold (not spiking)
        frame_ok &= (fdn.squeeze(-1) <= delta_thr)

        # Condition 2: at least min_fingers_in_contact in contact
        n_contact = (fc > 0.5).sum(axis=-1)  # [T, N] number of fingers in contact
        frame_ok &= (n_contact >= cfg.min_fingers_in_contact)

        # Window-level: require >= filter_ratio of frames to pass
        # Window spans [obs_start, obs_start + n_obs + horizon) — all frames
        # that the model sees (obs) or predicts (action)
        window_len = self.n_obs + self.H
        required_ok = max(1, int(cfg.filter_ratio * window_len))

        valid = []
        for t, e in candidates:
            obs_start = t - self.n_obs + 1
            window_end = obs_start + window_len
            window_frames = frame_ok[obs_start:window_end, e]
            n_ok = int(window_frames.sum())

            if n_ok >= required_ok:
                valid.append((t, e))

        n_filtered = len(candidates) - len(valid)
        if n_filtered > 0:
            pct = 100 * n_filtered / len(candidates)
            print(
                f"  Filter: {n_filtered:,}/{len(candidates):,} windows filtered ({pct:.1f}%) "
                f"— kept {len(valid):,}"
            )

        return valid

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        t, e = self.indices[idx]
        obs_start = t - self.n_obs + 1
        obs_slice = slice(obs_start, t + 1)
        act_slice = slice(obs_start, obs_start + self.H)

        return {
            "observation.state": torch.from_numpy(
                self.state[obs_slice, e].astype(np.float32)
            ),
            "observation.environment_state": torch.zeros(
                self.n_obs, 1, dtype=torch.float32
            ),
            "action": torch.from_numpy(
                self.action[act_slice, e].astype(np.float32)
            ),
            "action_is_pad": torch.zeros(self.H, dtype=torch.bool),
        }


# ── Quick test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Inspect H5 and test dataset loading")
    p.add_argument("h5_path", type=str, nargs="?", default=None)
    p.add_argument("--stride", type=int, default=3)
    p.add_argument("--n-obs", type=int, default=16)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--no-filter", action="store_true")
    args = p.parse_args()

    h5_path = args.h5_path or (
        "/home/rimlab/Code/Hand_Compliance_Control/"
        "finger_compliance_control/data/headless/"
        "collect_20260603_170927.h5"
    )

    print("=== H5 Metadata ===")
    meta = load_h5_metadata(h5_path)
    for k, v in meta.items():
        print(f"  {k}: shape={v['shape']} range=[{v['min']:.3f}, {v['max']:.3f}]")

    print(f"\n=== Loading with stride={args.stride} ===")
    train, val = load_data(h5_path, stride=args.stride)

    print(f"\n=== Building datasets (n_obs={args.n_obs}, H={args.horizon}) ===")
    filter_cfg = None if args.no_filter else FilterConfig()
    train_ds = FilteredWindowDataset(train, args.n_obs, args.horizon, filter_cfg)
    val_ds = FilteredWindowDataset(val, args.n_obs, args.horizon, filter_cfg)

    print(f"\nTrain windows: {len(train_ds):,}  Val windows: {len(val_ds):,}")

    # Check a sample
    sample = train_ds[0]
    print(f"\nSample shapes:")
    for k, v in sample.items():
        print(f"  {k}: {tuple(v.shape)}  dtype={v.dtype}")
