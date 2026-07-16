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
import os
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

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
    episode_id: np.ndarray | None         # [T, N]  reset boundary marker


def _quat_mul_np(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Quaternion multiplication (wxyz) in numpy. q1, q2: [..., 4]."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.stack([w, x, y, z], axis=-1)


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
        # Explicit hierarchical schema: state/action semantics are prepared
        # upstream and must not be rebuilt using the legacy finger-only layout.
        if "state" in f and "action" in f:
            state = np.asarray(f["state"][::stride], dtype=np.float32)
            act = np.asarray(f["action"][::stride], dtype=np.float32)

            def _direct_optional(name: str) -> np.ndarray | None:
                return (
                    np.asarray(f[name][::stride], dtype=np.float32)
                    if name in f else None
                )

            direct_quality = {
                "finger_contact": _direct_optional("finger_contact"),
                "contact_stability": _direct_optional("contact_stability"),
                "fsr_delta_norm": _direct_optional("fsr_delta_norm"),
                "episode_id": _direct_optional("episode_id"),
            }
            return _split_direct_data(state, act, direct_quality, val_envs, stride)

        n_total = f["q"].shape[0] # type: ignore
        all_envs = np.arange(f["q"].shape[1]) # type: ignore

        q = np.asarray(f["q"][::stride, :, :22], dtype=np.float32) # type: ignore
        fsr = np.asarray(f["fsr"][::stride, :, :16], dtype=np.float32) # type: ignore
        # 预测目标: 控制器输出的绝对值 action ∈ [-1,1] (16D)
        # DP学习"当前状态下控制器会输出多少驱动力"
        act = np.asarray(f["action"][::stride, :, :16], dtype=np.float32) # type: ignore

        # Optional quality fields
        def _try_load(name: str) -> np.ndarray | None:
            if name in f:
                return np.asarray(f[name][::stride], dtype=np.float32) # type: ignore
            return None

        finger_contact = _try_load("finger_contact")
        contact_stability = _try_load("contact_stability")
        fsr_delta_norm = _try_load("fsr_delta_norm")
        episode_id = _try_load("episode_id")

        # ── Spatial features ──
        palm_pose = _try_load("palm_pose_world")         # [T, E, 7] or None
        fingertip_force_3d = _try_load("fingertip_force_3d")

    # Clip FSR outliers
    if clip_fsr_pct < 100:
        fsr = np.clip(fsr, 0, float(np.percentile(fsr, clip_fsr_pct)))

    if fingertip_force_3d is None:
        fingertip_force_3d = np.zeros_like(fsr[..., :12])
    # Geometry: hardcoded capsule radius + half_height (2D)
    obj_geom = np.full((q.shape[0], q.shape[1], 2), [0.15, 0.08], dtype=np.float32)

    # ── Build state ──
    state_parts = [q, fsr, fingertip_force_3d]  # 50D base
    if palm_pose is not None:
        state_parts.append(palm_pose)            # 7D: palm in object frame
    if finger_contact is not None:
        state_parts.append(finger_contact)       # 4D: per-finger binary contact
    state_parts.append(obj_geom)                 # 2D: capsule geometry
    state = np.concatenate(state_parts, axis=-1).astype(np.float32)

    # ── Build action: [Δpalm_pos(3), Δpalm_rot(3), finger_action(16)] = 22D ──
    # Δpos = palm_pose[t+1, :3] - palm_pose[t, :3] in object frame
    # Δrot = imaginary part [x,y,z] of delta quaternion q_{t+1} * q_t^{-1}
    #   3 DOF rotation representation — no quaternion redundancy
    if palm_pose is not None:
        T = palm_pose.shape[0]
        dpalm_pos = np.zeros_like(palm_pose[..., :3])
        dpalm_pos[:-1] = palm_pose[1:, :, :3] - palm_pose[:-1, :, :3]
        q_t = palm_pose[:-1, :, 3:7]
        q_t1 = palm_pose[1:, :, 3:7]
        q_t_inv = q_t.copy()
        q_t_inv[..., 1:] *= -1.0
        dq = _quat_mul_np(q_t1, q_t_inv)
        dpalm_rot = np.zeros_like(palm_pose[..., :3])
        dpalm_rot[:-1] = dq[..., 1:4]  # imaginary part ≡ rotation vector for small angles
        act = np.concatenate([dpalm_pos, dpalm_rot, act], axis=-1).astype(np.float32)  # [T, E, 22]
    else:
        act = np.concatenate([np.zeros_like(act[..., :6]), act], axis=-1).astype(np.float32)

    # ── Goal conditioning: future palm pose (6D: pos + quat imaginary part) ──
    # goal = [pos(3), quat_imag(3)] — compact 6D, no quaternion redundancy
    GOAL_HORIZON = 48  # ~1.44s ahead (stride=3, dt=0.03s → 48 obs steps)
    if palm_pose is not None:
        T = palm_pose.shape[0]
        goal = np.zeros((T, palm_pose.shape[1], 6), dtype=np.float32)
        for t in range(T):
            future_t = min(t + GOAL_HORIZON, T - 1)
            goal[t, :, :3] = palm_pose[future_t, :, :3]       # pos
            goal[t, :, 3:6] = palm_pose[future_t, :, 4:7]     # quat imaginary [x,y,z]
        state = np.concatenate([state, goal], axis=-1).astype(np.float32)  # +6D
    else:
        state = np.concatenate([state, np.zeros_like(state[..., :6])], axis=-1).astype(np.float32)

    # Split by environment
    n_envs = state.shape[1]
    val_envs_set = set(val_envs)
    out_of_bounds = [e for e in val_envs_set if e >= n_envs]
    if out_of_bounds:
        raise IndexError(
            f"val_envs {sorted(out_of_bounds)} out of bounds: data has {n_envs} "
            f"environments (indices 0-{n_envs-1}). Pass --val-envs with valid indices."
        )
    train_envs = [e for e in range(n_envs) if e not in val_envs_set]
    val_envs_list = list(val_envs_set)

    def _split(arr: np.ndarray | None, envs: list[int]) -> np.ndarray | None:
        if arr is None:
            return None
        return arr[:, envs, :] if arr.ndim == 3 else arr[:, envs]

    train = DataDict(
        state=state[:, train_envs, :],
        action=act[:, train_envs, :],
        finger_contact=_split(finger_contact, train_envs),
        contact_stability=_split(contact_stability, train_envs),
        fsr_delta_norm=_split(fsr_delta_norm, train_envs),
        fsr=_split(fsr, train_envs),
        episode_id=_split(episode_id, train_envs),
    )
    val = DataDict(
        state=state[:, val_envs_list, :],
        action=act[:, val_envs_list, :],
        finger_contact=_split(finger_contact, val_envs_list),
        contact_stability=_split(contact_stability, val_envs_list),
        fsr_delta_norm=_split(fsr_delta_norm, val_envs_list),
        fsr=_split(fsr, val_envs_list),
        episode_id=_split(episode_id, val_envs_list),
    )

    s = state.shape
    print(f"Loaded H5: {n_total} total steps, stride={stride} → {s[0]} steps × {s[1]} envs")
    print(f"  state={s[-1]}d  action={act.shape[-1]}d (Δpalm(3) + finger(16))"
          f"  fsr∈[{fsr.min():.1f}, {fsr.max():.1f}]")
    print(f"  Train: {train.state.shape[0]} steps × {len(train_envs)} envs")
    print(f"  Val:   {val.state.shape[0]} steps × {len(val_envs_list)} envs")
    if palm_pose is not None:
        print(f"  Spatial features: palm_in_obj(7) + geom(2) + contact(4) = +13D")
    if finger_contact is not None:
        print(f"  Quality fields available: finger_contact, contact_stability, fsr_delta_norm")
    else:
        print(f"  Quality fields NOT in H5 — computing on the fly from FSR")

    return train, val


def _split_direct_data(
    state: np.ndarray,
    action: np.ndarray,
    quality: dict[str, np.ndarray | None],
    val_envs: Sequence[int],
    stride: int,
) -> tuple[DataDict, DataDict]:
    """Split an explicit state/action dataset without changing its semantics."""
    n_envs = state.shape[1]
    val_set = set(int(e) for e in val_envs)
    invalid = [e for e in val_set if e < 0 or e >= n_envs]
    if invalid:
        raise IndexError(
            f"val_envs {sorted(invalid)} out of bounds for {n_envs} episodes"
        )
    train_envs = [e for e in range(n_envs) if e not in val_set]
    val_list = sorted(val_set)
    if not train_envs or not val_list:
        raise ValueError(
            f"Need non-empty train and val episode sets; got train={train_envs}, val={val_list}"
        )

    def take(arr: np.ndarray | None, envs: list[int]) -> np.ndarray | None:
        if arr is None:
            return None
        return arr[:, envs, :] if arr.ndim == 3 else arr[:, envs]

    def make(envs: list[int]) -> DataDict:
        return DataDict(
            state=state[:, envs],
            action=action[:, envs],
            finger_contact=take(quality["finger_contact"], envs),
            contact_stability=take(quality["contact_stability"], envs),
            fsr_delta_norm=take(quality["fsr_delta_norm"], envs),
            fsr=None,
            episode_id=take(quality["episode_id"], envs),
        )

    schema_steps = state.shape[0]
    print(
        f"Loaded explicit state/action H5: stride={stride} -> "
        f"{schema_steps} steps x {n_envs} episodes"
    )
    print(f"  state={state.shape[-1]}d action={action.shape[-1]}d")
    print(f"  Train episodes={train_envs} Val episodes={val_list}")
    return make(train_envs), make(val_list)


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
    # 每指接触判定：用该指所有FSR传感器的均值 (近端+远端)
    # index=(4,5,6), middle=(7,8,9), ring=(10,11,12), thumb=(13,14,15)
    finger_force = np.stack([
        fsr_data[..., [4, 5, 6]].mean(axis=-1),    # index
        fsr_data[..., [7, 8, 9]].mean(axis=-1),    # middle
        fsr_data[..., [10, 11, 12]].mean(axis=-1),  # ring
        fsr_data[..., [13, 14, 15]].mean(axis=-1),  # thumb
    ], axis=-1)  # [..., 4]

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
    # ── 静止片段剔除 ──
    min_action_norm: float = 0.01          # window 内 action L2-norm (16D) 均值下限, 低于此值视为静止
    min_action_std: float = 0.005          # window 内 action L2-norm 的标准差下限, 低于此值视为无变化


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
        cache_path: str | None = None,
    ):
        self.state = data.state
        self.action = data.action
        self.n_obs = n_obs_steps
        self.H = horizon
        self.T = data.state.shape[0]
        self.N = data.state.shape[1]
        self.filter_cfg = filter_cfg

        # Standard LeRobot alignment: action index (n_obs-1) = current timestep t.
        # LeRobot's generate_actions hardcodes slicing action[n_obs-1 : n_obs-1+n_action_steps].
        # Must have enough future action frames: horizon >= n_obs_steps + n_action_steps - 1.
        # (n_action_steps is not available here, so we check the minimum safe bound.)
        if horizon <= n_obs_steps:
            raise ValueError(
                f"horizon ({horizon}) must be > n_obs_steps ({n_obs_steps}) — "
                f"need future action frames beyond the observation window. "
                f"Recommended: horizon >= n_obs_steps + n_action_steps - 1"
            )

        # Window-valid time range:
        # - Need n_obs_steps-1 frames before t (obs history)
        # - Need horizon frames starting from obs_start (action: [obs_start, obs_start+H))
        #   obs_start = t - n_obs + 1, so act_end = t - n_obs + 1 + H <= T
        #   → t <= T + n_obs - 1 - H
        t0 = n_obs_steps - 1
        t1 = self.T - horizon + n_obs_steps
        if t1 <= t0:
            raise ValueError(
                f"No valid windows: T={self.T} n_obs={n_obs_steps} H={horizon}"
            )

        # ── Cache: load pre-computed indices if available ──
        if cache_path and os.path.exists(cache_path):
            cached = np.load(cache_path, allow_pickle=True)
            self.indices = [(int(t), int(e)) for t, e in cached["indices"]]
            print(f"  Loaded {len(self.indices):,} cached windows from {cache_path}")
            return

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

        # ── Save cache ──
        if cache_path and len(self.indices) > 0:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.savez(cache_path, indices=np.array(self.indices, dtype=np.int32))
            print(f"  Cached window indices → {cache_path}")

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

        # Condition 3: window-level action magnitude (剔除静止片段)
        # Compute per-frame action L2 norm [T, N] for efficient window-level lookup
        action_norm_frame = np.linalg.norm(data.action, axis=-1)  # [T, N]

        # Window-level: require >= filter_ratio of frames to pass
        # Standard alignment: obs=[t-n_obs+1, t], action=[t-n_obs+1, t-n_obs+1+H)
        # Combined unique frames = max(n_obs, H)  (action covers obs when H >= n_obs)
        window_len = max(self.n_obs, self.H)
        required_ok = max(1, int(cfg.filter_ratio * window_len))

        n_filtered_contact = 0
        n_filtered_action = 0
        n_filtered_episode = 0
        valid = []
        _iter = tqdm(candidates, desc="Filtering windows", unit="w", dynamic_ncols=True) if tqdm else candidates
        for t, e in _iter:
            obs_start = t - self.n_obs + 1
            window_end = obs_start + window_len
            window_frames = frame_ok[obs_start:window_end, e]
            n_ok = int(window_frames.sum())

            if n_ok < required_ok:
                n_filtered_contact += 1
                continue

            # Episode boundary check: window must not cross a reset boundary
            if data.episode_id is not None:
                ep_slice = data.episode_id[obs_start:window_end, e]
                if ep_slice.min() != ep_slice.max():  # episode changed within window
                    n_filtered_episode += 1
                    continue

            # Action magnitude check: 剔除窗口内几乎不动的片段
            if cfg.min_action_norm > 0 or cfg.min_action_std > 0:
                act_slice = action_norm_frame[obs_start:window_end, e]  # [window_len]
                if act_slice.mean() < cfg.min_action_norm or act_slice.std() < cfg.min_action_std:
                    n_filtered_action += 1
                    continue

            valid.append((t, e))

        n_filtered = len(candidates) - len(valid)
        if n_filtered > 0:
            pct = 100 * n_filtered / len(candidates)
            print(
                f"  Filter: {n_filtered:,}/{len(candidates):,} windows filtered ({pct:.1f}%) "
                f"— kept {len(valid):,}"
            )
            if n_filtered_contact > 0:
                print(f"    contact filter removed: {n_filtered_contact:,}")
            if n_filtered_episode > 0:
                print(f"    episode boundary removed: {n_filtered_episode:,}")
            if n_filtered_action > 0:
                print(f"    action (static) removed: {n_filtered_action:,}")

        return valid

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        t, e = self.indices[idx]
        obs_start = t - self.n_obs + 1
        obs_slice = slice(obs_start, t + 1)
        # Standard LeRobot alignment: action shares the same starting point as obs.
        # Action indices [0, n_obs-1) overlap with observation history.
        # Action index (n_obs-1) = current timestep t.
        # Need: horizon >= n_obs_steps + n_action_steps - 1
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
        "./finger_compliance_control/data/train_dp/"
        "collect_20260609_215735.h5"
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
