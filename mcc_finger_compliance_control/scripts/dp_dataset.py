"""Window dataset for fingertip future-pose diffusion training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


ROBOT_STATE_FIELDS = (
    ("q_hand", 16),
    ("fingertip_force_object", 12),
    ("fingertip_contact_normal_object", 12),
)
ENV_STATE_FIELDS = (("palm_twist_object", 6),)
ROBOT_STATE_DIM = sum(size for _, size in ROBOT_STATE_FIELDS)
ENV_STATE_DIM = sum(size for _, size in ENV_STATE_FIELDS)
STATE_FIELDS = ROBOT_STATE_FIELDS + ENV_STATE_FIELDS
STATE_DIM = ROBOT_STATE_DIM + ENV_STATE_DIM
ACTION_DIM = 16


@dataclass
class Normalization:
    state_mean: np.ndarray
    state_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray


def _flat_feature(file: h5py.File, name: str, expected: int) -> np.ndarray:
    if name not in file:
        raise KeyError(
            f"Required field {name!r} is missing. Run invert_trajectories.py "
            "with the current pipeline first."
        )
    value = np.asarray(file[name], dtype=np.float32)
    value = value.reshape(value.shape[0] * value.shape[1], -1)
    if value.shape[-1] != expected:
        raise ValueError(f"{name}: expected {expected} values, got {value.shape}")
    return value


def load_episodes(path: str | Path, stride: int) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Load per-episode state and q_hand arrays without crossing reset boundaries."""
    with h5py.File(path, "r") as file:
        episode_id = np.asarray(file["episode_id"]).reshape(-1).astype(np.int64)
        q_hand = _flat_feature(file, "q_hand", ACTION_DIM)
        state = np.concatenate(
            [_flat_feature(file, name, size) for name, size in STATE_FIELDS],
            axis=-1,
        )
    episodes: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for eid in np.unique(episode_id):
        mask = episode_id == eid
        episodes[int(eid)] = (state[mask][::stride], q_hand[mask][::stride])
    return episodes


def split_episode_ids(
    episode_ids: list[int], val_ratio: float, seed: int
) -> tuple[list[int], list[int]]:
    ids = np.asarray(sorted(episode_ids), dtype=np.int64)
    if ids.size == 0:
        raise ValueError("The input H5 contains no episodes")
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    if val_ratio <= 0.0 or ids.size == 1:
        return sorted(ids.tolist()), []
    val_count = min(ids.size - 1, max(1, int(round(len(ids) * val_ratio))))
    return sorted(ids[val_count:].tolist()), sorted(ids[:val_count].tolist())


def compute_normalization(
    episodes: dict[int, tuple[np.ndarray, np.ndarray]], train_ids: list[int]
) -> Normalization:
    states = np.concatenate([episodes[eid][0] for eid in train_ids], axis=0).astype(
        np.float64
    )
    q_values = np.concatenate(
        [episodes[eid][1] for eid in train_ids], axis=0
    ).astype(np.float64)
    return Normalization(
        state_mean=states.mean(axis=0).astype(np.float32),
        state_std=np.maximum(states.std(axis=0), 1.0e-5).astype(np.float32),
        # The label is displacement from the current q to every future q,
        # not a one-step motor command. Center it at zero and scale by the
        # demonstrated joint-pose spread.
        action_mean=np.zeros(ACTION_DIM, dtype=np.float32),
        # LeRobot's diffusion scheduler clips the denoised action to [-1, 1].
        # A demonstrated joint range guarantees every q_future-q_current label
        # lies in that interval without clipping valid teacher motion.
        action_std=np.maximum(np.ptp(q_values, axis=0), 1.0e-4).astype(np.float32),
    )


class FingertipDiffusionDataset(Dataset):
    """History state -> future q_hand displacement sequence."""

    def __init__(
        self,
        episodes: dict[int, tuple[np.ndarray, np.ndarray]],
        episode_ids: list[int],
        normalization: Normalization,
        obs_horizon: int,
        pred_horizon: int,
    ):
        self.episodes = episodes
        self.normalization = normalization
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.windows: list[tuple[int, int]] = []
        for eid in episode_ids:
            length = len(episodes[eid][0])
            for current in range(obs_horizon - 1, length - pred_horizon):
                self.windows.append((eid, current))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        eid, current = self.windows[index]
        state, q_hand = self.episodes[eid]
        history = state[current - self.obs_horizon + 1 : current + 1]
        future = q_hand[current + 1 : current + 1 + self.pred_horizon]
        action = future - q_hand[current]
        history = (
            history - self.normalization.state_mean
        ) / self.normalization.state_std
        action = (
            action - self.normalization.action_mean
        ) / self.normalization.action_std
        history_tensor = torch.from_numpy(history.astype(np.float32))
        return {
            "observation.state": history_tensor[:, :ROBOT_STATE_DIM],
            "observation.environment_state": history_tensor[:, ROBOT_STATE_DIM:],
            "action": torch.from_numpy(action.astype(np.float32)),
            "action_is_pad": torch.zeros(self.pred_horizon, dtype=torch.bool),
        }
