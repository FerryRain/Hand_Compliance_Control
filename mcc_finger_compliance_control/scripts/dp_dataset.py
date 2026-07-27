"""Window dataset for fingertip future-pose diffusion training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


STATE_FIELDS_BY_SCHEMA = {
    "force_normal": {
        "object": (
            ("q_hand", 16),
            ("fingertip_force_object", 12),
            ("fingertip_contact_normal_object", 12),
            ("palm_twist_object", 6),
        ),
        "palm": (
            ("q_hand", 16),
            ("fingertip_force_palm", 12),
            ("fingertip_contact_normal_palm", 12),
            ("palm_relative_twist_palm", 6),
        ),
    },
    "contact_geometry": {
        "object": (
            ("q_hand", 16),
            ("fingertip_contact_pos_object", 12),
            ("fingertip_contact_normal_object", 12),
            ("fingertip_contact_mask", 4),
            ("palm_twist_object", 6),
        ),
        "palm": (
            ("q_hand", 16),
            ("fingertip_contact_pos_palm", 12),
            ("fingertip_contact_normal_palm", 12),
            ("fingertip_contact_mask", 4),
            ("palm_relative_twist_palm", 6),
        ),
    },
}
# Legacy defaults retained for loading existing force-input checkpoints.
ROBOT_STATE_DIM = 40
ENV_STATE_DIM = 6
STATE_DIM = ROBOT_STATE_DIM + ENV_STATE_DIM
ACTION_DIM = 16
ActionRepresentation = Literal["delta_q", "absolute_q"]
ACTION_REPRESENTATIONS: tuple[ActionRepresentation, ...] = (
    "delta_q",
    "absolute_q",
)


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


def input_frame(path: str | Path) -> str:
    """Return the explicitly declared DP coordinate frame."""
    with h5py.File(path, "r") as file:
        frame = str(file.attrs.get("dp_input_frame", "object"))
        if frame not in ("object", "palm"):
            raise ValueError(
                f"Unsupported dp_input_frame={frame!r}; "
                "expected one of ('object', 'palm')"
            )
        return frame


def state_schema(path: str | Path) -> str:
    with h5py.File(path, "r") as file:
        schema = str(file.attrs.get("dp_state_schema", "force_normal"))
    if schema not in STATE_FIELDS_BY_SCHEMA:
        raise ValueError(
            f"Unsupported dp_state_schema={schema!r}; "
            f"expected one of {tuple(STATE_FIELDS_BY_SCHEMA)}"
        )
    return schema


def state_fields(path: str | Path) -> tuple[tuple[str, int], ...]:
    return STATE_FIELDS_BY_SCHEMA[state_schema(path)][input_frame(path)]


def state_dimensions(path: str | Path) -> tuple[int, int, int]:
    fields = state_fields(path)
    environment_dim = fields[-1][1]
    total_dim = sum(size for _, size in fields)
    return total_dim - environment_dim, environment_dim, total_dim


def load_episodes(path: str | Path, stride: int) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Load per-episode state and q_hand arrays without crossing reset boundaries."""
    with h5py.File(path, "r") as file:
        frame = str(file.attrs.get("dp_input_frame", "object"))
        schema = str(file.attrs.get("dp_state_schema", "force_normal"))
        try:
            fields = STATE_FIELDS_BY_SCHEMA[schema][frame]
        except KeyError as error:
            raise ValueError(
                f"Unsupported DP state schema/frame: {schema!r}/{frame!r}"
            ) from error
        episode_id = np.asarray(file["episode_id"]).reshape(-1).astype(np.int64)
        q_hand = _flat_feature(file, "q_hand", ACTION_DIM)
        state = np.concatenate(
            [_flat_feature(file, name, size) for name, size in fields],
            axis=-1,
        )
    episodes: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for eid in np.unique(episode_id):
        mask = episode_id == eid
        episode_state = state[mask][::stride].copy()
        if schema == "contact_geometry":
            # Match deployment: an unavailable contact retains the last
            # reliable point/normal, while the explicit mask remains zero.
            contact_mask = episode_state[:, 40:44] > 0.5
            for finger in range(4):
                valid_indices = np.flatnonzero(contact_mask[:, finger])
                if not len(valid_indices):
                    continue
                last = int(valid_indices[0])
                pos_slice = slice(16 + 3 * finger, 19 + 3 * finger)
                normal_slice = slice(28 + 3 * finger, 31 + 3 * finger)
                episode_state[:last, pos_slice] = episode_state[last, pos_slice]
                episode_state[:last, normal_slice] = episode_state[
                    last, normal_slice
                ]
                for index in range(last + 1, len(episode_state)):
                    if contact_mask[index, finger]:
                        last = index
                    else:
                        episode_state[index, pos_slice] = episode_state[
                            last, pos_slice
                        ]
                        episode_state[index, normal_slice] = episode_state[
                            last, normal_slice
                        ]
        episodes[int(eid)] = (episode_state, q_hand[mask][::stride])
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
    episodes: dict[int, tuple[np.ndarray, np.ndarray]],
    train_ids: list[int],
    action_representation: ActionRepresentation = "delta_q",
    schema: str = "force_normal",
) -> Normalization:
    states = np.concatenate([episodes[eid][0] for eid in train_ids], axis=0).astype(
        np.float64
    )
    q_values = np.concatenate(
        [episodes[eid][1] for eid in train_ids], axis=0
    ).astype(np.float64)
    if action_representation == "delta_q":
        # q_future-q_current is centered at zero. A demonstrated joint range
        # guarantees that every valid displacement remains inside [-1, 1].
        action_mean = np.zeros(ACTION_DIM, dtype=np.float64)
        action_std = np.maximum(np.ptp(q_values, axis=0), 1.0e-4)
    elif action_representation == "absolute_q":
        # LeRobot clips denoised normalized actions to [-1, 1]. Map the
        # demonstrated absolute joint limits exactly into that interval.
        q_min = q_values.min(axis=0)
        q_max = q_values.max(axis=0)
        action_mean = 0.5 * (q_min + q_max)
        action_std = np.maximum(0.5 * (q_max - q_min), 1.0e-4)
    else:
        raise ValueError(
            f"Unsupported action_representation={action_representation!r}; "
            f"expected one of {ACTION_REPRESENTATIONS}"
        )
    state_mean = states.mean(axis=0)
    state_std = np.maximum(states.std(axis=0), 1.0e-5)
    if schema == "contact_geometry":
        # Strict teacher trajectories make these masks almost constant.  A
        # statistical standard deviation would make a live zero-mask an
        # enormous outlier, so encode contact explicitly as {-1, +1}.
        state_mean[40:44] = 0.5
        state_std[40:44] = 0.5
    return Normalization(
        state_mean=state_mean.astype(np.float32),
        state_std=state_std.astype(np.float32),
        action_mean=action_mean.astype(np.float32),
        action_std=action_std.astype(np.float32),
    )


class FingertipDiffusionDataset(Dataset):
    """History state -> future absolute-q or delta-q sequence."""

    def __init__(
        self,
        episodes: dict[int, tuple[np.ndarray, np.ndarray]],
        episode_ids: list[int],
        normalization: Normalization,
        obs_horizon: int,
        pred_horizon: int,
        robot_state_dim: int,
        state_schema: str = "force_normal",
        contact_dropout_probability: float = 0.0,
        max_contact_dropout_steps: int = 3,
        action_representation: ActionRepresentation = "delta_q",
    ):
        if action_representation not in ACTION_REPRESENTATIONS:
            raise ValueError(
                f"Unsupported action_representation={action_representation!r}"
            )
        self.episodes = episodes
        self.normalization = normalization
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.robot_state_dim = robot_state_dim
        self.state_schema = state_schema
        self.contact_dropout_probability = contact_dropout_probability
        self.max_contact_dropout_steps = max_contact_dropout_steps
        self.action_representation = action_representation
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
        if (
            self.state_schema == "contact_geometry"
            and self.contact_dropout_probability > 0.0
        ):
            history = history.copy()
            for finger in range(4):
                if np.random.random() >= self.contact_dropout_probability:
                    continue
                length = np.random.randint(
                    1,
                    min(self.max_contact_dropout_steps, self.obs_horizon - 1)
                    + 1,
                )
                stop = np.random.randint(length + 1, self.obs_horizon + 1)
                start = stop - length
                source = start - 1
                pos_slice = slice(16 + 3 * finger, 19 + 3 * finger)
                normal_slice = slice(28 + 3 * finger, 31 + 3 * finger)
                history[start:stop, pos_slice] = history[source, pos_slice]
                history[start:stop, normal_slice] = history[
                    source, normal_slice
                ]
                history[start:stop, 40 + finger] = 0.0
        future = q_hand[current + 1 : current + 1 + self.pred_horizon]
        if self.action_representation == "delta_q":
            action = future - q_hand[current]
        else:
            action = future
        history = (
            history - self.normalization.state_mean
        ) / self.normalization.state_std
        action = (
            action - self.normalization.action_mean
        ) / self.normalization.action_std
        history_tensor = torch.from_numpy(history.astype(np.float32))
        return {
            "observation.state": history_tensor[:, : self.robot_state_dim],
            "observation.environment_state": history_tensor[
                :, self.robot_state_dim :
            ],
            "action": torch.from_numpy(action.astype(np.float32)),
            "action_is_pad": torch.zeros(self.pred_horizon, dtype=torch.bool),
        }

    def current_q(self, index: int) -> np.ndarray:
        """Return the unnormalized joint pose at a window's observation end."""
        eid, current = self.windows[index]
        return self.episodes[eid][1][current]
