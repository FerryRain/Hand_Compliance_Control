"""Window dataset for fingertip future-pose diffusion training."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from dp_motion_features import (
    DUAL_TRACK_SCHEMA,
    DUAL_TRACK_V3_SCHEMA,
    MOTION_SCHEMA,
    MOTION_SCHEMAS,
    Q_VELOCITY_SLICE,
    TASK_EST_V4_SCHEMA,
    kinematic_q_baseline,
)
from palm_planner_features import planner_feature_dim


STATE_FIELDS_BY_SCHEMA = {
    "force_normal": {
        "palm": (
            ("q_hand", 16),
            ("fingertip_force_palm", 12),
            ("fingertip_contact_normal_palm", 12),
            ("palm_relative_twist_palm", 6),
        ),
    },
    "contact_geometry": {
        "palm": (
            ("q_hand", 16),
            ("fingertip_contact_pos_palm", 12),
            ("fingertip_contact_normal_palm", 12),
            ("fingertip_contact_mask", 4),
            ("palm_relative_twist_palm", 6),
        ),
    },
    "contact_geometry_planner": {
        "palm": (
            ("q_hand", 16),
            ("fingertip_contact_pos_palm", 12),
            ("fingertip_contact_normal_palm", 12),
            ("fingertip_contact_mask", 4),
            ("palm_relative_twist_palm", 6),
            ("planner_palm_delta_pose_palm", planner_feature_dim()),
        ),
    },
    MOTION_SCHEMA: {
        "palm": (
            ("q_hand", 16),
            ("fingertip_contact_pos_palm", 12),
            ("fingertip_contact_normal_palm", 12),
            ("fingertip_contact_mask", 4),
            ("q_velocity", 16),
            ("fingertip_contact_point_velocity_palm", 12),
            ("fingertip_contact_normal_angular_rate_palm", 12),
            ("palm_relative_twist_palm", 6),
            ("planner_palm_delta_pose_palm", planner_feature_dim()),
        ),
    },
    DUAL_TRACK_SCHEMA: {
        "palm": (
            # Keep task q first so absolute-q diagnostics and the optional
            # kinematic baseline are explicitly anchored to task intent.
            ("q_prior", 16),
            ("fingertip_contact_pos_palm", 12),
            ("fingertip_contact_normal_palm", 12),
            ("fingertip_contact_mask", 4),
            ("q_prior_velocity", 16),
            ("fingertip_contact_point_velocity_palm", 12),
            ("fingertip_contact_normal_angular_rate_palm", 12),
            # Execution feedback is a separate stream; do not overwrite the
            # autoregressive task prior with q_live.
            ("q_hand", 16),
            ("delta_q_comp", 16),
            ("e_servo", 16),
            ("palm_relative_twist_palm", 6),
            ("planner_palm_delta_pose_palm", planner_feature_dim()),
        ),
    },
    DUAL_TRACK_V3_SCHEMA: {
        "palm": (
            ("q_prior", 16),
            ("fingertip_contact_pos_palm", 12),
            ("fingertip_contact_normal_palm", 12),
            ("fingertip_contact_mask", 4),
            ("q_prior_velocity", 16),
            ("fingertip_contact_point_velocity_palm", 12),
            ("fingertip_contact_normal_angular_rate_palm", 12),
            ("q_hand", 16),
            ("q_live_velocity", 16),
            ("e_qdot", 16),
            ("delta_q_comp", 16),
            ("e_servo", 16),
            ("palm_relative_twist_palm", 6),
            ("planner_palm_delta_pose_palm", planner_feature_dim()),
        ),
    },
    TASK_EST_V4_SCHEMA: {
        "palm": (
            # Execution feedback with the known MCC command-space correction
            # removed.  This is a physical task-state estimate, not the
            # policy's previous nominal prediction.
            ("q_task_est", 16),
            ("fingertip_contact_pos_palm", 12),
            ("fingertip_contact_normal_palm", 12),
            ("fingertip_contact_mask", 4),
            ("q_task_est_velocity", 16),
            ("fingertip_contact_point_velocity_palm", 12),
            ("fingertip_contact_normal_angular_rate_palm", 12),
            ("palm_relative_twist_palm", 6),
            ("planner_palm_delta_pose_palm", planner_feature_dim()),
        ),
    },
    "contact_geometry_planner_manifold": {
        "palm": (
            ("q_hand", 16),
            ("fingertip_contact_pos_palm", 12),
            ("fingertip_contact_normal_palm", 12),
            ("fingertip_contact_mask", 4),
            ("palm_relative_twist_palm", 6),
            ("planner_palm_delta_pose_palm", planner_feature_dim()),
            ("surface_manifold_embedding", 32),
        ),
    },
}
ENVIRONMENT_FIELD_COUNTS = {
    "force_normal": 1,
    "contact_geometry": 1,
    "contact_geometry_planner": 2,
    MOTION_SCHEMA: 2,
    DUAL_TRACK_SCHEMA: 2,
    DUAL_TRACK_V3_SCHEMA: 2,
    TASK_EST_V4_SCHEMA: 2,
    "contact_geometry_planner_manifold": 3,
}
# Legacy defaults retained for loading existing force-input checkpoints.
ROBOT_STATE_DIM = 40
ENV_STATE_DIM = 6
STATE_DIM = ROBOT_STATE_DIM + ENV_STATE_DIM
ACTION_DIM = 16
ActionRepresentation = Literal["delta_q", "absolute_q", "kinematic_residual_q"]
ACTION_REPRESENTATIONS: tuple[ActionRepresentation, ...] = (
    "delta_q",
    "absolute_q",
    "kinematic_residual_q",
)

GEOMETRY_STATE_SCHEMAS = (
    "contact_geometry",
    "contact_geometry_planner",
    "contact_geometry_planner_manifold",
    MOTION_SCHEMA,
    DUAL_TRACK_SCHEMA,
    DUAL_TRACK_V3_SCHEMA,
    TASK_EST_V4_SCHEMA,
)
PLANNER_STATE_SCHEMAS = (
    "contact_geometry_planner",
    "contact_geometry_planner_manifold",
    MOTION_SCHEMA,
    DUAL_TRACK_SCHEMA,
    DUAL_TRACK_V3_SCHEMA,
    TASK_EST_V4_SCHEMA,
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
    """Validate and return the mandatory palm-frame DP contract."""
    with h5py.File(path, "r") as file:
        if "dp_input_frame" not in file.attrs:
            raise ValueError(
                "DP H5 has no explicit dp_input_frame. Re-export it with "
                "export_palm_dp.py; implicit legacy object/world frames are forbidden."
            )
        frame = str(file.attrs["dp_input_frame"])
        if frame != "palm":
            raise ValueError(
                f"DP training requires dp_input_frame='palm', got {frame!r}. "
                "Run invert_trajectories.py followed by export_palm_dp.py."
            )
        if str(file.attrs.get("palm_frame_body", "")) != "palm_lower":
            raise ValueError(
                "DP H5 must declare palm_frame_body='palm_lower' so training "
                "and deployment use the same physical frame."
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
    frame = input_frame(path)
    with h5py.File(path, "r") as file:
        schema = str(file.attrs.get("dp_state_schema", "force_normal"))
        fields = list(STATE_FIELDS_BY_SCHEMA[schema][frame])
        if schema in PLANNER_STATE_SCHEMAS:
            waypoint_count = int(file.attrs["planner_waypoints"])
            planner_index = -2 if schema.endswith("_manifold") else -1
            fields[planner_index] = (
                fields[planner_index][0], planner_feature_dim(waypoint_count)
            )
            if schema.endswith("_manifold"):
                fields[-1] = (
                    fields[-1][0], int(file.attrs["surface_manifold_embedding_dim"])
                )
    return tuple(fields)


def state_dimensions(path: str | Path) -> tuple[int, int, int]:
    fields = state_fields(path)
    environment_count = ENVIRONMENT_FIELD_COUNTS[state_schema(path)]
    environment_dim = sum(size for _, size in fields[-environment_count:])
    total_dim = sum(size for _, size in fields)
    return total_dim - environment_dim, environment_dim, total_dim


def state_field_groups(
    path: str | Path,
) -> tuple[tuple[tuple[str, int], ...], tuple[tuple[str, int], ...]]:
    fields = state_fields(path)
    environment_count = ENVIRONMENT_FIELD_COUNTS[state_schema(path)]
    return fields[:-environment_count], fields[-environment_count:]


def planner_configuration(path: str | Path) -> dict[str, int | float] | None:
    if state_schema(path) not in PLANNER_STATE_SCHEMAS:
        return None
    with h5py.File(path, "r") as file:
        return {
            "waypoints": int(file.attrs["planner_waypoints"]),
            "step_frames": int(file.attrs["planner_step_frames"]),
            "horizon_frames": int(file.attrs["planner_horizon_frames"]),
            "waypoint_dt": float(file.attrs["planner_waypoint_dt"]),
            "horizon_seconds": float(file.attrs["planner_horizon_seconds"]),
        }


def control_dt(path: str | Path) -> float:
    with h5py.File(path, "r") as file:
        value = float(file.attrs.get("control_dt", 0.01))
    if value <= 0.0:
        raise ValueError(f"{path}: control_dt must be positive")
    return value


def motion_configuration(path: str | Path) -> dict[str, int | float] | None:
    if state_schema(path) not in MOTION_SCHEMAS:
        return None
    with h5py.File(path, "r") as file:
        step_frames = int(file.attrs["motion_feature_step_frames"])
        feature_dt = float(file.attrs["motion_feature_dt"])
    if step_frames <= 0 or feature_dt <= 0.0:
        raise ValueError(f"{path}: invalid causal motion feature configuration")
    return {"step_frames": step_frames, "feature_dt": feature_dt}


def action_dimension(path: str | Path) -> int:
    """Return the per-step label dimensionality of the action channel.

    The dual-track file (export_dual_track.py) declares the action channel via
    file.attrs["action_field"] ("q_ref" 16D | "tip_delta_tangent_palm" 12D)
    and its dimensionality via file.attrs["action_dim"] (default 16)."""
    with h5py.File(path, "r") as file:
        return int(file.attrs.get("action_dim", ACTION_DIM))


def load_episodes(
    path: str | Path,
    stride: int,
    action_field: str | None = None,
    action_dim: int | None = None,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Load per-episode observation state and supervised action-q arrays.

    Clean teacher exports use ``q_hand`` for both quantities.  Aggregated
    closed-loop/DAgger exports may additionally provide ``action_q_hand``:
    observation ``q_hand`` then remains the causal live joint state while
    ``action_q_hand`` is the time-aligned successful teacher trajectory.
    Keeping those contracts separate prevents failed MCC commands or live
    tracking errors from becoming imitation labels.

    Dual-track exports (attrs ``action_field`` / ``action_dim``) load the label
    channel declared by the file: "q_ref" (16D joint intent) or
    "tip_delta_tangent_palm" (12D tangential fingertip intent).
    """
    frame = input_frame(path)
    with h5py.File(path, "r") as file:
        schema = str(file.attrs.get("dp_state_schema", "force_normal"))
        try:
            fields = list(STATE_FIELDS_BY_SCHEMA[schema][frame])
            if schema in PLANNER_STATE_SCHEMAS:
                waypoint_count = int(file.attrs["planner_waypoints"])
                planner_index = -2 if schema.endswith("_manifold") else -1
                fields[planner_index] = (
                    fields[planner_index][0],
                    planner_feature_dim(waypoint_count),
                )
                if schema.endswith("_manifold"):
                    fields[-1] = (
                        fields[-1][0],
                        int(file.attrs["surface_manifold_embedding_dim"]),
                    )
        except KeyError as error:
            raise ValueError(
                f"Unsupported DP state schema/frame: {schema!r}/{frame!r}"
            ) from error
        episode_id = np.asarray(file["episode_id"]).reshape(-1).astype(np.int64)
        q_hand = _flat_feature(file, "q_hand", ACTION_DIM)
        if action_field is None:
            action_field = str(file.attrs.get("action_field", "q_hand"))
        if action_dim is None:
            action_dim = int(file.attrs.get("action_dim", ACTION_DIM))
        action_q = _flat_feature(file, action_field, action_dim)
        state = np.concatenate(
            [_flat_feature(file, name, size) for name, size in fields],
            axis=-1,
        )
    episodes: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for eid in np.unique(episode_id):
        mask = episode_id == eid
        episode_state = state[mask][::stride].copy()
        if schema in GEOMETRY_STATE_SCHEMAS:
            offsets: dict[str, slice] = {}
            cursor = 0
            for name, size in fields:
                offsets[name] = slice(cursor, cursor + size)
                cursor += size
            mask_slice = offsets["fingertip_contact_mask"]
            position_base = offsets["fingertip_contact_pos_palm"].start
            normal_base = offsets["fingertip_contact_normal_palm"].start
            # Match deployment: an unavailable contact retains the last
            # reliable point/normal, while the explicit mask remains zero.
            contact_mask = episode_state[:, mask_slice] > 0.5
            for finger in range(4):
                valid_indices = np.flatnonzero(contact_mask[:, finger])
                if not len(valid_indices):
                    continue
                last = int(valid_indices[0])
                pos_slice = slice(
                    position_base + 3 * finger, position_base + 3 * finger + 3
                )
                normal_slice = slice(
                    normal_base + 3 * finger, normal_base + 3 * finger + 3
                )
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
        episodes[int(eid)] = (episode_state, action_q[mask][::stride])
    return episodes


def episode_domains(path: str | Path) -> dict[int, str]:
    """Return optional episode-to-object-domain labels for combined datasets."""
    with h5py.File(path, "r") as file:
        if "episode_domain_id" not in file:
            return {}
        episode_id = np.asarray(file["episode_id"]).reshape(-1).astype(np.int64)
        domain_id = np.asarray(file["episode_domain_id"]).reshape(-1).astype(np.int64)
        names = json.loads(str(file.attrs.get("domain_names_json", "[]")))
    mapping: dict[int, str] = {}
    for eid in np.unique(episode_id):
        values = np.unique(domain_id[episode_id == eid])
        if len(values) != 1:
            raise ValueError(f"episode {eid} spans multiple object domains: {values}")
        value = int(values[0])
        if value < 0 or value >= len(names):
            raise ValueError(f"episode {eid} has unknown domain id {value}")
        mapping[int(eid)] = str(names[value])
    return mapping


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
    obs_horizon: int = 16,
    pred_horizon: int = 32,
    waypoint_dt: float = 0.05,
    kinematic_velocity_clip: float = 1.0,
    action_dim: int = ACTION_DIM,
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
        action_mean = np.zeros(action_dim, dtype=np.float64)
        action_std = np.maximum(np.ptp(q_values, axis=0), 1.0e-4)
    elif action_representation == "absolute_q":
        # LeRobot clips denoised normalized actions to [-1, 1]. Map the
        # demonstrated absolute joint limits exactly into that interval.
        q_min = q_values.min(axis=0)
        q_max = q_values.max(axis=0)
        action_mean = 0.5 * (q_min + q_max)
        action_std = np.maximum(0.5 * (q_max - q_min), 1.0e-4)
    elif action_representation == "kinematic_residual_q":
        if action_dim != ACTION_DIM:
            raise ValueError(
                "kinematic_residual_q is joint-space only (16D); "
                f"action_dim={action_dim} requested"
            )
        if schema not in MOTION_SCHEMAS:
            raise ValueError(
                "kinematic_residual_q requires the explicit motion state schema"
            )
        residual_chunks = []
        for episode_id in train_ids:
            state, q = episodes[episode_id]
            current = np.arange(obs_horizon - 1, len(q) - pred_horizon)
            if not len(current):
                continue
            future_indices = current[:, None] + 1 + np.arange(pred_horizon)[None, :]
            future = q[future_indices]
            velocity = state[current, Q_VELOCITY_SLICE]
            time = waypoint_dt * np.arange(1, pred_horizon + 1, dtype=np.float32)
            observation_q = state[current, :ACTION_DIM]
            baseline = observation_q[:, None, :] + time[None, :, None] * np.clip(
                velocity[:, None, :],
                -kinematic_velocity_clip,
                kinematic_velocity_clip,
            )
            residual_chunks.append((future - baseline).reshape(-1, ACTION_DIM))
        if not residual_chunks:
            raise ValueError("No kinematic residuals are available for normalization")
        residual = np.concatenate(residual_chunks, axis=0).astype(np.float64)
        # Robustly use the available normalized action interval.  This fixes
        # the legacy delta representation, which divided milliradian targets
        # by the full demonstrated joint range and collapsed them near zero.
        action_mean = np.median(residual, axis=0)
        action_std = np.maximum(
            np.quantile(np.abs(residual - action_mean), 0.995, axis=0),
            1.0e-4,
        )
    else:
        raise ValueError(
            f"Unsupported action_representation={action_representation!r}; "
            f"expected one of {ACTION_REPRESENTATIONS}"
        )
    state_mean = states.mean(axis=0)
    state_std = np.maximum(states.std(axis=0), 1.0e-5)
    if schema in GEOMETRY_STATE_SCHEMAS:
        # Strict teacher trajectories make these masks almost constant.  A
        # statistical standard deviation would make a live zero-mask an
        # enormous outlier, so encode contact explicitly as {-1, +1}.
        fields = STATE_FIELDS_BY_SCHEMA[schema]["palm"]
        cursor = 0
        mask_slice = None
        for name, size in fields:
            if name == "fingertip_contact_mask":
                mask_slice = slice(cursor, cursor + size)
                break
            cursor += size
        if mask_slice is None:
            raise ValueError(f"{schema}: geometry schema has no contact mask")
        state_mean[mask_slice] = 0.5
        state_std[mask_slice] = 0.5
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
        waypoint_dt: float = 0.05,
        kinematic_velocity_clip: float = 1.0,
        bounded_action_episode_ids: set[int] | None = None,
        action_dim: int = ACTION_DIM,
        state_input_mask: np.ndarray | None = None,
        qhand_dropout_probability: float = 0.0,
        qhand_dropout_max_steps: int = 6,
        qhand_dropout_sigma: float = 0.02,
        bus_contact_dropout_probability: float = 0.0,
    ):
        if action_representation not in ACTION_REPRESENTATIONS:
            raise ValueError(
                f"Unsupported action_representation={action_representation!r}"
            )
        if action_dim != ACTION_DIM and action_representation != "absolute_q":
            raise ValueError(
                "Non-joint action channels (tip_delta_tangent_palm, 12D) only "
                f"support action_representation='absolute_q'; got "
                f"{action_representation!r} with action_dim={action_dim}"
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
        self.action_dim = action_dim
        self.waypoint_dt = waypoint_dt
        self.kinematic_velocity_clip = kinematic_velocity_clip
        self.bounded_action_episode_ids = set(bounded_action_episode_ids or ())
        self.state_input_mask = (
            None
            if state_input_mask is None
            else np.asarray(state_input_mask, dtype=np.float32).reshape(-1)
        )
        self.qhand_dropout_probability = qhand_dropout_probability
        self.qhand_dropout_max_steps = max(1, qhand_dropout_max_steps)
        self.qhand_dropout_sigma = qhand_dropout_sigma
        self.bus_contact_dropout_probability = bus_contact_dropout_probability
        if (
            self.state_input_mask is not None
            and self.state_input_mask.shape != self.normalization.state_mean.shape
        ):
            raise ValueError(
                "state_input_mask shape differs from state normalization: "
                f"{self.state_input_mask.shape} versus "
                f"{self.normalization.state_mean.shape}"
            )
        fields = STATE_FIELDS_BY_SCHEMA[state_schema]["palm"]
        self.field_slices: dict[str, slice] = {}
        cursor = 0
        for name, size in fields:
            self.field_slices[name] = slice(cursor, cursor + size)
            cursor += size
        self.windows: list[tuple[int, int]] = []
        for eid in episode_ids:
            length = len(episodes[eid][0])
            for current in range(obs_horizon - 1, length - pred_horizon):
                self.windows.append((eid, current))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        eid, current = self.windows[index]
        state, action_q = self.episodes[eid]
        history = state[current - self.obs_horizon + 1 : current + 1]
        if (
            self.state_schema in GEOMETRY_STATE_SCHEMAS
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
                position_base = self.field_slices[
                    "fingertip_contact_pos_palm"
                ].start
                normal_base = self.field_slices[
                    "fingertip_contact_normal_palm"
                ].start
                mask_base = self.field_slices["fingertip_contact_mask"].start
                pos_slice = slice(
                    position_base + 3 * finger, position_base + 3 * finger + 3
                )
                normal_slice = slice(
                    normal_base + 3 * finger, normal_base + 3 * finger + 3
                )
                history[start:stop, pos_slice] = history[source, pos_slice]
                history[start:stop, normal_slice] = history[
                    source, normal_slice
                ]
                history[start:stop, mask_base + finger] = 0.0
                if self.state_schema in MOTION_SCHEMAS:
                    point_base = self.field_slices[
                        "fingertip_contact_point_velocity_palm"
                    ].start
                    normal_rate_base = self.field_slices[
                        "fingertip_contact_normal_angular_rate_palm"
                    ].start
                    point_velocity_slice = slice(
                        point_base + 3 * finger, point_base + 3 * finger + 3
                    )
                    normal_rate_slice = slice(
                        normal_rate_base + 3 * finger,
                        normal_rate_base + 3 * finger + 3,
                    )
                    # The first recovered sample also has no valid causal
                    # endpoint pair because its preceding sample was dropped.
                    motion_stop = min(stop + 1, self.obs_horizon)
                    history[start:motion_stop, point_velocity_slice] = 0.0
                    history[start:motion_stop, normal_rate_slice] = 0.0
        # Moderate synthetic sensor failures (observation only; targets and
        # all non-corrupted channels stay untouched).
        # 1) q_hand encoder burst: one finger's 4-joint block gets additive
        # Gaussian noise over a short contiguous span (a glitch, not a bias).
        if (
            self.qhand_dropout_probability > 0.0
            and "q_hand" in self.field_slices
            and np.random.random() < self.qhand_dropout_probability
        ):
            history = history.copy()
            finger = int(np.random.randint(4))
            length = int(
                np.random.randint(
                    1, min(self.qhand_dropout_max_steps, self.obs_horizon) + 1
                )
            )
            stop = int(np.random.randint(length + 1, self.obs_horizon + 1))
            start = stop - length
            q_hand_base = self.field_slices["q_hand"].start
            block = slice(q_hand_base + 4 * finger, q_hand_base + 4 * finger + 4)
            history[start:stop, block] = (
                history[start:stop, block]
                + self.qhand_dropout_sigma
                * np.random.randn(length, 4).astype(np.float32)
            )
        # 2) Bus-level contact loss: every fingertip reports no contact for the
        # whole window; geometry freezes at the first-history value (sensor
        # dead since before the window) and contact motion reads zero.
        if (
            self.state_schema in GEOMETRY_STATE_SCHEMAS
            and self.bus_contact_dropout_probability > 0.0
            and np.random.random() < self.bus_contact_dropout_probability
        ):
            history = history.copy()
            position_base = self.field_slices[
                "fingertip_contact_pos_palm"
            ].start
            normal_base = self.field_slices[
                "fingertip_contact_normal_palm"
            ].start
            mask_base = self.field_slices["fingertip_contact_mask"].start
            pos_slice = slice(position_base, position_base + 12)
            normal_slice = slice(normal_base, normal_base + 12)
            history[:, pos_slice] = history[0, pos_slice][None, :]
            history[:, normal_slice] = history[0, normal_slice][None, :]
            history[:, mask_base : mask_base + 4] = 0.0
            if self.state_schema in MOTION_SCHEMAS:
                point_base = self.field_slices[
                    "fingertip_contact_point_velocity_palm"
                ].start
                normal_rate_base = self.field_slices[
                    "fingertip_contact_normal_angular_rate_palm"
                ].start
                history[:, point_base : point_base + 12] = 0.0
                history[:, normal_rate_base : normal_rate_base + 12] = 0.0
        future = action_q[current + 1 : current + 1 + self.pred_horizon]
        if self.action_representation == "delta_q":
            current_live_q = state[current, :ACTION_DIM]
            action = future - current_live_q
        elif self.action_representation == "kinematic_residual_q":
            action = future - self._kinematic_baseline(state, current)
        else:
            action = future
        history = (
            history - self.normalization.state_mean
        ) / self.normalization.state_std
        # Ablation profiles share one H5 and one network shape.  Mask after
        # normalization so a disabled channel is exactly zero rather than the
        # normalized representation of a physical zero.
        if self.state_input_mask is not None:
            history = history * self.state_input_mask
        action = (
            action - self.normalization.action_mean
        ) / self.normalization.action_std
        if eid in self.bounded_action_episode_ids:
            # The deployed LeRobot sampler clips normalized actions to this
            # interval.  A DAgger teacher farther away than one representable
            # step must therefore supply a bounded teacher-directed recovery
            # command; repeated replans complete the return instead of asking
            # the network to imitate an action it can never execute.
            action = np.clip(action, -1.0, 1.0)
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
        return self.episodes[eid][0][current, :ACTION_DIM]

    def future_q(self, index: int) -> np.ndarray:
        eid, current = self.windows[index]
        return self.episodes[eid][1][current + 1 : current + 1 + self.pred_horizon]

    def _kinematic_baseline(
        self, state: np.ndarray, current: int
    ) -> np.ndarray:
        return kinematic_q_baseline(
            state[current, :ACTION_DIM],
            state[current, Q_VELOCITY_SLICE],
            self.pred_horizon,
            self.waypoint_dt,
            self.kinematic_velocity_clip,
        )

    def action_base(self, index: int) -> np.ndarray:
        """Base added to the model's physical action output."""
        eid, current = self.windows[index]
        state, _ = self.episodes[eid]
        if self.action_representation == "absolute_q":
            return np.zeros((self.pred_horizon, self.action_dim), dtype=np.float32)
        if self.action_representation == "delta_q":
            return np.broadcast_to(
                state[current, :ACTION_DIM],
                (self.pred_horizon, self.action_dim),
            ).copy()
        return self._kinematic_baseline(state, current)

    def kinematic_baseline(self, index: int) -> np.ndarray:
        """Return the explicit local-motion baseline for diagnostics."""
        eid, current = self.windows[index]
        state, _ = self.episodes[eid]
        if self.action_dim != ACTION_DIM:
            # Cartesian intent channels (tip_delta) have no kinematic joint
            # baseline; diagnostics treat the teacher action itself as the
            # reference, i.e. a zero residual.
            return np.zeros((self.pred_horizon, self.action_dim), dtype=np.float32)
        if self.state_schema not in MOTION_SCHEMAS:
            return np.broadcast_to(
                state[current, :ACTION_DIM],
                (self.pred_horizon, self.action_dim),
            ).copy()
        return self._kinematic_baseline(state, current)
