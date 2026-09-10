"""Causal local-motion features shared by DP export and deployment."""

from __future__ import annotations

import numpy as np


MOTION_SCHEMA = "contact_geometry_planner_motion"
DUAL_TRACK_SCHEMA = "contact_geometry_planner_motion_dual_track"
DUAL_TRACK_V3_SCHEMA = "contact_geometry_planner_motion_dual_track_v3"
TASK_EST_V4_SCHEMA = "contact_geometry_planner_task_est_v4"
MOTION_SCHEMAS = (
    MOTION_SCHEMA,
    DUAL_TRACK_SCHEMA,
    DUAL_TRACK_V3_SCHEMA,
    TASK_EST_V4_SCHEMA,
)
Q_VELOCITY_SLICE = slice(44, 60)


def causal_motion_features(
    q_hand: np.ndarray,
    contact_position_palm: np.ndarray,
    contact_normal_palm: np.ndarray,
    contact_mask: np.ndarray,
    episode_id: np.ndarray,
    *,
    control_dt: float,
    step_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return causal q, contact-point and contact-normal rates.

    Inputs may contain interleaved vectorized environments.  ``episode_id`` is
    therefore used instead of assuming adjacent rows belong to one trajectory.
    Point/normal differences deliberately use the palm-frame values stored at
    each endpoint; live deployment applies the identical convention.
    """

    if control_dt <= 0.0:
        raise ValueError("control_dt must be positive")
    if step_frames <= 0:
        raise ValueError("step_frames must be positive")
    q = np.asarray(q_hand, dtype=np.float64)
    position = np.asarray(contact_position_palm, dtype=np.float64)
    normal = np.asarray(contact_normal_palm, dtype=np.float64)
    mask = np.asarray(contact_mask) > 0.5
    ids = np.asarray(episode_id).reshape(-1)
    base_shape = q.shape[:-1]
    q_flat = q.reshape(-1, 16)
    position_flat = position.reshape(-1, 4, 3)
    normal_flat = normal.reshape(-1, 4, 3)
    mask_flat = mask.reshape(-1, 4)
    if ids.shape[0] != q_flat.shape[0]:
        raise ValueError("episode_id and feature arrays have different lengths")

    q_velocity = np.zeros_like(q_flat, dtype=np.float32)
    point_velocity = np.zeros_like(position_flat, dtype=np.float32)
    normal_rate = np.zeros_like(normal_flat, dtype=np.float32)
    duration = step_frames * control_dt
    for episode in np.unique(ids):
        indices = np.flatnonzero(ids == episode)
        if len(indices) <= step_frames:
            continue
        current = indices[step_frames:]
        previous = indices[:-step_frames]
        q_velocity[current] = ((q_flat[current] - q_flat[previous]) / duration).astype(
            np.float32
        )
        valid = mask_flat[current] & mask_flat[previous]
        point_delta = (position_flat[current] - position_flat[previous]) / duration
        # For unit normals, n_prev x n_now is the small-angle angular-rate
        # vector.  Large mesh-seam changes remain visible but are masked when
        # either endpoint has no valid contact.
        angular = np.cross(normal_flat[previous], normal_flat[current]) / duration
        point_velocity[current] = np.where(
            valid[..., None], point_delta, 0.0
        ).astype(np.float32)
        normal_rate[current] = np.where(valid[..., None], angular, 0.0).astype(
            np.float32
        )

    return (
        q_velocity.reshape(*base_shape, 16),
        point_velocity.reshape(*base_shape, 4, 3),
        normal_rate.reshape(*base_shape, 4, 3),
    )


def kinematic_q_baseline(
    current_q: np.ndarray,
    current_q_velocity: np.ndarray,
    pred_horizon: int,
    waypoint_dt: float,
    velocity_clip: float,
) -> np.ndarray:
    """Constant-velocity local baseline, returned as absolute joint positions."""

    if pred_horizon <= 0 or waypoint_dt <= 0.0 or velocity_clip <= 0.0:
        raise ValueError("invalid kinematic baseline configuration")
    q = np.asarray(current_q, dtype=np.float32)
    velocity = np.clip(
        np.asarray(current_q_velocity, dtype=np.float32),
        -velocity_clip,
        velocity_clip,
    )
    time = waypoint_dt * np.arange(1, pred_horizon + 1, dtype=np.float32)
    return q[None, :] + time[:, None] * velocity[None, :]
