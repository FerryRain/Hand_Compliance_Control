"""Causal conditioning features derived from a future palm motion plan.

The upper-level planner is assumed to provide future palm poses.  This module
encodes those poses relative to the current palm, so the DP input does not
depend on an object/world coordinate frame.  During teacher-data export the
same signal is reconstructed from ``palm_pose_object``.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


# The deployment interface intentionally uses only the next local planner
# target.  Requiring a long future pose sequence creates an unnecessary
# train/deploy dependency on the upper-level planner.
DEFAULT_PLANNER_WAYPOINTS = 1
DEFAULT_PLANNER_STEP_FRAMES = 20
PLANNER_WAYPOINT_DIM = 6


def future_palm_delta_pose_palm(
    palm_pose_object: np.ndarray,
    episode_id: np.ndarray,
    *,
    waypoint_count: int = DEFAULT_PLANNER_WAYPOINTS,
    step_frames: int = DEFAULT_PLANNER_STEP_FRAMES,
) -> np.ndarray:
    """Encode future palm waypoints as current-palm-frame 6-D delta poses.

    For waypoint ``k`` the translation is

    ``R_object_from_palm(t).T @ (p(t+k) - p(t))``

    and the rotation is the rotation vector of

    ``R_object_from_palm(t).T @ R_object_from_palm(t+k)``.

    Targets are clamped at the end of each episode and never cross reset
    boundaries.  Quaternion order is wxyz in the input; rotation-vector units
    are radians.
    """

    if waypoint_count <= 0:
        raise ValueError("waypoint_count must be positive")
    if step_frames <= 0:
        raise ValueError("step_frames must be positive")
    pose = np.asarray(palm_pose_object, dtype=np.float64)
    if pose.ndim < 2 or pose.shape[-1] != 7:
        raise ValueError(f"Expected palm poses [..., 7], got {pose.shape}")
    base_shape = pose.shape[:-1]
    flat_pose = pose.reshape(-1, 7)
    flat_episode = np.asarray(episode_id).reshape(-1)
    if flat_episode.shape[0] != flat_pose.shape[0]:
        raise ValueError(
            f"episode_id has {flat_episode.shape[0]} samples but pose has "
            f"{flat_pose.shape[0]}"
        )
    if not np.all(np.isfinite(flat_pose)):
        raise ValueError("palm_pose_object contains non-finite values")

    output = np.empty(
        (len(flat_pose), waypoint_count, PLANNER_WAYPOINT_DIM),
        dtype=np.float32,
    )
    offsets = step_frames * np.arange(1, waypoint_count + 1, dtype=np.int64)
    for eid in np.unique(flat_episode):
        indices = np.flatnonzero(flat_episode == eid)
        if not len(indices):
            continue
        local_targets = np.minimum(
            np.arange(len(indices), dtype=np.int64)[:, None] + offsets[None, :],
            len(indices) - 1,
        )
        current = flat_pose[indices]
        future = flat_pose[indices[local_targets]]

        current_quat_xyzw = current[:, [4, 5, 6, 3]]
        future_quat_xyzw = future[..., [4, 5, 6, 3]]
        repeated_current_quat = np.broadcast_to(
            current_quat_xyzw[:, None, :], future_quat_xyzw.shape
        )
        current_rotation = Rotation.from_quat(
            repeated_current_quat.reshape(-1, 4)
        )
        future_rotation = Rotation.from_quat(future_quat_xyzw.reshape(-1, 4))

        translation_object = future[..., :3] - current[:, None, :3]
        translation_palm = current_rotation.inv().apply(
            translation_object.reshape(-1, 3)
        )
        rotation_vector_palm = (
            current_rotation.inv() * future_rotation
        ).as_rotvec()
        encoded = np.concatenate(
            (translation_palm, rotation_vector_palm), axis=-1
        ).reshape(len(indices), waypoint_count, PLANNER_WAYPOINT_DIM)
        output[indices] = encoded.astype(np.float32)

    return output.reshape(*base_shape, waypoint_count, PLANNER_WAYPOINT_DIM)


def planner_feature_dim(waypoint_count: int = DEFAULT_PLANNER_WAYPOINTS) -> int:
    if waypoint_count <= 0:
        raise ValueError("waypoint_count must be positive")
    return waypoint_count * PLANNER_WAYPOINT_DIM
