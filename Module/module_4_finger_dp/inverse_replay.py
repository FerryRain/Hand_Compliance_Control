"""SE(3) inverse proposals; these are proposals, never physical demonstrations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation


def _poses(value: ArrayLike, name: str) -> NDArray[np.float64]:
  result = np.asarray(value, dtype=np.float64)
  if result.ndim != 2 or result.shape[1] != 7 or not np.all(np.isfinite(result)):
    raise ValueError(f"{name} must be finite with shape (T,7)")
  norms = np.linalg.norm(result[:, 3:], axis=1)
  if np.any(np.abs(norms - 1.0) > 1e-6):
    raise ValueError(f"{name} quaternions must be unit [qw,qx,qy,qz]")
  return np.array(result, dtype=np.float64, copy=True)


def _pose(value: ArrayLike, name: str) -> NDArray[np.float64]:
  result = np.asarray(value, dtype=np.float64)
  if result.shape != (7,) or not np.all(np.isfinite(result)):
    raise ValueError(f"{name} must be finite with shape (7,)")
  if not np.isclose(np.linalg.norm(result[3:]), 1.0, atol=1e-6, rtol=0.0):
    raise ValueError(f"{name} quaternion must be unit [qw,qx,qy,qz]")
  return np.array(result, dtype=np.float64, copy=True)


def pose_to_matrix(pose: ArrayLike) -> NDArray[np.float64]:
  value = _pose(pose, "pose")
  matrix = np.eye(4, dtype=np.float64)
  matrix[:3, 3] = value[:3]
  matrix[:3, :3] = Rotation.from_quat(
    [value[4], value[5], value[6], value[3]]
  ).as_matrix()
  return matrix


def matrix_to_pose(matrix: ArrayLike) -> NDArray[np.float64]:
  value = np.asarray(matrix, dtype=np.float64)
  if value.shape != (4, 4) or not np.all(np.isfinite(value)):
    raise ValueError("matrix must be finite with shape (4,4)")
  if not np.allclose(value[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
    raise ValueError("matrix is not a homogeneous transform")
  xyzw = Rotation.from_matrix(value[:3, :3]).as_quat()
  return np.array(
    [value[0, 3], value[1, 3], value[2, 3], xyzw[3], xyzw[0], xyzw[1], xyzw[2]],
    dtype=np.float64,
  )


def relative_pose(parent_world: ArrayLike, child_world: ArrayLike) -> NDArray[np.float64]:
  transform = np.linalg.inv(pose_to_matrix(parent_world)) @ pose_to_matrix(child_world)
  return matrix_to_pose(transform)


@dataclass(frozen=True, slots=True)
class InverseReplayProposal:
  wrist_pose_world: NDArray[np.float64]
  object_pose_in_hand: NDArray[np.float64]
  fixed_object_pose_world: NDArray[np.float64]
  reverse_time: bool
  maximum_relative_transform_residual: float


def inverse_replay_wrist_proposal(
  forward_hand_pose_world: ArrayLike,
  forward_object_pose_world: ArrayLike,
  fixed_object_pose_world: ArrayLike,
  *,
  reverse_time: bool = False,
) -> InverseReplayProposal:
  """Convert object-in-hand motion into a fixed-object wrist-pose proposal.

  No force, contact or action label is transformed here.  Those channels must
  be measured again during closed-loop physical replay.  The v1 default is a
  *spatial* inversion: sample ``t`` remains sample ``t``.  ``reverse_time`` is
  an explicit, separate experimental choice and is never implied by changing
  the parent/child reference frame.
  """

  hand = _poses(forward_hand_pose_world, "forward_hand_pose_world")
  object_world = _poses(forward_object_pose_world, "forward_object_pose_world")
  fixed = _pose(fixed_object_pose_world, "fixed_object_pose_world")
  if hand.shape != object_world.shape:
    raise ValueError("forward hand and object trajectories must have equal shape")
  relative_matrices = np.stack(
    [
      np.linalg.inv(pose_to_matrix(hand[index])) @ pose_to_matrix(object_world[index])
      for index in range(len(hand))
    ]
  )
  if reverse_time:
    relative_matrices = relative_matrices[::-1].copy()
  fixed_matrix = pose_to_matrix(fixed)
  wrist_matrices = np.stack(
    [fixed_matrix @ np.linalg.inv(relative) for relative in relative_matrices]
  )
  wrist_poses = np.stack([matrix_to_pose(value) for value in wrist_matrices])
  relative_poses = np.stack([matrix_to_pose(value) for value in relative_matrices])
  # Make quaternion signs continuous for interpolation without changing poses.
  for trajectory in (wrist_poses, relative_poses):
    for index in range(1, len(trajectory)):
      if float(np.dot(trajectory[index - 1, 3:], trajectory[index, 3:])) < 0.0:
        trajectory[index, 3:] *= -1.0
  residual = 0.0
  for wrist_matrix, expected_relative in zip(wrist_matrices, relative_matrices):
    reconstructed = np.linalg.inv(wrist_matrix) @ fixed_matrix
    residual = max(residual, float(np.linalg.norm(reconstructed - expected_relative)))
  for value in (wrist_poses, relative_poses, fixed):
    value.setflags(write=False)
  return InverseReplayProposal(
    wrist_pose_world=wrist_poses,
    object_pose_in_hand=relative_poses,
    fixed_object_pose_world=fixed,
    reverse_time=reverse_time,
    maximum_relative_transform_residual=residual,
  )


def spatial_inverse_replay_proposal(
  forward_hand_pose_world: ArrayLike,
  forward_object_pose_world: ArrayLike,
  fixed_object_pose_world: ArrayLike,
) -> InverseReplayProposal:
  """Spatial role inversion with forward time order preserved.

  This named entry point is the frozen v1 inverse-data operation.  Finger
  commands are deliberately not arguments: their proposal mapping is the
  identity ``q_cmd_replay[t] = q_cmd_forward[t]`` and is handled by the
  physical-pair builder, where it can be audited independently of SE(3).
  """

  return inverse_replay_wrist_proposal(
    forward_hand_pose_world,
    forward_object_pose_world,
    fixed_object_pose_world,
    reverse_time=False,
  )


def temporal_reverse_replay_proposal(
  forward_hand_pose_world: ArrayLike,
  forward_object_pose_world: ArrayLike,
  fixed_object_pose_world: ArrayLike,
) -> InverseReplayProposal:
  """Optional temporal-reversal experiment; not the v1 spatial pipeline."""

  return inverse_replay_wrist_proposal(
    forward_hand_pose_world,
    forward_object_pose_world,
    fixed_object_pose_world,
    reverse_time=True,
  )
