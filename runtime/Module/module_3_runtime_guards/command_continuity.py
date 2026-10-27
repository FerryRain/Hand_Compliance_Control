"""Deterministic rate limits for controller/guard command ownership changes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _finite_vector(value: ArrayLike, length: int, name: str) -> NDArray[np.float64]:
  result = np.asarray(value, dtype=np.float64)
  if result.shape != (length,) or not np.all(np.isfinite(result)):
    raise ValueError(f"{name} must be finite with shape ({length},)")
  return np.array(result, dtype=np.float64, copy=True)


@dataclass(frozen=True, slots=True)
class CommandContinuityConfig:
  max_finger_step_rad: float = 0.001
  max_wrist_translation_step_m: float = 0.00008
  max_wrist_rotation_step_rad: float = 0.01

  def __post_init__(self) -> None:
    for name in (
      "max_finger_step_rad",
      "max_wrist_translation_step_m",
      "max_wrist_rotation_step_rad",
    ):
      value = float(getattr(self, name))
      if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


class CommandContinuityLimiter:
  """Bound per-tick finger and SE(3) target changes across authority switches."""

  def __init__(self, config: CommandContinuityConfig | None = None) -> None:
    self.config = config or CommandContinuityConfig()
    self._finger_command: NDArray[np.float64] | None = None
    self._wrist_pose: NDArray[np.float64] | None = None

  def reset(
    self,
    *,
    finger_command_rad: ArrayLike,
    wrist_pose_world: ArrayLike,
  ) -> None:
    self._finger_command = _finite_vector(finger_command_rad, 16, "finger_command_rad")
    self._wrist_pose = self._normalized_pose(wrist_pose_world)

  @staticmethod
  def _normalized_pose(value: ArrayLike) -> NDArray[np.float64]:
    pose = _finite_vector(value, 7, "wrist_pose_world")
    norm = float(np.linalg.norm(pose[3:]))
    if norm < 1e-12:
      raise ValueError("wrist pose quaternion must be non-zero")
    pose[3:] /= norm
    return pose

  @staticmethod
  def _bounded_quaternion(
    previous: NDArray[np.float64],
    target: NDArray[np.float64],
    maximum_angle: float,
  ) -> NDArray[np.float64]:
    aligned = target.copy()
    dot = float(np.dot(previous, aligned))
    if dot < 0.0:
      aligned *= -1.0
      dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    angle = 2.0 * float(np.arccos(dot))
    if angle <= maximum_angle or angle < 1e-12:
      return aligned
    fraction = maximum_angle / angle
    theta = float(np.arccos(dot))
    sin_theta = float(np.sin(theta))
    if sin_theta < 1e-9:
      blended = (1.0 - fraction) * previous + fraction * aligned
    else:
      blended = (
        np.sin((1.0 - fraction) * theta) / sin_theta * previous
        + np.sin(fraction * theta) / sin_theta * aligned
      )
    return blended / np.linalg.norm(blended)

  def limit_finger(
    self,
    proposed_command_rad: ArrayLike,
    maximum_step_rad: ArrayLike | float | None = None,
  ) -> NDArray[np.float64]:
    proposed = _finite_vector(proposed_command_rad, 16, "proposed_finger_command_rad")
    if self._finger_command is None:
      self._finger_command = proposed
      return proposed.copy()
    if maximum_step_rad is None:
      maximum_step = np.full(16, self.config.max_finger_step_rad, dtype=np.float64)
    else:
      candidate = np.asarray(maximum_step_rad, dtype=np.float64)
      if candidate.ndim == 0:
        maximum_step = np.full(16, float(candidate), dtype=np.float64)
      elif candidate.shape == (16,):
        maximum_step = np.array(candidate, dtype=np.float64, copy=True)
      else:
        raise ValueError("maximum_step_rad must be scalar or have shape (16,)")
      if not np.all(np.isfinite(maximum_step)) or np.any(maximum_step <= 0.0):
        raise ValueError("maximum_step_rad must contain finite positive values")
    delta = np.clip(
      proposed - self._finger_command,
      -maximum_step,
      maximum_step,
    )
    self._finger_command = self._finger_command + delta
    return self._finger_command.copy()

  def limit_wrist(
    self,
    proposed_pose_world: ArrayLike,
    *,
    normal_direction_world: ArrayLike | None = None,
    max_normal_step_m: float | None = None,
    max_tangent_step_m: float | None = None,
  ) -> NDArray[np.float64]:
    proposed = self._normalized_pose(proposed_pose_world)
    if self._wrist_pose is None:
      self._wrist_pose = proposed
      return proposed.copy()
    delta = proposed[:3] - self._wrist_pose[:3]
    if normal_direction_world is None:
      length = float(np.linalg.norm(delta))
      if length > self.config.max_wrist_translation_step_m:
        delta *= self.config.max_wrist_translation_step_m / length
    else:
      normal = _finite_vector(normal_direction_world, 3, "normal_direction_world")
      normal_norm = float(np.linalg.norm(normal))
      if normal_norm < 1e-12:
        raise ValueError("normal_direction_world must be non-zero")
      normal /= normal_norm
      normal_limit = (
        self.config.max_wrist_translation_step_m
        if max_normal_step_m is None
        else float(max_normal_step_m)
      )
      if not np.isfinite(normal_limit) or normal_limit <= 0.0:
        raise ValueError("max_normal_step_m must be finite and positive")
      tangent_limit = (
        self.config.max_wrist_translation_step_m
        if max_tangent_step_m is None
        else float(max_tangent_step_m)
      )
      if not np.isfinite(tangent_limit) or tangent_limit <= 0.0:
        raise ValueError("max_tangent_step_m must be finite and positive")
      normal_delta = float(np.clip(np.dot(delta, normal), -normal_limit, normal_limit))
      tangent_delta = delta - np.dot(delta, normal) * normal
      tangent_length = float(np.linalg.norm(tangent_delta))
      if tangent_length > tangent_limit:
        tangent_delta *= tangent_limit / tangent_length
      delta = tangent_delta + normal_delta * normal
    limited = self._wrist_pose.copy()
    limited[:3] += delta
    limited[3:] = self._bounded_quaternion(
      self._wrist_pose[3:],
      proposed[3:],
      self.config.max_wrist_rotation_step_rad,
    )
    self._wrist_pose = limited
    return limited.copy()


__all__ = ["CommandContinuityConfig", "CommandContinuityLimiter"]
