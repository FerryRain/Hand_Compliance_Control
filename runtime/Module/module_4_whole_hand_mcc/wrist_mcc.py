"""Six-dimensional Cartesian wrist admittance for the FR3 branch."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from numpy.typing import ArrayLike, NDArray


def _vector(value: ArrayLike, name: str, length: int) -> NDArray[np.float64]:
  result = np.asarray(value, dtype=np.float64)
  if result.shape != (length,) or not np.all(np.isfinite(result)):
    raise ValueError(f"{name} must be finite with shape ({length},)")
  return np.array(result, dtype=np.float64, copy=True)


@dataclass(frozen=True, slots=True)
class WristMCCConfig:
  virtual_mass: ArrayLike = (4.0, 4.0, 4.0, 0.20, 0.20, 0.20)
  damping: ArrayLike = (120.0, 120.0, 120.0, 5.0, 5.0, 5.0)
  stiffness: ArrayLike = (800.0, 800.0, 800.0, 25.0, 25.0, 25.0)
  dt_s: float = 0.02
  max_abs_offset: ArrayLike = (0.012, 0.012, 0.012, 0.10, 0.10, 0.10)
  max_abs_velocity: ArrayLike = (0.08, 0.08, 0.08, 0.5, 0.5, 0.5)
  max_abs_acceleration: ArrayLike = (1.5, 1.5, 1.5, 8.0, 8.0, 8.0)

  def __post_init__(self) -> None:
    for name in (
      "virtual_mass",
      "damping",
      "stiffness",
      "max_abs_offset",
      "max_abs_velocity",
      "max_abs_acceleration",
    ):
      value = _vector(getattr(self, name), name, 6)
      if name == "stiffness":
        if np.any(value < 0.0):
          raise ValueError("stiffness must be non-negative")
      elif np.any(value <= 0.0):
        raise ValueError(f"{name} must be positive")
      object.__setattr__(self, name, value)
    if not np.isfinite(self.dt_s) or self.dt_s <= 0.0:
      raise ValueError("dt_s must be finite and positive")


@dataclass(frozen=True, slots=True)
class WristMCCState:
  offset: NDArray[np.float64]
  velocity: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class WristMCCCommand:
  pose_command: NDArray[np.float64]
  offset: NDArray[np.float64]
  velocity: NDArray[np.float64]
  acceleration: NDArray[np.float64]
  wrench_error: NDArray[np.float64]
  selected_wrench_error: NDArray[np.float64]
  saturated_axes: tuple[int, ...]


def _quaternion_multiply(first: NDArray[np.float64], second: NDArray[np.float64]) -> NDArray[np.float64]:
  w1, x1, y1, z1 = first
  w2, x2, y2, z2 = second
  result = np.array(
    [
      w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
      w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
      w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
      w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ],
    dtype=np.float64,
  )
  return result / np.linalg.norm(result)


def _rotation_vector_quaternion(rotation_vector: NDArray[np.float64]) -> NDArray[np.float64]:
  angle = float(np.linalg.norm(rotation_vector))
  if angle < 1e-12:
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
  axis = rotation_vector / angle
  return np.array([np.cos(angle / 2.0), *(np.sin(angle / 2.0) * axis)], dtype=np.float64)


class WristMCC:
  """Stateful 6D admittance with an explicit power-space selection matrix."""

  def __init__(self, config: WristMCCConfig | None = None) -> None:
    self.config = config or WristMCCConfig()
    self._offset = np.zeros(6, dtype=np.float64)
    self._velocity = np.zeros(6, dtype=np.float64)

  @property
  def state(self) -> WristMCCState:
    return WristMCCState(self._offset.copy(), self._velocity.copy())

  def reset(self) -> None:
    self._offset[:] = 0.0
    self._velocity[:] = 0.0

  def step(
    self,
    planned_pose_world: ArrayLike,
    desired_hand_wrench_world: ArrayLike,
    measured_hand_wrench_world: ArrayLike,
    selection_matrix: ArrayLike,
  ) -> WristMCCCommand:
    pose = _vector(planned_pose_world, "planned_pose_world", 7)
    if not np.isclose(np.linalg.norm(pose[3:]), 1.0, atol=1e-6):
      raise ValueError("planned pose quaternion must be unit length")
    desired = _vector(desired_hand_wrench_world, "desired_hand_wrench_world", 6)
    measured = _vector(measured_hand_wrench_world, "measured_hand_wrench_world", 6)
    selection = np.asarray(selection_matrix, dtype=np.float64)
    if selection.shape != (6, 6) or not np.all(np.isfinite(selection)):
      raise ValueError("selection_matrix must be finite with shape (6,6)")
    if not np.allclose(selection, selection.T, atol=1e-8) or not np.allclose(
      selection @ selection,
      selection,
      atol=1e-6,
    ):
      raise ValueError("selection_matrix must be an orthogonal projector")

    error = desired - measured
    # A positive object-on-hand force error (desired > measured) requires the
    # palm to move *into* the object, opposite that reaction direction.  The
    # admittance drive therefore uses the negative selected wrench error.
    selected = -(selection @ error)
    raw_acceleration = (
      selected - self.config.damping * self._velocity - self.config.stiffness * self._offset
    ) / self.config.virtual_mass
    acceleration = np.clip(
      raw_acceleration,
      -self.config.max_abs_acceleration,
      self.config.max_abs_acceleration,
    )
    velocity_unclipped = self._velocity + acceleration * self.config.dt_s
    velocity = np.clip(
      velocity_unclipped,
      -self.config.max_abs_velocity,
      self.config.max_abs_velocity,
    )
    offset_unclipped = self._offset + velocity * self.config.dt_s
    offset = np.clip(
      offset_unclipped,
      -self.config.max_abs_offset,
      self.config.max_abs_offset,
    )
    saturated = np.flatnonzero(
      (acceleration != raw_acceleration)
      | (velocity != velocity_unclipped)
      | (offset != offset_unclipped)
    )
    for axis in saturated:
      if np.sign(velocity[axis]) == np.sign(offset_unclipped[axis] - offset[axis]):
        velocity[axis] = 0.0
    self._offset[:] = offset
    self._velocity[:] = velocity

    pose_command = pose.copy()
    pose_command[:3] += offset[:3]
    delta_quaternion = _rotation_vector_quaternion(offset[3:])
    pose_command[3:] = _quaternion_multiply(delta_quaternion, pose[3:])
    frozen = [pose_command, offset.copy(), velocity.copy(), acceleration.copy(), error, selected]
    for value in frozen:
      value.setflags(write=False)
    return WristMCCCommand(
      pose_command=frozen[0],
      offset=frozen[1],
      velocity=frozen[2],
      acceleration=frozen[3],
      wrench_error=frozen[4],
      selected_wrench_error=frozen[5],
      saturated_axes=tuple(int(value) for value in saturated),
    )
