"""Discrete normal-direction fingertip MCC.

``compliance_direction`` must point in the direction where positive displacement
increases normal contact force. For an object's outward SurfaceModel normal,
pass its negative.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _finite_scalar(value: float, name: str) -> float:
  result = float(value)
  if not np.isfinite(result):
    raise ValueError(f"{name} must be finite")
  return result


def _position3(value: ArrayLike, name: str) -> NDArray[np.float64]:
  array = np.asarray(value, dtype=np.float64)
  if array.shape != (3,) or not np.all(np.isfinite(array)):
    raise ValueError(f"{name} must be a finite vector with shape (3,)")
  return np.array(array, copy=True)


@dataclass(frozen=True, slots=True)
class MCCConfig:
  virtual_mass: float = 0.05
  damping: float = 10.0
  stiffness: float = 10.0
  dt_s: float = 0.002
  max_offset_m: float = 0.01
  max_velocity_m_s: float = 0.25
  max_acceleration_m_s2: float = 100.0
  normal_tolerance: float = 1e-6

  def __post_init__(self) -> None:
    positive = {
      "virtual_mass": self.virtual_mass,
      "damping": self.damping,
      "dt_s": self.dt_s,
      "max_offset_m": self.max_offset_m,
      "max_velocity_m_s": self.max_velocity_m_s,
      "max_acceleration_m_s2": self.max_acceleration_m_s2,
      "normal_tolerance": self.normal_tolerance,
    }
    for name, value in positive.items():
      if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    if not np.isfinite(self.stiffness) or self.stiffness < 0.0:
      raise ValueError("stiffness must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class MCCState:
  offset_m: float = 0.0
  velocity_m_s: float = 0.0


@dataclass(frozen=True, slots=True)
class MCCCommand:
  position_command: NDArray[np.float64]
  offset_m: float
  velocity_m_s: float
  acceleration_m_s2: float
  force_error_n: float
  saturated_limits: tuple[str, ...]


class FingertipMCC:
  """Stateful second-order normal compliance controller."""

  def __init__(self, config: MCCConfig | None = None) -> None:
    self.config = config or MCCConfig()
    self._state = MCCState()

  @property
  def state(self) -> MCCState:
    return self._state

  def reset(self, *, offset_m: float = 0.0, velocity_m_s: float = 0.0) -> None:
    offset = _finite_scalar(offset_m, "offset_m")
    velocity = _finite_scalar(velocity_m_s, "velocity_m_s")
    if abs(offset) > self.config.max_offset_m:
      raise ValueError("initial offset exceeds max_offset_m")
    if abs(velocity) > self.config.max_velocity_m_s:
      raise ValueError("initial velocity exceeds max_velocity_m_s")
    self._state = MCCState(offset, velocity)

  def step(
    self,
    planned_position: ArrayLike,
    compliance_direction: ArrayLike,
    desired_normal_force_n: float,
    measured_normal_force_n: float,
  ) -> MCCCommand:
    plan = _position3(planned_position, "planned_position")
    direction = _position3(compliance_direction, "compliance_direction")
    direction_norm = float(np.linalg.norm(direction))
    if abs(direction_norm - 1.0) > self.config.normal_tolerance:
      raise ValueError("compliance_direction must be unit length")
    direction /= direction_norm

    desired_force = _finite_scalar(desired_normal_force_n, "desired_normal_force_n")
    measured_force = _finite_scalar(measured_normal_force_n, "measured_normal_force_n")
    if desired_force < 0.0 or measured_force < 0.0:
      raise ValueError("normal forces must be non-negative")

    return self.step_force_error(
      planned_position=plan,
      compliance_direction=direction,
      force_error_n=desired_force - measured_force,
    )

  def step_force_error(
    self,
    planned_position: ArrayLike,
    compliance_direction: ArrayLike,
    force_error_n: float,
  ) -> MCCCommand:
    """Advance MCC from an already coordinated signed force error.

    ``E05-F-MCC`` calls :meth:`step` with the full local error.  In
    ``E05-H-MCC`` the Contact Force Coordinator owns the collective component
    and calls this method with only ``N_H e_lambda``.  This avoids fabricating
    desired/measured forces just to reuse the integrator.
    """

    plan = _position3(planned_position, "planned_position")
    direction = _position3(compliance_direction, "compliance_direction")
    direction_norm = float(np.linalg.norm(direction))
    if abs(direction_norm - 1.0) > self.config.normal_tolerance:
      raise ValueError("compliance_direction must be unit length")
    direction /= direction_norm
    force_error = _finite_scalar(force_error_n, "force_error_n")
    raw_acceleration = (
      force_error
      - self.config.damping * self._state.velocity_m_s
      - self.config.stiffness * self._state.offset_m
    ) / self.config.virtual_mass

    saturated: list[str] = []
    acceleration = float(
      np.clip(
        raw_acceleration,
        -self.config.max_acceleration_m_s2,
        self.config.max_acceleration_m_s2,
      )
    )
    if acceleration != raw_acceleration:
      saturated.append("acceleration")

    velocity_unclipped = self._state.velocity_m_s + acceleration * self.config.dt_s
    velocity = float(
      np.clip(
        velocity_unclipped,
        -self.config.max_velocity_m_s,
        self.config.max_velocity_m_s,
      )
    )
    if velocity != velocity_unclipped:
      saturated.append("velocity")

    offset_unclipped = self._state.offset_m + velocity * self.config.dt_s
    offset = float(
      np.clip(
        offset_unclipped,
        -self.config.max_offset_m,
        self.config.max_offset_m,
      )
    )
    if offset != offset_unclipped:
      saturated.append("offset")
      if np.sign(velocity) == np.sign(offset_unclipped - offset):
        velocity = 0.0

    self._state = MCCState(offset, velocity)
    return MCCCommand(
      position_command=plan + offset * direction,
      offset_m=offset,
      velocity_m_s=velocity,
      acceleration_m_s2=acceleration,
      force_error_n=force_error,
      saturated_limits=tuple(saturated),
    )
