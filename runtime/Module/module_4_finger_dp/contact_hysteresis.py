"""Time-confirmed real contact set for DP authority and logging."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from Module.module_4_finger_dp.contracts import NUM_FINGERS


@dataclass(frozen=True, slots=True)
class ContactHysteresisConfig:
  enter_force_n: float = 0.20
  exit_force_n: float = 0.10
  confirm_steps: int = 5
  release_steps: int = 5

  def __post_init__(self) -> None:
    if not 0.0 <= self.exit_force_n < self.enter_force_n:
      raise ValueError("contact forces must satisfy 0 <= exit < enter")
    if self.confirm_steps < 1 or self.release_steps < 1:
      raise ValueError("contact hysteresis step counts must be positive")


@dataclass(frozen=True, slots=True)
class ContactHysteresisOutput:
  actual_contact_mask: NDArray[np.bool_]
  make_mask: NDArray[np.bool_]
  break_mask: NDArray[np.bool_]


class MeasuredContactHysteresis:
  """Confirm MAKE/BREAK from force only; predictions have no authority."""

  def __init__(self, config: ContactHysteresisConfig | None = None) -> None:
    self.config = config or ContactHysteresisConfig()
    self.reset()

  def reset(self) -> None:
    self._active = np.zeros(NUM_FINGERS, dtype=bool)
    self._on_counter = np.zeros(NUM_FINGERS, dtype=np.int32)
    self._off_counter = np.zeros(NUM_FINGERS, dtype=np.int32)

  @property
  def actual_contact_mask(self) -> NDArray[np.bool_]:
    result = np.array(self._active, copy=True)
    result.setflags(write=False)
    return result

  def update(self, normal_force_n: ArrayLike) -> ContactHysteresisOutput:
    forces = np.asarray(normal_force_n, dtype=np.float64)
    if forces.shape != (NUM_FINGERS,) or not np.all(np.isfinite(forces)):
      raise ValueError("normal_force_n must be finite with shape (4,)")
    if np.any(forces < 0.0):
      raise ValueError("normal force magnitudes must be non-negative")
    previous = self._active.copy()
    entering = forces >= self.config.enter_force_n
    leaving = forces < self.config.exit_force_n
    self._on_counter = np.where(
      ~self._active & entering,
      self._on_counter + 1,
      0,
    )
    self._off_counter = np.where(
      self._active & leaving,
      self._off_counter + 1,
      0,
    )
    self._active |= self._on_counter >= self.config.confirm_steps
    self._active &= ~(self._off_counter >= self.config.release_steps)
    self._on_counter[self._active] = 0
    self._off_counter[~self._active] = 0
    make = self._active & ~previous
    broken = previous & ~self._active
    arrays = (self._active.copy(), make, broken)
    for value in arrays:
      value.setflags(write=False)
    return ContactHysteresisOutput(*arrays)
