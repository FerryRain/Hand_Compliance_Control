"""M02-FR3 moving-wrist adapter for four coordinated fingertip MCC loops."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from Module.module_2_fingertip_mcc.controller import FingertipMCC, MCCCommand


@dataclass(frozen=True, slots=True)
class CoordinatedFingerCommand:
  commands: tuple[MCCCommand, ...]
  active_mask: NDArray[np.bool_]
  force_error_n: NDArray[np.float64]


class FullRobotFingertipMCC:
  """Own four MCC states and handle active-contact transitions explicitly."""

  def __init__(self, controllers: tuple[FingertipMCC, ...]) -> None:
    if len(controllers) != 4:
      raise ValueError("exactly four FingertipMCC controllers are required")
    self.controllers = controllers
    self._active = np.zeros(4, dtype=np.bool_)

  @property
  def active_mask(self) -> NDArray[np.bool_]:
    return np.array(self._active, copy=True)

  def reset(self) -> None:
    self._active[:] = False
    for controller in self.controllers:
      controller.reset()

  def step(
    self,
    planned_positions: ArrayLike,
    compliance_directions: ArrayLike,
    force_errors_n: ArrayLike,
    active_mask: ArrayLike,
  ) -> CoordinatedFingerCommand:
    positions = np.asarray(planned_positions, dtype=np.float64)
    directions = np.asarray(compliance_directions, dtype=np.float64)
    errors = np.asarray(force_errors_n, dtype=np.float64)
    active = np.asarray(active_mask, dtype=np.bool_)
    if positions.shape != (4, 3) or directions.shape != (4, 3):
      raise ValueError("planned_positions and compliance_directions must be (4,3)")
    if errors.shape != (4,) or active.shape != (4,):
      raise ValueError("force_errors_n and active_mask must be (4,)")
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(directions)):
      raise ValueError("finger positions and directions must be finite")
    if not np.all(np.isfinite(errors)):
      raise ValueError("force_errors_n must be finite")

    commands: list[MCCCommand] = []
    for index, controller in enumerate(self.controllers):
      # A newly activated contact starts at zero offset/velocity.  A released
      # contact also resets, preventing a stale integrator from being applied
      # at the next MAKE confirmation.
      if bool(active[index]) != bool(self._active[index]):
        controller.reset()
      error = float(errors[index]) if active[index] else 0.0
      commands.append(
        controller.step_force_error(
          positions[index],
          directions[index],
          error,
        )
      )
    self._active[:] = active
    stored_errors = np.where(active, errors, 0.0).astype(np.float64)
    stored_errors.setflags(write=False)
    active_copy = np.array(active, copy=True)
    active_copy.setflags(write=False)
    return CoordinatedFingerCommand(tuple(commands), active_copy, stored_errors)
