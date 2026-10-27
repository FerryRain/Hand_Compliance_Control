"""Measured-state anchored relative command chunks for Finger DP."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from Module.module_4_finger_dp.contracts import NUM_FINGER_JOINTS


def _matrix(value: ArrayLike, name: str, columns: int) -> NDArray[np.float64]:
  result = np.asarray(value, dtype=np.float64)
  if result.ndim != 2 or result.shape[1] != columns or not np.all(np.isfinite(result)):
    raise ValueError(f"{name} must be finite with shape (T,{columns})")
  return np.array(result, dtype=np.float64, copy=True)


def _vector(value: ArrayLike, name: str) -> NDArray[np.float64]:
  result = np.asarray(value, dtype=np.float64)
  if result.shape != (NUM_FINGER_JOINTS,) or not np.all(np.isfinite(result)):
    raise ValueError(f"{name} must be finite with shape ({NUM_FINGER_JOINTS},)")
  return np.array(result, dtype=np.float64, copy=True)


@dataclass(frozen=True, slots=True)
class MeasuredAnchoredActionChunk:
  """A future command sequence whose single anchor is measured ``q_t``."""

  anchor_q_meas_rad: ArrayLike
  offsets_rad: ArrayLike
  policy_dt_s: float = 0.02

  def __post_init__(self) -> None:
    anchor = _vector(self.anchor_q_meas_rad, "anchor_q_meas_rad")
    offsets = _matrix(self.offsets_rad, "offsets_rad", NUM_FINGER_JOINTS)
    if offsets.shape[0] < 1:
      raise ValueError("an action chunk must contain at least one future command")
    if not np.isfinite(self.policy_dt_s) or self.policy_dt_s <= 0.0:
      raise ValueError("policy_dt_s must be finite and positive")
    anchor.setflags(write=False)
    offsets.setflags(write=False)
    object.__setattr__(self, "anchor_q_meas_rad", anchor)
    object.__setattr__(self, "offsets_rad", offsets)

  @property
  def horizon(self) -> int:
    return int(self.offsets_rad.shape[0])

  @property
  def nominal_commands_rad(self) -> NDArray[np.float64]:
    result = self.anchor_q_meas_rad[None, :] + self.offsets_rad
    result.setflags(write=False)
    return result

  def nominal_command(self, index: int) -> NDArray[np.float64]:
    if index < 0 or index >= self.horizon:
      raise IndexError("action-chunk index is outside the horizon")
    result = self.anchor_q_meas_rad + self.offsets_rad[index]
    result.setflags(write=False)
    return result

  def seam_blended_command(
    self,
    index: int,
    previous_executed_command_rad: ArrayLike,
    blend_steps: int,
  ) -> NDArray[np.float64]:
    if blend_steps < 1:
      raise ValueError("blend_steps must be at least one")
    previous = _vector(
      previous_executed_command_rad,
      "previous_executed_command_rad",
    )
    alpha = min(1.0, (index + 1) / blend_steps)
    result = (1.0 - alpha) * previous + alpha * self.nominal_command(index)
    return result


@dataclass(frozen=True, slots=True)
class TeacherCommandChunk:
  """One command-imitation label; never a future measured-state label."""

  start_index: int
  anchor_q_meas_rad: NDArray[np.float64]
  future_teacher_command_rad: NDArray[np.float64]
  target_offsets_rad: NDArray[np.float64]
  teacher_source: str
  repaired: bool

  def as_action_chunk(self, policy_dt_s: float = 0.02) -> MeasuredAnchoredActionChunk:
    return MeasuredAnchoredActionChunk(
      self.anchor_q_meas_rad,
      self.target_offsets_rad,
      policy_dt_s,
    )


def build_teacher_command_chunks(
  q_measured_rad: ArrayLike,
  q_teacher_command_rad: ArrayLike,
  *,
  horizon: int,
  stride: int = 1,
  usable_mask: ArrayLike | None = None,
  teacher_source: str = "VERIFIED_INVERSE",
  repair_mask: ArrayLike | None = None,
) -> tuple[TeacherCommandChunk, ...]:
  """Construct command chunks anchored at ``q_measured[t]``.

  The target at start ``t`` is exactly
  ``q_teacher_command[t+1:t+1+H] - q_measured[t]``.  A chunk is excluded when
  any future command is marked unusable, for example because Hard Guard owned
  execution authority.
  """

  measured = _matrix(q_measured_rad, "q_measured_rad", NUM_FINGER_JOINTS)
  teacher = _matrix(
    q_teacher_command_rad,
    "q_teacher_command_rad",
    NUM_FINGER_JOINTS,
  )
  if measured.shape != teacher.shape:
    raise ValueError("measured and teacher command trajectories must have equal shape")
  if horizon < 1 or stride < 1:
    raise ValueError("horizon and stride must be positive")
  if not teacher_source:
    raise ValueError("teacher_source must be non-empty")
  count = measured.shape[0]
  usable = np.ones(count, dtype=np.bool_)
  if usable_mask is not None:
    usable = np.asarray(usable_mask, dtype=np.bool_)
    if usable.shape != (count,):
      raise ValueError("usable_mask must have shape (T,)")
  repaired_steps = np.zeros(count, dtype=np.bool_)
  if repair_mask is not None:
    repaired_steps = np.asarray(repair_mask, dtype=np.bool_)
    if repaired_steps.shape != (count,):
      raise ValueError("repair_mask must have shape (T,)")

  chunks: list[TeacherCommandChunk] = []
  for start in range(0, count - horizon, stride):
    future_slice = slice(start + 1, start + 1 + horizon)
    if not bool(np.all(usable[future_slice])):
      continue
    anchor = measured[start].copy()
    future = teacher[future_slice].copy()
    target = future - anchor[None, :]
    for value in (anchor, future, target):
      value.setflags(write=False)
    chunks.append(
      TeacherCommandChunk(
        start_index=start,
        anchor_q_meas_rad=anchor,
        future_teacher_command_rad=future,
        target_offsets_rad=target,
        teacher_source=teacher_source,
        repaired=bool(np.any(repaired_steps[future_slice])),
      )
    )
  return tuple(chunks)
