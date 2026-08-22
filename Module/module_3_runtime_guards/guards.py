"""Runtime guards that use only signals available in the frozen contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from Module.common import ContactState, ExecutorState


def _finite_vector(value: ArrayLike, name: str) -> NDArray[np.float64]:
  array = np.asarray(value, dtype=np.float64)
  if array.ndim != 1 or not np.all(np.isfinite(array)):
    raise ValueError(f"{name} must be a finite one-dimensional array")
  result = np.array(array, copy=True)
  result.setflags(write=False)
  return result


class GuardReason(str, Enum):
  NONE = "NONE"
  TIP_OVERFORCE = "TIP_OVERFORCE"
  JOINT_LIMIT = "JOINT_LIMIT"
  SELF_COLLISION = "SELF_COLLISION"
  NO_PROGRESS = "NO_PROGRESS"
  SUSPECTED_OBJECT_BLOCKAGE = "SUSPECTED_OBJECT_BLOCKAGE"


class GuardSeverity(str, Enum):
  SAFE = "SAFE"
  SOFT_STOP = "SOFT_STOP"
  HARD_STOP = "HARD_STOP"


@dataclass(frozen=True, slots=True)
class RuntimeGuardConfig:
  joint_lower_rad: ArrayLike
  joint_upper_rad: ArrayLike
  dt_s: float = 0.01
  max_tip_force_n: float = 3.5
  joint_limit_margin_rad: float = 0.02
  min_self_collision_distance_m: float = 0.005
  command_speed_threshold_rad_s: float = 0.05
  actual_progress_threshold_rad_s: float = 0.005
  tip_quiet_force_n: float = 0.1
  stall_time_s: float = 0.15

  def __post_init__(self) -> None:
    lower = _finite_vector(self.joint_lower_rad, "joint_lower_rad")
    upper = _finite_vector(self.joint_upper_rad, "joint_upper_rad")
    if lower.shape != upper.shape or np.any(lower >= upper):
      raise ValueError("joint bounds must have equal shapes and lower < upper")
    positive = {
      "dt_s": self.dt_s,
      "max_tip_force_n": self.max_tip_force_n,
      "joint_limit_margin_rad": self.joint_limit_margin_rad,
      "min_self_collision_distance_m": self.min_self_collision_distance_m,
      "command_speed_threshold_rad_s": self.command_speed_threshold_rad_s,
      "actual_progress_threshold_rad_s": self.actual_progress_threshold_rad_s,
      "tip_quiet_force_n": self.tip_quiet_force_n,
      "stall_time_s": self.stall_time_s,
    }
    for name, value in positive.items():
      if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    object.__setattr__(self, "joint_lower_rad", lower)
    object.__setattr__(self, "joint_upper_rad", upper)


@dataclass(frozen=True, slots=True)
class GuardObservation:
  q_rad: ArrayLike
  qd_command_rad_s: ArrayLike
  qd_actual_rad_s: ArrayLike
  fingertip_forces_n: ArrayLike
  contact_states: tuple[ContactState | str, ...]
  min_self_collision_distance_m: float | None = None

  def __post_init__(self) -> None:
    q = _finite_vector(self.q_rad, "q_rad")
    qd_command = _finite_vector(self.qd_command_rad_s, "qd_command_rad_s")
    qd_actual = _finite_vector(self.qd_actual_rad_s, "qd_actual_rad_s")
    if q.shape != qd_command.shape or q.shape != qd_actual.shape:
      raise ValueError("q, qd_command, and qd_actual must have equal shapes")
    fingertip_forces = _finite_vector(self.fingertip_forces_n, "fingertip_forces_n")
    if np.any(fingertip_forces < 0.0):
      raise ValueError("fingertip forces must be non-negative")
    contact_states = tuple(ContactState(state) for state in self.contact_states)
    if len(contact_states) != len(fingertip_forces):
      raise ValueError("contact_states length must equal fingertip force length")
    if self.min_self_collision_distance_m is not None:
      distance = float(self.min_self_collision_distance_m)
      if not np.isfinite(distance):
        raise ValueError("min_self_collision_distance_m must be finite when provided")
      object.__setattr__(self, "min_self_collision_distance_m", distance)
    object.__setattr__(self, "q_rad", q)
    object.__setattr__(self, "qd_command_rad_s", qd_command)
    object.__setattr__(self, "qd_actual_rad_s", qd_actual)
    object.__setattr__(self, "fingertip_forces_n", fingertip_forces)
    object.__setattr__(self, "contact_states", contact_states)


@dataclass(frozen=True, slots=True)
class GuardEvidence:
  stall_duration_s: float
  commanded_speed_rad_s: float
  progress_speed_rad_s: float
  max_tip_force_n: float
  min_self_collision_distance_m: float | None
  joint_indices: tuple[int, ...]
  local_observation_only: bool = True

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)


@dataclass(frozen=True, slots=True)
class GuardDecision:
  reason: GuardReason
  severity: GuardSeverity
  executor_state: ExecutorState
  evidence: GuardEvidence

  @property
  def should_stop(self) -> bool:
    return self.severity is not GuardSeverity.SAFE

  def to_dict(self) -> dict[str, Any]:
    return {
      "reason": self.reason.value,
      "severity": self.severity.value,
      "executor_state": self.executor_state.value,
      "should_stop": self.should_stop,
      "evidence": self.evidence.to_dict(),
    }


class RuntimeGuards:
  """Stateful guard evaluator with deterministic reason priority.

  Priority is fingertip over-force, known self-collision, joint limit, then
  accumulated no-progress evidence. No unknown collision location or normal is
  synthesized.
  """

  def __init__(self, config: RuntimeGuardConfig) -> None:
    self.config = config
    self._stall_duration_s = 0.0

  @property
  def stall_duration_s(self) -> float:
    return self._stall_duration_s

  def reset(self) -> None:
    self._stall_duration_s = 0.0

  def evaluate(self, observation: GuardObservation) -> GuardDecision:
    if observation.q_rad.shape != self.config.joint_lower_rad.shape:
      raise ValueError("observation joint dimension does not match guard configuration")

    command_speed = float(np.linalg.norm(observation.qd_command_rad_s))
    if command_speed > 0.0:
      progress_speed = float(
        np.dot(observation.qd_actual_rad_s, observation.qd_command_rad_s)
        / command_speed
      )
    else:
      progress_speed = 0.0
    max_tip_force = (
      float(np.max(observation.fingertip_forces_n))
      if len(observation.fingertip_forces_n)
      else 0.0
    )

    if max_tip_force > self.config.max_tip_force_n:
      self.reset()
      return self._decision(
        GuardReason.TIP_OVERFORCE,
        GuardSeverity.HARD_STOP,
        observation,
        command_speed,
        progress_speed,
      )

    collision_distance = observation.min_self_collision_distance_m
    if (
      collision_distance is not None
      and collision_distance <= self.config.min_self_collision_distance_m
    ):
      self.reset()
      return self._decision(
        GuardReason.SELF_COLLISION,
        GuardSeverity.HARD_STOP,
        observation,
        command_speed,
        progress_speed,
      )

    below_lower = observation.q_rad < self.config.joint_lower_rad
    above_upper = observation.q_rad > self.config.joint_upper_rad
    toward_lower = (
      observation.q_rad <= self.config.joint_lower_rad + self.config.joint_limit_margin_rad
    ) & (observation.qd_command_rad_s < 0.0)
    toward_upper = (
      observation.q_rad >= self.config.joint_upper_rad - self.config.joint_limit_margin_rad
    ) & (observation.qd_command_rad_s > 0.0)
    limit_mask = below_lower | above_upper | toward_lower | toward_upper
    if np.any(limit_mask):
      self.reset()
      return self._decision(
        GuardReason.JOINT_LIMIT,
        GuardSeverity.HARD_STOP,
        observation,
        command_speed,
        progress_speed,
        joint_indices=tuple(int(index) for index in np.flatnonzero(limit_mask)),
      )

    attempted_motion = command_speed >= self.config.command_speed_threshold_rad_s
    stalled = (
      attempted_motion
      and progress_speed < self.config.actual_progress_threshold_rad_s
    )
    if stalled:
      self._stall_duration_s += self.config.dt_s
    else:
      self.reset()

    if self._stall_duration_s + 1e-12 >= self.config.stall_time_s:
      quiet_tip = max_tip_force <= self.config.tip_quiet_force_n
      reason = (
        GuardReason.SUSPECTED_OBJECT_BLOCKAGE
        if quiet_tip
        else GuardReason.NO_PROGRESS
      )
      return self._decision(
        reason,
        GuardSeverity.SOFT_STOP,
        observation,
        command_speed,
        progress_speed,
      )

    return self._decision(
      GuardReason.NONE,
      GuardSeverity.SAFE,
      observation,
      command_speed,
      progress_speed,
    )

  def _decision(
    self,
    reason: GuardReason,
    severity: GuardSeverity,
    observation: GuardObservation,
    command_speed: float,
    progress_speed: float,
    joint_indices: tuple[int, ...] = (),
  ) -> GuardDecision:
    evidence = GuardEvidence(
      stall_duration_s=self._stall_duration_s,
      commanded_speed_rad_s=command_speed,
      progress_speed_rad_s=progress_speed,
      max_tip_force_n=(
        float(np.max(observation.fingertip_forces_n))
        if len(observation.fingertip_forces_n)
        else 0.0
      ),
      min_self_collision_distance_m=observation.min_self_collision_distance_m,
      joint_indices=joint_indices,
    )
    executor_state = (
      ExecutorState.RUNNING
      if severity is GuardSeverity.SAFE
      else ExecutorState.BLOCKED
    )
    return GuardDecision(reason, severity, executor_state, evidence)
