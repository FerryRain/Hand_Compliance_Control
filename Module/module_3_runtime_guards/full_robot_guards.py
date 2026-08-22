"""M03-FR3 grouped safety guards for the 23-DoF full robot."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from Module.common import ExecutorState


def _vector(value: ArrayLike, name: str, length: int) -> NDArray[np.float64]:
  result = np.asarray(value, dtype=np.float64)
  if result.shape != (length,) or not np.all(np.isfinite(result)):
    raise ValueError(f"{name} must be finite with shape ({length},)")
  return np.array(result, dtype=np.float64, copy=True)


def _boolean(value: ArrayLike, name: str, length: int) -> NDArray[np.bool_]:
  result = np.asarray(value, dtype=np.bool_)
  if result.shape != (length,):
    raise ValueError(f"{name} must have shape ({length},)")
  return np.array(result, dtype=np.bool_, copy=True)


class FullRobotGuardReason(str, Enum):
  NONE = "NONE"
  SENSOR_INVALID = "SENSOR_INVALID"
  TIP_OVERFORCE = "TIP_OVERFORCE"
  WRIST_WRENCH_LIMIT = "WRIST_WRENCH_LIMIT"
  ARM_EXTERNAL_TORQUE_LIMIT = "ARM_EXTERNAL_TORQUE_LIMIT"
  ROBOT_COLLISION = "ROBOT_COLLISION"
  ARM_JOINT_LIMIT = "ARM_JOINT_LIMIT"
  FINGER_JOINT_LIMIT = "FINGER_JOINT_LIMIT"
  CONTROLLER_LIMIT = "CONTROLLER_LIMIT"
  ARM_ACTUATOR_SATURATION = "ARM_ACTUATOR_SATURATION"
  FINGER_ACTUATOR_SATURATION = "FINGER_ACTUATOR_SATURATION"
  ARM_NO_PROGRESS = "ARM_NO_PROGRESS"
  FINGER_NO_PROGRESS = "FINGER_NO_PROGRESS"
  SUSPECTED_FINGER_BLOCKAGE = "SUSPECTED_FINGER_BLOCKAGE"


class HoldScope(str, Enum):
  NONE = "NONE"
  FINGER_LOCAL = "FINGER_LOCAL"
  GLOBAL_SAFE_HOLD = "GLOBAL_SAFE_HOLD"


@dataclass(frozen=True, slots=True)
class FullRobotGuardConfig:
  arm_joint_lower_rad: ArrayLike
  arm_joint_upper_rad: ArrayLike
  finger_joint_lower_rad: ArrayLike
  finger_joint_upper_rad: ArrayLike
  dt_s: float = 0.002
  max_tip_force_n: float = 8.0
  max_abs_wrist_wrench: ArrayLike = (80.0, 80.0, 80.0, 8.0, 8.0, 8.0)
  max_abs_arm_external_torque_nm: ArrayLike = (40.0, 40.0, 40.0, 30.0, 10.0, 10.0, 10.0)
  joint_limit_margin_rad: float = 0.02
  min_robot_collision_distance_m: float = 0.003
  arm_command_speed_threshold_rad_s: float = 0.03
  finger_command_speed_threshold_rad_s: float = 0.05
  progress_threshold_rad_s: float = 0.004
  stall_time_s: float = 0.15
  saturation_time_s: float = 0.08
  sensor_max_age_s: float = 0.02
  max_wrist_offset: ArrayLike = (0.02, 0.02, 0.02, 0.15, 0.15, 0.15)
  max_finger_offset_m: float = 0.015

  def __post_init__(self) -> None:
    arm_lower = _vector(self.arm_joint_lower_rad, "arm_joint_lower_rad", 7)
    arm_upper = _vector(self.arm_joint_upper_rad, "arm_joint_upper_rad", 7)
    finger_lower = _vector(self.finger_joint_lower_rad, "finger_joint_lower_rad", 16)
    finger_upper = _vector(self.finger_joint_upper_rad, "finger_joint_upper_rad", 16)
    if np.any(arm_lower >= arm_upper) or np.any(finger_lower >= finger_upper):
      raise ValueError("all joint bounds must satisfy lower < upper")
    arrays = {
      "max_abs_wrist_wrench": _vector(self.max_abs_wrist_wrench, "max_abs_wrist_wrench", 6),
      "max_abs_arm_external_torque_nm": _vector(
        self.max_abs_arm_external_torque_nm,
        "max_abs_arm_external_torque_nm",
        7,
      ),
      "max_wrist_offset": _vector(self.max_wrist_offset, "max_wrist_offset", 6),
    }
    positives = {
      "dt_s": self.dt_s,
      "max_tip_force_n": self.max_tip_force_n,
      "joint_limit_margin_rad": self.joint_limit_margin_rad,
      "min_robot_collision_distance_m": self.min_robot_collision_distance_m,
      "arm_command_speed_threshold_rad_s": self.arm_command_speed_threshold_rad_s,
      "finger_command_speed_threshold_rad_s": self.finger_command_speed_threshold_rad_s,
      "progress_threshold_rad_s": self.progress_threshold_rad_s,
      "stall_time_s": self.stall_time_s,
      "saturation_time_s": self.saturation_time_s,
      "sensor_max_age_s": self.sensor_max_age_s,
      "max_finger_offset_m": self.max_finger_offset_m,
    }
    for name, value in positives.items():
      if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    if any(np.any(value <= 0.0) for value in arrays.values()):
      raise ValueError("wrench, torque and offset limits must be positive")
    object.__setattr__(self, "arm_joint_lower_rad", arm_lower)
    object.__setattr__(self, "arm_joint_upper_rad", arm_upper)
    object.__setattr__(self, "finger_joint_lower_rad", finger_lower)
    object.__setattr__(self, "finger_joint_upper_rad", finger_upper)
    for name, value in arrays.items():
      object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class FullRobotGuardObservation:
  arm_q_rad: ArrayLike
  arm_qd_command_rad_s: ArrayLike
  arm_qd_actual_rad_s: ArrayLike
  finger_q_rad: ArrayLike
  finger_qd_command_rad_s: ArrayLike
  finger_qd_actual_rad_s: ArrayLike
  fingertip_forces_n: ArrayLike
  wrist_wrench: ArrayLike
  arm_external_torque_nm: ArrayLike
  wrist_compliance_offset: ArrayLike
  finger_compliance_offsets_m: ArrayLike
  arm_actuator_saturated: ArrayLike = (False,) * 7
  finger_actuator_saturated: ArrayLike = (False,) * 16
  min_robot_collision_distance_m: float | None = None
  sensor_validity: Mapping[str, bool] = None  # type: ignore[assignment]
  sensor_age_s: float = 0.0

  def __post_init__(self) -> None:
    lengths = {
      "arm_q_rad": 7,
      "arm_qd_command_rad_s": 7,
      "arm_qd_actual_rad_s": 7,
      "finger_q_rad": 16,
      "finger_qd_command_rad_s": 16,
      "finger_qd_actual_rad_s": 16,
      "fingertip_forces_n": 4,
      "wrist_wrench": 6,
      "arm_external_torque_nm": 7,
      "wrist_compliance_offset": 6,
      "finger_compliance_offsets_m": 4,
    }
    for name, length in lengths.items():
      object.__setattr__(self, name, _vector(getattr(self, name), name, length))
    if np.any(self.fingertip_forces_n < 0.0):
      raise ValueError("fingertip forces must be non-negative")
    object.__setattr__(
      self,
      "arm_actuator_saturated",
      _boolean(self.arm_actuator_saturated, "arm_actuator_saturated", 7),
    )
    object.__setattr__(
      self,
      "finger_actuator_saturated",
      _boolean(self.finger_actuator_saturated, "finger_actuator_saturated", 16),
    )
    if self.min_robot_collision_distance_m is not None:
      distance = float(self.min_robot_collision_distance_m)
      if not np.isfinite(distance):
        raise ValueError("min_robot_collision_distance_m must be finite")
      object.__setattr__(self, "min_robot_collision_distance_m", distance)
    if not np.isfinite(self.sensor_age_s) or self.sensor_age_s < 0.0:
      raise ValueError("sensor_age_s must be finite and non-negative")
    validity = self.sensor_validity or {}
    object.__setattr__(self, "sensor_validity", {str(k): bool(v) for k, v in validity.items()})


@dataclass(frozen=True, slots=True)
class FullRobotGuardDecision:
  reason: FullRobotGuardReason
  hold_scope: HoldScope
  executor_state: ExecutorState
  affected_indices: tuple[int, ...]
  evidence: Mapping[str, Any]

  @property
  def should_stop(self) -> bool:
    return self.hold_scope is not HoldScope.NONE


class FullRobotRuntimeGuards:
  """Evaluate arm and each four-joint finger without norm masking."""

  def __init__(self, config: FullRobotGuardConfig) -> None:
    self.config = config
    self._arm_stall_s = 0.0
    self._finger_stall_s = np.zeros(4, dtype=np.float64)
    self._arm_saturation_s = 0.0
    self._finger_saturation_s = np.zeros(4, dtype=np.float64)

  def reset(self) -> None:
    self._arm_stall_s = 0.0
    self._finger_stall_s[:] = 0.0
    self._arm_saturation_s = 0.0
    self._finger_saturation_s[:] = 0.0

  @staticmethod
  def _progress(command: NDArray[np.float64], actual: NDArray[np.float64]) -> tuple[float, float]:
    speed = float(np.linalg.norm(command))
    progress = float(np.dot(actual, command) / speed) if speed > 0.0 else 0.0
    return speed, progress

  def _decision(
    self,
    reason: FullRobotGuardReason,
    scope: HoldScope,
    affected: tuple[int, ...] = (),
    **evidence: Any,
  ) -> FullRobotGuardDecision:
    return FullRobotGuardDecision(
      reason=reason,
      hold_scope=scope,
      executor_state=(ExecutorState.RUNNING if scope is HoldScope.NONE else ExecutorState.BLOCKED),
      affected_indices=affected,
      evidence=evidence,
    )

  def evaluate(self, observation: FullRobotGuardObservation) -> FullRobotGuardDecision:
    c = self.config
    invalid_channels = tuple(
      name for name, valid in observation.sensor_validity.items() if not valid
    )
    if invalid_channels or observation.sensor_age_s > c.sensor_max_age_s:
      return self._decision(
        FullRobotGuardReason.SENSOR_INVALID,
        HoldScope.GLOBAL_SAFE_HOLD,
        invalid_channels=invalid_channels,
        sensor_age_s=observation.sensor_age_s,
      )
    maximum_force = float(np.max(observation.fingertip_forces_n))
    if maximum_force > c.max_tip_force_n:
      finger = int(np.argmax(observation.fingertip_forces_n))
      return self._decision(
        FullRobotGuardReason.TIP_OVERFORCE,
        HoldScope.GLOBAL_SAFE_HOLD,
        (finger,),
        max_tip_force_n=maximum_force,
      )
    wrench_ratio = np.abs(observation.wrist_wrench) / c.max_abs_wrist_wrench
    if np.any(wrench_ratio > 1.0):
      axes = tuple(int(value) for value in np.flatnonzero(wrench_ratio > 1.0))
      return self._decision(
        FullRobotGuardReason.WRIST_WRENCH_LIMIT,
        HoldScope.GLOBAL_SAFE_HOLD,
        axes,
        wrench=observation.wrist_wrench.tolist(),
      )
    torque_ratio = np.abs(observation.arm_external_torque_nm) / c.max_abs_arm_external_torque_nm
    if np.any(torque_ratio > 1.0):
      joints = tuple(int(value) for value in np.flatnonzero(torque_ratio > 1.0))
      return self._decision(
        FullRobotGuardReason.ARM_EXTERNAL_TORQUE_LIMIT,
        HoldScope.GLOBAL_SAFE_HOLD,
        joints,
        external_torque_nm=observation.arm_external_torque_nm.tolist(),
      )
    distance = observation.min_robot_collision_distance_m
    if distance is not None and distance <= c.min_robot_collision_distance_m:
      return self._decision(
        FullRobotGuardReason.ROBOT_COLLISION,
        HoldScope.GLOBAL_SAFE_HOLD,
        collision_distance_m=distance,
      )

    def limit_indices(q: NDArray[np.float64], command: NDArray[np.float64], lower: NDArray[np.float64], upper: NDArray[np.float64]) -> tuple[int, ...]:
      mask = (
        (q < lower)
        | (q > upper)
        | ((q <= lower + c.joint_limit_margin_rad) & (command < 0.0))
        | ((q >= upper - c.joint_limit_margin_rad) & (command > 0.0))
      )
      return tuple(int(value) for value in np.flatnonzero(mask))

    arm_limits = limit_indices(
      observation.arm_q_rad,
      observation.arm_qd_command_rad_s,
      c.arm_joint_lower_rad,
      c.arm_joint_upper_rad,
    )
    if arm_limits:
      return self._decision(
        FullRobotGuardReason.ARM_JOINT_LIMIT,
        HoldScope.GLOBAL_SAFE_HOLD,
        arm_limits,
      )
    finger_limits = limit_indices(
      observation.finger_q_rad,
      observation.finger_qd_command_rad_s,
      c.finger_joint_lower_rad,
      c.finger_joint_upper_rad,
    )
    if finger_limits:
      fingers = tuple(sorted({index // 4 for index in finger_limits}))
      return self._decision(
        FullRobotGuardReason.FINGER_JOINT_LIMIT,
        HoldScope.FINGER_LOCAL,
        fingers,
        joint_indices=finger_limits,
      )
    if np.any(np.abs(observation.wrist_compliance_offset) > c.max_wrist_offset) or np.any(
      np.abs(observation.finger_compliance_offsets_m) > c.max_finger_offset_m
    ):
      return self._decision(
        FullRobotGuardReason.CONTROLLER_LIMIT,
        HoldScope.GLOBAL_SAFE_HOLD,
        wrist_offset=observation.wrist_compliance_offset.tolist(),
        finger_offsets=observation.finger_compliance_offsets_m.tolist(),
      )

    self._arm_saturation_s = (
      self._arm_saturation_s + c.dt_s
      if np.any(observation.arm_actuator_saturated)
      else 0.0
    )
    for finger in range(4):
      saturated = np.any(observation.finger_actuator_saturated[4 * finger:4 * finger + 4])
      self._finger_saturation_s[finger] = self._finger_saturation_s[finger] + c.dt_s if saturated else 0.0
    if self._arm_saturation_s + 1e-12 >= c.saturation_time_s:
      return self._decision(
        FullRobotGuardReason.ARM_ACTUATOR_SATURATION,
        HoldScope.GLOBAL_SAFE_HOLD,
        duration_s=self._arm_saturation_s,
      )
    saturated_fingers = tuple(int(value) for value in np.flatnonzero(self._finger_saturation_s + 1e-12 >= c.saturation_time_s))
    if saturated_fingers:
      return self._decision(
        FullRobotGuardReason.FINGER_ACTUATOR_SATURATION,
        HoldScope.FINGER_LOCAL,
        saturated_fingers,
        durations_s=self._finger_saturation_s.tolist(),
      )

    arm_speed, arm_progress = self._progress(
      observation.arm_qd_command_rad_s,
      observation.arm_qd_actual_rad_s,
    )
    arm_stalled = arm_speed >= c.arm_command_speed_threshold_rad_s and arm_progress < c.progress_threshold_rad_s
    self._arm_stall_s = self._arm_stall_s + c.dt_s if arm_stalled else 0.0
    for finger in range(4):
      indices = slice(4 * finger, 4 * finger + 4)
      speed, progress = self._progress(
        observation.finger_qd_command_rad_s[indices],
        observation.finger_qd_actual_rad_s[indices],
      )
      stalled = speed >= c.finger_command_speed_threshold_rad_s and progress < c.progress_threshold_rad_s
      self._finger_stall_s[finger] = self._finger_stall_s[finger] + c.dt_s if stalled else 0.0
    if self._arm_stall_s + 1e-12 >= c.stall_time_s:
      return self._decision(
        FullRobotGuardReason.ARM_NO_PROGRESS,
        HoldScope.GLOBAL_SAFE_HOLD,
        duration_s=self._arm_stall_s,
        command_speed_rad_s=arm_speed,
        progress_rad_s=arm_progress,
      )
    stalled_fingers = tuple(int(value) for value in np.flatnonzero(self._finger_stall_s + 1e-12 >= c.stall_time_s))
    if stalled_fingers:
      quiet = tuple(
        finger for finger in stalled_fingers if observation.fingertip_forces_n[finger] < 0.1
      )
      return self._decision(
        FullRobotGuardReason.SUSPECTED_FINGER_BLOCKAGE if quiet else FullRobotGuardReason.FINGER_NO_PROGRESS,
        HoldScope.FINGER_LOCAL,
        stalled_fingers,
        quiet_fingers=quiet,
        durations_s=self._finger_stall_s.tolist(),
      )
    return self._decision(
      FullRobotGuardReason.NONE,
      HoldScope.NONE,
      arm_stall_s=self._arm_stall_s,
      finger_stall_s=self._finger_stall_s.tolist(),
    )
