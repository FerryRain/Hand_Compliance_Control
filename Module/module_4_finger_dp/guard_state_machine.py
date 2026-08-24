"""DP-specific hard-release executor layered above M03 detection."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import ArrayLike, NDArray

from Module.module_4_finger_dp.contracts import NUM_FINGERS, NUM_FINGER_JOINTS


def _vector(value: ArrayLike, name: str, length: int) -> NDArray[np.float64]:
  result = np.asarray(value, dtype=np.float64)
  if result.shape != (length,) or not np.all(np.isfinite(result)):
    raise ValueError(f"{name} must be finite with shape ({length},)")
  return np.array(result, dtype=np.float64, copy=True)


class DPGuardState(str, Enum):
  INITIALIZE = "INITIALIZE"
  BUFFER_FILL = "BUFFER_FILL"
  DP_ACTIVE = "DP_ACTIVE"
  SOFT_RECOVERY = "SOFT_RECOVERY"
  HARD_RELEASE = "HARD_RELEASE"
  SAFE_HOLD = "SAFE_HOLD"
  BUFFER_RESET = "BUFFER_RESET"
  ABORTED = "ABORTED"


@dataclass(frozen=True, slots=True)
class DPGuardConfig:
  joint_lower_rad: ArrayLike
  joint_upper_rad: ArrayLike
  dt_s: float = 0.002
  soft_force_n: float = 6.0
  hard_force_n: float = 8.0
  recover_force_n: float = 2.5
  stable_time_s: float = 0.10
  hard_timeout_s: float = 0.50
  soft_dp_authority_scale: float = 0.25
  soft_wrist_velocity_scale: float = 0.25
  release_compression_step: float = 0.0005
  release_damping: float = 1e-8
  max_abs_release_delta_rad: float = 0.010
  joint_margin_rad: float = 0.02

  def __post_init__(self) -> None:
    lower = _vector(self.joint_lower_rad, "joint_lower_rad", NUM_FINGER_JOINTS)
    upper = _vector(self.joint_upper_rad, "joint_upper_rad", NUM_FINGER_JOINTS)
    if np.any(lower >= upper):
      raise ValueError("joint bounds must satisfy lower < upper")
    object.__setattr__(self, "joint_lower_rad", lower)
    object.__setattr__(self, "joint_upper_rad", upper)
    positives = {
      "dt_s": self.dt_s,
      "soft_force_n": self.soft_force_n,
      "hard_force_n": self.hard_force_n,
      "recover_force_n": self.recover_force_n,
      "stable_time_s": self.stable_time_s,
      "hard_timeout_s": self.hard_timeout_s,
      "release_compression_step": self.release_compression_step,
      "max_abs_release_delta_rad": self.max_abs_release_delta_rad,
      "joint_margin_rad": self.joint_margin_rad,
    }
    for name, value in positives.items():
      if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    if not self.recover_force_n < self.soft_force_n < self.hard_force_n:
      raise ValueError("force thresholds must satisfy recover < soft < hard")
    if self.release_damping < 0.0 or not np.isfinite(self.release_damping):
      raise ValueError("release_damping must be finite and non-negative")
    for name in ("soft_dp_authority_scale", "soft_wrist_velocity_scale"):
      value = float(getattr(self, name))
      if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be in (0,1)")


@dataclass(frozen=True, slots=True)
class DPGuardOutput:
  state: DPGuardState
  dp_authority_scale: float
  wrist_velocity_scale: float
  exploration_enabled: bool
  override_delta_rad: NDArray[np.float64] | None
  affected_fingers: tuple[int, ...]
  reset_history: bool
  terminate_episode: bool
  reason: str


class DPRuntimeGuardExecutor:
  """Turn guard evidence into recovery, release, hold and buffer-reset actions.

  ``signed_compression_jacobian[i]`` is defined so positive row motion raises
  compression.  Hard release therefore always solves ``J_s delta_q < 0`` and
  never depends on an outward-normal sign convention.
  """

  def __init__(self, config: DPGuardConfig) -> None:
    self.config = config
    self.reset()

  def reset(self) -> None:
    self.state = DPGuardState.INITIALIZE
    self._stable_elapsed_s = 0.0
    self._hard_elapsed_s = 0.0

  def _release_delta(
    self,
    current_q: NDArray[np.float64],
    signed_compression_jacobian: NDArray[np.float64],
    affected: NDArray[np.int64],
  ) -> NDArray[np.float64]:
    if len(affected) == 0:
      return np.zeros(NUM_FINGER_JOINTS, dtype=np.float64)
    jacobian = signed_compression_jacobian[affected]
    target = -np.full(len(affected), self.config.release_compression_step)
    gram = jacobian @ jacobian.T + self.config.release_damping * np.eye(len(affected))
    try:
      delta = jacobian.T @ np.linalg.solve(gram, target)
    except np.linalg.LinAlgError:
      delta = jacobian.T @ np.linalg.pinv(gram) @ target
    delta = np.clip(
      delta,
      -self.config.max_abs_release_delta_rad,
      self.config.max_abs_release_delta_rad,
    )
    command = np.clip(
      current_q + delta,
      self.config.joint_lower_rad + self.config.joint_margin_rad,
      self.config.joint_upper_rad - self.config.joint_margin_rad,
    )
    return command - current_q

  def _output(
    self,
    *,
    authority: float,
    wrist_scale: float,
    exploration: bool,
    override: NDArray[np.float64] | None = None,
    affected: tuple[int, ...] = (),
    reset_history: bool = False,
    terminate: bool = False,
    reason: str,
  ) -> DPGuardOutput:
    if override is not None:
      override = np.array(override, dtype=np.float64, copy=True)
      override.setflags(write=False)
    return DPGuardOutput(
      state=self.state,
      dp_authority_scale=float(authority),
      wrist_velocity_scale=float(wrist_scale),
      exploration_enabled=bool(exploration),
      override_delta_rad=override,
      affected_fingers=affected,
      reset_history=reset_history,
      terminate_episode=terminate,
      reason=reason,
    )

  def step(
    self,
    *,
    fingertip_force_n: ArrayLike,
    force_valid_mask: ArrayLike,
    history_ready: bool,
    current_q_rad: ArrayLike,
    signed_compression_jacobian: ArrayLike,
    fatal_error: bool = False,
  ) -> DPGuardOutput:
    forces = _vector(fingertip_force_n, "fingertip_force_n", NUM_FINGERS)
    if np.any(forces < 0.0):
      raise ValueError("fingertip_force_n must contain non-negative magnitudes")
    valid = np.asarray(force_valid_mask, dtype=np.bool_)
    if valid.shape != (NUM_FINGERS,):
      raise ValueError("force_valid_mask must have shape (4,)")
    current_q = _vector(current_q_rad, "current_q_rad", NUM_FINGER_JOINTS)
    jacobian = np.asarray(signed_compression_jacobian, dtype=np.float64)
    if jacobian.shape != (NUM_FINGERS, NUM_FINGER_JOINTS) or not np.all(np.isfinite(jacobian)):
      raise ValueError("signed_compression_jacobian must be finite with shape (4,16)")

    if fatal_error:
      self.state = DPGuardState.ABORTED
      return self._output(
        authority=0.0,
        wrist_scale=0.0,
        exploration=False,
        terminate=True,
        reason="FATAL_RUNTIME_ERROR",
      )
    if self.state is DPGuardState.ABORTED:
      return self._output(
        authority=0.0,
        wrist_scale=0.0,
        exploration=False,
        terminate=True,
        reason="ALREADY_ABORTED",
      )
    if not np.all(valid):
      self.state = DPGuardState.SAFE_HOLD
      self._stable_elapsed_s = 0.0
      return self._output(
        authority=0.0,
        wrist_scale=0.0,
        exploration=False,
        reason="FORCE_SENSOR_INVALID",
      )

    hard = forces >= self.config.hard_force_n
    soft = forces >= self.config.soft_force_n
    recovered = forces < self.config.recover_force_n

    if self.state is DPGuardState.INITIALIZE:
      self.state = DPGuardState.BUFFER_FILL
      return self._output(
        authority=0.0,
        wrist_scale=0.0,
        exploration=False,
        reason="INITIALIZE_COMPLETE",
      )

    if np.any(hard) and self.state not in {DPGuardState.HARD_RELEASE, DPGuardState.ABORTED}:
      self.state = DPGuardState.HARD_RELEASE
      self._hard_elapsed_s = 0.0
      self._stable_elapsed_s = 0.0

    if self.state is DPGuardState.HARD_RELEASE:
      self._hard_elapsed_s += self.config.dt_s
      if np.all(recovered):
        self.state = DPGuardState.SAFE_HOLD
        self._stable_elapsed_s = 0.0
        return self._output(
          authority=0.0,
          wrist_scale=0.0,
          exploration=False,
          reason="HARD_RELEASE_COMPLETE",
        )
      if self._hard_elapsed_s > self.config.hard_timeout_s:
        self.state = DPGuardState.ABORTED
        return self._output(
          authority=0.0,
          wrist_scale=0.0,
          exploration=False,
          terminate=True,
          reason="PERSISTENT_HARD_FORCE",
        )
      affected_array = np.flatnonzero(~recovered)
      release = self._release_delta(current_q, jacobian, affected_array)
      return self._output(
        authority=0.0,
        wrist_scale=0.0,
        exploration=False,
        override=release,
        affected=tuple(int(index) for index in affected_array),
        reason="DETERMINISTIC_COMPRESSION_RELEASE",
      )

    if self.state is DPGuardState.SAFE_HOLD:
      if np.all(recovered):
        self._stable_elapsed_s += self.config.dt_s
      else:
        self._stable_elapsed_s = 0.0
      if self._stable_elapsed_s >= self.config.stable_time_s:
        self.state = DPGuardState.BUFFER_RESET
        return self._output(
          authority=0.0,
          wrist_scale=0.0,
          exploration=False,
          reset_history=True,
          reason="RECOVERY_STABLE_RESET_HISTORY",
        )
      return self._output(
        authority=0.0,
        wrist_scale=0.0,
        exploration=False,
        reason="WAITING_FOR_STABLE_RECOVERY",
      )

    if self.state is DPGuardState.BUFFER_RESET:
      self.state = DPGuardState.BUFFER_FILL
      return self._output(
        authority=0.0,
        wrist_scale=0.0,
        exploration=False,
        reset_history=True,
        reason="BUFFER_RESET_COMPLETE",
      )

    if self.state is DPGuardState.BUFFER_FILL:
      if history_ready and not np.any(soft):
        self.state = DPGuardState.DP_ACTIVE
        return self._output(
          authority=1.0,
          wrist_scale=1.0,
          exploration=True,
          reason="FORCE_HISTORY_READY",
        )
      return self._output(
        authority=0.0,
        wrist_scale=0.0,
        exploration=False,
        reason="FILLING_FORCE_HISTORY",
      )

    if self.state is DPGuardState.DP_ACTIVE and np.any(soft):
      self.state = DPGuardState.SOFT_RECOVERY
      self._stable_elapsed_s = 0.0

    if self.state is DPGuardState.SOFT_RECOVERY:
      if not np.any(soft):
        self._stable_elapsed_s += self.config.dt_s
      else:
        self._stable_elapsed_s = 0.0
      if self._stable_elapsed_s >= self.config.stable_time_s:
        self.state = DPGuardState.DP_ACTIVE
        return self._output(
          authority=1.0,
          wrist_scale=1.0,
          exploration=True,
          reason="SOFT_RECOVERY_COMPLETE",
        )
      return self._output(
        authority=self.config.soft_dp_authority_scale,
        wrist_scale=self.config.soft_wrist_velocity_scale,
        exploration=False,
        reason="SOFT_FORCE_RECOVERY",
      )

    return self._output(
      authority=1.0,
      wrist_scale=1.0,
      exploration=True,
      reason="DP_AUTHORITY_ACTIVE",
    )
