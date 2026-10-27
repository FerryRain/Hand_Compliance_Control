"""Controller-independent fingertip over-force recovery executor.

M03 owns the deterministic safety authority shared by analytical Finger MCC
and learned Finger DP. A controller may propose a command, but this executor
can scale, hold, or replace it with a signed-compression release command.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import ArrayLike, NDArray


NUM_FINGERS = 4
NUM_FINGER_JOINTS = 16


def _vector(value: ArrayLike, name: str, length: int) -> NDArray[np.float64]:
  result = np.asarray(value, dtype=np.float64)
  if result.shape != (length,) or not np.all(np.isfinite(result)):
    raise ValueError(f"{name} must be finite with shape ({length},)")
  return np.array(result, dtype=np.float64, copy=True)


class ForceSafetyState(str, Enum):
  INITIALIZE = "INITIALIZE"
  BUFFER_FILL = "BUFFER_FILL"
  ACTIVE = "ACTIVE"
  DP_ACTIVE = "ACTIVE"
  SOFT_RECOVERY = "SOFT_RECOVERY"
  REENTRY_RAMP = "REENTRY_RAMP"
  HARD_RELEASE = "HARD_RELEASE"
  SAFE_HOLD = "SAFE_HOLD"
  BUFFER_RESET = "BUFFER_RESET"
  ABORTED = "ABORTED"


@dataclass(frozen=True, slots=True)
class ForceSafetyConfig:
  joint_lower_rad: ArrayLike
  joint_upper_rad: ArrayLike
  dt_s: float = 0.002
  soft_force_n: float = 6.0
  hard_force_n: float = 8.0
  recover_force_n: float = 2.5
  stable_time_s: float = 0.10
  reentry_ramp_time_s: float = 0.10
  hard_timeout_s: float = 0.50
  soft_finger_authority_scale: float = 0.25
  soft_wrist_velocity_scale: float = 0.25
  soft_release_gain: float = 0.25
  release_compression_step: float = 0.0005
  release_damping: float = 1e-8
  max_abs_release_delta_rad: float = 0.010
  joint_margin_rad: float = 0.02
  rapid_loading_rate_n_s: float = 500.0
  rapid_loading_min_force_n: float = 3.0

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
      "reentry_ramp_time_s": self.reentry_ramp_time_s,
      "release_compression_step": self.release_compression_step,
      "max_abs_release_delta_rad": self.max_abs_release_delta_rad,
      "joint_margin_rad": self.joint_margin_rad,
      "rapid_loading_rate_n_s": self.rapid_loading_rate_n_s,
      "rapid_loading_min_force_n": self.rapid_loading_min_force_n,
    }
    for name, value in positives.items():
      if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    if not self.recover_force_n < self.soft_force_n < self.hard_force_n:
      raise ValueError("force thresholds must satisfy recover < soft < hard")
    if self.release_damping < 0.0 or not np.isfinite(self.release_damping):
      raise ValueError("release_damping must be finite and non-negative")
    for name in ("soft_finger_authority_scale", "soft_wrist_velocity_scale"):
      value = float(getattr(self, name))
      if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be in (0,1)")
    if not 0.0 <= self.soft_release_gain <= 1.0:
      raise ValueError("soft_release_gain must be in [0,1]")

  @property
  def soft_dp_authority_scale(self) -> float:
    return self.soft_finger_authority_scale


@dataclass(frozen=True, slots=True)
class ForceSafetyOutput:
  state: ForceSafetyState
  finger_authority_scale: float
  wrist_velocity_scale: float
  exploration_enabled: bool
  override_delta_rad: NDArray[np.float64] | None
  affected_fingers: tuple[int, ...]
  reset_history: bool
  terminate_episode: bool
  reason: str

  @property
  def dp_authority_scale(self) -> float:
    return self.finger_authority_scale


class ForceSafetyExecutor:
  """Convert force evidence into bounded recovery, release, and hold actions.

  ``signed_compression_jacobian[i]`` is positive when finger motion raises
  compression. Hard release always solves ``J_s delta_q < 0``.
  """

  def __init__(self, config: ForceSafetyConfig) -> None:
    self.config = config
    self.reset()

  def reset(self) -> None:
    self.state = ForceSafetyState.INITIALIZE
    self._stable_elapsed_s = 0.0
    self._hard_elapsed_s = 0.0
    self._reentry_elapsed_s = 0.0
    self._reentry_start_finger_scale = 0.0
    self._reentry_start_wrist_scale = 0.0
    self._previous_force_n = np.zeros(NUM_FINGERS, dtype=np.float64)
    self._force_derivative_ready = False

  def _begin_reentry(
    self,
    *,
    finger_scale: float,
    wrist_scale: float,
    reason: str,
  ) -> ForceSafetyOutput:
    """Start authority restoration without a zero-to-full command edge."""

    self.state = ForceSafetyState.REENTRY_RAMP
    self._reentry_elapsed_s = 0.0
    self._reentry_start_finger_scale = float(finger_scale)
    self._reentry_start_wrist_scale = float(wrist_scale)
    return self._output(
      authority=finger_scale,
      wrist_scale=wrist_scale,
      exploration=False,
      reason=reason,
    )

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
    delta = np.clip(delta, -self.config.max_abs_release_delta_rad, self.config.max_abs_release_delta_rad)
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
  ) -> ForceSafetyOutput:
    if override is not None:
      override = np.array(override, dtype=np.float64, copy=True)
      override.setflags(write=False)
    return ForceSafetyOutput(
      state=self.state,
      finger_authority_scale=float(authority),
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
  ) -> ForceSafetyOutput:
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
      self.state = ForceSafetyState.ABORTED
      return self._output(authority=0.0, wrist_scale=0.0, exploration=False, terminate=True, reason="FATAL_RUNTIME_ERROR")
    if self.state is ForceSafetyState.ABORTED:
      return self._output(authority=0.0, wrist_scale=0.0, exploration=False, terminate=True, reason="ALREADY_ABORTED")
    if not np.all(valid):
      self.state = ForceSafetyState.SAFE_HOLD
      self._stable_elapsed_s = 0.0
      return self._output(authority=0.0, wrist_scale=0.0, exploration=False, reason="FORCE_SENSOR_INVALID")

    hard = forces >= self.config.hard_force_n
    soft = forces >= self.config.soft_force_n
    recovered = forces < self.config.recover_force_n
    if self._force_derivative_ready:
      loading_rate = (forces - self._previous_force_n) / self.config.dt_s
      rapid_loading = (
        (forces >= self.config.rapid_loading_min_force_n)
        & (loading_rate >= self.config.rapid_loading_rate_n_s)
      )
    else:
      rapid_loading = np.zeros(NUM_FINGERS, dtype=np.bool_)
    self._previous_force_n[:] = forces
    self._force_derivative_ready = True

    if self.state is ForceSafetyState.INITIALIZE:
      self.state = ForceSafetyState.BUFFER_FILL
      return self._output(authority=0.0, wrist_scale=0.0, exploration=False, reason="INITIALIZE_COMPLETE")

    if np.any(hard) and self.state not in {ForceSafetyState.HARD_RELEASE, ForceSafetyState.ABORTED}:
      self.state = ForceSafetyState.HARD_RELEASE
      self._hard_elapsed_s = 0.0
      self._stable_elapsed_s = 0.0
    elif (
      np.any(rapid_loading)
      and self.state in {ForceSafetyState.ACTIVE, ForceSafetyState.REENTRY_RAMP}
    ):
      self.state = ForceSafetyState.SOFT_RECOVERY
      self._stable_elapsed_s = 0.0

    if self.state is ForceSafetyState.HARD_RELEASE:
      self._hard_elapsed_s += self.config.dt_s
      if np.all(recovered):
        self.state = ForceSafetyState.SAFE_HOLD
        self._stable_elapsed_s = 0.0
        return self._output(authority=0.0, wrist_scale=0.0, exploration=False, reason="HARD_RELEASE_COMPLETE")
      if self._hard_elapsed_s > self.config.hard_timeout_s:
        self.state = ForceSafetyState.ABORTED
        return self._output(authority=0.0, wrist_scale=0.0, exploration=False, terminate=True, reason="PERSISTENT_HARD_FORCE")
      affected_array = np.flatnonzero(~recovered)
      return self._output(
        authority=0.0,
        wrist_scale=0.0,
        exploration=False,
        override=self._release_delta(current_q, jacobian, affected_array),
        affected=tuple(int(index) for index in affected_array),
        reason="DETERMINISTIC_COMPRESSION_RELEASE",
      )

    if self.state is ForceSafetyState.SAFE_HOLD:
      self._stable_elapsed_s = self._stable_elapsed_s + self.config.dt_s if np.all(recovered) else 0.0
      if self._stable_elapsed_s >= self.config.stable_time_s:
        self.state = ForceSafetyState.BUFFER_RESET
        return self._output(authority=0.0, wrist_scale=0.0, exploration=False, reset_history=True, reason="RECOVERY_STABLE_RESET_HISTORY")
      return self._output(authority=0.0, wrist_scale=0.0, exploration=False, reason="WAITING_FOR_STABLE_RECOVERY")

    if self.state is ForceSafetyState.BUFFER_RESET:
      self.state = ForceSafetyState.BUFFER_FILL
      return self._output(authority=0.0, wrist_scale=0.0, exploration=False, reset_history=True, reason="BUFFER_RESET_COMPLETE")

    if self.state is ForceSafetyState.BUFFER_FILL:
      if history_ready and not np.any(soft):
        return self._begin_reentry(
          finger_scale=0.0,
          wrist_scale=0.0,
          reason="FORCE_HISTORY_READY_BEGIN_REENTRY",
        )
      return self._output(authority=0.0, wrist_scale=0.0, exploration=False, reason="FILLING_FORCE_HISTORY")

    if self.state is ForceSafetyState.ACTIVE and np.any(soft):
      self.state = ForceSafetyState.SOFT_RECOVERY
      self._stable_elapsed_s = 0.0

    if self.state is ForceSafetyState.SOFT_RECOVERY:
      self._stable_elapsed_s = (
        self._stable_elapsed_s + self.config.dt_s
        if not np.any(soft)
        else 0.0
      )
      if self._stable_elapsed_s >= self.config.stable_time_s:
        return self._begin_reentry(
          finger_scale=self.config.soft_finger_authority_scale,
          wrist_scale=self.config.soft_wrist_velocity_scale,
          reason="SOFT_RECOVERY_COMPLETE_BEGIN_REENTRY",
        )
      if np.any(rapid_loading):
        # A fast rise on one pad often precedes an inter-finger load transfer.
        # Damping only the triggering pad can push the next physics tick's
        # resultant load into another contact, so include every currently
        # loaded pad in this short bounded soft-release action.
        affected_array = np.flatnonzero(forces > 0.0)
      else:
        affected_array = np.flatnonzero(soft)
      soft_override = (
        self.config.soft_release_gain
        * self._release_delta(current_q, jacobian, affected_array)
        if len(affected_array) and self.config.soft_release_gain > 0.0
        else None
      )
      return self._output(
        authority=self.config.soft_finger_authority_scale,
        wrist_scale=self.config.soft_wrist_velocity_scale,
        exploration=False,
        override=soft_override,
        affected=tuple(int(index) for index in affected_array),
        reason="SOFT_FORCE_RECOVERY",
      )

    if self.state is ForceSafetyState.REENTRY_RAMP:
      if np.any(soft | rapid_loading):
        self.state = ForceSafetyState.SOFT_RECOVERY
        self._stable_elapsed_s = 0.0
        return self._output(
          authority=self.config.soft_finger_authority_scale,
          wrist_scale=self.config.soft_wrist_velocity_scale,
          exploration=False,
          reason="REENTRY_INTERRUPTED_BY_SOFT_FORCE",
        )
      self._reentry_elapsed_s += self.config.dt_s
      alpha = min(1.0, self._reentry_elapsed_s / self.config.reentry_ramp_time_s)
      finger_scale = self._reentry_start_finger_scale + alpha * (
        1.0 - self._reentry_start_finger_scale
      )
      wrist_scale = self._reentry_start_wrist_scale + alpha * (
        1.0 - self._reentry_start_wrist_scale
      )
      if alpha >= 1.0:
        self.state = ForceSafetyState.ACTIVE
      return self._output(
        authority=finger_scale,
        wrist_scale=wrist_scale,
        exploration=alpha >= 1.0,
        reason="BOUNDED_AUTHORITY_REENTRY",
      )

    return self._output(authority=1.0, wrist_scale=1.0, exploration=True, reason="CONTROLLER_AUTHORITY_ACTIVE")
