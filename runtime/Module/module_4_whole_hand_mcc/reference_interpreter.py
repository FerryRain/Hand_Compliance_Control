"""Deterministic contact-role interpreter shared by Exp. 2 branches.

The learned or analytical reference source proposes a role. This module owns
the executable role, contact confirmation, force ramps, and minimum-contact
guard. Discrete role semantics therefore stay outside the diffusion head.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from numpy.typing import ArrayLike, NDArray


NUM_FINGERS = 4


class ContactRole(IntEnum):
  KEEP = 0
  RELEASE = 1
  FREE = 2
  MAKE = 3


@dataclass(frozen=True, slots=True)
class RoleInterpreterConfig:
  dt_s: float = 0.002
  minimum_confirmed_contacts: int = 1
  make_force_threshold_n: float = 0.20
  break_force_threshold_n: float = 0.10
  make_confirm_time_s: float = 0.030
  make_dropout_grace_s: float = 0.010
  break_confirm_time_s: float = 0.030
  load_ramp_time_s: float = 0.20
  release_ramp_time_s: float = 0.25
  request_confirm_time_s: float = 0.040
  make_confirmation_motion_scale: float = 0.50

  def __post_init__(self) -> None:
    for name in (
      "dt_s",
      "make_force_threshold_n",
      "break_force_threshold_n",
      "make_confirm_time_s",
      "make_dropout_grace_s",
      "break_confirm_time_s",
      "load_ramp_time_s",
      "release_ramp_time_s",
      "request_confirm_time_s",
    ):
      value = float(getattr(self, name))
      if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    if not 0.0 < self.make_confirmation_motion_scale <= 1.0:
      raise ValueError("make_confirmation_motion_scale must be in (0,1]")
    if self.break_force_threshold_n >= self.make_force_threshold_n:
      raise ValueError("break threshold must be below make threshold")
    if not 1 <= self.minimum_confirmed_contacts <= NUM_FINGERS:
      raise ValueError("minimum_confirmed_contacts must be in [1,4]")


@dataclass(frozen=True, slots=True)
class RoleInterpreterOutput:
  roles: NDArray[np.int64]
  physical_contact_confirmed: NDArray[np.bool_]
  desired_force_n: NDArray[np.float64]
  mcc_enabled: NDArray[np.bool_]
  full_reference_authority: NDArray[np.bool_]
  reference_motion_scale: NDArray[np.float64]
  tangential_scale: NDArray[np.float64]
  transition_approved: NDArray[np.bool_]
  transition_reason: tuple[str, ...]


def _roles(value: ArrayLike) -> NDArray[np.int64]:
  result = np.asarray(value, dtype=np.int64)
  if result.shape != (NUM_FINGERS,):
    raise ValueError("requested_roles must have shape (4,)")
  if np.any((result < int(ContactRole.KEEP)) | (result > int(ContactRole.MAKE))):
    raise ValueError("requested_roles contains an unknown role")
  return np.array(result, copy=True)


def _forces(value: ArrayLike, name: str) -> NDArray[np.float64]:
  result = np.asarray(value, dtype=np.float64)
  if result.shape != (NUM_FINGERS,) or not np.all(np.isfinite(result)):
    raise ValueError(f"{name} must be finite with shape (4,)")
  if np.any(result < 0.0):
    raise ValueError(f"{name} must be non-negative")
  return np.array(result, copy=True)


class ContactRoleInterpreter:
  """Approve role intentions and convert them to executable force semantics."""

  def __init__(self, config: RoleInterpreterConfig | None = None) -> None:
    self.config = config or RoleInterpreterConfig()
    self.reset()

  def reset(self, initial_roles: ArrayLike | None = None) -> None:
    if initial_roles is None:
      self._roles = np.full(NUM_FINGERS, int(ContactRole.MAKE), dtype=np.int64)
    else:
      self._roles = _roles(initial_roles)
    self._make_elapsed = np.zeros(NUM_FINGERS, dtype=np.float64)
    self._break_elapsed = np.zeros(NUM_FINGERS, dtype=np.float64)
    self._make_gap_elapsed = np.zeros(NUM_FINGERS, dtype=np.float64)
    self._role_elapsed = np.zeros(NUM_FINGERS, dtype=np.float64)
    self._release_start_force = np.zeros(NUM_FINGERS, dtype=np.float64)
    self._load_start_force = np.zeros(NUM_FINGERS, dtype=np.float64)
    self._physical_contact_confirmed = np.zeros(NUM_FINGERS, dtype=np.bool_)
    self._pending_request = np.full(NUM_FINGERS, -1, dtype=np.int64)
    self._request_elapsed = np.zeros(NUM_FINGERS, dtype=np.float64)

  @property
  def roles(self) -> NDArray[np.int64]:
    return self._roles.copy()

  def step(
    self,
    *,
    requested_roles: ArrayLike,
    measured_force_n: ArrayLike,
    target_force_n: ArrayLike,
  ) -> RoleInterpreterOutput:
    requested = _roles(requested_roles)
    measured = _forces(measured_force_n, "measured_force_n")
    target = _forces(target_force_n, "target_force_n")
    approved = np.zeros(NUM_FINGERS, dtype=np.bool_)
    reasons = ["UNCHANGED"] * NUM_FINGERS
    self._role_elapsed += self.config.dt_s
    same_request = requested == self._pending_request
    self._request_elapsed = np.where(
      same_request,
      self._request_elapsed + self.config.dt_s,
      self.config.dt_s,
    )
    self._pending_request[:] = requested
    request_confirmed = self._request_elapsed + 1e-12 >= self.config.request_confirm_time_s

    has_contact = measured >= self.config.make_force_threshold_n
    is_clear = measured < self.config.break_force_threshold_n
    self._make_gap_elapsed = np.where(
      has_contact,
      0.0,
      self._make_gap_elapsed + self.config.dt_s,
    )
    self._make_elapsed = np.where(
      has_contact,
      self._make_elapsed + self.config.dt_s,
      np.where(
        self._make_gap_elapsed < self.config.make_dropout_grace_s,
        self._make_elapsed,
        0.0,
      ),
    )
    self._break_elapsed = np.where(
      is_clear,
      self._break_elapsed + self.config.dt_s,
      0.0,
    )

    for index in range(NUM_FINGERS):
      role = ContactRole(int(self._roles[index]))
      request = ContactRole(int(requested[index]))
      if role is ContactRole.MAKE:
        if self._make_elapsed[index] >= self.config.make_confirm_time_s:
          self._roles[index] = int(ContactRole.KEEP)
          self._physical_contact_confirmed[index] = True
          self._load_start_force[index] = min(measured[index], target[index])
          self._role_elapsed[index] = 0.0
          approved[index] = True
          reasons[index] = "MAKE_CONTACT_CONFIRMED"
        elif request is ContactRole.FREE and is_clear[index]:
          self._roles[index] = int(ContactRole.FREE)
          self._physical_contact_confirmed[index] = False
          self._role_elapsed[index] = 0.0
          approved[index] = True
          reasons[index] = "MAKE_CANCELLED_CLEAR"
      elif (
        role is ContactRole.FREE
        and request is ContactRole.MAKE
        and request_confirmed[index]
      ):
        self._roles[index] = int(ContactRole.MAKE)
        self._physical_contact_confirmed[index] = False
        self._load_start_force[index] = 0.0
        self._role_elapsed[index] = 0.0
        self._make_elapsed[index] = 0.0
        self._make_gap_elapsed[index] = 0.0
        approved[index] = True
        reasons[index] = "MAKE_INTENTION_APPROVED"
      elif role is ContactRole.RELEASE:
        if self._break_elapsed[index] >= self.config.break_confirm_time_s:
          self._roles[index] = int(ContactRole.FREE)
          self._physical_contact_confirmed[index] = False
          self._role_elapsed[index] = 0.0
          approved[index] = True
          reasons[index] = "RELEASE_CLEAR_CONFIRMED"

    confirmed_keep = {
      int(index)
      for index in range(NUM_FINGERS)
      if ContactRole(int(self._roles[index])) is ContactRole.KEEP
      and self._physical_contact_confirmed[index]
    }
    for index in range(NUM_FINGERS):
      role = ContactRole(int(self._roles[index]))
      request = ContactRole(int(requested[index]))
      if (
        role is ContactRole.KEEP
        and request is ContactRole.RELEASE
        and request_confirmed[index]
      ):
        remaining = len(confirmed_keep - {index})
        if remaining >= self.config.minimum_confirmed_contacts:
          self._roles[index] = int(ContactRole.RELEASE)
          self._role_elapsed[index] = 0.0
          self._release_start_force[index] = min(measured[index], target[index])
          confirmed_keep.discard(index)
          approved[index] = True
          reasons[index] = "RELEASE_INTENTION_APPROVED"
        else:
          reasons[index] = "RELEASE_DENIED_MINIMUM_CONTACT"

    # Contact execution state is distinct from role intention. A KEEP finger
    # that loses physical contact remains semantically KEEP, but is removed
    # from the actual grasp map until a time-confirmed recontact. This prevents
    # Wrist MCC from assigning missing fingers' resultant load to the few pads
    # that remain in contact.
    for index in range(NUM_FINGERS):
      if ContactRole(int(self._roles[index])) is not ContactRole.KEEP:
        continue
      if (
        self._physical_contact_confirmed[index]
        and self._break_elapsed[index] >= self.config.break_confirm_time_s
      ):
        self._physical_contact_confirmed[index] = False
        self._load_start_force[index] = 0.0
        self._role_elapsed[index] = 0.0
        reasons[index] = "KEEP_CONTACT_LOST_RECOVERY"
      elif (
        not self._physical_contact_confirmed[index]
        and self._make_elapsed[index] >= self.config.make_confirm_time_s
      ):
        self._physical_contact_confirmed[index] = True
        self._load_start_force[index] = min(measured[index], target[index])
        self._role_elapsed[index] = 0.0
        reasons[index] = "KEEP_RECONTACT_CONFIRMED"

    desired = np.zeros(NUM_FINGERS, dtype=np.float64)
    mcc_enabled = np.zeros(NUM_FINGERS, dtype=np.bool_)
    full_authority = np.zeros(NUM_FINGERS, dtype=np.bool_)
    tangential_scale = np.ones(NUM_FINGERS, dtype=np.float64)
    reference_motion_scale = np.ones(NUM_FINGERS, dtype=np.float64)
    for index in range(NUM_FINGERS):
      role = ContactRole(int(self._roles[index]))
      if role is ContactRole.KEEP:
        if self._physical_contact_confirmed[index]:
          alpha = min(1.0, self._role_elapsed[index] / self.config.load_ramp_time_s)
          desired[index] = (
            self._load_start_force[index]
            + alpha * (target[index] - self._load_start_force[index])
          )
          mcc_enabled[index] = True
        else:
          full_authority[index] = True
          if self._make_elapsed[index] > 0.0:
            # Preserve a small preload while contact confirmation accumulates.
            # Freezing at the first force tick caused contact flicker and made
            # several fingers repeatedly miss the 30 ms confirmation window.
            reference_motion_scale[index] = self.config.make_confirmation_motion_scale
      elif role is ContactRole.RELEASE:
        alpha = min(1.0, self._role_elapsed[index] / self.config.release_ramp_time_s)
        desired[index] = (1.0 - alpha) * self._release_start_force[index]
        mcc_enabled[index] = True
        tangential_scale[index] = 0.0
      else:
        full_authority[index] = True
        if role is ContactRole.MAKE and self._make_elapsed[index] > 0.0:
          reference_motion_scale[index] = self.config.make_confirmation_motion_scale

    arrays = (
      self._roles.copy(),
      self._physical_contact_confirmed.copy(),
      desired,
      mcc_enabled,
      full_authority,
      reference_motion_scale,
      tangential_scale,
      approved,
    )
    for value in arrays:
      value.setflags(write=False)
    return RoleInterpreterOutput(
      roles=arrays[0],
      physical_contact_confirmed=arrays[1],
      desired_force_n=arrays[2],
      mcc_enabled=arrays[3],
      full_reference_authority=arrays[4],
      reference_motion_scale=arrays[5],
      tangential_scale=arrays[6],
      transition_approved=arrays[7],
      transition_reason=tuple(reasons),
    )


__all__ = [
  "ContactRole",
  "ContactRoleInterpreter",
  "RoleInterpreterConfig",
  "RoleInterpreterOutput",
]
