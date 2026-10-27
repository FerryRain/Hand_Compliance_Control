"""M0 contract for the 7-DoF FR3 plus 16-DoF Leap Hand plant."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, ClassVar, Mapping, TextIO

import numpy as np
from numpy.typing import ArrayLike, NDArray

from Module.common.contracts import ContactState, ExecutorState
from Module.fr3_leap.model import ARM_JOINT_NAMES, HAND_JOINT_NAMES


FULL_ROBOT_SCHEMA_VERSION = "fr3-leap-modules.v1"


def _array(value: ArrayLike, name: str, shape: tuple[int, ...]) -> NDArray[np.float64]:
  result = np.asarray(value, dtype=np.float64)
  if result.shape != shape or not np.all(np.isfinite(result)):
    raise ValueError(f"{name} must be finite with shape {shape}, got {result.shape}")
  result = np.array(result, dtype=np.float64, copy=True)
  result.setflags(write=False)
  return result


def _bool_array(value: ArrayLike, name: str, shape: tuple[int, ...]) -> NDArray[np.bool_]:
  result = np.asarray(value, dtype=np.bool_)
  if result.shape != shape:
    raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
  result = np.array(result, dtype=np.bool_, copy=True)
  result.setflags(write=False)
  return result


@dataclass(frozen=True, slots=True)
class FullRobotStateSnapshot:
  """One authoritative full-robot sample in explicit frames and joint groups.

  Contact force vectors are object-on-hand forces expressed in ``force_frame``.
  The wrist wrench uses the same acting-body convention and is referenced at
  ``wrench_reference``.  Predicted contact modes are deliberately absent from
  the authoritative :attr:`actual_contact_set` computation.
  """

  timestamp_s: float
  episode_id: str
  step: int
  seed: int
  surface_model_version: str
  palm_pose_world: ArrayLike
  palm_twist_world: ArrayLike
  wrist_wrench: ArrayLike
  arm_q_rad: ArrayLike
  arm_dq_rad_s: ArrayLike
  arm_command_rad: ArrayLike
  arm_external_torque_nm: ArrayLike
  finger_q_rad: ArrayLike
  finger_dq_rad_s: ArrayLike
  finger_command_rad: ArrayLike
  fingertip_positions_world_m: ArrayLike
  fingertip_force_vectors: ArrayLike
  fingertip_normal_forces_n: ArrayLike
  contact_positions_world_m: ArrayLike
  contact_normals_world: ArrayLike
  contact_states: tuple[ContactState | str, ...]
  contact_position_valid: ArrayLike
  sampling_period_s: float = 0.002
  force_frame: str = "world"
  wrench_frame: str = "world"
  wrench_reference: str = "fr3_palm_control_site"
  wrench_acting_on: str = "HAND"
  wrench_estimator: str = "FR3_JOINT_CONSTRAINT_TORQUE"
  wrist_compliance_offset: ArrayLike = (0.0,) * 6
  finger_compliance_offsets_m: ArrayLike = (0.0,) * 4
  arm_actuator_saturated: ArrayLike = (False,) * 7
  finger_actuator_saturated: ArrayLike = (False,) * 16
  sensor_validity: Mapping[str, bool] = field(default_factory=dict)
  executor_state: ExecutorState | str = ExecutorState.RUNNING
  guard_reason: str = "NONE"
  safety_override: str | None = None
  controller_mode: str = "PRESCRIBED_WRIST_FINGER_MCC"
  schema_version: ClassVar[str] = FULL_ROBOT_SCHEMA_VERSION
  arm_joint_names: ClassVar[tuple[str, ...]] = ARM_JOINT_NAMES
  finger_joint_names: ClassVar[tuple[str, ...]] = HAND_JOINT_NAMES

  def __post_init__(self) -> None:
    if not np.isfinite(self.timestamp_s) or self.timestamp_s < 0.0:
      raise ValueError("timestamp_s must be finite and non-negative")
    if self.step < 0 or not self.episode_id or not self.surface_model_version:
      raise ValueError("step, episode_id and surface_model_version are invalid")
    if not np.isfinite(self.sampling_period_s) or self.sampling_period_s <= 0.0:
      raise ValueError("sampling_period_s must be finite and positive")
    for name in (
      "force_frame",
      "wrench_frame",
      "wrench_reference",
      "wrench_acting_on",
      "wrench_estimator",
      "controller_mode",
      "guard_reason",
    ):
      if not str(getattr(self, name)):
        raise ValueError(f"{name} must be non-empty")
    if self.wrench_acting_on != "HAND":
      raise ValueError("wrench_acting_on must be HAND in this contract version")

    arrays = {
      "palm_pose_world": ((7,), self.palm_pose_world),
      "palm_twist_world": ((6,), self.palm_twist_world),
      "wrist_wrench": ((6,), self.wrist_wrench),
      "arm_q_rad": ((7,), self.arm_q_rad),
      "arm_dq_rad_s": ((7,), self.arm_dq_rad_s),
      "arm_command_rad": ((7,), self.arm_command_rad),
      "arm_external_torque_nm": ((7,), self.arm_external_torque_nm),
      "finger_q_rad": ((16,), self.finger_q_rad),
      "finger_dq_rad_s": ((16,), self.finger_dq_rad_s),
      "finger_command_rad": ((16,), self.finger_command_rad),
      "fingertip_positions_world_m": ((4, 3), self.fingertip_positions_world_m),
      "fingertip_force_vectors": ((4, 3), self.fingertip_force_vectors),
      "fingertip_normal_forces_n": ((4,), self.fingertip_normal_forces_n),
      "contact_positions_world_m": ((4, 3), self.contact_positions_world_m),
      "contact_normals_world": ((4, 3), self.contact_normals_world),
      "wrist_compliance_offset": ((6,), self.wrist_compliance_offset),
      "finger_compliance_offsets_m": ((4,), self.finger_compliance_offsets_m),
    }
    for name, (shape, value) in arrays.items():
      object.__setattr__(self, name, _array(value, name, shape))
    quaternion_norm = float(np.linalg.norm(self.palm_pose_world[3:]))
    if not np.isclose(quaternion_norm, 1.0, atol=1e-6, rtol=0.0):
      raise ValueError("palm quaternion must be unit length in [qw,qx,qy,qz] order")
    if np.any(self.fingertip_normal_forces_n < 0.0):
      raise ValueError("fingertip_normal_forces_n must be non-negative")

    contact_states = tuple(ContactState(value) for value in self.contact_states)
    if len(contact_states) != 4:
      raise ValueError("contact_states must contain four fingertips")
    valid = _bool_array(self.contact_position_valid, "contact_position_valid", (4,))
    if any(
      state is ContactState.CONTACT and not bool(is_valid)
      for state, is_valid in zip(contact_states, valid)
    ):
      raise ValueError("CONTACT requires a valid contact position")
    arm_saturated = _bool_array(
      self.arm_actuator_saturated,
      "arm_actuator_saturated",
      (7,),
    )
    finger_saturated = _bool_array(
      self.finger_actuator_saturated,
      "finger_actuator_saturated",
      (16,),
    )
    validity = {str(name): bool(value) for name, value in self.sensor_validity.items()}
    if not validity:
      raise ValueError("sensor_validity must explicitly identify available channels")
    object.__setattr__(self, "contact_states", contact_states)
    object.__setattr__(self, "contact_position_valid", valid)
    object.__setattr__(self, "arm_actuator_saturated", arm_saturated)
    object.__setattr__(self, "finger_actuator_saturated", finger_saturated)
    object.__setattr__(self, "sensor_validity", validity)
    object.__setattr__(self, "executor_state", ExecutorState(self.executor_state))

  @property
  def actual_contact_set(self) -> frozenset[int]:
    return frozenset(
      index + 1
      for index, state in enumerate(self.contact_states)
      if state is ContactState.CONTACT
    )

  @property
  def q_rad(self) -> NDArray[np.float64]:
    result = np.concatenate((self.arm_q_rad, self.finger_q_rad))
    result.setflags(write=False)
    return result

  def to_dict(self) -> dict[str, Any]:
    payload: dict[str, Any] = {
      "schema_version": self.schema_version,
      "timestamp_s": self.timestamp_s,
      "episode_id": self.episode_id,
      "step": self.step,
      "seed": self.seed,
      "surface_model_version": self.surface_model_version,
      "sampling_period_s": self.sampling_period_s,
      "force_frame": self.force_frame,
      "wrench_frame": self.wrench_frame,
      "wrench_reference": self.wrench_reference,
      "wrench_acting_on": self.wrench_acting_on,
      "wrench_estimator": self.wrench_estimator,
      "arm_joint_names": list(self.arm_joint_names),
      "finger_joint_names": list(self.finger_joint_names),
      "contact_states": [state.value for state in self.contact_states],
      "actual_contact_set": sorted(self.actual_contact_set),
      "contact_position_valid": self.contact_position_valid.tolist(),
      "arm_actuator_saturated": self.arm_actuator_saturated.tolist(),
      "finger_actuator_saturated": self.finger_actuator_saturated.tolist(),
      "sensor_validity": dict(self.sensor_validity),
      "executor_state": self.executor_state.value,
      "guard_reason": self.guard_reason,
      "safety_override": self.safety_override,
      "controller_mode": self.controller_mode,
    }
    for name in (
      "palm_pose_world",
      "palm_twist_world",
      "wrist_wrench",
      "arm_q_rad",
      "arm_dq_rad_s",
      "arm_command_rad",
      "arm_external_torque_nm",
      "finger_q_rad",
      "finger_dq_rad_s",
      "finger_command_rad",
      "fingertip_positions_world_m",
      "fingertip_force_vectors",
      "fingertip_normal_forces_n",
      "contact_positions_world_m",
      "contact_normals_world",
      "wrist_compliance_offset",
      "finger_compliance_offsets_m",
    ):
      payload[name] = getattr(self, name).tolist()
    return payload

  @classmethod
  def from_dict(cls, payload: Mapping[str, Any]) -> "FullRobotStateSnapshot":
    if payload.get("schema_version") != FULL_ROBOT_SCHEMA_VERSION:
      raise ValueError("unsupported full-robot schema_version")
    if tuple(payload.get("arm_joint_names", ())) != ARM_JOINT_NAMES:
      raise ValueError("arm_joint_names/order do not match FR3 contract")
    if tuple(payload.get("finger_joint_names", ())) != HAND_JOINT_NAMES:
      raise ValueError("finger_joint_names/order do not match Leap contract")
    field_names = {definition.name for definition in fields(cls) if definition.init}
    kwargs = {name: payload[name] for name in field_names if name in payload}
    return cls(**kwargs)


class FullRobotJsonlLogger:
  def __init__(self, path: str | Path) -> None:
    self.path = Path(path)
    self._stream: TextIO | None = None

  def __enter__(self) -> "FullRobotJsonlLogger":
    self.path.parent.mkdir(parents=True, exist_ok=True)
    self._stream = self.path.open("w", encoding="utf-8")
    return self

  def append(self, snapshot: FullRobotStateSnapshot) -> None:
    if self._stream is None:
      raise RuntimeError("logger must be used as a context manager")
    self._stream.write(json.dumps(snapshot.to_dict(), sort_keys=True) + "\n")
    self._stream.flush()

  def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
    if self._stream is not None:
      self._stream.close()
      self._stream = None


def load_full_robot_jsonl(path: str | Path) -> list[FullRobotStateSnapshot]:
  snapshots: list[FullRobotStateSnapshot] = []
  with Path(path).open("r", encoding="utf-8") as stream:
    for line_number, line in enumerate(stream, start=1):
      if not line.strip():
        continue
      try:
        snapshots.append(FullRobotStateSnapshot.from_dict(json.loads(line)))
      except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid full-robot snapshot at line {line_number}: {error}") from error
  return snapshots
