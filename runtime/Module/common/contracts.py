"""Minimal Phase-0 contracts shared by Modules 1--3.

All numeric fields use SI units. Wrist poses are represented as
``[x, y, z, qw, qx, qy, qz]`` in the right-handed world frame.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Iterable, Mapping, TextIO

import numpy as np
from numpy.typing import ArrayLike, NDArray


SCHEMA_VERSION = "hand-modules.v1"


class ContactState(str, Enum):
  """Directly measured fingertip contact state."""

  CONTACT = "CONTACT"
  FREE = "FREE"


class ExecutorState(str, Enum):
  """Execution state, intentionally separate from contact state."""

  RUNNING = "RUNNING"
  DONE = "DONE"
  BLOCKED = "BLOCKED"
  CANCELLED = "CANCELLED"


def _finite_array(
  value: ArrayLike,
  *,
  name: str,
  ndim: int | None = None,
  shape: tuple[int, ...] | None = None,
) -> NDArray[np.float64]:
  array = np.asarray(value, dtype=np.float64)
  if ndim is not None and array.ndim != ndim:
    raise ValueError(f"{name} must have {ndim} dimensions, got {array.ndim}")
  if shape is not None and array.shape != shape:
    raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
  if not np.all(np.isfinite(array)):
    raise ValueError(f"{name} must contain only finite values")
  array = np.array(array, dtype=np.float64, copy=True)
  array.setflags(write=False)
  return array


def _optional_finite(value: float | None, name: str) -> float | None:
  if value is None:
    return None
  result = float(value)
  if not np.isfinite(result):
    raise ValueError(f"{name} must be finite when provided")
  return result


def _json_records(
  value: Iterable[Mapping[str, Any]],
  name: str,
) -> tuple[dict[str, Any], ...]:
  records = tuple(dict(record) for record in value)
  try:
    json.dumps(records, allow_nan=False)
  except (TypeError, ValueError) as error:
    raise ValueError(f"{name} must be JSON serializable") from error
  return records


def _json_mapping(value: Mapping[str, Any], name: str) -> dict[str, Any]:
  result = dict(value)
  try:
    json.dumps(result, allow_nan=False)
  except (TypeError, ValueError) as error:
    raise ValueError(f"{name} must be JSON serializable") from error
  return result


@dataclass(frozen=True, slots=True)
class StateSnapshot:
  """Versioned real-state snapshot used at module boundaries.

  Finger identifiers in :attr:`actual_contact_set` are one-based to match the
  contact-mode notation used by the master plan.
  """

  timestamp_s: float
  episode_id: str
  step: int
  seed: int
  wrist_pose: ArrayLike
  q: ArrayLike
  dq: ArrayLike
  fingertip_positions: ArrayLike
  fingertip_forces: ArrayLike
  contact_states: tuple[ContactState | str, ...]
  surface_model_version: str
  failed_evidence: tuple[str, ...] = ()
  executor_state: ExecutorState | str = ExecutorState.RUNNING
  frame_id: str = "world"
  evaluator_version: str = "module-evaluator.v1"
  sampling_period_s: float = 0.01
  wrist_twist: ArrayLike = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
  wrist_wrench: ArrayLike = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
  previous_action: ArrayLike = ()
  current_command: ArrayLike = ()
  predicted_contact_set: tuple[int, ...] = ()
  planned_trajectory: tuple[Mapping[str, Any], ...] = ()
  committed_prefix: tuple[Mapping[str, Any], ...] = ()
  prediction_suffix: tuple[Mapping[str, Any], ...] = ()
  transaction_id: str | None = None
  micro_barrier_state: str = "IDLE"
  blocked_reason: str | None = None
  blocked_evidence: Mapping[str, Any] = field(default_factory=dict)
  latencies_s: Mapping[str, float] = field(default_factory=dict)
  collision_distance_m: float | None = None
  joint_margin_rad: float | None = None
  anchor_margin_m: float | None = None
  reach_margin_m: float | None = None
  safety_override: str | None = None
  contact_event: str | None = None
  certificate_id: str | None = None
  schema_version: ClassVar[str] = SCHEMA_VERSION

  def __post_init__(self) -> None:
    if not np.isfinite(self.timestamp_s) or self.timestamp_s < 0.0:
      raise ValueError("timestamp_s must be finite and non-negative")
    if not self.episode_id:
      raise ValueError("episode_id must be non-empty")
    if self.step < 0:
      raise ValueError("step must be non-negative")
    if not self.surface_model_version:
      raise ValueError("surface_model_version must be non-empty")
    if not self.frame_id or not self.evaluator_version:
      raise ValueError("frame_id and evaluator_version must be non-empty")
    if not np.isfinite(self.sampling_period_s) or self.sampling_period_s <= 0.0:
      raise ValueError("sampling_period_s must be finite and positive")
    if not self.micro_barrier_state:
      raise ValueError("micro_barrier_state must be non-empty")

    wrist_pose = _finite_array(self.wrist_pose, name="wrist_pose", shape=(7,))
    quaternion_norm = float(np.linalg.norm(wrist_pose[3:]))
    if not np.isclose(quaternion_norm, 1.0, atol=1e-6, rtol=0.0):
      raise ValueError("wrist quaternion must be unit length in [qw,qx,qy,qz] order")

    q = _finite_array(self.q, name="q", ndim=1)
    dq = _finite_array(self.dq, name="dq", ndim=1)
    if q.shape != dq.shape:
      raise ValueError("q and dq must have identical shapes")

    fingertip_positions = _finite_array(
      self.fingertip_positions,
      name="fingertip_positions",
      ndim=2,
    )
    if fingertip_positions.shape[1:] != (3,):
      raise ValueError("fingertip_positions must have shape (num_fingers, 3)")
    num_fingers = fingertip_positions.shape[0]
    fingertip_forces = _finite_array(
      self.fingertip_forces,
      name="fingertip_forces",
      shape=(num_fingers,),
    )

    contact_states = tuple(ContactState(state) for state in self.contact_states)
    if len(contact_states) != num_fingers:
      raise ValueError("contact_states length must equal number of fingertips")

    executor_state = ExecutorState(self.executor_state)
    failed_evidence = tuple(str(item) for item in self.failed_evidence)
    wrist_twist = _finite_array(self.wrist_twist, name="wrist_twist", shape=(6,))
    wrist_wrench = _finite_array(self.wrist_wrench, name="wrist_wrench", shape=(6,))
    previous_action = _finite_array(self.previous_action, name="previous_action", ndim=1)
    current_command = _finite_array(self.current_command, name="current_command", ndim=1)

    predicted_contact_set = tuple(int(index) for index in self.predicted_contact_set)
    if len(set(predicted_contact_set)) != len(predicted_contact_set):
      raise ValueError("predicted_contact_set must not contain duplicates")
    if any(index < 1 or index > num_fingers for index in predicted_contact_set):
      raise ValueError("predicted_contact_set contains an invalid one-based finger id")

    planned_trajectory = _json_records(self.planned_trajectory, "planned_trajectory")
    committed_prefix = _json_records(self.committed_prefix, "committed_prefix")
    prediction_suffix = _json_records(self.prediction_suffix, "prediction_suffix")
    blocked_evidence = _json_mapping(self.blocked_evidence, "blocked_evidence")
    latencies_s = {str(name): float(value) for name, value in self.latencies_s.items()}
    if any(not np.isfinite(value) or value < 0.0 for value in latencies_s.values()):
      raise ValueError("latencies_s values must be finite and non-negative")

    optional_strings = {
      "transaction_id": self.transaction_id,
      "blocked_reason": self.blocked_reason,
      "safety_override": self.safety_override,
      "contact_event": self.contact_event,
      "certificate_id": self.certificate_id,
    }
    for name, value in optional_strings.items():
      if value is not None and not str(value):
        raise ValueError(f"{name} must be non-empty when provided")

    object.__setattr__(self, "wrist_pose", wrist_pose)
    object.__setattr__(self, "wrist_twist", wrist_twist)
    object.__setattr__(self, "wrist_wrench", wrist_wrench)
    object.__setattr__(self, "q", q)
    object.__setattr__(self, "dq", dq)
    object.__setattr__(self, "previous_action", previous_action)
    object.__setattr__(self, "current_command", current_command)
    object.__setattr__(self, "fingertip_positions", fingertip_positions)
    object.__setattr__(self, "fingertip_forces", fingertip_forces)
    object.__setattr__(self, "contact_states", contact_states)
    object.__setattr__(self, "executor_state", executor_state)
    object.__setattr__(self, "failed_evidence", failed_evidence)
    object.__setattr__(self, "predicted_contact_set", predicted_contact_set)
    object.__setattr__(self, "planned_trajectory", planned_trajectory)
    object.__setattr__(self, "committed_prefix", committed_prefix)
    object.__setattr__(self, "prediction_suffix", prediction_suffix)
    object.__setattr__(self, "blocked_evidence", blocked_evidence)
    object.__setattr__(self, "latencies_s", latencies_s)
    object.__setattr__(
      self,
      "collision_distance_m",
      _optional_finite(self.collision_distance_m, "collision_distance_m"),
    )
    object.__setattr__(
      self,
      "joint_margin_rad",
      _optional_finite(self.joint_margin_rad, "joint_margin_rad"),
    )
    object.__setattr__(
      self,
      "anchor_margin_m",
      _optional_finite(self.anchor_margin_m, "anchor_margin_m"),
    )
    object.__setattr__(
      self,
      "reach_margin_m",
      _optional_finite(self.reach_margin_m, "reach_margin_m"),
    )

  @property
  def actual_contact_set(self) -> frozenset[int]:
    """Return authoritative, one-based contacts derived only from measurements."""

    return frozenset(
      index + 1
      for index, state in enumerate(self.contact_states)
      if state is ContactState.CONTACT
    )

  def to_dict(self) -> dict[str, Any]:
    return {
      "schema_version": self.schema_version,
      "timestamp_s": self.timestamp_s,
      "episode_id": self.episode_id,
      "step": self.step,
      "seed": self.seed,
      "frame_id": self.frame_id,
      "evaluator_version": self.evaluator_version,
      "sampling_period_s": self.sampling_period_s,
      "wrist_pose": self.wrist_pose.tolist(),
      "wrist_twist": self.wrist_twist.tolist(),
      "wrist_wrench": self.wrist_wrench.tolist(),
      "q": self.q.tolist(),
      "dq": self.dq.tolist(),
      "previous_action": self.previous_action.tolist(),
      "current_command": self.current_command.tolist(),
      "fingertip_positions": self.fingertip_positions.tolist(),
      "fingertip_forces": self.fingertip_forces.tolist(),
      "contact_states": [state.value for state in self.contact_states],
      "actual_contact_set": sorted(self.actual_contact_set),
      "predicted_contact_set": list(self.predicted_contact_set),
      "surface_model_version": self.surface_model_version,
      "failed_evidence": list(self.failed_evidence),
      "executor_state": self.executor_state.value,
      "planned_trajectory": list(self.planned_trajectory),
      "committed_prefix": list(self.committed_prefix),
      "prediction_suffix": list(self.prediction_suffix),
      "transaction_id": self.transaction_id,
      "micro_barrier_state": self.micro_barrier_state,
      "blocked_reason": self.blocked_reason,
      "blocked_evidence": dict(self.blocked_evidence),
      "latencies_s": dict(self.latencies_s),
      "collision_distance_m": self.collision_distance_m,
      "joint_margin_rad": self.joint_margin_rad,
      "anchor_margin_m": self.anchor_margin_m,
      "reach_margin_m": self.reach_margin_m,
      "safety_override": self.safety_override,
      "contact_event": self.contact_event,
      "certificate_id": self.certificate_id,
    }

  @classmethod
  def from_dict(cls, payload: Mapping[str, Any]) -> StateSnapshot:
    schema_version = payload.get("schema_version")
    if schema_version != SCHEMA_VERSION:
      raise ValueError(
        f"unsupported schema_version {schema_version!r}; expected {SCHEMA_VERSION!r}"
      )
    return cls(
      timestamp_s=float(payload["timestamp_s"]),
      episode_id=str(payload["episode_id"]),
      step=int(payload["step"]),
      seed=int(payload["seed"]),
      frame_id=str(payload.get("frame_id", "world")),
      evaluator_version=str(payload.get("evaluator_version", "module-evaluator.v1")),
      sampling_period_s=float(payload.get("sampling_period_s", 0.01)),
      wrist_pose=payload["wrist_pose"],
      wrist_twist=payload.get("wrist_twist", (0.0,) * 6),
      wrist_wrench=payload.get("wrist_wrench", (0.0,) * 6),
      q=payload["q"],
      dq=payload["dq"],
      previous_action=payload.get("previous_action", ()),
      current_command=payload.get("current_command", ()),
      fingertip_positions=payload["fingertip_positions"],
      fingertip_forces=payload["fingertip_forces"],
      contact_states=tuple(payload["contact_states"]),
      predicted_contact_set=tuple(payload.get("predicted_contact_set", ())),
      surface_model_version=str(payload["surface_model_version"]),
      failed_evidence=tuple(payload.get("failed_evidence", ())),
      executor_state=str(payload.get("executor_state", ExecutorState.RUNNING.value)),
      planned_trajectory=tuple(payload.get("planned_trajectory", ())),
      committed_prefix=tuple(payload.get("committed_prefix", ())),
      prediction_suffix=tuple(payload.get("prediction_suffix", ())),
      transaction_id=payload.get("transaction_id"),
      micro_barrier_state=str(payload.get("micro_barrier_state", "IDLE")),
      blocked_reason=payload.get("blocked_reason"),
      blocked_evidence=payload.get("blocked_evidence", {}),
      latencies_s=payload.get("latencies_s", {}),
      collision_distance_m=payload.get("collision_distance_m"),
      joint_margin_rad=payload.get("joint_margin_rad"),
      anchor_margin_m=payload.get("anchor_margin_m"),
      reach_margin_m=payload.get("reach_margin_m"),
      safety_override=payload.get("safety_override"),
      contact_event=payload.get("contact_event"),
      certificate_id=payload.get("certificate_id"),
    )


class JsonlEpisodeLogger:
  """Append-only JSONL logger for validated :class:`StateSnapshot` objects."""

  def __init__(self, path: str | Path) -> None:
    self.path = Path(path)
    self._stream: TextIO | None = None

  def __enter__(self) -> JsonlEpisodeLogger:
    self.path.parent.mkdir(parents=True, exist_ok=True)
    self._stream = self.path.open("w", encoding="utf-8")
    return self

  def append(self, snapshot: StateSnapshot) -> None:
    if self._stream is None:
      raise RuntimeError("logger must be used as a context manager")
    self._stream.write(json.dumps(snapshot.to_dict(), sort_keys=True) + "\n")
    self._stream.flush()

  def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
    if self._stream is not None:
      self._stream.close()
      self._stream = None


def load_jsonl_episode(path: str | Path) -> list[StateSnapshot]:
  """Load and validate every state snapshot in a JSONL episode."""

  snapshots: list[StateSnapshot] = []
  with Path(path).open("r", encoding="utf-8") as stream:
    for line_number, line in enumerate(stream, start=1):
      if not line.strip():
        continue
      try:
        payload = json.loads(line)
        snapshots.append(StateSnapshot.from_dict(payload))
      except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid snapshot at line {line_number}: {error}") from error
  return snapshots


def snapshots_to_dicts(snapshots: Iterable[StateSnapshot]) -> list[dict[str, Any]]:
  """Convenience conversion used by deterministic evaluators."""

  return [snapshot.to_dict() for snapshot in snapshots]
