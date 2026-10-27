"""Certificate-gated short-prefix executor for the explicit MCC baseline.

M06 deliberately knows nothing about beam search.  It accepts one audited
prefix, executes only that prefix, waits at a micro barrier, and exposes a new
measured snapshot.  Any prediction suffix remains outside the command path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from Module.module_2_fingertip_mcc import FullRobotFingertipMCC


NUM_FINGERS = 4
_CERTIFICATE_AUTHORITY = object()


def _finite_array(
  value: ArrayLike,
  name: str,
  *,
  shape: tuple[int, ...] | None = None,
  ndim: int | None = None,
) -> NDArray[np.float64]:
  array = np.asarray(value, dtype=np.float64)
  if shape is not None and array.shape != shape:
    raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
  if ndim is not None and array.ndim != ndim:
    raise ValueError(f"{name} must have ndim={ndim}, got {array.ndim}")
  if not np.all(np.isfinite(array)):
    raise ValueError(f"{name} must contain only finite values")
  result = np.array(array, copy=True)
  result.setflags(write=False)
  return result


def _contact_set(value: object, name: str) -> frozenset[int]:
  result = frozenset(int(finger) for finger in value)  # type: ignore[arg-type]
  if not result or any(finger < 1 or finger > NUM_FINGERS for finger in result):
    raise ValueError(f"{name} must be a nonempty subset of {{1,2,3,4}}")
  return result


class TransactionType(str, Enum):
  WRIST_ADJUST = "WRIST_ADJUST"
  FINGER_RECONFIGURE = "FINGER_RECONFIGURE"


class PrefixSource(str, Enum):
  OPTIMIZER_COMMIT_CANDIDATE = "OPTIMIZER_COMMIT_CANDIDATE"
  PREDICTION_SUFFIX = "PREDICTION_SUFFIX"
  MANUAL_TEST = "MANUAL_TEST"


class ParticipantState(str, Enum):
  RUNNING = "RUNNING"
  DONE = "DONE"
  BLOCKED = "BLOCKED"
  CANCELLED = "CANCELLED"


class TransactionState(str, Enum):
  IDLE = "IDLE"
  RUNNING = "RUNNING"
  DONE = "DONE"
  BLOCKED = "BLOCKED"
  CANCELLED = "CANCELLED"
  SAFE_HOLD = "SAFE_HOLD"


class BarrierState(str, Enum):
  IDLE = "IDLE"
  OPEN = "OPEN"
  WAITING_FOR_FRESH_OBSERVATION = "WAITING_FOR_FRESH_OBSERVATION"
  CLOSED = "CLOSED"
  SAFE_HOLD = "SAFE_HOLD"


@dataclass(frozen=True, slots=True)
class PrefixSample:
  """One time-parameterized sample of a committed Cartesian/joint prefix."""

  time_s: float
  wrist_position_m: ArrayLike
  fingertip_positions_m: ArrayLike
  joint_positions_rad: ArrayLike

  def __post_init__(self) -> None:
    if not np.isfinite(self.time_s) or self.time_s < 0.0:
      raise ValueError("time_s must be finite and non-negative")
    object.__setattr__(
      self,
      "wrist_position_m",
      _finite_array(self.wrist_position_m, "wrist_position_m", shape=(3,)),
    )
    object.__setattr__(
      self,
      "fingertip_positions_m",
      _finite_array(
        self.fingertip_positions_m,
        "fingertip_positions_m",
        shape=(NUM_FINGERS, 3),
      ),
    )
    object.__setattr__(
      self,
      "joint_positions_rad",
      _finite_array(self.joint_positions_rad, "joint_positions_rad", ndim=1),
    )

  def to_dict(self) -> dict[str, Any]:
    return {
      "time_s": self.time_s,
      "wrist_position_m": self.wrist_position_m.tolist(),
      "fingertip_positions_m": self.fingertip_positions_m.tolist(),
      "joint_positions_rad": self.joint_positions_rad.tolist(),
    }


@dataclass(frozen=True, slots=True)
class PlannedPrefix:
  """A short prefix; being optimized is not equivalent to being executable."""

  prefix_id: str
  transaction_type: TransactionType | str
  primitive_kind: str
  surface_model_version: str
  root_contact_set: frozenset[int] | set[int] | tuple[int, ...]
  expected_terminal_contact_set: frozenset[int] | set[int] | tuple[int, ...]
  samples: tuple[PrefixSample, ...]
  participating_fingers: tuple[int, ...] = ()
  anchor_fingers: tuple[int, ...] = ()
  finger_id: int | None = None
  topology_change_count: int = 0
  source: PrefixSource | str = PrefixSource.OPTIMIZER_COMMIT_CANDIDATE
  metadata: Mapping[str, Any] = field(default_factory=dict)

  def __post_init__(self) -> None:
    if not self.prefix_id or not self.primitive_kind or not self.surface_model_version:
      raise ValueError("prefix_id, primitive_kind, and surface_model_version are required")
    transaction_type = TransactionType(self.transaction_type)
    source = PrefixSource(self.source)
    root = _contact_set(self.root_contact_set, "root_contact_set")
    terminal = _contact_set(
      self.expected_terminal_contact_set,
      "expected_terminal_contact_set",
    )
    samples = tuple(self.samples)
    if len(samples) < 2:
      raise ValueError("a prefix needs at least two samples")
    if abs(samples[0].time_s) > 1e-12:
      raise ValueError("the first prefix sample must be at time zero")
    if any(right.time_s <= left.time_s for left, right in zip(samples, samples[1:])):
      raise ValueError("prefix sample times must be strictly increasing")
    joint_shape = samples[0].joint_positions_rad.shape
    if any(sample.joint_positions_rad.shape != joint_shape for sample in samples):
      raise ValueError("all prefix samples must use the same joint dimension")

    participating = tuple(sorted(set(int(finger) for finger in self.participating_fingers)))
    anchors = tuple(sorted(set(int(finger) for finger in self.anchor_fingers)))
    if any(finger < 1 or finger > NUM_FINGERS for finger in participating + anchors):
      raise ValueError("finger identifiers must be one-based in [1,4]")
    if set(participating) & set(anchors):
      raise ValueError("participating_fingers and anchor_fingers must be disjoint")
    if transaction_type is TransactionType.WRIST_ADJUST and participating:
      raise ValueError("WRIST_ADJUST cannot contain participating fingers")
    if transaction_type is TransactionType.FINGER_RECONFIGURE and not participating:
      raise ValueError("FINGER_RECONFIGURE needs at least one participating finger")
    if self.finger_id is not None and self.finger_id not in participating:
      raise ValueError("finger_id must be one of participating_fingers")
    if self.topology_change_count not in (0, 1):
      raise ValueError("a committed prefix may contain at most one topology change")
    metadata = dict(self.metadata)
    is_make_progress = bool(metadata.get("make_progress", False))
    if self.primitive_kind == "MAKE":
      expected_changes = 0 if is_make_progress else 1
      if self.topology_change_count != expected_changes:
        raise ValueError("MAKE_PROGRESS has zero topology changes; completed MAKE has one")
    elif self.primitive_kind == "BREAK" and self.topology_change_count != 1:
      raise ValueError("BREAK prefixes must declare one topology change")
    if self.primitive_kind not in {"MAKE", "BREAK"} and self.topology_change_count != 0:
      raise ValueError("only MAKE/BREAK may change topology")
    try:
      json.dumps(metadata, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as error:
      raise ValueError("metadata must be finite JSON data") from error

    object.__setattr__(self, "transaction_type", transaction_type)
    object.__setattr__(self, "source", source)
    object.__setattr__(self, "root_contact_set", root)
    object.__setattr__(self, "expected_terminal_contact_set", terminal)
    object.__setattr__(self, "samples", samples)
    object.__setattr__(self, "participating_fingers", participating)
    object.__setattr__(self, "anchor_fingers", anchors)
    object.__setattr__(self, "metadata", metadata)

  @property
  def duration_s(self) -> float:
    return self.samples[-1].time_s

  def to_dict(self) -> dict[str, Any]:
    return {
      "prefix_id": self.prefix_id,
      "transaction_type": self.transaction_type.value,
      "primitive_kind": self.primitive_kind,
      "surface_model_version": self.surface_model_version,
      "root_contact_set": sorted(self.root_contact_set),
      "expected_terminal_contact_set": sorted(self.expected_terminal_contact_set),
      "samples": [sample.to_dict() for sample in self.samples],
      "participating_fingers": list(self.participating_fingers),
      "anchor_fingers": list(self.anchor_fingers),
      "finger_id": self.finger_id,
      "topology_change_count": self.topology_change_count,
      "source": self.source.value,
      "metadata": dict(self.metadata),
    }


def prefix_digest(prefix: PlannedPrefix) -> str:
  payload = json.dumps(
    prefix.to_dict(),
    allow_nan=False,
    separators=(",", ":"),
    sort_keys=True,
  ).encode("utf-8")
  return hashlib.sha256(payload).hexdigest()


class ExecutionCertificate:
  """Opaque capability issued only through M10's exact-audit code path."""

  __slots__ = (
    "certificate_id",
    "prefix_id",
    "prefix_digest",
    "surface_model_version",
    "root_contact_set",
    "audit_version",
    "issued_at_s",
    "_authority",
    "_initialized",
  )

  def __init__(
    self,
    *,
    certificate_id: str,
    prefix_id: str,
    prefix_digest_value: str,
    surface_model_version: str,
    root_contact_set: frozenset[int],
    audit_version: str,
    issued_at_s: float,
    _authority: object | None = None,
  ) -> None:
    if _authority is not _CERTIFICATE_AUTHORITY:
      raise PermissionError("ExecutionCertificate can only be issued by ExactPrefixAudit")
    object.__setattr__(self, "certificate_id", certificate_id)
    object.__setattr__(self, "prefix_id", prefix_id)
    object.__setattr__(self, "prefix_digest", prefix_digest_value)
    object.__setattr__(self, "surface_model_version", surface_model_version)
    object.__setattr__(self, "root_contact_set", root_contact_set)
    object.__setattr__(self, "audit_version", audit_version)
    object.__setattr__(self, "issued_at_s", issued_at_s)
    object.__setattr__(self, "_authority", _authority)
    object.__setattr__(self, "_initialized", True)

  def __setattr__(self, name: str, value: object) -> None:
    if getattr(self, "_initialized", False):
      raise AttributeError("ExecutionCertificate is immutable")
    object.__setattr__(self, name, value)

  @property
  def authentic(self) -> bool:
    return self._authority is _CERTIFICATE_AUTHORITY

  def to_dict(self) -> dict[str, Any]:
    return {
      "certificate_id": self.certificate_id,
      "prefix_id": self.prefix_id,
      "prefix_digest": self.prefix_digest,
      "surface_model_version": self.surface_model_version,
      "root_contact_set": sorted(self.root_contact_set),
      "audit_version": self.audit_version,
      "issued_at_s": self.issued_at_s,
    }


def _issue_execution_certificate(
  *,
  certificate_id: str,
  prefix: PlannedPrefix,
  root_contact_set: frozenset[int],
  audit_version: str,
  issued_at_s: float,
) -> ExecutionCertificate:
  """Internal M06/M10 bridge; deliberately omitted from public exports."""

  return ExecutionCertificate(
    certificate_id=certificate_id,
    prefix_id=prefix.prefix_id,
    prefix_digest_value=prefix_digest(prefix),
    surface_model_version=prefix.surface_model_version,
    root_contact_set=root_contact_set,
    audit_version=audit_version,
    issued_at_s=issued_at_s,
    _authority=_CERTIFICATE_AUTHORITY,
  )


@dataclass(frozen=True, slots=True)
class ExecutorObservation:
  timestamp_s: float
  surface_model_version: str
  wrist_position_m: ArrayLike
  fingertip_positions_m: ArrayLike
  joint_positions_rad: ArrayLike
  fingertip_forces_n: ArrayLike
  outward_normals: ArrayLike
  actual_contact_set: frozenset[int] | set[int] | tuple[int, ...]
  blocked_fingers: Mapping[int, str] = field(default_factory=dict)
  global_safety_reason: str | None = None

  def __post_init__(self) -> None:
    if not np.isfinite(self.timestamp_s) or self.timestamp_s < 0.0:
      raise ValueError("timestamp_s must be finite and non-negative")
    if not self.surface_model_version:
      raise ValueError("surface_model_version must be nonempty")
    object.__setattr__(
      self,
      "wrist_position_m",
      _finite_array(self.wrist_position_m, "wrist_position_m", shape=(3,)),
    )
    object.__setattr__(
      self,
      "fingertip_positions_m",
      _finite_array(
        self.fingertip_positions_m,
        "fingertip_positions_m",
        shape=(NUM_FINGERS, 3),
      ),
    )
    object.__setattr__(
      self,
      "joint_positions_rad",
      _finite_array(self.joint_positions_rad, "joint_positions_rad", ndim=1),
    )
    forces = _finite_array(
      self.fingertip_forces_n,
      "fingertip_forces_n",
      shape=(NUM_FINGERS,),
    )
    if np.any(forces < 0.0):
      raise ValueError("fingertip_forces_n must be non-negative")
    normals = _finite_array(
      self.outward_normals,
      "outward_normals",
      shape=(NUM_FINGERS, 3),
    )
    norms = np.linalg.norm(normals, axis=1)
    if np.any(np.abs(norms - 1.0) > 1e-6):
      raise ValueError("outward_normals rows must be unit length")
    contacts = frozenset(int(finger) for finger in self.actual_contact_set)
    if any(finger < 1 or finger > NUM_FINGERS for finger in contacts):
      raise ValueError("actual_contact_set contains an invalid finger")
    blocked = {int(finger): str(reason) for finger, reason in self.blocked_fingers.items()}
    if any(finger < 1 or finger > NUM_FINGERS for finger in blocked):
      raise ValueError("blocked_fingers contains an invalid finger")
    if any(not reason for reason in blocked.values()):
      raise ValueError("blocked_fingers reasons must be nonempty")
    if self.global_safety_reason is not None and not self.global_safety_reason:
      raise ValueError("global_safety_reason must be nonempty when provided")
    object.__setattr__(self, "fingertip_forces_n", forces)
    object.__setattr__(self, "outward_normals", normals)
    object.__setattr__(self, "actual_contact_set", contacts)
    object.__setattr__(self, "blocked_fingers", blocked)


@dataclass(frozen=True, slots=True)
class ExecutorConfig:
  completion_tolerance_m: float = 0.00075
  wrist_completion_tolerance_m: float = 0.00075
  # Optional certified joint-space completion for a WRIST participant.  This
  # is disabled by default.  Full-robot integrations may use the leading arm
  # slice when finite contact load leaves a small Cartesian servo offset even
  # though the actuated arm has tracked the audited terminal q.
  wrist_joint_completion_tolerance_rad: float | None = None
  wrist_joint_dimension: int = 0
  default_timeout_s: float = 1.0
  desired_anchor_force_n: float = 2.0
  barrier_fresh_observations: int = 1
  root_state_tolerance: float = 1e-9
  # A contact-complete MAKE may safely stop at an earlier point of the audited
  # prefix instead of pushing through the physical surface to its Cartesian
  # terminal.  Disabled by default to preserve the frozen M06 module tests;
  # physical integrations must opt in and provide debounced measured contact.
  make_contact_is_terminal: bool = False
  # Some physical integrations use prefixes whose first and terminal
  # Cartesian samples are closer than the plant tolerance while their audited
  # joint/orientation sweep is still essential.  They can require a minimum
  # elapsed fraction before non-MAKE participants complete.  Zero preserves
  # all legacy M06 behavior.
  minimum_execution_fraction: float = 0.0

  def __post_init__(self) -> None:
    for name in (
      "completion_tolerance_m",
      "wrist_completion_tolerance_m",
      "default_timeout_s",
      "desired_anchor_force_n",
      "root_state_tolerance",
    ):
      value = float(getattr(self, name))
      if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    if self.barrier_fresh_observations < 1:
      raise ValueError("barrier_fresh_observations must be positive")
    if self.wrist_joint_completion_tolerance_rad is None:
      if self.wrist_joint_dimension != 0:
        raise ValueError(
          "wrist_joint_dimension must be zero when joint completion is disabled"
        )
    else:
      tolerance = float(self.wrist_joint_completion_tolerance_rad)
      if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError(
          "wrist_joint_completion_tolerance_rad must be finite and positive"
        )
      if self.wrist_joint_dimension < 1:
        raise ValueError(
          "wrist_joint_dimension must be positive when joint completion is enabled"
        )
    if not 0.0 <= self.minimum_execution_fraction <= 1.0:
      raise ValueError("minimum_execution_fraction must lie in [0,1]")


@dataclass(frozen=True, slots=True)
class ParticipantRecord:
  participant: str
  state: ParticipantState
  reason: str | None = None


@dataclass(frozen=True, slots=True)
class BarrierSnapshot:
  transaction_id: str
  timestamp_s: float
  surface_model_version: str
  wrist_position_m: NDArray[np.float64]
  fingertip_positions_m: NDArray[np.float64]
  joint_positions_rad: NDArray[np.float64]
  fingertip_forces_n: NDArray[np.float64]
  actual_contact_set: frozenset[int]
  transaction_state: TransactionState
  participants: tuple[ParticipantRecord, ...]
  blocked_evidence: Mapping[str, str]
  certificate_id: str


@dataclass(frozen=True, slots=True)
class ExecutorCommand:
  transaction_id: str
  generation: int
  target_wrist_position_m: NDArray[np.float64]
  # The complete audited state sample is exposed for full-robot integrations.
  # Legacy Cartesian runners may ignore it; I04 uses its arm slice so 6D palm
  # motion cannot bypass the M10 certificate through an out-of-band IK command.
  target_joint_positions_rad: NDArray[np.float64]
  nominal_fingertip_positions_m: NDArray[np.float64]
  commanded_fingertip_positions_m: NDArray[np.float64]
  mcc_active_mask: NDArray[np.bool_]
  participants: tuple[ParticipantRecord, ...]
  transaction_state: TransactionState
  barrier_state: BarrierState
  safe_hold: bool
  safety_reason: str | None
  certificate_id: str
  prefix_digest: str


class MCCBaselineAdapter:
  """Apply the existing four-finger MCC only to measured active contacts."""

  def __init__(self, controller: FullRobotFingertipMCC) -> None:
    self.controller = controller

  def apply(
    self,
    nominal_positions_m: NDArray[np.float64],
    observation: ExecutorObservation,
    *,
    desired_force_n: float,
    excluded_fingers: frozenset[int] = frozenset(),
  ) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    active = np.array(
      [
        finger in observation.actual_contact_set and finger not in excluded_fingers
        for finger in range(1, NUM_FINGERS + 1)
      ],
      dtype=np.bool_,
    )
    desired = np.full(NUM_FINGERS, desired_force_n, dtype=np.float64)
    errors = desired - observation.fingertip_forces_n
    output = self.controller.step(
      nominal_positions_m,
      -observation.outward_normals,
      errors,
      active,
    )
    commands = np.stack([command.position_command for command in output.commands])
    commands.setflags(write=False)
    active.setflags(write=False)
    return commands, active


@dataclass(slots=True)
class _TransactionRuntime:
  transaction_id: str
  generation: int
  prefix: PlannedPrefix
  certificate: ExecutionCertificate
  started_at_s: float
  timeout_s: float
  participants: dict[str, ParticipantRecord]
  state: TransactionState = TransactionState.RUNNING
  barrier_state: BarrierState = BarrierState.OPEN
  barrier_observations: int = 0
  safety_reason: str | None = None
  blocked_evidence: dict[str, str] = field(default_factory=dict)
  last_observation_s: float = 0.0
  barrier_started_at_s: float | None = None


class TransactionalPrefixExecutor:
  """Atomic, real-state-driven executor with explicit revocation semantics."""

  def __init__(
    self,
    config: ExecutorConfig | None = None,
    *,
    mcc_adapter: MCCBaselineAdapter | None = None,
  ) -> None:
    self.config = config or ExecutorConfig()
    self.mcc_adapter = mcc_adapter
    self._runtime: _TransactionRuntime | None = None
    self._generation = 0
    self._revoked_transaction_ids: set[str] = set()
    self._last_barrier_snapshot: BarrierSnapshot | None = None

  @property
  def current_transaction_id(self) -> str | None:
    return None if self._runtime is None else self._runtime.transaction_id

  @property
  def transaction_state(self) -> TransactionState:
    return TransactionState.IDLE if self._runtime is None else self._runtime.state

  @property
  def revoked_transaction_ids(self) -> frozenset[str]:
    return frozenset(self._revoked_transaction_ids)

  def commit(
    self,
    prefix: PlannedPrefix,
    certificate: ExecutionCertificate,
    observation: ExecutorObservation,
    *,
    timeout_s: float | None = None,
  ) -> str:
    self._validate_commit(prefix, certificate, observation)
    if self._runtime is not None:
      self._revoke_current()
    self._generation += 1
    transaction_id = f"tx-{self._generation:06d}-{prefix.prefix_id}"
    if prefix.transaction_type is TransactionType.WRIST_ADJUST:
      participants = {
        "WRIST": ParticipantRecord("WRIST", ParticipantState.RUNNING),
      }
    else:
      participants = {
        f"FINGER_{finger}": ParticipantRecord(
          f"FINGER_{finger}",
          ParticipantState.RUNNING,
        )
        for finger in prefix.participating_fingers
      }
    effective_timeout = self.config.default_timeout_s if timeout_s is None else float(timeout_s)
    if not np.isfinite(effective_timeout) or effective_timeout <= 0.0:
      raise ValueError("timeout_s must be finite and positive")
    self._runtime = _TransactionRuntime(
      transaction_id=transaction_id,
      generation=self._generation,
      prefix=prefix,
      certificate=certificate,
      started_at_s=observation.timestamp_s,
      timeout_s=effective_timeout,
      participants=participants,
      last_observation_s=observation.timestamp_s,
    )
    self._last_barrier_snapshot = None
    return transaction_id

  def _validate_commit(
    self,
    prefix: PlannedPrefix,
    certificate: ExecutionCertificate,
    observation: ExecutorObservation,
  ) -> None:
    if prefix.source is PrefixSource.PREDICTION_SUFFIX:
      raise PermissionError("prediction suffix has no execution authority")
    if not isinstance(certificate, ExecutionCertificate) or not certificate.authentic:
      raise PermissionError("an authentic ExactPrefixAudit certificate is required")
    if certificate.prefix_id != prefix.prefix_id:
      raise PermissionError("certificate prefix_id mismatch")
    if certificate.prefix_digest != prefix_digest(prefix):
      raise PermissionError("certificate prefix digest mismatch")
    if certificate.surface_model_version != prefix.surface_model_version:
      raise PermissionError("certificate/prefix model version mismatch")
    if observation.surface_model_version != prefix.surface_model_version:
      raise PermissionError("stale SurfaceModelVersion")
    if observation.actual_contact_set != prefix.root_contact_set:
      raise PermissionError("real root contact set differs from audited root")
    if certificate.root_contact_set != observation.actual_contact_set:
      raise PermissionError("certificate root contact set mismatch")
    root = prefix.samples[0]
    tolerance = self.config.root_state_tolerance
    if float(np.max(np.abs(root.joint_positions_rad - observation.joint_positions_rad))) > tolerance:
      raise PermissionError("real root joint state differs from audited prefix")
    if float(np.linalg.norm(root.wrist_position_m - observation.wrist_position_m)) > tolerance:
      raise PermissionError("real root wrist state differs from audited prefix")
    if float(
      np.max(
        np.linalg.norm(
          root.fingertip_positions_m - observation.fingertip_positions_m,
          axis=1,
        )
      )
    ) > tolerance:
      raise PermissionError("real root fingertip state differs from audited prefix")

  def _revoke_current(self) -> None:
    assert self._runtime is not None
    for name, record in tuple(self._runtime.participants.items()):
      if record.state is ParticipantState.RUNNING:
        self._runtime.participants[name] = ParticipantRecord(
          name,
          ParticipantState.CANCELLED,
          "REVOKED_BY_NEW_TRANSACTION",
        )
    self._runtime.state = TransactionState.CANCELLED
    self._runtime.barrier_state = BarrierState.CLOSED
    self._revoked_transaction_ids.add(self._runtime.transaction_id)

  def planner_timeout(self, observation: ExecutorObservation) -> ExecutorCommand:
    if self._runtime is None:
      raise RuntimeError("no active transaction")
    return self._enter_safe_hold(observation, "PLANNER_TIMEOUT")

  def step(self, observation: ExecutorObservation) -> ExecutorCommand:
    runtime = self._runtime
    if runtime is None:
      raise RuntimeError("commit an audited prefix before step")
    if runtime.state in {
      TransactionState.DONE,
      TransactionState.BLOCKED,
      TransactionState.CANCELLED,
      TransactionState.SAFE_HOLD,
    }:
      return self._hold_command(observation)
    if observation.surface_model_version != runtime.prefix.surface_model_version:
      return self._enter_safe_hold(observation, "SURFACE_MODEL_VERSION_DRIFT")
    if observation.global_safety_reason is not None:
      return self._enter_safe_hold(observation, observation.global_safety_reason)
    if not observation.actual_contact_set:
      return self._enter_safe_hold(observation, "LAST_CONTACT_LOST")
    if observation.timestamp_s < runtime.last_observation_s - 1e-12:
      return self._enter_safe_hold(observation, "NON_MONOTONIC_TIMESTAMP")
    elapsed = observation.timestamp_s - runtime.started_at_s
    if elapsed > runtime.timeout_s + 1e-12:
      return self._enter_safe_hold(observation, "EXECUTOR_TIMEOUT")

    sample = self._interpolate(runtime.prefix, max(0.0, elapsed))
    self._revalidate_topology_at_barrier(observation)
    self._update_participants(observation, sample)
    terminal = all(
      record.state is not ParticipantState.RUNNING
      for record in runtime.participants.values()
    )
    if terminal:
      if runtime.barrier_state is BarrierState.OPEN:
        runtime.barrier_state = BarrierState.WAITING_FOR_FRESH_OBSERVATION
        runtime.barrier_observations = 0
        runtime.barrier_started_at_s = observation.timestamp_s
      else:
        assert runtime.barrier_started_at_s is not None
        if observation.timestamp_s > runtime.barrier_started_at_s + 1e-12:
          runtime.barrier_observations += 1
          runtime.barrier_started_at_s = observation.timestamp_s
        if runtime.barrier_observations >= self.config.barrier_fresh_observations:
          runtime.state = (
            TransactionState.BLOCKED
            if any(
              record.state is ParticipantState.BLOCKED
              for record in runtime.participants.values()
            )
            else TransactionState.DONE
          )
          runtime.barrier_state = BarrierState.CLOSED
          self._last_barrier_snapshot = self._snapshot(observation)
    runtime.last_observation_s = observation.timestamp_s
    return self._command_for_sample(observation, sample)

  def _revalidate_topology_at_barrier(
    self,
    observation: ExecutorObservation,
  ) -> None:
    """A topology-changing contact must persist through the fresh barrier.

    A single collision sample is enough to stop an audited MAKE sweep, but it
    is not enough to close the transaction.  If that contact disappears before
    the next authenticated observation, reopen the participant and continue
    tracking the certified terminal sample.  BREAK uses the symmetric rule.
    """

    runtime = self._runtime
    assert runtime is not None
    if runtime.barrier_state is not BarrierState.WAITING_FOR_FRESH_OBSERVATION:
      return
    primitive = runtime.prefix.primitive_kind
    if primitive not in {"MAKE", "BREAK"}:
      return
    if bool(runtime.prefix.metadata.get("make_progress", False)):
      return
    reopened = False
    for name, record in tuple(runtime.participants.items()):
      if record.state is not ParticipantState.DONE or not name.startswith("FINGER_"):
        continue
      finger = int(name.rsplit("_", maxsplit=1)[1])
      contact_persists = finger in observation.actual_contact_set
      if primitive == "BREAK":
        contact_persists = not contact_persists
      if not contact_persists:
        runtime.participants[name] = ParticipantRecord(
          name,
          ParticipantState.RUNNING,
        )
        reopened = True
    if reopened:
      runtime.barrier_state = BarrierState.OPEN
      runtime.barrier_observations = 0
      runtime.barrier_started_at_s = None

  def consume_barrier_snapshot(self) -> BarrierSnapshot | None:
    snapshot = self._last_barrier_snapshot
    self._last_barrier_snapshot = None
    return snapshot

  def _interpolate(self, prefix: PlannedPrefix, elapsed_s: float) -> PrefixSample:
    if elapsed_s >= prefix.duration_s:
      return prefix.samples[-1]
    for left, right in zip(prefix.samples, prefix.samples[1:]):
      if elapsed_s <= right.time_s:
        alpha = (elapsed_s - left.time_s) / (right.time_s - left.time_s)
        alpha = float(np.clip(alpha, 0.0, 1.0))
        return PrefixSample(
          time_s=elapsed_s,
          wrist_position_m=(1.0 - alpha) * left.wrist_position_m
          + alpha * right.wrist_position_m,
          fingertip_positions_m=(1.0 - alpha) * left.fingertip_positions_m
          + alpha * right.fingertip_positions_m,
          joint_positions_rad=(1.0 - alpha) * left.joint_positions_rad
          + alpha * right.joint_positions_rad,
        )
    return prefix.samples[-1]

  def _update_participants(
    self,
    observation: ExecutorObservation,
    sample: PrefixSample,
  ) -> None:
    runtime = self._runtime
    assert runtime is not None
    terminal_sample = runtime.prefix.samples[-1]
    for name, record in tuple(runtime.participants.items()):
      if record.state is not ParticipantState.RUNNING:
        continue
      if name == "WRIST":
        distance = float(
          np.linalg.norm(
            observation.wrist_position_m - terminal_sample.wrist_position_m
          )
        )
        joint_at_target = False
        joint_tolerance = self.config.wrist_joint_completion_tolerance_rad
        if joint_tolerance is not None:
          dimension = self.config.wrist_joint_dimension
          if dimension > len(observation.joint_positions_rad):
            raise ValueError(
              "wrist_joint_dimension exceeds the observed joint dimension"
            )
          joint_at_target = bool(
            np.max(
              np.abs(
                observation.joint_positions_rad[:dimension]
                - terminal_sample.joint_positions_rad[:dimension]
              )
            )
            <= joint_tolerance
          )
        progressed = (
          sample.time_s + 1e-12
          >= self.config.minimum_execution_fraction * terminal_sample.time_s
        )
        retained_root_anchor = bool(
          runtime.prefix.root_contact_set & observation.actual_contact_set
        )
        if (
          (distance <= self.config.wrist_completion_tolerance_m or joint_at_target)
          and progressed
          and retained_root_anchor
        ):
          runtime.participants[name] = ParticipantRecord(name, ParticipantState.DONE)
        continue
      finger = int(name.rsplit("_", maxsplit=1)[1])
      if finger in observation.blocked_fingers:
        reason = observation.blocked_fingers[finger]
        runtime.participants[name] = ParticipantRecord(
          name,
          ParticipantState.BLOCKED,
          reason,
        )
        runtime.blocked_evidence[name] = reason
        continue
      distance = float(
        np.linalg.norm(
          observation.fingertip_positions_m[finger - 1]
          - terminal_sample.fingertip_positions_m[finger - 1]
        )
      )
      at_target = distance <= self.config.completion_tolerance_m
      primitive = runtime.prefix.primitive_kind
      if primitive == "MAKE":
        make_progress = bool(runtime.prefix.metadata.get("make_progress", False))
        contact_condition = make_progress or finger in observation.actual_contact_set
        if make_progress:
          at_target = at_target and (
            sample.time_s + 1e-12
            >= self.config.minimum_execution_fraction * terminal_sample.time_s
          )
        if self.config.make_contact_is_terminal and contact_condition:
          # Every earlier point is contained in the swept, certified prefix.
          # The executor therefore truncates safely at measured contact rather
          # than commanding further penetration solely to meet a Cartesian
          # endpoint behind the physical surface.
          if not make_progress:
            at_target = True
      elif primitive == "BREAK":
        contact_condition = finger not in observation.actual_contact_set
      else:
        contact_condition = True
        at_target = at_target and (
          sample.time_s + 1e-12
          >= self.config.minimum_execution_fraction * terminal_sample.time_s
        )
      if at_target and contact_condition:
        runtime.participants[name] = ParticipantRecord(name, ParticipantState.DONE)

  def _command_for_sample(
    self,
    observation: ExecutorObservation,
    sample: PrefixSample,
  ) -> ExecutorCommand:
    runtime = self._runtime
    assert runtime is not None
    nominal = np.array(sample.fingertip_positions_m, copy=True)
    terminal = runtime.prefix.samples[-1]
    for record in runtime.participants.values():
      if not record.participant.startswith("FINGER_"):
        continue
      finger = int(record.participant.rsplit("_", maxsplit=1)[1])
      if record.state is ParticipantState.DONE:
        nominal[finger - 1] = terminal.fingertip_positions_m[finger - 1]
      elif record.state in {ParticipantState.BLOCKED, ParticipantState.CANCELLED}:
        nominal[finger - 1] = observation.fingertip_positions_m[finger - 1]
    excluded = (
      frozenset(runtime.prefix.participating_fingers)
      if runtime.prefix.primitive_kind == "BREAK"
      else frozenset()
    )
    if self.mcc_adapter is None:
      commanded = np.array(nominal, copy=True)
      active = np.zeros(NUM_FINGERS, dtype=np.bool_)
    else:
      commanded, active = self.mcc_adapter.apply(
        nominal,
        observation,
        desired_force_n=self.config.desired_anchor_force_n,
        excluded_fingers=excluded,
      )
    nominal.setflags(write=False)
    commanded.setflags(write=False)
    active.setflags(write=False)
    return ExecutorCommand(
      transaction_id=runtime.transaction_id,
      generation=runtime.generation,
      target_wrist_position_m=np.array(sample.wrist_position_m, copy=True),
      target_joint_positions_rad=np.array(sample.joint_positions_rad, copy=True),
      nominal_fingertip_positions_m=nominal,
      commanded_fingertip_positions_m=commanded,
      mcc_active_mask=active,
      participants=tuple(runtime.participants.values()),
      transaction_state=runtime.state,
      barrier_state=runtime.barrier_state,
      safe_hold=runtime.state is TransactionState.SAFE_HOLD,
      safety_reason=runtime.safety_reason,
      certificate_id=runtime.certificate.certificate_id,
      prefix_digest=runtime.certificate.prefix_digest,
    )

  def _enter_safe_hold(
    self,
    observation: ExecutorObservation,
    reason: str,
  ) -> ExecutorCommand:
    runtime = self._runtime
    assert runtime is not None
    for name, record in tuple(runtime.participants.items()):
      if record.state is ParticipantState.RUNNING:
        runtime.participants[name] = ParticipantRecord(
          name,
          ParticipantState.CANCELLED,
          reason,
        )
    runtime.state = TransactionState.SAFE_HOLD
    runtime.barrier_state = BarrierState.SAFE_HOLD
    runtime.safety_reason = reason
    runtime.blocked_evidence["GLOBAL"] = reason
    self._revoked_transaction_ids.add(runtime.transaction_id)
    return self._hold_command(observation)

  def _hold_command(self, observation: ExecutorObservation) -> ExecutorCommand:
    runtime = self._runtime
    assert runtime is not None
    positions = np.array(observation.fingertip_positions_m, copy=True)
    active = np.zeros(NUM_FINGERS, dtype=np.bool_)
    positions.setflags(write=False)
    active.setflags(write=False)
    wrist = np.array(observation.wrist_position_m, copy=True)
    wrist.setflags(write=False)
    return ExecutorCommand(
      transaction_id=runtime.transaction_id,
      generation=runtime.generation,
      target_wrist_position_m=wrist,
      target_joint_positions_rad=np.array(
        observation.joint_positions_rad,
        copy=True,
      ),
      nominal_fingertip_positions_m=positions,
      commanded_fingertip_positions_m=positions,
      mcc_active_mask=active,
      participants=tuple(runtime.participants.values()),
      transaction_state=runtime.state,
      barrier_state=runtime.barrier_state,
      safe_hold=runtime.state is TransactionState.SAFE_HOLD,
      safety_reason=runtime.safety_reason,
      certificate_id=runtime.certificate.certificate_id,
      prefix_digest=runtime.certificate.prefix_digest,
    )

  def _snapshot(self, observation: ExecutorObservation) -> BarrierSnapshot:
    runtime = self._runtime
    assert runtime is not None
    return BarrierSnapshot(
      transaction_id=runtime.transaction_id,
      timestamp_s=observation.timestamp_s,
      surface_model_version=observation.surface_model_version,
      wrist_position_m=np.array(observation.wrist_position_m, copy=True),
      fingertip_positions_m=np.array(observation.fingertip_positions_m, copy=True),
      joint_positions_rad=np.array(observation.joint_positions_rad, copy=True),
      fingertip_forces_n=np.array(observation.fingertip_forces_n, copy=True),
      actual_contact_set=observation.actual_contact_set,
      transaction_state=runtime.state,
      participants=tuple(runtime.participants.values()),
      blocked_evidence=dict(runtime.blocked_evidence),
      certificate_id=runtime.certificate.certificate_id,
    )
