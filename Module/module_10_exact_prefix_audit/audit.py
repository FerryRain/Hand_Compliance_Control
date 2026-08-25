"""Exact, swept validation of the only trajectory allowed to execute."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
from time import perf_counter_ns
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from Module.module_1_oracle_surface_model import OracleSurfaceModel
from Module.module_6_prefix_executor import (
  ExecutionCertificate,
  PlannedPrefix,
  PrefixSource,
  TransactionType,
  prefix_digest,
)
from Module.module_6_prefix_executor.executor import _issue_execution_certificate
from Module.module_7_contact_mode_graph import (
  CommitContext,
  ContactModeGraph,
  ContactPrimitive,
  PrimitiveKind,
)
from Module.module_9_continuous_optimize import (
  LinearizedHandKinematics,
  PlannerState,
)


ClearanceFunction = Callable[
  [NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
  float,
]


class AuditVerdict(str, Enum):
  CERTIFIED = "CERTIFIED"
  REJECTED = "REJECTED"


@dataclass(frozen=True, slots=True)
class AuditConfig:
  audit_version: str = "exact-prefix-audit.v1"
  subdivisions_per_segment: int = 9
  minimum_self_collision_clearance_m: float = 0.0
  minimum_link_clearance_m: float = 0.0
  minimum_joint_margin_rad: float = 0.0
  anchor_tolerance_m: float = 0.00075
  kinematic_consistency_tolerance_m: float = 0.0001
  max_commit_displacement_m: float = 0.015
  root_state_tolerance: float = 1e-9

  def __post_init__(self) -> None:
    if not self.audit_version:
      raise ValueError("audit_version must be nonempty")
    if self.subdivisions_per_segment < 3:
      raise ValueError("subdivisions_per_segment must be >=3")
    for name in (
      "minimum_self_collision_clearance_m",
      "minimum_link_clearance_m",
      "minimum_joint_margin_rad",
      "anchor_tolerance_m",
      "kinematic_consistency_tolerance_m",
      "max_commit_displacement_m",
      "root_state_tolerance",
    ):
      value = float(getattr(self, name))
      if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    if (
      self.anchor_tolerance_m == 0.0
      or self.max_commit_displacement_m == 0.0
      or self.root_state_tolerance == 0.0
    ):
      raise ValueError("anchor tolerance and trust radius must be positive")


@dataclass(frozen=True, slots=True)
class AuditEnvironment:
  surface_model: OracleSurfaceModel
  kinematics: LinearizedHandKinematics
  link_clearance_fn: ClearanceFunction
  self_collision_clearance_fn: ClearanceFunction | None = None

  def link_clearance(
    self,
    q: NDArray[np.float64],
    wrist: NDArray[np.float64],
    tips: NDArray[np.float64],
  ) -> float:
    return float(self.link_clearance_fn(q, wrist, tips))

  def self_collision_clearance(
    self,
    q: NDArray[np.float64],
    wrist: NDArray[np.float64],
    tips: NDArray[np.float64],
  ) -> float:
    if self.self_collision_clearance_fn is None:
      return self.kinematics.self_collision_clearance(tips)
    return float(self.self_collision_clearance_fn(q, wrist, tips))


@dataclass(frozen=True, slots=True)
class AuditRequest:
  prefix: PlannedPrefix
  current_state: PlannerState
  commit_context: CommitContext
  issued_at_s: float

  def __post_init__(self) -> None:
    if not np.isfinite(self.issued_at_s) or self.issued_at_s < 0.0:
      raise ValueError("issued_at_s must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class AuditResult:
  verdict: AuditVerdict
  reasons: tuple[str, ...]
  certificate: ExecutionCertificate | None
  latency_s: float
  swept_samples: int
  minimum_joint_margin_rad: float
  minimum_self_collision_clearance_m: float
  minimum_link_clearance_m: float
  maximum_anchor_error_m: float
  maximum_trust_displacement_m: float
  maximum_kinematic_error_m: float

  @property
  def certified(self) -> bool:
    return self.verdict is AuditVerdict.CERTIFIED


class ExactPrefixAuditor:
  """Audit a committed prefix; prediction-only data is rejected explicitly."""

  def __init__(
    self,
    graph: ContactModeGraph,
    environment: AuditEnvironment,
    config: AuditConfig | None = None,
  ) -> None:
    self.graph = graph
    self.environment = environment
    self.config = config or AuditConfig()

  def audit(self, request: AuditRequest) -> AuditResult:
    started = perf_counter_ns()
    prefix = request.prefix
    state = request.current_state
    reasons: list[str] = []
    if not state.contact_set_is_measured:
      reasons.append("PREDICTED_ROOT_HAS_NO_AUDIT_AUTHORITY")
    if prefix.source is PrefixSource.PREDICTION_SUFFIX:
      reasons.append("PREDICTION_SUFFIX_HAS_NO_AUTHORITY")
    if prefix.surface_model_version != self.environment.surface_model.version:
      reasons.append("STALE_SURFACE_MODEL_VERSION")
    if prefix.surface_model_version != state.surface_model_version:
      reasons.append("ROOT_SURFACE_MODEL_VERSION_MISMATCH")
    if prefix.root_contact_set != state.actual_contact_set:
      reasons.append("ROOT_ACTUAL_CONTACT_SET_MISMATCH")
    if request.commit_context.actual_contact_set != state.actual_contact_set:
      reasons.append("COMMIT_CONTEXT_CONTACT_SET_MISMATCH")
    root = prefix.samples[0]
    root_tolerance = self.config.root_state_tolerance
    if root.joint_positions_rad.shape != state.joint_positions_rad.shape or (
      root.joint_positions_rad.shape == state.joint_positions_rad.shape
      and float(np.max(np.abs(root.joint_positions_rad - state.joint_positions_rad)))
      > root_tolerance
    ):
      reasons.append("ROOT_JOINT_STATE_MISMATCH")
    if float(np.linalg.norm(root.wrist_position_m - state.wrist_position_m)) > root_tolerance:
      reasons.append("ROOT_WRIST_STATE_MISMATCH")
    if float(
      np.max(
        np.linalg.norm(
          root.fingertip_positions_m - state.fingertip_positions_m,
          axis=1,
        )
      )
    ) > root_tolerance:
      reasons.append("ROOT_FINGERTIP_STATE_MISMATCH")

    try:
      primitive = ContactPrimitive(prefix.primitive_kind, prefix.finger_id)
    except ValueError:
      primitive = None
      reasons.append("UNKNOWN_PRIMITIVE")
    if primitive is not None:
      commit_legality = self.graph.commit_legal(
        state.mode,
        primitive,
        request.commit_context,
      )
      if not commit_legality.legal:
        reasons.append(commit_legality.reason)
      self._check_phase(prefix, primitive, reasons)
      predicted_terminal = self.graph.apply_predictive(state.mode, primitive).contacts
      make_progress = bool(prefix.metadata.get("make_progress", False))
      expected = state.actual_contact_set if make_progress else predicted_terminal
      if prefix.expected_terminal_contact_set != expected:
        reasons.append("EXPECTED_TERMINAL_MODE_MISMATCH")

    sweep = self._swept_states(prefix)
    min_joint = float("inf")
    min_self = float("inf")
    min_link = float("inf")
    max_anchor = 0.0
    max_trust = 0.0
    max_kinematic = 0.0
    first = prefix.samples[0]
    for index, (q, wrist, planned_tips) in enumerate(sweep):
      if q.shape != self.environment.kinematics.reference_q_rad.shape:
        reasons.append("JOINT_DIMENSION_MISMATCH")
        break
      recomputed_tips = self.environment.kinematics.forward(q, wrist)
      kinematic_error = float(
        np.max(np.linalg.norm(recomputed_tips - planned_tips, axis=1))
      )
      max_kinematic = max(max_kinematic, kinematic_error)
      joint_margin = self.environment.kinematics.joint_margin(q)
      self_clearance = self.environment.self_collision_clearance(q, wrist, recomputed_tips)
      link_clearance = self.environment.link_clearance(q, wrist, recomputed_tips)
      if not np.isfinite(self_clearance) or not np.isfinite(link_clearance):
        reasons.append(f"NONFINITE_CLEARANCE_AT_{index}")
        break
      min_joint = min(min_joint, joint_margin)
      min_self = min(min_self, self_clearance)
      min_link = min(min_link, link_clearance)
      if prefix.anchor_fingers:
        indices = np.asarray([finger - 1 for finger in prefix.anchor_fingers])
        anchor_error = float(
          np.max(
            np.linalg.norm(
              recomputed_tips[indices] - state.fingertip_positions_m[indices],
              axis=1,
            )
          )
        )
        max_anchor = max(max_anchor, anchor_error)
      if prefix.transaction_type is TransactionType.WRIST_ADJUST:
        trust = float(np.linalg.norm(wrist - first.wrist_position_m))
      else:
        trust = max(
          (
            float(
              np.linalg.norm(
                recomputed_tips[finger - 1]
                - first.fingertip_positions_m[finger - 1]
              )
            )
            for finger in prefix.participating_fingers
          ),
          default=0.0,
        )
      max_trust = max(max_trust, trust)

    if min_joint < self.config.minimum_joint_margin_rad - 1e-12:
      reasons.append("SWEPT_JOINT_LIMIT")
    if min_self < self.config.minimum_self_collision_clearance_m - 1e-12:
      reasons.append("SWEPT_SELF_COLLISION")
    if min_link < self.config.minimum_link_clearance_m - 1e-12:
      reasons.append("SWEPT_LINK_COLLISION")
    if max_anchor > self.config.anchor_tolerance_m + 1e-12:
      reasons.append("SWEPT_ANCHOR_ASSUMPTION_BROKEN")
    if max_trust > self.config.max_commit_displacement_m + 1e-12:
      reasons.append("TRUST_REGION_EXCEEDED")
    if max_kinematic > self.config.kinematic_consistency_tolerance_m + 1e-12:
      reasons.append("KINEMATIC_TRAJECTORY_MISMATCH")

    unique_reasons = tuple(dict.fromkeys(reasons))
    certificate = None
    if not unique_reasons:
      digest = prefix_digest(prefix)
      certificate_id = "cert-" + hashlib.sha256(
        f"{digest}:{request.issued_at_s:.9f}:{self.config.audit_version}".encode("utf-8")
      ).hexdigest()[:20]
      certificate = _issue_execution_certificate(
        certificate_id=certificate_id,
        prefix=prefix,
        root_contact_set=state.actual_contact_set,
        audit_version=self.config.audit_version,
        issued_at_s=request.issued_at_s,
      )
    return AuditResult(
      verdict=AuditVerdict.CERTIFIED if certificate is not None else AuditVerdict.REJECTED,
      reasons=unique_reasons,
      certificate=certificate,
      latency_s=(perf_counter_ns() - started) * 1e-9,
      swept_samples=len(sweep),
      minimum_joint_margin_rad=min_joint,
      minimum_self_collision_clearance_m=min_self,
      minimum_link_clearance_m=min_link,
      maximum_anchor_error_m=max_anchor,
      maximum_trust_displacement_m=max_trust,
      maximum_kinematic_error_m=max_kinematic,
    )

  def _check_phase(
    self,
    prefix: PlannedPrefix,
    primitive: ContactPrimitive,
    reasons: list[str],
  ) -> None:
    if primitive.kind is PrimitiveKind.WRIST_ADJUST:
      if prefix.transaction_type is not TransactionType.WRIST_ADJUST:
        reasons.append("WRIST_PRIMITIVE_IN_FINGER_PHASE")
      if prefix.participating_fingers:
        reasons.append("WRIST_AND_FINGER_PHASES_MIXED")
    else:
      if prefix.transaction_type is not TransactionType.FINGER_RECONFIGURE:
        reasons.append("FINGER_PRIMITIVE_IN_WRIST_PHASE")
      if primitive.finger_id not in prefix.participating_fingers:
        reasons.append("PRIMITIVE_FINGER_NOT_PARTICIPATING")
    if prefix.topology_change_count > 1:
      reasons.append("MULTIPLE_TOPOLOGY_CHANGES")

  def _swept_states(
    self,
    prefix: PlannedPrefix,
  ) -> tuple[
    tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    ...,
  ]:
    samples: list[
      tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
    ] = []
    for segment, (left, right) in enumerate(zip(prefix.samples, prefix.samples[1:])):
      alphas = np.linspace(0.0, 1.0, self.config.subdivisions_per_segment)
      if segment:
        alphas = alphas[1:]
      for alpha in alphas:
        q = (1.0 - alpha) * left.joint_positions_rad + alpha * right.joint_positions_rad
        wrist = (1.0 - alpha) * left.wrist_position_m + alpha * right.wrist_position_m
        tips = (
          (1.0 - alpha) * left.fingertip_positions_m
          + alpha * right.fingertip_positions_m
        )
        samples.append((q, wrist, tips))
    return tuple(samples)
