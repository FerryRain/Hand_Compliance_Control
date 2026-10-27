"""Fast, optimistic screening of contact-mode graph edges.

CheapCert intentionally accepts uncertain candidates near the boundary.  It is
a computational filter, never an execution certificate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from time import perf_counter_ns
from typing import Mapping

import numpy as np

from Module.module_7_contact_mode_graph import (
  ContactMode,
  ContactModeGraph,
  ContactPrimitive,
)


class CheapVerdict(str, Enum):
  SURVIVE = "SURVIVE"
  REJECT = "REJECT"


@dataclass(frozen=True, slots=True)
class FailedEdgeEvidence:
  edge_key: str
  reason: str
  surface_model_version: str
  hard_invalidating: bool
  attempted_direction: tuple[float, float, float] | None = None

  def __post_init__(self) -> None:
    if not self.edge_key or not self.reason or not self.surface_model_version:
      raise ValueError("failed-edge evidence requires key, reason, and model version")
    if self.attempted_direction is not None:
      direction = np.asarray(self.attempted_direction, dtype=np.float64)
      if direction.shape != (3,) or not np.all(np.isfinite(direction)):
        raise ValueError("attempted_direction must be a finite 3-vector")


@dataclass(frozen=True, slots=True)
class CheapCertInput:
  mode: ContactMode
  primitive: ContactPrimitive
  surface_model_version: str
  anchor_margin_m: float
  joint_margin_rad: float
  collision_margin_m: float
  reach_margin_m: float
  uncertainty_margin: float
  trust_margin_m: float
  failed_evidence: tuple[FailedEdgeEvidence, ...] = ()
  metadata: Mapping[str, float] = field(default_factory=dict)

  def __post_init__(self) -> None:
    if not self.surface_model_version:
      raise ValueError("surface_model_version must be nonempty")
    for name in (
      "anchor_margin_m",
      "joint_margin_rad",
      "collision_margin_m",
      "reach_margin_m",
      "uncertainty_margin",
      "trust_margin_m",
    ):
      if not np.isfinite(float(getattr(self, name))):
        raise ValueError(f"{name} must be finite")
    metadata = {str(name): float(value) for name, value in self.metadata.items()}
    if any(not np.isfinite(value) for value in metadata.values()):
      raise ValueError("metadata values must be finite")
    object.__setattr__(self, "failed_evidence", tuple(self.failed_evidence))
    object.__setattr__(self, "metadata", metadata)

  @property
  def edge_key(self) -> str:
    return f"{self.mode.mask}:{self.primitive.key}"


@dataclass(frozen=True, slots=True)
class CheapCertConfig:
  """Optimistic rejection bounds chosen to keep false negatives low."""

  reject_anchor_below_m: float = -0.002
  reject_joint_below_rad: float = -0.03
  reject_collision_below_m: float = -0.002
  reject_reach_below_m: float = -0.010
  reject_uncertainty_below: float = 0.0
  reject_trust_below_m: float = -0.005

  def __post_init__(self) -> None:
    for name in (
      "reject_anchor_below_m",
      "reject_joint_below_rad",
      "reject_collision_below_m",
      "reject_reach_below_m",
      "reject_uncertainty_below",
      "reject_trust_below_m",
    ):
      if not np.isfinite(float(getattr(self, name))):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True, slots=True)
class CheapCertResult:
  verdict: CheapVerdict
  reasons: tuple[str, ...]
  margins: Mapping[str, float]
  edge_key: str
  surface_model_version: str
  latency_s: float
  execution_authority: bool = False

  @property
  def survived(self) -> bool:
    return self.verdict is CheapVerdict.SURVIVE


class CheapCert:
  """Deterministic graph/margin/evidence screen."""

  def __init__(
    self,
    graph: ContactModeGraph,
    config: CheapCertConfig | None = None,
  ) -> None:
    self.graph = graph
    self.config = config or CheapCertConfig()

  def screen(self, candidate: CheapCertInput) -> CheapCertResult:
    started = perf_counter_ns()
    reasons: list[str] = []
    graph_legality = self.graph.predict_legal(candidate.mode, candidate.primitive)
    if not graph_legality.legal:
      reasons.append(graph_legality.reason)

    matching_evidence = sorted(
      (
        evidence
        for evidence in candidate.failed_evidence
        if evidence.edge_key == candidate.edge_key
        and evidence.surface_model_version == candidate.surface_model_version
      ),
      key=lambda evidence: (not evidence.hard_invalidating, evidence.reason),
    )
    reasons.extend(
      f"FAILED_EDGE_{evidence.reason}"
      + ("_HARD" if evidence.hard_invalidating else "_LOCAL")
      for evidence in matching_evidence
    )

    checks = (
      (
        "ANCHOR_MARGIN_CLEARLY_INFEASIBLE",
        candidate.anchor_margin_m,
        self.config.reject_anchor_below_m,
      ),
      (
        "JOINT_MARGIN_CLEARLY_INFEASIBLE",
        candidate.joint_margin_rad,
        self.config.reject_joint_below_rad,
      ),
      (
        "COLLISION_MARGIN_CLEARLY_INFEASIBLE",
        candidate.collision_margin_m,
        self.config.reject_collision_below_m,
      ),
      (
        "REACH_MARGIN_CLEARLY_INFEASIBLE",
        candidate.reach_margin_m,
        self.config.reject_reach_below_m,
      ),
      (
        "UNCERTAINTY_BUDGET_EXCEEDED",
        candidate.uncertainty_margin,
        self.config.reject_uncertainty_below,
      ),
      (
        "TRUST_REGION_CLEARLY_EXCEEDED",
        candidate.trust_margin_m,
        self.config.reject_trust_below_m,
      ),
    )
    reasons.extend(name for name, value, threshold in checks if value < threshold)
    latency_s = (perf_counter_ns() - started) * 1e-9
    margins = {
      "m_anchor": float(candidate.anchor_margin_m),
      "m_joint": float(candidate.joint_margin_rad),
      "m_collision": float(candidate.collision_margin_m),
      "m_reach": float(candidate.reach_margin_m),
      "m_uncertainty": float(candidate.uncertainty_margin),
      "m_trust": float(candidate.trust_margin_m),
    }
    return CheapCertResult(
      verdict=CheapVerdict.REJECT if reasons else CheapVerdict.SURVIVE,
      reasons=tuple(reasons),
      margins=margins,
      edge_key=candidate.edge_key,
      surface_model_version=candidate.surface_model_version,
      latency_s=latency_s,
    )
