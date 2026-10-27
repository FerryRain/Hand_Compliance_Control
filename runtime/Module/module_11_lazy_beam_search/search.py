"""Lazy receding-horizon beam search with mode-diversity quotas."""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter_ns
from typing import Callable, Protocol

import numpy as np

from Module.module_6_prefix_executor import PlannedPrefix, PrefixSource
from Module.module_7_contact_mode_graph import (
  ContactMode,
  ContactModeGraph,
  ContactPrimitive,
)
from Module.module_8_cheap_cert import CheapCert, CheapCertInput
from Module.module_9_continuous_optimize import (
  ContinuousOptimizer,
  OptimizationRequest,
  OptimizationResult,
  PlannerState,
)


@dataclass(frozen=True, slots=True)
class PlanningCandidate:
  cheap_input: CheapCertInput
  optimization_request: OptimizationRequest
  motion_cost: float = 0.0
  risk_cost: float = 0.0

  def __post_init__(self) -> None:
    if not np.isfinite(self.motion_cost) or self.motion_cost < 0.0:
      raise ValueError("motion_cost must be finite and non-negative")
    if not np.isfinite(self.risk_cost) or self.risk_cost < 0.0:
      raise ValueError("risk_cost must be finite and non-negative")


class CandidateFactory(Protocol):
  def __call__(
    self,
    state: PlannerState,
    primitive: ContactPrimitive,
    depth: int,
  ) -> PlanningCandidate | None: ...


@dataclass(frozen=True, slots=True)
class SearchWeights:
  progress: float = 8.0
  contact: float = 0.20
  motion: float = 1.0
  risk: float = 1.5
  switch: float = 0.08

  def __post_init__(self) -> None:
    for name in ("progress", "contact", "motion", "risk", "switch"):
      value = float(getattr(self, name))
      if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class BeamSearchConfig:
  horizon: int = 3
  beam_width: int = 8
  per_mode_quota: int = 2

  def __post_init__(self) -> None:
    if self.horizon < 1 or self.beam_width < 1 or self.per_mode_quota < 1:
      raise ValueError("horizon, beam_width, and per_mode_quota must be positive")


@dataclass(frozen=True, slots=True)
class SearchNode:
  state: PlannerState
  predicted_mode: ContactMode
  primitives: tuple[ContactPrimitive, ...]
  edge_prefixes: tuple[PlannedPrefix, ...]
  score: float
  progress_score: float
  contact_score: float
  motion_cost: float
  risk_cost: float
  switch_cost: float
  warm_start_matches: int = 0

  @property
  def sequence_key(self) -> tuple[str, ...]:
    return tuple(primitive.key for primitive in self.primitives)


@dataclass(frozen=True, slots=True)
class SearchResult:
  best_node: SearchNode | None
  committed_prefix_candidate: PlannedPrefix | None
  prediction_suffix: tuple[PlannedPrefix, ...]
  latency_s: float
  expanded_nodes: int
  enumerated_edges: int
  cheap_survivors: int
  optimized_edges: int
  retained_nodes_per_depth: tuple[int, ...]
  distinct_modes_per_depth: tuple[int, ...]
  shifted_suffix_matches: int

  @property
  def found(self) -> bool:
    return self.best_node is not None


TerminalViability = Callable[[PlannerState], bool]


class LazyBeamSearch:
  """Search predictive suffixes while exposing only edge zero for audit."""

  def __init__(
    self,
    graph: ContactModeGraph,
    cheap_cert: CheapCert,
    optimizer: ContinuousOptimizer,
    config: BeamSearchConfig | None = None,
    weights: SearchWeights | None = None,
  ) -> None:
    self.graph = graph
    self.cheap_cert = cheap_cert
    self.optimizer = optimizer
    self.config = config or BeamSearchConfig()
    self.weights = weights or SearchWeights()

  def search(
    self,
    root_state: PlannerState,
    candidate_factory: CandidateFactory,
    *,
    shifted_suffix: tuple[ContactPrimitive, ...] = (),
    terminal_viability: TerminalViability | None = None,
  ) -> SearchResult:
    started = perf_counter_ns()
    if not root_state.contact_set_is_measured:
      raise ValueError("beam root must be a measured barrier snapshot")
    root = SearchNode(
      state=root_state,
      predicted_mode=root_state.mode,
      primitives=(),
      edge_prefixes=(),
      score=0.0,
      progress_score=0.0,
      contact_score=0.0,
      motion_cost=0.0,
      risk_cost=0.0,
      switch_cost=0.0,
    )
    beam = [root]
    expanded_nodes = 0
    enumerated_edges = 0
    cheap_survivors = 0
    optimized_edges = 0
    retained_per_depth: list[int] = []
    modes_per_depth: list[int] = []
    total_warm_matches = 0
    for depth in range(self.config.horizon):
      children: list[SearchNode] = []
      warm_primitive = shifted_suffix[depth] if depth < len(shifted_suffix) else None
      for node in beam:
        expanded_nodes += 1
        edges = list(self.graph.edges_from(node.predicted_mode))
        edges.sort(
          key=lambda edge: (
            0 if warm_primitive is not None and edge.primitive == warm_primitive else 1,
            edge.primitive.key,
            edge.target.mask,
          )
        )
        for edge in edges:
          enumerated_edges += 1
          candidate = candidate_factory(node.state, edge.primitive, depth)
          if candidate is None:
            continue
          if candidate.cheap_input.mode != node.predicted_mode:
            raise ValueError("candidate cheap-input mode differs from search node")
          if candidate.optimization_request.primitive != edge.primitive:
            raise ValueError("candidate optimizer primitive differs from graph edge")
          cheap = self.cheap_cert.screen(candidate.cheap_input)
          if not cheap.survived:
            continue
          cheap_survivors += 1
          optimized = self.optimizer.optimize(candidate.optimization_request)
          if not optimized.feasible or optimized.prefix is None:
            continue
          optimized_edges += 1
          prefix = optimized.prefix
          if depth > 0:
            prefix = replace(prefix, source=PrefixSource.PREDICTION_SUFFIX)
          terminal_sample = prefix.samples[-1]
          child_state = PlannerState(
            joint_positions_rad=terminal_sample.joint_positions_rad,
            wrist_position_m=terminal_sample.wrist_position_m,
            fingertip_positions_m=terminal_sample.fingertip_positions_m,
            actual_contact_set=prefix.expected_terminal_contact_set,
            surface_model_version=prefix.surface_model_version,
            contact_set_authority="PREDICTED",
          )
          progress = float(optimized.achieved_progress_m)
          contact = len(prefix.expected_terminal_contact_set) / 4.0
          motion = float(candidate.motion_cost)
          risk = float(candidate.risk_cost)
          switch = float(edge.primitive.topology_change_count)
          increment = (
            self.weights.progress * progress
            + self.weights.contact * contact
            - self.weights.motion * motion
            - self.weights.risk * risk
            - self.weights.switch * switch
          )
          warm_match = int(warm_primitive is not None and edge.primitive == warm_primitive)
          total_warm_matches += warm_match
          children.append(
            SearchNode(
              state=child_state,
              predicted_mode=ContactMode(prefix.expected_terminal_contact_set),
              primitives=node.primitives + (edge.primitive,),
              edge_prefixes=node.edge_prefixes + (prefix,),
              score=node.score + increment,
              progress_score=node.progress_score + progress,
              contact_score=node.contact_score + contact,
              motion_cost=node.motion_cost + motion,
              risk_cost=node.risk_cost + risk,
              switch_cost=node.switch_cost + switch,
              warm_start_matches=node.warm_start_matches + warm_match,
            )
          )
      if not children:
        beam = []
        retained_per_depth.append(0)
        modes_per_depth.append(0)
        break
      children.sort(key=lambda node: (-node.score, node.sequence_key))
      retained: list[SearchNode] = []
      mode_counts: dict[int, int] = {}
      for child in children:
        mode = child.predicted_mode.mask
        if mode_counts.get(mode, 0) >= self.config.per_mode_quota:
          continue
        if depth == self.config.horizon - 1 and terminal_viability is not None:
          if not terminal_viability(child.state):
            continue
        retained.append(child)
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
        if len(retained) >= self.config.beam_width:
          break
      beam = retained
      retained_per_depth.append(len(beam))
      modes_per_depth.append(len(mode_counts))
      if not beam:
        break

    best = min(beam, key=lambda node: (-node.score, node.sequence_key)) if beam else None
    committed = None if best is None else best.edge_prefixes[0]
    suffix = () if best is None else best.edge_prefixes[1:]
    if committed is not None and committed.source is PrefixSource.PREDICTION_SUFFIX:
      raise AssertionError("edge zero must remain an audit candidate, never a suffix")
    if any(prefix.source is not PrefixSource.PREDICTION_SUFFIX for prefix in suffix):
      raise AssertionError("all nonzero-horizon prefixes must remain prediction-only")
    return SearchResult(
      best_node=best,
      committed_prefix_candidate=committed,
      prediction_suffix=suffix,
      latency_s=(perf_counter_ns() - started) * 1e-9,
      expanded_nodes=expanded_nodes,
      enumerated_edges=enumerated_edges,
      cheap_survivors=cheap_survivors,
      optimized_edges=optimized_edges,
      retained_nodes_per_depth=tuple(retained_per_depth),
      distinct_modes_per_depth=tuple(modes_per_depth),
      shifted_suffix_matches=total_warm_matches,
    )
