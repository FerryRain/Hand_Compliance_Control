"""Cheap one-step successor test used only to reject terminal dead ends."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from time import perf_counter_ns
from typing import Callable

from Module.module_7_contact_mode_graph import (
  ContactModeGraph,
  ContactPrimitive,
  PrimitiveKind,
)
from Module.module_8_cheap_cert import CheapCert
from Module.module_9_continuous_optimize import PlannerState
from Module.module_11_lazy_beam_search import CandidateFactory


class ShadowStatus(str, Enum):
  VIABLE = "VIABLE"
  NONVIABLE = "NONVIABLE"


@dataclass(frozen=True, slots=True)
class ShadowSuccessor:
  primitive: ContactPrimitive
  target_contact_set: frozenset[int]
  edge_key: str
  successor_finger: int | None


@dataclass(frozen=True, slots=True)
class ShadowResult:
  status: ShadowStatus
  successors: tuple[ShadowSuccessor, ...]
  distinct_successor_fingers: tuple[int, ...]
  tested_edges: int
  cheap_survivors: int
  latency_s: float
  reason: str
  execution_authority: bool = False

  @property
  def viable(self) -> bool:
    return self.status is ShadowStatus.VIABLE


class ShadowViabilityEvaluator:
  """Require a nontrivial cheap continuation without optimizing or certifying."""

  def __init__(self, graph: ContactModeGraph, cheap_cert: CheapCert) -> None:
    self.graph = graph
    self.cheap_cert = cheap_cert

  def evaluate(
    self,
    terminal_state: PlannerState,
    candidate_factory: CandidateFactory,
  ) -> ShadowResult:
    started = perf_counter_ns()
    singleton = len(terminal_state.actual_contact_set) == 1
    successors: list[ShadowSuccessor] = []
    tested = 0
    cheap_survivors = 0
    for edge in self.graph.edges_from(terminal_state.mode):
      primitive = edge.primitive
      if primitive.kind is PrimitiveKind.WRIST_ADJUST:
        continue
      if singleton and primitive.kind is not PrimitiveKind.MAKE:
        continue
      tested += 1
      candidate = candidate_factory(terminal_state, primitive, 0)
      if candidate is None:
        continue
      if candidate.cheap_input.mode != terminal_state.mode:
        raise ValueError("shadow candidate mode differs from terminal mode")
      screen = self.cheap_cert.screen(candidate.cheap_input)
      if not screen.survived:
        continue
      cheap_survivors += 1
      successors.append(
        ShadowSuccessor(
          primitive=primitive,
          target_contact_set=edge.target.contacts,
          edge_key=screen.edge_key,
          successor_finger=primitive.finger_id,
        )
      )
    fingers = tuple(
      sorted(
        {
          successor.successor_finger
          for successor in successors
          if successor.successor_finger is not None
        }
      )
    )
    viable = bool(successors)
    if viable:
      reason = "CHEAP_NONTRIVIAL_SUCCESSOR_EXISTS"
    elif singleton:
      reason = "SINGLETON_HAS_NO_CHEAP_FEASIBLE_MAKE"
    else:
      reason = "NO_CHEAP_NONTRIVIAL_SUCCESSOR"
    return ShadowResult(
      status=ShadowStatus.VIABLE if viable else ShadowStatus.NONVIABLE,
      successors=tuple(successors),
      distinct_successor_fingers=fingers,
      tested_edges=tested,
      cheap_survivors=cheap_survivors,
      latency_s=(perf_counter_ns() - started) * 1e-9,
      reason=reason,
    )

  def predicate(
    self,
    candidate_factory: CandidateFactory,
  ) -> Callable[[PlannerState], bool]:
    return lambda state: self.evaluate(state, candidate_factory).viable
