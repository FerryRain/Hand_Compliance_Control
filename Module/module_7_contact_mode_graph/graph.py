"""Finite contact-mode graph with separate predictive and commit legality."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Sequence


NUM_FINGERS = 4


class PrimitiveKind(str, Enum):
  WRIST_ADJUST = "WRIST_ADJUST"
  SLIDE = "SLIDE"
  REPOSITION = "REPOSITION"
  MAKE = "MAKE"
  BREAK = "BREAK"


@dataclass(frozen=True, slots=True)
class ContactMode:
  """One of the fifteen nonempty four-finger contact sets."""

  contacts: frozenset[int]

  def __post_init__(self) -> None:
    contacts = frozenset(int(finger) for finger in self.contacts)
    if not contacts or any(finger < 1 or finger > NUM_FINGERS for finger in contacts):
      raise ValueError("contacts must be a nonempty subset of {1,2,3,4}")
    object.__setattr__(self, "contacts", contacts)

  @property
  def mask(self) -> int:
    return sum(1 << (finger - 1) for finger in self.contacts)

  @classmethod
  def from_mask(cls, mask: int) -> ContactMode:
    if mask < 1 or mask >= 1 << NUM_FINGERS:
      raise ValueError("mask must identify one of the fifteen nonempty modes")
    return cls(frozenset(finger for finger in range(1, NUM_FINGERS + 1) if mask & (1 << (finger - 1))))


@dataclass(frozen=True, slots=True)
class ContactPrimitive:
  kind: PrimitiveKind | str
  finger_id: int | None = None

  def __post_init__(self) -> None:
    kind = PrimitiveKind(self.kind)
    if kind is PrimitiveKind.WRIST_ADJUST:
      if self.finger_id is not None:
        raise ValueError("WRIST_ADJUST has no finger_id")
    elif self.finger_id is None or self.finger_id < 1 or self.finger_id > NUM_FINGERS:
      raise ValueError(f"{kind.value} requires finger_id in [1,4]")
    object.__setattr__(self, "kind", kind)

  @property
  def topology_change_count(self) -> int:
    return int(self.kind in {PrimitiveKind.MAKE, PrimitiveKind.BREAK})

  @property
  def transaction_family(self) -> str:
    return "WRIST" if self.kind is PrimitiveKind.WRIST_ADJUST else "FINGER"

  @property
  def key(self) -> str:
    return self.kind.value if self.finger_id is None else f"{self.kind.value}({self.finger_id})"


@dataclass(frozen=True, slots=True)
class LegalityResult:
  legal: bool
  reason: str


@dataclass(frozen=True, slots=True)
class GraphEdge:
  source: ContactMode
  primitive: ContactPrimitive
  target: ContactMode

  @property
  def key(self) -> str:
    return f"{self.source.mask}:{self.primitive.key}:{self.target.mask}"


@dataclass(frozen=True, slots=True)
class CommitContext:
  """Measured facts required in addition to graph-predictive legality."""

  actual_contact_set: frozenset[int] | set[int] | tuple[int, ...]
  replacement_confirmation_s: Mapping[int, float] = field(default_factory=dict)
  minimum_confirmation_s: float = 0.05

  def __post_init__(self) -> None:
    actual = frozenset(int(finger) for finger in self.actual_contact_set)
    if not actual or any(finger < 1 or finger > NUM_FINGERS for finger in actual):
      raise ValueError("actual_contact_set must be a nonempty valid mode")
    confirmations = {
      int(finger): float(duration)
      for finger, duration in self.replacement_confirmation_s.items()
    }
    if any(finger < 1 or finger > NUM_FINGERS for finger in confirmations):
      raise ValueError("replacement confirmation contains an invalid finger")
    if any(duration < 0.0 for duration in confirmations.values()):
      raise ValueError("replacement confirmation durations must be non-negative")
    if self.minimum_confirmation_s <= 0.0:
      raise ValueError("minimum_confirmation_s must be positive")
    object.__setattr__(self, "actual_contact_set", actual)
    object.__setattr__(self, "replacement_confirmation_s", confirmations)


class ContactModeGraph:
  """Complete, deterministic graph over all nonempty four-finger modes."""

  def __init__(self) -> None:
    self._modes = tuple(ContactMode.from_mask(mask) for mask in range(1, 1 << NUM_FINGERS))

  @property
  def modes(self) -> tuple[ContactMode, ...]:
    return self._modes

  @property
  def primitives(self) -> tuple[ContactPrimitive, ...]:
    result = [ContactPrimitive(PrimitiveKind.WRIST_ADJUST)]
    for kind in (
      PrimitiveKind.SLIDE,
      PrimitiveKind.REPOSITION,
      PrimitiveKind.MAKE,
      PrimitiveKind.BREAK,
    ):
      result.extend(ContactPrimitive(kind, finger) for finger in range(1, NUM_FINGERS + 1))
    return tuple(result)

  def predict_legal(
    self,
    mode: ContactMode,
    primitive: ContactPrimitive,
  ) -> LegalityResult:
    contacts = mode.contacts
    finger = primitive.finger_id
    if primitive.kind is PrimitiveKind.WRIST_ADJUST:
      return LegalityResult(True, "LEGAL")
    assert finger is not None
    if primitive.kind is PrimitiveKind.SLIDE:
      return (
        LegalityResult(True, "LEGAL")
        if finger in contacts
        else LegalityResult(False, "SLIDE_REQUIRES_CONTACT")
      )
    if primitive.kind is PrimitiveKind.REPOSITION:
      return (
        LegalityResult(True, "LEGAL")
        if finger not in contacts
        else LegalityResult(False, "REPOSITION_REQUIRES_FREE_FINGER")
      )
    if primitive.kind is PrimitiveKind.MAKE:
      return (
        LegalityResult(True, "LEGAL")
        if finger not in contacts
        else LegalityResult(False, "MAKE_REQUIRES_FREE_FINGER")
      )
    if primitive.kind is PrimitiveKind.BREAK:
      if finger not in contacts:
        return LegalityResult(False, "BREAK_REQUIRES_CONTACT")
      if len(contacts) == 1:
        return LegalityResult(False, "EMPTY_CONTACT_MODE_FORBIDDEN")
      return LegalityResult(True, "LEGAL")
    raise AssertionError("unhandled primitive")

  def apply_predictive(
    self,
    mode: ContactMode,
    primitive: ContactPrimitive,
  ) -> ContactMode:
    legality = self.predict_legal(mode, primitive)
    if not legality.legal:
      raise ValueError(f"illegal predictive edge: {legality.reason}")
    contacts = set(mode.contacts)
    if primitive.kind is PrimitiveKind.MAKE:
      assert primitive.finger_id is not None
      contacts.add(primitive.finger_id)
    elif primitive.kind is PrimitiveKind.BREAK:
      assert primitive.finger_id is not None
      contacts.remove(primitive.finger_id)
    return ContactMode(frozenset(contacts))

  def edges_from(self, mode: ContactMode) -> tuple[GraphEdge, ...]:
    edges: list[GraphEdge] = []
    for primitive in self.primitives:
      if self.predict_legal(mode, primitive).legal:
        edges.append(GraphEdge(mode, primitive, self.apply_predictive(mode, primitive)))
    return tuple(edges)

  def commit_legal(
    self,
    mode: ContactMode,
    primitive: ContactPrimitive,
    context: CommitContext,
  ) -> LegalityResult:
    predictive = self.predict_legal(mode, primitive)
    if not predictive.legal:
      return predictive
    if context.actual_contact_set != mode.contacts:
      return LegalityResult(False, "ACTUAL_CONTACT_SET_MISMATCH")
    if primitive.kind is not PrimitiveKind.BREAK:
      return LegalityResult(True, "LEGAL")
    assert primitive.finger_id is not None
    remaining = context.actual_contact_set - {primitive.finger_id}
    confirmed = {
      finger
      for finger, duration in context.replacement_confirmation_s.items()
      if duration + 1e-12 >= context.minimum_confirmation_s
    }
    if not remaining & confirmed:
      return LegalityResult(False, "REPLACEMENT_CONTACT_NOT_CONFIRMED")
    return LegalityResult(True, "LEGAL")

  def validate_prefix(
    self,
    root: ContactMode,
    primitives: Sequence[ContactPrimitive],
  ) -> LegalityResult:
    sequence = tuple(primitives)
    if not sequence:
      return LegalityResult(False, "EMPTY_PREFIX")
    families = {primitive.transaction_family for primitive in sequence}
    if len(families) != 1:
      return LegalityResult(False, "WRIST_AND_FINGER_PHASES_MIXED")
    topology = [primitive.kind for primitive in sequence if primitive.topology_change_count]
    if len(topology) > 1:
      return LegalityResult(False, "MULTIPLE_TOPOLOGY_CHANGES")
    if PrimitiveKind.MAKE in topology and PrimitiveKind.BREAK in topology:
      return LegalityResult(False, "MAKE_AND_BREAK_MIXED")
    mode = root
    for primitive in sequence:
      legality = self.predict_legal(mode, primitive)
      if not legality.legal:
        return legality
      mode = self.apply_predictive(mode, primitive)
    return LegalityResult(True, "LEGAL")
