"""Versioned Oracle SurfaceModel built on analytic ground-truth shapes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from Module.module_1_oracle_surface_model.geometry import (
  AnalyticShape,
  SurfaceProjection,
  _vector3,
)


@dataclass(frozen=True, slots=True)
class CapsuleLink:
  """A link represented as a centerline segment swept by a sphere."""

  start: ArrayLike
  end: ArrayLike
  radius: float
  name: str = "link"

  def __post_init__(self) -> None:
    if self.radius < 0.0:
      raise ValueError("capsule radius must be non-negative")
    object.__setattr__(self, "start", _vector3(self.start, "start"))
    object.__setattr__(self, "end", _vector3(self.end, "end"))


@dataclass(frozen=True, slots=True)
class ClearanceResult:
  clearance: float
  link_name: str
  segment_index: int
  alpha: float
  link_point: NDArray[np.float64]
  surface_point: NDArray[np.float64]
  surface_normal: NDArray[np.float64]
  model_version: str


@dataclass(frozen=True, slots=True)
class ContactCandidateRequest:
  finger_id: int
  workspace_center: ArrayLike
  reach_radius: float
  count: int = 8
  seed: int = 7
  oversample_factor: int = 128

  def __post_init__(self) -> None:
    if self.finger_id <= 0:
      raise ValueError("finger_id must be one-based and positive")
    if self.reach_radius <= 0.0:
      raise ValueError("reach_radius must be positive")
    if self.count <= 0 or self.oversample_factor < 1:
      raise ValueError("count and oversample_factor must be positive")
    object.__setattr__(
      self,
      "workspace_center",
      _vector3(self.workspace_center, "workspace_center"),
    )


@dataclass(frozen=True, slots=True)
class ContactCandidate:
  finger_id: int
  position: NDArray[np.float64]
  outward_normal: NDArray[np.float64]
  reach_distance: float
  model_version: str


class OracleSurfaceModel:
  """Immutable, zero-uncertainty geometry oracle.

  The capsule query searches the analytic signed-distance field along each
  centerline. A deterministic coarse scan locates the best basin and a golden
  section search refines that interval.
  """

  def __init__(
    self,
    shape: AnalyticShape,
    *,
    version: str,
    clearance_samples: int = 257,
    refinement_iterations: int = 64,
  ) -> None:
    if not version:
      raise ValueError("version must be non-empty")
    if clearance_samples < 3 or clearance_samples % 2 == 0:
      raise ValueError("clearance_samples must be an odd integer >= 3")
    if refinement_iterations < 1:
      raise ValueError("refinement_iterations must be positive")
    self._shape = shape
    self._version = version
    self._clearance_samples = clearance_samples
    self._refinement_iterations = refinement_iterations

  @property
  def version(self) -> str:
    return self._version

  @property
  def shape(self) -> AnalyticShape:
    return self._shape

  def query_surface(self, point: ArrayLike) -> SurfaceProjection:
    return self._shape.query(point)

  def query_normal(self, point: ArrayLike) -> NDArray[np.float64]:
    return self._shape.query(point).normal.copy()

  def query_uncertainty(self, point: ArrayLike) -> float:
    _vector3(point, "point")
    return 0.0

  def query_clearance(
    self,
    link_or_swept_geometry: CapsuleLink | Sequence[CapsuleLink],
  ) -> ClearanceResult:
    if isinstance(link_or_swept_geometry, CapsuleLink):
      links = (link_or_swept_geometry,)
    else:
      links = tuple(link_or_swept_geometry)
    if not links:
      raise ValueError("at least one capsule link is required")

    results = [self._query_single_capsule(link, index) for index, link in enumerate(links)]
    return min(results, key=lambda result: result.clearance)

  def _query_single_capsule(self, link: CapsuleLink, segment_index: int) -> ClearanceResult:
    direction = link.end - link.start

    def evaluate(alpha: float) -> float:
      point = link.start + alpha * direction
      return self._shape.signed_distance(point) - link.radius

    if float(np.linalg.norm(direction)) <= 1e-15:
      best_alpha = 0.0
    else:
      alphas = np.linspace(0.0, 1.0, self._clearance_samples)
      values = np.array([evaluate(float(alpha)) for alpha in alphas])
      best_index = int(np.argmin(values))
      lower_index = max(0, best_index - 1)
      upper_index = min(self._clearance_samples - 1, best_index + 1)
      lower = float(alphas[lower_index])
      upper = float(alphas[upper_index])
      best_alpha = self._golden_section_minimum(evaluate, lower, upper)
      candidates = [
        (float(values[best_index]), float(alphas[best_index])),
        (evaluate(best_alpha), best_alpha),
        (evaluate(0.0), 0.0),
        (evaluate(1.0), 1.0),
      ]
      _, best_alpha = min(candidates, key=lambda item: item[0])

    link_point = link.start + best_alpha * direction
    query = self._shape.query(link_point)
    return ClearanceResult(
      clearance=query.signed_distance - link.radius,
      link_name=link.name,
      segment_index=segment_index,
      alpha=best_alpha,
      link_point=link_point,
      surface_point=query.point,
      surface_normal=query.normal,
      model_version=self._version,
    )

  def _golden_section_minimum(self, function, lower: float, upper: float) -> float:
    if upper <= lower:
      return lower
    ratio = (np.sqrt(5.0) - 1.0) / 2.0
    left = upper - ratio * (upper - lower)
    right = lower + ratio * (upper - lower)
    left_value = function(left)
    right_value = function(right)
    for _ in range(self._refinement_iterations):
      if left_value <= right_value:
        upper = right
        right = left
        right_value = left_value
        left = upper - ratio * (upper - lower)
        left_value = function(left)
      else:
        lower = left
        left = right
        left_value = right_value
        right = lower + ratio * (upper - lower)
        right_value = function(right)
    return 0.5 * (lower + upper)

  def sample_contact_candidates(
    self,
    request: ContactCandidateRequest,
  ) -> list[ContactCandidate]:
    rng = np.random.default_rng(request.seed)
    sample_count = max(request.count * request.oversample_factor, request.count)
    points = self._shape.sample_surface(sample_count, rng)
    distances = np.linalg.norm(points - request.workspace_center[None, :], axis=1)
    reachable = np.flatnonzero(distances <= request.reach_radius + 1e-12)
    ordered = reachable[np.argsort(distances[reachable], kind="stable")]

    candidates: list[ContactCandidate] = []
    for index in ordered[: request.count]:
      query = self._shape.query(points[index])
      candidates.append(
        ContactCandidate(
          finger_id=request.finger_id,
          position=query.point,
          outward_normal=query.normal,
          reach_distance=float(distances[index]),
          model_version=self._version,
        )
      )
    return candidates
