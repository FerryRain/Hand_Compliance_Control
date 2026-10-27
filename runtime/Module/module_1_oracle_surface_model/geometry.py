"""Analytic ground-truth shapes used by the Oracle SurfaceModel."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


_EPS = 1e-12


def _vector3(value: ArrayLike, name: str) -> NDArray[np.float64]:
  array = np.asarray(value, dtype=np.float64)
  if array.shape != (3,) or not np.all(np.isfinite(array)):
    raise ValueError(f"{name} must be a finite vector with shape (3,)")
  result = np.array(array, copy=True)
  result.setflags(write=False)
  return result


def _unit_vector(value: ArrayLike, name: str) -> NDArray[np.float64]:
  vector = _vector3(value, name)
  norm = float(np.linalg.norm(vector))
  if norm <= _EPS:
    raise ValueError(f"{name} must be non-zero")
  result = vector / norm
  result.setflags(write=False)
  return result


def _rotation_matrix(value: ArrayLike | None) -> NDArray[np.float64]:
  if value is None:
    result = np.eye(3, dtype=np.float64)
    result.setflags(write=False)
    return result
  rotation = np.asarray(value, dtype=np.float64)
  if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
    raise ValueError("rotation must be a finite 3x3 matrix")
  if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-9, rtol=0.0):
    raise ValueError("rotation must be orthonormal")
  if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-9, rtol=0.0):
    raise ValueError("rotation must be right-handed")
  result = np.array(rotation, copy=True)
  result.setflags(write=False)
  return result


def _tangent_basis(normal: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
  helper = np.array([1.0, 0.0, 0.0])
  if abs(float(np.dot(helper, normal))) > 0.9:
    helper = np.array([0.0, 1.0, 0.0])
  tangent_1 = np.cross(normal, helper)
  tangent_1 /= np.linalg.norm(tangent_1)
  tangent_2 = np.cross(normal, tangent_1)
  return tangent_1, tangent_2


@dataclass(frozen=True, slots=True)
class SurfaceProjection:
  """Closest surface point, outward normal, and signed distance."""

  point: NDArray[np.float64]
  normal: NDArray[np.float64]
  signed_distance: float


class AnalyticShape(ABC):
  """Interface implemented by exact analytic object geometries."""

  @abstractmethod
  def query(self, point: ArrayLike) -> SurfaceProjection:
    """Query closest surface point, outward normal, and signed distance."""

  @abstractmethod
  def sample_surface(
    self,
    count: int,
    rng: np.random.Generator,
  ) -> NDArray[np.float64]:
    """Return deterministic-on-seed surface samples in world coordinates."""

  def signed_distance(self, point: ArrayLike) -> float:
    return self.query(point).signed_distance


@dataclass(frozen=True, slots=True)
class Plane(AnalyticShape):
  point: ArrayLike
  normal: ArrayLike
  sample_extent: float = 0.15

  def __post_init__(self) -> None:
    if self.sample_extent <= 0.0:
      raise ValueError("sample_extent must be positive")
    object.__setattr__(self, "point", _vector3(self.point, "point"))
    object.__setattr__(self, "normal", _unit_vector(self.normal, "normal"))

  def query(self, point: ArrayLike) -> SurfaceProjection:
    x = _vector3(point, "point")
    signed_distance = float(np.dot(x - self.point, self.normal))
    closest = x - signed_distance * self.normal
    return SurfaceProjection(closest, self.normal.copy(), signed_distance)

  def sample_surface(
    self,
    count: int,
    rng: np.random.Generator,
  ) -> NDArray[np.float64]:
    if count <= 0:
      raise ValueError("count must be positive")
    tangent_1, tangent_2 = _tangent_basis(self.normal)
    coordinates = rng.uniform(-self.sample_extent, self.sample_extent, size=(count, 2))
    return (
      self.point[None, :]
      + coordinates[:, :1] * tangent_1[None, :]
      + coordinates[:, 1:] * tangent_2[None, :]
    )


@dataclass(frozen=True, slots=True)
class Sphere(AnalyticShape):
  center: ArrayLike
  radius: float

  def __post_init__(self) -> None:
    if self.radius <= 0.0:
      raise ValueError("radius must be positive")
    object.__setattr__(self, "center", _vector3(self.center, "center"))

  def query(self, point: ArrayLike) -> SurfaceProjection:
    x = _vector3(point, "point")
    radial = x - self.center
    distance = float(np.linalg.norm(radial))
    if distance <= _EPS:
      normal = np.array([1.0, 0.0, 0.0])
    else:
      normal = radial / distance
    closest = self.center + self.radius * normal
    return SurfaceProjection(closest, normal, distance - self.radius)

  def sample_surface(
    self,
    count: int,
    rng: np.random.Generator,
  ) -> NDArray[np.float64]:
    if count <= 0:
      raise ValueError("count must be positive")
    directions = rng.normal(size=(count, 3))
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    while np.any(norms <= _EPS):
      mask = norms[:, 0] <= _EPS
      directions[mask] = rng.normal(size=(int(np.sum(mask)), 3))
      norms = np.linalg.norm(directions, axis=1, keepdims=True)
    return self.center[None, :] + self.radius * directions / norms


@dataclass(frozen=True, slots=True)
class Cylinder(AnalyticShape):
  """Finite, capped cylinder."""

  center: ArrayLike
  axis: ArrayLike
  radius: float
  half_length: float

  def __post_init__(self) -> None:
    if self.radius <= 0.0 or self.half_length <= 0.0:
      raise ValueError("radius and half_length must be positive")
    object.__setattr__(self, "center", _vector3(self.center, "center"))
    object.__setattr__(self, "axis", _unit_vector(self.axis, "axis"))

  def query(self, point: ArrayLike) -> SurfaceProjection:
    x = _vector3(point, "point")
    relative = x - self.center
    axial_coordinate = float(np.dot(relative, self.axis))
    radial = relative - axial_coordinate * self.axis
    radial_distance = float(np.linalg.norm(radial))
    if radial_distance <= _EPS:
      radial_direction, _ = _tangent_basis(self.axis)
    else:
      radial_direction = radial / radial_distance

    radial_excess = radial_distance - self.radius
    axial_excess = abs(axial_coordinate) - self.half_length
    if radial_excess > 0.0 or axial_excess > 0.0:
      closest_radial_distance = min(radial_distance, self.radius)
      closest_axial = float(
        np.clip(axial_coordinate, -self.half_length, self.half_length)
      )
      closest = (
        self.center
        + closest_radial_distance * radial_direction
        + closest_axial * self.axis
      )
      delta = x - closest
      signed_distance = float(np.linalg.norm(delta))
      normal = delta / signed_distance
      return SurfaceProjection(closest, normal, signed_distance)

    side_gap = self.radius - radial_distance
    cap_gap = self.half_length - abs(axial_coordinate)
    if side_gap <= cap_gap:
      closest = (
        self.center
        + self.radius * radial_direction
        + axial_coordinate * self.axis
      )
      normal = radial_direction
      signed_distance = -side_gap
    else:
      cap_sign = 1.0 if axial_coordinate >= 0.0 else -1.0
      closest = self.center + radial + cap_sign * self.half_length * self.axis
      normal = cap_sign * self.axis
      signed_distance = -cap_gap
    return SurfaceProjection(closest, normal, float(signed_distance))

  def sample_surface(
    self,
    count: int,
    rng: np.random.Generator,
  ) -> NDArray[np.float64]:
    if count <= 0:
      raise ValueError("count must be positive")
    tangent_1, tangent_2 = _tangent_basis(self.axis)
    side_area = 4.0 * np.pi * self.radius * self.half_length
    cap_area = 2.0 * np.pi * self.radius**2
    side_probability = side_area / (side_area + cap_area)
    choose_side = rng.random(count) < side_probability
    angles = rng.uniform(0.0, 2.0 * np.pi, size=count)
    radial_directions = (
      np.cos(angles)[:, None] * tangent_1[None, :]
      + np.sin(angles)[:, None] * tangent_2[None, :]
    )
    points = np.empty((count, 3), dtype=np.float64)

    side_indices = np.flatnonzero(choose_side)
    side_axial = rng.uniform(-self.half_length, self.half_length, size=len(side_indices))
    points[side_indices] = (
      self.center[None, :]
      + self.radius * radial_directions[side_indices]
      + side_axial[:, None] * self.axis[None, :]
    )

    cap_indices = np.flatnonzero(~choose_side)
    cap_radii = self.radius * np.sqrt(rng.random(len(cap_indices)))
    cap_signs = rng.choice(np.array([-1.0, 1.0]), size=len(cap_indices))
    points[cap_indices] = (
      self.center[None, :]
      + cap_radii[:, None] * radial_directions[cap_indices]
      + (cap_signs * self.half_length)[:, None] * self.axis[None, :]
    )
    return points


@dataclass(frozen=True, slots=True)
class Box(AnalyticShape):
  center: ArrayLike
  half_extents: ArrayLike
  rotation: ArrayLike | None = None

  def __post_init__(self) -> None:
    center = _vector3(self.center, "center")
    half_extents = _vector3(self.half_extents, "half_extents")
    if np.any(half_extents <= 0.0):
      raise ValueError("half_extents must be positive")
    object.__setattr__(self, "center", center)
    object.__setattr__(self, "half_extents", half_extents)
    object.__setattr__(self, "rotation", _rotation_matrix(self.rotation))

  def query(self, point: ArrayLike) -> SurfaceProjection:
    x = _vector3(point, "point")
    local = self.rotation.T @ (x - self.center)
    closest_local = np.clip(local, -self.half_extents, self.half_extents)
    outside_delta = local - closest_local
    outside_distance = float(np.linalg.norm(outside_delta))
    if outside_distance > _EPS:
      normal_local = outside_delta / outside_distance
      signed_distance = outside_distance
    else:
      gaps = self.half_extents - np.abs(local)
      axis = int(np.argmin(gaps))
      sign = 1.0 if local[axis] >= 0.0 else -1.0
      closest_local = local.copy()
      closest_local[axis] = sign * self.half_extents[axis]
      normal_local = np.zeros(3, dtype=np.float64)
      normal_local[axis] = sign
      signed_distance = -float(gaps[axis])
    closest = self.center + self.rotation @ closest_local
    normal = self.rotation @ normal_local
    return SurfaceProjection(closest, normal, signed_distance)

  def sample_surface(
    self,
    count: int,
    rng: np.random.Generator,
  ) -> NDArray[np.float64]:
    if count <= 0:
      raise ValueError("count must be positive")
    hx, hy, hz = self.half_extents
    face_areas = np.array([hy * hz, hy * hz, hx * hz, hx * hz, hx * hy, hx * hy])
    faces = rng.choice(6, size=count, p=face_areas / np.sum(face_areas))
    local_points = rng.uniform(-1.0, 1.0, size=(count, 3)) * self.half_extents
    for face in range(6):
      indices = np.flatnonzero(faces == face)
      axis = face // 2
      sign = -1.0 if face % 2 == 0 else 1.0
      local_points[indices, axis] = sign * self.half_extents[axis]
    return self.center[None, :] + local_points @ self.rotation.T


@dataclass(frozen=True, slots=True)
class RoundedBox(AnalyticShape):
  """Minkowski sum of an oriented box and a sphere."""

  center: ArrayLike
  half_extents: ArrayLike
  corner_radius: float
  rotation: ArrayLike | None = None

  def __post_init__(self) -> None:
    if self.corner_radius <= 0.0:
      raise ValueError("corner_radius must be positive")
    core = Box(self.center, self.half_extents, self.rotation)
    object.__setattr__(self, "center", core.center)
    object.__setattr__(self, "half_extents", core.half_extents)
    object.__setattr__(self, "rotation", core.rotation)

  @property
  def core(self) -> Box:
    return Box(self.center, self.half_extents, self.rotation)

  def query(self, point: ArrayLike) -> SurfaceProjection:
    core_query = self.core.query(point)
    closest = core_query.point + self.corner_radius * core_query.normal
    return SurfaceProjection(
      closest,
      core_query.normal,
      core_query.signed_distance - self.corner_radius,
    )

  def sample_surface(
    self,
    count: int,
    rng: np.random.Generator,
  ) -> NDArray[np.float64]:
    core_points = self.core.sample_surface(count, rng)
    samples = np.empty_like(core_points)
    for index, point in enumerate(core_points):
      query = self.core.query(point)
      samples[index] = point + self.corner_radius * query.normal
    return samples
