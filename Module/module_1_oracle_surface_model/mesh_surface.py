"""Ground-truth triangle-mesh surface for presentation and mesh experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from numpy.typing import ArrayLike, NDArray

from Module.module_1_oracle_surface_model.geometry import (
  AnalyticShape,
  SurfaceProjection,
  _vector3,
)


@dataclass(frozen=True, slots=True)
class MeshScalePolicy:
  """Uniform scaling policy sized for whole-hand surface motion."""

  target_longest_extent_m: float = 0.30
  minimum_second_extent_m: float = 0.18

  def __post_init__(self) -> None:
    if self.target_longest_extent_m <= 0.0 or self.minimum_second_extent_m <= 0.0:
      raise ValueError("mesh target extents must be positive")
    if self.minimum_second_extent_m > self.target_longest_extent_m:
      raise ValueError("minimum_second_extent_m cannot exceed target_longest_extent_m")


class MeshSurface(AnalyticShape):
  """Scaled, grounded triangle mesh with local signed-distance queries.

  The sign is determined by the closest face normal. This is robust for the
  near-surface contact and clearance queries used by the demo, including open
  meshes such as the Stanford Bunny. It is not a global inside/outside oracle
  for arbitrary points behind mesh holes.
  """

  def __init__(
    self,
    mesh: trimesh.Trimesh,
    *,
    source_path: str | Path,
    source_up_axis: str,
    scale_policy: MeshScalePolicy,
    scale_factor: float,
  ) -> None:
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
      raise ValueError("mesh must contain vertices and triangular faces")
    self._mesh = mesh.copy()
    self._source_path = Path(source_path)
    self._source_up_axis = source_up_axis
    self._scale_policy = scale_policy
    self._scale_factor = float(scale_factor)

  @classmethod
  def from_file(
    cls,
    path: str | Path,
    *,
    source_up_axis: str = "z",
    scale_policy: MeshScalePolicy | None = None,
    process: bool = True,
  ) -> MeshSurface:
    source = Path(path)
    if not source.is_file():
      raise FileNotFoundError(source)
    if source_up_axis not in {"x", "y", "z"}:
      raise ValueError("source_up_axis must be one of: x, y, z")
    policy = scale_policy or MeshScalePolicy()
    loaded = trimesh.load(source, force="mesh", process=process)
    if not isinstance(loaded, trimesh.Trimesh):
      raise ValueError(f"{source} did not load as one triangle mesh")
    mesh = loaded.copy()

    rotation = np.eye(4)
    if source_up_axis == "y":
      rotation[:3, :3] = np.array(
        [
          [1.0, 0.0, 0.0],
          [0.0, 0.0, -1.0],
          [0.0, 1.0, 0.0],
        ]
      )
    elif source_up_axis == "x":
      rotation[:3, :3] = np.array(
        [
          [0.0, 0.0, -1.0],
          [0.0, 1.0, 0.0],
          [1.0, 0.0, 0.0],
        ]
      )
    mesh.apply_transform(rotation)

    sorted_extents = np.sort(np.asarray(mesh.extents, dtype=np.float64))[::-1]
    if sorted_extents[1] <= 0.0:
      raise ValueError("mesh needs at least two non-zero spatial extents")
    scale_factor = max(
      policy.target_longest_extent_m / sorted_extents[0],
      policy.minimum_second_extent_m / sorted_extents[1],
    )
    mesh.apply_scale(scale_factor)

    bounds = np.asarray(mesh.bounds)
    horizontal_center = 0.5 * (bounds[0, :2] + bounds[1, :2])
    mesh.apply_translation(
      np.array([-horizontal_center[0], -horizontal_center[1], -bounds[0, 2]])
    )
    mesh.remove_unreferenced_vertices()
    return cls(
      mesh,
      source_path=source,
      source_up_axis=source_up_axis,
      scale_policy=policy,
      scale_factor=scale_factor,
    )

  @property
  def source_path(self) -> Path:
    return self._source_path

  @property
  def source_up_axis(self) -> str:
    return self._source_up_axis

  @property
  def scale_policy(self) -> MeshScalePolicy:
    return self._scale_policy

  @property
  def scale_factor(self) -> float:
    return self._scale_factor

  @property
  def extents(self) -> NDArray[np.float64]:
    return np.asarray(self._mesh.extents, dtype=np.float64).copy()

  @property
  def bounds(self) -> NDArray[np.float64]:
    return np.asarray(self._mesh.bounds, dtype=np.float64).copy()

  @property
  def vertex_count(self) -> int:
    return len(self._mesh.vertices)

  @property
  def face_count(self) -> int:
    return len(self._mesh.faces)

  @property
  def is_watertight(self) -> bool:
    return bool(self._mesh.is_watertight)

  @property
  def mesh(self) -> trimesh.Trimesh:
    """Return a copy for rendering or export without mutating the oracle."""

    return self._mesh.copy()

  def query(self, point: ArrayLike) -> SurfaceProjection:
    x = _vector3(point, "point")
    closest, distances, triangle_ids = trimesh.proximity.closest_point(
      self._mesh,
      x[None, :],
    )
    closest_point = np.asarray(closest[0], dtype=np.float64)
    distance = float(distances[0])
    triangle_id = int(triangle_ids[0])
    if triangle_id < 0:
      raise RuntimeError("mesh proximity query did not return a triangle")
    normal = np.array(
      self._mesh.face_normals[triangle_id],
      dtype=np.float64,
      copy=True,
    )
    normal /= np.linalg.norm(normal)
    signed_offset = float(np.dot(x - closest_point, normal))
    if distance <= 1e-12:
      signed_distance = 0.0
    else:
      signed_distance = distance if signed_offset >= 0.0 else -distance
    return SurfaceProjection(closest_point, normal, signed_distance)

  def sample_surface(
    self,
    count: int,
    rng: np.random.Generator,
  ) -> NDArray[np.float64]:
    if count <= 0:
      raise ValueError("count must be positive")
    seed = int(rng.integers(0, np.iinfo(np.int32).max))
    points, _ = trimesh.sample.sample_surface(self._mesh, count, seed=seed)
    return np.asarray(points, dtype=np.float64)

  def export(self, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    self._mesh.export(destination)
    return destination
