"""Canonical Bunny geometry shared by the I01 physics scene and evaluator.

MuJoCo's native mesh collision representation is not used as an unreported
surrogate for the non-convex Stanford Bunny.  The exact transformed mesh is
kept for rendering and audit, while a deterministic upper-envelope height
field represents the exposed side used by the downward-facing fingertip pads.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
from pathlib import Path

import numpy as np
import trimesh
from numpy.typing import ArrayLike, NDArray

from Module.module_1_oracle_surface_model.mesh_demo import DEFAULT_BUNNY
from Module.module_1_oracle_surface_model.mesh_surface import MeshScalePolicy, MeshSurface


BUNNY_HFIELD_NROW = 181
BUNNY_HFIELD_NCOL = 181
BUNNY_SIDE_ROTATION_DEG = 90.0
BUNNY_BASE_DEPTH_M = 0.012
_EPSILON_M = 1e-6


def _readonly(array: NDArray[np.generic]) -> NDArray[np.generic]:
  result = np.array(array, copy=True)
  result.setflags(write=False)
  return result


@dataclass(frozen=True, slots=True)
class BunnyHeightField:
  """Exact visual mesh plus a regular upper-envelope collision surface."""

  source_path: Path
  source_sha256: str
  mesh: trimesh.Trimesh
  x_coordinates_m: NDArray[np.float64]
  y_coordinates_m: NDArray[np.float64]
  height_m: NDArray[np.float64]
  valid_mask: NDArray[np.bool_]
  dz_dx: NDArray[np.float64]
  dz_dy: NDArray[np.float64]

  @property
  def x_half_m(self) -> float:
    return 0.5 * float(self.x_coordinates_m[-1] - self.x_coordinates_m[0])

  @property
  def y_half_m(self) -> float:
    return 0.5 * float(self.y_coordinates_m[-1] - self.y_coordinates_m[0])

  @property
  def height_span_m(self) -> float:
    return float(np.max(self.height_m))

  @property
  def extents_m(self) -> NDArray[np.float64]:
    return np.asarray(self.mesh.extents, dtype=np.float64).copy()

  @property
  def coverage_fraction(self) -> float:
    return float(np.mean(self.valid_mask))

  def mujoco_elevation(self) -> NDArray[np.float64]:
    """Return normalized data in MuJoCo's +Y-first row order."""

    span = max(self.height_span_m, _EPSILON_M)
    return np.asarray((self.height_m / span)[::-1], dtype=np.float64)

  def query(
    self,
    x_m: float,
    y_m: float,
  ) -> tuple[float, NDArray[np.float64], bool]:
    """Bilinearly query height/normal and whether the cell belongs to Bunny."""

    x = float(x_m)
    y = float(y_m)
    if (
      x < self.x_coordinates_m[0]
      or x > self.x_coordinates_m[-1]
      or y < self.y_coordinates_m[0]
      or y > self.y_coordinates_m[-1]
    ):
      return 0.0, np.array([0.0, 0.0, 1.0]), False
    col_float = (x - self.x_coordinates_m[0]) / (
      self.x_coordinates_m[-1] - self.x_coordinates_m[0]
    ) * (len(self.x_coordinates_m) - 1)
    row_float = (y - self.y_coordinates_m[0]) / (
      self.y_coordinates_m[-1] - self.y_coordinates_m[0]
    ) * (len(self.y_coordinates_m) - 1)
    col0 = min(int(np.floor(col_float)), len(self.x_coordinates_m) - 2)
    row0 = min(int(np.floor(row_float)), len(self.y_coordinates_m) - 2)
    tx = col_float - col0
    ty = row_float - row0
    weights = np.array(
      [(1.0 - tx) * (1.0 - ty), tx * (1.0 - ty), (1.0 - tx) * ty, tx * ty],
      dtype=np.float64,
    )
    indices = ((row0, col0), (row0, col0 + 1), (row0 + 1, col0), (row0 + 1, col0 + 1))
    heights = np.array([self.height_m[index] for index in indices])
    dx = float(sum(weight * self.dz_dx[index] for weight, index in zip(weights, indices)))
    dy = float(sum(weight * self.dz_dy[index] for weight, index in zip(weights, indices)))
    normal = np.array([-dx, -dy, 1.0], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    # All four corners must come from a real vertical mesh intersection.  This
    # conservative rule prevents interpolation from filling the silhouette.
    valid = bool(all(self.valid_mask[index] for index in indices))
    return float(weights @ heights), normal, valid

  def mesh_residuals(self, points_local_m: ArrayLike) -> NDArray[np.float64]:
    """Return exact closest-triangle distances in bounded-memory chunks."""

    points = np.asarray(points_local_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
      raise ValueError("points_local_m must be finite with shape (N,3)")
    if len(points) == 0:
      return np.empty(0, dtype=np.float64)
    result = np.empty(len(points), dtype=np.float64)
    for start in range(0, len(points), 2048):
      stop = min(start + 2048, len(points))
      _, distances, _ = trimesh.proximity.closest_point(self.mesh, points[start:stop])
      result[start:stop] = distances
    return result

  def export_visual_mesh(self, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    self.mesh.export(destination)
    return destination


def _canonical_mesh(source: Path) -> trimesh.Trimesh:
  upright = MeshSurface.from_file(
    source,
    source_up_axis="y",
    scale_policy=MeshScalePolicy(0.30, 0.18),
  ).mesh
  rotation = trimesh.transformations.rotation_matrix(
    np.deg2rad(BUNNY_SIDE_ROTATION_DEG),
    [1.0, 0.0, 0.0],
  )
  upright.apply_transform(rotation)
  bounds = np.asarray(upright.bounds, dtype=np.float64)
  center_xy = 0.5 * (bounds[0, :2] + bounds[1, :2])
  upright.apply_translation([-center_xy[0], -center_xy[1], -bounds[0, 2]])
  upright.remove_unreferenced_vertices()
  return upright


def _upper_envelope(
  mesh: trimesh.Trimesh,
) -> tuple[
  NDArray[np.float64],
  NDArray[np.float64],
  NDArray[np.float64],
  NDArray[np.bool_],
]:
  bounds = np.asarray(mesh.bounds, dtype=np.float64)
  x = np.linspace(bounds[0, 0], bounds[1, 0], BUNNY_HFIELD_NCOL)
  y = np.linspace(bounds[0, 1], bounds[1, 1], BUNNY_HFIELD_NROW)
  grid_x, grid_y = np.meshgrid(x, y)
  ray_z = float(bounds[1, 2] + 0.02)
  origins = np.column_stack(
    (grid_x.ravel(), grid_y.ravel(), np.full(grid_x.size, ray_z))
  )
  directions = np.tile([0.0, 0.0, -1.0], (len(origins), 1))
  locations, ray_indices, _ = mesh.ray.intersects_location(
    origins,
    directions,
    multiple_hits=True,
  )
  top = np.full(len(origins), -np.inf, dtype=np.float64)
  np.maximum.at(top, ray_indices, locations[:, 2])
  valid = np.isfinite(top)
  top[~valid] = 0.0
  return x, y, top.reshape(grid_x.shape), valid.reshape(grid_x.shape)


@lru_cache(maxsize=1)
def canonical_bunny_heightfield() -> BunnyHeightField:
  source = DEFAULT_BUNNY.resolve()
  digest = hashlib.sha256(source.read_bytes()).hexdigest()
  mesh = _canonical_mesh(source)
  x, y, height, valid = _upper_envelope(mesh)
  dy, dx = np.gradient(
    height,
    float(y[1] - y[0]),
    float(x[1] - x[0]),
  )
  # Gradients across silhouette discontinuities are not physical surface
  # normals.  They remain unused because query() requires four valid corners.
  dx[~valid] = 0.0
  dy[~valid] = 0.0
  return BunnyHeightField(
    source_path=source,
    source_sha256=digest,
    mesh=mesh,
    x_coordinates_m=_readonly(x),
    y_coordinates_m=_readonly(y),
    height_m=_readonly(height),
    valid_mask=_readonly(valid),
    dz_dx=_readonly(dx),
    dz_dy=_readonly(dy),
  )
