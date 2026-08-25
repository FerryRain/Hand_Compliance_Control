"""Deterministic full-Bunny graph and state-rooted coverage ledger for I04.

The required point set is frozen by geometry alone.  Its visit order is not:
after every physical micro-barrier the route is rooted again at the closest
measured fingertip contact and the explicit planner chooses the next remaining
goal.  The Oracle never supplies a finger identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import trimesh
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree


FeasibilityScore = Callable[[int, float], float | None]


def _readonly(array: ArrayLike, *, dtype: np.dtype | None = None) -> NDArray:
  result = np.array(array, dtype=dtype, copy=True)
  result.setflags(write=False)
  return result


@dataclass(frozen=True, slots=True)
class SurfaceGoal:
  """No-finger-ID goal issued to either I04 method."""

  goal_id: int
  vertex_index: int
  position_local_m: NDArray[np.float64]
  normal_local: NDArray[np.float64]
  outgoing_tangent_local: NDArray[np.float64]
  geodesic_tolerance_m: float
  normal_tolerance_rad: float


@dataclass(frozen=True, slots=True)
class GoalSelection:
  goal: SurfaceGoal
  root_vertex: int
  bridge_vertices: NDArray[np.int32]
  bridge_length_m: float
  selection_score: float
  feasible_candidate_count: int
  considered_candidate_count: int


class BunnySurfaceGraph:
  """Exact mesh-edge geodesics, fixed FPS goals and area bookkeeping."""

  def __init__(
    self,
    mesh: trimesh.Trimesh,
    *,
    coverage_radius_m: float = 0.025,
    arrival_tolerance_m: float = 0.010,
    normal_tolerance_rad: float = np.deg2rad(55.0),
  ) -> None:
    if coverage_radius_m <= 0.0 or arrival_tolerance_m <= 0.0:
      raise ValueError("coverage and arrival radii must be positive")
    if not 0.0 < normal_tolerance_rad < np.pi:
      raise ValueError("normal_tolerance_rad must lie in (0, pi)")
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0:
      raise ValueError("mesh must be a nonempty Trimesh")
    self.mesh = mesh.copy()
    self.vertices_m = _readonly(self.mesh.vertices, dtype=np.float64)
    self.normals = _readonly(self.mesh.vertex_normals, dtype=np.float64)
    self.faces = _readonly(self.mesh.faces, dtype=np.int32)
    self.coverage_radius_m = float(coverage_radius_m)
    self.arrival_tolerance_m = float(arrival_tolerance_m)
    self.normal_tolerance_rad = float(normal_tolerance_rad)
    self._tree = cKDTree(self.vertices_m)
    edges = np.asarray(self.mesh.edges_unique, dtype=np.int32)
    weights = np.linalg.norm(
      self.vertices_m[edges[:, 0]] - self.vertices_m[edges[:, 1]],
      axis=1,
    )
    rows = np.concatenate((edges[:, 0], edges[:, 1]))
    cols = np.concatenate((edges[:, 1], edges[:, 0]))
    values = np.concatenate((weights, weights))
    self.adjacency: csr_matrix = coo_matrix(
      (values, (rows, cols)),
      shape=(len(self.vertices_m), len(self.vertices_m)),
    ).tocsr()
    seed = int(
      np.lexsort(
        (
          self.vertices_m[:, 1],
          self.vertices_m[:, 0],
          -self.vertices_m[:, 2],
        )
      )[0]
    )
    required, owner, radius = self._farthest_point_cover(seed)
    self.required_vertices = _readonly(required, dtype=np.int32)
    self.vertex_goal_owner = _readonly(owner, dtype=np.int32)
    self.realized_cover_radius_m = float(radius)
    vertex_area = np.zeros(len(self.vertices_m), dtype=np.float64)
    thirds = np.repeat(np.asarray(self.mesh.area_faces) / 3.0, 3)
    np.add.at(vertex_area, self.faces.ravel(), thirds)
    goal_area = np.zeros(len(required), dtype=np.float64)
    np.add.at(goal_area, owner, vertex_area)
    self.goal_area_m2 = _readonly(goal_area, dtype=np.float64)

  def _farthest_point_cover(
    self,
    seed_vertex: int,
  ) -> tuple[NDArray[np.int32], NDArray[np.int32], float]:
    minimum = np.full(len(self.vertices_m), np.inf, dtype=np.float64)
    owner = np.full(len(self.vertices_m), -1, dtype=np.int32)
    selected: list[int] = []
    current = int(seed_vertex)
    while True:
      goal_id = len(selected)
      selected.append(current)
      distances = np.asarray(
        dijkstra(self.adjacency, indices=current, directed=False),
        dtype=np.float64,
      )
      improved = distances < minimum
      minimum[improved] = distances[improved]
      owner[improved] = goal_id
      finite = np.isfinite(minimum)
      if not np.all(finite):
        raise ValueError("Bunny mesh graph must be one connected component")
      radius = float(np.max(minimum))
      if radius <= self.coverage_radius_m + 1e-12:
        return (
          np.asarray(selected, dtype=np.int32),
          owner,
          radius,
        )
      current = int(np.argmax(minimum))

  @property
  def required_goal_count(self) -> int:
    return len(self.required_vertices)

  @property
  def total_area_m2(self) -> float:
    return float(np.sum(self.goal_area_m2))

  def nearest_vertex(self, point_local_m: ArrayLike) -> tuple[int, float]:
    point = np.asarray(point_local_m, dtype=np.float64)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
      raise ValueError("point_local_m must be a finite 3-vector")
    distance, index = self._tree.query(point, k=1)
    return int(index), float(distance)

  def nearest_vertices(
    self,
    points_local_m: ArrayLike,
  ) -> tuple[NDArray[np.int32], NDArray[np.float64]]:
    points = np.asarray(points_local_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
      raise ValueError("points_local_m must be finite with shape (N,3)")
    distances, indices = self._tree.query(points, k=1)
    return (
      np.asarray(indices, dtype=np.int32),
      np.asarray(distances, dtype=np.float64),
    )

  def oriented_nearest_vertices(
    self,
    points_local_m: ArrayLike,
    normals_local: ArrayLike,
    *,
    candidate_count: int = 24,
    distance_band_m: float = 0.0015,
  ) -> tuple[NDArray[np.int32], NDArray[np.float64]]:
    """Disambiguate Euclidean-near but geodesically distant Bunny sheets."""

    points = np.asarray(points_local_m, dtype=np.float64)
    normals = np.asarray(normals_local, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or normals.shape != points.shape:
      raise ValueError("points/normals must both have shape (N,3)")
    if candidate_count < 1 or distance_band_m <= 0.0:
      raise ValueError("candidate_count and distance_band_m must be positive")
    normal_norms = np.linalg.norm(normals, axis=1)
    if np.any(normal_norms <= 1e-12) or not np.all(np.isfinite(normals)):
      raise ValueError("normals must be finite and nonzero")
    unit = normals / normal_norms[:, None]
    k = min(int(candidate_count), len(self.vertices_m))
    distances, indices = self._tree.query(points, k=k)
    if k == 1:
      distances = np.asarray(distances)[:, None]
      indices = np.asarray(indices)[:, None]
    selected = np.zeros(len(points), dtype=np.int32)
    residual = np.zeros(len(points), dtype=np.float64)
    for row in range(len(points)):
      local_distances = np.asarray(distances[row], dtype=np.float64)
      local_indices = np.asarray(indices[row], dtype=np.int32)
      eligible = local_distances <= local_distances[0] + distance_band_m
      alignment = self.normals[local_indices] @ unit[row]
      score = np.where(eligible, alignment, -np.inf)
      best = int(np.argmax(score))
      selected[row] = int(local_indices[best])
      residual[row] = float(local_distances[best])
    return selected, residual

  def distances_from(
    self,
    root_vertex: int,
    *,
    return_predecessors: bool = False,
  ) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.int32]]:
    if root_vertex < 0 or root_vertex >= len(self.vertices_m):
      raise ValueError("root_vertex is out of range")
    result = dijkstra(
      self.adjacency,
      indices=int(root_vertex),
      directed=False,
      return_predecessors=return_predecessors,
    )
    if return_predecessors:
      distances, predecessors = result
      return (
        np.asarray(distances, dtype=np.float64),
        np.asarray(predecessors, dtype=np.int32),
      )
    return np.asarray(result, dtype=np.float64)

  def reconstruct_path(
    self,
    root_vertex: int,
    target_vertex: int,
    predecessors: NDArray[np.int32] | None = None,
  ) -> NDArray[np.int32]:
    if predecessors is None:
      _, predecessors = self.distances_from(
        root_vertex,
        return_predecessors=True,
      )
    path = [int(target_vertex)]
    current = int(target_vertex)
    while current != int(root_vertex):
      current = int(predecessors[current])
      if current < 0:
        raise ValueError("target is disconnected from root")
      path.append(current)
    path.reverse()
    return np.asarray(path, dtype=np.int32)

  def decimate_path(
    self,
    path_vertices: ArrayLike,
    *,
    maximum_step_m: float,
  ) -> NDArray[np.int32]:
    path = np.asarray(path_vertices, dtype=np.int32)
    if path.ndim != 1 or len(path) == 0:
      raise ValueError("path_vertices must be a nonempty vector")
    if maximum_step_m <= 0.0:
      raise ValueError("maximum_step_m must be positive")
    retained = [int(path[0])]
    accumulated = 0.0
    previous = int(path[0])
    for vertex in path[1:]:
      current = int(vertex)
      accumulated += float(
        np.linalg.norm(self.vertices_m[current] - self.vertices_m[previous])
      )
      if accumulated >= maximum_step_m - 1e-12:
        retained.append(current)
        accumulated = 0.0
      previous = current
    if retained[-1] != int(path[-1]):
      retained.append(int(path[-1]))
    return np.asarray(retained, dtype=np.int32)

  def make_goal(
    self,
    goal_id: int,
    bridge_vertices: ArrayLike,
  ) -> SurfaceGoal:
    if goal_id < 0 or goal_id >= self.required_goal_count:
      raise ValueError("goal_id is out of range")
    path = np.asarray(bridge_vertices, dtype=np.int32)
    vertex = int(self.required_vertices[goal_id])
    tangent = np.zeros(3, dtype=np.float64)
    if len(path) >= 2:
      tangent = self.vertices_m[path[-1]] - self.vertices_m[path[-2]]
      normal = self.normals[vertex]
      tangent -= float(np.dot(tangent, normal)) * normal
    norm = float(np.linalg.norm(tangent))
    if norm <= 1e-12:
      normal = self.normals[vertex]
      basis = np.array([1.0, 0.0, 0.0])
      if abs(float(np.dot(basis, normal))) > 0.9:
        basis = np.array([0.0, 1.0, 0.0])
      tangent = basis - float(np.dot(basis, normal)) * normal
      norm = float(np.linalg.norm(tangent))
    tangent /= norm
    return SurfaceGoal(
      goal_id=int(goal_id),
      vertex_index=vertex,
      position_local_m=_readonly(self.vertices_m[vertex], dtype=np.float64),
      normal_local=_readonly(self.normals[vertex], dtype=np.float64),
      outgoing_tangent_local=_readonly(tangent, dtype=np.float64),
      geodesic_tolerance_m=self.arrival_tolerance_m,
      normal_tolerance_rad=self.normal_tolerance_rad,
    )


class CoverageLedger:
  """Persistent required-node ledger; bridge points never erase hard regions."""

  def __init__(self, graph: BunnySurfaceGraph) -> None:
    self.graph = graph
    self._visited = np.zeros(graph.required_goal_count, dtype=np.bool_)
    self._visit_order: list[int] = []

  @property
  def visited_mask(self) -> NDArray[np.bool_]:
    return np.array(self._visited, copy=True)

  @property
  def visit_order(self) -> tuple[int, ...]:
    return tuple(self._visit_order)

  @property
  def remaining_goal_ids(self) -> NDArray[np.int32]:
    return np.flatnonzero(~self._visited).astype(np.int32)

  @property
  def complete(self) -> bool:
    return bool(np.all(self._visited))

  @property
  def completion_fraction(self) -> float:
    return float(np.mean(self._visited))

  @property
  def covered_area_fraction(self) -> float:
    area = self.graph.goal_area_m2
    return float(np.sum(area[self._visited]) / np.sum(area))

  def mark_arrived(self, goal_id: int) -> bool:
    if goal_id < 0 or goal_id >= len(self._visited):
      raise ValueError("goal_id is out of range")
    if self._visited[goal_id]:
      return False
    self._visited[goal_id] = True
    self._visit_order.append(int(goal_id))
    return True

  def select_from_measured_contacts(
    self,
    contact_points_local_m: ArrayLike,
    *,
    contact_normals_local: ArrayLike | None = None,
    feasibility_score: FeasibilityScore | None = None,
    maximum_candidates: int = 24,
  ) -> GoalSelection:
    """Choose the next goal from the current real contact root.

    ``feasibility_score`` is owned by the explicit whole-hand planner.  It can
    inspect the current measured robot state and returns an additive cost or
    ``None`` for a candidate not reachable from this barrier.  No selected
    finger is included in the returned Oracle goal.
    """

    points = np.asarray(contact_points_local_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
      raise ValueError("at least one measured contact point is required")
    if maximum_candidates < 1:
      raise ValueError("maximum_candidates must be positive")
    if contact_normals_local is None:
      contact_vertices, _ = self.graph.nearest_vertices(points)
    else:
      contact_vertices, _ = self.graph.oriented_nearest_vertices(
        points,
        contact_normals_local,
      )
    best_root = int(contact_vertices[0])
    best_eccentricity = np.inf
    remaining_vertices = self.graph.required_vertices[self.remaining_goal_ids]
    for root in contact_vertices:
      distances = self.graph.distances_from(int(root))
      eccentricity = float(np.min(distances[remaining_vertices]))
      if eccentricity < best_eccentricity:
        best_eccentricity = eccentricity
        best_root = int(root)
    distances, predecessors = self.graph.distances_from(
      best_root,
      return_predecessors=True,
    )
    remaining = self.remaining_goal_ids
    ranked = sorted(
      (int(goal_id) for goal_id in remaining),
      key=lambda goal_id: (
        float(distances[int(self.graph.required_vertices[goal_id])]),
        goal_id,
      ),
    )
    considered = ranked[:maximum_candidates]
    feasible: list[tuple[float, int]] = []
    for goal_id in considered:
      distance = float(distances[int(self.graph.required_vertices[goal_id])])
      extra = 0.0 if feasibility_score is None else feasibility_score(goal_id, distance)
      if extra is not None and np.isfinite(extra):
        feasible.append((distance + float(extra), goal_id))
    if feasible:
      score, selected = min(feasible, key=lambda item: (item[0], item[1]))
    else:
      selected = considered[0]
      score = float(distances[int(self.graph.required_vertices[selected])])
    target = int(self.graph.required_vertices[selected])
    bridge = self.graph.reconstruct_path(best_root, target, predecessors)
    return GoalSelection(
      goal=self.graph.make_goal(selected, bridge),
      root_vertex=best_root,
      bridge_vertices=bridge,
      bridge_length_m=float(distances[target]),
      selection_score=float(score),
      feasible_candidate_count=len(feasible),
      considered_candidate_count=len(considered),
    )

  def arrival_fingers(
    self,
    goal: SurfaceGoal,
    contact_points_local_m: ArrayLike,
    contact_normals_local: ArrayLike,
  ) -> tuple[int, ...]:
    points = np.asarray(contact_points_local_m, dtype=np.float64)
    normals = np.asarray(contact_normals_local, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or normals.shape != points.shape:
      raise ValueError("contact points/normals must both have shape (N,3)")
    if len(points) == 0:
      return ()
    vertices, _ = self.graph.oriented_nearest_vertices(points, normals)
    distances = dijkstra(
      self.graph.adjacency,
      indices=goal.vertex_index,
      directed=False,
      limit=goal.geodesic_tolerance_m + 1e-12,
    )
    cosine = float(np.cos(goal.normal_tolerance_rad))
    arrived: list[int] = []
    for finger, (vertex, normal) in enumerate(zip(vertices, normals), start=1):
      normal_norm = float(np.linalg.norm(normal))
      if normal_norm <= 1e-12:
        continue
      alignment = float(np.dot(normal / normal_norm, goal.normal_local))
      if distances[int(vertex)] <= goal.geodesic_tolerance_m + 1e-12 and alignment >= cosine:
        arrived.append(finger)
    return tuple(arrived)
