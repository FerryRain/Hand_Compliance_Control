"""Privileged geometry targets and measured, geodesic surface coverage."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import numpy as np
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.transform import Rotation
import trimesh

from Module.i01_bunny_physics.surface import canonical_bunny_heightfield
from Module.i04_oracle_next_point.surface_graph import BunnySurfaceGraph
from .contracts import WristTarget, WholeHandTarget
from .physics import pose, tip_positions, rotation, quaternion, pad_support


class MeasuredCoverage:
    """Only actual pad contacts can add coverage; geometry witnesses cannot."""
    def __init__(self, graph, config):
        self.graph = graph
        self.config = config
        self.visited_vertices = np.zeros(len(graph.vertices_m), dtype=bool)
        self.vertex_area = np.zeros(len(graph.vertices_m))
        np.add.at(self.vertex_area, graph.faces.ravel(),
                  np.repeat(np.asarray(graph.mesh.area_faces) / 3, 3))
        self.visited_goals = np.zeros(graph.required_goal_count, dtype=bool)
        self.last_vertices = np.full(4, -1, dtype=int)
        self.normal_rejected_contacts = 0

    @lru_cache(maxsize=4096)
    def neighborhood(self, vertex):
        # Sparse truncated Dijkstra avoids counting the other side of an ear.
        distances = dijkstra(self.graph.adjacency, directed=False, indices=int(vertex),
                             limit=self.config.coverage_radius_m)
        return np.flatnonzero(np.isfinite(distances))

    def update(self, points_local, normals_world, actual_pad_contact):
        active = np.flatnonzero(actual_pad_contact)
        if not len(active):
            return
        vertices, residuals = self.graph.oriented_nearest_vertices(
            points_local[active], normals_world[active])
        for finger, vertex, residual in zip(active, vertices, residuals):
            alignment = float(self.graph.normals[vertex] @ normals_world[finger])
            if residual > self.config.mesh_residual_limit_m or alignment < self.config.normal_alignment_cos:
                self.normal_rejected_contacts += 1
                continue
            self.last_vertices[finger] = int(vertex)
            self.visited_vertices[self.neighborhood(int(vertex))] = True
        self.visited_goals = self.visited_vertices[self.graph.required_vertices]

    @property
    def area_fraction(self):
        return float(self.vertex_area @ self.visited_vertices / self.vertex_area.sum())


def align_vector(first, second):
    a = np.asarray(first) / np.linalg.norm(first)
    b = np.asarray(second) / np.linalg.norm(second)
    axis = np.cross(a, b)
    sine = np.linalg.norm(axis)
    cosine = float(np.clip(a @ b, -1, 1))
    if sine < 1e-10:
        if cosine > 0:
            return Rotation.identity()
        basis = np.eye(3)[np.argmin(np.abs(a))]
        axis = np.cross(a, basis)
        return Rotation.from_rotvec(np.pi * axis / np.linalg.norm(axis))
    return Rotation.from_rotvec(np.arctan2(sine, cosine) * axis / sine)


class GTCoverageOracle:
    """Full GT selects geometric targets; adapters strip privileged witnesses."""
    def __init__(self, handles, data, graph, config):
        self.h, self.graph, self.config = handles, graph, config
        self.object_position = handles.object_position_m.copy()
        self.initial_pose = pose(handles, data)
        self.initial_rotation = rotation(self.initial_pose[3:])
        tips = tip_positions(handles, data)
        self.tip_local = (tips - self.initial_pose[:3]) @ self.initial_rotation.as_matrix()
        projected, _, triangles = trimesh.proximity.closest_point(graph.mesh, tips - self.object_position)
        self.initial_anchor = np.mean(projected, axis=0)
        self.initial_normal = np.mean(graph.mesh.face_normals[triangles], axis=0)
        self.initial_normal /= np.linalg.norm(self.initial_normal)
        self.palm_offset = self.initial_pose[:3] - self.object_position - self.initial_anchor
        self.support = np.array([pad_support(handles, data, i, self.initial_normal) for i in range(4)])
        self.attempts = np.zeros(graph.required_goal_count, dtype=int)
        self.geometric_rejections = {}

    def latent_target(self, goal_id):
        vertex = int(self.graph.required_vertices[goal_id])
        center = self.graph.vertices_m[vertex]
        neighbors = self.graph._tree.query_ball_point(center, 0.008)
        normal = np.mean(self.graph.normals[neighbors], axis=0)
        normal /= np.linalg.norm(normal)
        aligned = align_vector(self.initial_normal, normal)
        target_rotation = aligned * self.initial_rotation
        palm = self.object_position + center + aligned.apply(self.palm_offset)
        wrist = WristTarget(np.r_[palm, quaternion(target_rotation)])
        raw = palm + target_rotation.apply(self.tip_local)
        points, _, triangles = trimesh.proximity.closest_point(self.graph.mesh, raw - self.object_position)
        normals = self.graph.mesh.face_normals[triangles]
        targets = self.object_position + points + (self.support[:, None] - 0.0003) * normals
        return WholeHandTarget(wrist, targets)

    def ordered_candidates(self, data, coverage):
        remaining = np.flatnonzero(~coverage.visited_goals)
        if not len(remaining):
            return []
        anchor = pose(self.h, data)[:3] - self.object_position - self.palm_offset
        points = self.graph.vertices_m[self.graph.required_vertices[remaining]]
        distances = np.linalg.norm(points - anchor, axis=1)
        # Deferred goals remain in the full denominator and are retried; no
        # unreachable region is erased from the requested surface set.
        score = distances + 0.035 * np.minimum(self.attempts[remaining], 8)
        return remaining[np.argsort(score)].tolist()

    def geometric_continuation(self, goal_id, witness_q, planner):
        """Privileged GT successor check; no contact answer reaches runtime."""
        vertex = int(self.graph.required_vertices[goal_id])
        points = self.graph.vertices_m[self.graph.required_vertices]
        distances = np.linalg.norm(points - self.graph.vertices_m[vertex], axis=1)
        for successor in np.argsort(distances)[1:5]:
            latent = self.latent_target(int(successor))
            q, ep, er, ef = planner.solve(witness_q, latent, iterations=80)
            if (ep <= self.config.goal_position_tolerance_m
                    and er <= self.config.goal_rotation_tolerance_rad
                    and ef <= self.config.fingertip_tolerance_m
                    and planner.collision_reason(q) is None):
                # Check a midpoint too; this is a geometry-only Oracle
                # witness, not a contact-preservation certificate.
                if planner.collision_reason(0.5 * (witness_q + q)) is None:
                    return int(successor)
        return None

    @staticmethod
    def expose(cell, latent):
        if cell == 'A-MCC':
            return WristTarget(latent.wrist.pose_world)
        if cell == 'B-MCC':
            return WholeHandTarget(latent.wrist, latent.fingertip_positions_world_m)
        raise ValueError('DPRef is deferred; only A-MCC and B-MCC are enabled')
