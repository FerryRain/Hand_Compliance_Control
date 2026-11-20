"""Measure object surface area swept by the recorded finger-pad contacts."""

from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.sparse import bmat, coo_matrix, csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import trimesh


REPO = Path(__file__).resolve().parents[2]
RADII_MM = (5, 10, 25)
MIN_FORCE_N = 0.05
MAX_PROJECTION_M = 0.005
MIN_NORMAL_ALIGNMENT = 0.5
SUBDIVISIONS = 2


class ContactSurface:
    def __init__(self, mesh: trimesh.Trimesh) -> None:
        self.original = mesh.copy()
        self.original.merge_vertices(merge_tex=True, merge_norm=True)
        self.original.update_faces(self.original.unique_faces() & self.original.nondegenerate_faces())
        self.original.remove_unreferenced_vertices()
        fingerprint = hashlib.sha256()
        fingerprint.update(np.asarray(self.original.vertices, dtype="<f8").tobytes())
        fingerprint.update(np.asarray(self.original.faces, dtype="<i8").tobytes())
        self.mesh_sha256 = fingerprint.hexdigest()
        self.mesh = self.original
        for _ in range(SUBDIVISIONS):
            self.mesh = self.mesh.subdivide()
        self.vertices = np.asarray(self.mesh.vertices)
        self.faces = np.asarray(self.mesh.faces)
        self.triangles = self.vertices[self.faces]
        self.centers = self.triangles.mean(axis=1)
        self.bounds = np.linalg.norm(self.triangles - self.centers[:, None], axis=-1).max(axis=1)
        self.candidate_radius = float(self.bounds.max()) + MAX_PROJECTION_M
        self.tree = cKDTree(self.centers)
        self.normals = np.asarray(self.mesh.face_normals)

        self.vertex_area = np.zeros(len(self.vertices))
        np.add.at(self.vertex_area, self.faces.ravel(), np.repeat(self.mesh.area_faces / 3, 3))
        self.area = float(self.vertex_area.sum())
        np.testing.assert_allclose(self.area, self.original.area, rtol=1e-9, atol=1e-12)
        lo, hi = self.mesh.edges_unique.T
        edge_lengths = np.linalg.norm(self.vertices[lo] - self.vertices[hi], axis=1)
        self.graph = coo_matrix(
            (np.tile(edge_lengths, 2), (np.r_[lo, hi], np.r_[hi, lo])),
            shape=(len(self.vertices), len(self.vertices)),
        ).tocsr()

    def project(self, points: np.ndarray, normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Register contacts to nearby, outward-facing triangles without filling gaps."""
        chosen = np.full(len(points), -1, dtype=np.int32)
        projected = np.full_like(points, np.nan)
        batch_size = 1024
        for start in range(0, len(points), batch_size):
            xyz = points[start : start + batch_size]
            outward = normals[start : start + batch_size]
            neighbors = self.tree.query_ball_point(xyz, self.candidate_radius)
            counts = np.fromiter(map(len, neighbors), dtype=int, count=len(xyz))
            if not counts.any():
                continue
            sample = np.repeat(np.arange(len(xyz)), counts)
            face = np.concatenate(neighbors).astype(np.int32, copy=False)
            separation = np.linalg.norm(xyz[sample] - self.centers[face], axis=1)
            compatible = (separation <= self.bounds[face] + MAX_PROJECTION_M) & (
                np.einsum("ij,ij->i", outward[sample], self.normals[face]) >= MIN_NORMAL_ALIGNMENT
            )
            sample, face = sample[compatible], face[compatible]
            if not len(face):
                continue
            nearest = trimesh.triangles.closest_point(self.triangles[face], xyz[sample])
            error = np.sum((nearest - xyz[sample]) ** 2, axis=1)
            ordered = np.lexsort((face, error, sample))
            first = np.r_[True, np.diff(sample[ordered]) != 0]
            pick = ordered[first]
            pick = pick[error[pick] <= MAX_PROJECTION_M**2]
            chosen[start + sample[pick]] = face[pick]
            projected[start + sample[pick]] = nearest[pick]
        return chosen, projected

    def distances(self, triangles: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Seed a multi-source mesh-edge search at the projected surface points."""
        offsets = np.full(len(self.vertices), np.inf)
        corners = self.faces[triangles]
        lengths = np.linalg.norm(self.vertices[corners] - points[:, None], axis=-1)
        np.minimum.at(offsets, corners.ravel(), lengths.ravel())
        sources = np.flatnonzero(np.isfinite(offsets))
        if not len(sources):
            return offsets
        source_column = csr_matrix(
            (np.maximum(offsets[sources], 1e-12), (sources, np.zeros_like(sources))),
            shape=(len(offsets), 1),
        )
        graph = bmat(
            [[self.graph, source_column], [source_column.T, csr_matrix((1, 1))]], format="csr"
        )
        return dijkstra(graph, directed=False, indices=len(offsets), limit=max(RADII_MM) / 1000)[:-1]

    def fractions(self, distances: np.ndarray) -> dict[str, float]:
        return {
            str(radius): float(self.vertex_area[distances <= radius / 1000].sum() / self.area)
            for radius in RADII_MM
        }


@lru_cache(maxsize=5)
def load_surface(object_id: str) -> ContactSurface:
    shared = str(REPO / "mcc_finger_compliance_control/scripts")
    if shared not in sys.path:
        sys.path.insert(0, shared)
    from object_catalog import MeshNormalOracle, load_object_config

    oracle = MeshNormalOracle.from_config(load_object_config(object_id), scale=1.0)
    if oracle is None:
        raise ValueError(f"No source surface mesh for {object_id}")
    return ContactSurface(oracle.mesh)


def measure(
    result_path: Path, common_frames: int | None = None
) -> tuple[dict, ContactSurface, np.ndarray]:
    """Return full-run and equal-time areas with the mesh and the coverage mask."""
    result = json.loads(result_path.read_bytes())
    request = json.loads(result_path.with_suffix(".request.json").read_bytes())
    if len(result["lanes"]) != 1 or len(request["rows"]) != 1:
        raise ValueError("Surface coverage requires one recorded object rollout")
    n = result["lanes"][0]["frames"]
    if n <= 0:
        raise ValueError("The object rollout contains no recorded frames")
    if common_frames is not None and not 1 <= common_frames <= n:
        raise ValueError("Equal-time window exceeds the recorded rollout")
    object_id = request["rows"][0]["object_id"]
    surface = load_surface(object_id)
    trace_path = REPO / result["trace"]
    with np.load(trace_path, allow_pickle=False) as trace:
        pose = trace["measured_object_pose_world"][:n, 0].astype(np.float64)
        np.testing.assert_allclose(pose, np.broadcast_to(pose[0], pose.shape), rtol=0, atol=1e-6)
        points = trace["physical_tip_points"][:n, 0].astype(np.float64)
        normals = -trace["physical_tip_inward"][:n, 0].astype(np.float64)
        normal_length = np.linalg.norm(normals, axis=-1)
        force = np.linalg.norm(trace["input_tip_force_world"][:n, 0], axis=-1)
        loaded = (
            trace["physical_tip_found"][:n, 0]
            & trace["input_tip_found"][:n, 0]
            & np.isfinite(force)
            & (force >= MIN_FORCE_N)
            & np.isfinite(points).all(axis=-1)
            & np.isfinite(normal_length)
            & (normal_length > 1e-6)
        )
    ticks, fingers = np.nonzero(loaded)
    world_to_object = Rotation.from_quat(pose[0, [4, 5, 6, 3]]).inv()
    points = world_to_object.apply(points[ticks, fingers] - pose[0, :3])
    unit = normals[ticks, fingers] / normal_length[ticks, fingers, None]
    normals = world_to_object.apply(unit)
    triangles, projected = surface.project(points, normals)
    valid = triangles >= 0
    if len(points) and not valid.any():
        raise ValueError(f"No loaded finger-pad contact registered to {object_id}")
    full = surface.distances(triangles[valid], projected[valid])
    fractions = surface.fractions(full)
    if not 0 <= fractions["5"] <= fractions["10"] <= fractions["25"] <= 1:
        raise ValueError("Coverage must be monotone with the radius")
    dt = result["source_config"]["control_dt"]
    metrics = {
        "object_id": object_id,
        "source_surface_mesh_sha256": surface.mesh_sha256,
        "object_surface_area_cm2": surface.area * 10000,
        "covered_surface_area_cm2_25mm": fractions["25"] * surface.area * 10000,
        "surface_coverage_by_radius_mm": fractions,
        "loaded_pad_contact_samples": int(len(ticks)),
        "registered_pad_contact_samples": int(valid.sum()),
        "maximum_registered_residual_mm": float(
            np.linalg.norm(projected[valid] - points[valid], axis=-1).max(initial=0) * 1000
        ),
    }
    if common_frames is not None:
        common = valid & (ticks < common_frames)
        common_area = surface.fractions(surface.distances(triangles[common], projected[common]))
        if common_area["25"] > fractions["25"] + 1e-12:
            raise ValueError("Equal-time coverage cannot exceed full-run coverage")
        metrics.update(equal_time_coverage_25mm=common_area["25"], equal_time_sim_seconds=common_frames * dt)
    return metrics, surface, full


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    metrics, _, _ = measure(args.result)
    report = json.dumps(metrics, indent=2) + "\n"
    if args.output is None:
        print(report, end="")
    else:
        args.output.write_text(report, encoding="utf8")
