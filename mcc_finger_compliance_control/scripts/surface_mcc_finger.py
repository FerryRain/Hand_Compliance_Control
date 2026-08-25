"""Four-fingertip surface MCC with an explicit geometry-provider boundary.

This is the fingertip part of ``full_hand_mcc`` adapted to the inverse replay
environment.  The controller itself does not know the object shape: callers
provide one target surface point and one outward normal per fingertip.  The
first experiment uses :class:`PrivilegedCapsuleSurfaceOracle`; a later sensor
adapter can replace it without changing the MCC or IK implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import daqp
import mink
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

# This is deliberately the same numerical fingertip admittance used by the
# full-hand MCC task.  Keeping the reference dynamics shared is important for
# an A/B experiment: only the replay adapter and the source of the surface
# plan differ, not the force-loop equations.
from mjlab.tasks.leaphand.full_hand_mcc_core import (
    FingertipAdmittanceGains,
    FingertipNormalAdmittance,
)


def _resolve_hand_xml() -> Path:
    """Resolve leap_hand_tactile.xml from cwd or repo root (any launch dir)."""
    candidates = [
        Path("src/mjlab/asset_zoo/robots/xarm6_leap_hand/leap_hand_tactile.xml"),
        Path(__file__).resolve().parents[2]
        / "src/mjlab/asset_zoo/robots/xarm6_leap_hand/leap_hand_tactile.xml",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"leap_hand_tactile.xml not found; tried {[str(c) for c in candidates]}"
    )


HAND_XML = _resolve_hand_xml()
TIP_NAMES = ("if_tip", "mf_tip", "rf_tip", "th_tip")
TIP_BODY_NAMES = (
    "fingertip",
    "fingertip_2",
    "fingertip_3",
    "thumb_fingertip",
)
TIP_SITE_LOCAL_POSITIONS = (
    (-0.0106151, -0.0326103, 0.0141088),
    (-0.0106151, -0.0326103, 0.0144487),
    (-0.0106151, -0.0326103, 0.0140386),
    (-0.0106383, -0.0453895, -0.0144321),
)
# MuJoCo tree/action order used by the standalone tactile hand.
HAND_JOINT_NAMES = (
    "1", "0", "2", "3",
    "5", "4", "6", "7",
    "9", "8", "10", "11",
    "12", "13", "14", "15",
)

# Thin-cylinder pinch posture, kept in sync with
# ``leaphand_mcc_finger_env_cfg.DEFAULT_PREGRASP_Q`` (side-axis pulls index
# and ring toward the middle finger; thumb opposes at 1.30 rad).  This is
# the grasp direction used by the full-hand search; large objects may
# restore the open posture via the per-object pregrasp override.
DEFAULT_GRASP_CLOSURE_Q = np.asarray(
    (
        1.05, 0.50, 0.85, 0.85,
        1.05, 0.00, 0.85, 0.85,
        1.05, -0.50, 0.85, 0.85,
        1.05, 1.30, 0.85, 0.85,
    ),
    dtype=np.float64,
)

# Open seed used only to construct the per-finger closure path.  Collection
# stops this path at the first surface intersection, or at a conservative
# fraction when no intersection exists.
DEFAULT_OPEN_Q = np.asarray(
    (
        0.85, 0.00, 0.45, 0.55,
        0.85, 0.00, 0.45, 0.55,
        0.85, 0.00, 0.45, 0.55,
        0.85, 1.57, 0.45, 0.55,
    ),
    dtype=np.float64,
)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    if np.any(norms < 1.0e-9):
        raise ValueError("Surface normals must be non-zero")
    return vectors / norms


def _clip_row_norm(vectors: np.ndarray, limit: float) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors * np.minimum(1.0, limit / np.maximum(norms, 1.0e-12))


def _quat_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm((w, x, y, z))
    if norm < 1.0e-12:
        raise ValueError("Palm quaternion must be non-zero")
    w, x, y, z = np.asarray((w, x, y, z)) / norm
    return np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def _fixed_hand_model() -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(HAND_XML))
    for exclude in list(spec.excludes):
        if exclude.bodyname1 == "thumb_pip" and exclude.bodyname2 == "pip4":
            spec.delete(exclude)
    free_joint = spec.joint("palm_base")
    if free_joint is not None:
        spec.delete(free_joint)
    palm = spec.body("palm_lower")
    palm.pos[:] = (0.0, 0.0, 0.0)
    palm.quat[:] = (1.0, 0.0, 0.0, 0.0)
    palm.alt.type = mujoco.mjtOrientation.mjORIENTATION_QUAT
    existing = {site.name for site in spec.sites}
    for body_name, site_name, site_pos in zip(
        TIP_BODY_NAMES, TIP_NAMES, TIP_SITE_LOCAL_POSITIONS
    ):
        if site_name not in existing:
            spec.body(body_name).add_site(
                name=site_name,
                pos=site_pos,
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=(0.004, 0.0, 0.0),
            )
    return spec.compile()


@dataclass(frozen=True)
class CapsuleSurfaceObservation:
    points_world: np.ndarray
    normals_world: np.ndarray
    signed_distance: np.ndarray


class PrivilegedCapsuleSurfaceOracle:
    """Analytic nearest-surface query for the replay capsule.

    ``signed_distance`` is positive outside the object and negative inside.
    This class is intentionally the only privileged component.
    """

    def __init__(
        self,
        radius: float = 0.15,
        half_height: float = 0.08,
        center_world: np.ndarray | None = None,
        rotation_world_from_object: np.ndarray | None = None,
    ) -> None:
        if radius <= 0.0 or half_height < 0.0:
            raise ValueError("Invalid capsule dimensions")
        self.radius = float(radius)
        self.half_height = float(half_height)
        self.center_world = np.asarray(
            np.zeros(3) if center_world is None else center_world,
            dtype=np.float64,
        ).reshape(3)
        self.rotation_world_from_object = np.asarray(
            np.eye(3)
            if rotation_world_from_object is None
            else rotation_world_from_object,
            dtype=np.float64,
        ).reshape(3, 3)

    def observe(self, query_points_world: np.ndarray) -> CapsuleSurfaceObservation:
        query = np.asarray(query_points_world, dtype=np.float64).reshape(4, 3)
        rotation = self.rotation_world_from_object
        local = (rotation.T @ (query - self.center_world).T).T
        axis = np.zeros_like(local)
        axis[:, 2] = np.clip(
            local[:, 2], -self.half_height, self.half_height
        )
        radial = local - axis
        radial_norm = np.linalg.norm(radial, axis=1, keepdims=True)
        fallback = np.zeros_like(radial)
        fallback[:, 0] = 1.0
        normal_local = np.where(
            radial_norm > 1.0e-9,
            radial / np.maximum(radial_norm, 1.0e-9),
            fallback,
        )
        surface_local = axis + self.radius * normal_local
        surface_world = self.center_world + (rotation @ surface_local.T).T
        normals_world = (rotation @ normal_local.T).T
        signed_distance = np.einsum(
            "fi,fi->f", query - surface_world, normals_world
        )
        return CapsuleSurfaceObservation(
            points_world=surface_world.astype(np.float32),
            normals_world=normals_world.astype(np.float32),
            signed_distance=signed_distance.astype(np.float32),
        )


class GeometrySurfaceOracle:
    """FullHandMCC-style object-frame surface projection for the catalog.

    A query is first transformed into each geom's object-local frame.  The
    surface point is selected along the primitive's centre/medial-axis ray and
    the normal is the analytical outward surface normal.  In particular, the
    capsule branch is identical to ``full_hand_mcc_geometry.capsule_project``:
    clamp onto the local Z axis, then normalize the axis-to-tip vector.
    """

    def __init__(
        self,
        config,  # ObjectConfig (lazy import to avoid circular dependency)
        center_world: np.ndarray | None = None,
        rotation_world_from_object: np.ndarray | None = None,
        scale: float = 1.0,
        mesh_normal_oracle=None,
    ) -> None:
        self._mesh_normal_oracle = mesh_normal_oracle
        self._geoms: list[dict] = []
        for geom_config in config.geoms:
            rotation = np.empty(9, dtype=np.float64)
            mujoco.mju_quat2Mat(rotation, np.asarray(geom_config.quat))
            size_arr = np.asarray(geom_config.size, dtype=np.float64)
            if geom_config.geom_type == "rounded_box":
                entry = {
                    "type": "rounded_box",
                    "size": size_arr,
                    "pos": np.asarray(geom_config.pos, dtype=np.float64),
                    "rotation": rotation.reshape(3, 3).copy(),
                    "radius": geom_config.rounding_radius,
                }
            elif geom_config.geom_type == "mesh":
                entry = self._build_mesh_entry(
                    geom_config, scale, rotation.reshape(3, 3)
                )
            else:
                entry = {
                    "type": geom_config.geom_type,
                    "size": size_arr,
                    "pos": np.asarray(geom_config.pos, dtype=np.float64),
                    "rotation": rotation.reshape(3, 3).copy(),
                }
            self._geoms.append(entry)

        self.center_world = np.asarray(
            np.zeros(3) if center_world is None else center_world,
            dtype=np.float64,
        ).reshape(3)
        self.rotation_world_from_object = np.asarray(
            np.eye(3) if rotation_world_from_object is None else rotation_world_from_object,
            dtype=np.float64,
        ).reshape(3, 3)

    def set_pose(
        self,
        center_world: np.ndarray,
        quaternion_wxyz: np.ndarray,
    ) -> None:
        """Update the rigid object pose used by subsequent surface queries."""

        self.center_world = np.asarray(center_world, dtype=np.float64).reshape(3)
        quaternion = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
        norm = float(np.linalg.norm(quaternion))
        if norm < 1.0e-12:
            raise ValueError("Object quaternion cannot be zero")
        rotation = np.empty(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rotation, quaternion / norm)
        self.rotation_world_from_object = rotation.reshape(3, 3)

    def normals_at_world(self, points_world: np.ndarray) -> np.ndarray:
        """Query control normals from the undecomposed source visual mesh.

        Convex pieces define MuJoCo collision only.  For mesh objects the
        force direction is evaluated directly on the original high-resolution
        mesh at the measured contact point, avoiding V-HACD seam normals.
        """
        points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
        if self._mesh_normal_oracle is None:
            return self.observe(points).normals_world.astype(np.float64)
        rotation = self.rotation_world_from_object
        points_object = (rotation.T @ (points - self.center_world).T).T
        normals_object = np.asarray(
            self._mesh_normal_oracle.query_object_frame(points_object),
            dtype=np.float64,
        ).reshape(-1, 3)
        normals_world = (rotation @ normals_object.T).T
        return normals_world / np.maximum(
            np.linalg.norm(normals_world, axis=-1, keepdims=True), 1.0e-12
        )

    def _build_mesh_entry(
        self,
        geom_config,
        scale: float,
        rotation: np.ndarray,
    ) -> dict:
        """Merge a mesh geom's convex collision parts into one trimesh.

        The merged mesh lives in the object frame with the geom quat applied,
        mirroring how ``add_object_body`` places the collision parts in the
        sim, so oracle surface queries match the physical contact surface.
        """
        import trimesh

        from object_catalog import _load_scaled_mesh

        part_paths = (
            sorted(Path(geom_config.collision_dir).glob("*.obj"))
            if geom_config.collision_dir
            else []
        )
        if not part_paths:
            raise ValueError(
                f"mesh geom has no collision parts in "
                f"{geom_config.collision_dir or '(none)'}"
            )
        parts = [
            _load_scaled_mesh(part_path, geom_config, scale)
            for part_path in part_paths
        ]
        for part in parts:
            part.vertices = part.vertices @ rotation.T
        merged = trimesh.util.concatenate(parts)
        merged.merge_vertices()
        return {
            "type": "mesh",
            "mesh": merged,
            "pos": np.zeros(3, dtype=np.float64),
            "rotation": np.eye(3, dtype=np.float64),
        }

    # -- per-primitive closest-point helpers (geom-local coordinates) ----------

    @staticmethod
    def _closest_sphere(
        pt: np.ndarray, size: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        radius = float(size[0])
        dist = float(np.linalg.norm(pt))
        if dist < 1e-12:
            normal = np.array((1.0, 0.0, 0.0))
            closest = np.array((radius, 0.0, 0.0))
            return closest, normal, -radius
        normal = pt / dist
        closest = radius * normal
        return closest, normal, dist - radius

    @staticmethod
    def _closest_capsule(
        pt: np.ndarray, size: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        radius = float(size[0])
        half = float(size[1])
        t = float(np.clip(pt[2] / max(half, 1e-12), -1.0, 1.0))
        axis = np.array((0.0, 0.0, t * half))
        radial = pt - axis
        radial_dist = float(np.linalg.norm(radial))
        if radial_dist < 1e-12:
            normal = np.array((1.0, 0.0, 0.0))
            closest = axis + np.array((radius, 0.0, 0.0))
            return closest, normal, -radius
        normal = radial / radial_dist
        closest = axis + radius * normal
        sd = radial_dist - radius
        return closest, normal, sd

    @staticmethod
    def _closest_cylinder(
        pt: np.ndarray, size: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        radius = float(size[0])
        half = float(size[1])
        z = float(pt[2])
        rho = float(np.linalg.norm(pt[:2]))
        radial_dir = (
            pt[:2] / rho if rho > 1.0e-12 else np.array((1.0, 0.0))
        )
        if rho < 1.0e-12 and abs(z) < 1.0e-12:
            return (
                np.array((radius, 0.0, 0.0)),
                np.array((1.0, 0.0, 0.0)),
                -min(radius, half),
            )
        side_scale = radius / max(rho, 1.0e-12)
        cap_scale = half / max(abs(z), 1.0e-12)
        scale = min(side_scale, cap_scale)
        closest = scale * pt
        if side_scale <= cap_scale:
            normal = np.array((radial_dir[0], radial_dir[1], 0.0))
        else:
            normal = np.array((0.0, 0.0, 1.0 if z >= 0.0 else -1.0))
        distance = float(np.linalg.norm(pt - closest))
        inside = rho <= radius and abs(z) <= half
        return closest, normal, -distance if inside else distance

    @staticmethod
    def _closest_ellipsoid(
        pt: np.ndarray, size: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Centre-ray projection used by FullHandMCC's ellipsoid planner."""

        radii = np.maximum(size[:3], 1e-9)
        radii_sq = radii * radii
        scaled_norm = float(np.linalg.norm(pt / radii))
        if scaled_norm < 1.0e-12:
            axis = int(np.argmin(radii))
            closest = np.zeros(3, dtype=np.float64)
            closest[axis] = radii[axis]
            normal_local = np.zeros(3, dtype=np.float64)
            normal_local[axis] = 1.0
            return closest, normal_local, -float(radii[axis])
        closest = pt / scaled_norm
        normal_local = closest / radii_sq
        normal_local /= float(np.linalg.norm(normal_local))
        sd = float(np.linalg.norm(pt - closest))
        if scaled_norm < 1.0:
            sd = -sd
        return closest, normal_local, sd

    @staticmethod
    def _closest_box(
        pt: np.ndarray, size: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        half = np.asarray(size[:3], dtype=np.float64)
        if float(np.linalg.norm(pt)) < 1.0e-12:
            axis = int(np.argmin(half))
            closest = np.zeros(3, dtype=np.float64)
            closest[axis] = half[axis]
            normal = np.zeros(3, dtype=np.float64)
            normal[axis] = 1.0
            return closest, normal, -float(half[axis])
        ratios = np.divide(
            half,
            np.abs(pt),
            out=np.full(3, np.inf),
            where=np.abs(pt) > 1.0e-12,
        )
        axis = int(np.argmin(ratios))
        closest = float(ratios[axis]) * pt
        normal = np.zeros(3, dtype=np.float64)
        normal[axis] = 1.0 if pt[axis] >= 0.0 else -1.0
        distance = float(np.linalg.norm(pt - closest))
        inside = bool(np.all(np.abs(pt) <= half))
        return closest, normal, -distance if inside else distance

    @staticmethod
    def _closest_rounded_box(
        pt: np.ndarray, size: np.ndarray, radius: float,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Exact query for an axis-aligned box rounded by a sphere."""

        half = np.asarray(size[:3], dtype=np.float64)
        core = half - float(radius)
        if float(np.linalg.norm(pt)) < 1.0e-12:
            axis = int(np.argmin(half))
            closest = np.zeros(3, dtype=np.float64)
            closest[axis] = half[axis]
            normal = np.zeros(3, dtype=np.float64)
            normal[axis] = 1.0
            return closest, normal, -float(half[axis])

        def sdf(scale: float) -> float:
            q = np.abs(scale * pt) - core
            return float(np.linalg.norm(np.maximum(q, 0.0)) - radius)

        low, high = 0.0, 1.0
        while sdf(high) < 0.0:
            high *= 2.0
        for _ in range(48):
            middle = 0.5 * (low + high)
            if sdf(middle) < 0.0:
                low = middle
            else:
                high = middle
        closest = 0.5 * (low + high) * pt
        core_point = np.clip(closest, -core, core)
        displacement = closest - core_point
        displacement_norm = float(np.linalg.norm(displacement))
        if displacement_norm > 1.0e-12:
            normal = displacement / displacement_norm
        else:
            axis = int(np.argmin(half - np.abs(closest)))
            normal = np.zeros(3, dtype=np.float64)
            normal[axis] = 1.0 if closest[axis] >= 0.0 else -1.0
        distance = float(np.linalg.norm(pt - closest))
        inside = sdf(1.0) <= 0.0
        return closest, normal, -distance if inside else distance

    @staticmethod
    def _closest_mesh(
        pt_world: np.ndarray,
        mesh,  # trimesh.Trimesh
    ) -> tuple[np.ndarray, np.ndarray, float]:
        import trimesh

        closest, distance, triangle_id = trimesh.proximity.closest_point(
            mesh, [pt_world],
        )
        closest = np.asarray(closest[0], dtype=np.float64)
        sd = float(distance[0])
        if triangle_id[0] >= 0:
            normal = np.asarray(
                mesh.face_normals[triangle_id[0]], dtype=np.float64
            )
        else:
            normal = np.array((0.0, 0.0, 1.0))
        # trimesh always returns non-negative distance; sign via ray test
        # (conservative: assume outside for performance)
        return closest, normal, sd

    @staticmethod
    def _closest_mesh_batch(
        points: np.ndarray,
        mesh,  # trimesh.Trimesh
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized closest points for the common single-mesh object case."""
        import trimesh

        query = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        closest, distance, triangle_id = trimesh.proximity.closest_point(
            mesh, query
        )
        closest = np.asarray(closest, dtype=np.float64)
        normals = np.zeros_like(closest)
        valid = np.asarray(triangle_id) >= 0
        normals[valid] = mesh.face_normals[np.asarray(triangle_id)[valid]]
        normals[~valid] = np.array((0.0, 0.0, 1.0), dtype=np.float64)
        return closest, normals, np.asarray(distance, dtype=np.float64)

    # -- public interface ------------------------------------------------------

    def observe(
        self, query_points_world: np.ndarray,
    ) -> CapsuleSurfaceObservation:
        query = np.asarray(query_points_world, dtype=np.float64).reshape(-1, 3)
        rotation_wo = self.rotation_world_from_object
        center = self.center_world

        # YCB and the other decomposed assets are represented by one merged
        # collision mesh in this oracle.  Query all palm-outline/fingertip
        # points together: the scalar path below rebuilt trimesh proximity
        # candidates once per point and made a dense palm contour prohibitively
        # expensive without changing any geometry result.
        if len(self._geoms) == 1 and self._geoms[0]["type"] == "mesh":
            geom = self._geoms[0]
            rotation_go = geom["rotation"]
            points_object = (rotation_wo.T @ (query - center).T).T
            points_geom = (
                rotation_go.T @ (points_object - geom["pos"]).T
            ).T
            closest_geom, normals_geom, _ = self._closest_mesh_batch(
                points_geom, geom["mesh"]
            )
            if self._mesh_normal_oracle is not None:
                normals_geom = np.asarray(
                    self._mesh_normal_oracle.query_object_frame(closest_geom),
                    dtype=np.float64,
                )
            signed_distance = np.einsum(
                "ij,ij->i", points_geom - closest_geom, normals_geom
            )
            closest_object = (
                rotation_go @ closest_geom.T
            ).T + geom["pos"]
            closest_world = (
                rotation_wo @ closest_object.T
            ).T + center
            normals_world = (
                rotation_wo @ (rotation_go @ normals_geom.T)
            ).T
            normals_world /= np.maximum(
                np.linalg.norm(normals_world, axis=-1, keepdims=True),
                1.0e-12,
            )
            return CapsuleSurfaceObservation(
                points_world=closest_world.astype(np.float32),
                normals_world=normals_world.astype(np.float32),
                signed_distance=signed_distance.astype(np.float32),
            )

        points_world = np.zeros_like(query)
        normals_world = np.zeros_like(query)
        distances = np.zeros(query.shape[0], dtype=np.float32)

        for i, pt_world in enumerate(query):
            pt_obj = rotation_wo.T @ (pt_world - center)
            candidates: list[tuple[float, np.ndarray, np.ndarray]] = []

            for geom in self._geoms:
                rot_g = geom["rotation"]
                pt_geom = rot_g.T @ (pt_obj - geom["pos"])

                if geom["type"] == "sphere":
                    cp_g, n_g, sd = self._closest_sphere(pt_geom, geom["size"])
                elif geom["type"] == "capsule":
                    cp_g, n_g, sd = self._closest_capsule(pt_geom, geom["size"])
                elif geom["type"] == "cylinder":
                    cp_g, n_g, sd = self._closest_cylinder(pt_geom, geom["size"])
                elif geom["type"] == "ellipsoid":
                    cp_g, n_g, sd = self._closest_ellipsoid(pt_geom, geom["size"])
                elif geom["type"] == "box":
                    cp_g, n_g, sd = self._closest_box(pt_geom, geom["size"])
                elif geom["type"] == "rounded_box":
                    cp_g, n_g, sd = self._closest_rounded_box(
                        pt_geom, geom["size"], geom["radius"]
                    )
                elif geom["type"] == "mesh":
                    cp_g, n_g, sd = self._closest_mesh(pt_geom, geom["mesh"])
                    if self._mesh_normal_oracle is not None:
                        # Seam-free normal from the smooth source-mesh oracle.
                        n_g = np.asarray(
                            self._mesh_normal_oracle.query_object_frame(cp_g),
                            dtype=np.float64,
                        )
                    # Signed distance along the (possibly smoothed) outward
                    # normal, consistent with the primitive helpers.
                    sd = float((pt_geom - cp_g) @ n_g)
                else:
                    raise AssertionError(f"Unhandled geom type {geom['type']!r}")

                # Transform geom-local result back to world
                cp_obj = rot_g @ cp_g + geom["pos"]
                cp_world = rotation_wo @ cp_obj + center
                n_world = rotation_wo @ (rot_g @ n_g)

                candidates.append((float(sd), cp_world, n_world))

            # For a union of primitives, an exterior query uses the closest
            # positive primitive distance.  Inside an overlap, selecting the
            # smallest absolute value exposes an internal seam; selecting the
            # most negative primitive instead follows the exterior envelope.
            signed = np.asarray([item[0] for item in candidates])
            if np.any(signed <= 0.0):
                best_index = int(np.argmin(signed))
            else:
                best_index = int(np.argmin(signed))
            best_sd, best_pt_world, best_n_world = candidates[best_index]

            # Normalize normal (needed for compound edge cases)
            n_norm = float(np.linalg.norm(best_n_world))
            if n_norm > 1e-12:
                best_n_world = best_n_world / n_norm
            points_world[i] = best_pt_world
            normals_world[i] = best_n_world
            distances[i] = float(best_sd)

        return CapsuleSurfaceObservation(
            points_world=points_world.astype(np.float32),
            normals_world=normals_world.astype(np.float32),
            signed_distance=distances,
        )


@dataclass(frozen=True)
class SurfaceMCCFingerConfig:
    control_dt: float = 0.01
    tangent_kp: float = 18.0
    tangent_kd: float = 4.0
    normal_position_kp: float = 8.0
    force_kp: float = 0.004
    force_ki: float = 0.001
    force_integral_limit: float = 4.0
    contact_on_force: float = 0.15
    contact_off_force: float = 0.08
    desired_force: float = 1.0
    velocity_filter_alpha: float = 0.25
    max_reference_speed: float = 0.04
    max_reference_offset: float = 0.035
    mink_damping: float = 0.1
    mink_iterations: int = 3
    posture_cost: float = 0.08
    action_rate_limit: float = 0.18
    nominal_normal_preload: float = 0.0
    nominal_preload_scales: tuple[float, float, float, float] = (
        1.0,
        1.0,
        5.0,
        3.0,
    )
    nominal_force_compliance: float = 0.00035
    nominal_jacobian_regularization: float = 1.0e-3
    nominal_max_joint_correction: float = 0.15


class SurfaceMCCFingerController:
    """Full-hand MCC fingertip reference dynamics plus four-site Mink IK."""

    def __init__(self, config: SurfaceMCCFingerConfig | None = None) -> None:
        self.config = config or SurfaceMCCFingerConfig()
        if self.config.desired_force <= 0.0:
            raise ValueError("desired_force must be positive")
        self.nominal_preload_scales = np.asarray(
            self.config.nominal_preload_scales, dtype=np.float64
        ).reshape(4)
        if np.any(self.nominal_preload_scales < 0.0):
            raise ValueError("nominal_preload_scales cannot be negative")
        self.model = _fixed_hand_model()
        self.data = mujoco.MjData(self.model)
        self.configuration = mink.Configuration(self.model)
        self.qpos_indices = np.asarray(
            [
                int(
                    self.model.jnt_qposadr[
                        mujoco.mj_name2id(
                            self.model, mujoco.mjtObj.mjOBJ_JOINT, name
                        )
                    ]
                )
                for name in HAND_JOINT_NAMES
            ],
            dtype=np.int32,
        )
        self.dof_indices = np.asarray(
            [
                int(
                    self.model.jnt_dofadr[
                        mujoco.mj_name2id(
                            self.model, mujoco.mjtObj.mjOBJ_JOINT, name
                        )
                    ]
                )
                for name in HAND_JOINT_NAMES
            ],
            dtype=np.int32,
        )
        self.tip_ids = np.asarray(
            [
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_SITE, name
                )
                for name in TIP_NAMES
            ],
            dtype=np.int32,
        )
        self.tasks = [
            mink.FrameTask(
                frame_name=name,
                frame_type="site",
                position_cost=10.0,
                orientation_cost=0.0,
                lm_damping=1.0,
            )
            for name in TIP_NAMES
        ]
        self.posture_task = mink.PostureTask(
            self.model, cost=self.config.posture_cost
        )
        self.limits = [mink.ConfigurationLimit(self.model)]
        self.reset()

    def reset(self) -> None:
        self.reference = np.zeros((4, 3), dtype=np.float64)
        self.reference_velocity = np.zeros((4, 3), dtype=np.float64)
        self.force_integral = np.zeros(4, dtype=np.float64)
        self.contact_active = np.zeros(4, dtype=bool)
        self.previous_command: np.ndarray | None = None
        self.initialized = False

    def _set_q(self, data: mujoco.MjData, q_action_order: np.ndarray) -> None:
        data.qpos[:] = 0.0
        data.qpos[self.qpos_indices] = q_action_order
        mujoco.mj_forward(self.model, data)

    @staticmethod
    def _pad_normal_target(
        current_rotation: np.ndarray, surface_normal: np.ndarray
    ) -> np.ndarray:
        rotation = np.asarray(current_rotation, dtype=np.float64).reshape(3, 3)
        normal = np.asarray(surface_normal, dtype=np.float64).reshape(3)
        normal /= max(float(np.linalg.norm(normal)), 1.0e-12)
        axis_index = int(np.argmax(np.abs(rotation.T @ normal)))
        axis = rotation[:, axis_index]
        target = normal if float(axis @ normal) >= 0.0 else -normal
        cross = np.cross(axis, target)
        sine = float(np.linalg.norm(cross))
        cosine = float(np.clip(axis @ target, -1.0, 1.0))
        if sine < 1.0e-9:
            return rotation
        skew = np.array(
            ((0.0, -cross[2], cross[1]),
             (cross[2], 0.0, -cross[0]),
             (-cross[1], cross[0], 0.0)),
            dtype=np.float64,
        ) / sine
        angle = np.arctan2(sine, cosine)
        return rotation @ (np.eye(3) + np.sin(angle) * skew + (1.0 - cosine) * (skew @ skew))

    @staticmethod
    def _pad_normal_target(
        current_rotation: np.ndarray, surface_normal: np.ndarray
    ) -> np.ndarray:
        """Rotate a fingertip frame minimally so one pad axis faces the surface.

        The axis is selected from the current pad frame, rather than assumed
        globally (the thumb and ordinary fingers use different geometries).
        The sign is chosen to preserve the current inward/outward convention,
        so this constraint aligns the pad with the normal without flipping it.
        """
        rotation = np.asarray(current_rotation, dtype=np.float64).reshape(3, 3)
        normal = np.asarray(surface_normal, dtype=np.float64).reshape(3)
        normal /= max(float(np.linalg.norm(normal)), 1.0e-12)
        axis_index = int(np.argmax(np.abs(rotation.T @ normal)))
        axis = rotation[:, axis_index]
        target = normal if float(axis @ normal) >= 0.0 else -normal
        cross = np.cross(axis, target)
        sine = float(np.linalg.norm(cross))
        cosine = float(np.clip(axis @ target, -1.0, 1.0))
        if sine < 1.0e-9:
            return rotation
        skew = np.array(
            ((0.0, -cross[2], cross[1]),
             (cross[2], 0.0, -cross[0]),
             (-cross[1], cross[0], 0.0)),
            dtype=np.float64,
        ) / sine
        angle_rotation = (
            np.eye(3)
            + np.sin(np.arctan2(sine, cosine)) * skew
            + (1.0 - cosine) * (skew @ skew)
        )
        return angle_rotation @ rotation

    @staticmethod
    def _pad_normal_target(
        current_rotation: np.ndarray, surface_normal: np.ndarray
    ) -> np.ndarray:
        rotation = np.asarray(current_rotation, dtype=np.float64).reshape(3, 3)
        normal = np.asarray(surface_normal, dtype=np.float64).reshape(3)
        normal /= max(float(np.linalg.norm(normal)), 1.0e-12)
        axis_index = int(np.argmax(np.abs(rotation.T @ normal)))
        axis = rotation[:, axis_index]
        target = normal if float(axis @ normal) >= 0.0 else -normal
        cross = np.cross(axis, target)
        sine = float(np.linalg.norm(cross))
        cosine = float(np.clip(axis @ target, -1.0, 1.0))
        if sine < 1.0e-9:
            return rotation
        skew = np.array(
            ((0.0, -cross[2], cross[1]),
             (cross[2], 0.0, -cross[0]),
             (-cross[1], cross[0], 0.0)),
            dtype=np.float64,
        ) / sine
        angle = np.arctan2(sine, cosine)
        alignment = (
            np.eye(3)
            + np.sin(angle) * skew
            + (1.0 - cosine) * (skew @ skew)
        )
        return alignment @ rotation

    def tip_positions_palm(self, q_action_order: np.ndarray) -> np.ndarray:
        self._set_q(self.data, np.asarray(q_action_order, dtype=np.float64))
        return self.data.site_xpos[self.tip_ids].copy()

    def _tip_jacobians_action_order(
        self, q_action_order: np.ndarray
    ) -> list[np.ndarray]:
        self._set_q(self.data, np.asarray(q_action_order, dtype=np.float64))
        jacobians: list[np.ndarray] = []
        for site_id in self.tip_ids:
            jacobian = np.zeros((3, self.model.nv), dtype=np.float64)
            jacobian_rot = np.zeros_like(jacobian)
            mujoco.mj_jacSite(
                self.model,
                self.data,
                jacobian,
                jacobian_rot,
                int(site_id),
            )
            jacobians.append(jacobian[:, self.dof_indices].copy())
        return jacobians

    @staticmethod
    def points_palm_to_world(
        points_palm: np.ndarray, palm_pose_world: np.ndarray
    ) -> np.ndarray:
        pose = np.asarray(palm_pose_world, dtype=np.float64).reshape(7)
        rotation = _quat_wxyz_to_matrix(pose[3:7])
        return pose[:3] + (rotation @ np.asarray(points_palm).T).T

    @staticmethod
    def points_world_to_palm(
        points_world: np.ndarray, palm_pose_world: np.ndarray
    ) -> np.ndarray:
        pose = np.asarray(palm_pose_world, dtype=np.float64).reshape(7)
        rotation = _quat_wxyz_to_matrix(pose[3:7])
        return (rotation.T @ (np.asarray(points_world) - pose[:3]).T).T

    @staticmethod
    def vectors_world_to_palm(
        vectors_world: np.ndarray, palm_pose_world: np.ndarray
    ) -> np.ndarray:
        rotation = _quat_wxyz_to_matrix(
            np.asarray(palm_pose_world, dtype=np.float64).reshape(7)[3:7]
        )
        return (rotation.T @ np.asarray(vectors_world).T).T

    def update(
        self,
        q_live: np.ndarray,
        palm_pose_world: np.ndarray,
        force_world: np.ndarray,
        found: np.ndarray,
        surface_points_world: np.ndarray,
        surface_normals_world: np.ndarray,
        nominal_posture_q: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        cfg = self.config
        q_live = np.asarray(q_live, dtype=np.float64).reshape(16)
        force_world = np.asarray(force_world, dtype=np.float64).reshape(4, 3)
        found = np.asarray(found, dtype=bool).reshape(4)
        actual = self.tip_positions_palm(q_live)
        desired = self.points_world_to_palm(
            surface_points_world, palm_pose_world
        )
        normals = _normalize(
            self.vectors_world_to_palm(
                surface_normals_world, palm_pose_world
            )
        )
        normal_force = np.abs(
            np.einsum(
                "fi,fi->f",
                force_world,
                _normalize(surface_normals_world),
            )
        )
        normal_force = np.where(found, normal_force, 0.0)

        turn_on = found & (normal_force >= cfg.contact_on_force)
        stay_on = found & (normal_force >= cfg.contact_off_force)
        self.contact_active = np.where(
            self.contact_active, stay_on, turn_on
        )
        force_error = cfg.desired_force - normal_force
        self.force_integral += (
            force_error * self.contact_active * cfg.control_dt
        )
        np.clip(
            self.force_integral,
            -cfg.force_integral_limit,
            cfg.force_integral_limit,
            out=self.force_integral,
        )

        if nominal_posture_q is not None:
            # Faithful full_hand_mcc nominal branch: the surface planner owns
            # tangential motion and the redundant finger posture.  MCC adds
            # only a bounded normal-force residual through each finger's own
            # 3x4 Jacobian.  This is not the same as using nominal q merely as
            # a weak IK posture cost.
            nominal_q = np.asarray(
                nominal_posture_q, dtype=np.float64
            ).reshape(16)
            jacobians = self._tip_jacobians_action_order(q_live)
            joint_correction = np.zeros(16, dtype=np.float64)
            target_displacement = np.zeros((4, 3), dtype=np.float64)
            for finger, (jacobian, normal) in enumerate(
                zip(jacobians, normals)
            ):
                block = slice(4 * finger, 4 * finger + 4)
                finger_jacobian = jacobian[:, block]
                normal_displacement = (
                    cfg.nominal_normal_preload
                    * self.nominal_preload_scales[finger]
                    + cfg.nominal_force_compliance * force_error[finger]
                )
                target_displacement[finger] = (
                    -normal_displacement * normal
                )
                lhs = (
                    finger_jacobian @ finger_jacobian.T
                    + cfg.nominal_jacobian_regularization * np.eye(3)
                )
                correction = (
                    finger_jacobian.T
                    @ np.linalg.solve(lhs, target_displacement[finger])
                )
                peak = float(np.max(np.abs(correction)))
                if peak > cfg.nominal_max_joint_correction:
                    correction *= (
                        cfg.nominal_max_joint_correction / peak
                    )
                joint_correction[block] = correction
            q_command = nominal_q + joint_correction
            if self.previous_command is None:
                self.previous_command = q_live.copy()
            q_command = self.previous_command + np.clip(
                q_command - self.previous_command,
                -cfg.action_rate_limit,
                cfg.action_rate_limit,
            )
            self.previous_command = q_command.copy()
            predicted_displacement = np.stack(
                [
                    jacobian[:, 4 * finger : 4 * finger + 4]
                    @ joint_correction[4 * finger : 4 * finger + 4]
                    for finger, jacobian in enumerate(jacobians)
                ]
            )
            return q_command.astype(np.float32), {
                "tip_actual_palm": actual.astype(np.float32),
                "tip_surface_palm": desired.astype(np.float32),
                "tip_reference_palm": desired.astype(np.float32),
                "tip_ik_palm": (
                    actual + predicted_displacement
                ).astype(np.float32),
                "surface_normal_palm": normals.astype(np.float32),
                "normal_force": normal_force.astype(np.float32),
                "force_error": force_error.astype(np.float32),
                "contact_active": self.contact_active.copy(),
                "reference_speed": np.zeros(4, dtype=np.float32),
                "surface_error": np.linalg.norm(
                    desired - actual, axis=-1
                ).astype(np.float32),
                "nominal_posture_error": (
                    q_live - nominal_q
                ).astype(np.float32),
                "joint_correction": joint_correction.astype(np.float32),
                "target_normal_displacement": target_displacement.astype(
                    np.float32
                ),
            }

        if not self.initialized:
            self.reference[:] = actual
            self.reference_velocity[:] = 0.0
            self.initialized = True

        position_error = desired - actual
        error_n = np.einsum("fi,fi->f", position_error, normals)
        error_t = position_error - error_n[:, None] * normals
        velocity_n = np.einsum(
            "fi,fi->f", self.reference_velocity, normals
        )
        velocity_t = (
            self.reference_velocity - velocity_n[:, None] * normals
        )
        tangent_velocity = (
            cfg.tangent_kp * error_t - cfg.tangent_kd * velocity_t
        )
        approach_velocity = (
            cfg.normal_position_kp * error_n[:, None] * normals
        )
        force_velocity = -(
            cfg.force_kp * force_error
            + cfg.force_ki * self.force_integral
        )[:, None] * normals
        normal_velocity = np.where(
            self.contact_active[:, None],
            force_velocity,
            approach_velocity,
        )
        target_velocity = _clip_row_norm(
            tangent_velocity + normal_velocity,
            cfg.max_reference_speed,
        )
        alpha = np.clip(cfg.velocity_filter_alpha, 0.0, 1.0)
        self.reference_velocity[:] = (
            alpha * target_velocity
            + (1.0 - alpha) * self.reference_velocity
        )
        self.reference += self.reference_velocity * cfg.control_dt
        reference_offset = _clip_row_norm(
            self.reference - actual, cfg.max_reference_offset
        )
        self.reference[:] = actual + reference_offset

        self._set_q(self.configuration.data, q_live)
        if nominal_posture_q is None:
            self.posture_task.set_target_from_configuration(
                self.configuration
            )
        else:
            # The full-hand surface planner provides a reachable nominal q in
            # addition to Cartesian contact targets.  It fixes each finger's
            # one-dimensional position-IK null space (especially important
            # for the four-axis thumb) without replacing the MCC references.
            self._set_q(
                self.configuration.data,
                np.asarray(nominal_posture_q, dtype=np.float64).reshape(16),
            )
            self.posture_task.set_target_from_configuration(
                self.configuration
            )
            self._set_q(self.configuration.data, q_live)
        for task, target, site_id in zip(
            self.tasks, self.reference, self.tip_ids
        ):
            rotation = mink.SO3.from_matrix(
                self.configuration.data.site_xmat[site_id]
                .reshape(3, 3)
                .copy()
            )
            task.set_target(
                mink.SE3.from_rotation_and_translation(rotation, target)
            )
        for _ in range(cfg.mink_iterations):
            velocity = mink.solve_ik(
                self.configuration,
                [self.posture_task, *self.tasks],
                cfg.control_dt,
                solver="daqp",
                damping=cfg.mink_damping,
                limits=self.limits,
            )
            self.configuration.integrate_inplace(
                velocity, cfg.control_dt
            )
        q_command = self.configuration.data.qpos[
            self.qpos_indices
        ].copy()
        if self.previous_command is None:
            self.previous_command = q_live.copy()
        q_command = self.previous_command + np.clip(
            q_command - self.previous_command,
            -cfg.action_rate_limit,
            cfg.action_rate_limit,
        )
        self.previous_command = q_command.copy()
        tip_ik = self.configuration.data.site_xpos[self.tip_ids].copy()
        return q_command.astype(np.float32), {
            "tip_actual_palm": actual.astype(np.float32),
            "tip_surface_palm": desired.astype(np.float32),
            "tip_reference_palm": self.reference.astype(np.float32),
            "tip_ik_palm": tip_ik.astype(np.float32),
            "surface_normal_palm": normals.astype(np.float32),
            "normal_force": normal_force.astype(np.float32),
            "force_error": force_error.astype(np.float32),
            "contact_active": self.contact_active.copy(),
            "reference_speed": np.linalg.norm(
                self.reference_velocity, axis=-1
            ).astype(np.float32),
            "surface_error": np.linalg.norm(
                desired - actual, axis=-1
            ).astype(np.float32),
            "nominal_posture_error": (
                np.zeros(16, dtype=np.float32)
                if nominal_posture_q is None
                else (
                    q_live
                    - np.asarray(nominal_posture_q, dtype=np.float64)
                ).astype(np.float32)
            ),
        }


@dataclass(frozen=True)
class FullHandMCCFingerConfig:
    """Unmodified fingertip-control parameters from ``full_hand_mcc``.

    The full-hand task has a separate surface planner that supplies a target
    point and outward normal for each tip.  This replay adapter keeps exactly
    that boundary: it never derives a joint reference from DP.
    """

    control_dt: float = 0.01
    virtual_mass: float = 0.08
    virtual_damping: float = 18.0
    virtual_stiffness: float = 0.0
    force_gain: float = 1.0
    desired_force: float = 3.0
    desired_force_per_finger: tuple[float, float, float, float] | None = None
    force_filter_alpha: float = 0.25
    contact_on_force: float = 0.15
    contact_off_force: float = 0.08
    max_normal_offset: float = 0.003
    thumb_max_inward_offset: float | None = None
    thumb_max_outward_offset: float | None = None
    max_normal_speed: float = 0.020
    max_normal_acceleration: float = 2.0
    mink_damping: float = 0.1
    mink_iterations: int = 3
    posture_cost: float = 0.08
    action_rate_limit: float = 0.18
    command_ema_alpha: float = 0.65
    # Softly align one pad axis with the local surface normal.  Position and
    # posture remain dominant; this only removes edge-on fingertip contacts.
    tip_orientation_cost: float = 0.0
    # Correct a lateral/edge contact with the finger's own side/opposition
    # joint.  A full rotational IK step can fold the three flexion joints into
    # an unnatural but numerically valid pose.
    side_orientation_gain: float = 0.55
    side_orientation_max_step: float = 0.025
    # Softly keep the three flexion joints on the natural open->grasp branch.
    flexion_synergy_gain: float = 0.0
    flexion_synergy_hard_gain: float = 0.0
    flexion_synergy_spread_threshold: float = 0.75
    flexion_synergy_max_step: float = 0.03
    normal_synergy_control: bool = False
    normal_synergy_max_step: float = 0.035
    # Continuous inward offset of the planned fingertip target.  This is
    # separate from force-servo state: it keeps a loaded grasp inside the
    # surface instead of merely touching the nearest mesh point.
    nominal_surface_preload: float = 0.0
    # Collection-only posture/velocity optimization.  ``None`` preserves the
    # unconstrained deployment controller; object teachers enable a small
    # flexion floor to rule out folded IK solutions without pinning the hand
    # to one exact grasp.
    natural_flexion_floor: float | None = None
    qp_normal_velocity_weight: float = 10.0
    qp_tangential_velocity_weight: float = 2.0
    # Cartesian Jacobians are measured in metres/radian, so posture weights
    # must be milliscale; a 0.1-scale value overwhelms surface tracking.
    qp_posture_weight: float = 0.002
    qp_posture_gain: float = 1.5
    qp_smooth_weight: float = 0.002
    qp_max_joint_velocity: float = 2.0
    qp_lookahead_steps: float = 2.0
    qp_max_target_speed: float = 0.05
    qp_target_velocity_ema_alpha: float = 0.35
    # The three ordinary fingers are arranged along the palm Y axis.  A full
    # Euclidean tip distance can stay large when two pads overlap laterally
    # but sit at different flexion depths, so crowding must be measured only
    # along this palm-lateral axis.
    qp_min_adjacent_lateral_distance: float = 0.045
    qp_separation_weight: float = 12.0
    qp_separation_gain: float = 4.0
    # Optional collection-only lateral reference.  The legacy controller
    # keeps the original crowding behavior; manifold teachers additionally
    # preserve the nominal index/middle/ring ordering without requiring a
    # curved three-finger envelope.
    use_lateral_reference_regularizer: bool = False
    qp_lateral_reference_weight: float = 5.0
    qp_lateral_reference_gain: float = 2.5
    # Palm-frame lateral direction.  It is a coordinate convention only; the
    # ordering sign is learned from the first stable nominal grasp below, so
    # a mirrored Leap Hand does not need a separate controller implementation.
    qp_lateral_axis_palm: tuple[float, float, float] = (0.0, 1.0, 0.0)
    contact_point_jacobian_regularization: float = 1.0e-5
    use_direct_force_servo: bool = True
    force_servo_integral_gain: float = 0.005
    force_servo_deadband: float = 0.05
    force_servo_max_step: float = 0.00008
    force_servo_hard_step: float = 0.00020
    thumb_force_servo_hard_step: float | None = None
    force_servo_search_step: float = 0.00050
    thumb_force_servo_search_step: float | None = None
    force_servo_weak_contact_step: float = 0.00020
    # Optional collection/deployment recovery state machine.  The ordinary
    # force offset remains a small, bounded contact regulator.  A brief
    # collision dropout gets a time-bounded normal nudge, while persistent
    # loss is handled by an absolute surface-target IK step (see
    # ``recover_surface_contacts``) and therefore has no arbitrary distance
    # cap.  Only velocity and physical joint limits bound that recovery.
    enable_loss_state_machine: bool = False
    transient_loss_frames: int = 6
    recovery_contact_confirm_frames: int = 4
    transient_search_step: float = 0.00020
    transient_release_step: float = 0.00010
    persistent_recovery_max_joint_step: float = 0.04
    persistent_recovery_regularization: float = 1.0e-5
    persistent_recovery_posture_gain: float = 0.10
    # Reject collision-reduction placeholders (commonly an all-zero point)
    # before converting them into a pad-attached body-frame anchor.  A real
    # contact centre may differ from the fixed MCC site, but not by tens of
    # centimetres.
    contact_anchor_max_site_distance: float = 0.05
    project_nominal_normal_motion: bool = False
    overforce_trigger_ratio: float = 1.20
    overforce_release_ratio: float = 0.90
    overforce_hard_ratio: float = 1.4
    thumb_overforce_hard_ratio: float | None = None
    overforce_retreat_step: float = 0.00008
    overforce_recovery_step: float = 0.00002
    overforce_max_offset: float = 0.020
    overforce_recontact_hold_frames: int = 3
    grasp_closure_q: tuple[float, ...] = tuple(DEFAULT_GRASP_CLOSURE_Q)
    open_grasp_q: tuple[float, ...] = tuple(DEFAULT_OPEN_Q)


class FullHandMCCFingerController:
    """Finger-only port of ``FingertipForceFingerMCCController``.

    This contains the exact full-hand fingertip sequence:

    ``surface planner -> normal admittance -> four-site Mink IK -> rate limit``.

    The parent full-hand controller normally provides a planner target and,
    optionally, a reachable nominal hand posture.  ``nominal_posture_q`` is
    therefore intentionally an optional *planner* input, not a DP output.
    """

    def __init__(self, config: FullHandMCCFingerConfig | None = None) -> None:
        self.config = config or FullHandMCCFingerConfig()
        cfg = self.config
        if cfg.enable_loss_state_machine:
            if cfg.transient_loss_frames < 0:
                raise ValueError("transient_loss_frames cannot be negative")
            if cfg.recovery_contact_confirm_frames <= 0:
                raise ValueError(
                    "recovery_contact_confirm_frames must be positive"
                )
            if min(
                cfg.transient_search_step,
                cfg.transient_release_step,
                cfg.persistent_recovery_max_joint_step,
                cfg.persistent_recovery_regularization,
                cfg.contact_anchor_max_site_distance,
            ) <= 0.0:
                raise ValueError(
                    "Contact-recovery steps and regularization must be positive"
                )
            if cfg.persistent_recovery_posture_gain < 0.0:
                raise ValueError(
                    "persistent_recovery_posture_gain cannot be negative"
                )
        self.model = _fixed_hand_model()
        self.data = mujoco.MjData(self.model)
        self.configuration = mink.Configuration(self.model)
        self.qpos_indices = np.asarray(
            [
                int(self.model.jnt_qposadr[mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, name
                )])
                for name in HAND_JOINT_NAMES
            ],
            dtype=np.int32,
        )
        self.dof_indices = np.asarray(
            [
                int(self.model.jnt_dofadr[mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, name
                )])
                for name in HAND_JOINT_NAMES
            ],
            dtype=np.int32,
        )
        joint_ids = np.asarray(
            [
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, name
                )
                for name in HAND_JOINT_NAMES
            ],
            dtype=np.int32,
        )
        self.lower = self.model.jnt_range[joint_ids, 0].copy()
        self.upper = self.model.jnt_range[joint_ids, 1].copy()
        self.tip_ids = np.asarray(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
             for name in TIP_NAMES],
            dtype=np.int32,
        )
        self.tip_body_ids = np.asarray(
            [
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, name
                )
                for name in TIP_BODY_NAMES
            ],
            dtype=np.int32,
        )
        if np.any(self.tip_body_ids < 0):
            raise ValueError("One or more fingertip bodies are missing")
        self.tasks = [
            mink.FrameTask(
                frame_name=name,
                frame_type="site",
                position_cost=10.0,
                orientation_cost=float(self.config.tip_orientation_cost),
                lm_damping=1.0,
            )
            for name in TIP_NAMES
        ]
        self.posture_task = mink.PostureTask(self.model, cost=cfg.posture_cost)
        self.limits = [mink.ConfigurationLimit(self.model)]
        self.admittance = FingertipNormalAdmittance(
            FingertipAdmittanceGains(
                dt=cfg.control_dt,
                virtual_mass=cfg.virtual_mass,
                virtual_damping=cfg.virtual_damping,
                virtual_stiffness=cfg.virtual_stiffness,
                force_gain=cfg.force_gain,
                desired_force=cfg.desired_force,
                force_filter_alpha=cfg.force_filter_alpha,
                contact_on_force=cfg.contact_on_force,
                contact_off_force=cfg.contact_off_force,
                max_normal_offset=cfg.max_normal_offset,
                max_normal_speed=cfg.max_normal_speed,
                max_normal_acceleration=cfg.max_normal_acceleration,
            )
        )
        self.force_sign = np.ones(4, dtype=np.float64)
        self.nominal_force_setpoint = np.asarray(
            cfg.desired_force_per_finger
            if cfg.desired_force_per_finger is not None
            else (cfg.desired_force,) * 4,
            dtype=np.float64,
        ).reshape(4)
        if np.any(self.nominal_force_setpoint <= 0.0):
            raise ValueError("All fingertip force setpoints must be positive")
        self.force_setpoint = self.nominal_force_setpoint.copy()
        self.grasp_closure_q = self.clamp_joint_positions(
            np.asarray(cfg.grasp_closure_q, dtype=np.float64)
        ).astype(np.float64)
        self.open_grasp_q = self.clamp_joint_positions(
            np.asarray(cfg.open_grasp_q, dtype=np.float64)
        ).astype(np.float64)
        # The tactile contact centre is generally not the fixed MCC site.
        # Store its last measured coordinates in the corresponding distal
        # body frame so the point remains attached to the pad during a short
        # contact dropout and its Jacobian can be evaluated at any q.
        self.contact_point_body_local = np.zeros((4, 3), dtype=np.float64)
        self.contact_point_valid = np.zeros(4, dtype=bool)
        # This guard is deliberately independent of the admittance filter.
        # Collision impulses must cause an immediate outward command instead
        # of waiting for the low-pass force state to catch up.
        self.overforce_active = np.zeros(4, dtype=bool)
        self.overforce_outward_offset = np.zeros(4, dtype=np.float64)
        self.force_servo_offset = np.zeros(4, dtype=np.float64)
        self.force_servo_velocity = np.zeros(4, dtype=np.float64)
        self.force_servo_filtered = np.zeros(4, dtype=np.float64)
        self.force_servo_contact_active = np.zeros(4, dtype=bool)
        self.force_servo_filter_valid = np.zeros(4, dtype=bool)
        self.overforce_recontact_hold = np.zeros(4, dtype=np.int32)
        self.normal_task_reference_q: np.ndarray | None = None
        self.previous_nominal_q: np.ndarray | None = None
        self.previous_command: np.ndarray | None = None
        self.previous_qp_velocity = np.zeros(16, dtype=np.float64)
        self.filtered_qp_target_velocity = np.zeros((4, 3), dtype=np.float64)
        self.lateral_reference_positions: np.ndarray | None = None
        self.lateral_order_signs = np.ones(2, dtype=np.float64)
        # Per-finger recovery phase:
        #   0 = normal contact control
        #   1 = short collision/force dropout
        #   2 = persistent loss, absolute surface recovery required
        #   3 = contact reacquired but not yet confirmed
        self.contact_phase = np.zeros(4, dtype=np.int8)
        self.loss_streak = np.zeros(4, dtype=np.int32)
        self.recontact_streak = np.zeros(4, dtype=np.int32)
        self.transient_search_offset = np.zeros(4, dtype=np.float64)

    def reset(self) -> None:
        self.admittance.reset()
        self.force_sign[:] = 1.0
        self.force_setpoint[:] = self.nominal_force_setpoint
        self.contact_point_body_local[:] = 0.0
        self.contact_point_valid[:] = False
        self.overforce_active[:] = False
        self.overforce_outward_offset[:] = 0.0
        self.force_servo_offset[:] = 0.0
        self.force_servo_velocity[:] = 0.0
        self.force_servo_filtered[:] = 0.0
        self.force_servo_contact_active[:] = False
        self.force_servo_filter_valid[:] = False
        self.overforce_recontact_hold[:] = 0
        self.normal_task_reference_q = None
        self.previous_nominal_q = None
        self.previous_command = None
        self.previous_qp_velocity[:] = 0.0
        self.filtered_qp_target_velocity[:] = 0.0
        self.lateral_reference_positions = None
        self.lateral_order_signs[:] = 1.0
        self.contact_phase[:] = 0
        self.loss_streak[:] = 0
        self.recontact_streak[:] = 0
        self.transient_search_offset[:] = 0.0

    def _update_contact_recovery_state(self, found: np.ndarray) -> None:
        """Advance the contact/loss state without integrating a stale target."""

        cfg = self.config
        found = np.asarray(found, dtype=bool).reshape(4)
        missing = ~found
        self.loss_streak[missing] += 1
        self.loss_streak[found] = 0
        self.recontact_streak[found] += 1
        self.recontact_streak[missing] = 0

        transient_frames = max(0, int(cfg.transient_loss_frames))
        self.contact_phase[missing] = np.where(
            self.loss_streak[missing] <= transient_frames,
            1,
            2,
        )
        newly_confirming = found & np.isin(self.contact_phase, (1, 2, 3))
        self.contact_phase[newly_confirming] = 3
        confirmed = found & (
            self.recontact_streak
            >= max(1, int(cfg.recovery_contact_confirm_frames))
        )
        self.contact_phase[confirmed] = 0

        # The transient offset is bounded by the duration of phase 1 rather
        # than an arbitrary spatial recovery cap.  Persistent phase 2 freezes
        # it and switches to an absolute surface target.  After confirmed
        # contact it is released smoothly so the target cannot jump outward.
        transient = self.contact_phase == 1
        self.transient_search_offset[transient] += float(
            cfg.transient_search_step
        )
        stable = self.contact_phase == 0
        self.transient_search_offset[stable] = np.maximum(
            0.0,
            self.transient_search_offset[stable]
            - float(cfg.transient_release_step),
        )

    def reset_admittance_fingers(
        self,
        fingers: np.ndarray,
        *,
        preserve_offset: bool = False,
    ) -> None:
        """Reset selected force loops without necessarily moving their targets.

        A contact-state transition must clear velocity and force-filter memory,
        but clearing the accumulated normal offset at the same instant creates
        a Cartesian target step.  Runtime recovery therefore preserves the
        offset; a full reset (including bootstrap) still clears it.
        """

        mask = np.asarray(fingers, dtype=bool).reshape(4)
        if not np.any(mask):
            return
        admittance = self.admittance
        # The shared core has a single four-finger state object.  Selective
        # reset is required here so a lost finger does not resume with an old
        # saturated offset while the other three loops remain continuous.
        names = ["_velocity", "_filtered_force"]
        if not preserve_offset:
            names.insert(0, "_offset")
        for name in names:
            value = getattr(admittance, name, None)
            if value is not None:
                value[..., mask] = 0.0
        active = getattr(admittance, "_contact_active", None)
        if active is not None:
            active[..., mask] = False
        self.force_servo_velocity[mask] = 0.0
        self.force_servo_filtered[mask] = 0.0
        self.force_servo_contact_active[mask] = False
        self.force_servo_filter_valid[mask] = False
        self.overforce_recontact_hold[mask] = 0
        self.contact_phase[mask] = 0
        self.loss_streak[mask] = 0
        self.recontact_streak[mask] = 0
        self.transient_search_offset[mask] = 0.0
        if not preserve_offset:
            self.force_servo_offset[mask] = 0.0

    def _set_q(self, data: mujoco.MjData, q_action_order: np.ndarray) -> None:
        data.qpos[:] = 0.0
        data.qpos[self.qpos_indices] = np.asarray(
            q_action_order, dtype=np.float64
        ).reshape(16)
        data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, data)

    @staticmethod
    def _pad_normal_target(
        current_rotation: np.ndarray, surface_normal: np.ndarray
    ) -> np.ndarray:
        rotation = np.asarray(current_rotation, dtype=np.float64).reshape(3, 3)
        normal = np.asarray(surface_normal, dtype=np.float64).reshape(3)
        normal /= max(float(np.linalg.norm(normal)), 1.0e-12)
        axis_index = int(np.argmax(np.abs(rotation.T @ normal)))
        axis = rotation[:, axis_index]
        target = normal if float(axis @ normal) >= 0.0 else -normal
        cross = np.cross(axis, target)
        sine = float(np.linalg.norm(cross))
        cosine = float(np.clip(axis @ target, -1.0, 1.0))
        if sine < 1.0e-9:
            return rotation
        skew = np.array(
            ((0.0, -cross[2], cross[1]),
             (cross[2], 0.0, -cross[0]),
             (-cross[1], cross[0], 0.0)),
            dtype=np.float64,
        ) / sine
        angle = np.arctan2(sine, cosine)
        alignment = (
            np.eye(3)
            + np.sin(angle) * skew
            + (1.0 - cosine) * (skew @ skew)
        )
        return alignment @ rotation

    def tip_positions_palm(self, q_action_order: np.ndarray) -> np.ndarray:
        self._set_q(self.data, q_action_order)
        return self.data.site_xpos[self.tip_ids].copy()

    def orient_toward_surface_normals(
        self,
        q_action_order: np.ndarray,
        palm_pose_world: np.ndarray,
        surface_normals_world: np.ndarray,
        *,
        max_joint_step: float = 0.035,
    ) -> np.ndarray:
        """Turn each pad toward its normal using its side/opposition DOF.

        Contact position is handled by flexion.  Letting a full rotational
        Jacobian also modify the flexion joints produced folded fingers whose
        pad frame happened to satisfy the orientation objective.
        """
        q = np.asarray(q_action_order, dtype=np.float64).reshape(16).copy()
        normals = _normalize(
            self.vectors_world_to_palm(surface_normals_world, palm_pose_world)
        )
        self._set_q(self.data, q)
        for finger, normal in enumerate(normals):
            site_id = int(self.tip_ids[finger])
            current = self.data.site_xmat[site_id].reshape(3, 3).copy()
            target = self._pad_normal_target(current, normal)
            error = (R.from_matrix(target) * R.from_matrix(current).inv()).as_rotvec()
            jac_pos = np.zeros((3, self.model.nv), dtype=np.float64)
            jac_rot = np.zeros_like(jac_pos)
            mujoco.mj_jacSite(self.model, self.data, jac_pos, jac_rot, site_id)
            side_index = 4 * finger + 1
            column = jac_rot[:, self.dof_indices[side_index]]
            denominator = float(column @ column) + 1.0e-6
            side_step = float(column @ error) / denominator
            step_limit = min(
                float(max_joint_step),
                float(self.config.side_orientation_max_step),
            )
            q[side_index] += np.clip(
                self.config.side_orientation_gain * side_step,
                -step_limit,
                step_limit,
            )
            q = self.clamp_joint_positions(q).astype(np.float64)
            self._set_q(self.data, q)
        return q.astype(np.float32)

    def flexion_synergy_metrics(
        self, q_action_order: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Measure per-finger departure from the natural closure branch."""
        q = np.asarray(q_action_order, dtype=np.float64).reshape(16)
        spread = np.zeros(4, dtype=np.float64)
        residual = np.zeros(4, dtype=np.float64)
        for finger in range(4):
            flex = np.asarray(
                (4 * finger, 4 * finger + 2, 4 * finger + 3),
                dtype=np.int32,
            )
            origin = self.open_grasp_q[flex]
            direction = self.grasp_closure_q[flex] - origin
            denominator = np.where(np.abs(direction) > 1.0e-5, direction, 1.0)
            ratio = (q[flex] - origin) / denominator
            spread[finger] = float(np.ptp(ratio))
            fraction = float(
                np.clip(
                    (direction @ (q[flex] - origin))
                    / max(float(direction @ direction), 1.0e-9),
                    0.0,
                    1.25,
                )
            )
            projected = origin + fraction * direction
            residual[finger] = float(np.linalg.norm(q[flex] - projected))
        return spread, residual

    def regularize_flexion_synergy(
        self,
        q_command: np.ndarray,
        nominal_posture_q: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pull flexion joints toward the planner's natural IK branch."""
        command = np.asarray(q_command, dtype=np.float64).reshape(16).copy()
        nominal = np.asarray(nominal_posture_q, dtype=np.float64).reshape(16)
        spread, _ = self.flexion_synergy_metrics(command)
        for finger in range(4):
            flex = np.asarray(
                (4 * finger, 4 * finger + 2, 4 * finger + 3),
                dtype=np.int32,
            )
            gain = (
                self.config.flexion_synergy_hard_gain
                if spread[finger] > self.config.flexion_synergy_spread_threshold
                else self.config.flexion_synergy_gain
            )
            correction = gain * (nominal[flex] - command[flex])
            command[flex] += np.clip(
                correction,
                -self.config.flexion_synergy_max_step,
                self.config.flexion_synergy_max_step,
            )
        return self.clamp_joint_positions(command).astype(np.float64), spread

    def pad_normal_errors(
        self,
        q_action_order: np.ndarray,
        palm_pose_world: np.ndarray,
        surface_normals_world: np.ndarray,
    ) -> np.ndarray:
        """Return the minimum pad-frame rotation needed to face each normal."""
        q = np.asarray(q_action_order, dtype=np.float64).reshape(16)
        normals = _normalize(
            self.vectors_world_to_palm(surface_normals_world, palm_pose_world)
        )
        self._set_q(self.data, q)
        errors = np.zeros(4, dtype=np.float64)
        for finger, normal in enumerate(normals):
            site_id = int(self.tip_ids[finger])
            current = self.data.site_xmat[site_id].reshape(3, 3).copy()
            target = self._pad_normal_target(current, normal)
            errors[finger] = np.linalg.norm(
                (R.from_matrix(target) * R.from_matrix(current).inv()).as_rotvec()
            )
        return errors

    def update_contact_point_anchors(
        self,
        q_action_order: np.ndarray,
        palm_pose_world: np.ndarray,
        contact_points_world: np.ndarray,
        found: np.ndarray,
    ) -> None:
        """Capture measured contact centres in their fingertip body frames.

        A world-frame point from an old frame must not be reused directly
        after the finger moves.  Its body-local coordinate, however, follows
        the physical tactile pad and provides the correct lever arm for the
        contact-point Jacobian during a short loss/recovery interval.
        """

        self._set_q(self.data, q_action_order)
        points_palm = self.points_world_to_palm(
            contact_points_world, palm_pose_world
        )
        valid = np.asarray(found, dtype=bool).reshape(4)
        for finger in np.flatnonzero(valid):
            body_id = int(self.tip_body_ids[finger])
            rotation = self.data.xmat[body_id].reshape(3, 3)
            origin = self.data.xpos[body_id]
            point = points_palm[finger]
            site = self.data.site_xpos[self.tip_ids[finger]]
            if (
                not np.all(np.isfinite(point))
                or np.linalg.norm(point - site)
                > float(self.config.contact_anchor_max_site_distance)
            ):
                # Keep the previous good anchor.  If none exists, persistent
                # recovery safely falls back to the fixed fingertip site.
                continue
            self.contact_point_body_local[finger] = rotation.T @ (
                point - origin
            )
            self.contact_point_valid[finger] = True

    def _finger_control_point_jacobian(self, finger: int) -> np.ndarray:
        """Return the 3x4 Jacobian at the measured pad contact centre.

        ``self.data`` must already contain the configuration at which the
        Jacobian is requested.  Before a finger has touched anything, the
        fixed MCC site is retained as a safe bootstrap fallback.
        """

        jacobian = np.zeros((3, self.model.nv), dtype=np.float64)
        jacobian_rot = np.zeros_like(jacobian)
        if self.contact_point_valid[finger]:
            body_id = int(self.tip_body_ids[finger])
            rotation = self.data.xmat[body_id].reshape(3, 3)
            point = self.data.xpos[body_id] + (
                rotation @ self.contact_point_body_local[finger]
            )
            mujoco.mj_jac(
                self.model,
                self.data,
                jacobian,
                jacobian_rot,
                point,
                body_id,
            )
        else:
            mujoco.mj_jacSite(
                self.model,
                self.data,
                jacobian,
                jacobian_rot,
                int(self.tip_ids[finger]),
            )
        block = slice(4 * finger, 4 * finger + 4)
        return jacobian[:, self.dof_indices[block]].copy()

    def clamp_joint_positions(self, q_action_order: np.ndarray) -> np.ndarray:
        """Clamp a hand command to the physical 16-DOF joint limits."""
        clipped = np.clip(
            np.asarray(q_action_order, dtype=np.float64).reshape(16),
            self.lower,
            self.upper,
        )
        if self.config.natural_flexion_floor is not None:
            # Each finger is [base flexion, side/opposition, middle, distal].
            # Only the two distal flexion coordinates are protected.  Side
            # motion and thumb opposition remain free for surface fitting.
            flexion = np.asarray((2, 3, 6, 7, 10, 11, 14, 15))
            clipped[flexion] = np.maximum(
                clipped[flexion],
                float(self.config.natural_flexion_floor),
            )
        return clipped.astype(np.float32)

    def solve_contact_velocity_qp(
        self,
        q_live: np.ndarray,
        target_velocity_palm: np.ndarray,
        surface_normals_palm: np.ndarray,
        nominal_posture_q: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, np.ndarray | float | int]]:
        """Predict a feasible hand posture from the moving contact manifold.

        The object motion is known to the privileged collection teacher.  We
        therefore solve a small 16-variable convex QP before the ordinary MCC
        correction.  Normal surface velocity is tracked strongly, tangential
        velocity softly, and the remaining one-DoF-per-finger redundancy is
        pulled toward a natural loaded grasp.  Joint position and velocity
        limits are hard constraints.

        This is deliberately a differential, contact-explicit teacher rather
        than a full contact-implicit DDP planner: all four contacts are meant
        to remain active, so contact-mode search would add cost without adding
        useful labels at this stage.
        """

        cfg = self.config
        q = np.asarray(q_live, dtype=np.float64).reshape(16)
        velocity = np.asarray(target_velocity_palm, dtype=np.float64).reshape(4, 3)
        speed = np.linalg.norm(velocity, axis=-1, keepdims=True)
        velocity = velocity * np.minimum(
            1.0,
            float(cfg.qp_max_target_speed) / np.maximum(speed, 1.0e-9),
        )
        alpha = float(np.clip(cfg.qp_target_velocity_ema_alpha, 0.0, 1.0))
        self.filtered_qp_target_velocity[:] = (
            alpha * velocity
            + (1.0 - alpha) * self.filtered_qp_target_velocity
        )
        velocity = self.filtered_qp_target_velocity.copy()
        normals = _normalize(
            np.asarray(surface_normals_palm, dtype=np.float64).reshape(4, 3)
        )
        nominal = np.asarray(nominal_posture_q, dtype=np.float64).reshape(16)
        self._set_q(self.data, q)
        nominal_tip_positions = None
        if cfg.use_lateral_reference_regularizer:
            self._set_q(self.data, nominal)
            nominal_tip_positions = self.data.site_xpos[self.tip_ids].copy()
            self._set_q(self.data, q)

        hessian = 1.0e-7 * np.eye(16, dtype=np.float64)
        gradient = np.zeros(16, dtype=np.float64)
        identity3 = np.eye(3, dtype=np.float64)
        finger_jacobians: list[np.ndarray] = []
        for finger in range(4):
            block = slice(4 * finger, 4 * finger + 4)
            jacobian = self._finger_control_point_jacobian(finger)
            finger_jacobians.append(jacobian)
            normal = normals[finger]
            weight = (
                cfg.qp_tangential_velocity_weight * identity3
                + (
                    cfg.qp_normal_velocity_weight
                    - cfg.qp_tangential_velocity_weight
                )
                * np.outer(normal, normal)
            )
            hessian[block, block] += 2.0 * jacobian.T @ weight @ jacobian
            gradient[block] += -2.0 * (
                jacobian.T @ weight @ velocity[finger]
            )

        # Adapt only the side-swing joints of index/middle/ring when two
        # neighbouring pads become crowded.  Measure spacing on the palm
        # lateral (+Y) axis, not as a 3-D Euclidean distance: two fingertips
        # at different flexion depths can otherwise appear safely separated
        # even though their pads overlap when viewed from the palm.
        tip_positions = self.data.site_xpos[self.tip_ids].copy()
        adjacent_distances = np.zeros(2, dtype=np.float64)
        separation_active = np.zeros(2, dtype=bool)
        lateral_axis = np.asarray(cfg.qp_lateral_axis_palm, dtype=np.float64)
        lateral_axis /= max(float(np.linalg.norm(lateral_axis)), 1.0e-12)
        # Capture the ordering from the first stable reference posture.  This
        # is deliberately not hard-coded as index<middle<ring: the same code
        # works if the palm coordinate convention is mirrored.  The captured
        # reference is held for the episode, preventing a transient crossing
        # from flipping the inequality and making the constraint meaningless.
        if nominal_tip_positions is not None and self.lateral_reference_positions is None:
            self.lateral_reference_positions = nominal_tip_positions[:3].copy()
            for pair_index, (left, right) in enumerate(((0, 1), (1, 2))):
                signed = float(
                    lateral_axis
                    @ (self.lateral_reference_positions[left]
                       - self.lateral_reference_positions[right])
                )
                self.lateral_order_signs[pair_index] = (
                    1.0 if signed >= 0.0 else -1.0
                )
        for pair_index, (left, right) in enumerate(((0, 1), (1, 2))):
            difference = tip_positions[left] - tip_positions[right]
            signed_distance = float(lateral_axis @ difference)
            ordering_sign = float(self.lateral_order_signs[pair_index])
            # Signed distance is the actual ordering margin.  Taking abs()
            # here would incorrectly declare a crossed index/ring pair safe.
            distance = ordering_sign * signed_distance
            adjacent_distances[pair_index] = abs(signed_distance)
            deficit = float(cfg.qp_min_adjacent_lateral_distance) - distance
            if deficit <= 0.0:
                continue
            # In manifold mode the nominal sign prevents a pair from crossing
            # and then declaring itself safe merely because its absolute
            # distance became large on the opposite side.
            direction = ordering_sign * lateral_axis
            row = np.zeros(16, dtype=np.float64)
            left_side = 4 * left + 1
            right_side = 4 * right + 1
            row[left_side] = float(
                direction @ finger_jacobians[left][:, 1]
            )
            row[right_side] = float(
                -direction @ finger_jacobians[right][:, 1]
            )
            desired_rate = float(cfg.qp_separation_gain) * deficit
            weight = float(cfg.qp_separation_weight)
            hessian += 2.0 * weight * np.outer(row, row)
            gradient += -2.0 * weight * desired_rate * row
            separation_active[pair_index] = True

        if nominal_tip_positions is not None:
            # Keep each ordinary finger near its own nominal lateral slot.
            # This prevents all three side joints from drifting together while
            # still allowing the whole trio to remain almost collinear.
            for finger in range(3):
                block = slice(4 * finger, 4 * finger + 4)
                current_lateral = float(lateral_axis @ tip_positions[finger])
                nominal_lateral = float(
                    lateral_axis @ nominal_tip_positions[finger]
                )
                desired_rate = float(cfg.qp_lateral_reference_gain) * (
                    nominal_lateral - current_lateral
                )
                row = np.zeros(16, dtype=np.float64)
                row[4 * finger + 1] = float(
                    lateral_axis @ finger_jacobians[finger][:, 1]
                )
                weight = float(cfg.qp_lateral_reference_weight)
                hessian += 2.0 * weight * np.outer(row, row)
                gradient += -2.0 * weight * desired_rate * row

        posture_velocity = cfg.qp_posture_gain * (nominal - q)
        hessian += 2.0 * cfg.qp_posture_weight * np.eye(16)
        gradient += -2.0 * cfg.qp_posture_weight * posture_velocity
        hessian += 2.0 * cfg.qp_smooth_weight * np.eye(16)
        gradient += -2.0 * (
            cfg.qp_smooth_weight * self.previous_qp_velocity
        )

        dt = float(cfg.control_dt)
        horizon_dt = dt * float(cfg.qp_lookahead_steps)
        max_velocity = float(cfg.qp_max_joint_velocity)
        lower_velocity = np.maximum(
            -max_velocity,
            (self.lower - q) / horizon_dt,
        )
        upper_velocity = np.minimum(
            max_velocity,
            (self.upper - q) / horizon_dt,
        )
        # Contact impulses can push a physical joint beyond its soft MuJoCo
        # range.  Requiring a one-horizon return then produces lower > upper
        # and makes the QP infeasible.  Outside the range, constrain only the
        # recovery direction; the posture objective returns it progressively.
        below_limit = q < self.lower
        above_limit = q > self.upper
        lower_velocity[below_limit] = 0.0
        upper_velocity[below_limit] = max_velocity
        lower_velocity[above_limit] = -max_velocity
        upper_velocity[above_limit] = 0.0
        # DAQP interprets the first n bounds as simple variable bounds when A
        # has zero rows.  This avoids constructing a redundant identity block.
        solution, _, exit_flag, info = daqp.solve(
            hessian,
            gradient,
            np.empty((0, 16), dtype=np.float64),
            upper_velocity,
            lower_velocity,
            primal_start=self.previous_qp_velocity,
        )
        if int(exit_flag) != 1 or not np.all(np.isfinite(solution)):
            solution = np.clip(
                posture_velocity,
                lower_velocity,
                upper_velocity,
            )
        self.previous_qp_velocity[:] = solution
        q_target = self.clamp_joint_positions(q + horizon_dt * solution)

        predicted_velocity = np.zeros((4, 3), dtype=np.float64)
        normal_error = np.zeros(4, dtype=np.float64)
        for finger in range(4):
            block = slice(4 * finger, 4 * finger + 4)
            jacobian = finger_jacobians[finger]
            predicted_velocity[finger] = jacobian @ solution[block]
            normal_error[finger] = float(
                normals[finger]
                @ (predicted_velocity[finger] - velocity[finger])
            )
        return q_target.astype(np.float32), {
            "joint_velocity": solution.astype(np.float32),
            "predicted_tip_velocity_palm": predicted_velocity.astype(np.float32),
            "target_tip_velocity_palm": velocity.astype(np.float32),
            "normal_velocity_error": normal_error.astype(np.float32),
            "adjacent_lateral_distance": adjacent_distances.astype(np.float32),
            "separation_active": separation_active,
            "exit_flag": int(exit_flag),
            "iterations": int(info.get("iterations", 0)),
            "solve_time_s": float(info.get("solve_time", 0.0)),
        }

    @staticmethod
    def points_palm_to_world(
        points_palm: np.ndarray, palm_pose_world: np.ndarray
    ) -> np.ndarray:
        pose = np.asarray(palm_pose_world, dtype=np.float64).reshape(7)
        return pose[:3] + (
            _quat_wxyz_to_matrix(pose[3:7]) @ np.asarray(points_palm).T
        ).T

    @staticmethod
    def points_world_to_palm(
        points_world: np.ndarray, palm_pose_world: np.ndarray
    ) -> np.ndarray:
        pose = np.asarray(palm_pose_world, dtype=np.float64).reshape(7)
        return (
            _quat_wxyz_to_matrix(pose[3:7]).T
            @ (np.asarray(points_world) - pose[:3]).T
        ).T

    @staticmethod
    def vectors_world_to_palm(
        vectors_world: np.ndarray, palm_pose_world: np.ndarray
    ) -> np.ndarray:
        rotation = _quat_wxyz_to_matrix(
            np.asarray(palm_pose_world, dtype=np.float64).reshape(7)[3:7]
        )
        return (rotation.T @ np.asarray(vectors_world).T).T

    @staticmethod
    def vectors_palm_to_world(
        vectors_palm: np.ndarray, palm_pose_world: np.ndarray
    ) -> np.ndarray:
        rotation = _quat_wxyz_to_matrix(
            np.asarray(palm_pose_world, dtype=np.float64).reshape(7)[3:7]
        )
        return (rotation @ np.asarray(vectors_palm).T).T

    def grasp_closure_directions_palm(
        self,
        q_action_order: np.ndarray,
    ) -> np.ndarray:
        """Return per-finger Cartesian directions toward the grasp posture.

        The direction is computed through each finger's own 3x4 Jacobian.
        This preserves the thumb's four-DoF kinematics and avoids treating a
        joint-space vector as if it were a Cartesian surface normal.
        """

        q = np.asarray(q_action_order, dtype=np.float64).reshape(16)
        self._set_q(self.data, q)
        directions = np.zeros((4, 3), dtype=np.float64)
        fallback_synergy = np.asarray((0.20, 0.0, 0.12, 0.12))
        for finger in range(4):
            block = slice(4 * finger, 4 * finger + 4)
            jacobian = np.zeros((3, self.model.nv), dtype=np.float64)
            jacobian_rot = np.zeros_like(jacobian)
            mujoco.mj_jacSite(
                self.model,
                self.data,
                jacobian,
                jacobian_rot,
                int(self.tip_ids[finger]),
            )
            finger_jacobian = jacobian[:, self.dof_indices[block]]
            joint_direction = self.grasp_closure_q[block] - q[block]
            direction = finger_jacobian @ joint_direction
            if np.linalg.norm(direction) < 1.0e-7:
                direction = finger_jacobian @ fallback_synergy
            norm = np.linalg.norm(direction)
            if norm < 1.0e-9:
                raise ValueError(
                    f"Finger {finger} has no usable grasp-closure direction"
                )
            directions[finger] = direction / norm
        return directions.astype(np.float32)

    def directional_search_delta(
        self,
        q_action_order: np.ndarray,
        palm_pose_world: np.ndarray,
        inward_directions_world: np.ndarray,
        missing: np.ndarray,
        inward_step: float,
        max_joint_step: float,
        contact_points_world: np.ndarray | None = None,
        contact_point_found: np.ndarray | None = None,
    ) -> np.ndarray:
        """Move missing tactile contact centres along the selected axes."""

        if contact_points_world is not None and contact_point_found is not None:
            self.update_contact_point_anchors(
                q_action_order,
                palm_pose_world,
                contact_points_world,
                contact_point_found,
            )
        self._set_q(self.data, q_action_order)
        directions = _normalize(
            self.vectors_world_to_palm(
                inward_directions_world, palm_pose_world
            )
        )
        delta = np.zeros(16, dtype=np.float64)
        for finger, is_missing in enumerate(np.asarray(missing, dtype=bool)):
            if not is_missing:
                continue
            block = slice(4 * finger, 4 * finger + 4)
            finger_jacobian = self._finger_control_point_jacobian(finger)
            target = float(inward_step) * directions[finger]
            lhs = finger_jacobian @ finger_jacobian.T + 1.0e-5 * np.eye(3)
            correction = finger_jacobian.T @ np.linalg.solve(lhs, target)
            delta[block] = np.clip(
                correction, -max_joint_step, max_joint_step
            )
        return delta.astype(np.float32)

    def grasp_synergy_search_delta(
        self,
        q_action_order: np.ndarray,
        missing: np.ndarray,
        cartesian_step: float,
        max_joint_step: float,
    ) -> np.ndarray:
        """Advance missing fingers directly toward the grasp posture.

        Unlike ``J^# (J dq_grasp)``, this preserves the original four-joint
        grasp synergy.  The Jacobian is used only to scale that joint-space
        direction to an approximate Cartesian step length.
        """

        q = np.asarray(q_action_order, dtype=np.float64).reshape(16)
        self._set_q(self.data, q)
        delta = np.zeros(16, dtype=np.float64)
        for finger, is_missing in enumerate(np.asarray(missing, dtype=bool)):
            if not is_missing:
                continue
            block = slice(4 * finger, 4 * finger + 4)
            joint_direction = self.grasp_closure_q[block] - q[block]
            if np.linalg.norm(joint_direction) < 1.0e-9:
                continue
            jacobian = np.zeros((3, self.model.nv), dtype=np.float64)
            jacobian_rot = np.zeros_like(jacobian)
            mujoco.mj_jacSite(
                self.model,
                self.data,
                jacobian,
                jacobian_rot,
                int(self.tip_ids[finger]),
            )
            finger_jacobian = jacobian[:, self.dof_indices[block]]
            cartesian_direction = finger_jacobian @ joint_direction
            cartesian_norm = np.linalg.norm(cartesian_direction)
            if cartesian_norm < 1.0e-9:
                continue
            correction = (
                float(cartesian_step) / cartesian_norm
            ) * joint_direction
            delta[block] = np.clip(
                correction, -max_joint_step, max_joint_step
            )
        return delta.astype(np.float32)

    def calibrate_force_sign(
        self,
        force_world: np.ndarray,
        found: np.ndarray,
        surface_normals_world: np.ndarray,
    ) -> None:
        """Match the sign calibration performed in full-hand MCC warm-up."""

        force = np.asarray(force_world, dtype=np.float64).reshape(4, 3)
        normal = _normalize(surface_normals_world)
        signed = np.einsum("fi,fi->f", force, normal)
        reliable = np.asarray(found, dtype=bool).reshape(4) & (np.abs(signed) >= 0.05)
        self.force_sign[reliable] = np.where(signed[reliable] >= 0.0, 1.0, -1.0)

    def calibrate_force_setpoint(
        self,
        force_world: np.ndarray,
        found: np.ndarray,
        surface_normals_world: np.ndarray,
        maximum_force: float = 12.0,
        capture_measured: bool = True,
    ) -> np.ndarray:
        """Calibrate force sign and optionally capture a loaded force point.

        Sensor-only replay uses ``capture_measured=False``: the first valid
        contact can transiently contain a large collision impulse, so that
        impulse must not become a permanent force setpoint.  The configured
        ``desired_force`` remains the target in that mode.
        """

        self.calibrate_force_sign(force_world, found, surface_normals_world)
        signed = np.einsum(
            "fi,fi->f",
            np.asarray(force_world, dtype=np.float64).reshape(4, 3),
            _normalize(surface_normals_world),
        ) * self.force_sign
        reliable = np.asarray(found, dtype=bool).reshape(4)
        if capture_measured:
            loaded = np.abs(signed)
            self.force_setpoint[reliable] = np.minimum(
                np.maximum(
                    loaded[reliable],
                    self.nominal_force_setpoint[reliable],
                ),
                maximum_force,
            )
        else:
            self.force_setpoint[reliable] = self.nominal_force_setpoint[reliable]
        return self.force_setpoint.copy()

    def normal_search_delta(
        self,
        q_action_order: np.ndarray,
        palm_pose_world: np.ndarray,
        surface_normals_world: np.ndarray,
        missing: np.ndarray,
        inward_step: float,
        max_joint_step: float,
    ) -> np.ndarray:
        """Per-pad precontact search used by the fullhandMCC demo.

        It moves only a missing finger along its own physical inward surface
        normal.  The palm stays fixed in this inverse evaluation.
        """

        self._set_q(self.data, q_action_order)
        normals = _normalize(
            self.vectors_world_to_palm(surface_normals_world, palm_pose_world)
        )
        delta = np.zeros(16, dtype=np.float64)
        for finger, is_missing in enumerate(np.asarray(missing, dtype=bool)):
            if not is_missing:
                continue
            jacobian = np.zeros((3, self.model.nv), dtype=np.float64)
            jacobian_rot = np.zeros_like(jacobian)
            mujoco.mj_jacSite(
                self.model, self.data, jacobian, jacobian_rot,
                int(self.tip_ids[finger]),
            )
            block = slice(4 * finger, 4 * finger + 4)
            finger_jacobian = jacobian[:, self.dof_indices[block]]
            target = -float(inward_step) * normals[finger]
            lhs = finger_jacobian @ finger_jacobian.T + 1.0e-5 * np.eye(3)
            correction = finger_jacobian.T @ np.linalg.solve(lhs, target)
            delta[block] = np.clip(
                correction, -max_joint_step, max_joint_step
            )
        return delta.astype(np.float32)

    def recover_surface_contacts(
        self,
        q_live: np.ndarray,
        base_command_q: np.ndarray,
        palm_pose_world: np.ndarray,
        surface_points_world: np.ndarray,
        surface_normals_world: np.ndarray,
        persistent_loss: np.ndarray,
        surface_preload_m: float,
        nominal_posture_q: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Drive persistently lost pad points to absolute surface targets.

        Unlike the small force-servo offset, this recovery has no accumulated
        displacement state and no metric distance cap.  Every frame recomputes
        the error between the actual pad-attached control point and a point
        just below the current surface, then takes a bounded differential IK
        step with that finger's own 3x4 Jacobian.  Repeated steps can traverse
        the complete physical workspace while joint limits and per-step speed
        remain hard safeguards.
        """

        cfg = self.config
        q = np.asarray(q_live, dtype=np.float64).reshape(16)
        command = np.asarray(base_command_q, dtype=np.float64).reshape(16).copy()
        persistent = np.asarray(persistent_loss, dtype=bool).reshape(4)
        recovery_nominal = (
            self.grasp_closure_q
            if nominal_posture_q is None
            else np.asarray(nominal_posture_q, dtype=np.float64).reshape(16)
        )
        # Recovery can be entered from a stale pregrasp/reference branch.  Do
        # not allow that branch to undo an object-specific thumb closure
        # adjustment: q12 stays at the open-root target and q15 carries the
        # extra distal curl configured in ``grasp_closure_q``.
        recovery_nominal = np.asarray(recovery_nominal, dtype=np.float64).copy()
        recovery_nominal[12] = self.grasp_closure_q[12]
        recovery_nominal[15] = self.grasp_closure_q[15]
        normals_world = _normalize(
            np.asarray(surface_normals_world, dtype=np.float64).reshape(4, 3)
        )
        target_world = (
            np.asarray(surface_points_world, dtype=np.float64).reshape(4, 3)
            - float(surface_preload_m) * normals_world
        )
        target_palm = self.points_world_to_palm(target_world, palm_pose_world)

        self._set_q(self.data, q)
        current_palm = np.zeros((4, 3), dtype=np.float64)
        delta_q = np.zeros(16, dtype=np.float64)
        error_palm = np.zeros((4, 3), dtype=np.float64)
        for finger in range(4):
            if self.contact_point_valid[finger]:
                body_id = int(self.tip_body_ids[finger])
                rotation = self.data.xmat[body_id].reshape(3, 3)
                current_palm[finger] = self.data.xpos[body_id] + (
                    rotation @ self.contact_point_body_local[finger]
                )
            else:
                current_palm[finger] = self.data.site_xpos[self.tip_ids[finger]]
            error_palm[finger] = target_palm[finger] - current_palm[finger]
            if not persistent[finger]:
                continue

            block = slice(4 * finger, 4 * finger + 4)
            jacobian = self._finger_control_point_jacobian(finger)
            lhs = (
                jacobian @ jacobian.T
                + float(cfg.persistent_recovery_regularization)
                * np.eye(3, dtype=np.float64)
            )
            jacobian_pinv = jacobian.T @ np.linalg.solve(lhs, np.eye(3))
            correction = jacobian_pinv @ error_palm[finger]

            # Use the one-dimensional null space to unfold a degenerate IK
            # branch without weakening the absolute Cartesian recovery task.
            nullspace = np.eye(4) - jacobian_pinv @ jacobian
            # Resolve Cartesian redundancy toward the *current closure-path
            # solution*.  Pulling every lost finger toward the fixed full
            # grasp prevents it from opening when the rolling object presses
            # across its path.
            posture_error = recovery_nominal[block] - q[block]
            correction += (
                float(cfg.persistent_recovery_posture_gain)
                * nullspace
                @ posture_error
            )
            correction = np.clip(
                correction,
                -float(cfg.persistent_recovery_max_joint_step),
                float(cfg.persistent_recovery_max_joint_step),
            )
            delta_q[block] = correction
            # Persistent recovery owns this finger.  Basing it on live q
            # prevents the ordinary QP/posture target from cancelling the
            # inward step in the same frame.
            command[block] = q[block] + correction

        command = self.clamp_joint_positions(command).astype(np.float64)
        return command.astype(np.float32), {
            "recovery_target_palm": target_palm.astype(np.float32),
            "recovery_control_point_palm": current_palm.astype(np.float32),
            "recovery_error_palm": error_palm.astype(np.float32),
            "recovery_error_norm": np.linalg.norm(
                error_palm, axis=-1
            ).astype(np.float32),
            "recovery_joint_step": delta_q.astype(np.float32),
            "persistent_loss": persistent.copy(),
        }

    def update(
        self,
        q_live: np.ndarray,
        palm_pose_world: np.ndarray,
        force_world: np.ndarray,
        found: np.ndarray,
        surface_points_world: np.ndarray,
        surface_normals_world: np.ndarray,
        nominal_posture_q: np.ndarray | None = None,
        force_magnitude_only: bool = False,
        contact_points_world: np.ndarray | None = None,
        use_contact_point_jacobian: bool = False,
        manage_contact_state: bool = True,
        contact_observed: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        cfg = self.config
        q_live = np.asarray(q_live, dtype=np.float64).reshape(16)
        found = np.asarray(found, dtype=bool).reshape(4)
        if cfg.enable_loss_state_machine and manage_contact_state:
            state_contact = (
                found
                if contact_observed is None
                else np.asarray(contact_observed, dtype=bool).reshape(4)
            )
            self._update_contact_recovery_state(state_contact)
        actual = self.tip_positions_palm(q_live)
        desired = self.points_world_to_palm(surface_points_world, palm_pose_world)
        normals = _normalize(
            self.vectors_world_to_palm(surface_normals_world, palm_pose_world)
        )
        raw_force_magnitude = np.linalg.norm(
            np.asarray(force_world, dtype=np.float64).reshape(4, 3),
            axis=-1,
        )
        if force_magnitude_only:
            # The admittance core only needs a scalar load along its control
            # axis.  Reconstruct that scalar from the 3-D force magnitude so
            # tangential friction and an uncertain surface normal cannot
            # reverse the contact correction.
            force_magnitude = np.linalg.norm(
                np.asarray(force_world, dtype=np.float64).reshape(4, 3),
                axis=-1,
            )
            force_magnitude[~found] = 0.0
            force_local = force_magnitude[:, None] * normals
        else:
            force_local = self.vectors_world_to_palm(force_world, palm_pose_world)
            force_local *= self.force_sign[:, None]
            force_local[~found] = 0.0
        if cfg.use_direct_force_servo:
            measurement = raw_force_magnitude.copy()
            measurement[~found] = 0.0
            new_filter = ~self.force_servo_filter_valid
            self.force_servo_filtered[new_filter] = measurement[new_filter]
            self.force_servo_filter_valid[new_filter] = True
            old_filter = ~new_filter
            alpha = float(np.clip(cfg.force_filter_alpha, 0.0, 1.0))
            self.force_servo_filtered[old_filter] = (
                alpha * measurement[old_filter]
                + (1.0 - alpha) * self.force_servo_filtered[old_filter]
            )
            turn_on = self.force_servo_filtered >= cfg.contact_on_force
            stay_on = self.force_servo_filtered >= cfg.contact_off_force
            self.force_servo_contact_active[:] = np.where(
                self.force_servo_contact_active, stay_on, turn_on
            )
            force_error = self.force_setpoint - self.force_servo_filtered
            effective_error = np.where(
                np.abs(force_error) <= cfg.force_servo_deadband,
                0.0,
                force_error,
            )
            offset_step = (
                cfg.force_servo_integral_gain
                * effective_error
                * cfg.control_dt
            )
            offset_step = np.clip(
                offset_step,
                -cfg.force_servo_max_step,
                cfg.force_servo_max_step,
            )
            missing = ~found
            weak_contact = (
                found
                & ~self.force_servo_contact_active
                & (self.overforce_recontact_hold == 0)
            )
            if cfg.enable_loss_state_machine:
                # Missing-contact recovery must not consume and saturate the
                # small force-regulation budget.  Phase 1 is represented by a
                # separate time-bounded transient offset; phase 2 is handled
                # by absolute surface IK outside this force loop.
                held_missing = np.zeros(4, dtype=bool)
                search_missing = np.zeros(4, dtype=bool)
                offset_step[missing] = 0.0
                offset_step[weak_contact] = np.maximum(
                    offset_step[weak_contact],
                    cfg.force_servo_weak_contact_step,
                )
                self.force_servo_filtered[missing] = 0.0
                self.force_servo_filter_valid[missing] = False
            else:
                # Legacy mode: use the force offset itself as the contact
                # search accumulator.  Kept for existing deployment presets;
                # collection enables the explicit state machine instead.
                held_missing = missing & (
                    self.overforce_recontact_hold > 0
                )
                search_missing = missing & ~held_missing
                offset_step[held_missing] = 0.0
                search_step = np.full(
                    4, cfg.force_servo_search_step, dtype=np.float64
                )
                if cfg.thumb_force_servo_search_step is not None:
                    search_step[3] = cfg.thumb_force_servo_search_step
                offset_step[search_missing] = np.maximum(
                    offset_step[search_missing], search_step[search_missing]
                )
                offset_step[weak_contact] = np.maximum(
                    offset_step[weak_contact],
                    cfg.force_servo_weak_contact_step,
                )
                self.force_servo_filtered[search_missing] = 0.0
                self.force_servo_filter_valid[search_missing] = False
            hard_ratio = np.full(4, cfg.overforce_hard_ratio, dtype=np.float64)
            if cfg.thumb_overforce_hard_ratio is not None:
                hard_ratio[3] = cfg.thumb_overforce_hard_ratio
            hard_overforce = found & (
                raw_force_magnitude >= hard_ratio * self.force_setpoint
            )
            self.overforce_recontact_hold[hard_overforce] = (
                cfg.overforce_recontact_hold_frames
            )
            decay_hold = found & ~hard_overforce
            self.overforce_recontact_hold[decay_hold] = np.maximum(
                self.overforce_recontact_hold[decay_hold] - 1,
                0,
            )
            self.overforce_recontact_hold[held_missing] = np.maximum(
                self.overforce_recontact_hold[held_missing] - 1,
                0,
            )
            hard_step = np.full(
                4, cfg.force_servo_hard_step, dtype=np.float64
            )
            if cfg.thumb_force_servo_hard_step is not None:
                hard_step[3] = cfg.thumb_force_servo_hard_step
            offset_step[hard_overforce] = np.minimum(
                offset_step[hard_overforce], -hard_step[hard_overforce]
            )
            lower_offset = np.full(4, -cfg.max_normal_offset)
            if cfg.thumb_max_outward_offset is not None:
                lower_offset[3] = -cfg.thumb_max_outward_offset
            upper_offset = np.full(4, cfg.max_normal_offset)
            if cfg.thumb_max_inward_offset is not None:
                upper_offset[3] = cfg.thumb_max_inward_offset
            self.force_servo_offset[:] = np.clip(
                self.force_servo_offset + offset_step,
                lower_offset,
                upper_offset,
            )
            self.force_servo_velocity[:] = offset_step / cfg.control_dt

            total_normal_offset = (
                self.force_servo_offset + self.transient_search_offset
            )
            total_normal_offset = (
                total_normal_offset + float(cfg.nominal_surface_preload)
            )
            command_points = desired - total_normal_offset[:, None] * normals
            measured_normal_force = raw_force_magnitude
            filtered_normal_force = self.force_servo_filtered.copy()
            contact_active = self.force_servo_contact_active.copy()
            normal_offset_debug = total_normal_offset.copy()
            normal_velocity_debug = self.force_servo_velocity.copy()
            normal_acceleration_debug = np.zeros(4, dtype=np.float64)
        else:
            finger_step = self.admittance.step(
                desired,
                normals,
                force_local,
                desired_force=self.force_setpoint,
            )
            # The shared full-hand core is batched even for this single replay
            # environment.  Unbatch here before feeding individual Mink tasks.
            command_points = finger_step.command_points[0]
            measured_normal_force = finger_step.measured_normal_force[0]
            filtered_normal_force = finger_step.filtered_normal_force[0]
            force_error = finger_step.force_error[0]
            contact_active = finger_step.contact_active[0]
            normal_offset_debug = finger_step.normal_offset[0]
            normal_velocity_debug = finger_step.normal_velocity[0]
            normal_acceleration_debug = finger_step.normal_acceleration[0]

        # This mirrors the full-hand adapter: use the external planner's
        # nominal q only to resolve fingertip IK redundancy; never integrate a
        # force correction into that nominal posture.
        nominal_q = (
            q_live
            if nominal_posture_q is None
            else np.asarray(nominal_posture_q, dtype=np.float64).reshape(16)
        )
        if use_contact_point_jacobian:
            if contact_points_world is None:
                raise ValueError(
                    "contact_points_world is required for contact-point control"
                )
            self.update_contact_point_anchors(
                q_live,
                palm_pose_world,
                contact_points_world,
                found,
            )
            # Evaluate every contact Jacobian at the physical state.  The DP
            # nominal is a posture/tangential preference; once contact is
            # established it must not retain ownership of the contact-normal
            # degree of freedom, otherwise its inward motion fights the force
            # loop and can exceed the admittance offset limit.
            normal_displacement = command_points - desired
            self._set_q(self.data, q_live)
            if cfg.project_nominal_normal_motion:
                if self.normal_task_reference_q is None:
                    # Start from the loaded physical posture.  Subsequent DP
                    # increments update only tangent/null-space coordinates;
                    # the force servo owns the contact-normal coordinate.
                    self.normal_task_reference_q = q_live.copy()
                    self.previous_nominal_q = nominal_q.copy()
                assert self.previous_nominal_q is not None
                nominal_increment = nominal_q - self.previous_nominal_q
                q_command = self.normal_task_reference_q.copy()
            else:
                nominal_increment = np.zeros(16, dtype=np.float64)
                q_command = nominal_q.copy()
            normal_projection = np.zeros(4, dtype=np.float64)
            if cfg.use_direct_force_servo:
                self.overforce_active[:] = found & (
                    raw_force_magnitude >= hard_ratio * self.force_setpoint
                )
                self.overforce_outward_offset[:] = 0.0
            else:
                trigger = cfg.overforce_trigger_ratio * self.force_setpoint
                release = cfg.overforce_release_ratio * self.force_setpoint
                self.overforce_active |= found & (
                    raw_force_magnitude >= trigger
                )
                self.overforce_active &= found & (
                    raw_force_magnitude > release
                )
                overforce_ratio = raw_force_magnitude / np.maximum(
                    self.force_setpoint, 1.0e-6
                )
                severity = np.clip(
                    (overforce_ratio - cfg.overforce_trigger_ratio)
                    / max(
                        cfg.overforce_hard_ratio
                        - cfg.overforce_trigger_ratio,
                        1.0e-6,
                    ),
                    0.0,
                    1.0,
                )
                self.overforce_outward_offset[self.overforce_active] += (
                    cfg.overforce_retreat_step
                    * (1.0 + 3.0 * severity[self.overforce_active])
                )
                recovering = ~self.overforce_active
                self.overforce_outward_offset[recovering] = np.maximum(
                    0.0,
                    self.overforce_outward_offset[recovering]
                    - cfg.overforce_recovery_step,
                )
                np.clip(
                    self.overforce_outward_offset,
                    0.0,
                    cfg.overforce_max_offset,
                    out=self.overforce_outward_offset,
                )
                normal_displacement = (
                    normal_displacement
                    + self.overforce_outward_offset[:, None] * normals
                )
            for finger in range(4):
                block = slice(4 * finger, 4 * finger + 4)
                jacobian = self._finger_control_point_jacobian(finger)
                # The force controller owns one scalar Cartesian DoF.  Solve
                # with J_n = n^T J directly instead of a 3-D pseudoinverse,
                # which would additionally (and unnecessarily) constrain two
                # tangent directions.  The scalar minimum-norm solution is
                # especially important for the thumb's four-joint geometry.
                normal_jacobian = normals[finger] @ jacobian
                normal_denominator = float(
                    np.dot(normal_jacobian, normal_jacobian)
                    + cfg.contact_point_jacobian_regularization
                )

                if cfg.project_nominal_normal_motion:
                    increment = nominal_increment[block]
                    nominal_cartesian = jacobian @ increment
                    normal_projection[finger] = float(
                        np.dot(nominal_cartesian, normals[finger])
                    )
                    if found[finger]:
                        # Remove the complete instantaneous normal component;
                        # the bidirectional force servo below follows both an
                        # approaching and a receding surface.  Joint null-space
                        # motion is retained by this minimum-norm projection.
                        increment = (
                            increment
                            - normal_projection[finger]
                            * normal_jacobian
                            / normal_denominator
                        )
                    # During geometry loss allow the full DP increment while
                    # the force offset and runtime search move inward.
                    q_command[block] += increment

                normal_distance = float(
                    np.dot(normal_displacement[finger], normals[finger])
                )
                if cfg.normal_synergy_control and found[finger]:
                    # One scalar closure coordinate owns normal pressure.
                    # All three flexion joints advance by the same normalized
                    # open->grasp fraction; the side joint is reserved for pad
                    # facing.  This prevents the minimum-norm Jacobian from
                    # closing one phalanx while opening the other two.
                    synergy = np.zeros(4, dtype=np.float64)
                    flex_local = np.asarray((0, 2, 3), dtype=np.int32)
                    synergy[flex_local] = (
                        self.grasp_closure_q[block][flex_local]
                        - self.open_grasp_q[block][flex_local]
                    )
                    normal_per_fraction = float(normal_jacobian @ synergy)
                    if abs(normal_per_fraction) > 1.0e-5:
                        correction = (
                            normal_distance / normal_per_fraction
                        ) * synergy
                        correction = np.clip(
                            correction,
                            -cfg.normal_synergy_max_step,
                            cfg.normal_synergy_max_step,
                        )
                    else:
                        correction = (
                            normal_distance
                            * normal_jacobian
                            / normal_denominator
                        )
                else:
                    correction = (
                        normal_distance
                        * normal_jacobian
                        / normal_denominator
                    )
                q_command[block] += correction
            if cfg.project_nominal_normal_motion:
                # Store the reference before adding this frame's force offset;
                # otherwise compliance would be integrated twice.
                reference = q_command.copy()
                for finger in range(4):
                    block = slice(4 * finger, 4 * finger + 4)
                    jacobian = self._finger_control_point_jacobian(finger)
                    normal_jacobian = normals[finger] @ jacobian
                    denominator = float(
                        np.dot(normal_jacobian, normal_jacobian)
                        + cfg.contact_point_jacobian_regularization
                    )
                    normal_distance = float(
                        np.dot(
                            normal_displacement[finger], normals[finger]
                        )
                    )
                    reference[block] -= (
                        normal_distance
                        * normal_jacobian
                        / denominator
                    )
                self.normal_task_reference_q = self.clamp_joint_positions(
                    reference
                ).astype(np.float64)
                self.previous_nominal_q = nominal_q.copy()
            q_command = self.clamp_joint_positions(q_command).astype(np.float64)
            self._set_q(self.configuration.data, q_command)
        else:
            self._set_q(self.configuration.data, nominal_q)
            self.posture_task.set_target_from_configuration(self.configuration)
            for finger, (task, target, site_id) in enumerate(zip(
                self.tasks, command_points, self.tip_ids, strict=True
            )):
                current_rotation = self.configuration.data.site_xmat[site_id].reshape(3, 3).copy()
                rotation = mink.SO3.from_matrix(
                    self._pad_normal_target(current_rotation, normals[finger])
                )
                task.set_target(
                    mink.SE3.from_rotation_and_translation(rotation, target)
                )
            for _ in range(cfg.mink_iterations):
                velocity = mink.solve_ik(
                    self.configuration,
                    [self.posture_task, *self.tasks],
                    cfg.control_dt,
                    solver="daqp",
                    damping=cfg.mink_damping,
                    limits=self.limits,
                )
                self.configuration.integrate_inplace(velocity, cfg.control_dt)
            q_command = self.configuration.data.qpos[self.qpos_indices].copy()
        q_command, flexion_synergy_spread = self.regularize_flexion_synergy(
            q_command, nominal_q
        )
        if self.previous_command is None:
            self.previous_command = q_live.copy()
        q_command = self.previous_command + np.clip(
            q_command - self.previous_command,
            -cfg.action_rate_limit,
            cfg.action_rate_limit,
        )
        # A light command-space EMA suppresses the residual discontinuity
        # when a fingertip switches between force regulation and re-contact.
        # It is intentionally applied after the hard rate limit, so neither
        # the filter nor an IK transient can violate the per-frame bound.
        alpha = float(np.clip(cfg.command_ema_alpha, 0.0, 1.0))
        filtered_command = self.previous_command + alpha * (
            q_command - self.previous_command
        )
        q_command = filtered_command
        self.previous_command = q_command.copy()
        self._set_q(self.configuration.data, q_command)
        tip_ik = self.configuration.data.site_xpos[self.tip_ids].copy()
        surface_error = np.linalg.norm(desired - actual, axis=-1)
        return q_command.astype(np.float32), {
            "tip_actual_palm": actual.astype(np.float32),
            "tip_surface_palm": desired.astype(np.float32),
            "tip_reference_palm": command_points.astype(np.float32),
            "tip_ik_palm": tip_ik.astype(np.float32),
            "surface_normal_palm": normals.astype(np.float32),
            "normal_force": measured_normal_force.astype(np.float32),
            "force_error": force_error.astype(np.float32),
            "contact_active": contact_active.copy(),
            "reference_speed": np.abs(
                normal_velocity_debug
            ).astype(np.float32),
            "surface_error": surface_error.astype(np.float32),
            "normal_offset": normal_offset_debug.astype(np.float32),
            "force_regulation_offset": self.force_servo_offset.astype(
                np.float32
            ),
            "transient_search_offset": self.transient_search_offset.astype(
                np.float32
            ),
            "contact_phase": self.contact_phase.copy(),
            "loss_streak": self.loss_streak.copy(),
            "recontact_streak": self.recontact_streak.copy(),
            "persistent_loss": (self.contact_phase == 2),
            "normal_velocity": normal_velocity_debug.astype(np.float32),
            "normal_acceleration": normal_acceleration_debug.astype(np.float32),
            "raw_force_magnitude": np.linalg.norm(
                np.asarray(force_world, dtype=np.float64).reshape(4, 3),
                axis=-1,
            ).astype(np.float32),
            "flexion_synergy_spread": flexion_synergy_spread.astype(np.float32),
            "overforce_active": self.overforce_active.copy(),
            "overforce_outward_offset": (
                self.overforce_outward_offset.copy().astype(np.float32)
            ),
            "normal_projection": (
                normal_projection.astype(np.float32)
                if use_contact_point_jacobian
                else np.zeros(4, dtype=np.float32)
            ),
            "nominal_posture_error": (q_live - nominal_q).astype(np.float32),
        }
