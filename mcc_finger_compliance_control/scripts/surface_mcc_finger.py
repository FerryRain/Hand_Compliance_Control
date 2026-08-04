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

import mink
import mujoco
import numpy as np

# This is deliberately the same numerical fingertip admittance used by the
# full-hand MCC task.  Keeping the reference dynamics shared is important for
# an A/B experiment: only the replay adapter and the source of the surface
# plan differ, not the force-loop equations.
from mjlab.tasks.leaphand.full_hand_mcc_core import (
    FingertipAdmittanceGains,
    FingertipNormalAdmittance,
)


HAND_XML = Path(
    "src/mjlab/asset_zoo/robots/xarm6_leap_hand/leap_hand_tactile.xml"
)
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

# Conservative hand-closing posture used by the standalone MCC collection
# task.  In normal-free deployment this is not commanded directly: its FK
# direction supplies one local inward search axis per finger.
DEFAULT_GRASP_CLOSURE_Q = np.asarray(
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
    ) -> None:
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

    # -- public interface ------------------------------------------------------

    def observe(
        self, query_points_world: np.ndarray,
    ) -> CapsuleSurfaceObservation:
        query = np.asarray(query_points_world, dtype=np.float64).reshape(-1, 3)
        rotation_wo = self.rotation_world_from_object
        center = self.center_world

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
    virtual_stiffness: float = 1000.0
    force_gain: float = 1.0
    desired_force: float = 3.0
    force_filter_alpha: float = 0.25
    contact_on_force: float = 0.15
    contact_off_force: float = 0.08
    max_normal_offset: float = 0.003
    max_normal_speed: float = 0.010
    max_normal_acceleration: float = 0.2
    mink_damping: float = 0.1
    mink_iterations: int = 3
    posture_cost: float = 0.08
    action_rate_limit: float = 0.18
    command_ema_alpha: float = 0.65
    contact_point_jacobian_regularization: float = 1.0e-5
    grasp_closure_q: tuple[float, ...] = tuple(DEFAULT_GRASP_CLOSURE_Q)


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
                orientation_cost=0.0,
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
        self.force_setpoint = np.full(4, cfg.desired_force, dtype=np.float64)
        self.grasp_closure_q = self.clamp_joint_positions(
            np.asarray(cfg.grasp_closure_q, dtype=np.float64)
        ).astype(np.float64)
        # The tactile contact centre is generally not the fixed MCC site.
        # Store its last measured coordinates in the corresponding distal
        # body frame so the point remains attached to the pad during a short
        # contact dropout and its Jacobian can be evaluated at any q.
        self.contact_point_body_local = np.zeros((4, 3), dtype=np.float64)
        self.contact_point_valid = np.zeros(4, dtype=bool)
        self.previous_command: np.ndarray | None = None

    def reset(self) -> None:
        self.admittance.reset()
        self.force_sign[:] = 1.0
        self.force_setpoint[:] = self.config.desired_force
        self.contact_point_body_local[:] = 0.0
        self.contact_point_valid[:] = False
        self.previous_command = None

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

    def _set_q(self, data: mujoco.MjData, q_action_order: np.ndarray) -> None:
        data.qpos[:] = 0.0
        data.qpos[self.qpos_indices] = np.asarray(
            q_action_order, dtype=np.float64
        ).reshape(16)
        data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, data)

    def tip_positions_palm(self, q_action_order: np.ndarray) -> np.ndarray:
        self._set_q(self.data, q_action_order)
        return self.data.site_xpos[self.tip_ids].copy()

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
            self.contact_point_body_local[finger] = rotation.T @ (
                points_palm[finger] - origin
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

        return np.clip(
            np.asarray(q_action_order, dtype=np.float64).reshape(16),
            self.lower,
            self.upper,
        ).astype(np.float32)

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
            self.force_setpoint[reliable] = np.clip(
                loaded[reliable], self.config.desired_force, maximum_force
            )
        else:
            self.force_setpoint[reliable] = self.config.desired_force
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
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        cfg = self.config
        q_live = np.asarray(q_live, dtype=np.float64).reshape(16)
        found = np.asarray(found, dtype=bool).reshape(4)
        actual = self.tip_positions_palm(q_live)
        desired = self.points_world_to_palm(surface_points_world, palm_pose_world)
        normals = _normalize(
            self.vectors_world_to_palm(surface_normals_world, palm_pose_world)
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
        finger_step = self.admittance.step(
            desired,
            normals,
            force_local,
            desired_force=self.force_setpoint,
        )
        # The shared full-hand core is batched even for this single replay
        # environment.  Unbatch here before feeding individual Mink tasks.
        command_points = finger_step.command_points[0]

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
            # FullHandMCC's scalar admittance state remains unchanged.  Only
            # its Cartesian-to-joint mapping is replaced: the normal offset
            # is now applied at the measured tactile contact centre rather
            # than at the fixed kinematic site.
            normal_displacement = command_points - desired
            self._set_q(self.data, nominal_q)
            q_command = nominal_q.copy()
            for finger in range(4):
                block = slice(4 * finger, 4 * finger + 4)
                jacobian = self._finger_control_point_jacobian(finger)
                lhs = jacobian @ jacobian.T + (
                    cfg.contact_point_jacobian_regularization * np.eye(3)
                )
                correction = jacobian.T @ np.linalg.solve(
                    lhs, normal_displacement[finger]
                )
                q_command[block] += correction
            q_command = self.clamp_joint_positions(q_command).astype(np.float64)
            self._set_q(self.configuration.data, q_command)
        else:
            self._set_q(self.configuration.data, nominal_q)
            self.posture_task.set_target_from_configuration(self.configuration)
            for task, target, site_id in zip(
                self.tasks, command_points, self.tip_ids, strict=True
            ):
                rotation = mink.SO3.from_matrix(
                    self.configuration.data.site_xmat[site_id].reshape(3, 3).copy()
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
        q_command = self.previous_command + alpha * (
            q_command - self.previous_command
        )
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
            "normal_force": finger_step.measured_normal_force[0].astype(np.float32),
            "force_error": finger_step.force_error[0].astype(np.float32),
            "contact_active": finger_step.contact_active[0].copy(),
            "reference_speed": np.abs(
                finger_step.normal_velocity[0]
            ).astype(np.float32),
            "surface_error": surface_error.astype(np.float32),
            "normal_offset": finger_step.normal_offset[0].astype(np.float32),
            "normal_velocity": finger_step.normal_velocity[0].astype(np.float32),
            "normal_acceleration": finger_step.normal_acceleration[0].astype(np.float32),
            "nominal_posture_error": (q_live - nominal_q).astype(np.float32),
        }
