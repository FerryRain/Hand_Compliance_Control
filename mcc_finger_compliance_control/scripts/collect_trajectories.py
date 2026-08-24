from __future__ import annotations

import argparse
import os
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
import time

import h5py
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
from tqdm.auto import tqdm

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.sensor import ContactSensor
from mjlab.tasks.leaphand.leaphand_mcc_finger_env_cfg import (
    HARD_CONTACT_SOLIMP,
    HARD_CONTACT_SOLREF,
    MCCLeapHandPositionControlCfg,
    mcc_finger_contact_env_cfg,
    mcc_palm_free_contact_env_cfg,
)
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

from object_catalog import (
    MeshNormalOracle,
    get_motion_config,
    load_object_config,
    object_local_aabb,
)
from surface_mcc_finger import (
    FullHandMCCFingerConfig,
    FullHandMCCFingerController,
    GeometrySurfaceOracle,
)


TASK_ID = "Leaphand-Finger-MCC-Position-Control"
TIP_SITES = ("if_tip", "mf_tip", "rf_tip", "th_tip")
TIP_COLORS = (
    (1.0, 0.30, 0.20, 0.95),
    (0.20, 0.70, 1.0, 0.95),
    (1.0, 0.80, 0.15, 0.95),
    (0.80, 0.30, 1.0, 0.95),
)
# Vertices of the actual ``palm_lower`` collision-mesh outline on its
# object-facing plane.  The long x=-0.036095 edge is the three-finger/root side
# from which the mustard cap approaches in the failure case.  Checking a palm
# side edge or a few FSR centres does not represent that collision geometry.
PALM_PROTECTION_OUTLINE_VERTICES_LOCAL = np.asarray(
    (
        (-0.100095, -0.027242, -0.0347224),
        (-0.100095, -0.054761, -0.0347224),
        (-0.093899, -0.080485, -0.0347224),
        (-0.071635, -0.093574, -0.0347224),
        (-0.044283, -0.096601, -0.0347224),
        (-0.036095, -0.078225, -0.0347224),
        (-0.036095, 0.004332, -0.0347224),
        (-0.042189, 0.025758, -0.0347224),
        (-0.065295, 0.015398, -0.0347224),
        (-0.082695, -0.005922, -0.0347224),
    ),
    dtype=np.float64,
)


def _sample_closed_outline(
    vertices: np.ndarray,
    maximum_spacing_m: float,
) -> np.ndarray:
    """Sample every edge of a closed 3-D polygon without duplicate vertices."""
    polygon = np.asarray(vertices, dtype=np.float64)
    if polygon.ndim != 2 or polygon.shape[1] != 3 or len(polygon) < 3:
        raise ValueError("Palm outline must contain at least three 3-D vertices")
    if maximum_spacing_m <= 0.0:
        raise ValueError("Palm-outline sample spacing must be positive")
    samples: list[np.ndarray] = []
    for start, end in zip(polygon, np.roll(polygon, -1, axis=0), strict=True):
        edge = end - start
        segments = max(1, int(np.ceil(np.linalg.norm(edge) / maximum_spacing_m)))
        # The endpoint belongs to the following edge, so exclude it here.
        samples.extend(
            start + (index / segments) * edge for index in range(segments)
        )
    return np.asarray(samples, dtype=np.float64)


# A 5 mm perimeter discretization is fine enough to catch the bottle-cap rim
# between neighbouring samples while remaining cheap for batched collection.
PALM_PROTECTION_POINTS_LOCAL = _sample_closed_outline(
    PALM_PROTECTION_OUTLINE_VERTICES_LOCAL,
    maximum_spacing_m=0.005,
)


def _fingertip_force_world(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Read the four live contact-sensor resultants in world coordinates."""
    forces: list[torch.Tensor] = []
    for site_name in TIP_SITES:
        sensor = env.scene[f"{site_name}_contact"]
        if not isinstance(sensor, ContactSensor):
            raise TypeError(f"{site_name}_contact is not a ContactSensor")
        force = sensor.data.force
        if force is None:
            forces.append(torch.zeros((env.num_envs, 3), device=env.device))
            continue
        if sensor.data.found is not None:
            force = torch.where(
                sensor.data.found.unsqueeze(-1) > 0,
                force,
                torch.zeros_like(force),
            )
        forces.append(force.sum(dim=1))
    return torch.stack(forces, dim=1)


def _wxyz_multiply(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = lhs.unbind(-1)
    w2, x2, y2, z2 = rhs.unbind(-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def _wxyz_apply(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by normalized wxyz quaternions."""
    quaternion = quaternion / torch.linalg.vector_norm(
        quaternion, dim=-1, keepdim=True
    ).clamp_min(1.0e-8)
    xyz = quaternion[..., 1:]
    cross = 2.0 * torch.linalg.cross(xyz, vector, dim=-1)
    return vector + quaternion[..., :1] * cross + torch.linalg.cross(
        xyz, cross, dim=-1
    )


def _rotvec_to_wxyz_quat(rotvec: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle rotation vectors to normalized wxyz quaternions."""
    theta = torch.linalg.vector_norm(rotvec, dim=-1)
    axis = rotvec / theta.clamp_min(1.0e-8)
    return torch.cat(
        (
            torch.cos(theta / 2).unsqueeze(-1),
            axis * torch.sin(theta / 2).unsqueeze(-1),
        ),
        dim=-1,
    )


def _wxyz_quat_to_rotvec(quat: torch.Tensor) -> torch.Tensor:
    """Convert normalized wxyz quaternions to axis-angle rotation vectors."""
    q = quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True).clamp_min(
        1.0e-8
    )
    axis_norm = torch.linalg.vector_norm(q[..., 1:], dim=-1)
    theta = 2.0 * torch.atan2(axis_norm, q[..., 0].abs().clamp_min(1.0e-8))
    axis = q[..., 1:] / axis_norm.clamp_min(1.0e-8)
    return theta.unsqueeze(-1) * axis


def _quaternion_path_angle_deg(quaternions_wxyz: np.ndarray) -> float:
    """Accumulate realized SO(3) travel, robust to quaternion sign flips."""
    quaternion = np.asarray(quaternions_wxyz, dtype=np.float64)
    if quaternion.ndim != 2 or quaternion.shape[1] != 4 or len(quaternion) < 2:
        return 0.0
    quaternion /= np.linalg.norm(quaternion, axis=1, keepdims=True).clip(1.0e-12)
    dots = np.abs(np.sum(quaternion[1:] * quaternion[:-1], axis=1))
    increments = 2.0 * np.arccos(np.clip(dots, 0.0, 1.0))
    return float(np.rad2deg(np.sum(increments)))


def _find_local_body_index(env: ManagerBasedRlEnv, suffix: str) -> int:
    names = [body.name or "" for body in env.scene["robot"].data.indexing.bodies]
    for index, name in enumerate(names):
        if name == suffix or name.endswith(f"/{suffix}"):
            return index
    raise ValueError(f"Body {suffix!r} not found; available={names}")


def _find_local_site_index(env: ManagerBasedRlEnv, suffix: str) -> int:
    names = [site.name or "" for site in env.scene["robot"].data.indexing.sites]
    for index, name in enumerate(names):
        if name == suffix or name.endswith(f"/{suffix}"):
            return index
    raise ValueError(f"Site {suffix!r} not found; available={names}")


def _sample_initial_quaternion(
    nominal_quat: torch.Tensor,
    mode: str,
    jitter_deg: float,
) -> torch.Tensor:
    """Sample the object orientation used by both collection and visualization."""
    num_envs = nominal_quat.shape[0]
    device = nominal_quat.device
    if mode == "uniform":
        quaternion = torch.randn((num_envs, 4), device=device)
        return quaternion / torch.linalg.vector_norm(
            quaternion, dim=-1, keepdim=True
        ).clamp_min(1.0e-8)
    if mode == "jitter":
        axes = torch.randn((num_envs, 3), device=device)
        axes /= torch.linalg.vector_norm(axes, dim=-1, keepdim=True).clamp_min(
            1.0e-8
        )
        limit = np.deg2rad(jitter_deg)
        angles = torch.empty(num_envs, device=device).uniform_(-limit, limit)
        jitter = torch.cat(
            (
                torch.cos(angles / 2).unsqueeze(-1),
                axes * torch.sin(angles / 2).unsqueeze(-1),
            ),
            dim=-1,
        )
        quaternion = _wxyz_multiply(nominal_quat, jitter)
        return quaternion / torch.linalg.vector_norm(
            quaternion, dim=-1, keepdim=True
        ).clamp_min(1.0e-8)
    return nominal_quat.clone()


def _sample_motion_axes(
    initial_quat: torch.Tensor,
    allowed_axes: list[str] | tuple[str, ...],
    *,
    output_frame: str,
    sampling: str,
) -> tuple[torch.Tensor, list[str]]:
    """Sample configured axes in either the object-local or initial world frame."""
    if output_frame not in ("object", "world"):
        raise ValueError("output_frame must be 'object' or 'world'")
    if sampling not in ("random", "stratified"):
        raise ValueError("axis sampling must be 'random' or 'stratified'")
    if not allowed_axes:
        raise ValueError("motion.allowed_axes cannot be empty")
    known = {"principal_x", "principal_y", "principal_z", "uniform_sphere"}
    unknown = sorted(set(allowed_axes) - known)
    if unknown:
        raise ValueError(f"Unknown motion axes {unknown}; available={sorted(known)}")

    num_envs = initial_quat.shape[0]
    device = initial_quat.device
    choice = (
        torch.arange(num_envs, device=device) % len(allowed_axes)
        if sampling == "stratified"
        else torch.randint(len(allowed_axes), (num_envs,), device=device)
    )
    result = torch.zeros((num_envs, 3), device=device)
    labels: list[str] = []
    principal = {
        "principal_x": (1.0, 0.0, 0.0),
        "principal_y": (0.0, 1.0, 0.0),
        "principal_z": (0.0, 0.0, 1.0),
    }
    for env_id in range(num_envs):
        label = str(allowed_axes[int(choice[env_id])])
        labels.append(label)
        if label == "uniform_sphere":
            local_axis = torch.randn(3, device=device)
            local_axis /= torch.linalg.vector_norm(local_axis).clamp_min(1.0e-8)
        else:
            local_axis = torch.tensor(
                principal[label], device=device, dtype=initial_quat.dtype
            )
        axis = (
            _wxyz_apply(initial_quat[env_id], local_axis)
            if output_frame == "world"
            else local_axis
        )
        result[env_id] = axis
    return result, labels


class ObjectMotionController:
    """One trajectory of config-driven object motion shared by every run mode."""

    def __init__(
        self,
        env: ManagerBasedRlEnv,
        target_mocap_idx: int,
        *,
        motion_start: int,
        motion_length: int,
        dt: float,
        initial_orientation_mode: str,
        initial_orientation_jitter_deg: float,
        rotation_enabled: bool,
        rotation_allowed_axes: list[str] | tuple[str, ...],
        angular_speed_min: float,
        angular_speed_max: float,
        rotation_acceleration_time_s: float,
        axis_sampling: str,
        rotation_axis_override_local: np.ndarray | None,
        translation_enabled: bool,
        translation_allowed_axes: list[str] | tuple[str, ...],
        trans_speed_min: float,
        trans_speed_max: float,
        trans_distance_mode: str,
        trans_distance_ratio_range: tuple[float, float],
        trans_absolute_max_m: float,
        object_extent: np.ndarray,
        rotation_axis_profiles: dict[str, dict[str, object]] | None = None,
        translation_axis_profiles: dict[str, dict[str, object]] | None = None,
        default_segment_move_steps: int | None = None,
        default_segment_hold_steps: int = 0,
        segment_move_steps_override: int | None = None,
        segment_hold_steps_override: int | None = None,
        contact_gated_motion: bool = False,
        lock_horizontal_lowest_point: bool = False,
        surface_points_object: np.ndarray | None = None,
        lowest_point_anchor_world: np.ndarray | None = None,
        lowest_point_band_m: float = 0.002,
        lowest_point_follow_max_speed_m_s: float = 0.015,
        lowest_point_follow_time_constant_s: float = 0.20,
    ) -> None:
        self.env = env
        self.target_mocap_idx = target_mocap_idx
        self.motion_start = int(motion_start)
        self.motion_length = int(motion_length)
        self.dt = float(dt)
        self.initial_orientation_mode = initial_orientation_mode
        self.initial_orientation_jitter_deg = float(initial_orientation_jitter_deg)
        self.rotation_enabled = bool(rotation_enabled)
        self.rotation_allowed_axes = tuple(rotation_allowed_axes)
        self.angular_speed_min = float(angular_speed_min)
        self.angular_speed_max = float(angular_speed_max)
        self.rotation_acceleration_time_s = float(rotation_acceleration_time_s)
        self.axis_sampling = axis_sampling
        self.rotation_axis_override_local = (
            None
            if rotation_axis_override_local is None
            else np.asarray(rotation_axis_override_local, dtype=np.float64).reshape(3)
        )
        self.translation_enabled = bool(translation_enabled)
        self.translation_allowed_axes = tuple(translation_allowed_axes)
        self.trans_speed_min = float(trans_speed_min)
        self.trans_speed_max = float(trans_speed_max)
        self.trans_distance_mode = trans_distance_mode
        self.trans_distance_ratio_range = trans_distance_ratio_range
        self.trans_absolute_max_m = float(trans_absolute_max_m)
        self.rotation_axis_profiles = rotation_axis_profiles or {}
        self.translation_axis_profiles = translation_axis_profiles or {}
        self.default_segment_move_steps = int(
            default_segment_move_steps or self.motion_length
        )
        self.default_segment_hold_steps = int(default_segment_hold_steps)
        self.segment_move_steps_override = segment_move_steps_override
        self.segment_hold_steps_override = segment_hold_steps_override
        self.contact_gated_motion = bool(contact_gated_motion)
        self.lock_horizontal_lowest_point = bool(
            lock_horizontal_lowest_point
        )
        self.lowest_point_band_m = float(lowest_point_band_m)
        if self.lowest_point_band_m <= 0.0:
            raise ValueError("lowest_point_band_m must be positive")
        self.lowest_point_follow_max_speed_m_s = float(
            lowest_point_follow_max_speed_m_s
        )
        self.lowest_point_follow_time_constant_s = float(
            lowest_point_follow_time_constant_s
        )
        if self.lowest_point_follow_max_speed_m_s <= 0.0:
            raise ValueError("lowest_point_follow_max_speed_m_s must be positive")
        if self.lowest_point_follow_time_constant_s < 0.0:
            raise ValueError(
                "lowest_point_follow_time_constant_s must be non-negative"
            )
        if self.lock_horizontal_lowest_point:
            points = np.asarray(surface_points_object, dtype=np.float32)
            anchors = np.asarray(lowest_point_anchor_world, dtype=np.float32)
            if (
                points.ndim != 2
                or points.shape[1] != 3
                or len(points) < 4
                or not np.all(np.isfinite(points))
            ):
                raise ValueError(
                    "Lowest-point lock requires finite object-frame surface points"
                )
            if anchors.shape != (env.num_envs, 3) or not np.all(
                np.isfinite(anchors)
            ):
                raise ValueError(
                    "lowest_point_anchor_world must have shape (num_envs, 3)"
                )
            self.surface_points_object = torch.as_tensor(
                points, device=env.device, dtype=torch.float32
            )
            self.lowest_point_anchor_world = torch.as_tensor(
                anchors, device=env.device, dtype=torch.float32
            )
        else:
            self.surface_points_object = torch.empty(
                (0, 3), device=env.device, dtype=torch.float32
            )
            self.lowest_point_anchor_world = torch.zeros(
                (env.num_envs, 3), device=env.device, dtype=torch.float32
            )
        self.lowest_point_world = torch.zeros_like(
            self.lowest_point_anchor_world
        )
        self.lowest_point_compensation_world = torch.zeros_like(
            self.lowest_point_anchor_world
        )
        self.lowest_point_follow_velocity_world = torch.zeros_like(
            self.lowest_point_anchor_world
        )
        self.lowest_point_follow_offset_world = torch.zeros_like(
            self.lowest_point_anchor_world
        )
        self.object_extent = torch.as_tensor(
            np.abs(object_extent), device=env.device, dtype=torch.float32
        )
        self.episode_step = 0
        self.initial_pos = torch.empty(0, device=env.device)
        self.initial_quat = torch.empty(0, device=env.device)
        self.rotation_axes = torch.empty(0, device=env.device)
        self.rotation_axis_labels: list[str] = []
        self.angular_speeds = torch.empty(0, device=env.device)
        self.current_angular_speeds = torch.empty(0, device=env.device)
        self.translation_axes = torch.empty(0, device=env.device)
        self.translation_axis_labels: list[str] = []
        self.translation_speeds = torch.empty(0, device=env.device)
        self.translation_amplitudes = torch.empty(0, device=env.device)
        self.translation_phases = torch.empty(0, device=env.device)
        self.translation_active_time = torch.empty(0, device=env.device)
        self.segment_move_steps = torch.empty(
            0, dtype=torch.long, device=env.device
        )
        self.segment_hold_steps = torch.empty(
            0, dtype=torch.long, device=env.device
        )
        self.motion_active = torch.empty(
            0, dtype=torch.bool, device=env.device
        )
        self.motion_contact_ready = torch.empty(
            0, dtype=torch.bool, device=env.device
        )
        self.motion_schedule_step = torch.empty(
            0, dtype=torch.long, device=env.device
        )

    def _horizontal_lowest_patch_world(
        self,
        position_world: torch.Tensor,
        quaternion_world: torch.Tensor,
    ) -> torch.Tensor:
        """Return the centroid of the lowest horizontal surface patch.

        Averaging every sampled surface point inside a thin Z band avoids an
        arbitrary vertex/collision-part switch on flat bottoms.  The returned
        X/Y is the patch centroid while Z remains the exact sampled minimum.
        """

        if not self.lock_horizontal_lowest_point:
            return torch.zeros_like(position_world)
        points = self.surface_points_object.unsqueeze(0).expand(
            position_world.shape[0], -1, -1
        )
        quaternion = quaternion_world.unsqueeze(1).expand(
            -1, points.shape[1], -1
        )
        rotated = _wxyz_apply(quaternion, points)
        minimum_z = torch.min(rotated[..., 2], dim=1).values
        in_band = rotated[..., 2] <= (
            minimum_z[:, None] + self.lowest_point_band_m
        )
        weights = in_band.to(rotated.dtype)
        centroid = torch.sum(rotated * weights[..., None], dim=1) / (
            torch.sum(weights, dim=1, keepdim=True).clamp_min(1.0)
        )
        centroid[:, 2] = minimum_z
        return position_world + centroid

    def _apply_horizontal_lowest_lock(
        self,
        position_world: torch.Tensor,
        quaternion_world: torch.Tensor,
        anchor_world: torch.Tensor,
        *,
        update_debug: bool,
        smooth: bool,
    ) -> torch.Tensor:
        if not self.lock_horizontal_lowest_point:
            return position_world
        lowest = self._horizontal_lowest_patch_world(
            position_world, quaternion_world
        )
        residual = anchor_world - lowest
        if smooth:
            residual_norm = torch.linalg.vector_norm(
                residual, dim=-1, keepdim=True
            )
            desired_velocity = residual / self.dt
            desired_speed = torch.linalg.vector_norm(
                desired_velocity, dim=-1, keepdim=True
            )
            desired_velocity *= torch.clamp(
                self.lowest_point_follow_max_speed_m_s
                / desired_speed.clamp_min(1.0e-8),
                max=1.0,
            )
            if self.lowest_point_follow_time_constant_s > 0.0:
                alpha = self.dt / (
                    self.lowest_point_follow_time_constant_s + self.dt
                )
            else:
                alpha = 1.0
            velocity = (
                (1.0 - alpha) * self.lowest_point_follow_velocity_world
                + alpha * desired_velocity
            )
            step = velocity * self.dt
            # Never cross the anchor when the residual becomes smaller than
            # one filtered step.  This removes the small limit cycle that a
            # rate-limited tracker otherwise develops around the target.
            step_norm = torch.linalg.vector_norm(step, dim=-1, keepdim=True)
            step *= torch.clamp(
                residual_norm / step_norm.clamp_min(1.0e-8), max=1.0
            )
            reached = residual_norm <= step_norm
            velocity = torch.where(
                reached,
                torch.zeros_like(velocity),
                velocity,
            )
            compensation = step
            self.lowest_point_follow_velocity_world = velocity
            self.lowest_point_follow_offset_world += compensation
        else:
            compensation = residual
            # Reset uses the exact solution once.  Future-pose preview also
            # requests an exact geometric solution but must remain read-only,
            # otherwise calling preview_pose() would restart the velocity
            # filter on every control tick.
            if update_debug:
                self.lowest_point_follow_velocity_world.zero_()
        corrected = position_world + compensation
        if update_debug:
            self.lowest_point_world = self._horizontal_lowest_patch_world(
                corrected, quaternion_world
            )
            self.lowest_point_compensation_world = compensation.clone()
        return corrected

    @staticmethod
    def _axis_profile(
        profiles: dict[str, dict[str, object]], label: str
    ) -> dict[str, object]:
        profile = profiles.get(label, {})
        if not isinstance(profile, dict):
            raise ValueError(f"Axis profile {label!r} must be a mapping")
        return profile

    @staticmethod
    def _range_from_profile(
        profile: dict[str, object], key: str, fallback: tuple[float, float]
    ) -> tuple[float, float]:
        value = profile.get(key, fallback)
        array = np.asarray(value, dtype=np.float64)
        if array.shape != (2,) or not np.all(np.isfinite(array)):
            raise ValueError(f"Axis profile {key} must contain two finite values")
        lower, upper = float(array[0]), float(array[1])
        if lower < 0.0 or upper < lower:
            raise ValueError(f"Invalid axis profile range {key}={value}")
        return lower, upper

    def reset(self) -> None:
        sim_data = self.env.sim.data
        num_envs = self.env.num_envs
        self.episode_step = 0
        self.lowest_point_follow_velocity_world.zero_()
        self.lowest_point_follow_offset_world.zero_()
        self.initial_pos = sim_data.mocap_pos[:, self.target_mocap_idx, :].clone()
        nominal_quat = sim_data.mocap_quat[:, self.target_mocap_idx, :].clone()
        self.initial_quat = _sample_initial_quaternion(
            nominal_quat,
            self.initial_orientation_mode,
            self.initial_orientation_jitter_deg,
        )
        if self.lock_horizontal_lowest_point:
            self.initial_pos = self._apply_horizontal_lowest_lock(
                self.initial_pos,
                self.initial_quat,
                self.lowest_point_anchor_world,
                update_debug=True,
                smooth=False,
            )
        sim_data.mocap_pos[:, self.target_mocap_idx, :] = self.initial_pos
        sim_data.mocap_quat[:, self.target_mocap_idx, :] = self.initial_quat

        self.rotation_axes, self.rotation_axis_labels = _sample_motion_axes(
            self.initial_quat,
            self.rotation_allowed_axes,
            output_frame="object",
            sampling=self.axis_sampling,
        )
        if self.rotation_axis_override_local is not None:
            override = torch.as_tensor(
                self.rotation_axis_override_local,
                device=self.env.device,
                dtype=self.initial_quat.dtype,
            )
            override /= torch.linalg.vector_norm(override).clamp_min(1.0e-8)
            self.rotation_axes[:] = override
            self.rotation_axis_labels = ["custom"] * num_envs
        angular_ranges = [
            self._range_from_profile(
                self._axis_profile(self.rotation_axis_profiles, label),
                "angular_speed_range_rad_s",
                (self.angular_speed_min, self.angular_speed_max),
            )
            for label in self.rotation_axis_labels
        ]
        angular_lower = torch.tensor(
            [value[0] for value in angular_ranges], device=self.env.device
        )
        angular_upper = torch.tensor(
            [value[1] for value in angular_ranges], device=self.env.device
        )
        self.angular_speeds = angular_lower + torch.rand(
            num_envs, device=self.env.device
        ) * (angular_upper - angular_lower)
        self.current_angular_speeds = torch.zeros_like(self.angular_speeds)

        self.translation_axes, self.translation_axis_labels = _sample_motion_axes(
            self.initial_quat,
            self.translation_allowed_axes,
            output_frame="world",
            sampling=self.axis_sampling,
        )
        translation_profiles = [
            self._axis_profile(self.translation_axis_profiles, label)
            for label in self.translation_axis_labels
        ]
        translation_ranges = [
            self._range_from_profile(
                profile,
                "speed_range_m_s",
                (self.trans_speed_min, self.trans_speed_max),
            )
            for profile in translation_profiles
        ]
        translation_lower = torch.tensor(
            [value[0] for value in translation_ranges], device=self.env.device
        )
        translation_upper = torch.tensor(
            [value[1] for value in translation_ranges], device=self.env.device
        )
        self.translation_speeds = translation_lower + torch.rand(
            num_envs, device=self.env.device
        ) * (translation_upper - translation_lower)
        if self.trans_distance_mode == "object_extent_ratio":
            ratio_ranges = [
                self._range_from_profile(
                    profile,
                    "distance_ratio_range",
                    self.trans_distance_ratio_range,
                )
                for profile in translation_profiles
            ]
            ratio_lower = torch.tensor(
                [value[0] for value in ratio_ranges], device=self.env.device
            )
            ratio_upper = torch.tensor(
                [value[1] for value in ratio_ranges], device=self.env.device
            )
            ratio = ratio_lower + torch.rand(
                num_envs, device=self.env.device
            ) * (ratio_upper - ratio_lower)
            extent_along_axis = torch.sum(
                self.object_extent * torch.abs(self.translation_axes), dim=-1
            )
            absolute_max = torch.tensor(
                [
                    float(profile.get("absolute_max_m", self.trans_absolute_max_m))
                    for profile in translation_profiles
                ],
                device=self.env.device,
            )
            self.translation_amplitudes = torch.clamp(
                0.5 * ratio * extent_along_axis,
                max=absolute_max,
            )
        else:
            self.translation_amplitudes = torch.full(
                (num_envs,), self.trans_absolute_max_m, device=self.env.device
            )
        # Start every translation at zero displacement.  A random phase here
        # would teleport the object by up to one amplitude at motion_start.
        self.translation_phases = torch.zeros(num_envs, device=self.env.device)
        self.translation_active_time = torch.zeros(
            num_envs, device=self.env.device
        )

        move_steps: list[int] = []
        hold_steps: list[int] = []
        for env_id in range(num_envs):
            active_profiles: list[dict[str, object]] = []
            if self.rotation_enabled:
                active_profiles.append(
                    self._axis_profile(
                        self.rotation_axis_profiles,
                        self.rotation_axis_labels[env_id],
                    )
                )
            if self.translation_enabled:
                active_profiles.append(translation_profiles[env_id])
            configured_moves = [
                int(profile.get("segment_move_steps", self.default_segment_move_steps))
                for profile in active_profiles
            ]
            configured_holds = [
                int(profile.get("segment_hold_steps", self.default_segment_hold_steps))
                for profile in active_profiles
            ]
            move = (
                int(self.segment_move_steps_override)
                if self.segment_move_steps_override is not None
                else min(configured_moves, default=self.default_segment_move_steps)
            )
            hold = (
                int(self.segment_hold_steps_override)
                if self.segment_hold_steps_override is not None
                else max(configured_holds, default=self.default_segment_hold_steps)
            )
            if move <= 0 or hold < 0:
                raise ValueError(
                    "Invalid segmented motion schedule "
                    f"move={move}, hold={hold}"
                )
            move_steps.append(move)
            hold_steps.append(hold)
        self.segment_move_steps = torch.tensor(
            move_steps, dtype=torch.long, device=self.env.device
        )
        self.segment_hold_steps = torch.tensor(
            hold_steps, dtype=torch.long, device=self.env.device
        )
        self.motion_active = torch.zeros(
            num_envs, dtype=torch.bool, device=self.env.device
        )
        self.motion_contact_ready = torch.ones(
            num_envs, dtype=torch.bool, device=self.env.device
        )
        self.motion_schedule_step = torch.zeros(
            num_envs, dtype=torch.long, device=self.env.device
        )

    def step(
        self,
        episode_step: int | None = None,
        contact_ready: np.ndarray | torch.Tensor | None = None,
    ) -> bool:
        if episode_step is None:
            episode_step = self.episode_step
            self.episode_step += 1
        else:
            self.episode_step = int(episode_step)
        in_window = self.motion_start <= episode_step < (
            self.motion_start + self.motion_length
        )
        if not in_window:
            self.current_angular_speeds.zero_()
            self.motion_active.zero_()
            return False
        if contact_ready is None or not self.contact_gated_motion:
            self.motion_contact_ready.fill_(True)
        else:
            self.motion_contact_ready = torch.as_tensor(
                contact_ready, dtype=torch.bool, device=self.env.device
            ).reshape(self.env.num_envs)
        cycle_steps = self.segment_move_steps + self.segment_hold_steps
        phase_steps = self.motion_schedule_step % cycle_steps
        planned_active = phase_steps < self.segment_move_steps
        self.motion_active = planned_active & self.motion_contact_ready
        # Holds always consume schedule time.  A planned move consumes time
        # only while all fingertips are ready, so loss freezes the object at
        # its current pose instead of silently skipping part of the path.
        schedule_advances = (~planned_active) | self.motion_contact_ready
        self.motion_schedule_step += schedule_advances.to(torch.long)
        sim_data = self.env.sim.data
        if self.rotation_enabled:
            if self.rotation_acceleration_time_s > 0.0:
                elapsed = (phase_steps.to(torch.float32) + 1.0) * self.dt
                remaining = (
                    self.segment_move_steps.to(torch.float32)
                    - phase_steps.to(torch.float32)
                ) * self.dt
                ramp = torch.minimum(
                    torch.ones_like(elapsed),
                    torch.minimum(
                        elapsed / self.rotation_acceleration_time_s,
                        remaining / self.rotation_acceleration_time_s,
                    ),
                )
                ramp = ramp * ramp * (3.0 - 2.0 * ramp)
            else:
                ramp = torch.ones_like(self.angular_speeds)
            self.current_angular_speeds = (
                self.angular_speeds * ramp * self.motion_active
            )
            angle = self.current_angular_speeds * self.dt
            delta = torch.cat(
                (
                    torch.cos(angle / 2).unsqueeze(-1),
                    self.rotation_axes * torch.sin(angle / 2).unsqueeze(-1),
                ),
                dim=-1,
            )
            quat = _wxyz_multiply(
                sim_data.mocap_quat[:, self.target_mocap_idx, :].clone(), delta
            )
            sim_data.mocap_quat[:, self.target_mocap_idx, :] = (
                quat
                / torch.linalg.vector_norm(quat, dim=-1, keepdim=True).clamp_min(
                    1.0e-8
                )
            )
        if self.translation_enabled:
            self.translation_active_time += (
                self.motion_active.to(torch.float32) * self.dt
            )
            omega = self.translation_speeds / self.translation_amplitudes.clamp_min(
                1.0e-8
            )
            offset = self.translation_amplitudes * torch.sin(
                omega * self.translation_active_time + self.translation_phases
            )
            sim_data.mocap_pos[:, self.target_mocap_idx, :] = (
                self.initial_pos + self.translation_axes * offset.unsqueeze(-1)
            )
            if self.lock_horizontal_lowest_point:
                sim_data.mocap_pos[:, self.target_mocap_idx, :] += (
                    self.lowest_point_follow_offset_world
                )
        if self.lock_horizontal_lowest_point:
            translation_offset = (
                self.translation_axes
                * (
                    self.translation_amplitudes
                    * torch.sin(
                        self.translation_speeds
                        / self.translation_amplitudes.clamp_min(1.0e-8)
                        * self.translation_active_time
                        + self.translation_phases
                    )
                ).unsqueeze(-1)
                if self.translation_enabled
                else torch.zeros_like(self.initial_pos)
            )
            anchor = self.lowest_point_anchor_world + translation_offset
            current_pos = sim_data.mocap_pos[:, self.target_mocap_idx, :].clone()
            current_quat = sim_data.mocap_quat[:, self.target_mocap_idx, :].clone()
            sim_data.mocap_pos[:, self.target_mocap_idx, :] = (
                self._apply_horizontal_lowest_lock(
                    current_pos,
                    current_quat,
                    anchor,
                    update_debug=True,
                    smooth=True,
                )
            )
        return bool(torch.any(self.motion_active))

    def preview_pose(self, horizon_s: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the privileged scheduled object pose ``horizon_s`` ahead.

        Collection owns the mocap trajectory, so its teacher can anticipate a
        rotating/translated surface instead of waiting for contact to push the
        palm away.  Preview is clipped at the current move-segment and global
        motion-window boundaries; hold frames therefore never get extrapolated.
        """
        sim_data = self.env.sim.data
        pos = sim_data.mocap_pos[:, self.target_mocap_idx, :].clone()
        quat = sim_data.mocap_quat[:, self.target_mocap_idx, :].clone()
        if horizon_s <= 0.0 or self.motion_active.numel() == 0:
            return pos, quat

        cycle_steps = self.segment_move_steps + self.segment_hold_steps
        phase_steps = self.motion_schedule_step % cycle_steps
        remaining_segment_steps = torch.clamp(
            self.segment_move_steps - phase_steps, min=0
        )
        remaining_window_steps = max(
            0, self.motion_start + self.motion_length - int(self.episode_step) - 1
        )
        preview_time = torch.minimum(
            torch.full_like(self.angular_speeds, float(horizon_s)),
            remaining_segment_steps.to(torch.float32) * self.dt,
        )
        preview_time = torch.minimum(
            preview_time,
            torch.full_like(preview_time, remaining_window_steps * self.dt),
        )
        preview_time *= self.motion_active.to(torch.float32)

        if self.rotation_enabled:
            angle = self.angular_speeds * preview_time
            delta = torch.cat(
                (
                    torch.cos(angle / 2).unsqueeze(-1),
                    self.rotation_axes * torch.sin(angle / 2).unsqueeze(-1),
                ),
                dim=-1,
            )
            quat = _wxyz_multiply(quat, delta)
            quat /= torch.linalg.vector_norm(
                quat, dim=-1, keepdim=True
            ).clamp_min(1.0e-8)
        if self.translation_enabled:
            omega = self.translation_speeds / self.translation_amplitudes.clamp_min(
                1.0e-8
            )
            future_active_time = self.translation_active_time + preview_time
            offset = self.translation_amplitudes * torch.sin(
                omega * future_active_time + self.translation_phases
            )
            future_pos = self.initial_pos + self.translation_axes * offset.unsqueeze(-1)
            pos = torch.where(self.motion_active[:, None], future_pos, pos)
        if self.lock_horizontal_lowest_point:
            translation_offset = (
                self.translation_axes * offset.unsqueeze(-1)
                if self.translation_enabled
                else torch.zeros_like(pos)
            )
            anchor = self.lowest_point_anchor_world + translation_offset
            corrected = self._apply_horizontal_lowest_lock(
                pos,
                quat,
                anchor,
                update_debug=False,
                smooth=False,
            )
            pos = torch.where(self.motion_active[:, None], corrected, pos)
        return pos, quat


class PlannedFixedPalmObjectController:
    """Invert a smooth palm-in-object trajectory into object mocap motion.

    This is an opt-in collection baseline.  First a virtual palm pose is
    planned in the object frame (smoothstep progress and a chosen local
    manifold direction).  It is then inverted at every frame with
    ``T_object_world = T_palm_world @ inv(T_palm_object)``.  The physical palm
    therefore stays on its fixed world MCC target while the mocap object
    follows exactly the planned relative trajectory.  Finger targets are
    generated by FullHandMCC's surface oracle and per-finger QP; legacy motion
    modes are untouched.
    """

    def __init__(
        self,
        env: ManagerBasedRlEnv,
        target_mocap_idx: int,
        palm_body_idx: int,
        fixed_palm_target_np: np.ndarray,
        object_initial_pos_local: np.ndarray,
        object_initial_quat_wxyz: np.ndarray,
        motion_start: int,
        motion_length: int,
        total_steps: int,
        dt: float,
        axis_local: np.ndarray,
        angle_deg: float,
        direction: int,
    ) -> None:
        if motion_length <= 0 or total_steps <= motion_start:
            raise ValueError("Invalid planned palm trajectory horizon")
        if angle_deg <= 0.0:
            raise ValueError("manifold angle must be positive")
        if direction not in (-1, 1):
            raise ValueError("manifold direction must be -1 or +1")
        axis = np.asarray(axis_local, dtype=np.float64).reshape(3)
        axis /= max(float(np.linalg.norm(axis)), 1.0e-12)
        self.env = env
        self.target_mocap_idx = int(target_mocap_idx)
        self.palm_body_idx = int(palm_body_idx)
        self.motion_start = int(motion_start)
        self.motion_length = int(motion_length)
        self.dt = float(dt)
        self.axis_local = axis.astype(np.float32)
        self.direction = int(direction)
        self.angle_deg = float(angle_deg)
        self._total_steps = int(total_steps)
        self.rotation_enabled = True
        self.translation_enabled = False
        self.lock_horizontal_lowest_point = False
        self.lowest_point_world = torch.full(
            (env.num_envs, 3), float("nan"), device=env.device
        )
        self.lowest_point_anchor_world = torch.full_like(
            self.lowest_point_world, float("nan")
        )
        self.lowest_point_compensation_world = torch.zeros_like(
            self.lowest_point_world
        )
        self.lowest_point_follow_velocity_world = torch.zeros_like(
            self.lowest_point_world
        )
        self.motion_active = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        self.motion_contact_ready = torch.ones_like(self.motion_active)
        self.motion_schedule_step = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
        self.segment_move_steps = torch.full(
            (env.num_envs,), motion_length, dtype=torch.long, device=env.device
        )
        self.segment_hold_steps = torch.zeros_like(self.segment_move_steps)
        self.translation_axes = torch.zeros(
            (env.num_envs, 3), device=env.device
        )
        self.translation_speeds = torch.zeros(
            env.num_envs, device=env.device
        )
        self.translation_amplitudes = torch.zeros(
            env.num_envs, device=env.device
        )
        self.angular_speeds = torch.full(
            (env.num_envs,),
            np.deg2rad(angle_deg) / (motion_length * dt),
            device=env.device,
        )
        self.current_angular_speeds = self.angular_speeds.clone()
        self.rotation_axes = torch.as_tensor(
            np.broadcast_to(axis, (env.num_envs, 3)),
            device=env.device,
            dtype=torch.float32,
        )
        self.rotation_axis_labels = ["manifold_local_axis"] * env.num_envs
        self.initial_pos = torch.empty(
            (env.num_envs, 3), device=env.device, dtype=torch.float32
        )
        self.initial_quat = torch.empty(
            (env.num_envs, 4), device=env.device, dtype=torch.float32
        )
        self._path = self._build_path(
            np.asarray(fixed_palm_target_np, dtype=np.float64),
            np.asarray(object_initial_pos_local, dtype=np.float64),
            np.asarray(object_initial_quat_wxyz, dtype=np.float64),
            total_steps,
        )
        self.reset()

    @staticmethod
    def _wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
        return R.from_quat(np.roll(quaternion / np.linalg.norm(quaternion), -1)).as_matrix()

    @staticmethod
    def _matrix_to_wxyz(rotation: np.ndarray) -> np.ndarray:
        q = R.from_matrix(rotation).as_quat()
        return q[[3, 0, 1, 2]]

    def _build_path(
        self,
        fixed_palm_target_np: np.ndarray,
        object_initial_pos_local: np.ndarray,
        object_initial_quat_wxyz: np.ndarray,
        total_steps: int,
    ) -> torch.Tensor:
        palm_pos_local = fixed_palm_target_np[:3]
        palm_rot_world = R.from_rotvec(fixed_palm_target_np[3:6]).as_matrix()
        object_rot = self._wxyz_to_matrix(object_initial_quat_wxyz)
        env_origins = self.env.scene.env_origins.detach().cpu().numpy()
        palm_pos_world = np.broadcast_to(
            palm_pos_local[None, :] + env_origins,
            (self.env.num_envs, 3),
        ).copy()
        palm_rot_world_batch = np.broadcast_to(
            palm_rot_world, (self.env.num_envs, 3, 3)
        ).copy()
        object_pos_world = np.broadcast_to(
            object_initial_pos_local[None, :] + env_origins,
            (self.env.num_envs, 3),
        ).copy()
        object_rot_batch = np.broadcast_to(
            object_rot, (self.env.num_envs, 3, 3)
        ).copy()
        return self._build_path_from_world_reference(
            palm_pos_world,
            palm_rot_world_batch,
            object_pos_world,
            object_rot_batch,
            total_steps,
        )

    def _build_path_from_world_reference(
        self,
        palm_pos_world: np.ndarray,
        palm_rot_world: np.ndarray,
        object_pos_world: np.ndarray,
        object_rot: np.ndarray,
        total_steps: int,
    ) -> torch.Tensor:
        """Build an inverted object path from a measured contact-frame pose."""
        palm_pos_world = np.asarray(palm_pos_world, dtype=np.float64).reshape(
            self.env.num_envs, 3
        )
        palm_rot_world = np.asarray(palm_rot_world, dtype=np.float64).reshape(
            self.env.num_envs, 3, 3
        )
        object_pos_world = np.asarray(object_pos_world, dtype=np.float64).reshape(
            self.env.num_envs, 3
        )
        object_rot = np.asarray(object_rot, dtype=np.float64).reshape(
            self.env.num_envs, 3, 3
        )
        palm_pos_object = np.einsum(
            "bij,bj->bi", np.swapaxes(object_rot, 1, 2),
            palm_pos_world - object_pos_world,
        )
        palm_rot_object = np.einsum(
            "bij,bjk->bik", np.swapaxes(object_rot, 1, 2), palm_rot_world
        )
        axis = self.axis_local.astype(np.float64)
        axis_skew = np.array(
            (
                (0.0, -axis[2], axis[1]),
                (axis[2], 0.0, -axis[0]),
                (-axis[1], axis[0], 0.0),
            )
        )
        path = np.zeros(
            (self.env.num_envs, total_steps + 1, 7), dtype=np.float32
        )
        palm_path_object = np.zeros(
            (self.env.num_envs, total_steps + 1, 3), dtype=np.float32
        )
        for env_id in range(self.env.num_envs):
            for step in range(total_steps + 1):
                progress = np.clip(
                    (step - self.motion_start) / self.motion_length,
                    0.0,
                    1.0,
                )
                smooth = progress * progress * (3.0 - 2.0 * progress)
                angle = self.direction * np.deg2rad(self.angle_deg) * smooth
                # Rodrigues rotation transports the palm frame in the object
                # chart without a quaternion sign discontinuity.
                rot_delta = (
                    np.eye(3)
                    + np.sin(angle) * axis_skew
                    + (1.0 - np.cos(angle)) * (axis_skew @ axis_skew)
                )
                # Plan both translation and orientation in the object frame.
                # This is a rigid surface-manifold orbit of the measured
                # contact pose, not a fixed-point wrist rotation.  The object
                # pose below is then obtained by exact SE(3) inversion.
                palm_pos_object_t = rot_delta @ palm_pos_object[env_id]
                palm_rot_object_t = rot_delta @ palm_rot_object[env_id]
                object_rot_t = palm_rot_world[env_id] @ palm_rot_object_t.T
                object_pos_t = palm_pos_world[env_id] - object_rot_t @ palm_pos_object_t
                palm_path_object[env_id, step] = palm_pos_object_t
                path[env_id, step, :3] = object_pos_t
                path[env_id, step, 3:] = self._matrix_to_wxyz(object_rot_t)
        self._palm_path_object = palm_path_object
        return torch.as_tensor(path, device=self.env.device)

    def reanchor_from_current_state(self) -> None:
        """Use the settled, measured palm/object pose as trajectory frame 0."""
        robot = self.env.scene["robot"]
        palm_pose = robot.data.body_link_pose_w[:, self.palm_body_idx, :]
        palm_pos = palm_pose[:, :3].detach().cpu().numpy()
        palm_quat = palm_pose[:, 3:7].detach().cpu().numpy()
        palm_rot = np.stack(
            [self._wxyz_to_matrix(quaternion) for quaternion in palm_quat], axis=0
        )
        object_pos = self.env.sim.data.mocap_pos[:, self.target_mocap_idx, :].detach().cpu().numpy()
        object_quat = self.env.sim.data.mocap_quat[:, self.target_mocap_idx, :].detach().cpu().numpy()
        object_rot = np.stack(
            [self._wxyz_to_matrix(quaternion) for quaternion in object_quat], axis=0
        )
        self._path = self._build_path_from_world_reference(
            palm_pos,
            palm_rot,
            object_pos,
            object_rot,
            self._total_steps,
        )
        self.initial_pos = self._path[:, 0, :3].clone()
        self.initial_quat = self._path[:, 0, 3:7].clone()
        print(
            "[MANIFOLD] reanchored from measured contact pose | "
            f"palm0={np.round(palm_pos[0], 4).tolist()} "
            f"object0={np.round(object_pos[0], 4).tolist()}"
        )

    def reset(self) -> None:
        self.motion_schedule_step.zero_()
        self.motion_active.zero_()
        self.initial_pos = self._path[:, 0, :3].clone()
        self.initial_quat = self._path[:, 0, 3:7].clone()
        self.env.sim.data.mocap_pos[:, self.target_mocap_idx, :] = self.initial_pos
        self.env.sim.data.mocap_quat[:, self.target_mocap_idx, :] = self.initial_quat

    def step(self, episode_step: int, contact_ready=None) -> bool:
        del contact_ready
        index = min(max(int(episode_step), 0), self._path.shape[1] - 1)
        pose = self._path[:, index]
        self.env.sim.data.mocap_pos[:, self.target_mocap_idx, :] = pose[:, :3]
        self.env.sim.data.mocap_quat[:, self.target_mocap_idx, :] = pose[:, 3:7]
        active = (episode_step >= self.motion_start) and (
            episode_step < self.motion_start + self.motion_length
        )
        self.motion_active.fill_(active)
        self.motion_schedule_step.fill_(max(0, episode_step - self.motion_start))
        return active

    def preview_pose(self, horizon_s: float) -> tuple[torch.Tensor, torch.Tensor]:
        index = int(self.motion_schedule_step[0].item()) + int(
            round(max(0.0, horizon_s) / self.dt)
        ) + self.motion_start
        index = min(max(index, 0), self._path.shape[1] - 1)
        return self._path[:, index, :3], self._path[:, index, 3:7]


class InversePlannedPalmObjectController:
    """Apply a planned object-frame palm path as fixed-palm object mocap motion."""

    def __init__(
        self,
        env: ManagerBasedRlEnv,
        target_mocap_idx: int,
        palm_body_idx: int,
        palm_pose_object: np.ndarray,
        motion_start: int,
        motion_length: int,
        total_steps: int,
        dt: float,
    ) -> None:
        plan = np.asarray(palm_pose_object, dtype=np.float64)
        if plan.ndim != 2 or plan.shape[1] != 7 or len(plan) < 2:
            raise ValueError("planner H5 must contain a (T,7) palm_pose_object path")
        self.env = env
        self.target_mocap_idx = int(target_mocap_idx)
        self.palm_body_idx = int(palm_body_idx)
        self.motion_start = int(motion_start)
        self.motion_length = int(motion_length)
        self.total_steps = int(total_steps)
        self.dt = float(dt)
        self.plan_pos = plan[:, :3]
        self.plan_rot = np.stack(
            [PlannedFixedPalmObjectController._wxyz_to_matrix(q) for q in plan[:, 3:]],
            axis=0,
        )
        self.plan_rot0_inv = self.plan_rot[0].T
        self.rotation_enabled = True
        self.translation_enabled = False
        self.lock_horizontal_lowest_point = False
        self.motion_active = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.motion_contact_ready = torch.ones_like(self.motion_active)
        self.motion_schedule_step = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.segment_move_steps = torch.full((env.num_envs,), motion_length, dtype=torch.long, device=env.device)
        self.segment_hold_steps = torch.zeros_like(self.segment_move_steps)
        self.translation_axes = torch.zeros((env.num_envs, 3), device=env.device)
        self.translation_speeds = torch.zeros(env.num_envs, device=env.device)
        self.translation_amplitudes = torch.zeros(env.num_envs, device=env.device)
        self.translation_axis_labels = ["none"] * env.num_envs
        self.rotation_axes = torch.zeros((env.num_envs, 3), device=env.device)
        self.rotation_axis_labels = ["planner_object_frame"] * env.num_envs
        self.angular_speeds = torch.zeros(env.num_envs, device=env.device)
        self.current_angular_speeds = self.angular_speeds.clone()
        self.lowest_point_world = torch.full((env.num_envs, 3), float("nan"), device=env.device)
        self.lowest_point_anchor_world = torch.full_like(self.lowest_point_world, float("nan"))
        self.lowest_point_compensation_world = torch.zeros_like(self.lowest_point_world)
        self.lowest_point_follow_velocity_world = torch.zeros_like(self.lowest_point_world)
        self._path: torch.Tensor | None = None
        self.initial_pos = torch.zeros((env.num_envs, 3), device=env.device)
        self.initial_quat = torch.zeros((env.num_envs, 4), device=env.device)
        self.initial_quat[:, 0] = 1.0

    def reset(self) -> None:
        self.motion_active.zero_()
        self.motion_schedule_step.zero_()
        self._path = None
        self.initial_pos = self.env.sim.data.mocap_pos[:, self.target_mocap_idx, :].clone()
        self.initial_quat = self.env.sim.data.mocap_quat[:, self.target_mocap_idx, :].clone()

    def _plan_index(self, episode_step: int) -> int:
        u = np.clip((episode_step - self.motion_start) / max(self.motion_length, 1), 0.0, 1.0)
        return int(round(u * (len(self.plan_pos) - 1)))

    def reanchor_from_current_state(self) -> None:
        """Invert the plan using only the stable measured palm pose.

        The plan is expressed as ``T_p^o`` (palm pose in object coordinates).
        Once the arm has settled, keep that measured ``T_p^w`` fixed and set
        the object to ``T_o^w = T_p^w inv(T_p^o)``.  Deliberately do not read
        the current object pose here: doing so would silently re-anchor the
        planned path to whatever object placement happened during prep.
        """
        robot = self.env.scene["robot"]
        palm = robot.data.body_link_pose_w[:, self.palm_body_idx, :].detach().cpu().numpy()
        palm_rot = np.stack([PlannedFixedPalmObjectController._wxyz_to_matrix(q) for q in palm[:, 3:]], axis=0)
        path = np.zeros((self.env.num_envs, self.total_steps + 1, 7), dtype=np.float32)
        for step in range(self.total_steps + 1):
            index = self._plan_index(step)
            for env_id in range(self.env.num_envs):
                relative_rot = self.plan_rot[index]
                relative_pos = self.plan_pos[index]
                object_rotation = palm_rot[env_id] @ relative_rot.T
                object_position = palm[env_id, :3] - object_rotation @ relative_pos
                path[env_id, step, :3] = object_position
                path[env_id, step, 3:] = PlannedFixedPalmObjectController._matrix_to_wxyz(object_rotation)
        self._path = torch.as_tensor(path, device=self.env.device)
        self.initial_pos = self._path[:, 0, :3].clone()
        self.initial_quat = self._path[:, 0, 3:].clone()
        print(
            f"[PLANNER-INVERSE] inverted planned path ({len(self.plan_pos)} frames) "
            "from stable palm pose only"
        )

    def step(self, episode_step: int, contact_ready=None) -> bool:
        del contact_ready
        if self._path is None:
            raise RuntimeError("planner trajectory was not reanchored before motion")
        # ``motion_start`` is deliberately delayed by planner_settle_steps.
        # Before that instant the inverted object must remain exactly at the
        # first planned pose; indexing by absolute episode_step would silently
        # advance the path during the finger-contact settling interval.
        if episode_step < self.motion_start:
            index = 0
        else:
            index = min(
                max(int(episode_step - self.motion_start), 0),
                self._path.shape[1] - 1,
            )
        pose = self._path[:, index]
        self.env.sim.data.mocap_pos[:, self.target_mocap_idx, :] = pose[:, :3]
        self.env.sim.data.mocap_quat[:, self.target_mocap_idx, :] = pose[:, 3:]
        active = self.motion_start <= episode_step < self.motion_start + self.motion_length
        self.motion_active.fill_(active)
        self.motion_schedule_step.fill_(max(0, episode_step - self.motion_start))
        return active

    def preview_pose(self, horizon_s: float) -> tuple[torch.Tensor, torch.Tensor]:
        if self._path is None:
            return self.initial_pos, self.initial_quat
        index = min(
            self._path.shape[1] - 1,
            self.motion_start + int(self.motion_schedule_step[0].item()) + int(round(horizon_s / self.dt)),
        )
        return self._path[:, index, :3], self._path[:, index, 3:]


class PalmOrbitController:
    """Contact-gated palm orbit about the object's long axis (privileged).

    The object stays still; the palm target is the calibrated fixed-target
    pose rigidly rotated by a triangle-wave phase about the object's long
    axis through its centre.  Rigid rotation keeps the hand-object relative
    geometry constant, so fingertip contacts stay on the pads by
    construction (the surface under each pad does not slide).

    Privileged information: the reset object pose (mocap centre) and the
    object-local AABB define centre, axis and orbit radius; the orbit axis
    is the AABB's longest local axis mapped to world by the reset
    quaternion.  Phase advances only while all four fingertips are loaded,
    mirroring ObjectMotionController's contact gate.
    """

    def __init__(
        self,
        env: ManagerBasedRlEnv,
        fixed_target: torch.Tensor,
        target_mocap_idx: int,
        object_extent_local: np.ndarray,
        angular_speed_min: float,
        angular_speed_max: float,
        amplitude_deg_min: float,
        amplitude_deg_max: float,
        motion_start: int,
        motion_length: int,
        dt: float,
        device: str,
    ) -> None:
        self.env = env
        self.fixed_target = fixed_target  # (B,6) world pos + rotvec, float32
        self.mocap_idx = int(target_mocap_idx)
        self.motion_start = int(motion_start)
        self.motion_length = int(motion_length)
        self.dt = float(dt)
        self.device = device
        num_envs = int(env.num_envs)
        self.long_axis_local = int(np.argmax(np.abs(object_extent_local)))
        uniform = torch.rand(num_envs, device=device)
        self.angular_speeds = (
            angular_speed_min + (angular_speed_max - angular_speed_min) * uniform
        ).to(torch.float32)
        uniform = torch.rand(num_envs, device=device)
        # torch.deg2rad (not np.deg2rad) so a CUDA device tensor works.
        self.amplitudes_rad = torch.deg2rad(
            amplitude_deg_min + (amplitude_deg_max - amplitude_deg_min) * uniform
        ).to(torch.float32)
        self.phase = torch.zeros(num_envs, device=device)
        self.direction = torch.ones(num_envs, device=device)
        self.current_x_des = self.fixed_target.clone()
        self.motion_active = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.motion_contact_ready = torch.zeros(
            num_envs, dtype=torch.bool, device=device
        )
        self.object_center = torch.zeros((num_envs, 3), device=device)
        self.orbit_axis_world = torch.zeros((num_envs, 3), device=device)
        self.offset0 = torch.zeros((num_envs, 3), device=device)
        self.rot0 = torch.zeros((num_envs, 4), device=device)  # wxyz quat
        self._captured = False

    def reset(self) -> None:
        """Reset phase/window state; the object pose is snapshotted lazily."""
        self.phase.zero_()
        self.direction.fill_(1.0)
        self.motion_active.zero_()
        self.motion_contact_ready.zero_()
        self._captured = False
        self.current_x_des = self.fixed_target.clone()

    def _capture(self) -> None:
        """Snapshot the reset object pose: centre, world long axis, radius."""
        sim = self.env.sim.data
        num_envs = int(self.env.num_envs)
        center = sim.mocap_pos[:, self.mocap_idx, :].clone().to(self.device)
        quat = sim.mocap_quat[:, self.mocap_idx, :].clone().to(self.device)  # wxyz
        axis_local = torch.zeros((num_envs, 3), device=self.device)
        axis_local[:, self.long_axis_local] = 1.0
        axis = _wxyz_apply(quat, axis_local)  # (B,3) world-frame long axis
        self.object_center = center.to(torch.float32)
        self.orbit_axis_world = axis.to(torch.float32)
        self.offset0 = (self.fixed_target[:, :3] - center).to(torch.float32)
        self.rot0 = _rotvec_to_wxyz_quat(self.fixed_target[:, 3:])
        self._captured = True

    def step(
        self,
        episode_step: int | None = None,
        contact_ready: np.ndarray | torch.Tensor | None = None,
    ) -> bool:
        """Advance the gated triangle-wave phase; return whether any env moved."""
        if episode_step is None:
            raise RuntimeError("PalmOrbitController requires the episode step")
        if not self._captured:
            # Lazy capture so any reset ordering (env / policy / motion
            # controller) still snapshots the final object pose.
            self._capture()
        in_window = self.motion_start <= episode_step < (
            self.motion_start + self.motion_length
        )
        if not in_window:
            self.motion_active.zero_()
            self.current_x_des = self.fixed_target.clone()
            return False
        if contact_ready is None:
            self.motion_contact_ready.fill_(True)
        else:
            self.motion_contact_ready = torch.as_tensor(
                contact_ready, dtype=torch.bool, device=self.device
            ).reshape(self.env.num_envs)
        self.motion_active = self.motion_contact_ready.clone()
        if not bool(torch.any(self.motion_active)):
            # Hold the palm where it is instead of snapping back to phase 0.
            return False

        self.phase = self.phase + self.direction * self.angular_speeds * self.dt
        over = self.phase > self.amplitudes_rad
        under = self.phase < -self.amplitudes_rad
        self.phase = torch.where(
            over, 2.0 * self.amplitudes_rad - self.phase, self.phase
        )
        self.phase = torch.where(
            under, -2.0 * self.amplitudes_rad - self.phase, self.phase
        )
        self.direction = torch.where(over | under, -self.direction, self.direction)

        # Rodrigues: rotate offset0 about the world axis, then the orientation.
        theta = self.phase  # (B,)
        axis = self.orbit_axis_world  # (B,3)
        o = self.offset0  # (B,3)
        c = torch.cos(theta)
        s = torch.sin(theta)
        cross = torch.linalg.cross(axis, o, dim=-1)
        dot = (axis * o).sum(dim=-1, keepdim=True)
        rot_offset = (
            o * c.unsqueeze(-1)
            + cross * s.unsqueeze(-1)
            + axis * dot * (1.0 - c).unsqueeze(-1)
        )
        pos = self.object_center + rot_offset
        q_axis = torch.cat(
            (
                torch.cos(theta / 2).unsqueeze(-1),
                axis * torch.sin(theta / 2).unsqueeze(-1),
            ),
            dim=-1,
        )
        q_orbit = _wxyz_multiply(q_axis, self.rot0)
        rotvec = _wxyz_quat_to_rotvec(q_orbit)
        self.current_x_des = torch.cat((pos, rotvec), dim=-1).to(torch.float32)
        return True


class FacingCenterOrbitController:
    """One-way palm orbit: constant surface clearance, normal faces centre.

    Unlike the rigid-body orbit (``PalmOrbitController``), the palm position
    follows the object's elliptical cross-section at a constant distance from
    the surface::

        pos = C + n(theta) * (r_surf(n(theta)) + h)

    where ``n`` is the radial direction in the plane perpendicular to the
    long axis and ``r_surf`` is the ellipse radial distance along ``n``.  The
    orientation is the calibrated pose rotated about the long axis only, so
    the palm normal (which already points at the object centre in the
    calibrated pose) keeps pointing at the centre for every theta.  Phase
    advances one-way with no triangle-wave reflection: the palm never
    reverses, so a return sweep can never squeeze the fingers against the
    surface.  Fingers keep their static pregrasp targets and slide along the
    surface as the palm carries them.
    """

    def __init__(
        self,
        env: ManagerBasedRlEnv,
        fixed_target: torch.Tensor,
        target_mocap_idx: int,
        object_extent_local: np.ndarray,
        angular_speed_min: float,
        angular_speed_max: float,
        surface_clearance_m: float | None,
        motion_start: int,
        motion_length: int,
        dt: float,
        device: str,
    ) -> None:
        self.env = env
        self.fixed_target = fixed_target  # (B,6) world pos + rotvec, float32
        self.mocap_idx = int(target_mocap_idx)
        self.motion_start = int(motion_start)
        self.motion_length = int(motion_length)
        self.dt = float(dt)
        self.device = device
        self.surface_clearance_override = (
            None if surface_clearance_m is None else float(surface_clearance_m)
        )
        num_envs = int(env.num_envs)
        self.object_extent_local = np.asarray(
            object_extent_local, dtype=np.float64
        )
        extent = self.object_extent_local
        self.long_axis_local = int(np.argmax(np.abs(extent)))
        self.ellipse_axes_local = [
            i for i in range(3) if i != self.long_axis_local
        ]
        uniform = torch.rand(num_envs, device=device)
        self.angular_speeds = (
            angular_speed_min + (angular_speed_max - angular_speed_min) * uniform
        ).to(torch.float32)
        self.phase = torch.zeros(num_envs, device=device)
        self.current_x_des = self.fixed_target.clone()
        self.motion_active = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.motion_contact_ready = torch.zeros(
            num_envs, dtype=torch.bool, device=device
        )
        self.object_center = torch.zeros((num_envs, 3), device=device)
        self.orbit_axis_world = torch.zeros((num_envs, 3), device=device)
        self.n0 = torch.zeros((num_envs, 3), device=device)
        self.s1hat = torch.zeros(3, device=device)
        self.s2hat = torch.zeros(3, device=device)
        self.ellipse_a = torch.zeros(1, device=device)
        self.ellipse_b = torch.zeros(1, device=device)
        self.surface_clearance = torch.zeros(num_envs, device=device)
        self.theta0 = torch.zeros(num_envs, device=device)
        self.rot0 = torch.zeros((num_envs, 4), device=device)  # wxyz quat
        self._captured = False

    def reset(self) -> None:
        """Reset phase/window state; the object pose is snapshotted lazily."""
        self.phase.zero_()
        self.motion_active.zero_()
        self.motion_contact_ready.zero_()
        self._captured = False
        self.current_x_des = self.fixed_target.clone()

    def _ellipse_radius(self, n: torch.Tensor) -> torch.Tensor:
        """Ellipse radial distance along unit direction ``n`` in the cross-plane."""
        n1 = (n * self.s1hat).sum(dim=-1)
        n2 = (n * self.s2hat).sum(dim=-1)
        return self.ellipse_a * self.ellipse_b / torch.sqrt(
            (self.ellipse_b * n1) ** 2 + (self.ellipse_a * n2) ** 2
        )

    def _capture(self) -> None:
        """Snapshot object pose, cross-section ellipse, and the initial pose."""
        sim = self.env.sim.data
        num_envs = int(self.env.num_envs)
        center = sim.mocap_pos[:, self.mocap_idx, :].clone().to(self.device)
        quat = sim.mocap_quat[:, self.mocap_idx, :].clone().to(self.device)  # wxyz
        axis_local = torch.zeros((num_envs, 3), device=self.device)
        axis_local[:, self.long_axis_local] = 1.0
        axis = _wxyz_apply(quat, axis_local)  # (B,3) world long axis
        self.object_center = center.to(torch.float32)
        self.orbit_axis_world = axis.to(torch.float32)

        offset0 = self.fixed_target[:, :3] - center  # (B,3)
        # Radial part perpendicular to the long axis.
        offset_radial = offset0 - axis * (offset0 * axis).sum(
            dim=-1, keepdim=True
        )
        n0 = torch.nn.functional.normalize(offset_radial, dim=-1)
        self.n0 = n0.to(torch.float32)

        # Cross-section ellipse: the two non-long local axes of the object
        # AABB, rotated into the world frame (object rotation from mocap quat).
        e1_local = torch.zeros(3, device=self.device)
        e1_local[self.ellipse_axes_local[0]] = 1.0
        e2_local = torch.zeros(3, device=self.device)
        e2_local[self.ellipse_axes_local[1]] = 1.0
        e1 = _wxyz_apply(quat[0], e1_local)
        e2 = _wxyz_apply(quat[0], e2_local)
        # Half-axes from the caller-supplied local AABB (same for all envs).
        half1 = 0.5 * float(
            np.abs(self.object_extent_local[self.ellipse_axes_local[0]])
        )
        half2 = 0.5 * float(
            np.abs(self.object_extent_local[self.ellipse_axes_local[1]])
        )
        self.s1hat = e1.to(torch.float32)
        self.s2hat = e2.to(torch.float32)
        self.ellipse_a = torch.tensor([half1], device=self.device)
        self.ellipse_b = torch.tensor([half2], device=self.device)

        r0 = self._ellipse_radius(n0)
        if self.surface_clearance_override is not None:
            h = torch.full(
                (num_envs,), self.surface_clearance_override, device=self.device
            )
        else:
            h = offset_radial.norm(dim=-1) - r0
        self.surface_clearance = h.to(torch.float32)
        n1v = (n0 * self.s1hat).sum(dim=-1)
        n2v = (n0 * self.s2hat).sum(dim=-1)
        self.theta0 = torch.atan2(n2v, n1v)
        self.rot0 = _rotvec_to_wxyz_quat(self.fixed_target[:, 3:])
        self._captured = True

    def step(
        self,
        episode_step: int | None = None,
        contact_ready: np.ndarray | torch.Tensor | None = None,
    ) -> bool:
        """Advance the gated one-way phase; return whether any env moved."""
        if episode_step is None:
            raise RuntimeError("FacingCenterOrbitController requires the episode step")
        if not self._captured:
            self._capture()
        in_window = self.motion_start <= episode_step < (
            self.motion_start + self.motion_length
        )
        if not in_window:
            self.motion_active.zero_()
            self.current_x_des = self.fixed_target.clone()
            return False
        if contact_ready is None:
            self.motion_contact_ready.fill_(True)
        else:
            self.motion_contact_ready = torch.as_tensor(
                contact_ready, dtype=torch.bool, device=self.device
            ).reshape(self.env.num_envs)
        self.motion_active = self.motion_contact_ready.clone()
        if not bool(torch.any(self.motion_active)):
            # Hold the palm where it is instead of snapping back to phase 0.
            return False

        # One-way phase advance: no triangle-wave reflection.
        self.phase = self.phase + self.angular_speeds * self.dt
        dth = self.phase - self.theta0  # (B,)

        # Position: ellipse radial + constant surface clearance.
        axis = self.orbit_axis_world
        n = self.n0
        c = torch.cos(dth)
        s = torch.sin(dth)
        cross = torch.linalg.cross(axis, n, dim=-1)
        dot = (axis * n).sum(dim=-1, keepdim=True)
        n_theta = (
            n * c.unsqueeze(-1)
            + cross * s.unsqueeze(-1)
            + axis * dot * (1.0 - c).unsqueeze(-1)
        )
        r = self._ellipse_radius(n_theta)
        pos = self.object_center + n_theta * (r + self.surface_clearance).unsqueeze(-1)

        # Orientation: rotate the calibrated pose about the long axis only,
        # which keeps the palm normal pointing at the object centre.
        q_axis = torch.cat(
            (
                torch.cos(dth / 2).unsqueeze(-1),
                axis * torch.sin(dth / 2).unsqueeze(-1),
            ),
            dim=-1,
        )
        q_orbit = _wxyz_multiply(q_axis, self.rot0)
        rotvec = _wxyz_quat_to_rotvec(q_orbit)
        self.current_x_des = torch.cat((pos, rotvec), dim=-1).to(torch.float32)
        return True


class PalmOrbitFixedPregraspPolicy:
    """fixed_pregrasp + ``--motion-mode orbit_palm`` teacher.

    Object stays still; the palm target orbits the object's long axis on a
    contact gate (all four fingertips loaded).  Fingers keep their static
    pregrasp targets and passively comply through the soft finger servo.
    """

    def __init__(
        self,
        env: ManagerBasedRlEnv,
        base_policy,
        orbit: PalmOrbitController,
        contact_threshold: float,
    ) -> None:
        self.env = env
        self.base_policy = base_policy
        self.orbit = orbit
        self.contact_threshold = float(contact_threshold)
        self.step_count = 0
        self.orbit_moving = False
        self.ready_for_motion = True
        self.last_debug: dict[str, torch.Tensor] = {}

    def reset(self) -> None:
        self.base_policy.reset()
        self.orbit.reset()
        self.step_count = 0
        self.orbit_moving = False
        self.last_debug = {}

    def __call__(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        forces = _fingertip_force_world(self.env)  # (B,4,3)
        loaded = torch.linalg.vector_norm(forces, dim=-1) >= self.contact_threshold
        contact_ready = loaded.all(dim=-1)  # (B,)
        self.orbit_moving = self.orbit.step(
            self.step_count, contact_ready=contact_ready
        )
        self.step_count += 1
        action = self.base_policy(obs, x_des=self.orbit.current_x_des)
        travel = (
            self.orbit.phase - self.orbit.theta0
            if hasattr(self.orbit, "theta0")
            else self.orbit.phase
        )
        self.last_debug = {
            **self.base_policy.last_debug,
            "palm_x_des_orbit": self.orbit.current_x_des,
            "orbit_contact_ready": contact_ready.float(),
            "orbit_phase_rad": self.orbit.phase,
            "orbit_moving": self.orbit.motion_active.float(),
            "orbit_surface_clearance_m": getattr(
                self.orbit,
                "surface_clearance",
                torch.zeros_like(self.orbit.phase),
            ),
            "orbit_travel_total_rad": travel,
        }
        return action


class FullHandMCCCollectionPolicy:
    """Privileged FullHandMCC teacher used only while generating data.

    The existing fixed-world palm MCC is retained.  Its fixed-pregrasp finger
    branch is replaced by four independent surface targets from the geometry
    oracle, normal admittance, and the FullHandMCC four-site IK.  Once a pad
    first settles, its surface point is stored in object coordinates so rigid
    object motion is projected against an independent planner query instead of
    letting the current (possibly lagging) fingertip redefine its own target.
    """

    def __init__(
        self,
        env: ManagerBasedRlEnv,
        base_policy,
        object_config,
        target_mocap_idx: int,
        palm_idx: int,
        tip_indices: list[int],
        *,
        surface_preload_m: float,
        anchor_force_threshold: float,
        precontact_force_threshold: float,
        anchor_settle_frames: int,
        surface_target_mode: str,
        contact_search_step_m: float,
        contact_search_step_rad: float,
        contact_search_limit_rad: float,
        contact_transient_loss_frames: int,
        contact_recovery_confirm_frames: int,
        contact_transient_search_step_m: float,
        contact_transient_release_step_m: float,
        persistent_recovery_max_joint_step_rad: float,
        nominal_grasp_q: np.ndarray | tuple[float, ...],
        differential_contact_qp: bool,
        fixed_grasp_fingers: bool = False,
        object_scale: float = 1.0,
        mesh_normal_oracle=None,
        palm_reposition_max_m: float | None = None,
        palm_normal_ema_alpha: float = 0.15,
        palm_follow_max_position_step_m: float = 0.0015,
        palm_follow_max_orientation_step_rad: float = np.deg2rad(0.5),
        palm_follow_max_tilt_from_fixed_rad: float = np.deg2rad(20.0),
        palm_motion_preview_s: float = 0.75,
        palm_protection_update_decimation: int = 5,
        palm_standoff_extra_m: float = 0.0,
        hand_shape_degradation_threshold_rad: float = 0.75,
        hand_shape_retreat_gain_m_per_rad: float = 0.025,
        hand_shape_retreat_max_m: float = 0.035,
        finger_retreat_compensation_max_m: float | None = 0.055,
        enable_privileged_palm_follow: bool = True,
        shape_regularization: bool = False,
        force_servo_integral_gain: float = 0.0,
        fixed_grasp_nominal_weight: float = 0.80,
        initial_pad_max_angle_rad: float = np.deg2rad(55.0),
        closure_path_fallback_fraction: float = 0.70,
        closure_path_samples: int = 25,
    ) -> None:
        self.env = env
        # Articulated FK inside the palm MCC is evaluated in one canonical
        # environment, while scene bodies/contact queries are reported in
        # replicated world coordinates.  Keep the translation that maps each
        # canonical palm target into its scene world; without this, env>0
        # adds a world-space retreat to env0's fixed target and eventually
        # drives every parallel hand away from its object.
        self.env_origins_world = (
            env.scene.env_origins.detach().cpu().numpy().astype(np.float64)
        )
        self.base_policy = base_policy
        self.palm_controller = base_policy.palm_controller
        self.target_mocap_idx = int(target_mocap_idx)
        self.palm_idx = int(palm_idx)
        self.tip_indices = tuple(int(index) for index in tip_indices)
        self.surface_preload_m = float(surface_preload_m)
        self.anchor_force_threshold = float(anchor_force_threshold)
        self.precontact_force_threshold = float(precontact_force_threshold)
        self.anchor_settle_frames = max(1, int(anchor_settle_frames))
        if surface_target_mode not in ("nearest_surface", "object_anchor"):
            raise ValueError(f"Unknown surface target mode {surface_target_mode!r}")
        self.surface_target_mode = surface_target_mode
        self.enable_privileged_palm_follow = bool(
            enable_privileged_palm_follow
        )
        # No object-independent displacement cap by default.  A cap cannot
        # represent both a bottle body and a much more prominent bottle cap;
        # the required retreat is determined by measured/previewed clearance.
        self.palm_reposition_max_m = (
            None
            if palm_reposition_max_m is None
            else max(0.0, float(palm_reposition_max_m))
        )
        self.palm_normal_ema_alpha = float(
            np.clip(palm_normal_ema_alpha, 0.0, 1.0)
        )
        self.palm_follow_max_position_step_m = float(
            palm_follow_max_position_step_m
        )
        self.palm_follow_max_orientation_step_rad = float(
            palm_follow_max_orientation_step_rad
        )
        self.palm_follow_max_tilt_from_fixed_rad = float(
            palm_follow_max_tilt_from_fixed_rad
        )
        self.palm_motion_preview_s = max(0.0, float(palm_motion_preview_s))
        self.palm_protection_update_decimation = max(
            1, int(palm_protection_update_decimation)
        )
        self.motion_preview_controller: ObjectMotionController | None = None
        self.palm_preview_object_pose_world = np.zeros(
            (env.num_envs, 7), dtype=np.float64
        )
        self.palm_standoff_extra_m = max(0.0, float(palm_standoff_extra_m))
        self.hand_shape_degradation_threshold_rad = max(
            0.0, float(hand_shape_degradation_threshold_rad)
        )
        self.hand_shape_retreat_gain_m_per_rad = max(
            0.0, float(hand_shape_retreat_gain_m_per_rad)
        )
        self.hand_shape_retreat_max_m = max(
            0.0, float(hand_shape_retreat_max_m)
        )
        self.finger_retreat_compensation_max_m = (
            None
            if finger_retreat_compensation_max_m is None
            else max(0.0, float(finger_retreat_compensation_max_m))
        )
        self.hand_shape_deviation_rad = np.zeros(
            env.num_envs, dtype=np.float64
        )
        self.hand_shape_retreat_m = np.zeros(env.num_envs, dtype=np.float64)
        self.unloaded_streak = np.zeros(env.num_envs, dtype=np.int32)
        # Privileged predictive palm standoff guide.  Once the initial grasp
        # is stable, the palm-outline clearances are calibrated as one-sided
        # safety constraints and checked against current/preview object poses.
        self.palm_correction_world = np.zeros(
            (env.num_envs, 3), dtype=np.float64
        )
        self.palm_surface_normal_world = np.zeros(
            (env.num_envs, 3), dtype=np.float64
        )
        self.palm_surface_normal_valid = np.zeros(env.num_envs, dtype=bool)
        self.palm_surface_query_world = np.zeros(
            (env.num_envs, 3), dtype=np.float64
        )
        self.palm_surface_query_valid = np.zeros(env.num_envs, dtype=bool)
        self.palm_initial_surface_normal_world = np.zeros(
            (env.num_envs, 3), dtype=np.float64
        )
        self.palm_surface_standoff_m = np.zeros(env.num_envs, dtype=np.float64)
        self.palm_protection_standoff_m = np.zeros(
            (env.num_envs, len(PALM_PROTECTION_POINTS_LOCAL)), dtype=np.float64
        )
        self.palm_protection_clearance_m = np.zeros_like(
            self.palm_protection_standoff_m
        )
        self.palm_surface_standoff_valid = np.zeros(env.num_envs, dtype=bool)
        self.palm_predicted_clearance_m = np.zeros(
            env.num_envs, dtype=np.float64
        )
        self.palm_predicted_intrusion_m = np.zeros(
            env.num_envs, dtype=np.float64
        )
        self.palm_target_rotvec = np.tile(
            np.asarray(self.palm_controller.fixed_target_np[3:6], dtype=np.float64),
            (env.num_envs, 1),
        )
        self.contact_search_step_m = float(contact_search_step_m)
        self.contact_search_step_rad = float(contact_search_step_rad)
        self.contact_search_limit_rad = float(contact_search_limit_rad)
        self.differential_contact_qp = bool(differential_contact_qp)
        self.fixed_grasp_fingers = bool(fixed_grasp_fingers)
        self.fixed_grasp_nominal_weight = float(
            np.clip(fixed_grasp_nominal_weight, 0.0, 1.0)
        )
        self.initial_pad_max_angle_rad = max(
            0.0, float(initial_pad_max_angle_rad)
        )
        self.closure_path_fallback_fraction = float(
            np.clip(closure_path_fallback_fraction, 0.0, 1.0)
        )
        self.closure_path_samples = max(3, int(closure_path_samples))
        self.shape_regularization = bool(shape_regularization)
        nominal_grasp = np.asarray(nominal_grasp_q, dtype=np.float64)
        if nominal_grasp.shape != (16,) or not np.all(np.isfinite(nominal_grasp)):
            raise ValueError("nominal_grasp_q must contain 16 finite joint values")
        self.controllers = [
            FullHandMCCFingerController(
                FullHandMCCFingerConfig(
                    control_dt=0.01,
                    posture_cost=0.15,
                    natural_flexion_floor=-0.10,
                    desired_force_per_finger=(3.0, 3.0, 3.0, 4.0),
                    max_normal_offset=0.003,
                    thumb_max_inward_offset=0.006,
                    enable_loss_state_machine=True,
                    transient_loss_frames=contact_transient_loss_frames,
                    recovery_contact_confirm_frames=(
                        contact_recovery_confirm_frames
                    ),
                    transient_search_step=contact_transient_search_step_m,
                    transient_release_step=contact_transient_release_step_m,
                    persistent_recovery_max_joint_step=(
                        persistent_recovery_max_joint_step_rad
                    ),
                    # Collection learns q trajectories, not force targets.
                    # Preserve the FullHandMCC per-finger normal search but
                    # do not let transient mesh/contact impulses command an
                    # outward dropout.  Missing contact still advances by the
                    # configured search step until geometry is recovered.
                    force_servo_integral_gain=float(force_servo_integral_gain),
                    overforce_hard_ratio=1000.0,
                    thumb_overforce_hard_ratio=1000.0,
                    use_lateral_reference_regularizer=self.shape_regularization,
                    qp_lateral_reference_weight=(10.0 if self.shape_regularization else 5.0),
                    qp_lateral_reference_gain=(2.5 if self.shape_regularization else 2.5),
                    qp_min_adjacent_lateral_distance=(
                        0.025 if self.shape_regularization else 0.045
                    ),
                    # Contact points remain the primary task, but the
                    # planner mode should stay close to the loaded grasp
                    # instead of taking a new IK branch every frame.
                    qp_posture_weight=(0.030 if self.shape_regularization else 0.002),
                    qp_tangential_velocity_weight=(
                        1.0 if self.shape_regularization else 2.0
                    ),
                    qp_target_velocity_ema_alpha=(
                        0.15 if self.shape_regularization else 0.35
                    ),
                    # Pad attitude is corrected only by each finger's own
                    # side/opposition joint.  The main site task controls
                    # position, not a six-DoF pose that folds flexion joints.
                    tip_orientation_cost=0.0,
                    flexion_synergy_gain=(
                        0.18 if self.shape_regularization else 0.0
                    ),
                    flexion_synergy_hard_gain=(
                        0.75 if self.shape_regularization else 0.0
                    ),
                    flexion_synergy_spread_threshold=0.75,
                    flexion_synergy_max_step=0.025,
                    normal_synergy_control=self.shape_regularization,
                    normal_synergy_max_step=0.025,
                    nominal_surface_preload=(
                        0.003 if self.shape_regularization else 0.0
                    ),
                    grasp_closure_q=tuple(float(value) for value in nominal_grasp),
                )
            )
            for _ in range(env.num_envs)
        ]
        self.oracles = [
            GeometrySurfaceOracle(
                object_config,
                scale=float(object_scale),
                mesh_normal_oracle=mesh_normal_oracle,
            )
            for _ in range(env.num_envs)
        ]
        self.anchor_points_object = np.zeros((env.num_envs, 4, 3), dtype=np.float64)
        self.anchor_valid = np.zeros((env.num_envs, 4), dtype=bool)
        # The most recent measured pad contact is a better persistent-loss
        # recovery seed than the fixed MCC site.  Store it in object
        # coordinates so it follows the known teacher object motion exactly.
        self.last_contact_points_object = np.zeros(
            (env.num_envs, 4, 3), dtype=np.float64
        )
        self.last_contact_valid = np.zeros((env.num_envs, 4), dtype=bool)
        self.last_surface_points_object = np.zeros(
            (env.num_envs, 4, 3), dtype=np.float64
        )
        self.last_surface_normals_object = np.zeros_like(
            self.last_surface_points_object
        )
        self.last_surface_valid = np.zeros((env.num_envs, 4), dtype=bool)
        self.site_standoff_m = np.zeros((env.num_envs, 4), dtype=np.float64)
        self.loaded_streak = np.zeros((env.num_envs, 4), dtype=np.int32)
        self.contact_settle_streak = np.zeros(env.num_envs, dtype=np.int32)
        self.contact_calibrated = np.zeros(env.num_envs, dtype=bool)
        self.precontact_base_q = np.zeros((env.num_envs, 16), dtype=np.float64)
        self.precontact_base_valid = np.zeros(env.num_envs, dtype=bool)
        self.precontact_closure = np.zeros((env.num_envs, 16), dtype=np.float64)
        self.loaded_nominal_q = np.zeros((env.num_envs, 16), dtype=np.float64)
        self.loaded_nominal_valid = np.zeros(env.num_envs, dtype=bool)
        # When the object is stationary and a finger is stably loaded, freeze
        # that finger's natural q reference.  The high-frequency loop then
        # owns only pressure along the measured source-mesh normal, instead of
        # repeatedly selecting a new tangential target.
        self.static_contact_q_ref = np.zeros(
            (env.num_envs, 16), dtype=np.float64
        )
        self.static_contact_q_valid = np.zeros(
            (env.num_envs, 4), dtype=bool
        )
        self.hybrid_grasp_q_ref = np.zeros(
            (env.num_envs, 16), dtype=np.float64
        )
        self.hybrid_grasp_valid = np.zeros(
            (env.num_envs, 4), dtype=bool
        )
        self.previous_target_palm = np.zeros(
            (env.num_envs, 4, 3), dtype=np.float64
        )
        self.previous_target_valid = np.zeros(env.num_envs, dtype=bool)
        self.planner_query_world = np.zeros((env.num_envs, 4, 3), dtype=np.float64)
        self.planner_query_valid = np.zeros(env.num_envs, dtype=bool)
        self.last_debug: dict[str, torch.Tensor] = {}
        self._call_count = 0
        self.reset()

    @staticmethod
    def _world_to_object(
        points_world: np.ndarray,
        center_world: np.ndarray,
        rotation_world_from_object: np.ndarray,
    ) -> np.ndarray:
        return (
            rotation_world_from_object.T
            @ (np.asarray(points_world) - center_world).T
        ).T

    @staticmethod
    def _object_to_world(
        points_object: np.ndarray,
        center_world: np.ndarray,
        rotation_world_from_object: np.ndarray,
    ) -> np.ndarray:
        return center_world + (
            rotation_world_from_object @ np.asarray(points_object).T
        ).T

    def _closure_surface_targets(
        self,
        controller: FullHandMCCFingerController,
        oracle: GeometrySurfaceOracle,
        palm_pose_world: np.ndarray,
        *,
        samples: int = 25,
        fallback_fraction: float = 0.70,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Plan each finger along open->grasp and stop at its first surface hit.

        The palm path is untouched.  This only chooses a per-finger MCC
        posture/target for the current palm pose.  A missing intersection uses
        a conservative partial closure rather than forcing the complete grasp
        posture, which preserves recovery workspace.
        """
        q_open = controller.open_grasp_q
        q_close = controller.grasp_closure_q
        fractions = np.linspace(0.0, 1.0, max(3, int(samples)))
        points = np.zeros((4, 3), dtype=np.float64)
        normals = np.zeros((4, 3), dtype=np.float64)
        target_q = q_open.copy()
        hit = np.zeros(4, dtype=bool)
        chosen_fraction = np.full(4, float(np.clip(fallback_fraction, 0.0, 1.0)))
        previous_sd: np.ndarray | None = None
        previous_obs = None
        for fraction in fractions:
            q = q_open + float(fraction) * (q_close - q_open)
            tip_palm = controller.tip_positions_palm(q)
            tip_world = controller.points_palm_to_world(tip_palm, palm_pose_world)
            observation = oracle.observe(tip_world)
            sd = np.asarray(observation.signed_distance, dtype=np.float64)
            if previous_sd is None:
                # If the fully open endpoint is already inside the object,
                # closing farther is exactly the wrong recovery direction:
                # the object is pressing this finger from above.  Treat the
                # open endpoint as the earliest intersection so the nominal
                # posture actively unfolds instead of selecting 100% grasp.
                crossed = sd <= 0.0
            else:
                crossed = (previous_sd > 0.0) & (sd <= 0.0)
            for finger in np.flatnonzero(crossed & ~hit):
                hit[finger] = True
                chosen_fraction[finger] = float(fraction)
                points[finger] = observation.points_world[finger]
                normals[finger] = observation.normals_world[finger]
            previous_sd = sd
            previous_obs = observation
        # Fingers without a geometric crossing use the partial-closure pose.
        for finger in range(4):
            block = slice(4 * finger, 4 * finger + 4)
            target_q[block] = q_open[block] + chosen_fraction[finger] * (
                q_close[block] - q_open[block]
            )
        if previous_obs is None:
            raise RuntimeError("Closure path produced no surface observations")
        fallback_tip = controller.points_palm_to_world(
            controller.tip_positions_palm(q_open + float(np.clip(fallback_fraction, 0.0, 1.0)) * (q_close - q_open)),
            palm_pose_world,
        )
        fallback_obs = oracle.observe(fallback_tip)
        for finger in np.flatnonzero(~hit):
            # No surface intersection means there is no valid Cartesian
            # contact target on this closure ray.  Hold the conservative
            # partial-grasp endpoint itself; chasing its nearest mesh point
            # can fold the finger underneath the object.
            points[finger] = fallback_tip[finger]
            normals[finger] = fallback_obs.normals_world[finger]
        for finger in np.flatnonzero(hit):
            block = slice(4 * finger, 4 * finger + 4)
            target_q[block] = q_open[block] + chosen_fraction[finger] * (
                q_close[block] - q_open[block]
            )
        norms = np.linalg.norm(normals, axis=-1, keepdims=True)
        normals = normals / np.maximum(norms, 1.0e-12)
        return points, normals, target_q.reshape(16), hit

    def reset(self) -> None:
        self.palm_controller.reset()
        self.base_policy.finger_controller.reset()
        for controller in self.controllers:
            controller.reset()
        self.anchor_points_object.fill(0.0)
        self.anchor_valid.fill(False)
        self.last_contact_points_object.fill(0.0)
        self.last_contact_valid.fill(False)
        self.last_surface_points_object.fill(0.0)
        self.last_surface_normals_object.fill(0.0)
        self.last_surface_valid.fill(False)
        self.site_standoff_m.fill(0.0)
        self.loaded_streak.fill(0)
        self.contact_settle_streak.fill(0)
        self.contact_calibrated.fill(False)
        self.precontact_base_q.fill(0.0)
        self.precontact_base_valid.fill(False)
        self.precontact_closure.fill(0.0)
        self.loaded_nominal_q.fill(0.0)
        self.loaded_nominal_valid.fill(False)
        self.static_contact_q_ref.fill(0.0)
        self.static_contact_q_valid.fill(False)
        self.hybrid_grasp_q_ref.fill(0.0)
        self.hybrid_grasp_valid.fill(False)
        self.previous_target_palm.fill(0.0)
        self.previous_target_valid.fill(False)
        self.planner_query_world.fill(0.0)
        self.planner_query_valid.fill(False)
        self.palm_correction_world.fill(0.0)
        self.palm_surface_normal_world.fill(0.0)
        self.palm_surface_normal_valid.fill(False)
        self.palm_surface_query_world.fill(0.0)
        self.palm_surface_query_valid.fill(False)
        self.palm_initial_surface_normal_world.fill(0.0)
        self.palm_surface_standoff_m.fill(0.0)
        self.palm_protection_standoff_m.fill(0.0)
        self.palm_protection_clearance_m.fill(0.0)
        self.palm_surface_standoff_valid.fill(False)
        self.palm_predicted_clearance_m.fill(0.0)
        self.palm_predicted_intrusion_m.fill(0.0)
        self.hand_shape_deviation_rad.fill(0.0)
        self.hand_shape_retreat_m.fill(0.0)
        self.palm_target_rotvec[:] = np.asarray(
            self.palm_controller.fixed_target_np[3:6], dtype=np.float64
        )
        self.palm_preview_object_pose_world.fill(0.0)
        self.unloaded_streak.fill(0)

    def set_motion_preview_controller(
        self, controller: ObjectMotionController
    ) -> None:
        """Attach the privileged object trajectory generator used by collection."""
        self.motion_preview_controller = controller

    @staticmethod
    def _align_world_vectors(source: np.ndarray, target: np.ndarray) -> R:
        """Return the minimum world-frame rotation mapping source to target."""
        source = np.asarray(source, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        source /= max(float(np.linalg.norm(source)), 1.0e-12)
        target /= max(float(np.linalg.norm(target)), 1.0e-12)
        cosine = float(np.clip(source @ target, -1.0, 1.0))
        cross = np.cross(source, target)
        sine = float(np.linalg.norm(cross))
        if sine > 1.0e-10:
            return R.from_rotvec((cross / sine) * np.arctan2(sine, cosine))
        if cosine > 0.0:
            return R.identity()
        # Antiparallel vectors have infinitely many solutions.  Pick a stable
        # axis orthogonal to source instead of letting numerical noise choose.
        basis = np.array((1.0, 0.0, 0.0), dtype=np.float64)
        if abs(float(source @ basis)) > 0.9:
            basis = np.array((0.0, 1.0, 0.0), dtype=np.float64)
        axis = np.cross(source, basis)
        axis /= max(float(np.linalg.norm(axis)), 1.0e-12)
        return R.from_rotvec(np.pi * axis)

    @property
    def ready_for_motion(self) -> bool:
        return bool(np.all(self.motion_ready_mask))

    @property
    def motion_ready_mask(self) -> np.ndarray:
        """Per-env measured-contact gate, valid only after palm preparation."""
        stable_loaded = np.all(
            self.loaded_streak >= self.anchor_settle_frames, axis=1
        )
        return self.contact_calibrated & stable_loaded

    def _live_contacts(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        force_t = _fingertip_force_world(self.env)
        # Keep all four sensor reductions on the simulation device.  The old
        # implementation copied each finger's mask and position separately,
        # which introduced up to eight host synchronizations per control tick.
        found_columns: list[torch.Tensor] = []
        position_columns: list[torch.Tensor] = []
        for finger, site_name in enumerate(TIP_SITES):
            sensor_data = self.env.scene[f"{site_name}_contact"].data
            if sensor_data.found is None:
                found_columns.append(
                    torch.zeros(
                        self.env.num_envs,
                        dtype=torch.bool,
                        device=force_t.device,
                    )
                )
                position_columns.append(
                    torch.zeros(
                        (self.env.num_envs, 3),
                        dtype=torch.float32,
                        device=force_t.device,
                    )
                )
                continue
            finger_found = (
                sensor_data.found > 0
            ).any(dim=1)
            found_columns.append(finger_found)
            if sensor_data.pos is not None:
                position_columns.append(torch.where(
                    finger_found[:, None],
                    sensor_data.pos[:, 0],
                    torch.zeros_like(sensor_data.pos[:, 0]),
                ))
            else:
                position_columns.append(torch.zeros(
                        (self.env.num_envs, 3),
                        dtype=torch.float32,
                        device=force_t.device,
                    ))
        found_t = torch.stack(found_columns, dim=1)
        positions_t = torch.stack(position_columns, dim=1)
        return (
            force_t.detach().cpu().numpy(),
            found_t.detach().cpu().numpy(),
            positions_t.detach().cpu().numpy(),
        )

    def __call__(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        # The surface-follow state is updated at the end of each control call
        # and applied here on the next 10 ms tick.  This one-step delay avoids
        # coupling the palm IK and fingertip oracle algebraically.
        fixed = np.asarray(
            self.palm_controller.fixed_target_np, dtype=np.float64
        ).reshape(6)
        palm_x_des_np = np.tile(fixed, (self.env.num_envs, 1)).copy()
        palm_x_des_np[:, :3] += self.palm_correction_world
        palm_x_des_np[:, 3:6] = self.palm_target_rotvec
        palm_output = self.palm_controller(
            {"palm": obs["palm"]},
            x_des=torch.as_tensor(
                palm_x_des_np,
                device=self.env.device,
                dtype=torch.float32,
            ),
        )
        palm_debug = self.palm_controller.last_debug
        palm_in_prep = palm_debug["palm_in_prep"][:, 0] > 0.5
        palm_site_batch = (
            palm_debug["palm_site_pos"].detach().cpu().numpy().astype(np.float64)
        )
        palm_site_batch += self.env_origins_world

        robot = self.env.scene["robot"]
        q_hand_batch = robot.data.joint_pos[:, 6:22].detach().cpu().numpy()
        palm_pose_batch = robot.data.body_link_pose_w[:, self.palm_idx].detach().cpu().numpy()
        site_pose = robot.data.site_pose_w
        tip_world_batch = torch.stack(
            [site_pose[:, index, :3] for index in self.tip_indices], dim=1
        ).detach().cpu().numpy()
        object_pos_batch = self.env.sim.data.mocap_pos[:, self.target_mocap_idx].detach().cpu().numpy()
        object_quat_batch = self.env.sim.data.mocap_quat[:, self.target_mocap_idx].detach().cpu().numpy()
        preview_object_pos = object_pos_batch.copy()
        preview_object_quat = object_quat_batch.copy()
        if self.motion_preview_controller is not None:
            preview_pos_t, preview_quat_t = (
                self.motion_preview_controller.preview_pose(
                    self.palm_motion_preview_s
                )
            )
            preview_object_pos = preview_pos_t.detach().cpu().numpy()
            preview_object_quat = preview_quat_t.detach().cpu().numpy()
        self.palm_preview_object_pose_world[:, :3] = preview_object_pos
        self.palm_preview_object_pose_world[:, 3:7] = preview_object_quat
        force_batch, found_batch, contact_pos_batch = self._live_contacts()
        # The force tensor is transferred to host once by ``_live_contacts``;
        # reduce the complete batch in one NumPy call instead of recomputing a
        # norm for every environment in the controller loop.
        force_magnitude_batch = np.linalg.norm(force_batch, axis=-1)
        # Copy the preview motion state once per control tick.  Calling
        # ``.item()`` inside the per-environment loop forces a separate GPU
        # synchronization for every environment.
        if self.motion_preview_controller is not None:
            motion_active_batch = (
                self.motion_preview_controller.motion_active
                .detach()
                .to(device="cpu", dtype=torch.bool)
                .reshape(-1)
                .numpy()
            )
        else:
            motion_active_batch = np.zeros(self.env.num_envs, dtype=bool)

        q_command_batch = q_hand_batch.copy().astype(np.float32)
        tip_surface_world = np.zeros((self.env.num_envs, 4, 3), dtype=np.float32)
        tip_reference_world = np.zeros_like(tip_surface_world)
        tip_ik_world = np.zeros_like(tip_surface_world)
        tip_surface_palm = np.zeros_like(tip_surface_world)
        tip_reference_palm = np.zeros_like(tip_surface_world)
        normal_world_batch = np.zeros_like(tip_surface_world)
        closure_target_q_batch = np.zeros((self.env.num_envs, 16), dtype=np.float64)
        closure_hit_batch = np.zeros((self.env.num_envs, 4), dtype=bool)
        normal_force_batch = np.zeros((self.env.num_envs, 4), dtype=np.float32)
        normal_offset_batch = np.zeros_like(normal_force_batch)
        contact_phase_batch = np.zeros(
            (self.env.num_envs, 4), dtype=np.float32
        )
        contact_loss_streak_batch = np.zeros_like(contact_phase_batch)
        transient_search_offset_batch = np.zeros_like(contact_phase_batch)
        persistent_recovery_error_batch = np.zeros_like(contact_phase_batch)
        persistent_recovery_joint_step = np.zeros(
            (self.env.num_envs, 16), dtype=np.float32
        )
        persistent_recovery_target_world = np.zeros_like(tip_surface_world)
        persistent_recovery_control_point_world = np.zeros_like(
            tip_surface_world
        )
        sd_debug = np.zeros((self.env.num_envs, 4), dtype=np.float32)
        qp_joint_velocity = np.zeros(
            (self.env.num_envs, 16), dtype=np.float32
        )
        qp_target_tip_velocity = np.zeros_like(tip_surface_world)
        qp_normal_velocity_error = np.zeros(
            (self.env.num_envs, 4), dtype=np.float32
        )
        qp_adjacent_lateral_distance = np.zeros(
            (self.env.num_envs, 2), dtype=np.float32
        )
        qp_separation_active = np.zeros(
            (self.env.num_envs, 2), dtype=np.float32
        )
        qp_exit_flag = np.zeros(self.env.num_envs, dtype=np.float32)
        qp_solve_time_us = np.zeros(self.env.num_envs, dtype=np.float32)

        for env_id, (controller, oracle) in enumerate(
            zip(self.controllers, self.oracles, strict=True)
        ):
            oracle.set_pose(object_pos_batch[env_id], object_quat_batch[env_id])
            current_rotation = oracle.rotation_world_from_object
            current_center = oracle.center_world
            measured_contact = found_batch[env_id]
            if np.any(measured_contact):
                self.last_contact_points_object[
                    env_id, measured_contact
                ] = self._world_to_object(
                    contact_pos_batch[env_id, measured_contact],
                    current_center,
                    current_rotation,
                )
                self.last_contact_valid[env_id, measured_contact] = True
            magnitude = force_magnitude_batch[env_id]
            loaded = found_batch[env_id] & (
                magnitude >= self.anchor_force_threshold
            )
            object_is_moving = False
            if self.motion_preview_controller is not None:
                object_is_moving = bool(motion_active_batch[env_id])
            synergy_spread, synergy_residual = (
                controller.flexion_synergy_metrics(q_hand_batch[env_id])
            )
            # This is intentionally a large-error detector, not a demand for
            # perfect kinematic synergy.  It catches the visually degenerate
            # branch where one phalanx is folded while the others are open.
            shape_degenerate = (
                (synergy_spread > 2.0) | (synergy_residual > 0.35)
            )
            precontact_loaded = found_batch[env_id] & (
                magnitude >= self.precontact_force_threshold
            )
            if not self.contact_calibrated[env_id]:
                # During grasp construction, query the object from the
                # reference pregrasp rather than from an arbitrary reset/live
                # hand posture.  This makes the geometry oracle propose four
                # candidate patches for the intended grasp, not four nearest
                # points belonging to a folded or side-on hand.
                nominal_tip_palm = controller.tip_positions_palm(
                    controller.grasp_closure_q
                )
                query_world = controller.points_palm_to_world(
                    nominal_tip_palm, palm_pose_batch[env_id]
                )
            elif (
                self.surface_target_mode == "nearest_surface"
                and self.loaded_nominal_valid[env_id]
            ):
                # Keep the surface planner coordinated with palm retreat.  A
                # world-fixed query becomes unreachable as soon as the arm
                # backs away from a protruding cap.  Instead, evaluate the
                # four fingertips of the once-calibrated natural grasp in the
                # *current palm frame* and project those independent probes
                # onto the current object surface.  The oracle chooses the
                # surface points; the nominal grasp only resolves which
                # nearby branch each finger should use and prevents folding.
                nominal_tip_palm = controller.tip_positions_palm(
                    self.loaded_nominal_q[env_id]
                )
                query_world = controller.points_palm_to_world(
                    nominal_tip_palm,
                    palm_pose_batch[env_id],
                )
            else:
                query_world = (
                    self.planner_query_world[env_id]
                    if self.planner_query_valid[env_id]
                    else tip_world_batch[env_id]
                )
            # Predict the phase transition before the controller consumes
            # this frame.  A persistent finger will use its last valid
            # material surface point below, rather than trusting a new
            # nearest-mesh query after it has already left the surface.
            predicted_persistent = (
                ~loaded
                & (
                    controller.loss_streak + 1
                    > controller.config.transient_loss_frames
                )
            )
            nearest = oracle.observe(query_world)
            sd_debug[env_id] = nearest.signed_distance
            rotation = oracle.rotation_world_from_object
            center = oracle.center_world

            # Finger targets come from the open->fixed-grasp closure path,
            # not from an arbitrary nearest point at the current live q.
            closure_surface, closure_normals, closure_q, closure_hit = (
                self._closure_surface_targets(
                    controller,
                    oracle,
                    palm_pose_batch[env_id],
                    samples=self.closure_path_samples,
                    fallback_fraction=self.closure_path_fallback_fraction,
                )
            )
            # Apply the normal-facing correction to the selected closure
            # posture itself, so direct force-servo mode does not discard the
            # orientation objective when it builds q_cmd from nominal_q.
            closure_q = controller.orient_toward_surface_normals(
                closure_q,
                palm_pose_batch[env_id],
                closure_normals,
            ).astype(np.float64)
            closure_target_q_batch[env_id] = closure_q
            closure_hit_batch[env_id] = closure_hit

            if bool(palm_in_prep[env_id]):
                self.loaded_streak[env_id].fill(0)
            else:
                self.loaded_streak[env_id] = np.where(
                    loaded,
                    self.loaded_streak[env_id] + 1,
                    0,
                )

            # Stationary contact is a force-regulation problem, not a new
            # nearest-point planning problem every frame.  Freeze the healthy
            # loaded q branch.  Motion, persistent loss, or a folded hand
            # shape releases it immediately and returns ownership to the
            # closure-path planner.
            if object_is_moving:
                self.static_contact_q_valid[env_id].fill(False)
            else:
                release_static = predicted_persistent | shape_degenerate
                self.static_contact_q_valid[env_id, release_static] = False
                acquire_static = (
                    (self.loaded_streak[env_id] >= self.anchor_settle_frames)
                    & ~shape_degenerate
                    & ~self.static_contact_q_valid[env_id]
                )
                for finger in np.flatnonzero(acquire_static):
                    block = slice(4 * finger, 4 * finger + 4)
                    self.static_contact_q_ref[env_id, block] = (
                        q_hand_batch[env_id, block]
                    )
                    self.static_contact_q_valid[env_id, finger] = True

            # Capture each material surface anchor once.  Updating it every
            # frame would make a lagging tip define its own target and erase
            # the passive-following trajectory that collection is meant to
            # generate.
            new_anchor = (
                (self.loaded_streak[env_id] >= self.anchor_settle_frames)
                & ~self.anchor_valid[env_id]
            )
            if np.any(new_anchor):
                self.anchor_points_object[env_id, new_anchor] = self._world_to_object(
                    closure_surface[new_anchor], center, rotation
                )
                self.site_standoff_m[env_id, new_anchor] = 0.0
                self.anchor_valid[env_id, new_anchor] = True

            transported = closure_surface.astype(np.float64)
            valid = self.anchor_valid[env_id]
            if self.surface_target_mode == "object_anchor" and np.any(valid):
                transported[valid] = self._object_to_world(
                    self.anchor_points_object[env_id, valid], center, rotation
                )
            # ``transported`` is already the selected material target.  The
            # previous implementation queried the geometry here and then
            # immediately overwrote both returned arrays with the closure
            # targets.  Avoid that redundant nearest-mesh query; retaining
            # the closure arrays is numerically identical.
            surface_points = transported
            normals = closure_normals.astype(np.float64, copy=False)
            # Force direction comes from the undecomposed source mesh at the
            # measured 3-D contact, never from a V-HACD part or its seam.
            if np.any(loaded):
                normals[loaded] = oracle.normals_at_world(
                    contact_pos_batch[env_id, loaded]
                )
            static_valid = self.static_contact_q_valid[env_id]
            if np.any(static_valid):
                # No tangential target motion while the object is stopped.
                # The frozen joint posture rejects drift; MCC only adds its
                # scalar inward force correction along the current mesh normal.
                surface_points[static_valid] = tip_world_batch[
                    env_id, static_valid
                ]
                for finger in np.flatnonzero(static_valid):
                    block = slice(4 * finger, 4 * finger + 4)
                    closure_q[block] = self.static_contact_q_ref[
                        env_id, block
                    ]
                closure_target_q_batch[env_id] = closure_q
            # Do not freeze a persistently lost finger to its previous
            # material contact.  The current open->grasp intersection is the
            # authoritative target and may move *outward* as the object rolls
            # over the finger.  Reusing the old anchor was what kept the ring
            # finger curled while the bottle pressed it underneath.

            remember_surface = loaded & ~predicted_persistent
            if np.any(remember_surface):
                self.last_surface_points_object[
                    env_id, remember_surface
                ] = self._world_to_object(
                    surface_points[remember_surface], center, rotation
                )
                self.last_surface_normals_object[
                    env_id, remember_surface
                ] = (rotation.T @ normals[remember_surface].T).T
                self.last_surface_valid[env_id, remember_surface] = True
            # Only a real path/surface intersection receives preload.  With
            # no intersection, ``surface_points`` is the 60--80% fallback FK
            # endpoint and must remain a posture hold, not a nearest-surface
            # recovery command.
            preload = self.surface_preload_m * closure_hit.astype(np.float64)
            kinematic_targets = surface_points + (
                self.site_standoff_m[env_id, :, None] - preload[:, None]
            ) * normals
            current_target_palm = controller.points_world_to_palm(
                kinematic_targets, palm_pose_batch[env_id]
            ).astype(np.float64)
            if self.previous_target_valid[env_id]:
                target_velocity_palm = (
                    current_target_palm - self.previous_target_palm[env_id]
                ) / float(controller.config.control_dt)
            else:
                target_velocity_palm = np.zeros((4, 3), dtype=np.float64)
            target_velocity_palm[static_valid] = 0.0

            # Privileged predictive palm standoff.  Query the densely sampled
            # outline of the object-facing palm plane against current and
            # preview object poses.  The outline point with the largest
            # safety-distance violation drives the normal correction.
            if (
                self.enable_privileged_palm_follow
                and not bool(palm_in_prep[env_id])
                and self.contact_calibrated[env_id]
                and (
                    not self.palm_surface_standoff_valid[env_id]
                    or self._call_count
                    % self.palm_protection_update_decimation
                    == 0
                )
            ):
                protection_world = controller.points_palm_to_world(
                    PALM_PROTECTION_POINTS_LOCAL,
                    palm_pose_batch[env_id],
                )

                def protection_clearance(surface):
                    points = surface.points_world.astype(np.float64)
                    normals_local = surface.normals_world.astype(np.float64)
                    normal_norm = np.linalg.norm(
                        normals_local, axis=-1, keepdims=True
                    )
                    normals_local /= np.maximum(normal_norm, 1.0e-12)
                    vectors = protection_world - points
                    flip = np.einsum(
                        "ij,ij->i", normals_local, vectors
                    ) < 0.0
                    normals_local[flip] *= -1.0
                    clearances = np.einsum(
                        "ij,ij->i", normals_local, vectors
                    )
                    return clearances, normals_local

                current_surface = oracle.observe(protection_world)
                current_clearance, current_normals = protection_clearance(
                    current_surface
                )
                if not self.palm_surface_standoff_valid[env_id]:
                    # Calibrate one distance between the object and the palm
                    # *footprint*, not one unrelated distance per contour
                    # sample.  Preserving every point's initial clearance
                    # makes an initially remote point (e.g. 50 mm from the
                    # bottle) demand that same 50 mm when the cap later passes
                    # it, driving the palm far outside fingertip reach.  The
                    # global minimum is the actual initial collision margin.
                    footprint_standoff = max(
                        0.0, float(np.min(current_clearance))
                    )
                    self.palm_protection_standoff_m[env_id].fill(
                        footprint_standoff
                    )
                    selected = int(np.argmin(current_clearance))
                    predicted_clearances = current_clearance
                    selected_normals = current_normals
                    self.palm_surface_standoff_valid[env_id] = True
                else:
                    # Compare the current and previewed object at every palm
                    # landmark.  Each point independently keeps the smaller
                    # clearance, so a local cap near one palm edge cannot be
                    # hidden by the centre or the opposite edge moving away.
                    oracle.set_pose(
                        preview_object_pos[env_id],
                        preview_object_quat[env_id],
                    )
                    preview_surface = oracle.observe(protection_world)
                    preview_clearance, preview_normals = protection_clearance(
                        preview_surface
                    )
                    use_current = current_clearance <= preview_clearance
                    predicted_clearances = np.where(
                        use_current, current_clearance, preview_clearance
                    )
                    selected_normals = np.where(
                        use_current[:, None], current_normals, preview_normals
                    )
                    desired = (
                        self.palm_protection_standoff_m[env_id]
                        + self.palm_standoff_extra_m
                    )
                    selected = int(
                        np.argmax(desired - predicted_clearances)
                    )

                desired_clearances = (
                    self.palm_protection_standoff_m[env_id]
                    + self.palm_standoff_extra_m
                )
                self.palm_protection_clearance_m[env_id] = predicted_clearances
                query = protection_world[selected]
                raw_normal = selected_normals[selected]
                predicted_clearance = float(predicted_clearances[selected])
                desired_clearance = float(desired_clearances[selected])
                predicted_intrusion = max(
                    0.0, desired_clearance - predicted_clearance
                )
                self.palm_surface_query_world[env_id] = query
                self.palm_surface_query_valid[env_id] = True
                if self.palm_surface_standoff_m[env_id] == 0.0:
                    self.palm_initial_surface_normal_world[env_id] = raw_normal
                self.palm_surface_standoff_m[env_id] = float(
                    np.min(self.palm_protection_standoff_m[env_id])
                )

                self.palm_predicted_clearance_m[env_id] = predicted_clearance
                self.palm_predicted_intrusion_m[env_id] = predicted_intrusion
                if self.loaded_nominal_valid[env_id]:
                    shape_deviation = float(
                        np.max(
                            np.abs(
                                q_hand_batch[env_id]
                                - self.loaded_nominal_q[env_id]
                            )
                        )
                    )
                else:
                    shape_deviation = 0.0
                shape_retreat = float(
                    np.clip(
                        (
                            shape_deviation
                            - self.hand_shape_degradation_threshold_rad
                        )
                        * self.hand_shape_retreat_gain_m_per_rad,
                        0.0,
                        self.hand_shape_retreat_max_m,
                    )
                )
                self.hand_shape_deviation_rad[env_id] = shape_deviation
                self.hand_shape_retreat_m[env_id] = shape_retreat
                # Hand shape is diagnostic here, not a retreat velocity.  A
                # direct retreat proportional to shape error is positive
                # feedback: retreat increases fingertip reach error, which
                # increases shape error and can drive the palm away forever.
                if self.palm_surface_normal_valid[env_id]:
                    previous_normal = self.palm_surface_normal_world[env_id]
                    if float(previous_normal @ raw_normal) < 0.0:
                        raw_normal *= -1.0
                    alpha = self.palm_normal_ema_alpha
                    filtered_normal = (
                        (1.0 - alpha) * previous_normal + alpha * raw_normal
                    )
                    filtered_normal /= max(
                        float(np.linalg.norm(filtered_normal)), 1.0e-12
                    )
                else:
                    filtered_normal = raw_normal
                    self.palm_surface_normal_valid[env_id] = True
                self.palm_surface_normal_world[env_id] = filtered_normal

                fixed_position_world = (
                    fixed[:3] + self.env_origins_world[env_id]
                )
                previous_position_target = (
                    fixed_position_world + self.palm_correction_world[env_id]
                )
                # Coordinate palm retreat with the four independent finger
                # workspaces.  Around the current palm pose, signed distance
                # changes as n^T dp.  A three-variable damped least-squares
                # step therefore places the once-loaded natural-grasp tips
                # near their calibrated surface offsets.  Afterwards project
                # the step onto every active palm-outline safety half-space.
                # This is the small translation block of the contact-planning
                # optimization; it replaces the unstable serial scheme where
                # the palm retreated first and the fingers chased unreachable
                # world-fixed targets afterwards.
                tip_valid = self.anchor_valid[env_id]
                palm_twist = np.zeros(6, dtype=np.float64)
                palm_quaternion_wxyz = palm_pose_batch[env_id, 3:7]
                palm_rotation = R.from_quat(
                    np.roll(palm_quaternion_wxyz, -1)
                )
                palm_facing_normal = palm_rotation.apply(
                    np.array((0.0, 0.0, -1.0), dtype=np.float64)
                )
                if np.any(tip_valid):
                    tip_residual = (
                        nearest.signed_distance.astype(np.float64)
                        - (
                            self.site_standoff_m[env_id]
                            - self.surface_preload_m
                        )
                    )
                    # A switched mesh branch must not create a single-frame
                    # centimetre-scale palm command; fingers handle the
                    # remaining local shape residual.
                    tip_residual = np.clip(tip_residual, -0.020, 0.020)
                    tip_lever = (
                        query_world[tip_valid] - palm_site_batch[env_id]
                    )
                    tip_rotation_jacobian = np.cross(
                        tip_lever, normals[tip_valid]
                    )
                    tip_jacobian = np.concatenate(
                        (normals[tip_valid], tip_rotation_jacobian),
                        axis=1,
                    )
                    lhs = (
                        tip_jacobian.T @ tip_jacobian
                        + np.diag((0.20, 0.20, 0.20, 0.005, 0.005, 0.005))
                    )
                    rhs = -(tip_jacobian.T @ tip_residual[tip_valid])
                    palm_twist = np.linalg.solve(lhs, rhs)
                    # The task requires normal alignment but no arbitrary
                    # wrist yaw.  Remove the component about palm-Z before
                    # composing the candidate orientation.
                    palm_twist[3:] -= (
                        palm_twist[3:] @ palm_facing_normal
                    ) * palm_facing_normal

                # Sequential projection gives the previewed palm footprint
                # priority over the soft fingertip objective.  Several passes
                # handle non-parallel cap/body normals while retaining the
                # minimum-norm tip-compatible part of the step.
                required_clearance = (
                    desired_clearances - predicted_clearances
                )
                for _ in range(3):
                    for point, normal, required in zip(
                        protection_world,
                        selected_normals,
                        required_clearance,
                        strict=True,
                    ):
                        lever = point - palm_site_batch[env_id]
                        rotation_row = np.cross(lever, normal)
                        rotation_row -= (
                            rotation_row @ palm_facing_normal
                        ) * palm_facing_normal
                        constraint_row = np.concatenate((normal, rotation_row))
                        deficit = float(required - constraint_row @ palm_twist)
                        if deficit > 0.0:
                            palm_twist += (
                                deficit
                                * constraint_row
                                / max(
                                    float(constraint_row @ constraint_row),
                                    1.0e-9,
                                )
                            )
                palm_twist[3:] -= (
                    palm_twist[3:] @ palm_facing_normal
                ) * palm_facing_normal
                translation_step = palm_twist[:3]
                optimized_rotation_step = palm_twist[3:]
                raw_position_target = (
                    palm_site_batch[env_id] + translation_step
                )
                position_step = raw_position_target - previous_position_target
                position_step_norm = float(np.linalg.norm(position_step))
                if position_step_norm > self.palm_follow_max_position_step_m:
                    position_step *= (
                        self.palm_follow_max_position_step_m / position_step_norm
                    )
                next_position_target = previous_position_target + position_step
                correction = next_position_target - fixed_position_world
                correction_norm = float(np.linalg.norm(correction))
                if (
                    self.palm_reposition_max_m is not None
                    and correction_norm > self.palm_reposition_max_m
                ):
                    correction *= self.palm_reposition_max_m / correction_norm
                self.palm_correction_world[env_id] = correction

                fixed_rotation = R.from_rotvec(fixed[3:6])
                desired_rotation = (
                    R.from_rotvec(optimized_rotation_step) * palm_rotation
                )
                # Enforce the no-yaw requirement again on the absolute target
                # and bound only the accumulated palm-normal tilt.
                fixed_facing_normal = fixed_rotation.apply(
                    np.array((0.0, 0.0, -1.0), dtype=np.float64)
                )
                relative_tilt = (
                    desired_rotation * fixed_rotation.inv()
                ).as_rotvec()
                relative_tilt -= (
                    relative_tilt @ fixed_facing_normal
                ) * fixed_facing_normal
                tilt_norm = float(np.linalg.norm(relative_tilt))
                if tilt_norm > self.palm_follow_max_tilt_from_fixed_rad:
                    relative_tilt *= (
                        self.palm_follow_max_tilt_from_fixed_rad / tilt_norm
                    )
                desired_rotation = R.from_rotvec(relative_tilt) * fixed_rotation
                previous_rotation = R.from_rotvec(
                    self.palm_target_rotvec[env_id]
                )
                rotation_step = (
                    desired_rotation * previous_rotation.inv()
                ).as_rotvec()
                rotation_step_norm = float(np.linalg.norm(rotation_step))
                if rotation_step_norm > self.palm_follow_max_orientation_step_rad:
                    rotation_step *= (
                        self.palm_follow_max_orientation_step_rad
                        / rotation_step_norm
                    )
                self.palm_target_rotvec[env_id] = (
                    R.from_rotvec(rotation_step) * previous_rotation
                ).as_rotvec()
                # Fingertip targets below are defined on the actual current
                # object, never on the preview pose.
                oracle.set_pose(
                    object_pos_batch[env_id], object_quat_batch[env_id]
                )

            tip_surface_world[env_id] = surface_points
            normal_world_batch[env_id] = normals
            tip_surface_palm[env_id] = controller.points_world_to_palm(
                surface_points, palm_pose_batch[env_id]
            )
            recovery_debug: dict[str, np.ndarray] | None = None

            if bool(palm_in_prep[env_id]):
                controller.reset()
                self.precontact_base_q[env_id] = q_hand_batch[env_id]
                self.precontact_base_valid[env_id] = True
                self.precontact_closure[env_id].fill(0.0)
                self.contact_settle_streak[env_id] = 0
                self.contact_calibrated[env_id] = False
                self.loaded_nominal_valid[env_id] = False
                self.previous_target_valid[env_id] = False
                self.planner_query_valid[env_id] = False
                q_command = q_hand_batch[env_id]
                tip_reference = kinematic_targets
                tip_ik = tip_world_batch[env_id]
            elif not self.contact_calibrated[env_id]:
                # FullHandMCC first obtains a surface-consistent IK posture.
                # The per-finger normal search is only a correction around
                # this reachable planner posture, never a blind closure from
                # the arbitrary live/reset joint configuration.
                controller.calibrate_force_sign(
                    force_batch[env_id], found_batch[env_id], normals
                )
                # First orient each fingertip toward its own oracle normal.
                # This prevents a side/edge contact from being accepted merely
                # because the contact sensor reports a nonzero load.
                q_orientation = controller.orient_toward_surface_normals(
                    q_hand_batch[env_id],
                    palm_pose_batch[env_id],
                    normals,
                )
                q_surface, debug = controller.update(
                    q_live=q_orientation,
                    palm_pose_world=palm_pose_batch[env_id],
                    force_world=force_batch[env_id],
                    found=found_batch[env_id],
                    surface_points_world=kinematic_targets,
                    surface_normals_world=normals,
                    nominal_posture_q=closure_target_q_batch[env_id],
                    force_magnitude_only=True,
                    contact_points_world=contact_pos_batch[env_id],
                    use_contact_point_jacobian=False,
                    manage_contact_state=False,
                )
                pad_angle = controller.pad_normal_errors(
                    q_surface,
                    palm_pose_batch[env_id],
                    normals,
                )
                orientation_ok = pad_angle <= self.initial_pad_max_angle_rad
                settled = precontact_loaded & orientation_ok
                if bool(np.all(settled)):
                    self.contact_settle_streak[env_id] += 1
                else:
                    self.contact_settle_streak[env_id] = 0
                    delta = controller.normal_search_delta(
                        q_hand_batch[env_id],
                        palm_pose_batch[env_id],
                        normals,
                        ~settled,
                        inward_step=self.contact_search_step_m,
                        max_joint_step=self.contact_search_step_rad,
                    )
                    self.precontact_closure[env_id] = np.clip(
                        self.precontact_closure[env_id] + delta,
                        -self.contact_search_limit_rad,
                        self.contact_search_limit_rad,
                    )
                q_command = controller.clamp_joint_positions(
                    q_surface + self.precontact_closure[env_id]
                )
                if (
                    os.environ.get("MCC_DEBUG_SETTLE")
                    and self._call_count % 100 == 0
                ):
                    gaps = np.linalg.norm(
                        tip_world_batch[env_id] - kinematic_targets, axis=-1
                    )
                    print(
                        f"[SETTLE] env={env_id} step={self._call_count} "
                        f"gap={np.round(gaps, 3).tolist()} "
                        f"found={found_batch[env_id].tolist()} "
                        f"|f|={np.round(np.linalg.norm(force_batch[env_id], axis=-1), 2).tolist()} "
                        f"closure={np.round(self.precontact_closure[env_id], 3).tolist()}",
                        flush=True,
                    )
                    print(
                        f"  q_surface={np.round(q_surface, 3).tolist()}\n"
                        f"  upper={np.round(controller.upper, 3).tolist()}\n"
                        f"  tip={np.round(tip_world_batch[env_id], 3).tolist()}\n"
                        f"  tgt={np.round(kinematic_targets, 3).tolist()}",
                        flush=True,
                    )
                controller.previous_command = np.asarray(
                    q_command, dtype=np.float64
                ).copy()
                tip_reference = controller.points_palm_to_world(
                    debug["tip_reference_palm"], palm_pose_batch[env_id]
                )
                tip_ik_palm = controller.tip_positions_palm(q_command)
                tip_ik = controller.points_palm_to_world(
                    tip_ik_palm, palm_pose_batch[env_id]
                )
                normal_force_batch[env_id] = debug["normal_force"]
                normal_offset_batch[env_id] = debug["normal_offset"]
                if self.contact_settle_streak[env_id] >= self.anchor_settle_frames:
                    controller.calibrate_force_setpoint(
                        force_batch[env_id],
                        settled,
                        normals,
                        capture_measured=False,
                    )
                    self.contact_calibrated[env_id] = True
                    # The first stable, physically loaded grasp becomes the
                    # soft null-space reference.  It preserves a natural hand
                    # shape while still allowing fingertip motion on the
                    # surface; using the deformed live q every frame would
                    # make an IK fold self-reinforcing.
                    self.loaded_nominal_q[env_id] = q_hand_batch[env_id]
                    self.loaded_nominal_valid[env_id] = True
                    # Seed the hybrid reference from the planner's natural
                    # closure branch, not from a potentially side-loaded or
                    # already folded physical settling pose.
                    self.hybrid_grasp_q_ref[env_id] = (
                        closure_target_q_batch[env_id]
                    )
                    self.hybrid_grasp_valid[env_id].fill(True)
                    # FullHandMCC plans from the true loaded configuration,
                    # not from a future noisy live state.
                    self.planner_query_world[env_id] = tip_world_batch[env_id]
                    self.planner_query_valid[env_id] = True
            else:
                controller.calibrate_force_sign(
                    force_batch[env_id], found_batch[env_id], normals
                )
                # The per-finger closure-path solution is the nominal MCC
                # posture.  It is intentionally not replaced by the fully
                # closed grasp after contact: a finger with no geometric
                # intersection remains at the conservative 70% closure.
                planned_nominal_q = closure_target_q_batch[env_id].copy()
                # Healthy contact stays on a slowly evolving grasp branch.
                # Surface planning may reshape that branch while the object
                # moves, but it cannot overwrite the whole finger in one
                # frame.  Loss/degeneration bypasses this memory and receives
                # the freshly replanned closure posture immediately.
                nominal_q = planned_nominal_q.copy()
                for finger in range(4):
                    block = slice(4 * finger, 4 * finger + 4)
                    healthy = bool(loaded[finger] and not shape_degenerate[finger])
                    if not self.hybrid_grasp_valid[env_id, finger]:
                        self.hybrid_grasp_q_ref[env_id, block] = (
                            planned_nominal_q[block]
                        )
                        self.hybrid_grasp_valid[env_id, finger] = True
                    if healthy and object_is_moving:
                        # A 2% update at 100 Hz has a ~0.5 s time constant:
                        # fast enough for surface curvature, slow enough to
                        # preserve fixed-grasp stability and avoid branch hops.
                        alpha_grasp = 0.035
                        self.hybrid_grasp_q_ref[env_id, block] += (
                            alpha_grasp
                            * (
                                planned_nominal_q[block]
                                - self.hybrid_grasp_q_ref[env_id, block]
                            )
                        )
                    elif not healthy:
                        self.hybrid_grasp_q_ref[env_id, block] = (
                            planned_nominal_q[block]
                        )
                    nominal_q[block] = self.hybrid_grasp_q_ref[env_id, block]
                predicted_targets = kinematic_targets
                if (
                    self.differential_contact_qp
                    and self.previous_target_valid[env_id]
                    and not self.shape_regularization
                ):
                    qp_nominal, qp_debug = controller.solve_contact_velocity_qp(
                        q_live=q_hand_batch[env_id],
                        target_velocity_palm=target_velocity_palm,
                        surface_normals_palm=controller.vectors_world_to_palm(
                            normals, palm_pose_batch[env_id]
                        ),
                        nominal_posture_q=nominal_q,
                    )
                    nominal_q = qp_nominal
                    # The geometric target is advanced by the same short
                    # horizon as the QP.  MCC then handles only force/contact
                    # residuals instead of reacting one frame after the
                    # moving surface has already escaped.
                    predicted_target_palm = current_target_palm + (
                        controller.config.control_dt
                        * controller.config.qp_lookahead_steps
                        * qp_debug["target_tip_velocity_palm"]
                    )
                    predicted_targets = controller.points_palm_to_world(
                        predicted_target_palm, palm_pose_batch[env_id]
                    )
                    qp_joint_velocity[env_id] = qp_debug["joint_velocity"]
                    qp_target_tip_velocity[env_id] = qp_debug[
                        "target_tip_velocity_palm"
                    ]
                    qp_normal_velocity_error[env_id] = qp_debug[
                        "normal_velocity_error"
                    ]
                    qp_adjacent_lateral_distance[env_id] = qp_debug[
                        "adjacent_lateral_distance"
                    ]
                    qp_separation_active[env_id] = qp_debug[
                        "separation_active"
                    ]
                    qp_exit_flag[env_id] = float(qp_debug["exit_flag"])
                    qp_solve_time_us[env_id] = (
                        1.0e6 * float(qp_debug["solve_time_s"])
                    )
                # Keep the physical pad point attached to its distal body.
                # The ordinary QP/MCC path remains unchanged while contact is
                # healthy; persistent recovery below uses this point and its
                # own per-finger 3x4 Jacobian instead of the fixed MCC site.
                controller.update_contact_point_anchors(
                    q_hand_batch[env_id],
                    palm_pose_batch[env_id],
                    contact_pos_batch[env_id],
                    # A geometry flag can be true while the reduced contact
                    # position is still its zero placeholder.  Only a loaded
                    # contact may update the persistent-recovery pad anchor;
                    # the controller additionally checks spatial plausibility.
                    loaded,
                )
                q_command, debug = controller.update(
                    q_live=q_hand_batch[env_id],
                    palm_pose_world=palm_pose_batch[env_id],
                    force_world=force_batch[env_id],
                    found=found_batch[env_id],
                    surface_points_world=predicted_targets,
                    surface_normals_world=normals,
                    nominal_posture_q=nominal_q,
                    force_magnitude_only=True,
                    contact_points_world=contact_pos_batch[env_id],
                    # Healthy fingers receive only a scalar normal correction
                    # around the hybrid grasp.  Persistent loss below is the
                    # only mode allowed to invoke full 3-D surface recovery.
                    use_contact_point_jacobian=True,
                    manage_contact_state=True,
                    contact_observed=loaded,
                )
                persistent_loss = np.asarray(
                    debug["persistent_loss"], dtype=bool
                ) & closure_hit
                if np.any(persistent_loss):
                    q_command, recovery_debug = (
                        controller.recover_surface_contacts(
                            q_live=q_hand_batch[env_id],
                            base_command_q=q_command,
                            palm_pose_world=palm_pose_batch[env_id],
                            surface_points_world=surface_points,
                            surface_normals_world=normals,
                            persistent_loss=persistent_loss,
                            surface_preload_m=self.surface_preload_m,
                            nominal_posture_q=nominal_q,
                        )
                    )
                    # The recovery layer has priority for lost fingers.  Keep
                    # the command filter continuous from the command actually
                    # sent, rather than letting the next MCC frame jump back
                    # to its pre-recovery value.
                    controller.previous_command = np.asarray(
                        q_command, dtype=np.float64
                    ).copy()
                tip_reference = controller.points_palm_to_world(
                    debug["tip_reference_palm"], palm_pose_batch[env_id]
                )
                tip_ik = controller.points_palm_to_world(
                    controller.tip_positions_palm(q_command),
                    palm_pose_batch[env_id],
                )
                normal_force_batch[env_id] = debug["normal_force"]
                normal_offset_batch[env_id] = debug["normal_offset"]
            contact_phase_batch[env_id] = controller.contact_phase
            contact_loss_streak_batch[env_id] = controller.loss_streak
            transient_search_offset_batch[env_id] = (
                controller.transient_search_offset
            )
            if recovery_debug is not None:
                persistent_recovery_error_batch[env_id] = recovery_debug[
                    "recovery_error_norm"
                ]
                persistent_recovery_joint_step[env_id] = recovery_debug[
                    "recovery_joint_step"
                ]
                persistent_recovery_target_world[env_id] = (
                    controller.points_palm_to_world(
                        recovery_debug["recovery_target_palm"],
                        palm_pose_batch[env_id],
                    )
                )
                persistent_recovery_control_point_world[env_id] = (
                    controller.points_palm_to_world(
                        recovery_debug["recovery_control_point_palm"],
                        palm_pose_batch[env_id],
                    )
                )
            # Store the unpredicted current target.  Finite differences on
            # this signal include both object motion and the small palm
            # correction, i.e. exactly the relative motion each fingertip
            # must realize in the next control period.
            self.previous_target_palm[env_id] = current_target_palm
            self.previous_target_valid[env_id] = not bool(
                palm_in_prep[env_id]
            )
            q_command_batch[env_id] = q_command
            tip_reference_world[env_id] = tip_reference
            tip_ik_world[env_id] = tip_ik
            tip_reference_palm[env_id] = controller.points_world_to_palm(
                tip_reference, palm_pose_batch[env_id]
            )

        q_command_t = torch.as_tensor(
            q_command_batch, device=self.env.device, dtype=torch.float32
        )
        # TEMP DEBUG: watch the pre-contact search on the collection console.
        self._call_count += 1
        if self._call_count % 100 == 0:
            loaded_counts = found_batch.sum(axis=1)
            print(
                f"[FULLHAND] call={self._call_count} "
                f"loaded={loaded_counts.tolist()} "
                f"calibrated={self.contact_calibrated.tolist()} "
                f"sd={np.round(sd_debug * 1000, 1).tolist()} "
                f"palm_corr={np.round(self.palm_correction_world * 1000, 1).tolist()} "
                f"palm_intrusion_mm={np.round(self.palm_predicted_intrusion_m * 1000, 1).tolist()} "
                f"shape_dev={np.round(self.hand_shape_deviation_rad, 2).tolist()} "
                f"shape_retreat_mm={np.round(self.hand_shape_retreat_m * 1000, 1).tolist()}"
            )
        q_hand_t = robot.data.joint_pos[:, 6:22]
        fixed_finger_debug: dict[str, torch.Tensor] | None = None
        if self.fixed_grasp_fingers:
            # Do not carry the fixed palm-frame tip targets away with an arm
            # retreat.  Measure how much of the commanded retreat the arm has
            # actually achieved, then translate all four tip targets by the
            # opposite amount in the palm frame.  The four independent finger
            # Jacobians convert this common Cartesian extension into their own
            # joint motions; no shared/thumb-invalid Jacobian is assumed.
            tip_offset_local = np.zeros(
                (self.env.num_envs, 3), dtype=np.float64
            )
            for env_id, controller in enumerate(self.controllers):
                outward_normal = self.palm_surface_normal_world[env_id]
                normal_norm = float(np.linalg.norm(outward_normal))
                if normal_norm <= 1.0e-9:
                    continue
                outward_normal = outward_normal / normal_norm
                fixed_position_world = (
                    fixed[:3] + self.env_origins_world[env_id]
                )
                actual_displacement = (
                    palm_site_batch[env_id] - fixed_position_world
                )
                achieved_retreat = max(
                    0.0, float(actual_displacement @ outward_normal)
                )
                if self.finger_retreat_compensation_max_m is not None:
                    achieved_retreat = min(
                        achieved_retreat,
                        self.finger_retreat_compensation_max_m,
                    )
                extension_world = -achieved_retreat * outward_normal
                tip_offset_local[env_id] = controller.vectors_world_to_palm(
                    extension_world[None, :], palm_pose_batch[env_id]
                )[0]
            self.base_policy.finger_controller.set_tip_target_offset_local(
                tip_offset_local
            )
            if bool(torch.all(palm_in_prep)):
                self.base_policy.finger_controller.reset()
                finger_action = self.base_policy.finger_controller(
                    {"policy": obs["finger"]}
                )
                finger_action.zero_()
                self.base_policy.finger_controller.reset()
            else:
                finger_action = self.base_policy.finger_controller(
                    {"policy": obs["finger"]}
                )
                finger_action = torch.where(
                    palm_in_prep.unsqueeze(-1),
                    torch.zeros_like(finger_action),
                    finger_action,
                )
            fixed_finger_debug = self.base_policy.finger_controller.last_debug
            # Hybrid fixed-grasp mode: the nominal grasp owns most of the
            # command, while the FullHand contact planner retains a small
            # Cartesian correction authority.  This lets an outlying finger
            # (e.g. ring finger on the flat mustard bottle) reach its planned
            # surface point without allowing all fingers to chase noisy
            # surface targets every frame.
            q_grasp = fixed_finger_debug["q_ref"]
            q_contact = q_command_t
            w_grasp = self.fixed_grasp_nominal_weight
            q_blended = w_grasp * q_grasp + (1.0 - w_grasp) * q_contact
            q_hand_now = robot.data.joint_pos[:, 6:22]
            action_cmd = torch.clamp((q_blended - q_hand_now) / 0.08, -1.0, 1.0)
            previous = self.base_policy.finger_controller.prev_action
            delta = torch.clamp(
                action_cmd - previous,
                -self.base_policy.finger_controller.action_rate_limit,
                self.base_policy.finger_controller.action_rate_limit,
            )
            finger_action = previous + delta
            self.base_policy.finger_controller.prev_action = finger_action.detach().clone()
            fixed_finger_debug["q_ref_blended"] = q_blended
            fixed_finger_debug["q_ref_contact_plan"] = q_contact
        else:
            finger_action = torch.clamp(
                (q_command_t - q_hand_t) / 0.08, -1.0, 1.0
            )
        action = torch.cat((palm_output[:, :6], finger_action), dim=-1)

        q_pre_debug = torch.as_tensor(
            np.stack(
                [controller.grasp_closure_q for controller in self.controllers]
            ),
            device=self.env.device,
            dtype=torch.float32,
        )
        tip_x_des_debug = torch.as_tensor(
            tip_surface_world, device=self.env.device
        )
        tip_x_ref_debug = torch.as_tensor(
            tip_reference_world, device=self.env.device
        )
        tip_x_ik_debug = torch.as_tensor(
            tip_ik_world, device=self.env.device
        )
        tip_x_des_palm_debug = torch.as_tensor(
            tip_surface_palm, device=self.env.device
        )
        tip_x_ref_palm_debug = torch.as_tensor(
            tip_reference_palm, device=self.env.device
        )
        if fixed_finger_debug is not None:
            q_pre_debug = fixed_finger_debug["q_pre"]
            tip_x_des_debug = fixed_finger_debug["tip_x_des"]
            tip_x_ref_debug = fixed_finger_debug["tip_x_ref"]
            tip_x_ik_debug = fixed_finger_debug["tip_x_ik"]
            tip_x_des_palm_debug = fixed_finger_debug["tip_x_des_palm"]
            tip_x_ref_palm_debug = fixed_finger_debug["tip_x_ref_palm"]

        self.last_debug = {
            "q_pre": q_pre_debug,
            "q_ref": q_command_t,
            "tip_x_des": tip_x_des_debug,
            "tip_x_ref": tip_x_ref_debug,
            "tip_x_ik": tip_x_ik_debug,
            "tip_x_des_palm": tip_x_des_palm_debug,
            "tip_x_ref_palm": tip_x_ref_palm_debug,
            "tip_surface_normal_world": torch.as_tensor(normal_world_batch, device=self.env.device),
            "tip_normal_force": torch.as_tensor(normal_force_batch, device=self.env.device),
            "tip_normal_offset": torch.as_tensor(normal_offset_batch, device=self.env.device),
            "tip_contact_phase": torch.as_tensor(
                contact_phase_batch, device=self.env.device
            ),
            "tip_contact_loss_streak": torch.as_tensor(
                contact_loss_streak_batch, device=self.env.device
            ),
            "tip_transient_search_offset": torch.as_tensor(
                transient_search_offset_batch, device=self.env.device
            ),
            "tip_persistent_recovery_error": torch.as_tensor(
                persistent_recovery_error_batch, device=self.env.device
            ),
            "tip_persistent_recovery_joint_step": torch.as_tensor(
                persistent_recovery_joint_step, device=self.env.device
            ),
            "tip_persistent_recovery_target_world": torch.as_tensor(
                persistent_recovery_target_world, device=self.env.device
            ),
            "tip_persistent_recovery_control_point_world": torch.as_tensor(
                persistent_recovery_control_point_world,
                device=self.env.device,
            ),
            "tip_anchor_valid": torch.as_tensor(self.anchor_valid, device=self.env.device),
            "fullhand_contact_calibrated": torch.as_tensor(
                self.contact_calibrated, device=self.env.device
            ),
            "contact_qp_joint_velocity": torch.as_tensor(
                qp_joint_velocity, device=self.env.device
            ),
            "contact_qp_target_tip_velocity_palm": torch.as_tensor(
                qp_target_tip_velocity, device=self.env.device
            ),
            "contact_qp_normal_velocity_error": torch.as_tensor(
                qp_normal_velocity_error, device=self.env.device
            ),
            "contact_qp_adjacent_lateral_distance": torch.as_tensor(
                qp_adjacent_lateral_distance, device=self.env.device
            ),
            "contact_qp_separation_active": torch.as_tensor(
                qp_separation_active, device=self.env.device
            ),
            "contact_qp_exit_flag": torch.as_tensor(
                qp_exit_flag, device=self.env.device
            ),
            "contact_qp_solve_time_us": torch.as_tensor(
                qp_solve_time_us, device=self.env.device
            ),
            "palm_surface_normal_world": torch.as_tensor(
                self.palm_surface_normal_world,
                device=self.env.device,
                dtype=torch.float32,
            ),
            "palm_surface_standoff_m": torch.as_tensor(
                self.palm_surface_standoff_m,
                device=self.env.device,
                dtype=torch.float32,
            ),
            "palm_protection_standoff_m": torch.as_tensor(
                self.palm_protection_standoff_m,
                device=self.env.device,
                dtype=torch.float32,
            ),
            "palm_protection_clearance_m": torch.as_tensor(
                self.palm_protection_clearance_m,
                device=self.env.device,
                dtype=torch.float32,
            ),
            "palm_predicted_clearance_m": torch.as_tensor(
                self.palm_predicted_clearance_m,
                device=self.env.device,
                dtype=torch.float32,
            ),
            "palm_predicted_intrusion_m": torch.as_tensor(
                self.palm_predicted_intrusion_m,
                device=self.env.device,
                dtype=torch.float32,
            ),
            "hand_shape_deviation_rad": torch.as_tensor(
                self.hand_shape_deviation_rad,
                device=self.env.device,
                dtype=torch.float32,
            ),
            "hand_shape_retreat_m": torch.as_tensor(
                self.hand_shape_retreat_m,
                device=self.env.device,
                dtype=torch.float32,
            ),
            "palm_surface_follow_valid": torch.as_tensor(
                self.palm_surface_standoff_valid,
                device=self.env.device,
            ),
            "palm_surface_query_world": torch.as_tensor(
                self.palm_surface_query_world,
                device=self.env.device,
                dtype=torch.float32,
            ),
            "palm_preview_object_pose_world": torch.as_tensor(
                self.palm_preview_object_pose_world,
                device=self.env.device,
                dtype=torch.float32,
            ),
            **palm_debug,
            # Override the canonical-model value with the replicated scene
            # world position used by every other ``*_world`` field.
            "palm_site_pos": torch.as_tensor(
                palm_site_batch,
                device=self.env.device,
                dtype=torch.float32,
            ),
            "fixed_palm_target": self.palm_controller.fixed_target,
        }
        return action


class StreamingH5:
    SHAPES = {
        "time": (),
        "episode_id": (),
        "episode_step": (),
        "record_step": (),
        "actual_motion_start_step": (),
        "actual_record_start_step": (),
        "q": (22,),
        "qvel": (22,),
        "q_hand": (16,),
        "q_pre": (16,),
        "q_ref": (16,),
        "arm_q_ref": (6,),
        "fixed_palm_target": (6,),
        "palm_control_pos_world": (3,),
        "action": (22,),
        "object_pose_world": (7,),
        "palm_pose_world": (7,),
        "fingertip_pose_world": (4, 7),
        "fingertip_force_world": (4, 3),
        "fingertip_force_local": (4, 3),
        "fingertip_contact_pos_world": (4, 3),
        "fingertip_contact_normal_world": (4, 3),
        "fingertip_contact_dist": (4,),
        "fingertip_collision_found": (4,),
        "fingertip_contact": (4,),
        "tip_x_des_world": (4, 3),
        "tip_x_ref_world": (4, 3),
        "tip_x_ik_world": (4, 3),
        "tip_x_des_palm": (4, 3),
        "tip_x_ref_palm": (4, 3),
        "oracle_surface_normal_world": (4, 3),
        "fullhand_normal_force": (4,),
        "fullhand_normal_offset": (4,),
        "fullhand_contact_phase": (4,),
        "fullhand_contact_loss_streak": (4,),
        "fullhand_transient_search_offset": (4,),
        "fullhand_persistent_recovery_error": (4,),
        "fullhand_persistent_recovery_joint_step": (16,),
        "fullhand_persistent_recovery_target_world": (4, 3),
        "fullhand_persistent_recovery_control_point_world": (4, 3),
        "fullhand_anchor_valid": (4,),
        "fullhand_contact_calibrated": (),
        "contact_qp_joint_velocity": (16,),
        "contact_qp_target_tip_velocity_palm": (4, 3),
        "contact_qp_normal_velocity_error": (4,),
        "contact_qp_adjacent_lateral_distance": (2,),
        "contact_qp_separation_active": (2,),
        "contact_qp_exit_flag": (),
        "contact_qp_solve_time_us": (),
        "palm_surface_normal_world": (3,),
        "palm_surface_standoff_m": (),
        "palm_protection_standoff_m": (len(PALM_PROTECTION_POINTS_LOCAL),),
        "palm_protection_clearance_m": (len(PALM_PROTECTION_POINTS_LOCAL),),
        "palm_predicted_clearance_m": (),
        "palm_predicted_intrusion_m": (),
        "hand_shape_deviation_rad": (),
        "hand_shape_retreat_m": (),
        "palm_surface_follow_valid": (),
        "palm_surface_query_world": (3,),
        "palm_preview_object_pose_world": (7,),
        "palm_x_des": (6,),
        "object_angular_velocity_world": (3,),
        "object_rotation_axis_local": (3,),
        "object_rotation_speed_target_rad_s": (),
        "object_translation_axis_world": (3,),
        "object_translation_amplitude_m": (),
        "object_translation_speed_target_m_s": (),
        "object_horizontal_lowest_point_world": (3,),
        "object_lowest_point_anchor_world": (3,),
        "object_lowest_point_compensation_world": (3,),
        "object_lowest_point_follow_velocity_world": (3,),
        "object_motion_active": (),
        "object_motion_contact_ready": (),
        "object_motion_schedule_step": (),
        "object_segment_move_steps": (),
        "object_segment_hold_steps": (),
        "palm_x_des_orbit": (6,),
        "palm_x_ref": (6,),
        "orbit_phase_rad": (),
        "orbit_axis_world": (3,),
        "orbit_amplitude_rad": (),
        "orbit_speed_target_rad_s": (),
        "orbit_moving": (),
        "orbit_surface_clearance_m": (),
        "orbit_travel_total_rad": (),
    }

    def __init__(self, path: Path, num_envs: int, q_dof: int = 22):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = h5py.File(path, "w")
        self.num_envs = num_envs
        self.step = 0
        self.datasets: dict[str, h5py.Dataset] = {}
        # q/qvel are 23-D in palm-free mode (free palm 7 + hand 16).
        self._shapes = dict(self.SHAPES)
        self._shapes["q"] = (q_dof,)
        self._shapes["qvel"] = (q_dof,)
        for name, tail in self._shapes.items():
            dtype = (
                "i4"
                if name
                in (
                    "episode_id",
                    "episode_step",
                    "record_step",
                    "actual_motion_start_step",
                    "actual_record_start_step",
                )
                else "f4"
            )
            shape = (0, num_envs, *tail)
            maxshape = (None, num_envs, *tail)
            chunks = (min(100, 2500), num_envs, *tail)
            self.datasets[name] = self.file.create_dataset(
                name, shape=shape, maxshape=maxshape, chunks=chunks, dtype=dtype
            )

    def append(self, values: dict[str, np.ndarray | torch.Tensor | float | int]) -> None:
        next_size = self.step + 1
        for dataset in self.datasets.values():
            dataset.resize((next_size, *dataset.shape[1:]))
        for name, value in values.items():
            if torch.is_tensor(value):
                value = value.detach().cpu().numpy()
            array = np.asarray(value)
            if array.ndim == len(self._shapes[name]):
                array = np.broadcast_to(array, (self.num_envs, *array.shape))
            self.datasets[name][self.step] = array
        self.step = next_size

    def truncate(self, size: int) -> None:
        """Roll back a rejected candidate trajectory."""
        if size < 0 or size > self.step:
            raise ValueError(f"Invalid truncate size {size} for current size {self.step}")
        for dataset in self.datasets.values():
            dataset.resize((size, *dataset.shape[1:]))
        self.step = size

    def close(self, attrs: dict[str, object]) -> None:
        for key, value in attrs.items():
            self.file.attrs[key] = value
        self.file.attrs["total_steps"] = self.step
        self.file.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect MCC fingertip position-teacher data.")
    parser.add_argument(
        "--viewer",
        choices=("headless", "native", "viser"),
        default="headless",
        help=(
            "headless writes H5 data; native/viser run one finite trajectory "
            "through the exact same environment and motion controller."
        ),
    )
    parser.add_argument("--viewer-fps", type=float, default=60.0)
    parser.add_argument("--viewer-env-index", type=int, default=0)
    parser.add_argument(
        "--visual-random-batch-index",
        type=int,
        default=0,
        help=(
            "Native/viser only: advance the seeded motion randomizer by this "
            "many complete parallel batches before displaying a trajectory. "
            "Together with --viewer-env-index, this reproduces an episode from "
            "a prior raw collection with the same seed and num-envs."
        ),
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument(
        "--trajectory-length",
        type=int,
        default=None,
        help=(
            "Total simulator steps per trajectory.  Default: 4000 for "
            "orbit_palm (long stroking window), 2500 otherwise."
        ),
    )
    parser.add_argument("--max-trajectories", type=int, default=5)
    parser.add_argument("--motion-start", type=int, default=1000)
    parser.add_argument(
        "--motion-length",
        type=int,
        default=None,
        help=(
            "Motion window length in simulator steps after motion-start.  "
            "Default: 3000 for orbit_palm, 1400 otherwise."
        ),
    )
    parser.add_argument(
        "--max-prep-wait-steps",
        type=int,
        default=1000,
        help=(
            "After motion-start (the earliest allowed start), wait at most this "
            "many additional simulator steps for the arm and all four fingertips "
            "to settle. Prep frames are never recorded."
        ),
    )
    parser.add_argument(
        "--fixed-motion-start",
        action="store_true",
        help=(
            "Start motion exactly at --motion-start without waiting for all "
            "four fingertips to settle. This is useful for unbiased raw SO(3) "
            "diagnostics: contact quality is measured from the fixed start and "
            "poor initial pose/axis combinations remain visible in the data."
        ),
    )
    parser.add_argument(
        "--record-start-step",
        type=int,
        default=None,
        help="First saved step; defaults to motion-start so prep is excluded.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help=(
            "Maximum candidate batches, including prep failures. Defaults to "
            "10x max-trajectories for online filtering or 10x the required "
            "number of parallel raw batches."
        ),
    )
    parser.add_argument(
        "--online-quality-gate",
        action="store_true",
        help="Reject whole trajectories online. Off by default for fast raw collection.",
    )
    parser.add_argument(
        "--min-realized-rotation-deg",
        type=float,
        default=30.0,
        help=(
            "Online gate: minimum accumulated object rotation during recorded "
            "frames. Ignored for translation-only collection."
        ),
    )
    parser.add_argument(
        "--min-active-motion-ratio",
        type=float,
        default=0.80,
        help=(
            "Online gate: minimum fraction of planned move frames that were "
            "actually allowed to advance; prevents high contact scores from "
            "a mostly frozen object."
        ),
    )
    parser.add_argument(
        "--object-id",
        default="capsule_medium",
        help="Object configuration id from the contact-object catalog.",
    )
    parser.add_argument(
        "--teacher-controller",
        choices=(
            "fullhand_mcc",
            "preview_fixed_grasp",
            "fixed_pregrasp",
        ),
        default="fullhand_mcc",
        help=(
            "fullhand_mcc uses privileged surface points/normals and normal "
            "admittance; preview_fixed_grasp uses the privileged predictive "
            "palm guide but keeps the legacy fixed finger grasp; "
            "fixed_pregrasp retains the fully legacy position teacher."
        ),
    )
    parser.add_argument(
        "--differential-contact-qp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For fullhand_mcc, predict fingertip motion from consecutive "
            "privileged surface targets with a constrained 16-DoF QP. "
            "Enabled by default; use --no-differential-contact-qp for A/B."
        ),
    )
    parser.add_argument(
        "--surface-preload-m",
        type=float,
        default=None,
        help=(
            "Override FullHandMCC inward surface preload; otherwise use the "
            "object YAML collection.surface_preload_m or 0.002 m."
        ),
    )
    parser.add_argument(
        "--palm-standoff-m",
        type=float,
        default=0.0,
        help=(
            "Shift the calibrated palm target along the palm facing axis "
            "(positive = pull the palm away from the object) so the object "
            "sits between the fingers."
        ),
    )
    parser.add_argument(
        "--mcc-tracking",
        action="store_true",
        help=(
            "Use the original MCC fingertip tracking (mass-damper reference "
            "toward the pregrasp tip targets, soft stop at obstacles) "
            "instead of the pure position-servo teacher."
        ),
    )
    parser.add_argument(
        "--no-contact-gate",
        action="store_true",
        help=(
            "Disable the contact gate: the object keeps rotating even when "
            "contact is lost (continuous, gapless trajectories)."
        ),
    )
    parser.add_argument(
        "--anchor-settle-frames",
        type=int,
        default=50,
        help=(
            "Consecutive loaded, correctly oriented contact frames required "
            "before accepting the initial grasp and starting motion."
        ),
    )
    parser.add_argument(
        "--initial-pad-max-angle-deg",
        type=float,
        default=55.0,
        help=(
            "Maximum fingertip-pad rotation error allowed during initial "
            "teacher grasp calibration. Side/edge contacts above this angle "
            "are rejected until the per-finger normal IK recovers them."
        ),
    )
    parser.add_argument(
        "--closure-path-fallback-fraction",
        type=float,
        default=0.70,
        help=(
            "Fraction of the open-to-grasp path used when a fingertip path "
            "does not intersect the object surface (0.60-0.80 recommended)."
        ),
    )
    parser.add_argument(
        "--closure-path-samples",
        type=int,
        default=25,
        help="Samples per fingertip open-to-grasp path for surface intersection.",
    )
    parser.add_argument(
        "--surface-target-mode",
        choices=("nearest_surface", "object_anchor"),
        default="nearest_surface",
        help=(
            "nearest_surface allows tangential slip while following local shape; "
            "object_anchor enforces sticking to the first settled material point."
        ),
    )
    parser.add_argument("--precontact-force-threshold", type=float, default=0.10)
    parser.add_argument("--contact-search-step-m", type=float, default=0.00015)
    parser.add_argument("--contact-search-step-rad", type=float, default=0.02)
    parser.add_argument("--contact-search-limit-rad", type=float, default=0.30)
    parser.add_argument(
        "--contact-transient-loss-frames",
        type=int,
        default=6,
        help=(
            "Loaded-contact dropout frames treated as sensor/contact jitter "
            "before absolute surface recovery starts."
        ),
    )
    parser.add_argument(
        "--contact-recovery-confirm-frames",
        type=int,
        default=4,
        help="Consecutive loaded frames required to leave recovery mode.",
    )
    parser.add_argument(
        "--contact-transient-search-step-m",
        type=float,
        default=0.00020,
        help="Per-step normal nudge during a transient dropout.",
    )
    parser.add_argument(
        "--contact-transient-release-step-m",
        type=float,
        default=0.00010,
        help="Per-step release of the transient nudge after stable recontact.",
    )
    parser.add_argument(
        "--persistent-recovery-max-joint-step-rad",
        type=float,
        default=0.04,
        help=(
            "Per-step joint-speed safeguard for absolute persistent-loss "
            "recovery; this is not a total recovery-distance cap."
        ),
    )
    parser.add_argument("--angular-speed-min", type=float, default=None)
    parser.add_argument("--angular-speed-max", type=float, default=None)
    parser.add_argument(
        "--rotation-axes",
        nargs="+",
        choices=("principal_x", "principal_y", "principal_z", "uniform_sphere"),
        default=None,
        help="Override the configured rotation axes for targeted collection.",
    )
    parser.add_argument(
        "--rotation-axis-vector",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help=(
            "Use one exact object-local rotation axis for every environment. "
            "The vector is normalized automatically and is intended for "
            "reproducing a recorded failure case."
        ),
    )
    parser.add_argument(
        "--axis-sampling",
        choices=("random", "stratified"),
        default="random",
        help=(
            "random samples configured axes independently; stratified cycles "
            "through them across parallel environments for baseline tests."
        ),
    )
    parser.add_argument(
        "--motion-mode",
        choices=(
            "rotation",
            "translation",
            "combined",
            "manifold_fixed_palm",
            "planner_inverse",
            "orbit_palm",
            "palm_orbit",
        ),
        default="rotation",
        help=(
            "Object excitation during the motion window. Use translation and "
            "rotation separately when building single-mode teacher datasets. "
            "orbit_palm keeps the object static and orbits the palm about the "
            "object's long axis (privileged object pose) so fingertip "
            "contacts stay on the pads. palm_orbit is the arm-free variant: "
            "the palm is a free 6-DoF body commanded directly, one-way, with "
            "constant surface clearance and its normal always facing the "
            "object centre."
        ),
    )
    parser.add_argument(
        "--planner-file",
        type=Path,
        default=None,
        help=(
            "Object-frame palm plan generated by generate_manifold_palm_plan.py. "
            "Required for --motion-mode planner_inverse. The collector inverts "
            "this planned palm trajectory to fixed-world-palm object mocap motion."
        ),
    )
    parser.add_argument(
        "--planner-settle-steps",
        type=int,
        default=200,
        help=(
            "For planner_inverse, hold the inverted initial object pose for "
            "this many control steps so FullHandMCC can establish contact "
            "before motion and recording begin (default: 200)."
        ),
    )
    parser.add_argument(
        "--manifold-axis",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 1.0),
        metavar=("X", "Y", "Z"),
        help="Object-local axis for manifold_fixed_palm (default: 0 0 1).",
    )
    parser.add_argument(
        "--manifold-angle-deg",
        type=float,
        default=60.0,
        help="Smooth palm-in-object transport angle in degrees.",
    )
    parser.add_argument(
        "--manifold-direction",
        type=int,
        choices=(-1, 1),
        default=1,
        help="Direction of manifold_fixed_palm transport (+1 or -1).",
    )
    parser.add_argument(
        "--lock-horizontal-lowest-point-to-palm",
        action="store_true",
        help=(
            "Experimental, opt-in object trajectory: after applying the "
            "requested rotation/translation, add a privileged mocap "
            "translation that keeps the centroid of the object's world-Z "
            "lowest surface patch at a fixed point above the palm target. "
            "The existing motion and palm-follow logic is unchanged unless "
            "this flag is present."
        ),
    )
    parser.add_argument(
        "--lowest-point-clearance-m",
        type=float,
        default=0.015,
        help=(
            "With --lock-horizontal-lowest-point-to-palm, vertical distance "
            "between the fixed palm control target and the locked lowest "
            "surface patch. Default: %(default)g m."
        ),
    )
    parser.add_argument(
        "--lowest-point-band-m",
        type=float,
        default=0.002,
        help=(
            "Thickness of the lowest-Z surface band whose X/Y centroid is "
            "locked, avoiding arbitrary vertex switches on a flat face."
        ),
    )
    parser.add_argument(
        "--lowest-point-follow-max-speed-m-s",
        type=float,
        default=0.015,
        help=(
            "Maximum translation speed used to bring the horizontal lowest "
            "surface patch back to its palm anchor. This only affects the "
            "opt-in lowest-point mode. Default: %(default)g m/s."
        ),
    )
    parser.add_argument(
        "--lowest-point-follow-time-constant-s",
        type=float,
        default=0.20,
        help=(
            "First-order velocity smoothing time constant for the optional "
            "lowest-point follower. Larger values react more slowly and "
            "smoothly. Default: %(default)g s."
        ),
    )
    parser.add_argument(
        "--object-initial-z-offset-m",
        type=float,
        default=0.0,
        help=(
            "Optional additive change to the configured object initial world "
            "height. It is independent from the lowest-point experiment and "
            "defaults to zero so existing trajectories are unchanged."
        ),
    )
    parser.add_argument(
        "--surface-clearance-m",
        type=float,
        default=None,
        help=(
            "palm_orbit: constant distance from the object surface along the "
            "radial direction (m). Default: back-computed from the calibrated "
            "palm pose minus the ellipse radius at the initial azimuth."
        ),
    )
    parser.add_argument(
        "--orbit-amplitude-deg",
        type=float,
        default=None,
        help=(
            "Palm-orbit half-amplitude per side in degrees. Default: half of "
            "rotation.angle_range_deg from the object yaml."
        ),
    )
    parser.add_argument(
        "--enable-translation",
        action="store_true",
        help=(
            "Deprecated compatibility alias: changes --motion-mode rotation "
            "to combined. Prefer --motion-mode combined."
        ),
    )
    parser.add_argument(
        "--trans-speed-min",
        type=float,
        default=None,
        help="Override translation speed lower bound from object config.",
    )
    parser.add_argument(
        "--translation-axes",
        nargs="+",
        choices=("principal_x", "principal_y", "principal_z", "uniform_sphere"),
        default=None,
        help="Override the configured translation axes for targeted collection.",
    )
    parser.add_argument(
        "--trans-speed-max",
        type=float,
        default=None,
        help="Override translation speed upper bound from object config.",
    )
    parser.add_argument(
        "--segment-move-steps",
        type=int,
        default=None,
        help="Override every axis profile's consecutive moving steps.",
    )
    parser.add_argument(
        "--segment-hold-steps",
        type=int,
        default=None,
        help="Override every axis profile's recovery hold steps.",
    )
    parser.add_argument(
        "--initial-orientation-mode",
        choices=("uniform", "jitter", "fixed"),
        default="uniform",
        help=(
            "Initial object orientation. 'uniform' samples the full SO(3) "
            "rotation space; 'jitter' applies the legacy bounded perturbation "
            "around the environment nominal pose; 'fixed' keeps the nominal pose."
        ),
    )
    parser.add_argument(
        "--initial-orientation-jitter-deg",
        type=float,
        default=10.0,
        help=(
            "Maximum legacy perturbation around the nominal object orientation; "
            "used only when --initial-orientation-mode jitter."
        ),
    )
    parser.add_argument(
        "--contact-threshold",
        type=float,
        default=0.05,
        help="Minimum full-fingertip 3-D resultant force magnitude in newtons.",
    )
    parser.add_argument(
        "--contact-stiffness",
        type=float,
        default=-HARD_CONTACT_SOLREF[0],
        help=(
            "Direct-format MuJoCo contact stiffness for the target object. "
            "Default: %(default)g N/m."
        ),
    )
    parser.add_argument(
        "--contact-damping",
        type=float,
        default=-HARD_CONTACT_SOLREF[1],
        help=(
            "Direct-format MuJoCo contact damping for the target object. "
            "Default: %(default)g N s/m."
        ),
    )
    parser.add_argument(
        "--contact-transition-width-m",
        type=float,
        default=HARD_CONTACT_SOLIMP[2],
        help=(
            "MuJoCo solimp transition width for hand/object contact. "
            "Keep margin/gap at zero for MJWarp; default: %(default)g m."
        ),
    )
    parser.add_argument(
        "--fullhand-finger-stiffness",
        type=float,
        default=35.0,
        help="Position-servo stiffness used by the FullHandMCC teacher.",
    )
    parser.add_argument(
        "--fullhand-finger-damping",
        type=float,
        default=2.5,
        help="Position-servo damping used by the FullHandMCC teacher.",
    )
    parser.add_argument(
        "--fullhand-finger-effort-limit",
        type=float,
        default=35.0,
        help="Per-joint effort limit used by the FullHandMCC teacher.",
    )
    parser.add_argument(
        "--fullhand-force-servo-gain",
        type=float,
        default=0.003,
        help=(
            "Direct fingertip force-servo integral gain for FullHandMCC. "
            "Set to zero only for position-only data collection (default: 0.003)."
        ),
    )
    parser.add_argument(
        "--physics-substeps",
        type=int,
        default=10,
        help=(
            "Physics steps per 10 ms control step. The default 10 gives a "
            "1 ms timestep and avoids single-frame convex-part contact "
            "dropouts in difficult mesh-contact teachers."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--filename", default=None)
    args = parser.parse_args()

    if args.contact_stiffness <= 0.0:
        raise ValueError("--contact-stiffness must be positive")
    if args.contact_damping <= 0.0:
        raise ValueError("--contact-damping must be positive")
    if args.contact_transition_width_m <= 0.0:
        raise ValueError("--contact-transition-width-m must be positive")
    if args.fullhand_finger_stiffness <= 0.0:
        raise ValueError("--fullhand-finger-stiffness must be positive")
    if args.fullhand_finger_damping < 0.0:
        raise ValueError("--fullhand-finger-damping must be non-negative")
    if args.fullhand_finger_effort_limit <= 0.0:
        raise ValueError("--fullhand-finger-effort-limit must be positive")
    if args.fullhand_force_servo_gain < 0.0:
        raise ValueError("--fullhand-force-servo-gain must be non-negative")
    if args.physics_substeps <= 0:
        raise ValueError("--physics-substeps must be positive")
    if args.planner_settle_steps < 0:
        raise ValueError("--planner-settle-steps cannot be negative")
    if not np.isfinite(args.object_initial_z_offset_m):
        raise ValueError("--object-initial-z-offset-m must be finite")
    if args.lowest_point_clearance_m < 0.0:
        raise ValueError("--lowest-point-clearance-m must be non-negative")
    if args.lowest_point_band_m <= 0.0:
        raise ValueError("--lowest-point-band-m must be positive")
    if args.lowest_point_follow_max_speed_m_s <= 0.0:
        raise ValueError("--lowest-point-follow-max-speed-m-s must be positive")
    if args.lowest_point_follow_time_constant_s < 0.0:
        raise ValueError(
            "--lowest-point-follow-time-constant-s must be non-negative"
        )
    if args.visual_random_batch_index < 0:
        raise ValueError("--visual-random-batch-index must be non-negative")
    if args.contact_transient_loss_frames < 0:
        raise ValueError("--contact-transient-loss-frames cannot be negative")
    if args.contact_recovery_confirm_frames <= 0:
        raise ValueError("--contact-recovery-confirm-frames must be positive")
    if min(
        args.contact_transient_search_step_m,
        args.contact_transient_release_step_m,
        args.persistent_recovery_max_joint_step_rad,
    ) <= 0.0:
        raise ValueError("Contact recovery step sizes must be positive")
    contact_solref = (-args.contact_stiffness, -args.contact_damping)
    contact_solimp = (
        HARD_CONTACT_SOLIMP[0],
        HARD_CONTACT_SOLIMP[1],
        args.contact_transition_width_m,
        HARD_CONTACT_SOLIMP[3],
        HARD_CONTACT_SOLIMP[4],
    )

    if args.rotation_axis_vector is not None:
        rotation_axis_vector = np.asarray(args.rotation_axis_vector, dtype=np.float64)
        if not np.all(np.isfinite(rotation_axis_vector)):
            raise ValueError("--rotation-axis-vector must contain finite values")
        if np.linalg.norm(rotation_axis_vector) < 1.0e-8:
            raise ValueError("--rotation-axis-vector cannot be the zero vector")
    else:
        rotation_axis_vector = None

    manifold_mode = args.motion_mode == "manifold_fixed_palm"
    planner_inverse_mode = args.motion_mode == "planner_inverse"
    orbit_mode = args.motion_mode in ("orbit_palm", "palm_orbit")
    palm_orbit_mode = args.motion_mode == "palm_orbit"
    # Mode-dependent defaults: orbit modes record a long stroking window
    # (40 s trajectory, 30 s motion) so the full palm sweep fits inside the
    # recording; other modes keep the shorter segment-based defaults.
    if args.trajectory_length is None:
        args.trajectory_length = 4000 if orbit_mode else 2500
    if args.motion_length is None:
        args.motion_length = 3000 if orbit_mode else 1400
    if args.enable_translation:
        if args.motion_mode == "translation":
            raise ValueError(
                "--enable-translation is redundant with --motion-mode translation"
            )
        if orbit_mode:
            raise ValueError(
                "--enable-translation is incompatible with --motion-mode "
                "orbit_palm"
            )
        args.motion_mode = "combined"
    if manifold_mode or planner_inverse_mode:
        if args.enable_translation or orbit_mode:
            raise ValueError(
                "fixed-palm planned modes are incompatible with explicit translation/orbit modes"
            )
        if args.initial_orientation_mode != "fixed":
            raise ValueError(
                "fixed-palm planned modes require --initial-orientation-mode fixed "
                "so the planned object-frame trajectory matches the reset pose"
            )
        if args.lock_horizontal_lowest_point_to_palm:
            raise ValueError(
                "fixed-palm planned modes cannot be combined with "
                "--lock-horizontal-lowest-point-to-palm"
            )
        if manifold_mode and args.manifold_angle_deg <= 0.0:
            raise ValueError("--manifold-angle-deg must be positive")
        if planner_inverse_mode:
            if args.planner_file is None:
                raise ValueError("--motion-mode planner_inverse requires --planner-file")
            if not args.planner_file.is_file():
                raise FileNotFoundError(f"planner file not found: {args.planner_file}")
            if args.teacher_controller not in ("fullhand_mcc", "preview_fixed_grasp"):
                raise ValueError(
                    "planner_inverse requires --teacher-controller fullhand_mcc "
                    "or preview_fixed_grasp"
                )
        manifold_axis = np.asarray(args.manifold_axis, dtype=np.float64)
        if manifold_axis.shape != (3,) or np.linalg.norm(manifold_axis) < 1.0e-8:
            raise ValueError("--manifold-axis must be a non-zero 3-vector")
    else:
        manifold_axis = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    rotation_enabled = args.motion_mode in ("rotation", "combined")
    translation_enabled = args.motion_mode in ("translation", "combined")
    if orbit_mode:
        if args.teacher_controller != "fixed_pregrasp":
            raise ValueError(
                "--motion-mode orbit_palm requires --teacher-controller "
                "fixed_pregrasp (FullHandMCC has its own palm admittance)"
            )
        if args.initial_orientation_mode == "uniform":
            raise ValueError(
                "--motion-mode orbit_palm starts from the calibrated world "
                "grasp; uniform object orientation would rotate the object "
                "out of the phase-0 grasp. Use --initial-orientation-mode "
                "fixed or jitter."
            )
    if args.lock_horizontal_lowest_point_to_palm and orbit_mode:
        raise ValueError(
            "--lock-horizontal-lowest-point-to-palm applies to object-motion "
            "modes, not palm-orbit modes"
        )

    # ---- resolve object & motion configuration ---------------------------------
    object_config = load_object_config(args.object_id)
    if args.object_initial_z_offset_m:
        shifted_position = list(object_config.initial_pos)
        shifted_position[2] += float(args.object_initial_z_offset_m)
        object_config = replace(
            object_config, initial_pos=tuple(shifted_position)
        )
        print(
            "[OBJECT] experimental initial Z offset: "
            f"{args.object_initial_z_offset_m * 1000:+.1f} mm -> "
            f"z={shifted_position[2]:.4f} m"
        )
    surface_preload_m = float(
        args.surface_preload_m
        if args.surface_preload_m is not None
        else object_config.collection.get("surface_preload_m", 0.002)
    )
    # Mesh objects sample an isotropic scale once per collection run so the
    # robustness sweep covers object size without re-tuning the geometry YAML.
    scale_range = object_config.collection.get("size_scale_range")
    if scale_range is not None:
        scale_lo, scale_hi = (float(value) for value in scale_range)
        if not 0.0 < scale_lo <= scale_hi:
            raise ValueError(
                f"object {object_id!r} size_scale_range must be positive "
                "and ordered (lo <= hi)"
            )
        object_scale = float(
            np.random.default_rng(args.seed).uniform(scale_lo, scale_hi)
        )
    else:
        object_scale = 1.0
    object_lower, object_upper = object_local_aabb(
        object_config, scale=object_scale
    )
    object_extent = object_upper - object_lower
    motion = get_motion_config(object_config)
    rot_cfg = motion["rotation"]
    trans_cfg = motion["translation"]
    rot_allowed_axes = (
        args.rotation_axes
        if args.rotation_axes is not None
        else rot_cfg.get("allowed_axes", ["uniform_sphere"])
    )

    angular_speed_min = (
        args.angular_speed_min
        if args.angular_speed_min is not None
        else rot_cfg["angular_speed_range_rad_s"][0]
    )
    angular_speed_max = (
        args.angular_speed_max
        if args.angular_speed_max is not None
        else rot_cfg["angular_speed_range_rad_s"][1]
    )
    if orbit_mode:
        # Orbit tuning can live in its own ``motion.orbit`` section and
        # falls back to the rotation section for objects without one.
        # (get_motion_config always returns an "orbit" entry, possibly {}.)
        orbit_cfg = motion.get("orbit") or rot_cfg
        if args.angular_speed_min is None:
            angular_speed_min = float(orbit_cfg["angular_speed_range_rad_s"][0])
        if args.angular_speed_max is None:
            angular_speed_max = float(orbit_cfg["angular_speed_range_rad_s"][1])
        orbit_amp_deg = orbit_cfg.get("angle_range_deg", (20.0, 100.0))
        if args.orbit_amplitude_deg is not None:
            amplitude_deg_min = amplitude_deg_max = float(args.orbit_amplitude_deg)
        else:
            amplitude_deg_min = float(orbit_amp_deg[0]) / 2.0
            amplitude_deg_max = float(orbit_amp_deg[1]) / 2.0
    trans_speed_min = (
        args.trans_speed_min
        if args.trans_speed_min is not None
        else trans_cfg["speed_range_m_s"][0]
    )
    trans_speed_max = (
        args.trans_speed_max
        if args.trans_speed_max is not None
        else trans_cfg["speed_range_m_s"][1]
    )
    trans_distance_ratio_min, trans_distance_ratio_max = trans_cfg.get(
        "distance_ratio_range", (0.1, 0.3)
    )
    trans_absolute_max_m = float(trans_cfg.get("absolute_max_m", 0.05))
    trans_allowed_axes = (
        args.translation_axes
        if args.translation_axes is not None
        else trans_cfg.get("allowed_axes", ["uniform_sphere"])
    )
    trans_distance_mode = trans_cfg.get("distance_mode", "object_extent_ratio")
    active_motion_configs = [
        config
        for enabled, config in (
            (rotation_enabled, rot_cfg),
            (translation_enabled, trans_cfg),
        )
        if enabled
    ]
    default_segment_move_steps = int(
        args.segment_move_steps
        if args.segment_move_steps is not None
        else min(
            (
                int(config.get("segment_move_steps", args.motion_length))
                for config in active_motion_configs
            ),
            default=args.motion_length,
        )
    )
    default_segment_hold_steps = int(
        args.segment_hold_steps
        if args.segment_hold_steps is not None
        else max(
            (
                int(config.get("segment_hold_steps", 0))
                for config in active_motion_configs
            ),
            default=0,
        )
    )
    contact_gated_motion = any(
        bool(config.get("contact_gated", False))
        for config in active_motion_configs
    )

    if angular_speed_min < 0.0 or angular_speed_max < angular_speed_min:
        raise ValueError("Invalid angular speed range")
    if trans_speed_min < 0.0 or trans_speed_max < trans_speed_min:
        raise ValueError("Invalid translation speed range")

    print(
        f"[OBJECT] id={args.object_id} family={object_config.family} "
        f"extent={np.round(object_extent, 3).tolist()}m "
        f"mass={object_config.total_mass_kg:.2f}kg"
        + (
            f" scale={object_scale:.3f}"
            if scale_range is not None
            else ""
        )
    )
    print(
        f"[MOTION] rotation angular_speed=[{angular_speed_min:.3f}, "
        f"{angular_speed_max:.3f}] rad/s axes={list(rot_allowed_axes)} "
        f"ramp={float(rot_cfg.get('acceleration_time_s', 0.0)):.2f}s | "
        f"mode={args.motion_mode} "
        f"segments=move{default_segment_move_steps}+"
        f"hold{default_segment_hold_steps}steps "
        f"contact_gate={'ON' if contact_gated_motion else 'OFF'} "
        + (
            f"speed=[{trans_speed_min:.4f}, {trans_speed_max:.4f}] m/s"
            if translation_enabled
            else ""
        )
    )
    # ---------------------------------------------------------------------------

    if args.motion_start < 0 or args.motion_start >= args.trajectory_length:
        raise ValueError("motion-start must lie inside the trajectory")
    if args.max_prep_wait_steps < 0:
        raise ValueError("max-prep-wait-steps must be non-negative")
    if not 0.0 <= args.closure_path_fallback_fraction <= 1.0:
        raise ValueError("closure-path-fallback-fraction must be in [0, 1]")
    if args.closure_path_samples < 3:
        raise ValueError("closure-path-samples must be at least 3")
    record_start = (
        args.motion_start if args.record_start_step is None else args.record_start_step
    )
    if record_start < 0 or record_start >= args.trajectory_length:
        raise ValueError("record-start-step must lie inside the trajectory")
    if record_start < args.motion_start:
        raise ValueError("record-start-step cannot precede motion-start")
    record_delay = record_start - args.motion_start
    # planner_inverse spends an additional settling interval with the
    # inverted object pose fixed. Those frames are excluded from recording.
    effective_record_delay = record_delay + (
        args.planner_settle_steps if planner_inverse_mode else 0
    )
    post_start_steps = args.trajectory_length - args.motion_start
    saved_frames_per_trajectory = post_start_steps - effective_record_delay
    if saved_frames_per_trajectory <= 0:
        raise ValueError(
            "trajectory-length leaves no frames after planner settling; "
            "increase --trajectory-length or reduce --planner-settle-steps"
        )
    if args.online_quality_gate and args.num_envs != 1:
        raise ValueError(
            "Online trajectory filtering requires --num-envs 1; collect raw in parallel "
            "and run filter_trajectories.py instead."
        )
    batches_needed = int(np.ceil(args.max_trajectories / args.num_envs))
    collected_target = batches_needed * args.num_envs
    if args.max_attempts is not None:
        max_attempts = args.max_attempts
    elif args.online_quality_gate:
        max_attempts = max(10 * args.max_trajectories, 1)
    elif args.fixed_motion_start:
        # Fixed-start raw collection never rejects a batch during preparation.
        max_attempts = batches_needed
    else:
        # A uniformly sampled SO(3) pose can be outside the four-tip
        # pre-contact basin.  Raw parallel collection must retry such a batch
        # instead of aborting the entire overnight run because one environment
        # did not settle.
        max_attempts = max(10 * batches_needed, 1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    if palm_orbit_mode:
        env_cfg = mcc_palm_free_contact_env_cfg(
            num_envs=args.num_envs,
            play=True,
            object_config=object_config,
            object_scale=object_scale,
        )
    else:
        fullhand_actuators = args.teacher_controller == "fullhand_mcc"
        env_cfg = mcc_finger_contact_env_cfg(
            num_envs=args.num_envs,
            play=True,
            object_config=object_config,
            object_scale=object_scale,
            finger_stiffness=(
                args.fullhand_finger_stiffness if fullhand_actuators else 5.0
            ),
            finger_damping=(
                args.fullhand_finger_damping if fullhand_actuators else 0.5
            ),
            finger_effort_limit=(
                args.fullhand_finger_effort_limit if fullhand_actuators else 10.0
            ),
            contact_solref=contact_solref,
            contact_solimp=contact_solimp,
            physics_substeps=args.physics_substeps,
        )
        if fullhand_actuators:
            print(
                "[CONTROLLER] FullHandMCC finger actuators: "
                f"stiffness={args.fullhand_finger_stiffness:g} "
                f"damping={args.fullhand_finger_damping:g} "
                f"effort_limit={args.fullhand_finger_effort_limit:g}"
            )
        print(
            "[CONTACT] target solref="
            f"({contact_solref[0]:g}, {contact_solref[1]:g}) "
            f"solimp_width={contact_solimp[2]:g}m margin=0 gap=0 "
            f"physics_substeps={args.physics_substeps}"
        )
    env_cfg.viewer.env_idx = max(
        0, min(int(args.viewer_env_index), args.num_envs - 1)
    )
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    # For mesh objects (convex-decomposed collision parts), MuJoCo contact
    # normals jump at part seams.  Replace them with smooth normals estimated
    # from the high-resolution source OBJ near each contact point.
    mesh_normal_oracle = MeshNormalOracle.from_config(
        object_config, scale=object_scale
    )
    if mesh_normal_oracle is not None:
        print(
            "[OBJECT] mesh normal oracle: "
            f"{len(mesh_normal_oracle.vertices)} source verts, "
            f"radius={mesh_normal_oracle.radius_m:.3f} m"
        )
    rl_cfg = MCCLeapHandPositionControlCfg()
    kwargs = asdict(rl_cfg)
    if palm_orbit_mode:
        # Palm-free collection: the palm is a free 6-DoF body commanded as an
        # absolute world pose; the hand joint block starts at qpos index 7.
        kwargs["palm_direct"] = True
        kwargs["hand_q_start"] = 7
        print(
            "[CONTROLLER] palm_direct: no xArm; palm absolute-pose servo, "
            "hand block at qpos 7"
        )
    # Per-object pregrasp override: pin a specific hand shape for an
    # object through ``collection.pregrasp_q`` in its object yaml.  The
    # YCB mustard bottle keeps the default adducted pinch (verified to
    # load all four pads on the enlarged surface); overrides are for
    # objects that genuinely need a different hand shape.
    object_pregrasp = object_config.collection.get("pregrasp_q")
    if object_pregrasp is not None:
        kwargs["pregrasp_q"] = tuple(float(v) for v in object_pregrasp)
        print(
            "[CONTROLLER] per-object pregrasp_q override from "
            f"{args.object_id}: {kwargs['pregrasp_q']}"
        )
    policy_class = kwargs.pop("policy_class")
    kwargs.pop("device", None)
    if args.mcc_tracking:
        kwargs["position_test_mode"] = False
        print("[CONTROLLER] MCC fingertip tracking enabled (original path)")
    base_policy = policy_class(device=device, num_envs=args.num_envs, **kwargs)
    if args.palm_standoff_m:
        facing = R.from_rotvec(
            base_policy.palm_controller.fixed_target_np[3:6]
        ).apply(np.array([0.0, 0.0, -1.0]))
        base_policy.palm_controller.fixed_target_np[:3] += (
            facing * float(args.palm_standoff_m)
        )
        print(
            "[CONTROLLER] palm standoff "
            f"+{args.palm_standoff_m * 1000:.0f} mm along the facing axis"
        )

    target_mocap_idx = int(env.scene["target"].data.indexing.mocap_id)
    palm_idx = _find_local_body_index(env, "palm_lower")
    tip_indices = [_find_local_site_index(env, name) for name in TIP_SITES]
    dt = float(env_cfg.decimation * env_cfg.sim.mujoco.timestep)
    if orbit_mode:
        if palm_orbit_mode:
            orbit_controller = FacingCenterOrbitController(
                env,
                fixed_target=base_policy.palm_controller.fixed_target,
                target_mocap_idx=target_mocap_idx,
                object_extent_local=object_extent,
                angular_speed_min=angular_speed_min,
                angular_speed_max=angular_speed_max,
                surface_clearance_m=args.surface_clearance_m,
                motion_start=args.motion_start,
                motion_length=args.motion_length,
                dt=dt,
                device=device,
            )
        else:
            orbit_controller = PalmOrbitController(
                env,
                fixed_target=base_policy.palm_controller.fixed_target,
                target_mocap_idx=target_mocap_idx,
                object_extent_local=object_extent,
                angular_speed_min=angular_speed_min,
                angular_speed_max=angular_speed_max,
                amplitude_deg_min=amplitude_deg_min,
                amplitude_deg_max=amplitude_deg_max,
                motion_start=args.motion_start,
                motion_length=args.motion_length,
                dt=dt,
                device=device,
            )
        policy = PalmOrbitFixedPregraspPolicy(
            env,
            base_policy,
            orbit_controller,
            contact_threshold=args.contact_threshold,
        )
    else:
        orbit_controller = None
        policy = (
            FullHandMCCCollectionPolicy(
                env,
                base_policy,
                object_config,
                target_mocap_idx,
                palm_idx,
                tip_indices,
                surface_preload_m=surface_preload_m,
                anchor_force_threshold=args.contact_threshold,
                precontact_force_threshold=args.precontact_force_threshold,
                anchor_settle_frames=args.anchor_settle_frames,
                surface_target_mode=args.surface_target_mode,
                contact_search_step_m=args.contact_search_step_m,
                contact_search_step_rad=args.contact_search_step_rad,
                contact_search_limit_rad=args.contact_search_limit_rad,
                contact_transient_loss_frames=(
                    args.contact_transient_loss_frames
                ),
                contact_recovery_confirm_frames=(
                    args.contact_recovery_confirm_frames
                ),
                contact_transient_search_step_m=(
                    args.contact_transient_search_step_m
                ),
                contact_transient_release_step_m=(
                    args.contact_transient_release_step_m
                ),
                persistent_recovery_max_joint_step_rad=(
                    args.persistent_recovery_max_joint_step_rad
                ),
                nominal_grasp_q=base_policy.finger_controller.pregrasp_q,
                differential_contact_qp=args.differential_contact_qp,
                fixed_grasp_fingers=(
                    args.teacher_controller == "preview_fixed_grasp"
                    and not planner_inverse_mode
                ),
                object_scale=object_scale,
                mesh_normal_oracle=mesh_normal_oracle,
                # The optional lowest-point experiment moves the object to
                # preserve palm clearance.  Do not simultaneously move the
                # palm with the legacy privileged preview, otherwise two
                # independent loops fight over the same relative geometry.
                # The default (flag absent) remains exactly the old behavior.
                enable_privileged_palm_follow=(
                    not args.lock_horizontal_lowest_point_to_palm
                    and not manifold_mode
                    and not planner_inverse_mode
                ),
                # The direct object-frame palm planner intentionally leaves
                # fingertip placement to FullHandMCC.  Keep its reference
                # grasp / lateral-spacing regularizer active so the fingers
                # do not collapse into a curled or side-swung configuration.
                shape_regularization=(manifold_mode or planner_inverse_mode),
                force_servo_integral_gain=args.fullhand_force_servo_gain,
                initial_pad_max_angle_rad=np.deg2rad(
                    args.initial_pad_max_angle_deg
                ),
                closure_path_fallback_fraction=args.closure_path_fallback_fraction,
                closure_path_samples=args.closure_path_samples,
            )
            if args.teacher_controller in (
                "fullhand_mcc",
                "preview_fixed_grasp",
            )
            else base_policy
        )
    print(
        f"[CONTROLLER] teacher={args.teacher_controller} "
        "privileged_oracle="
        f"{args.teacher_controller in ('fullhand_mcc', 'preview_fixed_grasp')} "
        f"surface_target={args.surface_target_mode} "
        f"differential_qp={args.differential_contact_qp}"
    )
    if planner_inverse_mode and args.teacher_controller == "preview_fixed_grasp":
        print(
            "[CONTROLLER] planner contact IK active | "
            "strong grasp-posture + fingertip-normal regularization"
        )
    if orbit_mode:
        print(
            f"[PALM-ORBIT] long_axis_local={orbit_controller.long_axis_local} "
            f"amplitude_deg=[{amplitude_deg_min:.1f},{amplitude_deg_max:.1f}] "
            f"speed_range=[{angular_speed_min:.4f},{angular_speed_max:.4f}]rad/s "
            f"window=[{args.motion_start},{args.motion_start + args.motion_length})"
        )

    lowest_surface_points_object: np.ndarray | None = None
    lowest_point_anchor_world: np.ndarray | None = None
    if args.lock_horizontal_lowest_point_to_palm:
        if mesh_normal_oracle is not None:
            # High-resolution visual vertices preserve the intended YCB
            # geometry; collision decomposition differs only at millimetre
            # scale and its part seams should not move the trajectory anchor.
            lowest_surface_points_object = np.asarray(
                mesh_normal_oracle.vertices, dtype=np.float32
            )
            lowest_geometry_source = "source_mesh_vertices"
        else:
            # Conservative fallback for procedural primitives/compounds.  It
            # keeps the feature available without changing their legacy
            # oracle; mesh objects use their full surface above.
            lowest_surface_points_object = np.asarray(
                [
                    (x, y, z)
                    for x in (object_lower[0], object_upper[0])
                    for y in (object_lower[1], object_upper[1])
                    for z in (object_lower[2], object_upper[2])
                ],
                dtype=np.float32,
            )
            lowest_geometry_source = "object_aabb_corners"
        fixed_palm_position = np.asarray(
            base_policy.palm_controller.fixed_target_np[:3], dtype=np.float32
        )
        lowest_point_anchor_world = (
            env.scene.env_origins.detach().cpu().numpy().astype(np.float32)
            + fixed_palm_position[None, :]
        )
        lowest_point_anchor_world[:, 2] += float(
            args.lowest_point_clearance_m
        )
        print(
            "[MOTION-EXPERIMENT] horizontal lowest-point lock ENABLED | "
            f"source={lowest_geometry_source} "
            f"points={len(lowest_surface_points_object)} "
            f"clearance={args.lowest_point_clearance_m * 1000:.1f}mm "
            f"band={args.lowest_point_band_m * 1000:.1f}mm "
            f"max_follow_speed="
            f"{args.lowest_point_follow_max_speed_m_s * 1000:.1f}mm/s "
            f"smooth_tau={args.lowest_point_follow_time_constant_s:.2f}s"
        )

    motion_controller = ObjectMotionController(
        env,
        target_mocap_idx,
        motion_start=args.motion_start,
        motion_length=args.motion_length,
        dt=dt,
        initial_orientation_mode=args.initial_orientation_mode,
        initial_orientation_jitter_deg=args.initial_orientation_jitter_deg,
        rotation_enabled=rotation_enabled,
        rotation_allowed_axes=rot_allowed_axes,
        angular_speed_min=angular_speed_min,
        angular_speed_max=angular_speed_max,
        rotation_acceleration_time_s=float(
            rot_cfg.get("acceleration_time_s", 0.0)
        ),
        axis_sampling=args.axis_sampling,
        rotation_axis_override_local=rotation_axis_vector,
        translation_enabled=translation_enabled,
        translation_allowed_axes=trans_allowed_axes,
        trans_speed_min=trans_speed_min,
        trans_speed_max=trans_speed_max,
        trans_distance_mode=trans_distance_mode,
        trans_distance_ratio_range=(
            float(trans_distance_ratio_min),
            float(trans_distance_ratio_max),
        ),
        trans_absolute_max_m=trans_absolute_max_m,
        object_extent=object_extent,
        rotation_axis_profiles=rot_cfg.get("axis_profiles", {}),
        translation_axis_profiles=trans_cfg.get("axis_profiles", {}),
        default_segment_move_steps=default_segment_move_steps,
        default_segment_hold_steps=default_segment_hold_steps,
        segment_move_steps_override=args.segment_move_steps,
        segment_hold_steps_override=args.segment_hold_steps,
        contact_gated_motion=contact_gated_motion,
        lock_horizontal_lowest_point=(
            args.lock_horizontal_lowest_point_to_palm
        ),
        surface_points_object=lowest_surface_points_object,
        lowest_point_anchor_world=lowest_point_anchor_world,
        lowest_point_band_m=args.lowest_point_band_m,
        lowest_point_follow_max_speed_m_s=(
            args.lowest_point_follow_max_speed_m_s
        ),
        lowest_point_follow_time_constant_s=(
            args.lowest_point_follow_time_constant_s
        ),
    )
    if manifold_mode:
        # Replace the ordinary object excitation only after constructing it;
        # this inverted trajectory is the actual mocap writer used below.
        motion_controller = PlannedFixedPalmObjectController(
            env,
            target_mocap_idx,
            palm_idx,
            base_policy.palm_controller.fixed_target_np,
            np.asarray(object_config.initial_pos, dtype=np.float64),
            np.asarray(object_config.initial_rot, dtype=np.float64),
            motion_start=args.motion_start,
            motion_length=args.motion_length,
            total_steps=(args.trajectory_length + args.max_prep_wait_steps + 2),
            dt=dt,
            axis_local=manifold_axis,
            angle_deg=args.manifold_angle_deg,
            direction=args.manifold_direction,
        )
        print(
            "[MANIFOLD] fixed world palm + inverted object trajectory | "
            f"axis={np.round(manifold_axis / np.linalg.norm(manifold_axis), 3).tolist()} "
            f"angle={args.manifold_angle_deg:.1f}deg "
            f"direction={args.manifold_direction:+d}"
        )
    elif planner_inverse_mode:
        with h5py.File(args.planner_file, "r") as planner_h5:
            if "palm_pose_object" not in planner_h5:
                raise KeyError(
                    f"{args.planner_file} has no palm_pose_object dataset"
                )
            planned_palm_pose = np.asarray(
                planner_h5["palm_pose_object"], dtype=np.float64
            )
        # Plan files are inverse-compatible and conventionally store T x 1 x 7.
        if planned_palm_pose.ndim == 3:
            planned_palm_pose = planned_palm_pose[:, 0, :]
        motion_controller = InversePlannedPalmObjectController(
            env,
            target_mocap_idx,
            palm_idx,
            planned_palm_pose,
            motion_start=args.motion_start,
            motion_length=args.motion_length,
            total_steps=(args.trajectory_length + args.max_prep_wait_steps + 2),
            dt=dt,
        )
        print(
            "[PLANNER-INVERSE] fixed world palm + smoothed object-frame palm plan | "
            f"file={args.planner_file} frames={len(planned_palm_pose)}"
        )
    if isinstance(policy, FullHandMCCCollectionPolicy):
        policy.set_motion_preview_controller(motion_controller)
        if policy.enable_privileged_palm_follow:
            print(
                "[CONTROLLER] privileged palm surface preview "
                f"enabled: {policy.palm_motion_preview_s:.2f}s"
            )
        else:
            print(
                "[MOTION-EXPERIMENT] legacy palm surface preview disabled; "
                "palm MCC retains its original fixed target"
            )

    if args.viewer != "headless":
        class VisualCollectionPolicy:
            """Same controller and object motion as H5 collection, with live contact UI."""

            def __init__(self) -> None:
                self.step_count = 0
                self.actual_motion_start: int | None = None
                self.recorded_frames = 0
                self.loaded_frames = np.zeros(4, dtype=np.int64)
                self.found_frames = np.zeros(4, dtype=np.int64)
                self.all4_frames = 0
                self.contact_pos = np.full((4, 3), np.nan, dtype=np.float64)
                self.tip_pos = np.full((4, 3), np.nan, dtype=np.float64)
                self.found = np.zeros(4, dtype=bool)
                self.loaded = np.zeros(4, dtype=bool)
                self.contact_phase = np.zeros(4, dtype=np.int32)
                self.recovery_error_m = np.zeros(4, dtype=np.float64)
                self.recovery_target = np.full(
                    (4, 3), np.nan, dtype=np.float64
                )
                self.object_lowest_point = np.full(
                    3, np.nan, dtype=np.float64
                )

            def reset(self) -> None:
                policy.reset()
                motion_controller.reset()
                motion_controller.motion_start = args.motion_start
                self.step_count = 0
                self.actual_motion_start = None
                self.planner_pose_initialized = False
                self.planner_settle_until = None
                self.recorded_frames = 0
                self.loaded_frames.fill(0)
                self.found_frames.fill(0)
                self.all4_frames = 0
                self.contact_pos.fill(np.nan)
                self.tip_pos.fill(np.nan)
                self.found.fill(False)
                self.loaded.fill(False)
                self.contact_phase.fill(0)
                self.recovery_error_m.fill(0.0)
                self.recovery_target.fill(np.nan)
                self.object_lowest_point.fill(np.nan)
                print(
                    "[VISUAL] orientation="
                    f"{motion_controller.initial_quat[env_cfg.viewer.env_idx].detach().cpu().numpy().round(4).tolist()} "
                    f"rotation_axis={motion_controller.rotation_axis_labels[env_cfg.viewer.env_idx]} "
                    f"speed={float(motion_controller.angular_speeds[env_cfg.viewer.env_idx]):.4f}rad/s"
                )
                if translation_enabled:
                    print(
                        "[VISUAL] translation_axis="
                        f"{motion_controller.translation_axis_labels[env_cfg.viewer.env_idx]} "
                        f"amplitude={float(motion_controller.translation_amplitudes[env_cfg.viewer.env_idx]):.4f}m "
                        f"speed={float(motion_controller.translation_speeds[env_cfg.viewer.env_idx]):.4f}m/s"
                    )

            def __call__(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
                ready = (
                    policy.ready_for_motion
                    if args.teacher_controller == "fullhand_mcc"
                    else True
                )
                if planner_inverse_mode:
                    if (
                        not self.planner_pose_initialized
                        and self.step_count >= args.motion_start
                        and (args.fixed_motion_start or ready)
                    ):
                        motion_controller.reanchor_from_current_state()
                        self.planner_pose_initialized = True
                        self.planner_settle_until = (
                            self.step_count + args.planner_settle_steps
                        )
                        motion_controller.motion_start = self.planner_settle_until
                        print(
                            "[VISUAL] inverted object pose initialized at step "
                            f"{self.step_count}; settling until "
                            f"{self.planner_settle_until}"
                        )
                    if (
                        self.planner_pose_initialized
                        and self.actual_motion_start is None
                        and self.step_count >= self.planner_settle_until
                    ):
                        self.actual_motion_start = self.step_count
                        print(
                            "[VISUAL] motion and recording start after planner "
                            f"settle at step {self.actual_motion_start}"
                        )
                elif (
                    self.actual_motion_start is None
                    and self.step_count >= args.motion_start
                    and (args.fixed_motion_start or ready)
                ):
                    self.actual_motion_start = self.step_count
                    motion_controller.motion_start = self.actual_motion_start
                    if manifold_mode:
                        motion_controller.reanchor_from_current_state()
                    print(
                        "[VISUAL] motion and recording window start at simulator "
                        f"step {self.actual_motion_start} "
                        f"({'fixed schedule' if args.fixed_motion_start else 'contact settled'})"
                    )
                if (
                    not planner_inverse_mode
                    and not args.fixed_motion_start
                    and
                    self.actual_motion_start is None
                    and self.step_count
                    >= args.motion_start + args.max_prep_wait_steps
                ):
                    raise RuntimeError(
                        "FullHandMCC did not settle all four fingertips within the "
                        "allowed prep window; fix the initial grasp/preload"
                    )
                contact_ready = (
                    policy.motion_ready_mask
                    if args.teacher_controller == "fullhand_mcc"
                    else None
                )
                if orbit_mode:
                    # Object stays still; the palm orbit is stepped inside the
                    # policy on the same contact gate.
                    moving = policy.orbit_moving
                elif planner_inverse_mode and self.planner_pose_initialized:
                    # During settle, step() keeps the inverted object at path[0].
                    moving = motion_controller.step(
                        self.step_count, contact_ready=contact_ready
                    )
                elif self.actual_motion_start is not None:
                    moving = motion_controller.step(
                        self.step_count, contact_ready=contact_ready
                    )
                else:
                    moving = False
                action = policy(obs)
                env_id = env_cfg.viewer.env_idx
                if args.teacher_controller == "fullhand_mcc":
                    phase_tensor = policy.last_debug.get("tip_contact_phase")
                    error_tensor = policy.last_debug.get(
                        "tip_persistent_recovery_error"
                    )
                    target_tensor = policy.last_debug.get(
                        "tip_persistent_recovery_target_world"
                    )
                    if phase_tensor is not None:
                        self.contact_phase[:] = (
                            phase_tensor[env_id]
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.int32)
                        )
                    if error_tensor is not None:
                        self.recovery_error_m[:] = (
                            error_tensor[env_id].detach().cpu().numpy()
                        )
                    if target_tensor is not None:
                        self.recovery_target[:] = (
                            target_tensor[env_id].detach().cpu().numpy()
                        )
                if motion_controller.lock_horizontal_lowest_point:
                    self.object_lowest_point[:] = (
                        motion_controller.lowest_point_world[env_id]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                force = _fingertip_force_world(env)[env_id]
                magnitude = torch.linalg.vector_norm(force, dim=-1)
                site_pose = env.scene["robot"].data.site_pose_w[env_id]
                self.tip_pos[:] = torch.stack(
                    [site_pose[index, :3] for index in tip_indices]
                ).detach().cpu().numpy()
                for finger, site_name in enumerate(TIP_SITES):
                    sensor_data = env.scene[f"{site_name}_contact"].data
                    found = bool(
                        sensor_data.found is not None
                        and torch.any(sensor_data.found[env_id] > 0)
                    )
                    self.found[finger] = found
                    if found and sensor_data.pos is not None:
                        self.contact_pos[finger] = (
                            sensor_data.pos[env_id, 0].detach().cpu().numpy()
                        )
                    else:
                        self.contact_pos[finger] = np.nan
                self.loaded[:] = self.found & (
                    magnitude.detach().cpu().numpy() >= args.contact_threshold
                )
                relative_step = (
                    self.step_count - self.actual_motion_start
                    if self.actual_motion_start is not None
                    else -1
                )
                should_record = (
                    relative_step >= record_delay
                    and self.recorded_frames < saved_frames_per_trajectory
                )
                if should_record:
                    self.found_frames += self.found
                    self.loaded_frames += self.loaded
                    self.all4_frames += int(np.all(self.loaded))
                    self.recorded_frames += 1
                if self.step_count % 100 == 0:
                    print(
                        f"[VISUAL] step={self.step_count:04d} moving={moving} "
                        f"found={self.found.astype(int).tolist()} "
                        f"loaded={self.loaded.astype(int).tolist()} "
                        f"phase={self.contact_phase.tolist()} "
                        f"recovery_err_mm="
                        f"{np.round(self.recovery_error_m * 1000, 1).tolist()} "
                        f"force_N={magnitude.detach().cpu().numpy().round(3).tolist()}",
                        flush=True,
                    )
                self.step_count += 1
                return action

        visual_policy = VisualCollectionPolicy()
        base_update_visualizers = env.update_visualizers

        def update_contact_visualizers(visualizer) -> None:
            base_update_visualizers(visualizer)
            if manifold_mode and hasattr(motion_controller, "_palm_path_object"):
                # Draw the planned palm-center path in the *current object
                # frame*.  The dots therefore move with the object and expose
                # the relative manifold trajectory directly.
                object_pos = (
                    env.sim.data.mocap_pos[0, target_mocap_idx]
                    .detach()
                    .cpu()
                    .numpy()
                )
                object_quat = (
                    env.sim.data.mocap_quat[0, target_mocap_idx]
                    .detach()
                    .cpu()
                    .numpy()
                )
                object_rot = R.from_quat(np.roll(object_quat, -1)).as_matrix()
                path = motion_controller._palm_path_object[0]
                samples = np.linspace(
                    motion_controller.motion_start,
                    min(
                        motion_controller.motion_start
                        + motion_controller.motion_length,
                        len(path) - 1,
                    ),
                    num=min(24, max(2, motion_controller.motion_length // 30)),
                ).astype(int)
                for path_index, sample in enumerate(samples):
                    world_point = object_pos + object_rot @ path[sample]
                    visualizer.add_sphere(
                        world_point,
                        radius=0.0035,
                        color=(0.10, 0.85, 1.0, 0.75),
                        label=f"manifold_palm_path_{path_index}",
                    )
            for finger, color in enumerate(TIP_COLORS):
                if visual_policy.found[finger] and np.all(
                    np.isfinite(visual_policy.contact_pos[finger])
                ):
                    live_color = (
                        (0.10, 1.0, 0.15, 1.0)
                        if visual_policy.loaded[finger]
                        else (1.0, 0.50, 0.05, 1.0)
                    )
                    visualizer.add_sphere(
                        visual_policy.contact_pos[finger],
                        radius=0.009,
                        color=live_color,
                        label=f"{TIP_SITES[finger]}_contact",
                    )
                elif np.all(np.isfinite(visual_policy.tip_pos[finger])):
                    visualizer.add_sphere(
                        visual_policy.tip_pos[finger],
                        radius=(
                            0.009
                            if visual_policy.contact_phase[finger] == 2
                            else 0.007
                        ),
                        color=(
                            (1.0, 0.05, 0.85, 1.0)
                            if visual_policy.contact_phase[finger] == 2
                            else (1.0, 0.05, 0.05, 1.0)
                        ),
                        label=f"{TIP_SITES[finger]}_lost",
                    )
                # A small finger-colored center remains visible under the
                # contact marker and makes finger identity unambiguous.
                if np.all(np.isfinite(visual_policy.tip_pos[finger])):
                    visualizer.add_sphere(
                        visual_policy.tip_pos[finger],
                        radius=0.003,
                        color=color,
                    )
                if (
                    visual_policy.contact_phase[finger] == 2
                    and np.all(
                        np.isfinite(visual_policy.recovery_target[finger])
                    )
                ):
                    visualizer.add_sphere(
                        visual_policy.recovery_target[finger],
                        radius=0.004,
                        color=(0.10, 0.95, 1.0, 0.95),
                        label=f"{TIP_SITES[finger]}_recovery_target",
                    )
            if np.all(np.isfinite(visual_policy.object_lowest_point)):
                visualizer.add_sphere(
                    visual_policy.object_lowest_point,
                    radius=0.008,
                    color=(0.05, 1.0, 1.0, 0.95),
                    label="object_horizontal_lowest_patch",
                )

        env.update_visualizers = update_contact_visualizers
        wrapped = RslRlVecEnvWrapper(env)
        # Reproduce a later batch from a seeded raw collection.  Each skipped
        # reset mirrors the exact env/policy/motion reset order in the headless
        # batch loop, including all random SO(3), axis, and speed draws.
        for skipped_batch in range(args.visual_random_batch_index):
            env.reset()
            policy.reset()
            motion_controller.reset()
            print(
                "[VISUAL] skipped seeded random batch "
                f"{skipped_batch + 1}/{args.visual_random_batch_index}"
            )
        env.reset()
        visual_policy.reset()
        print(
            f"[INFO] visual collection object={args.object_id} viewer={args.viewer} "
            f"steps={args.trajectory_length} device={device}"
        )
        print(
            "[INFO] contact markers: green=loaded, orange=geometry-only/weak, "
            "red=transient loss, magenta=persistent recovery"
        )
        try:
            if args.viewer == "native":
                NativeMujocoViewer(
                    wrapped, visual_policy, frame_rate=args.viewer_fps
                ).run(num_steps=args.trajectory_length + args.max_prep_wait_steps)
            else:
                ViserPlayViewer(
                    wrapped, visual_policy, frame_rate=args.viewer_fps
                ).run(num_steps=args.trajectory_length + args.max_prep_wait_steps)
        finally:
            saved_steps = max(visual_policy.recorded_frames, 1)
            print(
                "[VISUAL-RESULT] "
                f"frames={saved_steps} all4={visual_policy.all4_frames / saved_steps:.1%} "
                f"per_tip={(visual_policy.loaded_frames / saved_steps).round(3).tolist()}"
            )
            wrapped.close()
            # MuJoCo's passive viewer tears down its GLX context on a
            # background render thread. Exiting immediately after close()
            # can race that teardown: the process closes the X11 display
            # connection while the render thread still calls into X11
            # (XConnectionNumber) -> SIGSEGV at interpreter exit (observed
            # with NVIDIA GLX + conda libX11). Minimal reproducer segfaults
            # at 0s wait, exits cleanly at 0.25s; keep a 2s margin.
            time.sleep(2.0)
        return

    filename = args.filename or f"mcc_tip_{datetime.now():%Y%m%d_%H%M%S}"
    output = Path("mcc_finger_compliance_control/data/trajectories") / f"{filename}.h5"
    logger = StreamingH5(
        output, args.num_envs, q_dof=23 if palm_orbit_mode else 22
    )

    total_frames = saved_frames_per_trajectory * collected_target
    print(f"[INFO] task={TASK_ID} device={device} accepted_frames={total_frames} output={output}")
    if args.teacher_controller == "fullhand_mcc":
        print(
            "[INFO] FullHandMCC uses privileged oracle surface normals and "
            "live fingertip 3-D force feedback"
        )
    else:
        print(
            "[INFO] fixed-pregrasp controller uses ZERO measured force; "
            "fingertip force is record-only"
        )
    print(
        "[INFO] initial object orientation mode="
        f"{args.initial_orientation_mode}"
        + (
            f" jitter=+/-{args.initial_orientation_jitter_deg:.1f}deg"
            if args.initial_orientation_mode == "jitter"
            else ""
        )
    )
    if args.fixed_motion_start:
        if planner_inverse_mode:
            print(
                "[INFO] planner_inverse: object is placed at step "
                f"{args.motion_start}, held for {args.planner_settle_steps} "
                "steps, then motion/recording begin"
            )
        else:
            print(
                "[INFO] fixed motion start: no four-tip prep wait; motion and "
                f"recording start at step {args.motion_start}"
            )
    if args.online_quality_gate:
        print(
            "[INFO] ONLINE strict gate: all 4 complete fingertip geoms need "
            f"|F_3D| >= {args.contact_threshold:.3f} N on every saved frame"
        )
    else:
        print(
            "[INFO] RAW mode: no online rejection; run filter_trajectories.py later | "
            f"requested={args.max_trajectories}, actual={collected_target}, "
            f"parallel_envs={args.num_envs}"
        )

    hand_q_start = 7 if palm_orbit_mode else 6
    accepted = 0
    attempts = 0
    progress = tqdm(
        total=max_attempts * args.trajectory_length,
        desc="Collecting raw" if not args.online_quality_gate else "Collecting strict",
        unit="step",
        dynamic_ncols=True,
    )
    try:
        while accepted < collected_target and attempts < max_attempts:
            candidate_id = attempts
            attempts += 1
            obs, _ = env.reset()
            policy.reset()
            motion_controller.reset()
            motion_controller.motion_start = args.motion_start
            if translation_enabled:
                print(
                    "[TRANSLATION] "
                    f"axis={motion_controller.translation_axis_labels[0]} "
                    f"amplitude={motion_controller.translation_amplitudes[0].item():.4f}m "
                    f"speed={motion_controller.translation_speeds[0].item():.4f}m/s"
                )
            candidate_start = logger.step
            quality_forces: list[np.ndarray] = []
            quality_contacts: list[np.ndarray] = []
            quality_object_pose: list[np.ndarray] = []
            quality_motion_active: list[bool] = []
            quality_motion_window: list[bool] = []
            actual_motion_start: int | None = None
            actual_record_start: int | None = None
            planner_pose_initialized = False
            planner_settle_until: int | None = None
            recorded_frames = 0
            prep_failed = False
            prev_all4 = torch.ones(
                args.num_envs, dtype=torch.bool, device=device
            )

            for episode_step in range(
                args.trajectory_length + args.max_prep_wait_steps
            ):
                ready = (
                    policy.ready_for_motion
                    if args.teacher_controller == "fullhand_mcc"
                    else True
                )
                if planner_inverse_mode:
                    if (
                        not planner_pose_initialized
                        and episode_step >= args.motion_start
                        and (args.fixed_motion_start or ready)
                    ):
                        planner_settle_until = (
                            episode_step + args.planner_settle_steps
                        )
                        # The inverse path's time parameter is tied to
                        # ``motion_controller.motion_start``.  Set the
                        # post-settle start before building the path; doing
                        # this afterwards compresses the executed trajectory
                        # and makes a requested 180-degree plan stop early.
                        motion_controller.motion_start = planner_settle_until
                        motion_controller.reanchor_from_current_state()
                        planner_pose_initialized = True
                        print(
                            "[PLANNER-INVERSE] object initialized at step "
                            f"{episode_step}; settling until {planner_settle_until}"
                        )
                    if (
                        planner_pose_initialized
                        and actual_motion_start is None
                        and planner_settle_until is not None
                        and episode_step >= planner_settle_until
                    ):
                        actual_motion_start = episode_step
                        actual_record_start = episode_step + record_delay
                        print(
                            "[START] planner motion starts at simulator step "
                            f"{actual_motion_start}; recording starts at "
                            f"{actual_record_start} after settle"
                        )
                elif (
                    actual_motion_start is None
                    and episode_step >= args.motion_start
                    and (args.fixed_motion_start or ready)
                ):
                    actual_motion_start = episode_step
                    actual_record_start = episode_step + record_delay
                    motion_controller.motion_start = episode_step
                    if manifold_mode:
                        motion_controller.reanchor_from_current_state()
                    print(
                        "[START] motion starts at simulator step "
                        f"{actual_motion_start}; recording starts at "
                        f"{actual_record_start}; mode="
                        f"{'fixed_schedule' if args.fixed_motion_start else 'contact_settled'}"
                    )
                if (
                    not planner_inverse_mode
                    and not args.fixed_motion_start
                    and
                    actual_motion_start is None
                    and episode_step
                    >= args.motion_start + args.max_prep_wait_steps
                ):
                    ready_mask = np.asarray(
                        policy.motion_ready_mask, dtype=bool
                    ).reshape(-1)
                    pending_envs = np.flatnonzero(~ready_mask).tolist()
                    print(
                        "[PREP-REJECT] "
                        f"candidate={candidate_id} did not settle within "
                        f"{args.max_prep_wait_steps} extra steps; "
                        f"pending_envs={pending_envs}. Resampling the complete "
                        "parallel batch; no prep frames were recorded.",
                        flush=True,
                    )
                    prep_failed = True
                    break
                contact_ready = (
                    None
                    if args.no_contact_gate
                    else (
                        policy.motion_ready_mask
                        if args.teacher_controller == "fullhand_mcc"
                        else (prev_all4 if contact_gated_motion else None)
                    )
                )
                if orbit_mode:
                    # Object stays still; the palm orbit is stepped inside the
                    # policy on the same contact gate.
                    moving = policy.orbit_moving
                elif planner_inverse_mode and planner_pose_initialized:
                    # Keep path[0] during the contact-settling interval;
                    # motion_controller switches to path progression at the
                    # delayed motion_start assigned above.
                    moving = motion_controller.step(
                        episode_step, contact_ready=contact_ready
                    )
                elif actual_motion_start is not None:
                    moving = motion_controller.step(
                        episode_step, contact_ready=contact_ready
                    )
                else:
                    moving = False

                action = policy(obs)
                obs, *_ = env.step(action)
                progress.update(1)
                robot = env.scene["robot"]
                body_pose = robot.data.body_link_pose_w
                palm_pose = body_pose[:, palm_idx, :]
                site_pose = robot.data.site_pose_w
                tip_pose = torch.stack([site_pose[:, idx, :] for idx in tip_indices], dim=1)
                tip_force_local = obs["finger"][:, :12].reshape(args.num_envs, 4, 3)
                tip_force = _fingertip_force_world(env)
                contact_pos = torch.zeros_like(tip_force)
                contact_normal = torch.zeros_like(tip_force)
                contact_dist = torch.zeros((args.num_envs, 4), device=device)
                contact_found = torch.zeros(
                    (args.num_envs, 4), dtype=torch.bool, device=device
                )
                for tip_id, site_name in enumerate(TIP_SITES):
                    sensor_data = env.scene[f"{site_name}_contact"].data
                    if sensor_data.found is not None:
                        found = torch.any(sensor_data.found > 0, dim=1)
                    else:
                        found = torch.zeros(
                            args.num_envs, dtype=torch.bool, device=device
                        )
                    contact_found[:, tip_id] = found
                    if sensor_data.pos is not None:
                        selected_pos = sensor_data.pos[:, 0]
                        contact_pos[:, tip_id] = torch.where(
                            found.unsqueeze(-1), selected_pos, torch.zeros_like(selected_pos)
                        )
                    if sensor_data.normal is not None:
                        selected_normal = sensor_data.normal[:, 0]
                        contact_normal[:, tip_id] = torch.where(
                            found.unsqueeze(-1), selected_normal, torch.zeros_like(selected_normal)
                        )
                    if sensor_data.dist is not None:
                        selected_dist = sensor_data.dist[:, 0]
                        contact_dist[:, tip_id] = torch.where(
                            found, selected_dist, torch.zeros_like(selected_dist)
                        )
                # Contact gate for the fixed teacher: all four tips loaded.
                prev_all4 = (
                    contact_found
                    & (
                        torch.linalg.vector_norm(tip_force, dim=-1)
                        >= args.contact_threshold
                    )
                ).all(dim=1)
                obj_pose = torch.cat(
                    (
                        env.sim.data.mocap_pos[:, target_mocap_idx, :],
                        env.sim.data.mocap_quat[:, target_mocap_idx, :],
                    ),
                    dim=-1,
                )
                # Replace MuJoCo seam-contaminated contact normals with smooth
                # oracle normals estimated from the source OBJ point cloud at
                # the same contact positions (world -> object -> world).
                if mesh_normal_oracle is not None and contact_found.any():
                    found_flat = contact_found.reshape(-1)
                    env_idx = (
                        torch.arange(args.num_envs, device=device)
                        .repeat_interleave(4)[found_flat]
                    )
                    pts_world = contact_pos.reshape(-1, 3)[found_flat].cpu().numpy()
                    obj_pos = obj_pose[:, :3][env_idx].cpu().numpy()
                    obj_quat = obj_pose[:, 3:7][env_idx].cpu().numpy()
                    oracle_normals = torch.from_numpy(
                        mesh_normal_oracle.query_world(
                            pts_world, obj_pos, obj_quat
                        )
                    ).to(device=device, dtype=contact_normal.dtype)
                    contact_normal.reshape(-1, 3)[found_flat] = oracle_normals
                object_angular_velocity_world = (
                    _wxyz_apply(
                        obj_pose[:, 3:7], motion_controller.rotation_axes
                    )
                    * motion_controller.current_angular_speeds.unsqueeze(-1)
                    if moving and motion_controller.rotation_enabled
                    else torch.zeros_like(motion_controller.rotation_axes)
                )
                debug = policy.last_debug
                tip_force_magnitude = torch.linalg.vector_norm(tip_force, dim=-1)
                should_record = (
                    actual_record_start is not None
                    and episode_step >= actual_record_start
                    and recorded_frames < saved_frames_per_trajectory
                )
                if should_record:
                    quality_forces.append(
                        tip_force_magnitude[0].detach().cpu().numpy().copy()
                    )
                    quality_contacts.append(
                        contact_found[0].detach().cpu().numpy().copy()
                    )
                    quality_object_pose.append(
                        obj_pose[0].detach().cpu().numpy().copy()
                    )
                    quality_motion_active.append(
                        bool(motion_controller.motion_active[0])
                    )
                    quality_motion_window.append(
                        bool(
                            actual_motion_start is not None
                            and actual_motion_start
                            <= episode_step
                            < actual_motion_start + args.motion_length
                        )
                    )
                    logger.append(
                        {
                        "time": np.full(args.num_envs, logger.step * dt),
                        "episode_id": (
                            accepted + np.arange(args.num_envs, dtype=np.int32)
                        ),
                        "episode_step": np.full(args.num_envs, episode_step, dtype=np.int32),
                        "record_step": np.full(
                            args.num_envs, recorded_frames, dtype=np.int32
                        ),
                        "actual_motion_start_step": np.full(
                            args.num_envs, actual_motion_start, dtype=np.int32
                        ),
                        "actual_record_start_step": np.full(
                            args.num_envs, actual_record_start, dtype=np.int32
                        ),
                        "q": robot.data.joint_pos,
                        "qvel": robot.data.joint_vel,
                        "q_hand": robot.data.joint_pos[
                            :, hand_q_start : hand_q_start + 16
                        ],
                        "q_pre": debug["q_pre"],
                        "q_ref": debug["q_ref"],
                        "arm_q_ref": debug["palm_arm_q_ref"],
                        "palm_x_des": debug["palm_x_des"],
                        "palm_x_ref": debug["palm_x_ref"],
                        "fixed_palm_target": debug["fixed_palm_target"],
                        "palm_control_pos_world": debug["palm_site_pos"],
                        "action": action,
                        "object_pose_world": obj_pose,
                        "palm_pose_world": palm_pose,
                        "fingertip_pose_world": tip_pose,
                        "fingertip_force_world": tip_force,
                        "fingertip_force_local": tip_force_local,
                        "fingertip_contact_pos_world": contact_pos,
                        "fingertip_contact_normal_world": contact_normal,
                        "fingertip_contact_dist": contact_dist,
                        "fingertip_collision_found": contact_found.float(),
                        "fingertip_contact": (
                            contact_found
                            & (tip_force_magnitude >= args.contact_threshold)
                        ).float(),
                        "tip_x_des_world": debug["tip_x_des"],
                        "tip_x_ref_world": debug["tip_x_ref"],
                        "tip_x_ik_world": debug["tip_x_ik"],
                        "tip_x_des_palm": debug["tip_x_des_palm"],
                        "tip_x_ref_palm": debug["tip_x_ref_palm"],
                        "oracle_surface_normal_world": debug.get(
                            "tip_surface_normal_world", torch.zeros_like(tip_force)
                        ),
                        "fullhand_normal_force": debug.get(
                            "tip_normal_force",
                            torch.zeros((args.num_envs, 4), device=device),
                        ),
                        "fullhand_normal_offset": debug.get(
                            "tip_normal_offset",
                            torch.zeros((args.num_envs, 4), device=device),
                        ),
                        "fullhand_contact_phase": debug.get(
                            "tip_contact_phase",
                            torch.zeros((args.num_envs, 4), device=device),
                        ),
                        "fullhand_contact_loss_streak": debug.get(
                            "tip_contact_loss_streak",
                            torch.zeros((args.num_envs, 4), device=device),
                        ),
                        "fullhand_transient_search_offset": debug.get(
                            "tip_transient_search_offset",
                            torch.zeros((args.num_envs, 4), device=device),
                        ),
                        "fullhand_persistent_recovery_error": debug.get(
                            "tip_persistent_recovery_error",
                            torch.zeros((args.num_envs, 4), device=device),
                        ),
                        "fullhand_persistent_recovery_joint_step": debug.get(
                            "tip_persistent_recovery_joint_step",
                            torch.zeros((args.num_envs, 16), device=device),
                        ),
                        "fullhand_persistent_recovery_target_world": debug.get(
                            "tip_persistent_recovery_target_world",
                            torch.zeros((args.num_envs, 4, 3), device=device),
                        ),
                        "fullhand_persistent_recovery_control_point_world": debug.get(
                            "tip_persistent_recovery_control_point_world",
                            torch.zeros((args.num_envs, 4, 3), device=device),
                        ),
                        "fullhand_anchor_valid": debug.get(
                            "tip_anchor_valid",
                            torch.zeros((args.num_envs, 4), device=device),
                        ).float(),
                        "fullhand_contact_calibrated": debug.get(
                            "fullhand_contact_calibrated",
                            torch.zeros(args.num_envs, device=device),
                        ).float(),
                        "contact_qp_joint_velocity": debug.get(
                            "contact_qp_joint_velocity",
                            torch.zeros((args.num_envs, 16), device=device),
                        ),
                        "contact_qp_target_tip_velocity_palm": debug.get(
                            "contact_qp_target_tip_velocity_palm",
                            torch.zeros((args.num_envs, 4, 3), device=device),
                        ),
                        "contact_qp_normal_velocity_error": debug.get(
                            "contact_qp_normal_velocity_error",
                            torch.zeros((args.num_envs, 4), device=device),
                        ),
                        "contact_qp_adjacent_lateral_distance": debug.get(
                            "contact_qp_adjacent_lateral_distance",
                            torch.zeros((args.num_envs, 2), device=device),
                        ),
                        "contact_qp_separation_active": debug.get(
                            "contact_qp_separation_active",
                            torch.zeros((args.num_envs, 2), device=device),
                        ),
                        "contact_qp_exit_flag": debug.get(
                            "contact_qp_exit_flag",
                            torch.zeros(args.num_envs, device=device),
                        ),
                        "contact_qp_solve_time_us": debug.get(
                            "contact_qp_solve_time_us",
                            torch.zeros(args.num_envs, device=device),
                        ),
                        "palm_surface_normal_world": debug.get(
                            "palm_surface_normal_world",
                            torch.zeros((args.num_envs, 3), device=device),
                        ),
                        "palm_surface_standoff_m": debug.get(
                            "palm_surface_standoff_m",
                            torch.zeros(args.num_envs, device=device),
                        ),
                        "palm_protection_standoff_m": debug.get(
                            "palm_protection_standoff_m",
                            torch.zeros(
                                (args.num_envs, len(PALM_PROTECTION_POINTS_LOCAL)),
                                device=device,
                            ),
                        ),
                        "palm_protection_clearance_m": debug.get(
                            "palm_protection_clearance_m",
                            torch.zeros(
                                (args.num_envs, len(PALM_PROTECTION_POINTS_LOCAL)),
                                device=device,
                            ),
                        ),
                        "palm_predicted_clearance_m": debug.get(
                            "palm_predicted_clearance_m",
                            torch.zeros(args.num_envs, device=device),
                        ),
                        "palm_predicted_intrusion_m": debug.get(
                            "palm_predicted_intrusion_m",
                            torch.zeros(args.num_envs, device=device),
                        ),
                        "hand_shape_deviation_rad": debug.get(
                            "hand_shape_deviation_rad",
                            torch.zeros(args.num_envs, device=device),
                        ),
                        "hand_shape_retreat_m": debug.get(
                            "hand_shape_retreat_m",
                            torch.zeros(args.num_envs, device=device),
                        ),
                        "palm_surface_follow_valid": debug.get(
                            "palm_surface_follow_valid",
                            torch.zeros(args.num_envs, device=device),
                        ).float(),
                        "palm_surface_query_world": debug.get(
                            "palm_surface_query_world",
                            torch.zeros((args.num_envs, 3), device=device),
                        ),
                        "palm_preview_object_pose_world": debug.get(
                            "palm_preview_object_pose_world",
                            torch.zeros((args.num_envs, 7), device=device),
                        ),
                        "object_angular_velocity_world": (
                            object_angular_velocity_world
                        ),
                        "object_rotation_axis_local": (
                            motion_controller.rotation_axes
                        ),
                        "object_rotation_speed_target_rad_s": (
                            motion_controller.angular_speeds
                        ),
                        "object_translation_axis_world": (
                            motion_controller.translation_axes
                            if translation_enabled
                            else torch.zeros_like(
                                motion_controller.translation_axes
                            )
                        ),
                        "object_translation_amplitude_m": (
                            motion_controller.translation_amplitudes
                            if translation_enabled
                            else torch.zeros_like(
                                motion_controller.translation_amplitudes
                            )
                        ),
                        "object_translation_speed_target_m_s": (
                            motion_controller.translation_speeds
                            if translation_enabled
                            else torch.zeros_like(
                                motion_controller.translation_speeds
                            )
                        ),
                        "object_horizontal_lowest_point_world": (
                            motion_controller.lowest_point_world
                        ),
                        "object_lowest_point_anchor_world": (
                            motion_controller.lowest_point_anchor_world
                        ),
                        "object_lowest_point_compensation_world": (
                            motion_controller.lowest_point_compensation_world
                        ),
                        "object_lowest_point_follow_velocity_world": (
                            motion_controller.lowest_point_follow_velocity_world
                        ),
                        "object_motion_active": motion_controller.motion_active.float(),
                        "object_motion_contact_ready": (
                            motion_controller.motion_contact_ready.float()
                        ),
                        "object_motion_schedule_step": (
                            motion_controller.motion_schedule_step
                        ),
                        "object_segment_move_steps": motion_controller.segment_move_steps,
                        "object_segment_hold_steps": motion_controller.segment_hold_steps,
                        "palm_x_des_orbit": (
                            orbit_controller.current_x_des
                            if orbit_mode
                            else torch.zeros((args.num_envs, 6), device=device)
                        ),
                        "orbit_phase_rad": (
                            orbit_controller.phase
                            if orbit_mode
                            else torch.zeros(args.num_envs, device=device)
                        ),
                        "orbit_axis_world": (
                            orbit_controller.orbit_axis_world
                            if orbit_mode
                            else torch.zeros((args.num_envs, 3), device=device)
                        ),
                        "orbit_amplitude_rad": (
                            orbit_controller.amplitudes_rad
                            if orbit_mode and not palm_orbit_mode
                            else torch.zeros(args.num_envs, device=device)
                        ),
                        "orbit_speed_target_rad_s": (
                            orbit_controller.angular_speeds
                            if orbit_mode
                            else torch.zeros(args.num_envs, device=device)
                        ),
                        "orbit_moving": (
                            orbit_controller.motion_active.float()
                            if orbit_mode
                            else torch.zeros(args.num_envs, device=device)
                        ),
                        "orbit_surface_clearance_m": (
                            orbit_controller.surface_clearance
                            if palm_orbit_mode
                            else torch.zeros(args.num_envs, device=device)
                        ),
                        "orbit_travel_total_rad": (
                            orbit_controller.phase - orbit_controller.theta0
                            if palm_orbit_mode
                            else torch.zeros(args.num_envs, device=device)
                        ),
                        }
                    )
                    recorded_frames += 1
                    if recorded_frames >= saved_frames_per_trajectory:
                        break

            if prep_failed:
                # Normally candidate_start == logger.step because recording
                # cannot begin before every environment is ready.  Truncation
                # keeps this invariant explicit if the recording logic changes.
                logger.truncate(candidate_start)
                progress.set_postfix(
                    batch=f"retry {candidate_id + 1}/{max_attempts}",
                    trajectories=f"{accepted}/{collected_target}",
                )
                continue

            if recorded_frames != saved_frames_per_trajectory:
                raise RuntimeError(
                    f"Recorded {recorded_frames}/{saved_frames_per_trajectory} "
                    "post-prep frames"
                )

            pose_quality = np.asarray(quality_object_pose, dtype=np.float64)
            realized_rotation_deg = _quaternion_path_angle_deg(
                pose_quality[:, 3:7]
            )
            realized_translation_m = float(
                np.sum(
                    np.linalg.norm(
                        np.diff(pose_quality[:, :3], axis=0), axis=1
                    )
                )
            )
            active_flags = np.asarray(quality_motion_active, dtype=bool)
            motion_window_flags = np.asarray(quality_motion_window, dtype=bool)
            active_motion_ratio = float(
                np.mean(active_flags[motion_window_flags])
                if np.any(motion_window_flags)
                else 0.0
            )
            print(
                "[MOTION-QUALITY] "
                f"candidate={candidate_id} "
                f"rotation_path={realized_rotation_deg:.1f}deg "
                f"translation_path={1000.0 * realized_translation_m:.1f}mm "
                f"active_ratio={active_motion_ratio:.3f}",
                flush=True,
            )

            if not args.online_quality_gate:
                accepted += args.num_envs
                progress.set_postfix(
                    batch=candidate_id + 1,
                    trajectories=f"{accepted}/{collected_target}",
                )
                continue

            quality = np.asarray(quality_forces, dtype=np.float32)
            contact_quality = np.asarray(quality_contacts, dtype=bool)
            loaded_contact = contact_quality & (quality >= args.contact_threshold)
            per_tip_contact = np.mean(loaded_contact, axis=0)
            all_four_contact = np.all(loaded_contact, axis=1)
            contact_pass = bool(np.all(all_four_contact))
            rotation_pass = (
                not rotation_enabled
                or realized_rotation_deg >= args.min_realized_rotation_deg
            )
            motion_pass = active_motion_ratio >= args.min_active_motion_ratio
            strict_pass = bool(contact_pass and rotation_pass and motion_pass)
            min_force = np.min(quality, axis=0)
            first_loss = np.flatnonzero(~all_four_contact)
            first_loss_step = (
                int(actual_record_start + first_loss[0])
                if first_loss.size and actual_record_start is not None
                else None
            )
            summary = (
                f"candidate={candidate_id} pass={strict_pass} "
                f"tip_contact={np.round(per_tip_contact, 4).tolist()} "
                f"all4={float(np.mean(all_four_contact)):.4f} "
                f"rotation_deg={realized_rotation_deg:.1f} "
                f"translation_mm={1000.0 * realized_translation_m:.1f} "
                f"active_ratio={active_motion_ratio:.3f} "
                f"motion_pass={motion_pass and rotation_pass} "
                f"min_force_N={np.round(min_force, 3).tolist()} "
                f"first_loss_step={first_loss_step}"
            )
            if strict_pass:
                accepted += 1
                print(f"[ACCEPT] {summary} accepted={accepted}/{args.max_trajectories}")
            else:
                logger.truncate(candidate_start)
                print(f"[REJECT] {summary}")
    finally:
        progress.close()
        logger.close(
            {
                "task_id": TASK_ID,
                "schema_version": "mcc_tip_v1",
                "teacher_controller": args.teacher_controller,
                "force_feedback_enabled": args.teacher_controller == "fullhand_mcc",
                "privileged_surface_oracle": args.teacher_controller == "fullhand_mcc",
                "differential_contact_qp": bool(args.differential_contact_qp),
                "contact_recovery_state_machine": (
                    args.teacher_controller == "fullhand_mcc"
                ),
                "contact_recovery_transient_frames": (
                    args.contact_transient_loss_frames
                ),
                "contact_recovery_confirm_frames": (
                    args.contact_recovery_confirm_frames
                ),
                "contact_recovery_distance_cap_m": "none_absolute_surface_target",
                "contact_recovery_max_joint_step_rad": (
                    args.persistent_recovery_max_joint_step_rad
                ),
                "surface_preload_m": surface_preload_m,
                "surface_target_mode": args.surface_target_mode,
                "force_frame": "world",
                "pose_quaternion_order": "wxyz",
                "control_dt": dt,
                "motion_start": args.motion_start,
                "record_start_step": record_start,
                "motion_start_semantics": (
                    "fixed_schedule"
                    if args.fixed_motion_start
                    else "earliest_start_after_prep"
                ),
                "wait_for_four_tip_prep": not args.fixed_motion_start,
                "record_start_delay_steps": record_delay,
                "effective_record_delay_steps": effective_record_delay,
                "max_prep_wait_steps": args.max_prep_wait_steps,
                "trajectory_length": args.trajectory_length,
                "num_trajectories": accepted,
                "candidate_attempts": attempts,
                "strict_four_tip_continuous_contact": args.online_quality_gate,
                "minimum_realized_rotation_deg": args.min_realized_rotation_deg,
                "minimum_active_motion_ratio": args.min_active_motion_ratio,
                "contact_gate": "full_fingertip_geom_found_and_3d_force_magnitude",
                "contact_threshold": args.contact_threshold,
                "contact_stiffness": args.contact_stiffness,
                "contact_damping": args.contact_damping,
                "contact_transition_width_m": args.contact_transition_width_m,
                "fullhand_finger_stiffness": args.fullhand_finger_stiffness,
                "fullhand_finger_damping": args.fullhand_finger_damping,
                "fullhand_finger_effort_limit": args.fullhand_finger_effort_limit,
                "fullhand_force_servo_gain": args.fullhand_force_servo_gain,
                "physics_substeps": args.physics_substeps,
                "initial_orientation_mode": args.initial_orientation_mode,
                "object_angular_velocity_frame": "world",
                "object_rotation_axis_frame": "object_local",
                "object_translation_axis_frame": "world",
                "motion_mode": args.motion_mode,
                "manifold_fixed_palm": bool(manifold_mode),
                "planner_inverse": bool(planner_inverse_mode),
                "planner_settle_steps": int(args.planner_settle_steps),
                "planner_file": (
                    str(args.planner_file) if args.planner_file is not None else ""
                ),
                "manifold_axis_local": np.asarray(manifold_axis, dtype=np.float64),
                "manifold_angle_deg": float(args.manifold_angle_deg),
                "manifold_direction": int(args.manifold_direction),
                "horizontal_lowest_point_lock": bool(
                    args.lock_horizontal_lowest_point_to_palm
                ),
                "horizontal_lowest_point_clearance_m": float(
                    args.lowest_point_clearance_m
                ),
                "horizontal_lowest_point_band_m": float(
                    args.lowest_point_band_m
                ),
                "horizontal_lowest_point_follow_max_speed_m_s": float(
                    args.lowest_point_follow_max_speed_m_s
                ),
                "horizontal_lowest_point_follow_time_constant_s": float(
                    args.lowest_point_follow_time_constant_s
                ),
                "object_initial_z_offset_m": float(
                    args.object_initial_z_offset_m
                ),
                "contact_gated_motion": contact_gated_motion,
                "default_segment_move_steps": default_segment_move_steps,
                "default_segment_hold_steps": default_segment_hold_steps,
                "rotation_allowed_axes": ",".join(rot_allowed_axes),
                "translation_allowed_axes": ",".join(trans_allowed_axes),
                "rotation_acceleration_time_s": float(
                    rot_cfg.get("acceleration_time_s", 0.0)
                ),
                "object_id": args.object_id,
                "initial_orientation_jitter_deg": (
                    args.initial_orientation_jitter_deg
                ),
                "seed": args.seed,
            }
        )
        env.close()
    if accepted < collected_target:
        print(
            f"[WARNING] saved {accepted}/{collected_target} trajectories "
            f"after {attempts} attempts: {output}"
        )
    else:
        mode = "strict" if args.online_quality_gate else "raw"
        print(f"[SUCCESS] saved {accepted} {mode} trajectories to {output}")


if __name__ == "__main__":
    main()
