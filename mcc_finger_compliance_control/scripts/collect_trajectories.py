from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import h5py
import mujoco
import numpy as np
import torch
from tqdm.auto import tqdm

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.sensor import ContactSensor
from mjlab.tasks.leaphand.leaphand_mcc_finger_env_cfg import (
    MCCLeapHandPositionControlCfg,
    mcc_finger_contact_env_cfg,
)
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

from object_catalog import (
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
        self.initial_pos = sim_data.mocap_pos[:, self.target_mocap_idx, :].clone()
        nominal_quat = sim_data.mocap_quat[:, self.target_mocap_idx, :].clone()
        self.initial_quat = _sample_initial_quaternion(
            nominal_quat,
            self.initial_orientation_mode,
            self.initial_orientation_jitter_deg,
        )
        sim_data.mocap_pos[:, self.target_mocap_idx, :] = self.initial_pos
        sim_data.mocap_quat[:, self.target_mocap_idx, :] = self.initial_quat

        self.rotation_axes, self.rotation_axis_labels = _sample_motion_axes(
            self.initial_quat,
            self.rotation_allowed_axes,
            output_frame="object",
            sampling=self.axis_sampling,
        )
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
        return bool(torch.any(self.motion_active))


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
    ) -> None:
        self.env = env
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
        self.contact_search_step_m = float(contact_search_step_m)
        self.contact_search_step_rad = float(contact_search_step_rad)
        self.contact_search_limit_rad = float(contact_search_limit_rad)
        self.controllers = [
            FullHandMCCFingerController(
                FullHandMCCFingerConfig(control_dt=0.01)
            )
            for _ in range(env.num_envs)
        ]
        self.oracles = [GeometrySurfaceOracle(object_config) for _ in range(env.num_envs)]
        self.anchor_points_object = np.zeros((env.num_envs, 4, 3), dtype=np.float64)
        self.anchor_valid = np.zeros((env.num_envs, 4), dtype=bool)
        self.site_standoff_m = np.zeros((env.num_envs, 4), dtype=np.float64)
        self.loaded_streak = np.zeros((env.num_envs, 4), dtype=np.int32)
        self.contact_settle_streak = np.zeros(env.num_envs, dtype=np.int32)
        self.contact_calibrated = np.zeros(env.num_envs, dtype=bool)
        self.precontact_base_q = np.zeros((env.num_envs, 16), dtype=np.float64)
        self.precontact_base_valid = np.zeros(env.num_envs, dtype=bool)
        self.precontact_closure = np.zeros((env.num_envs, 16), dtype=np.float64)
        self.planner_query_world = np.zeros((env.num_envs, 4, 3), dtype=np.float64)
        self.planner_query_valid = np.zeros(env.num_envs, dtype=bool)
        self.last_debug: dict[str, torch.Tensor] = {}
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

    def reset(self) -> None:
        self.palm_controller.reset()
        for controller in self.controllers:
            controller.reset()
        self.anchor_points_object.fill(0.0)
        self.anchor_valid.fill(False)
        self.site_standoff_m.fill(0.0)
        self.loaded_streak.fill(0)
        self.contact_settle_streak.fill(0)
        self.contact_calibrated.fill(False)
        self.precontact_base_q.fill(0.0)
        self.precontact_base_valid.fill(False)
        self.precontact_closure.fill(0.0)
        self.planner_query_world.fill(0.0)
        self.planner_query_valid.fill(False)

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
        force = _fingertip_force_world(self.env).detach().cpu().numpy()
        found = np.zeros((self.env.num_envs, 4), dtype=bool)
        positions = np.zeros((self.env.num_envs, 4, 3), dtype=np.float32)
        for finger, site_name in enumerate(TIP_SITES):
            sensor_data = self.env.scene[f"{site_name}_contact"].data
            if sensor_data.found is None:
                continue
            finger_found = (
                sensor_data.found > 0
            ).any(dim=1)
            found[:, finger] = finger_found.detach().cpu().numpy()
            if sensor_data.pos is not None:
                positions[:, finger] = torch.where(
                    finger_found[:, None],
                    sensor_data.pos[:, 0],
                    torch.zeros_like(sensor_data.pos[:, 0]),
                ).detach().cpu().numpy()
        return force, found, positions

    def __call__(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        palm_output = self.palm_controller({"palm": obs["palm"]})
        palm_debug = self.palm_controller.last_debug
        palm_in_prep = palm_debug["palm_in_prep"][:, 0] > 0.5

        robot = self.env.scene["robot"]
        q_hand_batch = robot.data.joint_pos[:, 6:22].detach().cpu().numpy()
        palm_pose_batch = robot.data.body_link_pose_w[:, self.palm_idx].detach().cpu().numpy()
        site_pose = robot.data.site_pose_w
        tip_world_batch = torch.stack(
            [site_pose[:, index, :3] for index in self.tip_indices], dim=1
        ).detach().cpu().numpy()
        object_pos_batch = self.env.sim.data.mocap_pos[:, self.target_mocap_idx].detach().cpu().numpy()
        object_quat_batch = self.env.sim.data.mocap_quat[:, self.target_mocap_idx].detach().cpu().numpy()
        force_batch, found_batch, contact_pos_batch = self._live_contacts()

        q_command_batch = q_hand_batch.copy().astype(np.float32)
        tip_surface_world = np.zeros((self.env.num_envs, 4, 3), dtype=np.float32)
        tip_reference_world = np.zeros_like(tip_surface_world)
        tip_ik_world = np.zeros_like(tip_surface_world)
        tip_surface_palm = np.zeros_like(tip_surface_world)
        tip_reference_palm = np.zeros_like(tip_surface_world)
        normal_world_batch = np.zeros_like(tip_surface_world)
        normal_force_batch = np.zeros((self.env.num_envs, 4), dtype=np.float32)
        normal_offset_batch = np.zeros_like(normal_force_batch)

        for env_id, (controller, oracle) in enumerate(
            zip(self.controllers, self.oracles, strict=True)
        ):
            oracle.set_pose(object_pos_batch[env_id], object_quat_batch[env_id])
            query_world = (
                self.planner_query_world[env_id]
                if self.planner_query_valid[env_id]
                else tip_world_batch[env_id]
            )
            nearest = oracle.observe(query_world)
            rotation = oracle.rotation_world_from_object
            center = oracle.center_world
            magnitude = np.linalg.norm(force_batch[env_id], axis=-1)
            loaded = found_batch[env_id] & (
                magnitude >= self.anchor_force_threshold
            )
            precontact_loaded = found_batch[env_id] & (
                magnitude >= self.precontact_force_threshold
            )

            if bool(palm_in_prep[env_id]):
                self.loaded_streak[env_id].fill(0)
            else:
                self.loaded_streak[env_id] = np.where(
                    loaded,
                    self.loaded_streak[env_id] + 1,
                    0,
                )

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
                    nearest.points_world[new_anchor], center, rotation
                )
                self.site_standoff_m[env_id, new_anchor] = nearest.signed_distance[
                    new_anchor
                ]
                self.anchor_valid[env_id, new_anchor] = True

            transported = nearest.points_world.astype(np.float64)
            valid = self.anchor_valid[env_id]
            if self.surface_target_mode == "object_anchor" and np.any(valid):
                transported[valid] = self._object_to_world(
                    self.anchor_points_object[env_id, valid], center, rotation
                )
            target_surface = oracle.observe(transported)
            surface_points = target_surface.points_world.astype(np.float64)
            normals = target_surface.normals_world.astype(np.float64)
            kinematic_targets = surface_points + (
                self.site_standoff_m[env_id, :, None] - self.surface_preload_m
            ) * normals

            tip_surface_world[env_id] = surface_points
            normal_world_batch[env_id] = normals
            tip_surface_palm[env_id] = controller.points_world_to_palm(
                surface_points, palm_pose_batch[env_id]
            )

            if bool(palm_in_prep[env_id]):
                controller.reset()
                self.precontact_base_q[env_id] = q_hand_batch[env_id]
                self.precontact_base_valid[env_id] = True
                self.precontact_closure[env_id].fill(0.0)
                self.contact_settle_streak[env_id] = 0
                self.contact_calibrated[env_id] = False
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
                q_surface, debug = controller.update(
                    q_live=q_hand_batch[env_id],
                    palm_pose_world=palm_pose_batch[env_id],
                    force_world=force_batch[env_id],
                    found=found_batch[env_id],
                    surface_points_world=kinematic_targets,
                    surface_normals_world=normals,
                    nominal_posture_q=None,
                    force_magnitude_only=True,
                    contact_points_world=contact_pos_batch[env_id],
                    use_contact_point_jacobian=False,
                )
                settled = precontact_loaded
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
                    # FullHandMCC plans from the true loaded configuration,
                    # not from a future noisy live state.
                    self.planner_query_world[env_id] = tip_world_batch[env_id]
                    self.planner_query_valid[env_id] = True
            else:
                controller.calibrate_force_sign(
                    force_batch[env_id], found_batch[env_id], normals
                )
                q_command, debug = controller.update(
                    q_live=q_hand_batch[env_id],
                    palm_pose_world=palm_pose_batch[env_id],
                    force_world=force_batch[env_id],
                    found=found_batch[env_id],
                    surface_points_world=kinematic_targets,
                    surface_normals_world=normals,
                    nominal_posture_q=None,
                    force_magnitude_only=True,
                    contact_points_world=contact_pos_batch[env_id],
                    use_contact_point_jacobian=False,
                )
                tip_reference = controller.points_palm_to_world(
                    debug["tip_reference_palm"], palm_pose_batch[env_id]
                )
                tip_ik = controller.points_palm_to_world(
                    debug["tip_ik_palm"], palm_pose_batch[env_id]
                )
                normal_force_batch[env_id] = debug["normal_force"]
                normal_offset_batch[env_id] = debug["normal_offset"]
            q_command_batch[env_id] = q_command
            tip_reference_world[env_id] = tip_reference
            tip_ik_world[env_id] = tip_ik
            tip_reference_palm[env_id] = controller.points_world_to_palm(
                tip_reference, palm_pose_batch[env_id]
            )

        q_command_t = torch.as_tensor(
            q_command_batch, device=self.env.device, dtype=torch.float32
        )
        q_hand_t = robot.data.joint_pos[:, 6:22]
        finger_action = torch.clamp((q_command_t - q_hand_t) / 0.08, -1.0, 1.0)
        action = torch.cat((palm_output[:, :6], finger_action), dim=-1)

        self.last_debug = {
            "q_pre": torch.as_tensor(
                np.stack([controller.grasp_closure_q for controller in self.controllers]),
                device=self.env.device,
                dtype=torch.float32,
            ),
            "q_ref": q_command_t,
            "tip_x_des": torch.as_tensor(tip_surface_world, device=self.env.device),
            "tip_x_ref": torch.as_tensor(tip_reference_world, device=self.env.device),
            "tip_x_ik": torch.as_tensor(tip_ik_world, device=self.env.device),
            "tip_x_des_palm": torch.as_tensor(tip_surface_palm, device=self.env.device),
            "tip_x_ref_palm": torch.as_tensor(tip_reference_palm, device=self.env.device),
            "tip_surface_normal_world": torch.as_tensor(normal_world_batch, device=self.env.device),
            "tip_normal_force": torch.as_tensor(normal_force_batch, device=self.env.device),
            "tip_normal_offset": torch.as_tensor(normal_offset_batch, device=self.env.device),
            "tip_anchor_valid": torch.as_tensor(self.anchor_valid, device=self.env.device),
            "fullhand_contact_calibrated": torch.as_tensor(
                self.contact_calibrated, device=self.env.device
            ),
            **palm_debug,
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
        "fullhand_anchor_valid": (4,),
        "fullhand_contact_calibrated": (),
        "object_angular_velocity_world": (3,),
        "object_rotation_axis_local": (3,),
        "object_rotation_speed_target_rad_s": (),
        "object_translation_axis_world": (3,),
        "object_translation_amplitude_m": (),
        "object_translation_speed_target_m_s": (),
        "object_motion_active": (),
        "object_motion_contact_ready": (),
        "object_motion_schedule_step": (),
        "object_segment_move_steps": (),
        "object_segment_hold_steps": (),
    }

    def __init__(self, path: Path, num_envs: int):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = h5py.File(path, "w")
        self.num_envs = num_envs
        self.step = 0
        self.datasets: dict[str, h5py.Dataset] = {}
        for name, tail in self.SHAPES.items():
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
            if array.ndim == len(self.SHAPES[name]):
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
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--trajectory-length", type=int, default=2500)
    parser.add_argument("--max-trajectories", type=int, default=5)
    parser.add_argument("--motion-start", type=int, default=1000)
    parser.add_argument("--motion-length", type=int, default=1400)
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
        "--record-start-step",
        type=int,
        default=None,
        help="First saved step; defaults to motion-start so prep is excluded.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="Maximum candidate attempts; defaults to 10x max-trajectories.",
    )
    parser.add_argument(
        "--online-quality-gate",
        action="store_true",
        help="Reject whole trajectories online. Off by default for fast raw collection.",
    )
    parser.add_argument(
        "--object-id",
        default="capsule_medium",
        help="Object configuration id from the contact-object catalog.",
    )
    parser.add_argument(
        "--teacher-controller",
        choices=("fullhand_mcc", "fixed_pregrasp"),
        default="fullhand_mcc",
        help=(
            "fullhand_mcc uses privileged surface points/normals and normal "
            "admittance; fixed_pregrasp retains the legacy position teacher."
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
        "--anchor-settle-frames",
        type=int,
        default=3,
        help="Consecutive loaded-contact frames required before locking a material anchor.",
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
        choices=("rotation", "translation", "combined"),
        default="rotation",
        help=(
            "Object excitation during the motion window. Use translation and "
            "rotation separately when building single-mode teacher datasets."
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--filename", default=None)
    args = parser.parse_args()

    if args.enable_translation:
        if args.motion_mode == "translation":
            raise ValueError(
                "--enable-translation is redundant with --motion-mode translation"
            )
        args.motion_mode = "combined"
    rotation_enabled = args.motion_mode in ("rotation", "combined")
    translation_enabled = args.motion_mode in ("translation", "combined")

    # ---- resolve object & motion configuration ---------------------------------
    object_config = load_object_config(args.object_id)
    surface_preload_m = float(
        args.surface_preload_m
        if args.surface_preload_m is not None
        else object_config.collection.get("surface_preload_m", 0.002)
    )
    object_lower, object_upper = object_local_aabb(object_config)
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
    record_start = (
        args.motion_start if args.record_start_step is None else args.record_start_step
    )
    if record_start < 0 or record_start >= args.trajectory_length:
        raise ValueError("record-start-step must lie inside the trajectory")
    if record_start < args.motion_start:
        raise ValueError("record-start-step cannot precede motion-start")
    record_delay = record_start - args.motion_start
    post_start_steps = args.trajectory_length - args.motion_start
    saved_frames_per_trajectory = post_start_steps - record_delay
    if args.online_quality_gate and args.num_envs != 1:
        raise ValueError(
            "Online trajectory filtering requires --num-envs 1; collect raw in parallel "
            "and run filter_trajectories.py instead."
        )
    batches_needed = int(np.ceil(args.max_trajectories / args.num_envs))
    collected_target = batches_needed * args.num_envs
    max_attempts = (
        args.max_attempts or max(10 * args.max_trajectories, 1)
        if args.online_quality_gate
        else batches_needed
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    env_cfg = mcc_finger_contact_env_cfg(
        num_envs=args.num_envs, play=True, object_config=object_config,
    )
    env_cfg.viewer.env_idx = max(
        0, min(int(args.viewer_env_index), args.num_envs - 1)
    )
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    rl_cfg = MCCLeapHandPositionControlCfg()
    kwargs = asdict(rl_cfg)
    policy_class = kwargs.pop("policy_class")
    kwargs.pop("device", None)
    base_policy = policy_class(device=device, num_envs=args.num_envs, **kwargs)

    target_mocap_idx = int(env.scene["target"].data.indexing.mocap_id)
    palm_idx = _find_local_body_index(env, "palm_lower")
    tip_indices = [_find_local_site_index(env, name) for name in TIP_SITES]
    dt = float(env_cfg.decimation * env_cfg.sim.mujoco.timestep)
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
        )
        if args.teacher_controller == "fullhand_mcc"
        else base_policy
    )
    print(
        f"[CONTROLLER] teacher={args.teacher_controller} "
        f"privileged_oracle={args.teacher_controller == 'fullhand_mcc'} "
        f"surface_target={args.surface_target_mode}"
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

            def reset(self) -> None:
                policy.reset()
                motion_controller.reset()
                motion_controller.motion_start = args.motion_start
                self.step_count = 0
                self.actual_motion_start = None
                self.recorded_frames = 0
                self.loaded_frames.fill(0)
                self.found_frames.fill(0)
                self.all4_frames = 0
                self.contact_pos.fill(np.nan)
                self.tip_pos.fill(np.nan)
                self.found.fill(False)
                self.loaded.fill(False)
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
                if (
                    self.actual_motion_start is None
                    and self.step_count >= args.motion_start
                    and ready
                ):
                    self.actual_motion_start = self.step_count
                    motion_controller.motion_start = self.actual_motion_start
                    print(
                        "[VISUAL] prep complete: motion and recording window start "
                        f"at simulator step {self.actual_motion_start}"
                    )
                if (
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
                moving = (
                    motion_controller.step(
                        self.step_count, contact_ready=contact_ready
                    )
                    if self.actual_motion_start is not None
                    else False
                )
                action = policy(obs)
                env_id = env_cfg.viewer.env_idx
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
                        f"force_N={magnitude.detach().cpu().numpy().round(3).tolist()}",
                        flush=True,
                    )
                self.step_count += 1
                return action

        visual_policy = VisualCollectionPolicy()
        base_update_visualizers = env.update_visualizers

        def update_contact_visualizers(visualizer) -> None:
            base_update_visualizers(visualizer)
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
                        radius=0.007,
                        color=(1.0, 0.05, 0.05, 1.0),
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

        env.update_visualizers = update_contact_visualizers
        wrapped = RslRlVecEnvWrapper(env)
        env.reset()
        visual_policy.reset()
        print(
            f"[INFO] visual collection object={args.object_id} viewer={args.viewer} "
            f"steps={args.trajectory_length} device={device}"
        )
        print(
            "[INFO] contact markers: green=loaded, orange=geometry-only/weak, "
            "red=fingertip lost"
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
        return

    filename = args.filename or f"mcc_tip_{datetime.now():%Y%m%d_%H%M%S}"
    output = Path("mcc_finger_compliance_control/data/trajectories") / f"{filename}.h5"
    logger = StreamingH5(output, args.num_envs)

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
            actual_motion_start: int | None = None
            actual_record_start: int | None = None
            recorded_frames = 0

            for episode_step in range(
                args.trajectory_length + args.max_prep_wait_steps
            ):
                ready = (
                    policy.ready_for_motion
                    if args.teacher_controller == "fullhand_mcc"
                    else True
                )
                if (
                    actual_motion_start is None
                    and episode_step >= args.motion_start
                    and ready
                ):
                    actual_motion_start = episode_step
                    actual_record_start = episode_step + record_delay
                    motion_controller.motion_start = episode_step
                    print(
                        "[PREP] settled: motion starts at simulator step "
                        f"{actual_motion_start}; recording starts at "
                        f"{actual_record_start}"
                    )
                if (
                    actual_motion_start is None
                    and episode_step
                    >= args.motion_start + args.max_prep_wait_steps
                ):
                    raise RuntimeError(
                        "FullHandMCC did not settle all four fingertips within "
                        f"{args.max_prep_wait_steps} extra prep steps; no prep "
                        "frames were recorded"
                    )
                contact_ready = (
                    policy.motion_ready_mask
                    if args.teacher_controller == "fullhand_mcc"
                    else None
                )
                moving = (
                    motion_controller.step(
                        episode_step, contact_ready=contact_ready
                    )
                    if actual_motion_start is not None
                    else False
                )

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
                obj_pose = torch.cat(
                    (
                        env.sim.data.mocap_pos[:, target_mocap_idx, :],
                        env.sim.data.mocap_quat[:, target_mocap_idx, :],
                    ),
                    dim=-1,
                )
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
                        "q_hand": robot.data.joint_pos[:, 6:22],
                        "q_pre": debug["q_pre"],
                        "q_ref": debug["q_ref"],
                        "arm_q_ref": debug["palm_arm_q_ref"],
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
                        "fullhand_anchor_valid": debug.get(
                            "tip_anchor_valid",
                            torch.zeros((args.num_envs, 4), device=device),
                        ).float(),
                        "fullhand_contact_calibrated": debug.get(
                            "fullhand_contact_calibrated",
                            torch.zeros(args.num_envs, device=device),
                        ).float(),
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
                        "object_motion_active": motion_controller.motion_active.float(),
                        "object_motion_contact_ready": (
                            motion_controller.motion_contact_ready.float()
                        ),
                        "object_motion_schedule_step": (
                            motion_controller.motion_schedule_step
                        ),
                        "object_segment_move_steps": motion_controller.segment_move_steps,
                        "object_segment_hold_steps": motion_controller.segment_hold_steps,
                        }
                    )
                    recorded_frames += 1
                    if recorded_frames >= saved_frames_per_trajectory:
                        break

            if recorded_frames != saved_frames_per_trajectory:
                raise RuntimeError(
                    f"Recorded {recorded_frames}/{saved_frames_per_trajectory} "
                    "post-prep frames"
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
            strict_pass = bool(np.all(all_four_contact))
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
                "surface_preload_m": surface_preload_m,
                "surface_target_mode": args.surface_target_mode,
                "force_frame": "world",
                "pose_quaternion_order": "wxyz",
                "control_dt": dt,
                "motion_start": args.motion_start,
                "record_start_step": record_start,
                "motion_start_semantics": "earliest_start_after_prep",
                "record_start_delay_steps": record_delay,
                "max_prep_wait_steps": args.max_prep_wait_steps,
                "trajectory_length": args.trajectory_length,
                "num_trajectories": accepted,
                "candidate_attempts": attempts,
                "strict_four_tip_continuous_contact": args.online_quality_gate,
                "contact_gate": "full_fingertip_geom_found_and_3d_force_magnitude",
                "contact_threshold": args.contact_threshold,
                "initial_orientation_mode": args.initial_orientation_mode,
                "object_angular_velocity_frame": "world",
                "object_rotation_axis_frame": "object_local",
                "object_translation_axis_frame": "world",
                "motion_mode": args.motion_mode,
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
