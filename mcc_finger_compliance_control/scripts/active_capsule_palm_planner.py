"""Contact-gated palm-only capsule surface planner.

This is the deployment-side adapter of FullHandMCC's ``meridian_inward``
surface route.  It deliberately plans only the palm pose.  Finger posture is
left to DP and high-rate contact correction is left to FullHandMCC.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from mjlab.tasks.leaphand.full_hand_mcc_geometry import (
    capsule_meridian_coordinates,
    capsule_meridian_targets,
    capsule_project,
)

from palm_planner_features import future_palm_delta_pose_palm


@dataclass(frozen=True)
class ActiveCapsulePalmPlannerConfig:
    radius_m: float = 0.15
    half_height_m: float = 0.08
    surface_speed_m_s: float = 0.008
    travel_m: float = 0.040
    direction: int = 1
    control_dt: float = 0.01
    max_surface_acceleration_m_s2: float = 0.04
    min_contact_fingers: int = 3


def _pose_from_rt(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    quaternion_xyzw = Rotation.from_matrix(rotation).as_quat()
    return np.concatenate(
        (position, quaternion_xyzw[[3, 0, 1, 2]])
    ).astype(np.float32)


class ActiveCapsulePalmPlanner:
    """Move a palm guide along one capsule meridian with tactile gating.

    The initial palm-to-surface offset is expressed in the local capsule
    contact frame and parallel-transported along the meridian.  This is the
    palm-guide portion of FullHandMCC's surface planner, without importing its
    planned finger joint trajectory.
    """

    def __init__(
        self,
        initial_pose_object: np.ndarray,
        config: ActiveCapsulePalmPlannerConfig,
    ) -> None:
        self.config = config
        if config.radius_m <= 0.0 or config.half_height_m < 0.0:
            raise ValueError("Invalid capsule dimensions")
        if config.surface_speed_m_s <= 0.0 or config.travel_m <= 0.0:
            raise ValueError("Surface speed and travel must be positive")
        if config.direction not in (-1, 1):
            raise ValueError("direction must be -1 or +1")
        if config.control_dt <= 0.0:
            raise ValueError("control_dt must be positive")
        if not 1 <= config.min_contact_fingers <= 4:
            raise ValueError("min_contact_fingers must be in [1, 4]")

        pose = np.asarray(initial_pose_object, dtype=np.float64).reshape(7)
        self.initial_position = pose[:3].copy()
        self.initial_rotation = Rotation.from_quat(
            pose[[4, 5, 6, 3]]
        ).as_matrix()
        center = np.zeros(3, dtype=np.float64)
        object_rotation = np.eye(3, dtype=np.float64)
        surface, _ = capsule_project(
            self.initial_position[None],
            center,
            object_rotation,
            config.radius_m,
            config.half_height_m,
        )
        arc, azimuth = capsule_meridian_coordinates(
            surface,
            center,
            object_rotation,
            config.radius_m,
            config.half_height_m,
        )
        _, _, frame = capsule_meridian_targets(
            arc,
            azimuth,
            center,
            object_rotation,
            config.radius_m,
            config.half_height_m,
        )
        self.start_arc = float(arc[0])
        self.azimuth = float(azimuth[0])
        self.start_frame = np.asarray(frame[0], dtype=np.float64)
        self.frame_offset = self.start_frame.T @ (
            self.initial_position - np.asarray(surface[0], dtype=np.float64)
        )
        # Preserve the initial wrist orientation relative to the transported
        # local surface frame.  No arbitrary wrist-yaw lock is introduced.
        self.rotation_in_surface_frame = (
            self.start_frame.T @ self.initial_rotation
        )
        total_arc = np.pi * config.radius_m + 2.0 * config.half_height_m
        requested_end = self.start_arc + config.direction * config.travel_m
        self.end_arc = float(np.clip(requested_end, 0.0, total_arc))
        if abs(self.end_arc - self.start_arc) < 1.0e-6:
            raise ValueError("Initial palm pose leaves no travel in that direction")

        self.arc = self.start_arc
        self.surface_velocity = 0.0
        self.enabled = False
        self.paused_for_contact = False
        self.finished = False
        self.step_count = 0
        self.motion_steps = 0
        self.pause_steps = 0
        self._pose = self._pose_at(self.arc)
        self._twist = np.zeros(6, dtype=np.float32)

    def _pose_at(self, arc: float) -> np.ndarray:
        point, _, frame = capsule_meridian_targets(
            np.asarray([arc]),
            np.asarray([self.azimuth]),
            np.zeros(3),
            np.eye(3),
            self.config.radius_m,
            self.config.half_height_m,
        )
        local_frame = np.asarray(frame[0], dtype=np.float64)
        position = np.asarray(point[0], dtype=np.float64) + (
            local_frame @ self.frame_offset
        )
        rotation = local_frame @ self.rotation_in_surface_frame
        return _pose_from_rt(position, rotation)

    @property
    def pose_object(self) -> np.ndarray:
        return self._pose.copy()

    @property
    def twist_object(self) -> np.ndarray:
        return self._twist.copy()

    @property
    def progress_m(self) -> float:
        return abs(self.arc - self.start_arc)

    def planner_feature(
        self, *, waypoint_count: int, step_frames: int
    ) -> np.ndarray:
        poses = [self.pose_object]
        signed_speed = self.surface_velocity
        if self.enabled and not self.paused_for_contact and not self.finished:
            signed_speed = (
                self.config.direction * self.config.surface_speed_m_s
            )
        for waypoint in range(1, waypoint_count + 1):
            future_arc = self.arc + signed_speed * (
                waypoint * step_frames * self.config.control_dt
            )
            if self.config.direction > 0:
                future_arc = min(future_arc, self.end_arc)
            else:
                future_arc = max(future_arc, self.end_arc)
            poses.append(self._pose_at(float(future_arc)))
        pose_array = np.asarray(poses, dtype=np.float64)
        # Encode each requested waypoint relative to the same current pose.
        features = []
        for waypoint in range(1, waypoint_count + 1):
            pair = pose_array[[0, waypoint]]
            value = future_palm_delta_pose_palm(
                pair,
                np.zeros(2, dtype=np.int32),
                waypoint_count=1,
                step_frames=1,
            )[0, 0]
            features.append(value)
        return np.concatenate(features).astype(np.float32)

    def step(self, contact_count: int, *, enabled: bool) -> None:
        """Advance one control frame, or smoothly stop on contact loss."""

        self.step_count += 1
        self.enabled = bool(enabled)
        contact_ok = int(contact_count) >= self.config.min_contact_fingers
        self.paused_for_contact = self.enabled and not contact_ok
        if self.finished:
            target_velocity = 0.0
        elif self.enabled and contact_ok:
            target_velocity = (
                self.config.direction * self.config.surface_speed_m_s
            )
        else:
            target_velocity = 0.0
        max_delta = (
            self.config.max_surface_acceleration_m_s2
            * self.config.control_dt
        )
        self.surface_velocity += float(
            np.clip(
                target_velocity - self.surface_velocity,
                -max_delta,
                max_delta,
            )
        )

        previous_pose = self._pose.copy()
        candidate = self.arc + self.surface_velocity * self.config.control_dt
        reached = (
            candidate >= self.end_arc
            if self.config.direction > 0
            else candidate <= self.end_arc
        )
        if reached:
            candidate = self.end_arc
            self.surface_velocity = 0.0
            self.finished = True
        self.arc = float(candidate)
        self._pose = self._pose_at(self.arc)

        previous_rotation = Rotation.from_quat(
            previous_pose[[4, 5, 6, 3]]
        ).as_matrix()
        current_rotation = Rotation.from_quat(
            self._pose[[4, 5, 6, 3]]
        ).as_matrix()
        linear = (self._pose[:3] - previous_pose[:3]) / self.config.control_dt
        angular = Rotation.from_matrix(
            current_rotation @ previous_rotation.T
        ).as_rotvec() / self.config.control_dt
        self._twist = np.concatenate((linear, angular)).astype(np.float32)
        if abs(self.surface_velocity) > 1.0e-8:
            self.motion_steps += 1
        if self.paused_for_contact:
            self.pause_steps += 1
