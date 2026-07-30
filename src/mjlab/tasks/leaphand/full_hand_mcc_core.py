"""Pure numerical core for five-contact full-hand MCC.

The module deliberately has no MuJoCo, Mink, Torch, or MJLab dependency.  The
simulation adapter lives in :mod:`leaphand_full_hand_mcc_env_cfg`; keeping the
reference dynamics here makes the five controller variants cheap to unit test.

Sign convention
---------------
``surface_normals`` point out of the object.  Positive measured normal force is
therefore the force exerted by the object on the hand.  If measured force is
below the desired force, the controller moves along ``-normal``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


MCCVariant = Literal[
    "independent_mcc",
    "motor_torque_mcc",
    "hierarchical_mcc",
    "hybrid_force_position",
    "passivity_tank",
]

MCC_VARIANTS: tuple[MCCVariant, ...] = (
    "independent_mcc",
    "motor_torque_mcc",
    "hierarchical_mcc",
    "hybrid_force_position",
    "passivity_tank",
)


def _normalized(vectors: np.ndarray, eps: float = 1.0e-8) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    fallback = np.zeros_like(vectors)
    fallback[..., 2] = 1.0
    return np.where(norms > eps, vectors / np.maximum(norms, eps), fallback)


def _clip_vector_norm(vectors: np.ndarray, max_norm: float) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    scale = np.minimum(1.0, max_norm / np.maximum(norms, 1.0e-9))
    return vectors * scale


@dataclass(frozen=True)
class FullHandMCCGains:
    """Gains shared by the five controller variants.

    Contact index 0 is the palm root and indices 1..4 are fingertips.
    """

    dt: float = 0.01
    tangent_kp: float = 18.0
    tangent_kd: float = 4.0
    normal_position_kp: float = 8.0
    force_kp: float = 0.004
    force_ki: float = 0.001
    force_integral_limit: float = 4.0
    contact_on_force: float = 0.15
    contact_off_force: float = 0.08
    desired_palm_force: float = 3.0
    desired_fingertip_force: float = 1.0
    force_balance_gain: float = 0.002
    relative_position_gain: float = 12.0
    velocity_filter_alpha: float = 0.25
    max_reference_speed: float = 0.04
    max_reference_offset: float = 0.035
    energy_tank_initial: float = 0.35
    energy_tank_capacity: float = 2.0
    energy_tank_floor: float = 0.03
    passive_dissipation: float = 2.0

    @property
    def desired_normal_force(self) -> np.ndarray:
        return np.asarray(
            [self.desired_palm_force] + [self.desired_fingertip_force] * 4,
            dtype=np.float64,
        )


@dataclass(frozen=True)
class FullHandMCCStep:
    reference_points: np.ndarray
    reference_velocity: np.ndarray
    measured_normal_force: np.ndarray
    force_error: np.ndarray
    contact_active: np.ndarray
    energy_tank: np.ndarray
    passivity_scale: np.ndarray


@dataclass(frozen=True)
class FingertipAdmittanceGains:
    """Four independent normal-direction fingertip admittance loops.

    ``normal_offset`` is positive into the object.  With outward surface
    normals, the Cartesian command is ``planned - normal_offset * normal``.
    """

    dt: float = 0.01
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


@dataclass(frozen=True)
class FingertipAdmittanceStep:
    command_points: np.ndarray
    normal_offset: np.ndarray
    normal_velocity: np.ndarray
    normal_acceleration: np.ndarray
    measured_normal_force: np.ndarray
    filtered_normal_force: np.ndarray
    force_error: np.ndarray
    contact_active: np.ndarray


class FingertipNormalAdmittance:
    """Analytical Baseline-2 fingertip force controller.

    The four loops consume *measured fingertip forces*.  They never infer
    contact force from actuator current/torque.  The upper-level optimizer
    owns tangential motion; this controller only offsets each planned point
    along its outward surface normal.
    """

    def __init__(
        self,
        gains: FingertipAdmittanceGains | None = None,
    ) -> None:
        self.gains = gains or FingertipAdmittanceGains()
        self._offset: np.ndarray | None = None
        self._velocity: np.ndarray | None = None
        self._filtered_force: np.ndarray | None = None
        self._contact_active: np.ndarray | None = None
        self._filter_initialized = False

    def reset(self) -> None:
        self._offset = None
        self._velocity = None
        self._filtered_force = None
        self._contact_active = None
        self._filter_initialized = False

    def _ensure_state(self, batch: int) -> None:
        shape = (batch, 4)
        if self._offset is not None and self._offset.shape == shape:
            return
        self._offset = np.zeros(shape, dtype=np.float64)
        self._velocity = np.zeros(shape, dtype=np.float64)
        self._filtered_force = np.zeros(shape, dtype=np.float64)
        self._contact_active = np.zeros(shape, dtype=bool)
        self._filter_initialized = False

    def step(
        self,
        planned_points: np.ndarray,
        surface_normals: np.ndarray,
        measured_forces: np.ndarray,
        desired_force: np.ndarray | float | None = None,
    ) -> FingertipAdmittanceStep:
        """Advance four scalar admittance states by one finger-control period.

        Inputs use one common Cartesian frame and have shape ``(B,4,3)``.
        ``measured_forces`` must already use the calibrated sensor sign such
        that force exerted by the object on the fingertip projects positively
        on the outward surface normal.
        """

        planned = np.asarray(planned_points, dtype=np.float64)
        normals = _normalized(np.asarray(surface_normals, dtype=np.float64))
        forces = np.asarray(measured_forces, dtype=np.float64)
        if planned.ndim == 2:
            planned = planned[None, ...]
            normals = normals[None, ...]
            forces = forces[None, ...]
        expected = (planned.shape[0], 4, 3)
        for name, value in (
            ("planned_points", planned),
            ("surface_normals", normals),
            ("measured_forces", forces),
        ):
            if value.shape != expected:
                raise ValueError(
                    f"{name} must have shape {expected}, got {value.shape}"
                )
        if not all(
            np.all(np.isfinite(value)) for value in (planned, normals, forces)
        ):
            raise ValueError("Fingertip admittance inputs must be finite")

        self._ensure_state(planned.shape[0])
        assert self._offset is not None
        assert self._velocity is not None
        assert self._filtered_force is not None
        assert self._contact_active is not None

        g = self.gains
        if min(
            g.dt,
            g.virtual_mass,
            g.virtual_damping,
            g.virtual_stiffness,
            g.max_normal_offset,
            g.max_normal_speed,
            g.max_normal_acceleration,
        ) <= 0.0:
            raise ValueError("Fingertip admittance gains and limits must be positive")

        measured_normal = np.einsum("bfi,bfi->bf", forces, normals)
        if not self._filter_initialized:
            self._filtered_force[:] = measured_normal
            self._filter_initialized = True
        else:
            alpha = float(np.clip(g.force_filter_alpha, 0.0, 1.0))
            self._filtered_force[:] = (
                alpha * measured_normal
                + (1.0 - alpha) * self._filtered_force
            )

        turn_on = self._filtered_force >= g.contact_on_force
        stay_on = self._filtered_force >= g.contact_off_force
        self._contact_active[:] = np.where(
            self._contact_active, stay_on, turn_on
        )

        desired = (
            np.full_like(self._filtered_force, g.desired_force)
            if desired_force is None
            else np.broadcast_to(
                np.asarray(desired_force, dtype=np.float64),
                self._filtered_force.shape,
            )
        )
        if not np.all(np.isfinite(desired)):
            raise ValueError("desired_force must be finite")
        force_error = desired - self._filtered_force
        acceleration = (
            g.force_gain * force_error
            - g.virtual_damping * self._velocity
            - g.virtual_stiffness * self._offset
        ) / g.virtual_mass
        acceleration = np.clip(
            acceleration,
            -g.max_normal_acceleration,
            g.max_normal_acceleration,
        )
        self._velocity += acceleration * g.dt
        np.clip(
            self._velocity,
            -g.max_normal_speed,
            g.max_normal_speed,
            out=self._velocity,
        )
        proposed_offset = self._offset + self._velocity * g.dt
        clipped_offset = np.clip(
            proposed_offset,
            -g.max_normal_offset,
            g.max_normal_offset,
        )
        saturated_outward = (
            (proposed_offset > g.max_normal_offset) & (self._velocity > 0.0)
        ) | (
            (proposed_offset < -g.max_normal_offset) & (self._velocity < 0.0)
        )
        self._velocity[saturated_outward] = 0.0
        self._offset[:] = clipped_offset

        command = planned - self._offset[..., None] * normals
        return FingertipAdmittanceStep(
            command_points=command.copy(),
            normal_offset=self._offset.copy(),
            normal_velocity=self._velocity.copy(),
            normal_acceleration=acceleration.copy(),
            measured_normal_force=measured_normal.copy(),
            filtered_normal_force=self._filtered_force.copy(),
            force_error=force_error.copy(),
            contact_active=self._contact_active.copy(),
        )


@dataclass(frozen=True)
class WristAdmittanceGains:
    """Six-dimensional wrist reference dynamics for Baseline 2."""

    dt: float = 0.04
    translation_mass: float = 3.0
    rotation_inertia: tuple[float, float, float] = (0.30, 0.30, 0.30)
    normal_stiffness: float = 400.0
    tangent_stiffness: float = 800.0
    rotation_stiffness: float = 80.0
    damping_ratio: float = 1.0
    wrench_filter_alpha: float = 0.10
    max_force_error: float = 5.0
    max_torque_error: float = 0.8
    max_translation_offset: float = 0.003
    max_rotation_offset: float = 0.03
    max_translation_speed: float = 0.010
    max_rotation_speed: float = 0.10
    max_translation_acceleration: float = 0.10
    max_rotation_acceleration: float = 0.5


@dataclass(frozen=True)
class WristAdmittanceStep:
    reference_offset: np.ndarray
    reference_velocity: np.ndarray
    reference_acceleration: np.ndarray
    filtered_wrench_error: np.ndarray


class WristCartesianAdmittance:
    """External-wrench-driven virtual wrist reference integrator."""

    def __init__(self, gains: WristAdmittanceGains | None = None) -> None:
        self.gains = gains or WristAdmittanceGains()
        self._offset: np.ndarray | None = None
        self._velocity: np.ndarray | None = None
        self._filtered_wrench: np.ndarray | None = None
        self._filter_initialized = False

    def reset(self) -> None:
        self._offset = None
        self._velocity = None
        self._filtered_wrench = None
        self._filter_initialized = False

    def _ensure_state(self, batch: int) -> None:
        shape = (batch, 6)
        if self._offset is not None and self._offset.shape == shape:
            return
        self._offset = np.zeros(shape, dtype=np.float64)
        self._velocity = np.zeros(shape, dtype=np.float64)
        self._filtered_wrench = np.zeros(shape, dtype=np.float64)
        self._filter_initialized = False

    @staticmethod
    def _clip_radial_velocity(
        offset: np.ndarray,
        velocity: np.ndarray,
        limit: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        clipped = _clip_vector_norm(offset, limit)
        saturated = np.linalg.norm(offset, axis=-1) > limit + 1.0e-12
        for env_id in np.flatnonzero(saturated):
            direction = clipped[env_id] / max(
                np.linalg.norm(clipped[env_id]), 1.0e-12
            )
            outward_speed = float(np.dot(velocity[env_id], direction))
            if outward_speed > 0.0:
                velocity[env_id] -= outward_speed * direction
        return clipped, velocity

    def step(
        self,
        wrench_error: np.ndarray,
        surface_normal: np.ndarray,
    ) -> WristAdmittanceStep:
        wrench = np.asarray(wrench_error, dtype=np.float64)
        normal = _normalized(np.asarray(surface_normal, dtype=np.float64))
        if wrench.ndim == 1:
            wrench = wrench[None, ...]
            normal = normal[None, ...]
        if wrench.shape != (wrench.shape[0], 6):
            raise ValueError(f"wrench_error must be (B,6), got {wrench.shape}")
        if normal.shape != (wrench.shape[0], 3):
            raise ValueError(
                f"surface_normal must be {(wrench.shape[0], 3)}, got {normal.shape}"
            )
        if not np.all(np.isfinite(wrench)) or not np.all(np.isfinite(normal)):
            raise ValueError("Wrist admittance inputs must be finite")

        self._ensure_state(wrench.shape[0])
        assert self._offset is not None
        assert self._velocity is not None
        assert self._filtered_wrench is not None
        g = self.gains
        if min(
            g.dt,
            g.translation_mass,
            *g.rotation_inertia,
            g.normal_stiffness,
            g.tangent_stiffness,
            g.rotation_stiffness,
            g.damping_ratio,
            g.max_force_error,
            g.max_torque_error,
        ) <= 0.0:
            raise ValueError("Wrist admittance gains must be positive")

        if not self._filter_initialized:
            self._filtered_wrench[:] = wrench
            self._filter_initialized = True
        else:
            alpha = float(np.clip(g.wrench_filter_alpha, 0.0, 1.0))
            self._filtered_wrench[:] = (
                alpha * wrench + (1.0 - alpha) * self._filtered_wrench
            )
        self._filtered_wrench[:, :3] = _clip_vector_norm(
            self._filtered_wrench[:, :3], g.max_force_error
        )
        self._filtered_wrench[:, 3:] = _clip_vector_norm(
            self._filtered_wrench[:, 3:], g.max_torque_error
        )

        offset_t = self._offset[:, :3]
        velocity_t = self._velocity[:, :3]
        offset_n = np.einsum("bi,bi->b", offset_t, normal)
        velocity_n = np.einsum("bi,bi->b", velocity_t, normal)
        spring_t = (
            g.tangent_stiffness * offset_t
            + (g.normal_stiffness - g.tangent_stiffness)
            * offset_n[:, None]
            * normal
        )
        damping_tangent = (
            2.0
            * g.damping_ratio
            * np.sqrt(g.translation_mass * g.tangent_stiffness)
        )
        damping_normal = (
            2.0
            * g.damping_ratio
            * np.sqrt(g.translation_mass * g.normal_stiffness)
        )
        damping_t = (
            damping_tangent * velocity_t
            + (damping_normal - damping_tangent)
            * velocity_n[:, None]
            * normal
        )
        acceleration_t = (
            self._filtered_wrench[:, :3] - spring_t - damping_t
        ) / g.translation_mass
        acceleration_t = _clip_vector_norm(
            acceleration_t, g.max_translation_acceleration
        )

        inertia = np.asarray(g.rotation_inertia, dtype=np.float64)
        damping_r = (
            2.0
            * g.damping_ratio
            * np.sqrt(inertia * g.rotation_stiffness)
        )
        acceleration_r = (
            self._filtered_wrench[:, 3:]
            - g.rotation_stiffness * self._offset[:, 3:]
            - damping_r * self._velocity[:, 3:]
        ) / inertia
        acceleration_r = _clip_vector_norm(
            acceleration_r, g.max_rotation_acceleration
        )
        acceleration = np.concatenate((acceleration_t, acceleration_r), axis=1)

        self._velocity += acceleration * g.dt
        self._velocity[:, :3] = _clip_vector_norm(
            self._velocity[:, :3], g.max_translation_speed
        )
        self._velocity[:, 3:] = _clip_vector_norm(
            self._velocity[:, 3:], g.max_rotation_speed
        )
        self._offset += self._velocity * g.dt
        self._offset[:, :3], self._velocity[:, :3] = (
            self._clip_radial_velocity(
                self._offset[:, :3],
                self._velocity[:, :3],
                g.max_translation_offset,
            )
        )
        self._offset[:, 3:], self._velocity[:, 3:] = (
            self._clip_radial_velocity(
                self._offset[:, 3:],
                self._velocity[:, 3:],
                g.max_rotation_offset,
            )
        )
        return WristAdmittanceStep(
            reference_offset=self._offset.copy(),
            reference_velocity=self._velocity.copy(),
            reference_acceleration=acceleration.copy(),
            filtered_wrench_error=self._filtered_wrench.copy(),
        )


class FullHandMCCCore:
    """Five-point reference dynamics with five selectable MCC variants."""

    def __init__(
        self,
        variant: MCCVariant = "hybrid_force_position",
        gains: FullHandMCCGains | None = None,
    ) -> None:
        if variant not in MCC_VARIANTS:
            raise ValueError(f"Unknown MCC variant {variant!r}; choose from {MCC_VARIANTS}")
        self.variant = variant
        self.gains = gains or FullHandMCCGains()
        self._reference: np.ndarray | None = None
        self._velocity: np.ndarray | None = None
        self._force_integral: np.ndarray | None = None
        self._contact_active: np.ndarray | None = None
        self._energy_tank: np.ndarray | None = None

    def reset(self) -> None:
        self._reference = None
        self._velocity = None
        self._force_integral = None
        self._contact_active = None
        self._energy_tank = None

    def _ensure_state(self, points: np.ndarray) -> None:
        batch = points.shape[0]
        if self._reference is not None and self._reference.shape == points.shape:
            return
        self._reference = points.copy()
        self._velocity = np.zeros_like(points)
        self._force_integral = np.zeros((batch, 5), dtype=np.float64)
        self._contact_active = np.zeros((batch, 5), dtype=bool)
        self._energy_tank = np.full(
            batch, self.gains.energy_tank_initial, dtype=np.float64
        )

    def step(
        self,
        actual_points: np.ndarray,
        desired_points: np.ndarray,
        surface_normals: np.ndarray,
        measured_forces: np.ndarray,
    ) -> FullHandMCCStep:
        """Advance the five-point MCC reference by one control period.

        All array inputs have shape ``(batch, 5, 3)``.  The returned reference
        is always finite, velocity limited, and offset limited with respect to
        the measured kinematic points.
        """

        actual = np.asarray(actual_points, dtype=np.float64)
        desired = np.asarray(desired_points, dtype=np.float64)
        normals = _normalized(np.asarray(surface_normals, dtype=np.float64))
        forces = np.asarray(measured_forces, dtype=np.float64)
        if actual.ndim == 2:
            actual = actual[None, ...]
            desired = desired[None, ...]
            normals = normals[None, ...]
            forces = forces[None, ...]
        expected = (actual.shape[0], 5, 3)
        for name, value in (
            ("actual_points", actual),
            ("desired_points", desired),
            ("surface_normals", normals),
            ("measured_forces", forces),
        ):
            if value.shape != expected:
                raise ValueError(f"{name} must have shape {expected}, got {value.shape}")
        if not all(np.all(np.isfinite(value)) for value in (actual, desired, normals, forces)):
            raise ValueError("MCC inputs must contain only finite values")

        self._ensure_state(actual)
        assert self._reference is not None
        assert self._velocity is not None
        assert self._force_integral is not None
        assert self._contact_active is not None
        assert self._energy_tank is not None

        g = self.gains
        measured_normal = np.einsum("bci,bci->bc", forces, normals)
        # A motor-force estimate may flip sign during initial calibration.  The
        # contact magnitude is robust, while the signed value remains available
        # in adapter diagnostics for sign verification.
        measured_normal = np.maximum(measured_normal, 0.0)
        turn_on = measured_normal >= g.contact_on_force
        stay_on = measured_normal >= g.contact_off_force
        self._contact_active = np.where(self._contact_active, stay_on, turn_on)

        desired_force = np.broadcast_to(g.desired_normal_force, measured_normal.shape)
        force_error = desired_force - measured_normal
        self._force_integral += (
            force_error * self._contact_active * g.dt
        )
        np.clip(
            self._force_integral,
            -g.force_integral_limit,
            g.force_integral_limit,
            out=self._force_integral,
        )

        error = desired - actual
        error_n = np.einsum("bci,bci->bc", error, normals)
        error_t = error - error_n[..., None] * normals
        velocity_n = np.einsum("bci,bci->bc", self._velocity, normals)
        velocity_t = self._velocity - velocity_n[..., None] * normals

        tangent_velocity = (
            g.tangent_kp * error_t - g.tangent_kd * velocity_t
        )
        approach_velocity = g.normal_position_kp * error_n[..., None] * normals
        force_velocity = -(
            g.force_kp * force_error + g.force_ki * self._force_integral
        )[..., None] * normals
        normal_velocity = np.where(
            self._contact_active[..., None],
            force_velocity,
            approach_velocity,
        )

        if self.variant == "independent_mcc":
            command_velocity = tangent_velocity + normal_velocity
        elif self.variant == "motor_torque_mcc":
            # The adapter adds direct per-motor torque correction after IK.
            # Keep a gentle Cartesian attractor here to avoid drift.
            command_velocity = 0.55 * tangent_velocity + normal_velocity
        elif self.variant == "hierarchical_mcc":
            palm_velocity = tangent_velocity[:, :1] + normal_velocity[:, :1]
            desired_relative = desired[:, 1:] - desired[:, :1]
            actual_relative = actual[:, 1:] - actual[:, :1]
            relative_error = desired_relative - actual_relative
            relative_n = np.einsum(
                "bci,bci->bc", relative_error, normals[:, 1:]
            )
            relative_t = (
                relative_error - relative_n[..., None] * normals[:, 1:]
            )
            finger_velocity = (
                palm_velocity
                + g.relative_position_gain * relative_t
                + normal_velocity[:, 1:]
            )
            command_velocity = np.concatenate((palm_velocity, finger_velocity), axis=1)
        else:
            # Recommended hybrid mode: exact tangential position tracking and
            # force PI only in the normal direction.
            command_velocity = tangent_velocity + normal_velocity
            if self.variant == "hybrid_force_position":
                mean_tip_force = measured_normal[:, 1:].mean(axis=1, keepdims=True)
                balance_error = mean_tip_force - measured_normal[:, 1:]
                command_velocity[:, 1:] += (
                    -g.force_balance_gain
                    * balance_error[..., None]
                    * normals[:, 1:]
                )

        passivity_scale = np.ones(actual.shape[0], dtype=np.float64)
        if self.variant == "passivity_tank":
            # Positive injected_power means motion is commanded against the
            # measured environment force.  The tank bounds that injection.
            injected_power = np.maximum(
                0.0, np.sum(-forces * command_velocity, axis=(1, 2))
            )
            dissipated_power = (
                g.passive_dissipation
                * np.sum(self._velocity * self._velocity, axis=(1, 2))
            )
            available_power = np.maximum(
                0.0, (self._energy_tank - g.energy_tank_floor) / g.dt
            )
            need_scale = injected_power > available_power
            passivity_scale[need_scale] = (
                available_power[need_scale]
                / np.maximum(injected_power[need_scale], 1.0e-9)
            )
            command_velocity *= passivity_scale[:, None, None]
            actual_injected = injected_power * passivity_scale
            self._energy_tank += (dissipated_power - actual_injected) * g.dt
            np.clip(
                self._energy_tank,
                g.energy_tank_floor,
                g.energy_tank_capacity,
                out=self._energy_tank,
            )

        command_velocity = _clip_vector_norm(
            command_velocity, g.max_reference_speed
        )
        alpha = np.clip(g.velocity_filter_alpha, 0.0, 1.0)
        self._velocity = (
            alpha * command_velocity + (1.0 - alpha) * self._velocity
        )
        self._reference += self._velocity * g.dt

        offset = self._reference - actual
        offset = _clip_vector_norm(offset, g.max_reference_offset)
        self._reference = actual + offset

        return FullHandMCCStep(
            reference_points=self._reference.copy(),
            reference_velocity=self._velocity.copy(),
            measured_normal_force=measured_normal.copy(),
            force_error=force_error.copy(),
            contact_active=self._contact_active.copy(),
            energy_tank=self._energy_tank.copy(),
            passivity_scale=passivity_scale,
        )
