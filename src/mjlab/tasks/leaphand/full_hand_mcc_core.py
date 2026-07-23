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
