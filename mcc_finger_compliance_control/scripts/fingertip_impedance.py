"""Bounded normal-force compliance around a position-controlled DP pose.

The LeapHand interface used by replay accepts joint-position commands rather
than motor torques.  Each fingertip therefore adds one small Cartesian task to
the current DP reference::

    q_cmd = q_dp + dq_normal

The task uses the live tactile contact point and surface normal.  A bounded
preload offset is increased when normal force is too small, decreased when it
is too large, and held inside the target band.  It never changes the DP
reference itself and cannot integrate beyond ``max_normal_offset``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np


HAND_XML = Path(
    "src/mjlab/asset_zoo/robots/xarm6_leap_hand/leap_hand_tactile.xml"
)
TIP_BODY_NAMES = (
    "fingertip",
    "fingertip_2",
    "fingertip_3",
    "thumb_fingertip",
)
TIP_SITE_NAMES = ("if_tip", "mf_tip", "rf_tip", "th_tip")
TIP_SITE_LOCAL_POSITIONS = (
    (-0.0106151, -0.0326103, 0.0141088),
    (-0.0106151, -0.0326103, 0.0144487),
    (-0.0106151, -0.0326103, 0.0140386),
    (-0.0106383, -0.0453895, -0.0144321),
)


@dataclass
class FingertipImpedanceConfig:
    """Parameters for the bounded position-equivalent fingertip force loop."""

    control_dt: float = 0.01
    force_min: float = 2.2
    force_max: float = 3.5
    contact_on_force: float = 0.20
    contact_off_force: float = 0.10
    force_filter_alpha: float = 0.80
    normal_filter_alpha: float = 0.80
    force_error_full_scale: float = 2.2
    max_normal_offset: float = 0.006
    max_retreat_offset: float = 0.003
    max_offset_rate: float = 0.00005
    recovery_offset_rate: float = 0.00015
    max_recovery_offset_step: float = 0.003
    jacobian_damping: float = 0.01
    max_joint_correction: float = 0.08
    max_joint_rate: float = 0.03
    nominal_guard_enabled: bool = True
    nominal_release_rate: float = 0.003
    recovery_confirm_steps: int = 3
    joint_limit_margin: float = 0.03


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
        TIP_BODY_NAMES, TIP_SITE_NAMES, TIP_SITE_LOCAL_POSITIONS
    ):
        if site_name not in existing:
            spec.body(body_name).add_site(
                name=site_name,
                pos=site_pos,
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=(0.004, 0.0, 0.0),
            )
    return spec.compile()


def _quat_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(quaternion)
    if norm < 1.0e-12:
        return np.eye(3)
    w, x, y, z = quaternion / norm
    return np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


class FingertipImpedanceController:
    """Bounded four-fingertip force loop around a nominal DP joint pose."""

    def __init__(self, config: FingertipImpedanceConfig | None = None):
        self.config = config or FingertipImpedanceConfig()
        if not 0.0 <= self.config.force_filter_alpha < 1.0:
            raise ValueError("force_filter_alpha must lie in [0, 1)")
        if not 0.0 <= self.config.normal_filter_alpha < 1.0:
            raise ValueError("normal_filter_alpha must lie in [0, 1)")
        if self.config.force_min >= self.config.force_max:
            raise ValueError("force_min must be smaller than force_max")
        self.model = _fixed_hand_model()
        self.data = mujoco.MjData(self.model)
        if self.model.nq != 16 or self.model.nv != 16:
            raise ValueError(
                f"Expected fixed 16-DoF hand, got nq={self.model.nq}, "
                f"nv={self.model.nv}"
            )
        self.tip_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            for name in TIP_SITE_NAMES
        ]
        self.tip_body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in TIP_BODY_NAMES
        ]
        self.active_dofs = (
            np.asarray((0, 1, 2, 3)),
            np.asarray((4, 5, 6, 7)),
            np.asarray((8, 9, 10, 11)),
            np.asarray((12, 13, 14, 15)),
        )
        self.lower = np.full(16, -np.inf, dtype=np.float64)
        self.upper = np.full(16, np.inf, dtype=np.float64)
        for joint_id in range(self.model.njnt):
            address = int(self.model.jnt_qposadr[joint_id])
            if self.model.jnt_limited[joint_id]:
                self.lower[address], self.upper[address] = self.model.jnt_range[
                    joint_id
                ]
        self.reset()

    def reset(self) -> None:
        self.force_filtered = np.zeros((4, 3), dtype=np.float64)
        self.normal_filtered = np.zeros((4, 3), dtype=np.float64)
        self.normal_offset = np.zeros(4, dtype=np.float64)
        self.contact_state = np.zeros(4, dtype=bool)
        self.contact_point_palm = np.zeros((4, 3), dtype=np.float64)
        self.contact_point_valid = np.zeros(4, dtype=bool)
        self.normal_age = np.full(4, np.iinfo(np.int64).max // 2, dtype=np.int64)
        self.previous_correction = np.zeros(16, dtype=np.float64)
        self.previous_nominal: np.ndarray | None = None
        self.tracked_nominal: np.ndarray | None = None
        self.recovery_contact_steps = np.zeros(4, dtype=np.int64)
        self.initialized = False

    def _update_contact_measurement(
        self,
        force_world: np.ndarray,
        normal_world: np.ndarray,
        contact_pos_world: np.ndarray,
        found: np.ndarray,
        palm_position_world: np.ndarray,
        world_from_palm: np.ndarray,
    ) -> np.ndarray:
        cfg = self.config
        force_local = np.einsum("ij,fj->fi", world_from_palm.T, force_world)
        normal_local = np.einsum("ij,fj->fi", world_from_palm.T, normal_world)
        contact_pos_world = np.asarray(contact_pos_world, dtype=np.float64).reshape(4, 3)
        palm_position_world = np.asarray(palm_position_world, dtype=np.float64).reshape(3)
        if not self.initialized:
            self.force_filtered[:] = force_local
        alpha_f = cfg.force_filter_alpha
        self.force_filtered[:] = (
            alpha_f * self.force_filtered + (1.0 - alpha_f) * force_local
        )
        force_magnitude = np.linalg.norm(self.force_filtered, axis=-1)
        self.contact_state = np.where(
            self.contact_state,
            found & (force_magnitude >= cfg.contact_off_force),
            found & (force_magnitude >= cfg.contact_on_force),
        )
        self.normal_age += 1
        for finger in range(4):
            candidate = normal_local[finger]
            candidate_norm = np.linalg.norm(candidate)
            if found[finger] and candidate_norm > 0.5:
                # ContactSensor's geometry normal points primary -> secondary:
                # from the fingertip into the contacted object.  Store this
                # inward direction directly for every finger.
                candidate = candidate / candidate_norm
                previous = self.normal_filtered[finger]
                if np.linalg.norm(previous) > 0.5 and np.dot(candidate, previous) < 0.0:
                    candidate = -candidate
                if np.linalg.norm(previous) < 0.5:
                    self.normal_filtered[finger] = candidate
                else:
                    alpha_n = cfg.normal_filter_alpha
                    filtered = alpha_n * previous + (1.0 - alpha_n) * candidate
                    filtered_norm = np.linalg.norm(filtered)
                    if filtered_norm > 1.0e-8:
                        self.normal_filtered[finger] = filtered / filtered_norm
                point_world = contact_pos_world[finger]
                self.contact_point_palm[finger] = world_from_palm.T @ (
                    point_world - palm_position_world
                )
                self.contact_point_valid[finger] = True
                self.normal_age[finger] = 0

        # Control only the force component along the measured surface normal.
        # During brief contact loss the last normal is retained for bounded
        # contact search, but force is correctly treated as zero.
        controlled_force = np.zeros(4, dtype=np.float64)
        for finger in range(4):
            normal = self.normal_filtered[finger]
            if found[finger] and np.linalg.norm(normal) > 0.5:
                controlled_force[finger] = max(
                    0.0, float(np.dot(force_local[finger], normal))
                )
        self.initialized = True
        return controlled_force

    def _update_normal_offset(
        self,
        normal_force: np.ndarray,
        dp_normal_step: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Update a bounded preload relative to the current DP pose.

        This is deliberately first order.  It has no free virtual velocity,
        so contact switching cannot excite the oscillatory second-order state
        used by the previous controller. During weak/lost contact it also
        cancels an outward DP normal step before adding an inward recovery
        step. Thus the moving DP reference cannot outrun contact recovery.
        """
        cfg = self.config
        scale = max(cfg.force_error_full_scale, 1.0e-6)
        offset_step = np.zeros(4, dtype=np.float64)
        contact_mode = np.zeros(4, dtype=np.int8)
        for finger in range(4):
            has_normal = np.linalg.norm(self.normal_filtered[finger]) > 0.5
            if not has_normal:
                continue
            found_now = self.normal_age[finger] == 0
            stable = (
                found_now
                and self.contact_state[finger]
                and cfg.force_min <= normal_force[finger] <= cfg.force_max
            )
            if stable:
                contact_mode[finger] = 1
            elif found_now and normal_force[finger] < cfg.force_min:
                contact_mode[finger] = 2
            elif found_now:
                contact_mode[finger] = 4
            else:
                contact_mode[finger] = 3

            if not found_now:
                desired_normal_step = cfg.recovery_offset_rate
            elif normal_force[finger] < cfg.force_min:
                error = cfg.force_min - normal_force[finger]
                desired_normal_step = (
                    cfg.max_offset_rate * min(error / scale, 1.0)
                )
            elif normal_force[finger] > cfg.force_max:
                error = normal_force[finger] - cfg.force_max
                desired_normal_step = (
                    -cfg.max_offset_rate * min(error / scale, 1.0)
                )
            else:
                desired_normal_step = 0.0

            # Positive motion points into the object.  The force loop owns the
            # entire normal component, rather than waiting until contact has
            # already been lost.  Subtracting the DP normal step makes the
            # resulting command follow ``desired_normal_step`` while leaving
            # DP's tangential and null-space motion untouched.
            delta = desired_normal_step - dp_normal_step[finger]
            delta = np.clip(
                delta,
                -cfg.max_recovery_offset_step,
                cfg.max_recovery_offset_step,
            )

            previous = self.normal_offset[finger]
            self.normal_offset[finger] = np.clip(
                previous + delta,
                -cfg.max_retreat_offset,
                cfg.max_normal_offset,
            )
            offset_step[finger] = self.normal_offset[finger] - previous
        return offset_step, contact_mode

    def _normal_jacobians(
        self, q_nominal: np.ndarray
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        self.data.qpos[:] = q_nominal
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        reduced_jacobians: list[np.ndarray] = []
        normal_jacobians: list[np.ndarray] = []
        for finger, (site_id, dofs) in enumerate(
            zip(self.tip_ids, self.active_dofs)
        ):
            normal = self.normal_filtered[finger]
            reduced = np.zeros((3, len(dofs)), dtype=np.float64)
            reduced_jacobians.append(reduced)
            normal_jacobians.append(np.zeros(len(dofs), dtype=np.float64))
            if np.linalg.norm(normal) < 0.5:
                continue
            jacobian = np.zeros((3, self.model.nv), dtype=np.float64)
            jacobian_rot = np.zeros_like(jacobian)
            if self.contact_point_valid[finger] and self.normal_age[finger] == 0:
                mujoco.mj_jac(
                    self.model,
                    self.data,
                    jacobian,
                    jacobian_rot,
                    self.contact_point_palm[finger],
                    self.tip_body_ids[finger],
                )
            else:
                mujoco.mj_jacSite(
                    self.model, self.data, jacobian, jacobian_rot, site_id
                )
            reduced = jacobian[:, dofs]
            reduced_jacobians[-1] = reduced.copy()
            normal_jacobians[-1] = (normal @ reduced).copy()
        return reduced_jacobians, normal_jacobians

    def _contact_guarded_nominal(
        self,
        q_nominal: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Freeze one lost finger, then smoothly release it back to live DP.

        This anchor is per finger and temporary.  It does not freeze DP
        inference or the other fingers.  Contact recovery uses the low
        hysteretic contact threshold, not the target-force band.  Once contact
        has been present for several consecutive frames, the anchor follows
        the current DP pose with a bounded joint rate while the independent
        normal-force loop continues driving force toward its target band.
        """
        if self.tracked_nominal is None:
            self.tracked_nominal = q_nominal.copy()
        if not self.config.nominal_guard_enabled:
            self.tracked_nominal[:] = q_nominal
            self.recovery_contact_steps[:] = self.contact_state.astype(np.int64)
            return self.tracked_nominal.copy(), np.zeros(4, dtype=bool)
        frozen = np.ones(4, dtype=bool)
        for finger, dofs in enumerate(self.active_dofs):
            if self.contact_state[finger]:
                self.recovery_contact_steps[finger] += 1
            else:
                self.recovery_contact_steps[finger] = 0
            if (
                self.recovery_contact_steps[finger]
                >= self.config.recovery_confirm_steps
            ):
                delta = np.clip(
                    q_nominal[dofs] - self.tracked_nominal[dofs],
                    -self.config.nominal_release_rate,
                    self.config.nominal_release_rate,
                )
                self.tracked_nominal[dofs] += delta
                frozen[finger] = False
        return self.tracked_nominal.copy(), frozen

    def _joint_correction(
        self,
        reduced_jacobians: list[np.ndarray],
        normal_jacobians: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cfg = self.config
        correction = np.zeros(16, dtype=np.float64)
        for finger, (dofs, normal_jacobian) in enumerate(
            zip(self.active_dofs, normal_jacobians)
        ):
            if not np.any(normal_jacobian):
                continue
            # One scalar normal task per finger.  Tangential fingertip motion
            # remains governed by DP instead of being overwritten by a 3-D
            # contact optimizer.
            denominator = float(
                normal_jacobian @ normal_jacobian
                + cfg.jacobian_damping**2
            )
            delta = (
                normal_jacobian * self.normal_offset[finger] / denominator
            )
            correction[dofs] = delta
        correction = np.clip(
            correction, -cfg.max_joint_correction, cfg.max_joint_correction
        )
        correction = np.clip(
            correction,
            self.previous_correction - cfg.max_joint_rate,
            self.previous_correction + cfg.max_joint_rate,
        )
        predicted_tip_displacement = np.zeros((4, 3), dtype=np.float64)
        predicted_normal_displacement = np.zeros(4, dtype=np.float64)
        for finger, (dofs, reduced) in enumerate(
            zip(self.active_dofs, reduced_jacobians)
        ):
            displacement = reduced @ correction[dofs]
            predicted_tip_displacement[finger] = displacement
            predicted_normal_displacement[finger] = np.dot(
                displacement, self.normal_filtered[finger]
            )
        return (
            correction,
            predicted_tip_displacement,
            predicted_normal_displacement,
        )

    def prime(
        self,
        q_nominal: np.ndarray,
        force_world: np.ndarray,
        normal_world: np.ndarray,
        contact_pos_world: np.ndarray,
        found: np.ndarray,
        palm_position_world: np.ndarray,
        palm_quaternion_wxyz: np.ndarray,
    ) -> None:
        """Warm up force/normal state without applying a position correction."""
        rotation = _quat_wxyz_to_matrix(palm_quaternion_wxyz)
        self._update_contact_measurement(
            np.asarray(force_world, dtype=np.float64).reshape(4, 3),
            np.asarray(normal_world, dtype=np.float64).reshape(4, 3),
            np.asarray(contact_pos_world, dtype=np.float64).reshape(4, 3),
            np.asarray(found, dtype=bool).reshape(4),
            np.asarray(palm_position_world, dtype=np.float64).reshape(3),
            rotation,
        )
        self.normal_offset.fill(0.0)
        self.previous_correction.fill(0.0)
        self.previous_nominal = np.asarray(q_nominal, dtype=np.float64).copy()
        self.tracked_nominal = self.previous_nominal.copy()
        self.recovery_contact_steps.fill(0)

    def update(
        self,
        q_nominal: np.ndarray,
        force_world: np.ndarray,
        normal_world: np.ndarray,
        contact_pos_world: np.ndarray,
        found: np.ndarray,
        palm_position_world: np.ndarray,
        palm_quaternion_wxyz: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Return a safe position command and controller diagnostics."""
        q_nominal = np.asarray(q_nominal, dtype=np.float64)
        force_world = np.asarray(force_world, dtype=np.float64).reshape(4, 3)
        normal_world = np.asarray(normal_world, dtype=np.float64).reshape(4, 3)
        contact_pos_world = np.asarray(contact_pos_world, dtype=np.float64).reshape(4, 3)
        found = np.asarray(found, dtype=bool).reshape(4)
        if q_nominal.shape != (16,):
            raise ValueError(f"q_nominal must be 16-D, got {q_nominal.shape}")
        rotation = _quat_wxyz_to_matrix(palm_quaternion_wxyz)
        magnitude = self._update_contact_measurement(
            force_world,
            normal_world,
            contact_pos_world,
            found,
            palm_position_world,
            rotation,
        )
        guarded_nominal, nominal_frozen = self._contact_guarded_nominal(
            q_nominal,
        )
        reduced_jacobians, normal_jacobians = self._normal_jacobians(
            guarded_nominal
        )
        if self.previous_nominal is None:
            nominal_step = np.zeros(16, dtype=np.float64)
        else:
            nominal_step = guarded_nominal - self.previous_nominal
        dp_normal_step = np.asarray(
            [
                float(normal_jacobian @ nominal_step[dofs])
                for dofs, normal_jacobian in zip(
                    self.active_dofs, normal_jacobians
                )
            ],
            dtype=np.float64,
        )
        offset_step, contact_mode = self._update_normal_offset(
            magnitude, dp_normal_step
        )
        (
            correction,
            predicted_tip_displacement,
            predicted_normal_displacement,
        ) = self._joint_correction(reduced_jacobians, normal_jacobians)
        lower = self.lower + self.config.joint_limit_margin
        upper = self.upper - self.config.joint_limit_margin
        q_command = np.clip(guarded_nominal + correction, lower, upper)
        correction = q_command - guarded_nominal
        self.previous_correction[:] = correction
        self.previous_nominal = guarded_nominal.copy()
        return q_command.astype(np.float32), {
            "force_magnitude": magnitude.astype(np.float32),
            "normal_offset": self.normal_offset.astype(np.float32).copy(),
            "joint_correction": correction.astype(np.float32).copy(),
            "contact_state": self.contact_state.copy(),
            "contact_mode": contact_mode.copy(),
            "normal_local": self.normal_filtered.astype(np.float32).copy(),
            "dp_normal_step": dp_normal_step.astype(np.float32).copy(),
            "offset_step": offset_step.astype(np.float32).copy(),
            "guarded_nominal": guarded_nominal.astype(np.float32).copy(),
            "nominal_guard_correction": (
                guarded_nominal - q_nominal
            ).astype(np.float32),
            "nominal_frozen": nominal_frozen.copy(),
            "recovery_contact_steps": self.recovery_contact_steps.copy(),
            "predicted_tip_displacement": (
                predicted_tip_displacement.astype(np.float32).copy()
            ),
            "predicted_normal_displacement": (
                predicted_normal_displacement.astype(np.float32).copy()
            ),
        }
