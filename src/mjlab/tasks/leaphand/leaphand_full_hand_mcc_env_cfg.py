"""Full-hand MCC adapter for xArm6 + LEAP Hand.

Architecture:

* contact 0 (palm root): existing arm-torque MCC controller;
* contacts 1..4 (fingertips): 16 motor-torque residuals -> four Cartesian
  force estimates -> MCC references -> constrained multi-site IK;
* all five input points are checked by :class:`FivePointReachabilitySolver`
  against the actual MuJoCo/URDF kinematic chain and joint limits.

Surface-specific target generation intentionally remains outside this module.
See ``full_hand_mcc/scripts/demo_surface_slide.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, cast

import mink
import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
import mjlab.tasks.leaphand.leaphand_palm_mcc_env_cfg as _palm_mcc_module

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.leaphand.full_hand_mcc_core import (
    FullHandMCCCore,
    FullHandMCCGains,
    MCC_VARIANTS,
    MCCVariant,
)
from mjlab.tasks.leaphand.leaphand_mcc_finger_env_cfg import (
    DEFAULT_PREGRASP_Q,
    MCC_TARGET_ARM_Q,
    MCC_TIP_GEOM_NAMES,
    MCC_TIP_NAMES,
    _LEAPHAND_XML as _TACTILE_ROBOT_XML,
    _get_hard_contact_target_spec,
    _load_fixed_palm_mcc_hand_spec,
    _load_mcc_leaphand_spec,
    fingertip_force_3d,
    joint_pos,
    mcc_finger_contact_env_cfg,
)
from mjlab.tasks.leaphand.leaphand_palm_mcc_env_cfg import (
    MCCPalmComplianceController,
    joint_vel_arm,
    palm_jacobian,
    palm_jacobian_rot,
    palm_pos,
    palm_rot,
    qfrc_actuator_arm,
    qfrc_bias_arm,
    target_body_pos,
    target_body_rot,
)


PALM_CONTROL_SITE = "full_hand_palm_contact"
PALM_CONTROL_OFFSET_LOCAL = (-0.0559703, -0.04142053, -0.0340008)
FULL_HAND_CAPSULE_RADIUS = 0.02
FULL_HAND_CAPSULE_HALF_HEIGHT = 0.235
HAND_QPOS_NAMES = (
    "1", "0", "2", "3",
    "5", "4", "6", "7",
    "9", "8", "10", "11",
    "12", "13", "14", "15",
)


def _get_full_hand_contact_target_spec() -> mujoco.MjSpec:
    """Use a capsule that collides only with the four tactile fingertip pads."""

    spec = _get_hard_contact_target_spec()
    geom = spec.geom("target_capsule_medium_geom")
    if geom is None:
        raise ValueError("target_capsule_medium_geom is missing")
    geom.size[:] = (
        FULL_HAND_CAPSULE_RADIUS,
        FULL_HAND_CAPSULE_HALF_HEIGHT,
        0.0,
    )
    # Dedicated target collision bit.  Non-tip robot geoms retain the default
    # bit 1, while the four tactile pads opt into bit 2 below.
    geom.contype = 2
    geom.conaffinity = 0
    return spec


def _load_full_hand_robot_spec() -> mujoco.MjSpec:
    """Enable target collision only on the four measured fingertip geoms."""

    spec = _load_mcc_leaphand_spec()
    fingertip_geoms = set(MCC_TIP_GEOM_NAMES)
    found: set[str] = set()
    for geom in spec.geoms:
        name = geom.name or ""
        if name in fingertip_geoms:
            geom.conaffinity = int(geom.conaffinity) | 2
            found.add(name)
    missing = fingertip_geoms - found
    if missing:
        raise ValueError(f"Missing tactile fingertip geoms: {sorted(missing)}")
    return spec


def joint_vel_hand(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    return env.scene[asset_cfg.name].data.joint_vel[:, 6:22]


def qfrc_actuator_hand(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    return env.scene[asset_cfg.name].data.qfrc_actuator[:, 6:22]


def qfrc_bias_hand(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return the 16 hand bias torques with a batched MJWarp fallback."""

    asset = env.scene[asset_cfg.name]
    direct = getattr(asset.data, "qfrc_bias", None)
    if direct is not None:
        return direct[:, 6:22]
    if hasattr(env.sim.data, "struct") and hasattr(env.sim.data.struct, "qfrc_bias"):
        bias = env.sim.data.struct.qfrc_bias
        if not torch.is_tensor(bias):
            bias = torch.as_tensor(
                bias.numpy(), device=env.device, dtype=torch.float32
            )
        if bias.ndim == 1:
            bias = bias.unsqueeze(0)
        return bias[:, 6:22]
    return torch.zeros((env.num_envs, 16), device=env.device)


def full_hand_mcc_env_cfg(
    num_envs: int = 1,
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Reuse the tactile environment and expose both MCC feedback paths."""

    cfg = mcc_finger_contact_env_cfg(num_envs=num_envs, play=play)
    robot_cfg = cfg.scene.entities["robot"]
    robot_cfg.spec_fn = _load_full_hand_robot_spec
    # The standalone finger task intentionally uses a very soft 5 Nm/rad
    # position servo.  In the five-contact task that servo stalls against the
    # capsule before reaching the collision-consistent IK reference, which
    # looks like an uncommanded natural closure.  Keep the arm settings and
    # strengthen only the LEAP joints for reliable nominal tracking; MCC still
    # supplies the bounded force-feedback correction around that nominal pose.
    hand_actuator = robot_cfg.articulation.actuators[1]
    hand_actuator.stiffness = 35.0
    hand_actuator.damping = 2.5
    hand_actuator.effort_limit = 35.0
    robot_cfg.init_state.joint_pos.update(
        {
            **{
                f"joint{joint_id + 1}": float(value)
                for joint_id, value in enumerate(MCC_TARGET_ARM_Q)
            },
            **{
                joint_name: float(value)
                for joint_name, value in zip(
                    HAND_QPOS_NAMES, DEFAULT_PREGRASP_Q, strict=True
                )
            },
        }
    )
    cfg.scene.entities["target"].spec_fn = _get_full_hand_contact_target_spec
    # Put all four initial fingertip contacts near the lower end of the
    # straight cylinder.  A 200 mm slide then finishes near the upper end
    # without forcing any finger through a capsule end-cap curvature change.
    # Keep the original upper end fixed while extending the useful sliding
    # surface downward by 0.20 m.  This avoids introducing new target geometry
    # into the palm/proximal-link region above the fingertip contact band.
    cfg.scene.entities["target"].init_state.pos = (0.7007, 0.0003, 0.7077)
    cfg.actions["hand_delta"] = JointPositionActionCfg(
        entity_name="robot",
        actuator_names=(r"^[0-9]+$",),
        use_default_offset=False,
    )
    cfg.viewer.origin_type = cfg.viewer.OriginType.ASSET_BODY
    cfg.viewer.entity_name = "target"
    cfg.viewer.body_name = "target_ball"
    cfg.viewer.distance = 0.82
    cfg.viewer.elevation = -18.0
    cfg.viewer.azimuth = 145.0
    robot = SceneEntityCfg("robot")
    cfg.observations = {
        "palm": ObservationGroupCfg(
            {
                "joint_pos": ObservationTermCfg(func=joint_pos, params={"asset_cfg": robot}),
                "joint_vel_arm": ObservationTermCfg(
                    func=joint_vel_arm, params={"asset_cfg": robot}
                ),
                "qfrc_actuator_arm": ObservationTermCfg(
                    func=qfrc_actuator_arm, params={"asset_cfg": robot}
                ),
                "qfrc_bias_arm": ObservationTermCfg(
                    func=qfrc_bias_arm, params={"asset_cfg": robot}
                ),
                "palm_jacobian": ObservationTermCfg(
                    func=palm_jacobian,
                    params={"asset_cfg": robot, "body_name": "palm_lower"},
                ),
                "palm_jacobian_rot": ObservationTermCfg(
                    func=palm_jacobian_rot,
                    params={"asset_cfg": robot, "body_name": "palm_lower"},
                ),
                "palm_pos": ObservationTermCfg(
                    func=palm_pos,
                    params={"asset_cfg": robot, "body_name": "palm_lower"},
                ),
                "palm_rot": ObservationTermCfg(
                    func=palm_rot,
                    params={"asset_cfg": robot, "body_name": "palm_lower"},
                ),
                "target_pos": ObservationTermCfg(
                    func=target_body_pos,
                    params={"asset_cfg": robot, "body_name": "target_ball"},
                ),
                "target_rot": ObservationTermCfg(
                    func=target_body_rot,
                    params={"asset_cfg": robot, "body_name": "target_ball"},
                ),
            }
        ),
        # Layout: force(12), q(22), qd_hand(16), tau_motor(16), bias_hand(16).
        "finger": ObservationGroupCfg(
            {
                "fingertip_force_3d": ObservationTermCfg(func=fingertip_force_3d),
                "joint_pos": ObservationTermCfg(func=joint_pos, params={"asset_cfg": robot}),
                "joint_vel_hand": ObservationTermCfg(
                    func=joint_vel_hand, params={"asset_cfg": robot}
                ),
                "qfrc_actuator_hand": ObservationTermCfg(
                    func=qfrc_actuator_hand, params={"asset_cfg": robot}
                ),
                "qfrc_bias_hand": ObservationTermCfg(
                    func=qfrc_bias_hand, params={"asset_cfg": robot}
                ),
            }
        ),
    }
    return cfg


class ReachabilityResult(NamedTuple):
    accepted: bool
    joint_position: np.ndarray
    achieved_points: np.ndarray
    residual_m: np.ndarray
    iterations: int


class PalmPoseResult(NamedTuple):
    accepted: bool
    joint_position: np.ndarray
    position_error_m: float
    orientation_error_rad: float
    iterations: int


def _spec_with_palm_contact_site() -> mujoco.MjSpec:
    spec = _load_mcc_leaphand_spec()
    palm = spec.body("palm_lower")
    if palm is None:
        raise ValueError("palm_lower is missing from the xArm6 + LEAP model")
    if spec.site(PALM_CONTROL_SITE) is None:
        palm.add_site(
            name=PALM_CONTROL_SITE,
            pos=PALM_CONTROL_OFFSET_LOCAL,
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=(0.004, 0.0, 0.0),
            rgba=(0.2, 1.0, 0.2, 0.8),
        )
    return spec


class FivePointReachabilitySolver:
    """Joint-limit-aware five-site IK using the real robot model.

    A target is accepted only when all five residuals are below ``tolerance``.
    Rejected targets are never sent to the controller; callers can line-search
    between their last accepted surface target and the requested target.
    """

    def __init__(
        self,
        tolerance: float = 0.004,
        damping: float = 2.0e-3,
        max_iterations: int = 80,
        max_joint_step: float = 0.08,
        palm_weight: float = 2.0,
        posture_regularization: float = 1.0e-4,
    ) -> None:
        self.model = _spec_with_palm_contact_site().compile()
        self.data = mujoco.MjData(self.model)
        self.tolerance = float(tolerance)
        self.damping = float(damping)
        self.max_iterations = int(max_iterations)
        self.max_joint_step = float(max_joint_step)
        self.posture_regularization = float(posture_regularization)
        self.site_names = (PALM_CONTROL_SITE, *MCC_TIP_NAMES)
        self.site_ids = np.asarray(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
                for name in self.site_names
            ],
            dtype=np.int32,
        )
        if np.any(self.site_ids < 0):
            raise ValueError(f"Missing one of the five contact sites: {self.site_names}")
        self.palm_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "palm_lower"
        )
        if self.palm_body_id < 0:
            raise ValueError("palm_lower body is missing from the reachability model")

        joint_names = tuple(f"joint{i}" for i in range(1, 7)) + HAND_QPOS_NAMES
        self.qpos_indices: list[int] = []
        self.dof_indices: list[int] = []
        lower: list[float] = []
        upper: list[float] = []
        for name in joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"Joint {name!r} missing from robot model")
            self.qpos_indices.append(int(self.model.jnt_qposadr[jid]))
            self.dof_indices.append(int(self.model.jnt_dofadr[jid]))
            if self.model.jnt_limited[jid]:
                lo, hi = self.model.jnt_range[jid]
            else:
                lo, hi = -np.inf, np.inf
            lower.append(float(lo))
            upper.append(float(hi))
        self.qpos_indices_np = np.asarray(self.qpos_indices, dtype=np.int32)
        self.dof_indices_np = np.asarray(self.dof_indices, dtype=np.int32)
        self.lower = np.asarray(lower)
        self.upper = np.asarray(upper)
        self.weights = np.repeat(
            np.asarray([palm_weight, 1.0, 1.0, 1.0, 1.0]), 3
        )

    def forward_points(self, joint_position: np.ndarray) -> np.ndarray:
        q = np.asarray(joint_position, dtype=np.float64).reshape(22)
        self.data.qpos[self.qpos_indices_np] = q
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        return self.data.site_xpos[self.site_ids].copy()

    def forward_palm_pose(
        self, joint_position: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        self.forward_points(joint_position)
        return (
            self.data.xpos[self.palm_body_id].copy(),
            self.data.xmat[self.palm_body_id].reshape(3, 3).copy(),
        )

    def solve_palm_pose(
        self,
        target_position: np.ndarray,
        target_rotation: np.ndarray,
        seed_joint_position: np.ndarray,
        position_tolerance: float = 5.0e-4,
        orientation_tolerance: float = 2.0e-3,
        max_iterations: int = 80,
    ) -> PalmPoseResult:
        """Solve a fixed-orientation arm seed before simultaneous five-point IK."""

        target_pos = np.asarray(target_position, dtype=np.float64).reshape(3)
        target_rot = np.asarray(target_rotation, dtype=np.float64).reshape(3, 3)
        q = np.asarray(seed_joint_position, dtype=np.float64).reshape(22).copy()
        q = np.minimum(np.maximum(q, self.lower), self.upper)
        best_q = q.copy()
        best_cost = np.inf
        best_pos_error = np.inf
        best_rot_error = np.inf
        iteration = 0

        for iteration in range(1, max_iterations + 1):
            current_pos, current_rot = self.forward_palm_pose(q)
            pos_error = target_pos - current_pos
            rot_error = R.from_matrix(target_rot @ current_rot.T).as_rotvec()
            pos_norm = float(np.linalg.norm(pos_error))
            rot_norm = float(np.linalg.norm(rot_error))
            cost = pos_norm**2 + 0.2 * rot_norm**2
            if cost < best_cost:
                best_cost = cost
                best_q = q.copy()
                best_pos_error = pos_norm
                best_rot_error = rot_norm
            if (
                pos_norm <= position_tolerance
                and rot_norm <= orientation_tolerance
            ):
                return PalmPoseResult(True, q, pos_norm, rot_norm, iteration)

            jacp = np.zeros((3, self.model.nv), dtype=np.float64)
            jacr = np.zeros((3, self.model.nv), dtype=np.float64)
            mujoco.mj_jacBody(
                self.model,
                self.data,
                jacp,
                jacr,
                self.palm_body_id,
            )
            arm_dofs = self.dof_indices_np[:6]
            jacobian = np.vstack((jacp[:, arm_dofs], jacr[:, arm_dofs]))
            weights = np.asarray([1.0, 1.0, 1.0, 0.2, 0.2, 0.2])
            weighted_j = weights[:, None] * jacobian
            lhs = jacobian.T @ weighted_j + 1.0e-3 * np.eye(6)
            rhs = jacobian.T @ (
                weights * np.concatenate((pos_error, rot_error))
            )
            dq_arm = np.linalg.solve(lhs, rhs)
            dq_arm = np.clip(
                dq_arm,
                -self.max_joint_step,
                self.max_joint_step,
            )

            accepted_step = False
            scale = 1.0
            while scale >= 1.0 / 32.0:
                candidate = q.copy()
                candidate[:6] += scale * dq_arm
                candidate = np.minimum(
                    np.maximum(candidate, self.lower),
                    self.upper,
                )
                candidate_pos, candidate_rot = self.forward_palm_pose(candidate)
                candidate_pos_error = target_pos - candidate_pos
                candidate_rot_error = R.from_matrix(
                    target_rot @ candidate_rot.T
                ).as_rotvec()
                candidate_cost = float(
                    candidate_pos_error @ candidate_pos_error
                    + 0.2 * candidate_rot_error @ candidate_rot_error
                )
                if candidate_cost < cost:
                    q = candidate
                    accepted_step = True
                    break
                scale *= 0.5
            if not accepted_step:
                break

        return PalmPoseResult(
            False,
            best_q,
            best_pos_error,
            best_rot_error,
            iteration,
        )

    def _stacked_jacobian(self) -> np.ndarray:
        blocks: list[np.ndarray] = []
        for site_id in self.site_ids:
            jacp = np.zeros((3, self.model.nv), dtype=np.float64)
            jacr = np.zeros((3, self.model.nv), dtype=np.float64)
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, int(site_id))
            blocks.append(jacp[:, self.dof_indices_np])
        return np.vstack(blocks)

    def solve(
        self,
        target_points: np.ndarray,
        seed_joint_position: np.ndarray,
    ) -> ReachabilityResult:
        targets = np.asarray(target_points, dtype=np.float64).reshape(5, 3)
        q = np.asarray(seed_joint_position, dtype=np.float64).reshape(22).copy()
        q = np.minimum(np.maximum(q, self.lower), self.upper)
        posture_anchor = q.copy()
        best_q = q.copy()
        best_points = self.forward_points(q)
        best_cost = float(
            np.sum(self.weights * (targets - best_points).ravel() ** 2)
        )
        iterations = 0

        for iterations in range(1, self.max_iterations + 1):
            points = self.forward_points(q)
            error = (targets - points).ravel()
            residual = np.linalg.norm(error.reshape(5, 3), axis=1)
            if float(residual.max()) <= self.tolerance:
                return ReachabilityResult(True, q, points, residual, iterations)

            jacobian = self._stacked_jacobian()
            weighted_j = self.weights[:, None] * jacobian
            lhs = jacobian.T @ weighted_j
            lhs += (
                self.damping + self.posture_regularization
            ) * np.eye(lhs.shape[0])
            rhs = (
                jacobian.T @ (self.weights * error)
                - self.posture_regularization * (q - posture_anchor)
            )
            dq = np.linalg.solve(lhs, rhs)
            dq = np.clip(dq, -self.max_joint_step, self.max_joint_step)

            accepted_step = False
            scale = 1.0
            while scale >= 1.0 / 32.0:
                candidate = q + scale * dq
                candidate = np.minimum(np.maximum(candidate, self.lower), self.upper)
                candidate_points = self.forward_points(candidate)
                candidate_cost = float(
                    np.sum(
                        self.weights
                        * (targets - candidate_points).ravel() ** 2
                    )
                    + self.posture_regularization
                    * np.sum((candidate - posture_anchor) ** 2)
                )
                if candidate_cost < best_cost:
                    q = candidate
                    best_q = candidate.copy()
                    best_points = candidate_points.copy()
                    best_cost = candidate_cost
                    accepted_step = True
                    break
                scale *= 0.5
            if not accepted_step:
                break

        residual = np.linalg.norm(targets - best_points, axis=1)
        return ReachabilityResult(
            bool(float(residual.max()) <= self.tolerance),
            best_q,
            best_points,
            residual,
            iterations,
        )


class MotorForceFingerMCCController:
    """Four fingertip MCC driven by all 16 measured motor torque residuals."""

    def __init__(
        self,
        device: str,
        num_envs: int,
        variant: MCCVariant,
        **kwargs,
    ) -> None:
        self.device = device
        self.num_envs = int(num_envs)
        self.variant = variant
        self.control_dt = float(kwargs.get("control_dt", 0.01))
        self.mink_damping = float(kwargs.get("mink_damping", 0.1))
        self.mink_num_iter = int(kwargs.get("mink_num_iter", 3))
        self.action_rate_limit = float(kwargs.get("action_rate_limit", 0.18))
        self.motor_force_gain = float(kwargs.get("motor_force_gain", 0.015))
        self.force_regularization = float(kwargs.get("force_regularization", 1.0e-3))
        self.nominal_tracking_radius = float(
            kwargs.get("finger_mcc_tracking_radius", 0.15)
        )
        self.force_closure_gain = float(
            kwargs.get("finger_force_closure_gain", 0.02)
        )
        self.max_release_correction = float(
            kwargs.get("finger_max_release_correction", 0.01)
        )
        self.normal_preload_m = float(
            kwargs.get("finger_normal_preload_m", 0.0015)
        )
        self.normal_compliance_m_per_n = float(
            kwargs.get("finger_normal_compliance_m_per_n", 0.00035)
        )
        self.pregrasp_q = np.asarray(
            kwargs.get("pregrasp_q", DEFAULT_PREGRASP_Q), dtype=np.float64
        )
        self.model = _load_fixed_palm_mcc_hand_spec().compile()
        self.data = mujoco.MjData(self.model)
        self.config = mink.Configuration(self.model)
        self.tip_ids = np.asarray(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
                for name in MCC_TIP_NAMES
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
            for name in MCC_TIP_NAMES
        ]
        self.posture_task = mink.PostureTask(self.model, cost=0.08)
        self.limits = [mink.ConfigurationLimit(self.model)]

        self.world_model = _load_mcc_leaphand_spec().compile()
        self.world_data = mujoco.MjData(self.world_model)
        self.world_palm_id = mujoco.mj_name2id(
            self.world_model, mujoco.mjtObj.mjOBJ_BODY, "palm_lower"
        )
        gains = FullHandMCCGains(
            dt=self.control_dt,
            desired_palm_force=float(kwargs.get("palm_desired_force", 3.0)),
            desired_fingertip_force=float(kwargs.get("finger_desired_force", 1.0)),
            tangent_kp=float(kwargs.get("finger_tangent_kp", 18.0)),
            force_kp=float(kwargs.get("finger_force_kp", 0.004)),
            force_ki=float(kwargs.get("finger_force_ki", 0.001)),
            max_reference_speed=float(kwargs.get("max_tip_speed", 0.04)),
        )
        self.core = FullHandMCCCore(variant=variant, gains=gains)
        self.motor_force_setpoint = np.full(
            4,
            self.core.gains.desired_fingertip_force,
            dtype=np.float64,
        )
        self.prev_action = torch.zeros((num_envs, 16), device=device)
        self.prev_action_initialized = False
        self.last_debug: dict[str, torch.Tensor] = {}

    def reset(self) -> None:
        self.core.reset()
        self.prev_action.zero_()
        self.prev_action_initialized = False

    def calibrate_motor_force_setpoint(
        self, force_magnitude: np.ndarray
    ) -> None:
        """Use true-contact motor residuals as per-finger load references."""

        baseline = np.asarray(force_magnitude, dtype=np.float64).reshape(4)
        if not np.all(np.isfinite(baseline)):
            raise ValueError("motor force baseline must be finite")
        self.motor_force_setpoint = np.maximum(baseline, 0.1)

    def _set_hand_q(self, q_hand: np.ndarray) -> None:
        self.data.qpos[:] = q_hand
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _world_palm_pose(self, q_full: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.world_data.qpos[:] = q_full
        self.world_data.qvel[:] = 0.0
        mujoco.mj_forward(self.world_model, self.world_data)
        return (
            self.world_data.xpos[self.world_palm_id].copy(),
            self.world_data.xmat[self.world_palm_id].reshape(3, 3).copy(),
        )

    def _tip_positions_and_jacobians(self) -> tuple[np.ndarray, list[np.ndarray]]:
        positions = self.data.site_xpos[self.tip_ids].copy()
        jacobians: list[np.ndarray] = []
        for site_id in self.tip_ids:
            jacp = np.zeros((3, self.model.nv), dtype=np.float64)
            jacr = np.zeros((3, self.model.nv), dtype=np.float64)
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, int(site_id))
            jacobians.append(jacp.copy())
        return positions, jacobians

    def _motor_torque_to_tip_forces(
        self,
        jacobians: list[np.ndarray],
        external_motor_torque: np.ndarray,
    ) -> np.ndarray:
        forces = np.zeros((4, 3), dtype=np.float64)
        for finger, jacobian in enumerate(jacobians):
            active = np.linalg.norm(jacobian, axis=0) > 1.0e-9
            j_active = jacobian[:, active]
            tau_active = external_motor_torque[active]
            if j_active.shape[1] == 0:
                continue
            lhs = j_active @ j_active.T
            lhs += self.force_regularization * np.eye(3)
            forces[finger] = np.linalg.solve(lhs, j_active @ tau_active)
        return forces

    def __call__(
        self,
        finger_obs: torch.Tensor,
        palm_target_world: torch.Tensor,
        palm_normal_world: torch.Tensor,
        tip_targets_world: torch.Tensor,
        tip_normals_world: torch.Tensor,
        nominal_hand_q: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q_full_batch = finger_obs[:, 12:34].detach().cpu().numpy().astype(np.float64)
        tau_motor = finger_obs[:, 50:66].detach().cpu().numpy().astype(np.float64)
        bias_motor = finger_obs[:, 66:82].detach().cpu().numpy().astype(np.float64)
        targets_world = tip_targets_world.detach().cpu().numpy().astype(np.float64)
        normals_world = tip_normals_world.detach().cpu().numpy().astype(np.float64)
        palm_target_np = palm_target_world.detach().cpu().numpy().astype(np.float64)
        palm_normal_np = palm_normal_world.detach().cpu().numpy().astype(np.float64)
        nominal_hand_np = (
            nominal_hand_q.detach().cpu().numpy().astype(np.float64)
            if nominal_hand_q is not None
            else None
        )
        batch = q_full_batch.shape[0]

        palm_actual_local_batch = np.broadcast_to(
            np.asarray(PALM_CONTROL_OFFSET_LOCAL, dtype=np.float64),
            (batch, 1, 3),
        ).copy()
        palm_desired_local_batch = np.zeros((batch, 1, 3))
        palm_normals_local_batch = np.zeros((batch, 1, 3))
        palm_force_local_batch = np.zeros((batch, 1, 3))
        actual_local_batch = np.zeros((batch, 4, 3))
        desired_local_batch = np.zeros((batch, 4, 3))
        normals_local_batch = np.zeros((batch, 4, 3))
        forces_local_batch = np.zeros((batch, 4, 3))
        jacobians_batch: list[list[np.ndarray]] = []
        palm_poses: list[tuple[np.ndarray, np.ndarray]] = []
        external_tau_batch = -(tau_motor - bias_motor)

        for env_id in range(batch):
            q_full = q_full_batch[env_id]
            self._set_hand_q(q_full[6:22])
            positions, jacobians = self._tip_positions_and_jacobians()
            palm_origin, palm_rotation = self._world_palm_pose(q_full)
            actual_local_batch[env_id] = positions
            desired_local_batch[env_id] = (
                palm_rotation.T @ (targets_world[env_id] - palm_origin).T
            ).T
            palm_desired_local_batch[env_id, 0] = (
                palm_rotation.T @ (palm_target_np[env_id] - palm_origin)
            )
            normals_local_batch[env_id] = (
                palm_rotation.T @ normals_world[env_id].T
            ).T
            palm_normals_local_batch[env_id, 0] = (
                palm_rotation.T @ palm_normal_np[env_id]
            )
            palm_force_local_batch[env_id, 0] = (
                self.core.gains.desired_palm_force
                * palm_normals_local_batch[env_id, 0]
            )
            forces_local_batch[env_id] = self._motor_torque_to_tip_forces(
                jacobians, external_tau_batch[env_id]
            )
            jacobians_batch.append(jacobians)
            palm_poses.append((palm_origin, palm_rotation))

        # The existing palm MCC owns contact 0.  Its kinematic target is still
        # included here so hierarchical_mcc can express each fingertip relative
        # to the moving palm.  A synthetic force exactly at the palm setpoint
        # keeps this coordinator from duplicating the arm force loop.
        finger_step = self.core.step(
            np.concatenate((palm_actual_local_batch, actual_local_batch), axis=1),
            np.concatenate((palm_desired_local_batch, desired_local_batch), axis=1),
            np.concatenate((palm_normals_local_batch, normals_local_batch), axis=1),
            np.concatenate((palm_force_local_batch, forces_local_batch), axis=1),
        )
        reference_local_batch = finger_step.reference_points[:, 1:]

        q_ref_batch = np.zeros((batch, 16), dtype=np.float32)
        force_joint_correction_batch = np.zeros((batch, 16), dtype=np.float32)
        ik_local_batch = np.zeros((batch, 4, 3), dtype=np.float32)
        for env_id in range(batch):
            q_hand = q_full_batch[env_id, 6:22]
            self.config.data.qpos[:] = q_hand
            mujoco.mj_forward(self.config.model, self.config.data)
            self.posture_task.set_target_from_configuration(self.config)
            for task, target, site_id in zip(
                self.tasks, reference_local_batch[env_id], self.tip_ids
            ):
                rotation = mink.SO3.from_matrix(
                    self.config.data.site_xmat[site_id].reshape(3, 3).copy()
                )
                task.set_target(
                    mink.SE3.from_rotation_and_translation(rotation, target)
                )
            for _ in range(self.mink_num_iter):
                velocity = mink.solve_ik(
                    self.config,
                    [self.posture_task, *self.tasks],
                    self.control_dt,
                    solver="daqp",
                    damping=self.mink_damping,
                    limits=self.limits,
                )
                self.config.integrate_inplace(velocity, self.control_dt)
            q_ref = self.config.data.qpos.copy()

            if self.variant == "motor_torque_mcc":
                desired_external_tau = np.zeros(16, dtype=np.float64)
                for jacobian, normal in zip(
                    jacobians_batch[env_id], normals_local_batch[env_id]
                ):
                    desired_external_tau += jacobian.T @ (
                        self.core.gains.desired_fingertip_force * normal
                    )
                external_torque_error = (
                    desired_external_tau - external_tau_batch[env_id]
                )
                # The position actuator must generate the reaction torque
                # opposite to the desired environment torque.
                q_ref -= self.motor_force_gain * external_torque_error

            if nominal_hand_np is not None:
                # The simultaneous five-point solver owns all tangential
                # geometry.  Apply only a normal motor-force MCC correction
                # around it.  Reusing the unconstrained Cartesian reference
                # here allowed its internal state to drift into a natural
                # closure even after a valid five-contact pose was supplied.
                nominal_q = nominal_hand_np[env_id]
                joint_correction = np.zeros(16, dtype=np.float64)
                for finger, (normal, measured_force) in enumerate(zip(
                    normals_local_batch[env_id],
                    forces_local_batch[env_id],
                )):
                    desired_force = self.motor_force_setpoint[finger]
                    measured_normal = abs(float(np.dot(measured_force, normal)))
                    force_error = desired_force - measured_normal
                    normal_displacement = (
                        self.normal_preload_m
                        + self.normal_compliance_m_per_n * force_error
                    )
                    # Map the requested inward pad displacement through the
                    # actual per-finger Jacobian.  Adding the same angle to
                    # every flexion motor is not a surface-normal motion and
                    # can pull a curved fingertip away from the object.
                    base = 4 * finger
                    finger_jacobian = jacobians_batch[env_id][finger][
                        :, base : base + 4
                    ]
                    normal_displacement = np.clip(
                        normal_displacement,
                        -self.max_release_correction * 0.10,
                        self.normal_preload_m
                        + self.nominal_tracking_radius * 0.10,
                    )
                    target_displacement = -normal_displacement * normal
                    normal_matrix = (
                        finger_jacobian @ finger_jacobian.T
                        + self.force_regularization * np.eye(3)
                    )
                    finger_correction = (
                        finger_jacobian.T
                        @ np.linalg.solve(normal_matrix, target_displacement)
                    )
                    peak = float(np.max(np.abs(finger_correction)))
                    if peak > self.nominal_tracking_radius:
                        finger_correction *= (
                            self.nominal_tracking_radius / peak
                        )
                    joint_correction[base : base + 4] = finger_correction
                joint_correction = np.clip(
                    joint_correction,
                    -self.nominal_tracking_radius,
                    self.nominal_tracking_radius,
                )
                q_ref = nominal_q + joint_correction
                force_joint_correction_batch[env_id] = (
                    joint_correction.astype(np.float32)
                )

            q_ref_batch[env_id] = q_ref.astype(np.float32)
            ik_local_batch[env_id] = self.config.data.site_xpos[
                self.tip_ids
            ].astype(np.float32)

        q_hand_t = finger_obs[:, 18:34]
        q_ref_t = torch.as_tensor(q_ref_batch, device=self.device)
        command = q_ref_t
        if not self.prev_action_initialized:
            self.prev_action = q_hand_t.detach().clone()
            self.prev_action_initialized = True
        delta = torch.clamp(
            command - self.prev_action,
            -self.action_rate_limit,
            self.action_rate_limit,
        )
        action = self.prev_action + delta
        self.prev_action = action.detach().clone()
        self.last_debug = {
            "motor_external_torque": torch.as_tensor(
                external_tau_batch, device=self.device, dtype=torch.float32
            ),
            "tip_force_from_motors": torch.as_tensor(
                forces_local_batch, device=self.device, dtype=torch.float32
            ),
            "tip_reference_palm": torch.as_tensor(
                reference_local_batch, device=self.device, dtype=torch.float32
            ),
            "tip_ik_palm": torch.as_tensor(
                ik_local_batch, device=self.device, dtype=torch.float32
            ),
            "finger_joint_reference": torch.as_tensor(
                q_ref_batch, device=self.device, dtype=torch.float32
            ),
            "finger_force_joint_correction": torch.as_tensor(
                force_joint_correction_batch,
                device=self.device,
                dtype=torch.float32,
            ),
            "tip_contact_active": torch.as_tensor(
                finger_step.contact_active[:, 1:], device=self.device
            ),
            "energy_tank": torch.as_tensor(
                finger_step.energy_tank, device=self.device, dtype=torch.float32
            ),
            "passivity_scale": torch.as_tensor(
                finger_step.passivity_scale, device=self.device, dtype=torch.float32
            ),
        }
        return action


def _align_local_z_to_normal(
    current_rotvec: np.ndarray,
    outward_normal: np.ndarray,
) -> np.ndarray:
    """Apply the minimum rotation that aligns palm local +Z to the normal."""

    rotation = R.from_rotvec(current_rotvec)
    current_z = rotation.as_matrix()[:, 2]
    target = outward_normal / max(np.linalg.norm(outward_normal), 1.0e-9)
    axis = np.cross(current_z, target)
    sine = float(np.linalg.norm(axis))
    cosine = float(np.clip(np.dot(current_z, target), -1.0, 1.0))
    if sine > 1.0e-8:
        correction = R.from_rotvec(axis / sine * np.arctan2(sine, cosine))
        rotation = correction * rotation
    elif cosine < 0.0:
        rotation = R.from_rotvec(rotation.as_matrix()[:, 0] * np.pi) * rotation
    return rotation.as_rotvec().astype(np.float32)


class FullHandMCCController:
    """Compose arm/palm MCC and motor-force fingertip MCC."""

    def __init__(self, device: str, num_envs: int, **kwargs) -> None:
        variant = str(kwargs.get("variant", "hybrid_force_position"))
        if variant not in MCC_VARIANTS:
            raise ValueError(f"Unknown variant {variant!r}; choose from {MCC_VARIANTS}")
        self.device = device
        self.num_envs = int(num_envs)
        self.variant = cast(MCCVariant, variant)
        self.arm_trust_region = float(kwargs.get("arm_trust_region", 0.08))
        self.arm_mcc_correction_limit = float(
            kwargs.get("arm_mcc_correction_limit", 0.012)
        )
        palm_kwargs = dict(kwargs)
        palm_kwargs["f_desired_normal"] = float(
            kwargs.get("palm_desired_force", 3.0)
        )
        # The legacy palm module used a workstation-specific absolute path.
        # Point its observer at the repository-local no-limit model before the
        # first controller instance is constructed.
        _palm_mcc_module._LEAPHAND_XML = _TACTILE_ROBOT_XML.with_name(
            "xarm6_leap_hand_nolimit.xml"
        )
        _palm_mcc_module._OBSERVER_CACHE.clear()
        self.palm = MCCPalmComplianceController(
            device=device, num_envs=num_envs, **palm_kwargs
        )
        self._skip_legacy_palm_preparation()
        finger_kwargs = dict(kwargs)
        finger_kwargs.pop("variant", None)
        self.fingers = MotorForceFingerMCCController(
            device=device,
            num_envs=num_envs,
            variant=self.variant,
            **finger_kwargs,
        )
        self.last_debug: dict[str, torch.Tensor] = {}
        self._previous_action: torch.Tensor | None = None
        self._arm_anchor: torch.Tensor | None = None

    def _skip_legacy_palm_preparation(self) -> None:
        """Start MCC at the validated five-contact pose, not the old retreat pose."""

        for state in self.palm.states:
            state["prep_counter"] = self.palm.prep_steps

    def reset(self) -> None:
        self.palm._init_states()
        self._skip_legacy_palm_preparation()
        self.fingers.reset()
        self._previous_action = None
        self._arm_anchor = None

    def __call__(
        self,
        obs: dict[str, torch.Tensor],
        contact_points: torch.Tensor,
        surface_normals: torch.Tensor,
        joint_reference: torch.Tensor | None = None,
        kinematic_points: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if contact_points.shape[-2:] != (5, 3):
            raise ValueError(
                f"contact_points must be (B,5,3), got {tuple(contact_points.shape)}"
            )
        if surface_normals.shape != contact_points.shape:
            raise ValueError("surface_normals must have the same shape as contact_points")
        if kinematic_points is not None and kinematic_points.shape != contact_points.shape:
            raise ValueError(
                "kinematic_points must have the same shape as contact_points"
            )
        if joint_reference is not None and joint_reference.shape != (
            contact_points.shape[0],
            22,
        ):
            raise ValueError(
                "joint_reference must be (B,22), got "
                f"{tuple(joint_reference.shape)}"
            )

        palm_obs = obs["palm"]
        tracking_points = (
            kinematic_points if kinematic_points is not None else contact_points
        )
        current_rotvec = palm_obs[:, 79:82].detach().cpu().numpy()
        normals_np = surface_normals[:, 0].detach().cpu().numpy()
        palm_target = np.zeros((contact_points.shape[0], 6), dtype=np.float32)
        for env_id in range(contact_points.shape[0]):
            # Preserve the live palm orientation.  Reorienting local +Z to an
            # arbitrary surface normal can demand a large coupled arm motion
            # even when the contact site itself is already close to target.
            palm_target[env_id, 3:] = current_rotvec[env_id]
            target_rotation = R.from_rotvec(
                palm_target[env_id, 3:]
            ).as_matrix()
            palm_target[env_id, :3] = (
                tracking_points[env_id, 0].detach().cpu().numpy()
                - target_rotation
                @ np.asarray(PALM_CONTROL_OFFSET_LOCAL, dtype=np.float32)
            )

        palm_action_all = self.palm(
            {"policy": palm_obs},
            f_cmd=(
                -float(self.fingers.core.gains.desired_palm_force)
                * surface_normals[:, 0]
            ),
            x_des=torch.as_tensor(palm_target, device=self.device),
        )
        if self._arm_anchor is None:
            self._arm_anchor = palm_obs[:, :6].detach().clone()
        current_arm = palm_obs[:, :6]
        nominal_arm = (
            joint_reference[:, :6] if joint_reference is not None else self._arm_anchor
        )
        arm_mcc_delta = torch.clamp(
            palm_action_all[:, :6] - current_arm,
            -self.arm_mcc_correction_limit,
            self.arm_mcc_correction_limit,
        )
        arm_nominal_with_mcc = nominal_arm + arm_mcc_delta
        trust_center = (
            nominal_arm if joint_reference is not None else self._arm_anchor
        )
        arm_action = torch.maximum(
            torch.minimum(
                arm_nominal_with_mcc,
                trust_center + self.arm_trust_region,
            ),
            trust_center - self.arm_trust_region,
        )
        finger_action = self.fingers(
            obs["finger"],
            tracking_points[:, 0],
            surface_normals[:, 0],
            tracking_points[:, 1:],
            surface_normals[:, 1:],
            nominal_hand_q=(
                joint_reference[:, 6:22] if joint_reference is not None else None
            ),
        )
        action = torch.cat((arm_action, finger_action), dim=-1)

        if self.variant == "passivity_tank" and self._previous_action is not None:
            # The energy tank lives in fingertip Cartesian space.  A final
            # whole-hand rate limiter also prevents a large arm command from
            # bypassing that safety layer.
            action = self._previous_action + torch.clamp(
                action - self._previous_action, -0.04, 0.04
            )
        self._previous_action = action.detach().clone()
        self.last_debug = {
            **self.fingers.last_debug,
            "five_contact_targets": contact_points.detach().clone(),
            "five_kinematic_targets": tracking_points.detach().clone(),
            "five_surface_normals": surface_normals.detach().clone(),
            "nominal_joint_reference": (
                joint_reference.detach().clone()
                if joint_reference is not None
                else torch.cat((self._arm_anchor, palm_obs[:, 6:22]), dim=-1)
            ),
        }
        return action


@dataclass
class FullHandMCCControlCfg(RslRlOnPolicyRunnerCfg):
    seed: int = 42
    device: str = "cuda:0"
    policy_class: type = FullHandMCCController
    amplitude: float = 0.5
    variant: str = "hybrid_force_position"
    control_dt: float = 0.01
    prep_duration_s: float = 1.5

    # Palm MCC.
    mass_trans: float = 1.5
    inertia_diag: tuple[float, float, float] = (0.15, 0.15, 0.15)
    K_force: float = 25.0
    K_position: float = 180.0
    K_rot: float = 25.0
    palm_desired_force: float = 3.0
    contact_threshold: float = 0.4
    alpha_normal: float = 0.2
    alpha_tau: float = 0.2
    Kf_vel: float = 0.008
    Kif_vel: float = 0.001
    force_int_max_n: float = 4.0
    kd_normal: float = 90.0
    Kp_task: float = 1.2
    dls_lambda: float = 0.12
    k_posture: float = 0.0

    # Finger motor-force MCC.
    finger_desired_force: float = 1.0
    finger_tangent_kp: float = 18.0
    finger_force_kp: float = 0.004
    finger_force_ki: float = 0.001
    motor_force_gain: float = 0.015
    force_regularization: float = 1.0e-3
    max_tip_speed: float = 0.04
    mink_damping: float = 0.1
    mink_num_iter: int = 3
    action_rate_limit: float = 0.18
    arm_trust_region: float = 0.08
    arm_mcc_correction_limit: float = 0.012
    finger_mcc_tracking_radius: float = 0.15
    finger_force_closure_gain: float = 0.02
    finger_max_release_correction: float = 0.01
    pregrasp_q: tuple[float, ...] = DEFAULT_PREGRASP_Q
