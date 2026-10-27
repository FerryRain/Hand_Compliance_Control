"""Full-hand MCC adapter for Franka FR3 + LEAP Hand.

Architecture:

* contact 0 (palm root): FR3 wrench estimate -> Cartesian wrist admittance;
* contacts 1..4 (fingertips): four direct fingertip force measurements ->
  independent normal Cartesian admittance references -> constrained IK;
* all five input points are checked by :class:`FivePointReachabilitySolver`
  against the actual MuJoCo/URDF kinematic chain and joint limits.

Surface-specific target generation intentionally remains outside this module.
See ``full_hand_mcc/scripts/demo_surface_slide.py``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import NamedTuple

import mink
import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityCfg
from mjlab.entity.entity import EntityArticulationInfoCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.leaphand.full_hand_mcc_core import (
    FingertipAdmittanceGains,
    FingertipNormalAdmittance,
    WristAdmittanceGains,
    WristCartesianAdmittance,
)
from mjlab.tasks.leaphand.leaphand_direct_force_env import (
    DEFAULT_PREGRASP_Q,
    LEAP_HAND_TACTILE_XML,
    MCC_NON_TIP_HAND_GEOM_PATTERN,
    MCC_TIP_BODY_NAMES,
    MCC_TIP_GEOM_NAMES,
    MCC_TIP_SITE_LOCAL_POSITIONS,
    MCC_TIP_NAMES,
    direct_force_contact_env_cfg,
    fingertip_force_3d,
    joint_pos,
    load_fixed_palm_direct_force_hand_spec,
    make_hard_contact_target_spec,
)


PALM_CONTROL_SITE = "full_hand_palm_contact"
PALM_CONTROL_OFFSET_LOCAL = (0.0, 0.0, 0.0)
FULL_HAND_CAPSULE_RADIUS = 0.02
FULL_HAND_CAPSULE_HALF_HEIGHT = 0.235
FULL_HAND_OBJECT_SHAPE = "capsule"
FULL_HAND_COLLISION_MODE = "full_robot"
FULL_HAND_HIGHLIGHT_THUMB = True
ARM_DOF = 7
HAND_DOF = 16
TOTAL_DOF = ARM_DOF + HAND_DOF
ARM_JOINT_NAMES = tuple(f"fr3v2_joint{i}" for i in range(1, ARM_DOF + 1))
ARM_COLLISION_GEOM_PATTERN = re.compile(
    r"^(?:base_collision|link[1-6]_collision|"
    r"fr3v2_link[0-7]_collision)$"
)
FR3_HOME_Q = np.asarray(
    (0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853),
    dtype=np.float32,
)
_FR3_LEAP_XML = (
    LEAP_HAND_TACTILE_XML.parents[1]
    / "fr3_leap_hand"
    / "fr3v2_collision.xml"
)
_LEAP_HAND_ONLY_XML = LEAP_HAND_TACTILE_XML
# The LEAP model places the fingertip contact reference on the local -X face.
# This is the physical finger-pad outward normal; the opposite +X face is the
# nail/back side.
MCC_TIP_PAD_NORMAL_LOCAL = np.asarray(
    ((-1.0, 0.0, 0.0),) * 4,
    dtype=np.float64,
)
# The fixed-palm IK model and FR3-attached hand share the same palm axes.  Keep
# this explicit transform so every force/target conversion has one documented
# frame boundary.
FIXED_TO_ATTACHED_PALM_ROTATION = np.eye(3, dtype=np.float64)
HAND_QPOS_NAMES = (
    "1", "0", "2", "3",
    "5", "4", "6", "7",
    "9", "8", "10", "11",
    "12", "13", "14", "15",
)


def _configure_full_hand_target_geom(geom) -> None:
    """Apply the selected axisymmetric test surface to an MjSpec geom."""

    if FULL_HAND_OBJECT_SHAPE == "capsule":
        geom.type = mujoco.mjtGeom.mjGEOM_CAPSULE
        geom.size[:] = (
            FULL_HAND_CAPSULE_RADIUS,
            FULL_HAND_CAPSULE_HALF_HEIGHT,
            0.0,
        )
    elif FULL_HAND_OBJECT_SHAPE == "ellipsoid":
        geom.type = mujoco.mjtGeom.mjGEOM_ELLIPSOID
        geom.size[:] = (
            FULL_HAND_CAPSULE_RADIUS,
            FULL_HAND_CAPSULE_RADIUS,
            FULL_HAND_CAPSULE_HALF_HEIGHT,
        )
    else:
        raise ValueError(
            f"Unsupported full-hand object shape: {FULL_HAND_OBJECT_SHAPE!r}"
        )


def _get_full_hand_contact_target_spec() -> mujoco.MjSpec:
    """Use the selected surface with a dedicated robot/object collision bit."""

    spec = make_hard_contact_target_spec()
    geom = spec.geom("target_capsule_medium_geom")
    if geom is None:
        raise ValueError("target_capsule_medium_geom is missing")
    _configure_full_hand_target_geom(geom)
    # Dedicated target collision bit.  Every physical robot geom opts into
    # this bit below.  This keeps terrain/self collision filtering unchanged
    # while making arm, palm, and non-tip finger collisions physically real.
    geom.contype = 2
    geom.conaffinity = 0
    return spec


def _load_fr3_leaphand_spec() -> mujoco.MjSpec:
    """Attach the fixed-base LEAP hand to the official FR3 flange model."""

    fr3_spec = mujoco.MjSpec.from_file(str(_FR3_LEAP_XML))
    hand_spec = mujoco.MjSpec.from_file(str(_LEAP_HAND_ONLY_XML))
    # MjSpec attachment loses the source XML directory used for resolving
    # relative mesh paths. Resolve both asset sets before the two specs merge.
    for mesh in fr3_spec.meshes:
        mesh.file = str((_FR3_LEAP_XML.parent / mesh.file).resolve())
    for mesh in hand_spec.meshes:
        mesh.file = str((_LEAP_HAND_ONLY_XML.parent / mesh.file).resolve())

    # The standalone hand has a free palm joint.  Once attached to the FR3
    # flange the palm belongs to the serial chain and must not retain it.
    palm_base = hand_spec.joint("palm_base")
    if palm_base is not None:
        hand_spec.delete(palm_base)
    for exclude in list(hand_spec.excludes):
        if exclude.bodyname1 == "thumb_pip" and exclude.bodyname2 == "pip4":
            hand_spec.delete(exclude)

    palm = hand_spec.body("palm_lower")
    if palm is None:
        raise ValueError(f"palm_lower is missing from {_LEAP_HAND_ONLY_XML}")
    palm.pos = (0.0, 0.0, 0.0)
    palm.quat = (1.0, 0.0, 0.0, 0.0)
    for joint in hand_spec.joints:
        name = joint.name or ""
        if name.isdigit() and 0 <= int(name) < HAND_DOF:
            joint.damping = (0.03, 0.0, 0.0)
            joint.frictionloss = 0.001

    flange = fr3_spec.body("fr3v2_link8")
    if flange is None:
        raise ValueError("fr3v2_link8 is missing from the FR3 model")
    # Keep the validated pad-side convention while centring the hand on the
    # FR3 flange.  The small offset represents the rigid hand adapter.
    # The extra local-X half turn sends the fingers away from the FR3 wrist.
    # Without it the pre-grasp fingers point back toward links 6/7 and overlap
    # the arm collision meshes by as much as 49 mm.
    mount_rotation = R.from_euler(
        "xyz", (0.0, -np.pi, np.pi / 2.0)
    ) * R.from_euler("x", np.pi)
    mount_xyzw = mount_rotation.as_quat()
    mount = flange.add_site(
        name="leap_hand_mount",
        pos=(0.0, 0.0, 0.035),
        quat=tuple(float(value) for value in np.roll(mount_xyzw, 1)),
        size=(0.002, 0.0, 0.0),
        rgba=(0.2, 0.8, 1.0, 0.5),
    )
    fr3_spec.attach(hand_spec, prefix="", suffix="", site=mount)

    existing_sites = {site.name for site in fr3_spec.sites}
    for body_name, site_name, site_pos in zip(
        MCC_TIP_BODY_NAMES,
        MCC_TIP_NAMES,
        MCC_TIP_SITE_LOCAL_POSITIONS,
        strict=True,
    ):
        if site_name in existing_sites:
            continue
        body = fr3_spec.body(body_name)
        if body is None:
            raise ValueError(f"Fingertip body {body_name!r} is missing after attachment")
        body.add_site(
            name=site_name,
            pos=site_pos,
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=(0.004, 0.0, 0.0),
            rgba=(1.0, 0.8, 0.1, 0.8),
        )
    return fr3_spec


def _load_full_hand_robot_spec() -> mujoco.MjSpec:
    """Configure staged fingertip-only or complete-robot target collision."""

    spec = _load_fr3_leaphand_spec()
    fingertip_geoms = set(MCC_TIP_GEOM_NAMES)
    found: set[str] = set()
    for geom in spec.geoms:
        name = geom.name or ""
        if FULL_HAND_HIGHLIGHT_THUMB and name.startswith("thumb_"):
            geom.rgba = (
                (0.95, 0.18, 1.0, 1.0)
                if name == "thumb_fingertip_geom"
                else (0.48, 0.16, 0.68, 1.0)
            )
        if (
            FULL_HAND_COLLISION_MODE == "full_robot"
            and (geom.contype != 0 or geom.conaffinity != 0)
        ):
            geom.conaffinity = int(geom.conaffinity) | 2
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
    return env.scene[asset_cfg.name].data.joint_vel[:, ARM_DOF:TOTAL_DOF]


def qfrc_actuator_hand(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    return env.scene[asset_cfg.name].data.qfrc_actuator[:, ARM_DOF:TOTAL_DOF]


def qfrc_bias_hand(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return the 16 hand bias torques with a batched MJWarp fallback."""

    asset = env.scene[asset_cfg.name]
    direct = getattr(asset.data, "qfrc_bias", None)
    if direct is not None:
        return direct[:, ARM_DOF:TOTAL_DOF]
    if hasattr(env.sim.data, "struct") and hasattr(env.sim.data.struct, "qfrc_bias"):
        bias = env.sim.data.struct.qfrc_bias
        if not torch.is_tensor(bias):
            bias = torch.as_tensor(
                bias.numpy(), device=env.device, dtype=torch.float32
            )
        if bias.ndim == 1:
            bias = bias.unsqueeze(0)
        return bias[:, ARM_DOF:TOTAL_DOF]
    return torch.zeros((env.num_envs, HAND_DOF), device=env.device)


def joint_vel_arm_fr3(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    return env.scene[asset_cfg.name].data.joint_vel[:, :ARM_DOF]


def qfrc_actuator_arm_fr3(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    return env.scene[asset_cfg.name].data.qfrc_actuator[:, :ARM_DOF]


def qfrc_bias_arm_fr3(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    direct = getattr(asset.data, "qfrc_bias", None)
    if direct is not None:
        return direct[:, :ARM_DOF]
    if hasattr(env.sim.data, "struct") and hasattr(env.sim.data.struct, "qfrc_bias"):
        bias = env.sim.data.struct.qfrc_bias
        if not torch.is_tensor(bias):
            bias = torch.as_tensor(
                bias.numpy(), device=env.device, dtype=torch.float32
            )
        if bias.ndim == 1:
            bias = bias.unsqueeze(0)
        return bias[:, :ARM_DOF]
    return torch.zeros((env.num_envs, ARM_DOF), device=env.device)


def full_hand_mcc_env_cfg(
    num_envs: int = 1,
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Build the FR3 + LEAP Baseline-2 direct-force environment."""

    robot_cfg = EntityCfg(
        spec_fn=_load_full_hand_robot_spec,
        articulation=EntityArticulationInfoCfg(
            actuators=(
                BuiltinPositionActuatorCfg(
                    target_names_expr=(r"^fr3v2_joint[1-7]$",),
                    stiffness=2200.0,
                    damping=220.0,
                    effort_limit=87.0,
                ),
                BuiltinPositionActuatorCfg(
                    target_names_expr=(r"^[0-9]+$",),
                    stiffness=35.0,
                    damping=2.5,
                    effort_limit=35.0,
                    armature=0.0,
                    frictionloss=0.001,
                ),
            ),
        ),
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            # Anchored selectors are required because a bare numeric pattern
            # such as "1" also matches joints "10" through "15".
            joint_pos={
                **{
                    f"^{joint_name}$": float(value)
                    for joint_name, value in zip(
                        ARM_JOINT_NAMES, FR3_HOME_Q, strict=True
                    )
                },
                **{
                    f"^{joint_name}$": float(value)
                    for joint_name, value in zip(
                        HAND_QPOS_NAMES, DEFAULT_PREGRASP_Q, strict=True
                    )
                },
            },
        ),
    )
    target_cfg = EntityCfg(
        spec_fn=_get_full_hand_contact_target_spec,
        # Put all four initial fingertip contacts near the lower end of the
        # straight cylinder while retaining room to slide toward the top.
        init_state=EntityCfg.InitialStateCfg(pos=(0.7007, 0.0003, 0.7077)),
    )
    cfg = direct_force_contact_env_cfg(
        robot_cfg=robot_cfg,
        target_cfg=target_cfg,
        actions={
            "arm_pos": JointPositionActionCfg(
                entity_name="robot",
                actuator_names=(r"^fr3v2_joint[1-7]$",),
                use_default_offset=False,
            ),
            "hand_delta": JointPositionActionCfg(
                entity_name="robot",
                actuator_names=(r"^[0-9]+$",),
                use_default_offset=False,
            ),
        },
        num_envs=num_envs,
        play=play,
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
                    func=joint_vel_arm_fr3, params={"asset_cfg": robot}
                ),
                "qfrc_actuator_arm": ObservationTermCfg(
                    func=qfrc_actuator_arm_fr3, params={"asset_cfg": robot}
                ),
                "qfrc_bias_arm": ObservationTermCfg(
                    func=qfrc_bias_arm_fr3, params={"asset_cfg": robot}
                ),
            }
        ),
        # Layout: force(12), q(23), qd_hand(16), tau_motor(16), bias_hand(16).
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
    spec = _load_fr3_leaphand_spec()
    for geom in spec.geoms:
        if geom.contype != 0 or geom.conaffinity != 0:
            geom.conaffinity = int(geom.conaffinity) | 2
    palm = spec.body("palm_lower")
    if palm is None:
        raise ValueError("palm_lower is missing from the FR3 + LEAP model")
    if spec.site(PALM_CONTROL_SITE) is None:
        palm.add_site(
            name=PALM_CONTROL_SITE,
            pos=PALM_CONTROL_OFFSET_LOCAL,
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=(0.004, 0.0, 0.0),
            rgba=(0.2, 1.0, 0.2, 0.8),
        )
    collision_target = spec.worldbody.add_body(
        name="planner_collision_target",
        mocap=True,
        pos=(10.0, 10.0, 10.0),
    )
    collision_geom = collision_target.add_geom(
        name="planner_collision_target_geom",
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        size=(FULL_HAND_CAPSULE_RADIUS, FULL_HAND_CAPSULE_HALF_HEIGHT, 0.0),
        contype=2,
        conaffinity=0,
        mass=1.0,
    )
    _configure_full_hand_target_geom(collision_geom)
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
        self.collision_target_geom_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "planner_collision_target_geom",
        )
        collision_target_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "planner_collision_target",
        )
        if self.collision_target_geom_id < 0 or collision_target_body_id < 0:
            raise ValueError("Planner collision target is missing")
        self.collision_target_mocap_id = int(
            self.model.body_mocapid[collision_target_body_id]
        )
        fingertip_geoms = set(MCC_TIP_GEOM_NAMES)
        self.tip_geom_ids = np.asarray(
            [
                mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    name,
                )
                for name in MCC_TIP_GEOM_NAMES
            ],
            dtype=np.int32,
        )
        if np.any(self.tip_geom_ids < 0):
            raise ValueError("One or more fingertip collision geoms are missing")
        self.tip_body_ids = np.asarray(
            [
                mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    name,
                )
                for name in MCC_TIP_BODY_NAMES
            ],
            dtype=np.int32,
        )
        if np.any(self.tip_body_ids < 0):
            raise ValueError("One or more fingertip bodies are missing")
        named_robot_geom_ids = [
            geom_id
            for geom_id in range(self.model.ngeom)
            if geom_id != self.collision_target_geom_id
            and (self.model.geom(geom_id).name or "")
        ]
        self.non_tip_geom_ids = np.asarray(
            [
                geom_id
                for geom_id in named_robot_geom_ids
                if (self.model.geom(geom_id).name or "")
                not in fingertip_geoms
            ],
            dtype=np.int32,
        )
        self.arm_geom_ids = np.asarray(
            [
                geom_id
                for geom_id in self.non_tip_geom_ids
                if ARM_COLLISION_GEOM_PATTERN.fullmatch(
                    self.model.geom(int(geom_id)).name or ""
                )
            ],
            dtype=np.int32,
        )
        arm_geom_id_set = set(int(geom_id) for geom_id in self.arm_geom_ids)
        self.hand_non_tip_geom_ids = np.asarray(
            [
                geom_id
                for geom_id in self.non_tip_geom_ids
                if int(geom_id) not in arm_geom_id_set
            ],
            dtype=np.int32,
        )
        if not len(self.arm_geom_ids):
            raise ValueError("No FR3 collision geoms were found")
        if not len(self.hand_non_tip_geom_ids):
            raise ValueError("No non-tip LEAP Hand collision geoms were found")
        uncovered_hand_geoms = [
            self.model.geom(int(geom_id)).name or ""
            for geom_id in self.hand_non_tip_geom_ids
            if re.fullmatch(
                MCC_NON_TIP_HAND_GEOM_PATTERN,
                self.model.geom(int(geom_id)).name or "",
            )
            is None
        ]
        if uncovered_hand_geoms:
            raise ValueError(
                "Runtime incidental-contact sensors do not cover LEAP Hand "
                f"geoms: {uncovered_hand_geoms}"
            )

        joint_names = ARM_JOINT_NAMES + HAND_QPOS_NAMES
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
        q = np.asarray(joint_position, dtype=np.float64).reshape(TOTAL_DOF)
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

    def fingertip_pad_normals(
        self,
        joint_position: np.ndarray,
    ) -> np.ndarray:
        """Return the four physical finger-pad outward normals in world axes."""

        self.forward_points(joint_position)
        return np.asarray(
            [
                self.data.xmat[int(body_id)].reshape(3, 3)
                @ local_normal
                for body_id, local_normal in zip(
                    self.tip_body_ids,
                    MCC_TIP_PAD_NORMAL_LOCAL,
                    strict=True,
                )
            ],
            dtype=np.float64,
        )

    def minimum_non_tip_clearance(
        self,
        joint_position: np.ndarray,
        object_center: np.ndarray,
        object_rotation: np.ndarray,
        distance_limit: float = 1.0,
    ) -> tuple[float, str]:
        """Return legacy clearance to any non-tip robot geom.

        New full-hand planning should use :meth:`minimum_arm_clearance` for
        strict FR3 avoidance and :meth:`minimum_hand_clearance` only to bound
        allowed incidental LEAP Hand penetration.
        """

        return self._minimum_clearance(
            self.non_tip_geom_ids,
            joint_position,
            object_center,
            object_rotation,
            distance_limit,
        )

    def minimum_arm_clearance(
        self,
        joint_position: np.ndarray,
        object_center: np.ndarray,
        object_rotation: np.ndarray,
        distance_limit: float = 1.0,
    ) -> tuple[float, str]:
        """Return signed FR3/object clearance; FR3 contact is forbidden."""

        return self._minimum_clearance(
            self.arm_geom_ids,
            joint_position,
            object_center,
            object_rotation,
            distance_limit,
        )

    def minimum_hand_clearance(
        self,
        joint_position: np.ndarray,
        object_center: np.ndarray,
        object_rotation: np.ndarray,
        distance_limit: float = 1.0,
    ) -> tuple[float, str]:
        """Return signed non-tip LEAP Hand/object clearance.

        A negative value is allowed up to the configured incidental-contact
        penetration limit; it is not an FR3 collision.
        """

        return self._minimum_clearance(
            self.hand_non_tip_geom_ids,
            joint_position,
            object_center,
            object_rotation,
            distance_limit,
        )

    def _minimum_clearance(
        self,
        geom_ids: np.ndarray,
        joint_position: np.ndarray,
        object_center: np.ndarray,
        object_rotation: np.ndarray,
        distance_limit: float,
    ) -> tuple[float, str]:
        """Return the closest signed distance for an explicit geometry set."""

        center = np.asarray(object_center, dtype=np.float64).reshape(3)
        rotation = np.asarray(object_rotation, dtype=np.float64).reshape(3, 3)
        quat_xyzw = R.from_matrix(rotation).as_quat()
        self.data.mocap_pos[self.collision_target_mocap_id] = center
        self.data.mocap_quat[self.collision_target_mocap_id] = np.roll(
            quat_xyzw, 1
        )
        self.forward_points(joint_position)
        best_distance = float(distance_limit)
        best_name = ""
        for geom_id in geom_ids:
            distance = float(
                mujoco.mj_geomDistance(
                    self.model,
                    self.data,
                    int(geom_id),
                    self.collision_target_geom_id,
                    distance_limit,
                    None,
                )
            )
            if distance < best_distance:
                best_distance = distance
                best_name = self.model.geom(int(geom_id)).name or ""
        return best_distance, best_name

    def geometry_group_clearances(
        self,
        joint_position: np.ndarray,
        object_center: np.ndarray,
        object_rotation: np.ndarray,
        distance_limit: float = 1.0,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        tuple[str, ...],
        np.ndarray,
        tuple[str, ...],
    ]:
        """Return signed tip, FR3, and incidental-hand object distances."""

        center = np.asarray(object_center, dtype=np.float64).reshape(3)
        rotation = np.asarray(object_rotation, dtype=np.float64).reshape(3, 3)
        quat_xyzw = R.from_matrix(rotation).as_quat()
        self.data.mocap_pos[self.collision_target_mocap_id] = center
        self.data.mocap_quat[self.collision_target_mocap_id] = np.roll(
            quat_xyzw, 1
        )
        self.forward_points(joint_position)

        def distances(geom_ids: np.ndarray) -> np.ndarray:
            return np.asarray(
                [
                    mujoco.mj_geomDistance(
                        self.model,
                        self.data,
                        int(geom_id),
                        self.collision_target_geom_id,
                        distance_limit,
                        None,
                    )
                    for geom_id in geom_ids
                ],
                dtype=np.float64,
            )

        arm_names = tuple(
            self.model.geom(int(geom_id)).name or ""
            for geom_id in self.arm_geom_ids
        )
        hand_names = tuple(
            self.model.geom(int(geom_id)).name or ""
            for geom_id in self.hand_non_tip_geom_ids
        )
        return (
            distances(self.tip_geom_ids),
            distances(self.arm_geom_ids),
            arm_names,
            distances(self.hand_non_tip_geom_ids),
            hand_names,
        )

    def geometry_clearances(
        self,
        joint_position: np.ndarray,
        object_center: np.ndarray,
        object_rotation: np.ndarray,
        distance_limit: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
        """Return signed tip and non-tip distances to the planning capsule."""

        center = np.asarray(object_center, dtype=np.float64).reshape(3)
        rotation = np.asarray(object_rotation, dtype=np.float64).reshape(3, 3)
        quat_xyzw = R.from_matrix(rotation).as_quat()
        self.data.mocap_pos[self.collision_target_mocap_id] = center
        self.data.mocap_quat[self.collision_target_mocap_id] = np.roll(
            quat_xyzw, 1
        )
        self.forward_points(joint_position)

        def distances(geom_ids: np.ndarray) -> np.ndarray:
            return np.asarray(
                [
                    mujoco.mj_geomDistance(
                        self.model,
                        self.data,
                        int(geom_id),
                        self.collision_target_geom_id,
                        distance_limit,
                        None,
                    )
                    for geom_id in geom_ids
                ],
                dtype=np.float64,
            )

        non_tip_names = tuple(
            self.model.geom(int(geom_id)).name or ""
            for geom_id in self.non_tip_geom_ids
        )
        return (
            distances(self.tip_geom_ids),
            distances(self.non_tip_geom_ids),
            non_tip_names,
        )

    def self_collision_contacts(
        self,
        joint_position: np.ndarray,
    ) -> tuple[tuple[tuple[int, int], ...], np.ndarray]:
        """Return active robot/robot contact pairs with the target moved away."""

        self.data.mocap_pos[self.collision_target_mocap_id] = (
            5.0,
            5.0,
            5.0,
        )
        self.forward_points(joint_position)
        pairs: list[tuple[int, int]] = []
        distances: list[float] = []
        for contact_id in range(self.data.ncon):
            contact = self.data.contact[contact_id]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            body1 = int(self.model.geom_bodyid[geom1])
            body2 = int(self.model.geom_bodyid[geom2])
            if body1 == 0 or body2 == 0:
                continue
            if (
                geom1 == self.collision_target_geom_id
                or geom2 == self.collision_target_geom_id
            ):
                continue
            pair = tuple(sorted((geom1, geom2)))
            if pair in pairs:
                continue
            pairs.append(pair)
            distances.append(float(contact.dist))
        return tuple(pairs), np.asarray(distances, dtype=np.float64)

    def geometry_pair_distances(
        self,
        joint_position: np.ndarray,
        geom_pairs: tuple[tuple[int, int], ...],
        distance_limit: float = 0.03,
    ) -> np.ndarray:
        """Return signed distances for a fixed set of self-collision pairs."""

        self.forward_points(joint_position)
        return np.asarray(
            [
                mujoco.mj_geomDistance(
                    self.model,
                    self.data,
                    geom1,
                    geom2,
                    distance_limit,
                    None,
                )
                for geom1, geom2 in geom_pairs
            ],
            dtype=np.float64,
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
        q = np.asarray(seed_joint_position, dtype=np.float64).reshape(TOTAL_DOF).copy()
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
            arm_dofs = self.dof_indices_np[:ARM_DOF]
            jacobian = np.vstack((jacp[:, arm_dofs], jacr[:, arm_dofs]))
            weights = np.asarray([1.0, 1.0, 1.0, 0.2, 0.2, 0.2])
            weighted_j = weights[:, None] * jacobian
            lhs = jacobian.T @ weighted_j + 1.0e-3 * np.eye(ARM_DOF)
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
                candidate[:ARM_DOF] += scale * dq_arm
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
        q = np.asarray(seed_joint_position, dtype=np.float64).reshape(TOTAL_DOF).copy()
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

    def solve_fingertips_fixed_arm(
        self,
        target_points: np.ndarray,
        seed_joint_position: np.ndarray,
        tolerance: float = 2.5e-4,
    ) -> ReachabilityResult:
        """Solve four fingertip targets using only the 16 hand joints.

        This is the appropriate warm start for surface MPC states whose palm
        travel ratio is zero.  The generic five-point solver intentionally
        stops once it reaches its runtime IK tolerance (normally several
        millimetres), which creates a no-motion dead zone for small MPC
        increments.  Here the seven arm joints are held exactly fixed and the
        tighter seed tolerance forces real fingertip articulation.
        """

        targets = np.asarray(target_points, dtype=np.float64).reshape(5, 3)
        q = np.asarray(seed_joint_position, dtype=np.float64).reshape(TOTAL_DOF).copy()
        q = np.minimum(np.maximum(q, self.lower), self.upper)
        fixed_arm = q[:ARM_DOF].copy()
        posture_anchor = q[ARM_DOF:].copy()
        best_q = q.copy()
        best_points = self.forward_points(q)
        tip_error = (targets[1:] - best_points[1:]).ravel()
        best_cost = float(tip_error @ tip_error)
        iterations = 0
        hand_dofs = self.dof_indices_np[ARM_DOF:]
        damping = max(0.05 * self.damping, 1.0e-6)
        regularization = min(self.posture_regularization, 1.0e-7)

        for iterations in range(1, self.max_iterations + 1):
            points = self.forward_points(q)
            error = (targets[1:] - points[1:]).ravel()
            tip_residual = np.linalg.norm(error.reshape(4, 3), axis=1)
            if float(tip_residual.max()) <= tolerance:
                residual = np.concatenate(
                    (
                        np.asarray(
                            [np.linalg.norm(targets[0] - points[0])],
                            dtype=np.float64,
                        ),
                        tip_residual,
                    )
                )
                return ReachabilityResult(
                    True,
                    q,
                    points,
                    residual,
                    iterations,
                )

            jacobian_blocks: list[np.ndarray] = []
            for site_id in self.site_ids[1:]:
                jacp = np.zeros((3, self.model.nv), dtype=np.float64)
                jacr = np.zeros((3, self.model.nv), dtype=np.float64)
                mujoco.mj_jacSite(
                    self.model,
                    self.data,
                    jacp,
                    jacr,
                    int(site_id),
                )
                jacobian_blocks.append(jacp[:, hand_dofs])
            jacobian = np.vstack(jacobian_blocks)
            lhs = jacobian.T @ jacobian
            lhs += (damping + regularization) * np.eye(16)
            rhs = (
                jacobian.T @ error
                - regularization * (q[ARM_DOF:] - posture_anchor)
            )
            dq_hand = np.linalg.solve(lhs, rhs)
            dq_hand = np.clip(
                dq_hand,
                -self.max_joint_step,
                self.max_joint_step,
            )

            accepted_step = False
            scale = 1.0
            while scale >= 1.0 / 64.0:
                candidate = q.copy()
                candidate[:ARM_DOF] = fixed_arm
                candidate[ARM_DOF:] += scale * dq_hand
                candidate = np.minimum(
                    np.maximum(candidate, self.lower),
                    self.upper,
                )
                candidate[:ARM_DOF] = fixed_arm
                candidate_points = self.forward_points(candidate)
                candidate_error = (
                    targets[1:] - candidate_points[1:]
                ).ravel()
                candidate_cost = float(
                    candidate_error @ candidate_error
                    + regularization
                    * np.sum((candidate[ARM_DOF:] - posture_anchor) ** 2)
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

        residual = np.concatenate(
            (
                np.asarray(
                    [np.linalg.norm(targets[0] - best_points[0])],
                    dtype=np.float64,
                ),
                np.linalg.norm(
                    targets[1:] - best_points[1:],
                    axis=1,
                ),
            )
        )
        return ReachabilityResult(
            bool(float(residual.max()) <= tolerance),
            best_q,
            best_points,
            residual,
            iterations,
        )


class FingertipForceFingerMCCController:
    """Four normal admittance loops driven by direct fingertip forces."""

    def __init__(
        self,
        device: str,
        num_envs: int,
        **kwargs,
    ) -> None:
        self.device = device
        self.num_envs = int(num_envs)
        self.control_dt = float(kwargs.get("control_dt", 0.01))
        self.mink_damping = float(kwargs.get("mink_damping", 0.1))
        self.mink_num_iter = int(kwargs.get("mink_num_iter", 3))
        self.action_rate_limit = float(kwargs.get("action_rate_limit", 0.18))
        self.force_regularization = float(kwargs.get("force_regularization", 1.0e-3))
        self.nominal_tracking_radius = float(
            kwargs.get("finger_mcc_tracking_radius", 0.15)
        )
        self.pregrasp_q = np.asarray(
            kwargs.get("pregrasp_q", DEFAULT_PREGRASP_Q), dtype=np.float64
        )
        self.model = load_fixed_palm_direct_force_hand_spec().compile()
        self.data = mujoco.MjData(self.model)
        self.config = mink.Configuration(self.model)
        # The environment/action order is HAND_QPOS_NAMES, whereas MuJoCo's
        # compiled fixed-palm model stores qpos/dofs in XML tree order.  They
        # are not interchangeable (notably the first two axes of the three
        # non-thumb fingers are swapped).  Keep every fixed-palm Jacobian and
        # configuration conversion explicitly in action order.
        self.hand_qpos_indices = np.asarray(
            [
                int(
                    self.model.jnt_qposadr[
                        mujoco.mj_name2id(
                            self.model,
                            mujoco.mjtObj.mjOBJ_JOINT,
                            name,
                        )
                    ]
                )
                for name in HAND_QPOS_NAMES
            ],
            dtype=np.int32,
        )
        self.hand_dof_indices = np.asarray(
            [
                int(
                    self.model.jnt_dofadr[
                        mujoco.mj_name2id(
                            self.model,
                            mujoco.mjtObj.mjOBJ_JOINT,
                            name,
                        )
                    ]
                )
                for name in HAND_QPOS_NAMES
            ],
            dtype=np.int32,
        )
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

        self.world_model = _load_fr3_leaphand_spec().compile()
        self.world_data = mujoco.MjData(self.world_model)
        world_joint_names = ARM_JOINT_NAMES + HAND_QPOS_NAMES
        self.world_qpos_indices = np.asarray(
            [
                int(
                    self.world_model.jnt_qposadr[
                        mujoco.mj_name2id(
                            self.world_model,
                            mujoco.mjtObj.mjOBJ_JOINT,
                            name,
                        )
                    ]
                )
                for name in world_joint_names
            ],
            dtype=np.int32,
        )
        self.world_dof_indices = np.asarray(
            [
                int(
                    self.world_model.jnt_dofadr[
                        mujoco.mj_name2id(
                            self.world_model,
                            mujoco.mjtObj.mjOBJ_JOINT,
                            name,
                        )
                    ]
                )
                for name in world_joint_names
            ],
            dtype=np.int32,
        )
        self.world_palm_id = mujoco.mj_name2id(
            self.world_model, mujoco.mjtObj.mjOBJ_BODY, "palm_lower"
        )
        self.world_tip_ids = np.asarray(
            [
                mujoco.mj_name2id(
                    self.world_model,
                    mujoco.mjtObj.mjOBJ_SITE,
                    name,
                )
                for name in MCC_TIP_NAMES
            ],
            dtype=np.int32,
        )
        if np.any(self.world_tip_ids < 0):
            raise ValueError("One or more world-model fingertip sites are missing")
        self.fixed_to_attached_palm_rotation = (
            FIXED_TO_ATTACHED_PALM_ROTATION.copy()
        )
        admittance_gains = FingertipAdmittanceGains(
            dt=self.control_dt,
            virtual_mass=float(kwargs.get("finger_virtual_mass", 0.08)),
            virtual_damping=float(kwargs.get("finger_virtual_damping", 18.0)),
            virtual_stiffness=float(
                kwargs.get("finger_virtual_stiffness", 1000.0)
            ),
            force_gain=float(kwargs.get("finger_force_gain", 1.0)),
            desired_force=float(kwargs.get("finger_desired_force", 3.0)),
            force_filter_alpha=float(
                kwargs.get("finger_force_filter_alpha", 0.25)
            ),
            contact_on_force=float(
                kwargs.get("finger_contact_on_force", 0.15)
            ),
            contact_off_force=float(
                kwargs.get("finger_contact_off_force", 0.08)
            ),
            max_normal_offset=float(
                kwargs.get("finger_max_normal_offset_m", 0.003)
            ),
            max_normal_speed=float(kwargs.get("max_tip_speed", 0.010)),
            max_normal_acceleration=float(
                kwargs.get("finger_max_normal_acceleration", 0.2)
            ),
        )
        self.admittance = FingertipNormalAdmittance(admittance_gains)
        self.fingertip_force_setpoint = np.full(
            4,
            admittance_gains.desired_force,
            dtype=np.float64,
        )
        self.minimum_fingertip_force_setpoint = float(
            admittance_gains.desired_force
        )
        self.maximum_fingertip_force_setpoint = float(
            kwargs.get("finger_max_calibrated_force", 12.0)
        )
        if (
            self.maximum_fingertip_force_setpoint
            < self.minimum_fingertip_force_setpoint
        ):
            raise ValueError(
                "finger_max_calibrated_force must be at least "
                "finger_desired_force"
            )
        self.fingertip_force_sign = np.ones(4, dtype=np.float64)
        self.prev_action = torch.zeros((num_envs, 16), device=device)
        self.prev_action_initialized = False
        self.last_debug: dict[str, torch.Tensor] = {}

    def reset(self) -> None:
        self.admittance.reset()
        self.prev_action.zero_()
        self.prev_action_initialized = False

    def calibrate_fingertip_force_sign(
        self, signed_normal_force: np.ndarray
    ) -> None:
        """Calibrate direct-sensor signs without changing the force targets."""

        baseline = np.asarray(signed_normal_force, dtype=np.float64).reshape(4)
        if not np.all(np.isfinite(baseline)):
            raise ValueError("fingertip force baseline must be finite")
        reliable = np.abs(baseline) >= 0.05
        self.fingertip_force_sign[reliable] = np.where(
            baseline[reliable] >= 0.0, 1.0, -1.0
        )

    def calibrate_fingertip_force_setpoint(
        self, signed_normal_force: np.ndarray
    ) -> None:
        """Capture a bounded loaded force target after calibrating signs."""

        self.calibrate_fingertip_force_sign(signed_normal_force)
        baseline = np.abs(
            np.asarray(signed_normal_force, dtype=np.float64).reshape(4)
        )
        self.fingertip_force_setpoint = np.clip(
            baseline,
            self.minimum_fingertip_force_setpoint,
            self.maximum_fingertip_force_setpoint,
        )

    def _set_hand_q(self, q_hand: np.ndarray) -> None:
        self.data.qpos[:] = 0.0
        self.data.qpos[self.hand_qpos_indices] = q_hand
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _world_kinematics(
        self,
        q_full: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return palm pose, tip rotations, and the world-frame arm Jacobian."""

        self.world_data.qpos[:] = 0.0
        self.world_data.qpos[self.world_qpos_indices] = q_full
        self.world_data.qvel[:] = 0.0
        mujoco.mj_forward(self.world_model, self.world_data)
        jacp = np.zeros((3, self.world_model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.world_model.nv), dtype=np.float64)
        mujoco.mj_jacBody(
            self.world_model,
            self.world_data,
            jacp,
            jacr,
            self.world_palm_id,
        )
        arm_dofs = self.world_dof_indices[:ARM_DOF]
        return (
            self.world_data.xpos[self.world_palm_id].copy(),
            self.world_data.xmat[self.world_palm_id].reshape(3, 3).copy(),
            self.world_data.site_xmat[self.world_tip_ids]
            .reshape(4, 3, 3)
            .copy(),
            np.vstack((jacp[:, arm_dofs], jacr[:, arm_dofs])),
        )

    def _world_palm_pose(self, q_full: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        position, rotation, _, _ = self._world_kinematics(q_full)
        return position, rotation

    def _tip_positions_and_jacobians(self) -> tuple[np.ndarray, list[np.ndarray]]:
        positions = self.data.site_xpos[self.tip_ids].copy()
        jacobians: list[np.ndarray] = []
        for site_id in self.tip_ids:
            jacp = np.zeros((3, self.model.nv), dtype=np.float64)
            jacr = np.zeros((3, self.model.nv), dtype=np.float64)
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, int(site_id))
            jacobians.append(jacp[:, self.hand_dof_indices].copy())
        return positions, jacobians

    def _motor_torque_to_tip_forces_diagnostic(
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
        # Baseline 2 assigns palm/wrist motion to the wrist loop.  These
        # arguments stay in the signature for the five-point controller API.
        del palm_target_world, palm_normal_world
        direct_force_tip_local_batch = (
            finger_obs[:, :12]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
            .reshape(-1, 4, 3)
        )
        q_full_batch = (
            finger_obs[:, 12 : 12 + TOTAL_DOF]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        motor_offset = 12 + TOTAL_DOF + HAND_DOF
        tau_motor = (
            finger_obs[:, motor_offset : motor_offset + HAND_DOF]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        bias_motor = (
            finger_obs[
                :, motor_offset + HAND_DOF : motor_offset + 2 * HAND_DOF
            ]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        targets_world = tip_targets_world.detach().cpu().numpy().astype(np.float64)
        normals_world = tip_normals_world.detach().cpu().numpy().astype(np.float64)
        nominal_hand_np = (
            nominal_hand_q.detach().cpu().numpy().astype(np.float64)
            if nominal_hand_q is not None
            else None
        )
        batch = q_full_batch.shape[0]

        desired_local_batch = np.zeros((batch, 4, 3))
        normals_local_batch = np.zeros((batch, 4, 3))
        direct_forces_local_raw_batch = np.zeros((batch, 4, 3))
        motor_forces_diagnostic_batch = np.zeros((batch, 4, 3))
        external_tau_batch = -(tau_motor - bias_motor)

        for env_id in range(batch):
            q_full = q_full_batch[env_id]
            self._set_hand_q(q_full[ARM_DOF:TOTAL_DOF])
            _, jacobians = self._tip_positions_and_jacobians()
            (
                palm_origin,
                palm_rotation,
                tip_rotations_world,
                _,
            ) = self._world_kinematics(q_full)
            world_to_fixed = (
                self.fixed_to_attached_palm_rotation.T
                @ palm_rotation.T
            )
            desired_local_batch[env_id] = (
                world_to_fixed
                @ (targets_world[env_id] - palm_origin).T
            ).T
            normals_local_batch[env_id] = (
                world_to_fixed @ normals_world[env_id].T
            ).T
            direct_force_world = np.einsum(
                "fij,fj->fi",
                tip_rotations_world,
                direct_force_tip_local_batch[env_id],
            )
            direct_forces_local_raw_batch[env_id] = (
                world_to_fixed @ direct_force_world.T
            ).T
            motor_forces_diagnostic_batch[env_id] = (
                self._motor_torque_to_tip_forces_diagnostic(
                    jacobians,
                    external_tau_batch[env_id],
                )
            )

        signed_normal_force_raw = np.einsum(
            "bfi,bfi->bf",
            direct_forces_local_raw_batch,
            normals_local_batch,
        )
        direct_forces_local_batch = (
            direct_forces_local_raw_batch
            * self.fingertip_force_sign[None, :, None]
        )
        finger_step = self.admittance.step(
            desired_local_batch,
            normals_local_batch,
            direct_forces_local_batch,
            desired_force=self.fingertip_force_setpoint[None, :],
        )
        reference_local_batch = finger_step.command_points

        q_ref_batch = np.zeros((batch, 16), dtype=np.float32)
        force_joint_correction_batch = np.zeros((batch, 16), dtype=np.float32)
        ik_local_batch = np.zeros((batch, 4, 3), dtype=np.float32)
        for env_id in range(batch):
            q_hand = q_full_batch[env_id, ARM_DOF:TOTAL_DOF]
            nominal_q = (
                nominal_hand_np[env_id]
                if nominal_hand_np is not None
                else q_hand
            )
            self.config.data.qpos[:] = 0.0
            self.config.data.qpos[self.hand_qpos_indices] = nominal_q
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
            q_ref = self.config.data.qpos[
                self.hand_qpos_indices
            ].copy()
            if nominal_hand_np is not None:
                joint_correction = q_ref - nominal_q
                peak = float(np.max(np.abs(joint_correction)))
                if peak > self.nominal_tracking_radius:
                    joint_correction *= self.nominal_tracking_radius / peak
                q_ref = nominal_q + joint_correction
                force_joint_correction_batch[env_id] = (
                    joint_correction.astype(np.float32)
                )

            q_ref_batch[env_id] = q_ref.astype(np.float32)
            self._set_hand_q(q_ref)
            ik_local_batch[env_id] = self.data.site_xpos[self.tip_ids].astype(
                np.float32
            )

        q_hand_t = finger_obs[
            :, 12 + ARM_DOF : 12 + TOTAL_DOF
        ]
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
            "tip_force_from_motors_diagnostic": torch.as_tensor(
                motor_forces_diagnostic_batch,
                device=self.device,
                dtype=torch.float32,
            ),
            "tip_force_direct_raw_palm": torch.as_tensor(
                direct_forces_local_raw_batch,
                device=self.device,
                dtype=torch.float32,
            ),
            "tip_force_direct_palm": torch.as_tensor(
                direct_forces_local_batch,
                device=self.device,
                dtype=torch.float32,
            ),
            "tip_normal_force_signed_raw": torch.as_tensor(
                signed_normal_force_raw,
                device=self.device,
                dtype=torch.float32,
            ),
            "tip_normal_force": torch.as_tensor(
                finger_step.measured_normal_force,
                device=self.device,
                dtype=torch.float32,
            ),
            "tip_normal_force_filtered": torch.as_tensor(
                finger_step.filtered_normal_force,
                device=self.device,
                dtype=torch.float32,
            ),
            "tip_force_error": torch.as_tensor(
                finger_step.force_error,
                device=self.device,
                dtype=torch.float32,
            ),
            "finger_normal_admittance_offset_m": torch.as_tensor(
                finger_step.normal_offset,
                device=self.device,
                dtype=torch.float32,
            ),
            "finger_normal_admittance_velocity_m_s": torch.as_tensor(
                finger_step.normal_velocity,
                device=self.device,
                dtype=torch.float32,
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
                finger_step.contact_active, device=self.device
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
    """Baseline 2: wrist admittance plus four fingertip-force admittances."""

    def __init__(self, device: str, num_envs: int, **kwargs) -> None:
        self.device = device
        self.num_envs = int(num_envs)
        self.action_rate_limit = float(kwargs.get("action_rate_limit", 0.18))
        if self.action_rate_limit <= 0.0:
            raise ValueError("action_rate_limit must be positive")
        self.arm_trust_region = float(kwargs.get("arm_trust_region", 0.08))
        self.arm_mcc_correction_limit = float(
            kwargs.get("arm_mcc_correction_limit", 0.012)
        )
        self.wrist_update_decimation = max(
            1, int(kwargs.get("wrist_update_decimation", 4))
        )
        self.wrist_wrench_force_regularization = float(
            kwargs.get("wrist_wrench_force_regularization", 1.0e-3)
        )
        self.wrist_wrench_torque_regularization = float(
            kwargs.get("wrist_wrench_torque_regularization", 1.0e-2)
        )
        self.wrist_ik_damping = float(
            kwargs.get("wrist_ik_damping", 2.0e-3)
        )
        control_dt = float(kwargs.get("control_dt", 0.01))
        wrist_gains = WristAdmittanceGains(
            dt=control_dt * self.wrist_update_decimation,
            translation_mass=float(kwargs.get("mass_trans", 3.0)),
            rotation_inertia=tuple(
                float(value)
                for value in kwargs.get(
                    "inertia_diag", (0.30, 0.30, 0.30)
                )
            ),
            normal_stiffness=float(kwargs.get("K_force", 400.0)),
            tangent_stiffness=float(kwargs.get("K_position", 800.0)),
            rotation_stiffness=float(kwargs.get("K_rot", 80.0)),
            damping_ratio=float(kwargs.get("wrist_damping_ratio", 1.0)),
            wrench_filter_alpha=float(kwargs.get("alpha_tau", 0.10)),
            max_force_error=float(
                kwargs.get("wrist_max_force_error_n", 5.0)
            ),
            max_torque_error=float(
                kwargs.get("wrist_max_torque_error_nm", 0.8)
            ),
            max_translation_offset=float(
                kwargs.get("wrist_max_translation_offset_m", 0.003)
            ),
            max_rotation_offset=float(
                kwargs.get("wrist_max_rotation_offset_rad", 0.03)
            ),
            max_translation_speed=float(
                kwargs.get("wrist_max_translation_speed_m_s", 0.010)
            ),
            max_rotation_speed=float(
                kwargs.get("wrist_max_rotation_speed_rad_s", 0.10)
            ),
            max_translation_acceleration=float(
                kwargs.get("wrist_max_translation_acceleration", 0.10)
            ),
            max_rotation_acceleration=float(
                kwargs.get("wrist_max_rotation_acceleration", 0.5)
            ),
        )
        self.wrist_admittance = WristCartesianAdmittance(wrist_gains)
        self.fingers = FingertipForceFingerMCCController(
            device=device,
            num_envs=num_envs,
            **kwargs,
        )
        self.last_debug: dict[str, torch.Tensor] = {}
        self._previous_action: torch.Tensor | None = None
        self._arm_anchor: torch.Tensor | None = None
        self._arm_external_wrench_setpoint: np.ndarray | None = None
        self._wrist_last_step = None
        self._wrist_step_count = 0

    def reset(self) -> None:
        self.fingers.reset()
        self.wrist_admittance.reset()
        self._previous_action = None
        self._arm_anchor = None
        self._arm_external_wrench_setpoint = None
        self._wrist_last_step = None
        self._wrist_step_count = 0

    @staticmethod
    def _arm_external_torque(palm_obs: torch.Tensor) -> torch.Tensor:
        """Estimate seven FR3 external joint torques from actuator/bias residuals."""

        actuator_start = TOTAL_DOF + ARM_DOF
        bias_start = actuator_start + ARM_DOF
        actuator_torque = palm_obs[:, actuator_start:bias_start]
        bias_torque = palm_obs[:, bias_start : bias_start + ARM_DOF]
        return -(actuator_torque - bias_torque)

    def _arm_external_wrench(
        self,
        q_full_batch: np.ndarray,
        external_torque_batch: np.ndarray,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """Estimate world-frame palm wrench from FR3 external joint torque."""

        wrench_batch = np.zeros(
            (q_full_batch.shape[0], 6), dtype=np.float64
        )
        jacobians: list[np.ndarray] = []
        regularization = np.diag(
            [
                self.wrist_wrench_force_regularization,
                self.wrist_wrench_force_regularization,
                self.wrist_wrench_force_regularization,
                self.wrist_wrench_torque_regularization,
                self.wrist_wrench_torque_regularization,
                self.wrist_wrench_torque_regularization,
            ]
        )
        for env_id, (q_full, external_torque) in enumerate(
            zip(q_full_batch, external_torque_batch, strict=True)
        ):
            _, _, _, jacobian = self.fingers._world_kinematics(q_full)
            lhs = jacobian @ jacobian.T + regularization
            wrench_batch[env_id] = np.linalg.solve(
                lhs,
                jacobian @ external_torque,
            )
            jacobians.append(jacobian)
        return wrench_batch, jacobians

    def calibrate_arm_force_setpoint(
        self,
        palm_obs: torch.Tensor,
    ) -> None:
        """Capture the loaded four-contact world-frame wrist wrench setpoint."""

        q_full = (
            palm_obs[:, :TOTAL_DOF]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        external_torque = (
            self._arm_external_torque(palm_obs)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        self._arm_external_wrench_setpoint = self._arm_external_wrench(
            q_full,
            external_torque,
        )[0]
        self.wrist_admittance.reset()
        self._wrist_last_step = None
        self._wrist_step_count = 0

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
            TOTAL_DOF,
        ):
            raise ValueError(
                f"joint_reference must be (B,{TOTAL_DOF}), got "
                f"{tuple(joint_reference.shape)}"
            )

        palm_obs = obs["palm"]
        tracking_points = (
            kinematic_points if kinematic_points is not None else contact_points
        )
        if self._arm_anchor is None:
            self._arm_anchor = palm_obs[:, :ARM_DOF].detach().clone()
        current_arm = palm_obs[:, :ARM_DOF]
        nominal_arm = (
            joint_reference[:, :ARM_DOF]
            if joint_reference is not None
            else self._arm_anchor
        )
        arm_external_torque = self._arm_external_torque(palm_obs)
        q_full_np = (
            palm_obs[:, :TOTAL_DOF]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        arm_external_torque_np = (
            arm_external_torque.detach().cpu().numpy().astype(np.float64)
        )
        arm_external_wrench_np, _ = self._arm_external_wrench(
            q_full_np,
            arm_external_torque_np,
        )
        if (
            self.arm_mcc_correction_limit > 0.0
            and self._arm_external_wrench_setpoint is not None
        ):
            arm_wrench_error_np = (
                arm_external_wrench_np
                - self._arm_external_wrench_setpoint
            )
            if (
                self._wrist_last_step is None
                or self._wrist_step_count % self.wrist_update_decimation == 0
            ):
                self._wrist_last_step = self.wrist_admittance.step(
                    arm_wrench_error_np,
                    surface_normals[:, 0]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64),
                )
            assert self._wrist_last_step is not None
            wrist_reference_offset_np = (
                self._wrist_last_step.reference_offset
            )
            nominal_full_np = (
                joint_reference.detach().cpu().numpy().astype(np.float64)
                if joint_reference is not None
                else q_full_np.copy()
            )
            nominal_full_np[:, :ARM_DOF] = (
                nominal_arm.detach().cpu().numpy().astype(np.float64)
            )
            arm_mcc_delta_np = np.zeros(
                (q_full_np.shape[0], ARM_DOF),
                dtype=np.float64,
            )
            for env_id, nominal_full in enumerate(nominal_full_np):
                _, _, _, nominal_jacobian = self.fingers._world_kinematics(
                    nominal_full
                )
                lhs = (
                    nominal_jacobian @ nominal_jacobian.T
                    + self.wrist_ik_damping * np.eye(6)
                )
                joint_delta = nominal_jacobian.T @ np.linalg.solve(
                    lhs,
                    wrist_reference_offset_np[env_id],
                )
                peak = float(np.max(np.abs(joint_delta)))
                if peak > self.arm_mcc_correction_limit:
                    joint_delta *= self.arm_mcc_correction_limit / peak
                arm_mcc_delta_np[env_id] = joint_delta
            arm_mcc_delta = torch.as_tensor(
                arm_mcc_delta_np,
                device=self.device,
                dtype=torch.float32,
            )
        else:
            arm_wrench_error_np = np.zeros_like(arm_external_wrench_np)
            wrist_reference_offset_np = np.zeros_like(arm_external_wrench_np)
            arm_mcc_delta = torch.zeros_like(current_arm)
        self._wrist_step_count += 1
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
                joint_reference[:, ARM_DOF:TOTAL_DOF]
                if joint_reference is not None
                else None
            ),
        )
        action = torch.cat((arm_action, finger_action), dim=-1)

        if self._previous_action is not None:
            # Apply one controller-independent joint-command rate limit to the
            # arm and fingers after both admittance corrections are composed.
            action = self._previous_action + torch.clamp(
                action - self._previous_action,
                -self.action_rate_limit,
                self.action_rate_limit,
            )
        self._previous_action = action.detach().clone()
        self.last_debug = {
            **self.fingers.last_debug,
            "five_contact_targets": contact_points.detach().clone(),
            "five_kinematic_targets": tracking_points.detach().clone(),
            "five_surface_normals": surface_normals.detach().clone(),
            "arm_external_torque": arm_external_torque.detach().clone(),
            "arm_external_wrench": torch.as_tensor(
                arm_external_wrench_np,
                device=self.device,
                dtype=torch.float32,
            ),
            "arm_wrench_error": torch.as_tensor(
                arm_wrench_error_np,
                device=self.device,
                dtype=torch.float32,
            ),
            "wrist_admittance_reference_offset": torch.as_tensor(
                wrist_reference_offset_np,
                device=self.device,
                dtype=torch.float32,
            ),
            "arm_force_joint_correction": arm_mcc_delta.detach().clone(),
            "nominal_joint_reference": (
                joint_reference.detach().clone()
                if joint_reference is not None
                else torch.cat(
                    (self._arm_anchor, palm_obs[:, ARM_DOF:TOTAL_DOF]),
                    dim=-1,
                )
            ),
        }
        return action


@dataclass
class FullHandMCCControlCfg(RslRlOnPolicyRunnerCfg):
    seed: int = 42
    device: str = "cuda:0"
    policy_class: type = FullHandMCCController
    amplitude: float = 0.5
    control_dt: float = 0.01
    prep_duration_s: float = 1.5

    # Palm MCC.
    mass_trans: float = 3.0
    inertia_diag: tuple[float, float, float] = (0.30, 0.30, 0.30)
    K_force: float = 400.0
    K_position: float = 800.0
    K_rot: float = 80.0
    palm_desired_force: float = 3.0
    contact_threshold: float = 0.4
    alpha_normal: float = 0.2
    alpha_tau: float = 0.10
    Kf_vel: float = 0.008
    Kif_vel: float = 0.001
    force_int_max_n: float = 4.0
    kd_normal: float = 90.0
    Kp_task: float = 1.2
    dls_lambda: float = 0.12
    k_posture: float = 0.0

    # Baseline-2 fingertip admittance. Direct fingertip forces are primary;
    # motor loads are retained only as diagnostics.
    finger_desired_force: float = 3.0
    finger_virtual_mass: float = 0.08
    finger_virtual_damping: float = 18.0
    finger_virtual_stiffness: float = 1000.0
    finger_force_gain: float = 1.0
    finger_max_calibrated_force: float = 12.0
    finger_force_filter_alpha: float = 0.25
    finger_contact_on_force: float = 0.15
    finger_contact_off_force: float = 0.08
    finger_max_normal_offset_m: float = 0.003
    finger_max_normal_acceleration: float = 0.2
    force_regularization: float = 1.0e-3
    max_tip_speed: float = 0.010
    mink_damping: float = 0.1
    mink_num_iter: int = 3
    action_rate_limit: float = 0.18

    # Baseline-2 wrist admittance. The default 4x decimation gives 25 Hz at
    # the 100 Hz controller rate, slower than the fingertip loops.
    arm_trust_region: float = 0.08
    arm_mcc_correction_limit: float = 0.003
    wrist_update_decimation: int = 4
    wrist_wrench_force_regularization: float = 1.0e-3
    wrist_wrench_torque_regularization: float = 1.0e-2
    wrist_ik_damping: float = 2.0e-3
    wrist_damping_ratio: float = 1.0
    wrist_max_force_error_n: float = 5.0
    wrist_max_torque_error_nm: float = 0.8
    wrist_max_translation_offset_m: float = 0.003
    wrist_max_rotation_offset_rad: float = 0.03
    wrist_max_translation_speed_m_s: float = 0.010
    wrist_max_rotation_speed_rad_s: float = 0.10
    wrist_max_translation_acceleration: float = 0.10
    wrist_max_rotation_acceleration: float = 0.5

    finger_mcc_tracking_radius: float = 0.15
    pregrasp_q: tuple[float, ...] = DEFAULT_PREGRASP_Q
