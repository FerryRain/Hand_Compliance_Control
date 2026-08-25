from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TypedDict

import mink
import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityCfg
from mjlab.entity.entity import EntityArticulationInfoCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg, JointRelativePositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensor, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig
from mjlab.utils.lab_api.math import quat_apply_inverse
from mcc_finger_compliance_control.scripts.object_catalog import (
    ObjectConfig,
    add_object_body,
    load_object_config,
)

_LEAPHAND_XML = (
    Path(__file__).resolve().parents[2]
    / "asset_zoo/robots/xarm6_leap_hand/xarm6_leap_hand_tactile.xml"
)
_LEAPHAND_HAND_ONLY_XML = _LEAPHAND_XML.with_name("leap_hand_tactile.xml")

MCC_TIP_NAMES = ("if_tip", "mf_tip", "rf_tip", "th_tip")
MCC_TIP_BODY_NAMES = (
    "fingertip",
    "fingertip_2",
    "fingertip_3",
    "thumb_fingertip",
)
MCC_TIP_GEOM_NAMES = (
    r"fingertip_geom",
    r"fingertip_2_geom",
    r"fingertip_3_geom",
    r"thumb_fingertip_geom",
)
MCC_NON_TIP_HAND_GEOM_PATTERN = (
    r"^(?:palm_lower_collision|"
    r"mcp_joint(?:_[23])?_geom|"
    r"pip(?:_[234])?_geom|"
    r"dip(?:_[23])?_geom|"
    r"thumb_(?:pip|dip)_geom)$"
)
MCC_TIP_SITE_LOCAL_POSITIONS = (
    (-0.0106151, -0.0326103, 0.0141088),
    (-0.0106151, -0.0326103, 0.0144487),
    (-0.0106151, -0.0326103, 0.0140386),
    (-0.0106383, -0.0453895, -0.0144321),
)


def joint_pos(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    return env.scene[asset_cfg.name].data.joint_pos

# MuJoCo has no perfectly rigid, zero-penetration contact constraint. Use the
# direct (negative) solref format with a moderately stiff response. The
# previous (-100000, -1000) / 0.1 mm transition was too abrupt for the finger
# position servo and caused repeated contact/separation switching.  A 2 mm
# transition removed enough single-frame convex-part seam flicker for the
# deterministic mustard-cap failure case to exceed 99% raw four-tip contact;
# 3 mm softened the boundary too much and reduced contact again.
# Keep margin/gap at zero: MJWarp MULTICCD does not support non-zero margins,
# and the rendered surface itself should be the contact boundary.
HARD_CONTACT_SOLREF = (-20_000.0, -400.0)  # stiffness, damping
HARD_CONTACT_SOLIMP = (0.90, 0.98, 0.002, 0.5, 2.0)
HARD_CONTACT_MARGIN = 0.0


def _apply_hard_contact(
    geom: mujoco.MjsGeom,
    *,
    solref: tuple[float, float] = HARD_CONTACT_SOLREF,
    solimp: tuple[float, float, float, float, float] = HARD_CONTACT_SOLIMP,
) -> None:
    """Apply the hard-contact approximation used for hand/object collision."""
    if geom.contype == 0 and geom.conaffinity == 0:
        return
    geom.solref[:] = solref
    geom.solimp[:] = solimp
    geom.margin = HARD_CONTACT_MARGIN
    geom.gap = 0.0
    # A higher priority prevents a softer material on the other geom from
    # being mixed into this contact pair.
    geom.priority = 10

# qpos/action order in xarm6_leap_hand_0.xml is four blocks of
# [flexion/abduction, side-axis, middle, distal].  Default pre-grasp is a
# "thin-cylinder pinch": index/ring fingers are drawn inward toward the
# middle finger with the side axis, all three fingers are curled deeper
# (middle/distal ~0.85 rad), and the thumb opposes at 1.30 rad instead of
# the fully open 1.57.  FK check: if-rf span shrinks from 90.9 mm to
# 55.3 mm and the fingertip enveloping radius from 56 to 34 mm, so small
# or flat objects stay within the grasp.  The adduction is what keeps the
# fingertip pads loaded as the palm orbits an object (verified on the
# scaled YCB mustard bottle: 3.6-10.5 N per tip, tighter than the open
# hand).  Per-object ``collection.pregrasp_q`` overrides remain available
# for objects that genuinely need a different hand shape.
DEFAULT_PREGRASP_Q = (
    1.05, 0.50, 0.85, 0.85,
    1.05, 0.00, 0.85, 0.85,
    1.05, -0.50, 0.85, 0.85,
    1.05, 1.30, 0.85, 0.85,
)

# Arm pose whose palm-control world pose remains the fixed MCC target.
MCC_TARGET_ARM_Q = np.array(
    (0.0, 1.183, -3.1416, 3.1415, 1.183, -1.569), dtype=np.float32
)

# Collision-free reset/preparation pose used by the combined environment.
COMBINE_INITIAL_ARM_Q = np.array(
    (0.0, 1.183, -1.541, 3.1415, 2.742, -1.569), dtype=np.float32
)


def _load_mcc_leaphand_spec() -> mujoco.MjSpec:
    """Load the original robot and add MCC fingertip sites.

    The source XML contains one stale contact exclude referring to ``pip4``.
    mjlab attachment tolerates it, but compiling the private observer model
    used by Mink does not, so it is removed here.
    """
    spec = mujoco.MjSpec.from_file(str(_LEAPHAND_XML))
    for exclude in list(spec.excludes):
        if exclude.bodyname1 == "thumb_pip" and exclude.bodyname2 == "pip4":
            spec.delete(exclude)

    # Match the original MCC LeapHand passive joint dynamics. The xArm joints
    # keep their source-model parameters; only numeric hand joints 0..15 are
    # changed here.
    for joint in spec.joints:
        joint_name = joint.name or ""
        if joint_name.isdigit() and 0 <= int(joint_name) < 16:
            joint.damping = (0.03, 0.0, 0.0)
            joint.frictionloss = 0.001

    _add_tip_sites(spec)
    return spec


def _add_tip_sites(spec: mujoco.MjSpec, xml_path: Path | None = None) -> None:
    """Add the four MCC fingertip reference sites to an existing spec."""
    xml_path = xml_path or _LEAPHAND_XML
    existing_sites = {site.name for site in spec.sites}
    for body_name, site_name, site_pos in zip(
        MCC_TIP_BODY_NAMES, MCC_TIP_NAMES, MCC_TIP_SITE_LOCAL_POSITIONS
    ):
        if site_name in existing_sites:
            continue
        body = spec.body(body_name)
        if body is None:
            raise ValueError(f"Fingertip body {body_name!r} not found in {xml_path}")
        body.add_site(
            name=site_name,
            pos=site_pos,
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=(0.004, 0.0, 0.0),
            rgba=(1.0, 0.8, 0.1, 0.8),
        )


def _get_hard_contact_target_spec(
    object_config: ObjectConfig | None = None,
    object_scale: float = 1.0,
    contact_solref: tuple[float, float] = HARD_CONTACT_SOLREF,
    contact_solimp: tuple[float, float, float, float, float] = HARD_CONTACT_SOLIMP,
) -> mujoco.MjSpec:
    """Build a configured target with the collection contact material."""
    object_config = object_config or load_object_config("capsule_medium")
    spec = mujoco.MjSpec()
    add_object_body(
        spec,
        object_config,
        body_name="target_ball",
        mocap=True,
        scale=object_scale,
    )
    for geom in spec.geoms:
        _apply_hard_contact(
            geom,
            solref=contact_solref,
            solimp=contact_solimp,
        )
    return spec


def _load_fixed_palm_mcc_hand_spec() -> mujoco.MjSpec:
    """Build the fixed-palm 16-DoF IK model used by MCC LeapHand.

    All fingertip references in this model are expressed directly in the palm
    frame.  Consequently the arm/root world pose cannot leak into finger IK.
    """
    spec = mujoco.MjSpec.from_file(str(_LEAPHAND_HAND_ONLY_XML))
    for exclude in list(spec.excludes):
        if exclude.bodyname1 == "thumb_pip" and exclude.bodyname2 == "pip4":
            spec.delete(exclude)
    palm_free_joint = spec.joint("palm_base")
    if palm_free_joint is not None:
        spec.delete(palm_free_joint)
    palm = spec.body("palm_lower")
    palm.pos[:] = (0.0, 0.0, 0.0)
    palm.quat[:] = (1.0, 0.0, 0.0, 0.0)
    _add_tip_sites(spec, xml_path=_LEAPHAND_HAND_ONLY_XML)
    return spec


def _load_palm_free_leaphand_spec() -> mujoco.MjSpec:
    """Hand-only XML with the ``palm_base`` free joint kept.

    The palm is a free-floating 6-DoF body driven directly by a stiff
    position servo (palm-free collection: no xArm, the controller commands
    the palm's absolute world pose).  Hand joints 0..15 keep the MCC soft
    servo dynamics; a ``palm_control_site`` is added as in the arm version.
    """
    spec = mujoco.MjSpec.from_file(str(_LEAPHAND_HAND_ONLY_XML))
    for exclude in list(spec.excludes):
        if exclude.bodyname1 == "thumb_pip" and exclude.bodyname2 == "pip4":
            spec.delete(exclude)
    for joint in spec.joints:
        joint_name = joint.name or ""
        if joint_name.isdigit() and 0 <= int(joint_name) < 16:
            joint.damping = (0.03, 0.0, 0.0)
            joint.frictionloss = 0.001
    _add_tip_sites(spec, xml_path=_LEAPHAND_HAND_ONLY_XML)
    return spec


def _tip_sensor_cfgs() -> tuple[ContactSensorCfg, ...]:
    tip_sensors = tuple(
        ContactSensorCfg(
            name=f"{site_name}_contact",
            # Read the exact 3-D contact resultant on the visible fingertip geom.
            # The tactile task XML contains no FSR bodies, geoms, or mesh assets.
            primary=ContactMatch(mode="geom", pattern=f"^{geom_name}$", entity="robot"),
            secondary=ContactMatch(mode="body", pattern="^target_ball$", entity="target"),
            fields=("found", "force", "dist", "pos", "normal", "tangent"),
            reduce="netforce",
            num_slots=1,
        )
        for site_name, geom_name in zip(MCC_TIP_NAMES, MCC_TIP_GEOM_NAMES)
    )
    arm_object_guard = ContactSensorCfg(
        name="arm_object_collision",
        primary=ContactMatch(
            mode="geom",
            pattern=(
                r"^(?:base_collision|link[1-6]_collision|"
                r"fr3v2_link[0-7]_collision)$"
            ),
            entity="robot",
        ),
        secondary=ContactMatch(
            mode="body",
            pattern=r"^target_ball$",
            entity="target",
        ),
        fields=("found", "force", "dist", "pos", "normal"),
        reduce="mindist",
        num_slots=1,
    )
    incidental_hand_primary = ContactMatch(
        mode="geom",
        pattern=MCC_NON_TIP_HAND_GEOM_PATTERN,
        entity="robot",
    )
    incidental_hand_secondary = ContactMatch(
        mode="body",
        pattern=r"^target_ball$",
        entity="target",
    )
    incidental_hand_depth = ContactSensorCfg(
        name="incidental_hand_object_contact_depth",
        primary=incidental_hand_primary,
        secondary=incidental_hand_secondary,
        fields=("found", "force", "dist", "pos", "normal"),
        reduce="mindist",
        num_slots=1,
    )
    incidental_hand_force = ContactSensorCfg(
        name="incidental_hand_object_contact_force",
        primary=ContactMatch(
            mode="geom",
            pattern=MCC_NON_TIP_HAND_GEOM_PATTERN,
            entity="robot",
        ),
        secondary=ContactMatch(
            mode="body",
            pattern=r"^target_ball$",
            entity="target",
        ),
        fields=("found", "force", "dist", "pos", "normal"),
        reduce="maxforce",
        num_slots=1,
    )
    return (
        *tip_sensors,
        arm_object_guard,
        incidental_hand_depth,
        incidental_hand_force,
    )


def fingertip_force_3d_world(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Exact 3-D resultant on each visible fingertip geom in world frame."""
    forces: list[torch.Tensor] = []
    for site_name in MCC_TIP_NAMES:
        sensor = env.scene[f"{site_name}_contact"]
        assert isinstance(sensor, ContactSensor)
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
    return torch.cat(forces, dim=-1)


def fingertip_force_3d(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Four 3-D fingertip resultants expressed in their local site frames."""
    force_world = fingertip_force_3d_world(env).reshape(env.num_envs, 4, 3)
    robot = env.scene["robot"]
    site_names = [site.name or "" for site in robot.data.indexing.sites]
    site_indices = []
    for wanted in MCC_TIP_NAMES:
        matches = [
            index
            for index, name in enumerate(site_names)
            if name == wanted or name.endswith(f"/{wanted}")
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one site {wanted!r}, got {matches}")
        site_indices.append(matches[0])
    site_quat = robot.data.site_pose_w[:, site_indices, 3:7]
    return quat_apply_inverse(
        site_quat.reshape(-1, 4), force_world.reshape(-1, 3)
    ).reshape(env.num_envs, 12)


def mcc_finger_contact_env_cfg(
    num_envs: int = 1,
    play: bool = False,
    object_id: str = "capsule_medium",
    object_config: ObjectConfig | None = None,
    object_scale: float = 1.0,
    finger_stiffness: float = 5.0,
    finger_damping: float = 0.5,
    finger_effort_limit: float = 10.0,
    contact_solref: tuple[float, float] = HARD_CONTACT_SOLREF,
    contact_solimp: tuple[float, float, float, float, float] = HARD_CONTACT_SOLIMP,
    physics_substeps: int = 10,
) -> ManagerBasedRlEnvCfg:
    if object_config is not None and object_id != "capsule_medium":
        raise ValueError("Pass either object_id or object_config, not both")
    resolved_object = object_config or load_object_config(object_id)
    if physics_substeps <= 0:
        raise ValueError("physics_substeps must be positive")
    control_dt = 0.01
    robot_cfg = EntityCfg(
        spec_fn=_load_mcc_leaphand_spec,
        articulation=EntityArticulationInfoCfg(
            actuators=(
                BuiltinPositionActuatorCfg(
                    target_names_expr=(r"^(joint[1-6])$",),
                    # Match the stable Strict palm environment.  The arm
                    # receives absolute joint targets produced by palm MCC.
                    # A very large damping term made the arm lag a moving
                    # standoff target by several centimetres.  Keep the loop
                    # stiff but let it execute previewed retreat promptly.
                    stiffness=8000.0,
                    damping=160.0,
                    effort_limit=1500.0,
                ),
                BuiltinPositionActuatorCfg(
                    target_names_expr=(r"^[0-9]+$",),
                    stiffness=float(finger_stiffness),
                    damping=float(finger_damping),
                    effort_limit=float(finger_effort_limit),
                    armature=0.0,
                    frictionloss=0.001,
                ),
            ),
        ),
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={
                "joint1": 0.0,
                "joint2": 1.183,
                "joint3": -1.541,
                "joint4": 3.1415,
                "joint5": 2.742,
                "joint6": -1.569,
                "13": 1.57,
            },
        ),
    )
    target_cfg = EntityCfg(
        spec_fn=partial(
            _get_hard_contact_target_spec,
            resolved_object,
            object_scale,
            contact_solref,
            contact_solimp,
        ),
        init_state=EntityCfg.InitialStateCfg(
            pos=resolved_object.initial_pos,
            rot=resolved_object.initial_rot,
        ),
    )

    observations = {
        # Minimal palm state: fixed-pose IK only needs q and gravity bias.
        "palm": ObservationGroupCfg(
            {
                "joint_pos": ObservationTermCfg(
                    func=joint_pos,
                    params={"asset_cfg": SceneEntityCfg("robot")},
                ),
            }
        ),
        # Existing 34-D layout consumed by MCCLeapHandPositionController.
        "finger": ObservationGroupCfg(
            {
                "fingertip_force_3d": ObservationTermCfg(func=fingertip_force_3d),
                "joint_pos": ObservationTermCfg(
                    func=joint_pos,
                    params={"asset_cfg": SceneEntityCfg("robot")},
                ),
            }
        )
    }
    actions: dict[str, ActionTermCfg] = {
        "arm_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(r"^joint[1-6]$",),
            use_default_offset=False,
        ),
        "hand_delta": JointRelativePositionActionCfg(
            entity_name="robot",
            actuator_names=(r"^[0-9]+$",),
            scale=0.08,
            use_default_offset=False,
        )
    }
    return ManagerBasedRlEnvCfg(
        decimation=int(physics_substeps),
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={"robot": robot_cfg, "target": target_cfg},
            sensors=_tip_sensor_cfgs(),
            num_envs=num_envs,
            env_spacing=2.0,
        ),
        observations=observations,
        actions=actions,
        rewards={},
        terminations={},
        sim=SimulationCfg(
            mujoco=MujocoCfg(
                timestep=control_dt / float(physics_substeps),
                gravity=(0.0, 0.0, -9.81),
                ccd_iterations=200,
                solver="newton",
            ),
            njmax=1000,
            nconmax=500,
        ),
        viewer=ViewerConfig(entity_name="robot", body_name="palm_lower", distance=1.2),
        episode_length_s=1e10 if play else 50.0,
    )


# Calibrated palm world pose when the xArm sits at MCC_TARGET_ARM_Q (palm_lower
# body, not the control site).  The palm-free environment starts the hand here
# so its absolute pose matches the legacy arm calibration; the fixed target
# used by the palm-direct controller is the same constant.
PALM_FREE_INIT_POS = (0.707417, -0.029887, 0.635323)
PALM_FREE_INIT_ROTVEC = (-np.pi, 0.0, 0.0)


def _free_joint_quat_pos(
    rotvec: tuple[float, float, float],
    pos: tuple[float, float, float],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Split a (rotvec, xyz) palm pose into free-joint qpos pieces."""
    quat = R.from_rotvec(np.asarray(rotvec)).as_quat()  # (x, y, z, w)
    return (float(quat[3]), float(quat[0]), float(quat[1]), float(quat[2])), tuple(pos)


def mcc_palm_free_contact_env_cfg(
    num_envs: int = 1,
    play: bool = False,
    object_id: str = "capsule_medium",
    object_config: ObjectConfig | None = None,
    object_scale: float = 1.0,
    palm_init_pos: tuple[float, float, float] = PALM_FREE_INIT_POS,
    palm_init_rotvec: tuple[float, float, float] = PALM_FREE_INIT_ROTVEC,
    pregrasp_q: tuple[float, ...] = DEFAULT_PREGRASP_Q,
) -> ManagerBasedRlEnvCfg:
    """Hand-only environment with the palm as a free 6-DoF body.

    No xArm: the ``palm_base`` free joint is driven by a stiff position
    servo so the collection script commands the palm's absolute world pose
    directly.  Fingers keep the same 16-DoF soft position servo as the arm
    environment.  Joint qpos order is 7 (free palm: wxyz + xyz) then the 16
    hand joints, so ``q`` observations are 23-D and the hand block starts
    at index 7 (controllers use ``hand_q_start=7``).
    """
    if object_config is not None and object_id != "capsule_medium":
        raise ValueError("Pass either object_id or object_config, not both")
    resolved_object = object_config or load_object_config(object_id)
    palm_qpos_quat, palm_qpos_pos = _free_joint_quat_pos(
        palm_init_rotvec, palm_init_pos
    )
    hand_init = dict(
        (str(index), float(value))
        for index, value in enumerate(pregrasp_q)
    )
    robot_cfg = EntityCfg(
        spec_fn=_load_palm_free_leaphand_spec,
        articulation=EntityArticulationInfoCfg(
            actuators=(
                BuiltinPositionActuatorCfg(
                    target_names_expr=(r"^palm_base$",),
                    # Stiff 6-DoF servo on the free palm: this is the direct
                    # absolute-pose control channel (no arm in the model).
                    stiffness=3000.0,
                    damping=300.0,
                    effort_limit=500.0,
                ),
                BuiltinPositionActuatorCfg(
                    target_names_expr=(r"^[0-9]+$",),
                    stiffness=5.0,
                    damping=0.5,
                    effort_limit=10.0,
                    armature=0.0,
                    frictionloss=0.001,
                ),
            ),
        ),
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={"palm_base": (*palm_qpos_quat, *palm_qpos_pos), **hand_init},
        ),
    )
    target_cfg = EntityCfg(
        spec_fn=partial(
            _get_hard_contact_target_spec, resolved_object, object_scale
        ),
        init_state=EntityCfg.InitialStateCfg(
            pos=resolved_object.initial_pos,
            rot=resolved_object.initial_rot,
        ),
    )
    observations = {
        "palm": ObservationGroupCfg(
            {
                "joint_pos": ObservationTermCfg(
                    func=joint_pos,
                    params={"asset_cfg": SceneEntityCfg("robot")},
                ),
            }
        ),
        "finger": ObservationGroupCfg(
            {
                "fingertip_force_3d": ObservationTermCfg(func=fingertip_force_3d),
                "joint_pos": ObservationTermCfg(
                    func=joint_pos,
                    params={"asset_cfg": SceneEntityCfg("robot")},
                ),
            }
        ),
    }
    actions: dict[str, ActionTermCfg] = {
        "palm_pose": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(r"^palm_base$",),
            use_default_offset=False,
        ),
        "hand_delta": JointRelativePositionActionCfg(
            entity_name="robot",
            actuator_names=(r"^[0-9]+$",),
            scale=0.08,
            use_default_offset=False,
        ),
    }
    return ManagerBasedRlEnvCfg(
        decimation=5,
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={"robot": robot_cfg, "target": target_cfg},
            sensors=_tip_sensor_cfgs(),
            num_envs=num_envs,
            env_spacing=2.0,
        ),
        observations=observations,
        actions=actions,
        rewards={},
        terminations={},
        sim=SimulationCfg(
            mujoco=MujocoCfg(
                timestep=0.002,
                gravity=(0.0, 0.0, -9.81),
                ccd_iterations=200,
                solver="newton",
            ),
            njmax=1000,
            nconmax=500,
        ),
        viewer=ViewerConfig(entity_name="robot", body_name="palm_lower", distance=1.2),
        episode_length_s=1e10 if play else 50.0,
    )


class _FingerReferenceState(TypedDict):
    initialized: bool
    x_ref_local: np.ndarray
    v_ref_local: np.ndarray


class _PalmReferenceState(TypedDict):
    prep_counter: int
    q_init: np.ndarray | None
    initialized: bool
    x_ref: np.ndarray
    v_ref: np.ndarray


class MCCLeapHandPositionController:
    """Four-site MCC position controller with force feedback disabled.

    A fixed pre-grasp is converted to four fingertip targets expressed in the
    palm frame.  The MCC spring-damper reference is integrated in that local
    frame with zero measured/commanded wrench.  Before position-only multi-site
    Mink IK, targets are transformed by the current palm pose into the world
    frame. Contact forces are observed and logged but deliberately not used by
    this first controller version.
    """

    def __init__(self, device: str, num_envs: int, **kwargs):
        self.device = device
        self.num_envs = int(num_envs)
        self.control_dt = float(kwargs.get("control_dt", 0.01))
        self.mass = float(kwargs.get("mass", 1.0))
        self.kp_pos = float(kwargs.get("kp_pos", 100.0))
        kd_default = 2.0 * np.sqrt(self.mass * self.kp_pos)
        self.kd_pos = float(kwargs.get("kd_pos", kd_default))
        self.mink_damping = float(kwargs.get("mink_damping", 0.1))
        self.mink_num_iter = int(kwargs.get("mink_num_iter", 3))
        self.position_test_mode = bool(kwargs.get("position_test_mode", False))
        self.action_rate_limit = float(kwargs.get("action_rate_limit", 0.25))
        # A fingertip-position-only IK has one redundant DoF per finger.  If
        # that null space is left unconstrained, the side-swing joints can
        # migrate to a different IK branch while an object pushes the pads.
        # Keep those joints at the nominal grasp values; flexion remains free
        # to realize the Cartesian MCC target and to deflect physically at the
        # low-gain joint servo.
        self.side_joint_anchor_strength = float(
            np.clip(kwargs.get("finger_side_joint_anchor_strength", 0.0), 0.0, 1.0)
        )
        self.side_joint_indices = np.asarray((1, 5, 9, 13), dtype=np.int32)
        self.pregrasp_q = np.asarray(
            kwargs.get("pregrasp_q", DEFAULT_PREGRASP_Q), dtype=np.float64
        )
        if self.pregrasp_q.shape != (16,):
            raise ValueError(f"pregrasp_q must be 16-D, got {self.pregrasp_q.shape}")
        # Observation layout: joint_pos block starts at ``hand_q_start`` (6 with
        # the xArm, 7 when a free palm_base joint precedes the hand).
        self.hand_q_start = int(kwargs.get("hand_q_start", 6))

        # Control/IK model: fixed palm, 16 hand joints, palm-frame targets.
        spec = _load_fixed_palm_mcc_hand_spec()
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.config = mink.Configuration(self.model)
        self.tip_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            for name in MCC_TIP_NAMES
        ]
        # Full model is observer-only and never participates in finger IK.  It
        # supplies current palm world pose for diagnostics/H5 world fields.
        self.world_model = _load_mcc_leaphand_spec().compile()
        self.world_data = mujoco.MjData(self.world_model)
        self.world_palm_body_id = mujoco.mj_name2id(
            self.world_model, mujoco.mjtObj.mjOBJ_BODY, "palm_lower"
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
        self.posture_task = mink.PostureTask(self.model, cost=0.1)
        self.limits = [mink.ConfigurationLimit(self.model)]

        self._states: list[_FingerReferenceState] = [
            {
                "initialized": False,
                "x_ref_local": np.zeros((4, 3), dtype=np.float64),
                "v_ref_local": np.zeros((4, 3), dtype=np.float64),
            }
            for _ in range(self.num_envs)
        ]
        self.prev_action = torch.zeros((self.num_envs, 16), device=device)
        self.tip_target_offset_local = np.zeros(
            (self.num_envs, 3), dtype=np.float64
        )
        self.last_debug: dict[str, torch.Tensor] = {}

    def reset(self) -> None:
        for state in self._states:
            state["initialized"] = False
            state["x_ref_local"][:] = 0.0
            state["v_ref_local"][:] = 0.0
        self.prev_action.zero_()
        self.tip_target_offset_local.fill(0.0)

    def set_tip_target_offset_local(self, offset_local: np.ndarray) -> None:
        """Apply one palm-local Cartesian extension to all four tip targets."""
        offset = np.asarray(offset_local, dtype=np.float64)
        expected = (self.num_envs, 3)
        if offset.shape != expected:
            raise ValueError(
                f"tip target offset must have shape {expected}, got {offset.shape}"
            )
        self.tip_target_offset_local[:] = offset

    def _set_qpos(self, q_hand: np.ndarray) -> None:
        self.data.qpos[:] = q_hand
        mujoco.mj_forward(self.model, self.data)

    def _tip_positions(self) -> np.ndarray:
        return np.stack([self.data.site_xpos[sid].copy() for sid in self.tip_ids])

    def _world_palm_pose(
        self, q_full: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if q_full.shape[0] == 23:
            # Palm-free env: qpos starts with the free palm (wxyz quat + xyz).
            quat = q_full[:4]
            rotation = R.from_quat(
                (quat[1], quat[2], quat[3], quat[0])
            ).as_matrix()
            return q_full[4:7].copy(), rotation
        self.world_data.qpos[:] = q_full
        mujoco.mj_forward(self.world_model, self.world_data)
        position = self.world_data.xpos[self.world_palm_body_id].copy()
        rotation = self.world_data.xmat[self.world_palm_body_id].reshape(3, 3).copy()
        return position, rotation

    @staticmethod
    def _world_to_local(
        points_world: np.ndarray,
        origin_world: np.ndarray,
        rotation_world_from_local: np.ndarray,
    ) -> np.ndarray:
        return (rotation_world_from_local.T @ (points_world - origin_world).T).T

    @staticmethod
    def _local_to_world(
        points_local: np.ndarray,
        origin_world: np.ndarray,
        rotation_world_from_local: np.ndarray,
    ) -> np.ndarray:
        return origin_world + (rotation_world_from_local @ points_local.T).T

    def __call__(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        policy_obs = obs["policy"]
        batch = int(policy_obs.shape[0])
        if batch != self.num_envs:
            raise ValueError(f"Controller configured for {self.num_envs} envs, got {batch}")
        # Finger obs layout: [fingertip force (12), full joint_pos (22 arm /
        # 23 palm-free)].  q_actual keeps the FULL qpos so the world model
        # palm pose stays available; the hand block is sliced per env below.
        q_start = 12
        q_actual = policy_obs[
            :, q_start : q_start + (23 if self.hand_q_start == 7 else 22)
        ].detach().cpu().numpy().astype(np.float64)

        q_ref_batch = np.zeros((batch, 16), dtype=np.float32)
        x_des_batch = np.zeros((batch, 4, 3), dtype=np.float32)
        x_ref_batch = np.zeros((batch, 4, 3), dtype=np.float32)
        x_ik_batch = np.zeros((batch, 4, 3), dtype=np.float32)
        x_des_local_batch = np.zeros((batch, 4, 3), dtype=np.float32)
        x_ref_local_batch = np.zeros((batch, 4, 3), dtype=np.float32)

        for env_id in range(batch):
            q_now = q_actual[env_id].copy()
            q_hand_now = q_now[self.hand_q_start : self.hand_q_start + 16]

            # In the fixed-palm model site_xpos is already palm-local.
            self._set_qpos(self.pregrasp_q)
            x_des_local = (
                self._tip_positions()
                + self.tip_target_offset_local[env_id][None, :]
            )
            self.config.data.qpos[:] = self.pregrasp_q
            mujoco.mj_forward(self.config.model, self.config.data)
            # Never turn a collision-deformed live posture into the next IK
            # nominal.  Doing so caused a self-reinforcing folded hand shape.
            self.posture_task.set_target(self.pregrasp_q)

            self._set_qpos(q_hand_now)
            x_live_local = self._tip_positions()
            palm_pos, palm_rot = self._world_palm_pose(q_now)
            x_des_world = self._local_to_world(
                x_des_local, palm_pos, palm_rot
            )

            if self.position_test_mode:
                # MCC use_compliance=False semantics: the nominal motor pose
                # is the target.  This is the correct teacher for fixed
                # pre-grasp + passive physical deflection.  It also removes
                # multi-site IK null-space variation with palm motion.
                q_ref = self.pregrasp_q.copy()
                x_ref_local = x_des_local.copy()
                x_ref_world = x_des_world.copy()
                self.config.data.qpos[:] = self.pregrasp_q
                mujoco.mj_forward(self.config.model, self.config.data)
            else:
                state = self._states[env_id]
                if not bool(state["initialized"]):
                    state["x_ref_local"][:] = x_live_local
                    state["v_ref_local"][:] = 0.0
                    state["initialized"] = True

                x_ref_local = state["x_ref_local"]
                v_ref_local = state["v_ref_local"]
                acceleration = (
                    self.kp_pos * (x_des_local - x_ref_local)
                    - self.kd_pos * v_ref_local
                ) / self.mass
                v_ref_local[:] = v_ref_local + acceleration * self.control_dt
                x_ref_local[:] = x_ref_local + v_ref_local * self.control_dt
                x_ref_world = self._local_to_world(
                    x_ref_local, palm_pos, palm_rot
                )

                self.config.data.qpos[:] = q_hand_now
                mujoco.mj_forward(self.config.model, self.config.data)
                for task, target_pos, sid in zip(self.tasks, x_ref_local, self.tip_ids):
                    rot = mink.SO3.from_matrix(
                        self.config.data.site_xmat[sid].reshape(3, 3).copy()
                    )
                    task.set_target(
                        mink.SE3.from_rotation_and_translation(rot, target_pos)
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

                mujoco.mj_forward(self.config.model, self.config.data)
                q_ref = self.config.data.qpos.copy()

                side = self.side_joint_indices
                alpha = self.side_joint_anchor_strength
                q_ref[side] = (
                    (1.0 - alpha) * q_ref[side]
                    + alpha * self.pregrasp_q[side]
                )

            q_ref_batch[env_id] = q_ref.astype(np.float32)
            x_des_batch[env_id] = x_des_world.astype(np.float32)
            x_ref_batch[env_id] = x_ref_world.astype(np.float32)
            x_des_local_batch[env_id] = x_des_local.astype(np.float32)
            x_ref_local_batch[env_id] = x_ref_local.astype(np.float32)
            x_ik_local = np.stack(
                [self.config.data.site_xpos[sid].copy() for sid in self.tip_ids]
            )
            x_ik_batch[env_id] = self._local_to_world(
                x_ik_local, palm_pos, palm_rot
            ).astype(np.float32)

        q_hand = policy_obs[:, 18:34]
        q_ref_t = torch.as_tensor(q_ref_batch, device=self.device)
        action_cmd = torch.clamp((q_ref_t - q_hand) / 0.08, -1.0, 1.0)
        delta = torch.clamp(
            action_cmd - self.prev_action,
            -self.action_rate_limit,
            self.action_rate_limit,
        )
        action = self.prev_action + delta
        self.prev_action = action.detach().clone()

        self.last_debug = {
            "q_pre": torch.as_tensor(self.pregrasp_q, device=self.device, dtype=torch.float32)
            .unsqueeze(0)
            .repeat(batch, 1),
            "q_ref": q_ref_t,
            "tip_x_des": torch.as_tensor(x_des_batch, device=self.device),
            "tip_x_ref": torch.as_tensor(x_ref_batch, device=self.device),
            "tip_x_ik": torch.as_tensor(x_ik_batch, device=self.device),
            "tip_x_des_palm": torch.as_tensor(x_des_local_batch, device=self.device),
            "tip_x_ref_palm": torch.as_tensor(x_ref_local_batch, device=self.device),
            "tip_force_used": torch.zeros((batch, 4, 3), device=self.device),
        }
        return action


class MCCFixedPalmPositionController:
    """Task-local Cartesian spring-damper controller for a fixed palm pose.

    This controller is intentionally independent of every combined/FSR task.
    It uses the FSR-free tactile XML for FK and Mink IK, performs a collision-
    free joint-space preparation, then tracks one immutable world-frame palm
    pose through a second-order Cartesian reference. External-force estimation
    is deliberately absent in this first data-collection controller.
    """

    def __init__(self, device: str, num_envs: int, **kwargs):
        self.device = device
        self.num_envs = int(num_envs)
        self.control_dt = float(kwargs.get("control_dt", 0.01))
        self.prep_steps = max(
            1,
            int(float(kwargs.get("prep_duration_s", 1.5)) / self.control_dt),
        )
        self.kp_pos = float(kwargs.get("K_position", 100.0))
        self.kd_pos = 2.0 * np.sqrt(self.kp_pos)
        self.kp_rot = float(kwargs.get("K_rot", 10.0))
        self.kd_rot = 2.0 * np.sqrt(self.kp_rot)
        self.mink_damping = float(kwargs.get("mink_damping", 0.1))
        self.mink_num_iter = int(kwargs.get("palm_mink_num_iter", 12))
        self.grav_comp_gain = float(kwargs.get("grav_comp_gain", 1.0))
        self.arm_servo_stiffness = float(
            kwargs.get("arm_servo_stiffness", 5000.0)
        )
        # palm_direct: hand-only env with a free palm_base joint; the palm 6-DoF
        # target is commanded as an absolute world pose (no arm, no Mink IK).
        # The fixed target is then the calibrated palm_lower world pose, and the
        # joint block containing the free palm starts the qpos layout.
        self.palm_direct = bool(kwargs.get("palm_direct", False))
        self.hand_q_start = int(kwargs.get("hand_q_start", 6))
        self.control_point_local = np.asarray(
            kwargs.get(
                "palm_control_offset_local",
                (-0.0559703, -0.04142053, -0.0340008),
            ),
            dtype=np.float64,
        ).reshape(3)

        if self.palm_direct:
            spec = _load_palm_free_leaphand_spec()
        else:
            spec = _load_mcc_leaphand_spec()
        palm = spec.body("palm_lower")
        if palm is None:
            raise ValueError("palm_lower body is missing from tactile robot XML")
        if spec.site("palm_control_site") is None:
            palm.add_site(
                name="palm_control_site",
                pos=tuple(float(value) for value in self.control_point_local),
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=(0.003, 0.0, 0.0),
                rgba=(0.1, 1.0, 0.1, 0.8),
            )
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.config = mink.Configuration(self.model)
        self.site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "palm_control_site"
        )
        self.arm_qpos_indices: list[int] = []
        self.arm_dof_indices: list[int] = []
        self.arm_ranges: list[np.ndarray] = []
        if not self.palm_direct:
            for name in (f"joint{i}" for i in range(1, 7)):
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if jid < 0:
                    raise ValueError(f"Arm joint {name!r} missing from tactile XML")
                self.arm_qpos_indices.append(int(self.model.jnt_qposadr[jid]))
                self.arm_dof_indices.append(int(self.model.jnt_dofadr[jid]))
                self.arm_ranges.append(self.model.jnt_range[jid].copy())
        self.arm_qpos_indices_np = np.asarray(self.arm_qpos_indices, dtype=np.int32)
        self.arm_dof_indices_np = np.asarray(self.arm_dof_indices, dtype=np.int32)
        self.arm_ranges_np = np.asarray(self.arm_ranges, dtype=np.float64)
        self.site_task = mink.FrameTask(
            frame_name="palm_control_site",
            frame_type="site",
            position_cost=float(kwargs.get("palm_mink_position_cost", 50.0)),
            orientation_cost=np.asarray(
                kwargs.get("mink_orientation_cost", (10.0, 10.0, 10.0)),
                dtype=np.float64,
            ),
            lm_damping=1.0,
        )
        self.posture_task = mink.PostureTask(
            self.model, cost=float(kwargs.get("palm_mink_posture_cost", 0.02))
        )
        self.limits = [mink.ConfigurationLimit(self.model)]

        if self.palm_direct:
            # Calibrated palm_lower world pose (xArm at MCC_TARGET_ARM_Q).
            self.fixed_target_np = np.concatenate(
                (
                    np.asarray(PALM_FREE_INIT_POS, dtype=np.float64),
                    np.asarray(PALM_FREE_INIT_ROTVEC, dtype=np.float64),
                )
            ).astype(np.float32)
        else:
            target_q = np.zeros(self.model.nq, dtype=np.float64)
            target_q[self.arm_qpos_indices_np] = MCC_TARGET_ARM_Q
            self.data.qpos[:] = target_q
            mujoco.mj_forward(self.model, self.data)
            target_pos = self.data.site_xpos[self.site_id].copy()
            target_rotvec = R.from_matrix(
                self.data.site_xmat[self.site_id].reshape(3, 3)
            ).as_rotvec()
            self.fixed_target_np = np.concatenate((target_pos, target_rotvec)).astype(
                np.float32
            )
        self.fixed_target = torch.as_tensor(
            self.fixed_target_np, device=device, dtype=torch.float32
        ).unsqueeze(0).repeat(self.num_envs, 1)
        self.states: list[_PalmReferenceState] = []
        self.last_debug: dict[str, torch.Tensor] = {}
        self.reset()
        print(
            "[MCC-Fixed-Palm] independent FSR-free controller | "
            f"target={np.round(self.fixed_target_np, 5)} prep={self.prep_steps} steps"
        )

    def reset(self) -> None:
        self.states = [
            {
                "prep_counter": 0,
                "q_init": None,
                "initialized": False,
                "x_ref": np.zeros(6, dtype=np.float32),
                "v_ref": np.zeros(6, dtype=np.float32),
            }
            for _ in range(self.num_envs)
        ]

    def _sync(
        self, qpos: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.data.qpos[:] = qpos
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        pos = self.data.site_xpos[self.site_id].copy().astype(np.float32)
        rotvec = R.from_matrix(
            self.data.site_xmat[self.site_id].reshape(3, 3)
        ).as_rotvec().astype(np.float32)
        bias = self.data.qfrc_bias[self.arm_dof_indices_np].copy().astype(np.float32)
        return pos, rotvec, bias

    def _palm_direct_pose(
        self, qpos: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Palm_lower free-joint pose plus the control-site pose, no arm."""
        q7 = np.asarray(qpos, dtype=np.float64)[:7]  # wxyz quat + xyz
        quat = q7[:4]
        pos = q7[4:7]
        rotmat = R.from_quat((quat[1], quat[2], quat[3], quat[0])).as_matrix()
        rotvec = R.from_matrix(rotmat).as_rotvec().astype(np.float32)
        site_pos = (pos + rotmat @ self.control_point_local).astype(np.float32)
        return pos.astype(np.float32), rotvec, site_pos

    def __call__(
        self,
        obs: dict[str, torch.Tensor],
        x_des: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Track ``x_des`` (B,6 world pos+rotvec) when given, else the fixed target.

        The external override lets the collection script drive the palm around
        a stationary object (palm-orbit teacher) while fingers keep their
        static pregrasp targets.
        """
        policy_obs = obs["palm"]
        q_all = (
            policy_obs[:, :23]
            if self.palm_direct
            else policy_obs[:, :22]
        ).detach().cpu().numpy().astype(np.float32)
        output = q_all.copy()
        batch = q_all.shape[0]
        if x_des is None:
            target_batch = np.tile(self.fixed_target_np, (batch, 1))
        else:
            target_batch = x_des.detach().cpu().numpy().astype(np.float32)
            if target_batch.shape != (batch, 6):
                raise ValueError(
                    f"external palm x_des must be ({batch},6), got "
                    f"{target_batch.shape}"
                )
        debug = {
            "palm_x_des": target_batch.copy(),
            "palm_x_ref": np.zeros((batch, 6), dtype=np.float32),
            "palm_site_pos": np.zeros((batch, 3), dtype=np.float32),
            "palm_site_rotvec": np.zeros((batch, 3), dtype=np.float32),
            "palm_arm_q_ref": np.zeros((batch, 6), dtype=np.float32),
            "palm_arm_q_actual": q_all[:, :6].copy(),
            "palm_fk_residual": np.zeros((batch, 1), dtype=np.float32),
            "palm_tracking_error": np.zeros((batch, 1), dtype=np.float32),
            "palm_in_prep": np.zeros((batch, 1), dtype=np.float32),
        }

        for env_id in range(batch):
            state = self.states[env_id]
            q_now = q_all[env_id]
            if self.palm_direct:
                palm_pos, palm_rotvec, site_pos = self._palm_direct_pose(q_now)
                gravity_bias = np.zeros(6, dtype=np.float32)
                site_rotvec = R.from_rotvec(palm_rotvec).as_rotvec().astype(
                    np.float32
                )
                current_pose = np.concatenate((palm_pos, palm_rotvec)).astype(
                    np.float32
                )
            else:
                site_pos, site_rotvec, gravity_bias = self._sync(q_now)
            debug["palm_site_pos"][env_id] = site_pos
            debug["palm_site_rotvec"][env_id] = site_rotvec
            debug["palm_tracking_error"][env_id, 0] = np.linalg.norm(
                target_batch[env_id, :3] - site_pos
            )

            prep_counter = int(state["prep_counter"])
            if prep_counter < self.prep_steps:
                if prep_counter == 0:
                    state["q_init"] = (
                        current_pose.copy()
                        if self.palm_direct
                        else q_now[:6].copy()
                    )
                prep_counter += 1
                state["prep_counter"] = prep_counter
                blend = prep_counter / self.prep_steps
                q_init = state["q_init"]
                if q_init is None:
                    raise RuntimeError("Palm preparation state has no initial q")
                if self.palm_direct:
                    output[env_id, :6] = (
                        (1.0 - blend) * q_init
                        + blend * self.fixed_target_np
                    )
                else:
                    output[env_id, :6] = (
                        (1.0 - blend) * q_init
                        + blend * MCC_TARGET_ARM_Q
                        + self.grav_comp_gain
                        * gravity_bias
                        / self.arm_servo_stiffness
                    )
                debug["palm_in_prep"][env_id, 0] = 1.0
                debug["palm_x_ref"][env_id, :3] = site_pos
                debug["palm_x_ref"][env_id, 3:] = site_rotvec
                debug["palm_arm_q_ref"][env_id] = output[env_id, :6]
                continue

            if not bool(state["initialized"]):
                state["x_ref"] = (
                    current_pose.copy()
                    if self.palm_direct
                    else np.concatenate((site_pos, site_rotvec)).astype(np.float32)
                )
                state["v_ref"] = np.zeros(6, dtype=np.float32)
                state["initialized"] = True

            x_ref = np.asarray(state["x_ref"], dtype=np.float32)
            v_ref = np.asarray(state["v_ref"], dtype=np.float32)
            target_np = target_batch[env_id]
            pos_error = target_np[:3] - x_ref[:3]
            lin_acc = self.kp_pos * pos_error - self.kd_pos * v_ref[:3]
            v_ref[:3] += lin_acc * self.control_dt
            x_ref[:3] += v_ref[:3] * self.control_dt

            current_ref_rot = R.from_rotvec(x_ref[3:])
            target_rot = R.from_rotvec(target_np[3:])
            ori_error = (target_rot * current_ref_rot.inv()).as_rotvec().astype(
                np.float32
            )
            ang_acc = self.kp_rot * ori_error - self.kd_rot * v_ref[3:]
            v_ref[3:] += ang_acc * self.control_dt
            x_ref[3:] = (
                R.from_rotvec(v_ref[3:] * self.control_dt) * current_ref_rot
            ).as_rotvec().astype(np.float32)
            state["x_ref"] = x_ref
            state["v_ref"] = v_ref

            if self.palm_direct:
                # Absolute palm pose: the reference is the command itself.
                output[env_id, :6] = x_ref
                debug["palm_x_ref"][env_id] = x_ref
                debug["palm_arm_q_ref"][env_id] = x_ref
                debug["palm_fk_residual"][env_id, 0] = 0.0
                continue

            self.config.data.qpos[:] = q_now
            mujoco.mj_forward(self.config.model, self.config.data)
            self.posture_task.set_target_from_configuration(self.config)
            target = mink.SE3.from_rotation_and_translation(
                mink.SO3.from_matrix(R.from_rotvec(x_ref[3:]).as_matrix()),
                x_ref[:3],
            )
            self.site_task.set_target(target)
            for _ in range(self.mink_num_iter):
                velocity = mink.solve_ik(
                    self.config,
                    [self.posture_task, self.site_task],
                    self.control_dt,
                    solver="daqp",
                    damping=self.mink_damping,
                    limits=self.limits,
                )
                self.config.integrate_inplace(velocity, self.control_dt)
            q_ref = self.config.data.qpos[self.arm_qpos_indices_np].copy()
            q_ref = np.clip(
                q_ref, self.arm_ranges_np[:, 0], self.arm_ranges_np[:, 1]
            ).astype(np.float32)
            q_ref += (
                self.grav_comp_gain
                * gravity_bias
                / self.arm_servo_stiffness
            )
            output[env_id, :6] = q_ref
            debug["palm_x_ref"][env_id] = x_ref
            debug["palm_arm_q_ref"][env_id] = q_ref
            mink_pos = self.config.data.site_xpos[self.site_id]
            debug["palm_fk_residual"][env_id, 0] = np.linalg.norm(
                mink_pos - x_ref[:3]
            )

        self.last_debug = {
            name: torch.as_tensor(value, device=self.device)
            for name, value in debug.items()
        }
        return torch.as_tensor(output, device=self.device, dtype=torch.float32)


class MCCFixedWorldPalmFingerController:
    """Strict fixed-world-pose palm hold plus fixed-palm finger MCC.

    The world pose corresponding to ``MCC_TARGET_ARM_Q`` remains the immutable
    target. The robot resets at the collision-free combined pose, interpolates
    toward the legacy target joint pose during Strict preparation, then keeps
    tracking the same Cartesian target.
    """

    def __init__(self, device: str, num_envs: int, **kwargs):
        self.device = device
        self.num_envs = int(num_envs)
        self.palm_controller = MCCFixedPalmPositionController(
            device=device, num_envs=num_envs, **kwargs
        )
        self.finger_controller = MCCLeapHandPositionController(
            device=device, num_envs=num_envs, **kwargs
        )
        if kwargs.get("palm_direct", False):
            print(
                "[MCC-Fixed-World] palm_direct: no xArm, palm 6-DoF absolute "
                f"pose target pos={self.palm_controller.fixed_target_np[:3]} "
                f"rotvec={self.palm_controller.fixed_target_np[3:6]}"
            )
        self.fixed_palm_target: torch.Tensor | None = None
        self.last_debug: dict[str, torch.Tensor] = {}

    def reset(self) -> None:
        self.palm_controller.reset()
        self.finger_controller.reset()
        self.fixed_palm_target = None

    def __call__(
        self,
        obs: dict[str, torch.Tensor],
        x_des: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward external palm ``x_des`` (palm-orbit teacher) when given."""
        if "palm" not in obs or "finger" not in obs:
            raise KeyError("Expected observation groups 'palm' and 'finger'")

        if self.fixed_palm_target is None:
            self.fixed_palm_target = self.palm_controller.fixed_target.clone()
            target = self.fixed_palm_target[0]
            print(
                "[MCC-Fixed-World] fixed target from legacy arm pose "
                f"pos={target[:3].cpu().numpy()} "
                f"rotvec={target[3:6].cpu().numpy()}"
            )

        palm_output = self.palm_controller({"palm": obs["palm"]}, x_des=x_des)
        palm_debug = self.palm_controller.last_debug
        in_prep = palm_debug["palm_in_prep"][:, 0] > 0.5

        if bool(torch.all(in_prep)):
            # Run once for diagnostics/collection fields, but discard both the
            # action and controller state so no hidden closure accumulates.
            self.finger_controller.reset()
            finger_action = self.finger_controller({"policy": obs["finger"]})
            finger_action.zero_()
            self.finger_controller.reset()
        else:
            finger_action = self.finger_controller({"policy": obs["finger"]})
            finger_action = torch.where(
                in_prep.unsqueeze(-1), torch.zeros_like(finger_action), finger_action
            )

        action = torch.cat((palm_output[:, :6], finger_action), dim=-1)
        self.last_debug = {
            **self.finger_controller.last_debug,
            **palm_debug,
            "fixed_palm_target": (
                self.fixed_palm_target
                if self.fixed_palm_target is not None
                else palm_debug["palm_x_des"]
            ),
        }
        return action


@dataclass
class MCCLeapHandPositionControlCfg(RslRlOnPolicyRunnerCfg):
    seed: int = 42
    device: str = "cuda:0"
    policy_class: type = MCCFixedWorldPalmFingerController
    amplitude: float = 0.5
    control_dt: float = 0.01
    mass: float = 1.0
    kp_pos: float = 100.0
    kd_pos: float = 20.0
    mink_damping: float = 0.1
    mink_num_iter: int = 3
    position_test_mode: bool = True
    action_rate_limit: float = 0.25
    # Optional null-space guard for the four side/opposition joints.  Keep it
    # disabled until an object's signed natural-splay posture is calibrated;
    # using an unverified sign here can move a pad away from the object.
    finger_side_joint_anchor_strength: float = 0.0
    pregrasp_q: tuple[float, ...] = DEFAULT_PREGRASP_Q
    # Palm-free collection: no xArm, palm absolute pose commanded directly.
    palm_direct: bool = False
    hand_q_start: int = 6

    # Fixed-pose palm controller (no wrench/force estimation).
    K_position: float = 100.0
    K_rot: float = 10.0
    palm_mink_num_iter: int = 12
    palm_mink_position_cost: float = 50.0
    palm_mink_posture_cost: float = 0.02
    arm_servo_stiffness: float = 8000.0
    mink_orientation_cost: tuple[float, float, float] = (10.0, 10.0, 10.0)
    grav_comp_gain: float = 1.0
    palm_control_offset_local: tuple[float, float, float] = (
        -0.0559703,
        -0.04142053,
        -0.0340008,
    )
    prep_duration_s: float = 1.5
