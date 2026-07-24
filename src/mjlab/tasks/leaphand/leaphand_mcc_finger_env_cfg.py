from __future__ import annotations

from dataclasses import dataclass
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
# position servo and caused repeated contact/separation switching.
# Keep margin/gap at zero: MJWarp MULTICCD does not support non-zero margins,
# and the rendered surface itself should be the contact boundary.
HARD_CONTACT_SOLREF = (-20_000.0, -400.0)  # stiffness, damping
HARD_CONTACT_SOLIMP = (0.90, 0.98, 0.001, 0.5, 2.0)
HARD_CONTACT_MARGIN = 0.0


def _apply_hard_contact(geom: mujoco.MjsGeom) -> None:
    """Apply the hard-contact approximation used for hand/object collision."""
    if geom.contype == 0 and geom.conaffinity == 0:
        return
    geom.solref[:] = HARD_CONTACT_SOLREF
    geom.solimp[:] = HARD_CONTACT_SOLIMP
    geom.margin = HARD_CONTACT_MARGIN
    geom.gap = 0.0
    # A higher priority prevents a softer material on the other geom from
    # being mixed into this contact pair.
    geom.priority = 10

# qpos/action order in xarm6_leap_hand_0.xml is four blocks of
# [flexion/abduction, side-axis, middle, distal].  This is a deliberately
# conservative pre-grasp; the second thumb axis keeps the existing 1.57 rad
# opposition used by the original finger environment.
DEFAULT_PREGRASP_Q = (
    0.85, 0.00, 0.45, 0.55,
    0.85, 0.00, 0.45, 0.55,
    0.85, 0.00, 0.45, 0.55,
    0.85, 1.57, 0.45, 0.55,
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

    existing_sites = {site.name for site in spec.sites}
    for body_name, site_name, site_pos in zip(
        MCC_TIP_BODY_NAMES, MCC_TIP_NAMES, MCC_TIP_SITE_LOCAL_POSITIONS
    ):
        if site_name in existing_sites:
            continue
        body = spec.body(body_name)
        if body is None:
            raise ValueError(f"Fingertip body {body_name!r} not found in {_LEAPHAND_XML}")
        body.add_site(
            name=site_name,
            pos=site_pos,
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=(0.004, 0.0, 0.0),
            rgba=(1.0, 0.8, 0.1, 0.8),
        )
    return spec


def _get_hard_contact_target_spec() -> mujoco.MjSpec:
    """Target whose high-priority material makes only hand/object contact hard."""
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="target_ball", mocap=True)
    body.add_geom(
        name="target_capsule_medium_geom",
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        size=(0.15, 0.08, 0.0),
        rgba=(0.2, 0.6, 1.0, 1.0),
        mass=1.0,
    )
    for geom in spec.geoms:
        _apply_hard_contact(geom)
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
    existing_sites = {site.name for site in spec.sites}
    for body_name, site_name, site_pos in zip(
        MCC_TIP_BODY_NAMES, MCC_TIP_NAMES, MCC_TIP_SITE_LOCAL_POSITIONS
    ):
        if site_name not in existing_sites:
            spec.body(body_name).add_site(
                name=site_name,
                pos=site_pos,
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=(0.004, 0.0, 0.0),
                rgba=(1.0, 0.8, 0.1, 0.8),
            )
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
        reduce="none",
        num_slots=1,
    )
    non_tip_hand_object_guard = ContactSensorCfg(
        name="non_tip_hand_object_collision",
        primary=ContactMatch(
            mode="geom",
            pattern=(
                r"^(?:palm_lower_collision|"
                r"mcp_joint(?:_[23])?_geom|"
                r"pip(?:_[234])?_geom|"
                r"dip(?:_[23])?_geom|"
                r"thumb_(?:pip|dip)_geom)$"
            ),
            entity="robot",
        ),
        secondary=ContactMatch(
            mode="body",
            pattern=r"^target_ball$",
            entity="target",
        ),
        fields=("found", "force", "dist", "pos", "normal"),
        reduce="none",
        num_slots=1,
    )
    return (*tip_sensors, arm_object_guard, non_tip_hand_object_guard)


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
) -> ManagerBasedRlEnvCfg:
    robot_cfg = EntityCfg(
        spec_fn=_load_mcc_leaphand_spec,
        articulation=EntityArticulationInfoCfg(
            actuators=(
                BuiltinPositionActuatorCfg(
                    target_names_expr=(r"^(joint[1-6])$",),
                    # Match the stable Strict palm environment.  The arm
                    # receives absolute joint targets produced by palm MCC.
                    stiffness=3000.0,
                    damping=300.0,
                    effort_limit=500.0,
                ),
                BuiltinPositionActuatorCfg(
                    target_names_expr=(r"^[0-9]+$",),
                    # Requested intermediate position-servo setting.
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
        spec_fn=_get_hard_contact_target_spec,
        init_state=EntityCfg.InitialStateCfg(pos=(0.7007, 0.0003, 0.8377)),
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
        self.pregrasp_q = np.asarray(
            kwargs.get("pregrasp_q", DEFAULT_PREGRASP_Q), dtype=np.float64
        )
        if self.pregrasp_q.shape != (16,):
            raise ValueError(f"pregrasp_q must be 16-D, got {self.pregrasp_q.shape}")

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
        self.last_debug: dict[str, torch.Tensor] = {}

    def reset(self) -> None:
        for state in self._states:
            state["initialized"] = False
            state["x_ref_local"][:] = 0.0
            state["v_ref_local"][:] = 0.0
        self.prev_action.zero_()

    def _set_qpos(self, q_hand: np.ndarray) -> None:
        self.data.qpos[:] = q_hand
        mujoco.mj_forward(self.model, self.data)

    def _tip_positions(self) -> np.ndarray:
        return np.stack([self.data.site_xpos[sid].copy() for sid in self.tip_ids])

    def _world_palm_pose(
        self, q_full: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
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
        q_actual = policy_obs[:, 12:34].detach().cpu().numpy().astype(np.float64)

        q_ref_batch = np.zeros((batch, 16), dtype=np.float32)
        x_des_batch = np.zeros((batch, 4, 3), dtype=np.float32)
        x_ref_batch = np.zeros((batch, 4, 3), dtype=np.float32)
        x_ik_batch = np.zeros((batch, 4, 3), dtype=np.float32)
        x_des_local_batch = np.zeros((batch, 4, 3), dtype=np.float32)
        x_ref_local_batch = np.zeros((batch, 4, 3), dtype=np.float32)

        for env_id in range(batch):
            q_now = q_actual[env_id].copy()
            q_hand_now = q_now[6:22]

            # In the fixed-palm model site_xpos is already palm-local.
            self._set_qpos(self.pregrasp_q)
            x_des_local = self._tip_positions()
            self.config.data.qpos[:] = self.pregrasp_q
            mujoco.mj_forward(self.config.model, self.config.data)
            self.posture_task.set_target_from_configuration(self.config)

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
        self.mink_num_iter = int(kwargs.get("mink_num_iter", 3))
        self.grav_comp_gain = float(kwargs.get("grav_comp_gain", 1.0))
        self.arm_servo_stiffness = 3000.0
        self.control_point_local = np.asarray(
            kwargs.get(
                "palm_control_offset_local",
                (-0.0559703, -0.04142053, -0.0340008),
            ),
            dtype=np.float64,
        ).reshape(3)

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
            position_cost=10.0,
            orientation_cost=np.asarray(
                kwargs.get("mink_orientation_cost", (10.0, 10.0, 10.0)),
                dtype=np.float64,
            ),
            lm_damping=1.0,
        )
        self.posture_task = mink.PostureTask(self.model, cost=0.1)
        self.limits = [mink.ConfigurationLimit(self.model)]

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

    def __call__(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        policy_obs = obs["palm"]
        q_all = policy_obs[:, :22].detach().cpu().numpy().astype(np.float32)
        output = q_all.copy()
        batch = q_all.shape[0]
        debug = {
            "palm_x_des": np.tile(self.fixed_target_np, (batch, 1)),
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
            site_pos, site_rotvec, gravity_bias = self._sync(q_now)
            debug["palm_site_pos"][env_id] = site_pos
            debug["palm_site_rotvec"][env_id] = site_rotvec
            debug["palm_tracking_error"][env_id, 0] = np.linalg.norm(
                self.fixed_target_np[:3] - site_pos
            )

            prep_counter = int(state["prep_counter"])
            if prep_counter < self.prep_steps:
                if prep_counter == 0:
                    state["q_init"] = q_now[:6].copy()
                prep_counter += 1
                state["prep_counter"] = prep_counter
                blend = prep_counter / self.prep_steps
                q_init = state["q_init"]
                if q_init is None:
                    raise RuntimeError("Palm preparation state has no initial q")
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
                state["x_ref"] = np.concatenate((site_pos, site_rotvec)).astype(
                    np.float32
                )
                state["v_ref"] = np.zeros(6, dtype=np.float32)
                state["initialized"] = True

            x_ref = np.asarray(state["x_ref"], dtype=np.float32)
            v_ref = np.asarray(state["v_ref"], dtype=np.float32)
            pos_error = self.fixed_target_np[:3] - x_ref[:3]
            lin_acc = self.kp_pos * pos_error - self.kd_pos * v_ref[:3]
            v_ref[:3] += lin_acc * self.control_dt
            x_ref[:3] += v_ref[:3] * self.control_dt

            current_ref_rot = R.from_rotvec(x_ref[3:])
            target_rot = R.from_rotvec(self.fixed_target_np[3:])
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
        self.fixed_palm_target: torch.Tensor | None = None
        self.last_debug: dict[str, torch.Tensor] = {}

    def reset(self) -> None:
        self.palm_controller.reset()
        self.finger_controller.reset()
        self.fixed_palm_target = None

    def __call__(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
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

        palm_output = self.palm_controller({"palm": obs["palm"]})
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
    pregrasp_q: tuple[float, ...] = DEFAULT_PREGRASP_Q

    # Fixed-pose palm controller (no wrench/force estimation).
    K_position: float = 100.0
    K_rot: float = 10.0
    mink_orientation_cost: tuple[float, float, float] = (10.0, 10.0, 10.0)
    grav_comp_gain: float = 1.0
    palm_control_offset_local: tuple[float, float, float] = (
        -0.0559703,
        -0.04142053,
        -0.0340008,
    )
    prep_duration_s: float = 1.5
