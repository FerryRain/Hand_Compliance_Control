from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityCfg
from mjlab.entity.entity import EntityArticulationInfoCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointRelativePositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensor, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

_LEAPHAND_XML = Path("/home/rimlab/Code/Hand_Compliance_Control/src/mjlab/asset_zoo/robots/xarm6_leap_hand/xarm6_leap_hand_nolimit.xml")
_ENABLE_HAND_OBJECT_ONLY_COLLISION = False
_HAND_CONTYPE = 2
_HAND_CONAFFINITY = 4
_OBJECT_CONTYPE = 4
_OBJECT_CONAFFINITY = 2

_FSR_CACHE = {}
_FSR_COLOR_FIELDS_READY = set()


def _load_leaphand_spec(enable_hand_object_only_collision: bool = False) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(_LEAPHAND_XML))
    if not enable_hand_object_only_collision:
        return spec

    hand_geom_regex = re.compile(
        r"^(?:"
        r"palm_.*|mcp_.*|pip(?:_\d+)?_geom|dip(?:_\d+)?_geom|"
        r"fingertip(?:_\d+)?_geom|thumb_.*|tip_\d+_fsr_geom|"
        r"pip_\d+_fsr_geom|dip_\d+_fsr_geom"
        r")$"
    )


    for geom in spec.geoms:
        geom_name = geom.name or ""
        if hand_geom_regex.fullmatch(geom_name):
            geom.contype = _HAND_CONTYPE
            geom.conaffinity = _HAND_CONAFFINITY

    return spec


def fsr_force_and_visual_logic(
    env: ManagerBasedRlEnv,
    sensor_name: str = "fsr_contact",
    fsr_regex: str = r".*_fsr_geom$",
    contact_rgba: tuple[float, float, float, float] = (0.2, 1.0, 0.2, 0.9),
    default_rgba: tuple[float, float, float, float] = (1.0, 0.2, 0.2, 0.9),
    display_forces: bool = True,
    display_every: int = 5,
    display_top_k: int = 8,
    expected_num_fsrs: int = 16,
) -> torch.Tensor:
    """Read FSR forces from ContactSensor and keep optional FSR geom coloring."""
    _ = (display_forces, display_every, display_top_k)

    m = env.sim.mj_model

    sensor = env.scene[sensor_name]
    assert isinstance(sensor, ContactSensor), (
        f"{sensor_name} must be ContactSensor, got {type(sensor).__name__}"
    )
    sensor_data = sensor.data
    assert sensor_data.force is not None, "ContactSensor must expose 'force' field"

    # force: [B, N, 3], convert to per-FSR magnitude [B, N].
    forces = torch.linalg.vector_norm(sensor_data.force, dim=-1)

    num_envs = forces.shape[0]
    num_fsrs = forces.shape[1]
    forces_tensor = torch.zeros(
        (num_envs, expected_num_fsrs),
        device=forces.device,
        dtype=forces.dtype,
    )
    copy_count = min(expected_num_fsrs, num_fsrs)
    forces_tensor[:, :copy_count] = forces[:, :copy_count]

    env_ptr = id(env)

    if env_ptr not in _FSR_COLOR_FIELDS_READY:
        env.sim.expand_model_fields(("geom_rgba",))
        _FSR_COLOR_FIELDS_READY.add(env_ptr)

    if env_ptr not in _FSR_CACHE:
        pattern = re.compile(fsr_regex)
        _FSR_CACHE[env_ptr] = [
            gid for gid in range(m.ngeom)
            if (name := mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid)) and pattern.match(name)
        ]
    fsr_ids = _FSR_CACHE[env_ptr]

    if num_envs > 0 and len(fsr_ids) > 0:
        sim_geom_rgba = env.sim.model.geom_rgba
        c_rgba_t = torch.as_tensor(contact_rgba, device=forces.device, dtype=sim_geom_rgba.dtype)
        d_rgba_t = torch.as_tensor(default_rgba, device=forces.device, dtype=sim_geom_rgba.dtype)

        for fsr_idx, gid in enumerate(fsr_ids):
            if fsr_idx >= expected_num_fsrs:
                break

            if sim_geom_rgba.ndim == 2:
                # Shared visual buffer: use env-0 state as fallback.
                color = c_rgba_t if forces_tensor[0, fsr_idx] > 0.0 else d_rgba_t
                sim_geom_rgba[gid] = color
            else:
                # Per-env visual buffer: color each environment independently.
                active_env_mask = forces_tensor[:, fsr_idx] > 0.0
                sim_geom_rgba[active_env_mask, gid] = c_rgba_t
                sim_geom_rgba[~active_env_mask, gid] = d_rgba_t

    return forces_tensor

def joint_pos(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Return absolute robot joint positions with shape [num_envs, num_joints]."""
    asset = env.scene[asset_cfg.name]

    return asset.data.joint_pos


def qfrc_actuator_arm(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """返回机械臂前 6 个关节的执行器输出力矩 [num_envs, 6]"""
    asset = env.scene[asset_cfg.name]
    return asset.data.qfrc_actuator[:, 0:6]

def qfrc_bias_arm(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """返回机械臂前 6 个关节的重力与科氏力偏置力矩 [num_envs, 6]"""
    # 提取底层 MuJoCo 仿真中的偏置力矩
    if hasattr(env.sim, "mj_data"):
        bias = env.sim.mj_data.qfrc_bias
        if isinstance(bias, np.ndarray):
            bias_t = torch.from_numpy(bias).to(device=env.device, dtype=torch.float32)
        else:
            bias_t = torch.as_tensor(bias, device=env.device, dtype=torch.float32)
        if bias_t.ndim == 1:
            bias_t = bias_t.unsqueeze(0)
        return bias_t[:, 0:6]
    return torch.zeros((env.num_envs, 6), device=env.device)

def _find_body_id(m, body_name: str, env_idx: int, num_envs: int) -> int:
    """模糊匹配 Body，兼容 'robot/palm_lower' 等任意前缀"""
    names_to_try = [
        f"env_{env_idx}/{body_name}",
        f"robot_{env_idx}/{body_name}",
        f"env_{env_idx}/robot/{body_name}",
        f"robot/{body_name}",
        body_name
    ]
    for name in names_to_try:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid != -1:
            return bid
            
    # 后缀模糊匹配保底
    suffix = f"/{body_name}"
    for b_idx in range(m.nbody):
        b_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b_idx)
        if b_name and (b_name == body_name or b_name.endswith(suffix)):
            return b_idx
    return -1


def _find_joint_id(m, joint_name: str, env_idx: int, num_envs: int) -> int:
    """模糊匹配关节，兼容 'robot/joint1' 等任意前缀"""
    names_to_try = [
        f"env_{env_idx}/{joint_name}",
        f"robot_{env_idx}/{joint_name}",
        f"env_{env_idx}/robot/{joint_name}",
        f"robot/{joint_name}",
        joint_name
    ]
    for name in names_to_try:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid != -1:
            return jid
            
    # 后缀模糊匹配保底
    suffix = f"/{joint_name}"
    for j_idx in range(m.njnt):
        j_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j_idx)
        if j_name and (j_name == joint_name or j_name.endswith(suffix)):
            return j_idx
    return -1

def _compute_palm_jacobians(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    body_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute both Jacobians and world-frame position for the palm body.

    Returns:
        jac_p:    Translational Jacobian [num_envs, 18] (flattened 3x6)
        jac_r:    Rotational Jacobian     [num_envs, 18] (flattened 3x6)
        palm_pos: Palm body position       [num_envs, 3]  (world frame)
    """
    m = env.sim.mj_model
    d = env.sim.mj_data
    num_envs = env.num_envs
    asset = env.scene[asset_cfg.name]
    joint_pos_gpu = asset.data.joint_pos
    joint_vel_gpu = asset.data.joint_vel
    jacs_p = []
    jacs_r = []
    positions = []

    for i in range(num_envs):
        bid = _find_body_id(m, body_name, i, num_envs)
        if bid == -1:
            all_bodies = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, j) for j in range(m.nbody)]
            raise ValueError(f"cannot find body '{body_name}' for env index {i}. "
                             f"Tried multiple naming patterns. Available bodies: {all_bodies}")

        d.qpos[:] = joint_pos_gpu[i].cpu().numpy()
        d.qvel[:] = joint_vel_gpu[i].cpu().numpy()
        mujoco.mj_forward(m, d)

        jac_p = np.zeros((3, m.nv))
        jac_r = np.zeros((3, m.nv))
        palm_pos = d.xpos[bid]
        mujoco.mj_jac(m, d, jac_p, jac_r, palm_pos, bid)

        positions.append(palm_pos.copy())  # store world-frame position

        jid = _find_joint_id(m, "joint1", i, num_envs)
        if jid == -1:
            all_joints = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)]
            raise ValueError(f"cannot find joint 'joint1' for env index {i}. "
                             f"Tried multiple naming patterns. Available joints: {all_joints}")

        dof_adr = m.jnt_dofadr[jid]
        jacs_p.append(jac_p[:, dof_adr : dof_adr + 6])
        jacs_r.append(jac_r[:, dof_adr : dof_adr + 6])

    jacs_p_t = torch.tensor(np.array(jacs_p), device=env.device, dtype=torch.float32)
    jacs_r_t = torch.tensor(np.array(jacs_r), device=env.device, dtype=torch.float32)
    pos_t = torch.tensor(np.array(positions), device=env.device, dtype=torch.float32)
    return jacs_p_t.view(num_envs, -1), jacs_r_t.view(num_envs, -1), pos_t


# def palm_jacobian(
#     env: ManagerBasedRlEnv,
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     body_name: str = "palm_lower",
# ) -> torch.Tensor:
#     """Return palm translational Jacobian w.r.t. arm joints, flattened to [num_envs, 18]."""
#     jac_p, _, _ = _compute_palm_jacobians(env, asset_cfg, body_name)
#     return jac_p


# def palm_jacobian_rot(
#     env: ManagerBasedRlEnv,
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     body_name: str = "palm_lower",
# ) -> torch.Tensor:
#     """Return palm rotational Jacobian w.r.t. arm joints, flattened to [num_envs, 18]."""
#     _, jac_r, _ = _compute_palm_jacobians(env, asset_cfg, body_name)
#     return jac_r


def palm_pos(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_name: str = "palm_lower",
) -> torch.Tensor:
    """Return palm body world-frame position [num_envs, 3]."""
    _, _, pos = _compute_palm_jacobians(env, asset_cfg, body_name)
    return pos


def palm_jacobian(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_name: str = "palm_lower",
) -> torch.Tensor:
    """Return palm translational Jacobian w.r.t. arm joints, flattened to [num_envs, 18]."""
    jac_p, _ ,_ = _compute_palm_jacobians(env, asset_cfg, body_name)
    return jac_p


def palm_jacobian_rot(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_name: str = "palm_lower",
) -> torch.Tensor:
    """Return palm rotational Jacobian w.r.t. arm joints, flattened to [num_envs, 18]."""
    _, jac_r, _ = _compute_palm_jacobians(env, asset_cfg, body_name)
    return jac_r



def joint_vel_arm(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """返回机械臂前 6 个关节的角速度 [num_envs, 6]"""
    asset = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, 0:6]


# --- Entity 配置 ---

def _get_target_box_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    # Mocap body is kinematic: mouse perturbation can place it directly.
    body = spec.worldbody.add_body(name="target_ball", mocap=True)
    target_geoms = (
        body.add_geom(
            name="target_capsule_medium_geom",
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            size=[0.15, 0.08],
            rgba=[0.2, 0.6, 1.0, 1.0],
            mass=1,
        ),
        # body.add_geom(
        #     name="target_box_medium_geom",
        #     type=mujoco.mjtGeom.mjGEOM_BOX,
        #     size=[0.15, 0.12, 0.12],
        #     rgba=[0.2, 1.0, 0.5, 1.0],
        #     mass=1,
        # ),
        # body.add_geom(
        #     name="target_cylinder_medium_geom",
        #     type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        #     size=[0.12, 0.15],
        #     rgba=[0.3, 0.7, 1.0, 1.0],
        #     mass=1,
        # ),
        # body.add_geom(
        #     name="target_ellipsoid_medium_geom",
        #     type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
        #     size=[0.18, 0.15, 0.12],
        #     rgba=[0.25, 0.8, 0.9, 1.0],
        #     mass=1,
        # ),
    )
    for geom in target_geoms:
        if _ENABLE_HAND_OBJECT_ONLY_COLLISION:
            geom.contype = _OBJECT_CONTYPE
            geom.conaffinity = _OBJECT_CONAFFINITY
    return spec

# --- 环境配置构建 ---

def _make_env_cfg(num_envs: int = 1, play: bool = False) -> ManagerBasedRlEnvCfg:
    robot_cfg = EntityCfg(
        spec_fn=lambda: _load_leaphand_spec(
            enable_hand_object_only_collision=_ENABLE_HAND_OBJECT_ONLY_COLLISION
        ),
        articulation=EntityArticulationInfoCfg(
            actuators=(
                BuiltinPositionActuatorCfg(
                    # LeapHand joints are named "0".."15" in this XML.
                    target_names_expr=(r"^(joint[1-6]|[0-9]+)$",), 
                    stiffness=1000.0,
                    damping=100.0,
                    effort_limit=500.0,
                ),
            ),
        ),
        init_state=EntityCfg.InitialStateCfg(
            pos=(0, 0, 0),
            joint_pos={
                "joint1": 0.0,      # 机械臂 joint1
                "joint2": 1.183,    # 机械臂 joint2
                "joint3": -3.1416,  # 机械臂 joint3
                "joint4": 3.1416,   # 机械臂 joint4
                "joint5": 1.183,    # 机械臂 joint5
                "joint6": -1.569,   # 机械臂 joint6
                "13": 1.57,         # 手指关节 (保持您原本的设置)
            },
        ),
    )
    
    target_cfg = EntityCfg(
        spec_fn=_get_target_box_spec,
        init_state=EntityCfg.InitialStateCfg(pos=(0.7007, 0.5003, 0.8377)),
    )

    fsr_contact_cfg = ContactSensorCfg(
        name="fsr_contact",
        primary=ContactMatch(mode="geom", pattern=r".*_fsr_geom$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="target_ball", entity="target"),
        fields=("force",),
        reduce="netforce",
        num_slots=1,
    )

    observations = {
        "policy": ObservationGroupCfg({
            # "fsr_forces": ObservationTermCfg(
            #     func=fsr_force_and_visual_logic,
            #     params={
            #         "sensor_name": "fsr_contact",
            #         "fsr_regex": r".*_fsr_geom$",
            #         "display_forces": True,
            #         "display_every": 5,
            #         "display_top_k": 8,
            #     },
            # ),
            "joint_pos": ObservationTermCfg(
                func=joint_pos,
                params={"asset_cfg": SceneEntityCfg("robot")},
            ),
            "joint_vel_arm": ObservationTermCfg(
                func=joint_vel_arm,
                params={"asset_cfg": SceneEntityCfg("robot")},
            ),
            "qfrc_actuator_arm": ObservationTermCfg(
                func=qfrc_actuator_arm,
                params={"asset_cfg": SceneEntityCfg("robot")},
            ),
            "qfrc_bias_arm": ObservationTermCfg(
                func=qfrc_bias_arm,
                params={"asset_cfg": SceneEntityCfg("robot")},
            ),
            "palm_jacobian": ObservationTermCfg(
                func=palm_jacobian,
                params={
                    "asset_cfg": SceneEntityCfg("robot"),
                    "body_name": "palm_lower",
                },
            ),
            "palm_jacobian_rot": ObservationTermCfg(
                func=palm_jacobian_rot,
                params={
                    "asset_cfg": SceneEntityCfg("robot"),
                    "body_name": "palm_lower",
                },
            ),
            "palm_pos": ObservationTermCfg(
                func=palm_pos,
                params={
                    "asset_cfg": SceneEntityCfg("robot"),
                    "body_name": "palm_lower",
                },
            ),
        })
    }

    actions: dict[str, ActionTermCfg] = {
        "arm_delta": JointRelativePositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=0.08,
            use_default_offset=False
        )
    }

    return ManagerBasedRlEnvCfg(
        decimation=5, # type: ignore
        scene=SceneCfg(
            # terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={"robot": robot_cfg, "target": target_cfg},
            sensors=(),
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
                gravity=(0.0, 0.0, 0.0),
                ccd_iterations=200,
                solver="newton",
            ),
            njmax=1000,
            nconmax=500,
        ),
        viewer=ViewerConfig(
            entity_name="robot",
            body_name="base_link",
            distance=2.0,
        ),
        episode_length_s=1e10 if play else 50.0,
    )

def leaphand_palm_contact_env_cfg(num_envs: int = 1, play: bool = False) -> ManagerBasedRlEnvCfg:
    return _make_env_cfg(num_envs=num_envs, play=play)


class LeapHandPalmComplianceController:
    """Palm compliance controller with full task-space impedance dynamics.

    Inspired by "Minimalist Compliance Control", this controller implements:

    1. Full 6-DOF wrench estimation (force + torque) from joint torques
       via regularized least squares on both translational and rotational Jacobians.

    2. Virtual mass-spring-damper impedance dynamics in task space:
         M * a + D * v + K * (x - x_des) = f_ext + f_cmd
       This is a 2nd-order system — it has inertia, restoring force, and damping,
       unlike the simpler 1st-order admittance (dx = gain * f) which drifts freely.

    3. Direction-dependent compliance — high stiffness on the normal axis
       (to maintain contact / regulate force) and low stiffness on tangent axes
       (to allow sliding along the surface).

    Observation layout (76 dims):
      [0:22]   joint_pos          (22)  — full robot joint positions
      [22:28]  joint_vel_arm      (6)   — arm joint velocities
      [28:34]  qfrc_actuator_arm  (6)   — arm actuator torques
      [34:40]  qfrc_bias_arm      (6)   — arm bias (gravity+Coriolis) torques
      [40:58]  palm_jacobian      (18)  — translational Jacobian (3x6 flattened)
      [58:76]  palm_jacobian_rot  (18)  — rotational Jacobian (3x6 flattened)
    """

    def __init__(self, device: str, num_envs: int, **kwargs):
        self.device = device
        self.num_envs = num_envs
        B = num_envs

        # ── Impedance parameters (task-space mass-spring-damper) ──
        # Translation
        self.mass = float(kwargs.get("mass_trans", 1.0))            # kg
        self.K_normal = float(kwargs.get("K_normal", 200.0))        # N/m  (normal dir)
        self.K_tangent = float(kwargs.get("K_tangent", 20.0))       # N/m  (tangent dirs)
        self.D_normal = float(kwargs.get("D_normal", 28.0))         # Ns/m (≈ 2*sqrt(m*K))
        self.D_tangent = float(kwargs.get("D_tangent", 9.0))        # Ns/m

        # Rotation (optional — currently damped-only)
        self.inertia_rot = float(kwargs.get("inertia_rot", 0.1))    # kg·m²
        self.K_rot = float(kwargs.get("K_rot", 30.0))               # Nm/rad
        self.D_rot = float(kwargs.get("D_rot", 5.0))                # Nms/rad

        # ── Task-space tracking gain (makes robot follow virtual reference) ──
        self.Kp_task = float(kwargs.get("Kp_task", 20.0))           # 1/s (15-30 Hz typical)

        # ── Filtering ──
        self.alpha_tau = float(kwargs.get("alpha_tau", 0.1))        # motor torque EMA
        self.alpha_wrench = float(kwargs.get("alpha_wrench", 0.15)) # wrench estimate EMA

        # ── Estimation regularization ──
        self.lambda_force = float(kwargs.get("lambda_force", 1e-3))
        self.lambda_torque = float(kwargs.get("lambda_torque", 1e-2))

        # ── DLS projection damping ──
        self.dls_lambda = float(kwargs.get("dls_lambda", 0.05))

        # ── Action interface ──
        self.action_scale_arm = float(kwargs.get("action_scale_arm", 0.08))
        self.action_rate_limit = float(kwargs.get("action_rate_limit", 0.15))

        # ── Control period (s) — must match decimation * sim_timestep ──
        self.control_dt = float(kwargs.get("control_dt", 0.01))

        # ── Normal axis (world frame) ──
        normal_axis = str(kwargs.get("normal_axis", "z")).lower()
        axis_map = {"x": 0, "y": 1, "z": 2}
        if normal_axis not in axis_map:
            raise ValueError(f"normal_axis must be 'x', 'y', or 'z', got {normal_axis!r}")
        self.normal_idx = axis_map[normal_axis]
        self.tangent_indices = [i for i in range(3) if i != self.normal_idx]

        # ── Default command force ──
        default_force = kwargs.get("f_cmd_default", [0.0, 0.0, 0.0])
        self.f_cmd_default = torch.tensor(default_force, device=device, dtype=torch.float32).unsqueeze(0)

        # ── Internal state ──
        self.tau_smoothed = torch.zeros(B, 6, device=device)
        self.f_ext_filtered = torch.zeros(B, 3, device=device)
        self.tau_ext_filtered = torch.zeros(B, 3, device=device)
        self.x_ref = torch.zeros(B, 3, device=device)        # virtual mass position
        self.v_ref = torch.zeros(B, 3, device=device)        # virtual mass velocity
        self.x_des = torch.zeros(B, 3, device=device)        # equilibrium (spring rest) position
        self.prev_action_arm = torch.zeros(B, 6, device=device)
        self._initialized = False

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    def _resize_state(self, B: int) -> None:
        """Re-allocate state tensors when the batch size changes."""
        self.num_envs = B
        self.tau_smoothed = torch.zeros(B, 6, device=self.device)
        self.f_ext_filtered = torch.zeros(B, 3, device=self.device)
        self.tau_ext_filtered = torch.zeros(B, 3, device=self.device)
        self.x_ref = torch.zeros(B, 3, device=self.device)
        self.v_ref = torch.zeros(B, 3, device=self.device)
        self.x_des = torch.zeros(B, 3, device=self.device)
        self.prev_action_arm = torch.zeros(B, 6, device=self.device)

    def _init_state(
        self, palm_pos: torch.Tensor, qfrc_actuator_arm: torch.Tensor
    ) -> None:
        """Initialise impedance state at the actual palm position.

        CRITICAL: x_des MUST be set to the actual palm position, NOT zero.
        Setting x_des=0 anchors the spring at the world origin, generating
        huge spurious forces (~140 N) that fight gravity and dominate control.

        Also seeds the torque EMA with the first motor torque observation
        to prevent a large spurious transient while the filter converges.
        """
        self.x_ref = palm_pos.clone()
        self.v_ref = torch.zeros_like(palm_pos)
        self.x_des = palm_pos.clone()
        self.tau_smoothed = qfrc_actuator_arm.clone()
        self._initialized = True

    def _build_impedance_matrices(self, B: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Build direction-dependent diagonal stiffness and damping matrices.

        Normal axis gets high stiffness/damping (maintain contact);
        tangent axes get low stiffness/damping (allow sliding).
        """
        K = torch.zeros(B, 3, 3, device=self.device)
        D = torch.zeros(B, 3, 3, device=self.device)

        K[:, self.normal_idx, self.normal_idx] = self.K_normal
        D[:, self.normal_idx, self.normal_idx] = self.D_normal

        for t in self.tangent_indices:
            K[:, t, t] = self.K_tangent
            D[:, t, t] = self.D_tangent

        return K, D

    @staticmethod
    def _solve_wrench_component(
        J: torch.Tensor, tau_ext: torch.Tensor, lam: float
    ) -> torch.Tensor:
        """Regularized least-squares:  f = (J J^T + λI)^{-1} J τ_ext

        Args:
            J:      Jacobian slice [B, 3, 6]
            tau_ext: external joint torque [B, 6]
            lam:    Tikhonov regularisation

        Returns:
            wrench component [B, 3]
        """
        B, device = J.shape[0], J.device
        J_T = J.transpose(1, 2)                                  # [B, 6, 3]
        A = torch.bmm(J, J_T)                                    # [B, 3, 3]
        A = A + lam * torch.eye(3, device=device).unsqueeze(0)   # regularise
        b = torch.bmm(J, tau_ext.unsqueeze(-1))                  # [B, 3, 1]
        return torch.linalg.solve(A, b).squeeze(-1)              # [B, 3]

    # ------------------------------------------------------------------
    #  Main control call
    # ------------------------------------------------------------------

    def __call__(
        self, obs: dict[str, torch.Tensor], f_cmd: torch.Tensor | None = None
    ) -> torch.Tensor:
        policy_obs = obs["policy"]
        B = policy_obs.shape[0]
        if B != self.num_envs:
            self._resize_state(B)

        # ---- parse observations (see class docstring for layout) ----
        # joint_pos      = policy_obs[:,  0:22]   # not directly used
        # joint_vel_arm  = policy_obs[:, 22:28]
        qfrc_actuator_arm = policy_obs[:, 28:34]
        qfrc_bias_arm     = policy_obs[:, 34:40]
        J_p = policy_obs[:, 40:58].view(B, 3, 6)   # translational Jacobian
        J_r = policy_obs[:, 58:76].view(B, 3, 6)   # rotational Jacobian
        palm_pos = policy_obs[:, 76:79]             # actual palm position (world)

        # ---- initialise virtual state at current palm position ----
        if not self._initialized:
            self._init_state(palm_pos, qfrc_actuator_arm)

        # ---- 1. EMA-smooth motor torques (suppress high-freq noise) ----
        tau_raw = (
            self.alpha_tau * qfrc_actuator_arm
            + (1.0 - self.alpha_tau) * self.tau_smoothed
        )
        self.tau_smoothed = tau_raw

        # ---- 2. External joint torque (bias - actuator) ----
        # tau_ext represents the torque that must originate from external
        # contacts (i.e. what the surface pushes back with).
        tau_ext = qfrc_bias_arm - tau_raw                             # [B, 6]

        # ---- 3. Full 6-DOF wrench estimation ----
        f_ext = self._solve_wrench_component(J_p, tau_ext, self.lambda_force)
        m_ext = self._solve_wrench_component(J_r, tau_ext, self.lambda_torque)

        # ---- 4. Low-pass filter the estimated wrench ----
        self.f_ext_filtered = (
            self.alpha_wrench * self.f_ext_filtered
            + (1.0 - self.alpha_wrench) * f_ext
        )
        self.tau_ext_filtered = (
            self.alpha_wrench * self.tau_ext_filtered
            + (1.0 - self.alpha_wrench) * m_ext
        )

        # ---- 5. Net force (external + commanded) ----
        if f_cmd is None:
            f_cmd_active = self.f_cmd_default.expand(B, -1)
        elif f_cmd.ndim == 1:
            f_cmd_active = f_cmd.unsqueeze(0).expand(B, -1)
        else:
            f_cmd_active = f_cmd.to(device=self.device)

        f_net = self.f_ext_filtered + f_cmd_active                   # [B, 3]

        # ---- 6. Direction-dependent impedance matrices ----
        Kp, Kd = self._build_impedance_matrices(B)

        # ---- 7. Task-space impedance dynamics (virtual mass-spring-damper) ----
        #  M * a  +  D * v  +  K * (x_ref - x_des)  =  f_net
        #  →  a = (K*(x_des - x_ref)  -  D*v  +  f_net) / M
        pos_error_spring = self.x_des - self.x_ref                    # [B, 3]

        f_spring = torch.bmm(Kp, pos_error_spring.unsqueeze(-1)).squeeze(-1)  # [B, 3]
        f_damp   = torch.bmm(Kd, self.v_ref.unsqueeze(-1)).squeeze(-1)        # [B, 3]

        a_lin = (f_spring - f_damp + f_net) / self.mass               # [B, 3]

        # Safety clip acceleration
        a_lin = torch.clamp(a_lin, min=-50.0, max=50.0)

        # Semi-implicit Euler integration (update virtual reference)
        self.v_ref = self.v_ref + a_lin * self.control_dt
        self.v_ref = self.v_ref * 0.998   # tiny drag to prevent unbounded drift
        self.x_ref = self.x_ref + self.v_ref * self.control_dt

        # ---- 8. Task-space PD: track virtual reference with actual robot ----
        #  dx = (Kp_task * (x_ref - x_actual) + v_ref) * dt
        #
        # The Kp_task term provides CLOSED-LOOP position feedback: without it,
        # the controller only outputs feedforward velocity and the actual robot
        # never catches up to x_ref after a disturbance.
        pos_error_track = self.x_ref - palm_pos                       # [B, 3]
        dx_vel = self.Kp_task * pos_error_track + self.v_ref          # [B, 3] m/s
        dx_task = dx_vel * self.control_dt                            # [B, 3] m

        # ---- 9. Task-space → joint-space via damped least squares ----
        J_p_T = J_p.transpose(1, 2)                                   # [B, 6, 3]
        A_dls = torch.bmm(J_p, J_p_T)                                 # [B, 3, 3]
        A_dls = A_dls + self.dls_lambda * torch.eye(3, device=self.device).unsqueeze(0)
        dq_arm = torch.bmm(
            J_p_T, torch.linalg.solve(A_dls, dx_task.unsqueeze(-1))
        ).squeeze(-1)                                                 # [B, 6]

        # ---- 10. (Optional) rotational compliance ----
        if self.K_rot > 0:
            d_ori = self.tau_ext_filtered / self.D_rot * self.control_dt * 0.1
            J_r_T = J_r.transpose(1, 2)
            A_r_dls = torch.bmm(J_r, J_r_T) + self.dls_lambda * torch.eye(3, device=self.device).unsqueeze(0)
            dq_rot = torch.bmm(
                J_r_T, torch.linalg.solve(A_r_dls, d_ori.unsqueeze(-1))
            ).squeeze(-1)
            dq_arm = dq_arm + dq_rot

        # ---- 11. Convert to standardised action command [-1, +1] ----
        action_arm_cmd = dq_arm / self.action_scale_arm
        action_arm_cmd = torch.clamp(action_arm_cmd, min=-1.0, max=1.0)

        # ---- 12. Rate limiting for smoothness ----
        if self.prev_action_arm.shape[0] != B:
            self.prev_action_arm = torch.zeros(B, 6, device=self.device)
        action_delta = torch.clamp(
            action_arm_cmd - self.prev_action_arm,
            min=-self.action_rate_limit,
            max=self.action_rate_limit,
        )
        action_arm = self.prev_action_arm + action_delta
        self.prev_action_arm = action_arm

        # ---- 13. Hand joints stay still ----
        action_hand = torch.zeros(B, 16, device=self.device)

        # ---- Debug logging (rate-limited) ----
        if not hasattr(self, "_debug_counter"):
            self._debug_counter = 0
        self._debug_counter += 1
        if self._debug_counter % 200 == 0:
            f_norm = torch.linalg.vector_norm(self.f_ext_filtered, dim=-1).mean().item()
            v_norm = torch.linalg.vector_norm(self.v_ref, dim=-1).mean().item()
            track_err = torch.linalg.vector_norm(pos_error_track, dim=-1).mean().item()
            spring_err = torch.linalg.vector_norm(pos_error_spring, dim=-1).mean().item()
            dq_norm = torch.linalg.vector_norm(dq_arm, dim=-1).mean().item()
            print(
                f"[PalmCtrl] |f|={f_norm:.2f}N  |v_ref|={v_norm:.3f}m/s  "
                f"|track|={track_err:.4f}m  |spring|={spring_err:.4f}m  "
                f"|dq|={dq_norm:.4f}rad  |act|={action_arm_cmd.abs().max().item():.2f}"
            )

        return torch.cat([action_arm, action_hand], dim=-1)

    
    
class NullComplianceController:
    """一个不做任何补偿的控制器，用于对比测试"""
    def __init__(self, device: str, num_envs: int, **kwargs):
        self.device = device

    def __call__(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size = obs["policy"].shape[0]
        return torch.zeros((batch_size, 22), device=self.device)

@dataclass
class LeapHandPalmControlCfg(RslRlOnPolicyRunnerCfg):
    seed: int = 42
    device: str = "cuda:0"
    """用于传递给采集脚本的配置"""
    # policy_class: type = NullComplianceController
    policy_class: type = LeapHandPalmComplianceController
    amplitude: float = 0.5
