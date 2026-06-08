from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
import mujoco
from scipy.spatial.transform import Rotation as R

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityCfg
from mjlab.entity.entity import EntityArticulationInfoCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
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
    _ = (display_forces, display_every, display_top_k)

    m = env.sim.mj_model
    sensor = env.scene[sensor_name]
    assert isinstance(sensor, ContactSensor)
    sensor_data = sensor.data
    assert sensor_data.force is not None

    forces = torch.linalg.vector_norm(sensor_data.force, dim=-1)
    num_envs = forces.shape[0]
    num_fsrs = forces.shape[1]
    forces_tensor = torch.zeros((num_envs, expected_num_fsrs), device=forces.device, dtype=forces.dtype)
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
                color = c_rgba_t if forces_tensor[0, fsr_idx] > 0.0 else d_rgba_t
                sim_geom_rgba[gid] = color
            else:
                active_env_mask = forces_tensor[:, fsr_idx] > 0.0
                sim_geom_rgba[active_env_mask, gid] = c_rgba_t
                sim_geom_rgba[~active_env_mask, gid] = d_rgba_t

    return forces_tensor


def joint_pos(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return asset.data.joint_pos


def qfrc_actuator_arm(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return asset.data.qfrc_actuator[:, 0:6]


def qfrc_bias_arm(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    if hasattr(env.sim, "mj_data"):
        bias = env.sim.mj_data.qfrc_bias
        bias_t = torch.as_tensor(bias, device=env.device, dtype=torch.float32)
        if bias_t.ndim == 1:
            bias_t = bias_t.unsqueeze(0)
        return bias_t[:, 0:6]
    return torch.zeros((env.num_envs, 6), device=env.device)

# def qfrc_bias_arm(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
#     m = env.sim.mj_model
#     d = env.sim.mj_data
#     asset = env.scene[asset_cfg.name]
    
#     # 显式同步：从 GPU asset 拉取当前 qpos/qvel 到 CPU mj_data → mj_forward
#     joint_pos = asset.data.joint_pos[0].cpu().numpy()   # num_envs=1
#     joint_vel = asset.data.joint_vel[0].cpu().numpy()
#     d.qpos[:] = joint_pos
#     d.qvel[:] = joint_vel
#     mujoco.mj_forward(m, d)
    
#     bias = d.qfrc_bias
#     bias_t = torch.as_tensor(bias, device=env.device, dtype=torch.float32)
#     return bias_t.unsqueeze(0)[:, 0:6]

def _find_body_id(m, body_name: str, env_idx: int, num_envs: int) -> int:
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
    suffix = f"/{body_name}"
    for b_idx in range(m.nbody):
        b_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b_idx)
        if b_name and (b_name == body_name or b_name.endswith(suffix)):
            return b_idx
    return -1


def _find_joint_id(m, joint_name: str, env_idx: int, num_envs: int) -> int:
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
            raise ValueError(f"cannot find body '{body_name}'")

        d.qpos[:] = joint_pos_gpu[i].cpu().numpy()
        d.qvel[:] = joint_vel_gpu[i].cpu().numpy()
        mujoco.mj_forward(m, d)

        jac_p = np.zeros((3, m.nv))
        jac_r = np.zeros((3, m.nv))
        palm_pos = d.xpos[bid]
        mujoco.mj_jac(m, d, jac_p, jac_r, palm_pos, bid)
        positions.append(palm_pos.copy())

        jid = _find_joint_id(m, "joint1", i, num_envs)
        if jid == -1:
            raise ValueError("cannot find joint 'joint1'")

        dof_adr = m.jnt_dofadr[jid]
        jacs_p.append(jac_p[:, dof_adr : dof_adr + 6])
        jacs_r.append(jac_r[:, dof_adr : dof_adr + 6])

    jacs_p_t = torch.tensor(np.array(jacs_p), device=env.device, dtype=torch.float32)
    jacs_r_t = torch.tensor(np.array(jacs_r), device=env.device, dtype=torch.float32)
    pos_t = torch.tensor(np.array(positions), device=env.device, dtype=torch.float32)
    return jacs_p_t.view(num_envs, -1), jacs_r_t.view(num_envs, -1), pos_t


def palm_pos(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), body_name: str = "palm_lower") -> torch.Tensor:
    _, _, pos = _compute_palm_jacobians(env, asset_cfg, body_name)
    return pos


def palm_jacobian(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), body_name: str = "palm_lower") -> torch.Tensor:
    jac_p, _, _ = _compute_palm_jacobians(env, asset_cfg, body_name)
    return jac_p


def palm_jacobian_rot(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), body_name: str = "palm_lower") -> torch.Tensor:
    _, jac_r, _ = _compute_palm_jacobians(env, asset_cfg, body_name)
    return jac_r


def joint_vel_arm(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, 0:6]


def _get_target_box_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="target_ball", mocap=True)
    body.add_geom(
        name="target_capsule_medium_geom",
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        size=[0.15, 0.08],
        rgba=[0.2, 0.6, 1.0, 1.0],
        mass=1,
    )
    return spec


def _make_env_cfg(num_envs: int = 1, play: bool = False) -> ManagerBasedRlEnvCfg:
    robot_cfg = EntityCfg(
        spec_fn=lambda: _load_leaphand_spec(enable_hand_object_only_collision=_ENABLE_HAND_OBJECT_ONLY_COLLISION),
        articulation=EntityArticulationInfoCfg(
            actuators=(
                # 给机械臂足够高的刚度，消除底层物理滞后（内环必须足够硬）
                BuiltinPositionActuatorCfg(
                    target_names_expr=(r"^(joint[1-6])$",), 
                    stiffness=2000.0,  # 增大刚度
                    damping=200.0,
                    effort_limit=500.0,
                ),
                # 给灵巧手符合其物理质量的极低刚度，消除高频噪声
                BuiltinPositionActuatorCfg(
                    target_names_expr=(r"^[0-9]+$",), # 手指关节
                    stiffness=20.0,    # 缩小 100 倍
                    damping=2.0,
                    effort_limit=5.0,
                ),
            ),
        ),
        init_state=EntityCfg.InitialStateCfg(
            pos=(0, 0, 0),
            joint_pos={
                "joint1": 0.0,
                "joint2": 1.183,
                "joint3": -1.541,
                "joint4": 3.1415,       # away from π (singularity)
                "joint5": 2.742,
                "joint6": -1.569,
                "13": 1.57,
            },
        ),
    )
    
    target_cfg = EntityCfg(
        spec_fn=_get_target_box_spec,
        init_state=EntityCfg.InitialStateCfg(pos=(0.7007, 0.5003, 0.8377)),
    )

    observations = {
        "policy": ObservationGroupCfg({
            "joint_pos": ObservationTermCfg(func=joint_pos, params={"asset_cfg": SceneEntityCfg("robot")}),
            "joint_vel_arm": ObservationTermCfg(func=joint_vel_arm, params={"asset_cfg": SceneEntityCfg("robot")}),
            "qfrc_actuator_arm": ObservationTermCfg(func=qfrc_actuator_arm, params={"asset_cfg": SceneEntityCfg("robot")}),
            "qfrc_bias_arm": ObservationTermCfg(func=qfrc_bias_arm, params={"asset_cfg": SceneEntityCfg("robot")}),
            "palm_jacobian": ObservationTermCfg(func=palm_jacobian, params={"asset_cfg": SceneEntityCfg("robot"), "body_name": "palm_lower"}),
            "palm_jacobian_rot": ObservationTermCfg(func=palm_jacobian_rot, params={"asset_cfg": SceneEntityCfg("robot"), "body_name": "palm_lower"}),
            "palm_pos": ObservationTermCfg(func=palm_pos, params={"asset_cfg": SceneEntityCfg("robot"), "body_name": "palm_lower"}),
        })
    }

    actions: dict[str, ActionTermCfg] = {
        "arm_delta": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            use_default_offset=False,
        )
    }

    return ManagerBasedRlEnvCfg(
        decimation=5, 
        scene=SceneCfg(
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
                gravity=(0.0, 0.0, -9.81), 
                ccd_iterations=200,
                solver="newton",
            ),
            njmax=1000,
            nconmax=500,
        ),
        viewer=ViewerConfig(entity_name="robot", body_name="base_link", distance=2.0),
        episode_length_s=1e10 if play else 50.0,
    )


def leaphand_palm_contact_env_cfg(num_envs: int = 1, play: bool = False) -> ManagerBasedRlEnvCfg:
    return _make_env_cfg(num_envs=num_envs, play=play)



# # ==============================================================================
# #  Observer helpers: independent CPU MuJoCo model (MCC WrenchSim pattern)
# # ==============================================================================

# def _build_observer() -> tuple[mujoco.MjModel, mujoco.MjData, np.ndarray, int]:
#     """Create an independent CPU MuJoCo instance for bias/Jacobian computation.

#     This mirrors MCC's WrenchSim: a dedicated model that is explicitly synced
#     with qpos each step, with qvel=0 so that qfrc_bias = pure gravity.

#     Returns:
#         model, data, arm_dof_idx (6,), palm_bid
#     """
#     model = mujoco.MjModel.from_xml_path(str(_LEAPHAND_XML))
#     data = mujoco.MjData(model)

#     # Resolve arm joint DOF indices: joint1..joint6 (all hinge → 1 DOF each)
#     dof_indices: list[int] = []
#     for joint_name in [f"joint{i}" for i in range(1, 7)]:
#         jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
#         if jid < 0:
#             raise ValueError(f"Joint '{joint_name}' not found in observer model.")
#         dof_indices.append(int(model.jnt_dofadr[jid]))
#     arm_dof_idx = np.array(dof_indices, dtype=np.int32)

#     # Find palm body
#     palm_bid = -1
#     for name in ["palm_lower", "palm", "robot/palm_lower"]:
#         bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
#         if bid >= 0:
#             palm_bid = int(bid)
#             break
#     if palm_bid < 0:
#         raise ValueError("Palm body not found in observer model.")

#     return model, data, arm_dof_idx, palm_bid


# # Singleton observer shared across all controller instances
# _OBSERVER = None  # (model, data, arm_dof_idx, palm_bid)


# def _get_observer():
#     global _OBSERVER
#     if _OBSERVER is None:
#         _OBSERVER = _build_observer()
#     return _OBSERVER


# def _sync_and_compute(
#     qpos_np: np.ndarray,
# ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#     """Sync observer with qpos (qvel=0) and return bias, J_p, J_r, palm_pos.

#     This is the MCC-equivalent of::

#         wrench_sim.set_qpos(qpos)
#         wrench_sim.forward()
#         bias = wrench_sim.bias_torque()
#         jacp, jacr = wrench_sim.site_jacobian(site)

#     Args:
#         qpos_np: full joint positions (nq,) as numpy array.

#     Returns:
#         bias:    (6,)  generalized bias force (gravity only) on arm DOFs
#         jac_p:   (3,6) position Jacobian at palm body
#         jac_r:   (3,6) rotation Jacobian at palm body
#         palm_pos: (3,)  world-frame palm body position
#     """
#     model, data, arm_dof_idx, palm_bid = _get_observer()

#     data.qpos[:] = qpos_np
#     data.qvel[:] = 0.0  # ← KEY: zero velocity → pure gravity bias
#     mujoco.mj_forward(model, data)

#     bias = data.qfrc_bias[arm_dof_idx].copy().astype(np.float64)

#     jac_p = np.zeros((3, model.nv), dtype=np.float64)
#     jac_r = np.zeros((3, model.nv), dtype=np.float64)
#     palm_pos = data.xpos[palm_bid].copy()
#     mujoco.mj_jac(model, data, jac_p, jac_r, palm_pos, palm_bid)

#     jac_p_arm = jac_p[:, arm_dof_idx].copy()
#     jac_r_arm = jac_r[:, arm_dof_idx].copy()

#     return bias, jac_p_arm, jac_r_arm, palm_pos


# ==============================================================================
#  数学基础工具 (NumPy 临界阻尼计算，与 MCC 完美一致)
# ==============================================================================

def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)

def _matrix_sqrt(matrix: np.ndarray) -> np.ndarray:
    sym = _symmetrize(matrix)
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals_clipped = np.clip(eigvals, 0.0, None)
    sqrt_matrix = eigvecs @ np.diag(np.sqrt(eigvals_clipped)) @ eigvecs.T
    return _symmetrize(sqrt_matrix)

def get_damping_matrix(stiffness: np.ndarray, inertia_like: np.ndarray) -> np.ndarray:
    mass_sqrt = _matrix_sqrt(inertia_like)
    stiffness_sqrt = _matrix_sqrt(stiffness)
    damping = 2.0 * (mass_sqrt @ stiffness_sqrt)
    return _symmetrize(damping)


# ==============================================================================
#  Observer 单例支持（ bias & Jacobian & joint limits）
# ==============================================================================

_LEAPHAND_XML = Path("/home/rimlab/Code/Hand_Compliance_Control/src/mjlab/asset_zoo/robots/xarm6_leap_hand/xarm6_leap_hand_nolimit.xml")
_OBSERVER = None 

def _build_observer() -> tuple[mujoco.MjModel, mujoco.MjData, np.ndarray, int, np.ndarray]:
    model = mujoco.MjModel.from_xml_path(str(_LEAPHAND_XML))
    data = mujoco.MjData(model)

    dof_indices: list[int] = []
    jnt_ranges = []
    for joint_name in [f"joint{i}" for i in range(1, 7)]:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0:
            raise ValueError(f"Joint '{joint_name}' not found in observer model.")
        dof_indices.append(int(model.jnt_dofadr[jid]))
        jnt_ranges.append(model.jnt_range[jid].copy())
    arm_dof_idx = np.array(dof_indices, dtype=np.int32)
    arm_jnt_ranges = np.array(jnt_ranges, dtype=np.float32)

    palm_bid = -1
    for name in ["palm_lower", "palm", "robot/palm_lower"]:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            palm_bid = int(bid)
            break
    if palm_bid < 0:
        raise ValueError("Palm body not found in observer model.")

    return model, data, arm_dof_idx, palm_bid, arm_jnt_ranges


def _get_observer():
    global _OBSERVER
    if _OBSERVER is None:
        _OBSERVER = _build_observer()
    return _OBSERVER


def _sync_and_compute(
    qpos_np: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """同步实际物理状态"""
    model, data, arm_dof_idx, palm_bid, _ = _get_observer()

    data.qpos[:] = qpos_np
    data.qvel[:] = 0.0  # 静力学观测：速度置零以得到纯重力/科氏力
    mujoco.mj_forward(model, data)

    bias = data.qfrc_bias[arm_dof_idx].copy().astype(np.float32)

    jac_p = np.zeros((3, model.nv), dtype=np.float64)
    jac_r = np.zeros((3, model.nv), dtype=np.float64)
    palm_pos = data.xpos[palm_bid].copy().astype(np.float32)
    mujoco.mj_jac(model, data, jac_p, jac_r, palm_pos, palm_bid)

    jac_p_arm = jac_p[:, arm_dof_idx].copy().astype(np.float32)
    jac_r_arm = jac_r[:, arm_dof_idx].copy().astype(np.float32)
    palm_rot = data.xmat[palm_bid].reshape(3, 3).copy().astype(np.float32)

    return bias, jac_p_arm, jac_r_arm, palm_pos, palm_rot


def _get_virtual_palm_fk(
    q_arm_np: np.ndarray,
    full_qpos_np: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """计算完全解耦的虚拟关节状态下的 FK 和 雅可比"""
    model, data, arm_dof_idx, palm_bid, _ = _get_observer()

    qpos_mix = full_qpos_np.copy()
    qpos_mix[:6] = q_arm_np
    
    data.qpos[:] = qpos_mix
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    jac_p = np.zeros((3, model.nv), dtype=np.float64)
    jac_r = np.zeros((3, model.nv), dtype=np.float64)
    palm_pos = data.xpos[palm_bid].copy().astype(np.float32)
    mujoco.mj_jac(model, data, jac_p, jac_r, palm_pos, palm_bid)

    jac_p_arm = jac_p[:, arm_dof_idx].copy().astype(np.float32)
    jac_r_arm = jac_r[:, arm_dof_idx].copy().astype(np.float32)
    palm_rot = data.xmat[palm_bid].reshape(3, 3).copy().astype(np.float32)

    return palm_pos, palm_rot, jac_p_arm, jac_r_arm

# ==============================================================================
#  MCC-style compliance controller
# ==============================================================================

class LeapHandPalmComplianceController:
    """掌部合规控制器（完全对齐 MCC 设计规范）"""

    def __init__(self, device: str, num_envs: int, **kwargs):
        self.device = device
        self.num_envs = num_envs
        self.control_dt = float(kwargs.get("control_dt", 0.01))

        # ── 物理参数 ──
        self.mass = float(kwargs.get("mass_trans", 3.0))
        self.inertia_diag = np.array(kwargs.get("inertia_diag", [0.1, 0.1, 0.1]), dtype=np.float32)

        # ── 刚度参数 ──
        self.K_normal = float(kwargs.get("K_normal", 200.0))
        self.K_tangent = float(kwargs.get("K_tangent", 20.0))
        self.K_rot = float(kwargs.get("K_rot", 30.0))

        normal_axis = str(kwargs.get("normal_axis", "z")).lower()
        axis_map = {"x": 0, "y": 1, "z": 2}
        self.normal_idx = axis_map[normal_axis]
        self.tangent_indices = [i for i in range(3) if i != self.normal_idx]

        # ── 构建阻尼和刚度矩阵 (通过临界阻尼法自适应计算，与 MCC get_damping_matrix 严格一致) ──
        k_pos_diagonal = np.zeros(3, dtype=np.float32)
        k_pos_diagonal[self.normal_idx] = self.K_normal
        for t in self.tangent_indices:
            k_pos_diagonal[t] = self.K_tangent

        self.kp_pos = np.diag(k_pos_diagonal)
        self.kd_pos = get_damping_matrix(self.kp_pos, np.eye(3, dtype=np.float32) * self.mass)

        self.kp_rot = np.eye(3, dtype=np.float32) * self.K_rot
        self.kd_rot = get_damping_matrix(self.kp_rot, np.diag(self.inertia_diag))

        # ── 跟踪控制参数 (DLS IK & Nullspace) ──
        self.Kp_task = float(kwargs.get("Kp_task", 10.0))
        self.dls_lambda = float(kwargs.get("dls_lambda", 0.05))
        self._k_posture = float(kwargs.get("k_posture", 0.15))

        # ── 滤波与正则化参数 ──
        self.alpha_tau = float(kwargs.get("alpha_tau", 0.1))
        self.alpha_wrench = float(kwargs.get("alpha_wrench", 0.03))
        self.lambda_force = float(kwargs.get("lambda_force", 1e-3))
        self.lambda_torque = float(kwargs.get("lambda_torque", 1e-2))

        # ── 引导(Preparation)阶段参数 ──
        self.prep_duration_s = float(kwargs.get("prep_duration_s", 1.5))
        self.prep_steps = int(self.prep_duration_s / self.control_dt)

        # ── 预设力指令 (Feedforward) ──
        default_force = kwargs.get("f_cmd_default", [0.0, 0.0, 0.0])
        self.f_cmd_default = np.array(default_force, dtype=np.float32)

        # ── 环境级 CPU 动力学状态字典 ──
        self._init_states()
        self._step_count = 0

        # 热启动 observer 获取约束限制
        _, _, _, _, self.arm_jnt_ranges = _get_observer()

    def _init_states(self) -> None:
        self.states = []
        for _ in range(self.num_envs):
            self.states.append({
                "initialized": False,
                "prep_counter": 0,
                "q_init": None,          # 过渡初始关节位置
                "q_posture": None,       # 期望零空间标定姿态
                "q_ref": None,           # 核心：完全解耦的虚拟集成关节位置
                "x_des": None,           # 指令笛卡尔目标姿态 [x, y, z, rx, ry, rz]
                "x_ref": None,           # 合规积分参考状态
                "v_ref": np.zeros(6, dtype=np.float32),
                "tau_smoothed": np.zeros(6, dtype=np.float32),
                "bias_smoothed": np.zeros(6, dtype=np.float32),
                "f_ext_filtered": np.zeros(3, dtype=np.float32),
                "tau_ext_filtered": np.zeros(3, dtype=np.float32),
            })

    def __call__(
        self, obs: dict[str, torch.Tensor], f_cmd: torch.Tensor | None = None
    ) -> torch.Tensor:
        policy_obs = obs["policy"]
        B = policy_obs.shape[0]
        if B != self.num_envs:
            self.num_envs = B
            self._init_states()

        self._step_count += 1

        # 转换物理输入到 CPU 进行精确、稳定的高频动力学计算
        joint_pos_all_gpu = policy_obs[:, 0:22]
        joint_pos_all_np = joint_pos_all_gpu.cpu().numpy().astype(np.float32)
        qfrc_actuator_arm_np = policy_obs[:, 28:34].cpu().numpy().astype(np.float32)

        if f_cmd is not None:
            f_cmd_np = f_cmd.cpu().numpy().astype(np.float32)
            if f_cmd_np.ndim == 1:
                f_cmd_np = np.tile(f_cmd_np, (B, 1))
        else:
            f_cmd_np = np.tile(self.f_cmd_default, (B, 1))

        output_joint_targets_np = np.zeros_like(joint_pos_all_np)

        # ── 并行遍历每个仿真环境 ──
        for i in range(B):
            state = self.states[i]
            qpos_full = joint_pos_all_np[i]

            # 1. 物理同步获取动力学偏置力矩与传感器物理量
            bias, jp, jr, palm_pos, palm_rot = _sync_and_compute(qpos_full)

            # 2. 引导阶段逻辑（Preparation Phase，平滑消除开机位置瞬变冲击）
            if state["prep_counter"] < self.prep_steps:
                if state["prep_counter"] == 0:
                    state["q_init"] = qpos_full[:6].copy()
                    state["q_posture"] = np.array([0.0, 1.183, -1.541, 3.1415, 2.742, -1.569], dtype=np.float32)
                
                state["prep_counter"] += 1
                t = state["prep_counter"] / self.prep_steps
                
                # 在关节空间执行极小震荡的纯过渡插值
                output_joint_targets_np[i, :6] = (1.0 - t) * state["q_init"] + t * state["q_posture"]
                output_joint_targets_np[i, 6:] = qpos_full[6:]
                continue

            # 3. 初始合规状态对齐阶段（Alignment Phase）
            if not state["initialized"]:
                state["q_ref"] = qpos_full[:6].copy()
                state["q_posture"] = qpos_full[:6].copy()
                state["tau_smoothed"] = qfrc_actuator_arm_np[i].copy()
                state["bias_smoothed"] = bias.copy()

                rot_vec = R.from_matrix(palm_rot).as_rotvec().astype(np.float32)
                state["x_des"] = np.concatenate([palm_pos, rot_vec])
                state["x_ref"] = state["x_des"].copy()
                state["v_ref"] = np.zeros(6, dtype=np.float32)
                state["initialized"] = True

            # 4. 力矩一阶 EMA 滤波 (严格遵守 MCC tau_ext 符号定义：tau_ext = -(tau_raw - bias))
            state["tau_smoothed"] = self.alpha_tau * qfrc_actuator_arm_np[i] + (1.0 - self.alpha_tau) * state["tau_smoothed"]
            state["bias_smoothed"] = self.alpha_tau * bias + (1.0 - self.alpha_tau) * state["bias_smoothed"]
            tau_ext = -(state["tau_smoothed"] - state["bias_smoothed"])

            # 5. 外力/力矩独立轴向投影估计 (Wrench Independent Estimation)
            A_p = jp @ jp.T + self.lambda_force * np.eye(3)
            f_ext_raw = np.linalg.solve(A_p, jp @ tau_ext)

            A_r = jr @ jr.T + self.lambda_torque * np.eye(3)
            tau_ext_raw = np.linalg.solve(A_r, jr @ tau_ext)

            state["f_ext_filtered"] = self.alpha_wrench * f_ext_raw + (1.0 - self.alpha_wrench) * state["f_ext_filtered"]
            state["tau_ext_filtered"] = self.alpha_wrench * tau_ext_raw + (1.0 - self.alpha_wrench) * state["tau_ext_filtered"]

            # 6. Admittance 动力学积分求解（通过临界阻尼矩阵迭代计算合规目标 x_ref）
            pos_prev = state["x_ref"][:3]
            vel_prev = state["v_ref"][:3]
            pos_des = state["x_des"][:3]
            pos_error = pos_des - pos_prev

            f_net = state["f_ext_filtered"] + f_cmd_np[i]
            kp_term_pos = self.kp_pos @ pos_error
            kd_term_pos = self.kd_pos @ vel_prev
            lin_acc = (kp_term_pos - kd_term_pos + f_net) / self.mass

            vel_next = vel_prev + lin_acc * self.control_dt
            pos_next = pos_prev + vel_next * self.control_dt

            # 旋转合规动力学 (采用 3D Rotation Vector 进行几何代数更新)
            ori_prev = R.from_rotvec(state["x_ref"][3:6])
            omega_prev = state["v_ref"][3:6]
            ori_des = R.from_rotvec(state["x_des"][3:6])
            ori_error = (ori_des * ori_prev.inv()).as_rotvec()

            kp_term_rot = self.kp_rot @ ori_error
            kd_term_rot = self.kd_rot @ omega_prev
            ang_acc = (kp_term_rot - kd_term_rot + state["tau_ext_filtered"]) / self.inertia_diag

            omega_next = omega_prev + ang_acc * self.control_dt
            ori_next = (R.from_rotvec(omega_next * self.control_dt) * ori_prev).as_rotvec()

            state["x_ref"][:3] = pos_next
            state["x_ref"][3:6] = ori_next
            state["v_ref"][:3] = vel_next
            state["v_ref"][3:6] = omega_next

            # 7. 虚拟关节跟踪解耦环路（通过内存中仿真虚拟配置 q_ref，完全斩断物理外力负反馈链条）
            v_palm_pos, v_palm_rot, v_jp, v_jr = _get_virtual_palm_fk(state["q_ref"], qpos_full)

            v_e_pos = state["x_ref"][:3] - v_palm_pos
            v_e_rot = (R.from_rotvec(state["x_ref"][3:6]) * R.from_matrix(v_palm_rot).inv()).as_rotvec()

            v_dx_task = np.concatenate([v_e_pos, v_e_rot]) * self.Kp_task * self.control_dt

            # 虚拟 DLS-IK
            v_J = np.vstack([v_jp, v_jr])
            A_dls = v_J @ v_J.T + self.dls_lambda * np.eye(6)
            v_J_pinv = v_J.T @ np.linalg.inv(A_dls)

            dq_primary = v_J_pinv @ v_dx_task

            # 虚拟 Nullspace Posture Pull
            N_space = np.eye(6) - v_J_pinv @ v_J
            dq_posture = self._k_posture * (state["q_posture"] - state["q_ref"])
            dq_null = N_space @ dq_posture

            # 更新解耦关节参考目标，并约束安全限幅
            state["q_ref"] = np.clip(
                state["q_ref"] + dq_primary + dq_null,
                self.arm_jnt_ranges[:, 0],
                self.arm_jnt_ranges[:, 1]
            )

            # 最终物理指令（物理 PD 用 1000 的完整刚度驱动，稳态误差趋于零）
            output_joint_targets_np[i, :6] = state["q_ref"].copy()
            output_joint_targets_np[i, 6:] = qpos_full[6:]

            # 8. 周期日志反馈
            if self._step_count % 300 == 0 and i == 0:
                f_n = np.linalg.norm(state["f_ext_filtered"])
                t_err = np.linalg.norm(state["x_ref"][:3] - palm_pos)
                print(
                    f"[MCC-Strict] Step={self._step_count} | F_ext={f_n:.2f}N | "
                    f"Cart_Err={t_err:.4f}m | Ref_Z={state['x_ref'][2]:.4f} | Obs_Z={palm_pos[2]:.4f}"
                )

        # 返回 GPU 执行高刚度 PD 输出
        return torch.as_tensor(output_joint_targets_np, device=self.device, dtype=torch.float32)

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
    policy_class: type = LeapHandPalmComplianceController
    amplitude: float = 0.5