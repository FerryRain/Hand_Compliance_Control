from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

# 将项目根目录加入 sys.path，使 minimalist_compliance_control 可被导入
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

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
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

# ── MCC 核心数学组件 ──
from minimalist_compliance_control.minimalist_compliance_control.wrench_estimation import (
    WrenchEstimateConfig,
    estimate_wrench,
)
from minimalist_compliance_control.minimalist_compliance_control.utils import get_damping_matrix

# ==============================================================================
#  常量 & 工具函数（复用自 leaphand_palm_env_cfg）
# ==============================================================================

_LEAPHAND_XML = Path(
    "/home/rimlab/Code/Hand_Compliance_Control/src/mjlab/asset_zoo/robots/xarm6_leap_hand/xarm6_leap_hand_nolimit.xml"
)
_ENABLE_HAND_OBJECT_ONLY_COLLISION = False
_HAND_CONTYPE = 2
_HAND_CONAFFINITY = 4
_OBJECT_CONTYPE = 4
_OBJECT_CONAFFINITY = 2


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


def _find_body_id(m, body_name: str, env_idx: int, num_envs: int) -> int:
    names_to_try = [
        f"env_{env_idx}/{body_name}",
        f"robot_{env_idx}/{body_name}",
        f"env_{env_idx}/robot/{body_name}",
        f"robot/{body_name}",
        body_name,
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
        joint_name,
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


def _compute_palm_state(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    body_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """一次性计算 palm 的雅可比、位置、姿态，避免重复 mj_forward。

    Returns:
        jacs_p:      (num_envs, 3*6)  position Jacobian (arm columns only)
        jacs_r:      (num_envs, 3*6)  rotation Jacobian (arm columns only)
        positions:   (num_envs, 3)    world-frame palm position
        rotvecs:     (num_envs, 3)    world-frame palm rotation vector
    """
    m = env.sim.mj_model
    d = env.sim.mj_data
    num_envs = env.num_envs
    asset = env.scene[asset_cfg.name]
    joint_pos_gpu = asset.data.joint_pos
    joint_vel_gpu = asset.data.joint_vel

    jacs_p, jacs_r, positions, rotvecs = [], [], [], []

    for i in range(num_envs):
        bid = _find_body_id(m, body_name, i, num_envs)
        if bid == -1:
            raise ValueError(f"cannot find body '{body_name}'")

        d.qpos[:] = joint_pos_gpu[i].cpu().numpy()
        d.qvel[:] = joint_vel_gpu[i].cpu().numpy()
        mujoco.mj_forward(m, d)

        jac_p = np.zeros((3, m.nv), dtype=np.float64)
        jac_r = np.zeros((3, m.nv), dtype=np.float64)
        palm_pos = d.xpos[bid].copy()
        mujoco.mj_jac(m, d, jac_p, jac_r, palm_pos, bid)

        positions.append(palm_pos)
        rotmat = d.xmat[bid].reshape(3, 3).copy()
        rotvecs.append(R.from_matrix(rotmat).as_rotvec().astype(np.float32))

        jid = _find_joint_id(m, "joint1", i, num_envs)
        if jid == -1:
            raise ValueError("cannot find joint 'joint1'")
        dof_adr = m.jnt_dofadr[jid]
        jacs_p.append(jac_p[:, dof_adr : dof_adr + 6].copy().astype(np.float32))
        jacs_r.append(jac_r[:, dof_adr : dof_adr + 6].copy().astype(np.float32))

    jacs_p_t = torch.tensor(np.array(jacs_p), device=env.device, dtype=torch.float32)
    jacs_r_t = torch.tensor(np.array(jacs_r), device=env.device, dtype=torch.float32)
    pos_t = torch.tensor(np.array(positions), device=env.device, dtype=torch.float32)
    rot_t = torch.tensor(np.array(rotvecs), device=env.device, dtype=torch.float32)
    return (
        jacs_p_t.view(num_envs, -1),
        jacs_r_t.view(num_envs, -1),
        pos_t,
        rot_t,
    )


# ==============================================================================
#  表面跟踪工具：Capsule 射线交点（独立于控制器，外部调用）
# ==============================================================================

def capsule_surface_intersection(
    center: np.ndarray,
    rotmat: np.ndarray,
    radius: float,
    half_height: float,
    point: np.ndarray,
) -> np.ndarray:
    """计算 capsule 中心到 point 连线与 capsule 表面的交点。

    Capsule 轴线沿 geom 局部 Z 轴 (rotmat[:, 2])。
    可用于外部实时计算 x_des 后传入控制器。

    Args:
        center:      capsule 世界坐标中心 (3,)
        rotmat:      capsule 世界坐标系旋转矩阵 (3,3)
        radius:      capsule 半径
        half_height: capsule 半高
        point:       查询点世界坐标 (3,)（如 palm_pos）

    Returns:
        表面交点世界坐标 (3,)
    """
    D = point - center
    d = float(np.linalg.norm(D))
    if d < 1e-9:
        return center + np.array([radius, 0.0, 0.0], dtype=np.float32)

    v = D / d
    a = rotmat[:, 2].astype(np.float32)
    v_dot_a = float(np.dot(v, a))
    sin_alpha_sq = 1.0 - v_dot_a * v_dot_a
    sin_alpha = np.sqrt(max(sin_alpha_sq, 0.0))

    h = half_height
    r = radius
    t_candidates = []

    # 圆柱体部分
    if sin_alpha > 1e-8:
        t_cyl = r / sin_alpha
        if t_cyl * abs(v_dot_a) <= h + 1e-6:
            t_candidates.append(t_cyl)

    # +h 半球帽
    b = -2.0 * h * v_dot_a
    c = h * h - r * r
    disc = b * b - 4.0 * c
    if disc >= 0.0:
        sqrt_disc = np.sqrt(disc)
        for t_cap in ((-b + sqrt_disc) / 2.0, (-b - sqrt_disc) / 2.0):
            if t_cap > 0.0 and t_cap * v_dot_a >= h - 1e-6:
                t_candidates.append(t_cap)

    # -h 半球帽
    b = 2.0 * h * v_dot_a
    c = h * h - r * r
    disc = b * b - 4.0 * c
    if disc >= 0.0:
        sqrt_disc = np.sqrt(disc)
        for t_cap in ((-b + sqrt_disc) / 2.0, (-b - sqrt_disc) / 2.0):
            if t_cap > 0.0 and t_cap * v_dot_a <= -h + 1e-6:
                t_candidates.append(t_cap)

    # 退化：球近似
    t_hit = r if not t_candidates else min(t_candidates)
    return (center + t_hit * v).astype(np.float32)


# ==============================================================================
#  观测函数
# ==============================================================================

def joint_pos(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return asset.data.joint_pos


def joint_vel_arm(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, 0:6]


def qfrc_actuator_arm(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return asset.data.qfrc_actuator[:, 0:6]


def qfrc_bias_arm(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """获取 arm 关节的静力学偏置力（重力+科氏力）。

    与 MCC 保持一致：在另一个独立 observer 中 qvel=0 得到纯重力偏置。
    这里为了重用现有 env 的 mj_data，直接取 qfrc_bias（包含速度相关项）。
    """
    if hasattr(env.sim, "mj_data"):
        bias = env.sim.mj_data.qfrc_bias
        bias_t = torch.as_tensor(bias, device=env.device, dtype=torch.float32)
        if bias_t.ndim == 1:
            bias_t = bias_t.unsqueeze(0)
        return bias_t[:, 0:6]
    return torch.zeros((env.num_envs, 6), device=env.device)


def palm_jacobian(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_name: str = "palm_lower",
) -> torch.Tensor:
    jac_p, _, _, _ = _compute_palm_state(env, asset_cfg, body_name)
    return jac_p


def palm_jacobian_rot(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_name: str = "palm_lower",
) -> torch.Tensor:
    _, jac_r, _, _ = _compute_palm_state(env, asset_cfg, body_name)
    return jac_r


def palm_pos(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_name: str = "palm_lower",
) -> torch.Tensor:
    _, _, pos, _ = _compute_palm_state(env, asset_cfg, body_name)
    return pos


def palm_rot(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_name: str = "palm_lower",
) -> torch.Tensor:
    """返回手掌 body 的世界坐标系旋转向量 (num_envs, 3)。"""
    _, _, _, rot = _compute_palm_state(env, asset_cfg, body_name)
    return rot


def _compute_target_state(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    body_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """获取 target body 的世界位置和旋转向量。

    复用 robot 关节状态做 forward，确保 target xpos 正确。
    """
    m = env.sim.mj_model
    d = env.sim.mj_data
    num_envs = env.num_envs
    asset = env.scene[asset_cfg.name]
    joint_pos_gpu = asset.data.joint_pos
    joint_vel_gpu = asset.data.joint_vel

    positions, rotvecs = [], []
    for i in range(num_envs):
        bid = _find_body_id(m, body_name, i, num_envs)
        if bid == -1:
            positions.append(np.zeros(3, dtype=np.float32))
            rotvecs.append(np.zeros(3, dtype=np.float32))
            continue
        d.qpos[:] = joint_pos_gpu[i].cpu().numpy()
        d.qvel[:] = joint_vel_gpu[i].cpu().numpy()
        mujoco.mj_forward(m, d)
        positions.append(d.xpos[bid].copy())
        rotmat = d.xmat[bid].reshape(3, 3).copy()
        rotvecs.append(R.from_matrix(rotmat).as_rotvec().astype(np.float32))

    pos_t = torch.tensor(np.array(positions), device=env.device, dtype=torch.float32)
    rot_t = torch.tensor(np.array(rotvecs), device=env.device, dtype=torch.float32)
    return pos_t, rot_t


def target_body_pos(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_name: str = "target_ball",
) -> torch.Tensor:
    """target body 世界位置 (num_envs, 3)。"""
    pos, _ = _compute_target_state(env, asset_cfg, body_name)
    return pos


def target_body_rot(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_name: str = "target_ball",
) -> torch.Tensor:
    """target body 世界旋转向量 (num_envs, 3)。"""
    _, rot = _compute_target_state(env, asset_cfg, body_name)
    return rot


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


# ==============================================================================
#  环境配置（与 leaphand_palm_env_cfg 相同，增加 palm_rot 观测）
# ==============================================================================

def _make_env_cfg(num_envs: int = 1, play: bool = False) -> ManagerBasedRlEnvCfg:
    robot_cfg = EntityCfg(
        spec_fn=lambda: _load_leaphand_spec(
            enable_hand_object_only_collision=_ENABLE_HAND_OBJECT_ONLY_COLLISION
        ),
        articulation=EntityArticulationInfoCfg(
            actuators=(
                BuiltinPositionActuatorCfg(
                    target_names_expr=(r"^(joint[1-6])$",),
                    stiffness=1000.0,
                    damping=100.0,
                    effort_limit=500.0,
                ),
                BuiltinPositionActuatorCfg(
                    target_names_expr=(r"^[0-9]+$",),
                    stiffness=20.0,
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
                "joint4": 3.1415,
                "joint5": 2.742,
                "joint6": -1.569,
                "13": 1.57,
            },
        ),
    )

    target_cfg = EntityCfg(
        spec_fn=_get_target_box_spec,
        init_state=EntityCfg.InitialStateCfg(pos=(0.7007, 0.0003, 0.8377)),
    )

    observations = {
        "policy": ObservationGroupCfg({
            "joint_pos": ObservationTermCfg(
                func=joint_pos, params={"asset_cfg": SceneEntityCfg("robot")}
            ),
            "joint_vel_arm": ObservationTermCfg(
                func=joint_vel_arm, params={"asset_cfg": SceneEntityCfg("robot")}
            ),
            "qfrc_actuator_arm": ObservationTermCfg(
                func=qfrc_actuator_arm, params={"asset_cfg": SceneEntityCfg("robot")}
            ),
            "qfrc_bias_arm": ObservationTermCfg(
                func=qfrc_bias_arm, params={"asset_cfg": SceneEntityCfg("robot")}
            ),
            "palm_jacobian": ObservationTermCfg(
                func=palm_jacobian,
                params={"asset_cfg": SceneEntityCfg("robot"), "body_name": "palm_lower"},
            ),
            "palm_jacobian_rot": ObservationTermCfg(
                func=palm_jacobian_rot,
                params={"asset_cfg": SceneEntityCfg("robot"), "body_name": "palm_lower"},
            ),
            "palm_pos": ObservationTermCfg(
                func=palm_pos,
                params={"asset_cfg": SceneEntityCfg("robot"), "body_name": "palm_lower"},
            ),
            "palm_rot": ObservationTermCfg(
                func=palm_rot,
                params={"asset_cfg": SceneEntityCfg("robot"), "body_name": "palm_lower"},
            ),
            "target_pos": ObservationTermCfg(
                func=target_body_pos,
                params={"asset_cfg": SceneEntityCfg("robot"), "body_name": "target_ball"},
            ),
            "target_rot": ObservationTermCfg(
                func=target_body_rot,
                params={"asset_cfg": SceneEntityCfg("robot"), "body_name": "target_ball"},
            ),
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


def mcc_palm_contact_env_cfg(num_envs: int = 1, play: bool = False) -> ManagerBasedRlEnvCfg:
    return _make_env_cfg(num_envs=num_envs, play=play)

# ---- 观测张量布局 ----
# 0:22   joint_pos          (22,)
# 22:28  joint_vel_arm       (6,)
# 28:34  qfrc_actuator_arm   (6,)
# 34:40  qfrc_bias_arm       (6,)
# 40:58  palm_jacobian       (18,) → reshape (3,6)
# 58:76  palm_jacobian_rot   (18,) → reshape (3,6)
# 76:79  palm_pos            (3,)
# 79:82  palm_rot            (3,)
# 82:85  target_pos          (3,)
# 85:88  target_rot          (3,)


# ==============================================================================
#  Observer 工具函数（控制器内部使用，等价于 MCC WrenchSim）
# ==============================================================================

_OBSERVER_CACHE: dict = {}


def _get_or_build_observer() -> tuple[mujoco.MjModel, mujoco.MjData, np.ndarray, int, np.ndarray]:
    """构建/返回独立的 MuJoCo observer 模型（等价于 MCC 的 WrenchSim）。

    该 observer 独立于环境 mj_data，用于：
      - 零速偏置力矩计算 (qfrc_bias with qvel=0)
      - 雅可比矩阵计算
      - 虚拟关节状态的 FK 计算（解耦跟踪环路）
    """
    if _OBSERVER_CACHE:
        return (
            _OBSERVER_CACHE["model"],
            _OBSERVER_CACHE["data"],
            _OBSERVER_CACHE["arm_dof_idx"],
            _OBSERVER_CACHE["palm_bid"],
            _OBSERVER_CACHE["arm_jnt_ranges"],
        )
    model = mujoco.MjModel.from_xml_path(str(_LEAPHAND_XML))
    data = mujoco.MjData(model)

    dof_indices: list[int] = []
    jnt_ranges: list[np.ndarray] = []
    for joint_name in [f"joint{i}" for i in range(1, 7)]:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0:
            raise ValueError(f"Joint '{joint_name}' not found in observer model.")
        dof_indices.append(int(model.jnt_dofadr[jid]))
        jnt_ranges.append(model.jnt_range[jid].copy())
    arm_dof_idx = np.array(dof_indices, dtype=np.int32)
    arm_jnt_ranges = np.array(jnt_ranges, dtype=np.float32)

    palm_bid = -1
    for name in ("palm_lower", "palm", "robot/palm_lower"):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            palm_bid = int(bid)
            break
    if palm_bid < 0:
        raise ValueError("Palm body not found in observer model.")

    _OBSERVER_CACHE["model"] = model
    _OBSERVER_CACHE["data"] = data
    _OBSERVER_CACHE["arm_dof_idx"] = arm_dof_idx
    _OBSERVER_CACHE["palm_bid"] = palm_bid
    _OBSERVER_CACHE["arm_jnt_ranges"] = arm_jnt_ranges
    return model, data, arm_dof_idx, palm_bid, arm_jnt_ranges


# ==============================================================================
#  MCCPalmComplianceController
# ==============================================================================

class MCCPalmComplianceController:
    """直接应用 MCC 核心数学管线的掌部笛卡尔顺应控制器。

    与 MCC (minimalist_compliance_control) 保持一致的：
      - wrench estimation (estimate_wrench, 正则化最小二乘)
      - 临界阻尼矩阵 (get_damping_matrix)
      - admittance 积分公式 (ComplianceReference.integrate_commands)
      - DLS-IK + nullspace posture

    仅控制 xarm6 的 6 个关节；手指关节保持当前位置不动。
    """

    def __init__(self, device: str, num_envs: int, **kwargs):
        self.device = device
        self.num_envs = num_envs

        # ── 控制频率 ──
        self.control_dt = float(kwargs.get("control_dt", 0.01))

        # ── 物理参数（等价于 MCC RefConfig.mass / inertia_diag）──
        self.mass = float(kwargs.get("mass_trans", 1.0))
        self.inertia_diag = np.array(
            kwargs.get("inertia_diag", (0.1, 0.1, 0.1)), dtype=np.float32
        )

        # ── 各向异性刚度：法向力控（软）+ 切向位置跟踪（硬），对齐 MCC ──
        # K_force:   接触法向刚度 → 极低，让法向退化为力控
        # K_position: 切向刚度     → 高，保持位置跟踪精度
        self.K_force = float(kwargs.get("K_force", 20.0))
        self.K_position = float(kwargs.get("K_position", 200.0))
        self.K_rot = float(kwargs.get("K_rot", 30.0))

        # 无接触时的默认法向（世界坐标系，手掌按压方向）
        normal_axis = str(kwargs.get("normal_axis", "z")).lower()
        axis_map = {"x": 0, "y": 1, "z": 2}
        self._default_normal_idx = axis_map.get(normal_axis, 2)
        _default_normal = np.zeros(3, dtype=np.float32)
        _default_normal[self._default_normal_idx] = 1.0
        self._default_contact_normal = _default_normal

        # 转动刚度和阻尼（保持各向同性）
        self.kp_rot = np.eye(3, dtype=np.float32) * self.K_rot
        self.kd_rot = get_damping_matrix(self.kp_rot, np.diag(self.inertia_diag))

        # ── 跟踪控制参数 (DLS-IK & Nullspace) ──
        self.Kp_task = float(kwargs.get("Kp_task", 0.8))
        self.dls_lambda = float(kwargs.get("dls_lambda", 0.1))
        self.k_posture = float(kwargs.get("k_posture", 0.1))

        # ── 力矩 EMA 滤波 ──
        self.alpha_tau = float(kwargs.get("alpha_tau", 0.1))

        # ── 接触后纯力控 PI 参数（仅接触时激活，接近阶段完全不动）──
        self.Kf_vel = float(kwargs.get("Kf_vel", 0.005))       # 力→速度 P 增益 (m/s/N)
        self.Kif_vel = float(kwargs.get("Kif_vel", 0.002))      # 力→速度 I 增益
        self._force_int_max_n = float(kwargs.get("force_int_max_n", 5.0))  # 积分抗饱和
        self._kd_normal = float(kwargs.get("kd_normal", 80.0))  # 接触法向显式阻尼 (K=0 时用)

        # ── MCC wrench estimation 配置 ──
        self._wrench_config = WrenchEstimateConfig(
            force_reg=float(kwargs.get("lambda_force", 1e-3)),
            torque_reg=float(kwargs.get("lambda_torque", 1e-2)),
        )

        # ── 引导 (Preparation) 阶段参数 ──
        self.prep_duration_s = float(kwargs.get("prep_duration_s", 1.5))
        self.prep_steps = max(1, int(self.prep_duration_s / self.control_dt))

        # ── 力控参数（法向力跟踪，对齐 MCC wrench_command）──
        # 期望法向接触力 (N)，正=推入表面（f_cmd 指向 -n，与 f_ext 方向相反）。0=纯位置伺服
        self.f_desired_normal = float(kwargs.get("f_desired_normal", 0.0))
        # 默认 f_cmd（世界坐标系 3D 向量），仅当 f_desired_normal=0 且未传入 f_cmd 时生效
        default_force = kwargs.get("f_cmd_default", (0.0, 0.0, 0.0))
        self.f_cmd_default = np.array(default_force, dtype=np.float32).reshape(3)

        # ── 接触法向估计参数 ──
        # 外力幅值超过此阈值时，用 f_ext 方向估计接触法向；否则使用 default_normal_axis
        self.contact_threshold = float(kwargs.get("contact_threshold", 0.5))
        # 接触法向 EMA 平滑因子（0-1，越大响应越快）
        self.alpha_normal = float(kwargs.get("alpha_normal", 0.1))

        # ── 构建独立 observer 模型 ──
        (
            self._obs_model,
            self._obs_data,
            self._arm_dof_idx,
            self._palm_bid,
            self._arm_jnt_ranges,
        ) = _get_or_build_observer()

        # ── 每环境状态字典 ──
        self._init_states()
        self._step_count = 0

        # 一次性诊断：确认关键力控参数
        print(
            f"[MCC-Palm] Init | f_desired_normal={self.f_desired_normal:.1f}N | "
            f"K_force={self.K_force:.0f} K_position={self.K_position:.0f} | "
            f"Kf_vel={self.Kf_vel:.4f} Kif_vel={self.Kif_vel:.4f} kd_n={self._kd_normal:.0f} | "
            f"contact_threshold={self.contact_threshold:.1f}N alpha_n={self.alpha_normal:.2f}"
        )

    # ------------------------------------------------------------------
    #  内部 observer 操作（等价于 MCC WrenchSim.set_qpos / forward / site_jacobian / bias_torque）
    # ------------------------------------------------------------------

    def _sync_observer(
        self, qpos_np: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """同步 observer 并返回 (bias, J_p, J_r, palm_pos, palm_rotmat)。

        关键：qvel=0 以得到纯重力偏置，与 MCC 的 wrench_sim.forward() 行为一致。
        """
        self._obs_data.qpos[:] = qpos_np
        self._obs_data.qvel[:] = 0.0
        mujoco.mj_forward(self._obs_model, self._obs_data)

        bias = self._obs_data.qfrc_bias[self._arm_dof_idx].copy().astype(np.float32)

        jac_p = np.zeros((3, self._obs_model.nv), dtype=np.float64)
        jac_r = np.zeros((3, self._obs_model.nv), dtype=np.float64)
        palm_pos = self._obs_data.xpos[self._palm_bid].copy().astype(np.float32)
        mujoco.mj_jac(
            self._obs_model, self._obs_data,
            jac_p, jac_r, palm_pos, self._palm_bid,
        )

        jac_p_arm = jac_p[:, self._arm_dof_idx].copy().astype(np.float32)
        jac_r_arm = jac_r[:, self._arm_dof_idx].copy().astype(np.float32)
        palm_rotmat = (
            self._obs_data.xmat[self._palm_bid].reshape(3, 3).copy().astype(np.float32)
        )

        return bias, jac_p_arm, jac_r_arm, palm_pos, palm_rotmat

    def _virtual_palm_fk(
        self, q_arm_np: np.ndarray, full_qpos_np: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """计算虚拟（解耦）关节状态下的 palm FK 和雅可比。

        仅修改 arm 关节 (qpos[:6])，手指关节保持物理实际值，从而斩断
        外力对虚拟参考的负反馈链条。
        """
        qpos_mix = full_qpos_np.copy()
        qpos_mix[:6] = q_arm_np

        self._obs_data.qpos[:] = qpos_mix
        self._obs_data.qvel[:] = 0.0
        mujoco.mj_forward(self._obs_model, self._obs_data)

        jac_p = np.zeros((3, self._obs_model.nv), dtype=np.float64)
        jac_r = np.zeros((3, self._obs_model.nv), dtype=np.float64)
        palm_pos = self._obs_data.xpos[self._palm_bid].copy().astype(np.float32)
        mujoco.mj_jac(
            self._obs_model, self._obs_data,
            jac_p, jac_r, palm_pos, self._palm_bid,
        )

        jac_p_arm = jac_p[:, self._arm_dof_idx].copy().astype(np.float32)
        jac_r_arm = jac_r[:, self._arm_dof_idx].copy().astype(np.float32)
        palm_rotmat = (
            self._obs_data.xmat[self._palm_bid].reshape(3, 3).copy().astype(np.float32)
        )

        return palm_pos, palm_rotmat, jac_p_arm, jac_r_arm

    # ------------------------------------------------------------------
    #  状态管理
    # ------------------------------------------------------------------

    def _init_states(self) -> None:
        self.states: list[dict] = []
        for _ in range(self.num_envs):
            self.states.append({
                "initialized": False,
                "prep_counter": 0,
                "q_init": None,           # preparation 初始关节位置
                "q_posture": None,        # 期望零空间标定姿态
                "q_ref": None,            # 虚拟解耦关节位置参考
                "x_des": None,            # 笛卡尔目标位姿 [pos(3), rotvec(3)]
                "x_ref": None,            # admittance 积分参考状态
                "v_ref": np.zeros(6, dtype=np.float32),
                "tau_smoothed": np.zeros(6, dtype=np.float32),
                "bias_smoothed": np.zeros(6, dtype=np.float32),
                # 接触法向 EMA 估计（初始化为默认法向）
                "contact_normal": self._default_contact_normal.copy(),
                # 接触后纯力控 PI 状态
                "force_error_integral_n": 0.0,   # 法向力误差积分（标量）
                "was_in_contact": False,          # 上一帧是否在接触中
                "approached": False,              # cart_err 曾经 < 阈值（真正接近过表面）
            })

    # ------------------------------------------------------------------
    #  主调用入口
    # ------------------------------------------------------------------

    def __call__(
        self,
        obs: dict[str, torch.Tensor],
        f_cmd: torch.Tensor | None = None,
        x_des: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """每控制周期调用一次，返回关节位置指令 (B, 22)。

        Args:
            obs:   观测字典
            f_cmd: 外部力指令 (B,3) 世界坐标系，None 则用内部默认
            x_des: 外部位置目标 (B,3) 世界坐标系，None 则用内部 x_des
        """
        policy_obs = obs["policy"]
        B = policy_obs.shape[0]
        if B != self.num_envs:
            self.num_envs = B
            self._init_states()

        self._step_count += 1

        # 从 policy 观测中提取所需物理量 (CPU)
        joint_pos_all = policy_obs[:, 0:22].cpu().numpy().astype(np.float32)
        qfrc_actuator_arm = policy_obs[:, 28:34].cpu().numpy().astype(np.float32)

        # 外部 x_des 覆盖：(B,3)=仅位置, (B,6)=位置+姿态
        if x_des is not None:
            x_des_np = x_des.cpu().numpy().astype(np.float32)
            if x_des_np.ndim == 1:
                x_des_np = x_des_np.reshape(1, -1)
            if x_des_np.shape[1] == 3:
                x_des_np = np.pad(x_des_np, ((0,0),(0,3)), constant_values=0.0)
            B_des = x_des_np.shape[0]
            if B_des == 1 and B > 1:
                x_des_np = np.tile(x_des_np, (B, 1))
            _use_external_x_des = True
        else:
            x_des_np = np.zeros((B, 6), dtype=np.float32)
            _use_external_x_des = False

        if f_cmd is not None:
            f_cmd_np = f_cmd.cpu().numpy().astype(np.float32)
            if f_cmd_np.ndim == 1:
                f_cmd_np = np.tile(f_cmd_np.reshape(1, 3), (B, 1))
            _use_explicit_f_cmd = True
        else:
            # placeholder，每环境根据 contact_normal 动态构建
            f_cmd_np = np.zeros((B, 3), dtype=np.float32)
            _use_explicit_f_cmd = False

        output = np.zeros_like(joint_pos_all)

        for i in range(B):
            state = self.states[i]
            qpos_full = joint_pos_all[i]

            # ---- Step 1: 物理同步 (等价于 MCC Controller.step 中的 sync_qpos) ----
            bias, jp, jr, palm_pos, palm_rotmat = self._sync_observer(qpos_full)

            # ---- Step 2: Preparation 阶段 ----
            if state["prep_counter"] < self.prep_steps:
                if state["prep_counter"] == 0:
                    state["q_init"] = qpos_full[:6].copy()
                    state["q_posture"] = np.array(
                        [0.0, 1.183, -1.541, 3.1415, 2.742, -1.569],
                        dtype=np.float32,
                    )
                state["prep_counter"] += 1
                t = state["prep_counter"] / self.prep_steps
                output[i, :6] = (
                    (1.0 - t) * state["q_init"] + t * state["q_posture"]
                )
                output[i, 6:] = qpos_full[6:]
                continue

            # ---- Step 3: Alignment 阶段（首次初始化状态）----
            if not state["initialized"]:
                state["q_ref"] = qpos_full[:6].copy()
                state["q_posture"] = qpos_full[:6].copy()
                state["tau_smoothed"] = qfrc_actuator_arm[i].copy()
                state["bias_smoothed"] = bias.copy()

                palm_rotvec = (
                    R.from_matrix(palm_rotmat).as_rotvec().astype(np.float32)
                )
                state["x_des"] = np.concatenate([palm_pos, palm_rotvec])
                state["x_ref"] = state["x_des"].copy()
                state["v_ref"] = np.zeros(6, dtype=np.float32)
                state["initialized"] = True

            # ---- Step 3.5: 外部 x_des 覆盖 ----
            if _use_external_x_des:
                state["x_des"][:3] = x_des_np[i, :3]
                # 仅在外部传入 6D 时覆盖姿态（rotvec 非全零则生效）
                if not np.allclose(x_des_np[i, 3:6], 0.0):
                    state["x_des"][3:6] = x_des_np[i, 3:6]

            # ---- Step 4: 力矩 EMA 滤波（等价于 MCC _smooth_motor_torques）----
            state["tau_smoothed"] = (
                self.alpha_tau * qfrc_actuator_arm[i]
                + (1.0 - self.alpha_tau) * state["tau_smoothed"]
            )
            state["bias_smoothed"] = (
                self.alpha_tau * bias
                + (1.0 - self.alpha_tau) * state["bias_smoothed"]
            )
            # MCC 符号约定: tau_ext = -(tau_raw - bias)
            tau_ext = -(state["tau_smoothed"] - state["bias_smoothed"])

            # ---- Step 5: MCC wrench estimation（核心：estimate_wrench）----
            wrench = estimate_wrench(
                jp, jr, tau_ext, palm_rotmat, self._wrench_config,
            )
            f_ext = wrench[:3].astype(np.float32)
            tau_ext_wrench = wrench[3:6].astype(np.float32)

            # ---- Step 6: 各向异性 Admittance（法向力控 + 切向位置跟踪，对齐 MCC）----
            # 6a. 估计接触法向（外力方向 EMA 平滑；无接触时用手掌自身朝向）
            f_ext_norm = float(np.linalg.norm(f_ext))
            if f_ext_norm > self.contact_threshold:
                raw_normal = f_ext / f_ext_norm
            else:
                # 手掌局部 normal_axis 在世界系的朝向（每步随姿态更新）
                raw_normal = palm_rotmat[:, self._default_normal_idx].astype(np.float32)
            state["contact_normal"] = (
                self.alpha_normal * raw_normal
                + (1.0 - self.alpha_normal) * state["contact_normal"]
            )
            n = state["contact_normal"] / np.linalg.norm(state["contact_normal"])

            # 6b. 构建刚度/阻尼矩阵（接近原样，接触后法向切 K=0 + 显式阻尼）
            nnT = np.outer(n, n)
            I3 = np.eye(3, dtype=np.float32)
            # 如果 cart_err 曾小于阈值，说明真正接近过表面（防初始力读数误触发 PI）
            cart_err_now = float(np.linalg.norm(state["x_ref"][:3] - palm_pos))
            if cart_err_now < 0.02:
                state["approached"] = True
            in_contact = (f_ext_norm > self.f_desired_normal) and state["approached"]
            if in_contact:
                # 接触：法向 K=0（纯力控，无弹簧回拉） + 显式阻尼
                K_normal = 0.0
                kp_pos_dyn = (self.K_position * (I3 - nnT)).astype(np.float32)
                kd_tangent = get_damping_matrix(
                    self.K_position * (I3 - nnT), I3 * self.mass
                )
                kd_pos_dyn = (kd_tangent + self._kd_normal * nnT).astype(np.float32)
            else:
                # 接近：原始行为不变
                K_normal = self.K_force
                kp_pos_dyn = (
                    self.K_position * I3 + (K_normal - self.K_position) * nnT
                ).astype(np.float32)
                kd_pos_dyn = get_damping_matrix(kp_pos_dyn, I3 * self.mass)

            # 6c. 构建 f_cmd：用户显式传入 > 法向标量力 > 默认零
            # 注意: f_cmd 指向 -n (推入表面)，因为 n 是表面外法向 (f_ext 方向)
            if _use_explicit_f_cmd:
                f_cmd_i = f_cmd_np[i]
            elif self.f_desired_normal != 0.0:
                f_cmd_i = -self.f_desired_normal * n
            else:
                f_cmd_i = self.f_cmd_default

            # 6d. 平动 admittance 积分
            pos_prev = state["x_ref"][:3]
            vel_prev = state["v_ref"][:3]
            pos_des = state["x_des"][:3]
            pos_error = pos_des - pos_prev

            kp_term = kp_pos_dyn @ pos_error
            kd_term = kd_pos_dyn @ vel_prev
            # 接近阶段仅用 f_cmd_i 前馈（避免微弱预接触力扰动导致螺旋）
            # 接触后加入 f_ext 实现力控闭环
            f_net = (f_ext + f_cmd_i) if in_contact else f_cmd_i
            lin_acc = (kp_term - kd_term + f_net) / self.mass
            vel_next = vel_prev + lin_acc * self.control_dt
            pos_next = pos_prev + vel_next * self.control_dt

            # 转动 (rotation vector 几何代数更新)
            ori_prev = R.from_rotvec(state["x_ref"][3:6])
            omega_prev = state["v_ref"][3:6]
            ori_des = R.from_rotvec(state["x_des"][3:6])
            ori_error = (ori_des * ori_prev.inv()).as_rotvec().astype(np.float32)

            kp_term_rot = self.kp_rot @ ori_error
            kd_term_rot = self.kd_rot @ omega_prev
            ang_acc = (
                (kp_term_rot - kd_term_rot + tau_ext_wrench) / self.inertia_diag
            )
            omega_next = omega_prev + ang_acc * self.control_dt
            ori_next = (
                R.from_rotvec(omega_next * self.control_dt) * ori_prev
            ).as_rotvec().astype(np.float32)

            state["x_ref"][:3] = pos_next
            state["x_ref"][3:6] = ori_next
            state["v_ref"][:3] = vel_next
            state["v_ref"][3:6] = omega_next

            # ---- Step 7: DLS-IK + Nullspace（接触后法向切 PI 力控）----
            e_rot = (
                R.from_rotvec(state["x_ref"][3:6])
                * R.from_matrix(palm_rotmat).inv()
            ).as_rotvec().astype(np.float32)

            if in_contact:
                # 接触：法向由 PI 力控接管，切向保持 admittance 位置跟踪
                e_pos_full = state["x_ref"][:3] - palm_pos
                e_pos_n = float(np.dot(e_pos_full, n))
                e_pos_t = e_pos_full - e_pos_n * n

                # 法向力反馈 PI
                f_ext_n = float(np.dot(f_ext, n))
                f_err_n = f_ext_n - self.f_desired_normal
                if not state["was_in_contact"]:
                    state["force_error_integral_n"] = 0.0  # 首帧复位
                state["force_error_integral_n"] += f_err_n * self.control_dt
                state["force_error_integral_n"] = float(np.clip(
                    state["force_error_integral_n"],
                    -self._force_int_max_n, self._force_int_max_n,
                ))
                vel_force_n = (
                    self.Kf_vel * f_err_n
                    + self.Kif_vel * state["force_error_integral_n"]
                )
                dx_force_n = n * (vel_force_n * self.control_dt)
                dx_pos = e_pos_t * self.Kp_task * self.control_dt + dx_force_n
            else:
                # 接近：原始 3D 位置跟踪，PI 静默
                e_pos = state["x_ref"][:3] - palm_pos
                dx_pos = e_pos * self.Kp_task * self.control_dt
            state["was_in_contact"] = in_contact

            dx_task = np.concatenate([dx_pos, e_rot * self.Kp_task * self.control_dt])

            J = np.vstack([jp, jr]).astype(np.float32)
            A_dls = J @ J.T + self.dls_lambda * np.eye(6, dtype=np.float32)
            J_pinv = J.T @ np.linalg.inv(A_dls)
            dq_primary = J_pinv @ dx_task

            # Nullspace Posture（在真实 Jacobian 零空间做）
            N_space = np.eye(6, dtype=np.float32) - J_pinv @ J
            dq_posture = self.k_posture * (state["q_posture"] - state["q_ref"])
            dq_null = N_space @ dq_posture

            state["q_ref"] = np.clip(
                state["q_ref"] + dq_primary + dq_null,
                self._arm_jnt_ranges[:, 0],
                self._arm_jnt_ranges[:, 1],
            )

            # ---- Step 8: 输出 ----
            output[i, :6] = state["q_ref"]
            output[i, 6:] = qpos_full[6:]

            # ---- 周期日志 ----
            if self._step_count % 300 == 0 and i == 0:
                f_norm = float(np.linalg.norm(f_ext))
                dz_ref = float(state["x_ref"][2] - state["x_des"][2])  # x_ref 相对 x_des 的 Z 偏移
                dz_palm = float(palm_pos[2] - state["x_des"][2])  # palm 相对 x_des 的 Z 偏移
                cart_err = float(np.linalg.norm(state["x_ref"][:3] - palm_pos))
                dq_norm = float(np.linalg.norm(dq_primary))
                dq_total_norm = float(np.linalg.norm(dq_primary + dq_null))
                dq_null_norm = float(np.linalg.norm(dq_null))
                q_ref_drift = float(np.linalg.norm(state["q_ref"] - state["q_posture"]))
                q_actual0 = float(qpos_full[0])
                q_ref0 = float(state["q_ref"][0])
                f_ext_n_log = float(np.dot(f_ext, n))
                f_err_n_log = f_ext_n_log - self.f_desired_normal
                print(
                    f"[MCC-Palm] Step={self._step_count} | "
                    f"|F_ext|={f_norm:.2f}N F_ext_n={f_ext_n_log:+.2f}N err={f_err_n_log:+.2f}N | "
                    f"∫err={state['force_error_integral_n']:+.2f} {'PI' if in_contact else 'POS'} | "
                    f"|dq|={dq_norm:.4f} |dq_tot|={dq_total_norm:.4f} |dq_null|={dq_null_norm:.4f} "
                    f"drift={q_ref_drift:.3f} | "
                    f"q_ref0={q_ref0:.3f} q_act0={q_actual0:.3f} | "
                    f"dz_ref={dz_ref:+.4f} dz_palm={dz_palm:+.4f} | "
                    f"Cart_Err={cart_err:.4f}m"
                )

        return torch.as_tensor(output, device=self.device, dtype=torch.float32)




class NullComplianceController:
    """零输出占位控制器，用于对比测试。"""

    def __init__(self, device: str, num_envs: int, **kwargs):
        self.device = device

    def __call__(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size = obs["policy"].shape[0]
        return torch.zeros((batch_size, 22), device=self.device)


# ==============================================================================
#  RL Config
# ==============================================================================

@dataclass
class MCCPalmControlCfg(RslRlOnPolicyRunnerCfg):
    seed: int = 42
    device: str = "cuda:0"
    policy_class: type = MCCPalmComplianceController
    amplitude: float = 0.5
    # ── 可通过 RL cfg 透传的控制器参数 ──
    control_dt: float = 0.01
    mass_trans: float = 1.0
    inertia_diag: tuple = (0.1, 0.1, 0.1)
    # ── 各向异性刚度 ──
    K_force: float = 20.0        # 接触法向刚度（低→力控主导）
    K_position: float = 200.0    # 切向刚度（高→位置跟踪）
    K_rot: float = 30.0
    normal_axis: str = "z"       # 无接触时的默认法向
    # ── 力控 ──
    f_desired_normal: float = 5.0  # 期望法向力 (N)，0=纯位置伺服
    f_cmd_default: tuple = (0.0, 0.0, 0.0)
    # ── 接触后纯力控 PI（仅接触时激活，接近阶段保持原样）──
    Kf_vel: float = 0.03           # 力误差→法向速度 P 增益 (m/s/N)
    Kif_vel: float = 0.002          # 力误差→法向速度 I 增益
    force_int_max_n: float = 5.0    # 法向力积分抗饱和 (±N·s)
    kd_normal: float = 80.0         # 接触法向显式阻尼 N·s/m（K=0 时代替临界阻尼）
    # ── 接触法向估计 ──
    contact_threshold: float = 4.0  # |F_ext| 阈值
    alpha_normal: float = 0.3       # 法向 EMA 平滑
    # ── 力矩估计与滤波 ──
    alpha_tau: float = 0.3
    lambda_force: float = 1e-3
    lambda_torque: float = 1e-2
    # ── IK 跟踪 ──
    Kp_task: float = 2.0
    dls_lambda: float = 0.1
    k_posture: float = 0
    # ── 其他 ──
    prep_duration_s: float = 1.5
