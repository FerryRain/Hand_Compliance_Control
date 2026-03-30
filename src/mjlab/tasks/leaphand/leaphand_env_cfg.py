from __future__ import annotations

import re
from pathlib import Path

import mujoco
import numpy as np
import torch
from dataclasses import dataclass

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityCfg
from mjlab.entity.entity import EntityArticulationInfoCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg, JointRelativePositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

_LEAPHAND_XML = Path("/home/rimlab/Code/MojocoLab/mjlab/src/mjlab/asset_zoo/robots/xarm6_leap_hand/xarm6_leap_hand.xml")
_ENABLE_HAND_OBJECT_ONLY_COLLISION = False
_HAND_CONTYPE = 2
_HAND_CONAFFINITY = 4
_OBJECT_CONTYPE = 4
_OBJECT_CONAFFINITY = 2

_FSR_CACHE = {}
_FSR_COLOR_FIELDS_READY = set()
_FSR_INDEX_LOGGED = set()
_THUMB_ROOT_JOINT_IDX_CACHE = {}


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
    fsr_regex: str = r".*_fsr_geom$",
    contact_rgba: tuple[float, float, float, float] = (0.2, 1.0, 0.2, 0.9),
    default_rgba: tuple[float, float, float, float] = (1.0, 0.2, 0.2, 0.9),
    display_forces: bool = True,
    display_every: int = 5,
    display_top_k: int = 8,
) -> torch.Tensor:
    """
    计算 FSR 受力并实时改变颜色。
    返回: [num_envs, num_fsrs] 的受力张量。
    """
    m = env.sim.mj_model
    d = env.sim.mj_data
    sim_data = env.sim.data

    env_ptr = id(env)
    if env_ptr not in _FSR_COLOR_FIELDS_READY:
        # Native viewer only syncs visual model fields per-env when the field is
        # expanded in sim.model (otherwise v.sync uses state_only=True).
        env.sim.expand_model_fields(("geom_rgba",))
        _FSR_COLOR_FIELDS_READY.add(env_ptr)

    # Keep CPU mjData in sync with the current sim state so contact queries are valid.
    # This term is currently used with num_envs=1.
    d.qpos[:] = sim_data.qpos[0].cpu().numpy()
    d.qvel[:] = sim_data.qvel[0].cpu().numpy()
    if m.nu > 0:
        d.ctrl[:] = sim_data.ctrl[0].cpu().numpy()
    if m.nmocap > 0:
        d.mocap_pos[:] = sim_data.mocap_pos[0].cpu().numpy()
        d.mocap_quat[:] = sim_data.mocap_quat[0].cpu().numpy()
    mujoco.mj_forward(m, d)
    
    if env_ptr not in _FSR_CACHE:
        pattern = re.compile(fsr_regex)
        _FSR_CACHE[env_ptr] = [
            i
            for i in range(m.ngeom)
            if (name := mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i))
            and pattern.match(name)
        ]
    
    fsr_ids = _FSR_CACHE[env_ptr]

    if env_ptr not in _FSR_INDEX_LOGGED:
        fsr_names = [
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid) or "<unnamed>"
            for gid in fsr_ids
        ]
        rows = [
            (idx, name, gid)
            for idx, (name, gid) in enumerate(zip(fsr_names, fsr_ids, strict=False))
        ]

        idx_w = max(len("idx"), max((len(str(idx)) for idx, _, _ in rows), default=1))
        name_w = max(len("fsr_geom"), max((len(name) for _, name, _ in rows), default=1))
        gid_w = max(len("geom_id"), max((len(str(gid)) for _, _, gid in rows), default=1))

        sep = f"+{'-' * (idx_w + 2)}+{'-' * (name_w + 2)}+{'-' * (gid_w + 2)}+"
        header = (
            f"| {'idx'.rjust(idx_w)} | {'fsr_geom'.ljust(name_w)} | "
            f"{'geom_id'.rjust(gid_w)} |"
        )

        print("[INFO] fsr_forces index mapping")
        print(sep)
        print(header)
        print(sep)
        for idx, name, gid in rows:
            print(
                f"| {str(idx).rjust(idx_w)} | {name.ljust(name_w)} | "
                f"{str(gid).rjust(gid_w)} |"
            )
        print(sep)
        _FSR_INDEX_LOGGED.add(env_ptr)

    fsr_index_by_gid = {gid: idx for idx, gid in enumerate(fsr_ids)}
    num_fsrs = len(fsr_ids)
    forces_tensor = torch.zeros((env.num_envs, num_fsrs), device=env.device)
    
    # 2. 计算力并变色
    active_gids = set()
    for i in range(d.ncon):
        con = d.contact[i]
        hit_gid = con.geom1 if con.geom1 in fsr_index_by_gid else con.geom2
        if hit_gid in fsr_index_by_gid:
            idx = fsr_index_by_gid[hit_gid]
            c_force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(m, d, i, c_force)
            # 转换类型以适配 mjlab/Pylance
            force_val = float(np.linalg.norm(c_force[:3]))
            forces_tensor[:, idx] += force_val # 简化：假设单环境
            active_gids.add(hit_gid)

    sim_geom_rgba = env.sim.model.geom_rgba
    contact_rgba_t = torch.tensor(
        contact_rgba, device=env.device, dtype=sim_geom_rgba.dtype
    )
    default_rgba_t = torch.tensor(
        default_rgba, device=env.device, dtype=sim_geom_rgba.dtype
    )

    for gid in fsr_ids:
        if gid in active_gids:
            sim_geom_rgba[0, gid] = contact_rgba_t
        else:
            sim_geom_rgba[0, gid] = default_rgba_t

    # Real-time console monitor for FSR forces (single-line refresh).
    if display_forces and env.num_envs > 0 and num_fsrs > 0:
        step = getattr(env, "_fsr_display_step", 0)
        every = max(1, int(display_every))
        if step % every == 0:
            vals = forces_tensor[0].detach().cpu()
            k = min(max(1, int(display_top_k)), num_fsrs)
            top_vals, top_idxs = torch.topk(vals, k=k)
            parts = [
                f"{int(idx):02d}:{float(val):.2f}"
                for idx, val in zip(top_idxs.tolist(), top_vals.tolist(), strict=False)
            ]
            print(
                "\r[FSR live] top "
                f"{k} | "
                + "  ".join(parts),
                end="",
                flush=True,
            )
        setattr(env, "_fsr_display_step", step + 1)

    return forces_tensor

def joint_pos(
    env: ManagerBasedRlEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """获取机器人的绝对关节位置 [num_envs, num_joints]"""
    asset = env.scene[asset_cfg.name]

    return asset.data.joint_pos

# --- Entity 配置 ---

def _get_target_box_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    # Mocap body is kinematic: mouse perturbation can place it directly.
    body = spec.worldbody.add_body(name="target_ball", mocap=True)
    ball = body.add_geom(
        name="ball_geom",
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        size=[0.08, 0.04],
        rgba=[0.2, 0.6, 1.0, 1.0],
        mass=1,
    )
    if _ENABLE_HAND_OBJECT_ONLY_COLLISION:
        ball.contype = _OBJECT_CONTYPE
        ball.conaffinity = _OBJECT_CONAFFINITY
    return spec

# --- 环境配置构建 ---

def _make_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    # hand
    robot_cfg = EntityCfg(
        spec_fn=lambda: _load_leaphand_spec(
            enable_hand_object_only_collision=_ENABLE_HAND_OBJECT_ONLY_COLLISION
        ),
        articulation=EntityArticulationInfoCfg(
            actuators=(
                BuiltinPositionActuatorCfg(
                    # LeapHand joints are named "0".."15" in this XML.
                    target_names_expr=(r"^[0-9]+$",),
                    stiffness=20.0,
                    damping=2.0,
                    effort_limit=500.0,
                ),
            ),
        ),
        init_state=EntityCfg.InitialStateCfg(
            pos=(0, 0, 0),
            joint_pos={"13": 1.57},
        ),
    )
    
    # ball
    target_cfg = EntityCfg(
        spec_fn=_get_target_box_spec,
        init_state=EntityCfg.InitialStateCfg(pos=(0.65, 0.0, 0.35)),
    )

    # obervation 
    observations = {
        "policy": ObservationGroupCfg({
            "fsr_forces": ObservationTermCfg(
                func=fsr_force_and_visual_logic,
                params={
                    "fsr_regex": r".*_fsr_geom$",
                    "display_forces": True,
                    "display_every": 5,
                    "display_top_k": 8,
                },
            ),
            "joint_pos": ObservationTermCfg(
                func=joint_pos,
                params={"asset_cfg": SceneEntityCfg("robot")},
            ),
        })
    }

    actions: dict[str, ActionTermCfg] = {
        "hand_delta": JointRelativePositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=0.05, 
            use_default_offset=False
        )
    }
    

    return ManagerBasedRlEnvCfg(

        decimation=10,
        scene=SceneCfg(
            terrain=None,
            entities={"robot": robot_cfg, "target": target_cfg},
            num_envs=1,
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
        viewer=ViewerConfig(
            entity_name="robot",
            body_name="base_link", 
            distance=2.0,
        ),
        episode_length_s=1e10 if play else 50.0,
    )

def leaphand_contact_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    return _make_env_cfg(play=play)

class LeapHandComplianceController:
    """
    针对 22 维架构 (6臂+16手) 的底层顺应性控制器
    
    修改说明：
    1. 解锁逻辑：仅当手掌 (FSR 0-3) 检测到压力时，手指才启动顺应性调节。
    2. 拇指统一：拇指根部关节 (18, 19) 采用与其他手指相同的回缩/下压逻辑。
    """
    def __init__(self, device: str, num_envs: int, **kwargs):
        self.device = device
        self.num_envs = num_envs
        self.step_count = 0

        # --- 1. 核心控制参数 ---
        self.S_min = 0.6            # 舒适区间下限 (N)
        self.S_max = 1.5            # 舒适区间上限 (N)
        self.K_prox = 0.15          # 根部/近端增益
        self.K_mid  = 0.08          # 中段关节增益
        self.K_dist = 0.04          # 末端关节增益
        self.D_force = 0.03         # 力微分阻尼
        
        # --- 2. 安全锁与手掌参数 ---
        self.S_palm_threshold = 0.2     # 手掌解锁门限 (N)
        self.S_contact_threshold = 0.1  # 接触判定门限 (用于回弹逻辑)
        self.reset_speed = 0.1         # 未触发时的回弹速度

        # --- 3. 状态变量 ---
        self.prev_fsr = torch.zeros((num_envs, 16), device=device)
        self.q_nom = torch.zeros((num_envs, 22), device=device)
        self.is_init = False
        
        # --- 4. 传感器与关节拓扑映射 ---
        # 手掌 FSR 索引
        self.palm_fsr_idx = [0, 1, 2, 3]
        
        # 定义四根手指的统一结构: [根部, 中段, 末端] 关节映射
        # 这里假设拇指的 18, 19 对应根部回缩逻辑
        self.finger_configs = [
            {"name": "index",  "j": [6, 8, 9],    "p_fsr": [4, 5], "d_fsr": [6]},   
            {"name": "middle", "j": [10, 12, 13], "p_fsr": [7, 8], "d_fsr": [9]},   
            {"name": "ring",   "j": [14, 16, 17], "p_fsr": [10, 11], "d_fsr": [12]},
            {"name": "thumb",  "j": [18, 20, 21], "p_fsr": [13, 14], "d_fsr": [15]} # 拇指统一化
        ]
        # 注：拇指 19 轴若作为辅助旋转轴，可根据需要放入 j[0] 或单独处理，这里暂将其根部主轴放入 j[0]

    def _compute_interval_error(self, s):
        """区间误差计算: 低于 S_min 下压(+), 高于 S_max 回缩(-)"""
        error = torch.zeros_like(s)
        low_mask = s < self.S_min
        high_mask = s > self.S_max
        error[low_mask] = self.S_min - s[low_mask]
        error[high_mask] = self.S_max - s[high_mask]
        return error

    def __call__(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        policy_obs = obs["policy"]
        fsr = policy_obs[:, :16]
        q_curr = policy_obs[:, 16:38]
        
        if not self.is_init:
            self.q_nom[:] = q_curr.clone()
            self.is_init = True

        dot_fsr = (fsr - self.prev_fsr)
        self.prev_fsr = fsr.clone()

        # 最终补偿增量
        delta_comp = torch.zeros_like(q_curr)

        # --- A. 计算解锁状态 (基于手掌 4 个 FSR) ---
        # 计算手掌平均受力
        palm_force = torch.mean(fsr[:, self.palm_fsr_idx], dim=1)
        # 解锁掩码: [num_envs]
        is_unlocked = palm_force > self.S_palm_threshold

        # --- B. 遍历四指执行统一顺应逻辑 ---
        for config in self.finger_configs:
            j_idx = config["j"]
            # 提取该手指的力
            s_p = torch.mean(fsr[:, config["p_fsr"]], dim=1)  # 近端力
            s_d = fsr[:, config["d_fsr"]].squeeze(-1)        # 远端力
            ds_p = torch.mean(dot_fsr[:, config["p_fsr"]], dim=1)

            # 1. 根部关节顺应 (对应你要求的 18, 19 或其他手指根部)
            e_p = self._compute_interval_error(s_p)
            comp_p = self.K_prox * e_p - self.D_force * ds_p

            # 2. 中段与末端关节 (包裹逻辑)
            e_d = self._compute_interval_error(s_d)
            # 指尖力过大时的抬起分量
            wrapping_factor = torch.clamp(s_d - s_p, min=0)
            adj_e_d = e_d - 0.5 * wrapping_factor
            
            comp_m = self.K_mid * adj_e_d
            comp_d = self.K_dist * adj_e_d

            # 3. 应用安全锁与动作分配
            # 如果未解锁：手指缓慢回到名义位置
            # 如果已解锁：执行顺应性 delta
            for i, joint_idx in enumerate(j_idx):
                target_comp = 0.0
                if i == 0: target_comp = comp_p
                elif i == 1: target_comp = comp_m
                else: target_comp = comp_d

                # 使用 torch.where 根据解锁状态切换逻辑
                reset_delta = self.reset_speed * (self.q_nom[:, joint_idx] - q_curr[:, joint_idx])
                delta_comp[:, joint_idx] = torch.where(is_unlocked, target_comp, reset_delta)

        # 对于未在 finger_configs 中定义的关节 (如 19 轴等)，保持原位或执行回弹
        # 如果你希望 19 轴也跟随 18 轴运动，可以把 19 加入 config["j"]
        unused_hand_joints = [7, 11, 15, 19] # 假设这些是侧摆轴
        for uj in unused_hand_joints:
             delta_comp[:, uj] = self.reset_speed * (self.q_nom[:, uj] - q_curr[:, uj])

        # --- C. Action 合成 ---
        scale = 0.05
        action = delta_comp / scale
        
        # 返回手部的 16 个关节
        return torch.clamp(action[:, 6:], -1.0, 1.0)

class NullComplianceController:
    """一个不做任何补偿的控制器，用于对比测试"""
    def __init__(self, device: str, num_envs: int, **kwargs):
        pass

    def __call__(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        # 直接输出零增量
        batch_size = obs["policy"].shape[0]
        return torch.zeros((batch_size, 16))
    
@dataclass
class LeapHandControlCfg(RslRlOnPolicyRunnerCfg):
    seed: int = 42
    device: str = "cuda:0"
    """用于传递给采集脚本的配置"""
    # policy_class: type = NullComplianceController
    policy_class: type = LeapHandComplianceController
    amplitude: float = 0.5


'''
PYTHONPATH=src python -m mjlab.scripts.collect_data Leaphand-Contact-Relocation --collect False --viewer native
'''