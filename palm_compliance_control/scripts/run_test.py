from __future__ import annotations

from dataclasses import asdict, dataclass
import sys
from typing import Any, Literal

import mujoco
import numpy as np
import torch
import tyro
from scipy.spatial.transform import Rotation as R

# ── mjlab 核心功能与 GUI 视窗导入 ──
from mjlab.envs import ManagerBasedRlEnv, types as env_types
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

# ── Capsule 表面交点工具 ──
from mjlab.tasks.leaphand.leaphand_palm_mcc_env_cfg import capsule_surface_intersection


@dataclass
class CollectConfig:
    """物理控制与交互 GUI 视窗测试脚本参数配置"""
    num_envs: int = 1
    """并行物理仿真环境数量"""
    use_compliance: bool = True
    """是否激活注册任务中的底层主动柔顺控制器"""
    device: str | None = None
    """运行物理仿真的物理设备 (cuda 或 cpu)"""
    viewer: Literal["native", "viser"] = "native"
    """渲染窗口类型: native (MuJoCo 官方原生视窗) 或 viser (网页端可视)"""
    surface_track: bool = False
    """是否将 x_des 实时投影到 capsule 表面（手掌→物体射线，需要观测中包含 target_pos/rot）"""
    bottom_surface_target: bool = False
    """将 x_des 固定在物体正下方表面点（世界-Z 射线，采集数据用）"""
    surface_offset: float = 0.0
    """目标点沿接触方向内推的距离 (m)，正值=推入表面，增大接触力"""
    x_des_offset_x: float = 0.0
    """手动修正 x_des 的 X 方向偏移 (m)"""
    x_des_offset_y: float = 0.0
    """手动修正 x_des 的 Y 方向偏移 (m)"""
    use_fsr_center: bool = True
    """将手掌 FSR 中心（4个 palm FSR 均值）作为控制点，替代 palm_lower body 原点"""
    # ── 物体旋转/平移策略（采集数据用）──
    rotate_object: bool = False
    """是否让 capsule 物体绕随机轴持续旋转"""
    translate_object: bool = False
    """是否让 capsule 物体做正弦振荡平移"""
    rotation_speed: float = 0.3
    """物体旋转速度 (rad/s)"""
    translation_amplitude: float = 0.01
    """物体平移振幅 (m)，正弦振荡幅值"""
    resample_interval: int = 1000
    """每隔 N 步随机切换旋转轴和振荡参数（0=不切换）"""
    # ── 手指控制实验开关 ──
    match_single_finger_control: bool = True
    """在 run_test 中把 strict-finger 手指控制对齐到单独手指柔顺控制器"""
    finger_gain_scale_override: float = 1.0
    """覆盖 strict-finger 外层手指动作缩放；1.0 表示不额外缩放"""
    finger_smooth_alpha_override: float = 0.0
    """覆盖 strict-finger 外层 EMA；0.0 表示不做外层 EMA 平滑"""
    hand_effort_limit_override: float = 500.0
    """覆盖手指 actuator effort_limit，使其与单独手指环境一致"""


class NullComplianceController:
    """不做任何柔顺动作的全 0 占位物理控制器"""
    def __init__(self, device: str, num_envs: int, **kwargs):
        self.device = device

    def __call__(self, obs: env_types.VecEnvObs, **kwargs) -> torch.Tensor:
        # 获取当前 batch 大小
        policy_obs = obs["policy"]
        batch_size = policy_obs.shape[0] if isinstance(policy_obs, torch.Tensor) else next(iter(policy_obs.values())).shape[0]
        # 默认返回 22 维 (后续由 Adapter 自动匹配底层实际维度)
        return torch.zeros((batch_size, 22), device=self.device)


class MocapObjectRotator:
    """物体旋转+平移策略，参考 collect_data_headless.py"""

    def __init__(self, num_envs: int, device: str, dt: float,
                 mocap_idx: int, rotate: bool = True, translate: bool = True,
                 rotation_speed: float = 0.3, translation_amplitude: float = 0.01):
        self.num_envs = num_envs
        self.device = device
        self.dt = dt
        self.mocap_idx = mocap_idx
        self.rotate = rotate
        self.translate = translate
        self._rotation_speed = rotation_speed
        self._translation_amplitude = translation_amplitude
        self._step_count = 0

        # 旋转参数
        self.axes = torch.zeros((num_envs, 3), device=device)
        self.speeds = torch.zeros(num_envs, device=device)

        # 平移正弦振荡参数
        self.trans_amp = torch.zeros((num_envs, 3), device=device)
        self.trans_freq = torch.zeros((num_envs, 3), device=device)
        self.trans_phase = torch.zeros((num_envs, 3), device=device)
        self.trans_base = torch.zeros((num_envs, 3), device=device)

        self.resample_axes()

    def resample_axes(self):
        """随机切换旋转轴、转速和振荡参数"""
        if self.rotate:
            axes = torch.randn((self.num_envs, 3), device=self.device)
            self.axes = axes / torch.norm(axes, dim=-1, keepdim=True)
            self.speeds = (
                torch.rand(self.num_envs, device=self.device) * self._rotation_speed * 0.7
                + self._rotation_speed * 0.3
            )

        if self.translate:
            amp = self._translation_amplitude
            self.trans_amp = torch.rand(self.num_envs, 3, device=self.device) * amp * 0.8 + amp * 0.2
            self.trans_freq = torch.rand(self.num_envs, 3, device=self.device) * 0.3 + 0.1
            self.trans_phase = (
                torch.rand(self.num_envs, 3, device=self.device) * 2 * np.pi
            )
        self._step_count = 0

    def set_base_position(self, pos: torch.Tensor):
        """记录振荡中心（当前物体位置）"""
        self.trans_base = pos.clone()

    def step(self, env: ManagerBasedRlEnv):
        """执行一步旋转+平移，直接写入 mocap"""
        self._step_count += 1
        t_sec = self._step_count * self.dt

        if self.rotate:
            current_quat = env.sim.data.mocap_quat[:, self.mocap_idx, :].clone()
            theta = self.speeds * self.dt
            cos_t = torch.cos(theta / 2).unsqueeze(-1)
            sin_t = torch.sin(theta / 2).unsqueeze(-1)
            delta_quat = torch.cat([cos_t, self.axes * sin_t], dim=-1)
            w1, x1, y1, z1 = current_quat.unbind(-1)
            w2, x2, y2, z2 = delta_quat.unbind(-1)
            new_quat = torch.stack([
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ], dim=-1)
            env.sim.data.mocap_quat[:, self.mocap_idx, :] = (
                new_quat / torch.norm(new_quat, dim=-1, keepdim=True)
            )

        if self.translate:
            osc = torch.sin(
                2.0 * np.pi * self.trans_freq * t_sec + self.trans_phase
            )
            offset = self.trans_amp * osc
            env.sim.data.mocap_pos[:, self.mocap_idx, :] = self.trans_base + offset


def _adapt_action_dim(action: torch.Tensor, target_dim: int) -> torch.Tensor:
    """【维度自适应】：根据 ActionManager 的实际需求，自动对 action 维度进行截断或零填充"""
    current_dim = int(action.shape[-1])
    if current_dim == target_dim:
        return action
    if target_dim == 0:
        return action.new_zeros((action.shape[0], 0))
    if current_dim > target_dim:
        return action[:, :target_dim]
    # 当前维度小于目标维度，补 0
    pad = action.new_zeros((action.shape[0], target_dim - current_dim))
    return torch.cat((action, pad), dim=-1)


def _policy_obs_tensor(obs: env_types.VecEnvObs) -> torch.Tensor:
    """从统一观测对象中解析出 policy 特征张量。

    支持单组任务 (obs["policy"]) 和多组任务 (obs["palm"] / obs["finger"])。
    """
    # 优先返回 "policy" 组（单组任务）
    if "policy" in obs:
        policy_obs = obs["policy"]
        if isinstance(policy_obs, torch.Tensor):
            return policy_obs
        if "fsr_forces" in policy_obs:
            return policy_obs["fsr_forces"]
        if len(policy_obs) == 1:
            return next(iter(policy_obs.values()))
        keys = ", ".join(sorted(policy_obs.keys()))
        raise ValueError(f"无法自动推断 policy tensor。当前包含项: {keys}")

    # 多组任务：返回 "palm" 组用于显示维度（与旧 palm 布局兼容）
    if "palm" in obs:
        return obs["palm"] # type: ignore

    # 兜底：取第一个可用的组
    for key, val in obs.items():
        if isinstance(val, torch.Tensor):
            return val
    raise ValueError(f"观测对象中无可用的张量组。键: {list(obs.keys())}")


def _log_observation_action_dims(env: ManagerBasedRlEnv, obs: env_types.VecEnvObs) -> int:
    policy_obs = _policy_obs_tensor(obs)
    action_dim = env.action_manager.total_action_dim
    print(f"[INFO] 任务观测特征维度: {int(policy_obs.shape[-1])}")
    print(f"[INFO] 任务动作空间维度: {action_dim}")
    return action_dim


_SINGLE_FINGER_CONTROLLER_PARAMS = {
    "S_min": 0.6,
    "S_max": 1.5,
    "K_prox": 1.2,
    "K_mid": 0.5,
    "K_dist": 0.35,
    "D_force": 1.8,
    "K_limit_spring": 0.3,
    "q_pre_grasp_list": [0.8, 0.4, 0.3],
    "S_contact_threshold": 0.15,
    "reset_speed": 0.1,
    "alpha_obs": 0.4,
    "alpha_ctrl": 0.15,
    "contact_on_threshold": 0.20,
    "contact_off_threshold": 0.12,
    "error_deadband": 0.03,
    "ds_clip": 0.2,
    "action_rate_limit": 0.15,
}


def _match_single_finger_env_params(env_cfg: Any, cfg: CollectConfig) -> None:
    """Make run_test's combined hand actuator limits match the standalone finger env."""
    if not cfg.match_single_finger_control:
        return
    try:
        robot_cfg = env_cfg.scene.entities["robot"]
        actuators = robot_cfg.articulation.actuators
    except Exception as exc:
        print(f"[WARN] 无法访问 robot actuator 配置，跳过 effort_limit 对齐: {exc}")
        return

    patched = 0
    for actuator_cfg in actuators:
        target_expr = getattr(actuator_cfg, "target_names_expr", ())
        if target_expr == (r"^[0-9]+$",):
            old = getattr(actuator_cfg, "effort_limit", None)
            setattr(actuator_cfg, "effort_limit", cfg.hand_effort_limit_override)
            patched += 1
            print(
                f"[INFO] 手指 actuator effort_limit 对齐单独手指环境: "
                f"{old} -> {cfg.hand_effort_limit_override}"
            )
    if patched == 0:
        print("[WARN] 未找到手指 actuator (target_names_expr='^[0-9]+$')，effort_limit 未修改")


def _match_single_finger_policy_params(policy: Any, cfg: CollectConfig) -> None:
    """Remove strict-finger outer scaling/EMA and align inner finger controller gains."""
    if not cfg.match_single_finger_control:
        return

    # MCCPalmStrictController(enable_finger_control=True) uses these private attrs.
    if hasattr(policy, "_finger_gain_scale"):
        old = getattr(policy, "_finger_gain_scale")
        setattr(policy, "_finger_gain_scale", cfg.finger_gain_scale_override)
        print(f"[INFO] strict-finger 外层 finger_gain_scale: {old} -> {cfg.finger_gain_scale_override}")
    if hasattr(policy, "_finger_smooth_alpha"):
        old = getattr(policy, "_finger_smooth_alpha")
        setattr(policy, "_finger_smooth_alpha", cfg.finger_smooth_alpha_override)
        setattr(policy, "_finger_delta_ema", None)
        print(f"[INFO] strict-finger 外层 finger_smooth_alpha: {old} -> {cfg.finger_smooth_alpha_override}")

    # CombinedMCCFingerController path, kept robust for future use.
    if hasattr(policy, "finger_gain_scale"):
        old = getattr(policy, "finger_gain_scale")
        setattr(policy, "finger_gain_scale", cfg.finger_gain_scale_override)
        print(f"[INFO] combined 外层 finger_gain_scale: {old} -> {cfg.finger_gain_scale_override}")

    finger_ctrl = getattr(policy, "_finger_ctrl", None)
    if finger_ctrl is None:
        finger_ctrl = getattr(policy, "finger_controller", None)
    if finger_ctrl is None:
        print("[WARN] 当前 policy 未暴露手指控制器实例；仅完成外层参数对齐")
        return

    changed = []
    for name, value in _SINGLE_FINGER_CONTROLLER_PARAMS.items():
        if hasattr(finger_ctrl, name):
            old = getattr(finger_ctrl, name)
            setattr(finger_ctrl, name, value)
            changed.append(f"{name}: {old}->{value}")
    print(
        "[INFO] 手指控制器参数已对齐单独 LeapHandComplianceController: "
        + (", ".join(changed) if changed else "无可修改字段")
    )


def _build_registered_policy(task_id: str, cfg: CollectConfig, num_envs: int) -> Any:
    """纯动态从注册任务的 rl_cfg 中安全提取并实例化对应的 Compliance 控制器"""
    policy_cfg = load_rl_cfg(task_id)
    policy_class = getattr(policy_cfg, "policy_class", None)
    if policy_class is None:
        raise ValueError(f"任务 '{task_id}' 在其注册的 rl_cfg 中未配置 'policy_class'。")

    cfg_dict = asdict(policy_cfg)
    cfg_dict.pop("policy_class", None)
    cfg_dict.pop("device", None)

    policy_device = cfg.device or getattr(policy_cfg, "device", None) or ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 成功自适应载入注册物理控制器: {policy_class.__name__}")
    return policy_class(device=policy_device, num_envs=num_envs, **cfg_dict)


def run_collect(task_name: str, cfg: CollectConfig) -> None:
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 正在载入任务配置: {task_name}...")

    # 1. 动态加载环境配置与策略配置
    env_cfg = load_env_cfg(task_name, play=True)
    env_cfg.scene.num_envs = cfg.num_envs
    agent_cfg = load_rl_cfg(task_name)
    _match_single_finger_env_params(env_cfg, cfg)

    # 2. 实例化 ManagerBasedRlEnv 环境
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)

    # 3. 动态配置对应的底层物理控制器（全动态解析，零硬编码导入）
    if cfg.use_compliance:
        raw_policy = _build_registered_policy(task_name, cfg, env.num_envs)
        _match_single_finger_policy_params(raw_policy, cfg)
    else:
        raw_policy = NullComplianceController(device=device, num_envs=env.num_envs)

    # 4. 获取第一步状态并确定所需动作维度
    current_obs, _ = env.reset()
    action_dim = _log_observation_action_dims(env, current_obs)

    # 4.5 物体旋转/平移策略
    target_entity = env.scene["target"]
    target_mocap_idx = int(target_entity.data.indexing.mocap_id)

    object_rotator = None
    if cfg.rotate_object or cfg.translate_object:
        dt = env.sim.model.opt.timestep * env.cfg.decimation
        object_rotator = MocapObjectRotator(
            num_envs=env.num_envs, device=device, dt=dt,
            mocap_idx=target_mocap_idx,
            rotate=cfg.rotate_object,
            translate=cfg.translate_object,
            rotation_speed=cfg.rotation_speed,
            translation_amplitude=cfg.translation_amplitude,
        )
        object_rotator.set_base_position(
            env.sim.data.mocap_pos[:, target_mocap_idx, :].clone()
        )
        print(f"[INFO] 物体策略: 旋转={'ON' if cfg.rotate_object else 'OFF'} "
              f"(~{cfg.rotation_speed:.1f} rad/s) | "
              f"平移={'ON' if cfg.translate_object else 'OFF'} "
              f"(~{cfg.translation_amplitude*1000:.1f} mm)")

        # Monkey-patch env.step 注入 rotator + 周期重采样
        _original_step = env.step
        _resample_interval = cfg.resample_interval
        _step_counter = [0]  # mutable counter

        def _patched_step(action):
            object_rotator.step(env)
            _step_counter[0] += 1
            if _resample_interval > 0 and _step_counter[0] % _resample_interval == 0:
                object_rotator.resample_axes()
                object_rotator.set_base_position(
                    env.sim.data.mocap_pos[:, target_mocap_idx, :].clone()
                )
            return _original_step(action)

        env.step = _patched_step

    # 5. 按照统一格式封装环境为包装器
    viewer_env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))

    class PolicyWithActionAdapter:
        def __init__(self):
            self._surface_track = cfg.surface_track
            self._bottom_target = cfg.bottom_surface_target
            self._surface_offset = cfg.surface_offset
            self._x_des_offset = np.array(
                [cfg.x_des_offset_x, cfg.x_des_offset_y, 0.0],
                dtype=np.float32,
            )
            self._use_fsr_center = cfg.use_fsr_center
            # palm_3_fsr（中指下方）在 palm_lower 局部坐标系中的位置
            self._fsr_center_local = np.array(
                [-0.047, -0.037, -0.034], dtype=np.float32,
            )
            # 上帧手掌旋转矩阵（每 env），用于平滑方向追踪，防止 X/Y 轴突变翻转
            self._prev_ori_rotmat: list[np.ndarray | None] = []

        def __call__(self, obs):
            if self._surface_track:
                x_des = self._compute_surface_x_des(obs)
                action = raw_policy(obs, x_des=x_des)
            elif self._bottom_target:
                x_des = self._compute_bottom_surface_x_des(obs)
                action = raw_policy(obs, x_des=x_des)
            else:
                action = raw_policy(obs)

            return _adapt_action_dim(action, action_dim)

        def reset(self) -> None:
            self._prev_ori_rotmat = []
            if hasattr(raw_policy, "reset"):
                raw_policy.reset()

        def _smooth_orient_to_normal(
            self,
            env_idx: int,
            target_z: np.ndarray,
            current_rotmat: np.ndarray,
        ) -> np.ndarray:
            """从当前姿态最小旋转对齐法向，不生成 wrist-yaw 目标。"""
            _ = env_idx
            target_z = target_z / np.linalg.norm(target_z)
            source_rotmat = current_rotmat.astype(np.float64)
            source_z = source_rotmat[:, 2]
            dot_z = float(np.clip(np.dot(source_z, target_z), -1.0, 1.0))
            rot_axis = np.cross(source_z, target_z)
            axis_norm = float(np.linalg.norm(rot_axis))
            if axis_norm > 1e-8:
                rot_axis /= axis_norm
                angle = np.arctan2(axis_norm, dot_z)
                correction = R.from_rotvec(rot_axis * angle)
                new_rotmat = (
                    (correction * R.from_matrix(source_rotmat))
                    .as_matrix()
                    .astype(np.float32)
                )
            elif dot_z < 0.0:
                tangent_axis = source_rotmat[:, 0]
                correction = R.from_rotvec(tangent_axis * np.pi)
                new_rotmat = (
                    (correction * R.from_matrix(source_rotmat))
                    .as_matrix()
                    .astype(np.float32)
                )
            else:
                new_rotmat = source_rotmat.astype(np.float32, copy=True)
            return new_rotmat

        def _compute_surface_x_des(self, obs: dict) -> torch.Tensor:
            """从观测中计算 capsule 表面交点 + 手掌朝向目标 (B,6)。

            支持单组 obs["policy"] 和多组 obs["palm"] 两种布局。
            """
            # 读取手掌相关观测 — 优先 palm 组（多组任务），回退 policy 组
            if "palm" in obs:
                palm_obs = obs["palm"]
            elif "policy" in obs:
                palm_obs = obs["policy"]
            else:
                palm_obs = _policy_obs_tensor(obs)

            # 布局: 76:79 palm_pos, 79:82 palm_rot, 82:85 target_pos, 85:88 target_rot
            palm_pos = palm_obs[:, 76:79].cpu().numpy().astype(np.float32)
            target_pos = palm_obs[:, 82:85].cpu().numpy().astype(np.float32)
            target_rotvec = palm_obs[:, 85:88].cpu().numpy().astype(np.float32)

            B = palm_pos.shape[0]
            x_des = np.zeros((B, 6), dtype=np.float32)
            for i in range(B):
                t_rotmat = R.from_rotvec(target_rotvec[i]).as_matrix().astype(np.float32)
                surf_pt = capsule_surface_intersection(
                    center=target_pos[i],
                    rotmat=t_rotmat,
                    radius=0.15,
                    half_height=0.08,
                    point=palm_pos[i],
                )
                x_des[i, :3] = surf_pt

                # 姿态：手掌 local Z 指向表面法向（surf → palm，即接触法向）
                approach = surf_pt - palm_pos[i]
                d = float(np.linalg.norm(approach))
                if d > 1e-6:
                    target_z = -approach / d  # 反转：palm local Z 指向 palm→surf 的反方向
                    current_rotmat = R.from_rotvec(
                        palm_obs[i, 79:82].cpu().numpy().astype(np.float64)
                    ).as_matrix()
                    rotmat = self._smooth_orient_to_normal(
                        i, target_z, current_rotmat,
                    )
                    x_des[i, 3:6] = R.from_matrix(rotmat).as_rotvec().astype(np.float32)

            return torch.tensor(x_des, device=palm_obs.device)

        def _compute_bottom_surface_x_des(self, obs: dict) -> torch.Tensor:
            """通用射线追踪：从物体下方打射线，命中任意几何体的底面。

            使用 MuJoCo mj_ray (flg_static=1)，适用于任意 static 几何体。
            手掌 local Z 对齐表面外法向（世界 +Z，底面朝上）。
            """
            if "palm" in obs:
                palm_obs = obs["palm"]
            elif "policy" in obs:
                palm_obs = obs["policy"]
            else:
                palm_obs = _policy_obs_tensor(obs)

            # 布局: 0:22 joint_pos
            joint_pos = palm_obs[:, 0:22].cpu().numpy().astype(np.float64)
            palm_rotvec = palm_obs[:, 79:82].cpu().numpy().astype(np.float64)

            m = env.sim.mj_model
            d = env.sim.mj_data

            # 同步 mocap 到 mj_data（warp → CPU），否则 target body 位置不准
            mocap_pos_np = env.sim.data.mocap_pos[:, target_mocap_idx, :].cpu().numpy()
            mocap_quat_np = env.sim.data.mocap_quat[:, target_mocap_idx, :].cpu().numpy()

            B = joint_pos.shape[0]
            x_des = np.zeros((B, 6), dtype=np.float32)

            for i in range(B):
                # 同步 mj_data：机械臂关节 + 物体 mocap
                d.qpos[:] = joint_pos[i]
                d.mocap_pos[target_mocap_idx] = mocap_pos_np[i]
                d.mocap_quat[target_mocap_idx] = mocap_quat_np[i]
                mujoco.mj_forward(m, d)

                # 射线从物体中心向世界 -Z 发射 → 命中底面
                center = d.xpos[
                    mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "target_ball")
                ].copy().astype(np.float64)
                ray_start = center.astype(np.float64)
                ray_dir = np.array([0.0, 0.0, -1.0], dtype=np.float64)
                geomid_out = np.zeros(1, dtype=np.int32)
                normal_out = np.zeros(3, dtype=np.float64)

                dist = mujoco.mj_ray(m, d, ray_start, ray_dir, None,
                                     1,     # flg_static=1: 只检测 static 几何体
                                     -1,    # bodyexclude: 不排除
                                     geomid_out,
                                     normal_out)

                if geomid_out[0] >= 0 and dist > 0:
                    surf_pt = (ray_start + ray_dir * dist).astype(np.float32)
                    surface_normal = normal_out.astype(np.float32)
                else:
                    surf_pt = (center + np.array([0.0, 0.0, -0.5], dtype=np.float32)).astype(np.float32)
                    surface_normal = np.array([0.0, 0.0, -1.0], dtype=np.float32)

                if i == 0 and (not hasattr(self, '_debug_count')):
                    self._debug_count = 0
                if i == 0:
                    self._debug_count += 1
                    if self._debug_count <= 5 or self._debug_count % 300 == 1:
                        diff = surf_pt - center.astype(np.float32)
                        gid = int(geomid_out[0])
                        gname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid) if gid >= 0 else 'MISS'
                        print(
                            f"[RAY #{self._debug_count}] "
                            f"center=({center[0]:.3f},{center[1]:.3f},{center[2]:.3f}) "
                            f"hit={gname} "
                            f"surf=({surf_pt[0]:.3f},{surf_pt[1]:.3f},{surf_pt[2]:.3f}) "
                            f"diff=({diff[0]:.3f},{diff[1]:.3f},{diff[2]:.3f}) "
                            f"(expect Z=-0.23)"
                        )

                if self._surface_offset != 0.0:
                    x_des[i, :3] = surf_pt - self._surface_offset * surface_normal
                else:
                    x_des[i, :3] = surf_pt
                x_des[i, :3] += self._x_des_offset

                current_rotmat = R.from_rotvec(palm_rotvec[i]).as_matrix()
                ori_rotmat = self._smooth_orient_to_normal(
                    i, surface_normal, current_rotmat,
                )
                x_des[i, 3:6] = R.from_matrix(ori_rotmat).as_rotvec().astype(np.float32)

            return torch.tensor(x_des, device=palm_obs.device)

    policy = PolicyWithActionAdapter()

    print("\n" + "="*45)
    print("         ★ Task Simulation Viewer Started ★")
    print("="*45)
    print(f"- 运行任务 ID:  {task_name}")
    print(f"- 仿真环境数:   {cfg.num_envs}")
    print(f"- 柔顺物理控制: {'已激活 (Active)' if cfg.use_compliance else '无补偿 (Disabled)'}")
    print(f"- 渲染视窗类型: {cfg.viewer.upper()} Viewer")
    print("="*45 + "\n")

    try:
        # 6. 使用 GUI 窗口原生运行仿真主环路
        if cfg.viewer == "native":
            NativeMujocoViewer(viewer_env, policy).run()
        else:
            ViserPlayViewer(viewer_env, policy).run()
    finally:
        viewer_env.close()
        print("[INFO] 物理仿真窗口已安全关闭，底层资源已成功释放。")


def main() -> None:
    # 确保加载所有注册任务
    import mjlab.tasks
    all_tasks = list_tasks()

    # 解析命令行所选任务
    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(all_tasks),
        add_help=False,
        return_unknown_args=True,
        config=mjlab.TYRO_FLAGS,
    )

    # 解析收集参数
    args = tyro.cli(
        CollectConfig,
        args=remaining_args,
        default=CollectConfig(),
        prog=sys.argv[0] + f" {chosen_task}",
        config=mjlab.TYRO_FLAGS,
    )

    # 执行
    run_collect(chosen_task, args)


if __name__ == "__main__":
    main()
