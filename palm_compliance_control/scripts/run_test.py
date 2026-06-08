from __future__ import annotations

from dataclasses import asdict, dataclass
import sys
from typing import Any, Literal

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
    """是否将 x_des 实时投影到 capsule 表面（需要观测中包含 target_pos/rot）"""


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
    """从统一观测对象中解析出 policy 特征张量"""
    policy_obs = obs["policy"]
    if isinstance(policy_obs, torch.Tensor):
        return policy_obs
    if "fsr_forces" in policy_obs:
        return policy_obs["fsr_forces"]
    if len(policy_obs) == 1:
        return next(iter(policy_obs.values()))
    keys = ", ".join(sorted(policy_obs.keys()))
    raise ValueError(f"无法自动推断 policy tensor。当前包含项: {keys}")


def _log_observation_action_dims(env: ManagerBasedRlEnv, obs: env_types.VecEnvObs) -> int:
    policy_obs = _policy_obs_tensor(obs)
    action_dim = env.action_manager.total_action_dim
    print(f"[INFO] 任务观测特征维度: {int(policy_obs.shape[-1])}")
    print(f"[INFO] 任务动作空间维度: {action_dim}")
    return action_dim


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

    # 2. 实例化 ManagerBasedRlEnv 环境
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)

    # 3. 动态配置对应的底层物理控制器（全动态解析，零硬编码导入）
    if cfg.use_compliance:
        raw_policy = _build_registered_policy(task_name, cfg, env.num_envs)
    else:
        raw_policy = NullComplianceController(device=device, num_envs=env.num_envs)

    # 4. 获取第一步状态并确定所需动作维度
    current_obs, _ = env.reset()
    action_dim = _log_observation_action_dims(env, current_obs)

    # 5. 按照统一格式封装环境为包装器
    viewer_env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))

    class PolicyWithActionAdapter:
        def __init__(self):
            self._surface_track = cfg.surface_track

        def __call__(self, obs):
            policy_obs = _policy_obs_tensor(obs)

            if self._surface_track:
                x_des = self._compute_surface_x_des(policy_obs)
                action = raw_policy(obs, x_des=x_des)
            else:
                action = raw_policy(obs)

            return _adapt_action_dim(action, action_dim)

        @staticmethod
        def _compute_surface_x_des(policy_obs: torch.Tensor) -> torch.Tensor:
            """从观测中计算 capsule 表面交点 + 手掌朝向目标 (B,6)。"""
            # 布局: 76:79 palm_pos, 79:82 palm_rot, 82:85 target_pos, 85:88 target_rot
            palm_pos = policy_obs[:, 76:79].cpu().numpy().astype(np.float32)
            target_pos = policy_obs[:, 82:85].cpu().numpy().astype(np.float32)
            target_rotvec = policy_obs[:, 85:88].cpu().numpy().astype(np.float32)

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
                    # 选一个不平行于 target_z 的参考轴来构造正交基
                    ref = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                    if abs(np.dot(ref, target_z)) > 0.99:
                        ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                    target_x = np.cross(ref, target_z)
                    target_x /= np.linalg.norm(target_x)
                    target_y = np.cross(target_z, target_x)
                    rotmat = np.column_stack([target_x, target_y, target_z])
                    x_des[i, 3:6] = R.from_matrix(rotmat).as_rotvec().astype(np.float32)

            return torch.tensor(x_des, device=policy_obs.device)

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