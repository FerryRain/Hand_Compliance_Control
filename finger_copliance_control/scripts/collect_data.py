"""Collect trajectory data by running the policy registered on a task."""

import csv  # 新增
from dataclasses import asdict, dataclass
from datetime import datetime
import os
import sys
from typing import Any, Literal

import h5py
import torch
import tyro
import numpy as np

import mjlab
from mjlab.envs import ManagerBasedRlEnv, types as env_types
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


@dataclass(frozen=True)
class CollectConfig:
    output_dir: str = "./finger_copliance_control/data"
    filename: str | None = None
    device: str | None = None
    viewer: Literal["native", "viser"] = "native"
    collect: bool = True
    record_forces: bool = True
    fsr_dims: int = 16 # 修改为16，对应手部传感器数量


class CSVDataLogger:
    """新增的 CSV 记录器，用于记录详细的每帧信息"""
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.file = open(filepath, 'w', newline='')
        self.writer = csv.writer(self.file)
        
        # 构建表头
        header = ["step"]
        header += [f"fsr_{i}" for i in range(16)]          # 16个FSR
        header += [f"action_{i}" for i in range(16)]       # 16个手部Action
        header += [f"pos_{i}" for i in range(22)]          # 22个关节Position
        header += ["is_unlocked"]                          # 解锁状态
        self.writer.writerow(header)
        self.step_idx = 0

    def log(self, fsr, action, pos, is_unlocked):
        # 假设 num_envs = 1，取第0个环境的数据
        row = [self.step_idx]
        row += fsr[0].detach().cpu().numpy().tolist()
        row += action[0].detach().cpu().numpy().tolist()
        row += pos[0].detach().cpu().numpy().tolist()
        row += [int(is_unlocked[0].item())]
        self.writer.writerow(row)
        self.step_idx += 1

    def close(self):
        self.file.close()


class H5DataLogger:
    def __init__(self, filepath: str):
        self.file = h5py.File(filepath, "w")
        self.group = self.file.create_group("data")
        self.step_idx = 0

    def log(
        self,
        obs: env_types.VecEnvObs,
        action: torch.Tensor,
        reward: torch.Tensor,
        forces: torch.Tensor | None = None,
    ) -> None:
        step_grp = self.group.create_group(f"step_{self.step_idx}")
        policy_obs = _policy_obs_tensor(obs)
        step_grp.create_dataset("obs", data=policy_obs.detach().cpu().numpy())
        step_grp.create_dataset("action", data=action.detach().cpu().numpy())
        step_grp.create_dataset("reward", data=reward.detach().cpu().numpy())
        if forces is not None:
            step_grp.create_dataset("fsr_forces", data=forces.detach().cpu().numpy())
        self.step_idx += 1

    def close(self) -> None:
        self.file.close()


def _policy_obs_tensor(obs: env_types.VecEnvObs) -> torch.Tensor:
    policy_obs = obs["policy"]
    if isinstance(policy_obs, torch.Tensor):
        return policy_obs
    if "fsr_forces" in policy_obs:
        return policy_obs["fsr_forces"]
    if len(policy_obs) == 1:
        return next(iter(policy_obs.values()))
    keys = ", ".join(sorted(policy_obs.keys()))
    raise ValueError(f"Cannot infer policy tensor. Available terms: {keys}.")


def _build_registered_policy(task_id: str, cfg: CollectConfig, num_envs: int) -> Any:
    policy_cfg = load_rl_cfg(task_id)
    policy_class = getattr(policy_cfg, "policy_class", None)
    if policy_class is None:
        raise ValueError(f"Task '{task_id}' has no 'policy_class' in rl_cfg.")

    cfg_dict = asdict(policy_cfg)
    cfg_dict.pop("policy_class", None)
    cfg_dict.pop("device", None)

    policy_device = cfg.device or getattr(policy_cfg, "device", None) or ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Loading registered policy: {policy_class.__name__}")
    return policy_class(device=policy_device, num_envs=num_envs, **cfg_dict)


def _adapt_action_dim(action: torch.Tensor, target_dim: int) -> torch.Tensor:
    current_dim = int(action.shape[-1])
    if current_dim == target_dim: return action
    if target_dim == 0: return action.new_zeros((action.shape[0], 0))
    if current_dim > target_dim: return action[:, :target_dim]
    pad = action.new_zeros((action.shape[0], target_dim - current_dim))
    return torch.cat((action, pad), dim=-1)


def _log_joint_action_mapping(env: ManagerBasedRlEnv) -> None:
    try:
        joint_action = env.action_manager.get_term("hand_delta")
    except Exception: return
    target_names = getattr(joint_action, "target_names", None)
    target_ids = getattr(joint_action, "target_ids", None)
    if target_names is None or target_ids is None: return
    if hasattr(target_ids, "tolist"): target_ids = target_ids.tolist()
    print("[INFO] hand_delta action mapping logic executed...")


def _log_observation_action_dims(env: ManagerBasedRlEnv, obs: env_types.VecEnvObs) -> int:
    policy_obs = _policy_obs_tensor(obs)
    action_dim = env.action_manager.total_action_dim
    print(f"[INFO] Policy observation dim: {int(policy_obs.shape[-1])}")
    return action_dim


def run_collect(task_id: str, cfg: CollectConfig) -> None:
    configure_torch_backends()

    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env_cfg = load_env_cfg(task_id, play=True)
    agent_cfg = load_rl_cfg(task_id)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    _log_joint_action_mapping(env)

    logger: H5DataLogger | None = None
    csv_logger: CSVDataLogger | None = None  # 新增
    
    if cfg.collect:
        os.makedirs(cfg.output_dir, exist_ok=True)
        base_name = cfg.filename or f"collect_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # H5 保持原有逻辑
        logger = H5DataLogger(os.path.join(cfg.output_dir, f"{base_name}.h5"))
        # 新增 CSV 记录
        csv_path = os.path.join(cfg.output_dir, f"{base_name}.csv")
        csv_logger = CSVDataLogger(csv_path)

    raw_policy = _build_registered_policy(task_id, cfg, env.num_envs)

    original_step = env.step
    current_obs, _ = env.reset()
    action_dim = _log_observation_action_dims(env, current_obs)

    # 预定义镜像的解锁逻辑参数（需与控制器一致）
    PALM_FSR_IDX = [0, 1, 2, 3]
    PALM_THRESHOLD = 0.2

    def step_with_logging(action: torch.Tensor):
        nonlocal current_obs
        # 1. 执行物理步
        next_obs, reward, terminated, truncated, info = original_step(action)
        
        # 2. 提取数据
        policy_obs = _policy_obs_tensor(next_obs)
        fsr_data = policy_obs[:, :16]
        pos_data = policy_obs[:, 16:38]
        
        # 3. 镜像计算解锁状态
        palm_force = torch.mean(fsr_data[:, PALM_FSR_IDX], dim=1)
        is_unlocked = palm_force > PALM_THRESHOLD

        # 4. 记录数据
        if logger is not None:
            logger.log(current_obs, action, reward, fsr_data)
        
        if csv_logger is not None:
            # 仅记录手部 16 维 action (假设 action 是适配后的 [envs, 22]，取后 16 位)
            # 或者直接取 raw_policy 的输出，这里我们取传入 step 的 action 的后 16 位
            hand_action = action[:, 6:] 
            csv_logger.log(fsr_data, hand_action, pos_data, is_unlocked)

        current_obs = next_obs
        return next_obs, reward, terminated, truncated, info

    env.step = step_with_logging
    viewer_env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))

    class PolicyWithActionAdapter:
        def __call__(self, obs):
            return _adapt_action_dim(raw_policy(obs), action_dim)

    policy = PolicyWithActionAdapter()

    try:
        if cfg.viewer == "native":
            NativeMujocoViewer(viewer_env, policy).run()
        else:
            ViserPlayViewer(viewer_env, policy).run()
    finally:
        if logger is not None: logger.close()
        if csv_logger is not None: csv_logger.close()
        viewer_env.close()
        saved_steps = csv_logger.step_idx if csv_logger is not None else 0
        print(f"[SUCCESS] Saved {saved_steps} steps to CSV and H5")


def main() -> None:
    import mjlab.tasks
    all_tasks = list_tasks()
    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(all_tasks),
        add_help=False,
        return_unknown_args=True,
        config=mjlab.TYRO_FLAGS,
    )
    args = tyro.cli(
        CollectConfig,
        args=remaining_args,
        default=CollectConfig(),
        prog=sys.argv[0] + f" {chosen_task}",
        config=mjlab.TYRO_FLAGS,
    )
    run_collect(chosen_task, args)


if __name__ == "__main__":
    main()