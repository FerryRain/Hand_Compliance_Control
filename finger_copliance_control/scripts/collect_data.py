"""Collect trajectory data by running the policy registered on a task."""

import csv  # 新增
from dataclasses import asdict, dataclass
from datetime import datetime
import os
import sys
from typing import Any, Literal

import glfw
import h5py
import torch
import tyro
import numpy as np
import mujoco

import mjlab
from mjlab.envs import ManagerBasedRlEnv, types as env_types
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


@dataclass(frozen=True)
class CollectConfig:
    output_dir: str = "./finger_copliance_control/data"
    num_envs: int = 16
    filename: str | None = None
    device: str | None = None
    viewer: Literal["native", "viser"] = "native"
    collect: bool = True
    record_forces: bool = True
    fsr_dims: int = 16 # 修改为16，对应手部传感器数量

class MocapObjectRotator:
    def __init__(self, num_envs: int, device: str, dt: float, mocap_idx: int):
        self.num_envs = num_envs
        self.device = device
        self.dt = dt
        self.mocap_idx = mocap_idx
        # 随机旋转轴
        axes = torch.randn((self.num_envs, 3), device=device)
        self.axes = axes / torch.norm(axes, dim=-1, keepdim=True)
        # speeds: [num_envs]
        self.speeds = torch.rand(self.num_envs, device=device) * 0.3 + 0.2

    def step(self, env: ManagerBasedRlEnv):
        # 获取当前四元数 [num_envs, 4] -> (w, x, y, z)
        current_quat = env.sim.data.mocap_quat[:, self.mocap_idx, :].clone()
        
        # 计算旋转增量
        theta = self.speeds * self.dt
        cos_t = torch.cos(theta / 2).unsqueeze(-1) # [num_envs, 1]
        sin_t = torch.sin(theta / 2).unsqueeze_(-1) # [num_envs, 1]
        delta_quat = torch.cat([cos_t, self.axes * sin_t], dim=-1)
        
        # 四元数乘法 (w1,x1,y1,z1) * (w2,x2,y2,z2)
        w1, x1, y1, z1 = current_quat.unbind(-1)
        w2, x2, y2, z2 = delta_quat.unbind(-1)
        new_quat = torch.stack([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ], dim=-1)
        # 写回数据
        env.sim.data.mocap_quat[:, self.mocap_idx, :] = (
            new_quat / torch.norm(new_quat, dim=-1, keepdim=True)
        )

class CSVDataLogger:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.file = open(filepath, 'w', newline='')
        self.writer = csv.writer(self.file)
        
        header = ["step", "time"]
        # FSR (16)
        header += [f"fsr_{i}" for i in range(16)]
        # Joint Positions (22: 6 arm + 16 hand)
        header += [f"q_{i}" for i in range(22)]
        # Object Pose (7)
        header += ["obj_px", "obj_py", "obj_pz", "obj_qw", "obj_qx", "obj_qy", "obj_qz"]
        # Palm Base Pose (7) - 关键：记录手掌基座在世界系下的位置
        header += ["palm_px", "palm_py", "palm_pz", "palm_qw", "palm_qx", "palm_qy", "palm_qz"]
        
        self.writer.writerow(header)
        self.step_idx = 0

    def log(self, time, fsr, q, obj_pose, palm_pose):
        row = [self.step_idx, time]
        row += fsr[0].detach().cpu().numpy().tolist()
        row += q[0].detach().cpu().numpy().tolist()
        row += obj_pose[0].detach().cpu().numpy().tolist()
        row += palm_pose[0].detach().cpu().numpy().tolist()
        self.writer.writerow(row)
        self.step_idx += 1

    def close(self):
        self.file.close()

class H5DataLogger:
    def __init__(self, filepath: str, num_envs: int, fsr_dim: int = 16, q_dim: int = 22, action_dim: int = 16):
        self.file = h5py.File(filepath, "w")
        self.num_envs = num_envs
        self.step_idx = 0
        
        chunk_size = 100 
        # 增加 dtype="f4" 修复警告
        self.dsets = {
            "fsr": self.file.create_dataset("fsr", (0, num_envs, fsr_dim), 
                                            maxshape=(None, num_envs, fsr_dim), chunks=(chunk_size, num_envs, fsr_dim), dtype="f4"),
            "q": self.file.create_dataset("q", (0, num_envs, q_dim), 
                                          maxshape=(None, num_envs, q_dim), chunks=(chunk_size, num_envs, q_dim), dtype="f4"),
            "action": self.file.create_dataset("action", (0, num_envs, action_dim), 
                                               maxshape=(None, num_envs, action_dim), chunks=(chunk_size, num_envs, action_dim), dtype="f4"),
            "obj_pose": self.file.create_dataset("obj_pose", (0, num_envs, 7), 
                                                 maxshape=(None, num_envs, 7), chunks=(chunk_size, num_envs, 7), dtype="f4"),
            "palm_pose": self.file.create_dataset("palm_pose", (0, num_envs, 7), 
                                                  maxshape=(None, num_envs, 7), chunks=(chunk_size, num_envs, 7), dtype="f4"),
            "time": self.file.create_dataset(
                "time",
                (0, num_envs),
                maxshape=(None, num_envs),
                chunks=(chunk_size, num_envs),
                dtype="f4",
            )
        }

    @staticmethod
    def _to_host(data):
        """Convert tensors/proxies to CPU scalar or numpy array for h5py."""
        # mjlab TorchArray proxy exposes the backing tensor as _tensor.
        if hasattr(data, "_tensor"):
            data = data._tensor

        if torch.is_tensor(data):
            data = data.detach().cpu()
            return data.item() if data.ndim == 0 else data.numpy()

        # Handle Python numeric values and numpy arrays uniformly.
        arr = np.asarray(data)
        return arr.item() if arr.ndim == 0 else arr

    def log(self, time, fsr, q, action, obj_pose, palm_pose):

        # 扩展数据集长度
        new_size = self.step_idx + 1
        for name, dset in self.dsets.items():
            if name == "time":
                dset.resize((new_size, self.num_envs))
            else:
                dset.resize((new_size, self.num_envs, dset.shape[2]))

        # 写入数据
        self.dsets["time"][self.step_idx] = self._to_host(time)
        self.dsets["fsr"][self.step_idx] = self._to_host(fsr)
        self.dsets["q"][self.step_idx] = self._to_host(q)
        self.dsets["action"][self.step_idx] = self._to_host(action)
        self.dsets["obj_pose"][self.step_idx] = self._to_host(obj_pose)
        self.dsets["palm_pose"][self.step_idx] = self._to_host(palm_pose)
        
        self.step_idx += 1
        
    def close(self):
        # 存储一些元数据方便后续分析
        self.file.attrs["num_envs"] = self.num_envs
        self.file.attrs["total_steps"] = self.step_idx
        self.file.close()

def get_random_quats(num_envs: int, device: str = "cuda:0") -> torch.Tensor:
    q = torch.randn((num_envs, 4), device=device)
    return q / torch.norm(q, dim=-1, keepdim=True)


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


def _resolve_target_mocap_idx(env: ManagerBasedRlEnv) -> int:
    target = env.scene["target"]
    mocap_id = target.data.indexing.mocap_id
    if isinstance(mocap_id, torch.Tensor):
        return int(mocap_id.item())
    return int(mocap_id)


def _resolve_palm_body_local_idx(env: ManagerBasedRlEnv) -> int:
    # Resolve against robot-local body names, which is robust to global name scoping.
    body_name_candidates = ("palm_lower", "base_link", "link6", "link_base")
    robot = env.scene["robot"]
    local_names = [body.name or "" for body in robot.data.indexing.bodies]

    for name in body_name_candidates:
        if name in local_names:
            local_idx = local_names.index(name)
            print(f"[INFO] Logging palm pose from body '{name}' (local_idx={local_idx})")
            return int(local_idx)

    # Fallback: support scoped names such as 'robot/palm_lower'.
    for name in body_name_candidates:
        for local_idx, local_name in enumerate(local_names):
            if local_name.endswith(f"/{name}") or local_name.endswith(name):
                print(
                    "[INFO] Logging palm pose from scoped body "
                    f"'{local_name}' (local_idx={local_idx})"
                )
                return int(local_idx)

    tried = ", ".join(body_name_candidates)
    sample = ", ".join(local_names[:12])
    raise ValueError(
        f"Could not find palm body. Tried: {tried}. "
        f"Robot body names sample: {sample}"
    )


def run_collect(task_id: str, cfg: CollectConfig) -> None:
    configure_torch_backends()

    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env_cfg = load_env_cfg(task_id, play=True)
    env_cfg.scene.num_envs = cfg.num_envs
    agent_cfg = load_rl_cfg(task_id)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    _log_joint_action_mapping(env)

    dt = env.sim.model.opt.timestep * env.cfg.decimation
    target_mocap_idx = _resolve_target_mocap_idx(env)
    palm_body_local_idx = _resolve_palm_body_local_idx(env)
    rotator = MocapObjectRotator(
        num_envs=env.num_envs,
        device=device,
        dt=dt,
        mocap_idx=target_mocap_idx,
    )

    logger: H5DataLogger | None = None
    csv_logger: CSVDataLogger | None = None  # 新增
    
    if cfg.collect:
        os.makedirs(cfg.output_dir, exist_ok=True)
        base_name = cfg.filename or f"collect_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # H5 保持原有逻辑
        h5_path = os.path.join(cfg.output_dir, f"{base_name}.h5")
        logger = H5DataLogger(
            h5_path, 
            num_envs=env.num_envs, 
            fsr_dim=16, 
            q_dim=22, 
            action_dim=16
        )
        # 新增 CSV 记录
        csv_path = os.path.join(cfg.output_dir, f"{base_name}.csv")
        # csv_logger = CSVDataLogger(csv_path)

    raw_policy = _build_registered_policy(task_id, cfg, env.num_envs)

    original_step = env.step
    original_reset = env.reset

    def reset_with_random_orientation(*args, **kwargs):
        obs, info = original_reset(*args, **kwargs)
        initial_quats = get_random_quats(env.num_envs, device=device)
        env.sim.data.mocap_quat[:, target_mocap_idx, :] = initial_quats
        env.sim.forward()  # 确保状态更新
        return obs, info

    env.reset = reset_with_random_orientation
    current_obs, _ = env.reset()

    action_dim = _log_observation_action_dims(env, current_obs)

    def step_with_logging(action: torch.Tensor):
        nonlocal current_obs
        rotator.step(env)

        # 1. 执行物理步
        next_obs, reward, terminated, truncated, info = original_step(action)
        
        # 2. 提取全量环境数据 (注意使用切片获取所有 env)
        policy_obs = _policy_obs_tensor(next_obs)
        fsr_data = policy_obs[:, :16]
        # 获取全部 22 个关节（包含机械臂基座和手）
        q_data = env.scene["robot"].data.joint_pos 

        # 获取物体位姿 (Mocap) [num_envs, 7]
        obj_p = env.sim.data.mocap_pos[:, target_mocap_idx, :].clone()
        obj_q = env.sim.data.mocap_quat[:, target_mocap_idx, :].clone()
        obj_pose = torch.cat((obj_p, obj_q), dim=-1)

        # 获取手掌基座的世界位姿 (从全局 xpos/xquat 提取)
        palm_pose_w = env.scene["robot"].data.body_link_pose_w[:, palm_body_local_idx, :]
        palm_p = palm_pose_w[:, :3].clone()
        palm_q = palm_pose_w[:, 3:].clone()
        palm_pose = torch.cat((palm_p, palm_q), dim=-1)

        # 3. H5 记录
        if logger is not None:
            logger.log(
                time=env.sim.data.time,
                fsr=fsr_data,
                q=q_data,
                action=action,
                obj_pose=obj_pose,
                palm_pose=palm_pose
            )

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
        saved_steps = logger.step_idx if logger is not None else saved_steps
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