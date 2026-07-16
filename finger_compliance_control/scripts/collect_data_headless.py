"""Headless automatic data collection for registered task policies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from contextlib import contextmanager
import os
import sys
from typing import Any
from typing import Literal

import warp as wp

wp.config.quiet = True  # 关闭 Warp 的大部分初始化和运行时提示

import mujoco

# 尝试再次设置，但这次是针对 MuJoCo 的用户 warning 回调。
mujoco.set_mju_user_warning(lambda *_args: None)

import h5py
import numpy as np
import torch
import tyro
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - fallback when tqdm is unavailable
    tqdm = None

import mjlab
from mjlab.envs import ManagerBasedRlEnv, types as env_types
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


@contextmanager
def _suppress_mujoco_output():
    sys.stdout.flush()
    sys.stderr.flush()
    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    with open(os.devnull, "w") as devnull:
        try:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(stdout_fd, 1)
            os.dup2(stderr_fd, 2)
            os.close(stdout_fd)
            os.close(stderr_fd)


@dataclass(frozen=True)
class HeadlessCollectConfig:
    output_dir: str = "./finger_compliance_control/data/headless"
    num_envs: int = 16
    episode_length: int = 2000
    max_episodes: int = 600
    total_steps: int | None = None
    reset_interval: int = 0
    resample_axes_interval: int = 1000  # 每隔N步随机切换物体转轴（0=不切换）
    filename: str | None = None
    device: str | None = None
    viewer: Literal["headless", "native", "viser"] = "headless"
    fsr_dims: int = 16
    ccd_iterations: int = 200
    contact_threshold: float = 0.20
    stability_window: int = 20
    action_noise_std: float = 0
    target_anchor: Literal["palm", "origin"] = "origin"
    target_offset: tuple[float, float, float] = (0.10, 0.0, 0.0)
    fsr_source: Literal["policy", "sensor"] = "policy"
    # Keep target position fixed to env init pose; only orientation is varied.
    randomize_object_orientation: bool = True
    # Comma-separated profile names, e.g. "capsule_medium,box_medium".
    object_profiles: str = "capsule_medium"
    randomize_object_profile: bool = True
    # Path to LeRobot pretrained DiffusionPolicy directory (overrides registered policy).
    dp_model_dir: str | None = None


class MocapObjectRotator:
    def __init__(self, num_envs: int, device: str, dt: float, mocap_idx: int):
        self.num_envs = num_envs
        self.device = device
        self.dt = dt
        self.mocap_idx = mocap_idx
        self._episode_step = 0  # 当期已跑步数，reset时归零

        self.axes = torch.zeros((self.num_envs, 3), device=device)
        self.speeds = torch.zeros(self.num_envs, device=device)
        # 正弦振荡参数
        self.trans_amp = torch.zeros((self.num_envs, 3), device=device)    # 各轴振幅 (m)
        self.trans_freq = torch.zeros((self.num_envs, 3), device=device)   # 各轴频率 (Hz)
        self.trans_phase = torch.zeros((self.num_envs, 3), device=device)  # 各轴相位 (rad)
        self.trans_base = torch.zeros((self.num_envs, 3), device=device)   # 振荡中心 (m)
        self.resample_axes()

    def resample_axes(self):
        """每期随机切换旋转轴、转速和振荡参数，增加轨迹多样性。"""
        axes = torch.randn((self.num_envs, 3), device=self.device)
        self.axes = axes / torch.norm(axes, dim=-1, keepdim=True)
        self.speeds = torch.rand(self.num_envs, device=self.device) * 0.5 + 0.3

        self.trans_amp = torch.rand(self.num_envs, 3, device=self.device) * 0.02 + 0.005  # 0.5-2.5 cm
        self.trans_freq = torch.rand(self.num_envs, 3, device=self.device) * 0.3 + 0.1   # 0.2-0.7 Hz
        self.trans_phase = torch.rand(self.num_envs, 3, device=self.device) * 2 * 3.14159  # 0-2π
        self._episode_step = 0
        print(f"[ROTATOR] resample_axes: "
              f"speeds∈[{self.speeds.min().item():.3f},{self.speeds.max().item():.3f}] rad/s, "
              f"amp∈[{self.trans_amp.min().item():.4f},{self.trans_amp.max().item():.4f}] m")

    def set_base_position(self, pos: torch.Tensor):
        """在reset后调用，记录当期振荡中心（当前物体位置）。"""
        self.trans_base = pos.clone()

    def step(self, env: ManagerBasedRlEnv):
        self._episode_step += 1
        t_sec = (self._episode_step * self.dt).unsqueeze(-1)  # type: ignore # 当期已过秒数, (num_envs,1)

        # ── 旋转 ──
        current_quat = env.sim.data.mocap_quat[:, self.mocap_idx, :].clone()
        theta = self.speeds * self.dt
        cos_t = torch.cos(theta / 2).unsqueeze(-1)
        sin_t = torch.sin(theta / 2).unsqueeze(-1)
        delta_quat = torch.cat([cos_t, self.axes * sin_t], dim=-1)
        w1, x1, y1, z1 = current_quat.unbind(-1)
        w2, x2, y2, z2 = delta_quat.unbind(-1)
        new_quat = torch.stack(
            [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
             w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
             w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
             w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], dim=-1)
        env.sim.data.mocap_quat[:, self.mocap_idx, :] = (
            new_quat / torch.norm(new_quat, dim=-1, keepdim=True)
        )

        # ── 正弦振荡：围绕base位置做3轴Lissajous运动 ──
        osc = torch.sin(2.0 * 3.14159 * self.trans_freq * t_sec + self.trans_phase)
        offset = self.trans_amp * osc  # [B, 3] * [B, 3]
        env.sim.data.mocap_pos[:, self.mocap_idx, :] = self.trans_base + offset


FINGER_FSR_IDS: tuple[tuple[int, ...], ...] = (
    (4, 5, 6),
    (7, 8, 9),
    (10, 11, 12),
    (13, 14, 15),
)

_TARGET_PROFILE_TO_GEOM_NAME: dict[str, str] = {
    "capsule_medium": "target_capsule_medium_geom",
    "box_medium": "target_box_medium_geom",
    "cylinder_medium": "target_cylinder_medium_geom",
    "ellipsoid_medium": "target_ellipsoid_medium_geom",
}


def _parse_object_profiles(value: str) -> tuple[str, ...]:
    profiles = tuple(part.strip() for part in value.split(",") if part.strip())
    if not profiles:
        raise ValueError("object_profiles must contain at least one profile name.")
    return profiles


class TargetObjectProfileRandomizer:
    def __init__(self, env: ManagerBasedRlEnv, cfg: HeadlessCollectConfig, device: str):
        self.env = env
        self.cfg = cfg
        self.device = device
        self._warned_global = False

        requested_profiles = _parse_object_profiles(cfg.object_profiles)
        unknown = sorted(set(requested_profiles) - set(_TARGET_PROFILE_TO_GEOM_NAME.keys()))
        if unknown:
            valid = ", ".join(sorted(_TARGET_PROFILE_TO_GEOM_NAME.keys()))
            raise ValueError(
                f"Unknown object profile(s): {', '.join(unknown)}. Valid profiles: {valid}."
            )

        target_entity = env.scene["target"]
        local_geom_names = [geom.name or "" for geom in target_entity.data.indexing.geoms]
        local_geom_ids = target_entity.data.indexing.geom_ids
        if isinstance(local_geom_ids, torch.Tensor):
            geom_ids = [int(gid) for gid in local_geom_ids.tolist()]
        else:
            geom_ids = [int(gid) for gid in local_geom_ids]
        geom_name_to_id = {name: gid for name, gid in zip(local_geom_names, geom_ids)}

        self.profiles: tuple[str, ...] = requested_profiles
        self.profile_to_geom_id: dict[str, int] = {}
        for profile in self.profiles:
            geom_name = _TARGET_PROFILE_TO_GEOM_NAME[profile]
            resolved_name = geom_name if geom_name in geom_name_to_id else None
            if resolved_name is None:
                for local_name in local_geom_names:
                    if local_name.endswith(f"/{geom_name}") or local_name.endswith(geom_name):
                        resolved_name = local_name
                        break
            if resolved_name is None:
                available = ", ".join(local_geom_names)
                raise ValueError(
                    f"Target geom '{geom_name}' for profile '{profile}' was not found. "
                    f"Available target geoms: {available}"
                )
            self.profile_to_geom_id[profile] = geom_name_to_id[resolved_name]

        self.geom_ids = torch.tensor(
            [self.profile_to_geom_id[p] for p in self.profiles],
            dtype=torch.long,
            device=device,
        )

        env.sim.expand_model_fields(("geom_size", "geom_rgba"))
        self.model = env.sim.model

        geom_size = self.model.geom_size
        geom_rgba = self.model.geom_rgba

        if geom_size.ndim == 3:
            self.base_size = geom_size[0, self.geom_ids].clone()
        else:
            self.base_size = geom_size[self.geom_ids].clone()

        if geom_rgba.ndim == 3:
            self.base_rgba = geom_rgba[0, self.geom_ids].clone()
        else:
            self.base_rgba = geom_rgba[self.geom_ids].clone()

        print(f"[INFO] Enabled object profiles: {', '.join(self.profiles)}")

    def randomize(self) -> None:
        num_envs = self.env.num_envs
        n_profiles = len(self.profiles)
        if n_profiles <= 1:
            return  # 单物体不需要随机化

        if self.cfg.randomize_object_profile:
            profile_idx = torch.randint(
                low=0, high=n_profiles, size=(num_envs,),
                device=self.device,
            )
        else:
            profile_idx = torch.zeros((num_envs,), dtype=torch.long, device=self.device)

        geom_size = self.model.geom_size
        geom_rgba = self.model.geom_rgba
        env_ids = torch.arange(num_envs, device=self.device, dtype=torch.long)

        # ── 每个环境只保留选中的物体，其余完全隐藏 ──
        if geom_size.ndim == 3 and geom_rgba.ndim == 3:
            # 先把所有物体缩小到不可见
            hidden = self.base_size.unsqueeze(0).repeat(num_envs, 1, 1) * 1e-6
            geom_size[env_ids.unsqueeze(-1), self.geom_ids.unsqueeze(0)] = hidden
            # rgba: alpha=0 完全透明
            zero_rgba = torch.zeros_like(
                self.base_rgba.unsqueeze(0).repeat(num_envs, 1, 1)
            )
            geom_rgba[env_ids.unsqueeze(-1), self.geom_ids.unsqueeze(0)] = zero_rgba

            # 恢复每个环境选中的物体
            selected = self.geom_ids[profile_idx]  # [num_envs]
            geom_size[env_ids, selected] = self.base_size[profile_idx]
            geom_rgba[env_ids, selected] = self.base_rgba[profile_idx]

        else:
            # 全局 fallback：逐个环境设置（效率低但可靠）
            if not self._warned_global:
                print(
                    "[WARN] geom fields not per-env expanded; "
                    "using env-loop fallback. Object randomization is GLOBAL."
                )
                self._warned_global = True

            # 全部缩小
            geom_size[self.geom_ids] = self.base_size * 1e-6
            geom_rgba[self.geom_ids] = 0.0

            # 只恢复第一个环境的选中物体
            chosen = int(profile_idx[0].item())
            selected_geom_id = int(self.geom_ids[chosen].item())
            geom_size[selected_geom_id] = self.base_size[chosen]
            geom_rgba[selected_geom_id] = self.base_rgba[chosen]

        # ── 物理刷新：确保几何变化生效 ──
        if hasattr(self.env.sim, 'forward'):
            self.env.sim.forward()


class H5DataLogger:
    def __init__(
        self,
        filepath: str,
        num_envs: int,
        fsr_dim: int = 16,
        q_dim: int = 22,
        action_dim: int = 16,
        qfrc_dim: int = 16,
    ):
        self.file = h5py.File(filepath, "w")
        self.num_envs = num_envs
        self.step_idx = 0

        chunk_size = 100
        self.dsets = {
            "fsr": self.file.create_dataset(
                "fsr",
                (0, num_envs, fsr_dim),
                maxshape=(None, num_envs, fsr_dim),
                chunks=(chunk_size, num_envs, fsr_dim),
                dtype="f4",
            ),
            "q": self.file.create_dataset(
                "q",
                (0, num_envs, q_dim),
                maxshape=(None, num_envs, q_dim),
                chunks=(chunk_size, num_envs, q_dim),
                dtype="f4",
            ),
            "action": self.file.create_dataset(
                "action",
                (0, num_envs, action_dim),
                maxshape=(None, num_envs, action_dim),
                chunks=(chunk_size, num_envs, action_dim),
                dtype="f4",
            ),
            "qfrc_actuator": self.file.create_dataset(
                "qfrc_actuator",
                (0, num_envs, qfrc_dim),
                maxshape=(None, num_envs, qfrc_dim),
                chunks=(chunk_size, num_envs, qfrc_dim),
                dtype="f4",
            ),
            "obj_pose": self.file.create_dataset(
                "obj_pose",
                (0, num_envs, 7),
                maxshape=(None, num_envs, 7),
                chunks=(chunk_size, num_envs, 7),
                dtype="f4",
            ),
            "palm_pose": self.file.create_dataset(
                "palm_pose",
                (0, num_envs, 7),
                maxshape=(None, num_envs, 7),
                chunks=(chunk_size, num_envs, 7),
                dtype="f4",
            ),
            "time": self.file.create_dataset(
                "time",
                (0, num_envs),
                maxshape=(None, num_envs),
                chunks=(chunk_size, num_envs),
                dtype="f4",
            ),
            "episode_id": self.file.create_dataset(
                "episode_id",
                (0, num_envs),
                maxshape=(None, num_envs),
                chunks=(chunk_size, num_envs),
                dtype="i4",
            ),
            "finger_force": self.file.create_dataset(
                "finger_force",
                (0, num_envs, 4),
                maxshape=(None, num_envs, 4),
                chunks=(chunk_size, num_envs, 4),
                dtype="f4",
            ),
            "finger_contact": self.file.create_dataset(
                "finger_contact",
                (0, num_envs, 4),
                maxshape=(None, num_envs, 4),
                chunks=(chunk_size, num_envs, 4),
                dtype="f4",
            ),
            "full_contact": self.file.create_dataset(
                "full_contact",
                (0, num_envs, 1),
                maxshape=(None, num_envs, 1),
                chunks=(chunk_size, num_envs, 1),
                dtype="f4",
            ),
            "contact_stability": self.file.create_dataset(
                "contact_stability",
                (0, num_envs, 1),
                maxshape=(None, num_envs, 1),
                chunks=(chunk_size, num_envs, 1),
                dtype="f4",
            ),
            "force_balance": self.file.create_dataset(
                "force_balance",
                (0, num_envs, 1),
                maxshape=(None, num_envs, 1),
                chunks=(chunk_size, num_envs, 1),
                dtype="f4",
            ),
            "fsr_delta_norm": self.file.create_dataset(
                "fsr_delta_norm",
                (0, num_envs, 1),
                maxshape=(None, num_envs, 1),
                chunks=(chunk_size, num_envs, 1),
                dtype="f4",
            ),
            # 指尖原始3D力矢量（非归一化）：4指 × 3D = 12D
            # index=(0:3), middle=(3:6), ring=(6:9), thumb=(9:12)
            # 从 ContactSensor 远端传感器 (6,9,12,15) 读取原始 force 向量
            "fingertip_force_3d": self.file.create_dataset(
                "fingertip_force_3d",
                (0, num_envs, 12),
                maxshape=(None, num_envs, 12),
                chunks=(chunk_size, num_envs, 12),
                dtype="f4",
            ),
            "fingertip_pos": self.file.create_dataset(
                "fingertip_pos",
                (0, num_envs, 12),
                maxshape=(None, num_envs, 12),
                chunks=(chunk_size, num_envs, 12),
                dtype="f4",
            ),
        }

    @staticmethod
    def _to_host(data: Any):
        if hasattr(data, "_tensor"):
            data = data._tensor
        if torch.is_tensor(data):
            data = data.detach().cpu()
            return data.item() if data.ndim == 0 else data.numpy()
        arr = np.asarray(data)
        return arr.item() if arr.ndim == 0 else arr

    def log(self, time, fsr, q, action, qfrc_actuator, obj_pose, palm_pose, quality,
            fingertip_force=None, episode_id=None, fingertip_pos=None):
        new_size = self.step_idx + 1
        for name, dset in self.dsets.items():
            if name in ("time", "episode_id"):
                dset.resize((new_size, self.num_envs))
            else:
                dset.resize((new_size, self.num_envs, dset.shape[2]))

        self.dsets["time"][self.step_idx] = self._to_host(time)
        if episode_id is not None:
            self.dsets["episode_id"][self.step_idx] = self._to_host(episode_id)
        self.dsets["fsr"][self.step_idx] = self._to_host(fsr)
        self.dsets["q"][self.step_idx] = self._to_host(q)
        self.dsets["action"][self.step_idx] = self._to_host(action)
        self.dsets["qfrc_actuator"][self.step_idx] = self._to_host(qfrc_actuator)
        self.dsets["obj_pose"][self.step_idx] = self._to_host(obj_pose)
        self.dsets["palm_pose"][self.step_idx] = self._to_host(palm_pose)
        self.dsets["finger_force"][self.step_idx] = self._to_host(quality["finger_force"])
        self.dsets["finger_contact"][self.step_idx] = self._to_host(
            quality["finger_contact"]
        )
        self.dsets["full_contact"][self.step_idx] = self._to_host(
            quality["full_contact"]
        )
        self.dsets["contact_stability"][self.step_idx] = self._to_host(
            quality["contact_stability"]
        )
        self.dsets["force_balance"][self.step_idx] = self._to_host(
            quality["force_balance"]
        )
        self.dsets["fsr_delta_norm"][self.step_idx] = self._to_host(
            quality["fsr_delta_norm"]
        )
        if fingertip_force is not None:
            self.dsets["fingertip_force_3d"][self.step_idx] = self._to_host(fingertip_force)
        if fingertip_pos is not None:
            self.dsets["fingertip_pos"][self.step_idx] = self._to_host(fingertip_pos)
        self.step_idx += 1

    def close(self, source_robot_joint_names: list[str] | None = None):
        self.file.attrs["num_envs"] = self.num_envs
        self.file.attrs["total_steps"] = self.step_idx
        if source_robot_joint_names is not None:
            self.file.attrs["source_robot_joint_names"] = np.array(
                source_robot_joint_names, dtype=h5py.string_dtype()
            )
        self.file.close()


def _get_random_quats(num_envs: int, device: str) -> torch.Tensor:
    q = torch.randn((num_envs, 4), device=device)
    return q / torch.norm(q, dim=-1, keepdim=True)


def _reset_target_pose(
    env: ManagerBasedRlEnv,
    target_mocap_idx: int,
    base_pos: torch.Tensor,
    palm_body_local_idx: int,
    cfg: HeadlessCollectConfig,
    device: str,
    object_profile_randomizer: TargetObjectProfileRandomizer | None = None,
) -> None:
    if cfg.target_anchor == "palm":
        palm_pose_w = env.scene["robot"].data.body_link_pose_w[:, palm_body_local_idx, :]
        anchor_pos = palm_pose_w[:, :3]
        offset = torch.tensor(cfg.target_offset, device=device, dtype=anchor_pos.dtype)
        env.sim.data.mocap_pos[:, target_mocap_idx, :] = anchor_pos + offset.unsqueeze(0)
    else:
        # Keep the position produced by env.reset() to match headed collector behavior.
        _ = base_pos

    if cfg.randomize_object_orientation:
        env.sim.data.mocap_quat[:, target_mocap_idx, :] = _get_random_quats(
            env.num_envs, device=device
        )

    if object_profile_randomizer is not None:
        object_profile_randomizer.randomize()

    env.sim.forward()


def _compute_contact_quality(
    fsr_data: torch.Tensor,
    prev_fsr_data: torch.Tensor,
    full_contact_run: torch.Tensor,
    cfg: HeadlessCollectConfig,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    # 指尖接触判定：仅用远端指尖传感器 (index=6, middle=9, ring=12, thumb=15)
    # finger_force = fsr_data[:, [6, 9, 12, 15]]  # [B, 4]
    finger_force = torch.stack([
        fsr_data[:, ids].mean(dim=1) for ids in [
            (4,5,6), (7,8,9), (10,11,12), (13,14,15)
        ]], dim=1)
    finger_contact = (finger_force >= cfg.contact_threshold).to(fsr_data.dtype)
    full_contact = torch.all(finger_contact > 0.5, dim=1, keepdim=True).to(fsr_data.dtype)

    full_contact_bool = full_contact.squeeze(-1) > 0.5
    full_contact_run = torch.where(
        full_contact_bool,
        full_contact_run + 1,
        torch.zeros_like(full_contact_run),
    )
    contact_stability = torch.clamp(
        full_contact_run.to(fsr_data.dtype) / max(1, cfg.stability_window),
        min=0.0,
        max=1.0,
    ).unsqueeze(-1)

    # Lower std means forces are distributed more evenly across fingers.
    force_balance = torch.std(finger_force, dim=1, keepdim=True)
    fsr_delta_norm = torch.linalg.vector_norm(
        fsr_data - prev_fsr_data,
        dim=1,
        keepdim=True,
    )

    quality = {
        "finger_force": finger_force,
        "finger_contact": finger_contact,
        "full_contact": full_contact,
        "contact_stability": contact_stability,
        "force_balance": force_balance,
        "fsr_delta_norm": fsr_delta_norm,
    }
    return quality, full_contact_run


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


def _read_fsr_data(
    env: ManagerBasedRlEnv,
    obs: env_types.VecEnvObs,
    fsr_dims: int,
    source: Literal["policy", "sensor"] = "policy",
) -> torch.Tensor:
    """Read FSR magnitudes robustly.

    Prefer direct ContactSensor force to avoid stale/ambiguous observation packing.
    Fallback to policy observation slicing when the sensor path is unavailable.
    """
    if source == "sensor":
        try:
            sensor = env.scene["fsr_contact"]
            sensor_data = sensor.data
            force = sensor_data.force
            if force is not None:
                mag = torch.linalg.vector_norm(force, dim=-1)
                fsr = torch.zeros((mag.shape[0], fsr_dims), device=mag.device, dtype=mag.dtype)
                copy_count = min(fsr_dims, mag.shape[1])
                fsr[:, :copy_count] = mag[:, :copy_count]
                return fsr
        except Exception:
            pass

    policy_obs = _policy_obs_tensor(obs)
    if policy_obs.ndim != 2 or policy_obs.shape[-1] < fsr_dims:
        raise ValueError(
            "Cannot recover FSR from policy observation. "
            f"Got shape={tuple(policy_obs.shape)}, expected second dim >= {fsr_dims}."
        )
    return policy_obs[:, :fsr_dims]


# 指尖远端 FSR 传感器索引（index=6, middle=9, ring=12, thumb=15）
_DISTAL_FSR_IDS = (6, 9, 12, 15)

# 指尖 body 名称（对应 index, middle, ring, thumb）
_FINGERTIP_BODY_NAMES = ("fingertip", "fingertip_2", "fingertip_3", "thumb_fingertip")


def _read_fingertip_positions(
    env: ManagerBasedRlEnv,
    palm_body_local_idx: int,
) -> torch.Tensor:
    """Read 4 fingertip body positions relative to palm.

    Returns: [B, 12] — index=(0:3), middle=(3:6), ring=(6:9), thumb=(9:12)
    Positions in world frame, will be converted to palm-relative later.
    """
    robot = env.scene["robot"]
    body_poses = robot.data.body_link_pose_w  # [B, n_bodies, 7]
    local_names = [body.name or "" for body in robot.data.indexing.bodies]

    result = torch.zeros(body_poses.shape[0], 12, device=body_poses.device, dtype=body_poses.dtype)
    palm_pos = body_poses[:, palm_body_local_idx, :3]  # [B, 3]

    for i, name in enumerate(_FINGERTIP_BODY_NAMES):
        if name in local_names:
            ft_local_idx = local_names.index(name)
            ft_world = body_poses[:, ft_local_idx, :3]  # [B, 3]
            result[:, i * 3 : (i + 1) * 3] = ft_world - palm_pos  # palm-relative
    return result


def _read_fingertip_force(
    env: ManagerBasedRlEnv,
) -> torch.Tensor:
    """Read raw 3D force vectors for the 4 distal FSR sensors.

    Returns:
        [B, 12] — index=(0:3), middle=(3:6), ring=(6:9), thumb=(9:12)
        Zero vector when no contact.
        Magnitude = |force_3d|, direction = normalize(force_3d).
    """
    zero = torch.zeros(1, 12, device=env.device)
    try:
        sensor = env.scene["fsr_contact"]
        force = sensor.data.force  # [B, N, 3]
        if force is None:
            return zero.expand(1, 12)
    except Exception:
        return zero

    B = force.shape[0]
    n_sensors = force.shape[1]
    result = torch.zeros(B, 12, device=force.device, dtype=force.dtype)

    for i, idx in enumerate(_DISTAL_FSR_IDS):
        if idx < n_sensors:
            result[:, i * 3 : (i + 1) * 3] = force[:, idx, :]
    return result


def _build_registered_policy(task_id: str, cfg: HeadlessCollectConfig, num_envs: int) -> Any:
    policy_cfg = load_rl_cfg(task_id)
    policy_class = getattr(policy_cfg, "policy_class", None)
    if policy_class is None:
        raise ValueError(f"Task '{task_id}' has no 'policy_class' in rl_cfg.")

    cfg_dict = asdict(policy_cfg)
    cfg_dict.pop("policy_class", None)
    cfg_dict.pop("device", None)

    policy_device = cfg.device or getattr(policy_cfg, "device", None) or (
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    print(f"[INFO] Loading registered policy: {policy_class.__name__}")
    return policy_class(device=policy_device, num_envs=num_envs, **cfg_dict)


def _adapt_action_dim(action: torch.Tensor, target_dim: int) -> torch.Tensor:
    current_dim = int(action.shape[-1])
    if current_dim == target_dim:
        return action
    if target_dim == 0:
        return action.new_zeros((action.shape[0], 0))
    if current_dim > target_dim:
        return action[:, :target_dim]
    pad = action.new_zeros((action.shape[0], target_dim - current_dim))
    return torch.cat((action, pad), dim=-1)


def _resolve_target_mocap_idx(env: ManagerBasedRlEnv) -> int:
    target = env.scene["target"]
    mocap_id = target.data.indexing.mocap_id
    if isinstance(mocap_id, torch.Tensor):
        return int(mocap_id.item())
    return int(mocap_id)


def _resolve_palm_body_local_idx(env: ManagerBasedRlEnv) -> int:
    body_name_candidates = ("palm_lower", "base_link", "link6", "link_base")
    robot = env.scene["robot"]
    local_names = [body.name or "" for body in robot.data.indexing.bodies]

    for name in body_name_candidates:
        if name in local_names:
            local_idx = local_names.index(name)
            print(f"[INFO] Logging palm pose from body '{name}' (local_idx={local_idx})")
            return int(local_idx)

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


def run_headless_collect(task_id: str, cfg: HeadlessCollectConfig) -> None:
    configure_torch_backends()

    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    total_steps = cfg.total_steps or (cfg.episode_length * cfg.max_episodes)
    reset_interval = cfg.reset_interval
    resample_axes_interval = cfg.resample_axes_interval

    with _suppress_mujoco_output():
        env_cfg = load_env_cfg(task_id, play=True)
        env_cfg.scene.num_envs = cfg.num_envs
        if hasattr(env_cfg, "sim") and hasattr(env_cfg.sim, "mujoco"):
            env_cfg.sim.mujoco.ccd_iterations = cfg.ccd_iterations
        env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    agent_cfg = load_rl_cfg(task_id)

    action_dim = env.action_manager.total_action_dim
    target_mocap_idx = _resolve_target_mocap_idx(env)
    palm_body_local_idx = _resolve_palm_body_local_idx(env)
    base_target_pos = env.sim.data.mocap_pos[:, target_mocap_idx, :].clone()
    object_profile_randomizer = TargetObjectProfileRandomizer(env, cfg, device)
    dt = env.sim.model.opt.timestep * env.cfg.decimation
    rotator = MocapObjectRotator(
        num_envs=env.num_envs,
        device=device,
        dt=dt,
        mocap_idx=target_mocap_idx,
    )

    os.makedirs(cfg.output_dir, exist_ok=True)
    base_name = cfg.filename or f"collect_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    h5_path = os.path.join(cfg.output_dir, f"{base_name}.h5")
    logger = H5DataLogger(
        h5_path,
        num_envs=env.num_envs,
        fsr_dim=cfg.fsr_dims,
        q_dim=22,
        action_dim=action_dim,
        qfrc_dim=16,
    )

    raw_policy = _build_registered_policy(task_id, cfg, env.num_envs)
    # If use fingertip control, uncomment the following line to give the policy access to the env for FSR reading.
    raw_policy._env = env  # type: ignore

    with _suppress_mujoco_output():
        obs, _ = env.reset()
    _reset_target_pose(
        env,
        target_mocap_idx,
        base_target_pos,
        palm_body_local_idx,
        cfg,
        device,
        object_profile_randomizer,
    )

    rotator.set_base_position(
        env.sim.data.mocap_pos[:, target_mocap_idx, :].clone()
    )

    prev_fsr_data = torch.zeros((env.num_envs, cfg.fsr_dims), device=env.device)
    full_contact_run = torch.zeros((env.num_envs,), dtype=torch.int32, device=env.device)
    episode_id = torch.zeros((env.num_envs,), dtype=torch.int32, device=env.device)
    fsr_nonzero_steps = 0
    viewer_step_count = 0  # viewer模式下全局步数计数

    try:
        if cfg.viewer == "headless":
            step_iter = range(total_steps)
            if tqdm is not None:
                step_iter = tqdm(step_iter, total=total_steps, desc="Collecting", unit="step")

            for step in step_iter:
                rotator.step(env)
                # 每 N 步随机切换物体转轴（不重置环境）
                if resample_axes_interval > 0 and (step + 1) % resample_axes_interval == 0:
                    rotator.resample_axes()
                    rotator.set_base_position(
                        env.sim.data.mocap_pos[:, target_mocap_idx, :].clone()
                    )
                action = _adapt_action_dim(raw_policy(obs), action_dim)
                if cfg.action_noise_std > 0:
                    action = torch.clamp(
                        action + torch.randn_like(action) * cfg.action_noise_std,
                        min=-1.0,
                        max=1.0,
                    )
                with _suppress_mujoco_output():
                    next_obs, _, _, _, _ = env.step(action)

                fsr_data = _read_fsr_data(
                    env,
                    next_obs,
                    cfg.fsr_dims,
                    source=cfg.fsr_source,
                )
                q_data = env.scene["robot"].data.joint_pos
                qfrc_hand = env.scene["robot"].data.qfrc_actuator[:, 6:22]

                if torch.any(fsr_data > 0):
                    fsr_nonzero_steps += 1

                obj_p = env.sim.data.mocap_pos[:, target_mocap_idx, :].clone()
                obj_q = env.sim.data.mocap_quat[:, target_mocap_idx, :].clone()
                obj_pose = torch.cat((obj_p, obj_q), dim=-1)

                palm_pose_w = env.scene["robot"].data.body_link_pose_w[:, palm_body_local_idx, :]
                palm_pose = palm_pose_w.clone()

                quality, full_contact_run = _compute_contact_quality(
                    fsr_data,
                    prev_fsr_data,
                    full_contact_run,
                    cfg,
                )
                prev_fsr_data = fsr_data.clone()
                fingertip_force = _read_fingertip_force(env)
                fingertip_pos = _read_fingertip_positions(env, palm_body_local_idx)

                logger.log(
                    time=env.sim.data.time,
                    fsr=fsr_data,
                    q=q_data,
                    action=action,
                    qfrc_actuator=qfrc_hand,
                    obj_pose=obj_pose,
                    palm_pose=palm_pose,
                    quality=quality,
                    fingertip_force=fingertip_force,
                    episode_id=episode_id,
                    fingertip_pos=fingertip_pos,
                )

                obs = next_obs

                if reset_interval > 0 and (step + 1) % reset_interval == 0:
                    with _suppress_mujoco_output():
                        obs, _ = env.reset()
                    _reset_target_pose(
                        env,
                        target_mocap_idx,
                        base_target_pos,
                        palm_body_local_idx,
                        cfg,
                        device,
                        object_profile_randomizer,
                    )
                    prev_fsr_data.zero_()
                    full_contact_run.zero_()
                    episode_id += 1
                    rotator.resample_axes()  # 每期随机新转轴+转速
                    rotator.set_base_position(
                        env.sim.data.mocap_pos[:, target_mocap_idx, :].clone()
                    )
        else:
            original_step = env.step
            original_reset = env.reset

            def reset_with_target_pose(*args, **kwargs):
                reset_obs, info = original_reset(*args, **kwargs)
                _reset_target_pose(
                    env,
                    target_mocap_idx,
                    base_target_pos,
                    palm_body_local_idx,
                    cfg,
                    device,
                    object_profile_randomizer,
                )
                prev_fsr_data.zero_()
                full_contact_run.zero_()
                rotator.resample_axes()
                rotator.set_base_position(
                    env.sim.data.mocap_pos[:, target_mocap_idx, :].clone()
                )
                return reset_obs, info

            def step_with_logging(action: torch.Tensor):
                nonlocal fsr_nonzero_steps, prev_fsr_data, full_contact_run, viewer_step_count
                viewer_step_count += 1
                rotator.step(env)
                # 每 N 步随机切换物体转轴（不重置环境）
                if resample_axes_interval > 0 and viewer_step_count % resample_axes_interval == 0:
                    rotator.resample_axes()
                    rotator.set_base_position(
                        env.sim.data.mocap_pos[:, target_mocap_idx, :].clone()
                    )
                next_obs, reward, terminated, truncated, info = original_step(action)

                fsr_data = _read_fsr_data(
                    env,
                    next_obs,
                    cfg.fsr_dims,
                    source=cfg.fsr_source,
                )
                q_data = env.scene["robot"].data.joint_pos
                qfrc_hand = env.scene["robot"].data.qfrc_actuator[:, 6:22]

                if torch.any(fsr_data > 0):
                    fsr_nonzero_steps += 1

                obj_p = env.sim.data.mocap_pos[:, target_mocap_idx, :].clone()
                obj_q = env.sim.data.mocap_quat[:, target_mocap_idx, :].clone()
                obj_pose = torch.cat((obj_p, obj_q), dim=-1)

                palm_pose_w = env.scene["robot"].data.body_link_pose_w[:, palm_body_local_idx, :]
                palm_pose = palm_pose_w.clone()

                quality, full_contact_run_new = _compute_contact_quality(
                    fsr_data,
                    prev_fsr_data,
                    full_contact_run,
                    cfg,
                )
                full_contact_run = full_contact_run_new
                prev_fsr_data = fsr_data.clone()
                fingertip_force = _read_fingertip_force(env)
                fingertip_pos = _read_fingertip_positions(env, palm_body_local_idx)

                logger.log(
                    time=env.sim.data.time,
                    fsr=fsr_data,
                    q=q_data,
                    action=action,
                    qfrc_actuator=qfrc_hand,
                    obj_pose=obj_pose,
                    palm_pose=palm_pose,
                    quality=quality,
                    fingertip_force=fingertip_force,
                    episode_id=torch.zeros(env.num_envs, dtype=torch.int32, device=env.device),
                    fingertip_pos=fingertip_pos,
                )
                return next_obs, reward, terminated, truncated, info

            class PolicyWithActionAdapter:
                def __call__(self, obs):
                    action = _adapt_action_dim(raw_policy(obs), action_dim)
                    if cfg.action_noise_std > 0:
                        action = torch.clamp(
                            action + torch.randn_like(action) * cfg.action_noise_std,
                            min=-1.0,
                            max=1.0,
                        )
                    return action

            env.reset = reset_with_target_pose
            env.step = step_with_logging
            viewer_env = RslRlVecEnvWrapper(
                env,
                clip_actions=getattr(agent_cfg, "clip_actions", None),
            )
            policy = PolicyWithActionAdapter()

            if cfg.viewer == "native":
                NativeMujocoViewer(viewer_env, policy).run()
            else:
                ViserPlayViewer(viewer_env, policy).run()
            viewer_env.close()
    finally:
        joint_names_list = list(env.scene["robot"].joint_names)
        logger.close(source_robot_joint_names=joint_names_list)
        env.close()

    contact_ratio = float(fsr_nonzero_steps) / max(1, int(logger.step_idx))
    if contact_ratio < 0.01:
        print(
            "[WARN] FSR is near-zero for almost all steps "
            f"(nonzero-step ratio={contact_ratio:.4f}). "
            "Likely no contact occurred; consider moving target closer or increasing exploration."
        )

    print(f"[SUCCESS] Saved {logger.step_idx} steps to {h5_path}")


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
        HeadlessCollectConfig,
        args=remaining_args,
        default=HeadlessCollectConfig(),
        prog=sys.argv[0] + f" {chosen_task}",
        config=mjlab.TYRO_FLAGS,
    )
    run_headless_collect(chosen_task, args)


if __name__ == "__main__":
    main()
