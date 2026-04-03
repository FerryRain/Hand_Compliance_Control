from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Literal

import h5py
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
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import NativeMujocoViewer, ViewerConfig, ViserPlayViewer


LEAP_HAND_XML = Path(
    "./src/mjlab/asset_zoo/robots/xarm6_leap_hand/leap_hand.xml"
)


def _load_hand_only_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(LEAP_HAND_XML))


def _get_fixed_target_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="target_ball")
    body.add_geom(
        name="ball_geom",
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        size=[0.15, 0.08],
        rgba=[0.2, 0.6, 1.0, 1.0],
        mass=1.0,
    )
    return spec


def _dummy_joint_obs(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    return env.scene[asset_cfg.name].data.joint_pos


def _build_hand_replay_env_cfg() -> ManagerBasedRlEnvCfg:
    robot_cfg = EntityCfg(
        spec_fn=_load_hand_only_spec,
        articulation=EntityArticulationInfoCfg(
            actuators=(
                BuiltinPositionActuatorCfg(
                    target_names_expr=(r"^[0-9]+$",),
                    stiffness=20.0,
                    damping=2.0,
                    effort_limit=500.0,
                ),
            ),
        ),
        init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), joint_pos={"13": 1.57}),
    )

    target_cfg = EntityCfg(
        spec_fn=_get_fixed_target_spec,
        init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    observations = {
        "policy": ObservationGroupCfg(
            {
                "joint_pos": ObservationTermCfg(
                    func=_dummy_joint_obs,
                    params={"asset_cfg": SceneEntityCfg("robot")},
                )
            }
        )
    }

    actions: dict[str, ActionTermCfg] = {
        "hand_delta": JointRelativePositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=0.05,
            use_default_offset=False,
        )
    }

    return ManagerBasedRlEnvCfg(
        decimation=5,
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
            body_name="palm_lower",
            distance=0.8,
        ),
        episode_length_s=1e10,
    )


def _select_hand_joint_slice(q_step: np.ndarray) -> np.ndarray:
    # Recorded q can be either 22D (xarm6 + hand) or 16D (hand only).
    if q_step.shape[-1] == 22:
        return q_step[..., -16:]
    if q_step.shape[-1] == 16:
        return q_step
    raise ValueError(f"Unsupported q dimension {q_step.shape[-1]}; expected 16 or 22.")


def _decode_str_attr(values: np.ndarray) -> list[str]:
    decoded: list[str] = []
    for value in values.tolist():
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8"))
        else:
            decoded.append(str(value))
    return decoded


def _build_q_mapper(
    source_joint_names: list[str] | None,
    target_joint_names: tuple[str, ...],
    q_dim: int,
):
    if q_dim == 16:
        print("[INFO] q has 16 dims; using direct hand joint replay.")
        return lambda q_step: q_step

    if q_dim == 22 and source_joint_names:
        name_to_src = {name: i for i, name in enumerate(source_joint_names)}
        missing = [name for name in target_joint_names if name not in name_to_src]
        if not missing:
            src_ids = [name_to_src[name] for name in target_joint_names]
            print("[INFO] Using name-based q remapping from source to replay hand model.")
            return lambda q_step: q_step[src_ids]

    if q_dim == 22:
        print("[WARN] Missing source joint metadata; fallback to last 16 dims.")
        return lambda q_step: _select_hand_joint_slice(q_step)

    raise ValueError(f"Unsupported q dimension {q_dim}; expected 16 or 22.")


def run_replay(
    file_path: str,
    device: str | None = None,
    env_idx: int = 0,
    viewer: Literal["native", "viser"] = "native",
) -> None:
    if not file_path:
        print("[ERROR] No inverted H5 file provided.")
        return

    with h5py.File(file_path, "r") as f:
        q_traj = np.array(f["q"], dtype=np.float32)
        palm_traj = np.array(f["palm_pose_world"], dtype=np.float32)
        obj_traj = np.array(f["obj_pose_world"], dtype=np.float32)
        num_steps = int(q_traj.shape[0])
        source_joint_names = None
        if "source_robot_joint_names" in f.attrs:
            source_joint_names = _decode_str_attr(
                np.asarray(f.attrs["source_robot_joint_names"])
            )

    num_envs_in_file = int(q_traj.shape[1])
    if not (0 <= env_idx < num_envs_in_file):
        raise ValueError(
            f"env_idx={env_idx} out of range [0, {num_envs_in_file - 1}]"
        )

    run_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env_cfg = _build_hand_replay_env_cfg()
    env = ManagerBasedRlEnv(cfg=env_cfg, device=run_device)
    viewer_env = RslRlVecEnvWrapper(env)
    action_dim = int(env.action_manager.total_action_dim)
    target_joint_names = tuple(env.scene["robot"].joint_names)
    q_mapper = _build_q_mapper(
        source_joint_names=source_joint_names,
        target_joint_names=target_joint_names,
        q_dim=int(q_traj.shape[-1]),
    )

    dist = np.linalg.norm(palm_traj[:, env_idx, :3] - obj_traj[:, env_idx, :3], axis=-1)
    print(
        "[INFO] palm-object distance stats "
        f"mean={float(dist.mean()):.4f}, min={float(dist.min()):.4f}, max={float(dist.max()):.4f}"
    )

    class ReplayPolicy:
        def __init__(self):
            self.t = 0

        def __call__(self, obs):
            _ = obs
            t = self.t % num_steps
            env_idx_in_h5 = env_idx

            palm_pose = torch.from_numpy(
                palm_traj[t, env_idx_in_h5]
            ).to(env.device)
            root_state = torch.cat(
                [palm_pose, torch.zeros(6, device=env.device, dtype=palm_pose.dtype)]
            ).unsqueeze(0)
            env.scene["robot"].write_root_state_to_sim(root_state)

            q_vals = torch.from_numpy(
                q_mapper(q_traj[t, env_idx_in_h5])
            ).to(env.device)
            zero_vel = torch.zeros_like(q_vals).unsqueeze(0)
            env.scene["robot"].write_joint_state_to_sim(
                position=q_vals.unsqueeze(0),
                velocity=zero_vel,
            )

            env.sim.forward()

            if env.sim.model.nmocap > 0:
                obj_pose = torch.from_numpy(
                    obj_traj[t, env_idx_in_h5]
                ).to(env.device)
                env.sim.data.mocap_pos[:, 0, :] = obj_pose[:3]
                env.sim.data.mocap_quat[:, 0, :] = obj_pose[3:]

            self.t += 1
            return torch.zeros((1, action_dim), device=env.device)

    print(f"[INFO] Replaying from: {os.path.basename(file_path)}")
    try:
        if viewer == "native":
            NativeMujocoViewer(viewer_env, ReplayPolicy()).run()
        else:
            ViserPlayViewer(viewer_env, ReplayPolicy()).run()
    finally:
        viewer_env.close()


def _resolve_input_file(explicit_path: str | None) -> str | None:
    if explicit_path:
        return explicit_path
    inverted_files = glob.glob("./finger_copliance_control/data/*_inverted.h5")
    if not inverted_files:
        return None
    return max(inverted_files, key=os.path.getctime)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay inverted hand trajectory with leap_hand.xml (hand-only)."
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to *_inverted.h5. If omitted, use latest one in data folder.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device, e.g. cuda:0 or cpu. Default: auto-select.",
    )
    parser.add_argument(
        "--env-idx",
        type=int,
        default=0,
        help="Environment index from multi-env H5 to replay.",
    )
    parser.add_argument(
        "--viewer",
        type=str,
        choices=("native", "viser"),
        default="native",
        help="Viewer backend. Use 'viser' to avoid GLX/OpenGL context issues.",
    )
    args = parser.parse_args()

    file_path = _resolve_input_file(args.file)
    if file_path is None:
        print("[ERROR] No *_inverted.h5 files found in ./finger_copliance_control/data/")
        return

    run_replay(
        file_path=file_path,
        device=args.device,
        env_idx=args.env_idx,
        viewer=args.viewer,
    )


if __name__ == "__main__":
    main()