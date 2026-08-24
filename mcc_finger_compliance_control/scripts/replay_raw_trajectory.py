"""Directly visualize one recorded raw trajectory without rerunning its controller.

The replay is kinematic: every viewer step writes the recorded arm/hand joint
state and object mocap pose into a single environment, then calls forward
kinematics only.  No dynamics step or controller is executed, so the displayed
motion is the H5 trajectory itself and does not drift away from the recording.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.leaphand.leaphand_mcc_finger_env_cfg import (
    HARD_CONTACT_SOLIMP,
    mcc_finger_contact_env_cfg,
)
from mjlab.viewer import NativeMujocoViewer

from object_catalog import load_object_config


TIP_SITES = ("if_tip", "mf_tip", "rf_tip", "th_tip")
TIP_COLORS = (
    (1.0, 0.30, 0.20, 0.95),
    (0.20, 0.70, 1.0, 0.95),
    (1.0, 0.80, 0.15, 0.95),
    (0.80, 0.30, 1.0, 0.95),
)


def _episode(file: h5py.File, episode_id: int, name: str) -> np.ndarray:
    ids = np.asarray(file["episode_id"], dtype=np.int64)
    locations = np.argwhere(ids == episode_id)
    if locations.size == 0:
        available = np.unique(ids)
        raise ValueError(
            f"episode_id={episode_id} not found; available range="
            f"[{available.min()}, {available.max()}]"
        )
    record_steps = np.asarray(
        [file["record_step"][t, e] for t, e in locations], dtype=np.int64
    )
    locations = locations[np.argsort(record_steps)]
    dataset = file[name]
    return np.stack(
        [dataset[t, e] for t, e in locations], axis=0
    ).astype(np.float32)


def _fixed_object_scale(object_config, file: h5py.File) -> float:
    if "object_scale" in file.attrs:
        return float(file.attrs["object_scale"])
    scale_range = np.asarray(
        object_config.collection.get("size_scale_range", (1.0, 1.0)),
        dtype=np.float64,
    )
    if scale_range.shape != (2,) or not np.isclose(scale_range[0], scale_range[1]):
        raise ValueError(
            "The H5 predates object_scale metadata and its object configuration "
            "uses a randomized scale, so exact replay is ambiguous."
        )
    return float(scale_range[0])


class _KinematicViewerEnv:
    """Minimal viewer adapter whose step intentionally performs no physics."""

    def __init__(self, env: ManagerBasedRlEnv) -> None:
        self.unwrapped = env
        self.cfg = env.cfg
        self.num_envs = env.num_envs

    def get_observations(self) -> dict[str, torch.Tensor]:
        return {}

    def step(self, _action: torch.Tensor) -> None:
        return None

    def reset(self):
        return self.unwrapped.reset()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Direct single-environment replay of a raw collection H5 episode."
    )
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--episode-id", type=int, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--viewer-fps", type=float, default=60.0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Number of recorded frames to replay; zero means through the end.",
    )
    parser.add_argument(
        "--zero-velocity",
        action="store_true",
        help="Write zero qvel instead of the recorded velocity (pose is unchanged).",
    )
    args = parser.parse_args()

    if args.viewer_fps <= 0.0:
        raise ValueError("--viewer-fps must be positive")
    if args.start_frame < 0:
        raise ValueError("--start-frame must be non-negative")
    if args.max_frames < 0:
        raise ValueError("--max-frames must be non-negative")

    with h5py.File(args.file, "r") as file:
        object_id = str(file.attrs["object_id"])
        object_config = load_object_config(object_id)
        object_scale = _fixed_object_scale(object_config, file)
        q = _episode(file, args.episode_id, "q")
        qvel = _episode(file, args.episode_id, "qvel")
        object_pose = _episode(file, args.episode_id, "object_pose_world")
        tip_pose = _episode(file, args.episode_id, "fingertip_pose_world")
        contact_pos = _episode(file, args.episode_id, "fingertip_contact_pos_world")
        contact = _episode(file, args.episode_id, "fingertip_contact") > 0.5
        force = _episode(file, args.episode_id, "fingertip_force_world")
        horizontal_lowest_lock = bool(
            file.attrs.get("horizontal_lowest_point_lock", False)
        )
        if (
            horizontal_lowest_lock
            and "object_lowest_point_anchor_world" in file
            and "fixed_palm_target" in file
        ):
            lowest_anchor = _episode(
                file, args.episode_id, "object_lowest_point_anchor_world"
            )
            fixed_palm_target = _episode(
                file, args.episode_id, "fixed_palm_target"
            )
            lowest_clearance = float(
                file.attrs.get("horizontal_lowest_point_clearance_m", 0.0)
            )
        else:
            lowest_anchor = None
            fixed_palm_target = None
            lowest_clearance = 0.0
        contact_stiffness = float(file.attrs.get("contact_stiffness", 20_000.0))
        contact_damping = float(file.attrs.get("contact_damping", 400.0))
        contact_width = float(file.attrs.get("contact_transition_width_m", 0.002))
        finger_stiffness = float(file.attrs.get("fullhand_finger_stiffness", 35.0))
        finger_damping = float(file.attrs.get("fullhand_finger_damping", 2.5))
        finger_effort = float(file.attrs.get("fullhand_finger_effort_limit", 35.0))
        physics_substeps = int(file.attrs.get("physics_substeps", 10))

    total_frames = len(q)
    if args.start_frame >= total_frames:
        raise ValueError(
            f"--start-frame={args.start_frame} exceeds episode length {total_frames}"
        )
    stop = total_frames
    if args.max_frames:
        stop = min(stop, args.start_frame + args.max_frames)
    selection = slice(args.start_frame, stop)
    q = q[selection]
    qvel = qvel[selection]
    object_pose = object_pose[selection]
    tip_pose = tip_pose[selection]
    contact_pos = contact_pos[selection]
    contact = contact[selection]
    force = force[selection]
    frames = len(q)

    contact_solimp = (
        HARD_CONTACT_SOLIMP[0],
        HARD_CONTACT_SOLIMP[1],
        contact_width,
        HARD_CONTACT_SOLIMP[3],
        HARD_CONTACT_SOLIMP[4],
    )
    env_cfg = mcc_finger_contact_env_cfg(
        num_envs=1,
        play=True,
        object_config=object_config,
        object_scale=object_scale,
        finger_stiffness=finger_stiffness,
        finger_damping=finger_damping,
        finger_effort_limit=finger_effort,
        contact_solref=(-contact_stiffness, -contact_damping),
        contact_solimp=contact_solimp,
        physics_substeps=physics_substeps,
    )
    run_device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env = ManagerBasedRlEnv(cfg=env_cfg, device=run_device)
    env.reset()
    robot = env.scene["robot"]
    target_mocap_idx = int(env.scene["target"].data.indexing.mocap_id)

    # Keep the recorded relative palm/object placement.  In planner_inverse
    # trajectories the first object pose is deliberately reanchored after
    # contact preparation and is therefore *not* object_config.initial_pos.
    # Inferring a source origin from the YAML pose moves the object away from
    # the recorded arm q and produces a visibly dislocated replay.
    replay_origin = env.scene.env_origins[0].detach().cpu().numpy()
    object_pose[:, :3] += replay_origin
    tip_pose[:, :, :3] += replay_origin
    contact_pos += replay_origin

    class ReplayPolicy:
        def __init__(self) -> None:
            self.frame = 0
            self.display_frame = 0

        def reset(self) -> None:
            self.frame = 0
            self.display_frame = 0

        def __call__(self, _obs: dict[str, torch.Tensor]) -> torch.Tensor:
            t = min(self.frame, frames - 1)
            joint_position = torch.as_tensor(q[t], device=env.device).unsqueeze(0)
            joint_velocity = (
                torch.zeros_like(joint_position)
                if args.zero_velocity
                else torch.as_tensor(qvel[t], device=env.device).unsqueeze(0)
            )
            robot.write_joint_state_to_sim(joint_position, joint_velocity)
            env.sim.data.mocap_pos[0, target_mocap_idx, :] = torch.as_tensor(
                object_pose[t, :3], device=env.device
            )
            env.sim.data.mocap_quat[0, target_mocap_idx, :] = torch.as_tensor(
                object_pose[t, 3:7], device=env.device
            )
            env.sim.forward()
            self.display_frame = t
            if t % 100 == 0:
                loaded = contact[t].astype(np.int32).tolist()
                magnitude = np.linalg.norm(force[t], axis=-1).round(2).tolist()
                print(
                    f"[RAW-REPLAY] frame={args.start_frame + t:4d} "
                    f"loaded={loaded} force_N={magnitude}",
                    flush=True,
                )
            self.frame += 1
            return torch.zeros((1, 22), device=env.device)

    policy = ReplayPolicy()
    base_update_visualizers = env.update_visualizers

    def update_visualizers(visualizer) -> None:
        base_update_visualizers(visualizer)
        t = policy.display_frame
        for finger, color in enumerate(TIP_COLORS):
            if contact[t, finger]:
                visualizer.add_sphere(
                    contact_pos[t, finger],
                    radius=0.009,
                    color=(0.10, 1.0, 0.15, 1.0),
                    label=f"{TIP_SITES[finger]}_contact",
                )
            else:
                visualizer.add_sphere(
                    tip_pose[t, finger, :3],
                    radius=0.007,
                    color=(1.0, 0.05, 0.05, 1.0),
                    label=f"{TIP_SITES[finger]}_lost",
                )
            visualizer.add_sphere(
                tip_pose[t, finger, :3], radius=0.003, color=color
            )

    env.update_visualizers = update_visualizers
    viewer_env = _KinematicViewerEnv(env)
    all4 = float(np.mean(np.all(contact, axis=1)))
    ge3 = float(np.mean(np.sum(contact, axis=1) >= 3))
    print(
        f"[RAW-REPLAY] file={args.file} episode={args.episode_id} "
        f"frames={frames} all4={all4:.1%} ge3={ge3:.1%} device={run_device}"
    )
    print("[RAW-REPLAY] green=recorded contact, red=recorded loss")
    try:
        NativeMujocoViewer(
            viewer_env,
            policy,
            frame_rate=args.viewer_fps,
            enable_perturbations=False,
        ).run(num_steps=frames)
    finally:
        env.close()
        # MuJoCo's passive viewer releases its native GLX context on a
        # background render thread.  Exiting Python immediately after the
        # finite replay can let glfw terminate first, after which that thread
        # calls GLFW again and ends in GLFW_NOT_INITIALIZED / SIGSEGV.  The
        # collection viewer uses the same grace period for this driver stack.
        time.sleep(2.0)
        print("[RAW-REPLAY] viewer and simulation resources released")


if __name__ == "__main__":
    main()
