from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import h5py
import mujoco
import numpy as np
import torch
from tqdm.auto import tqdm

from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor import ContactSensor
from mjlab.tasks.leaphand.leaphand_mcc_finger_env_cfg import (
    MCCLeapHandPositionControlCfg,
    mcc_finger_contact_env_cfg,
)


TASK_ID = "Leaphand-Finger-MCC-Position-Control"
TIP_SITES = ("if_tip", "mf_tip", "rf_tip", "th_tip")


def _fingertip_force_world(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Read the four live contact-sensor resultants in world coordinates."""
    forces: list[torch.Tensor] = []
    for site_name in TIP_SITES:
        sensor = env.scene[f"{site_name}_contact"]
        if not isinstance(sensor, ContactSensor):
            raise TypeError(f"{site_name}_contact is not a ContactSensor")
        force = sensor.data.force
        if force is None:
            forces.append(torch.zeros((env.num_envs, 3), device=env.device))
            continue
        if sensor.data.found is not None:
            force = torch.where(
                sensor.data.found.unsqueeze(-1) > 0,
                force,
                torch.zeros_like(force),
            )
        forces.append(force.sum(dim=1))
    return torch.stack(forces, dim=1)


def _wxyz_multiply(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = lhs.unbind(-1)
    w2, x2, y2, z2 = rhs.unbind(-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def _find_local_body_index(env: ManagerBasedRlEnv, suffix: str) -> int:
    names = [body.name or "" for body in env.scene["robot"].data.indexing.bodies]
    for index, name in enumerate(names):
        if name == suffix or name.endswith(f"/{suffix}"):
            return index
    raise ValueError(f"Body {suffix!r} not found; available={names}")


def _find_local_site_index(env: ManagerBasedRlEnv, suffix: str) -> int:
    names = [site.name or "" for site in env.scene["robot"].data.indexing.sites]
    for index, name in enumerate(names):
        if name == suffix or name.endswith(f"/{suffix}"):
            return index
    raise ValueError(f"Site {suffix!r} not found; available={names}")


class StreamingH5:
    SHAPES = {
        "time": (),
        "episode_id": (),
        "episode_step": (),
        "q": (22,),
        "qvel": (22,),
        "q_hand": (16,),
        "q_pre": (16,),
        "q_ref": (16,),
        "arm_q_ref": (6,),
        "fixed_palm_target": (6,),
        "palm_control_pos_world": (3,),
        "action": (22,),
        "object_pose_world": (7,),
        "palm_pose_world": (7,),
        "fingertip_pose_world": (4, 7),
        "fingertip_force_world": (4, 3),
        "fingertip_force_local": (4, 3),
        "fingertip_contact_pos_world": (4, 3),
        "fingertip_contact_normal_world": (4, 3),
        "fingertip_contact_dist": (4,),
        "fingertip_collision_found": (4,),
        "fingertip_contact": (4,),
        "tip_x_des_world": (4, 3),
        "tip_x_ref_world": (4, 3),
        "tip_x_ik_world": (4, 3),
        "tip_x_des_palm": (4, 3),
        "tip_x_ref_palm": (4, 3),
        "object_angular_velocity_world": (3,),
    }

    def __init__(self, path: Path, num_envs: int):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = h5py.File(path, "w")
        self.num_envs = num_envs
        self.step = 0
        self.datasets: dict[str, h5py.Dataset] = {}
        for name, tail in self.SHAPES.items():
            dtype = "i4" if name in ("episode_id", "episode_step") else "f4"
            shape = (0, num_envs, *tail)
            maxshape = (None, num_envs, *tail)
            chunks = (min(100, 2500), num_envs, *tail)
            self.datasets[name] = self.file.create_dataset(
                name, shape=shape, maxshape=maxshape, chunks=chunks, dtype=dtype
            )

    def append(self, values: dict[str, np.ndarray | torch.Tensor | float | int]) -> None:
        next_size = self.step + 1
        for dataset in self.datasets.values():
            dataset.resize((next_size, *dataset.shape[1:]))
        for name, value in values.items():
            if torch.is_tensor(value):
                value = value.detach().cpu().numpy()
            array = np.asarray(value)
            if array.ndim == len(self.SHAPES[name]):
                array = np.broadcast_to(array, (self.num_envs, *array.shape))
            self.datasets[name][self.step] = array
        self.step = next_size

    def truncate(self, size: int) -> None:
        """Roll back a rejected candidate trajectory."""
        if size < 0 or size > self.step:
            raise ValueError(f"Invalid truncate size {size} for current size {self.step}")
        for dataset in self.datasets.values():
            dataset.resize((size, *dataset.shape[1:]))
        self.step = size

    def close(self, attrs: dict[str, object]) -> None:
        for key, value in attrs.items():
            self.file.attrs[key] = value
        self.file.attrs["total_steps"] = self.step
        self.file.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect MCC fingertip position-teacher data.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--trajectory-length", type=int, default=2500)
    parser.add_argument("--max-trajectories", type=int, default=5)
    parser.add_argument("--motion-start", type=int, default=350)
    parser.add_argument("--motion-length", type=int, default=1800)
    parser.add_argument(
        "--record-start-step",
        type=int,
        default=None,
        help="First saved step; defaults to motion-start so prep is excluded.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="Maximum candidate attempts; defaults to 10x max-trajectories.",
    )
    parser.add_argument(
        "--online-quality-gate",
        action="store_true",
        help="Reject whole trajectories online. Off by default for fast raw collection.",
    )
    parser.add_argument("--angular-speed-min", type=float, default=0.35)
    parser.add_argument("--angular-speed-max", type=float, default=0.70)
    parser.add_argument(
        "--initial-orientation-jitter-deg",
        type=float,
        default=10.0,
        help=(
            "Random rotation about the environment's nominal object orientation. "
            "This deliberately avoids an unconstrained SO(3) reset that can make "
            "the fixed pre-grasp geometrically unable to contact all fingertips."
        ),
    )
    parser.add_argument(
        "--contact-threshold",
        type=float,
        default=0.05,
        help="Minimum full-fingertip 3-D resultant force magnitude in newtons.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--filename", default=None)
    args = parser.parse_args()

    if args.motion_start < 0 or args.motion_start >= args.trajectory_length:
        raise ValueError("motion-start must lie inside the trajectory")
    record_start = (
        args.motion_start if args.record_start_step is None else args.record_start_step
    )
    if record_start < 0 or record_start >= args.trajectory_length:
        raise ValueError("record-start-step must lie inside the trajectory")
    if args.online_quality_gate and args.num_envs != 1:
        raise ValueError(
            "Online trajectory filtering requires --num-envs 1; collect raw in parallel "
            "and run filter_trajectories.py instead."
        )
    batches_needed = int(np.ceil(args.max_trajectories / args.num_envs))
    collected_target = batches_needed * args.num_envs
    max_attempts = (
        args.max_attempts or max(10 * args.max_trajectories, 1)
        if args.online_quality_gate
        else batches_needed
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    env_cfg = mcc_finger_contact_env_cfg(num_envs=args.num_envs, play=True)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    rl_cfg = MCCLeapHandPositionControlCfg()
    kwargs = asdict(rl_cfg)
    policy_class = kwargs.pop("policy_class")
    kwargs.pop("device", None)
    policy = policy_class(device=device, num_envs=args.num_envs, **kwargs)

    target_mocap_idx = int(env.scene["target"].data.indexing.mocap_id)
    palm_idx = _find_local_body_index(env, "palm_lower")
    tip_indices = [_find_local_site_index(env, name) for name in TIP_SITES]
    dt = float(env_cfg.decimation * env_cfg.sim.mujoco.timestep)

    filename = args.filename or f"mcc_tip_{datetime.now():%Y%m%d_%H%M%S}"
    output = Path("mcc_finger_compliance_control/data/trajectories") / f"{filename}.h5"
    logger = StreamingH5(output, args.num_envs)

    saved_frames_per_trajectory = args.trajectory_length - record_start
    total_frames = saved_frames_per_trajectory * collected_target
    print(f"[INFO] task={TASK_ID} device={device} accepted_frames={total_frames} output={output}")
    print("[INFO] controller measured-force input is ZERO; fingertip force is record-only")
    if args.online_quality_gate:
        print(
            "[INFO] ONLINE strict gate: all 4 complete fingertip geoms need "
            f"|F_3D| >= {args.contact_threshold:.3f} N on every saved frame"
        )
    else:
        print(
            "[INFO] RAW mode: no online rejection; run filter_trajectories.py later | "
            f"requested={args.max_trajectories}, actual={collected_target}, "
            f"parallel_envs={args.num_envs}"
        )

    accepted = 0
    attempts = 0
    progress = tqdm(
        total=max_attempts * args.trajectory_length,
        desc="Collecting raw" if not args.online_quality_gate else "Collecting strict",
        unit="step",
        dynamic_ncols=True,
    )
    try:
        while accepted < collected_target and attempts < max_attempts:
            candidate_id = attempts
            attempts += 1
            obs, _ = env.reset()
            policy.reset()
            initial_pos = env.sim.data.mocap_pos[:, target_mocap_idx, :].clone()
            nominal_quat = env.sim.data.mocap_quat[:, target_mocap_idx, :].clone()
            jitter_axes = torch.randn((args.num_envs, 3), device=device)
            jitter_axes /= torch.linalg.vector_norm(
                jitter_axes, dim=-1, keepdim=True
            ).clamp_min(1.0e-8)
            jitter_limit = np.deg2rad(args.initial_orientation_jitter_deg)
            jitter_angle = torch.empty(args.num_envs, device=device).uniform_(
                -jitter_limit, jitter_limit
            )
            jitter_quat = torch.cat(
                (
                    torch.cos(jitter_angle / 2).unsqueeze(-1),
                    jitter_axes * torch.sin(jitter_angle / 2).unsqueeze(-1),
                ),
                dim=-1,
            )
            initial_quat = _wxyz_multiply(nominal_quat, jitter_quat)
            initial_quat /= torch.linalg.vector_norm(
                initial_quat, dim=-1, keepdim=True
            )
            env.sim.data.mocap_pos[:, target_mocap_idx, :] = initial_pos
            env.sim.data.mocap_quat[:, target_mocap_idx, :] = initial_quat

            axes = torch.randn((args.num_envs, 3), device=device)
            axes /= torch.linalg.vector_norm(axes, dim=-1, keepdim=True)
            speeds = torch.empty(args.num_envs, device=device).uniform_(
                args.angular_speed_min, args.angular_speed_max
            )
            candidate_start = logger.step
            quality_forces: list[np.ndarray] = []
            quality_contacts: list[np.ndarray] = []

            for episode_step in range(args.trajectory_length):
                moving = args.motion_start <= episode_step < min(
                    args.motion_start + args.motion_length, args.trajectory_length
                )
                if moving:
                    angle = speeds * dt
                    delta = torch.cat(
                        (torch.cos(angle / 2).unsqueeze(-1), axes * torch.sin(angle / 2).unsqueeze(-1)),
                        dim=-1,
                    )
                    quat = env.sim.data.mocap_quat[:, target_mocap_idx, :].clone()
                    quat = _wxyz_multiply(quat, delta)
                    env.sim.data.mocap_quat[:, target_mocap_idx, :] = (
                        quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True)
                    )

                action = policy(obs)
                obs, *_ = env.step(action)
                progress.update(1)
                robot = env.scene["robot"]
                body_pose = robot.data.body_link_pose_w
                palm_pose = body_pose[:, palm_idx, :]
                site_pose = robot.data.site_pose_w
                tip_pose = torch.stack([site_pose[:, idx, :] for idx in tip_indices], dim=1)
                tip_force_local = obs["finger"][:, :12].reshape(args.num_envs, 4, 3)
                tip_force = _fingertip_force_world(env)
                contact_pos = torch.zeros_like(tip_force)
                contact_normal = torch.zeros_like(tip_force)
                contact_dist = torch.zeros((args.num_envs, 4), device=device)
                contact_found = torch.zeros(
                    (args.num_envs, 4), dtype=torch.bool, device=device
                )
                for tip_id, site_name in enumerate(TIP_SITES):
                    sensor_data = env.scene[f"{site_name}_contact"].data
                    if sensor_data.force is not None:
                        slot_magnitude = torch.linalg.vector_norm(
                            sensor_data.force, dim=-1
                        )
                    else:
                        slot_count = (
                            sensor_data.found.shape[1]
                            if sensor_data.found is not None
                            else 1
                        )
                        slot_magnitude = torch.zeros(
                            (args.num_envs, slot_count), device=device
                        )
                    if sensor_data.found is not None:
                        slot_found = sensor_data.found > 0
                        slot_magnitude = torch.where(
                            slot_found,
                            slot_magnitude,
                            torch.full_like(slot_magnitude, -1.0),
                        )
                        found = torch.any(slot_found, dim=1)
                    else:
                        found = torch.zeros(
                            args.num_envs, dtype=torch.bool, device=device
                        )
                    contact_found[:, tip_id] = found
                    strongest_slot = torch.argmax(slot_magnitude, dim=1)
                    batch_ids = torch.arange(args.num_envs, device=device)
                    if sensor_data.pos is not None:
                        selected_pos = sensor_data.pos[batch_ids, strongest_slot]
                        contact_pos[:, tip_id] = torch.where(
                            found.unsqueeze(-1), selected_pos, torch.zeros_like(selected_pos)
                        )
                    if sensor_data.normal is not None:
                        selected_normal = sensor_data.normal[batch_ids, strongest_slot]
                        contact_normal[:, tip_id] = torch.where(
                            found.unsqueeze(-1), selected_normal, torch.zeros_like(selected_normal)
                        )
                    if sensor_data.dist is not None:
                        selected_dist = sensor_data.dist[batch_ids, strongest_slot]
                        contact_dist[:, tip_id] = torch.where(
                            found, selected_dist, torch.zeros_like(selected_dist)
                        )
                obj_pose = torch.cat(
                    (
                        env.sim.data.mocap_pos[:, target_mocap_idx, :],
                        env.sim.data.mocap_quat[:, target_mocap_idx, :],
                    ),
                    dim=-1,
                )
                debug = policy.last_debug
                tip_force_magnitude = torch.linalg.vector_norm(tip_force, dim=-1)
                if episode_step >= record_start:
                    quality_forces.append(
                        tip_force_magnitude[0].detach().cpu().numpy().copy()
                    )
                    quality_contacts.append(
                        contact_found[0].detach().cpu().numpy().copy()
                    )
                    logger.append(
                        {
                        "time": np.full(args.num_envs, logger.step * dt),
                        "episode_id": (
                            accepted + np.arange(args.num_envs, dtype=np.int32)
                        ),
                        "episode_step": np.full(args.num_envs, episode_step, dtype=np.int32),
                        "q": robot.data.joint_pos,
                        "qvel": robot.data.joint_vel,
                        "q_hand": robot.data.joint_pos[:, 6:22],
                        "q_pre": debug["q_pre"],
                        "q_ref": debug["q_ref"],
                        "arm_q_ref": debug["palm_arm_q_ref"],
                        "fixed_palm_target": debug["fixed_palm_target"],
                        "palm_control_pos_world": debug["palm_site_pos"],
                        "action": action,
                        "object_pose_world": obj_pose,
                        "palm_pose_world": palm_pose,
                        "fingertip_pose_world": tip_pose,
                        "fingertip_force_world": tip_force,
                        "fingertip_force_local": tip_force_local,
                        "fingertip_contact_pos_world": contact_pos,
                        "fingertip_contact_normal_world": contact_normal,
                        "fingertip_contact_dist": contact_dist,
                        "fingertip_collision_found": contact_found.float(),
                        "fingertip_contact": (
                            contact_found
                            & (tip_force_magnitude >= args.contact_threshold)
                        ).float(),
                        "tip_x_des_world": debug["tip_x_des"],
                        "tip_x_ref_world": debug["tip_x_ref"],
                        "tip_x_ik_world": debug["tip_x_ik"],
                        "tip_x_des_palm": debug["tip_x_des_palm"],
                        "tip_x_ref_palm": debug["tip_x_ref_palm"],
                        "object_angular_velocity_world": axes * speeds.unsqueeze(-1) if moving else torch.zeros_like(axes),
                        }
                    )

            if not args.online_quality_gate:
                accepted += args.num_envs
                progress.set_postfix(
                    batch=candidate_id + 1,
                    trajectories=f"{accepted}/{collected_target}",
                )
                continue

            quality = np.asarray(quality_forces, dtype=np.float32)
            contact_quality = np.asarray(quality_contacts, dtype=bool)
            loaded_contact = contact_quality & (quality >= args.contact_threshold)
            per_tip_contact = np.mean(loaded_contact, axis=0)
            all_four_contact = np.all(loaded_contact, axis=1)
            strict_pass = bool(np.all(all_four_contact))
            min_force = np.min(quality, axis=0)
            first_loss = np.flatnonzero(~all_four_contact)
            first_loss_step = (
                int(record_start + first_loss[0]) if first_loss.size else None
            )
            summary = (
                f"candidate={candidate_id} pass={strict_pass} "
                f"tip_contact={np.round(per_tip_contact, 4).tolist()} "
                f"all4={float(np.mean(all_four_contact)):.4f} "
                f"min_force_N={np.round(min_force, 3).tolist()} "
                f"first_loss_step={first_loss_step}"
            )
            if strict_pass:
                accepted += 1
                print(f"[ACCEPT] {summary} accepted={accepted}/{args.max_trajectories}")
            else:
                logger.truncate(candidate_start)
                print(f"[REJECT] {summary}")
    finally:
        progress.close()
        logger.close(
            {
                "task_id": TASK_ID,
                "schema_version": "mcc_tip_v1",
                "force_feedback_enabled": False,
                "force_frame": "world",
                "pose_quaternion_order": "wxyz",
                "control_dt": dt,
                "motion_start": args.motion_start,
                "record_start_step": record_start,
                "trajectory_length": args.trajectory_length,
                "num_trajectories": accepted,
                "candidate_attempts": attempts,
                "strict_four_tip_continuous_contact": args.online_quality_gate,
                "contact_gate": "full_fingertip_geom_found_and_3d_force_magnitude",
                "contact_threshold": args.contact_threshold,
                "initial_orientation_jitter_deg": (
                    args.initial_orientation_jitter_deg
                ),
                "seed": args.seed,
            }
        )
        env.close()
    if accepted < collected_target:
        print(
            f"[WARNING] saved {accepted}/{collected_target} trajectories "
            f"after {attempts} attempts: {output}"
        )
    else:
        mode = "strict" if args.online_quality_gate else "raw"
        print(f"[SUCCESS] saved {accepted} {mode} trajectories to {output}")


if __name__ == "__main__":
    main()
