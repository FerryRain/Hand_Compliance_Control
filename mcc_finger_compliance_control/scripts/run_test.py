from __future__ import annotations

import argparse
from dataclasses import asdict

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.leaphand.leaphand_mcc_finger_env_cfg import (
    MCCLeapHandPositionControlCfg,
    mcc_finger_contact_env_cfg,
)
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


TASK_ID = "Leaphand-Finger-MCC-Position-Control"


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize MCC fingertip position control.")
    parser.add_argument("--viewer", choices=("native", "viser"), default="native")
    parser.add_argument("--device", default=None)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument(
        "--rotate-object",
        action="store_true",
        help="Use the same randomized object motion as trajectory collection.",
    )
    parser.add_argument("--motion-start", type=int, default=350)
    parser.add_argument("--motion-length", type=int, default=1800)
    parser.add_argument("--angular-speed-min", type=float, default=0.06)
    parser.add_argument("--angular-speed-max", type=float, default=0.12)
    parser.add_argument(
        "--initial-orientation-mode",
        choices=("uniform", "jitter", "fixed"),
        default="uniform",
    )
    parser.add_argument("--initial-orientation-jitter-deg", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=20260716)
    args = parser.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env_cfg = mcc_finger_contact_env_cfg(num_envs=1, play=True)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    wrapped = RslRlVecEnvWrapper(env)

    rl_cfg = MCCLeapHandPositionControlCfg()
    kwargs = asdict(rl_cfg)
    policy_class = kwargs.pop("policy_class")
    kwargs.pop("device", None)
    controller = policy_class(device=device, num_envs=1, **kwargs)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    target_mocap_idx = int(env.scene["target"].data.indexing.mocap_id)
    nominal_quat = env.sim.data.mocap_quat[:, target_mocap_idx, :].clone()
    if args.initial_orientation_mode == "uniform":
        initial_quat = torch.randn((1, 4), device=device)
        initial_quat /= torch.linalg.vector_norm(
            initial_quat, dim=-1, keepdim=True
        ).clamp_min(1.0e-8)
    elif args.initial_orientation_mode == "jitter":
        jitter_axis = torch.randn((1, 3), device=device)
        jitter_axis /= torch.linalg.vector_norm(
            jitter_axis, dim=-1, keepdim=True
        ).clamp_min(1.0e-8)
        jitter_limit = np.deg2rad(args.initial_orientation_jitter_deg)
        jitter_angle = torch.empty(1, device=device).uniform_(
            -jitter_limit, jitter_limit
        )
        jitter_quat = torch.cat(
            (
                torch.cos(jitter_angle / 2).unsqueeze(-1),
                jitter_axis * torch.sin(jitter_angle / 2).unsqueeze(-1),
            ),
            dim=-1,
        )
        initial_quat = _wxyz_multiply(nominal_quat, jitter_quat)
        initial_quat /= torch.linalg.vector_norm(
            initial_quat, dim=-1, keepdim=True
        ).clamp_min(1.0e-8)
    else:
        initial_quat = nominal_quat
    env.sim.data.mocap_quat[:, target_mocap_idx, :] = initial_quat
    rotation_axis = torch.randn((1, 3), device=device)
    rotation_axis /= torch.linalg.vector_norm(
        rotation_axis, dim=-1, keepdim=True
    ).clamp_min(1.0e-8)
    angular_speed = torch.empty(1, device=device).uniform_(
        args.angular_speed_min, args.angular_speed_max
    )
    dt = float(env_cfg.decimation * env_cfg.sim.mujoco.timestep)

    class DiagnosticPolicy:
        def __init__(self):
            self.step = 0

        def __call__(self, obs):
            moving = (
                args.rotate_object
                and args.motion_start
                <= self.step
                < args.motion_start + args.motion_length
            )
            if moving:
                angle = angular_speed * dt
                delta = torch.cat(
                    (
                        torch.cos(angle / 2).unsqueeze(-1),
                        rotation_axis * torch.sin(angle / 2).unsqueeze(-1),
                    ),
                    dim=-1,
                )
                quat = env.sim.data.mocap_quat[:, target_mocap_idx, :].clone()
                quat = _wxyz_multiply(quat, delta)
                env.sim.data.mocap_quat[:, target_mocap_idx, :] = (
                    quat
                    / torch.linalg.vector_norm(
                        quat, dim=-1, keepdim=True
                    )
                )
            action = controller(obs)
            if self.step % max(args.print_every, 1) == 0:
                forces = obs["finger"][0, :12].reshape(4, 3)
                magnitude = torch.linalg.vector_norm(forces, dim=-1)
                found = []
                for site_name in ("if_tip", "mf_tip", "rf_tip", "th_tip"):
                    sensor = env.scene[f"{site_name}_contact"]
                    sensor_found = sensor.data.found
                    found.append(
                        bool(sensor_found is not None and sensor_found[0].any())
                    )
                debug = controller.last_debug
                ik_error = torch.linalg.vector_norm(
                    debug["tip_x_ik"] - debug["tip_x_ref"], dim=-1
                )[0]
                print(
                    f"[MCC-TIP] step={self.step:05d} "
                    f"force_N={magnitude.detach().cpu().numpy().round(3).tolist()} "
                    f"found={found} moving={moving} "
                    f"ik_err_mm={(ik_error.detach().cpu().numpy()*1000).round(2).tolist()} "
                    f"palm_err_mm={float(debug['palm_tracking_error'][0, 0])*1000:.2f}",
                    flush=True,
                )
            self.step += 1
            return action

    policy = DiagnosticPolicy()
    print(f"[INFO] task={TASK_ID} device={device} viewer={args.viewer}")
    if args.rotate_object:
        print(
            "[INFO] Collection motion enabled | "
            f"speed={float(angular_speed):.4f}rad/s "
            f"axis={rotation_axis[0].cpu().numpy().round(3).tolist()} "
            f"start={args.motion_start} length={args.motion_length} "
            f"initial_orientation={args.initial_orientation_mode}"
            + (
                f" jitter=+/-{args.initial_orientation_jitter_deg:.1f}deg"
                if args.initial_orientation_mode == "jitter"
                else ""
            )
        )
    print("[INFO] 3-D fingertip force is measured but NOT fed into the controller")
    print("[INFO] Native: use Ctrl+C for the most conservative shutdown path")
    try:
        if args.viewer == "native":
            NativeMujocoViewer(wrapped, policy).run()
        else:
            ViserPlayViewer(wrapped, policy).run()
    finally:
        wrapped.close()
        print("[INFO] Simulation resources released")


if __name__ == "__main__":
    main()
