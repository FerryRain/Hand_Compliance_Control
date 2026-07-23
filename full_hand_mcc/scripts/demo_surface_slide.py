"""Five reachable surface points + full-hand MCC surface-sliding demo.

Run from the repository root:

    python full_hand_mcc/scripts/demo_surface_slide.py \
        --variant hybrid_force_position --viewer native

The five target points are kept on the capsule analytically.  Every proposed
slide increment is then accepted only after the real 22-DoF model reaches all
five points within the configured tolerance.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.leaphand.full_hand_mcc_core import MCC_VARIANTS
from mjlab.tasks.leaphand.full_hand_mcc_geometry import (
    capsule_project,
    rotate_about_capsule_axis,
)
from mjlab.tasks.leaphand.leaphand_full_hand_mcc_env_cfg import (
    FULL_HAND_CAPSULE_HALF_HEIGHT,
    FULL_HAND_CAPSULE_RADIUS,
    FivePointReachabilitySolver,
    FullHandMCCControlCfg,
    full_hand_mcc_env_cfg,
)
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


CAPSULE_RADIUS = FULL_HAND_CAPSULE_RADIUS
CAPSULE_HALF_HEIGHT = FULL_HAND_CAPSULE_HALF_HEIGHT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=MCC_VARIANTS, default="hybrid_force_position")
    parser.add_argument(
        "--viewer", choices=("native", "viser", "video"), default="native"
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--slide-speed", type=float, default=0.06, help="rad/s")
    parser.add_argument("--motion-start", type=int, default=100)
    parser.add_argument("--ik-tolerance-mm", type=float, default=4.0)
    parser.add_argument("--contact-failure-window", type=int, default=20)
    parser.add_argument("--min-contact-force-n", type=float, default=0.10)
    parser.add_argument("--min-contact-ratio", type=float, default=0.99)
    parser.add_argument("--contact-settle-frames", type=int, default=3)
    parser.add_argument("--contact-calibration-start", type=int, default=15)
    parser.add_argument("--contact-tracking-radius-rad", type=float, default=0.12)
    parser.add_argument("--finger-force-n", type=float, default=12.0)
    parser.add_argument("--arm-mcc-correction-rad", type=float, default=0.002)
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("full_hand_mcc/output/full_hand_mcc_surface_slide.mp4"),
    )
    args = parser.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env_cfg = full_hand_mcc_env_cfg(num_envs=1, play=True)
    if args.viewer == "video":
        env_cfg.viewer.width = args.width
        env_cfg.viewer.height = args.height
    env = ManagerBasedRlEnv(
        cfg=env_cfg,
        device=device,
        render_mode="rgb_array" if args.viewer == "video" else None,
    )
    wrapped = RslRlVecEnvWrapper(env)
    cfg = FullHandMCCControlCfg(
        variant=args.variant,
        device=device,
        finger_desired_force=args.finger_force_n,
        arm_mcc_correction_limit=args.arm_mcc_correction_rad,
    )
    kwargs = asdict(cfg)
    policy_class = kwargs.pop("policy_class")
    kwargs.pop("device", None)
    controller = policy_class(device=device, num_envs=1, **kwargs)
    reachability = FivePointReachabilitySolver(
        tolerance=args.ik_tolerance_mm / 1000.0
    )
    target_mocap_idx = int(env.scene["target"].data.indexing.mocap_id)
    dt = float(env_cfg.decimation * env_cfg.sim.mujoco.timestep)

    class SurfaceSlidePolicy:
        def __init__(self) -> None:
            self.step = 0
            self.targets: np.ndarray | None = None
            self.kinematic_targets: np.ndarray | None = None
            self.normals: np.ndarray | None = None
            self.reachable_q: np.ndarray | None = None
            self.last_residual = np.full(5, np.inf)
            self.rejected_steps = 0
            self.actual_points = np.zeros((5, 3))
            self.tracking_error = np.full(5, np.inf)
            self.surface_error = np.full(5, np.inf)
            self.joint_error = np.full(22, np.inf)
            self.tactile_force = np.zeros(4)
            self.contact_frames = np.zeros(4, dtype=np.int64)
            self.evaluated_frames = 0
            self.bad_contact_streak = 0
            self.contact_settle_streak = 0
            self.contact_calibrated = False

        def _object_pose(self, obs) -> tuple[np.ndarray, np.ndarray]:
            _ = obs
            center = (
                env.sim.data.mocap_pos[0, target_mocap_idx]
                .detach()
                .cpu()
                .numpy()
            )
            quat_wxyz = (
                env.sim.data.mocap_quat[0, target_mocap_idx]
                .detach()
                .cpu()
                .numpy()
            )
            quat_xyzw = np.roll(quat_wxyz, -1)
            return center, R.from_quat(quat_xyzw).as_matrix()

        def _initialize(self, obs) -> None:
            q = obs["palm"][0, :22].detach().cpu().numpy()
            center, rotation = self._object_pose(obs)
            live_points = reachability.forward_points(q)
            surface_targets, normals = capsule_project(
                live_points,
                center,
                rotation,
                CAPSULE_RADIUS,
                CAPSULE_HALF_HEIGHT,
            )
            result = reachability.solve(surface_targets, q)
            if not result.accepted:
                raise RuntimeError(
                    "Initial five-point surface target is unreachable: "
                    f"residual_mm={(result.residual_m * 1000).round(2).tolist()}"
                )
            self.targets = surface_targets
            self.kinematic_targets = surface_targets.copy()
            self.normals = normals
            self.reachable_q = result.joint_position
            self.last_residual = result.residual_m

        def _capture_contact_calibration(
            self,
            obs,
            live_q: np.ndarray,
            live_points: np.ndarray,
        ) -> None:
            center, rotation = self._object_pose(obs)
            surface_targets, normals = capsule_project(
                live_points,
                center,
                rotation,
                CAPSULE_RADIUS,
                CAPSULE_HALF_HEIGHT,
            )
            # The input points are the true object-surface contacts.  The
            # kinematic sites sit inside the fingertip meshes, so their
            # collision-consistent targets retain the measured per-pad
            # standoff instead of being pulled through the capsule.
            self.targets = surface_targets
            self.kinematic_targets = live_points.copy()
            self.normals = normals
            self.reachable_q = live_q.copy()
            self.last_residual = np.zeros(5)
            self.surface_error = np.linalg.norm(
                live_points - surface_targets, axis=1
            )
            self.contact_calibrated = True
            controller.reset()
            controller.fingers.nominal_tracking_radius = (
                args.contact_tracking_radius_rad
            )
            print(
                "[CONTACT-CALIBRATION] captured collision-consistent "
                f"site_standoff_mm="
                f"{(self.surface_error[1:] * 1000).round(2).tolist()} "
                f"tactile_force_N={self.tactile_force.round(2).tolist()}",
                flush=True,
            )

        def _try_slide(self, obs) -> None:
            assert self.targets is not None
            assert self.kinematic_targets is not None
            assert self.reachable_q is not None
            center, rotation = self._object_pose(obs)
            full_step = args.slide_speed * dt
            for fraction in (1.0, 0.5, 0.25, 0.125):
                candidate_surface = rotate_about_capsule_axis(
                    self.targets, center, rotation, full_step * fraction
                )
                candidate_surface, normals = capsule_project(
                    candidate_surface,
                    center,
                    rotation,
                    CAPSULE_RADIUS,
                    CAPSULE_HALF_HEIGHT,
                )
                candidate_kinematic = rotate_about_capsule_axis(
                    self.kinematic_targets,
                    center,
                    rotation,
                    full_step * fraction,
                )
                result = reachability.solve(
                    candidate_kinematic, self.reachable_q
                )
                if result.accepted:
                    self.targets = candidate_surface
                    self.kinematic_targets = candidate_kinematic
                    self.normals = normals
                    self.reachable_q = result.joint_position
                    self.last_residual = result.residual_m
                    return
            self.rejected_steps += 1

        def __call__(self, obs):
            if self.targets is None:
                self._initialize(obs)
            assert self.targets is not None
            assert self.kinematic_targets is not None
            assert self.normals is not None
            assert self.reachable_q is not None
            live_q = obs["palm"][0, :22].detach().cpu().numpy()
            self.tactile_force = (
                torch.linalg.vector_norm(
                    obs["finger"][0, :12].reshape(4, 3), dim=-1
                )
                .detach()
                .cpu()
                .numpy()
            )
            self.actual_points = reachability.forward_points(live_q)
            if (
                not self.contact_calibrated
                and self.step >= args.contact_calibration_start
            ):
                if bool(np.all(self.tactile_force >= args.min_contact_force_n)):
                    self.contact_settle_streak += 1
                else:
                    self.contact_settle_streak = 0
                if self.contact_settle_streak >= args.contact_settle_frames:
                    self._capture_contact_calibration(
                        obs, live_q, self.actual_points
                    )
            if self.step >= args.motion_start:
                if not self.contact_calibrated:
                    raise RuntimeError(
                        "All four fingertips did not establish real contact "
                        "before surface sliding"
                    )
                self._try_slide(obs)
            assert self.targets is not None
            assert self.kinematic_targets is not None
            assert self.reachable_q is not None
            self.joint_error = self.reachable_q - live_q
            self.tracking_error = np.linalg.norm(
                self.actual_points - self.kinematic_targets, axis=1
            )
            center, rotation = self._object_pose(obs)
            projected_actual, _ = capsule_project(
                self.actual_points,
                center,
                rotation,
                CAPSULE_RADIUS,
                CAPSULE_HALF_HEIGHT,
            )
            self.surface_error = np.linalg.norm(
                self.actual_points - projected_actual, axis=1
            )
            if self.step >= args.motion_start:
                tip_contact = self.tactile_force >= args.min_contact_force_n
                self.contact_frames += tip_contact.astype(np.int64)
                self.evaluated_frames += 1
                if bool(np.all(tip_contact)):
                    self.bad_contact_streak = 0
                else:
                    self.bad_contact_streak += 1
                if self.bad_contact_streak > args.contact_failure_window:
                    raise RuntimeError(
                        "Continuous fingertip contact validation failed: "
                        f"site_standoff_mm="
                        f"{(self.surface_error[1:] * 1000).round(2).tolist()} "
                        f"kinematic_tracking_error_mm="
                        f"{(self.tracking_error[1:] * 1000).round(2).tolist()} "
                        f"tactile_force_N="
                        f"{self.tactile_force.round(2).tolist()} "
                        f"arm_joint_error_rad="
                        f"{self.joint_error[:6].round(3).tolist()} "
                        f"finger_joint_error_rad="
                        f"{self.joint_error[6:].round(3).tolist()}"
                    )
            target_t = torch.as_tensor(
                self.targets[None], device=device, dtype=torch.float32
            )
            normal_t = torch.as_tensor(
                self.normals[None], device=device, dtype=torch.float32
            )
            kinematic_target_t = torch.as_tensor(
                self.kinematic_targets[None], device=device, dtype=torch.float32
            )
            joint_reference_t = (
                torch.as_tensor(
                    self.reachable_q[None],
                    device=device,
                    dtype=torch.float32,
                )
                if self.contact_calibrated
                else None
            )
            action = controller(
                obs,
                contact_points=target_t,
                surface_normals=normal_t,
                joint_reference=joint_reference_t,
                kinematic_points=kinematic_target_t,
            )
            if self.step % max(args.print_every, 1) == 0:
                debug = controller.last_debug
                motor_force = torch.linalg.vector_norm(
                    debug["tip_force_from_motors"][0], dim=-1
                )
                print(
                    f"[FULL-HAND-MCC] step={self.step:05d} variant={args.variant} "
                    f"reachable_err_mm={(self.last_residual * 1000).round(2).tolist()} "
                    f"actual_tip_surface_mm="
                    f"{(self.surface_error[1:] * 1000).round(2).tolist()} "
                    f"actual_tip_kinematic_target_mm="
                    f"{(self.tracking_error[1:] * 1000).round(2).tolist()} "
                    f"max_joint_error_rad="
                    f"[{np.max(np.abs(self.joint_error[:6])):.3f},"
                    f"{np.max(np.abs(self.joint_error[6:])):.3f}] "
                    f"tip_force_N={motor_force.cpu().numpy().round(2).tolist()} "
                    f"tactile_force_N={self.tactile_force.round(2).tolist()} "
                    f"tank={float(debug['energy_tank'][0]):.3f} "
                    f"rejected_slide_steps={self.rejected_steps}",
                    flush=True,
                )
            self.step += 1
            return action

    policy = SurfaceSlidePolicy()

    base_update_visualizers = env.update_visualizers

    def update_demo_visualizers(visualizer) -> None:
        base_update_visualizers(visualizer)
        if policy.targets is None or policy.normals is None:
            return
        colors = (
            (0.2, 1.0, 0.2, 0.95),
            (1.0, 0.3, 0.2, 0.95),
            (0.2, 0.7, 1.0, 0.95),
            (1.0, 0.8, 0.15, 0.95),
            (0.8, 0.3, 1.0, 0.95),
        )
        for point, normal, color in zip(policy.targets, policy.normals, colors):
            visualizer.add_sphere(point, radius=0.007, color=color)
            visualizer.add_arrow(
                point,
                point + 0.035 * normal,
                color=color,
                width=0.003,
            )

    env.update_visualizers = update_demo_visualizers
    print(
        f"[INFO] Full-hand MCC demo | variant={args.variant} "
        f"device={device} viewer={args.viewer}"
    )
    print("[INFO] Targets are surface-projected and URDF/joint-limit checked every step")
    try:
        if args.viewer == "native":
            NativeMujocoViewer(wrapped, policy).run()
        elif args.viewer == "viser":
            ViserPlayViewer(wrapped, policy).run()
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            obs, _ = wrapped.reset()
            next_frame_time = 0.0
            frames_written = 0
            with imageio.get_writer(
                args.output,
                fps=args.fps,
                codec="libx264",
                quality=8,
                macro_block_size=None,
            ) as writer:
                for step in range(args.steps):
                    action = policy(obs)
                    obs, _, _, _ = wrapped.step(action)
                    sim_time = (step + 1) * dt
                    if sim_time + 1.0e-9 >= next_frame_time:
                        frame = env.render()
                        if frame is not None:
                            writer.append_data(np.asarray(frame, dtype=np.uint8))
                            frames_written += 1
                        next_frame_time += 1.0 / args.fps
            if policy.evaluated_frames <= 0:
                raise RuntimeError("No sliding frames were evaluated for contact")
            contact_ratio = (
                policy.contact_frames / float(policy.evaluated_frames)
            )
            if np.any(contact_ratio < args.min_contact_ratio):
                raise RuntimeError(
                    "Fingertip continuous-contact ratio below required "
                    f"{args.min_contact_ratio:.1%}: "
                    f"{contact_ratio.round(4).tolist()}"
                )
            print(
                f"[VIDEO] saved={args.output.resolve()} frames={frames_written} "
                f"duration_s={args.steps * dt:.2f} fps={args.fps:.1f} "
                f"tip_contact_ratio={contact_ratio.round(4).tolist()} "
                f"final_tip_site_standoff_mm="
                f"{(policy.surface_error[1:] * 1000).round(2).tolist()}",
                flush=True,
            )
    finally:
        wrapped.close()


if __name__ == "__main__":
    main()
