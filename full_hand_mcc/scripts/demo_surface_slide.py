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
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.leaphand.full_hand_mcc_core import MCC_VARIANTS
from mjlab.tasks.leaphand.full_hand_mcc_geometry import (
    capsule_meridian_coordinates,
    capsule_meridian_targets,
    capsule_project,
)
from mjlab.tasks.leaphand.leaphand_full_hand_mcc_env_cfg import (
    FULL_HAND_CAPSULE_HALF_HEIGHT,
    FULL_HAND_CAPSULE_RADIUS,
    FivePointReachabilitySolver,
    FullHandMCCControlCfg,
    full_hand_mcc_env_cfg,
)
from mjlab.tasks.leaphand.leaphand_mcc_finger_env_cfg import MCC_TIP_NAMES
import mjlab.tasks.leaphand.leaphand_full_hand_mcc_env_cfg as full_hand_env_module
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


CAPSULE_RADIUS = FULL_HAND_CAPSULE_RADIUS
CAPSULE_HALF_HEIGHT = FULL_HAND_CAPSULE_HALF_HEIGHT


def main() -> None:
    global CAPSULE_RADIUS, CAPSULE_HALF_HEIGHT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=MCC_VARIANTS, default="hybrid_force_position")
    parser.add_argument(
        "--viewer", choices=("native", "viser", "video"), default="native"
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--axial-travel-m", type=float, default=0.20)
    parser.add_argument(
        "--object-radius-m",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--object-half-height-m",
        type=float,
        default=0.26,
    )
    parser.add_argument(
        "--object-center-x-m",
        type=float,
        default=0.85,
    )
    parser.add_argument(
        "--object-center-y-m",
        type=float,
        default=0.0003,
    )
    parser.add_argument(
        "--object-center-z-m",
        type=float,
        default=0.7077,
    )
    parser.add_argument(
        "--planner",
        choices=(
            "staged_inward",
            "continuous_inward",
            "meridian_inward",
            "adaptive_surface_mpc",
            "circumferential_surface_mpc",
        ),
        default="adaptive_surface_mpc",
    )
    parser.add_argument("--waypoint-count", type=int, default=5)
    parser.add_argument("--stage-move-fraction", type=float, default=0.78)
    parser.add_argument("--surface-preload-mm", type=float, default=2.0)
    parser.add_argument("--max-plan-joint-step-rad", type=float, default=0.03)
    parser.add_argument("--motion-start", type=int, default=1000)
    parser.add_argument("--ik-tolerance-mm", type=float, default=5.0)
    parser.add_argument("--ik-max-iterations", type=int, default=160)
    parser.add_argument("--palm-ik-weight", type=float, default=5.0)
    parser.add_argument(
        "--palm-path-tolerance-mm",
        type=float,
        default=50.0,
        help=(
            "Maximum deviation from the nominal non-contact palm path. "
            "The accepted palm plan point is always replaced by the actual "
            "URDF-reachable point."
        ),
    )
    parser.add_argument("--mpc-keyframes", type=int, default=40)
    parser.add_argument("--mpc-max-nfev", type=int, default=120)
    parser.add_argument("--mpc-progress-tolerance-mm", type=float, default=4.0)
    parser.add_argument(
        "--mpc-intermediate-progress-tolerance-mm",
        type=float,
        default=4.0,
    )
    parser.add_argument("--mpc-normal-tolerance-mm", type=float, default=3.0)
    parser.add_argument("--contact-failure-window", type=int, default=20)
    parser.add_argument("--min-contact-force-n", type=float, default=0.10)
    parser.add_argument("--min-contact-ratio", type=float, default=0.99)
    parser.add_argument(
        "--max-contact-penetration-mm",
        type=float,
        default=1.0,
        help=(
            "Reject the video if MuJoCo reports deeper fingertip/object "
            "penetration at any time."
        ),
    )
    parser.add_argument("--contact-settle-frames", type=int, default=3)
    parser.add_argument("--contact-calibration-start", type=int, default=15)
    parser.add_argument("--preshape-frames", type=int, default=0)
    parser.add_argument("--object-approach-frames", type=int, default=300)
    parser.add_argument("--object-retreat-distance-m", type=float, default=0.20)
    parser.add_argument(
        "--planning-center-shift-m",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--axial-direction",
        type=float,
        choices=(-1.0, 1.0),
        default=-1.0,
        help="-1 follows the xArm-reachable downward object meridian.",
    )
    parser.add_argument("--contact-tracking-radius-rad", type=float, default=0.05)
    parser.add_argument(
        "--precontact-tracking-radius-rad",
        type=float,
        default=0.02,
    )
    parser.add_argument("--contact-search-step-rad", type=float, default=0.02)
    parser.add_argument("--contact-search-limit-rad", type=float, default=0.30)
    parser.add_argument("--finger-force-n", type=float, default=12.0)
    parser.add_argument(
        "--finger-normal-preload-mm",
        type=float,
        default=0.0,
        help="Constant inward Cartesian pad preload around the URDF plan.",
    )
    parser.add_argument(
        "--finger-normal-compliance-mm-per-n",
        type=float,
        default=0.05,
        help="Motor-force error to inward Cartesian pad displacement gain.",
    )
    parser.add_argument(
        "--finger-max-release-correction-rad",
        type=float,
        default=0.0,
    )
    parser.add_argument("--palm-force-n", type=float, default=0.0)
    parser.add_argument("--arm-mcc-correction-rad", type=float, default=0.0)
    parser.add_argument(
        "--arm-trajectory-tracking-gain",
        type=float,
        default=2.0,
        help="Joint-space lead compensation for the loaded moving arm.",
    )
    parser.add_argument(
        "--finger-trajectory-tracking-gain",
        type=float,
        default=0.5,
        help="Joint-space lead compensation for loaded finger servos.",
    )
    parser.add_argument(
        "--arm-servo-load-scale",
        type=float,
        default=1.0,
        help="Scale the calibrated arm command-to-loaded-position offset.",
    )
    parser.add_argument(
        "--finger-servo-load-scale",
        type=float,
        default=1.5,
        help="Scale the calibrated finger contact preload offset.",
    )
    parser.add_argument("--print-every", type=int, default=300)
    parser.add_argument("--steps", type=int, default=7000)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera-distance-m", type=float, default=1.12)
    parser.add_argument("--camera-elevation-deg", type=float, default=-18.0)
    parser.add_argument("--camera-azimuth-deg", type=float, default=325.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "full_hand_mcc/outputs/"
            "thick_object_slow_surface_slide.mp4"
        ),
    )
    parser.add_argument(
        "--plan-output",
        type=Path,
        default=Path(
            "full_hand_mcc/outputs/"
            "thick_object_slow_surface_slide_plan.npz"
        ),
    )
    parser.add_argument(
        "--reuse-plan",
        type=Path,
        default=None,
        help="Reuse a previously validated NPZ plan for controller tuning.",
    )
    args = parser.parse_args()
    if args.steps <= args.motion_start:
        raise ValueError("--steps must be greater than --motion-start")
    if args.axial_travel_m <= 0.0:
        raise ValueError("--axial-travel-m must be positive")
    if args.object_radius_m <= 0.0:
        raise ValueError("--object-radius-m must be positive")
    if args.object_half_height_m < 0.0:
        raise ValueError("--object-half-height-m cannot be negative")
    if args.waypoint_count < 1:
        raise ValueError("--waypoint-count must be at least one")
    if not 0.0 < args.stage_move_fraction <= 1.0:
        raise ValueError("--stage-move-fraction must be in (0, 1]")
    if args.surface_preload_mm < 0.0:
        raise ValueError("--surface-preload-mm cannot be negative")
    if args.mpc_keyframes < 2:
        raise ValueError("--mpc-keyframes must be at least two")
    if args.palm_path_tolerance_mm < args.ik_tolerance_mm:
        raise ValueError(
            "--palm-path-tolerance-mm must be at least --ik-tolerance-mm"
        )
    if args.contact_search_step_rad <= 0.0:
        raise ValueError("--contact-search-step-rad must be positive")
    if args.contact_search_limit_rad <= 0.0:
        raise ValueError("--contact-search-limit-rad must be positive")
    if args.finger_max_release_correction_rad < 0.0:
        raise ValueError(
            "--finger-max-release-correction-rad cannot be negative"
        )
    if args.finger_normal_preload_mm < 0.0:
        raise ValueError("--finger-normal-preload-mm cannot be negative")
    if args.finger_normal_compliance_mm_per_n < 0.0:
        raise ValueError(
            "--finger-normal-compliance-mm-per-n cannot be negative"
        )
    if args.arm_trajectory_tracking_gain < 0.0:
        raise ValueError("--arm-trajectory-tracking-gain cannot be negative")
    if args.finger_trajectory_tracking_gain < 0.0:
        raise ValueError(
            "--finger-trajectory-tracking-gain cannot be negative"
        )
    if args.arm_servo_load_scale < 0.0:
        raise ValueError("--arm-servo-load-scale cannot be negative")
    if args.finger_servo_load_scale < 0.0:
        raise ValueError("--finger-servo-load-scale cannot be negative")
    if args.max_contact_penetration_mm < 0.0:
        raise ValueError("--max-contact-penetration-mm cannot be negative")

    CAPSULE_RADIUS = args.object_radius_m
    CAPSULE_HALF_HEIGHT = args.object_half_height_m
    full_hand_env_module.FULL_HAND_CAPSULE_RADIUS = CAPSULE_RADIUS
    full_hand_env_module.FULL_HAND_CAPSULE_HALF_HEIGHT = CAPSULE_HALF_HEIGHT

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env_cfg = full_hand_mcc_env_cfg(num_envs=1, play=True)
    env_cfg.scene.entities["target"].init_state.pos = (
        args.object_center_x_m,
        args.object_center_y_m,
        args.object_center_z_m,
    )
    env_cfg.viewer.distance = args.camera_distance_m
    env_cfg.viewer.elevation = args.camera_elevation_deg
    env_cfg.viewer.azimuth = args.camera_azimuth_deg
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
        finger_max_release_correction=(
            args.finger_max_release_correction_rad
        ),
        palm_desired_force=args.palm_force_n,
        arm_mcc_correction_limit=args.arm_mcc_correction_rad,
    )
    kwargs = asdict(cfg)
    policy_class = kwargs.pop("policy_class")
    kwargs.pop("device", None)
    controller = policy_class(device=device, num_envs=1, **kwargs)
    controller.fingers.normal_preload_m = (
        args.finger_normal_preload_mm / 1000.0
    )
    controller.fingers.normal_compliance_m_per_n = (
        args.finger_normal_compliance_mm_per_n / 1000.0
    )
    controller.fingers.nominal_tracking_radius = (
        args.precontact_tracking_radius_rad
    )
    reachability = FivePointReachabilitySolver(
        tolerance=args.ik_tolerance_mm / 1000.0,
        max_iterations=args.ik_max_iterations,
        palm_weight=args.palm_ik_weight,
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
            self.contact_distance_m = np.full(4, np.nan)
            self.max_penetration_m = np.zeros(4)
            self.contact_frames = np.zeros(4, dtype=np.int64)
            self.evaluated_frames = 0
            self.bad_contact_streak = 0
            self.contact_settle_streak = 0
            self.contact_calibrated = False
            self.motor_force_recalibrated = False
            self.max_force_correction_rad = 0.0
            self.precontact_closure = np.zeros(16, dtype=np.float32)
            self.last_command_q: np.ndarray | None = None
            self.contact_servo_offset_q = np.zeros(22, dtype=np.float32)
            self.plan_surface: np.ndarray | None = None
            self.plan_kinematic: np.ndarray | None = None
            self.plan_normals: np.ndarray | None = None
            self.plan_q: np.ndarray | None = None
            self.plan_residual: np.ndarray | None = None
            self.plan_distance: np.ndarray | None = None
            self.plan_index = -1
            self.plan_direction = 1.0
            self.planned_axial_travel = 0.0
            self.executed_axial_travel = 0.0
            self.object_final_center: np.ndarray | None = None
            self.object_retreat_center: np.ndarray | None = None

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
            self.object_final_center = center.copy()
            self.object_retreat_center = (
                center + args.object_retreat_distance_m * rotation[:, 0]
            )
            live_points = reachability.forward_points(q)
            surface_targets, normals = capsule_project(
                live_points,
                center,
                rotation,
                CAPSULE_RADIUS,
                CAPSULE_HALF_HEIGHT,
            )
            approach_targets = surface_targets.copy()
            approach_targets[0] = live_points[0]
            approach = reachability.solve(approach_targets, q)
            if not approach.accepted:
                raise RuntimeError(
                    "Tactile pre-contact IK is unreachable: "
                    f"residual_mm="
                    f"{(approach.residual_m * 1000).round(2).tolist()}"
                )
            self.targets = surface_targets
            # Start from the simulator's current, joint-limit-valid posture.
            # Three pads normally already touch the capsule; moving all five
            # sites through a projected IK target before tactile calibration
            # unnecessarily releases those contacts.  The per-finger tactile
            # search below moves only pads that are actually separated.
            self.kinematic_targets = approach.achieved_points.copy()
            self.normals = normals
            self.reachable_q = approach.joint_position.copy()
            self.last_residual = approach.residual_m.copy()

        def _update_object_approach(self) -> None:
            if (
                self.object_final_center is None
                or self.object_retreat_center is None
            ):
                return
            phase = min(
                max(self.step - args.preshape_frames, 0)
                / float(max(args.object_approach_frames, 1)),
                1.0,
            )
            smooth_phase = phase * phase * (3.0 - 2.0 * phase)
            center = (
                (1.0 - smooth_phase) * self.object_retreat_center
                + smooth_phase * self.object_final_center
            )
            env.sim.data.mocap_pos[0, target_mocap_idx] = torch.as_tensor(
                center,
                device=env.sim.data.mocap_pos.device,
                dtype=env.sim.data.mocap_pos.dtype,
            )

        def _capture_contact_calibration(
            self,
            obs,
            live_q: np.ndarray,
            live_points: np.ndarray,
        ) -> None:
            center, rotation = self._object_pose(obs)
            center = (
                center
                + args.planning_center_shift_m * rotation[:, 2]
            )
            env.sim.data.mocap_pos[0, target_mocap_idx] = torch.as_tensor(
                center,
                device=env.sim.data.mocap_pos.device,
                dtype=env.sim.data.mocap_pos.dtype,
            )
            self.object_final_center = center.copy()
            self.object_retreat_center = center.copy()
            surface_targets, normals = capsule_project(
                live_points,
                center,
                rotation,
                CAPSULE_RADIUS,
                CAPSULE_HALF_HEIGHT,
            )
            # Preserve the servo deflection that produced the measured
            # contact forces as feed-forward, while planning from the true
            # loaded configuration.  Using the unloaded command itself as the
            # physical MPC state corrupts surface progress by several mm.
            contact_command_q = (
                self.last_command_q.copy()
                if self.last_command_q is not None
                else live_q.copy()
            )
            calibrated_offset = contact_command_q - live_q
            self.contact_servo_offset_q[:6] = (
                args.arm_servo_load_scale * calibrated_offset[:6]
            )
            self.contact_servo_offset_q[6:22] = (
                args.finger_servo_load_scale * calibrated_offset[6:22]
            )
            self.targets = surface_targets
            self.kinematic_targets = live_points.copy()
            self.normals = normals
            self.reachable_q = live_q.copy()
            self.last_residual = np.zeros(5)
            self.surface_error = np.linalg.norm(
                live_points - surface_targets, axis=1
            )
            self._build_axial_plan(center, rotation)
            self.contact_calibrated = True
            motor_baseline = torch.linalg.vector_norm(
                controller.last_debug["tip_force_from_motors"][0],
                dim=-1,
            ).detach().cpu().numpy()
            controller.reset()
            controller.fingers.calibrate_motor_force_setpoint(motor_baseline)
            controller.fingers.nominal_tracking_radius = (
                args.contact_tracking_radius_rad
            )
            print(
                "[CONTACT-CALIBRATION] captured collision-consistent "
                f"site_standoff_mm="
                f"{(self.surface_error[1:] * 1000).round(2).tolist()} "
                f"tactile_force_N={self.tactile_force.round(2).tolist()} "
                f"motor_force_baseline_N="
                f"{motor_baseline.round(2).tolist()} "
                f"servo_deflection_rad="
                f"{self.contact_servo_offset_q.round(3).tolist()}",
                flush=True,
            )

        def _build_axial_plan(
            self,
            center: np.ndarray,
            rotation: np.ndarray,
        ) -> None:
            assert self.targets is not None
            assert self.kinematic_targets is not None
            assert self.reachable_q is not None
            if args.planner == "circumferential_surface_mpc":
                self._build_circumferential_surface_mpc_plan(
                    center=center,
                    rotation=rotation,
                    frame_count=args.steps - args.motion_start,
                )
                return
            if args.reuse_plan is not None:
                cached = np.load(args.reuse_plan)
                joint_plan = np.asarray(
                    cached["joint_positions_rad"], dtype=np.float32
                )
                expected_frames = args.steps - args.motion_start
                if joint_plan.shape != (expected_frames, 22):
                    raise RuntimeError(
                        "Cached plan frame shape mismatch: "
                        f"got={joint_plan.shape} "
                        f"expected={(expected_frames, 22)}"
                    )
                self.plan_surface = np.asarray(
                    cached["surface_points_m"], dtype=np.float32
                )
                self.plan_kinematic = np.asarray(
                    cached["kinematic_points_m"], dtype=np.float32
                )
                _, cached_normals = capsule_project(
                    self.plan_kinematic.reshape(-1, 3),
                    center,
                    rotation,
                    CAPSULE_RADIUS,
                    CAPSULE_HALF_HEIGHT,
                )
                self.plan_normals = cached_normals.reshape(
                    expected_frames, 5, 3
                )
                self.plan_q = joint_plan
                self.plan_residual = np.asarray(
                    cached["progress_residual_m"], dtype=np.float32
                )
                self.plan_distance = np.asarray(
                    cached["axial_distance_m"], dtype=np.float32
                )
                self.planned_axial_travel = float(self.plan_distance[-1])
                start_delta = float(
                    np.max(np.abs(joint_plan[0] - self.reachable_q))
                )
                if start_delta > args.max_plan_joint_step_rad:
                    raise RuntimeError(
                        "Cached plan does not match calibrated start pose: "
                        f"max_joint_delta={start_delta:.5f}rad"
                    )
                print(
                    "[AXIAL-PLAN] reused validated plan | "
                    f"frames={expected_frames} "
                    f"travel_m={self.planned_axial_travel:.4f} "
                    f"start_joint_delta_rad={start_delta:.5f} "
                    f"source={args.reuse_plan.resolve()}",
                    flush=True,
                )
                return
            start_arc, azimuth = capsule_meridian_coordinates(
                self.targets,
                center,
                rotation,
                CAPSULE_RADIUS,
                CAPSULE_HALF_HEIGHT,
            )
            start_surface, _, start_frames = capsule_meridian_targets(
                start_arc,
                azimuth,
                center,
                rotation,
                CAPSULE_RADIUS,
                CAPSULE_HALF_HEIGHT,
            )
            standoff_components = np.einsum(
                "nji,nj->ni",
                start_frames,
                self.kinematic_targets - start_surface,
            )
            total_arc = np.pi * CAPSULE_RADIUS + 2.0 * CAPSULE_HALF_HEIGHT
            # The palm root is a non-contact arm/MCC reference.  Choose and
            # bound the surface direction from the four physical contacts.
            direction = float(args.axial_direction)
            available = (
                total_arc - float(start_arc[1:].max())
                if direction > 0.0
                else float(start_arc[1:].min())
            )
            if args.axial_travel_m > available - 1.0e-6:
                raise RuntimeError(
                    "Requested end-to-end travel exceeds the capsule surface: "
                    f"requested={args.axial_travel_m:.4f}m "
                    f"available={available:.4f}m"
                )
            frame_count = args.steps - args.motion_start
            surface_plan = np.zeros((frame_count, 5, 3), dtype=np.float32)
            kinematic_plan = np.zeros_like(surface_plan)
            normal_plan = np.zeros_like(surface_plan)
            joint_plan = np.zeros((frame_count, 22), dtype=np.float32)
            residual_plan = np.zeros((frame_count, 5), dtype=np.float32)
            distance_plan = np.zeros(frame_count, dtype=np.float32)
            palm_pose_error_plan = np.zeros((frame_count, 2), dtype=np.float32)
            seed = self.reachable_q.copy()
            start_kinematic = self.kinematic_targets.copy()
            axis_world = rotation[:, 2]
            preload = args.surface_preload_mm / 1000.0
            start_palm_position, start_palm_rotation = (
                reachability.forward_palm_pose(self.reachable_q)
            )

            if args.planner == "adaptive_surface_mpc":
                self._build_adaptive_surface_mpc_plan(
                    center=center,
                    rotation=rotation,
                    start_arc=start_arc,
                    start_azimuth=azimuth,
                    start_surface=start_surface,
                    start_frames=start_frames,
                    direction=direction,
                    available=available,
                    frame_count=frame_count,
                )
                return

            for frame in range(frame_count):
                phase = (frame + 1) / float(frame_count)
                if args.planner == "staged_inward":
                    segment_phase = min(
                        phase * args.waypoint_count,
                        float(args.waypoint_count),
                    )
                    segment = min(
                        int(segment_phase),
                        args.waypoint_count - 1,
                    )
                    local_phase = segment_phase - segment
                    move_phase = min(
                        local_phase / args.stage_move_fraction,
                        1.0,
                    )
                    smooth_local = move_phase * move_phase * (
                        3.0 - 2.0 * move_phase
                    )
                    smooth_phase = (
                        segment + smooth_local
                    ) / args.waypoint_count
                else:
                    smooth_phase = phase * phase * (3.0 - 2.0 * phase)
                distance = args.axial_travel_m * smooth_phase
                if args.planner == "meridian_inward":
                    arc = start_arc + direction * distance
                    surface, normals, frames = capsule_meridian_targets(
                        arc,
                        azimuth,
                        center,
                        rotation,
                        CAPSULE_RADIUS,
                        CAPSULE_HALF_HEIGHT,
                    )
                    kinematic = (
                        surface
                        + np.einsum(
                            "nij,nj->ni",
                            frames,
                            standoff_components,
                        )
                        - preload * normals
                    )
                else:
                    translation = direction * distance * axis_world
                    surface, normals = capsule_project(
                        start_surface + translation,
                        center,
                        rotation,
                        CAPSULE_RADIUS,
                        CAPSULE_HALF_HEIGHT,
                    )
                    # Preserve the calibrated hand shape and translate it as a
                    # rigid five-site group.  Projected surface points remain
                    # exact visualization/input targets; the kinematic sites
                    # are intentionally biased inward to maintain contact.
                    kinematic = (
                        start_kinematic
                        + translation
                        - preload * normals
                    )
                palm_translation = direction * distance * axis_world
                palm_pose = reachability.solve_palm_pose(
                    start_palm_position + palm_translation,
                    start_palm_rotation,
                    seed,
                )
                # This arm-only pose solve is a seed, not an acceptance
                # constraint.  Even its best sub-millimetre result can unlock
                # the strict simultaneous five-point solve.
                result = reachability.solve(
                    kinematic,
                    palm_pose.joint_position,
                )
                if not result.accepted:
                    result = reachability.solve(kinematic, seed)
                if not result.accepted:
                    raise RuntimeError(
                        "End-to-end plan is not simultaneously reachable: "
                        f"frame={frame}/{frame_count} "
                        f"distance_m={distance:.4f} "
                        f"residual_mm="
                        f"{(result.residual_m * 1000).round(2).tolist()}"
                    )
                q = result.joint_position
                if np.any(q < reachability.lower - 1.0e-8) or np.any(
                    q > reachability.upper + 1.0e-8
                ):
                    raise RuntimeError(
                        f"Planner produced a joint-limit violation at frame {frame}"
                    )
                surface_plan[frame] = surface
                kinematic_plan[frame] = kinematic
                normal_plan[frame] = normals
                joint_plan[frame] = q
                residual_plan[frame] = result.residual_m
                distance_plan[frame] = distance
                palm_pose_error_plan[frame] = (
                    palm_pose.position_error_m,
                    palm_pose.orientation_error_rad,
                )
                seed = q

            self.plan_surface = surface_plan
            self.plan_kinematic = kinematic_plan
            self.plan_normals = normal_plan
            self.plan_q = joint_plan
            self.plan_residual = residual_plan
            self.plan_distance = distance_plan
            self.plan_direction = direction
            self.planned_axial_travel = float(distance_plan[-1])
            q_with_seed = np.vstack((self.reachable_q[None], joint_plan))
            max_joint_step = float(np.max(np.abs(np.diff(q_with_seed, axis=0))))
            if max_joint_step > args.max_plan_joint_step_rad:
                raise RuntimeError(
                    "Planned joint step exceeds the configured bound: "
                    f"observed={max_joint_step:.5f}rad "
                    f"limit={args.max_plan_joint_step_rad:.5f}rad"
                )
            start_local = (rotation.T @ (surface_plan[0] - center).T).T
            end_local = (rotation.T @ (surface_plan[-1] - center).T).T
            args.plan_output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                args.plan_output,
                surface_points_m=surface_plan,
                kinematic_points_m=kinematic_plan,
                joint_positions_rad=joint_plan,
                residual_m=residual_plan,
                axial_distance_m=distance_plan,
                planner=np.asarray(args.planner),
                surface_preload_mm=np.asarray(args.surface_preload_mm),
                object_radius_m=np.asarray(CAPSULE_RADIUS),
                object_half_height_m=np.asarray(CAPSULE_HALF_HEIGHT),
                max_joint_step_rad=np.asarray(max_joint_step),
                palm_pose_error=palm_pose_error_plan,
                start_surface_local_m=start_local,
                end_surface_local_m=end_local,
            )
            print(
                "[AXIAL-PLAN] all frames simultaneously reachable | "
                f"planner={args.planner} "
                f"frames={frame_count} travel_m={self.planned_axial_travel:.4f} "
                f"preload_mm={args.surface_preload_mm:.2f} "
                f"max_joint_step_rad={max_joint_step:.5f} "
                f"direction={direction:+.0f} "
                f"start_z_m={start_local[:, 2].round(4).tolist()} "
                f"end_z_m={end_local[:, 2].round(4).tolist()} "
                f"max_residual_mm="
                f"{float(residual_plan.max() * 1000):.2f} "
                f"saved={args.plan_output.resolve()}",
                flush=True,
            )

        def _build_circumferential_surface_mpc_plan(
            self,
            *,
            center: np.ndarray,
            rotation: np.ndarray,
            frame_count: int,
        ) -> None:
            """Move the rigid five-point constellation around a thick capsule.

            The four tactile pads remain on their initial cylindrical rings.
            The palm-root point is rotated with the hand as a kinematic
            reference, but is never treated as a physical contact.
            """

            assert self.targets is not None
            assert self.kinematic_targets is not None
            assert self.reachable_q is not None
            start_surface = self.targets.copy()
            start_points = self.kinematic_targets.copy()
            start_local = (rotation.T @ (start_surface - center).T).T
            if np.any(
                np.abs(start_local[1:, 2])
                > CAPSULE_HALF_HEIGHT - 1.0e-4
            ):
                raise RuntimeError(
                    "Circumferential surface MPC requires all fingertip "
                    "contacts on the capsule cylinder, not an end cap: "
                    f"local_z_m={start_local[1:, 2].round(4).tolist()}"
                )

            keyframe_count = min(args.mpc_keyframes, frame_count)
            coarse_distance = np.linspace(
                0.0,
                args.axial_travel_m,
                keyframe_count + 1,
            )
            coarse_q = np.zeros((keyframe_count + 1, 22), dtype=np.float64)
            coarse_q[0] = self.reachable_q
            coarse_residual = np.zeros((keyframe_count + 1, 5), dtype=np.float64)
            coarse_nfev = np.zeros(keyframe_count + 1, dtype=np.int32)
            previous_q = self.reachable_q.copy()
            preload = args.surface_preload_mm / 1000.0

            def rotated_targets(
                distance: float,
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
                theta = (
                    float(args.axial_direction)
                    * distance
                    / CAPSULE_RADIUS
                )
                world_rotation = (
                    rotation
                    @ R.from_rotvec((0.0, 0.0, theta)).as_matrix()
                    @ rotation.T
                )
                surface = (
                    center
                    + (world_rotation @ (start_surface - center).T).T
                )
                points = (
                    center
                    + (world_rotation @ (start_points - center).T).T
                )
                _, normals = capsule_project(
                    surface,
                    center,
                    rotation,
                    CAPSULE_RADIUS,
                    CAPSULE_HALF_HEIGHT,
                )
                preload_fraction = min(
                    distance / max(0.025, args.axial_travel_m),
                    1.0,
                )
                points[1:] -= (
                    preload_fraction * preload * normals[1:]
                )
                return surface, points, normals

            for keyframe in range(1, keyframe_count + 1):
                distance = float(coarse_distance[keyframe])
                _, desired_points, _ = rotated_targets(distance)
                result = reachability.solve(desired_points, previous_q)
                tip_error = float(result.residual_m[1:].max())
                palm_error = float(result.residual_m[0])
                if (
                    tip_error > args.ik_tolerance_mm / 1000.0
                    or palm_error > args.palm_path_tolerance_mm / 1000.0
                ):
                    raise RuntimeError(
                        "Circumferential surface MPC found an unreachable "
                        f"five-point state at keyframe={keyframe}/"
                        f"{keyframe_count}, distance_m={distance:.4f}, "
                        f"residual_mm="
                        f"{(result.residual_m * 1000).round(2).tolist()}"
                    )
                coarse_q[keyframe] = result.joint_position
                coarse_residual[keyframe, 1:] = result.residual_m[1:]
                coarse_nfev[keyframe] = result.iterations
                previous_q = result.joint_position
                print(
                    "[CIRCUMFERENTIAL-MPC] "
                    f"keyframe={keyframe:02d}/{keyframe_count} "
                    f"surface_travel_m={distance:.4f} "
                    f"five_point_error_mm="
                    f"{(result.residual_m * 1000).round(2).tolist()} "
                    f"iterations={result.iterations}",
                    flush=True,
                )

            sample_coordinate = np.linspace(
                0.0,
                float(keyframe_count),
                frame_count,
            )
            joint_plan = np.column_stack(
                [
                    np.interp(
                        sample_coordinate,
                        np.arange(keyframe_count + 1),
                        coarse_q[:, joint],
                    )
                    for joint in range(22)
                ]
            ).astype(np.float32)
            distance_plan = np.linspace(
                0.0,
                args.axial_travel_m,
                frame_count,
                dtype=np.float32,
            )
            surface_plan = np.zeros((frame_count, 5, 3), dtype=np.float32)
            kinematic_plan = np.zeros_like(surface_plan)
            normal_plan = np.zeros_like(surface_plan)
            residual_plan = np.zeros((frame_count, 5), dtype=np.float32)
            for frame, distance in enumerate(distance_plan):
                surface, desired_points, normals = rotated_targets(
                    float(distance)
                )
                achieved_points = reachability.forward_points(
                    joint_plan[frame]
                )
                # The palm root is a non-contact load-feedback coordinate.
                # Replace its nominal rigid orbit with the point actually
                # achieved by this joint-limit-valid arm state.  Fingertip
                # surface targets remain unchanged and strictly checked.
                desired_points[0] = achieved_points[0]
                surface[0] = achieved_points[0]
                surface_plan[frame] = surface
                kinematic_plan[frame] = desired_points
                normal_plan[frame] = normals
                residual_plan[frame] = np.linalg.norm(
                    achieved_points - desired_points,
                    axis=1,
                )

            max_joint_step = float(
                np.max(np.abs(np.diff(joint_plan, axis=0)))
            )
            if max_joint_step > args.max_plan_joint_step_rad:
                raise RuntimeError(
                    "Circumferential surface MPC joint step exceeds bound: "
                    f"{max_joint_step:.5f}rad > "
                    f"{args.max_plan_joint_step_rad:.5f}rad"
                )
            if (
                float(residual_plan[-1, 1:].max())
                > args.ik_tolerance_mm / 1000.0
            ):
                raise RuntimeError(
                    "Circumferential surface MPC interpolation failed final "
                    f"fingertip reachability: "
                    f"{(residual_plan[-1] * 1000).round(2).tolist()}mm"
                )

            self.plan_surface = surface_plan
            self.plan_kinematic = kinematic_plan
            self.plan_normals = normal_plan
            self.plan_q = joint_plan
            self.plan_residual = residual_plan
            self.plan_distance = distance_plan
            self.planned_axial_travel = float(distance_plan[-1])
            end_local = (
                rotation.T @ (surface_plan[-1] - center).T
            ).T
            args.plan_output.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                args.plan_output,
                surface_points_m=surface_plan,
                kinematic_points_m=kinematic_plan,
                joint_positions_rad=joint_plan,
                progress_m=np.repeat(
                    distance_plan[:, None], 5, axis=1
                ),
                progress_residual_m=residual_plan,
                normal_error_m=np.zeros_like(residual_plan),
                axial_distance_m=distance_plan,
                planner=np.asarray(args.planner),
                surface_preload_mm=np.asarray(args.surface_preload_mm),
                max_joint_step_rad=np.asarray(max_joint_step),
                coarse_joint_positions_rad=coarse_q,
                coarse_progress_m=np.repeat(
                    coarse_distance[:, None], 5, axis=1
                ),
                coarse_normal_error_m=np.zeros_like(coarse_residual),
                coarse_cost=np.zeros(keyframe_count + 1),
                coarse_nfev=coarse_nfev,
                start_surface_local_m=start_local,
                end_surface_local_m=end_local,
            )
            print(
                "[CIRCUMFERENTIAL-PLAN] passed | "
                f"frames={frame_count} keyframes={keyframe_count} "
                f"surface_travel_m={self.planned_axial_travel:.4f} "
                f"rotation_deg="
                f"{np.degrees(args.axial_travel_m / CAPSULE_RADIUS):.2f} "
                f"max_joint_step_rad={max_joint_step:.5f} "
                f"max_five_point_error_mm="
                f"{float(residual_plan.max() * 1000):.2f} "
                f"saved={args.plan_output.resolve()}",
                flush=True,
            )

        def _build_adaptive_surface_mpc_plan(
            self,
            *,
            center: np.ndarray,
            rotation: np.ndarray,
            start_arc: np.ndarray,
            start_azimuth: np.ndarray,
            start_surface: np.ndarray,
            start_frames: np.ndarray,
            direction: float,
            available: float,
            frame_count: int,
        ) -> None:
            """Optimize a reachable contact path directly in joint space.

            Unlike rigid Cartesian transport, every plan state is a real URDF
            configuration.  Per-contact meridian progress is a soft MPC
            objective, so fingers may lead or lag slightly when the coupled
            kinematics require it while all contacts still cross the object.
            """

            assert self.kinematic_targets is not None
            assert self.reachable_q is not None
            if args.axial_travel_m > available - 1.0e-6:
                raise RuntimeError(
                    "Requested adaptive-MPC travel exceeds the capsule surface: "
                    f"requested={args.axial_travel_m:.4f}m "
                    f"available={available:.4f}m"
                )

            initial_site_offset = self.kinematic_targets - start_surface
            initial_signed_standoff = np.einsum(
                "ni,ni->n",
                initial_site_offset,
                start_frames[:, :, 0],
            )
            target_signed_standoff = (
                initial_signed_standoff - args.surface_preload_mm / 1000.0
            )
            lower = np.where(
                np.isfinite(reachability.lower),
                reachability.lower + 1.0e-7,
                -20.0,
            )
            upper = np.where(
                np.isfinite(reachability.upper),
                reachability.upper - 1.0e-7,
                20.0,
            )
            keyframe_count = min(args.mpc_keyframes, frame_count)
            coarse_q = np.zeros((keyframe_count + 1, 22), dtype=np.float64)
            coarse_q[0] = np.minimum(
                np.maximum(self.reachable_q, lower),
                upper,
            )
            coarse_distance = np.linspace(
                0.0,
                args.axial_travel_m,
                keyframe_count + 1,
            )
            coarse_progress = np.zeros((keyframe_count + 1, 5), dtype=np.float64)
            coarse_normal_error = np.zeros_like(coarse_progress)
            coarse_cost = np.zeros(keyframe_count + 1, dtype=np.float64)
            coarse_nfev = np.zeros(keyframe_count + 1, dtype=np.int32)

            def contact_state(
                q: np.ndarray,
            ) -> tuple[
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
            ]:
                points = reachability.forward_points(q)
                surface, normals = capsule_project(
                    points,
                    center,
                    rotation,
                    CAPSULE_RADIUS,
                    CAPSULE_HALF_HEIGHT,
                )
                arc, azimuth = capsule_meridian_coordinates(
                    surface,
                    center,
                    rotation,
                    CAPSULE_RADIUS,
                    CAPSULE_HALF_HEIGHT,
                )
                # The palm root is not a contact.  Its fifth planning
                # coordinate is exact axial URDF displacement rather than a
                # projection around an irrelevant capsule end cap.
                palm_axis_displacement = float(
                    np.dot(
                        points[0] - self.kinematic_targets[0],
                        rotation[:, 2],
                    )
                )
                arc[0] = start_arc[0] + palm_axis_displacement
                signed_standoff = np.einsum(
                    "ni,ni->n",
                    points - surface,
                    normals,
                )
                return points, surface, normals, arc, np.column_stack(
                    (azimuth, signed_standoff)
                )

            start_q = coarse_q[0].copy()
            initial_palm_position, initial_palm_rotation = (
                reachability.forward_palm_pose(start_q)
            )
            previous_q = start_q.copy()
            previous_delta = np.zeros(22, dtype=np.float64)
            for keyframe in range(1, keyframe_count + 1):
                desired_distance = float(coarse_distance[keyframe])
                desired_arc = start_arc + direction * desired_distance
                active_progress_tolerance_mm = (
                    args.mpc_progress_tolerance_mm
                    if keyframe == keyframe_count
                    else args.mpc_intermediate_progress_tolerance_mm
                )
                minimum_progress = (
                    coarse_progress[keyframe - 1]
                )
                preload_fraction = min(
                    desired_distance / max(0.025, args.axial_travel_m),
                    1.0,
                )
                desired_standoff = (
                    initial_signed_standoff
                    + preload_fraction
                    * (target_signed_standoff - initial_signed_standoff)
                )
                palm_target = (
                    self.kinematic_targets[0]
                    + direction
                    * desired_distance
                    * rotation[:, 2]
                )

                def residual(
                    q: np.ndarray,
                    *,
                    joint_regularization: float,
                    progress_scale: float,
                    normal_scale: float,
                ) -> np.ndarray:
                    points, _, _, arc, auxiliary = contact_state(q)
                    azimuth = auxiliary[:, 0]
                    signed_standoff = auxiliary[:, 1]
                    azimuth_error = (
                        azimuth - start_azimuth + np.pi
                    ) % (2.0 * np.pi) - np.pi
                    progress_error = direction * (arc - desired_arc)
                    normal_error = signed_standoff - desired_standoff
                    # Treat tracking bounds as a feasibility region instead
                    # of forcing progress and normal errors to compete in a
                    # single weighted sum.  Once an error enters the inner
                    # band, smoothness and azimuth drift select the solution.
                    progress_band = (
                        0.80
                        * active_progress_tolerance_mm
                        / 1000.0
                    )
                    normal_band = (
                        0.80
                        * args.mpc_normal_tolerance_mm
                        / 1000.0
                    )
                    progress_violation = np.sign(progress_error) * np.maximum(
                        np.abs(progress_error) - progress_band,
                        0.0,
                    )
                    tip_normal_error = normal_error[1:]
                    normal_violation = np.sign(tip_normal_error) * np.maximum(
                        np.abs(tip_normal_error) - normal_band,
                        0.0,
                    )
                    achieved_progress = direction * (arc - start_arc)
                    monotonic_violation = np.maximum(
                        minimum_progress - achieved_progress,
                        0.0,
                    )
                    _, palm_rotation = reachability.forward_palm_pose(q)
                    palm_orientation_error = R.from_matrix(
                        initial_palm_rotation @ palm_rotation.T
                    ).as_rotvec()
                    return np.concatenate(
                        (
                            progress_scale * progress_violation
                            + 1.0 * progress_error,
                            normal_scale * normal_violation
                            + 0.3 * tip_normal_error,
                            150.0 * monotonic_violation,
                            5.0 * (points[0] - palm_target),
                            0.02 * palm_orientation_error,
                            0.01 * (q[6:] - start_q[6:]),
                            0.02 * CAPSULE_RADIUS * azimuth_error,
                            joint_regularization * (q - previous_q),
                            0.0008 * (q - previous_q - previous_delta),
                        )
                    )

                candidates = []
                extrapolated_seed = np.minimum(
                    np.maximum(previous_q + previous_delta, lower),
                    upper,
                )
                rigid_seed_input = previous_q.copy()
                rigid_seed_input[6:] = start_q[6:]
                rigid_arm_result = reachability.solve_palm_pose(
                    initial_palm_position
                    + direction
                    * desired_distance
                    * rotation[:, 2],
                    initial_palm_rotation,
                    rigid_seed_input,
                    position_tolerance=1.0e-3,
                    orientation_tolerance=3.0e-3,
                    max_iterations=args.mpc_max_nfev,
                )
                rigid_arm_seed = rigid_arm_result.joint_position
                candidates = []
                _, _, _, rigid_arc, rigid_aux = contact_state(rigid_arm_seed)
                rigid_progress_error = np.abs(
                    direction * (rigid_arc - desired_arc)
                )
                rigid_normal_error = np.abs(
                    rigid_aux[:, 1] - desired_standoff
                )
                rigid_monotonic_error = np.maximum(
                    minimum_progress
                    - direction * (rigid_arc - start_arc),
                    0.0,
                )
                if (
                    float(rigid_progress_error.max())
                    <= active_progress_tolerance_mm / 1000.0
                    and float(rigid_normal_error[1:].max())
                    <= args.mpc_normal_tolerance_mm / 1000.0
                    and float(rigid_monotonic_error.max()) <= 2.0e-4
                ):
                    candidates.append(
                        (
                            -1.0,
                            SimpleNamespace(
                                x=rigid_arm_seed,
                                cost=0.0,
                                nfev=rigid_arm_result.iterations,
                            ),
                            rigid_progress_error,
                            rigid_normal_error,
                            rigid_arc,
                        )
                    )
                else:
                    print(
                        "[RIGID-SEED-REJECTED] "
                        f"keyframe={keyframe}/{keyframe_count} "
                        f"progress_error_mm="
                        f"{(rigid_progress_error * 1000).round(2).tolist()} "
                        f"tip_normal_error_mm="
                        f"{(rigid_normal_error[1:] * 1000).round(2).tolist()} "
                        f"palm_pos_error_mm="
                        f"{rigid_arm_result.position_error_m * 1000:.2f} "
                        f"palm_rot_error_rad="
                        f"{rigid_arm_result.orientation_error_rad:.4f}",
                        flush=True,
                    )
                for seed, regularization, progress_scale, normal_scale in (
                    (rigid_arm_seed, 0.0001, 24.0, 24.0),
                    (extrapolated_seed, 0.0003, 24.0, 24.0),
                    (previous_q, 0.0001, 32.0, 20.0),
                    (previous_q, 0.0001, 20.0, 32.0),
                    (previous_q, 0.0, 40.0, 40.0),
                ):
                    result = least_squares(
                        lambda q, reg=regularization, ps=progress_scale, ns=normal_scale: residual(
                            q,
                            joint_regularization=reg,
                            progress_scale=ps,
                            normal_scale=ns,
                        ),
                        seed,
                        bounds=(lower, upper),
                        max_nfev=args.mpc_max_nfev,
                        xtol=1.0e-8,
                        ftol=1.0e-8,
                        gtol=1.0e-8,
                        x_scale="jac",
                    )
                    _, _, _, candidate_arc, candidate_aux = contact_state(
                        result.x
                    )
                    progress_error = np.abs(
                        direction * (candidate_arc - desired_arc)
                    )
                    normal_error = np.abs(
                        candidate_aux[:, 1] - desired_standoff
                    )
                    monotonic_error = np.maximum(
                        minimum_progress
                        - direction * (candidate_arc - start_arc),
                        0.0,
                    )
                    score = (
                        1000.0 * float(monotonic_error.max())
                        +
                        1000.0
                        * max(
                            float(progress_error.max())
                            - active_progress_tolerance_mm / 1000.0,
                            0.0,
                        )
                        + 1000.0
                        * max(
                            float(normal_error[1:].max())
                            - args.mpc_normal_tolerance_mm / 1000.0,
                            0.0,
                        )
                        + 8.0 * float(progress_error.max())
                        + 5.0 * float(normal_error[1:].max())
                        + 0.02 * float(
                            np.max(np.abs(result.x - previous_q))
                        )
                        + 1.0e-3 * float(result.cost)
                    )
                    candidates.append(
                        (
                            score,
                            result,
                            progress_error,
                            normal_error,
                            candidate_arc,
                        )
                    )
                (
                    _,
                    best,
                    progress_error,
                    normal_error,
                    achieved_arc,
                ) = min(candidates, key=lambda item: item[0])
                if (
                    float(progress_error.max())
                    > active_progress_tolerance_mm / 1000.0
                    or float(normal_error[1:].max())
                    > args.mpc_normal_tolerance_mm / 1000.0
                ):
                    # Constraint-repair pass: warm-start from the best
                    # compromise and sharply penalize only band violations.
                    # This avoids loosening tolerances for a single contact
                    # that misses the feasible set by a fraction of a mm.
                    repair_seed = best.x.copy()
                    for progress_scale, normal_scale in (
                        (100.0, 60.0),
                        (60.0, 100.0),
                        (120.0, 120.0),
                    ):
                        repaired = least_squares(
                            lambda q, ps=progress_scale, ns=normal_scale: residual(
                                q,
                                joint_regularization=0.0,
                                progress_scale=ps,
                                normal_scale=ns,
                            ),
                            repair_seed,
                            bounds=(lower, upper),
                            max_nfev=args.mpc_max_nfev,
                            xtol=1.0e-9,
                            ftol=1.0e-9,
                            gtol=1.0e-9,
                            x_scale="jac",
                        )
                        _, _, _, repaired_arc, repaired_aux = contact_state(
                            repaired.x
                        )
                        repaired_progress_error = np.abs(
                            direction * (repaired_arc - desired_arc)
                        )
                        repaired_normal_error = np.abs(
                            repaired_aux[:, 1] - desired_standoff
                        )
                        repaired_monotonic_error = np.maximum(
                            minimum_progress
                            - direction * (repaired_arc - start_arc),
                            0.0,
                        )
                        repaired_score = (
                            1000.0
                            * float(repaired_monotonic_error.max())
                            +
                            1000.0
                            * max(
                                float(repaired_progress_error.max())
                                - active_progress_tolerance_mm / 1000.0,
                                0.0,
                            )
                            + 1000.0
                            * max(
                                float(repaired_normal_error[1:].max())
                                - args.mpc_normal_tolerance_mm / 1000.0,
                                0.0,
                            )
                            + 8.0 * float(repaired_progress_error.max())
                            + 5.0
                            * float(repaired_normal_error[1:].max())
                            + 0.02
                            * float(
                                np.max(np.abs(repaired.x - previous_q))
                            )
                            + 1.0e-3 * float(repaired.cost)
                        )
                        candidates.append(
                            (
                                repaired_score,
                                repaired,
                                repaired_progress_error,
                                repaired_normal_error,
                                repaired_arc,
                            )
                        )
                        repair_seed = repaired.x
                    (
                        _,
                        best,
                        progress_error,
                        normal_error,
                        achieved_arc,
                    ) = min(candidates, key=lambda item: item[0])
                achieved_progress = direction * (
                    achieved_arc - start_arc
                )
                monotonic_error = np.maximum(
                    minimum_progress - achieved_progress,
                    0.0,
                )
                if float(monotonic_error.max()) > 2.0e-4:
                    raise RuntimeError(
                        "Adaptive surface MPC violated monotonic progress: "
                        f"keyframe={keyframe}/{keyframe_count} "
                        f"error_mm="
                        f"{(monotonic_error * 1000).round(2).tolist()}"
                    )
                if (
                    float(progress_error.max())
                    > active_progress_tolerance_mm / 1000.0
                ):
                    raise RuntimeError(
                        "Adaptive surface MPC missed longitudinal progress: "
                        f"keyframe={keyframe}/{keyframe_count} "
                        f"distance_m={desired_distance:.4f} "
                        f"error_mm={(progress_error * 1000).round(2).tolist()}"
                    )
                if (
                    float(normal_error[1:].max())
                    > args.mpc_normal_tolerance_mm / 1000.0
                ):
                    raise RuntimeError(
                        "Adaptive surface MPC missed fingertip contact standoff: "
                        f"keyframe={keyframe}/{keyframe_count} "
                        f"distance_m={desired_distance:.4f} "
                        f"error_mm="
                        f"{(normal_error[1:] * 1000).round(2).tolist()}"
                    )
                q = best.x
                coarse_q[keyframe] = q
                coarse_progress[keyframe] = direction * (
                    achieved_arc - start_arc
                )
                coarse_normal_error[keyframe] = normal_error
                coarse_cost[keyframe] = float(best.cost)
                coarse_nfev[keyframe] = int(best.nfev)
                previous_delta = q - previous_q
                previous_q = q
                print(
                    "[ADAPTIVE-MPC] "
                    f"keyframe={keyframe:02d}/{keyframe_count} "
                    f"travel_m={desired_distance:.4f} "
                    f"progress_mm={(coarse_progress[keyframe] * 1000).round(1).tolist()} "
                    f"tip_normal_error_mm="
                    f"{(normal_error[1:] * 1000).round(2).tolist()} "
                    f"nfev={best.nfev}",
                    flush=True,
                )

            sample_coordinate = np.linspace(
                0.0,
                float(keyframe_count),
                frame_count + 1,
            )[1:]
            surface_plan = np.zeros((frame_count, 5, 3), dtype=np.float32)
            kinematic_plan = np.zeros_like(surface_plan)
            normal_plan = np.zeros_like(surface_plan)
            joint_plan = np.zeros((frame_count, 22), dtype=np.float32)
            residual_plan = np.zeros((frame_count, 5), dtype=np.float32)
            distance_plan = np.zeros(frame_count, dtype=np.float32)
            progress_plan = np.zeros((frame_count, 5), dtype=np.float32)
            normal_error_plan = np.zeros_like(progress_plan)
            for frame, coordinate in enumerate(sample_coordinate):
                left = min(int(np.floor(coordinate)), keyframe_count - 1)
                blend = coordinate - left
                blend = blend * blend * (3.0 - 2.0 * blend)
                q = (1.0 - blend) * coarse_q[left] + blend * coarse_q[left + 1]
                points, surface, normals, arc, auxiliary = contact_state(q)
                progress = direction * (arc - start_arc)
                desired_distance = args.axial_travel_m * (
                    (frame + 1) / float(frame_count)
                )
                surface_plan[frame] = surface
                kinematic_plan[frame] = points
                normal_plan[frame] = normals
                joint_plan[frame] = q
                residual_plan[frame] = np.abs(progress - desired_distance)
                progress_plan[frame] = progress
                normal_error_plan[frame] = np.abs(
                    auxiliary[:, 1]
                    - (
                        initial_signed_standoff
                        + min(
                            desired_distance
                            / max(0.025, args.axial_travel_m),
                            1.0,
                        )
                        * (
                            target_signed_standoff
                            - initial_signed_standoff
                        )
                    )
                )
                distance_plan[frame] = float(np.min(progress))

            final_progress_error = np.abs(
                progress_plan[-1] - args.axial_travel_m
            )
            if float(final_progress_error.max()) > (
                args.mpc_progress_tolerance_mm / 1000.0
            ):
                raise RuntimeError(
                    "Adaptive surface MPC interpolation failed final progress: "
                    f"error_mm="
                    f"{(final_progress_error * 1000).round(2).tolist()}"
                )
            q_with_seed = np.vstack((start_q[None], joint_plan))
            max_joint_step = float(np.max(np.abs(np.diff(q_with_seed, axis=0))))
            if max_joint_step > args.max_plan_joint_step_rad:
                raise RuntimeError(
                    "Adaptive surface MPC joint step exceeds bound: "
                    f"observed={max_joint_step:.5f}rad "
                    f"limit={args.max_plan_joint_step_rad:.5f}rad"
                )

            self.plan_surface = surface_plan
            self.plan_kinematic = kinematic_plan
            self.plan_normals = normal_plan
            self.plan_q = joint_plan
            self.plan_residual = residual_plan
            self.plan_distance = distance_plan
            self.plan_direction = direction
            self.planned_axial_travel = float(distance_plan[-1])
            start_local = (rotation.T @ (surface_plan[0] - center).T).T
            end_local = (rotation.T @ (surface_plan[-1] - center).T).T
            args.plan_output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                args.plan_output,
                surface_points_m=surface_plan,
                kinematic_points_m=kinematic_plan,
                joint_positions_rad=joint_plan,
                progress_m=progress_plan,
                progress_residual_m=residual_plan,
                normal_error_m=normal_error_plan,
                axial_distance_m=distance_plan,
                planner=np.asarray(args.planner),
                surface_preload_mm=np.asarray(args.surface_preload_mm),
                object_radius_m=np.asarray(CAPSULE_RADIUS),
                object_half_height_m=np.asarray(CAPSULE_HALF_HEIGHT),
                max_joint_step_rad=np.asarray(max_joint_step),
                coarse_joint_positions_rad=coarse_q,
                coarse_progress_m=coarse_progress,
                coarse_normal_error_m=coarse_normal_error,
                coarse_cost=coarse_cost,
                coarse_nfev=coarse_nfev,
                start_surface_local_m=start_local,
                end_surface_local_m=end_local,
            )
            print(
                "[AXIAL-PLAN] adaptive surface MPC passed | "
                f"frames={frame_count} "
                f"keyframes={keyframe_count} "
                f"min_travel_m={self.planned_axial_travel:.4f} "
                f"per_contact_travel_m={progress_plan[-1].round(4).tolist()} "
                f"max_joint_step_rad={max_joint_step:.5f} "
                f"max_progress_error_mm="
                f"{float(residual_plan.max() * 1000):.2f} "
                f"max_tip_normal_error_mm="
                f"{float(normal_error_plan[:, 1:].max() * 1000):.2f} "
                f"start_z_m={start_local[:, 2].round(4).tolist()} "
                f"end_z_m={end_local[:, 2].round(4).tolist()} "
                f"saved={args.plan_output.resolve()}",
                flush=True,
            )

        def _advance_axial_plan(self) -> None:
            assert self.plan_surface is not None
            assert self.plan_kinematic is not None
            assert self.plan_normals is not None
            assert self.plan_q is not None
            assert self.plan_residual is not None
            assert self.plan_distance is not None
            index = min(
                max(self.step - args.motion_start, 0),
                self.plan_surface.shape[0] - 1,
            )
            self.plan_index = index
            self.targets = self.plan_surface[index]
            self.kinematic_targets = self.plan_kinematic[index]
            self.normals = self.plan_normals[index]
            self.reachable_q = self.plan_q[index]
            self.last_residual = self.plan_residual[index]
            self.executed_axial_travel = float(self.plan_distance[index])

        def __call__(self, obs):
            if self.targets is None:
                self._initialize(obs)
            self._update_object_approach()
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
            for finger, site_name in enumerate(MCC_TIP_NAMES):
                sensor_data = env.scene[f"{site_name}_contact"].data
                if (
                    sensor_data.found is not None
                    and sensor_data.dist is not None
                    and bool(sensor_data.found[0, 0].item())
                ):
                    distance = float(sensor_data.dist[0, 0].item())
                    self.contact_distance_m[finger] = distance
                    self.max_penetration_m[finger] = max(
                        self.max_penetration_m[finger],
                        max(-distance, 0.0),
                    )
                else:
                    self.contact_distance_m[finger] = np.nan
            if not self.contact_calibrated:
                for finger, tactile_force in enumerate(self.tactile_force):
                    if tactile_force < args.min_contact_force_n:
                        base = 4 * finger
                        flexion = [base, base + 2, base + 3]
                        self.precontact_closure[flexion] = np.minimum(
                            self.precontact_closure[flexion]
                            + args.contact_search_step_rad,
                            args.contact_search_limit_rad,
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
                self._advance_axial_plan()
            elif (
                self.contact_calibrated
                and not self.motor_force_recalibrated
                and self.step == args.motion_start - 1
            ):
                settled_motor_force = torch.linalg.vector_norm(
                    controller.last_debug["tip_force_from_motors"][0],
                    dim=-1,
                ).detach().cpu().numpy()
                controller.fingers.calibrate_motor_force_setpoint(
                    settled_motor_force
                )
                self.motor_force_recalibrated = True
                print(
                    "[MOTOR-FORCE-RECALIBRATION] settled per-finger "
                    f"setpoint_N={settled_motor_force.round(2).tolist()}",
                    flush=True,
                )
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
            command_q = self.reachable_q.copy()
            if not self.contact_calibrated:
                command_q[6:22] += self.precontact_closure
                command_q = np.minimum(
                    np.maximum(command_q, reachability.lower),
                    reachability.upper,
                )
                command_q[6:22] = live_q[6:22] + np.clip(
                    command_q[6:22] - live_q[6:22],
                    -args.contact_search_step_rad,
                    args.contact_search_step_rad,
                )
            else:
                command_q += self.contact_servo_offset_q
                command_q[:6] += (
                    args.arm_trajectory_tracking_gain
                    * (self.reachable_q[:6] - live_q[:6])
                )
                command_q[6:22] += (
                    args.finger_trajectory_tracking_gain
                    * (self.reachable_q[6:22] - live_q[6:22])
                )
                command_q = np.minimum(
                    np.maximum(command_q, reachability.lower),
                    reachability.upper,
                )
            self.last_command_q = command_q.copy()
            joint_reference_t = torch.as_tensor(
                command_q[None],
                device=device,
                dtype=torch.float32,
            )
            action = controller(
                obs,
                contact_points=target_t,
                surface_normals=normal_t,
                joint_reference=joint_reference_t,
                kinematic_points=kinematic_target_t,
            )
            if self.contact_calibrated:
                force_correction = controller.last_debug[
                    "finger_force_joint_correction"
                ]
                self.max_force_correction_rad = max(
                    self.max_force_correction_rad,
                    float(torch.max(torch.abs(force_correction)).item()),
                )
            if not self.contact_calibrated:
                # Do not let a motor-residual force estimate release fingers
                # before true tactile contact has been established.  This
                # phase is absolute position hold plus independent tactile
                # search; motor-force MCC starts after calibration.
                action[:, :6] = joint_reference_t[:, :6]
                action[:, 6:22] = joint_reference_t[:, 6:22]
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
                    f"contact_dist_mm="
                    f"{(self.contact_distance_m * 1000).round(3).tolist()} "
                    f"tank={float(debug['energy_tank'][0]):.3f} "
                    f"axial_travel_m={self.executed_axial_travel:.4f} "
                    f"plan_frame={self.plan_index}",
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
        f"device={device} viewer={args.viewer} "
        f"object_radius_m={CAPSULE_RADIUS:.4f} "
        f"object_half_height_m={CAPSULE_HALF_HEIGHT:.4f}"
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
            if policy.executed_axial_travel < 0.99 * args.axial_travel_m:
                raise RuntimeError(
                    "End-to-end trajectory was not fully executed: "
                    f"executed={policy.executed_axial_travel:.4f}m "
                    f"requested={args.axial_travel_m:.4f}m"
                )
            contact_ratio = (
                policy.contact_frames / float(policy.evaluated_frames)
            )
            if np.any(contact_ratio < args.min_contact_ratio):
                raise RuntimeError(
                    "Fingertip continuous-contact ratio below required "
                    f"{args.min_contact_ratio:.1%}: "
                    f"{contact_ratio.round(4).tolist()}"
                )
            max_penetration_mm = policy.max_penetration_m * 1000.0
            if np.any(max_penetration_mm > args.max_contact_penetration_mm):
                raise RuntimeError(
                    "Fingertip/object penetration exceeded required limit "
                    f"{args.max_contact_penetration_mm:.3f}mm: "
                    f"{max_penetration_mm.round(3).tolist()}"
                )
            print(
                f"[VIDEO] saved={args.output.resolve()} frames={frames_written} "
                f"duration_s={args.steps * dt:.2f} fps={args.fps:.1f} "
                f"tip_contact_ratio={contact_ratio.round(4).tolist()} "
                f"axial_travel_m={policy.executed_axial_travel:.4f} "
                f"max_motor_force_correction_rad="
                f"{policy.max_force_correction_rad:.6f} "
                f"max_contact_penetration_mm="
                f"{max_penetration_mm.round(3).tolist()} "
                f"final_tip_site_standoff_mm="
                f"{(policy.surface_error[1:] * 1000).round(2).tolist()}",
                flush=True,
            )
    finally:
        wrapped.close()


if __name__ == "__main__":
    main()
