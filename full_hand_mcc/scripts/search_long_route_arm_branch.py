"""Find an FR3 IK branch that preserves a grasp over a long surface route.

The four LeapHand joint configurations and object-relative contact geometry
remain unchanged.  Only the seven arm joints are replaced by another inverse-
kinematics branch that realizes the same initial palm pose and can follow the
requested collision-avoidance palm trajectory end to end.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from mjlab.tasks.leaphand.full_hand_mcc_geometry import (
    capsule_meridian_coordinates,
    capsule_meridian_targets,
    capsule_project,
)
from mjlab.tasks.leaphand.leaphand_full_hand_mcc_env_cfg import (
    ARM_DOF,
    FivePointReachabilitySolver,
)
import mjlab.tasks.leaphand.leaphand_full_hand_mcc_env_cfg as env_module


def _smoothstep(value: float) -> float:
    phase = float(np.clip(value, 0.0, 1.0))
    return phase * phase * (3.0 - 2.0 * phase)


def _mean_contact_frame(frames: np.ndarray) -> np.ndarray:
    normal = np.mean(frames[1:, :, 0], axis=0)
    normal /= max(float(np.linalg.norm(normal)), 1.0e-12)
    meridian = np.mean(frames[1:, :, 2], axis=0)
    meridian -= normal * float(np.dot(normal, meridian))
    meridian /= max(float(np.linalg.norm(meridian)), 1.0e-12)
    azimuth = np.cross(meridian, normal)
    azimuth /= max(float(np.linalg.norm(azimuth)), 1.0e-12)
    meridian = np.cross(normal, azimuth)
    return np.column_stack((normal, azimuth, meridian))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-grasp", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--travel-m", type=float, default=0.40)
    parser.add_argument(
        "--world-offset-z-m",
        type=float,
        default=0.0,
        help=(
            "Translate the complete initial hand/object arrangement in world "
            "Z before searching. A negative value creates upward workspace "
            "without changing any object-relative fingertip contacts."
        ),
    )
    parser.add_argument("--keyframes", type=int, default=113)
    parser.add_argument("--random-starts", type=int, default=400)
    parser.add_argument("--random-seed", type=int, default=20260724)
    parser.add_argument("--clearance-mm", type=float, default=2.0)
    parser.add_argument("--clearance-lift-m", type=float, default=0.018)
    parser.add_argument("--clearance-ramp-m", type=float, default=0.04)
    parser.add_argument(
        "--follow-surface-frame",
        action="store_true",
        help=(
            "Transport the palm with the mean four-fingertip surface frame. "
            "This preserves pad orientation through curved end caps."
        ),
    )
    parser.add_argument("--surface-frame-gain", type=float, default=1.0)
    parser.add_argument("--clearance-tilt-deg", type=float, default=20.0)
    parser.add_argument("--tilt-start-m", type=float, default=0.0)
    parser.add_argument("--tilt-ramp-m", type=float, default=0.04)
    parser.add_argument("--tilt-release-start-m", type=float, default=1.0)
    parser.add_argument("--tilt-release-ramp-m", type=float, default=0.04)
    parser.add_argument("--secondary-lift-m", type=float, default=0.006)
    parser.add_argument("--secondary-start-m", type=float, default=0.10)
    parser.add_argument("--secondary-ramp-m", type=float, default=0.04)
    args = parser.parse_args()

    if args.travel_m <= 0.0:
        raise ValueError("--travel-m must be positive")
    if args.keyframes < 2:
        raise ValueError("--keyframes must be at least two")
    if args.random_starts < 1:
        raise ValueError("--random-starts must be positive")
    if not 0.0 <= args.surface_frame_gain <= 1.0:
        raise ValueError("--surface-frame-gain must be in [0, 1]")
    if args.tilt_start_m < 0.0:
        raise ValueError("--tilt-start-m cannot be negative")
    if args.tilt_ramp_m <= 0.0:
        raise ValueError("--tilt-ramp-m must be positive")

    source = np.load(args.seed_grasp)
    q_seed = np.asarray(source["joint_position_rad"], dtype=np.float64)
    source_center = np.asarray(
        source["object_center_m"],
        dtype=np.float64,
    )
    world_offset = np.asarray(
        (0.0, 0.0, args.world_offset_z_m),
        dtype=np.float64,
    )
    center = source_center + world_offset
    rotation = np.asarray(source["object_rotation"], dtype=np.float64)
    radius = float(source["object_radius_m"])
    half_height = float(source["object_half_height_m"])

    env_module.FULL_HAND_OBJECT_SHAPE = "capsule"
    env_module.FULL_HAND_CAPSULE_RADIUS = radius
    env_module.FULL_HAND_CAPSULE_HALF_HEIGHT = half_height
    solver = FivePointReachabilitySolver(
        tolerance=2.5e-4,
        max_iterations=300,
        max_joint_step=0.08,
        posture_regularization=1.0e-7,
    )
    source_points = solver.forward_points(q_seed)
    source_palm_body_position, initial_palm_rotation = (
        solver.forward_palm_pose(q_seed)
    )
    initial_points = source_points + world_offset
    initial_palm_body_position = (
        source_palm_body_position + world_offset
    )
    palm_site_offset = (
        initial_palm_rotation.T
        @ (initial_points[0] - initial_palm_body_position)
    )
    initial_surface, _ = capsule_project(
        initial_points,
        center,
        rotation,
        radius,
        half_height,
    )
    initial_arc, initial_azimuth = capsule_meridian_coordinates(
        initial_surface,
        center,
        rotation,
        radius,
        half_height,
    )
    _, _, initial_frames = capsule_meridian_targets(
        initial_arc,
        initial_azimuth,
        center,
        rotation,
        radius,
        half_height,
    )
    initial_contact_frame = _mean_contact_frame(initial_frames)
    initial_patch_center = np.mean(initial_surface[1:], axis=0)
    initial_palm_patch_offset = (
        initial_contact_frame.T
        @ (initial_points[0] - initial_patch_center)
    )

    def minimum_arm_object_clearance(q: np.ndarray) -> float:
        """Return clearance for the FR3 base/links, excluding hand geoms."""

        _, non_tip_distance, non_tip_names = solver.geometry_clearances(
            q,
            center,
            rotation,
        )
        arm_mask = np.asarray(
            [
                name.startswith("fr3v2_link")
                for name in non_tip_names
            ],
            dtype=bool,
        )
        return float(np.min(non_tip_distance[arm_mask]))

    def palm_target(distance: float) -> tuple[np.ndarray, np.ndarray]:
        clearance_phase = _smoothstep(distance / args.clearance_ramp_m)
        secondary_phase = _smoothstep(
            (
                distance - args.secondary_start_m
            )
            / args.secondary_ramp_m
        )
        tilt_phase = _smoothstep(
            (distance - args.tilt_start_m) / args.tilt_ramp_m
        )
        tilt_release_phase = _smoothstep(
            (
                distance - args.tilt_release_start_m
            )
            / args.tilt_release_ramp_m
        )
        transported_contact_frame = initial_contact_frame
        base_rotation = initial_palm_rotation
        if args.follow_surface_frame:
            desired_tip_arc = initial_arc[1:] + distance
            desired_tip_surface, _, desired_tip_frames = (
                capsule_meridian_targets(
                    desired_tip_arc,
                    initial_azimuth[1:],
                    center,
                    rotation,
                    radius,
                    half_height,
                )
            )
            desired_frames = np.concatenate(
                (initial_frames[:1], desired_tip_frames),
                axis=0,
            )
            desired_contact_frame = _mean_contact_frame(desired_frames)
            full_transport = desired_contact_frame @ initial_contact_frame.T
            frame_transport = R.from_rotvec(
                args.surface_frame_gain
                * R.from_matrix(full_transport).as_rotvec()
            ).as_matrix()
            transported_contact_frame = (
                frame_transport @ initial_contact_frame
            )
            base_rotation = frame_transport @ initial_palm_rotation
            site_position = (
                np.mean(desired_tip_surface, axis=0)
                + transported_contact_frame @ initial_palm_patch_offset
            )
        else:
            site_position = (
                initial_points[0] + distance * rotation[:, 2]
            )
        desired_rotation = (
            R.from_rotvec(
                -np.deg2rad(args.clearance_tilt_deg)
                * tilt_phase
                * (1.0 - tilt_release_phase)
                * transported_contact_frame[:, 1]
            ).as_matrix()
            @ base_rotation
        )
        site_position += (
            args.clearance_lift_m * clearance_phase
            + args.secondary_lift_m * secondary_phase
        ) * transported_contact_frame[:, 0]
        body_position = site_position - desired_rotation @ palm_site_offset
        return body_position, desired_rotation

    rng = np.random.default_rng(args.random_seed)
    finite_lower = solver.lower[:ARM_DOF]
    finite_upper = solver.upper[:ARM_DOF]
    arm_seeds = [q_seed[:ARM_DOF].copy()]
    # FR3 redundancy is continuous rather than the periodic xArm wrist
    # branches used by the previous model. Structured shoulder/elbow/tool
    # perturbations cover useful null-space seeds before random sampling.
    for joint1_offset in (0.0, -0.8, 0.8):
        for joint3_offset in (0.0, -0.8, 0.8):
            for joint7_offset in (0.0, -1.0, 1.0):
                candidate = q_seed[:ARM_DOF].copy()
                candidate[0] += joint1_offset
                candidate[2] += joint3_offset
                candidate[6] += joint7_offset
                arm_seeds.append(
                    np.minimum(
                        np.maximum(candidate, finite_lower),
                        finite_upper,
                    )
                )
    arm_seeds.extend(
        rng.uniform(
            finite_lower,
            finite_upper,
            size=(args.random_starts, ARM_DOF),
        )
    )

    branches: list[np.ndarray] = []
    for seed_index, arm_seed in enumerate(arm_seeds):
        q0 = q_seed.copy()
        q0[:ARM_DOF] = arm_seed
        result = solver.solve_palm_pose(
            initial_palm_body_position,
            initial_palm_rotation,
            q0,
            position_tolerance=2.5e-4,
            orientation_tolerance=1.0e-3,
            max_iterations=300,
        )
        if not result.accepted:
            continue
        candidate = result.joint_position
        if any(
            float(
                np.linalg.norm(
                    candidate[:ARM_DOF] - branch[:ARM_DOF]
                )
            )
            < 0.05
            for branch in branches
        ):
            continue
        clearance, _ = solver.minimum_non_tip_clearance(
            candidate,
            center,
            rotation,
        )
        self_pairs, _ = solver.self_collision_contacts(candidate)
        if clearance < args.clearance_mm / 1000.0 or self_pairs:
            continue
        branches.append(candidate)
        print(
            "[START-BRANCH] "
            f"seed={seed_index} branch={len(branches) - 1} "
            f"arm_q={candidate[:ARM_DOF].round(5).tolist()} "
            f"clearance_mm={clearance * 1000:.3f}",
            flush=True,
        )

    if not branches:
        raise RuntimeError("No collision-free alternative start IK branch found")

    successful: list[tuple[float, float, np.ndarray]] = []
    distances = np.linspace(0.0, args.travel_m, args.keyframes + 1)[1:]
    for branch_index, branch in enumerate(branches):
        previous = branch.copy()
        minimum_clearance = np.inf
        maximum_step = 0.0
        reached = 0.0
        failed_reason = ""
        for frame_index, distance in enumerate(distances, start=1):
            target_position, target_rotation = palm_target(float(distance))
            result = solver.solve_palm_pose(
                target_position,
                target_rotation,
                previous,
                position_tolerance=2.5e-4,
                orientation_tolerance=1.0e-3,
                max_iterations=300,
            )
            if not result.accepted:
                failed_reason = (
                    f"IK pos_mm={result.position_error_m * 1000:.3f} "
                    f"rot_rad={result.orientation_error_rad:.5f}"
                )
                break
            candidate = result.joint_position
            segment_clearance = np.inf
            segment_self_pairs = 0
            for fraction in np.linspace(0.2, 1.0, 5):
                sample = (
                    (1.0 - fraction) * previous
                    + fraction * candidate
                )
                clearance = minimum_arm_object_clearance(sample)
                segment_clearance = min(segment_clearance, clearance)
                self_pairs, _ = solver.self_collision_contacts(sample)
                segment_self_pairs += len(self_pairs)
            if (
                segment_clearance < args.clearance_mm / 1000.0
                or segment_self_pairs
            ):
                failed_reason = (
                    f"collision clearance_mm={segment_clearance * 1000:.3f} "
                    f"self_pairs={segment_self_pairs}"
                )
                break
            maximum_step = max(
                maximum_step,
                float(
                    np.max(
                        np.abs(
                            candidate[:ARM_DOF]
                            - previous[:ARM_DOF]
                        )
                    )
                ),
            )
            minimum_clearance = min(minimum_clearance, segment_clearance)
            previous = candidate
            reached = float(distance)
        print(
            "[BRANCH-AUDIT] "
            f"branch={branch_index} reached_m={reached:.4f} "
            f"min_clearance_mm={minimum_clearance * 1000:.3f} "
            f"max_arm_step_rad={maximum_step:.5f} "
            f"reason={failed_reason or 'passed'}",
            flush=True,
        )
        if reached >= args.travel_m - 1.0e-9:
            successful.append((minimum_clearance, -maximum_step, branch))

    if not successful:
        raise RuntimeError(
            "No start IK branch completed the requested collision-free route"
        )
    # Every accepted branch already has ample arm/object clearance. Prefer
    # the smoothest branch first, then use clearance as a tie-breaker.
    successful.sort(key=lambda item: (item[1], item[0]), reverse=True)
    best_clearance, neg_best_step, best_branch = successful[0]
    output_q = q_seed.copy()
    output_q[:ARM_DOF] = best_branch[:ARM_DOF]
    payload = {name: source[name] for name in source.files}
    payload["joint_position_rad"] = output_q
    payload["object_center_m"] = center
    payload["long_route_world_offset_z_m"] = np.asarray(
        args.world_offset_z_m
    )
    payload["long_route_travel_m"] = np.asarray(args.travel_m)
    payload["long_route_min_clearance_m"] = np.asarray(best_clearance)
    payload["long_route_max_arm_step_rad"] = np.asarray(-neg_best_step)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **payload)
    print(
        "[LONG-ROUTE-BRANCH] "
        f"saved={args.output.resolve()} "
        f"arm_q={best_branch[:ARM_DOF].round(6).tolist()} "
        f"min_clearance_mm={best_clearance * 1000:.3f} "
        f"max_arm_step_rad={-neg_best_step:.5f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
