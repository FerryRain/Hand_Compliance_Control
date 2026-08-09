"""Jointly optimize the arm/hand pose and axisymmetric-object relative pose."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

from mjlab.tasks.leaphand.full_hand_mcc_geometry import (
    capsule_project,
    ellipsoid_project,
)
from mjlab.tasks.leaphand.leaphand_full_hand_mcc_env_cfg import (
    FR3_HOME_Q,
    TOTAL_DOF,
    FivePointReachabilitySolver,
)
from mjlab.tasks.leaphand.leaphand_direct_force_env import (
    DEFAULT_PREGRASP_Q,
)
import mjlab.tasks.leaphand.leaphand_full_hand_mcc_env_cfg as env_module


JOINT_MARGIN_RESIDUAL_SCALE_RAD = 0.05


def _select_physical_reference_q(
    nominal_reference_q: np.ndarray,
    *,
    seed_only: bool,
    seed_q: np.ndarray | None,
) -> np.ndarray:
    """Select the posture prior used by the physical refinement stage."""

    nominal = np.asarray(nominal_reference_q, dtype=np.float64).reshape(-1)
    if not seed_only:
        return nominal.copy()
    if seed_q is None:
        raise ValueError("seed-only physical refinement requires a seed q")
    seed = np.asarray(seed_q, dtype=np.float64).reshape(-1)
    if seed.shape != nominal.shape:
        raise ValueError(
            "seed q and nominal reference q must have the same shape: "
            f"{seed.shape} != {nominal.shape}"
        )
    return seed.copy()


def _joint_margin_rad(
    q: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Return each joint's signed distance to its nearest position limit."""

    joint_position = np.asarray(q, dtype=np.float64)
    lower_limit = np.asarray(lower, dtype=np.float64)
    upper_limit = np.asarray(upper, dtype=np.float64)
    return np.minimum(
        joint_position - lower_limit,
        upper_limit - joint_position,
    )


def _joint_margin_soft_residual(
    q: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    target_margin_rad: float,
) -> np.ndarray:
    """Return a dimensionless per-joint hinge below the soft margin."""

    target = float(target_margin_rad)
    if not np.isfinite(target) or target < 0.0:
        raise ValueError("target joint margin must be finite and non-negative")
    if target == 0.0:
        # Preserve the exact legacy residual shape as well as its value.
        return np.zeros(0, dtype=np.float64)
    return np.maximum(
        target - _joint_margin_rad(q, lower, upper),
        0.0,
    ) / JOINT_MARGIN_RESIDUAL_SCALE_RAD


def _joint_margin_hard_ok(
    joint_margin_rad: np.ndarray,
    minimum_accepted_margin_rad: float,
) -> bool:
    """Audit the hard joint-margin threshold without rejecting +inf."""

    margins = np.asarray(joint_margin_rad, dtype=np.float64).reshape(-1)
    return bool(
        margins.size
        and not np.any(np.isnan(margins))
        and float(np.min(margins))
        >= float(minimum_accepted_margin_rad)
    )


def _least_squares_result_is_acceptable(result: object) -> bool:
    """Accept finite convergence or a finite max-evaluation candidate.

    SciPy reports ``status == 0`` and ``success == False`` when the evaluation
    budget is exhausted.  Such a state may still satisfy every independent
    physical hard gate below, so retain it for that audit.  Non-finite state
    and actual solver failures remain unusable.
    """

    try:
        finite = bool(
            np.all(np.isfinite(np.asarray(getattr(result, "x"))))
            and np.all(np.isfinite(np.asarray(getattr(result, "fun"))))
            and np.isfinite(float(getattr(result, "cost")))
            and np.isfinite(float(getattr(result, "optimality")))
        )
        if not finite:
            return False
        if bool(getattr(result, "success")):
            return True
        return int(getattr(result, "status")) == 0
    except (AttributeError, TypeError, ValueError):
        return False


def _optimizer_and_joint_margin_hard_ok(
    result: object,
    joint_margin_rad: np.ndarray,
    minimum_accepted_margin_rad: float,
) -> bool:
    """Return whether a candidate may pass the final solver/margin gate."""

    return _least_squares_result_is_acceptable(
        result
    ) and _joint_margin_hard_ok(
        joint_margin_rad,
        minimum_accepted_margin_rad,
    )


def main() -> None:
    global capsule_project
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--object-shape",
        choices=("capsule", "ellipsoid"),
        default="capsule",
    )
    parser.add_argument("--radius", type=float, default=0.10)
    parser.add_argument("--half-height", type=float, default=0.17)
    parser.add_argument("--clearance-mm", type=float, default=5.0)
    parser.add_argument(
        "--self-clearance-mm",
        type=float,
        default=0.5,
        help=(
            "Soft distance target for the three non-adjacent MCP-to-DIP "
            "geometry pairs that limit flexed-finger surface motion."
        ),
    )
    parser.add_argument(
        "--minimum-accepted-self-clearance-mm",
        type=float,
        default=0.0,
        help=(
            "Hard accepted margin for the MCP-to-DIP pairs. Use a positive "
            "value when optimizing a grasp for subsequent finger sliding."
        ),
    )
    parser.add_argument(
        "--tip-preload-mm",
        type=float,
        nargs=4,
        default=(0.25, 0.25, 0.75, 0.25),
        metavar=("INDEX", "MIDDLE", "RING", "THUMB"),
        help=(
            "Small signed collision-geometry preload for robust dynamic "
            "contact. The ring finger default is larger because its loaded "
            "servo pose otherwise separates by about 0.4 mm."
        ),
    )
    parser.add_argument("--max-nfev", type=int, default=500)
    parser.add_argument(
        "--max-pad-angle-deg",
        type=float,
        default=45.0,
        help=(
            "Maximum angle between each physical finger-pad normal and the "
            "local inward object-surface normal."
        ),
    )
    parser.add_argument(
        "--optimization-pad-angle-deg",
        type=float,
        default=35.0,
        help=(
            "Tighter pad cone used as the optimizer target, leaving margin "
            "inside --max-pad-angle-deg for physical collision refinement."
        ),
    )
    parser.add_argument(
        "--optimization-joint-margin-rad",
        type=float,
        default=0.0,
        help=(
            "Soft per-joint distance-to-limit target used by both grasp "
            "refinement stages. Zero preserves the legacy objective."
        ),
    )
    parser.add_argument(
        "--minimum-accepted-joint-margin-rad",
        type=float,
        default=0.0,
        help=(
            "Hard minimum distance of every saved joint from its nearest "
            "position limit. Zero preserves the legacy feasibility gate."
        ),
    )
    parser.add_argument(
        "--minimum-accepted-clearance-mm",
        type=float,
        default=4.0,
        help=(
            "Hard acceptance threshold. --clearance-mm remains the tighter "
            "soft optimization target."
        ),
    )
    parser.add_argument(
        "--pad-site-standoff-mm",
        type=float,
        default=1.0,
        help=(
            "Desired distance of the legacy physical tip-FSR center outside "
            "the object surface during smooth pad-side preconditioning."
        ),
    )
    parser.add_argument(
        "--tip-distance-tolerance-mm",
        type=float,
        default=0.6,
        help=(
            "Collision-geometry tolerance before the dynamic normal contact "
            "search. Real tactile contact is still mandatory before motion."
        ),
    )
    parser.add_argument(
        "--desired-tip-local-z-m",
        type=float,
        default=None,
        help=(
            "Optional mean fingertip height in the capsule frame. Use a "
            "positive value to leave enough straight cylindrical surface for "
            "a long downward slide without entering the end cap."
        ),
    )
    parser.add_argument(
        "--stage1-plan",
        type=Path,
        default=Path(
            "full_hand_mcc/outputs/"
            "stage1_tip_only_end_to_end_200mm_plan.npz"
        ),
    )
    parser.add_argument(
        "--seed-grasp",
        type=Path,
        default=None,
        help=(
            "Optional near-feasible NPZ used as the first local refinement "
            "seed; active robot self-collision pairs are explicitly removed."
        ),
    )
    parser.add_argument(
        "--seed-only",
        action="store_true",
        help="Refine only --seed-grasp instead of running all global starts.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=None,
        help=(
            "Select one indexed optimizer start after the optional seed is "
            "inserted; useful for reproducing and refining a specific arm "
            "kinematic branch."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "full_hand_mcc/outputs/"
            "full_robot_collision_free_grasp.npz"
        ),
    )
    args = parser.parse_args()
    if any(value < 0.0 for value in args.tip_preload_mm):
        raise ValueError("--tip-preload-mm cannot be negative")
    if not 0.0 < args.max_pad_angle_deg < 90.0:
        raise ValueError("--max-pad-angle-deg must be in (0, 90)")
    if not 0.0 < args.optimization_pad_angle_deg <= args.max_pad_angle_deg:
        raise ValueError(
            "--optimization-pad-angle-deg must be in "
            "(0, --max-pad-angle-deg]"
        )
    if not (
        np.isfinite(args.optimization_joint_margin_rad)
        and np.isfinite(args.minimum_accepted_joint_margin_rad)
        and 0.0 <= args.minimum_accepted_joint_margin_rad
        <= args.optimization_joint_margin_rad
    ):
        raise ValueError(
            "joint margins must satisfy 0 <= "
            "--minimum-accepted-joint-margin-rad <= "
            "--optimization-joint-margin-rad"
        )
    if not 0.0 <= args.minimum_accepted_clearance_mm <= args.clearance_mm:
        raise ValueError(
            "--minimum-accepted-clearance-mm must be in "
            "[0, --clearance-mm]"
        )
    if not (
        0.0
        <= args.minimum_accepted_self_clearance_mm
        <= args.self_clearance_mm
    ):
        raise ValueError(
            "--minimum-accepted-self-clearance-mm must be in "
            "[0, --self-clearance-mm]"
        )
    if args.tip_distance_tolerance_mm < 0.0:
        raise ValueError("--tip-distance-tolerance-mm cannot be negative")
    if args.seed_only and args.seed_grasp is None:
        raise ValueError("--seed-only requires --seed-grasp")

    if args.object_shape == "ellipsoid":
        if args.half_height <= 0.0:
            raise ValueError("Ellipsoid --half-height must be positive")
        capsule_project = ellipsoid_project
    env_module.FULL_HAND_OBJECT_SHAPE = args.object_shape
    env_module.FULL_HAND_CAPSULE_RADIUS = args.radius
    env_module.FULL_HAND_CAPSULE_HALF_HEIGHT = args.half_height
    solver = FivePointReachabilitySolver(
        tolerance=0.005,
        max_iterations=160,
        palm_weight=5.0,
    )
    rotation = np.eye(3)
    clearance = args.clearance_mm / 1000.0
    minimum_pad_alignment = float(
        np.cos(np.deg2rad(args.max_pad_angle_deg))
    )
    optimization_pad_alignment = float(
        np.cos(np.deg2rad(args.optimization_pad_angle_deg))
    )
    minimum_accepted_clearance = (
        args.minimum_accepted_clearance_mm / 1000.0
    )
    self_clearance = args.self_clearance_mm / 1000.0
    minimum_accepted_self_clearance = (
        args.minimum_accepted_self_clearance_mm / 1000.0
    )
    protected_self_pair_names = (
        ("mcp_joint_geom", "dip_geom"),
        ("mcp_joint_2_geom", "dip_2_geom"),
        ("mcp_joint_3_geom", "dip_3_geom"),
    )
    protected_self_pairs = tuple(
        (
            mujoco.mj_name2id(
                solver.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                first,
            ),
            mujoco.mj_name2id(
                solver.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                second,
            ),
        )
        for first, second in protected_self_pair_names
    )
    if any(first < 0 or second < 0 for first, second in protected_self_pairs):
        raise ValueError("One or more protected MCP-to-DIP geoms are missing")
    desired_tip_distance = (
        -np.asarray(args.tip_preload_mm, dtype=np.float64) / 1000.0
    )
    desired_pad_site_standoff = args.pad_site_standoff_mm / 1000.0
    desired_site_standoff = np.asarray(
        (0.0145, 0.0118, 0.0140, 0.0145),
        dtype=np.float64,
    )
    reference_q: np.ndarray | None = None
    if args.stage1_plan.exists():
        stage1 = np.load(args.stage1_plan)
        stage1_q = np.asarray(
            stage1["joint_positions_rad"][0],
            dtype=np.float64,
        )
        if stage1_q.shape == (TOTAL_DOF,):
            reference_q = stage1_q
        else:
            print(
                "[SEED] ignoring legacy stage-1 plan with "
                f"{stage1_q.size} DoF; FR3 requires {TOTAL_DOF}",
                flush=True,
            )
    if reference_q is None:
        reference_q = np.concatenate(
            (
                FR3_HOME_Q.astype(np.float64),
                np.asarray(DEFAULT_PREGRASP_Q, dtype=np.float64),
            )
        )
 
    starts = [
        np.asarray((0.56, -0.04, 0.48)),
        np.asarray((0.58, -0.02, 0.50)),
        np.asarray((0.54, -0.06, 0.50)),
        np.asarray((0.60, -0.08, 0.48)),
        np.asarray((0.54, 0.02, 0.48)),
        np.asarray((0.62, -0.02, 0.52)),
        np.asarray((0.50, -0.04, 0.52)),
        np.asarray((0.58, -0.10, 0.52)),
    ]
    q_starts: list[np.ndarray] = []
    arm_seed_offsets = (
        (0.0, 0.0, 0.0),
        (0.4, 0.0, 0.0),
        (-0.4, 0.0, 0.0),
        (0.0, 0.5, 0.0),
        (0.0, -0.5, 0.0),
        (0.0, 0.0, 0.8),
        (0.0, 0.0, -0.8),
        (0.3, -0.4, 0.6),
    )
    for joint1_offset, joint3_offset, joint7_offset in arm_seed_offsets:
        q_seed = reference_q.copy()
        q_seed[0] += joint1_offset
        q_seed[2] += joint3_offset
        q_seed[6] += joint7_offset
        q_seed = np.clip(
            q_seed,
            solver.lower + 1.0e-6,
            solver.upper - 1.0e-6,
        )
        q_starts.append(q_seed)
    if args.seed_grasp is not None:
        seed_grasp = np.load(args.seed_grasp)
        starts.insert(
            0,
            np.asarray(seed_grasp["object_center_m"], dtype=np.float64),
        )
        q_starts.insert(
            0,
            np.asarray(seed_grasp["joint_position_rad"], dtype=np.float64),
        )
        if args.seed_only:
            starts = starts[:1]
            q_starts = q_starts[:1]
    if args.start_index is not None:
        if not 0 <= args.start_index < len(starts):
            raise ValueError(
                f"--start-index must be in [0, {len(starts) - 1}]"
            )
        starts = [starts[args.start_index]]
        q_starts = [q_starts[args.start_index]]
    physical_reference_q = _select_physical_reference_q(
        reference_q,
        seed_only=args.seed_only,
        seed_q=q_starts[0] if args.seed_only else None,
    )
    lower = np.concatenate(
        (solver.lower, np.asarray((0.30, -0.50, 0.30)))
    )
    upper = np.concatenate(
        (solver.upper, np.asarray((1.00, 0.50, 0.90)))
    )
    results: list[tuple[float, np.ndarray, object]] = []

    for start_index, (start_center, q0) in enumerate(
        zip(starts, q_starts, strict=True)
    ):
        q0 = q0.copy()
        x0 = np.concatenate((q0, start_center))
        eval_count = 0
        active_self_collision_pairs, active_self_distances = (
            solver.self_collision_contacts(q0)
        )
        self_collision_pairs = tuple(
            sorted(
                set(active_self_collision_pairs)
                | set(protected_self_pairs)
            )
        )
        initial_self_distances = solver.geometry_pair_distances(
            q0,
            self_collision_pairs,
        )
        if active_self_collision_pairs:
            print(
                f"start={start_index} active_self_collision_pairs="
                f"{len(active_self_collision_pairs)} "
                f"deepest_mm={active_self_distances.min() * 1000:.3f}",
                flush=True,
            )
        print(
            f"start={start_index} protected_self_clearance_mm="
            f"{(solver.geometry_pair_distances(q0, protected_self_pairs) * 1000).round(3).tolist()}",
            flush=True,
        )

        def pad_state(
            q: np.ndarray,
            center: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            points = solver.forward_points(q)
            surface, surface_normals = capsule_project(
                points,
                center,
                rotation,
                args.radius,
                args.half_height,
            )
            pad_normals = solver.fingertip_pad_normals(q)
            pad_alignment = np.einsum(
                "ij,ij->i",
                pad_normals,
                -surface_normals[1:],
            )
            signed_site_standoff = np.einsum(
                "ij,ij->i",
                points[1:] - surface[1:],
                surface_normals[1:],
            )
            return (
                points,
                surface_normals,
                pad_alignment,
                signed_site_standoff,
            )

        def smooth_pad_residual(x: np.ndarray) -> np.ndarray:
            q = x[:TOTAL_DOF]
            center = x[TOTAL_DOF : TOTAL_DOF + 3]
            (
                points,
                _,
                pad_alignment,
                signed_site_standoff,
            ) = pad_state(q, center)
            alignment_barrier = np.maximum(
                optimization_pad_alignment - pad_alignment,
                0.0,
            )
            axial_layout_error = np.zeros(0, dtype=np.float64)
            if args.desired_tip_local_z_m is not None:
                local_points = (
                    rotation.T @ (points[1:] - center).T
                ).T
                axial_layout_error = 12.0 * (
                    local_points[:, 2] - args.desired_tip_local_z_m
                )
            return np.concatenate(
                (
                    220.0
                    * (
                        signed_site_standoff
                        - desired_pad_site_standoff
                    ),
                    12.0 * alignment_barrier,
                    1.5 * (1.0 - pad_alignment),
                    axial_layout_error,
                    120.0
                    * np.maximum(
                        self_clearance
                        - solver.geometry_pair_distances(
                            q,
                            self_collision_pairs,
                        ),
                        0.0,
                    ),
                    _joint_margin_soft_residual(
                        q,
                        solver.lower,
                        solver.upper,
                        args.optimization_joint_margin_rad,
                    ),
                    0.001 * (q - q0),
                    0.001 * (center - start_center),
                )
            )

        smooth_result = least_squares(
            smooth_pad_residual,
            x0,
            bounds=(lower, upper),
            max_nfev=min(args.max_nfev, 500),
            xtol=1.0e-9,
            ftol=1.0e-9,
            gtol=1.0e-9,
            x_scale="jac",
        )
        if not _least_squares_result_is_acceptable(smooth_result):
            print(
                f"start={start_index} optimizer_rejected_stage=smooth "
                f"success={bool(getattr(smooth_result, 'success', False))} "
                f"status={getattr(smooth_result, 'status', 'missing')}",
                flush=True,
            )
            continue
        if not smooth_result.success:
            print(
                f"start={start_index} optimizer_max_nfev_stage=smooth "
                "continuing_with_finite_seed=true",
                flush=True,
            )
        x0 = smooth_result.x
        (
            _,
            _,
            smooth_alignment,
            smooth_standoff,
        ) = pad_state(
            x0[:TOTAL_DOF],
            x0[TOTAL_DOF : TOTAL_DOF + 3],
        )
        print(
            f"start={start_index} smooth_pad_cost="
            f"{smooth_result.cost:.6f} nfev={smooth_result.nfev} "
            f"alignment={smooth_alignment.round(4).tolist()} "
            f"pad_angle_deg="
            f"{np.degrees(np.arccos(np.clip(smooth_alignment, -1, 1))).round(2).tolist()} "
            f"site_standoff_mm="
            f"{(smooth_standoff * 1000).round(3).tolist()}",
            flush=True,
        )

        def residual(x: np.ndarray) -> np.ndarray:
            nonlocal eval_count
            eval_count += 1
            q = x[:TOTAL_DOF]
            center = x[TOTAL_DOF : TOTAL_DOF + 3]
            (
                points,
                surface_normals,
                pad_alignment,
                signed_site_standoff,
            ) = pad_state(q, center)
            surface = points.copy()
            surface[1:] -= (
                signed_site_standoff[:, None]
                * surface_normals[1:]
            )
            site_standoff = np.linalg.norm(
                points[1:] - surface[1:],
                axis=1,
            )
            tip_distance, non_tip_distance, _ = (
                solver.geometry_clearances(q, center, rotation)
            )
            collision_barrier = np.maximum(
                clearance - non_tip_distance,
                0.0,
            )
            pad_alignment_barrier = np.maximum(
                optimization_pad_alignment - pad_alignment,
                0.0,
            )
            axial_layout_error = np.zeros(0, dtype=np.float64)
            if args.desired_tip_local_z_m is not None:
                local_points = (
                    rotation.T @ (points[1:] - center).T
                ).T
                axial_layout_error = 10.0 * (
                    local_points[:, 2] - args.desired_tip_local_z_m
                )
            return np.concatenate(
                (
                    tip_distance_weight
                    * (tip_distance - desired_tip_distance),
                    8.0
                    * (
                        signed_site_standoff
                        - desired_pad_site_standoff
                    ),
                    object_clearance_weight * collision_barrier,
                    pad_barrier_weight * pad_alignment_barrier,
                    0.5 * (1.0 - pad_alignment),
                    axial_layout_error,
                    self_collision_weight
                    * np.maximum(
                        self_clearance
                        - solver.geometry_pair_distances(
                            q,
                            self_collision_pairs,
                        ),
                        0.0,
                    ),
                    _joint_margin_soft_residual(
                        q,
                        solver.lower,
                        solver.upper,
                        args.optimization_joint_margin_rad,
                    ),
                    0.015 * (q - physical_reference_q),
                    0.01 * (center - start_center),
                )
            )

        physical_seed = x0
        self_collision_weight = 220.0
        tip_distance_weight = 240.0
        pad_barrier_weight = 8.0
        object_clearance_weight = 260.0
        result = None
        for collision_pass in range(10):
            candidate_result = least_squares(
                residual,
                physical_seed,
                bounds=(lower, upper),
                max_nfev=args.max_nfev,
                xtol=1.0e-10,
                ftol=1.0e-10,
                gtol=1.0e-10,
                verbose=1 if collision_pass == 0 else 0,
            )
            if not _least_squares_result_is_acceptable(candidate_result):
                print(
                    f"start={start_index} "
                    "optimizer_rejected_stage=physical "
                    f"pass={collision_pass + 1} "
                    "success="
                    f"{bool(getattr(candidate_result, 'success', False))} "
                    f"status={getattr(candidate_result, 'status', 'missing')}",
                    flush=True,
                )
                result = None
                break
            if not candidate_result.success:
                print(
                    f"start={start_index} "
                    "optimizer_max_nfev_stage=physical "
                    f"pass={collision_pass + 1} "
                    "continuing_to_hard_audit=true",
                    flush=True,
                )
            result = candidate_result
            remaining_pairs, remaining_distances = (
                solver.self_collision_contacts(result.x[:TOTAL_DOF])
            )
            (
                restore_tip_distance,
                restore_non_tip_distance,
                _,
            ) = solver.geometry_clearances(
                result.x[:TOTAL_DOF],
                result.x[TOTAL_DOF : TOTAL_DOF + 3],
                rotation,
            )
            _, _, restore_alignment, _ = pad_state(
                result.x[:TOTAL_DOF],
                result.x[TOTAL_DOF : TOTAL_DOF + 3],
            )
            restore_protected_self_distance = (
                solver.geometry_pair_distances(
                    result.x[:TOTAL_DOF],
                    protected_self_pairs,
                )
            )
            restore_joint_margin = _joint_margin_rad(
                result.x[:TOTAL_DOF],
                solver.lower,
                solver.upper,
            )
            tip_error = float(
                np.max(
                    np.abs(
                        restore_tip_distance - desired_tip_distance
                    )
                )
            )
            print(
                f"start={start_index} self_collision_pass="
                f"{collision_pass + 1} remaining_pairs="
                f"{len(remaining_pairs)} deepest_mm="
                f"{(remaining_distances.min() * 1000 if len(remaining_distances) else 0.0):.3f} "
                f"weight={self_collision_weight:.1f} "
                f"tip_error_mm={tip_error * 1000:.3f} "
                f"min_pad_alignment={restore_alignment.min():.4f} "
                f"object_clearance_mm="
                f"{restore_non_tip_distance.min() * 1000:.3f} "
                f"protected_self_clearance_mm="
                f"{(restore_protected_self_distance * 1000).round(3).tolist()} "
                f"min_joint_margin_rad="
                f"{float(np.min(restore_joint_margin)):.6f}",
                flush=True,
            )
            contact_and_pad_restored = bool(
                tip_error <= args.tip_distance_tolerance_mm / 1000.0
                and float(restore_alignment.min())
                >= minimum_pad_alignment
                and float(restore_non_tip_distance.min())
                >= minimum_accepted_clearance
                and float(restore_protected_self_distance.min())
                >= minimum_accepted_self_clearance
                and _joint_margin_hard_ok(
                    restore_joint_margin,
                    args.minimum_accepted_joint_margin_rad,
                )
            )
            if not remaining_pairs and contact_and_pad_restored:
                break
            self_collision_pairs = tuple(
                sorted(set(self_collision_pairs) | set(remaining_pairs))
            )
            if remaining_pairs:
                self_collision_weight *= 3.0
            else:
                # All discovered self-collisions are now separated. Restore
                # exact physical tip contact and pad orientation while keeping
                # those fixed pair distances inside the residual.
                self_collision_weight = max(
                    self_collision_weight,
                    5000.0,
                )
                tip_distance_weight = 1500.0
                pad_barrier_weight = 40.0
                object_clearance_weight = 5000.0
            physical_seed = result.x
        if result is None:
            continue
        q = result.x[:TOTAL_DOF]
        center = result.x[TOTAL_DOF : TOTAL_DOF + 3]
        tip_distance, non_tip_distance, non_tip_names = (
            solver.geometry_clearances(q, center, rotation)
        )
        points = solver.forward_points(q)
        surface, surface_normals = capsule_project(
            points,
            center,
            rotation,
            args.radius,
            args.half_height,
        )
        site_standoff = np.linalg.norm(
            points[1:] - surface[1:],
            axis=1,
        )
        local_tip_z = (
            rotation.T @ (points[1:] - center).T
        ).T[:, 2]
        pad_normals = solver.fingertip_pad_normals(q)
        pad_alignment = np.einsum(
            "ij,ij->i",
            pad_normals,
            -surface_normals[1:],
        )
        pad_angle_deg = np.degrees(
            np.arccos(np.clip(pad_alignment, -1.0, 1.0))
        )
        nearest_index = int(np.argmin(non_tip_distance))
        remaining_self_pairs, remaining_self_distances = (
            solver.self_collision_contacts(q)
        )
        protected_self_distances = solver.geometry_pair_distances(
            q,
            protected_self_pairs,
        )
        joint_margin = _joint_margin_rad(
            q,
            solver.lower,
            solver.upper,
        )
        optimizer_and_joint_margin_ok = _optimizer_and_joint_margin_hard_ok(
            result,
            joint_margin,
            args.minimum_accepted_joint_margin_rad,
        )
        feasible = bool(
            np.max(np.abs(tip_distance - desired_tip_distance))
            <= args.tip_distance_tolerance_mm / 1000.0
            and non_tip_distance[nearest_index] >= minimum_accepted_clearance
            and float(pad_alignment.min()) >= minimum_pad_alignment
            and len(remaining_self_pairs) == 0
            and float(protected_self_distances.min())
            >= minimum_accepted_self_clearance
            and optimizer_and_joint_margin_ok
        )
        score = float(
            np.max(np.abs(tip_distance - desired_tip_distance))
            + 10.0
            * max(
                minimum_accepted_clearance
                - non_tip_distance[nearest_index],
                0.0,
            )
            + 10.0
            * max(minimum_pad_alignment - float(pad_alignment.min()), 0.0)
            + 10.0
            * max(
                minimum_accepted_self_clearance
                - float(protected_self_distances.min()),
                0.0,
            )
            + 0.01 * (1.0 - float(pad_alignment.min()))
            + 10.0
            * max(
                args.minimum_accepted_joint_margin_rad
                - float(np.min(joint_margin)),
                0.0,
            )
        )
        if not feasible:
            score += 100.0
        print(
            f"start={start_index} feasible={feasible} score={score:.6f} "
            f"nfev={result.nfev} center={center.round(5).tolist()} "
            f"tip_distance_mm="
            f"{(tip_distance * 1000).round(3).tolist()} "
            f"target_tip_distance_mm="
            f"{(desired_tip_distance * 1000).round(3).tolist()} "
            f"site_standoff_mm="
            f"{(site_standoff * 1000).round(3).tolist()} "
            f"tip_local_z_m={local_tip_z.round(4).tolist()} "
            f"pad_alignment={pad_alignment.round(4).tolist()} "
            f"pad_angle_deg={pad_angle_deg.round(2).tolist()} "
            f"min_non_tip_clearance_mm="
            f"{non_tip_distance[nearest_index] * 1000:.3f} "
            f"self_collision_pairs={len(remaining_self_pairs)} "
            f"protected_self_clearance_mm="
            f"{(protected_self_distances * 1000).round(3).tolist()} "
            f"min_joint_margin_rad={float(np.min(joint_margin)):.6f} "
            f"deepest_self_mm="
            f"{(remaining_self_distances.min() * 1000 if len(remaining_self_distances) else 0.0):.3f} "
            f"nearest={non_tip_names[nearest_index]}"
        )
        results.append((score, result.x.copy(), result))

    feasible_results = [
        item
        for item in results
        if float(item[0]) < 1.0
    ]
    if not feasible_results:
        raise RuntimeError(
            "No physical finger-pad-side collision-free grasp was found; "
            "no invalid candidate was saved."
        )
    feasible_results.sort(key=lambda item: item[0])
    _, best_x, best_result = feasible_results[0]
    best_q = best_x[:TOTAL_DOF]
    best_center = best_x[TOTAL_DOF : TOTAL_DOF + 3]
    tip_distance, non_tip_distance, non_tip_names = (
        solver.geometry_clearances(best_q, best_center, rotation)
    )
    best_points = solver.forward_points(best_q)
    _, best_surface_normals = capsule_project(
        best_points,
        best_center,
        rotation,
        args.radius,
        args.half_height,
    )
    best_pad_normals = solver.fingertip_pad_normals(best_q)
    best_pad_alignment = np.einsum(
        "ij,ij->i",
        best_pad_normals,
        -best_surface_normals[1:],
    )
    best_protected_self_distances = solver.geometry_pair_distances(
        best_q,
        protected_self_pairs,
    )
    best_joint_margin = _joint_margin_rad(
        best_q,
        solver.lower,
        solver.upper,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        joint_position_rad=best_q,
        object_center_m=best_center,
        object_rotation=rotation,
        object_shape=np.asarray(args.object_shape),
        object_radius_m=np.asarray(args.radius),
        object_half_height_m=np.asarray(args.half_height),
        tip_distance_m=tip_distance,
        desired_tip_distance_m=desired_tip_distance,
        tip_local_z_m=(
            rotation.T
            @ (
                solver.forward_points(best_q)[1:]
                - best_center
            ).T
        ).T[:, 2],
        pad_normals_world=best_pad_normals,
        inward_surface_normals_world=-best_surface_normals[1:],
        pad_alignment=best_pad_alignment,
        max_pad_angle_deg=np.asarray(args.max_pad_angle_deg),
        non_tip_distance_m=non_tip_distance,
        non_tip_geom_names=np.asarray(non_tip_names),
        protected_self_pair_names=np.asarray(
            protected_self_pair_names,
        ),
        protected_self_distance_m=best_protected_self_distances,
        joint_margin_rad=best_joint_margin,
        minimum_joint_margin_rad=np.asarray(
            float(np.min(best_joint_margin))
        ),
        optimization_joint_margin_rad=np.asarray(
            args.optimization_joint_margin_rad
        ),
        minimum_accepted_joint_margin_rad=np.asarray(
            args.minimum_accepted_joint_margin_rad
        ),
        optimizer_cost=np.asarray(best_result.cost),
        optimizer_optimality=np.asarray(best_result.optimality),
        optimizer_success=np.asarray(bool(best_result.success)),
        optimizer_status=np.asarray(int(best_result.status)),
        optimizer_nfev=np.asarray(int(best_result.nfev)),
        optimizer_hard_feasible_max_nfev_override=np.asarray(
            not best_result.success and int(best_result.status) == 0
        ),
    )
    nearest_index = int(np.argmin(non_tip_distance))
    print(
        f"saved={args.output.resolve()} "
        f"center={best_center.round(6).tolist()} "
        f"max_tip_distance_mm="
        f"{np.max(np.abs(tip_distance - desired_tip_distance)) * 1000:.3f} "
        f"min_non_tip_clearance_mm="
        f"{non_tip_distance[nearest_index] * 1000:.3f} "
        f"protected_self_clearance_mm="
        f"{(best_protected_self_distances * 1000).round(3).tolist()} "
        f"min_joint_margin_rad={float(np.min(best_joint_margin)):.6f} "
        f"pad_alignment={best_pad_alignment.round(4).tolist()} "
        f"pad_angle_deg="
        f"{np.degrees(np.arccos(np.clip(best_pad_alignment, -1, 1))).round(2).tolist()} "
        f"nearest={non_tip_names[nearest_index]}"
    )


if __name__ == "__main__":
    main()
