"""Add a privileged four-fingertip posture plan to a palm trajectory.

The object is fixed at the object-frame origin.  For sparse palm-pose knots,
the solver jointly selects four reachable surface contacts and a healthy Leap
Hand posture from the exact catalog geometry.  It is deliberately independent
of an open-to-grasp intersection path: the first knot uses structured posture
multi-starts, and later knots use temporal continuation.  Dense per-frame
targets are then reconstructed after monotone joint interpolation.

This is an offline teacher-data tool: use of exact object geometry is
intentional and never becomes a deployment-policy input.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path

import h5py
import numpy as np
from scipy.interpolate import PchipInterpolator

from object_catalog import MeshNormalOracle, load_object_config
from surface_mcc_finger import (
    FullHandMCCFingerConfig,
    FullHandMCCFingerController,
    GeometrySurfaceOracle,
)


def _pose_rotation(quaternion_wxyz: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
    quaternion /= max(float(np.linalg.norm(quaternion)), 1.0e-12)
    w, x, y, z = quaternion
    return np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def _palm_to_object(points_palm: np.ndarray, palm_pose_object: np.ndarray) -> np.ndarray:
    pose = np.asarray(palm_pose_object, dtype=np.float64).reshape(7)
    return pose[:3] + (_pose_rotation(pose[3:7]) @ np.asarray(points_palm).T).T


def _object_to_palm(points_object: np.ndarray, palm_pose_object: np.ndarray) -> np.ndarray:
    pose = np.asarray(palm_pose_object, dtype=np.float64).reshape(7)
    rotation = _pose_rotation(pose[3:7])
    return (rotation.T @ (np.asarray(points_object) - pose[:3]).T).T


def _solve_frame(
    controller: FullHandMCCFingerController,
    oracle: GeometrySurfaceOracle,
    palm_pose_object: np.ndarray,
    seed_q: np.ndarray,
    nominal_q: np.ndarray,
    *,
    preload_m: float,
    projection_iterations: int,
    posture_gain: float,
    pad_alignment_gain: float,
    multistart: bool,
) -> tuple[np.ndarray, dict[str, np.ndarray | float | int]]:
    # The geometry oracle is already expressed in object coordinates.  Treat
    # that frame as world for this offline solve and pass the palm pose
    # directly to the shared planner used by online recovery.
    q, result = controller.solve_geometry_contact_posture(
        oracle,
        palm_pose_object,
        seed_q,
        nominal_q,
        preload_m=preload_m,
        projection_iterations=projection_iterations,
        posture_gain=posture_gain,
        pad_alignment_gain=pad_alignment_gain,
        multistart=multistart,
    )
    signed_distance = np.asarray(result["signed_distance"], dtype=np.float32)
    return q.astype(np.float32), {
        "surface_point": np.asarray(
            result["surface_point_world"], dtype=np.float32
        ),
        "surface_normal": np.asarray(
            result["surface_normal_world"], dtype=np.float32
        ),
        "signed_distance": signed_distance,
        "ik_residual": np.abs(signed_distance + float(preload_m)).astype(
            np.float32
        ),
        "pad_normal_error": np.asarray(
            result["pad_normal_error"], dtype=np.float32
        ),
        "synergy_spread": np.asarray(
            result["synergy_spread"], dtype=np.float32
        ),
        "synergy_residual": np.asarray(
            result["synergy_residual"], dtype=np.float32
        ),
        "isotropic_reachability": np.asarray(
            result["isotropic_reachability"], dtype=np.float32
        ),
        "manipulability": np.asarray(
            result["manipulability"], dtype=np.float32
        ),
        "ik_iterations": int(projection_iterations),
        "planner_valid": bool(result["valid"]),
        "planner_score": float(result["score"]),
        "true_tip_geometry_valid": bool(result["valid"]),
        "connector_distance": np.asarray(result.get("connector_distance", np.full(4, np.inf))),
        "self_penetration": float(result.get("self_penetration", 0.)),
        "top_fraction": np.asarray(result.get("top_fraction", np.full(4, np.nan))),
        "non_tip_distance": np.asarray(result.get("non_tip_distance", np.zeros(0))),
    }


def _evaluate_frame(
    controller: FullHandMCCFingerController,
    oracle: GeometrySurfaceOracle,
    palm_pose_object: np.ndarray,
    q: np.ndarray,
) -> dict[str, np.ndarray]:
    """Evaluate the interpolated posture itself without another IK solve."""

    tip_palm = controller.tip_positions_palm(q)
    tip_object = _palm_to_object(tip_palm, palm_pose_object)
    observation = oracle.observe(tip_object)
    normals_object = np.asarray(observation.normals_world, dtype=np.float64)
    normals_palm = (
        _pose_rotation(palm_pose_object[3:7]).T @ normals_object.T
    ).T
    palm_pose_identity = np.asarray(
        (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0), dtype=np.float64
    )
    pad_error = controller.pad_normal_errors(
        q, palm_pose_identity, normals_palm
    )
    synergy_spread, synergy_residual = controller.flexion_synergy_metrics(q)
    health_maps = controller._kinematic_health_maps()
    controller._set_q(controller.data, q)
    isotropic_reachability = np.zeros(4, dtype=np.float64)
    manipulability = np.zeros(4, dtype=np.float64)
    for finger in range(4):
        isotropic_reachability[finger], manipulability[finger] = (
            controller._finger_kinematic_health(
                finger, tip_palm[finger], health_maps[finger]
            )
        )
    result = {
        "surface_point": np.asarray(observation.points_world, dtype=np.float32),
        "surface_normal": normals_object.astype(np.float32),
        "signed_distance": np.asarray(observation.signed_distance, dtype=np.float32),
        "pad_normal_error": pad_error.astype(np.float32),
        "synergy_spread": synergy_spread.astype(np.float32),
        "synergy_residual": synergy_residual.astype(np.float32),
        "isotropic_reachability": isotropic_reachability.astype(np.float32),
        "manipulability": manipulability.astype(np.float32),
    }
    if getattr(controller, 'non_tip_clearance_m', None) is not None:
        geometry = controller.geometry.evaluate(q, palm_pose_object,
            self_collision=True, non_tip_clearance=True)
        result.update(signed_distance=geometry['distance'].astype(np.float32),
            non_tip_distance=geometry['non_tip_distance'].astype(np.float32),
            self_penetration=geometry['self_penetration'],
            true_tip_geometry_valid=controller.geometry_valid(geometry))
    return result


def _screening_violation(
    debug: dict[str, np.ndarray | float | int], args: argparse.Namespace
) -> float:
    """Dimensionless violation used to trigger/choose a branch replan."""
    signed = np.asarray(debug["signed_distance"], dtype=np.float64)
    pad = np.asarray(debug["pad_normal_error"], dtype=np.float64)
    isotropic = np.asarray(debug["isotropic_reachability"], dtype=np.float64)
    manipulability = np.asarray(debug["manipulability"], dtype=np.float64)
    outside = np.maximum(signed - float(args.max_outside_distance_m), 0.0)
    inside = np.maximum(-float(args.max_inside_distance_m) - signed, 0.0)
    pad_limit = np.deg2rad(float(args.max_pad_normal_error_deg))
    pad_excess = np.maximum(pad - pad_limit, 0.0)
    isotropic_shortfall = np.maximum(
        float(args.min_isotropic_reachability) - isotropic, 0.0
    )
    manipulability_shortfall = np.maximum(
        float(args.min_manipulability) - manipulability, 0.0
    )
    extra = 0.
    if args.true_tip_waypoints:
        extra = (0. if debug['true_tip_geometry_valid'] else 1.)
        extra += float(np.sum((np.maximum(-.0005-debug['connector_distance'], 0)/.001)**2))
        clearance = getattr(args, 'non_tip_clearance_m', None)
        if clearance is not None:
            extra += float(np.sum((np.maximum(clearance-debug['non_tip_distance'],0)/.001)**2))
            extra += float((max(debug['self_penetration']-args.max_cross_finger_penetration_m,0)/.001)**2)
    return float(
        np.sum((outside / 0.001) ** 2)
        + np.sum((inside / 0.001) ** 2)
        + np.sum((pad_excess / np.deg2rad(5.0)) ** 2)
        + np.sum((isotropic_shortfall / 1.0e-3) ** 2)
        + np.sum((manipulability_shortfall / 0.01) ** 2)+extra
    )


def optimize(args: argparse.Namespace) -> None:
    with h5py.File(args.input, "r") as source:
        if "palm_pose_object" not in source:
            raise KeyError(f"{args.input} has no palm_pose_object dataset")
        palm_pose = np.asarray(source["palm_pose_object"], dtype=np.float64)
        if palm_pose.ndim == 3:
            if palm_pose.shape[1] != 1:
                raise ValueError("optimize one palm trajectory at a time")
            palm_pose = palm_pose[:, 0]
        source_datasets = {
            name: np.asarray(dataset)
            for name, dataset in source.items()
            if name not in {
                "finger_q_plan",
                "finger_tip_surface_object",
                "finger_surface_normal_object",
                "finger_tip_signed_distance",
                "finger_ik_residual",
                "finger_pad_normal_error",
                "finger_synergy_spread",
                "finger_synergy_residual",
                "finger_isotropic_reachability",
                "finger_manipulability",
                "finger_plan_feasible",
                "grasp_keyframe_frame_index",
                "grasp_keyframe_q",
                "grasp_keyframe_contact_point_object",
                "grasp_keyframe_normal_object",
                "grasp_keyframe_signed_distance",
                "grasp_keyframe_pad_normal_error",
                "grasp_keyframe_synergy_residual",
                "grasp_keyframe_isotropic_reachability",
                "grasp_keyframe_manipulability",
                "grasp_keyframe_lateral_margin",
                "grasp_keyframe_posture_deviation",
                "grasp_keyframe_valid",
                "grasp_keyframe_true_tip_geometry_valid",
                "grasp_keyframe_tip_top_fraction",
                "grasp_keyframe_non_tip_distance",
                "grasp_keyframe_cross_finger_penetration",
                "non_tip_geom_name",
                "finger_non_tip_distance",
                "finger_cross_finger_penetration",
                "finger_true_tip_geometry_valid",
            }
        }
        source_attrs = dict(source.attrs)

    config = load_object_config(args.object_id)
    mesh_normals = MeshNormalOracle.from_config(config, scale=args.object_scale)
    oracle = GeometrySurfaceOracle(
        config,
        scale=args.object_scale,
        mesh_normal_oracle=mesh_normals,
    )
    oracle.set_pose(np.zeros(3), np.asarray((1.0, 0.0, 0.0, 0.0)))
    nominal = np.asarray(
        config.collection.get(
            "pregrasp_q",
            (0.8, 0.0, 0.4, 0.3) * 4,
        ),
        dtype=np.float64,
    ).reshape(16)
    controller = FullHandMCCFingerController(
        FullHandMCCFingerConfig(
            grasp_closure_q=tuple(float(v) for v in nominal),
            posture_cost=0.15,
            tip_orientation_cost=0.0,
            use_lateral_reference_regularizer=True,
            flexion_synergy_gain=0.18,
            flexion_synergy_hard_gain=0.75,
            flexion_synergy_spread_threshold=0.75,
            flexion_synergy_max_step=0.025,
        )
    )
    if args.true_tip_waypoints:
        from true_tip_waypoint_optimizer import TrueTipWaypointController
        controller = TrueTipWaypointController(controller, config, args.object_scale,
            args.tip_top_weight, args.tip_top_preferred_fraction,
            non_tip_clearance_m=getattr(args, 'non_tip_clearance_m', None),
            max_cross_finger_penetration_m=getattr(args, 'max_cross_finger_penetration_m', .0001),
            clearance_polish=getattr(args, 'clearance_polish', True),
            preferred_non_tip_clearance_m=getattr(args, 'non_tip_preferred_clearance_m', None))
        controller.measured_seed_mode = args.reference_trajectory is not None or args.reference_bank is not None
        # Whole fingertip orientation is free. Geometry/contact health gates
        # remain; angular measurements are diagnostics rather than front-axis
        # acceptance thresholds.
        args.max_pad_normal_error_deg = 180.
    nominal = controller.clamp_joint_positions(nominal).astype(np.float64)
    knot_indices = np.unique(
        np.concatenate(
            (
                np.arange(0, len(palm_pose), max(1, args.knot_stride)),
                np.asarray(getattr(args, 'extra_keyframe_frames', None) or [], dtype=np.int32),
                np.asarray((len(palm_pose) - 1,)),
            )
        )
    ).astype(np.int32)
    knot_q = np.zeros((len(knot_indices), 16), dtype=np.float32)
    knot_surface = np.zeros((len(knot_indices), 4, 3), dtype=np.float32)
    knot_normal = np.zeros_like(knot_surface)
    knot_signed_distance = np.zeros((len(knot_indices), 4), dtype=np.float32)
    knot_pad_error = np.zeros_like(knot_signed_distance)
    knot_synergy_residual = np.zeros_like(knot_signed_distance)
    knot_isotropic_reachability = np.zeros_like(knot_signed_distance)
    knot_manipulability = np.zeros_like(knot_signed_distance)
    knot_lateral_margin = np.zeros((len(knot_indices), 2), dtype=np.float32)
    knot_posture_deviation = np.zeros(len(knot_indices), dtype=np.float32)
    true_tip_geometry_valid = np.ones(len(knot_indices), dtype=bool)
    knot_top_fraction = np.full((len(knot_indices), 4), np.nan, dtype=np.float32)
    all_link_clearance = getattr(args, 'non_tip_clearance_m', None) is not None
    if all_link_clearance:
        knot_non_tip_distance = np.full((len(knot_indices), len(controller.geometry.non_tip_geoms)),
                                        np.nan, dtype=np.float32)
        knot_cross_finger_penetration = np.full(len(knot_indices), np.nan, dtype=np.float32)
    lateral_axis = np.asarray(
        controller.config.qp_lateral_axis_palm, dtype=np.float64
    )
    lateral_axis /= max(float(np.linalg.norm(lateral_axis)), 1.0e-12)
    nominal_tip = controller.tip_positions_palm(nominal)[:3]
    nominal_signed = np.asarray(
        [
            lateral_axis @ (nominal_tip[left] - nominal_tip[right])
            for left, right in ((0, 1), (1, 2))
        ],
        dtype=np.float64,
    )
    lateral_order_sign = np.where(nominal_signed >= 0.0, 1.0, -1.0)
    if args.true_tip_waypoints:
        controller.health_limits = dict(posture=args.max_posture_deviation_rad,
            eta=args.min_isotropic_reachability,mu=args.min_manipulability,
            lateral=args.min_lateral_margin_m,axis=lateral_axis,order=lateral_order_sign)
    # The first contact is solved directly from exact geometry with a
    # structured posture bank.  No open->grasp path or path/surface crossing
    # participates in contact selection.
    q_previous = nominal.copy()
    reference_q = None
    reference_bank = None
    if args.reference_bank is not None:
        reference_bank = np.load(args.reference_bank)
    if args.reference_trajectory is not None:
        with h5py.File(args.reference_trajectory, 'r') as reference:
            reference_q = np.asarray(reference['q_hand'], dtype=np.float64).reshape(-1, 16)
        if len(reference_q) != len(palm_pose):
            raise ValueError('reference trajectory must have the same full control horizon')
    for knot_id, frame in enumerate(knot_indices):
        if reference_q is not None:
            q_previous = reference_q[frame].copy()
        elif reference_bank is not None and not (
                getattr(args, 'temporal_seed_first', False) and knot_id > 0):
            poses = reference_bank['palm_pose_object']
            distance = (np.linalg.norm(poses[:, :3]-palm_pose[frame, :3], axis=-1)/.03
                + 2*np.arccos(np.clip(np.abs(poses[:, 3:] @ palm_pose[frame, 3:]), 0, 1)))
            candidates = np.argsort(distance)[:4]
            best_q, best_cost = q_previous, float('inf')
            for index in candidates:
                value = reference_bank['q_hand'][index]
                geometry = controller.geometry.evaluate(value, palm_pose[frame])
                cost = (np.sum((geometry['distance']+args.preload_m)**2)
                    + 4*np.sum(np.maximum(-.0005-geometry['connector_distance'], 0)**2))
                if cost < best_cost: best_q, best_cost = value, cost
            q_previous = np.asarray(best_q).copy()
        q_seed = q_previous.copy()
        q_previous, debug = _solve_frame(
            controller,
            oracle,
            palm_pose[frame],
            q_previous,
            nominal,
            preload_m=args.preload_m,
            projection_iterations=args.projection_iterations,
            posture_gain=args.ik_posture_gain,
            pad_alignment_gain=args.ik_pad_weight,
            multistart=(knot_id == 0),
        )
        violation = _screening_violation(debug, args)
        if args.retry_multistart and knot_id > 0 and violation > 0.0:
            retry_q, retry_debug = _solve_frame(
                controller,
                oracle,
                palm_pose[frame],
                q_seed,
                nominal,
                preload_m=args.preload_m,
                projection_iterations=args.projection_iterations,
                posture_gain=args.ik_posture_gain,
                pad_alignment_gain=args.ik_pad_weight,
                multistart=True,
            )
            retry_violation = _screening_violation(retry_debug, args)
            if (
                retry_violation < violation
                or (
                    np.isclose(retry_violation, violation)
                    and float(retry_debug["planner_score"])
                    < float(debug["planner_score"])
                )
            ):
                q_previous, debug = retry_q, retry_debug
        if (reference_bank is not None and getattr(args, 'temporal_seed_first', False)
                and knot_id > 0 and _screening_violation(debug, args) > 0):
            # Retry the bank only after temporal continuation fails. The bank
            # supplies initial guesses, never replacement trajectory labels.
            poses = reference_bank['palm_pose_object']
            distances = (np.linalg.norm(poses[:, :3]-palm_pose[frame, :3], axis=-1)/.03
                + 2*np.arccos(np.clip(np.abs(poses[:, 3:] @ palm_pose[frame, 3:]), 0, 1)))
            for index in np.argsort(distances)[:2]:
                retry_q, retry_debug = _solve_frame(controller, oracle, palm_pose[frame],
                    reference_bank['q_hand'][index], nominal, preload_m=args.preload_m,
                    projection_iterations=args.projection_iterations,
                    posture_gain=args.ik_posture_gain, pad_alignment_gain=args.ik_pad_weight,
                    multistart=False)
                old_violation = _screening_violation(debug, args)
                new_violation = _screening_violation(retry_debug, args)
                if new_violation < old_violation:
                    q_previous, debug = retry_q, retry_debug
                if _screening_violation(debug, args) == 0:
                    break
        knot_q[knot_id] = q_previous
        knot_surface[knot_id] = debug["surface_point"]
        knot_normal[knot_id] = debug["surface_normal"]
        knot_signed_distance[knot_id] = debug["signed_distance"]
        knot_pad_error[knot_id] = debug["pad_normal_error"]
        knot_synergy_residual[knot_id] = debug["synergy_residual"]
        knot_isotropic_reachability[knot_id] = debug[
            "isotropic_reachability"
        ]
        knot_manipulability[knot_id] = debug["manipulability"]
        tip_palm = controller.tip_positions_palm(q_previous)[:3]
        knot_lateral_margin[knot_id] = np.asarray(
            [
                lateral_order_sign[pair] * lateral_axis
                @ (tip_palm[left] - tip_palm[right])
                for pair, (left, right) in enumerate(((0, 1), (1, 2)))
            ],
            dtype=np.float32,
        )
        knot_posture_deviation[knot_id] = float(
            np.max(np.abs(np.asarray(q_previous) - nominal))
        )
        if args.true_tip_waypoints:
            true_tip_geometry_valid[knot_id] = debug['true_tip_geometry_valid']
            knot_top_fraction[knot_id] = debug['top_fraction']
        if all_link_clearance:
            knot_non_tip_distance[knot_id] = debug['non_tip_distance']
            knot_cross_finger_penetration[knot_id] = debug['self_penetration']
        if knot_id % max(1, len(knot_indices) // 10) == 0:
            print(
                f"[CONTACT-PLAN] knot={knot_id + 1}/{len(knot_indices)} "
                f"frame={frame}",
                flush=True,
            )

    dense_axis = np.arange(len(palm_pose), dtype=np.float64)
    q_plan = np.stack(
        [
            PchipInterpolator(knot_indices, knot_q[:, joint])(dense_axis)
            for joint in range(16)
        ],
        axis=1,
    )
    q_plan = np.stack(
        [controller.clamp_joint_positions(q) for q in q_plan], axis=0
    ).astype(np.float32)

    surface = np.zeros((len(palm_pose), 4, 3), dtype=np.float32)
    normal = np.zeros_like(surface)
    signed_distance = np.zeros((len(palm_pose), 4), dtype=np.float32)
    ik_residual = np.zeros_like(signed_distance)
    pad_error = np.zeros_like(signed_distance)
    synergy_spread = np.zeros_like(signed_distance)
    synergy_residual = np.zeros_like(signed_distance)
    isotropic_reachability = np.zeros_like(signed_distance)
    manipulability = np.zeros_like(signed_distance)
    if all_link_clearance:
        dense_non_tip_distance = np.full((len(palm_pose), len(controller.geometry.non_tip_geoms)),
                                         np.nan, dtype=np.float32)
        dense_cross_finger_penetration = np.full(len(palm_pose), np.nan, dtype=np.float32)
        dense_geometry_valid = np.zeros(len(palm_pose), dtype=bool)
    # Sparse collection consumes only these26 boundary conditions. Avoid
    # running2500 diagnostic geometry queries in keyframe-only mode.
    dense_frames = [] if args.keyframes_only else zip(palm_pose, q_plan, strict=True)
    for frame, (pose, q) in enumerate(dense_frames):
        debug = _evaluate_frame(controller, oracle, pose, q)
        surface[frame] = debug["surface_point"]
        normal[frame] = debug["surface_normal"]
        signed_distance[frame] = debug["signed_distance"]
        ik_residual[frame] = np.abs(
            signed_distance[frame] + float(args.preload_m)
        )
        pad_error[frame] = debug["pad_normal_error"]
        synergy_spread[frame] = debug["synergy_spread"]
        synergy_residual[frame] = debug["synergy_residual"]
        isotropic_reachability[frame] = debug["isotropic_reachability"]
        manipulability[frame] = debug["manipulability"]
        if all_link_clearance:
            dense_non_tip_distance[frame] = debug['non_tip_distance']
            dense_cross_finger_penetration[frame] = debug['self_penetration']
            dense_geometry_valid[frame] = debug['true_tip_geometry_valid']

    feasible = (
        (signed_distance <= args.max_outside_distance_m).all(axis=1)
        & (signed_distance >= -args.max_inside_distance_m).all(axis=1)
        & (pad_error <= np.deg2rad(args.max_pad_normal_error_deg)).all(axis=1)
        & (isotropic_reachability >= args.min_isotropic_reachability).all(axis=1)
        & (manipulability >= args.min_manipulability).all(axis=1)
    )
    if all_link_clearance:
        feasible &= dense_geometry_valid
    knot_valid = (
        (knot_signed_distance <= args.max_outside_distance_m).all(axis=1)
        & (knot_signed_distance >= -args.max_inside_distance_m).all(axis=1)
        & (knot_pad_error <= np.deg2rad(args.max_pad_normal_error_deg)).all(axis=1)
        & (
            knot_isotropic_reachability >= args.min_isotropic_reachability
        ).all(axis=1)
        & (knot_manipulability >= args.min_manipulability).all(axis=1)
        & (knot_lateral_margin >= args.min_lateral_margin_m).all(axis=1)
        & (knot_posture_deviation <= args.max_posture_deviation_rad)
        & true_tip_geometry_valid
    )

    if args.keyframes_only:
        # Screening mode: evaluate the identical criteria on the 26 grasp
        # keyframes only (what the inverse collector actually consumes) and
        # skip the dense PCHIP interpolation / per-frame evaluation.  Dense
        # q is retained for visualization only; online control connects
        # adjacent stable contacts with a local tangent-plane QP instead.
        max_step = 0.0
        feasible = np.zeros(len(palm_pose), dtype=bool)
        print(
            "[GRASP-KEYFRAMES] "
            f"count={len(knot_indices)} valid={float(np.mean(knot_valid)):.1%} "
            f"lateral_min_mm={float(np.min(knot_lateral_margin)) * 1000.0:.1f} "
            f"posture_max={float(np.max(knot_posture_deviation)):.3f}rad"
        )
        print(
            f"[KEYFRAMES-ONLY] dense frame evaluation skipped for {args.output}"
        )
    else:
        dense_axis = np.arange(len(palm_pose), dtype=np.float64)
        q_plan = np.stack(
            [
                PchipInterpolator(knot_indices, knot_q[:, joint])(dense_axis)
                for joint in range(16)
            ],
            axis=1,
        )
        q_plan = np.stack(
            [controller.clamp_joint_positions(q) for q in q_plan], axis=0
        ).astype(np.float32)

        surface = np.zeros((len(palm_pose), 4, 3), dtype=np.float32)
        normal = np.zeros_like(surface)
        signed_distance = np.zeros((len(palm_pose), 4), dtype=np.float32)
        ik_residual = np.zeros_like(signed_distance)
        pad_error = np.zeros_like(signed_distance)
        synergy_spread = np.zeros_like(signed_distance)
        synergy_residual = np.zeros_like(signed_distance)
        isotropic_reachability = np.zeros_like(signed_distance)
        manipulability = np.zeros_like(signed_distance)
        for frame, (pose, q) in enumerate(
            zip(palm_pose, q_plan, strict=True)
        ):
            debug = _evaluate_frame(controller, oracle, pose, q)
            surface[frame] = debug["surface_point"]
            normal[frame] = debug["surface_normal"]
            signed_distance[frame] = debug["signed_distance"]
            ik_residual[frame] = np.abs(
                signed_distance[frame] + float(args.preload_m)
            )
            pad_error[frame] = debug["pad_normal_error"]
            synergy_spread[frame] = debug["synergy_spread"]
            synergy_residual[frame] = debug["synergy_residual"]
            isotropic_reachability[frame] = debug["isotropic_reachability"]
            manipulability[frame] = debug["manipulability"]

        feasible = (
            (signed_distance <= args.max_outside_distance_m).all(axis=1)
            & (signed_distance >= -args.max_inside_distance_m).all(axis=1)
            & (pad_error <= np.deg2rad(args.max_pad_normal_error_deg)).all(axis=1)
            & (isotropic_reachability >= args.min_isotropic_reachability).all(
                axis=1
            )
            & (manipulability >= args.min_manipulability).all(axis=1)
        )
        if all_link_clearance:
            feasible &= dense_geometry_valid
        max_step = float(np.max(np.abs(np.diff(q_plan, axis=0))))
        print(
            "[CONTACT-PLAN-RESULT] "
            f"frames={len(q_plan)} feasible={float(np.mean(feasible)):.1%} "
            f"all_feasible={bool(np.all(feasible))} "
            f"outside_p95_mm={np.percentile(np.maximum(signed_distance, 0), 95) * 1000:.2f} "
            f"inside_p95_mm={np.percentile(np.maximum(-signed_distance, 0), 95) * 1000:.2f} "
            f"pad_p95_deg={np.degrees(np.percentile(pad_error, 95)):.1f} "
            f"eta_p05={np.percentile(isotropic_reachability, 5):.3f} "
            f"mu_p05={np.percentile(manipulability, 5):.3f} "
            f"max_q_step={max_step:.4f}rad"
        )
        print(
            "[GRASP-KEYFRAMES] "
            f"count={len(knot_indices)} valid={float(np.mean(knot_valid)):.1%} "
            f"lateral_min_mm={float(np.min(knot_lateral_margin)) * 1000.0:.1f} "
            f"posture_max={float(np.max(knot_posture_deviation)):.3f}rad"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output, "w") as output:
        for name, value in source_datasets.items():
            output.create_dataset(name, data=value)
        if not args.keyframes_only:
            # Dense finger_* datasets are only meaningful when the dense
            # pass ran; screening outputs keep the source plan plus the
            # sparse grasp-keyframe boundary conditions consumed by the
            # inverse collector.
            output.create_dataset("finger_q_plan", data=q_plan)
            output.create_dataset("finger_tip_surface_object", data=surface)
            output.create_dataset("finger_surface_normal_object", data=normal)
            output.create_dataset(
                "finger_tip_signed_distance", data=signed_distance
            )
            output.create_dataset("finger_ik_residual", data=ik_residual)
            output.create_dataset("finger_pad_normal_error", data=pad_error)
            output.create_dataset("finger_synergy_spread", data=synergy_spread)
            output.create_dataset(
                "finger_synergy_residual", data=synergy_residual
            )
            output.create_dataset(
                "finger_isotropic_reachability",
                data=isotropic_reachability,
            )
            output.create_dataset("finger_manipulability", data=manipulability)
            output.create_dataset(
                "finger_plan_feasible", data=feasible.astype(np.uint8)
            )
        # These sparse arrays are the boundary conditions consumed by the
        # inverse collector.  Dense q is retained for visualization only;
        # online control connects adjacent stable contacts with a local
        # tangent-plane QP instead of tracking dense IK joint positions.
        output.create_dataset("grasp_keyframe_frame_index", data=knot_indices)
        output.create_dataset("grasp_keyframe_q", data=knot_q)
        output.create_dataset(
            "grasp_keyframe_contact_point_object", data=knot_surface
        )
        output.create_dataset("grasp_keyframe_normal_object", data=knot_normal)
        output.create_dataset(
            "grasp_keyframe_signed_distance", data=knot_signed_distance
        )
        output.create_dataset(
            "grasp_keyframe_pad_normal_error", data=knot_pad_error
        )
        output.create_dataset(
            "grasp_keyframe_synergy_residual", data=knot_synergy_residual
        )
        output.create_dataset(
            "grasp_keyframe_isotropic_reachability",
            data=knot_isotropic_reachability,
        )
        output.create_dataset(
            "grasp_keyframe_manipulability", data=knot_manipulability
        )
        output.create_dataset(
            "grasp_keyframe_lateral_margin", data=knot_lateral_margin
        )
        output.create_dataset(
            "grasp_keyframe_posture_deviation", data=knot_posture_deviation
        )
        output.create_dataset(
            "grasp_keyframe_valid", data=knot_valid.astype(np.uint8)
        )
        if args.true_tip_waypoints:
            output.create_dataset("grasp_keyframe_true_tip_geometry_valid", data=true_tip_geometry_valid.astype(np.uint8))
            output.create_dataset("grasp_keyframe_tip_top_fraction", data=knot_top_fraction)
        if all_link_clearance:
            output.create_dataset('grasp_keyframe_non_tip_distance', data=knot_non_tip_distance)
            output.create_dataset('grasp_keyframe_cross_finger_penetration', data=knot_cross_finger_penetration)
            output.create_dataset('non_tip_geom_name', data=np.asarray(controller.geometry.non_tip_names,
                                  dtype=h5py.string_dtype('utf-8')))
            if not args.keyframes_only:
                output.create_dataset('finger_non_tip_distance', data=dense_non_tip_distance)
                output.create_dataset('finger_cross_finger_penetration', data=dense_cross_finger_penetration)
                output.create_dataset('finger_true_tip_geometry_valid', data=dense_geometry_valid.astype(np.uint8))
        for key, value in source_attrs.items():
            output.attrs[key] = value
        output.attrs["finger_contact_plan"] = True
        if args.true_tip_waypoints:
            output.attrs["collision_geometry"] = "true_fingertip_contact_v1_split"
            output.attrs["tip_top_preference_weight"] = args.tip_top_weight
            output.attrs["tip_top_preferred_fraction"] = args.tip_top_preferred_fraction
            output.attrs["tip_top_preference_is_hard_gate"] = False
            output.attrs["fixed_pad_orientation_constraint"] = False
            output.attrs["sampling_required_valid_ratio"] = 1.
        output.attrs['finger_contact_posture_gain'] = float(args.ik_posture_gain)
        if all_link_clearance:
            output.attrs['non_tip_clearance_m'] = float(args.non_tip_clearance_m)
            if getattr(args, 'non_tip_preferred_clearance_m', None) is not None:
                output.attrs['non_tip_preferred_clearance_m'] = float(args.non_tip_preferred_clearance_m)
            output.attrs['max_cross_finger_penetration_m'] = float(args.max_cross_finger_penetration_m)
            output.attrs['non_tip_clearance_scope'] = 'all active hand collision geoms except the four split true-tip pads'
            output.attrs['non_tip_clearance_acceptance'] = 'strict final geometry rejection; least-squares penalties do not guarantee feasibility'
            output.attrs['clearance_constrained_polish'] = bool(getattr(args,'clearance_polish',True))
            output.attrs['clearance_acceptance_representation'] = 'float32 joint waypoint, normalized float64 geometry pose'
            output.attrs['clearance_solver_sha256'] = hashlib.sha256(
                Path(__file__).with_name('true_tip_waypoint_optimizer.py').read_bytes()).hexdigest()
        output.attrs["finger_contact_knot_stride"] = int(args.knot_stride)
        output.attrs["finger_contact_preload_m"] = float(args.preload_m)
        output.attrs["grasp_keyframe_valid_ratio"] = float(np.mean(knot_valid))
        if not args.keyframes_only:
            output.attrs["finger_contact_feasible_ratio"] = float(
                np.mean(feasible)
            )
            output.attrs["finger_contact_all_feasible"] = bool(
                np.all(feasible)
            )
            output.attrs["finger_contact_max_q_step_rad"] = max_step
        else:
            output.attrs["finger_contact_feasible_ratio"] = float(
                np.mean(knot_valid)
            )
            output.attrs["finger_contact_all_feasible"] = bool(
                np.all(knot_valid)
            )
            output.attrs["finger_contact_max_q_step_rad"] = 0.0
            output.attrs["finger_contact_keyframes_only"] = True
    print(f"[SAVED] {args.output}")
    if (args.true_tip_waypoints and args.reference_bank is not None
            and args.reference_trajectory is None and not knot_valid.all()
            and args.extract_feasible_segment):
        from true_tip_plan_segment import write_feasible_segment
        segment = args.output.with_name(args.output.stem+'_segment_seed.h5')
        if write_feasible_segment(args.input, segment, knot_indices, knot_valid, knot_q, controller.base):
            retry=copy.copy(args)
            retry.input=segment;retry.reference_trajectory=segment;retry.reference_bank=None
            print('[TRUE-TIP-SEGMENT] replanning a continuous feasible interval over the full2500-frame horizon',flush=True)
            optimize(retry)
            segment.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument('--reference-trajectory', type=Path,
        help='Optional measured teacher q seed for each waypoint; palm path and acceptance remain unchanged')
    parser.add_argument('--reference-bank', type=Path,
        help='Measured healthy q/palm pose seeds for new true-tip waypoint optimization')
    parser.add_argument('--extract-feasible-segment', action=argparse.BooleanOptionalAction, default=True,
        help='For new bank-seeded paths, replan a continuous healthy interval as a full-length path if needed; record source interval')
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--object-id", default="ycb_mustard")
    parser.add_argument("--object-scale", type=float, default=1.0)
    parser.add_argument("--true-tip-waypoints", action="store_true",
        help="refine using split collision geometry, connector clearance and soft distal-top preference")
    parser.add_argument("--tip-top-weight", type=float, default=.05)
    parser.add_argument("--tip-top-preferred-fraction", type=float, default=.75)
    parser.add_argument('--non-tip-clearance-m', type=float, default=None,
        help='Opt-in positive clearance for connector/MCP/PIP/DIP/palm collision geoms; final acceptance is strict')
    parser.add_argument('--non-tip-preferred-clearance-m', type=float,
        help='Soft planning reserve above the hard non-tip margin, to accommodate execution deviation')
    parser.add_argument('--max-cross-finger-penetration-m', type=float, default=.0001,
        help='Cross-finger assembly penetration tolerance when full non-tip clearance is enabled')
    parser.add_argument('--clearance-polish', action=argparse.BooleanOptionalAction, default=True,
        help='Polish infeasible clearance candidates using constrained SLSQP; final geometry is always rechecked')
    parser.add_argument(
        "--keyframes-only",
        action="store_true",
        help=(
            "Screening mode: evaluate the feasibility criteria on the grasp "
            "keyframes only (the boundary conditions the inverse collector "
            "consumes) and skip the dense per-frame pass.  Fast enough to "
            "run thousands of plans; dense evaluation can be re-run on the "
            "few plans that pass screening."
        ),
    )
    parser.add_argument("--knot-stride", type=int, default=100)
    parser.add_argument('--extra-keyframe-frames', type=int, nargs='+',
        help='Additional control frames for adaptive interpolation repair')
    parser.add_argument('--temporal-seed-first', action='store_true',
        help='With a reference bank, use it for the first knot, then continue the previous solution')
    parser.add_argument("--projection-iterations", type=int, default=5)
    parser.add_argument(
        "--retry-multistart",
        action="store_true",
        help=(
            "When temporal continuation violates screening constraints, retry "
            "that keyframe with the structured posture bank and keep the less "
            "violating branch. Intended for shoulder/cap transitions."
        ),
    )
    parser.add_argument("--preload-m", type=float, default=0.003)
    parser.add_argument(
        "--ik-pad-weight",
        type=float,
        default=0.8,
        help=(
            "Weight on fingertip-pad-to-surface-normal alignment in the "
            "geometry contact IK (default 0.8, matching online recovery)."
        ),
    )
    parser.add_argument(
        "--ik-posture-gain",
        type=float,
        default=0.03,
        help=(
            "Direct Module-style pull toward the natural grasp; unlike a "
            "null-space-only term it remains active for a full-rank task."
        ),
    )
    parser.add_argument("--max-outside-distance-m", type=float, default=0.004)
    parser.add_argument("--max-inside-distance-m", type=float, default=0.012)
    parser.add_argument("--max-pad-normal-error-deg", type=float, default=85.0)
    parser.add_argument("--min-isotropic-reachability", type=float, default=1.0e-4)
    parser.add_argument("--min-manipulability", type=float, default=0.03)
    parser.add_argument("--min-lateral-margin-m", type=float, default=0.012)
    parser.add_argument("--max-posture-deviation-rad", type=float, default=1.20)
    args = parser.parse_args()
    if args.knot_stride <= 0 or args.projection_iterations <= 0:
        raise ValueError("stride and projection iterations must be positive")
    if args.non_tip_clearance_m is not None and not args.true_tip_waypoints:
        parser.error('non-tip clearance requires --true-tip-waypoints')
    if args.extra_keyframe_frames:
        with h5py.File(args.input) as handle:
            total = len(handle['palm_pose_object'])
        if min(args.extra_keyframe_frames) < 0 or max(args.extra_keyframe_frames) >= total:
            parser.error('extra keyframes must lie within the input trajectory')
    optimize(args)


if __name__ == "__main__":
    main()
