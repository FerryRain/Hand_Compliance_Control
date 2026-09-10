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
    return {
        "surface_point": np.asarray(observation.points_world, dtype=np.float32),
        "surface_normal": normals_object.astype(np.float32),
        "signed_distance": np.asarray(observation.signed_distance, dtype=np.float32),
        "pad_normal_error": pad_error.astype(np.float32),
        "synergy_spread": synergy_spread.astype(np.float32),
        "synergy_residual": synergy_residual.astype(np.float32),
        "isotropic_reachability": isotropic_reachability.astype(np.float32),
        "manipulability": manipulability.astype(np.float32),
    }


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
    nominal = controller.clamp_joint_positions(nominal).astype(np.float64)
    knot_indices = np.unique(
        np.concatenate(
            (
                np.arange(0, len(palm_pose), max(1, args.knot_stride)),
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
    # The first contact is normally solved directly from exact geometry with
    # a structured posture bank (no open->grasp path or path/surface
    # crossing participates in contact selection). When --recovery-state is
    # given, we already have a live, physically-realized hand posture at
    # state A (extract_recovery_state.py's q_live) -- warm-start knot 0 from
    # that instead of the blind multistart bank, matching the Guide's
    # requirement that the expert reference start from q_live(A), not a
    # freshly-chosen key posture.
    recovery_warm_start = getattr(args, "recovery_state", None) is not None
    if recovery_warm_start:
        with h5py.File(args.recovery_state, "r") as state:
            if str(state.attrs.get("format", "")) != "mcc_recovery_state_v1":
                raise ValueError(
                    f"{args.recovery_state}: expected attrs['format']="
                    "'mcc_recovery_state_v1' (extract_recovery_state.py "
                    f"output), got {state.attrs.get('format')!r}"
                )
            q_live = np.asarray(state["q_live"], dtype=np.float64)
        q_previous = controller.clamp_joint_positions(q_live).astype(np.float64)
        print(
            f"[INFO] --recovery-state given: warm-starting knot 0 from "
            f"q_live(A) in {args.recovery_state} instead of the structured "
            "posture multistart bank."
        )
    else:
        q_previous = nominal.copy()
    for knot_id, frame in enumerate(knot_indices):
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
            multistart=(knot_id == 0 and not recovery_warm_start),
        )
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
    for frame, (pose, q) in enumerate(zip(palm_pose, q_plan, strict=True)):
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
        & (isotropic_reachability >= args.min_isotropic_reachability).all(axis=1)
        & (manipulability >= args.min_manipulability).all(axis=1)
    )
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
        for key, value in source_attrs.items():
            output.attrs[key] = value
        output.attrs["finger_contact_plan"] = True
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--object-id", default="ycb_mustard")
    parser.add_argument("--object-scale", type=float, default=1.0)
    parser.add_argument(
        "--recovery-state",
        type=Path,
        default=None,
        help=(
            "extract_recovery_state.py output (state A). When given, knot 0 "
            "is warm-started from q_live(A) instead of the structured "
            "posture multistart bank -- pair with a --input plan produced "
            "by generate_manifold_palm_plan.py --recovery-state."
        ),
    )
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
    parser.add_argument("--projection-iterations", type=int, default=5)
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
    optimize(args)


if __name__ == "__main__":
    main()
