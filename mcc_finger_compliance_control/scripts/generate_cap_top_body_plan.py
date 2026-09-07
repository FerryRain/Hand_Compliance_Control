"""Generate a cap-top pinch -> bottle-body palm plan for YCB-style bottles.

The generic longitudinal ellipse follows the nearest mesh normal.  Near a
small bottle cap the nearest triangle can lie on the rim, so its terminal palm
pose is oblique even when the control point is above the cap.  This targeted
offline teacher-data utility replaces that terminal segment with an explicit
cap-top pose:

* palm +Z is aligned with the object's PCA long axis (palm -Z faces the cap);
* the palm control point is centred above the cap-top patch;
* position and orientation corrections use quintic endpoint blending;
* the completed body -> cap path is reversed to obtain cap -> body motion.

Run ``optimize_contact_plan.py`` on the output afterwards.  Keeping palm-path
generation separate from fingertip IK makes the endpoint geometry auditable
and lets the contact solver use a structured multistart at the cap endpoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R

from generate_manifold_palm_plan import (
    PALM_CENTER_LOCAL,
    palm_outline_clearance_stats,
    palm_plan_motion_metrics,
    pose_to_rt,
    rt_to_pose,
)
from object_catalog import MeshNormalOracle, load_object_config


def _quintic(progress: np.ndarray) -> np.ndarray:
    progress = np.asarray(progress, dtype=np.float64)
    return progress**3 * (10.0 - 15.0 * progress + 6.0 * progress**2)


def _positive_pca_long_axis(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.mean(vertices, axis=0)
    _, _, vectors = np.linalg.svd(vertices - center, full_matrices=False)
    long_axis = vectors[0]
    long_axis /= max(float(np.linalg.norm(long_axis)), 1.0e-12)
    dominant = int(np.argmax(np.abs(long_axis)))
    if long_axis[dominant] < 0.0:
        long_axis *= -1.0
    return center, long_axis


def _cap_top_frame(
    vertices: np.ndarray,
    *,
    band_m: float,
    yaw_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center, long_axis = _positive_pca_long_axis(vertices)
    longitudinal = (vertices - center) @ long_axis
    cap_mask = longitudinal >= float(longitudinal.max()) - float(band_m)
    if int(cap_mask.sum()) < 16:
        raise ValueError(
            f"cap-top band contains only {int(cap_mask.sum())} vertices; "
            "increase --cap-top-band-m"
        )
    cap_center = np.mean(vertices[cap_mask], axis=0)

    x_axis = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
    x_axis -= long_axis * float(x_axis @ long_axis)
    if float(np.linalg.norm(x_axis)) < 1.0e-8:
        x_axis = np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
        x_axis -= long_axis * float(x_axis @ long_axis)
    x_axis /= max(float(np.linalg.norm(x_axis)), 1.0e-12)
    x_axis = R.from_rotvec(
        long_axis * np.deg2rad(float(yaw_deg))
    ).apply(x_axis)
    y_axis = np.cross(long_axis, x_axis)
    y_axis /= max(float(np.linalg.norm(y_axis)), 1.0e-12)
    x_axis = np.cross(y_axis, long_axis)
    rotation = np.column_stack((x_axis, y_axis, long_axis))
    return cap_center, long_axis, rotation


def _twist_from_pose(pose: np.ndarray, control_dt: float) -> np.ndarray:
    position, rotation = pose_to_rt(pose)
    twist = np.zeros((len(pose), 6), dtype=np.float32)
    if len(pose) > 1:
        twist[1:, :3] = ((position[1:] - position[:-1]) / control_dt).astype(
            np.float32
        )
        twist[1:, 3:] = (
            R.from_matrix(rotation[1:] @ np.swapaxes(rotation[:-1], -1, -2))
            .as_rotvec()
            / control_dt
        ).astype(np.float32)
        twist[0] = twist[1]
    return twist


def generate(args: argparse.Namespace) -> None:
    with h5py.File(args.input, "r") as source:
        if "palm_pose_object" not in source:
            raise KeyError(f"{args.input} has no palm_pose_object")
        source_pose = np.asarray(source["palm_pose_object"], dtype=np.float64)
        if source_pose.ndim == 3:
            if source_pose.shape[1] != 1:
                raise ValueError("generate one palm trajectory at a time")
            source_pose = source_pose[:, 0]
        source_attrs = dict(source.attrs)
        q_hand = np.asarray(source["q_hand"])
        fingertip_pose = np.asarray(source["fingertip_pose_object"])

    frames = len(source_pose)
    if not 8 <= args.blend_frames <= frames:
        raise ValueError("--blend-frames must lie in [8, trajectory length]")

    config = load_object_config(args.object_id)
    oracle = MeshNormalOracle.from_config(config, scale=args.object_scale)
    if oracle is None:
        raise ValueError("cap-top planning requires a source mesh")
    cap_center, long_axis, cap_rotation = _cap_top_frame(
        oracle.vertices,
        band_m=args.cap_top_band_m,
        yaw_deg=args.cap_top_yaw_deg,
    )
    cap_control = cap_center + float(args.cap_top_standoff_m) * long_axis

    body_position, body_rotation = pose_to_rt(source_pose)
    control_position = body_position + np.einsum(
        "nij,j->ni", body_rotation, PALM_CENTER_LOCAL
    )
    blend_start = frames - args.blend_frames
    blend = _quintic(np.linspace(0.0, 1.0, args.blend_frames))

    # Apply a smooth endpoint displacement to the existing collision-safe
    # route.  Both displacement and its first two derivatives vanish at the
    # splice, while the final point lands exactly over the cap centre.
    endpoint_delta = cap_control - control_position[-1]
    body_to_cap_control = control_position.copy()
    body_to_cap_control[blend_start:] += blend[:, None] * endpoint_delta

    # Correct the old endpoint frame to the explicit cap-top frame without
    # choosing a new wrist branch at every intermediate mesh triangle.
    correction = cap_rotation @ body_rotation[-1].T
    correction_rotvec = R.from_matrix(correction).as_rotvec()
    body_to_cap_rotation = body_rotation.copy()
    for local_index, weight in enumerate(blend):
        frame = blend_start + local_index
        body_to_cap_rotation[frame] = (
            R.from_rotvec(weight * correction_rotvec).as_matrix()
            @ body_rotation[frame]
        )
    # A lateral endpoint correction can make the interpolated palm outline
    # pass a few millimetres closer to the shoulder even though both endpoint
    # poses are safe.  Repair only that intermediate clearance with a smooth
    # outward scalar field.  The cap endpoint itself is held fixed: raising it
    # to solve an intermediate clearance violation makes the index/ring tips
    # unable to reach the small cap.
    clearance_repair = np.zeros(frames, dtype=np.float64)
    for _ in range(5):
        body_to_cap_position = body_to_cap_control - np.einsum(
            "nij,j->ni", body_to_cap_rotation, PALM_CENTER_LOCAL
        )
        current_min = np.asarray(
            [
                palm_outline_clearance_stats(position, rotation, oracle)[1]
                for position, rotation in zip(
                    body_to_cap_position,
                    body_to_cap_rotation,
                    strict=True,
                )
            ],
            dtype=np.float64,
        )
        deficit = (
            float(args.min_outline_clearance_m) + 0.00025 - current_min
        )
        deficit = np.maximum(deficit, 0.0)
        if float(np.max(deficit)) <= 1.0e-6:
            break
        correction = gaussian_filter1d(deficit, sigma=8.0, mode="nearest")
        active = deficit > 0.0
        scale = float(
            np.max(
                deficit[active]
                / np.maximum(correction[active], 1.0e-12)
            )
        )
        correction *= 1.02 * max(1.0, scale)
        # Both original path endpoints already satisfy the clearance.  Keep
        # them exact so the first cap pinch and final body grasp do not move.
        correction[0] = 0.0
        correction[-1] = 0.0
        body_to_cap_control += (
            correction[:, None] * body_to_cap_rotation[:, :, 2]
        )
        clearance_repair += correction

    body_to_cap_position = body_to_cap_control - np.einsum(
        "nij,j->ni", body_to_cap_rotation, PALM_CENTER_LOCAL
    )
    body_to_cap_pose = np.stack(
        [
            rt_to_pose(position, rotation)
            for position, rotation in zip(
                body_to_cap_position, body_to_cap_rotation, strict=True
            )
        ],
        axis=0,
    )

    # The requested teacher direction begins in the top pinch and opens
    # toward the body.  Contact optimization is intentionally run after this
    # reversal so its structured multistart solves the difficult cap posture.
    cap_to_body_pose = body_to_cap_pose[::-1].copy()
    cap_position, cap_rotation_sequence = pose_to_rt(cap_to_body_pose)
    min_clearance = np.zeros(frames, dtype=np.float32)
    mean_clearance = np.zeros(frames, dtype=np.float32)
    for frame in range(frames):
        mean_value, min_value, _ = palm_outline_clearance_stats(
            cap_position[frame], cap_rotation_sequence[frame], oracle
        )
        mean_clearance[frame] = mean_value
        min_clearance[frame] = min_value
    minimum = float(np.min(min_clearance))
    if minimum < float(args.min_outline_clearance_m):
        bad_frame = int(np.argmin(min_clearance))
        raise ValueError(
            "cap-top blend violates palm-outline clearance: "
            f"frame={bad_frame}, min={minimum * 1000.0:.2f} mm < "
            f"required={args.min_outline_clearance_m * 1000.0:.2f} mm"
        )

    control_dt = float(source_attrs.get("control_dt", 0.01))
    twist = _twist_from_pose(cap_to_body_pose, control_dt)
    motion = palm_plan_motion_metrics(cap_to_body_pose)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output, "w") as output:
        output.create_dataset(
            "palm_pose_object", data=cap_to_body_pose[:, None].astype(np.float32)
        )
        output.create_dataset(
            "planner_palm_outline_min_clearance_object",
            data=min_clearance[:, None],
        )
        output.create_dataset(
            "planner_palm_outline_mean_clearance_object",
            data=mean_clearance[:, None],
        )
        output.create_dataset("palm_twist_object", data=twist[:, None])
        output.create_dataset("q_hand", data=q_hand[::-1])
        output.create_dataset("qvel", data=np.zeros_like(q_hand))
        output.create_dataset(
            "fingertip_pose_object", data=fingertip_pose[::-1]
        )
        fixed_object = np.zeros((frames, 1, 7), dtype=np.float32)
        fixed_object[..., 3] = 1.0
        output.create_dataset("object_pose_world", data=fixed_object)
        output.create_dataset(
            "episode_id", data=np.zeros((frames, 1), dtype=np.int64)
        )
        output.create_dataset("record_step", data=np.arange(frames)[:, None])
        for key, value in source_attrs.items():
            output.attrs[key] = value
        output.attrs["object_id"] = args.object_id
        output.attrs["object_scale"] = float(args.object_scale)
        output.attrs["direction"] = "cap_to_body"
        output.attrs["plan_name"] = args.output.stem
        output.attrs["cap_top_endpoint"] = True
        output.attrs["cap_top_center_object"] = cap_center
        output.attrs["cap_top_long_axis_object"] = long_axis
        output.attrs["cap_top_standoff_m"] = float(args.cap_top_standoff_m)
        output.attrs["cap_top_yaw_deg"] = float(args.cap_top_yaw_deg)
        output.attrs["cap_top_band_m"] = float(args.cap_top_band_m)
        output.attrs["cap_top_blend_frames"] = int(args.blend_frames)
        output.attrs["cap_top_clearance_repair_max_m"] = float(
            np.max(clearance_repair)
        )
        output.attrs["cap_top_source_plan"] = str(args.input)
        output.attrs["planner_palm_outline_clearance_m"] = float(
            args.min_outline_clearance_m
        )
        output.attrs["planner_palm_control_point_local"] = PALM_CENTER_LOCAL
        for key, value in motion.items():
            output.attrs[f"planner_realized_{key}"] = float(value)

    print(
        "[CAP-TOP-PLAN] "
        f"frames={frames} top_vertices_standoff={args.cap_top_standoff_m * 1000.0:.1f}mm "
        f"outline_min={minimum * 1000.0:.2f}mm "
        f"translation_path={motion['translation_path_m']:.3f}m "
        f"rotation_path={motion['rotation_path_deg']:.1f}deg"
    )
    print(f"[SAVED] {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--object-id", default="ycb_mustard")
    parser.add_argument("--object-scale", type=float, default=1.0)
    parser.add_argument("--cap-top-standoff-m", type=float, default=0.035)
    parser.add_argument("--cap-top-yaw-deg", type=float, default=90.0)
    parser.add_argument("--cap-top-band-m", type=float, default=0.004)
    parser.add_argument("--blend-frames", type=int, default=700)
    parser.add_argument("--min-outline-clearance-m", type=float, default=0.030)
    args = parser.parse_args()
    generate(args)


if __name__ == "__main__":
    main()
