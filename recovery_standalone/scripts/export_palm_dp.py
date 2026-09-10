"""Export a compact palm-frame H5 for fingertip diffusion-policy training."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from dp_motion_features import (
    DUAL_TRACK_SCHEMA,
    DUAL_TRACK_V3_SCHEMA,
    MOTION_SCHEMA,
    MOTION_SCHEMAS,
    causal_motion_features,
)
from palm_planner_features import (
    DEFAULT_PLANNER_STEP_FRAMES,
    DEFAULT_PLANNER_WAYPOINTS,
    future_palm_delta_pose_palm,
)


def _wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Return rotation matrices that map palm vectors into the object frame."""
    quaternion = np.asarray(quaternion, dtype=np.float64)
    quaternion /= np.maximum(
        np.linalg.norm(quaternion, axis=-1, keepdims=True), 1.0e-12
    )
    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    matrix = np.empty((*quaternion.shape[:-1], 3, 3), dtype=np.float64)
    matrix[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    matrix[..., 0, 1] = 2.0 * (x * y - z * w)
    matrix[..., 0, 2] = 2.0 * (x * z + y * w)
    matrix[..., 1, 0] = 2.0 * (x * y + z * w)
    matrix[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    matrix[..., 1, 2] = 2.0 * (y * z - x * w)
    matrix[..., 2, 0] = 2.0 * (x * z - y * w)
    matrix[..., 2, 1] = 2.0 * (y * z + x * w)
    matrix[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return matrix


def _create_like(
    target: h5py.File,
    name: str,
    shape: tuple[int, ...],
    dtype: np.dtype | str,
) -> h5py.Dataset:
    chunk_time = min(4096, shape[0])
    return target.create_dataset(
        name,
        shape=shape,
        dtype=dtype,
        chunks=(chunk_time, *shape[1:]),
    )


def export(
    input_path: Path,
    output_path: Path,
    block_size: int = 4096,
    state_schema: str = "force_normal",
    planner_waypoints: int = DEFAULT_PLANNER_WAYPOINTS,
    planner_step_frames: int = DEFAULT_PLANNER_STEP_FRAMES,
    motion_feature_step_frames: int = 5,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(input_path, "r") as source, h5py.File(output_path, "w") as target:
        if str(source.attrs.get("contact_mask_source", "")) != (
            "same_frame_collision_force_and_tip_surface_geometry"
        ):
            raise ValueError(
                "The inverted H5 predates same-frame contact correction. "
                "Re-run invert_trajectories.py on the raw trajectory before "
                "exporting DP data."
            )
        required = (
            "episode_id",
            "episode_step",
            "q_hand",
            "palm_pose_object",
            "palm_twist_object",
            "fingertip_contact_normal_object",
        )
        if state_schema == "force_normal":
            required += ("fingertip_force_object",)
        elif state_schema in (
            "contact_geometry",
            "contact_geometry_planner",
            MOTION_SCHEMA,
            DUAL_TRACK_SCHEMA,
            DUAL_TRACK_V3_SCHEMA,
        ):
            required += (
                "fingertip_contact_pos_object",
                "fingertip_contact",
            )
            if state_schema == DUAL_TRACK_SCHEMA:
                required += (
                    "q_prior",
                    "q_cmd",
                    "delta_q_comp",
                    "e_servo",
                )
        else:
            raise ValueError(f"Unsupported state_schema={state_schema!r}")
        missing = [name for name in required if name not in source]
        if missing:
            raise KeyError(f"Missing required inverted fields: {missing}")

        for name in ("episode_id", "episode_step", "q_hand"):
            dataset = source[name]
            target.create_dataset(name, data=np.asarray(dataset))
        if state_schema == DUAL_TRACK_SCHEMA:
            for name in ("q_prior", "q_cmd", "delta_q_comp", "e_servo"):
                target.create_dataset(name, data=np.asarray(source[name]))

        base_shape = source["q_hand"].shape[:2]
        force_out = None
        position_out = None
        mask_out = None
        if state_schema == "force_normal":
            force_out = _create_like(
                target, "fingertip_force_palm", (*base_shape, 4, 3), "f4"
            )
        else:
            position_out = _create_like(
                target,
                "fingertip_contact_pos_palm",
                (*base_shape, 4, 3),
                "f4",
            )
            mask_out = _create_like(
                target,
                "fingertip_contact_mask",
                (*base_shape, 4),
                "f4",
            )
        normal_out = _create_like(
            target,
            "fingertip_contact_normal_palm",
            (*base_shape, 4, 3),
            "f4",
        )
        twist_out = _create_like(
            target, "palm_relative_twist_palm", (*base_shape, 6), "f4"
        )
        if state_schema in (
            "contact_geometry_planner",
            MOTION_SCHEMA,
            DUAL_TRACK_SCHEMA,
            DUAL_TRACK_V3_SCHEMA,
        ):
            planner_out = _create_like(
                target,
                "planner_palm_delta_pose_palm",
                (*base_shape, planner_waypoints, 6),
                "f4",
            )
            planner_out[:] = future_palm_delta_pose_palm(
                np.asarray(source["palm_pose_object"], dtype=np.float64),
                np.asarray(source["episode_id"], dtype=np.int64),
                waypoint_count=planner_waypoints,
                step_frames=planner_step_frames,
            )

        total = base_shape[0]
        for start in range(0, total, block_size):
            stop = min(start + block_size, total)
            selection = np.s_[start:stop]
            palm_pose = np.asarray(
                source["palm_pose_object"][selection], dtype=np.float64
            )
            object_from_palm = _wxyz_to_matrix(palm_pose[..., 3:7])
            palm_from_object = np.swapaxes(object_from_palm, -1, -2)

            normal_object = np.asarray(
                source["fingertip_contact_normal_object"][selection],
                dtype=np.float64,
            )
            twist_object = np.asarray(
                source["palm_twist_object"][selection], dtype=np.float64
            )
            if force_out is not None:
                force_object = np.asarray(
                    source["fingertip_force_object"][selection],
                    dtype=np.float64,
                )
                force_out[selection] = np.einsum(
                    "...ij,...fj->...fi", palm_from_object, force_object
                ).astype(np.float32)
            else:
                contact_object = np.asarray(
                    source["fingertip_contact_pos_object"][selection],
                    dtype=np.float64,
                )
                contact_mask = np.asarray(
                    source["fingertip_contact"][selection],
                    dtype=np.float32,
                )
                contact_palm = np.einsum(
                    "...ij,...fj->...fi",
                    palm_from_object,
                    contact_object - palm_pose[..., None, :3],
                )
                position_out[selection] = np.where(
                    contact_mask[..., None] > 0.5,
                    contact_palm,
                    0.0,
                ).astype(np.float32)
                mask_out[selection] = contact_mask
            normal_out[selection] = np.einsum(
                "...ij,...fj->...fi", palm_from_object, normal_object
            ).astype(np.float32)
            twist_out[start:stop, ..., :3] = np.einsum(
                "...ij,...j->...i", palm_from_object, twist_object[..., :3]
            ).astype(np.float32)
            twist_out[start:stop, ..., 3:] = np.einsum(
                "...ij,...j->...i", palm_from_object, twist_object[..., 3:]
            ).astype(np.float32)

        if state_schema in MOTION_SCHEMAS:
            control_dt = float(source.attrs.get("control_dt", 0.01))
            q_velocity, point_velocity, normal_rate = causal_motion_features(
                np.asarray(
                    source[
                        "q_prior" if state_schema == DUAL_TRACK_SCHEMA else "q_hand"
                    ],
                    dtype=np.float32,
                ),
                np.asarray(position_out, dtype=np.float32),
                np.asarray(normal_out, dtype=np.float32),
                np.asarray(mask_out, dtype=np.float32),
                np.asarray(source["episode_id"], dtype=np.int64),
                control_dt=control_dt,
                step_frames=motion_feature_step_frames,
            )
            velocity_name = (
                "q_prior_velocity"
                if state_schema == DUAL_TRACK_SCHEMA
                else "q_velocity"
            )
            target.create_dataset(velocity_name, data=q_velocity)
            target.create_dataset(
                "fingertip_contact_point_velocity_palm", data=point_velocity
            )
            target.create_dataset(
                "fingertip_contact_normal_angular_rate_palm", data=normal_rate
            )

        for key, value in source.attrs.items():
            target.attrs[key] = value
        target.attrs["schema_version"] = (
            "mcc_tip_palm_dp_dual_track_v1"
            if state_schema == DUAL_TRACK_SCHEMA
            else "mcc_tip_palm_dp_v3"
        )
        target.attrs["dp_input_frame"] = "palm"
        target.attrs["palm_frame_body"] = "palm_lower"
        target.attrs["dp_state_schema"] = state_schema
        target.attrs["source_file"] = str(input_path)
        target.attrs["action_field"] = (
            "q_prior" if state_schema == DUAL_TRACK_SCHEMA else "q_hand"
        )
        target.attrs["action_dim"] = 16
        target.attrs["action_representation"] = "absolute_q"
        target.attrs["action_coordinate_space"] = "joint_position_rad"
        if state_schema == "force_normal":
            target.attrs["state_fields"] = (
                "q_hand,fingertip_force_palm,"
                "fingertip_contact_normal_palm,palm_relative_twist_palm"
            )
        elif state_schema == "contact_geometry":
            target.attrs["state_fields"] = (
                "q_hand,fingertip_contact_pos_palm,"
                "fingertip_contact_normal_palm,fingertip_contact_mask,"
                "palm_relative_twist_palm"
            )
        elif state_schema == "contact_geometry_planner":
            target.attrs["state_fields"] = (
                "q_hand,fingertip_contact_pos_palm,"
                "fingertip_contact_normal_palm,fingertip_contact_mask,"
                "palm_relative_twist_palm,planner_palm_delta_pose_palm"
            )
            target.attrs["planner_waypoints"] = planner_waypoints
            target.attrs["planner_step_frames"] = planner_step_frames
            target.attrs["planner_horizon_frames"] = (
                planner_waypoints * planner_step_frames
            )
            target.attrs["planner_feature"] = (
                "future palm delta position and rotation vector in the "
                "current palm frame; waypoint shape [K,6]"
            )
            control_dt = float(source.attrs.get("control_dt", 0.01))
            target.attrs["planner_waypoint_dt"] = planner_step_frames * control_dt
            target.attrs["planner_horizon_seconds"] = (
                planner_waypoints * planner_step_frames * control_dt
            )
        elif state_schema == MOTION_SCHEMA:
            target.attrs["state_fields"] = (
                "q_hand,fingertip_contact_pos_palm,"
                "fingertip_contact_normal_palm,fingertip_contact_mask,"
                "q_velocity,fingertip_contact_point_velocity_palm,"
                "fingertip_contact_normal_angular_rate_palm,"
                "palm_relative_twist_palm,planner_palm_delta_pose_palm"
            )
            target.attrs["planner_waypoints"] = planner_waypoints
            target.attrs["planner_step_frames"] = planner_step_frames
            target.attrs["planner_horizon_frames"] = (
                planner_waypoints * planner_step_frames
            )
            target.attrs["planner_feature"] = (
                "future palm delta position and rotation vector in the "
                "current palm frame; waypoint shape [K,6]"
            )
            control_dt = float(source.attrs.get("control_dt", 0.01))
            target.attrs["planner_waypoint_dt"] = planner_step_frames * control_dt
            target.attrs["planner_horizon_seconds"] = (
                planner_waypoints * planner_step_frames * control_dt
            )
            target.attrs["motion_feature_step_frames"] = motion_feature_step_frames
            target.attrs["motion_feature_dt"] = (
                motion_feature_step_frames * control_dt
            )
            target.attrs["motion_feature_convention"] = (
                "strictly causal backward difference in per-frame palm coordinates; "
                "contact rates are zero unless both endpoints have valid contact"
            )
        else:
            target.attrs["state_fields"] = (
                "q_prior,fingertip_contact_pos_palm,"
                "fingertip_contact_normal_palm,fingertip_contact_mask,"
                "q_prior_velocity,fingertip_contact_point_velocity_palm,"
                "fingertip_contact_normal_angular_rate_palm,q_hand,"
                "delta_q_comp,e_servo,palm_relative_twist_palm,"
                "planner_palm_delta_pose_palm"
            )
            target.attrs["planner_waypoints"] = planner_waypoints
            target.attrs["planner_step_frames"] = planner_step_frames
            target.attrs["planner_horizon_frames"] = (
                planner_waypoints * planner_step_frames
            )
            control_dt = float(source.attrs.get("control_dt", 0.01))
            target.attrs["planner_waypoint_dt"] = planner_step_frames * control_dt
            target.attrs["planner_horizon_seconds"] = (
                planner_waypoints * planner_step_frames * control_dt
            )
            target.attrs["motion_feature_step_frames"] = motion_feature_step_frames
            target.attrs["motion_feature_dt"] = (
                motion_feature_step_frames * control_dt
            )
            target.attrs["dual_track_contract"] = (
                "q_prior/q_cmd pre-step; q_hand/e_servo/tactile post-step; "
                "delta_q_comp=q_cmd-q_prior; action=future q_prior"
            )
        target.attrs["palm_frame_transform"] = (
            "per-frame T_palm_from_object: points use R^T(p-p_palm); "
            "force/normal/linear_velocity/angular_velocity use R^T v"
        )
        target.attrs["dp_coordinate_contract"] = (
            "q_hand=joint rad; fingertip_contact_pos_palm=point in palm_lower; "
            "fingertip_contact_normal_palm and fingertip_force_palm=vectors in "
            "palm_lower; palm_relative_twist_palm=[linear,angular] in palm_lower; "
            "planner_palm_delta_pose_palm=[translation,rotvec] in current palm_lower; "
            "explicit motion features are causal rates"
        )
    validate_palm_dp_file(output_path)
    print(f"[SUCCESS] palm-frame DP data saved to {output_path}")


def validate_palm_dp_file(path: Path) -> None:
    """Reject mixed-frame or numerically invalid DP exports immediately."""

    with h5py.File(path, "r") as file:
        if str(file.attrs.get("dp_input_frame", "")) != "palm":
            raise ValueError(f"{path}: dp_input_frame must be 'palm'")
        if str(file.attrs.get("palm_frame_body", "")) != "palm_lower":
            raise ValueError(f"{path}: palm_frame_body must be 'palm_lower'")
        state_fields = str(file.attrs.get("state_fields", ""))
        if "_world" in state_fields or "_object" in state_fields:
            raise ValueError(f"{path}: mixed-frame state fields: {state_fields}")
        required = ["q_hand", "fingertip_contact_normal_palm", "palm_relative_twist_palm"]
        schema = str(file.attrs.get("dp_state_schema", ""))
        if schema == "force_normal":
            required.append("fingertip_force_palm")
        elif schema in (
            "contact_geometry",
            "contact_geometry_planner",
            MOTION_SCHEMA,
            DUAL_TRACK_SCHEMA,
            DUAL_TRACK_V3_SCHEMA,
        ):
            required.extend(("fingertip_contact_pos_palm", "fingertip_contact_mask"))
        else:
            raise ValueError(f"{path}: unsupported dp_state_schema={schema!r}")
        if schema in (
            "contact_geometry_planner",
            MOTION_SCHEMA,
            DUAL_TRACK_SCHEMA,
            DUAL_TRACK_V3_SCHEMA,
        ):
            required.append("planner_palm_delta_pose_palm")
        if schema in MOTION_SCHEMAS:
            required.extend(
                (
                    (
                        "q_prior_velocity"
                        if schema in (DUAL_TRACK_SCHEMA, DUAL_TRACK_V3_SCHEMA)
                        else "q_velocity"
                    ),
                    "fingertip_contact_point_velocity_palm",
                    "fingertip_contact_normal_angular_rate_palm",
                )
            )
            if int(file.attrs.get("motion_feature_step_frames", 0)) <= 0:
                raise ValueError(f"{path}: invalid motion_feature_step_frames")
        if schema in (DUAL_TRACK_SCHEMA, DUAL_TRACK_V3_SCHEMA):
            required.extend(("q_prior", "q_cmd", "delta_q_comp", "e_servo"))
            if schema == DUAL_TRACK_V3_SCHEMA:
                required.extend(("q_live_velocity", "e_qdot"))
            q_prior = np.asarray(file["q_prior"], dtype=np.float64)
            q_cmd = np.asarray(file["q_cmd"], dtype=np.float64)
            q_live = np.asarray(file["q_hand"], dtype=np.float64)
            delta = np.asarray(file["delta_q_comp"], dtype=np.float64)
            servo = np.asarray(file["e_servo"], dtype=np.float64)
            if np.max(np.abs((q_cmd - q_prior) - delta)) > 1.0e-6:
                raise ValueError(f"{path}: inconsistent delta_q_comp")
            if np.max(np.abs((q_live - q_cmd) - servo)) > 1.0e-6:
                raise ValueError(f"{path}: inconsistent e_servo")
        for name in required:
            if name not in file:
                raise KeyError(f"{path}: missing required palm-frame field {name!r}")
            if not np.isfinite(np.asarray(file[name])).all():
                raise ValueError(f"{path}: non-finite values in {name}")
        if "fingertip_contact_pos_palm" in file:
            point = np.asarray(file["fingertip_contact_pos_palm"], dtype=np.float64)
            mask = np.asarray(file["fingertip_contact_mask"]) > 0.5
            if np.any(mask):
                maximum_radius = float(np.linalg.norm(point, axis=-1)[mask].max())
                if maximum_radius > 0.25:
                    raise ValueError(
                        f"{path}: palm-frame contact radius {maximum_radius:.3f} m "
                        "is physically implausible"
                    )
        normal = np.asarray(file["fingertip_contact_normal_palm"], dtype=np.float64)
        nonzero = np.linalg.norm(normal, axis=-1) > 1.0e-6
        if np.any(nonzero):
            error = float(np.max(np.abs(np.linalg.norm(normal, axis=-1)[nonzero] - 1.0)))
            if error > 5.0e-3:
                raise ValueError(f"{path}: non-unit palm-frame contact normal")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--block-size", type=int, default=4096)
    parser.add_argument(
        "--state-schema",
        choices=(
            "force_normal",
            "contact_geometry",
            "contact_geometry_planner",
            MOTION_SCHEMA,
            DUAL_TRACK_SCHEMA,
        ),
        default="contact_geometry_planner",
    )
    parser.add_argument("--planner-waypoints", type=int, default=DEFAULT_PLANNER_WAYPOINTS)
    parser.add_argument(
        "--planner-step-frames", type=int, default=DEFAULT_PLANNER_STEP_FRAMES
    )
    parser.add_argument("--motion-feature-step-frames", type=int, default=5)
    args = parser.parse_args()
    if args.block_size <= 0:
        raise ValueError("--block-size must be positive")
    if args.motion_feature_step_frames <= 0:
        raise ValueError("--motion-feature-step-frames must be positive")
    output = args.output or args.file.with_name(f"{args.file.stem}_palm_dp.h5")
    export(
        args.file,
        output,
        args.block_size,
        args.state_schema,
        args.planner_waypoints,
        args.planner_step_frames,
        args.motion_feature_step_frames,
    )


if __name__ == "__main__":
    main()
