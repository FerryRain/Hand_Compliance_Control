"""Export a compact palm-frame H5 for fingertip diffusion-policy training."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

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
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(input_path, "r") as source, h5py.File(output_path, "w") as target:
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
        elif state_schema in ("contact_geometry", "contact_geometry_planner"):
            required += (
                "fingertip_contact_pos_object",
                "fingertip_contact",
            )
        else:
            raise ValueError(f"Unsupported state_schema={state_schema!r}")
        missing = [name for name in required if name not in source]
        if missing:
            raise KeyError(f"Missing required inverted fields: {missing}")

        for name in ("episode_id", "episode_step", "q_hand"):
            dataset = source[name]
            target.create_dataset(name, data=np.asarray(dataset))

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
        if state_schema == "contact_geometry_planner":
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

        for key, value in source.attrs.items():
            target.attrs[key] = value
        target.attrs["schema_version"] = "mcc_tip_palm_dp_v2"
        target.attrs["dp_input_frame"] = "palm"
        target.attrs["dp_state_schema"] = state_schema
        target.attrs["source_file"] = str(input_path)
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
        else:
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
        target.attrs["palm_frame_transform"] = (
            "per-frame R_object_from_palm transpose; force/normal/twist vectors"
        )
    print(f"[SUCCESS] palm-frame DP data saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--block-size", type=int, default=4096)
    parser.add_argument(
        "--state-schema",
        choices=("force_normal", "contact_geometry", "contact_geometry_planner"),
        default="force_normal",
    )
    parser.add_argument("--planner-waypoints", type=int, default=DEFAULT_PLANNER_WAYPOINTS)
    parser.add_argument(
        "--planner-step-frames", type=int, default=DEFAULT_PLANNER_STEP_FRAMES
    )
    args = parser.parse_args()
    if args.block_size <= 0:
        raise ValueError("--block-size must be positive")
    output = args.output or args.file.with_name(f"{args.file.stem}_palm_dp.h5")
    export(
        args.file,
        output,
        args.block_size,
        args.state_schema,
        args.planner_waypoints,
        args.planner_step_frames,
    )


if __name__ == "__main__":
    main()
