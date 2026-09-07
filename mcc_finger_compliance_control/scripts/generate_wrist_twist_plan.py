"""Generate an in-place palm wrist-yaw plan from a feasible grasp anchor.

The caller can keep either the palm body origin or the palm contact-control
point fixed in the object frame while the complete palm rotates about its local
surface normal.  A previously screened plan supplies the position,
surface-facing orientation, and healthy grasp seed; exact four-finger
feasibility is still checked later by optimize_contact_plan.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation


DEFAULT_CONTROL_POINT_LOCAL = np.asarray(
    (-0.0559703, -0.04142053, -0.0340008), dtype=np.float64
)


def _pose_to_rt(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pose = np.asarray(pose, dtype=np.float64)
    rotation = Rotation.from_quat(pose[3:7][[1, 2, 3, 0]]).as_matrix()
    return pose[:3], rotation


def _rt_to_pose(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    quat_xyzw = Rotation.from_matrix(rotation).as_quat()
    return np.concatenate((position, quat_xyzw[[3, 0, 1, 2]])).astype(np.float32)


def _smooth_progress(frames: int) -> np.ndarray:
    u = np.linspace(0.0, 1.0, frames, dtype=np.float64)
    # Quintic smoothstep: zero angular velocity and acceleration at both ends.
    return u**3 * (10.0 + u * (-15.0 + 6.0 * u))


def _cyclic_yaw(
    frames: int,
    amplitude_rad: float,
    cycles: int,
) -> np.ndarray:
    """Return stopped, smooth 0 -> +A -> -A -> 0 wrist cycles."""

    if cycles < 1:
        raise ValueError("cycles must be positive")
    waypoints = [0.0]
    for _ in range(cycles):
        waypoints.extend((amplitude_rad, -amplitude_rad, 0.0))
    waypoints = np.asarray(waypoints, dtype=np.float64)
    phase = np.linspace(0.0, len(waypoints) - 1, frames, dtype=np.float64)
    segment = np.minimum(phase.astype(np.int64), len(waypoints) - 2)
    local = phase - segment
    blend = local**3 * (10.0 + local * (-15.0 + 6.0 * local))
    return waypoints[segment] + blend * (
        waypoints[segment + 1] - waypoints[segment]
    )


def generate(
    source: Path,
    output: Path,
    *,
    anchor_fraction: float,
    angle_deg: float,
    direction: int,
    frames: int,
    control_dt: float,
    pivot: str,
    cycles: int,
) -> None:
    with h5py.File(source, "r") as handle:
        palm = np.asarray(handle["palm_pose_object"], dtype=np.float64)
        if palm.ndim == 3:
            if palm.shape[1] != 1:
                raise ValueError("source must contain one plan environment")
            palm = palm[:, 0]
        attrs = dict(handle.attrs)
        control_local = np.asarray(
            attrs.get("planner_palm_control_point_local", DEFAULT_CONTROL_POINT_LOCAL),
            dtype=np.float64,
        )
        source_q = np.asarray(handle["q_hand"], dtype=np.float64)
        if source_q.ndim == 3:
            source_q = source_q[:, 0]
        clearance_min = (
            np.asarray(handle["planner_palm_outline_min_clearance_object"])
            if "planner_palm_outline_min_clearance_object" in handle
            else None
        )
        clearance_mean = (
            np.asarray(handle["planner_palm_outline_mean_clearance_object"])
            if "planner_palm_outline_mean_clearance_object" in handle
            else None
        )
        keyframe_index = (
            np.asarray(handle["grasp_keyframe_frame_index"], dtype=np.int64)
            if "grasp_keyframe_frame_index" in handle
            else None
        )
        keyframe_q = (
            np.asarray(handle["grasp_keyframe_q"], dtype=np.float64)
            if "grasp_keyframe_q" in handle
            else None
        )

    if frames < 2:
        raise ValueError("frames must be at least two")
    if not 0.0 <= anchor_fraction <= 1.0:
        raise ValueError("anchor-fraction must lie in [0, 1]")
    if direction not in (-1, 1):
        raise ValueError("direction must be -1 or +1")
    anchor = int(round(anchor_fraction * (len(palm) - 1)))
    position0, rotation0 = _pose_to_rt(palm[anchor])
    control_object = position0 + rotation0 @ control_local
    # The planner convention is palm -Z toward the surface.  Either +Z or -Z
    # defines the same yaw axis; +Z gives a right-handed positive wrist yaw.
    yaw_axis_object = rotation0[:, 2]
    amplitude_rad = direction * np.deg2rad(float(angle_deg))
    if cycles > 0:
        yaw = _cyclic_yaw(frames, amplitude_rad, cycles)
    else:
        yaw = amplitude_rad * _smooth_progress(frames)

    poses = np.zeros((frames, 7), dtype=np.float32)
    rotations = np.zeros((frames, 3, 3), dtype=np.float64)
    positions = np.zeros((frames, 3), dtype=np.float64)
    for frame, angle in enumerate(yaw):
        rotation = Rotation.from_rotvec(yaw_axis_object * angle).as_matrix() @ rotation0
        if pivot == "control_point":
            position = control_object - rotation @ control_local
        elif pivot == "palm_origin":
            position = position0
        else:
            raise ValueError(f"unsupported pivot: {pivot}")
        rotations[frame] = rotation
        positions[frame] = position
        poses[frame] = _rt_to_pose(position, rotation)

    twist = np.zeros((frames, 6), dtype=np.float32)
    twist[1:, :3] = np.diff(positions, axis=0) / control_dt
    relative = rotations[1:] @ np.swapaxes(rotations[:-1], -1, -2)
    twist[1:, 3:] = Rotation.from_matrix(relative).as_rotvec() / control_dt
    twist[0] = twist[1]

    if keyframe_index is not None and keyframe_q is not None:
        q_anchor = keyframe_q[int(np.argmin(np.abs(keyframe_index - anchor)))]
    else:
        q_anchor = source_q[anchor]
    q_hand = np.broadcast_to(q_anchor, (frames, 16)).astype(np.float32).copy()
    qvel = np.zeros_like(q_hand)

    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as target:
        target.create_dataset("palm_pose_object", data=poses[:, None])
        target.create_dataset("palm_twist_object", data=twist[:, None])
        target.create_dataset("q_hand", data=q_hand[:, None])
        target.create_dataset("qvel", data=qvel[:, None])
        fixed_object = np.zeros((frames, 1, 7), dtype=np.float32)
        fixed_object[..., 3] = 1.0
        target.create_dataset("object_pose_world", data=fixed_object)
        target.create_dataset("episode_id", data=np.zeros((frames, 1), dtype=np.int64))
        target.create_dataset("record_step", data=np.arange(frames)[:, None])
        if clearance_min is not None:
            value = float(np.ravel(clearance_min)[anchor])
            target.create_dataset(
                "planner_palm_outline_min_clearance_object",
                data=np.full((frames, 1), value, dtype=np.float32),
            )
        if clearance_mean is not None:
            value = float(np.ravel(clearance_mean)[anchor])
            target.create_dataset(
                "planner_palm_outline_mean_clearance_object",
                data=np.full((frames, 1), value, dtype=np.float32),
            )
        for key, value in attrs.items():
            target.attrs[key] = value
        target.attrs["planner_path_mode"] = "wrist_twist"
        target.attrs["planner_wrist_twist_angle_deg"] = float(direction * angle_deg)
        target.attrs["planner_wrist_twist_amplitude_deg"] = float(angle_deg)
        target.attrs["planner_wrist_twist_cycles"] = int(cycles)
        target.attrs["planner_wrist_twist_anchor_fraction"] = float(anchor_fraction)
        target.attrs["planner_wrist_twist_anchor_frame"] = int(anchor)
        target.attrs["planner_wrist_twist_axis_object"] = yaw_axis_object
        target.attrs["planner_wrist_twist_pivot"] = pivot
        target.attrs["planner_wrist_twist_control_point_object"] = control_object
        target.attrs["planner_realized_translation_path_m"] = float(
            np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()
        )
        realized_rotation_deg = float(
            np.degrees(
                np.linalg.norm(Rotation.from_matrix(relative).as_rotvec(), axis=1)
            ).sum()
        )
        target.attrs["planner_realized_rotation_path_deg"] = realized_rotation_deg
        target.attrs["planner_realized_max_translation_step_m"] = float(
            np.linalg.norm(np.diff(positions, axis=0), axis=1).max()
        )
        target.attrs["planner_realized_max_rotation_step_deg"] = float(
            np.degrees(
                np.linalg.norm(Rotation.from_matrix(relative).as_rotvec(), axis=1)
            ).max()
        )
        target.attrs["planner_wrist_twist_source"] = str(source)
        target.attrs["control_dt"] = float(control_dt)
    print(
        f"[SUCCESS] wrist twist amplitude={angle_deg:.1f} deg "
        f"cycles={cycles}, total={realized_rotation_deg:.1f} deg, "
        f"fixed {pivot} -> {output}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--anchor-fraction", type=float, default=0.5)
    parser.add_argument("--angle-deg", type=float, default=30.0)
    parser.add_argument("--direction", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--frames", type=int, default=2500)
    parser.add_argument("--control-dt", type=float, default=0.01)
    parser.add_argument(
        "--cycles",
        type=int,
        default=0,
        help="Number of smooth 0 -> +A -> -A -> 0 cycles; zero is one-way.",
    )
    parser.add_argument(
        "--pivot",
        choices=("palm_origin", "control_point"),
        default="palm_origin",
        help="Point kept fixed in the object frame while the wrist rotates.",
    )
    args = parser.parse_args()
    generate(
        args.source,
        args.output,
        anchor_fraction=args.anchor_fraction,
        angle_deg=args.angle_deg,
        direction=args.direction,
        frames=args.frames,
        control_dt=args.control_dt,
        pivot=args.pivot,
        cycles=args.cycles,
    )


if __name__ == "__main__":
    main()
