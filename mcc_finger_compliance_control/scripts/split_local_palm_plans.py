"""Split broad Mustard palm paths into smooth local contact windows.

The historical ``regional`` plans still cover hundreds of millimetres and
large accumulated rotations.  They are useful surface sweeps, but they are
not individual teacher episodes: a single unreachable section invalidates an
otherwise useful path.  This tool cuts them by *accumulated* palm translation
and rotation, optionally adds the reverse direction, and time-resamples every
window without changing its geometric path.

The output is deliberately a palm-only plan.  Run ``optimize_contact_plan.py``
on each result afterwards; fingertip feasibility is a property of the local
window and must not be interpolated across a discarded section.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation as R, Slerp


def _as_pose_batch(value: np.ndarray) -> np.ndarray:
    pose = np.asarray(value, dtype=np.float64)
    if pose.ndim == 2 and pose.shape[1] == 7:
        return pose[:, None, :]
    if pose.ndim == 3 and pose.shape[2] == 7:
        return pose
    raise ValueError(
        f"palm_pose_object must have shape (T,7) or (T,E,7), got {pose.shape}"
    )


def _continuous_quaternions_wxyz(quaternions: np.ndarray) -> np.ndarray:
    result = np.asarray(quaternions, dtype=np.float64).copy()
    result /= np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1.0e-12)
    for index in range(1, len(result)):
        if float(result[index - 1] @ result[index]) < 0.0:
            result[index] *= -1.0
    return result


def _edge_motion(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    translation = np.linalg.norm(np.diff(pose[:, :3], axis=0), axis=1)
    quaternion = _continuous_quaternions_wxyz(pose[:, 3:7])
    rotation = R.from_quat(quaternion[:, (1, 2, 3, 0)])
    angle = (rotation[:-1].inv() * rotation[1:]).magnitude()
    return translation, angle


def _window_ranges(
    pose: np.ndarray,
    *,
    max_translation_m: float,
    max_rotation_rad: float,
    overlap_fraction: float,
    min_translation_m: float,
    min_rotation_rad: float,
) -> list[tuple[int, int]]:
    translation, rotation = _edge_motion(pose)
    frame_count = len(pose)
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < frame_count - 1:
        end = start + 1
        translation_sum = 0.0
        rotation_sum = 0.0
        while end < frame_count:
            next_translation = translation_sum + float(translation[end - 1])
            next_rotation = rotation_sum + float(rotation[end - 1])
            exceeds = (
                next_translation > max_translation_m
                or next_rotation > max_rotation_rad
            )
            if exceeds and end > start + 1:
                break
            translation_sum = next_translation
            rotation_sum = next_rotation
            end += 1
            if exceeds:
                break
        end = min(end, frame_count)
        if (
            translation_sum >= min_translation_m
            or rotation_sum >= min_rotation_rad
        ):
            ranges.append((start, end))
        width = max(2, end - start)
        advance = max(1, int(round(width * (1.0 - overlap_fraction))))
        next_start = start + advance
        if end == frame_count and next_start + 1 >= frame_count:
            break
        start = next_start

    # Guarantee coverage of the final path section without duplicating an
    # already identical terminal range.
    if ranges and ranges[-1][1] < frame_count:
        last_width = ranges[-1][1] - ranges[-1][0]
        terminal = (max(0, frame_count - last_width), frame_count)
        if terminal != ranges[-1]:
            ranges.append(terminal)
    return ranges


def _resample_pose(pose: np.ndarray, frame_count: int) -> np.ndarray:
    if len(pose) < 2:
        raise ValueError("a local window must contain at least two poses")
    source_time = np.linspace(0.0, 1.0, len(pose))
    target_time = np.linspace(0.0, 1.0, int(frame_count))
    position = np.stack(
        [
            PchipInterpolator(source_time, pose[:, axis])(target_time)
            for axis in range(3)
        ],
        axis=1,
    )
    quaternion = _continuous_quaternions_wxyz(pose[:, 3:7])
    rotation = R.from_quat(quaternion[:, (1, 2, 3, 0)])
    sampled_xyzw = Slerp(source_time, rotation)(target_time).as_quat()
    sampled_wxyz = sampled_xyzw[:, (3, 0, 1, 2)]
    sampled_wxyz = _continuous_quaternions_wxyz(sampled_wxyz)
    return np.concatenate((position, sampled_wxyz), axis=1)


def _twist_from_pose(pose: np.ndarray, dt: float) -> np.ndarray:
    twist = np.zeros((len(pose), 6), dtype=np.float64)
    twist[:, :3] = np.gradient(pose[:, :3], dt, axis=0)
    quaternion = _continuous_quaternions_wxyz(pose[:, 3:7])
    rotation = R.from_quat(quaternion[:, (1, 2, 3, 0)])
    edge_velocity = (rotation[:-1].inv() * rotation[1:]).as_rotvec() / dt
    if len(edge_velocity):
        twist[:-1, 3:] = edge_velocity
        twist[-1, 3:] = edge_velocity[-1]
    return twist


def _write_plan(
    output: Path,
    pose: np.ndarray,
    *,
    source: Path,
    source_env: int,
    source_start: int,
    source_end: int,
    direction: str,
    dt: float,
    source_attrs: dict[str, object],
) -> None:
    translation, rotation = _edge_motion(pose)
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as handle:
        handle.create_dataset("palm_pose_object", data=pose.astype(np.float32))
        handle.create_dataset(
            "palm_twist_object",
            data=_twist_from_pose(pose, dt).astype(np.float32),
        )
        handle.create_dataset(
            "record_step", data=np.arange(len(pose), dtype=np.int32)
        )
        handle.create_dataset(
            "episode_id", data=np.zeros(len(pose), dtype=np.int32)
        )
        for key, value in source_attrs.items():
            try:
                handle.attrs[key] = value
            except (TypeError, ValueError):
                continue
        handle.attrs["local_window"] = True
        handle.attrs["local_source_file"] = str(source)
        handle.attrs["local_source_env"] = int(source_env)
        handle.attrs["local_source_start"] = int(source_start)
        handle.attrs["local_source_end_exclusive"] = int(source_end)
        handle.attrs["local_direction"] = direction
        handle.attrs["local_translation_path_m"] = float(np.sum(translation))
        handle.attrs["local_rotation_path_rad"] = float(np.sum(rotation))
        handle.attrs["control_dt"] = float(dt)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    # With the current DP defaults (stride=5, obs=16, pred=32), one training
    # sample spans up to 235 raw control frames.  A 1200-frame episode provides
    # roughly 193 distinct stride-rate windows and enough geometric evolution
    # to learn more than a nearly static local linearization.
    parser.add_argument("--max-translation-mm", type=float, default=120.0)
    parser.add_argument("--max-rotation-deg", type=float, default=45.0)
    parser.add_argument("--min-translation-mm", type=float, default=50.0)
    parser.add_argument("--min-rotation-deg", type=float, default=15.0)
    parser.add_argument("--overlap", type=float, default=0.50)
    parser.add_argument("--frames", type=int, default=1200)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--include-reverse", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_translation_mm <= 0.0 or args.max_rotation_deg <= 0.0:
        raise ValueError("maximum local motion bounds must be positive")
    if not 0.0 <= args.overlap < 1.0:
        raise ValueError("--overlap must be in [0,1)")
    if args.frames < 2 or args.dt <= 0.0:
        raise ValueError("--frames must be >=2 and --dt must be positive")

    written = 0
    for source in args.inputs:
        with h5py.File(source, "r") as handle:
            if "palm_pose_object" not in handle:
                raise KeyError(f"{source} has no palm_pose_object")
            pose_batch = _as_pose_batch(handle["palm_pose_object"][:])
            source_attrs = dict(handle.attrs)
        for env_index in range(pose_batch.shape[1]):
            pose = pose_batch[:, env_index]
            ranges = _window_ranges(
                pose,
                max_translation_m=args.max_translation_mm * 1.0e-3,
                max_rotation_rad=np.deg2rad(args.max_rotation_deg),
                overlap_fraction=args.overlap,
                min_translation_m=args.min_translation_mm * 1.0e-3,
                min_rotation_rad=np.deg2rad(args.min_rotation_deg),
            )
            for local_index, (start, end) in enumerate(ranges):
                raw = pose[start:end]
                variants = [("forward", raw)]
                if args.include_reverse:
                    variants.append(("reverse", raw[::-1].copy()))
                for direction, variant in variants:
                    sampled = _resample_pose(variant, args.frames)
                    suffix = "fwd" if direction == "forward" else "rev"
                    output = args.output_dir / (
                        f"{source.stem}_e{env_index:02d}_w{local_index:03d}_{suffix}.h5"
                    )
                    _write_plan(
                        output,
                        sampled,
                        source=source,
                        source_env=env_index,
                        source_start=start,
                        source_end=end,
                        direction=direction,
                        dt=args.dt,
                        source_attrs=source_attrs,
                    )
                    written += 1
            print(
                f"[LOCAL-SPLIT] {source.name} env={env_index} "
                f"windows={len(ranges)} variants={len(ranges) * (2 if args.include_reverse else 1)}",
                flush=True,
            )
    print(f"[SAVED] {written} local plans -> {args.output_dir}")


if __name__ == "__main__":
    main()
