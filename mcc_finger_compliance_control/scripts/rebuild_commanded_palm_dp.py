"""Replace future-realized palm conditioning with the preissued palm plan.

This is for the frozen planner_inverse collection, whose complete plan was
loaded before motion.  It does not claim that an object-frame plan is available
from a future real-world planner.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation


def commanded_delta(
    current_pose: np.ndarray, future_plan_pose: np.ndarray
) -> np.ndarray:
    """Return a planned target relative to the measured current palm frame."""
    current = np.asarray(current_pose, dtype=np.float64)
    future = np.asarray(future_plan_pose, dtype=np.float64)
    if current.shape != future.shape or current.ndim != 2 or current.shape[1] != 7:
        raise ValueError(f"expected matching (T,7) poses, got {current.shape}, {future.shape}")
    current_rotation = Rotation.from_quat(current[:, [4, 5, 6, 3]])
    future_rotation = Rotation.from_quat(future[:, [4, 5, 6, 3]])
    translation = current_rotation.inv().apply(future[:, :3] - current[:, :3])
    rotation = (current_rotation.inv() * future_rotation).as_rotvec()
    return np.concatenate((translation, rotation), axis=1).astype(np.float32)


def plan_indices(
    schedule: np.ndarray, horizon: int, motion_length: int, plan_length: int,
    endpoint_sample_count: bool = False,
) -> np.ndarray:
    """Match InversePlannedPalmObjectController.source_plan_index at speed 1."""
    if horizon <= 0 or motion_length <= 0 or plan_length < 2:
        raise ValueError("horizon and motion_length must be positive; plan needs 2 poses")
    last = max(motion_length-1,1) if endpoint_sample_count else motion_length
    executed = np.clip(np.asarray(schedule, dtype=np.int64) + horizon, 0, last)
    return np.rint(executed / last * (plan_length - 1)).astype(np.int64)


def _decode(value: bytes | str) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def rebuild(
    base_dp: Path, inverted: Path, selected_dirs: list[Path], output: Path
) -> dict[str, float | int]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    with h5py.File(base_dp, "r") as base, h5py.File(inverted, "r") as inv:
        horizon = int(base.attrs["planner_step_frames"])
        waypoints = int(base.attrs["planner_waypoints"])
        ids = np.asarray(base["episode_id"], dtype=np.int64).reshape(-1)
        inv_ids = np.asarray(inv["episode_id"], dtype=np.int64).reshape(-1)
        if not np.array_equal(ids, inv_ids):
            raise ValueError("DP and inverted episode IDs differ")
        labels = [_decode(value) for value in inv["source_trajectory"][:]]
        names = [Path(label.split("#env=", 1)[0]).stem for label in labels]
        if "source_plan_name" in base:
            base_names = [_decode(value) for value in base["source_plan_name"][:]]
            if base_names != names:
                raise ValueError("DP and inverted source trajectory order differs")
        if not np.array_equal(np.unique(ids), np.arange(len(names))):
            raise ValueError("episode IDs must be contiguous and match source names")

        selected = {}
        for directory in selected_dirs:
            for path in directory.glob("*.h5"):
                if path.stem in selected:
                    raise ValueError(f"duplicate selected source name {path.stem}")
                selected[path.stem] = path
        features = np.empty((len(ids), 1, waypoints, 6), dtype=np.float32)
        future_realized = np.asarray(base["planner_palm_delta_pose_palm"])
        if future_realized.shape != features.shape:
            raise ValueError(f"unexpected planner feature shape {future_realized.shape}")
        translation_error: list[np.ndarray] = []
        rotation_error: list[np.ndarray] = []
        cursor = 0
        for episode, name in enumerate(names):
            if name not in selected:
                raise FileNotFoundError(f"no selected source for {name}")
            source_path = selected[name]
            with h5py.File(source_path, "r") as raw:
                if not bool(raw.attrs.get("planner_inverse", False)):
                    raise ValueError(f"{source_path}: not a planner_inverse collection")
                plan = np.asarray(raw["palm_command_pose_object"], dtype=np.float64)
                if plan.ndim == 3 and plan.shape[1] == 1:
                    plan = plan[:, 0]
                if plan.ndim != 2 or plan.shape[1] != 7:
                    raise ValueError(f"{source_path}: invalid plan shape {plan.shape}")
                schedule = np.asarray(raw["object_motion_schedule_step"][:, 0], dtype=np.int64)
                motion_length_values = np.unique(raw["object_segment_move_steps"][:, 0])
                if len(motion_length_values) != 1:
                    raise ValueError(f"{source_path}: changing motion_length")
                motion_length = int(motion_length_values[0])
                episode_steps = np.asarray(raw["episode_step"][:, 0], dtype=np.int64)
                motion_starts = np.asarray(raw["actual_motion_start_step"][:, 0], dtype=np.int64)
                if not np.array_equal(schedule, episode_steps - motion_starts):
                    raise ValueError(f"{source_path}: schedule and episode steps disagree")
                count = len(schedule)
                if not np.all(ids[cursor : cursor + count] == episode):
                    raise ValueError(f"{source_path}: DP episode length/order mismatch")
                actual = np.asarray(
                    inv["palm_pose_object"][cursor : cursor + count, 0],
                    dtype=np.float64,
                )
                for waypoint in range(waypoints):
                    target = plan[
                        plan_indices(
                            schedule, horizon * (waypoint + 1), motion_length, len(plan),
                            raw.attrs.get('planner_motion_indexing') == 'endpoint_sample_count_minus_one_v1',
                        )
                    ]
                    new_feature = commanded_delta(actual, target)
                    features[cursor : cursor + count, 0, waypoint] = new_feature
                    difference = (
                        new_feature
                        - future_realized[cursor : cursor + count, 0, waypoint]
                    )
                    translation_error.append(np.linalg.norm(difference[:, :3], axis=1))
                    rotation_error.append(np.linalg.norm(difference[:, 3:], axis=1))
                cursor += count
        if cursor != len(ids):
            raise ValueError(f"only aligned {cursor}/{len(ids)} DP rows")

    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(base_dp, output)
    try:
        with h5py.File(output, "r+") as target:
            dataset = target["planner_palm_delta_pose_palm"]
            dataset[:] = features
            target.attrs["planner_feature_source"] = "preissued_palm_command_pose_object"
            target.attrs["planner_feature_anchor"] = "same_frame_executed_palm_pose_object"
            target.attrs["planner_command_frame"] = "object"
            target.attrs["planner_speed_factor_assumed"] = 1.0
            target.attrs["planner_feature_causal_scope"] = "frozen_plan_known_before_episode"
            target.attrs["planner_feature_not_proven_hardware_deployable"] = True
            target.attrs["palm_twist_deployment_scope"] = (
                "fixed_object_active_palm_equivalent_only; moving_object_needs_relative_motion_estimate"
            )
            target.attrs["planner_feature_base_dp"] = str(base_dp.resolve())
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    translation = np.concatenate(translation_error)
    rotation = np.concatenate(rotation_error)
    return {
        "episodes": len(names),
        "frames": len(ids),
        "delta_position_mean_m": float(np.mean(translation)),
        "delta_position_p95_m": float(np.percentile(translation, 95)),
        "delta_rotation_vector_mean_rad": float(np.mean(rotation)),
        "delta_rotation_vector_p95_rad": float(np.percentile(rotation, 95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dp", type=Path, required=True)
    parser.add_argument("--inverted", type=Path, required=True)
    parser.add_argument("--selected-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(rebuild(args.base_dp, args.inverted, args.selected_dir, args.output))


if __name__ == "__main__":
    main()
