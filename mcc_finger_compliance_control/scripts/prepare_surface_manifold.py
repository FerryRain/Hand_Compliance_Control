"""Build a causal GP-surface dataset for PointNet pretraining.

The compact output is sampled at the DP observation stride.  Every GP point
set uses only the preceding contact history, transformed into the current
palm frame.  Future contact geometry is stored solely as a pretraining target.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import h5py
import numpy as np
from tqdm.auto import tqdm

from palm_planner_features import future_palm_delta_pose_palm
from surface_manifold_gp import GPManifoldConfig, local_gp_point_features


def _wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    quaternion /= np.maximum(np.linalg.norm(quaternion, axis=-1, keepdims=True), 1e-12)
    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    matrix = np.empty((*quaternion.shape[:-1], 3, 3), dtype=np.float64)
    matrix[..., 0, 0] = 1 - 2 * (y * y + z * z)
    matrix[..., 0, 1] = 2 * (x * y - z * w)
    matrix[..., 0, 2] = 2 * (x * z + y * w)
    matrix[..., 1, 0] = 2 * (x * y + z * w)
    matrix[..., 1, 1] = 1 - 2 * (x * x + z * z)
    matrix[..., 1, 2] = 2 * (y * z - x * w)
    matrix[..., 2, 0] = 2 * (x * z - y * w)
    matrix[..., 2, 1] = 2 * (y * z + x * w)
    matrix[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return matrix


def _points_to_current_palm(
    points_object: np.ndarray, current_pose_object: np.ndarray
) -> np.ndarray:
    palm_from_object = _wxyz_to_matrix(current_pose_object[3:7]).T
    return np.einsum(
        "ij,...j->...i", palm_from_object, points_object - current_pose_object[:3]
    )


def _vectors_to_current_palm(
    vectors_object: np.ndarray, current_pose_object: np.ndarray
) -> np.ndarray:
    palm_from_object = _wxyz_to_matrix(current_pose_object[3:7]).T
    return np.einsum("ij,...j->...i", palm_from_object, vectors_object)


def prepare(
    source_path: Path,
    output_path: Path,
    stride: int,
    future_steps: int,
    config: GPManifoldConfig,
) -> None:
    config.validate()
    if stride <= 0 or future_steps <= 0:
        raise ValueError("stride and future_steps must be positive")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(source_path, "r") as source:
        episode_id = np.asarray(source["episode_id"]).reshape(-1).astype(np.int32)
        unique_ids = np.unique(episode_id)
        sampled_count = sum(len(np.flatnonzero(episode_id == eid)[::stride]) for eid in unique_ids)
        with h5py.File(output_path, "w") as target:
            datasets = {
                "episode_id": target.create_dataset("episode_id", (sampled_count,), dtype="i4"),
                "episode_step": target.create_dataset("episode_step", (sampled_count,), dtype="i4"),
                "raw_index": target.create_dataset("raw_index", (sampled_count,), dtype="i8"),
                "q_hand": target.create_dataset("q_hand", (sampled_count, 16), dtype="f4"),
                "planner_command": target.create_dataset("planner_command", (sampled_count, 6), dtype="f4"),
                "gp_points": target.create_dataset(
                    "gp_points",
                    (sampled_count, 4, config.query_count, 10),
                    dtype="f4",
                    chunks=(min(512, sampled_count), 4, config.query_count, 10),
                    compression="gzip",
                    compression_opts=2,
                ),
                "future_contact_delta": target.create_dataset(
                    "future_contact_delta", (sampled_count, 4, 3), dtype="f4"
                ),
                "future_contact_normal": target.create_dataset(
                    "future_contact_normal", (sampled_count, 4, 3), dtype="f4"
                ),
                "future_contact_mask": target.create_dataset(
                    "future_contact_mask", (sampled_count, 4), dtype="f4"
                ),
            }
            cursor = 0
            for eid in tqdm(unique_ids, desc="Causal GP manifolds", unit="episode"):
                raw_indices = np.flatnonzero(episode_id == eid)
                sampled_raw = raw_indices[::stride]
                pose = np.asarray(source["palm_pose_object"][sampled_raw, 0], dtype=np.float64)
                position = np.asarray(
                    source["fingertip_contact_pos_object"][sampled_raw, 0], dtype=np.float64
                )
                normal = np.asarray(
                    source["fingertip_contact_normal_object"][sampled_raw, 0], dtype=np.float64
                )
                mask = np.asarray(source["fingertip_contact"][sampled_raw, 0], dtype=bool)
                q_hand = np.asarray(source["q_hand"][sampled_raw, 0], dtype=np.float32)
                planner = future_palm_delta_pose_palm(
                    pose,
                    np.zeros(len(pose), dtype=np.int32),
                    waypoint_count=1,
                    step_frames=future_steps,
                )[:, 0]
                count = len(sampled_raw)
                gp_points = np.zeros((count, 4, config.query_count, 10), dtype=np.float32)
                future_delta = np.zeros((count, 4, 3), dtype=np.float32)
                future_normal = np.zeros((count, 4, 3), dtype=np.float32)
                future_mask = np.zeros((count, 4), dtype=np.float32)
                for time_index in range(count):
                    history_start = max(0, time_index - config.history_steps + 1)
                    history = slice(history_start, time_index + 1)
                    current_pose = pose[time_index]
                    history_position = _points_to_current_palm(position[history], current_pose)
                    history_normal = _vectors_to_current_palm(normal[history], current_pose)
                    for finger in range(4):
                        gp_points[time_index, finger] = local_gp_point_features(
                            history_position[:, finger],
                            history_normal[:, finger],
                            mask[history, finger],
                            config,
                        )
                    target_index = min(time_index + future_steps, count - 1)
                    current_position = _points_to_current_palm(
                        position[time_index], current_pose
                    )
                    target_position = _points_to_current_palm(
                        position[target_index], current_pose
                    )
                    target_normal = _vectors_to_current_palm(normal[target_index], current_pose)
                    target_valid = mask[time_index] & mask[target_index]
                    future_delta[time_index] = target_position - current_position
                    future_normal[time_index] = target_normal
                    future_mask[time_index] = target_valid.astype(np.float32)

                selection = slice(cursor, cursor + count)
                datasets["episode_id"][selection] = int(eid)
                datasets["episode_step"][selection] = np.asarray(
                    source["episode_step"][sampled_raw, 0], dtype=np.int32
                )
                datasets["raw_index"][selection] = sampled_raw
                datasets["q_hand"][selection] = q_hand
                datasets["planner_command"][selection] = planner.astype(np.float32)
                datasets["gp_points"][selection] = gp_points
                datasets["future_contact_delta"][selection] = future_delta
                datasets["future_contact_normal"][selection] = future_normal
                datasets["future_contact_mask"][selection] = future_mask
                cursor += count
            target.attrs["source_file"] = str(source_path)
            target.attrs["source_stride"] = stride
            target.attrs["effective_dt"] = float(source.attrs.get("control_dt", 0.01)) * stride
            target.attrs["future_steps"] = future_steps
            target.attrs["future_seconds"] = target.attrs["effective_dt"] * future_steps
            target.attrs["gp_config"] = json.dumps(asdict(config), sort_keys=True)
            target.attrs["causal"] = True
            target.attrs["future_fields_are_targets_only"] = True
    print(f"[SUCCESS] causal GP manifold data saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--history-steps", type=int, default=16)
    parser.add_argument("--future-steps", type=int, default=4)
    parser.add_argument("--query-count", type=int, default=8)
    parser.add_argument("--query-radius", type=float, default=0.006)
    parser.add_argument("--length-scale", type=float, default=0.008)
    parser.add_argument("--signal-std", type=float, default=0.004)
    parser.add_argument("--noise-std", type=float, default=0.0005)
    args = parser.parse_args()
    prepare(
        args.file,
        args.output,
        args.stride,
        args.future_steps,
        GPManifoldConfig(
            history_steps=args.history_steps,
            query_count=args.query_count,
            query_radius=args.query_radius,
            length_scale=args.length_scale,
            signal_std=args.signal_std,
            noise_std=args.noise_std,
        ),
    )


if __name__ == "__main__":
    main()
