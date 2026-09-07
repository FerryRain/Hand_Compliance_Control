"""Convert executed-q_live collection rollouts into a DP training file.

Input: per-episode rollout H5s written by `--mode collect_executed`
(deploy_dp_inverse.py), each with q_live (actual executed joints under the
deployment MCC), palm-frame contact geometry, and palm pose/twist per frame.

Output: a palm-frame DP training file with the same schema as the D1 motion96
file (dp_state_schema=contact_geometry_planner_motion): q_hand = q_live (obs
AND action), contact pos/normal/mask, palm_relative_twist_palm,
planner_palm_delta_pose_palm (recomputed from palm_pose_object), and the
strictly causal motion features.  Because q_live is the executed trajectory
under the deployment MCC, the compensation distribution the policy trains on
now matches the one it deploys with.

A per-episode JSON report records the q_live-vs-teacher deviation and contact
retention so the data quality is auditable before training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from dp_motion_features import MOTION_SCHEMA, causal_motion_features
from export_palm_dp import validate_palm_dp_file, _wxyz_to_matrix
from palm_planner_features import (
    DEFAULT_PLANNER_STEP_FRAMES,
    DEFAULT_PLANNER_WAYPOINTS,
    future_palm_delta_pose_palm,
)


def _flat(path: Path, key: str) -> np.ndarray:
    with h5py.File(path, "r") as file:
        return np.asarray(file[key], dtype=np.float32)


def export(
    rollout_dir: Path,
    teacher_path: Path,
    output_path: Path,
    state_schema: str = MOTION_SCHEMA,
    planner_waypoints: int = DEFAULT_PLANNER_WAYPOINTS,
    planner_step_frames: int = DEFAULT_PLANNER_STEP_FRAMES,
    motion_feature_step_frames: int = 5,
    block_size: int = 4096,
) -> dict[str, object]:
    rollout_paths = sorted(
        rollout_dir.glob("ep*.h5"),
        key=lambda path: int(path.stem[2:]),
    )
    if not rollout_paths:
        raise FileNotFoundError(f"No ep*.h5 rollouts in {rollout_dir}")
    if state_schema != MOTION_SCHEMA:
        raise ValueError(f"Only {MOTION_SCHEMA} output is supported")

    episode_frames: list[np.ndarray] = []
    for path in rollout_paths:
        with h5py.File(path, "r") as file:
            episode_frames.append(
                np.asarray(file["episode_step"], dtype=np.int32).reshape(-1)
            )
    lengths = [len(steps) for steps in episode_frames]
    total = int(sum(lengths))
    print(
        f"[EXPORT] {len(rollout_paths)} episodes, {total} frames; "
        f"frames per episode {min(lengths)}..{max(lengths)}",
        flush=True,
    )
    if any(length < 10 for length in lengths):
        raise ValueError(
            f"Episode {rollout_paths[int(np.argmin(lengths))].name} has only "
            f"{min(lengths)} frames; rollout is incomplete"
        )

    with h5py.File(teacher_path, "r") as file:
        teacher_ids = np.asarray(file["episode_id"], dtype=np.int64)[:, 0]
        teacher_q = np.asarray(file["q_hand"], dtype=np.float32).reshape(-1, 16)
        teacher_steps = np.asarray(file["episode_step"], dtype=np.int64)[:, 0]
        teacher_attrs = {key: value for key, value in file.attrs.items()}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Pre-allocate the merged arrays first so every dataset is created exactly
    # once; episodes are written as contiguous slices.
    merged_id = np.empty((total, 1), dtype=np.int32)
    merged_steps = np.empty((total, 1), dtype=np.int32)
    merged_q = np.empty((total, 16), dtype=np.float32)
    merged_pos = np.empty((total, 4, 3), dtype=np.float32)
    merged_normal = np.empty((total, 4, 3), dtype=np.float32)
    merged_mask = np.empty((total, 4), dtype=np.float32)
    merged_twist = np.empty((total, 6), dtype=np.float32)
    merged_pose = np.empty((total, 7), dtype=np.float32)

    offset = 0
    per_episode: list[dict[str, object]] = []
    for path, steps in zip(rollout_paths, episode_frames, strict=True):
        episode_id = int(path.stem[2:])
        count = len(steps)
        selection = np.s_[offset : offset + count]
        offset += count

        merged_id[selection, 0] = episode_id
        merged_steps[selection, 0] = steps
        q_live = _flat(path, "q_live").reshape(-1, 16).astype(np.float32)
        contact_pos = _flat(path, "fingertip_contact_pos_palm").reshape(-1, 4, 3)
        contact_normal = _flat(path, "fingertip_contact_normal_palm").reshape(-1, 4, 3)
        contact_mask = _flat(path, "fingertip_contact_mask").reshape(-1, 4)
        palm_pose = _flat(path, "palm_pose_object").reshape(-1, 7)
        palm_twist = _flat(path, "palm_twist_object").reshape(-1, 6)
        if count != len(q_live):
            raise ValueError(
                f"{path.name}: episode_step {count} != q_live {len(q_live)}"
            )
        merged_q[selection] = q_live
        merged_pos[selection] = contact_pos
        merged_normal[selection] = contact_normal
        merged_mask[selection] = contact_mask
        merged_pose[selection] = palm_pose
        # palm_relative_twist_palm: R^T of the object-frame causal twist.
        object_from_palm = _wxyz_to_matrix(palm_pose[:, 3:7])
        palm_from_object = np.swapaxes(object_from_palm, -1, -2)
        merged_twist[selection, :3] = np.einsum(
            "...ij,...j->...i", palm_from_object, palm_twist[:, :3]
        )
        merged_twist[selection, 3:] = np.einsum(
            "...ij,...j->...i", palm_from_object, palm_twist[:, 3:]
        )

        # Teacher comparison for the quality report.
        teacher_rows = np.flatnonzero(teacher_ids == episode_id)
        teacher_q_episode = teacher_q[teacher_rows]
        q_deviation = q_live - teacher_q_episode[:count]
        loaded_3 = (contact_mask.sum(axis=-1) >= 3).mean()
        loaded_4 = (contact_mask.sum(axis=-1) >= 4).mean()
        per_episode.append(
            {
                "episode_id": episode_id,
                "frames": count,
                "teacher_step_start": int(teacher_steps[teacher_rows[0]]),
                "q_live_vs_teacher_rms_rad": float(
                    np.sqrt(np.mean(np.square(q_deviation)))
                ),
                "q_live_vs_teacher_max_rad": float(np.max(np.abs(q_deviation))),
                "q_live_vs_teacher_p95_rad": float(
                    np.quantile(np.abs(q_deviation), 0.95)
                ),
                "contact_loaded_3_fraction": float(loaded_3),
                "contact_loaded_4_fraction": float(loaded_4),
                "per_finger_loaded_fraction": np.round(
                    contact_mask.mean(axis=0), 4
                ).tolist(),
                "contact_mean_force_n": float(
                    np.linalg.norm(_flat(path, "fingertip_force_palm"), axis=-1).mean()
                ),
            }
        )
        print(
            f"[EXPORT] ep {episode_id}: n={count} "
            f"q_rms={per_episode[-1]['q_live_vs_teacher_rms_rad']:.4f}rad "
            f"q_max={per_episode[-1]['q_live_vs_teacher_max_rad']:.4f}rad "
            f"loaded3={loaded_3:.1%} loaded4={loaded_4:.1%}",
            flush=True,
        )

    with h5py.File(output_path, "w") as target:
        chunk = min(4096, total)
        target.create_dataset(
            "episode_id", data=merged_id, chunks=(chunk, 1)
        )
        target.create_dataset(
            "episode_step", data=merged_steps, chunks=(chunk, 1)
        )
        target.create_dataset("q_hand", data=merged_q[:, None, :], chunks=(chunk, 1, 16))
        target.create_dataset(
            "fingertip_contact_pos_palm",
            data=merged_pos[:, None, ...],
            chunks=(chunk, 1, 4, 3),
        )
        target.create_dataset(
            "fingertip_contact_normal_palm",
            data=merged_normal[:, None, ...],
            chunks=(chunk, 1, 4, 3),
        )
        target.create_dataset(
            "fingertip_contact_mask",
            data=merged_mask[:, None, :],
            chunks=(chunk, 1, 4),
        )
        target.create_dataset(
            "palm_relative_twist_palm",
            data=merged_twist[:, None, :],
            chunks=(chunk, 1, 6),
        )

        # Planner feature over the merged file, respecting episode boundaries.
        planner = future_palm_delta_pose_palm(
            merged_pose.astype(np.float64),
            merged_id[:, 0].astype(np.int64),
            waypoint_count=planner_waypoints,
            step_frames=planner_step_frames,
        )
        target.create_dataset(
            "planner_palm_delta_pose_palm",
            data=planner.astype(np.float32)[:, None, ...],
            chunks=(chunk, 1, planner_waypoints, 6),
        )
        q_velocity, point_velocity, normal_rate = causal_motion_features(
            merged_q,
            merged_pos,
            merged_normal,
            merged_mask,
            merged_id[:, 0].astype(np.int64),
            control_dt=float(teacher_attrs.get("control_dt", 0.01)),
            step_frames=motion_feature_step_frames,
        )
        target.create_dataset(
            "q_velocity",
            data=q_velocity[:, None, :],
            chunks=(chunk, 1, 16),
        )
        target.create_dataset(
            "fingertip_contact_point_velocity_palm",
            data=point_velocity[:, None, ...],
            chunks=(chunk, 1, 4, 3),
        )
        target.create_dataset(
            "fingertip_contact_normal_angular_rate_palm",
            data=normal_rate[:, None, ...],
            chunks=(chunk, 1, 4, 3),
        )

        for key, value in teacher_attrs.items():
            target.attrs[key] = value
        target.attrs["schema_version"] = "mcc_tip_palm_dp_v3"
        target.attrs["dp_input_frame"] = "palm"
        target.attrs["palm_frame_body"] = "palm_lower"
        target.attrs["dp_state_schema"] = state_schema
        target.attrs["state_fields"] = (
            "q_hand,fingertip_contact_pos_palm,"
            "fingertip_contact_normal_palm,fingertip_contact_mask,"
            "q_velocity,fingertip_contact_point_velocity_palm,"
            "fingertip_contact_normal_angular_rate_palm,"
            "palm_relative_twist_palm,planner_palm_delta_pose_palm"
        )
        target.attrs["action_field"] = "q_hand"
        target.attrs["action_representation"] = "absolute_q"
        target.attrs["action_coordinate_space"] = "joint_position_rad"
        target.attrs["planner_waypoints"] = planner_waypoints
        target.attrs["planner_step_frames"] = planner_step_frames
        target.attrs["planner_horizon_frames"] = planner_waypoints * planner_step_frames
        target.attrs["planner_feature"] = (
            "future palm delta position and rotation vector in the "
            "current palm frame; waypoint shape [K,6]"
        )
        control_dt = float(teacher_attrs.get("control_dt", 0.01))
        target.attrs["planner_waypoint_dt"] = planner_step_frames * control_dt
        target.attrs["planner_horizon_seconds"] = (
            planner_waypoints * planner_step_frames * control_dt
        )
        target.attrs["motion_feature_step_frames"] = motion_feature_step_frames
        target.attrs["motion_feature_dt"] = motion_feature_step_frames * control_dt
        target.attrs["motion_feature_convention"] = (
            "strictly causal backward difference in per-frame palm coordinates; "
            "contact rates are zero unless both endpoints have valid contact"
        )
        target.attrs["dp_coordinate_contract"] = (
            "q_hand=joint rad; fingertip_contact_pos_palm=point in palm_lower; "
            "fingertip_contact_normal_palm and fingertip_force_palm=vectors in "
            "palm_lower; palm_relative_twist_palm=[linear,angular] in palm_lower; "
            "planner_palm_delta_pose_palm=[translation,rotvec] in current palm_lower; "
            "explicit motion features are causal rates"
        )
        target.attrs["palm_frame_transform"] = (
            "per-frame T_palm_from_object: points use R^T(p-p_palm); "
            "force/normal/linear_velocity/angular_velocity use R^T v"
        )
        # Provenance: this file is relabeled from an executed-q_live collection
        # under the deployment MCC, not from the original raw recording.
        target.attrs["relabel_source"] = "collect_executed_q_live"
        target.attrs["relabel_rollout_dir"] = str(rollout_dir)
        target.attrs["relabel_q_live"] = True
        target.attrs["relabel_obs_q_source"] = "q_live_history"
        target.attrs["relabel_action_q_source"] = "executed_q_live"

    validate_palm_dp_file(output_path)
    print(f"[SUCCESS] executed-q_live DP data saved to {output_path}", flush=True)

    per_episode_ids = [int(row["episode_id"]) for row in per_episode]
    ordered = sorted(
        range(len(per_episode)), key=lambda index: per_episode_ids[index]
    )
    per_episode_sorted = [per_episode[index] for index in ordered]
    q_rms = np.asarray(
        [row["q_live_vs_teacher_rms_rad"] for row in per_episode_sorted]
    )
    q_max = np.asarray(
        [row["q_live_vs_teacher_max_rad"] for row in per_episode_sorted]
    )
    loaded3 = np.asarray(
        [row["contact_loaded_3_fraction"] for row in per_episode_sorted]
    )
    report = {
        "rollout_dir": str(rollout_dir),
        "teacher_file": str(teacher_path),
        "output_file": str(output_path),
        "episodes": len(rollout_paths),
        "frames": total,
        "q_live_vs_teacher_rad": {
            "rms_median": float(np.median(q_rms)),
            "rms_p90": float(np.quantile(q_rms, 0.90)),
            "max_median": float(np.median(q_max)),
            "max_p90": float(np.quantile(q_max, 0.90)),
        },
        "contact_loaded_3_fraction": {
            "median": float(np.median(loaded3)),
            "min": float(np.min(loaded3)),
            "p10": float(np.quantile(loaded3, 0.10)),
        },
        "per_episode": per_episode_sorted,
    }
    report_path = output_path.with_suffix(".relabel_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[EXPORT] relabel report -> {report_path}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True,
                        help="Inverted teacher H5 (provenance attrs + comparison)")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--planner-waypoints", type=int, default=DEFAULT_PLANNER_WAYPOINTS)
    parser.add_argument("--planner-step-frames", type=int, default=DEFAULT_PLANNER_STEP_FRAMES)
    parser.add_argument("--motion-feature-step-frames", type=int, default=5)
    args = parser.parse_args()
    output = args.output or args.rollout_dir.with_name(
        f"{args.rollout_dir.name}_executed_q_live_dp.h5"
    )
    export(
        args.rollout_dir,
        args.teacher,
        output,
        MOTION_SCHEMA,
        args.planner_waypoints,
        args.planner_step_frames,
        args.motion_feature_step_frames,
    )


if __name__ == "__main__":
    main()
