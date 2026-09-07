"""Export randomized dual-track collection data into the 242-D v3 DP H5.

The exporter consumes only the single-source ``batch_*_collected.h5`` chain.
It preserves the causal controller split recorded by collection::

    q_prior, q_cmd                         pre-step
    q_hand, tactile, e_servo, qvel         post-step

The primary label is the v2-B 12-D tangential fingertip intent.  ``q_prior``
is retained in the same file as the 16-D absolute-intent ablation label.
Every output episode is reindexed to ``0..N-1`` and carries its actual
execution-randomization metadata.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np

from dp_motion_features import DUAL_TRACK_V3_SCHEMA, causal_motion_features
from export_palm_dp import _wxyz_to_matrix, validate_palm_dp_file
from invert_trajectories import (
    _backward_palm_twist,
    _enforce_normal_sign_continuity,
    _relative_pose,
)
from palm_planner_features import future_palm_delta_pose_palm


STATE_FIELDS = (
    ("q_prior", 16),
    ("fingertip_contact_pos_palm", 12),
    ("fingertip_contact_normal_palm", 12),
    ("fingertip_contact_mask", 4),
    ("q_prior_velocity", 16),
    ("fingertip_contact_point_velocity_palm", 12),
    ("fingertip_contact_normal_angular_rate_palm", 12),
    ("q_hand", 16),
    ("q_live_velocity", 16),
    ("e_qdot", 16),
    ("delta_q_comp", 16),
    ("e_servo", 16),
    ("palm_relative_twist_palm", 6),
    ("planner_palm_delta_pose_palm", 72),
)


def _strings(values: list[str]) -> np.ndarray:
    return np.asarray(values, dtype=h5py.string_dtype(encoding="utf-8"))


def _source_plan_names(collection: h5py.File, collection_path: Path) -> list[str]:
    configured = Path(str(collection.attrs.get("planner_file", "")))
    candidates = [configured, collection_path.with_name(collection_path.name.replace("_collected", ""))]
    plan_path = next((value for value in candidates if value.is_file()), None)
    if plan_path is None:
        raise FileNotFoundError(
            f"{collection_path}: cannot resolve planner bundle from {candidates}"
        )
    with h5py.File(plan_path, "r") as planner:
        source = np.asarray(planner["source_plan"])
    return [
        Path(value.decode() if isinstance(value, bytes) else str(value)).stem
        for value in source
    ]


def _quality_rows(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as stream:
        return {row["plan_name"]: row for row in csv.DictReader(stream)}


def _tip_delta_tangent(
    target_palm: np.ndarray,
    actual_palm: np.ndarray,
    normal_palm: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Vectorized parity implementation of the frozen v2-B label formula."""

    output = np.zeros_like(target_palm, dtype=np.float32)
    raw = target_palm[1:] - actual_palm[:-1]
    normal = normal_palm[:-1]
    norm = np.linalg.norm(normal, axis=-1, keepdims=True)
    unit = np.divide(
        normal,
        np.maximum(norm, 1.0e-12),
        out=np.zeros_like(normal),
        where=norm > 1.0e-6,
    )
    tangent = raw - np.sum(raw * unit, axis=-1, keepdims=True) * unit
    output[:-1] = np.where(mask[:-1, ..., None] > 0.5, tangent, 0.0)
    return output.astype(np.float32)


def _create_dataset(
    file: h5py.File, name: str, total: int, tail: tuple[int, ...], dtype: str = "f4"
) -> h5py.Dataset:
    return file.create_dataset(
        name,
        shape=(total, 1, *tail),
        dtype=dtype,
        chunks=(min(2048, total), 1, *tail),
        compression="lzf",
    )


def export(
    trajectory_dir: Path,
    output_path: Path,
    *,
    planner_waypoints: int = 12,
    planner_step_frames: int = 6,
    motion_feature_step_frames: int = 5,
) -> dict[str, object]:
    files = sorted(trajectory_dir.glob("batch_*_collected.h5"))
    if not files:
        raise FileNotFoundError(f"no batch_*_collected.h5 under {trajectory_dir}")
    layouts: list[tuple[Path, int, int]] = []
    for path in files:
        with h5py.File(path, "r") as source:
            shape = source["q_hand"].shape
            layouts.append((path, int(shape[0]), int(shape[1])))
    episode_count = sum(envs for _, _, envs in layouts)
    total = sum(frames * envs for _, frames, envs in layouts)
    if episode_count != 227:
        raise ValueError(
            f"frozen v3 source must contain 227 episodes, found {episode_count}"
        )

    quality = _quality_rows(trajectory_dir / "contact_rates.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    episode_metadata: list[dict[str, object]] = []
    velocity_audit: list[tuple[float, float, float, float]] = []
    identity_max = {"delta_q_comp": 0.0, "e_servo": 0.0}

    tails = {
        "episode_id": ((), "i4"),
        "episode_step": ((), "i4"),
        "episode_domain_id": ((), "i4"),
        "q_prior": ((16,), "f4"),
        "q_cmd": ((16,), "f4"),
        "q_hand": ((16,), "f4"),
        "q_ref": ((16,), "f4"),
        "q_pre": ((16,), "f4"),
        "q_prior_velocity": ((16,), "f4"),
        "q_live_velocity": ((16,), "f4"),
        "e_qdot": ((16,), "f4"),
        "delta_q_comp": ((16,), "f4"),
        "e_servo": ((16,), "f4"),
        "fingertip_contact_pos_palm": ((4, 3), "f4"),
        "fingertip_contact_normal_palm": ((4, 3), "f4"),
        "fingertip_contact_mask": ((4,), "f4"),
        "fingertip_contact_point_velocity_palm": ((4, 3), "f4"),
        "fingertip_contact_normal_angular_rate_palm": ((4, 3), "f4"),
        "palm_relative_twist_palm": ((6,), "f4"),
        "planner_palm_delta_pose_palm": ((planner_waypoints, 6), "f4"),
        "tip_x_des_palm": ((4, 3), "f4"),
        "tip_actual_palm": ((4, 3), "f4"),
        "tip_delta_tangent_palm": ((4, 3), "f4"),
        "execution_perturbation_q": ((16,), "f4"),
        "execution_perturbation_target_norm_rad": ((), "f4"),
    }

    with h5py.File(output_path, "w") as target:
        outputs = {
            name: _create_dataset(target, name, total, tail, dtype)
            for name, (tail, dtype) in tails.items()
        }
        cursor = 0
        episode_id = 0
        for file_index, (path, frames, envs) in enumerate(layouts):
            with h5py.File(path, "r") as source:
                timing = str(source.attrs.get("dual_track_timing", ""))
                if timing != "q_prior_and_q_cmd_pre_step__q_live_and_tactile_post_step":
                    raise ValueError(f"{path}: unsupported dual-track timing {timing!r}")
                required = (
                    "q_prior", "q_cmd", "q_hand", "q_ref", "q_pre", "qvel",
                    "delta_q_comp", "e_servo", "palm_pose_world",
                    "object_pose_world", "fingertip_pose_world",
                    "fingertip_contact_pos_world", "oracle_surface_normal_world",
                    "fingertip_contact", "tip_x_des_palm",
                    "execution_perturbation_q",
                    "execution_perturbation_target_norm_rad",
                )
                missing = [name for name in required if name not in source]
                if missing:
                    raise KeyError(f"{path}: missing v3 fields {missing}")
                plan_names = _source_plan_names(source, path)
                if len(plan_names) != envs:
                    raise ValueError(
                        f"{path}: {len(plan_names)} plans for {envs} environments"
                    )
                control_dt = float(source.attrs.get("control_dt", 0.01))
                batch_perturb_config = {
                    key: source.attrs[key].item()
                    if isinstance(source.attrs[key], np.generic)
                    else source.attrs[key]
                    for key in (
                        "execution_clean_env_fraction",
                        "execution_perturbation_min_norm_rad",
                        "execution_perturbation_max_norm_rad",
                        "execution_perturbation_time_constant_s",
                        "execution_perturbation_ramp_steps",
                    )
                    if key in source.attrs
                }

                for env in range(envs):
                    selection = slice(cursor, cursor + frames)
                    q_prior = np.asarray(source["q_prior"][:, env], dtype=np.float32)
                    q_cmd = np.asarray(source["q_cmd"][:, env], dtype=np.float32)
                    q_live = np.asarray(source["q_hand"][:, env], dtype=np.float32)
                    delta = np.asarray(source["delta_q_comp"][:, env], dtype=np.float32)
                    servo = np.asarray(source["e_servo"][:, env], dtype=np.float32)
                    perturb = np.asarray(
                        source["execution_perturbation_q"][:, env], dtype=np.float32
                    )
                    identity_max["delta_q_comp"] = max(
                        identity_max["delta_q_comp"],
                        float(np.max(np.abs(delta - (q_cmd - q_prior)))),
                    )
                    identity_max["e_servo"] = max(
                        identity_max["e_servo"],
                        float(np.max(np.abs(servo - (q_live - q_cmd)))),
                    )

                    object_pose = np.asarray(
                        source["object_pose_world"][:, env], dtype=np.float64
                    )
                    palm_pose = np.asarray(
                        source["palm_pose_world"][:, env], dtype=np.float64
                    )
                    palm_pose_object = _relative_pose(object_pose, palm_pose)
                    local_episode = np.full(frames, episode_id, dtype=np.int32)
                    palm_twist_object = _backward_palm_twist(
                        palm_pose_object, local_episode, control_dt
                    )
                    world_from_palm = _wxyz_to_matrix(palm_pose[:, 3:7])
                    palm_from_world = np.swapaxes(world_from_palm, -1, -2)

                    contact_mask = np.asarray(
                        source["fingertip_contact"][:, env], dtype=np.float32
                    )
                    contact_world = np.asarray(
                        source["fingertip_contact_pos_world"][:, env],
                        dtype=np.float64,
                    )
                    contact_palm = np.einsum(
                        "tij,tfj->tfi",
                        palm_from_world,
                        contact_world - palm_pose[:, None, :3],
                    )
                    contact_palm = np.where(
                        contact_mask[..., None] > 0.5, contact_palm, 0.0
                    ).astype(np.float32)
                    # The source mesh oracle is object-outward; DP uses the
                    # ContactSensor-compatible fingertip->object polarity.
                    normal_world = -np.asarray(
                        source["oracle_surface_normal_world"][:, env],
                        dtype=np.float64,
                    )
                    normal_world /= np.maximum(
                        np.linalg.norm(normal_world, axis=-1, keepdims=True), 1.0e-12
                    )
                    normal_palm = np.einsum(
                        "tij,tfj->tfi", palm_from_world, normal_world
                    )
                    normal_palm = np.where(
                        contact_mask[..., None] > 0.5, normal_palm, 0.0
                    )
                    normal_palm = _enforce_normal_sign_continuity(
                        normal_palm, contact_mask > 0.5, local_episode
                    ).astype(np.float32)

                    q_prior_velocity, point_velocity, normal_rate = (
                        causal_motion_features(
                            q_prior,
                            contact_palm,
                            normal_palm,
                            contact_mask,
                            local_episode,
                            control_dt=control_dt,
                            step_frames=motion_feature_step_frames,
                        )
                    )
                    qvel = np.asarray(source["qvel"][:, env], dtype=np.float32)
                    if qvel.shape[-1] != 22:
                        raise ValueError(f"{path}: expected 22-D qvel, got {qvel.shape}")
                    q_live_velocity = qvel[:, -16:]
                    e_qdot = q_live_velocity - q_prior_velocity

                    palm_twist_palm = np.empty_like(palm_twist_object)
                    palm_twist_palm[:, :3] = np.einsum(
                        "tij,tj->ti", palm_from_world, np.einsum(
                            "tij,tj->ti", _wxyz_to_matrix(object_pose[:, 3:7]),
                            palm_twist_object[:, :3],
                        )
                    )
                    palm_twist_palm[:, 3:] = np.einsum(
                        "tij,tj->ti", palm_from_world, np.einsum(
                            "tij,tj->ti", _wxyz_to_matrix(object_pose[:, 3:7]),
                            palm_twist_object[:, 3:],
                        )
                    )
                    planner = future_palm_delta_pose_palm(
                        palm_pose_object,
                        local_episode,
                        waypoint_count=planner_waypoints,
                        step_frames=planner_step_frames,
                    )

                    tip_pose_world = np.asarray(
                        source["fingertip_pose_world"][:, env, :, :3],
                        dtype=np.float64,
                    )
                    tip_actual = np.einsum(
                        "tij,tfj->tfi",
                        palm_from_world,
                        tip_pose_world - palm_pose[:, None, :3],
                    ).astype(np.float32)
                    tip_target = np.asarray(
                        source["tip_x_des_palm"][:, env], dtype=np.float32
                    )
                    tip_delta = _tip_delta_tangent(
                        tip_target, tip_actual, normal_palm, contact_mask
                    )

                    q_fd = np.zeros_like(q_live)
                    q_fd[1:] = (q_live[1:] - q_live[:-1]) / control_dt
                    sample = slice(1, None, 20)
                    fd = q_fd[sample].reshape(-1)
                    vel = q_live_velocity[sample].reshape(-1)
                    velocity_audit.append(
                        (
                            float(np.sum(fd * vel)),
                            float(np.sum(fd * fd)),
                            float(np.sum(vel * vel)),
                            float(np.sum(np.square(fd - vel))),
                        )
                    )

                    payload = {
                        "episode_id": np.full(frames, episode_id, dtype=np.int32),
                        "episode_step": np.arange(frames, dtype=np.int32),
                        "q_prior": q_prior,
                        "q_cmd": q_cmd,
                        "q_hand": q_live,
                        "q_ref": np.asarray(source["q_ref"][:, env]),
                        "q_pre": np.asarray(source["q_pre"][:, env]),
                        "q_prior_velocity": q_prior_velocity,
                        "q_live_velocity": q_live_velocity,
                        "e_qdot": e_qdot,
                        "delta_q_comp": delta,
                        "e_servo": servo,
                        "fingertip_contact_pos_palm": contact_palm,
                        "fingertip_contact_normal_palm": normal_palm,
                        "fingertip_contact_mask": contact_mask,
                        "fingertip_contact_point_velocity_palm": point_velocity,
                        "fingertip_contact_normal_angular_rate_palm": normal_rate,
                        "palm_relative_twist_palm": palm_twist_palm,
                        "planner_palm_delta_pose_palm": planner,
                        "tip_x_des_palm": tip_target,
                        "tip_actual_palm": tip_actual,
                        "tip_delta_tangent_palm": tip_delta,
                        "execution_perturbation_q": perturb,
                        "execution_perturbation_target_norm_rad": np.asarray(
                            source["execution_perturbation_target_norm_rad"][:, env]
                        ),
                    }
                    perturb_norm = np.linalg.norm(perturb, axis=-1)
                    is_clean = bool(float(np.max(perturb_norm)) < 1.0e-7)
                    payload["episode_domain_id"] = np.full(
                        frames, 0 if is_clean else 1, dtype=np.int32
                    )
                    for name, value in payload.items():
                        outputs[name][selection, 0] = value

                    plan_name = plan_names[env]
                    quality_row = quality.get(plan_name, {})
                    episode_metadata.append(
                        {
                            "episode_id": episode_id,
                            "source_file": str(path),
                            "source_env": env,
                            "plan_name": plan_name,
                            "domain": "clean" if is_clean else "perturbed",
                            "perturbation_actual_max_norm_rad": float(
                                np.max(perturb_norm)
                            ),
                            "perturbation_actual_p95_norm_rad": float(
                                np.quantile(perturb_norm, 0.95)
                            ),
                            "all4_contact_rate": float(
                                quality_row.get("all4_contact_rate", "nan")
                            ),
                            "quality_status": quality_row.get("status", "unknown"),
                            "batch_perturbation_config": batch_perturb_config,
                        }
                    )
                    cursor += frames
                    episode_id += 1

        metadata = target.create_group("episode_metadata")
        metadata.create_dataset(
            "episode_id",
            data=np.arange(episode_count, dtype=np.int32),
        )
        metadata.create_dataset(
            "source_file", data=_strings([str(row["source_file"]) for row in episode_metadata])
        )
        metadata.create_dataset(
            "source_env", data=np.asarray([row["source_env"] for row in episode_metadata], dtype=np.int32)
        )
        metadata.create_dataset(
            "plan_name", data=_strings([str(row["plan_name"]) for row in episode_metadata])
        )
        metadata.create_dataset(
            "domain", data=_strings([str(row["domain"]) for row in episode_metadata])
        )
        for name in (
            "perturbation_actual_max_norm_rad",
            "perturbation_actual_p95_norm_rad",
            "all4_contact_rate",
        ):
            metadata.create_dataset(
                name,
                data=np.asarray([row[name] for row in episode_metadata], dtype=np.float32),
            )
        metadata.create_dataset(
            "quality_status",
            data=_strings([str(row["quality_status"]) for row in episode_metadata]),
        )
        metadata.create_dataset(
            "perturbation_config_json",
            data=_strings(
                [json.dumps(row["batch_perturbation_config"], sort_keys=True) for row in episode_metadata]
            ),
        )

        target.attrs["schema_version"] = "mcc_tip_dual_track_v3"
        target.attrs["dp_state_schema"] = DUAL_TRACK_V3_SCHEMA
        target.attrs["dp_input_frame"] = "palm"
        target.attrs["palm_frame_body"] = "palm_lower"
        target.attrs["state_fields"] = ",".join(name for name, _ in STATE_FIELDS)
        target.attrs["state_dim"] = sum(size for _, size in STATE_FIELDS)
        target.attrs["action_field"] = "tip_delta_tangent_palm"
        target.attrs["action_dim"] = 12
        target.attrs["action_representation"] = "absolute_q"
        target.attrs["action_coordinate_space"] = "palm_tangent_displacement_m"
        target.attrs["alternate_action_field"] = "q_prior"
        target.attrs["alternate_action_dim"] = 16
        target.attrs["dual_track_timing"] = (
            "q_prior_and_q_cmd_pre_step__q_live_and_tactile_post_step"
        )
        target.attrs["tip_intent_timing"] = (
            "tip_x_des_palm and q_prior are from the same pre-step policy update"
        )
        target.attrs["tip_delta_tangent_convention"] = (
            "v2 parity: proj_tan(tip_x_des_palm[t+1]-tip_actual_palm[t], "
            "contact_normal_palm[t]); zero without contact"
        )
        target.attrs["q_live_velocity_source"] = (
            "same-frame post-step qvel[...,6:22]"
        )
        target.attrs["q_prior_velocity_source"] = (
            "causal backward difference over motion_feature_step_frames"
        )
        target.attrs["contact_normal_source"] = (
            "actual-contact-gated undecomposed source-mesh oracle"
        )
        target.attrs["contact_normal_polarity"] = "primary_fingertip_to_object"
        target.attrs["contact_normal_normalized"] = True
        target.attrs["domain_names_json"] = json.dumps(["clean", "perturbed"])
        target.attrs["planner_waypoints"] = planner_waypoints
        target.attrs["planner_step_frames"] = planner_step_frames
        target.attrs["planner_horizon_frames"] = planner_waypoints * planner_step_frames
        target.attrs["planner_waypoint_dt"] = planner_step_frames * 0.01
        target.attrs["planner_horizon_seconds"] = planner_waypoints * planner_step_frames * 0.01
        target.attrs["motion_feature_step_frames"] = motion_feature_step_frames
        target.attrs["motion_feature_dt"] = motion_feature_step_frames * 0.01
        target.attrs["control_dt"] = 0.01
        target.attrs["episode_count"] = episode_count
        target.attrs["source_files_json"] = json.dumps([str(path) for path, _, _ in layouts])
        target.attrs["identity_delta_q_comp_max_abs_rad"] = identity_max["delta_q_comp"]
        target.attrs["identity_e_servo_max_abs_rad"] = identity_max["e_servo"]

    validate_palm_dp_file(output_path)
    dot = sum(value[0] for value in velocity_audit)
    fd2 = sum(value[1] for value in velocity_audit)
    vel2 = sum(value[2] for value in velocity_audit)
    mse = sum(value[3] for value in velocity_audit) / max(
        1, len(velocity_audit) * 125 * 16
    )
    report = {
        "output": str(output_path),
        "episodes": episode_count,
        "frames": total,
        "clean_episodes": sum(row["domain"] == "clean" for row in episode_metadata),
        "perturbed_episodes": sum(row["domain"] == "perturbed" for row in episode_metadata),
        "state_dim": sum(size for _, size in STATE_FIELDS),
        "identity_max_abs_rad": identity_max,
        "qvel_same_frame_cosine": float(dot / max(np.sqrt(fd2 * vel2), 1.0e-12)),
        "qvel_vs_backward_difference_rmse_rad_s": float(np.sqrt(mse)),
        "all4_contact_rate": {
            "median": float(np.nanmedian([row["all4_contact_rate"] for row in episode_metadata])),
            "mean": float(np.nanmean([row["all4_contact_rate"] for row in episode_metadata])),
            "min": float(np.nanmin([row["all4_contact_rate"] for row in episode_metadata])),
        },
    }
    report_path = output_path.with_suffix(".audit.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"[SUCCESS] v3 dual-track export -> {output_path}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--planner-waypoints", type=int, default=12)
    parser.add_argument("--planner-step-frames", type=int, default=6)
    parser.add_argument("--motion-feature-step-frames", type=int, default=5)
    args = parser.parse_args()
    export(
        args.trajectory_dir,
        args.output,
        planner_waypoints=args.planner_waypoints,
        planner_step_frames=args.planner_step_frames,
        motion_feature_step_frames=args.motion_feature_step_frames,
    )


if __name__ == "__main__":
    main()
