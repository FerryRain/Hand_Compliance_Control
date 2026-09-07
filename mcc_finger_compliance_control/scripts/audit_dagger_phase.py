"""Audit whether time-indexed DAgger labels conflict with local teacher phase.

For every retained DAgger state, search the corresponding clean teacher episode
inside a local time window.  The audit compares three matching metrics:

* q: nominal joint state only;
* q_tactile: nominal q plus live/teacher contact geometry;
* full: q, tactile geometry, and the local planner command.

The script never changes labels.  It produces per-sample CSV files and a JSON
summary that can be used to decide whether phase-aware relabeling is justified.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


Q_SCALE_RAD = 0.05
CONTACT_POSITION_SCALE_M = 0.01
CONTACT_NORMAL_SCALE = 0.25
PLANNER_TRANSLATION_SCALE_M = 0.005
PLANNER_ROTATION_SCALE_RAD = 0.05


@dataclass(frozen=True)
class TeacherEpisode:
    steps: np.ndarray
    q: np.ndarray
    contact_pos: np.ndarray
    contact_normal: np.ndarray
    contact_mask: np.ndarray
    planner: np.ndarray


def _flat(array: np.ndarray) -> np.ndarray:
    return np.asarray(array, dtype=np.float32).reshape(len(array), -1)


def _rms_scaled(delta: np.ndarray, scale: float) -> np.ndarray:
    return np.sqrt(np.mean(np.square(delta / scale), axis=-1))


def _tactile_distance(
    live_pos: np.ndarray,
    live_normal: np.ndarray,
    live_mask: np.ndarray,
    teacher_pos: np.ndarray,
    teacher_normal: np.ndarray,
    teacher_mask: np.ndarray,
) -> np.ndarray:
    """Return one equally weighted contact-geometry distance per candidate."""
    live_mask = live_mask > 0.5
    teacher_mask = teacher_mask > 0.5
    common = teacher_mask & live_mask[None, :]
    common_count = common.sum(axis=1)

    pos_sq = np.square(
        (teacher_pos - live_pos[None, :, :]) / CONTACT_POSITION_SCALE_M
    ).mean(axis=2)
    normal_sq = np.square(
        (teacher_normal - live_normal[None, :, :]) / CONTACT_NORMAL_SCALE
    ).mean(axis=2)
    denominator = np.maximum(common_count, 1)
    pos_distance = np.sqrt((pos_sq * common).sum(axis=1) / denominator)
    normal_distance = np.sqrt((normal_sq * common).sum(axis=1) / denominator)

    # If no pad is valid in both states, geometry is undefined and should not
    # create an artificially good match just because invalid vectors are zero.
    pos_distance = np.where(common_count > 0, pos_distance, 2.0)
    normal_distance = np.where(common_count > 0, normal_distance, 2.0)
    mask_distance = np.mean(teacher_mask != live_mask[None, :], axis=1)
    return (pos_distance + normal_distance + mask_distance) / 3.0


def _planner_distance(live: np.ndarray, teacher: np.ndarray) -> np.ndarray:
    translation = _rms_scaled(
        teacher[:, :3] - live[None, :3], PLANNER_TRANSLATION_SCALE_M
    )
    rotation = _rms_scaled(
        teacher[:, 3:] - live[None, 3:], PLANNER_ROTATION_SCALE_RAD
    )
    return 0.5 * (translation + rotation)


def _best_offset(
    distances: np.ndarray, offsets: np.ndarray, same_index: int
) -> tuple[int, float, float, bool]:
    # A tiny time-distance term resolves flat/tied regions toward zero phase.
    regularized = distances + 1.0e-7 * np.abs(offsets)
    best_index = int(np.argmin(regularized))
    same_distance = float(distances[same_index])
    best_distance = float(distances[best_index])
    improvement = (same_distance - best_distance) / max(same_distance, 1.0e-8)
    return (
        int(offsets[best_index]),
        same_distance,
        best_distance,
        bool(improvement >= 0.02),
    )


def _load_teacher_episode(
    file: h5py.File, episode_rows: dict[int, np.ndarray], episode_id: int
) -> TeacherEpisode:
    rows = episode_rows[episode_id]
    return TeacherEpisode(
        # Rollout source_episode_step is the local index into load_episode(),
        # while the exported DP file retains the pre-inversion/raw episode step
        # (for Mustard it starts at 1200).  DAgger labels use the former.
        steps=np.arange(len(rows), dtype=np.int32),
        q=_flat(file["q_hand"][rows]),
        contact_pos=np.asarray(file["fingertip_contact_pos_palm"][rows])
        .reshape(-1, 4, 3)
        .astype(np.float32),
        contact_normal=np.asarray(file["fingertip_contact_normal_palm"][rows])
        .reshape(-1, 4, 3)
        .astype(np.float32),
        contact_mask=_flat(file["fingertip_contact_mask"][rows]),
        planner=_flat(file["planner_palm_delta_pose_palm"][rows]),
    )


def _summarize(offsets: np.ndarray, identifiable: np.ndarray, boundary: np.ndarray) -> dict:
    result: dict[str, object] = {
        "samples": int(len(offsets)),
        "phase_frames_mean": float(np.mean(offsets)),
        "phase_frames_median": float(np.median(offsets)),
        "abs_phase_frames_median": float(np.median(np.abs(offsets))),
        "abs_phase_frames_p90": float(np.quantile(np.abs(offsets), 0.90)),
        "abs_phase_frames_p95": float(np.quantile(np.abs(offsets), 0.95)),
        "fraction_abs_gt_5": float(np.mean(np.abs(offsets) > 5)),
        "fraction_abs_gt_10": float(np.mean(np.abs(offsets) > 10)),
        "fraction_abs_gt_20": float(np.mean(np.abs(offsets) > 20)),
        "boundary_fraction": float(np.mean(boundary)),
        "identifiable_fraction": float(np.mean(identifiable)),
    }
    selected = offsets[identifiable]
    if len(selected):
        result["identifiable"] = {
            "samples": int(len(selected)),
            "phase_frames_mean": float(np.mean(selected)),
            "phase_frames_median": float(np.median(selected)),
            "abs_phase_frames_median": float(np.median(np.abs(selected))),
            "abs_phase_frames_p90": float(np.quantile(np.abs(selected), 0.90)),
            "abs_phase_frames_p95": float(np.quantile(np.abs(selected), 0.95)),
            "fraction_abs_gt_10": float(np.mean(np.abs(selected) > 10)),
            "fraction_abs_gt_20": float(np.mean(np.abs(selected) > 20)),
        }
    else:
        result["identifiable"] = {"samples": 0}
    return result


def _audit_one(
    dagger_path: Path,
    teacher_file: h5py.File,
    episode_rows: dict[int, np.ndarray],
    radius: int,
    sample_stride: int,
    label_stride: int,
    label_horizon: int,
    output_csv: Path,
) -> dict:
    with h5py.File(dagger_path, "r") as dagger:
        source_episode = _flat(dagger["source_episode_id"])[:, 0].astype(np.int32)
        source_step = _flat(dagger["source_episode_step"])[:, 0].astype(np.int32)
        q = _flat(dagger["q_hand"])
        contact_pos = np.asarray(dagger["fingertip_contact_pos_palm"])
        contact_pos = contact_pos.reshape(-1, 4, 3).astype(np.float32)
        contact_normal = np.asarray(dagger["fingertip_contact_normal_palm"])
        contact_normal = contact_normal.reshape(-1, 4, 3).astype(np.float32)
        contact_mask = _flat(dagger["fingertip_contact_mask"])
        planner = _flat(dagger["planner_palm_delta_pose_palm"])

    teacher_cache: dict[int, TeacherEpisode] = {}
    rows_out: list[dict[str, object]] = []
    for sample_index in range(0, len(q), sample_stride):
        episode_id = int(source_episode[sample_index])
        step = int(source_step[sample_index])
        if episode_id not in teacher_cache:
            teacher_cache[episode_id] = _load_teacher_episode(
                teacher_file, episode_rows, episode_id
            )
        teacher = teacher_cache[episode_id]
        valid = np.flatnonzero(
            (teacher.steps >= step - radius) & (teacher.steps <= step + radius)
        )
        if not len(valid):
            continue
        same_candidates = np.flatnonzero(teacher.steps[valid] == step)
        if len(same_candidates) != 1:
            raise RuntimeError(
                f"Expected one same-time teacher row for episode={episode_id} step={step}"
            )
        same_index = int(same_candidates[0])
        offsets = teacher.steps[valid] - step

        q_distance = _rms_scaled(teacher.q[valid] - q[sample_index], Q_SCALE_RAD)
        tactile_distance = _tactile_distance(
            contact_pos[sample_index],
            contact_normal[sample_index],
            contact_mask[sample_index],
            teacher.contact_pos[valid],
            teacher.contact_normal[valid],
            teacher.contact_mask[valid],
        )
        planner_distance = _planner_distance(
            planner[sample_index], teacher.planner[valid]
        )
        variants = {
            "q": q_distance,
            "q_tactile": 0.5 * (q_distance + tactile_distance),
            "full": (q_distance + tactile_distance + planner_distance) / 3.0,
        }
        row: dict[str, object] = {
            "sample_index": sample_index,
            "source_episode_id": episode_id,
            "source_episode_step": step,
        }
        for name, distance in variants.items():
            phase, same_d, best_d, identifiable = _best_offset(
                distance, offsets, same_index
            )
            row[f"{name}_phase_frames"] = phase
            row[f"{name}_same_distance"] = same_d
            row[f"{name}_best_distance"] = best_d
            row[f"{name}_improvement_fraction"] = (
                (same_d - best_d) / max(same_d, 1.0e-8)
            )
            row[f"{name}_identifiable"] = int(identifiable)
            row[f"{name}_boundary"] = int(abs(phase) == radius)

        full_phase = int(row["full_phase_frames"])
        future_offsets = np.arange(1, label_horizon + 1) * label_stride
        same_future = step + future_offsets
        matched_future = step + full_phase + future_offsets
        future_valid = (
            (same_future >= 0)
            & (same_future < len(teacher.q))
            & (matched_future >= 0)
            & (matched_future < len(teacher.q))
        )
        if np.all(future_valid):
            label_delta = teacher.q[matched_future] - teacher.q[same_future]
            row["full_phase_future_q_rms_rad"] = float(
                np.sqrt(np.mean(np.square(label_delta)))
            )
            row["full_phase_future_q_max_rad"] = float(
                np.max(np.abs(label_delta))
            )
        else:
            row["full_phase_future_q_rms_rad"] = float("nan")
            row["full_phase_future_q_max_rad"] = float("nan")
        rows_out.append(row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows_out[0]))
        writer.writeheader()
        writer.writerows(rows_out)

    summary: dict[str, object] = {
        "dagger_file": str(dagger_path),
        "output_csv": str(output_csv),
        "sample_stride": sample_stride,
        "samples": len(rows_out),
        "variants": {},
    }
    for name in ("q", "q_tactile", "full"):
        offsets = np.asarray([row[f"{name}_phase_frames"] for row in rows_out])
        identifiable = np.asarray(
            [row[f"{name}_identifiable"] for row in rows_out], dtype=bool
        )
        boundary = np.asarray(
            [row[f"{name}_boundary"] for row in rows_out], dtype=bool
        )
        summary["variants"][name] = _summarize(offsets, identifiable, boundary)
    full_offsets = np.asarray([row["full_phase_frames"] for row in rows_out])
    full_identifiable = np.asarray(
        [row["full_identifiable"] for row in rows_out], dtype=bool
    )
    future_q_rms = np.asarray(
        [row["full_phase_future_q_rms_rad"] for row in rows_out], dtype=float
    )
    future_q_max = np.asarray(
        [row["full_phase_future_q_max_rad"] for row in rows_out], dtype=float
    )
    phase_conflict = (
        full_identifiable & (np.abs(full_offsets) > 10) & np.isfinite(future_q_rms)
    )
    summary["full_phase_future_label_difference"] = {
        "label_stride_frames": label_stride,
        "label_horizon": label_horizon,
        "phase_conflict_samples": int(np.sum(phase_conflict)),
    }
    if np.any(phase_conflict):
        summary["full_phase_future_label_difference"].update(
            {
                "q_rms_rad_median": float(np.median(future_q_rms[phase_conflict])),
                "q_rms_rad_p90": float(np.quantile(future_q_rms[phase_conflict], 0.9)),
                "q_rms_rad_p95": float(np.quantile(future_q_rms[phase_conflict], 0.95)),
                "q_max_rad_median": float(np.median(future_q_max[phase_conflict])),
                "q_max_rad_p95": float(np.quantile(future_q_max[phase_conflict], 0.95)),
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dp", type=Path, required=True)
    parser.add_argument("--dagger", type=Path, nargs="+", required=True)
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--search-radius-frames", type=int, default=50)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--label-stride-frames", type=int, default=5)
    parser.add_argument("--label-horizon", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.search_radius_frames <= 0
        or args.sample_stride <= 0
        or args.label_stride_frames <= 0
        or args.label_horizon <= 0
    ):
        raise ValueError("Search radius, strides, and horizon must be positive")
    if args.labels is not None and len(args.labels) not in (0, len(args.dagger)):
        raise ValueError("--labels must contain exactly one label per --dagger file")

    labels = args.labels or [path.stem for path in args.dagger]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, object] = {}
    with h5py.File(args.reference_dp, "r") as teacher_file:
        episode_ids = _flat(teacher_file["episode_id"])[:, 0].astype(np.int32)
        episode_rows = {
            int(episode_id): np.flatnonzero(episode_ids == episode_id)
            for episode_id in np.unique(episode_ids)
        }
        for label, dagger_path in zip(labels, args.dagger, strict=True):
            print(f"[PHASE-AUDIT] {label}: {dagger_path}", flush=True)
            summaries[label] = _audit_one(
                dagger_path,
                teacher_file,
                episode_rows,
                args.search_radius_frames,
                args.sample_stride,
                args.label_stride_frames,
                args.label_horizon,
                args.output_dir / f"{label}_phase_samples.csv",
            )

    report = {
        "reference_dp": str(args.reference_dp),
        "search_radius_frames": args.search_radius_frames,
        "label_stride_frames": args.label_stride_frames,
        "label_horizon": args.label_horizon,
        "physical_scales": {
            "q_rad": Q_SCALE_RAD,
            "contact_position_m": CONTACT_POSITION_SCALE_M,
            "contact_normal_vector": CONTACT_NORMAL_SCALE,
            "planner_translation_m": PLANNER_TRANSLATION_SCALE_M,
            "planner_rotation_rad": PLANNER_ROTATION_SCALE_RAD,
        },
        "identifiable_min_distance_improvement_fraction": 0.02,
        "datasets": summaries,
    }
    report_path = args.output_dir / "phase_audit.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[PHASE-AUDIT] report -> {report_path}", flush=True)


if __name__ == "__main__":
    main()
