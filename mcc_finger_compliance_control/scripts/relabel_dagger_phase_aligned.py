"""Constrained Viterbi phase-aligned relabeling of a DAgger H5 file.

The DAgger label contract is: observation is the causal state actually fed to
DP, and action_q_hand is the time-indexed successful teacher q.  When the
rollout has drifted from the teacher phase, same-time labels conflict with the
state the policy actually visits.  The audit (audit_dagger_phase.py) measures
this phase conflict but never changes labels.

This script changes exactly one variable: the label.  For every contiguous
segment (one source episode, consecutive source steps) it finds the teacher
phase path s_t inside a local window [step-R, step+R] that minimizes

    sum_t C(t, s_t) + transition penalties

where C is the same full distance (q + tactile + planner, normalized by the
audit's physical scales) and the transition penalty keeps the path smooth,
roughly velocity-matched and monotone:

    delta = phase(t+1) - phase(t) = (offset_{t+1} - offset_t) + 1
    penalty = lambda_speed * (offset_{t+1} - offset_t)^2
            + lambda_backward * max(0, -delta)
            + lambda_jump * max(0, |delta| - jump_cap)^2

Independent per-frame argmin produces non-monotonic, jittery phase paths, so
the whole segment is solved with one Viterbi pass; a tiny |offset|
regularization anchors flat regions toward the same-time phase, as in the
audit.

The output H5 is drop-in compatible with the input DAgger H5: every dataset
and attribute is copied verbatim except action_q_hand, which is replaced by
the phase-matched teacher future q (future offsets 1..horizon times the label
stride, as in the audit), and a new dataset source_teacher_step records the
matched teacher local step for auditing.  Observations are never modified, so
training the same checkpoint with and without this file is a causal A/B of the
label phase alone.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_dagger_phase import (  # noqa: E402
    Q_SCALE_RAD,
    TeacherEpisode,
    _load_teacher_episode,
    _planner_distance,
    _rms_scaled,
    _tactile_distance,
)


@dataclass(frozen=True)
class ViterbiParams:
    radius: int
    lambda_speed: float
    lambda_backward: float
    lambda_jump: float
    jump_cap: int
    lambda_time: float


def _full_distance(
    live: dict[str, np.ndarray],
    teacher: TeacherEpisode,
    candidate_rows: np.ndarray,
) -> np.ndarray:
    """Full normalized distance from one live state to every candidate teacher row."""
    q_distance = _rms_scaled(teacher.q[candidate_rows] - live["q"], Q_SCALE_RAD)
    tactile_distance = _tactile_distance(
        live["contact_pos"],
        live["contact_normal"],
        live["contact_mask"],
        teacher.contact_pos[candidate_rows],
        teacher.contact_normal[candidate_rows],
        teacher.contact_mask[candidate_rows],
    )
    planner_distance = _planner_distance(
        live["planner"], teacher.planner[candidate_rows]
    )
    return (q_distance + tactile_distance + planner_distance) / 3.0


def _viterbi_phase_path(
    steps: np.ndarray,
    lives: list[dict[str, np.ndarray]],
    teacher: TeacherEpisode,
    params: ViterbiParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (matched_steps, aligned_distance, same_distance, argmin_offsets)."""
    n = len(steps)
    offsets: list[np.ndarray] = []
    costs: list[np.ndarray] = []
    argmin_offsets = np.zeros(n, dtype=np.int32)
    for t, step in enumerate(steps):
        lo = max(-params.radius, -int(step))
        hi = min(params.radius, int(len(teacher.q) - 1 - step))
        cand = np.arange(lo, hi + 1, dtype=np.int32)
        cost = _full_distance(lives[t], teacher, step + cand)
        regularized = cost + params.lambda_time * np.abs(cand)
        argmin_offsets[t] = cand[int(np.argmin(regularized))]
        offsets.append(cand)
        costs.append(regularized)

    prev_offset = offsets[0]
    prev_dp = costs[0].copy()
    backptr: list[np.ndarray] = []
    cap = float(params.jump_cap)
    for t in range(1, n):
        cur_offset = offsets[t]
        d_off = cur_offset[:, None].astype(np.float64) - prev_offset[None, :]
        # delta = phase(t) - phase(t-1) = d_off + 1 (consecutive source steps)
        delta = d_off + 1.0
        transition = (
            params.lambda_speed * d_off ** 2
            + params.lambda_backward * np.maximum(0.0, -delta)
            + params.lambda_jump * np.maximum(0.0, np.abs(delta) - cap) ** 2
        )
        total = prev_dp[None, :] + transition
        best_j = np.argmin(total, axis=1)
        prev_dp = costs[t] + total[np.arange(len(cur_offset)), best_j]
        prev_offset = cur_offset
        backptr.append(best_j)

    matched_offsets = np.zeros(n, dtype=np.int32)
    index = int(np.argmin(prev_dp))
    for t in range(n - 1, 0, -1):
        matched_offsets[t] = offsets[t][index]
        index = int(backptr[t - 1][index])
    matched_offsets[0] = offsets[0][index]
    matched_steps = steps + matched_offsets

    aligned_distance = np.zeros(n, dtype=np.float32)
    same_distance = np.zeros(n, dtype=np.float32)
    for t in range(n):
        aligned_distance[t] = float(
            _full_distance(lives[t], teacher, np.asarray([matched_steps[t]]))[0]
        )
        same_distance[t] = float(
            _full_distance(lives[t], teacher, np.asarray([steps[t]]))[0]
        )
    return matched_steps, aligned_distance, same_distance, argmin_offsets


def _load_segments(
    dagger_path: Path,
) -> tuple[list[tuple[int, np.ndarray, np.ndarray]], dict[str, np.ndarray], dict]:
    """Return (episode_id, row_index, source_step) triples of contiguous runs."""
    with h5py.File(dagger_path, "r") as f:
        src_id = np.asarray(f["source_episode_id"])[:, 0].astype(np.int32)
        src_step = np.asarray(f["source_episode_step"])[:, 0].astype(np.int32)
        fields = {
            "q": np.asarray(f["q_hand"], dtype=np.float32).reshape(-1, 16),
            "contact_pos": np.asarray(f["fingertip_contact_pos_palm"], dtype=np.float32)
            .reshape(-1, 4, 3),
            "contact_normal": np.asarray(
                f["fingertip_contact_normal_palm"], dtype=np.float32
            ).reshape(-1, 4, 3),
            "contact_mask": np.asarray(
                f["fingertip_contact_mask"], dtype=np.float32
            ).reshape(-1, 4),
            "planner": np.asarray(f["planner_palm_delta_pose_palm"], dtype=np.float32)
            .reshape(-1, 6),
        }
        attrs = {key: value for key, value in f.attrs.items()}

    segments: list[tuple[int, np.ndarray, np.ndarray]] = []
    for episode_id in np.unique(src_id):
        idx = np.flatnonzero(src_id == episode_id)
        order = idx[np.argsort(src_step[idx])]
        st = src_step[order]
        breaks = np.flatnonzero(np.diff(st) != 1) + 1
        for run in np.split(order, breaks):
            if len(run) >= 2:
                # ``run`` contains original file rows. Using st[:len(run)]
                # silently assigns the first run's timestamps to every later
                # run from the same source episode.
                segments.append(
                    (int(episode_id), run, src_step[run].astype(np.int32))
                )
    return segments, fields, attrs


def _summarize_offsets(offsets: np.ndarray) -> dict[str, object]:
    return {
        "phase_frames_mean": float(np.mean(offsets)),
        "phase_frames_median": float(np.median(offsets)),
        "abs_phase_frames_median": float(np.median(np.abs(offsets))),
        "abs_phase_frames_p90": float(np.quantile(np.abs(offsets), 0.90)),
        "abs_phase_frames_p95": float(np.quantile(np.abs(offsets), 0.95)),
        "fraction_abs_gt_5": float(np.mean(np.abs(offsets) > 5)),
        "fraction_abs_gt_10": float(np.mean(np.abs(offsets) > 10)),
        "fraction_abs_gt_20": float(np.mean(np.abs(offsets) > 20)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dagger", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--search-radius-frames", type=int, default=50)
    parser.add_argument("--label-stride-frames", type=int, default=5)
    parser.add_argument("--label-horizon", type=int, default=8)
    # Calibrated against the audit's per-sample distance improvement of
    # roughly 0.008 (median ~5% of ~0.12 full distance): phase glides of up
    # to ~4 frames per sample must be cheaper than that gain, while a
    # one-step phase reversal and large jumps stay expensive.
    parser.add_argument("--lambda-speed", type=float, default=0.0004)
    parser.add_argument("--lambda-backward", type=float, default=0.05)
    parser.add_argument("--lambda-jump", type=float, default=0.01)
    parser.add_argument("--jump-cap", type=int, default=3)
    parser.add_argument("--lambda-time", type=float, default=1.0e-4)
    args = parser.parse_args()

    params = ViterbiParams(
        radius=args.search_radius_frames,
        lambda_speed=args.lambda_speed,
        lambda_backward=args.lambda_backward,
        lambda_jump=args.lambda_jump,
        jump_cap=args.jump_cap,
        lambda_time=args.lambda_time,
    )

    segments, fields, attrs = _load_segments(args.dagger)
    n_total = sum(len(run) for _, run, _ in segments)
    print(
        f"[PHASE-ALIGN] {args.dagger.name}: {n_total} frames, "
        f"{len(segments)} segments (>=2 frames each)",
        flush=True,
    )

    all_matched: list[np.ndarray] = []
    all_argmin_offsets: list[np.ndarray] = []
    all_aligned_distance: list[np.ndarray] = []
    all_same_distance: list[np.ndarray] = []
    all_runs: list[np.ndarray] = []
    teacher_q_by_episode: dict[int, np.ndarray] = {}
    per_segment: list[dict[str, object]] = []

    with h5py.File(args.teacher, "r") as teacher_file:
        episode_ids = np.asarray(teacher_file["episode_id"])[:, 0].astype(np.int32)
        episode_rows = {
            int(episode_id): np.flatnonzero(episode_ids == episode_id)
            for episode_id in np.unique(episode_ids)
        }
        for episode_id, run, steps in segments:
            teacher = _load_teacher_episode(teacher_file, episode_rows, episode_id)
            teacher_q_by_episode[episode_id] = teacher.q
            lives = [
                {
                    "q": fields["q"][i],
                    "contact_pos": fields["contact_pos"][i],
                    "contact_normal": fields["contact_normal"][i],
                    "contact_mask": fields["contact_mask"][i],
                    "planner": fields["planner"][i],
                }
                for i in run
            ]
            matched_s, aligned_d, same_d, argmin = _viterbi_phase_path(
                steps.astype(np.int32), lives, teacher, params
            )
            all_matched.append(matched_s)
            all_argmin_offsets.append(argmin)
            all_aligned_distance.append(aligned_d)
            all_same_distance.append(same_d)
            all_runs.append(run)

            offset = matched_s - steps
            delta_phase = np.diff(matched_s)
            per_segment.append(
                {
                    "source_episode_id": episode_id,
                    "frames": int(len(run)),
                    "step_range": [int(steps[0]), int(steps[-1])],
                    "offsets": _summarize_offsets(offset),
                    "backward_transitions": int(np.sum(delta_phase < 0)),
                    "max_abs_delta_phase": int(np.max(np.abs(delta_phase)))
                    if len(delta_phase)
                    else 0,
                    "same_distance_mean": float(np.mean(same_d)),
                    "aligned_distance_mean": float(np.mean(aligned_d)),
                    "improved_fraction": float(np.mean(aligned_d < same_d)),
                    "viterbi_vs_argmin_abs_mean": float(
                        np.mean(np.abs(matched_s - steps - argmin))
                    ),
                }
            )
            print(
                f"[PHASE-ALIGN] ep {episode_id}: n={len(run)} "
                f"offset med={np.median(offset):+.1f} "
                f"|offset|>10 {np.mean(np.abs(offset) > 10):.1%} "
                f"backward={int(np.sum(delta_phase < 0))} "
                f"same_d {np.mean(same_d):.3f} -> aligned_d {np.mean(aligned_d):.3f}",
                flush=True,
            )

    # ---- build phase-aligned single-frame labels ----
    # The DAgger action is a per-frame teacher q sequence; the training
    # window (dp_dataset.FingertipDiffusionDataset) constructs the future
    # pred_horizon targets from the following frames of this sequence, so a
    # smooth phase path yields phase-consistent future labels automatically.
    with h5py.File(args.dagger, "r") as src:
        new_action = np.asarray(src["action_q_hand"], dtype=np.float32).copy()
    fallback_count = 0
    source_teacher_step = np.zeros(n_total, dtype=np.int32)
    future_frames = args.label_horizon * args.label_stride_frames
    for (episode_id, run, steps), matched_s in zip(segments, all_matched, strict=True):
        teacher_q = teacher_q_by_episode[episode_id]
        for j, (row, matched_step) in enumerate(zip(run, matched_s, strict=True)):
            source_teacher_step[row] = int(matched_step)
            end = int(matched_step) + future_frames
            if int(matched_step) < 0 or end > len(teacher_q):
                fallback_count += 1
                continue  # keep the original same-time label
            new_action[row, 0, :] = teacher_q[int(matched_step)]

    # ---- write drop-in compatible output ----
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.dagger, "r") as src, h5py.File(args.output, "w") as dst:
        for key in src.keys():
            src.copy(key, dst)
        for key, value in src.attrs.items():
            dst.attrs[key] = value
        dst["action_q_hand"][:] = new_action
        dst.create_dataset(
            "source_teacher_step", data=source_teacher_step.reshape(-1, 1), dtype="i4"
        )
        dst.attrs["dagger_phase_aligned"] = True
        dst.attrs["dagger_phase_alignment_params"] = json.dumps(
            {
                "method": "per_segment_constrained_viterbi_full_distance",
                "search_radius_frames": params.radius,
                "lambda_speed": params.lambda_speed,
                "lambda_backward": params.lambda_backward,
                "lambda_jump": params.lambda_jump,
                "jump_cap": params.jump_cap,
                "lambda_time": params.lambda_time,
                "label_stride_frames": args.label_stride_frames,
                "label_horizon": args.label_horizon,
            }
        )
        dst.attrs["dagger_phase_alignment_fallback_count"] = int(fallback_count)

    # ---- report ----
    all_offsets = np.concatenate(
        [matched - steps for (_, _, steps), matched in zip(segments, all_matched)]
    )
    all_delta = np.concatenate([np.diff(matched) for matched in all_matched])
    same_d_all = np.concatenate(all_same_distance)
    aligned_d_all = np.concatenate(all_aligned_distance)
    conflict = np.abs(all_offsets) > 10
    if np.any(conflict):
        conflict_same_mean = float(np.mean(same_d_all[conflict]))
        conflict_aligned_mean = float(np.mean(aligned_d_all[conflict]))
        conflict_improved_fraction = float(
            np.mean(aligned_d_all[conflict] < same_d_all[conflict])
        )
    else:
        conflict_same_mean = None
        conflict_aligned_mean = None
        conflict_improved_fraction = None
    report = {
        "dagger_file": str(args.dagger),
        "teacher_file": str(args.teacher),
        "output_file": str(args.output),
        "frames": n_total,
        "segments": len(segments),
        "offsets": _summarize_offsets(all_offsets),
        "path": {
            "backward_transitions": int(np.sum(all_delta < 0)),
            "max_abs_delta_phase": int(np.max(np.abs(all_delta))),
            "mean_abs_delta_phase": float(np.mean(np.abs(all_delta))),
        },
        "distance": {
            "same_mean": float(np.mean(same_d_all)),
            "aligned_mean": float(np.mean(aligned_d_all)),
            "improved_fraction": float(np.mean(aligned_d_all < same_d_all)),
            "conflict_same_mean": conflict_same_mean,
            "conflict_aligned_mean": conflict_aligned_mean,
            "conflict_improved_fraction": conflict_improved_fraction,
        },
        "fallback_same_time_labels": fallback_count,
        "per_segment": per_segment,
    }
    report_path = args.output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[PHASE-ALIGN] report -> {report_path}", flush=True)
    print(
        f"[PHASE-ALIGN] |offset|>10 {report['offsets']['fraction_abs_gt_10']:.1%} "
        f"backward {report['path']['backward_transitions']} "
        f"same_d {report['distance']['same_mean']:.3f} -> "
        f"aligned_d {report['distance']['aligned_mean']:.3f} "
        f"(conflict: {report['distance']['conflict_same_mean']} -> "
        f"{report['distance']['conflict_aligned_mean']})",
        flush=True,
    )


if __name__ == "__main__":
    main()
