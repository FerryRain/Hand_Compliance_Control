"""Build local-stability data from causally aligned closed-loop DP rollouts.

The default observation is the nominal state actually fed back to DP, not the
MCC-compensated live joint state.  The action is always the time-indexed
successful teacher q.  This targets nominal autoregressive exposure bias
without asking DP to reproduce or cancel low-level MCC compensation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from dp_dataset import ACTION_DIM, state_fields, state_schema
from surface_mcc_finger import (
    FullHandMCCFingerConfig,
    FullHandMCCFingerController,
)


IDENTITY_PALM_POSE = np.asarray(
    (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0), dtype=np.float64
)


def _close_short_false_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    """Fill only bounded false runs; leading/trailing invalid data stay invalid."""

    result = np.asarray(mask, dtype=bool).copy()
    if max_gap <= 0:
        return result
    index = 0
    while index < len(result):
        if result[index]:
            index += 1
            continue
        stop = index + 1
        while stop < len(result) and not result[stop]:
            stop += 1
        if index > 0 and stop < len(result) and stop - index <= max_gap:
            result[index:stop] = True
        index = stop
    return result


def _true_runs(mask: np.ndarray, minimum_length: int) -> list[tuple[int, int]]:
    padded = np.pad(np.asarray(mask, dtype=np.int8), (1, 1))
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    return [
        (int(start), int(stop))
        for start, stop in zip(starts, stops, strict=True)
        if stop - start >= minimum_length
    ]


def _controller_health(
    controller: FullHandMCCFingerController,
    q_live: np.ndarray,
    contact_mask: np.ndarray,
    contact_pos_palm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return pad-valid count, synergy maximum and normalized joint margin."""

    frame_count = len(q_live)
    pad_count = np.zeros(frame_count, dtype=np.int16)
    synergy_max = np.zeros(frame_count, dtype=np.float32)
    joint_margin = np.zeros(frame_count, dtype=np.float32)
    travel = np.maximum(controller.upper - controller.lower, 1.0e-8)
    normalized = (q_live - controller.lower[None, :]) / travel[None, :]
    joint_margin[:] = np.min(
        np.minimum(normalized, 1.0 - normalized), axis=1
    ).astype(np.float32)
    for frame in range(frame_count):
        found = np.asarray(contact_mask[frame], dtype=bool)
        valid, _ = controller.pad_contact_validity(
            q_live[frame],
            IDENTITY_PALM_POSE,
            contact_pos_palm[frame],
            found,
        )
        pad_count[frame] = int(np.sum(valid))
        synergy_max[frame] = float(
            np.max(controller.flexion_synergy_metrics(q_live[frame])[0])
        )
    return pad_count, synergy_max, joint_margin


def build_dataset(args: argparse.Namespace) -> dict[str, object]:
    fields = state_fields(args.reference_dp)
    schema = state_schema(args.reference_dp)
    if schema != "contact_geometry_planner_motion":
        raise ValueError(
            "The pilot DAgger builder currently requires the 96-D explicit-motion schema"
        )
    expected_state_dim = sum(size for _, size in fields)
    controller = FullHandMCCFingerController(FullHandMCCFingerConfig())

    episodes: list[dict[str, np.ndarray | int | str]] = []
    rollout_stats: list[dict[str, object]] = []
    next_episode_id = int(args.episode_id_offset)
    required_frames = max(
        int(args.minimum_segment_frames),
        (int(args.obs_horizon) + int(args.pred_horizon)) * int(args.stride),
    )

    for rollout_path in args.rollout:
        with h5py.File(rollout_path, "r") as file:
            state_key = (
                "dp_observation_state"
                if args.observation_state_source == "dp"
                else "live_dp_state"
            )
            if state_key not in file:
                raise ValueError(
                    f"{rollout_path}: missing {state_key!r}; recollect this "
                    "rollout with the current deploy_dp_inverse.py"
                )
            state = np.asarray(file[state_key], dtype=np.float32)
            q_live = np.asarray(file["q_live"], dtype=np.float32)
            teacher_q = np.asarray(file["teacher_q_hand"], dtype=np.float32)
            contact_mask = np.asarray(
                file["fingertip_contact_mask"], dtype=np.float32
            )
            contact_pos = np.asarray(
                file["fingertip_contact_pos_palm"], dtype=np.float32
            )
            force = np.linalg.norm(
                np.asarray(file["fingertip_force_palm"], dtype=np.float32),
                axis=-1,
            )
            source_steps = np.asarray(file["episode_step"], dtype=np.int32).reshape(-1)
            source_episode = int(file.attrs["source_episode_id"])
            bootstrap = int(file.attrs.get("bootstrap_frames", 0))
            rollout_schema = str(file.attrs.get("state_schema", ""))
            rollout_q_source = str(
                file.attrs.get("dp_history_q_source", "unknown")
            )
        if state.ndim != 2 or state.shape[1] != expected_state_dim:
            raise ValueError(
                f"{rollout_path}: expected {state_key} (*,{expected_state_dim}), "
                f"got {state.shape}"
            )
        if rollout_schema != schema:
            raise ValueError(
                f"{rollout_path}: state_schema={rollout_schema!r}, expected {schema!r}"
            )
        if not (
            len(state)
            == len(q_live)
            == len(teacher_q)
            == len(contact_mask)
            == len(contact_pos)
        ):
            raise ValueError(f"{rollout_path}: time-axis length mismatch")
        if (
            args.observation_state_source == "dp"
            and args.require_dp_history_q_source != "any"
            and rollout_q_source != args.require_dp_history_q_source
        ):
            raise ValueError(
                f"{rollout_path}: dp_history_q_source={rollout_q_source!r}, "
                f"expected {args.require_dp_history_q_source!r}"
            )

        observation_q = state[:, :ACTION_DIM]
        if args.observation_state_source == "live":
            q_state_error = float(np.max(np.abs(observation_q - q_live)))
            if q_state_error > 2.0e-5:
                raise ValueError(
                    f"{rollout_path}: live state q and q_live disagree by "
                    f"{q_state_error:g}"
                )

        # Default: nominal-to-teacher error. q_live is used only for physical
        # safety and hand-health gates below.
        q_mae = np.mean(np.abs(observation_q - teacher_q), axis=1)
        pad_count, synergy_max, joint_margin = _controller_health(
            controller, q_live, contact_mask, contact_pos
        )
        valid = np.arange(len(state)) >= bootstrap
        valid &= q_mae <= float(args.max_teacher_q_mae_rad)
        valid &= pad_count >= int(args.min_valid_pad_contacts)
        valid &= np.max(force, axis=1) <= float(args.max_force_n)
        valid &= synergy_max <= float(args.max_synergy_spread)
        valid &= joint_margin >= float(args.min_normalized_joint_margin)
        valid = _close_short_false_gaps(valid, int(args.max_invalid_gap_frames))
        runs = _true_runs(valid, required_frames)

        aligned_runs: list[tuple[int, int]] = []
        for start, stop in runs:
            # dp_observation_state is updated on this grid and held between
            # samples. Align the segment so load_episodes(...)[::stride]
            # always selects real DP history states rather than held frames.
            relative = np.flatnonzero(
                source_steps[start:stop] % int(args.stride) == 0
            )
            if not len(relative):
                continue
            aligned_start = start + int(relative[0])
            if stop - aligned_start >= required_frames:
                aligned_runs.append((aligned_start, stop))

        for run_index, (start, stop) in enumerate(aligned_runs):
            episodes.append(
                {
                    "episode_id": next_episode_id,
                    "state": state[start:stop].copy(),
                    "teacher_q": teacher_q[start:stop].copy(),
                    "source_step": source_steps[start:stop].copy(),
                    "source_episode": source_episode,
                    "source_rollout": str(rollout_path),
                }
            )
            next_episode_id += 1
        rollout_stats.append(
            {
                "rollout": str(rollout_path),
                "source_episode": source_episode,
                "observation_state_source": args.observation_state_source,
                "dp_history_q_source": rollout_q_source,
                "frames": int(len(state)),
                "valid_frames": int(np.sum(valid)),
                "valid_fraction": float(np.mean(valid)),
                "segments": len(aligned_runs),
                "segment_lengths": [stop - start for start, stop in aligned_runs],
                "q_mae_p50_rad": float(np.median(q_mae[valid])) if np.any(valid) else None,
                "q_mae_p95_rad": float(np.percentile(q_mae[valid], 95)) if np.any(valid) else None,
                "pad4_fraction": float(np.mean(pad_count == 4)),
                "max_synergy_spread": float(np.max(synergy_max)),
                "min_joint_margin": float(np.min(joint_margin)),
            }
        )

    if not episodes:
        raise RuntimeError(
            "No causally usable DAgger segment passed the configured gates; "
            "inspect the generated JSON report thresholds"
        )

    total = sum(len(np.asarray(episode["state"])) for episode in episodes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.reference_dp, "r") as reference, h5py.File(
        args.output, "w"
    ) as output:
        for key, value in reference.attrs.items():
            output.attrs[key] = value
        output.attrs["action_field"] = "action_q_hand"
        output.attrs["observation_q_field"] = "q_hand"
        output.attrs["dagger_dataset"] = True
        # Local deviations should remain inside the clean action scale. Do
        # not silently turn a distant one-shot return into a clipped label.
        output.attrs["bound_normalized_action"] = False
        output.attrs["dagger_observation_state_source"] = (
            args.observation_state_source
        )
        output.attrs["dagger_required_dp_history_q_source"] = (
            args.require_dp_history_q_source
        )
        output.attrs["dagger_label_contract"] = (
            "observation is the causal state actually fed to DP; nominal q "
            "is isolated from MCC compensation; action_q_hand is the "
            "time-indexed successful teacher q"
        )
        output.attrs["dagger_rollout_count"] = len(args.rollout)
        output.attrs["dagger_segment_count"] = len(episodes)
        output.attrs["dagger_gate_json"] = json.dumps(
            {
                "max_teacher_q_mae_rad": args.max_teacher_q_mae_rad,
                "min_valid_pad_contacts": args.min_valid_pad_contacts,
                "max_force_n": args.max_force_n,
                "max_synergy_spread": args.max_synergy_spread,
                "min_normalized_joint_margin": args.min_normalized_joint_margin,
                "max_invalid_gap_frames": args.max_invalid_gap_frames,
                "required_frames": required_frames,
            },
            sort_keys=True,
        )

        flat_state = np.concatenate(
            [np.asarray(episode["state"]) for episode in episodes], axis=0
        )
        cursor = 0
        for name, size in fields:
            tail_shape = tuple(reference[name].shape[2:])
            values = flat_state[:, cursor : cursor + size].reshape(
                total, 1, *tail_shape
            )
            output.create_dataset(
                name, data=values.astype(np.float32), compression="gzip"
            )
            cursor += size
        output.create_dataset(
            "action_q_hand",
            data=np.concatenate(
                [
                    np.asarray(episode["teacher_q"], dtype=np.float32)[:, None, :]
                    for episode in episodes
                ],
                axis=0,
            ),
            compression="gzip",
        )
        output.create_dataset(
            "episode_id",
            data=np.concatenate(
                [
                    np.full((len(np.asarray(episode["state"])), 1), int(episode["episode_id"]), np.int32)
                    for episode in episodes
                ],
                axis=0,
            ),
        )
        output.create_dataset(
            "episode_step",
            data=np.concatenate(
                [
                    np.arange(len(np.asarray(episode["state"])), dtype=np.int32)[:, None]
                    for episode in episodes
                ],
                axis=0,
            ),
        )
        output.create_dataset(
            "source_episode_id",
            data=np.concatenate(
                [
                    np.full((len(np.asarray(episode["state"])), 1), int(episode["source_episode"]), np.int32)
                    for episode in episodes
                ],
                axis=0,
            ),
        )
        output.create_dataset(
            "source_episode_step",
            data=np.concatenate(
                [np.asarray(episode["source_step"], dtype=np.int32)[:, None] for episode in episodes],
                axis=0,
            ),
        )

    report = {
        "output": str(args.output),
        "reference_dp": str(args.reference_dp),
        "state_schema": schema,
        "state_dim": expected_state_dim,
        "observation_state_source": args.observation_state_source,
        "required_dp_history_q_source": args.require_dp_history_q_source,
        "segments": len(episodes),
        "frames": total,
        "required_segment_frames": required_frames,
        "rollouts": rollout_stats,
    }
    report_path = args.report or args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dp", type=Path, required=True)
    parser.add_argument("--rollout", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--obs-horizon", type=int, default=16)
    parser.add_argument("--pred-horizon", type=int, default=8)
    parser.add_argument("--minimum-segment-frames", type=int, default=120)
    parser.add_argument("--episode-id-offset", type=int, default=1_000_000)
    parser.add_argument(
        "--observation-state-source",
        choices=("dp", "live"),
        default="dp",
        help=(
            "Use the exact state fed to DP (default) or the raw live state. "
            "Nominal-stability DAgger should use dp."
        ),
    )
    parser.add_argument(
        "--require-dp-history-q-source",
        choices=("nominal", "live", "any"),
        default="nominal",
        help="Reject rollouts collected with the wrong DP feedback contract.",
    )
    parser.add_argument("--max-teacher-q-mae-rad", type=float, default=0.015)
    parser.add_argument("--min-valid-pad-contacts", type=int, default=3)
    parser.add_argument("--max-force-n", type=float, default=12.0)
    parser.add_argument("--max-synergy-spread", type=float, default=0.60)
    parser.add_argument("--min-normalized-joint-margin", type=float, default=0.10)
    parser.add_argument("--max-invalid-gap-frames", type=int, default=3)
    args = parser.parse_args()
    if args.stride <= 0 or args.obs_horizon <= 0 or args.pred_horizon <= 0:
        raise ValueError("stride/obs-horizon/pred-horizon must be positive")
    report = build_dataset(args)
    print(
        f"[DAGGER] wrote {report['segments']} segments / {report['frames']} frames "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
