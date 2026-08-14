"""Offline strict filtering for parallel MCC fingertip trajectory H5 files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np


def _flatten_env(data: np.ndarray) -> np.ndarray:
    """Flatten [batch_time, env, ...] while preserving complete episode IDs."""
    return data.reshape(data.shape[0] * data.shape[1], *data.shape[2:])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--contact-threshold", type=float, default=None)
    parser.add_argument(
        "--min-all4-ratio",
        type=float,
        default=1.0,
        help="Minimum fraction of frames where all four fingertips are loaded.",
    )
    parser.add_argument(
        "--min-per-tip-ratio",
        type=float,
        default=1.0,
        help="Minimum loaded-contact ratio required for every fingertip.",
    )
    parser.add_argument(
        "--max-loss-run",
        type=int,
        default=0,
        help="Maximum consecutive frames without simultaneous four-tip contact.",
    )
    args = parser.parse_args()
    if not 0.0 <= args.min_all4_ratio <= 1.0:
        raise ValueError("--min-all4-ratio must lie in [0, 1]")
    if not 0.0 <= args.min_per_tip_ratio <= 1.0:
        raise ValueError("--min-per-tip-ratio must lie in [0, 1]")
    if args.max_loss_run < 0:
        raise ValueError("--max-loss-run must be non-negative")

    output = args.output or args.input.with_name(f"{args.input.stem}_strict4tip.h5")
    report = args.report or args.input.with_name(
        f"{args.input.stem}_strict4tip_report.csv"
    )

    with h5py.File(args.input, "r") as src:
        threshold = float(
            args.contact_threshold
            if args.contact_threshold is not None
            else src.attrs.get("contact_threshold", 0.05)
        )
        episode_id = _flatten_env(np.asarray(src["episode_id"])).astype(np.int64)
        episode_step = _flatten_env(np.asarray(src["episode_step"])).astype(np.int64)
        force = _flatten_env(np.asarray(src["fingertip_force_world"]))
        force_norm = np.linalg.norm(force, axis=-1)
        if "fingertip_collision_found" in src:
            found = _flatten_env(
                np.asarray(src["fingertip_collision_found"])
            ) > 0.5
        else:
            found = force_norm > 0.0
        loaded_contact = found & (force_norm >= threshold)

        rows: list[dict[str, object]] = []
        accepted_ids: list[int] = []
        for eid in np.unique(episode_id):
            mask = episode_id == eid
            per_tip = loaded_contact[mask].mean(axis=0)
            all_four = loaded_contact[mask].all(axis=1)
            lost = np.flatnonzero(~all_four)
            loss_runs: list[int] = []
            current_run = 0
            for is_lost in ~all_four:
                if is_lost:
                    current_run += 1
                elif current_run:
                    loss_runs.append(current_run)
                    current_run = 0
            if current_run:
                loss_runs.append(current_run)
            max_loss_run = max(loss_runs, default=0)
            passed = bool(
                all_four.mean() >= args.min_all4_ratio
                and per_tip.min() >= args.min_per_tip_ratio
                and max_loss_run <= args.max_loss_run
            )
            if passed:
                accepted_ids.append(int(eid))
            rows.append(
                {
                    "episode_id": int(eid),
                    "strict_pass": int(passed),
                    "frames": int(mask.sum()),
                    "all4_ratio": float(all_four.mean()),
                    "max_loss_run": int(max_loss_run),
                    "loss_events": int(len(loss_runs)),
                    "tip0_ratio": float(per_tip[0]),
                    "tip1_ratio": float(per_tip[1]),
                    "tip2_ratio": float(per_tip[2]),
                    "tip3_ratio": float(per_tip[3]),
                    "tip0_min_force_N": float(force_norm[mask, 0].min()),
                    "tip1_min_force_N": float(force_norm[mask, 1].min()),
                    "tip2_min_force_N": float(force_norm[mask, 2].min()),
                    "tip3_min_force_N": float(force_norm[mask, 3].min()),
                    "first_loss_step": (
                        int(episode_step[mask][lost[0]]) if lost.size else -1
                    ),
                }
            )

        report.parent.mkdir(parents=True, exist_ok=True)
        with report.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        keep = np.isin(episode_id, np.asarray(accepted_ids, dtype=np.int64))
        output.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(output, "w") as dst:
            for key, value in src.attrs.items():
                dst.attrs[key] = value
            strict = (
                args.min_all4_ratio == 1.0
                and args.min_per_tip_ratio == 1.0
                and args.max_loss_run == 0
            )
            dst.attrs["quality_filter"] = (
                "strict_continuous_four_fingertips_v2"
                if strict
                else "relaxed_four_fingertips_v2"
            )
            dst.attrs["quality_contact_threshold"] = threshold
            dst.attrs["quality_min_all4_ratio"] = args.min_all4_ratio
            dst.attrs["quality_min_per_tip_ratio"] = args.min_per_tip_ratio
            dst.attrs["quality_max_loss_run"] = args.max_loss_run
            dst.attrs["quality_source_file"] = args.input.name
            dst.attrs["quality_total_trajectories"] = len(rows)
            dst.attrs["quality_selected_trajectories"] = len(accepted_ids)
            dst.attrs["num_trajectories"] = len(accepted_ids)
            dst.attrs["strict_four_tip_continuous_contact"] = strict

            for name, source in src.items():
                flat = _flatten_env(np.asarray(source))
                selected = flat[keep, None, ...]
                dataset_kwargs: dict[str, object] = {}
                if selected.shape[0] > 0:
                    dataset_kwargs["chunks"] = (
                        min(100, selected.shape[0]),
                        1,
                        *selected.shape[2:],
                    )
                dst.create_dataset(
                    name, data=selected, dtype=source.dtype, **dataset_kwargs
                )

    print(
        f"[FILTER] pass={len(accepted_ids)}/{len(rows)} "
        f"threshold={threshold:.3f}N all4>={args.min_all4_ratio:.4f} "
        f"per_tip>={args.min_per_tip_ratio:.4f} max_loss_run<={args.max_loss_run}"
    )
    print(f"[FILTER] output: {output}")
    print(f"[FILTER] report: {report}")


if __name__ == "__main__":
    main()
