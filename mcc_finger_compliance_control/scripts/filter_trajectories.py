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
    args = parser.parse_args()

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
            passed = bool(all_four.all())
            if passed:
                accepted_ids.append(int(eid))
            rows.append(
                {
                    "episode_id": int(eid),
                    "strict_pass": int(passed),
                    "frames": int(mask.sum()),
                    "all4_ratio": float(all_four.mean()),
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
            dst.attrs["quality_filter"] = "strict_continuous_four_fingertips_v1"
            dst.attrs["quality_contact_threshold"] = threshold
            dst.attrs["quality_source_file"] = args.input.name
            dst.attrs["quality_total_trajectories"] = len(rows)
            dst.attrs["quality_selected_trajectories"] = len(accepted_ids)
            dst.attrs["num_trajectories"] = len(accepted_ids)
            dst.attrs["strict_four_tip_continuous_contact"] = True

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
        f"[FILTER] strict pass={len(accepted_ids)}/{len(rows)} "
        f"threshold={threshold:.3f} N"
    )
    print(f"[FILTER] output: {output}")
    print(f"[FILTER] report: {report}")


if __name__ == "__main__":
    main()
