"""Analyze contact quality in raw multi-object trajectory H5 files.

Quality is evaluated from both collision state and the 3-D fingertip force
threshold.  Family-specific targets come from the resolved object YAML:

* minimum fraction of frames with at least three loaded fingertips;
* minimum fraction of frames with all four loaded fingertips;
* maximum consecutive run with fewer than three loaded fingertips;
* maximum recorded fingertip force.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from object_catalog import get_motion_config, load_object_config


TIP_LABELS = ("IF", "MF", "RF", "TH")


def _flatten_env(data: np.ndarray) -> np.ndarray:
    return data.reshape(data.shape[0] * data.shape[1], *data.shape[2:])


def _max_true_run(mask: np.ndarray) -> int:
    best = current = 0
    for value in mask:
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


@dataclass(frozen=True)
class EpisodeQuality:
    object_id: str
    episode_id: int
    frames: int
    contact3_ratio: float
    contact4_ratio: float
    initial_contact3_ratio: float
    first_below3_frame: int
    below3_max_run: int
    below4_max_run: int
    tip_ratios: tuple[float, float, float, float]
    force_p95_N: float
    force_max_N: float
    angular_speed_mean: float
    rotation_axis_label: str
    rotation_speed_target: float
    passed: bool


def _quality_limits(object_id: str) -> dict[str, float]:
    config = load_object_config(object_id)
    raw = config.resolved.get("collection", {}).get("quality", {})
    return {
        "min3": float(raw.get("min_three_finger_ratio", 0.85)),
        "min4": float(raw.get("min_four_finger_ratio", 0.45)),
        "max_loss": int(raw.get("max_contact_loss_frames", 25)),
        "max_force": float(raw.get("max_force_N", np.inf)),
    }


def _read_h5(
    path: Path,
    threshold_override: float | None,
    enforce_force_limit: bool,
) -> list[EpisodeQuality]:
    with h5py.File(path, "r") as source:
        object_id = str(source.attrs.get("object_id", ""))
        if not object_id:
            raise ValueError(f"{path} has no object_id attribute")
        threshold = float(
            threshold_override
            if threshold_override is not None
            else source.attrs.get("contact_threshold", 0.05)
        )
        episode_id = _flatten_env(np.asarray(source["episode_id"])).astype(np.int64)
        force = _flatten_env(np.asarray(source["fingertip_force_world"]))
        force_norm = np.linalg.norm(force, axis=-1)
        if "fingertip_collision_found" in source:
            found = _flatten_env(
                np.asarray(source["fingertip_collision_found"])
            ) > 0.5
        else:
            found = force_norm > 0.0
        loaded = found & (force_norm >= threshold)
        if "object_angular_velocity_world" in source:
            angular_speed = np.linalg.norm(
                _flatten_env(np.asarray(source["object_angular_velocity_world"])),
                axis=-1,
            )
        else:
            angular_speed = np.zeros_like(episode_id, dtype=np.float32)
        rotation_axis = (
            _flatten_env(np.asarray(source["object_rotation_axis_local"]))
            if "object_rotation_axis_local" in source
            else np.zeros((episode_id.size, 3), dtype=np.float32)
        )
        rotation_speed_target = (
            _flatten_env(
                np.asarray(source["object_rotation_speed_target_rad_s"])
            )
            if "object_rotation_speed_target_rad_s" in source
            else angular_speed
        )

    limits = _quality_limits(object_id)
    rows: list[EpisodeQuality] = []
    for eid in np.unique(episode_id):
        mask = episode_id == eid
        episode_loaded = loaded[mask]
        count = episode_loaded.sum(axis=1)
        contact3 = float(np.mean(count >= 3))
        contact4 = float(np.mean(count == 4))
        initial_contact3 = float(np.mean(count[: min(100, count.size)] >= 3))
        below3_indices = np.flatnonzero(count < 3)
        below3_run = _max_true_run(count < 3)
        below4_run = _max_true_run(count < 4)
        tip_ratios = tuple(float(value) for value in episode_loaded.mean(axis=0))
        episode_force = force_norm[mask]
        moving_speed = angular_speed[mask]
        moving_speed = moving_speed[moving_speed > 1.0e-8]
        force_p95 = float(np.percentile(episode_force, 95.0))
        force_max = float(np.max(episode_force))
        axis = rotation_axis[mask][0]
        principal_index = int(np.argmax(np.abs(axis)))
        if np.isclose(abs(axis[principal_index]), 1.0, atol=1.0e-4) and np.count_nonzero(
            np.abs(axis) > 1.0e-4
        ) == 1:
            sign = "+" if axis[principal_index] >= 0.0 else "-"
            axis_label = f"{sign}{'xyz'[principal_index]}"
        else:
            axis_label = "uniform"
        passed = bool(
            contact3 >= limits["min3"]
            and contact4 >= limits["min4"]
            and below3_run <= limits["max_loss"]
            and (not enforce_force_limit or force_max <= limits["max_force"])
        )
        rows.append(
            EpisodeQuality(
                object_id=object_id,
                episode_id=int(eid),
                frames=int(mask.sum()),
                contact3_ratio=contact3,
                contact4_ratio=contact4,
                initial_contact3_ratio=initial_contact3,
                first_below3_frame=(
                    int(below3_indices[0]) if below3_indices.size else -1
                ),
                below3_max_run=below3_run,
                below4_max_run=below4_run,
                tip_ratios=tip_ratios,
                force_p95_N=force_p95,
                force_max_N=force_max,
                angular_speed_mean=(
                    float(np.mean(moving_speed)) if moving_speed.size else 0.0
                ),
                rotation_axis_label=axis_label,
                rotation_speed_target=float(rotation_speed_target[mask][0]),
                passed=passed,
            )
        )
    return rows


def _write_episode_report(path: Path, rows: list[EpisodeQuality]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "object_id", "episode_id", "frames", "passed", "contact3_ratio",
        "contact4_ratio", "initial_contact3_ratio", "first_below3_frame",
        "below3_max_run", "below4_max_run", "tip0_ratio",
        "tip1_ratio", "tip2_ratio", "tip3_ratio", "force_p95_N",
        "force_max_N", "angular_speed_mean",
        "rotation_axis_label", "rotation_speed_target",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "object_id": row.object_id,
                    "episode_id": row.episode_id,
                    "frames": row.frames,
                    "passed": int(row.passed),
                    "contact3_ratio": row.contact3_ratio,
                    "contact4_ratio": row.contact4_ratio,
                    "initial_contact3_ratio": row.initial_contact3_ratio,
                    "first_below3_frame": row.first_below3_frame,
                    "below3_max_run": row.below3_max_run,
                    "below4_max_run": row.below4_max_run,
                    **{
                        f"tip{finger}_ratio": row.tip_ratios[finger]
                        for finger in range(4)
                    },
                    "force_p95_N": row.force_p95_N,
                    "force_max_N": row.force_max_N,
                    "angular_speed_mean": row.angular_speed_mean,
                    "rotation_axis_label": row.rotation_axis_label,
                    "rotation_speed_target": row.rotation_speed_target,
                }
            )


def _print_summary(
    rows: list[EpisodeQuality], *, enforce_force_limit: bool
) -> None:
    grouped: dict[str, list[EpisodeQuality]] = {}
    for row in rows:
        grouped.setdefault(row.object_id, []).append(row)

    header = (
        f"{'Object':<22} {'Pass':>7} {'C3 med':>8} {'C3 min':>8} "
        f"{'C4 med':>8} {'C4 min':>8} {'loss3':>7} "
        + " ".join(f"{tip:>7}" for tip in TIP_LABELS)
        + f" {'F95':>7} {'Fmax':>7}"
    )
    print(header)
    print("-" * len(header))
    for object_id, object_rows in grouped.items():
        contact3 = np.asarray([row.contact3_ratio for row in object_rows])
        contact4 = np.asarray([row.contact4_ratio for row in object_rows])
        tip = np.asarray([row.tip_ratios for row in object_rows])
        passed = sum(row.passed for row in object_rows)
        print(
            f"{object_id:<22} {passed:>2}/{len(object_rows):<4} "
            f"{np.median(contact3):>7.1%} {contact3.min():>7.1%} "
            f"{np.median(contact4):>7.1%} {contact4.min():>7.1%} "
            f"{max(row.below3_max_run for row in object_rows):>7d} "
            + " ".join(f"{value:>6.1%}" for value in np.median(tip, axis=0))
            + f" {np.median([r.force_p95_N for r in object_rows]):>6.2f} "
            f"{max(r.force_max_N for r in object_rows):>6.2f}"
        )

    print("\nPer-object diagnosis")
    for object_id, object_rows in grouped.items():
        limits = _quality_limits(object_id)
        config = load_object_config(object_id)
        motion = get_motion_config(config)
        contact3 = np.asarray([row.contact3_ratio for row in object_rows])
        contact4 = np.asarray([row.contact4_ratio for row in object_rows])
        tip = np.asarray([row.tip_ratios for row in object_rows])
        peak = max(row.force_max_N for row in object_rows)
        passed = sum(row.passed for row in object_rows)
        problems: list[str] = []
        diagnostics: list[str] = []
        if np.median(contact3) < limits["min3"]:
            problems.append(
                f"median C3 {np.median(contact3):.1%} < {limits['min3']:.1%}"
            )
        if np.median(contact4) < limits["min4"]:
            problems.append(
                f"median C4 {np.median(contact4):.1%} < {limits['min4']:.1%}"
            )
        loss3 = max(row.below3_max_run for row in object_rows)
        if loss3 > limits["max_loss"]:
            problems.append(f"<3-tip loss run {loss3} > {limits['max_loss']} frames")
        if peak > limits["max_force"]:
            qualifier = "rejects trajectory" if enforce_force_limit else "diagnostic only"
            diagnostics.append(
                f"peak force {peak:.2f} > {limits['max_force']:.2f} N ({qualifier})"
            )
        weakest = int(np.argmin(np.median(tip, axis=0)))
        if np.median(tip[:, weakest]) < limits["min3"]:
            problems.append(
                f"weakest {TIP_LABELS[weakest]} median={np.median(tip[:, weakest]):.1%}"
            )
        status = "PASS" if passed == len(object_rows) else "TUNE"
        print(f"  {object_id}: {status} ({passed}/{len(object_rows)})")
        for problem in problems:
            print(f"    - {problem}")
        for diagnostic in diagnostics:
            print(f"    - {diagnostic}")
        if problems:
            bad_initial = [
                row for row in object_rows
                if row.initial_contact3_ratio < limits["min3"]
            ]
            if bad_initial:
                detail = ", ".join(
                    f"ep{row.episode_id}/{row.rotation_axis_label}"
                    for row in bad_initial
                )
                print(
                    "    - initial contact is already invalid in "
                    f"{detail}; adjust/resample the initial object pose before "
                    "changing motion speed"
                )
            if len(bad_initial) < len(object_rows):
                speed = motion["rotation"]["angular_speed_range_rad_s"]
                print(
                    f"    - trajectories that start in contact but lose it use "
                    f"rotation speed={speed} rad/s; retest those pose/axis pairs "
                    "with a shorter path or lower upper speed"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Raw trajectory H5 files")
    parser.add_argument("--contact-threshold", type=float, default=None)
    parser.add_argument(
        "--enforce-force-limit",
        action="store_true",
        help="Also reject episodes whose peak force exceeds the family limit.",
    )
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    rows: list[EpisodeQuality] = []
    for path in args.inputs:
        object_rows = _read_h5(
            path, args.contact_threshold, args.enforce_force_limit
        )
        rows.extend(object_rows)
        print(f"[QUALITY] {path.name}: {len(object_rows)} episodes")
    _print_summary(rows, enforce_force_limit=args.enforce_force_limit)
    if args.report is not None:
        _write_episode_report(args.report, rows)
        print(f"[QUALITY] report: {args.report}")


if __name__ == "__main__":
    main()
