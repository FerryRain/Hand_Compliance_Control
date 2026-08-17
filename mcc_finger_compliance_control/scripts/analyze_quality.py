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
    source_file: str
    object_id: str
    motion_mode: str
    episode_id: int
    frames: int
    contact3_ratio: float
    contact4_ratio: float
    initial_contact3_ratio: float
    initial_contact4_ratio: float
    first_below3_frame: int
    below3_max_run: int
    below4_max_run: int
    tip_ratios: tuple[float, float, float, float]
    force_p95_N: float
    force_max_N: float
    angular_speed_mean: float
    rotation_axis_label: str
    rotation_speed_target: float
    translation_axis_label: str
    translation_amplitude_m: float
    passed: bool


def _axis_label(axis: np.ndarray, *, inactive: bool = False) -> str:
    if inactive or np.linalg.norm(axis) < 1.0e-8:
        return "off"
    principal_index = int(np.argmax(np.abs(axis)))
    if np.isclose(abs(axis[principal_index]), 1.0, atol=1.0e-4) and np.count_nonzero(
        np.abs(axis) > 1.0e-4
    ) == 1:
        sign = "+" if axis[principal_index] >= 0.0 else "-"
        return f"{sign}{'xyz'[principal_index]}"
    return "uniform"


def _quality_limits(
    object_id: str, quality_profile: str = "strict99"
) -> dict[str, float | int | str]:
    if quality_profile == "strict99":
        return {
            "min3": 0.99,
            "min4": 0.99,
            "min_tip": 0.99,
            "max_loss": 5,
            "loss_level": "four",
            "max_force": np.inf,
        }
    if quality_profile == "strict100":
        return {
            "min3": 1.0,
            "min4": 1.0,
            "min_tip": 1.0,
            "max_loss": 0,
            "loss_level": "four",
            "max_force": np.inf,
        }
    if quality_profile != "family":
        raise ValueError(f"Unknown quality profile {quality_profile!r}")
    config = load_object_config(object_id)
    raw = config.resolved.get("collection", {}).get("quality", {})
    return {
        "min3": float(raw.get("min_three_finger_ratio", 0.85)),
        "min4": float(raw.get("min_four_finger_ratio", 0.45)),
        "min_tip": 0.0,
        "max_loss": int(raw.get("max_contact_loss_frames", 25)),
        "loss_level": "three",
        "max_force": float(raw.get("max_force_N", np.inf)),
    }


def _read_h5(
    path: Path,
    threshold_override: float | None,
    enforce_force_limit: bool,
    quality_profile: str,
) -> list[EpisodeQuality]:
    with h5py.File(path, "r") as source:
        object_id = str(source.attrs.get("object_id", ""))
        if not object_id:
            raise ValueError(f"{path} has no object_id attribute")
        motion_mode = str(source.attrs.get("motion_mode", "rotation"))
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
        translation_axis = (
            _flatten_env(np.asarray(source["object_translation_axis_world"]))
            if "object_translation_axis_world" in source
            else np.zeros((episode_id.size, 3), dtype=np.float32)
        )
        translation_amplitude = (
            _flatten_env(np.asarray(source["object_translation_amplitude_m"]))
            if "object_translation_amplitude_m" in source
            else np.zeros_like(episode_id, dtype=np.float32)
        )

    limits = _quality_limits(object_id, quality_profile)
    rows: list[EpisodeQuality] = []
    for eid in np.unique(episode_id):
        mask = episode_id == eid
        episode_loaded = loaded[mask]
        count = episode_loaded.sum(axis=1)
        contact3 = float(np.mean(count >= 3))
        contact4 = float(np.mean(count == 4))
        initial_contact3 = float(np.mean(count[: min(100, count.size)] >= 3))
        initial_contact4 = float(np.mean(count[: min(100, count.size)] == 4))
        below3_indices = np.flatnonzero(count < 3)
        below3_run = _max_true_run(count < 3)
        below4_run = _max_true_run(count < 4)
        tip_ratios = tuple(float(value) for value in episode_loaded.mean(axis=0))
        episode_force = force_norm[mask]
        moving_speed = angular_speed[mask]
        moving_speed = moving_speed[moving_speed > 1.0e-8]
        force_p95 = float(np.percentile(episode_force, 95.0))
        force_max = float(np.max(episode_force))
        axis_label = _axis_label(
            rotation_axis[mask][0], inactive=motion_mode == "translation"
        )
        translation_label = _axis_label(
            translation_axis[mask][0], inactive=motion_mode == "rotation"
        )
        passed = bool(
            contact3 >= limits["min3"]
            and contact4 >= limits["min4"]
            and min(tip_ratios) >= limits["min_tip"]
            and (
                below4_run if limits["loss_level"] == "four" else below3_run
            ) <= limits["max_loss"]
            and (not enforce_force_limit or force_max <= limits["max_force"])
        )
        rows.append(
            EpisodeQuality(
                source_file=path.name,
                object_id=object_id,
                motion_mode=motion_mode,
                episode_id=int(eid),
                frames=int(mask.sum()),
                contact3_ratio=contact3,
                contact4_ratio=contact4,
                initial_contact3_ratio=initial_contact3,
                initial_contact4_ratio=initial_contact4,
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
                translation_axis_label=translation_label,
                translation_amplitude_m=float(translation_amplitude[mask][0]),
                passed=passed,
            )
        )
    return rows


def _write_episode_report(path: Path, rows: list[EpisodeQuality]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "source_file", "object_id", "motion_mode", "episode_id", "frames", "passed", "contact3_ratio",
        "contact4_ratio", "initial_contact3_ratio", "initial_contact4_ratio",
        "first_below3_frame",
        "below3_max_run", "below4_max_run", "tip0_ratio",
        "tip1_ratio", "tip2_ratio", "tip3_ratio", "force_p95_N",
        "force_max_N", "angular_speed_mean",
        "rotation_axis_label", "rotation_speed_target",
        "translation_axis_label", "translation_amplitude_m",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "source_file": row.source_file,
                    "object_id": row.object_id,
                    "motion_mode": row.motion_mode,
                    "episode_id": row.episode_id,
                    "frames": row.frames,
                    "passed": int(row.passed),
                    "contact3_ratio": row.contact3_ratio,
                    "contact4_ratio": row.contact4_ratio,
                    "initial_contact3_ratio": row.initial_contact3_ratio,
                    "initial_contact4_ratio": row.initial_contact4_ratio,
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
                    "translation_axis_label": row.translation_axis_label,
                    "translation_amplitude_m": row.translation_amplitude_m,
                }
            )


def _print_summary(
    rows: list[EpisodeQuality], *, enforce_force_limit: bool, quality_profile: str
) -> None:
    grouped: dict[str, list[EpisodeQuality]] = {}
    for row in rows:
        grouped.setdefault(row.object_id, []).append(row)

    loss_label = "loss4" if quality_profile != "family" else "loss3"
    header = (
        f"{'Object':<22} {'Pass':>7} {'C3 med':>8} {'C3 min':>8} "
        f"{'C4 med':>8} {'C4 min':>8} {loss_label:>7} "
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
            f"{max((row.below4_max_run if quality_profile != 'family' else row.below3_max_run) for row in object_rows):>7d} "
            + " ".join(f"{value:>6.1%}" for value in np.median(tip, axis=0))
            + f" {np.median([r.force_p95_N for r in object_rows]):>6.2f} "
            f"{max(r.force_max_N for r in object_rows):>6.2f}"
        )

    print("\nPer-object diagnosis")
    for object_id, object_rows in grouped.items():
        limits = _quality_limits(object_id, quality_profile)
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
        loss = max(
            row.below4_max_run
            if limits["loss_level"] == "four"
            else row.below3_max_run
            for row in object_rows
        )
        if loss > limits["max_loss"]:
            label = "four-tip loss" if limits["loss_level"] == "four" else "<3-tip loss"
            problems.append(f"{label} run {loss} > {limits['max_loss']} frames")
        if peak > limits["max_force"]:
            qualifier = "rejects trajectory" if enforce_force_limit else "diagnostic only"
            diagnostics.append(
                f"peak force {peak:.2f} > {limits['max_force']:.2f} N ({qualifier})"
            )
        weakest = int(np.argmin(np.median(tip, axis=0)))
        if np.median(tip[:, weakest]) < limits["min_tip"]:
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
            bad_early_motion = [
                row for row in object_rows
                if (
                    row.initial_contact4_ratio
                    if limits["loss_level"] == "four"
                    else row.initial_contact3_ratio
                ) < (
                    limits["min4"]
                    if limits["loss_level"] == "four"
                    else limits["min3"]
                )
            ]
            if bad_early_motion:
                detail = ", ".join(
                    f"ep{row.episode_id}/{row.motion_mode}/"
                    f"{row.rotation_axis_label if row.motion_mode != 'translation' else row.translation_axis_label}"
                    for row in bad_early_motion
                )
                print(
                    "    - diagnostic only: the first 100 RECORDED motion "
                    f"frames are below the contact target in {detail}; "
                    "unrecorded arm-prep frames are not analyzed"
                )
            if len(bad_early_motion) < len(object_rows):
                losing_modes = {
                    row.motion_mode
                    for row in object_rows
                    if row not in bad_early_motion and not row.passed
                }
                detail: list[str] = []
                if losing_modes & {"rotation", "combined"}:
                    detail.append(
                        "rotation speed="
                        f"{motion['rotation']['angular_speed_range_rad_s']} rad/s"
                    )
                if losing_modes & {"translation", "combined"}:
                    detail.append(
                        "translation speed="
                        f"{motion['translation']['speed_range_m_s']} m/s"
                    )
                print(
                    "    - trajectories that start in contact but lose it use "
                    + " and ".join(detail)
                    + "; retest those pose/axis pairs with a shorter path or "
                    "lower upper speed/amplitude"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Raw trajectory H5 files")
    parser.add_argument("--contact-threshold", type=float, default=None)
    parser.add_argument(
        "--quality-profile",
        choices=("strict99", "strict100", "family"),
        default="strict99",
        help=(
            "strict99 (default): all-four and every-tip ratios >=99%% with "
            "at most 5 consecutive lost frames; family enables the older "
            "shape-family diagnostic thresholds."
        ),
    )
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
            path,
            args.contact_threshold,
            args.enforce_force_limit,
            args.quality_profile,
        )
        rows.extend(object_rows)
        print(f"[QUALITY] {path.name}: {len(object_rows)} episodes")
    print(f"[QUALITY] profile={args.quality_profile}")
    _print_summary(
        rows,
        enforce_force_limit=args.enforce_force_limit,
        quality_profile=args.quality_profile,
    )
    if args.report is not None:
        _write_episode_report(args.report, rows)
        print(f"[QUALITY] report: {args.report}")


if __name__ == "__main__":
    main()
