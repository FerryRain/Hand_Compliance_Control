"""Offline strict filtering for parallel MCC fingertip trajectory H5 files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np


DEFAULT_BACK_CONTACT_X_LIMIT_M = np.asarray(
    (0.012, 0.012, 0.012, 0.016), dtype=np.float64
)
DEFAULT_CONTACT_POINT_RADIUS_LIMIT_M = 0.05


def _flatten_env(data: np.ndarray) -> np.ndarray:
    """Flatten ``[time, env, ...]`` into contiguous per-env trajectories."""

    # A direct reshape is time-major and interleaves episodes as
    # e0/t0, e1/t0, ..., e0/t1.  That is harmless for aggregate ratios but
    # corrupts temporal training windows after the filtered data is written
    # as a single environment.  Move env before time first.
    env_major = np.swapaxes(data, 0, 1)
    return env_major.reshape(data.shape[1] * data.shape[0], *data.shape[2:])


def _select_dataset(
    data: np.ndarray,
    keep: np.ndarray,
    *,
    num_steps: int,
    num_envs: int,
    name: str,
) -> np.ndarray:
    """Select accepted episodes without confusing static env data with time.

    Streaming datasets use ``(T, E, ...)``.  A few planner constants, most
    importantly ``q_nominal``, are written once as ``(E, ...)``.  Flattening
    both layouts as if they were time series made a four-environment file
    produce 64 nominal values for a 10000-frame boolean mask.

    The filtered file is a single-env chronological stream.  Time-varying
    datasets therefore retain a singleton environment axis.  ``q_nominal``
    is expanded to one value per retained frame as ``(T_selected, 16)`` so it
    can represent different nominal grasps after several source environments
    or episodes have been concatenated.
    """

    array = np.asarray(data)
    if array.ndim >= 2 and array.shape[:2] == (num_steps, num_envs):
        return _flatten_env(array)[keep, None, ...]
    if array.ndim >= 1 and array.shape[0] == num_envs:
        expanded = np.broadcast_to(array[None, ...], (num_steps, *array.shape))
        selected = _flatten_env(expanded)[keep]
        if name == "q_nominal":
            return selected
        return selected[:, None, ...]
    # Global metadata datasets that are neither streamed nor per-env should
    # remain byte-for-byte unchanged.
    return array


def _run_statistics(mask: np.ndarray) -> tuple[int, int]:
    """Return (number of true runs, longest run) for a 1-D boolean mask."""
    runs: list[int] = []
    current = 0
    for active in np.asarray(mask, dtype=bool).reshape(-1):
        if active:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return len(runs), max(runs, default=0)


def _contact_points_tip_local(
    fingertip_pose_world: np.ndarray,
    contact_points_world: np.ndarray,
) -> np.ndarray:
    """Transform world contact points into their MCC fingertip site frames."""

    pose = np.asarray(fingertip_pose_world, dtype=np.float64)
    points = np.asarray(contact_points_world, dtype=np.float64)
    quaternion = pose[..., 3:7].copy()
    quaternion /= np.maximum(
        np.linalg.norm(quaternion, axis=-1, keepdims=True), 1.0e-12
    )
    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    rotation = np.empty((*quaternion.shape[:-1], 3, 3), dtype=np.float64)
    rotation[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    rotation[..., 0, 1] = 2.0 * (x * y - z * w)
    rotation[..., 0, 2] = 2.0 * (x * z + y * w)
    rotation[..., 1, 0] = 2.0 * (x * y + z * w)
    rotation[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    rotation[..., 1, 2] = 2.0 * (y * z - x * w)
    rotation[..., 2, 0] = 2.0 * (x * z - y * w)
    rotation[..., 2, 1] = 2.0 * (y * z + x * w)
    rotation[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return np.einsum(
        "...ji,...j->...i", rotation, points - pose[..., :3]
    )


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
    parser.add_argument(
        "--shape-deviation-threshold-rad",
        type=float,
        default=0.75,
        help=(
            "A frame is hand-shape-degraded when any joint differs from the "
            "initial stable-contact posture by at least this amount."
        ),
    )
    parser.add_argument(
        "--max-shape-degraded-ratio",
        type=float,
        default=1.0,
        help=(
            "Maximum degraded-frame ratio. The default reports the metric "
            "without rejecting; use e.g. 0.02 for quality filtering."
        ),
    )
    parser.add_argument(
        "--max-shape-degraded-run",
        type=int,
        default=-1,
        help=(
            "Maximum consecutive degraded frames; -1 reports without using "
            "this metric as a rejection condition."
        ),
    )
    parser.add_argument(
        "--max-back-contact-ratio",
        type=float,
        default=0.0,
        help=(
            "Maximum loaded-frame ratio on the back half of any fingertip "
            "mesh. Zero is appropriate for strict teacher data."
        ),
    )
    args = parser.parse_args()
    if not 0.0 <= args.min_all4_ratio <= 1.0:
        raise ValueError("--min-all4-ratio must lie in [0, 1]")
    if not 0.0 <= args.min_per_tip_ratio <= 1.0:
        raise ValueError("--min-per-tip-ratio must lie in [0, 1]")
    if args.max_loss_run < 0:
        raise ValueError("--max-loss-run must be non-negative")
    if args.shape_deviation_threshold_rad <= 0.0:
        raise ValueError("--shape-deviation-threshold-rad must be positive")
    if not 0.0 <= args.max_shape_degraded_ratio <= 1.0:
        raise ValueError("--max-shape-degraded-ratio must lie in [0, 1]")
    if args.max_shape_degraded_run < -1:
        raise ValueError("--max-shape-degraded-run must be -1 or non-negative")
    if not 0.0 <= args.max_back_contact_ratio <= 1.0:
        raise ValueError("--max-back-contact-ratio must lie in [0, 1]")

    output = args.output or args.input.with_name(f"{args.input.stem}_strict4tip.h5")
    report = args.report or args.input.with_name(
        f"{args.input.stem}_strict4tip_report.csv"
    )

    with h5py.File(args.input, "r") as src:
        num_steps, num_envs = src["episode_id"].shape[:2]
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
        raw_loaded_contact = found & (force_norm >= threshold)
        # Prefer a same-frame reconstruction.  Older collection files logged
        # ``fingertip_contact_pos_tip`` from the pre-step policy debug while
        # world contact and tip pose were recorded post-step.  At contact
        # onset that one-tick mismatch can turn a previous zero placeholder
        # into a metre-scale local point and falsely classify one frame.
        if (
            "fingertip_pose_world" in src
            and "fingertip_contact_pos_world" in src
        ):
            contact_pos_tip = _contact_points_tip_local(
                _flatten_env(np.asarray(src["fingertip_pose_world"])),
                _flatten_env(np.asarray(src["fingertip_contact_pos_world"])),
            )
        elif "fingertip_contact_pos_tip" in src:
            contact_pos_tip = _flatten_env(
                np.asarray(src["fingertip_contact_pos_tip"])
            )
        else:
            contact_pos_tip = None
        if contact_pos_tip is not None:
            plausible_contact_point = (
                np.isfinite(contact_pos_tip).all(axis=-1)
                & (
                    np.linalg.norm(contact_pos_tip, axis=-1)
                    <= DEFAULT_CONTACT_POINT_RADIUS_LIMIT_M
                )
            )
            invalid_contact_point = raw_loaded_contact & ~plausible_contact_point
            back_contact = raw_loaded_contact & plausible_contact_point & (
                contact_pos_tip[:, :, 0]
                > DEFAULT_BACK_CONTACT_X_LIMIT_M[None, :]
            )
        else:
            plausible_contact_point = np.ones_like(raw_loaded_contact)
            invalid_contact_point = np.zeros_like(raw_loaded_contact)
            back_contact = np.zeros_like(raw_loaded_contact)
        loaded_contact = (
            raw_loaded_contact & plausible_contact_point & ~back_contact
        )
        q_hand = (
            _flatten_env(np.asarray(src["q_hand"]))
            if "q_hand" in src
            else None
        )
        q_ref = (
            _flatten_env(np.asarray(src["q_ref"]))
            if "q_ref" in src
            else None
        )
        q_pre = (
            _flatten_env(np.asarray(src["q_pre"]))
            if "q_pre" in src
            else None
        )
        palm_tracking_error = None
        if "palm_control_pos_world" in src and "palm_x_des" in src:
            palm_tracking_error = np.linalg.norm(
                _flatten_env(np.asarray(src["palm_control_pos_world"]))
                - _flatten_env(np.asarray(src["palm_x_des"]))[:, :3],
                axis=-1,
            )

        rows: list[dict[str, object]] = []
        accepted_ids: list[int] = []
        for eid in np.unique(episode_id):
            mask = episode_id == eid
            per_tip = loaded_contact[mask].mean(axis=0)
            raw_per_tip = raw_loaded_contact[mask].mean(axis=0)
            back_ratio = back_contact[mask].mean(axis=0)
            invalid_point_ratio = invalid_contact_point[mask].mean(axis=0)
            contact_x_p95_mm = np.full(4, np.nan, dtype=np.float64)
            if contact_pos_tip is not None:
                episode_contact_pos = contact_pos_tip[mask]
                episode_raw_loaded = raw_loaded_contact[mask]
                for finger in range(4):
                    values = episode_contact_pos[
                        episode_raw_loaded[:, finger], finger, 0
                    ]
                    if values.size:
                        contact_x_p95_mm[finger] = (
                            np.nanpercentile(values, 95) * 1000.0
                        )
            all_four = loaded_contact[mask].all(axis=1)
            lost = np.flatnonzero(~all_four)
            loss_events, max_loss_run = _run_statistics(~all_four)

            episode_q = q_hand[mask] if q_hand is not None else None
            episode_q_ref = q_ref[mask] if q_ref is not None else None
            episode_q_pre = q_pre[mask] if q_pre is not None else None
            if episode_q is not None:
                # Recording begins after preparation/contact settling.  Use a
                # robust short-window median so a single collision impulse
                # cannot redefine the nominal stable hand shape.
                baseline_frames = min(25, len(episode_q))
                stable_q = np.median(episode_q[:baseline_frames], axis=0)
                baseline_q_deviation = np.max(
                    np.abs(episode_q - stable_q), axis=-1
                )
                shape_degraded = (
                    baseline_q_deviation
                    >= args.shape_deviation_threshold_rad
                )
                shape_events, max_shape_run = _run_statistics(shape_degraded)
                shape_ratio = float(shape_degraded.mean())
                shape_p95 = float(np.percentile(baseline_q_deviation, 95))
                shape_max = float(baseline_q_deviation.max())
            else:
                shape_events = max_shape_run = 0
                shape_ratio = shape_p95 = shape_max = float("nan")

            if episode_q_ref is not None and episode_q_pre is not None:
                controller_deviation = np.max(
                    np.abs(episode_q_ref - episode_q_pre), axis=-1
                )
                controller_deviation_p95 = float(
                    np.percentile(controller_deviation, 95)
                )
                controller_deviation_max = float(controller_deviation.max())
            else:
                controller_deviation_p95 = controller_deviation_max = float("nan")
            if episode_q is not None and episode_q_ref is not None:
                physical_deflection = np.max(
                    np.abs(episode_q - episode_q_ref), axis=-1
                )
                physical_deflection_p95 = float(
                    np.percentile(physical_deflection, 95)
                )
                physical_deflection_max = float(physical_deflection.max())
            else:
                physical_deflection_p95 = physical_deflection_max = float("nan")

            shape_ratio_ok = (
                not np.isfinite(shape_ratio)
                or shape_ratio <= args.max_shape_degraded_ratio
            )
            shape_run_ok = (
                args.max_shape_degraded_run < 0
                or max_shape_run <= args.max_shape_degraded_run
            )
            passed = bool(
                all_four.mean() >= args.min_all4_ratio
                and per_tip.min() >= args.min_per_tip_ratio
                and max_loss_run <= args.max_loss_run
                and back_ratio.max() <= args.max_back_contact_ratio
                and shape_ratio_ok
                and shape_run_ok
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
                    "loss_events": int(loss_events),
                    "tip0_ratio": float(per_tip[0]),
                    "tip1_ratio": float(per_tip[1]),
                    "tip2_ratio": float(per_tip[2]),
                    "tip3_ratio": float(per_tip[3]),
                    "tip0_raw_ratio": float(raw_per_tip[0]),
                    "tip1_raw_ratio": float(raw_per_tip[1]),
                    "tip2_raw_ratio": float(raw_per_tip[2]),
                    "tip3_raw_ratio": float(raw_per_tip[3]),
                    "tip0_back_ratio": float(back_ratio[0]),
                    "tip1_back_ratio": float(back_ratio[1]),
                    "tip2_back_ratio": float(back_ratio[2]),
                    "tip3_back_ratio": float(back_ratio[3]),
                    "tip0_invalid_point_ratio": float(invalid_point_ratio[0]),
                    "tip1_invalid_point_ratio": float(invalid_point_ratio[1]),
                    "tip2_invalid_point_ratio": float(invalid_point_ratio[2]),
                    "tip3_invalid_point_ratio": float(invalid_point_ratio[3]),
                    "tip0_contact_x_p95_mm": float(contact_x_p95_mm[0]),
                    "tip1_contact_x_p95_mm": float(contact_x_p95_mm[1]),
                    "tip2_contact_x_p95_mm": float(contact_x_p95_mm[2]),
                    "tip3_contact_x_p95_mm": float(contact_x_p95_mm[3]),
                    "tip0_min_force_N": float(force_norm[mask, 0].min()),
                    "tip1_min_force_N": float(force_norm[mask, 1].min()),
                    "tip2_min_force_N": float(force_norm[mask, 2].min()),
                    "tip3_min_force_N": float(force_norm[mask, 3].min()),
                    "first_loss_step": (
                        int(episode_step[mask][lost[0]]) if lost.size else -1
                    ),
                    "shape_degraded_ratio": shape_ratio,
                    "shape_degraded_max_run": int(max_shape_run),
                    "shape_degraded_events": int(shape_events),
                    "shape_deviation_p95_rad": shape_p95,
                    "shape_deviation_max_rad": shape_max,
                    # q_ref-q_pre diagnoses controller/IK folding; q-q_ref
                    # diagnoses physical deformation or inadequate palm
                    # withdrawal.  Keeping both avoids blaming the palm for a
                    # target that had already collapsed.
                    "controller_deviation_p95_rad": controller_deviation_p95,
                    "controller_deviation_max_rad": controller_deviation_max,
                    "physical_deflection_p95_rad": physical_deflection_p95,
                    "physical_deflection_max_rad": physical_deflection_max,
                    "palm_tracking_error_p95_mm": (
                        float(np.percentile(palm_tracking_error[mask], 95) * 1000.0)
                        if palm_tracking_error is not None
                        else float("nan")
                    ),
                    "palm_tracking_error_max_mm": (
                        float(palm_tracking_error[mask].max() * 1000.0)
                        if palm_tracking_error is not None
                        else float("nan")
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
                "strict_continuous_front_pad_four_fingertips_v3"
                if strict
                else "relaxed_front_pad_four_fingertips_v3"
            )
            dst.attrs["quality_contact_threshold"] = threshold
            dst.attrs["quality_min_all4_ratio"] = args.min_all4_ratio
            dst.attrs["quality_min_per_tip_ratio"] = args.min_per_tip_ratio
            dst.attrs["quality_max_loss_run"] = args.max_loss_run
            dst.attrs["quality_shape_deviation_threshold_rad"] = (
                args.shape_deviation_threshold_rad
            )
            dst.attrs["quality_max_shape_degraded_ratio"] = (
                args.max_shape_degraded_ratio
            )
            dst.attrs["quality_max_shape_degraded_run"] = (
                args.max_shape_degraded_run
            )
            dst.attrs["quality_max_back_contact_ratio"] = (
                args.max_back_contact_ratio
            )
            dst.attrs["quality_back_contact_x_limit_m"] = (
                DEFAULT_BACK_CONTACT_X_LIMIT_M
            )
            dst.attrs["quality_contact_point_radius_limit_m"] = (
                DEFAULT_CONTACT_POINT_RADIUS_LIMIT_M
            )
            dst.attrs["quality_source_file"] = args.input.name
            dst.attrs["quality_total_trajectories"] = len(rows)
            dst.attrs["quality_selected_trajectories"] = len(accepted_ids)
            dst.attrs["num_trajectories"] = len(accepted_ids)
            dst.attrs["strict_four_tip_continuous_contact"] = strict

            for name, source in src.items():
                selected = _select_dataset(
                    np.asarray(source),
                    keep,
                    num_steps=num_steps,
                    num_envs=num_envs,
                    name=name,
                )
                dataset_kwargs: dict[str, object] = {}
                if selected.ndim > 0 and selected.shape[0] > 0:
                    dataset_kwargs["chunks"] = (
                        min(100, selected.shape[0]),
                        *selected.shape[1:],
                    )
                dst.create_dataset(
                    name, data=selected, dtype=source.dtype, **dataset_kwargs
                )

    print(
        f"[FILTER] pass={len(accepted_ids)}/{len(rows)} "
        f"threshold={threshold:.3f}N all4>={args.min_all4_ratio:.4f} "
        f"per_tip>={args.min_per_tip_ratio:.4f} "
        f"max_loss_run<={args.max_loss_run} "
        f"max_back_ratio<={args.max_back_contact_ratio:.4f}"
    )
    print(f"[FILTER] output: {output}")
    print(f"[FILTER] report: {report}")


if __name__ == "__main__":
    main()
