"""Audit fingertip contact location and folded-back contact in trajectory H5s.

The tactile/front face of the current Leap fingertips lies on the negative-X
side of the MCC fingertip site frame.  A sufficiently large positive local X
is therefore an unambiguous rear-shell contact.  Local Y/Z are deliberately
not constrained: side contacts are valid for the current task.

This script never edits trajectory files.  It writes a CSV plus passing and
rejected manifests, so transient one-frame collision-reduction changes can be
distinguished from sustained contact by a folded fingertip.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np


FINGER_NAMES = ("index", "middle", "ring", "thumb")
BACK_CONTACT_X_LIMIT_M = np.asarray(
    (0.012, 0.012, 0.012, 0.016), dtype=np.float64
)

# q_hand order: 1,0,2,3, 5,4,6,7, 9,8,10,11, 12,13,14,15.
JOINT_LOWER = np.asarray(
    (
        -0.314, -1.047, -0.506, -0.366,
        -0.314, -1.047, -0.506, -0.366,
        -0.314, -1.047, -0.506, -0.366,
        -0.349, -0.470, -1.200, -1.340,
    ),
    dtype=np.float64,
)
JOINT_UPPER = np.asarray(
    (
        2.230, 1.047, 1.885, 2.042,
        2.230, 1.047, 1.885, 2.042,
        2.230, 1.047, 1.885, 2.042,
        2.094, 2.443, 1.900, 1.880,
    ),
    dtype=np.float64,
)
FLEXION_INDICES = np.asarray(
    ((0, 2, 3), (4, 6, 7), (8, 10, 11), (12, 14, 15)),
    dtype=np.int64,
)


def _longest_true_run(mask: np.ndarray) -> int:
    best = current = 0
    for value in np.asarray(mask, dtype=bool).reshape(-1):
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def _discover_h5(inputs: list[Path]) -> list[Path]:
    paths: set[Path] = set()
    for item in inputs:
        if item.is_file() and item.suffix == ".h5":
            paths.add(item.resolve())
        elif item.is_dir():
            paths.update(path.resolve() for path in item.glob("*.h5"))
        else:
            raise FileNotFoundError(item)
    return sorted(paths)


def _episode_array(dataset: h5py.Dataset, env: int, env_count: int) -> np.ndarray:
    value = np.asarray(dataset)
    if value.ndim >= 2 and value.shape[1] == env_count:
        return value[:, env]
    if env_count == 1 and value.ndim >= 1:
        return value
    raise ValueError(f"unsupported dataset layout {dataset.name}: {value.shape}")


def _world_points_in_tip_frames(
    fingertip_pose_world: np.ndarray,
    contact_points_world: np.ndarray,
) -> np.ndarray:
    """Transform same-frame world points into wxyz fingertip site frames."""

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
    return np.einsum("...ji,...j->...i", rotation, points - pose[..., :3])


def _audit_env(
    source: h5py.File,
    path: Path,
    env: int,
    args: argparse.Namespace,
) -> dict[str, object]:
    env_count = int(source["q_hand"].shape[1])
    q = _episode_array(source["q_hand"], env, env_count).astype(np.float64)
    force = _episode_array(
        source["fingertip_force_world"], env, env_count
    ).astype(np.float64)
    force_norm = np.linalg.norm(force, axis=-1)
    threshold = float(
        args.contact_threshold
        if args.contact_threshold is not None
        else source.attrs.get("contact_threshold", 0.05)
    )
    if "fingertip_collision_found" in source:
        collision = _episode_array(
            source["fingertip_collision_found"], env, env_count
        ) > 0.5
    else:
        collision = force_norm > 0.0
    raw_loaded = collision & (force_norm >= threshold)

    if "fingertip_contact_pos_tip" not in source:
        raise KeyError(f"{path}: missing fingertip_contact_pos_tip")
    stored_contact_tip = _episode_array(
        source["fingertip_contact_pos_tip"], env, env_count
    ).astype(np.float64)
    # Older collectors logged the policy debug point from before env.step but
    # logged world point and fingertip pose after env.step.  Prefer a
    # same-frame reconstruction whenever those two authoritative fields exist.
    if (
        "fingertip_contact_pos_world" in source
        and "fingertip_pose_world" in source
    ):
        contact_tip = _world_points_in_tip_frames(
            _episode_array(source["fingertip_pose_world"], env, env_count),
            _episode_array(source["fingertip_contact_pos_world"], env, env_count),
        )
        stored_point_mismatch = (
            raw_loaded
            & np.isfinite(stored_contact_tip).all(axis=-1)
            & (
                np.linalg.norm(stored_contact_tip - contact_tip, axis=-1)
                > args.stored_point_mismatch_threshold
            )
        )
        contact_point_source = "aligned_world_pose"
    else:
        contact_tip = stored_contact_tip
        stored_point_mismatch = np.zeros_like(raw_loaded)
        contact_point_source = "stored_tip_local"
    finite_point = np.isfinite(contact_tip).all(axis=-1)
    back = (
        raw_loaded
        & finite_point
        & (contact_tip[..., 0] > BACK_CONTACT_X_LIMIT_M[None, :])
    )

    travel = np.maximum(JOINT_UPPER - JOINT_LOWER, 1.0e-9)
    normalized_q = (q - JOINT_LOWER[None, :]) / travel[None, :]
    flexion = normalized_q[:, FLEXION_INDICES]
    synergy_spread = np.ptp(flexion, axis=-1)
    mean_flexion = np.mean(flexion, axis=-1)
    folded_shape = (
        (synergy_spread > args.synergy_spread_threshold)
        | (mean_flexion > args.deep_flexion_threshold)
    )
    folded_back = back & folded_shape

    if "fingertip_contact" in source:
        effective_contact = _episode_array(
            source["fingertip_contact"], env, env_count
        ) > 0.5
    else:
        effective_contact = raw_loaded & ~back
    back_mask_leak = back & effective_contact

    frames = len(q)
    back_ratio = back.mean(axis=0)
    folded_back_ratio = folded_back.mean(axis=0)
    back_runs = np.asarray(
        [_longest_true_run(back[:, finger]) for finger in range(4)]
    )
    folded_back_runs = np.asarray(
        [_longest_true_run(folded_back[:, finger]) for finger in range(4)]
    )
    leak_frames = int(back_mask_leak.sum())

    reject_reasons: list[str] = []
    if float(np.max(back_ratio)) > args.max_back_ratio:
        reject_reasons.append("back_ratio")
    if int(np.max(back_runs)) > args.max_back_run:
        reject_reasons.append("back_run")
    if float(np.max(folded_back_ratio)) > args.max_folded_back_ratio:
        reject_reasons.append("folded_back_ratio")
    if int(np.max(folded_back_runs)) > args.max_folded_back_run:
        reject_reasons.append("folded_back_run")
    if leak_frames:
        reject_reasons.append("back_mask_leak")

    row: dict[str, object] = {
        "file": str(path),
        "env": env,
        "frames": frames,
        "surface_pass": int(not reject_reasons),
        "reject_reason": "+".join(reject_reasons),
        "raw_loaded_finger_frames": int(raw_loaded.sum()),
        "back_finger_frames": int(back.sum()),
        "folded_back_finger_frames": int(folded_back.sum()),
        "back_mask_leak_frames": leak_frames,
        "contact_point_source": contact_point_source,
        "stored_point_mismatch_frames": int(stored_point_mismatch.sum()),
        "max_back_ratio": float(np.max(back_ratio)),
        "max_back_run": int(np.max(back_runs)),
        "max_folded_back_ratio": float(np.max(folded_back_ratio)),
        "max_folded_back_run": int(np.max(folded_back_runs)),
        "max_synergy_spread": float(np.max(synergy_spread)),
    }
    for finger, name in enumerate(FINGER_NAMES):
        contact_x = contact_tip[raw_loaded[:, finger], finger, 0]
        row[f"{name}_back_frames"] = int(back[:, finger].sum())
        row[f"{name}_back_ratio"] = float(back_ratio[finger])
        row[f"{name}_back_max_run"] = int(back_runs[finger])
        row[f"{name}_folded_back_frames"] = int(
            folded_back[:, finger].sum()
        )
        row[f"{name}_folded_back_max_run"] = int(folded_back_runs[finger])
        row[f"{name}_contact_x_p95_mm"] = (
            float(np.nanpercentile(contact_x, 95) * 1000.0)
            if contact_x.size
            else float("nan")
        )
        row[f"{name}_contact_x_max_mm"] = (
            float(np.nanmax(contact_x) * 1000.0)
            if contact_x.size
            else float("nan")
        )
        row[f"{name}_synergy_spread_p95"] = float(
            np.percentile(synergy_spread[:, finger], 95)
        )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument(
        "--report", type=Path, default=Path("contact_surface_audit.csv")
    )
    parser.add_argument("--passing-list", type=Path, default=None)
    parser.add_argument("--rejected-list", type=Path, default=None)
    parser.add_argument("--contact-threshold", type=float, default=None)
    parser.add_argument("--synergy-spread-threshold", type=float, default=0.75)
    parser.add_argument("--deep-flexion-threshold", type=float, default=0.92)
    parser.add_argument(
        "--stored-point-mismatch-threshold",
        type=float,
        default=0.05,
        help=(
            "Report old pre/post-step local-point mismatches above this distance "
            "without rejecting the physical trajectory (default 50 mm)."
        ),
    )
    parser.add_argument(
        "--max-back-ratio",
        type=float,
        default=0.005,
        help="Maximum rear-shell loaded ratio for any finger (default 0.5%%).",
    )
    parser.add_argument(
        "--max-back-run",
        type=int,
        default=3,
        help="Maximum consecutive rear-shell frames (default 3).",
    )
    parser.add_argument(
        "--max-folded-back-ratio",
        type=float,
        default=0.001,
        help="Maximum folded + rear-shell ratio for any finger (default 0.1%%).",
    )
    parser.add_argument(
        "--max-folded-back-run",
        type=int,
        default=2,
        help="Maximum consecutive folded + rear-shell frames (default 2).",
    )
    args = parser.parse_args()
    if args.max_back_run < 0 or args.max_folded_back_run < 0:
        raise ValueError("run thresholds must be non-negative")

    paths = _discover_h5(args.inputs)
    if not paths:
        raise RuntimeError("no H5 files found")
    rows: list[dict[str, object]] = []
    for path in paths:
        with h5py.File(path, "r") as source:
            required = {
                "q_hand",
                "fingertip_force_world",
                "fingertip_contact_pos_tip",
            }
            missing = sorted(required.difference(source.keys()))
            if missing:
                print(f"[AUDIT] skip {path}: missing {','.join(missing)}")
                continue
            env_count = int(source["q_hand"].shape[1])
            for env in range(env_count):
                rows.append(_audit_env(source, path, env, args))
    if not rows:
        raise RuntimeError("no auditable episodes found")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    passing = [row for row in rows if row["surface_pass"]]
    rejected = [row for row in rows if not row["surface_pass"]]
    passing_list = args.passing_list or args.report.with_name(
        f"{args.report.stem}_passing.txt"
    )
    rejected_list = args.rejected_list or args.report.with_name(
        f"{args.report.stem}_rejected.txt"
    )
    passing_list.write_text(
        "\n".join(f"{row['file']}#env={row['env']}" for row in passing) + "\n",
        encoding="utf-8",
    )
    rejected_list.write_text(
        "\n".join(
            f"{row['file']}#env={row['env']} reason={row['reject_reason']}"
            for row in rejected
        )
        + ("\n" if rejected else ""),
        encoding="utf-8",
    )

    total_frames = sum(int(row["frames"]) * 4 for row in rows)
    back_frames = sum(int(row["back_finger_frames"]) for row in rows)
    folded_back_frames = sum(
        int(row["folded_back_finger_frames"]) for row in rows
    )
    print(
        f"[AUDIT] pass={len(passing)}/{len(rows)} "
        f"back={back_frames}/{total_frames} finger-frames "
        f"({back_frames / max(total_frames, 1):.6%}) "
        f"folded_back={folded_back_frames}/{total_frames} "
        f"({folded_back_frames / max(total_frames, 1):.6%})"
    )
    print(f"[AUDIT] report: {args.report}")
    print(f"[AUDIT] passing: {passing_list}")
    print(f"[AUDIT] rejected: {rejected_list}")
    if rejected:
        for row in rejected[:20]:
            print(
                f"[AUDIT-REJECT] {Path(str(row['file'])).name} "
                f"env={row['env']} reason={row['reject_reason']} "
                f"back={row['back_finger_frames']} "
                f"folded_back={row['folded_back_finger_frames']}"
            )


if __name__ == "__main__":
    main()
