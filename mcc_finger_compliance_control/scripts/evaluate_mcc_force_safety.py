"""Run and summarize live-DP + FullHandMCC force-safety validation."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


FINGER_LABELS = ("index", "middle", "ring", "thumb")


def _validation_ids(checkpoint_path: Path) -> list[int]:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    try:
        return [int(value) for value in checkpoint["val_episode_ids"]]
    finally:
        del checkpoint


def _summarize_episode(path: Path, bootstrap_end: int) -> dict[str, float | int]:
    data = pd.read_csv(path)
    data = data[data["frame"] >= bootstrap_end]
    if data.empty:
        raise ValueError(f"No post-bootstrap rows in {path}")
    force_columns = [f"{label}_force_raw_N" for label in FINGER_LABELS]
    forces = data[force_columns].to_numpy(dtype=np.float64)
    frame_max = forces.max(axis=1)
    result: dict[str, float | int] = {
        "episode_id": int(path.stem.removeprefix("ep")),
        "frames": int(len(data)),
        "contact3": float(np.mean(data["found_contacts"] >= 3)),
        "contact4": float(np.mean(data["found_contacts"] == 4)),
        "force_p95_N": float(np.percentile(frame_max, 95)),
        "force_p99_N": float(np.percentile(frame_max, 99)),
        "force_max_N": float(frame_max.max()),
        "force_over_2x_ratio": float(np.mean(frame_max > 2.0)),
        "q_mae_rad": float(data["q_teacher_mae_rad"].mean()),
    }
    for index, label in enumerate(FINGER_LABELS):
        result[f"{label}_contact"] = float(
            data[f"{label}_found"].mean()
        )
        result[f"{label}_force_max_N"] = float(forces[:, index].max())
    return result


def _write_summary(
    output_dir: Path,
    rows: list[dict[str, float | int]],
    desired_force: float,
    force_limit: float,
) -> None:
    fieldnames = list(rows[0])
    with (output_dir / "summary_per_episode.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    weights = np.asarray([int(row["frames"]) for row in rows], dtype=float)
    contact4 = np.asarray([float(row["contact4"]) for row in rows])
    contact3 = np.asarray([float(row["contact3"]) for row in rows])
    force_max = np.asarray([float(row["force_max_N"]) for row in rows])
    force_p95 = np.asarray([float(row["force_p95_N"]) for row in rows])
    aggregate = {
        "episodes": len(rows),
        "frames": int(weights.sum()),
        "desired_force_N": desired_force,
        "force_limit_N": force_limit,
        "contact3_weighted": float(np.average(contact3, weights=weights)),
        "contact4_weighted": float(np.average(contact4, weights=weights)),
        "episodes_contact4_ge_90pct": int(np.sum(contact4 >= 0.90)),
        "force_p95_episode_median_N": float(np.median(force_p95)),
        "force_max_all_N": float(force_max.max()),
        "episodes_force_max_le_2x": int(
            np.sum(force_max <= force_limit)
        ),
        "contact_gate_pass": bool(np.average(contact4, weights=weights) >= 0.90),
        "force_gate_pass": bool(force_max.max() <= force_limit),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    lines = [
        "# MCC force-safety validation",
        "",
        f"- episodes: {aggregate['episodes']}",
        f"- post-bootstrap frames: {aggregate['frames']}",
        f"- contact >=3: {100.0 * aggregate['contact3_weighted']:.2f}%",
        f"- four-finger contact: {100.0 * aggregate['contact4_weighted']:.2f}%",
        f"- global maximum force: {aggregate['force_max_all_N']:.3f} N",
        f"- force limit: {aggregate['force_limit_N']:.3f} N",
        f"- contact gate: {'PASS' if aggregate['contact_gate_pass'] else 'FAIL'}",
        f"- force gate: {'PASS' if aggregate['force_gate_pass'] else 'FAIL'}",
    ]
    (output_dir / "REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(aggregate, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--inference-steps", type=int, default=10)
    parser.add_argument("--dp-replan-interval", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--desired-force", type=float, default=1.0)
    parser.add_argument(
        "--force-limit",
        type=float,
        default=None,
        help="Validation ceiling in newtons (default: 2 * desired force).",
    )
    parser.add_argument("--hand-servo-stiffness", type=float, default=8.0)
    parser.add_argument("--hand-servo-damping", type=float, default=1.3)
    parser.add_argument("--hand-servo-effort-limit", type=float, default=12.0)
    parser.add_argument("--thumb-servo-stiffness", type=float, default=8.0)
    parser.add_argument("--thumb-servo-damping", type=float, default=1.3)
    parser.add_argument("--thumb-servo-effort-limit", type=float, default=12.0)
    parser.add_argument("--mcc-max-normal-offset-mm", type=float, default=20.0)
    parser.add_argument(
        "--mcc-thumb-max-outward-offset-mm", type=float, default=20.0
    )
    parser.add_argument("--mcc-finger-servo-load-scale", type=float, default=0.0)
    parser.add_argument("--mcc-finger-tracking-gain", type=float, default=0.0)
    parser.add_argument("--mcc-force-servo-hard-step-mm", type=float, default=0.20)
    parser.add_argument("--mcc-force-servo-search-step-mm", type=float, default=0.50)
    parser.add_argument(
        "--mcc-force-servo-weak-contact-step-mm", type=float, default=0.20
    )
    parser.add_argument("--mcc-overforce-hard-ratio", type=float, default=1.4)
    parser.add_argument("--mcc-thumb-overforce-hard-ratio", type=float, default=1.4)
    parser.add_argument("--bootstrap-end", type=int, default=75)
    parser.add_argument("--episode-ids", type=int, nargs="*", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_ids = args.episode_ids or _validation_ids(args.model)
    deploy = Path(__file__).with_name("deploy_dp_inverse.py")
    for number, episode_id in enumerate(episode_ids, start=1):
        report = output_dir / f"ep{episode_id}.csv"
        if report.exists() and args.skip_existing:
            print(
                f"[{number:02d}/{len(episode_ids):02d}] ep{episode_id}: cached",
                flush=True,
            )
            continue
        command = [
            sys.executable,
            str(deploy),
            "--file", str(args.file),
            "--model", str(args.model),
            "--episode-id", str(episode_id),
            "--mode", "live_dp",
            "--viewer", "headless",
            "--device", args.device,
            "--inference-steps", str(args.inference_steps),
            "--seed", str(args.seed),
            "--contact-threshold", "0.05",
            "--execution-layer", "fullhand_mcc",
            "--mcc-direction-source", "hybrid",
            "--chunk-execution",
            "--dp-replan-interval", str(args.dp_replan_interval),
            "--mcc-desired-force", str(args.desired_force),
            "--hand-servo-stiffness", str(args.hand_servo_stiffness),
            "--hand-servo-damping", str(args.hand_servo_damping),
            "--hand-servo-effort-limit", str(args.hand_servo_effort_limit),
            "--thumb-servo-stiffness", str(args.thumb_servo_stiffness),
            "--thumb-servo-damping", str(args.thumb_servo_damping),
            "--thumb-servo-effort-limit", str(args.thumb_servo_effort_limit),
            "--mcc-max-normal-offset-mm", str(args.mcc_max_normal_offset_mm),
            "--mcc-thumb-max-outward-offset-mm",
            str(args.mcc_thumb_max_outward_offset_mm),
            "--mcc-finger-servo-load-scale",
            str(args.mcc_finger_servo_load_scale),
            "--mcc-finger-tracking-gain", str(args.mcc_finger_tracking_gain),
            "--mcc-force-servo-hard-step-mm",
            str(args.mcc_force_servo_hard_step_mm),
            "--mcc-force-servo-search-step-mm",
            str(args.mcc_force_servo_search_step_mm),
            "--mcc-force-servo-weak-contact-step-mm",
            str(args.mcc_force_servo_weak_contact_step_mm),
            "--mcc-overforce-hard-ratio",
            str(args.mcc_overforce_hard_ratio),
            "--mcc-thumb-overforce-hard-ratio",
            str(args.mcc_thumb_overforce_hard_ratio),
            "--report", str(report),
        ]
        process = subprocess.run(
            command, text=True, capture_output=True, check=False
        )
        result_lines = [
            line for line in process.stdout.splitlines()
            if line.startswith("[RESULT]")
        ]
        if process.returncode != 0:
            print(process.stdout)
            print(process.stderr, file=sys.stderr)
            raise subprocess.CalledProcessError(process.returncode, command)
        result = result_lines[-1] if result_lines else "completed"
        print(
            f"[{number:02d}/{len(episode_ids):02d}] ep{episode_id}: {result}",
            flush=True,
        )

    rows = [
        _summarize_episode(output_dir / f"ep{episode_id}.csv", args.bootstrap_end)
        for episode_id in episode_ids
    ]
    force_limit = (
        2.0 * args.desired_force
        if args.force_limit is None
        else float(args.force_limit)
    )
    _write_summary(output_dir, rows, args.desired_force, force_limit)


if __name__ == "__main__":
    main()
