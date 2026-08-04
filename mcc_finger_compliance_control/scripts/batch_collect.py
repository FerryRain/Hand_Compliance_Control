"""Batch trajectory collection across all objects in the contact-object catalog.

Runs ``collect_trajectories.py`` for each configured object (or a subset) in
sequence with conservative per-object defaults.  Output files are named by
object id and timestamp so they can be tracked independently.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from object_catalog import list_object_ids


SCRIPTS = Path(__file__).resolve().parent
COLLECT_SCRIPT = SCRIPTS / "collect_trajectories.py"
FILTER_SCRIPT = SCRIPTS / "filter_trajectories.py"
DATA_DIR = SCRIPTS.parent / "data" / "trajectories"

# Baseline defaults that produce reasonable data without blowing up disk usage.
# These can be overridden per-object via CLI.  Translation is off by default
# because it roughly doubles collection time and the first pass should verify
# that rotation-only collection is stable across all shapes.
DEFAULTS = {
    "num_envs": 4,
    "trajectory_length": 2500,
    "max_trajectories": 64,
    "motion_start": 1000,
    "motion_length": 1400,
    "record_start_step": 1000,
    "initial_orientation_mode": "uniform",
    "contact_threshold": 0.05,
    "online_quality_gate": False,
}


def _run(cmd: list[str], desc: str) -> int:
    print(f"\n{'='*70}")
    print(f"[BATCH] {desc}")
    print(f"[BATCH] {' '.join(cmd)}")
    sys.stdout.flush()
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[BATCH] FAILED (code={result.returncode}): {desc}")
    else:
        print(f"[BATCH] OK: {desc}")
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--objects",
        nargs="*",
        default=None,
        help="Object ids to collect; defaults to all catalogued objects.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=DEFAULTS["num_envs"])
    parser.add_argument("--trajectory-length", type=int, default=DEFAULTS["trajectory_length"])
    parser.add_argument("--max-trajectories", type=int, default=DEFAULTS["max_trajectories"])
    parser.add_argument("--motion-start", type=int, default=DEFAULTS["motion_start"])
    parser.add_argument("--motion-length", type=int, default=DEFAULTS["motion_length"])
    parser.add_argument("--record-start-step", type=int, default=DEFAULTS["record_start_step"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--initial-orientation-mode",
        choices=("uniform", "jitter", "fixed"),
        default=DEFAULTS["initial_orientation_mode"],
    )
    parser.add_argument(
        "--teacher-controller",
        choices=("fullhand_mcc", "fixed_pregrasp"),
        default="fullhand_mcc",
    )
    parser.add_argument(
        "--axis-sampling",
        choices=("random", "stratified"),
        default="stratified",
        help="Use stratified axes for small cross-object baseline runs.",
    )
    parser.add_argument(
        "--enable-translation",
        action="store_true",
        help="Enable sinusoidal translation for every object.",
    )
    parser.add_argument(
        "--filter-after",
        action="store_true",
        help="Run filter_trajectories.py --min-all4-ratio 0.99 after each object.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip objects whose output H5 already exists.",
    )
    args = parser.parse_args()

    object_ids = list(args.objects if args.objects is not None else list_object_ids())
    if not object_ids:
        print("[BATCH] No objects selected.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    failed: list[str] = []
    collected: list[Path] = []

    for index, object_id in enumerate(object_ids):
        seed = args.seed if args.seed is not None else (42 + index * 1000)
        filename = f"{object_id}_{timestamp}"
        output = DATA_DIR / f"{filename}.h5"

        if args.skip_existing and output.exists():
            print(f"[BATCH] [{index+1}/{len(object_ids)}] SKIP {object_id} (exists: {output})")
            collected.append(output)
            continue

        print(f"\n[BATCH] [{index+1}/{len(object_ids)}] {object_id}")

        cmd = [
            sys.executable, str(COLLECT_SCRIPT),
            "--object-id", object_id,
            "--device", args.device,
            "--num-envs", str(args.num_envs),
            "--trajectory-length", str(args.trajectory_length),
            "--max-trajectories", str(args.max_trajectories),
            "--motion-start", str(args.motion_start),
            "--motion-length", str(args.motion_length),
            "--record-start-step", str(args.record_start_step),
            "--initial-orientation-mode", args.initial_orientation_mode,
            "--teacher-controller", args.teacher_controller,
            "--surface-target-mode", "nearest_surface",
            "--contact-threshold", str(DEFAULTS["contact_threshold"]),
            "--axis-sampling", args.axis_sampling,
            "--seed", str(seed),
            "--filename", filename,
        ]
        if args.enable_translation:
            cmd.append("--enable-translation")

        rc = _run(cmd, f"collect {object_id}")
        if rc != 0:
            failed.append(object_id)
            continue
        collected.append(output)

        if args.filter_after:
            filtered = DATA_DIR / f"{filename}_relaxed99.h5"
            cmd = [
                sys.executable, str(FILTER_SCRIPT),
                str(output),
                "--output", str(filtered),
                "--contact-threshold", str(DEFAULTS["contact_threshold"]),
                "--min-all4-ratio", "0.99",
                "--min-per-tip-ratio", "0.99",
                "--max-loss-run", "5",
            ]
            _run(cmd, f"filter {object_id}")

    print(f"\n[BATCH] {'='*70}")
    print(f"[BATCH] DONE  collected={len(collected)}/{len(object_ids)}")
    if failed:
        print(f"[BATCH] FAILED objects: {failed}")
    print(f"[BATCH] Output directory: {DATA_DIR}")


if __name__ == "__main__":
    main()
