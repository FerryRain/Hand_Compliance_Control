"""End-to-end, unattended, multi-object data pipeline.

Chains together the existing per-stage scripts in this folder:

    collect_trajectories.py  -> filter_trajectories.py
        -> invert_trajectories.py -> export_palm_dp.py
        -> replay_inverted.py (headless, auto QC on a few sampled episodes)

for every object in the catalog (or a subset you name), one object after
another, writing a JSON summary and a full stdout/stderr log at the end.
It does NOT run anything in parallel on purpose: collection and replay share
the same GPU, so objects are processed strictly one after another. This is
what makes it safe to launch once and walk away.

Drop this file next to collect_trajectories.py / filter_trajectories.py /
invert_trajectories.py / export_palm_dp.py / replay_inverted.py, i.e. into:

    mcc_finger_compliance_control/scripts/run_full_pipeline.py

Example (background, unattended, all catalogued objects):

    cd ~/Code/Hand_Compliance_Control
    conda activate mjlab
    MPLCONFIGDIR=/tmp/matplotlib WARP_CACHE_PATH=/tmp/warp \\
    nohup python mcc_finger_compliance_control/scripts/run_full_pipeline.py \\
        --device cuda:0 --num-envs 8 --max-trajectories 128 \\
        --skip-existing \\
        > pipeline_console.log 2>&1 &

    # check progress any time with:
    tail -f pipeline_console.log

Example (only a couple of objects, quick smoke test):

    python mcc_finger_compliance_control/scripts/run_full_pipeline.py \\
        --objects sphere_medium capsule_medium \\
        --device cuda:0 --num-envs 2 --max-trajectories 4 \\
        --trajectory-length 500 --motion-start 150 --motion-length 300 \\
        --record-start-step 150 --qc-episodes 1

Everything this script calls is a *subprocess*: no team code is imported or
modified. If any stage's CLI changes upstream, only the argument lists below
need updating.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from object_catalog import list_object_ids

SCRIPTS = Path(__file__).resolve().parent
# mcc_finger_compliance_control/scripts -> mcc_finger_compliance_control -> repo root
REPO_ROOT = SCRIPTS.parents[1]
COLLECT = SCRIPTS / "collect_trajectories.py"
FILTER = SCRIPTS / "filter_trajectories.py"
INVERT = SCRIPTS / "invert_trajectories.py"
EXPORT_DP = SCRIPTS / "export_palm_dp.py"
REPLAY = SCRIPTS / "replay_inverted.py"
TRAJ_DIR = SCRIPTS.parent / "data" / "trajectories"
INV_DIR = SCRIPTS.parent / "data" / "inverted"

RESULT_RE = re.compile(
    r"\[RESULT\].*tip_error_mean=([0-9.]+)mm.*tip_error_p95=([0-9.]+)mm"
)


def _subprocess_env() -> dict[str, str]:
    """Inherit the current environment but guarantee the repo root is on
    PYTHONPATH, so ``import mcc_finger_compliance_control...`` and similar
    top-level-package imports work in every stage script regardless of
    whether the caller's shell happened to export PYTHONPATH."""
    import os

    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    repo_root_str = str(REPO_ROOT)
    parts = [p for p in existing.split(":") if p]
    if repo_root_str not in parts:
        parts.insert(0, repo_root_str)
    env["PYTHONPATH"] = ":".join(parts)
    return env


def frame_count(h5_path: Path, dataset_key: str = "episode_id") -> int:
    """Return how many recorded frames an h5 file holds, or 0 if unreadable."""
    import h5py

    try:
        with h5py.File(h5_path, "r") as handle:
            if dataset_key not in handle:
                return 0
            return int(handle[dataset_key].shape[0])
    except OSError:
        return 0


def run(cmd: list[str], desc: str, log) -> tuple[bool, str]:
    print(f"\n{'=' * 70}\n[PIPELINE] {desc}\n[PIPELINE] {' '.join(cmd)}", flush=True)
    result = subprocess.run(
        cmd, capture_output=True, text=True, env=_subprocess_env(), cwd=str(REPO_ROOT)
    )
    log.write(f"\n### {desc}\n$ {' '.join(cmd)}\n--- stdout ---\n{result.stdout}\n"
              f"--- stderr ---\n{result.stderr}\n")
    log.flush()
    ok = result.returncode == 0
    print(f"[PIPELINE] {'OK' if ok else f'FAILED (code={result.returncode})'}: {desc}")
    if not ok:
        # Surface the tail of stderr immediately so a long unattended run's
        # console log still shows *why* an object was skipped.
        print("\n".join(result.stderr.strip().splitlines()[-15:]))
    return ok, result.stdout


def sample_episode_ids(report_csv: Path, n: int) -> list[int]:
    """Pick up to n *passing* episode ids, spread across the file."""
    if not report_csv.exists():
        return []
    with report_csv.open() as handle:
        passing = [
            int(row["episode_id"])
            for row in csv.DictReader(handle)
            if int(row.get("strict_pass", 0)) == 1
        ]
    if not passing:
        return []
    step = max(1, len(passing) // n)
    return passing[::step][:n]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects", nargs="*", default=None,
                         help="Object ids to run; default = every id in "
                              "configs/objects/*.yaml")
    parser.add_argument("--motion-modes", nargs="+", default=["rotation"],
                         choices=("rotation", "translation", "combined"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--trajectory-length", type=int, default=2500)
    parser.add_argument("--max-trajectories", type=int, default=128)
    parser.add_argument("--motion-start", type=int, default=1000)
    parser.add_argument("--motion-length", type=int, default=1400)
    parser.add_argument("--record-start-step", type=int, default=1000)
    parser.add_argument("--max-prep-wait-steps", type=int, default=1000)
    parser.add_argument("--initial-orientation-mode", default="uniform",
                         choices=("uniform", "jitter", "fixed"))
    parser.add_argument("--teacher-controller", default="fullhand_mcc",
                         choices=("fullhand_mcc", "fixed_pregrasp"))
    parser.add_argument("--axis-sampling", default="stratified",
                         choices=("random", "stratified"))
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--min-all4-ratio", type=float, default=0.99)
    parser.add_argument("--min-per-tip-ratio", type=float, default=0.99)
    parser.add_argument("--max-loss-run", type=int, default=5)
    parser.add_argument("--qc-episodes", type=int, default=3,
                         help="How many passing episodes per object to "
                              "auto-verify with a headless teacher replay.")
    parser.add_argument("--qc-max-tip-error-mm", type=float, default=5.0,
                         help="An object is flagged qc_all_pass=false if any "
                              "sampled episode's mean tip error exceeds this.")
    parser.add_argument("--skip-existing", action="store_true",
                         help="Skip an object/motion-mode if its exported "
                              "*_dp.h5 already exists.")
    args = parser.parse_args()

    object_ids = list(args.objects) if args.objects else list(list_object_ids())
    TRAJ_DIR.mkdir(parents=True, exist_ok=True)
    INV_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = SCRIPTS.parent / "data" / f"pipeline_log_{timestamp}.txt"
    summary: list[dict] = []

    with log_path.open("w") as log:
        for object_index, object_id in enumerate(object_ids):
            for motion_mode in args.motion_modes:
                filename = f"{object_id}_{motion_mode}_{timestamp}"
                raw_h5 = TRAJ_DIR / f"{filename}.h5"
                filtered_h5 = TRAJ_DIR / f"{filename}_relaxed99.h5"
                report_csv = TRAJ_DIR / f"{filename}_relaxed99_report.csv"
                inverted_h5 = INV_DIR / f"{filename}_relaxed99_inverted.h5"
                dp_h5 = INV_DIR / f"{filename}_relaxed99_dp.h5"
                entry = {"object_id": object_id, "motion_mode": motion_mode}

                if args.skip_existing and dp_h5.exists():
                    print(f"[PIPELINE] SKIP {object_id}/{motion_mode} (dp file exists)")
                    summary.append({**entry, "status": "skipped_existing"})
                    continue

                seed = args.seed + object_index * 1000

                ok, _ = run([
                    sys.executable, str(COLLECT),
                    "--object-id", object_id,
                    "--motion-mode", motion_mode,
                    "--device", args.device,
                    "--num-envs", str(args.num_envs),
                    "--trajectory-length", str(args.trajectory_length),
                    "--max-trajectories", str(args.max_trajectories),
                    "--motion-start", str(args.motion_start),
                    "--motion-length", str(args.motion_length),
                    "--record-start-step", str(args.record_start_step),
                    "--max-prep-wait-steps", str(args.max_prep_wait_steps),
                    "--initial-orientation-mode", args.initial_orientation_mode,
                    "--teacher-controller", args.teacher_controller,
                    "--axis-sampling", args.axis_sampling,
                    "--seed", str(seed),
                    "--filename", filename,
                ], f"[1/4 collect] {object_id}/{motion_mode}", log)
                if not ok:
                    summary.append({**entry, "status": "collect_failed"})
                    continue

                ok, _ = run([
                    sys.executable, str(FILTER), str(raw_h5),
                    "--output", str(filtered_h5),
                    "--report", str(report_csv),
                    "--contact-threshold", "0.05",
                    "--min-all4-ratio", str(args.min_all4_ratio),
                    "--min-per-tip-ratio", str(args.min_per_tip_ratio),
                    "--max-loss-run", str(args.max_loss_run),
                ], f"[2/4 filter] {object_id}/{motion_mode}", log)
                if not ok or not filtered_h5.exists():
                    summary.append({**entry, "status": "filter_failed_or_all_rejected"})
                    continue

                passing_count = 0
                if report_csv.exists():
                    with report_csv.open() as handle:
                        passing_count = sum(
                            int(row.get("strict_pass", 0))
                            for row in csv.DictReader(handle)
                        )
                if passing_count == 0:
                    print(f"[PIPELINE] {object_id}/{motion_mode}: 0 episodes passed "
                          f"the filter thresholds, skipping invert/export/QC.")
                    summary.append({**entry, "status": "filter_zero_pass",
                                    "report": str(report_csv)})
                    continue

                ok, _ = run([
                    sys.executable, str(INVERT),
                    "--file", str(filtered_h5),
                    "--output", str(inverted_h5),
                ], f"[3/4 invert] {object_id}/{motion_mode}", log)
                if not ok:
                    summary.append({**entry, "status": "invert_failed"})
                    continue

                if frame_count(inverted_h5) == 0:
                    print(f"[PIPELINE] {object_id}/{motion_mode}: 0 episodes survived "
                          f"filtering (see {report_csv.name}) — skipping export/QC, "
                          f"not treating this as a crash.")
                    summary.append({
                        **entry,
                        "status": "zero_episodes_passed_filter",
                        "report_csv": str(report_csv),
                    })
                    continue

                ok, _ = run([
                    sys.executable, str(EXPORT_DP),
                    "--file", str(inverted_h5),
                    "--output", str(dp_h5),
                ], f"[4/4 export_palm_dp] {object_id}/{motion_mode}", log)
                if not ok:
                    summary.append({**entry, "status": "export_dp_failed"})
                    continue

                episode_ids = sample_episode_ids(report_csv, args.qc_episodes) or [0]
                qc_results = []
                for episode_id in episode_ids:
                    ok, out = run([
                        sys.executable, str(REPLAY),
                        "--file", str(inverted_h5),
                        "--episode-id", str(episode_id),
                        "--viewer", "headless",
                        "--mode", "teacher",
                        "--device", args.device,
                        "--max-steps", str(args.trajectory_length - args.record_start_step
                                            + args.motion_start),
                        "--contact-threshold", "0.05",
                    ], f"[QC replay] {object_id}/{motion_mode} ep={episode_id}", log)
                    match = RESULT_RE.search(out)
                    if ok and match:
                        mean_mm = float(match.group(1))
                        qc_results.append({
                            "episode_id": episode_id,
                            "tip_error_mean_mm": mean_mm,
                            "tip_error_p95_mm": float(match.group(2)),
                            "pass": mean_mm <= args.qc_max_tip_error_mm,
                        })
                    else:
                        qc_results.append({"episode_id": episode_id, "pass": False})

                summary.append({
                    **entry,
                    "status": "done",
                    "dp_file": str(dp_h5),
                    "inverted_file": str(inverted_h5),
                    "qc": qc_results,
                    "qc_all_pass": all(r["pass"] for r in qc_results),
                })

    summary_path = SCRIPTS.parent / "data" / f"pipeline_summary_{timestamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"\n{'=' * 70}\n[PIPELINE] DONE\n[PIPELINE] summary -> {summary_path}\n"
          f"[PIPELINE] full log -> {log_path}\n")
    for item in summary:
        line = f"  {item['object_id']}/{item['motion_mode']}: {item['status']}"
        if item["status"] == "done":
            line += f" qc_all_pass={item['qc_all_pass']}"
        print(line)


if __name__ == "__main__":
    main()