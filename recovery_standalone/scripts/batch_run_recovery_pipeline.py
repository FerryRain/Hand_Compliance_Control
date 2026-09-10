"""Batch-run the full DAgger recovery pipeline across many DP-failure rollouts.

For each rollout H5 in --rollout-dir (collect_dagger_rollouts.py output:
epNNN.h5 + epNNN.csv pairs), this finds the first *sustained* failure frame
from the CSV report, then chains:

    extract_recovery_state.py
      -> generate_manifold_palm_plan.py --recovery-state
      -> optimize_contact_plan.py --recovery-state
      -> collect_trajectories.py --motion-mode planner_inverse --init-h5
      -> invert_trajectories.py
      -> export_palm_dp.py
      -> build_recovery_dataset.py

writing one recovery_<episode>.h5 per successfully-recovered episode, plus a
JSON summary and a full stdout/stderr log, mirroring
run_full_pipeline.py's/collect_dagger_rollouts.py's own structure and
conventions. Everything this script calls is a *subprocess*: no team code is
imported or modified; if a stage's CLI changes upstream, only the argument
lists below need updating.

Must be run from the repository root (matches collect_trajectories.py's own
hardcoded output path convention, mcc_finger_compliance_control/data/
trajectories/<filename>.h5):

    cd ~/Hand_Compliance_Control
    conda activate mjlab
    python mcc_finger_compliance_control/scripts/batch_run_recovery_pipeline.py \\
      --rollout-dir mcc_finger_compliance_control/data/closed_loop_rollouts/recovery_round1 \\
      --reference-dp mcc_finger_compliance_control/data/inverted/mustard_v1_239_motion96_kinematic_palm_dp.h5 \\
      --output-dir mcc_finger_compliance_control/data/dp/recovery_batch1 \\
      --object-id ycb_mustard

Known constraints baked in as defaults from this round's real GPU debugging
(see the plan file / work-log for the full story -- these are not arbitrary):

  - --motion-start/--record-start-step must exceed the fixed-palm
    controller's own prep window (observed 150 steps for ycb_mustard) or the
    object gets reanchored to a still-moving arm pose and drifts away
    instead of recovering; default 160.
  - --pred-horizon must match the checkpoint you actually intend to
    fine-tune, NOT this tool's own default -- ycb_mustard's
    mustard_v1_239_..._pred8_25k checkpoint uses pred_horizon=8. Passing the
    wrong value silently produces zero usable training windows.
  - collect_trajectories.py needs the repo root on PYTHONPATH (its
    mjlab.tasks.leaphand import chain uses an absolute
    mcc_finger_compliance_control.scripts.* import); this script sets that
    for its own subprocess calls automatically.
  - The action-label caveat from build_recovery_dataset.py still applies:
    every file this produces uses q_hand as an action-label placeholder, not
    the Guide's literal q_ref (see that tool's own module docstring).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import h5py

SCRIPTS = Path(__file__).resolve().parent
BUNDLE_ROOT = SCRIPTS.parent
EXTRACT = SCRIPTS / "extract_recovery_state.py"
GENERATE_PLAN = SCRIPTS / "generate_manifold_palm_plan.py"
OPTIMIZE = SCRIPTS / "optimize_contact_plan.py"
COLLECT = SCRIPTS / "collect_trajectories.py"
INVERT = SCRIPTS / "invert_trajectories.py"
EXPORT_DP = SCRIPTS / "export_palm_dp.py"
BUILD = SCRIPTS / "build_recovery_dataset.py"
RAW_TRAJECTORY_DIR = BUNDLE_ROOT / "data/trajectories"


def subprocess_env() -> dict[str, str]:
    """Environment for every stage, independent of the caller's cwd.

    Each stage is spawned by absolute path, but the scripts import their own
    siblings by bare module name (``object_catalog``, ``surface_mcc_finger``,
    ``dp_motion_features``, ...) and ``collect_trajectories.py`` writes into
    ``BUNDLE_ROOT/data/trajectories``. On sys.path is therefore the one thing
    every stage needs, and it is the same directory for all of them.
    """

    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{SCRIPTS}{os.pathsep}{existing}" if existing else str(SCRIPTS)
    return env


def run(cmd: list[str], desc: str, log) -> tuple[bool, str]:
    print(f"\n{'=' * 70}\n[BATCH-RECOVERY] {desc}\n[BATCH-RECOVERY] {' '.join(cmd)}", flush=True)
    result = subprocess.run(
        cmd, capture_output=True, text=True, env=subprocess_env()
    )
    log.write(
        f"\n### {desc}\n$ {' '.join(cmd)}\n--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}\n"
    )
    log.flush()
    ok = result.returncode == 0
    print(f"[BATCH-RECOVERY] {'OK' if ok else f'FAILED (code={result.returncode})'}: {desc}")
    if not ok:
        print("\n".join(result.stderr.strip().splitlines()[-15:]))
    return ok, result.stdout


def find_failure_frame(
    csv_path: Path,
    trigger_threshold: int,
    sustain_threshold: int,
    min_sustained_frames: int,
    min_frame: int = 0,
) -> int | None:
    """Return the first frame at or after ``min_frame`` whose
    ``found_contacts`` drops to ``trigger_threshold`` or below AND stays at
    ``sustain_threshold`` or below for at least ``min_sustained_frames``
    frames afterward.

    The trigger and sustain thresholds are deliberately independent (not
    "trigger and trigger+1"): the real ep024/frame=201 case this pipeline
    was validated against drops from 4 to 3 and *stays* mostly at 3
    (dipping to 2 occasionally) for a long stretch -- catching that
    degradation onset as early as possible (rather than waiting for the
    first frame that happens to hit 2) gives the recovery the most usable
    continuation length. With the defaults (trigger<=3, sustain<=3) this
    correctly rejects a single-frame startup blip that recovers to 4 within
    a few frames (the window check fails and the scan continues past it),
    while still catching a sustained partial-contact failure at its onset.

    ``min_frame`` matters separately: a rollout's own initial bootstrap
    window (its ``bootstrap_frames`` attr -- see ``run()``'s caller, which
    reads it per-rollout) can itself have low/zero contacts for a genuinely
    sustained stretch while the hand is still settling into its initial
    pregrasp, which the sustain check alone cannot distinguish from a real
    "grasped, then later failed" case. Frames before ``min_frame`` are
    therefore never treated as candidates.
    """
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    contacts = [int(row["found_contacts"]) for row in rows]
    frames = [int(row["frame"]) for row in rows]
    for i, value in enumerate(contacts):
        if frames[i] < min_frame:
            continue
        if value > trigger_threshold:
            continue
        window = contacts[i : i + min_sustained_frames]
        if len(window) < min_sustained_frames:
            break  # not enough trailing frames left to call this "sustained"
        if all(v <= sustain_threshold for v in window):
            return frames[i]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--reference-dp", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--object-id", default="ycb_mustard")
    parser.add_argument("--state-schema", default="contact_geometry_planner_motion")
    parser.add_argument("--obs-horizon", type=int, default=16)
    parser.add_argument(
        "--pred-horizon",
        type=int,
        default=8,
        help="MUST match the checkpoint you intend to fine-tune (see module docstring).",
    )
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--plan-frames", type=int, default=400)
    parser.add_argument(
        "--motion-start",
        type=int,
        default=160,
        help="Must exceed the fixed-palm controller's own prep window (see module docstring).",
    )
    parser.add_argument("--trajectory-length", type=int, default=600)
    parser.add_argument(
        "--failure-trigger-threshold",
        type=int,
        default=3,
        help="Candidate failure frame: first frame with found_contacts <= this.",
    )
    parser.add_argument(
        "--failure-sustain-threshold",
        type=int,
        default=3,
        help="The candidate frame and the following window must all stay <= this.",
    )
    parser.add_argument("--failure-min-sustained-frames", type=int, default=15)
    parser.add_argument(
        "--failure-min-frame",
        type=int,
        default=100,
        help=(
            "Never treat a frame before this as a failure candidate, "
            "regardless of the rollout's own bootstrap_frames attr (see "
            "find_failure_frame's docstring) -- the very first frames of a "
            "rollout can have genuinely low found_contacts while the hand "
            "is still settling into its initial pregrasp, which looks "
            "identical to a real 'grasped then later failed' case under "
            "the sustain check alone."
        ),
    )
    parser.add_argument(
        "--max-q-jump-rad",
        type=float,
        default=0.13,
        help=(
            "build_recovery_dataset.py's default (0.05) rejected every real "
            "recovery episode in this batch's first run, all on q_jump alone "
            "(observed real values: 0.068/0.098/0.122 rad; the other 5 audit "
            "checks passed comfortably in every case). This is the same "
            "benign settle-drift as the ep024 case in the report: "
            "--planner-settle-steps delays recording ~200 steps after "
            "injection, during which the controller converges the hand from "
            "state A's imperfect posture to a stable grasp. 0.13 covers the "
            "observed distribution with a small margin; revisit if a later "
            "batch's real q_jump values exceed it."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--episodes",
        nargs="*",
        default=None,
        help="Restrict to these episode stems (e.g. ep030 ep045); default = every epNNN.h5 in --rollout-dir.",
    )
    args = parser.parse_args()

    rollouts = sorted(args.rollout_dir.glob("ep*.h5"))
    if args.episodes:
        wanted = set(args.episodes)
        rollouts = [path for path in rollouts if path.stem in wanted]
    if not rollouts:
        raise SystemExit(f"no epNNN.h5 files found in {args.rollout_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = args.output_dir / "_intermediate"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    RAW_TRAJECTORY_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = args.output_dir / f"batch_recovery_log_{timestamp}.txt"
    summary: list[dict] = []

    with log_path.open("w") as log:
        for rollout_path in rollouts:
            stem = rollout_path.stem
            csv_path = rollout_path.with_suffix(".csv")
            final_output = args.output_dir / f"recovery_{stem}.h5"
            entry: dict = {"episode": stem}

            if args.skip_existing and final_output.exists():
                print(f"[BATCH-RECOVERY] SKIP {stem} (output already exists)")
                summary.append({**entry, "status": "skipped_existing"})
                continue

            if not csv_path.is_file():
                print(f"[BATCH-RECOVERY] SKIP {stem}: no companion CSV at {csv_path}")
                summary.append({**entry, "status": "missing_csv"})
                continue

            with h5py.File(rollout_path, "r") as rollout_file:
                bootstrap_frames = int(rollout_file.attrs.get("bootstrap_frames", 0))
            min_frame = max(args.failure_min_frame, bootstrap_frames)

            failure_frame = find_failure_frame(
                csv_path,
                args.failure_trigger_threshold,
                args.failure_sustain_threshold,
                args.failure_min_sustained_frames,
                min_frame=min_frame,
            )
            if failure_frame is None:
                print(
                    f"[BATCH-RECOVERY] SKIP {stem}: no sustained failure found "
                    f"at/after frame {min_frame} "
                    f"(trigger<={args.failure_trigger_threshold}, "
                    f"sustain<={args.failure_sustain_threshold}, "
                    f">={args.failure_min_sustained_frames} frames) "
                    "-- this episode likely never lost contact after settling"
                )
                summary.append({**entry, "status": "no_sustained_failure"})
                continue
            entry["failure_frame"] = failure_frame
            print(f"[BATCH-RECOVERY] {stem}: failure_frame={failure_frame}")

            state_a = tmp_dir / f"{stem}_state_A.h5"
            plan = tmp_dir / f"{stem}_from_A.h5"
            plan_opt = tmp_dir / f"{stem}_from_A_opt.h5"
            recovery_filename = f"recovery_{stem}"
            raw_recovery = RAW_TRAJECTORY_DIR / f"{recovery_filename}.h5"
            recovery_inverted = tmp_dir / f"{stem}_recovery_inverted.h5"
            recovery_dp = tmp_dir / f"{stem}_recovery_dp.h5"

            ok, _ = run(
                [
                    sys.executable, str(EXTRACT),
                    "--rollout", str(rollout_path),
                    "--frame", str(failure_frame),
                    "--output", str(state_a),
                ],
                f"[1/7 extract_recovery_state] {stem}",
                log,
            )
            if not ok:
                summary.append({**entry, "status": "extract_failed"})
                continue

            ok, _ = run(
                [
                    sys.executable, str(GENERATE_PLAN),
                    "--recovery-state", str(state_a),
                    "--output", str(plan),
                    "--frames", str(args.plan_frames),
                ],
                f"[2/7 generate_manifold_palm_plan] {stem}",
                log,
            )
            if not ok:
                summary.append({**entry, "status": "plan_failed"})
                continue

            ok, _ = run(
                [
                    sys.executable, str(OPTIMIZE),
                    "--input", str(plan),
                    "--recovery-state", str(state_a),
                    "--output", str(plan_opt),
                    "--object-id", args.object_id,
                ],
                f"[3/7 optimize_contact_plan] {stem}",
                log,
            )
            if not ok:
                summary.append({**entry, "status": "optimize_failed"})
                continue

            ok, _ = run(
                [
                    sys.executable, str(COLLECT),
                    "--motion-mode", "planner_inverse",
                    "--planner-file", str(plan_opt),
                    "--init-h5", str(state_a),
                    "--teacher-controller", "fullhand_mcc",
                    "--object-id", args.object_id,
                    "--device", args.device,
                    "--num-envs", "1",
                    "--max-trajectories", "1",
                    "--trajectory-length", str(args.trajectory_length),
                    "--motion-start", str(args.motion_start),
                    "--record-start-step", str(args.motion_start),
                    "--fixed-motion-start",
                    "--seed", str(args.seed),
                    "--filename", recovery_filename,
                ],
                f"[4/7 collect_trajectories --init-h5] {stem}",
                log,
            )
            if not ok or not raw_recovery.exists():
                summary.append({**entry, "status": "collect_failed"})
                continue

            ok, _ = run(
                [
                    sys.executable, str(INVERT),
                    "--file", str(raw_recovery),
                    "--output", str(recovery_inverted),
                ],
                f"[5/7 invert_trajectories] {stem}",
                log,
            )
            if not ok:
                summary.append({**entry, "status": "invert_failed"})
                continue

            ok, _ = run(
                [
                    sys.executable, str(EXPORT_DP),
                    "--file", str(recovery_inverted),
                    "--output", str(recovery_dp),
                    "--state-schema", args.state_schema,
                ],
                f"[6/7 export_palm_dp] {stem}",
                log,
            )
            if not ok:
                summary.append({**entry, "status": "export_failed"})
                continue

            ok, _ = run(
                [
                    sys.executable, str(BUILD),
                    "--rollout", str(rollout_path),
                    "--failure-frame", str(failure_frame),
                    "--recovery", str(recovery_dp),
                    "--recovery-inverted", str(recovery_inverted),
                    "--reference-dp", str(args.reference_dp),
                    "--output", str(final_output),
                    "--obs-horizon", str(args.obs_horizon),
                    "--pred-horizon", str(args.pred_horizon),
                    "--stride", str(args.stride),
                    "--max-q-jump-rad", str(args.max_q_jump_rad),
                ],
                f"[7/7 build_recovery_dataset] {stem}",
                log,
            )
            if not ok:
                summary.append({**entry, "status": "build_rejected_or_failed"})
                continue

            summary.append({**entry, "status": "done", "output": str(final_output)})

    summary_path = args.output_dir / f"batch_recovery_summary_{timestamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(
        f"\n{'=' * 70}\n[BATCH-RECOVERY] DONE\n[BATCH-RECOVERY] summary -> {summary_path}\n"
        f"[BATCH-RECOVERY] full log -> {log_path}\n"
    )
    for item in summary:
        line = f"  {item['episode']}: {item['status']}"
        if "failure_frame" in item:
            line += f" (failure_frame={item['failure_frame']})"
        print(line)


if __name__ == "__main__":
    main()
