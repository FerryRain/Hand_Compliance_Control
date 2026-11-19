"""Finite Mustard full-trajectory evaluation of explicit LEAP actuator limits.

Existing complete legacy trials are retained as references. Their initial
physical states need not match the new hardware-limited settling states;
configuration-level comparisons are not identical-state causal branches.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import h5py
import numpy as np

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parents[1]
DATA = SCRIPTS.parent / "data"
DEFAULT_IDS = [11, 81, 62, 46, 5, 40, 47, 80, 104, 105, 34, 42]


def save(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    tmp.replace(path)


def torque_report(folder, start, limit):
    path = folder / "physics_trace.npz"
    if not path.exists():
        return dict(available=False)
    with np.load(path) as archive:
        if "actuator_force" not in archive:
            return dict(available=False, reason="Legacy trace did not record actuator torque")
        mask = archive["step"] > start if start is not None else np.zeros(len(archive["step"]), bool)
        tau = archive["actuator_force"][mask, 0, -16:]
        velocity = archive["qvel"][mask, 0, -16:]
    if not len(tau):
        return dict(available=False, reason="No motion samples")
    actual = np.abs(tau)
    peak = float(actual.max())
    return dict(available=True, samples=len(tau), maximum_nm=peak,
                per_joint_max_nm=actual.max(0).tolist(), per_joint_rms_nm=np.sqrt((tau*tau).mean(0)).tolist(),
                p95_joint_sample_nm=float(np.quantile(actual, .95)),
                joint_sample_fraction_above_estimated_continuous=float((actual > .186+1e-5).mean()),
                joint_sample_fraction_at_cap=float((actual >= limit-1e-5).mean()),
                torque_cap_verified=bool(peak <= limit+1e-4),
                mechanical_power_absolute_mean_w=float(abs(tau*velocity).sum(-1).mean()),
                thermal_model=False)


def analyze(job):
    from run_collection_force_suite import analyze_job
    folder = Path(job["output"])
    row = analyze_job(dict(job, case=f'candidate_{job["candidate_id"]:04d}',
                           arm=job["variant"], geometry_group="same_healthy_reference"))
    summary = json.loads((folder / "summary.json").read_text())
    if (folder / "collected.h5").exists():
        with h5py.File(folder / "collected.h5") as handle:
            phase = handle["fullhand_contact_phase"][:, 0]
            command = handle["q_cmd"][:, 0]
        transitions = (phase[:-1] == 2) & (phase[1:] == 3)
        relapse = transitions[:-1] & (phase[2:] == 1)
        row["handover"] = dict(per_finger_count=transitions.sum(0).tolist(),
                                immediate_relapse_count=relapse.sum(0).tolist(),
                                command_jump_p95_rad=float(np.quantile(abs(np.diff(command, axis=0)), .95))
                                    if len(command) > 1 else 0.)
    row.update(candidate_id=job["candidate_id"], variant=job["variant"],
               output=str(folder), seed=job["seed"], tube=job["tube"])
    limit = summary.get("finger_actuator_config", {}).get("effort_limit_nm", 35.)
    row["torque"] = torque_report(folder, summary.get("actual_motion_start_step"), limit)
    row["comparison_scope"] = "Same reference/seed/tube; new actuator profile applies during settling, so initial physical states can differ"
    save(folder / "hardware_metrics.json", row)
    return row


def aggregate(rows):
    result = {}
    for variant in sorted({row["variant"] for row in rows}):
        group = [r for r in rows if r["variant"] == variant]
        complete = [r for r in group if r.get("status") == "completed"]
        result[variant] = dict(attempted=len(group), completed=len(complete),
            stable_starts=sum(r.get("stable_start", False) for r in group),
            strict_four_mean_full=float(np.mean([r.get("strict_healthy_four_rate_full", 0.) for r in group])),
            found_four_mean_full=float(np.mean([r.get("found_four_rate_full", 0.) for r in group])),
            over100_cases=sum((r.get("force_events") or {}).get("total_overforce_ms", 0) > 0 for r in group),
            strict95_complete_cases=sum(r.get("strict_healthy_four_rate_full", 0.) >= .95 for r in complete),
            max_force_n=max([r.get("peak_stop_sensor_force_n", 0.) for r in group]+[0.]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DATA/"leap_hardware_evaluation_20261016/full")
    parser.add_argument("--ids", type=int, nargs="+", default=DEFAULT_IDS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--profiles", nargs="+", choices=("continuous", "peak"), default=["continuous", "peak"])
    parser.add_argument("--interpolation-ablation", action="store_true")
    parser.add_argument("--leap-kp", type=float, default=35.)
    parser.add_argument("--leap-kd", type=float, default=.2)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = DATA / "healthy_regional_collection_20261016"
    jobs, rows = [], []
    for cid in args.ids:
        old = json.loads((source/"results"/f"candidate_{cid:04d}_r0.json").read_text())
        with h5py.File(old["reference"]) as handle:
            frames = len(handle["palm_pose_object"])
        base = dict(candidate_id=cid, frames=frames, seed=old["seed"], tube=old["tube"],
                    reference=old["reference"])
        legacy = dict(old, candidate_id=cid, variant="legacy", output=str(source/"rollouts"/f"candidate_{cid:04d}_r0"),
                      baseline_reused=True, comparison_scope="Historical complete same-reference rollout; physical initialization can differ")
        rows.append(legacy)
        variants = [(profile, True) for profile in args.profiles]
        if args.interpolation_ablation and cid in (62, 81) and "peak" in args.profiles:
            variants.append(("peak", False))
        for profile, interpolate in variants:
            variant = profile+"_interpolated" if interpolate else profile+"_staircase"
            folder = out/f"candidate_{cid:04d}_{variant}"
            command = [sys.executable, "-u", str(SCRIPTS/"collect_leap_hardware_probe.py"),
                       "--leap-profile", profile, "--plan", old["reference"], "--output", str(folder),
                       "--leap-kp", str(args.leap_kp), "--leap-kd", str(args.leap_kd),
                       "--frames", str(frames), "--stable-initialization", "--initialization-extra-steps", "1200",
                       "--force-profile", "regulated", "--force-audit", "--force-parts", "--substep-force-safety",
                       "--overforce-response", "record", "--seed", str(old["seed"])]
            if interpolate:
                command.append("--object-substep-interpolation")
            if old["tube"]:
                command += ["--execution-randomization", "--execution-clean-env-fraction", "0"]
            jobs.append(dict(base, variant=variant, output=str(folder), command=command))
    files = ("run_leap_hardware_evaluation.py", "collect_leap_hardware_probe.py", "leap_hardware_profile.py",
             "collect_long_oracle_probe.py", "collect_trajectories.py", "surface_mcc_finger.py",
             "collection_force_regulation.py", "collect_regional_healthy_reference.py")
    protocol = dict(jobs=jobs, source_sha256={name: hashlib.sha256((SCRIPTS/name).read_bytes()).hexdigest() for name in files},
                    primary_metrics="Unchanged true-tip strict contact and found, full-length denominator, 100N record-and-continue",
                    attribution="Combined actuator/profile and object motion change; two staircase ablations; no identical-state causal claim",
                    production_collection_resumed=False)
    path = out/"protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise RuntimeError("Protocol changed; select a new output directory")
    save(path, protocol)
    if args.prepare_only:
        print(json.dumps(dict(jobs=len(jobs), legacy_references=len(rows), output=str(out))))
        return
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               MPLCONFIGDIR="/tmp/matplotlib", WARP_CACHE_PATH="/tmp/warp", TQDM_DISABLE="1")
    env["PYTHONPATH"] = str(ROOT/"src")+os.pathsep+env.get("PYTHONPATH", "")
    started = time.monotonic()

    def run(job):
        folder = Path(job["output"])
        folder.mkdir(exist_ok=True)
        save(folder/"launch.json", job)
        if not (folder/"summary.json").exists():
            print("[LEAP] running", job["candidate_id"], job["variant"], flush=True)
            with (folder/"run.log").open("w") as log:
                completed = subprocess.run(job["command"], cwd=ROOT, env=env, stdout=log,
                                           stderr=subprocess.STDOUT, timeout=2400)
            if not (folder/"summary.json").exists():
                raise RuntimeError(f"Exit {completed.returncode}: {folder}/run.log")
        row = analyze(job)
        print("[LEAP] done", job["candidate_id"], job["variant"], row["status"],
              f'strict={row["strict_healthy_four_rate_full"]:.4f} found={row["found_four_rate_full"]:.4f}', flush=True)
        return row

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                rows.append(future.result())
            except Exception as error:
                rows.append(dict(candidate_id=job["candidate_id"], variant=job["variant"], error=str(error)))
            result = dict(completed_new=len(rows)-len(args.ids), total_new=len(jobs),
                          elapsed_s=time.monotonic()-started, aggregates=aggregate(rows), rows=rows)
            save(out/"status.json", result)
    save(out/"analysis.json", result)


if __name__ == "__main__":
    main()
