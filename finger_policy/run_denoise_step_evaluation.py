"""Evaluate three DDPM budgets on the same five frozen long references."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from pipeline import CHECKPOINTS, DATA, REPO, SCRIPTS, check_hash, digest, protocol, read, runtime_environment
from pipeline import write


GROUP = "single_all"
TRAIN_SEED = 20261116
STEPS = (10, 20, 50)
REFERENCES = {
    "mustard": "long_natural",
    "pear": "ycb_016_pear_forward",
    "bleach": "ycb_021_bleach_cleanser",
    "box": "ycb_003_cracker_box",
    "can": "ycb_002_master_chef_can",
}


def freeze():
    previous = Path(read(DATA / "provenance.json")["source_protocol"]).parent
    common = previous / "fresh_common_long_videos_20261118" / "protocol.json"
    final_video = previous / "realtime_long_rollout_videos_20261118" / "progress.json"
    archived = read(common)
    video_index = read(final_video)["videos"]
    training = protocol()
    index = read(CHECKPOINTS / "manifest.json")
    weights = index[f"{GROUP}_seed{TRAIN_SEED}"]
    checkpoint = CHECKPOINTS.parent / weights["path"]
    frozen = archived["frozen"].copy()
    frozen.update({
        str(path): digest(path) for path in (
            common, final_video, Path(__file__).resolve(),
            SCRIPTS / "controlled_policy_ablation.py",
            SCRIPTS / "evaluate_expanded_independent_job.py",
            SCRIPTS / "evaluate_controlled_policy_ablation.py",
        )
    })
    frozen[str(checkpoint)] = weights["sha256"]
    template_rows, cases = {}, {}
    for case, reference in REFERENCES.items():
        template_rows[reference] = training["request_templates"][reference]
        path = Path(template_rows[reference]["request"])
        request = read(path)
        result_path = Path(read(video_index[f"{case}/{GROUP}"]["metadata"])["result"])
        previous_request_path = result_path.with_suffix(".request.json")
        previous_request = read(previous_request_path)
        expected = dict(request, arm=f"{GROUP}_seed{TRAIN_SEED}_{reference}",
                        single_finger_policy=previous_request["single_finger_policy"],
                        comparison_checkpoint_sha256=weights["sha256"], output=previous_request["output"],
                        matched_progress_noise=False)
        expected.pop("finger_checkpoint_hashes", None)
        if (expected != previous_request or request["residual_mode"] != "off"
                or request["reference_prefix_only"]
                or request["seed"] != archived["references"][case]["deployment_seed"]
                or request["rows"][0]["frames"] != archived["references"][case]["frames"]):
            raise ValueError(f"Reference differs from the previously reviewed deployment: {case}")
        frozen[str(path)] = digest(path)
        frozen[str(previous_request_path)] = digest(previous_request_path)
        cases[case] = dict(reference=reference, seed=request["seed"],
                           frames=request["rows"][0]["frames"], source_request=str(previous_request_path))
    return dict(group=GROUP, training_seed=TRAIN_SEED, checkpoint=str(checkpoint),
                checkpoint_sha256=weights["sha256"], denoise_steps=list(STEPS),
                request_templates=template_rows, cases=cases, frozen_sha256=frozen,
                metric_scope="Full executed frames include startup and built-in recovery. "
                             "Common prefix ends at each run's first arrival at the earliest final reference cursor.",
                noise="Same seed at each shared progress key; different DDPM schedules use a different number of draws.",
                latency="DP provider wall time per replan after initialization. "
                        "The GPU is shared by independently executing batch-1 simulator processes.",
                added_correction=False)


def verify_frozen(frozen):
    for path, expected in frozen["frozen_sha256"].items():
        check_hash(path, expected)


def run_job(out, frozen, case, steps, canary):
    root = out / (f"preflight_{steps}" if canary else f"steps_{steps}")
    root.mkdir(exist_ok=True)
    protocol_path = root / "protocol.json"
    # The coordinator prepares these files before dispatching concurrent jobs.
    assert protocol_path.is_file()
    reference = REFERENCES[case]
    folder = root / ("canary_deployment" if canary else "evaluation") / f"{GROUP}_seed{TRAIN_SEED}"
    folder.mkdir(parents=True, exist_ok=True)
    args = [sys.executable, "-u", str(SCRIPTS / "evaluate_expanded_independent_job.py"),
            str(root), GROUP, str(TRAIN_SEED), reference,
            "--checkpoint", frozen["checkpoint"], "--denoise-steps", str(steps), "--record-latency"]
    if canary:
        args.append("--canary")
    verify_frozen(frozen)
    with (folder / f"{reference}.job.log").open("x") as log:
        subprocess.run(args, cwd=REPO, env=runtime_environment(), stdout=log, stderr=subprocess.STDOUT, check=True)
    return check_result(folder / f"{reference}.json", frozen, case, steps, canary)


def check_result(path, frozen, case, steps, canary):
    result = read(path)
    meta = result["single_finger_policy"]
    lane = result["lanes"][0]
    reference = frozen["cases"][case]
    request = read(path.with_suffix(".request.json"))
    previous = read(reference["source_request"])
    previous["output"] = request["output"]
    previous["single_finger_policy"].update(comparison_checkpoint=frozen["checkpoint"],
                                            denoise_steps=steps, record_latency=True)
    if canary:
        previous["reference_prefix_only"] = True
        previous["rows"][0]["frames"] = 200
        previous["palm_reset"]["extra_frames"] = 500
    if (request != previous or meta["sha256"] != frozen["checkpoint_sha256"]
            or meta["denoise_steps"] != steps or result["deployment"]["seed"] != reference["seed"]):
        raise ValueError(f"Execution settings differ: {path}")
    plan_events = meta["noise_schedule"]
    if not plan_events or len(plan_events) != meta["calls"] or any(
            event["branch_execution"] != "cuda_graph_streams" for event in plan_events):
        raise ValueError(f"Missing CUDA Graph inference records: {path}")
    latency = np.asarray([event["dp_latency_ms"] for event in plan_events])
    if not np.isfinite(latency).all() or np.any(latency <= 0):
        raise ValueError(f"Invalid inference timings: {path}")
    audit = read(path.with_suffix(".comparison_audit.json"))
    cursor = audit["last_reference_frame"]
    total = request["rows"][0]["frames"]
    if lane["complete"] != (cursor == total - 1) or bool(audit["canary_prefix_only"]) != canary:
        raise ValueError(f"Invalid reference completion record: {path}")
    full = lane["full"]
    return dict(case=case, steps=steps, result=str(path), trace=result["trace"],
                result_sha256=digest(path), trace_sha256=digest(result["trace"]),
                complete=lane["complete"], last_reference_cursor=cursor,
                wanted_reference_frames=total, actual_control_frames=lane["frames"],
                completion=(cursor + 1) / total, wall_duration_s=result["deployment"]["duration_s"],
                terminal_event=result["palm_reset"]["events"][-1],
                strict_four=full["strict_healthy_four_rate"], found_four=full["found_four_rate"],
                latency=dict(calls=len(latency), mean_ms=float(latency.mean()),
                             p50_ms=float(np.percentile(latency, 50)), p95_ms=float(np.percentile(latency, 95)),
                             max_ms=float(latency.max()), over_100_ms=int((latency > 100).sum())))


def paired_summary(rows):
    comparisons = {}
    for case in REFERENCES:
        group = {row["steps"]: row for row in rows if row["case"] == case}
        end = min(row["last_reference_cursor"] for row in group.values())
        common_prefix = {}
        progress_seeds = {}
        for steps, row in group.items():
            n = row["actual_control_frames"]
            with np.load(row["trace"]) as data:
                cursor = data["reset_reference_frame"][:n, 0]
                stop = int(np.flatnonzero(cursor >= end)[0]) + 1
                strict = data["strict"][:n, 0].all(-1)
                found = data["found"][:n, 0].all(-1)
                if not (np.isclose(strict.mean(), row["strict_four"])
                        and np.isclose(found.mean(), row["found_four"])):
                    raise ValueError(f"Contact rates do not match raw traces: {row['trace']}")
                common_prefix[steps] = dict(frames=stop, strict_four=float(strict[:stop].mean()),
                                            found_four=float(found[:stop].mean()))
            events = read(row["result"])["single_finger_policy"]["noise_schedule"]
            progress_seeds[steps] = {event["noise_key"]: event["seed"] for event in events}
        shared = []
        for a, b in combinations(STEPS, 2):
            common = progress_seeds[a].keys() & progress_seeds[b].keys()
            if not common or any(progress_seeds[a][key] != progress_seeds[b][key] for key in common):
                raise ValueError(f"DP seeds differ at shared progress keys: {case}, {a}/{b}")
            shared.append(dict(steps=[a, b], shared_progress_noise_keys=len(common)))
        comparisons[case] = dict(runs=group, common_prefix_reference_frames=end + 1,
                                 common_prefix=common_prefix, progress_seed_checks=shared)
    return comparisons


def main(out, workers):
    frozen = freeze()
    verify_frozen(frozen)
    out.mkdir(parents=True, exist_ok=False)
    write(out / "protocol.json", frozen)
    for steps in STEPS:
        for tag in ("preflight", "steps"):
            folder = out / f"{tag}_{steps}"
            folder.mkdir()
            write(folder / "protocol.json", frozen)
    started = time.time()
    state = dict(stage="preflight", started_at=started, target=15, preflight=[], results=[], errors=[])

    def save_progress():
        target = out / "progress.json"
        temp = target.with_suffix(".tmp")
        write(temp, state)
        temp.replace(target)

    save_progress()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for canary in (True, False):
            jobs = [("mustard", steps) for steps in STEPS] if canary else [
                (case, STEPS[(i + j) % len(STEPS)])
                for i, case in enumerate(REFERENCES) for j in range(len(STEPS))]
            state.update(stage="preflight" if canary else "running", scheduled=[list(job) for job in jobs])
            save_progress()
            tasks = {pool.submit(run_job, out, frozen, case, steps, canary): (case, steps)
                     for case, steps in jobs}
            for future in as_completed(tasks):
                case, steps = tasks[future]
                try:
                    row = future.result()
                    if canary and not row["complete"]:
                        raise ValueError("The 200-frame interface check did not finish")
                    state["preflight" if canary else "results"].append(row)
                    print(f"{case} {steps} steps: {'preflight' if canary else 'evaluation'} "
                          f"recorded, path {row['completion']:.1%}", flush=True)
                except Exception as exc:
                    state["errors"].append(dict(case=case, steps=steps, preflight=canary, error=repr(exc)))
                    print(f"{case} {steps} steps: {exc}", file=sys.stderr, flush=True)
                save_progress()
            if state["errors"]:
                state["stage"] = "failed"
                save_progress()
                raise RuntimeError(f"Evaluation failed. See {out / 'progress.json'}")
    verify_frozen(frozen)
    summary = dict(protocol=str(out / "protocol.json"), elapsed_s=time.time() - started,
                   result_count=len(state["results"]), comparison=paired_summary(state["results"]))
    write(out / "summary.json", summary)
    state.update(stage="complete", elapsed_s=summary["elapsed_s"])
    save_progress()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=4)
    args = parser.parse_args()
    main(args.out.resolve(), args.workers)
