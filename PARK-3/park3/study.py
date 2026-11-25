"""Run only registered PARK-3 acquisitions and score them with the historical evaluator."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from itertools import count
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
from scipy.spatial import cKDTree

from Module.palmreach6.__main__ import FIXTURES
from .common import HERE, OUTPUT, PREVIEW, ROOT, digest, planning, read, write, table
from .trajectory import extrema, limits

METHODS = {"AUDE": "aude", "PalmReach-6": "palmreach-6", "PARK-OLD": "park-D"}


def cells(cfg, suite):
    settings = cfg["experiments"]
    fixtures = read(FIXTURES / "fixtures.json")["objects"]
    if suite in ("main", "all"):
        for seconds, repeats in settings["main"]:
            for repeat in range(repeats):
                for obj in settings["objects"]:
                    yield seconds, obj, "D", fixtures[obj]["seed"]+repeat
    if suite in ("ablation", "all"):
        for repeat in range(settings["ablation_repeats"]):
            for variant in settings["variants"]:
                for obj in settings["objects"]:
                    yield settings["ablation_seconds"], obj, variant, fixtures[obj]["seed"]+repeat


def directory(root, cell):
    seconds, obj, variant, seed = cell
    return root / "trials" / f"{seconds}s" / obj / variant / f"seed-{seed}"


def check_run(source, obj, seconds, seed, method, variant=None):
    run = read(source / "run.json")
    fixture = read(FIXTURES / "fixtures.json")
    objspec = fixture["objects"][obj]
    sig = run["signature"]
    if (sig.get("obj", sig.get("object")) != obj or sig["seconds"] != seconds or sig["seed"] != seed
            or sig.get("paced", False) or run["steps"] != 100*seconds or run["seed"] != seed
            or run["initial_sha256"] != objspec["integration_state"] or run["model_sha256"] != objspec["model.mjb"]
            or run["offline_mesh_sha256"] != objspec["surface.obj"]):
        raise ValueError(f"Unmatched paired trial at {source}")
    if method == "AUDE":
        if sig["planner_feedback"] != "physical" or sig["period"] != 90 or sig["wrist_mcc"] or sig["finger_mcc"]:
            raise ValueError(f"Incorrect AUDE controller at {source}")
    else:
        p = run["policy"]
        if (sig["start"] != objspec["start"] or run["initial_file_sha256"] != objspec["initial.npz"]
                or run["robot_sha256"] != fixture["hand_sha256"] or p["checkpoint_sha256"] != fixture["checkpoint"]
                or p["inference_seed"] != seed or p["schedule"] != "100ms chunks, 50ms buffered handoff"
                or p["action_points"] != 16):
            raise ValueError(f"Unmatched deployed controller at {source}")
        if method == "PARK-OLD" and sig["arm"] != "park-D-independent-dp":
            raise ValueError(f"Unmatched archived PARK arm at {source}")
        if method == "PalmReach-6" and "palm_progress_recontact.py" not in sig["code"]:
            raise ValueError(f"Unmatched PalmReach-6 controller at {source}")
        if method == "PARK-3" and (sig["arm"] != f"PARK-3-{variant}-dp" or sig["code"]["method"] != method):
            raise ValueError(f"Unmatched PARK-3 arm at {source}")
    if (digest(source / "trace.npz") != run["trace_sha256"] or digest(source / "plans.jsonl") !=
            (run.get("plan_sha256") or run["hashes"]["plans.jsonl"])):
        raise ValueError(f"Raw trial checksum mismatch at {source}")
    return run


def archived(obj, seconds, seed, method):
    from Module.park import metrics
    key = METHODS[method]
    path = ROOT/"outputs/PARK/experiments"/f"{seconds}s"/obj/key/f"seed-{seed}"
    meta, score = read(path/"trial.json"), read(path/"metrics.json")
    source = (path/meta["source"]).resolve()
    run = check_run(source, obj, seconds, seed, method)
    if (meta != score["meta"] or meta["method"] != key or meta["object"] != obj or meta["seed"] != seed
            or meta["seconds"] != seconds or meta["trace_sha256"] != run["trace_sha256"]
            or score["signature"]["trace"] != run["trace_sha256"] or score["signature"]["plans"] != digest(source/"plans.jsonl")
            or score["signature"]["metrics"] != digest(Path(metrics.__file__))
            or score["signature"]["gpis"] != digest(ROOT/"outputs/PARK/models/gpis.pt")):
        raise ValueError(f"Unmatched historical score at {path}")
    return path, source, score, run


def log_rows(source):
    with (source/"plans.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def rate_check(source, cfg):
    with np.load(source/"trace.npz", allow_pickle=False) as trace:
        commands = trace["palm_command"]
        np.testing.assert_array_equal(trace["published_preview"][1:-PREVIEW], commands[1+PREVIEW:])
        peak = extrema(np.vstack((commands[:1], commands)))
        if (peak > limits(cfg)*(1+2e-6)).any():
            raise ValueError(f"Published path exceeded rate bound: {peak}")
        return peak.tolist()


def telemetry(source, method, initial):
    from Module.park.metrics import distribution
    from collections import Counter
    rows = list(log_rows(source))
    keys = ["planning_ms", "candidate_ms", "evaluation_ms", "gpis_update_ms", "gpis_query_ms", "kde_ms", "representation_ms"]
    stats = {key: distribution([r[key] for r in rows if r.get(key, 0) > 0]) for key in keys}
    with np.load(source/"control-timing.npz", allow_pickle=False) as data:
        stats["servo_service_ms"] = distribution(data["service_ms"])
    return stats, dict(Counter(r["reason"] for r in rows)), None


def patch_outcomes(source, radius):
    selected = [r for r in log_rows(source) if r["reason"] == "new_target"]
    with np.load(source/"trace.npz", allow_pickle=False) as trace:
        points = trace["points_m"]
        valid = trace["pad_valid"] & (trace["region_load_n"].max(1) < 8)[:, None]
    patches = []
    for i, row in enumerate(selected):
        start = row["splice_tick"]
        stop = min(len(points)-1, row["end_tick"], selected[i+1]["splice_tick"] if i+1<len(selected) else len(points)-1)
        if start >= stop:
            continue
        window = slice(start+1, stop+1)
        hits = points[window][valid[window]]
        observed = 0.
        if len(hits):
            distance = cKDTree(hits).query(row["surface"])[0]
            observed = float(np.average(distance <= radius, weights=row["variance_m2"]))
        patches.append(dict(tick=row["tick"], start=start, stop=stop, trigger=row["trigger"],
                            predicted=row["predicted"], observed=observed))
    return patches


def score_one(source, dest, cell, models, cfg):
    from Module.park import metrics
    import torch
    torch.set_num_threads(1)
    seconds, obj, variant, seed = cell
    run = check_run(source, obj, seconds, seed, "PARK-3", variant)
    signed = run["signature"]["code"]
    if signed["config"] != cfg["planner"] or any(digest(models/k) != sha for k, sha in signed["models"].items()):
        raise ValueError(f"Trial settings or frozen models changed at {source}")
    dest.mkdir(parents=True, exist_ok=True)
    extra = dest/"deployment.json"
    if extra.is_file() and (dest/"metrics.json").is_file():
        cached, result = read(extra), read(dest/"metrics.json")
        if (cached["trace_sha256"] == result["signature"]["trace"] == run["trace_sha256"]
                and cached["adapter_sha256"] == digest(Path(__file__))
                and all((dest/k).is_file() for k in result["files"])):
            return
    peak = rate_check(source, planning(cfg))
    first = read(FIXTURES/"fixtures.json")["objects"][obj]["seed"]
    meta = dict(method="PARK-3" if variant == "D" else f"PARK-3-{variant}", object=obj, seconds=seconds, seed=seed,
                repeat=seed-first, variant=variant, source=os.path.relpath(source, dest), archived=False,
                trace_sha256=run["trace_sha256"], initial_sha256=run["initial_sha256"], model_sha256=run["model_sha256"])
    # The exact same offline evaluator and evaluation GP are used for every method.
    with patch.object(metrics, "timing", telemetry):
        metrics.score(source, dest, meta, ROOT/"outputs/PARK/models")
    with np.load(models/"index.npz", allow_pickle=False) as data:
        radius = float(data["radius"])
    patches = patch_outcomes(source, radius)
    write(extra, dict(trace_sha256=run["trace_sha256"], adapter_sha256=digest(Path(__file__)),
                      rate_maxima=peak, radius_m=radius, patches=patches))
    write(dest/"trial.json", meta)


@contextmanager
def claim(folder):
    # Atomic mkdir prevents duplicate launches from two matrix invocations.
    lock = folder/"in-progress"
    lock.mkdir()
    try:
        yield
    finally:
        lock.rmdir()


def acquire(cell, args, cfg):
    root = args.out or OUTPUT
    where = directory(root, cell)
    where.mkdir(parents=True, exist_ok=True)
    seconds, obj, variant, seed = cell
    completed = where/"acquired.json"
    with claim(where):
        if completed.is_file():
            native = where/read(completed)["source"]
            check_run(native, obj, seconds, seed, "PARK-3", variant)
        else:
            attempt = next(where/f"attempt-{i:03d}" for i in count(1) if not (where/f"attempt-{i:03d}").exists())
            attempt.mkdir()
            args_run = [sys.executable, "-B", "-u", str(HERE/"run.py"), "run", "--object", obj, "--variant", variant,
                        "--seconds", str(seconds), "--seed", str(seed), "--out", str(attempt/"native"),
                        "--models", str(args.models), "--config", str(args.config)]
            with (attempt/"acquisition.log").open("w", encoding="utf-8") as stream:
                code = subprocess.call(args_run, stdout=stream, stderr=subprocess.STDOUT)
            if code:
                write(attempt/"failure.json", dict(exit_code=code, cell=cell))
                raise RuntimeError(f"Acquisition failed at {attempt}")
            native = attempt/"native"
            run = check_run(native, obj, seconds, seed, "PARK-3", variant)
            write(completed, dict(source=native.relative_to(where).as_posix(), trace_sha256=run["trace_sha256"]))
        print(f"Acquired {obj} {variant} {seconds}s {seed}", flush=True)


def matrix(args, cfg):
    if not 1 <= args.jobs <= 8:
        raise ValueError("Choose 1 to 8 native acquisition processes")
    work = list(cells(cfg, args.suite))
    failures = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        from concurrent.futures import as_completed
        futures = {pool.submit(acquire, c, args, cfg): c for c in work}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as err:
                failures.append((futures[future], str(err)))
                print(failures[-1], flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} registered acquisitions failed")


def score_existing(args, cfg):
    from time import sleep
    root = args.out or OUTPUT
    pending = list(cells(cfg, args.suite))
    while pending:
        waiting = []
        for cell in pending:
            folder = directory(root, cell)
            if not (folder/"acquired.json").is_file():
                waiting.append(cell)
                continue
            score_one(folder/read(folder/"acquired.json")["source"], folder/"score", cell, args.models, cfg)
            print("Scored", *cell, flush=True)
        if not args.follow:
            break
        pending = waiting
        if pending:
            sleep(12)
