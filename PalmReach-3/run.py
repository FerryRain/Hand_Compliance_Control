"""Run the controlled planner replacement and compare matched physical trials."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from functools import partial
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "PARK-3"))

from park3.common import configuration, digest, planning, read, rot, write

CONFIG = read(HERE / "config.json")
OUT = ROOT / "outputs/PalmReach-3"
MODELS = ROOT / CONFIG["models"]


def frozen_sources():
    from Module.palmreach6.__main__ import ROOT as release
    local = read(release / "Module/palmreach6/source-sha256.json")
    local.update(read(ROOT / "outputs/PARK-NEW/execution-contract.json")["original_files"])
    if bad := [name for name, expected in local.items() if digest(release / name) != expected]:
        raise ValueError(f"Native controller source differs from release: {bad}")
    planner_files = [ROOT / "PARK-3/park3" / (name + ".py") for name in
                     ("__init__", "belief", "common", "deployment", "geometry", "planner", "probe", "trajectory")]
    files = [*planner_files, ROOT / CONFIG["planner"],
             *(MODELS / n for n in ("gpis.pt", "index.npz", "middle.npz", "ring.npz", "thumb.npz")),
             HERE / "config.json", HERE / "planner_adapter.py"]
    return local, files


def acquire(obj, seconds, repeat, dest, smoke=False, revision="v1"):
    from Module.palmreach6.__main__ import FIXTURES
    from Module.park.compare import run_baseline
    from Module.wrist_active import palm_exploration, palm_progress_recontact, policy_reach, tactile_field
    from planner_adapter import PalmWrist, PalmWorker
    from park3.study import check_run

    local, files = frozen_sources()
    wrist_type, worker_type, extra = PalmWrist, PalmWorker, {}
    revision_files = ()
    if revision in ("v2","v3"):
        from frontier_planner import FrontierWrist, FrontierWorker
        revision_files = (HERE / "frontier_planner.py", HERE / "frontier.json")
        wrist_type, worker_type = FrontierWrist, FrontierWorker
        if revision == "v3":
            from measured_start import MeasuredStartWrist, MeasuredStartWorker
            revision_files += (HERE / "measured_start.py",)
            wrist_type, worker_type = MeasuredStartWrist, MeasuredStartWorker
        files.extend(revision_files)
        extra = dict(frontier=read(HERE / "frontier.json"))
    entrypoint_sha = digest(Path(__file__))
    fixture = read(FIXTURES / "fixtures.json")["objects"][obj]
    seed = fixture["seed"] + repeat
    signature = dict(protocol_version=1, method=CONFIG["method"], variant=CONFIG["variant"], object=obj, seconds=seconds, seed=seed,
                     files={p.relative_to(ROOT).as_posix(): digest(p) for p in files}, native=local)
    if revision != "v1":
        signature["revision"] = revision
    if dest.exists():
        record = read(dest / "adapter.json")
        previous = dict(record["signature"])
        # Early v1 receipts also signed every plotting/reporting script in the
        # planner directory. Compare the physical inputs across both v1 formats.
        previous.setdefault("protocol_version", 1)
        previous["files"] = {k: previous["files"].get(k) for k in signature["files"]}
        if previous != signature:
            raise ValueError(f"Different run already exists at {dest}")
        check_run(dest, obj, seconds, seed, CONFIG["method"])
        for name, expected in record["telemetry"].items():
            if digest(dest / name) != expected:
                raise ValueError(f"Telemetry checksum mismatch: {dest / name}")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=dest.parent) as tmp:
        stage = Path(tmp)
        native = stage / "native"
        global_log = stage / "global-plans.jsonl"
        raw_run = tactile_field.run
        cfg = planning(configuration(ROOT / CONFIG["planner"]))
        with global_log.open("w", encoding="utf-8") as events, ExitStack() as scope:
            record = SimpleNamespace(nominal=[], events=events)
            def integrated(*args, wrist_cls, worker_cls, source_files, release, **kwargs):
                if wrist_cls.func is not palm_progress_recontact.ProgressWrist or worker_cls is not palm_progress_recontact.ProgressWorker:
                    raise ValueError("Unexpected native controller entry point")
                controller = partial(wrist_type, models=MODELS, cfg=cfg, variant=CONFIG["variant"], recorder=record, **extra)
                return raw_run(*args, wrist_cls=controller, worker_cls=worker_type,
                               source_files=(*source_files, HERE / "planner_adapter.py", HERE / "config.json", *revision_files),
                               release=f"palmreach-3-{revision+'-' if revision != 'v1' else ''}{release}", **kwargs)
            scope.enter_context(patch.object(tactile_field, "run", integrated))
            if smoke:
                # Tripwires cover both the ordinary selection and a fallback from mesh recontact.
                def forbidden(*args, **kwargs):
                    raise AssertionError("PalmReach-6 global action selection was invoked")
                for owner, key in ((palm_exploration, "palm_headings"), (policy_reach.PolicyReach, "predict"),
                                   (palm_exploration, "joint_gain")):
                    scope.enter_context(patch.object(owner, key, forbidden))
            # Reuse the frozen fixture, cracker profile and both original seed
            # injection points. The hook above is reached before constructing a wrist.
            run_baseline(obj, "palmreach-6", seconds, native, seed)
        run = check_run(native, obj, seconds, seed, CONFIG["method"])
        if read(native / "clock.json")["release"][:12] != "palmreach-3-" or not record.nominal:
            raise ValueError("PalmReach-3 did not execute")
        nominal = np.stack(record.nominal)
        with np.load(native / "trace.npz", allow_pickle=False) as trace:
            actual, command = trace["qpos"][:, :7], trace["palm_command"]
        if nominal.shape != actual.shape or not np.isfinite(nominal).all():
            raise ValueError("Incomplete nominal wrist telemetry")
        translation = np.linalg.norm(nominal[:, :3] - command[:, :3], axis=1)
        angle = (rot(command) * rot(nominal).inv()).magnitude()
        np.savez_compressed(native / "wrist.npz", nominal=nominal, corrected_command=command, actual=actual,
                            nominal_command_distance_m=translation, nominal_command_angle_rad=angle)
        global_log.rename(native / global_log.name)
        rows = [json.loads(line) for line in (native / "global-plans.jsonl").read_bytes().splitlines()]
        with (native / "plans.jsonl").open(encoding="utf-8") as stream:
            fitted = sum("palm_height_fit_mm" in json.loads(line) for line in stream)
        accepted = sum(row["reason"] == "new_target" for row in rows)
        if smoke and (not accepted or not fitted):
            raise ValueError("Smoke trial did not exercise a selected target and native palm support fit")
        write(native / "adapter.json", dict(signature=signature, entrypoint_sha256=entrypoint_sha,
              trace_sha256=run["trace_sha256"],
              telemetry={n: digest(native / n) for n in ("global-plans.jsonl", "wrist.npz")},
              smoke_tripwires=smoke, decisions=len(rows), new_targets=accepted, support_fits=fitted))
        native.rename(dest)


def trial_dir(root, seconds, obj, repeat):
    from Module.palmreach6.__main__ import FIXTURES
    seed = read(FIXTURES / "fixtures.json")["objects"][obj]["seed"] + repeat
    return root / "trials" / f"{seconds}s" / obj / f"seed-{seed}"


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action", choices=("smoke", "trial", "matrix", "score", "report", "video", "compare"))
    cases = cli.add_mutually_exclusive_group()
    cases.add_argument("--object", choices=CONFIG["objects"])
    cases.add_argument("--objects", nargs="+", choices=CONFIG["objects"], help="objects for a matched baseline comparison")
    cli.add_argument("--seconds", type=int, default=240)
    cli.add_argument("--repeat", type=int, choices=range(CONFIG["repeats"]), default=0)
    cli.add_argument("--budgets", nargs="+", type=int, choices=CONFIG["budgets_s"], default=CONFIG["budgets_s"])
    cli.add_argument("--jobs", type=int, choices=range(1, 7), default=2)
    cli.add_argument("--revision", choices=("v1","v2","v3"), default="v3",
                     help="v3 replans from the measured wrist when nominal start clearance fails")
    cli.add_argument("--out", type=Path)
    args = cli.parse_args()
    args.out = args.out or (OUT if args.revision == "v1" else OUT.with_name(f"PalmReach-3-{args.revision}"))
    if args.objects and args.action != "compare":
        cli.error("--objects is available for the compare action")
    if args.object is None and args.action not in ("video","compare"):
        args.object = "cracker"
    if args.seconds <= 0:
        cli.error("--seconds must be positive")
    if args.action == "smoke":
        acquire(args.object, 8, args.repeat, args.out / "smoke" / args.object, smoke=True, revision=args.revision)
    elif args.action in ("trial", "score"):
        dest = trial_dir(args.out, args.seconds, args.object, args.repeat)
        if args.action == "trial":
            acquire(args.object, args.seconds, args.repeat, dest / "native", revision=args.revision)
        from comparison import score_trial
        score_trial(dest)
    elif args.action == "matrix":
        cells = [(s, obj, r) for s in dict.fromkeys(args.budgets) for r in range(CONFIG["repeats"])
                 for obj in CONFIG["objects"]]
        def execute(cell):
            s, obj, r = cell
            folder = trial_dir(args.out, s, obj, r)
            folder.mkdir(parents=True, exist_ok=True)
            with (folder / "trial.log").open("a", encoding="utf-8") as log:
                subprocess.run([sys.executable, "-B", "-u", str(HERE / "run.py"), "trial", "--object", obj,
                                "--seconds", str(s), "--repeat", str(r), "--out", str(args.out),"--revision",args.revision],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            print(f"Scored {obj}, {s}s, repeat {r}", flush=True)
        with ThreadPoolExecutor(args.jobs) as pool:
            list(pool.map(execute, cells))
    elif args.action == "video":
        from replays import render
        render(args.out, args.seconds, args.object, args.repeat)
    elif args.action == "compare":
        from baseline_comparison import compare
        objects = list(dict.fromkeys(args.objects or ([args.object] if args.object else CONFIG["objects"])))
        compare(args.out, args.seconds, objects, args.repeat, args.revision)
    else:
        from comparison import report
        report(args.out, args.revision)


if __name__ == "__main__":
    main()
