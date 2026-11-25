"""Acquire and score matched, resumable five-object experiments using native executors."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

from Module.palmreach6 import __main__ as release

from .settings import write_json


BASELINES = {"palmreach-6": "profiled", "facing-4": "facing4", "aude": "aude"}
MAIN = ("park-D", *BASELINES)
ABLATIONS = ("park-A", "park-B", "park-C", "park-D", "park-max")


def native_directory(trial, meta):
    source = meta.get("source", ".").replace("\\", "/")
    return (trial / source).resolve()


def run_baseline(obj, method, seconds, out, seed):
    from Module.acitive_02 import route_free_inputs
    from Module.wrist_active import informative_run, tactile_field

    manifest_path = release.FIXTURES / "fixtures.json"
    manifest = json.loads(manifest_path.read_bytes())
    native_seed = manifest["objects"][obj]["seed"]
    if seed == native_seed:
        return release.run(obj, out, seconds, method)
    # The release CLI binds each fixture to its published seed. Override only that
    # experimental parameter at the two native read boundaries, in this trial process.
    manifest["objects"][obj]["seed"] = seed
    manifest["fixtures"][obj + "/start-02"]["seed"] = seed
    def with_seed(original):
        def read(path):
            return manifest if Path(path).resolve() == manifest_path.resolve() else original(path)
        return read
    with (patch.object(informative_run, "read", with_seed(informative_run.read)),
          patch.object(route_free_inputs, "read_json", with_seed(route_free_inputs.read_json)),
          patch.object(informative_run, "seed_for", return_value=seed),
          patch.object(tactile_field, "seed_for", return_value=seed)):
        return release.run(obj, out, seconds, method)


def describe(args, obj, method, seconds, repeat):
    spec = json.loads((release.FIXTURES / "fixtures.json").read_bytes())
    seed = spec["objects"][obj]["seed"] + repeat
    directory = args.out / f"{seconds}s" / obj / method / f"seed-{seed}"
    source = directory / "native"
    archived = False
    if repeat == 0 and method in BASELINES and args.archive_root:
        archive = args.archive_root / "runs/full" / obj / BASELINES[method] / f"{seconds}s"
        if (archive / "run.json").exists():
            archived, source = True, archive
    return seed, directory, source, archived


def trial(args):
    import torch
    torch.set_num_threads(1)
    method, obj, seconds = args.method, args.object, args.seconds
    seed, directory, source, archived = describe(args, obj, method, seconds, args.repeat)
    directory.mkdir(parents=True, exist_ok=True)
    if args.phase != "score" and not archived:
        if method in BASELINES:
            run_baseline(obj, method, seconds, source, seed)
        else:
            from .deployment import run
            variant = "D" if method == "park-max" else method[-1]
            run(obj, source, seconds, args.models, seed=seed, variant=variant,
                coverage="max" if method == "park-max" else "independent", config=args.config)
    rec = json.loads((source / "run.json").read_bytes())
    expected = json.loads((release.FIXTURES / "fixtures.json").read_bytes())
    fixture = expected["objects"][obj]
    if (rec["seed"] != seed or rec["steps"] != seconds * 100 or rec["model_sha256"] != fixture["model.mjb"]
            or rec["initial_sha256"] != fixture["integration_state"] or rec["offline_mesh_sha256"] != fixture["surface.obj"]):
        raise ValueError(f"Unmatched acquisition at {source}")
    if method != "aude" and (rec["policy"]["checkpoint_sha256"] != expected["checkpoint"]
                              or rec["policy"]["inference_seed"] != seed):
        raise ValueError("Different Finger Policy or sampler seed")
    meta = dict(method=method, object=obj, seconds=seconds, seed=seed, repeat=args.repeat,
                source=Path(os.path.relpath(source, directory)).as_posix(),
                archived=archived, trace_sha256=rec["trace_sha256"],
                initial_sha256=rec["initial_sha256"], model_sha256=rec["model_sha256"])
    write_json(directory / "trial.json", meta)
    if args.phase != "acquire":
        from .metrics import score
        score(source, directory, meta, args.models)
    print(f"{method} {obj} {seconds}s seed {seed} {args.phase}", flush=True)


def cells(args):
    if args.suite in ("main", "all"):
        for seconds in args.budgets:
            for repeat in range(args.repeats):
                for obj in release.OBJECTS:
                    for method in MAIN:
                        yield obj, method, seconds, repeat
    if args.suite in ("ablations", "all"):
        for repeat in range(args.repeats):
            for obj in release.OBJECTS:
                for method in ABLATIONS:
                    yield obj, method, args.ablation_seconds, repeat


def matrix(args):
    args.out.mkdir(parents=True, exist_ok=True)
    logs = args.out / "logs"
    logs.mkdir(exist_ok=True)
    def execute(cell):
        obj, method, seconds, repeat = cell
        name = f"{obj}-{method}-{seconds}-r{repeat}"
        command = [sys.executable, "-B", "-u", "-m", "Module.park.compare", "trial", "--object", obj,
                   "--method", method, "--seconds", str(seconds), "--repeat", str(repeat),
                   "--models", str(args.models.resolve()), "--out", str(args.out.resolve()), "--phase", args.phase]
        for flag, value in (("--archive-root", args.archive_root), ("--config", args.config)):
            if value:
                command.extend((flag, str(value.resolve())))
        log = logs / f"{args.phase}-{name}.log"
        with log.open("w", encoding="utf-8") as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
        return name
    errors = []
    todo = list(dict.fromkeys(cells(args)))
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        tasks = {pool.submit(execute, cell): cell for cell in todo}
        for i, task in enumerate(as_completed(tasks), 1):
            try:
                print(i, "/", len(todo), task.result(), flush=True)
            except subprocess.CalledProcessError:
                errors.append(tasks[task])
                print("FAILED", tasks[task], flush=True)
    if errors:
        raise RuntimeError(f"Unfinished trials: {errors}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    one, all_jobs = sub.add_parser("trial"), sub.add_parser("matrix")
    for s in (one, all_jobs):
        s.add_argument("--out", required=True, type=Path)
        s.add_argument("--models", required=True, type=Path)
        s.add_argument("--archive-root", type=Path, help="Extracted retained_cracker_full_20261125 directory")
        s.add_argument("--config", type=Path)
        s.add_argument("--phase", choices=("acquire", "score", "both"), default="both")
    one.add_argument("--object", choices=release.OBJECTS, required=True)
    one.add_argument("--method", choices=tuple(dict.fromkeys((*MAIN, *ABLATIONS))), required=True)
    one.add_argument("--seconds", type=int, required=True)
    one.add_argument("--repeat", type=int, default=0)
    all_jobs.add_argument("--suite", choices=("main", "ablations", "all"), default="all")
    all_jobs.add_argument("--budgets", nargs="+", type=int, default=[240, 600])
    all_jobs.add_argument("--ablation-seconds", type=int, default=60)
    all_jobs.add_argument("--repeats", type=int, default=2)
    all_jobs.add_argument("--jobs", type=int, default=2)
    args = p.parse_args()
    if args.action == "matrix":
        if min(*args.budgets, args.ablation_seconds, args.repeats, args.jobs) <= 0:
            p.error("Durations, repeats and workers must be positive")
        matrix(args)
    else:
        if args.seconds <= 0 or args.repeat < 0:
            p.error("Trial duration must be positive and repeat index nonnegative")
        trial(args)


if __name__ == "__main__":
    main()

