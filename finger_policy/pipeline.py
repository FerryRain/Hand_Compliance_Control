"""Run the frozen finger-policy workflow from this directory."""

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


HOME_DIR = Path(__file__).resolve().parent
REPO = HOME_DIR.parent
SCRIPTS = HOME_DIR / "scripts"
SHARED = REPO / "mcc_finger_compliance_control" / "scripts"
DATA = HOME_DIR / "data"
PROTOCOLS = DATA / "protocols"
LOGS = HOME_DIR / "logs"
CHECKPOINTS = HOME_DIR / "checkpoints"
GROUPS = ("single_all", "single_common", "flat_common")
DEFAULT_SEED = 20261116


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check_hash(path, expected):
    if digest(path) != expected:
        raise ValueError(f"File differs from recorded SHA-256: {path}")


def verify():
    integrity = read(DATA / "integrity.json")
    checkpoints = read(CHECKPOINTS / "manifest.json")
    for name, expected in integrity.items():
        check_hash(HOME_DIR / name, expected)
    for info in checkpoints.values():
        check_hash(HOME_DIR / info["path"], info["sha256"])
    raw = read(DATA / "raw_collection" / "manifest.json")
    for info in raw.values():
        if not (HOME_DIR / info["link"] / "rollout.h5").is_file():
            raise FileNotFoundError(info["link"])
    print(f"Verified {len(integrity)} files, {len(checkpoints)} checkpoints and {len(raw)} raw-source links")


def runtime_environment():
    paths = [str(SCRIPTS), str(SHARED), str(REPO / "src")]
    if os.environ.get("PYTHONPATH"):
        paths.append(os.environ["PYTHONPATH"])
    return dict(os.environ, PYTHONPATH=os.pathsep.join(paths), MUJOCO_GL="egl")


def run_script(name, *args, stdout=None):
    subprocess.run(
        [sys.executable, "-u", str(SCRIPTS / name), *map(str, args)],
        cwd=REPO, env=runtime_environment(), check=True, stdout=stdout,
        stderr=subprocess.STDOUT if stdout else None,
    )


def run_id(value):
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", value):
        raise argparse.ArgumentTypeError("Use a simple run name with letters, digits, dot, dash or underscore.")
    return value


def list_collection():
    recipes = read(PROTOCOLS / "collection" / "recipes.json")
    for recipe, jobs in recipes.items():
        print(recipe, *(job["id"] for job in jobs), sep="\n  ")


def collect(args):
    recipes = read(PROTOCOLS / "collection" / "recipes.json")
    run = LOGS / "collection_runs" / args.run
    assets = DATA / "collection_runs" / args.run
    queue_path = run / "queue.json"
    if args.resume:
        queue = read(queue_path)
        if queue["recipe"] != args.recipe or args.job or args.prepare_only:
            raise ValueError("Resume uses the prepared recipe and its existing job list.")
    else:
        jobs = recipes[args.recipe]
        if args.job:
            requested = set(args.job)
            jobs = [job for job in jobs if job["id"] in requested]
            if {job["id"] for job in jobs} != requested:
                raise ValueError("Unknown collection job. See collection-list.")
        if run.exists() or assets.exists():
            raise FileExistsError(f"Existing collection run: {args.run}")
        plans = []
        for job in jobs:
            plan = read(HOME_DIR / job["protocol"])
            original_root = Path(plan["collector"]["root"])
            for name, expected in plan["source_hashes"].items():
                check_hash(SCRIPTS / name, expected)
            plans.append((job, plan, original_root))
        roots = {root for _, _, root in plans}
        if len(roots) != 1:
            raise ValueError("A collection run must use a common object pool.")
        root = roots.pop()
        assets.mkdir(parents=True)
        run.mkdir(parents=True)
        for name in ("objects", "object_manifest.json"):
            original = root / name
            (assets / name).symlink_to(os.path.relpath(original, assets), target_is_directory=original.is_dir())
        queued = []
        for job, plan, _ in plans:
            plan["collector"]["root"] = str(assets)
            plan_path = run / (job["id"] + ".protocol.json")
            write(plan_path, plan)
            folder = assets / Path(job["folder"]).relative_to(root)
            queued.append(dict(job, protocol=str(plan_path), folder=str(folder)))
        queue = dict(recipe=args.recipe, jobs=queued)
        write(queue_path, queue)
    if args.prepare_only:
        print(queue_path)
        return
    options = ["--queue", queue_path]
    if args.resume:
        options.append("--resume")
    if args.limit is not None:
        options.extend(("--limit", args.limit))
    run_script("run_contact_wm_collection_queue.py", *options)
    status = read(run / "status.json")
    if status.get("error") or any(not job["audited"] for job in status["jobs"]):
        raise RuntimeError(f"Collection or audit failed. See {run / 'status.json'}")


def protocol():
    value = read(PROTOCOLS / "training.json")
    for entry in value["request_templates"].values():
        entry["request"] = str(HOME_DIR / entry["request"])
    return value


def append_exported(manifest, extra_paths):
    """Extend the frozen split with approved measured rollouts."""
    known = {row["record_key"] for row in manifest["records"]}
    held = set(manifest["heldout_objects"])
    dev = manifest["validation_object"]
    trials = defaultdict(dict)
    for path in extra_paths:
        export = read(path)
        if not export["complete"]:
            raise ValueError(f"Incomplete measured-label export: {path}")
        for source in export["records"]:
            if not source["training_admissible"]:
                continue
            row = dict(source)
            key = f'{row["source_sha256"]}:{row["lane"]}:{row["finger"]}'
            if key in known:
                continue
            known.add(key)
            obj = row["object_parent"]
            split = "heldout" if obj in held else "val" if obj == dev else (
                "train" if row["split"] == "train" else "excluded_original_val"
            )
            row.update(record_key=key, assigned_split=split)
            row["shard"] = str((Path(path).parent / row["shard"]).resolve(strict=True))
            manifest["records"].append(row)
            trials[(row["source_sha256"], int(row["lane"]))][int(row["finger"])] = row

    if not trials:
        return

    import numpy as np

    for (source_hash, lane), fingers in trials.items():
        if set(fingers) != {0, 1, 2, 3}:
            continue
        rows = [fingers[i] for i in range(4)]
        obj, split = rows[0]["object_parent"], rows[0]["assigned_split"]
        palm = roots = None
        for row in rows:
            if (row["object_parent"], row["assigned_split"]) != (obj, split):
                raise ValueError("Same physical trial has inconsistent splits or objects.")
            with np.load(row["shard"]) as shard:
                current = shard["observation"][:, 33:]
                current_roots = shard["roots"]
                if palm is None:
                    palm, roots = current, current_roots
                elif not (np.array_equal(palm, current) and np.array_equal(roots, current_roots)):
                    raise ValueError("Four-hand trial has mismatched palm history or time roots.")
        manifest["whole_trials"].append(dict(
            object=obj, assigned_split=split, record_keys=[row["record_key"] for row in rows],
            source_sha256=source_hash, lane=lane,
        ))


def prepare_training(run, exports):
    data_index = DATA / "expanded" / "manifest.json"
    manifest = read(data_index)
    for row in manifest["records"]:
        row["shard"] = str(data_index.parent / row["shard"])
    append_exported(manifest, exports)
    p = protocol()
    local_index = run / "manifest.json"
    write(local_index, manifest)
    p.update(data_directory=str(DATA / "old222"), expansion_manifest=str(local_index))
    inputs = read(DATA / "integrity.json")
    frozen = {str(HOME_DIR / name): expected for name, expected in inputs.items()
              if name.startswith(("data/old222/", "data/expanded/", "data/protocols/"))}
    sources = [SCRIPTS / name for name in (
        "single_finger_dp.py", "expanded_policy_data.py",
        "controlled_policy_ablation.py", "train_expanded_policy_comparison.py",
    )] + [SHARED / "train_dp.py", SHARED / "dp_dataset.py", Path(__file__).resolve()]
    frozen.update({str(path): digest(path) for path in [local_index, *exports, *sources]})
    for row in manifest["records"]:
        shard = row["shard"]
        if shard not in frozen:
            frozen[shard] = digest(shard)
    p["frozen_sha256"] = frozen
    path = run / "protocol.json"
    write(path, p)
    return path


def train(args):
    run = LOGS / "training_runs" / args.run
    out = CHECKPOINTS / "runs" / args.run
    if out.exists():
        raise FileExistsError(out)
    run.mkdir(parents=True, exist_ok=False)
    params = prepare_training(run, args.add_shards)
    with (run / "training.log").open("x") as log:
        run_script("train_expanded_policy_comparison.py",
                   "--protocol", params, "--group", args.group, "--seed", args.seed,
                   "--output", out, "--device", args.device,
                   "--canary-steps", args.canary_steps, stdout=log)
    print(out / "best.pt")


def evaluate(args):
    os.environ.update(runtime_environment())
    sys.path[:0] = [str(SCRIPTS), str(SHARED), str(REPO / "src")]
    from evaluate_expanded_independent_job import evaluate as run_evaluation

    p = protocol()
    if args.reference not in p["request_templates"]:
        raise ValueError("Unknown reference. See reference-list.")
    if args.checkpoint is None:
        weights = read(CHECKPOINTS / "manifest.json")
        key = f"{args.group}_seed{args.seed}"
        if key not in weights:
            raise ValueError(f"No local checkpoint for {key}. Pass --checkpoint with the archived weight path.")
        info = weights[key]
        checkpoint = HOME_DIR / info["path"]
        check_hash(checkpoint, info["sha256"])
    else:
        checkpoint = args.checkpoint.resolve(strict=True)
    out = DATA / "evaluation_runs" / args.run
    report = run_evaluation(out, p, args.group, args.seed, checkpoint, args.reference, canary=args.canary)
    print(json.dumps({k: report[k] for k in (
        "source", "requested_reference_frames", "last_reference_frame", "reference_completion_fraction",
    )}, indent=2))


def absolute(value):
    return Path(value).expanduser().resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify", help="Verify packaged inputs and checkpoints.")
    commands.add_parser("collection-list", help="List captured collection recipe names.")
    commands.add_parser("reference-list", help="List deployment evaluation references.")
    p = commands.add_parser("collect", help="Execute or stage a fixed mocap collection recipe.")
    p.add_argument("recipe", choices=("single_region", "single_banana", "multi", "whole"))
    p.add_argument("--run", type=run_id, required=True)
    p.add_argument("--job", action="append")
    p.add_argument("--limit", type=int)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--resume", action="store_true")
    p = commands.add_parser("index", help="Review audited measured-q candidates.")
    p.add_argument("--audit-dirs", type=absolute, nargs="+", required=True)
    p.add_argument("--output", type=absolute, required=True)
    p = commands.add_parser("export", help="Export admitted measured trajectories as finger shards.")
    p.add_argument("--manifest", type=absolute, required=True)
    p.add_argument("--output", type=absolute, required=True)
    p = commands.add_parser("train", help="Train with the frozen budget and split.")
    p.add_argument("--group", choices=GROUPS, default="single_all")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--run", type=run_id, required=True)
    p.add_argument("--canary-steps", type=int, default=0)
    p.add_argument("--add-shards", type=absolute, action="append", default=[])
    p = commands.add_parser("evaluate", help="Deploy and score a checkpoint using a frozen long reference.")
    p.add_argument("--group", choices=GROUPS, default="single_all")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--run", type=run_id, required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--checkpoint", type=absolute)
    p.add_argument("--canary", action="store_true")
    p = commands.add_parser("render", help="Record a 1x measured-state replay.")
    p.add_argument("--result", type=absolute, required=True)
    p.add_argument("--output", type=absolute, required=True)
    p.add_argument("--title", default="Finger policy")
    args = parser.parse_args()
    if args.command == "verify":
        verify()
    elif args.command == "collection-list":
        list_collection()
    elif args.command == "reference-list":
        for name in protocol()["request_templates"]:
            print(name)
    elif args.command == "collect":
        collect(args)
    elif args.command == "index":
        run_script("prepare_expanded_finger_candidates.py",
                   "--audit-dirs", *args.audit_dirs, "--output", args.output)
    elif args.command == "export":
        run_script("export_wm_finger_candidates.py", "--manifest", args.manifest, "--output", args.output)
    elif args.command == "train":
        train(args)
    elif args.command == "evaluate":
        evaluate(args)
    elif args.command == "render":
        args.output.mkdir(parents=True, exist_ok=False)
        run_script("render_focused_policy_video.py", "--result", args.result, "--output", args.output,
                   "--title", args.title)


if __name__ == "__main__":
    main()
