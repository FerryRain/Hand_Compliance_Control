"""Audit all five conditions, the paired recordings, and the two new controller interventions."""

import argparse
import csv
import json
import sys
from contextlib import ExitStack
from pathlib import Path
from zipfile import Path as ZipPath
from zipfile import ZipFile

from PARK.identity import read

from . import report as base
from . import report_extended as extended
from .height_evidence import add_baseline_height
from .verify import checked_bytes, verify_trial


def check_deployment(run, obj, condition, release):
    pinned = release["runtime_dependencies"]
    fixtures = base.FIXTURES.relative_to(base.ROOT).as_posix()
    if (run["initial_file_sha256"] != pinned[f"{fixtures}/{obj}/initial.npz"]
            or run["robot_sha256"] != pinned[f"{fixtures}/hand.mjb"]):
        raise ValueError(f"Wrong physical initial file or kinematic hand model for {obj}/{condition}")
    if condition in extended.MCC_CONDITIONS:
        policy = run["policy"]
        expected = len(range(policy["bootstrap_frames"], base.TICKS - 5, 10))
        if (run["actuation"] != "LEAP position servos with fingertip ForceFeedback" or
                policy["bootstrap_frames"] != 75 or policy["queries"] != expected):
            raise ValueError(f"Wrong released LEAP actuator or single_all handoff schedule for {obj}")
    elif run["actuation"] != "direct torque on original LEAP hinges":
        raise ValueError(f"Wrong lower controller for the pure AUDE height experiment on {obj}")


def open_runs(stack, source):
    root = ZipPath(stack.enter_context(ZipFile(source))) if source.is_file() else source
    return root / "runs" if (root / "runs").is_dir() else root


def verify_tables(folder, rows, result):
    public = read(folder / "results.json")
    if any(public.get(k) != v for k, v in result.items()):
        raise ValueError(f"The {len(rows)}-run summary differs from the verified individual scores")
    if "pear_video" in public:
        pear = {r["method"]: r for r in rows if r["object"] == "pear"}
        for name, clip in public["pear_video"].items():
            if clip["trace_sha256"] != pear[name]["trace_sha256"] or clip["endpoint_sha256"] != pear[name]["endpoint_sha256"]:
                raise ValueError(f"Pear video differs from the scored trajectory for {name}")
    index = {(r["object"], r["method"]): r for r in rows}
    columns = set().union(*(r.keys() for r in rows))
    with (folder / "trials.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if set(reader.fieldnames or []) != columns:
            raise ValueError(f"The {len(rows)}-run CSV columns differ from the scoring receipts")
        for row in reader:
            source = index.pop((row["object"], row["method"]))
            expected = {k: "" if source.get(k) is None else str(source[k]) for k in columns}
            if expected != row:
                raise ValueError(f"CSV evidence differs for {row['object']}/{row['method']}")
    if index:
        raise ValueError(f"Missing acquisitions from the {len(rows)}-run CSV")


def verify_additional_trial(folder, obj, condition, release, checkpoint, *, geometry=False):
    run = json.loads((folder / "run.json").read_bytes())
    check_deployment(run, obj, condition, release)
    for name, fingerprint in run["hashes"].items():
        if Path(name).name != name:
            raise ValueError(f"Nonportable raw asset name at {obj}/{condition}")
        if name not in ("plans.jsonl", "trace.npz", "timing.npz"):
            checked_bytes(folder / name, fingerprint)
    return verify_trial(folder, obj, condition, release, checkpoint, geometry=geometry,
                        controller_check=extended.validate_controller, trace_check=extended.condition_evidence)


def verify_all(original, additional, *, geometry=False):
    release = read(base.ROOT / "PARK/release.json")
    checkpoint = read(base.ROOT / "PARK/contract.json")["finger_policy"]["checkpoint_sha256"]
    rows = []
    for obj in base.OBJECTS:
        signatures = set()
        for condition in extended.CONDITIONS:
            if condition in extended.ADDITIONAL:
                folder = additional / obj / condition
                row, signature = verify_additional_trial(folder, obj, condition, release, checkpoint, geometry=geometry)
            else:
                folder = original / obj / condition
                row, signature = verify_trial(folder, obj, condition, release, checkpoint, geometry=geometry)
            if condition == "planned":
                row = add_baseline_height(row, folder)
            rows.append(row)
            signatures.add(signature)
        if len(signatures) != 1:
            raise ValueError(f"The initial physical measurements or AUDE planner differ across conditions for {obj}")
    return rows, extended.summarize(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=base.ROOT_OUT / "runs",
                        help="Directory or trajectory ZIP with the original 15 runs")
    parser.add_argument("--new", type=Path, default=extended.ROOT_OUT / "runs",
                        help="Directory or trajectory ZIP with the ten additional runs")
    parser.add_argument("--report", type=Path, help="Also verify extended results.json and trials.csv")
    parser.add_argument("--geometry", action="store_true", help="Recompute coverage and reconstruction independently")
    args = parser.parse_args()
    with ExitStack() as stack:
        original = open_runs(stack, args.base)
        additional = original if args.base == args.new else open_runs(stack, args.new)
        rows, result = verify_all(original, additional, geometry=args.geometry)
    if args.report:
        verify_tables(args.report, rows, result)
    fields = ("c25", "f5", "ge2", "ge3", "precision5", "recall5")
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(("object", "method", *fields))
    for r in rows:
        writer.writerow((r["object"], r["method"], *(r[k] for k in fields)))
    for name, mean in result["macro_mean"].items():
        writer.writerow(("mean", name, *(mean[k] for k in fields)))


if __name__ == "__main__":
    main()
