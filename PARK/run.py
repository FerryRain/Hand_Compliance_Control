"""Run and verify the final PARK controller with its published experimental inputs."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    hasher = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            hasher.update(block)
    return hasher.hexdigest()


def verify_files(directory, expected):
    for name, checksum in expected.items():
        path = directory / name
        if not path.is_file() or digest(path) != checksum:
            raise ValueError(f"PARK release checksum mismatch: {path}")


def verify_results(release):
    for trial in release["published_runs"]:
        folder = ROOT / release["results_directory"] / trial["path"]
        receipt = read(folder / "native/adapter.json")
        sig = receipt["signature"]
        if (sig["method"], sig["revision"], sig["object"], sig["seconds"], sig["seed"]) != (
            release["recorded_method"], release["revision"], trial["object"], trial["seconds"], trial["seed"]
        ):
            raise ValueError(f"Trial identity changed: {folder}")
        if sig["files"] != release["planner_inputs"] or sig["native"] != release["controller_inputs"]:
            raise ValueError(f"Trial used different release inputs: {folder}")
        verify_files(folder, trial["files"])


def main():
    release = read(HERE / "release.json")
    cli = argparse.ArgumentParser(prog="PARK", description=__doc__)
    actions = cli.add_subparsers(dest="action", required=True)
    actions.add_parser("verify", help="Verify the release and the published raw traces, scores and meshes.")
    for action, help_text in (("trial", "Run and score PARK."), ("score", "Score an existing PARK acquisition.")):
        parser = actions.add_parser(action, help=help_text)
        parser.add_argument("--object", required=True, choices=release["seeds"])
        parser.add_argument("--seconds", type=int, default=600, choices=release["budgets_s"])
        parser.add_argument("--repeat", type=int, choices=(0, 1), default=0)
        parser.add_argument("--out", type=Path, default=ROOT / release["results_directory"])
    args = cli.parse_args()

    from Module.palmreach6.__main__ import ROOT as native_root

    verify_files(ROOT, release["planner_inputs"])
    verify_files(native_root, release["controller_inputs"])
    verify_files(ROOT, release["workspace_dependencies"])
    verify_files(native_root, release["runtime_dependencies"])
    if args.action == "verify":
        verify_results(release)
        print(f"PARK {release['revision']}: verified controller, models and {len(release['published_runs'])} published trials.")
        return

    # Use the exact acquisition entry point that produced the signed v3 records.
    sys.path.insert(0, str(ROOT / "PalmReach-3"))
    from run import acquire, trial_dir
    from comparison import score_trial

    folder = trial_dir(args.out.resolve(), args.seconds, args.object, args.repeat)
    if args.action == "trial":
        acquire(args.object, args.seconds, args.repeat, folder / "native", revision=release["revision"])
    signature = read(folder / "native/adapter.json")["signature"]
    expected = (release["recorded_method"], release["revision"], args.object, args.seconds,
                release["seeds"][args.object] + args.repeat)
    actual = (signature["method"], signature.get("revision"), signature["object"], signature["seconds"], signature["seed"])
    if (actual != expected or signature["files"] != release["planner_inputs"]
            or signature["native"] != release["controller_inputs"]):
        raise ValueError(f"Different PARK release or trial at {folder}")
    score_trial(folder)
    result = read(folder / "score/metrics.json")
    print(json.dumps(dict(method=release["name"], revision=release["revision"], object=args.object,
                          seconds=args.seconds, seed=signature["seed"], metrics=result["final"],
                          trial=str(folder)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
