"""Run and verify PARK on its published fixtures."""
import argparse
import json
from pathlib import Path

from .identity import read, verify_files, verify_identity

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def verify_results(release):
    for trial in release["published_runs"]:
        folder = ROOT / release["results_directory"] / trial["path"]
        verify_files(folder, trial["files"])
        receipt = read(folder / "native/adapter.json")
        verify_identity(receipt, folder / "native", HERE / "release.json")
        signature = receipt["signature"]
        if any(signature[k] != trial[k] for k in ("object", "seconds", "seed")):
            raise ValueError(f"Unexpected PARK trial: {folder}")


def main():
    release = read(HERE / "release.json")
    parser = argparse.ArgumentParser(prog="PARK", description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    actions.add_parser("verify", help="Check the controller, models and all published trials.")
    for action, help_text in (("trial", "Run and score PARK."), ("score", "Score a PARK acquisition.")):
        cmd = actions.add_parser(action, help=help_text)
        cmd.add_argument("--object", required=True, choices=release["seeds"])
        cmd.add_argument("--seconds", type=int, default=600, choices=release["budgets_s"])
        cmd.add_argument("--repeat", type=int, default=0, choices=(0, 1))
        cmd.add_argument("--out", type=Path, default=ROOT / release["results_directory"])
    args = parser.parse_args()
    from Module.palmreach6.__main__ import ROOT as native_root

    verify_files(ROOT, release["planner_inputs"])
    verify_files(native_root, release["controller_inputs"])
    verify_files(ROOT, release["workspace_dependencies"])
    verify_files(native_root, release["runtime_dependencies"])
    if args.action == "verify":
        verify_results(release)
        print(f"PARK: verified controller, models and {len(release['published_runs'])} published trials.")
        return

    from .acquisition import acquire, trial_dir
    from .comparison import score_trial

    folder = trial_dir(args.out.resolve(), args.seconds, args.object, args.repeat)
    if args.action == "trial":
        acquire(args.object, args.seconds, args.repeat, folder / "native")
    receipt = read(folder / "native/adapter.json")
    verify_identity(receipt, folder / "native", HERE / "release.json")
    sig = receipt["signature"]
    if (sig["object"], sig["seconds"], sig["seed"]) != (
            args.object, args.seconds, release["seeds"][args.object] + args.repeat):
        raise ValueError(f"Unexpected trial at {folder}")
    score_trial(folder)
    result = read(folder / "score/metrics.json")
    print(json.dumps(dict(method="PARK", object=args.object, seconds=args.seconds, seed=sig["seed"],
                          metrics=result["final"], trial=str(folder)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
