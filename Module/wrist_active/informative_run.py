"""Run an information-selected wrist with the frozen single_all/MCC on a paired fixture."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from . import experiment
from . import sweep_experiment as runtime
from .experiment import digest, seed_for
from .face_regression_suite import REPO, read, write
from .informative_actions import HISTORY
from .informative_wrist import InformativeWorker, InformativeWrist
from .sweep import DT
from .ycb_fixtures import OBJECTS, fixture
from .ycb_native import Clock


PROTOCOL = REPO / "Module/generated/ycb_frontier_active_20261123/validated/protocol.json"
RELEASE = "global-gpis-information-wrist"


def source_hashes():
    files = [
        Path(__file__).with_name(name)
        for name in (
            "informative_run.py", "informative_actions.py", "informative_wrist.py", "global_gpis.py", "contact_trail.py",
            "history.py", "sweep.py", "sweep_experiment.py", "experiment.py",
            "planner.py", "buffered_fingers.py", "wrist_worker.py", "ycb_fixtures.py",
        )
    ]
    files.extend([HISTORY, REPO / "Module/acitive_02/active_gpis.py",
                  REPO / "Module/i04_geometric_traversal/fingertip_traversal.py"])
    return {p.name: digest(p) for p in files}


def run(folder: Path, obj: str, seconds: float = 600.0):
    folder = Path(folder)
    n = round(seconds / DT)
    if n < 1 or abs(n * DT - seconds) > 1e-8:
        raise ValueError("Duration must be a positive integral number of 10 ms ticks")
    start = OBJECTS[obj][2]
    spec = read(PROTOCOL)
    reference = spec["fixtures"][f"{obj}/{start}"]
    code = source_hashes()
    seed = seed_for(obj, start)
    if folder.exists():
        receipt = read(folder / "clock.json")
        old = read(folder / "run.json")
        if (old["signature"]["code"] != code or receipt["seconds"] != seconds
                or receipt["obj"] != obj or receipt["release"] != RELEASE
                or receipt["trace_sha256"] != digest(folder / "trace.npz")
                or receipt["clock_sha256"] != digest(folder / "native-clock.npz")
                or old["plan_sha256"] != digest(folder / "plans.jsonl")
                or old["timing_sha256"] != digest(folder / "control-timing.npz")):
            raise ValueError(f"Different completed run at {folder}")
        return folder

    folder.parent.mkdir(parents=True, exist_ok=True)
    timer = Clock(obj, RELEASE)

    class TimedWorker(InformativeWorker):
        def submit(self, *args):
            timer.begin()
            return super().submit(*args)

    # The existing rollout constructs a separate geometry-free hand for the
    # controller. Its full physical model and mesh file stay on the plant side.
    with TemporaryDirectory(dir=folder.parent) as temporary:
        trial = Path(temporary) / "trial"
        with (
            patch.object(experiment, "scenario", return_value=fixture(obj)),
            patch.object(runtime, "SweepWrist", InformativeWrist),
            patch.object(runtime, "WristWorker", TimedWorker),
            patch.object(runtime, "robot_state", timer.measured(runtime.robot_state)),
            patch.object(runtime, "hashes", return_value=code),
        ):
            result = runtime.run(trial, obj, start, "information-wrist-dp", seconds, seed=seed)
        if (result["initial_sha256"] != reference["integration_state"]
                or result["model_sha256"] != reference["model.mjb"]
                or result["offline_mesh_sha256"] != reference["surface.obj"]
                or result["policy"]["checkpoint_sha256"] != spec["checkpoint"]
                or result["seed"] != reference["seed"]):
            raise ValueError("Fixture, policy checkpoint or seed differs from the release")
        elapsed = np.asarray(timer.post_step)
        if len(elapsed) != n + 1 or not np.all(np.diff(elapsed) > 0):
            raise ValueError("Physical and exploration clocks disagree")
        np.savez_compressed(trial / "native-clock.npz", first_plan_to_post_step_s=elapsed)
        receipt = dict(obj=obj, arm="ours", release=RELEASE, seconds=seconds,
                       trace_sha256=result["trace_sha256"], clock_sha256=digest(trial / "native-clock.npz"),
                       all_wall_s=float(elapsed[-1]))
        if n >= 24000:
            receipt["first_240_wall_s"] = float(elapsed[24000])
        write(trial / "clock.json", receipt)
        trial.rename(folder)
    return folder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", required=True, choices=tuple(OBJECTS))
    parser.add_argument("--seconds", type=float, default=600.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(run(args.out, args.object, args.seconds), flush=True)


if __name__ == "__main__":
    main()
