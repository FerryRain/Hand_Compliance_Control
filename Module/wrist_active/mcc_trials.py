"""Run the frozen wrist Active with fixed nominal fingers and tactile MCC."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from . import sweep_experiment as runtime
from .experiment import digest
from .load_guard import GuardedWrist
from .sweep_score import score
from .sweep_suite import jobs


RESULTS = runtime.GENERATED / "gpis_wrist_information_20261121"


def read(path):
    return json.loads(path.read_bytes())


def run_case(root, obj, start, seconds):
    parent = read(runtime.OUT / "protocol.json")
    sources = read(runtime.OUT / "force-priority/protocol.json")["source"]
    actual = runtime.hashes()
    for filename in ("load_guard.py", "sweep_guard_trials.py"):
        actual[filename] = digest(Path(__file__).with_name(filename))
    if actual != sources:
        raise ValueError("Frozen wrist source has changed")
    folder = root / "runs" / obj / start / "sweep-mcc"
    with patch.object(runtime, "SweepWrist", GuardedWrist), patch.object(runtime, "hashes", return_value=sources):
        receipt = runtime.run(folder, obj, start, "sweep-mcc", seconds)
    fixture = parent["fixtures"][f"{obj}/{start}"]
    if (receipt["model_sha256"] != fixture["model.mjb"] or receipt["initial_file_sha256"] != fixture["initial.npz"]
            or receipt["policy"] is not None or receipt["installs"] != 0):
        raise ValueError("Fixture or pure MCC configuration differs")
    with np.load(folder / "trace.npz", allow_pickle=False) as trace:
        nominal = trace["finger_nominal"]
        if not np.all(nominal == nominal[0]):
            raise ValueError("Pure MCC received changing nominal finger targets")
        if obj == "pear":
            archive = runtime.OUT / "force-priority/runs/pear/active-02/sweep-mcc"
            if digest(archive / "trace.npz") != read(archive / "run.json")["trace_sha256"]:
                raise ValueError("Archived pear trace changed")
            with np.load(archive / "trace.npz", allow_pickle=False) as old:
                for field in ("qpos", "finger_nominal", "palm_command", "points_m", "pad_valid", "region_load_n"):
                    np.testing.assert_array_equal(trace[field][:len(old[field])], old[field], err_msg=field)
    times = [s for s in (60., 120., 240., 600., 1129.32) if s <= seconds]
    score(folder, folder, obj, start, times)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=RESULTS)
    parser.add_argument("--group", choices=("pear", "ellipsoid", "meat-can", *runtime.SMOOTH), required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    for case in jobs():
        if case["obj"] == args.group and case["series"] == "runs" and case["arm"] == "sweep-dp":
            run_case(args.root, case["obj"], case["start"], case["seconds"])


if __name__ == "__main__":
    main()
