"""Run the main PalmReach-6 experiment or its matched Facing-4 and Aude controls."""

import argparse
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
OBJECTS = ("coffee", "cracker", "mustard", "bleach", "pear")
FIXTURES = ROOT / "Module/generated/palm_progress_full_20261125/paired-fixtures"


def run(obj, out, seconds, method="palmreach-6"):
    import torch
    from Module.wrist_active import informative_run, palm_cracker_parameters, palm_unload, ycb_fixtures
    from Module.wrist_active.experiment import digest

    torch.set_num_threads(1)
    protocol = FIXTURES / "fixtures.json"
    spec = json.loads(protocol.read_bytes())
    fixture = spec["objects"][obj]
    case = FIXTURES / obj
    if fixture["start"] != "start-02" or digest(FIXTURES / "hand.mjb") != spec["hand_sha256"]:
        raise ValueError("The fixtures differ from the formal paired start")
    for name in ("model.mjb", "initial.npz", "surface.obj"):
        if digest(case / name) != fixture[name]:
            raise ValueError(f"Fixture checksum mismatch for {obj}/{name}")
    if method == "aude":
        from Module.acitive_02 import author_control
        if seconds != int(seconds):
            raise ValueError("The native Aude runner takes whole seconds")
        author_control.run(obj, FIXTURES, out, int(seconds), planner_feedback="physical")
        return out
    controllers = {"palmreach-6": palm_cracker_parameters.run, "facing-4": palm_unload.run}
    with ExitStack() as context:
        context.enter_context(patch.dict(ycb_fixtures.OBJECTS, {obj: (*ycb_fixtures.OBJECTS[obj][:2], fixture["start"])}))
        context.enter_context(patch.object(informative_run, "fixture", return_value=(case, case / "surface.obj")))
        context.enter_context(patch.object(informative_run, "PROTOCOL", protocol))
        return controllers[method](out, obj, seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", choices=OBJECTS, required=True)
    parser.add_argument("--seconds", type=float, default=240)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--method", choices=("palmreach-6", "facing-4", "aude"), default="palmreach-6")
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0 or abs(round(args.seconds * 100) - args.seconds * 100) > 1e-6:
        parser.error("--seconds must span a positive integral number of 100 Hz steps")
    if args.method == "aude" and not args.seconds.is_integer():
        parser.error("--method aude takes whole seconds")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    print(run(args.object, args.out.resolve(), args.seconds, args.method), flush=True)


if __name__ == "__main__":
    main()
