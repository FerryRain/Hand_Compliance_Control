"""Run one Active-3 finger condition on the formal AUDE paired start."""

import argparse
import json
from pathlib import Path

from PARK.identity import digest

from ..palmreach6.__main__ import FIXTURES, OBJECTS, ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=FIXTURES)
    parser.add_argument("--object", choices=OBJECTS, required=True)
    parser.add_argument("--condition", choices=("planned", "single_all", "compliance", "aude_mcc", "aude_height",
                                                "aude_mcc_height"),
                        required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=240)
    parser.add_argument("--acquire-only", action="store_true", help="Save the raw trajectory for batch evaluation")
    args = parser.parse_args()
    if args.seconds < 1:
        parser.error("--seconds must be positive")
    if args.seconds != 240 and not args.acquire_only:
        parser.error("Active-3 tables use 240 seconds; use --acquire-only for a shorter or longer rollout")

    released = json.loads((ROOT / "PARK/release.json").read_bytes())["runtime_dependencies"]
    fixture_key = (FIXTURES.relative_to(ROOT) / "fixtures.json").as_posix()
    if digest(args.fixtures / "fixtures.json") != released[fixture_key]:
        parser.error("--fixtures must contain the formal five-object AUDE start-02 fixtures")

    import torch

    from ..acitive_02 import author_control
    from ..active_1_b import mustard_recovery, no_tip_target, single_all

    torch.set_num_threads(1)
    if args.condition in ("aude_mcc", "aude_mcc_height"):
        from . import mcc

        mcc.run(args.object, args.fixtures, args.out, args.seconds, constrain_palm=args.condition == "aude_mcc_height")
    elif args.condition == "aude_height":
        from . import height

        height.run(args.object, args.fixtures, args.out, args.seconds)
    elif args.condition == "planned":
        author_control.run(args.object, args.fixtures, args.out, args.seconds, planner_feedback="physical")
    elif args.condition == "compliance":
        no_tip_target.run(args.object, args.fixtures, args.out, args.seconds)
    elif args.object == "mustard":
        mustard_recovery.run(
            args.fixtures, args.out, fingers="single_all", radius_mm=0, seed=None,
            seconds=args.seconds, max_finger_step_rad=0.015, object_name=args.object,
        )
    else:
        single_all.rollout(args.object, args.fixtures, args.out, args.seconds)

    if not args.acquire_only:
        if args.condition == "aude_mcc_height":
            from .report_fingerall import score_trial
        elif args.condition in ("aude_mcc", "aude_height"):
            from .report_extended import score_trial
        else:
            from .report import score_trial

        result = score_trial(args.out, args.object, args.condition)
        print(f"{args.object}/{args.condition}  C25 {result['c25']:.2%}  F@5 {result['f5']:.2%}", flush=True)


if __name__ == "__main__":
    main()
