"""Run and score one paired Active-2 trajectory."""

import argparse
from pathlib import Path

from ..acitive_02.route_free_inputs import OBJECTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--object", choices=OBJECTS, required=True)
    parser.add_argument("--condition", choices=("planned", "single_all", "compliance"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=240)
    args = parser.parse_args()
    if args.seconds < 1:
        parser.error("--seconds must be positive")

    import torch

    from ..acitive_02 import author_control
    from ..acitive_02.route_free_score import score
    from ..active_1_b import mustard_recovery, no_tip_target, single_all

    torch.set_num_threads(1)
    if args.condition == "planned":
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

    print(score(args.out, args.fixtures / args.object / "surface.obj"), flush=True)


if __name__ == "__main__":
    main()
