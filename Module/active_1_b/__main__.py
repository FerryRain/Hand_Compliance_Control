"""Run Active-1-B on one registered four-contact LEAP fixture."""
import argparse
from pathlib import Path

import torch

from ..acitive_02.author_control import run
from ..acitive_02.route_free_inputs import OBJECTS
from ..acitive_02.route_free_score import score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--object", choices=OBJECTS, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=240)
    args = parser.parse_args()
    if args.seconds < 1:
        parser.error("--seconds must be positive")
    torch.set_num_threads(1)
    run(args.object, args.fixtures, args.out, args.seconds, planner_feedback="physical")
    print(score(args.out, args.fixtures / args.object / "surface.obj"), flush=True)


if __name__ == "__main__":
    main()
