"""Run PalmReach-6, the selected whole-palm exploration controller."""

import argparse
from pathlib import Path

from .ycb_fixtures import OBJECTS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", choices=tuple(OBJECTS), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=240)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    legacy = parser.add_argument_group("Historical twelve-direction experiments")
    legacy.add_argument("--legacy-12", action="store_true", help="Run the original twelve-direction baseline")
    legacy.add_argument("--prior", type=Path, help="Run the twelve-direction intervention with this tactile history")
    legacy.add_argument("--reach-model", type=Path, help="Run the twelve-direction experiment with this reach model")
    legacy.add_argument("--single-support-risk", type=Path, help="Run the twelve-direction single-pad risk experiment")
    args = parser.parse_args(argv)
    if args.legacy_12 or any(p is not None for p in (args.prior, args.reach_model, args.single_support_risk)):
        from . import palm_exploration, policy_reach

        return palm_exploration.run(args.out, args.object, args.seconds, args.device, args.prior,
                                    args.reach_model or policy_reach.MODEL, risk_path=args.single_support_risk)
    from . import palm_cracker_parameters

    return palm_cracker_parameters.run(args.out, args.object, args.seconds, args.device)


if __name__ == "__main__":
    print(main(), flush=True)
