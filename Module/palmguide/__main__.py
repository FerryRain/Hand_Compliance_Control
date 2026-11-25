"""Run the published PalmGuide planner on one prepared paired fixture."""
import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", choices=("pear", "mustard", "cracker", "bleach", "coffee"), required=True)
    parser.add_argument("--seconds", type=float, default=240)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    # The archived CPU numerical kernels used one thread per library. Override
    # machine-specific defaults before importing NumPy, SciPy and PyTorch.
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    # This CLI runs the historical Facing-4 snapshot. Its seven changed wrist
    # modules have their original bytes; the other modules are shared.
    from Module import wrist_active
    wrist_active.__path__.insert(0, str(Path(__file__).with_name("facing4")))
    from Module.wrist_active.palm_unload import run

    print(run(args.out.resolve(), args.object, args.seconds), flush=True)


if __name__ == "__main__":
    main()
