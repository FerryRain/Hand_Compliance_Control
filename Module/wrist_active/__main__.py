"""Run the adaptive wrist planner on a retained physical start and score its trace."""

import argparse
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

from . import sweep_experiment as runtime
from .experiment import digest
from .face_wrist import FaceWorker, FaceWrist
from .load_guard import GuardedWrist
from .mcc_trials import read
from .sweep_paced import timer_resolution
from .sweep_score import score
from .wrist_worker import WristWorker

RESULTS = runtime.GENERATED / "gpis_wrist_face_transfer_20261121"


def sources():
  files = runtime.hashes()
  for name in ("face_wrist.py", "load_guard.py", "__main__.py"):
    files[name] = digest(Path(__file__).with_name(name))
  return files


def run(
  root, obj, start, seconds=240.0, *, arm="sweep-dp", seed=None, fast=False, paced=False, original=False
):
  folder = root / "runs" / obj / start / arm
  fingerprints = (
    dict(runtime.hashes(), **{"load_guard.py": digest(Path(__file__).with_name("load_guard.py"))})
    if original
    else sources()
  )
  if (root / "protocol.json").is_file() and read(root / "protocol.json")["source"] != fingerprints:
    raise ValueError("Planner differs from the registered source")
  fingerprints["controller"] = "GuardedWrist" if original else "FaceWrist"
  with (
    patch.object(runtime, "SweepWrist", GuardedWrist if original else FaceWrist),
    patch.object(runtime, "WristWorker", WristWorker if original else FaceWorker),
    patch.object(runtime, "hashes", return_value=fingerprints),
    timer_resolution() if paced else nullcontext(),
  ):
    receipt = runtime.run(folder, obj, start, arm, seconds, paced=paced, seed=seed)
  fixture = read(runtime.OUT / "protocol.json")["fixtures"][f"{obj}/{start}"]
  if (
    receipt["model_sha256"] != fixture["model.mjb"]
    or receipt["initial_file_sha256"] != fixture["initial.npz"]
  ):
    raise ValueError("Physical fixture differs from the original trial")
  times = sorted({s for s in (60.0, 120.0, 240.0, seconds) if s <= seconds})
  return score(folder, folder, obj, start, times, fast=fast)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--root", type=Path, default=RESULTS)
  parser.add_argument("--case", default="pear/active-02")
  parser.add_argument("--seconds", type=float, default=240.0)
  parser.add_argument("--seed", type=int)
  parser.add_argument("--arm", choices=("sweep-dp", "sweep-mcc"), default="sweep-dp")
  parser.add_argument("--fast", action="store_true")
  parser.add_argument("--paced", action="store_true")
  parser.add_argument(
    "--original", action="store_true", help="run the preceding guarded wrist for a paired repeat"
  )
  args = parser.parse_args()
  obj, start = args.case.split("/")
  run(
    args.root,
    obj,
    start,
    args.seconds,
    seed=args.seed,
    arm=args.arm,
    fast=args.fast,
    paced=args.paced,
    original=args.original,
  )


if __name__ == "__main__":
  main()
