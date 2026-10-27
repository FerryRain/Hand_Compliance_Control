"""Run the frozen MCC-only E05-F/H physics evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from Module.module_4_whole_hand_mcc.benchmark import (
  DEFAULT_OUTPUT_DIR,
  run_formal_benchmark,
)
from Module.module_4_whole_hand_mcc.runner import E05MCCConfig


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
  parser.add_argument(
    "--quick",
    action="store_true",
    help="run a 3 s smoke protocol; cannot be written to the formal default directory",
  )
  args = parser.parse_args()
  configs = None
  if args.quick:
    if args.output_dir.resolve() == DEFAULT_OUTPUT_DIR.resolve():
      raise SystemExit("--quick requires a non-formal --output-dir")
    configs = {}
    for mode in ("E05-F-MCC", "E05-H-MCC"):
      configs[mode] = (
        (
          "quick_smoke",
          E05MCCConfig(
            mode=mode,
            duration_s=3.0,
            settling_time_s=0.6,
            pose_step_time_s=2.0,
            traversal_y_m=0.03,
            lateral_primary_amplitude_m=0.004,
            lateral_secondary_amplitude_m=0.002,
          ),
        ),
      )
  result = run_formal_benchmark(args.output_dir, episode_configs=configs)
  print(json.dumps(result["cells"], indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
