"""Run the frozen E05-PHY benchmark and write machine-readable artifacts."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")
cache_directory = Path(tempfile.gettempdir()) / "handcomp-mesa"
cache_directory.mkdir(parents=True, exist_ok=True)
os.environ["XDG_CACHE_HOME"] = str(cache_directory)

from Module.e05_physics.benchmark import (
  DEFAULT_OUTPUT_DIR,
  run_physics_evaluation,
  write_results,
)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
  args = parser.parse_args()
  summary, rows, traces = run_physics_evaluation()
  artifacts = write_results(args.output_dir, summary, rows, traces)
  from Module.e05_physics.visual_demo import run_visual_demo

  visuals = run_visual_demo(args.output_dir)
  print(
    json.dumps(
      {
        "evaluation_status": summary["evaluation_status"],
        "evaluation_completed": summary["evaluation_completed"],
        "benchmark_verdict": summary["benchmark_verdict"],
        "all_thresholds_met": summary["all_thresholds_met"],
        "physics_engine": summary["physics_engine"],
        "finger_dp_evaluated": False,
        "artifacts": {key: str(value) for key, value in artifacts.items()},
        "visuals": visuals,
      },
      indent=2,
      ensure_ascii=False,
    )
  )


if __name__ == "__main__":
  main()
