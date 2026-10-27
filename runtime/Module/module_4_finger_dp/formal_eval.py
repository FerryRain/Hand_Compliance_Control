"""CUDA-only held-out physical validation for a formal Dataset-I checkpoint."""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import json
from pathlib import Path

import numpy as np

from Module.module_4_finger_dp.track_d_closed_loop import (
  TrackDClosedLoopConfig,
  run_track_d_closed_loop,
)


def run_formal_heldout(
  checkpoint_path: str | Path,
  teacher_trace_path: str | Path,
  source_config_path: str | Path,
  output_directory: str | Path,
  *,
  dataset_scale: str,
) -> dict:
  source = json.loads(Path(source_config_path).read_text(encoding="utf-8"))["config"]
  config = TrackDClosedLoopConfig(
    duration_s=10.5,
    dp_activation_s=1.0,
    desired_force_n=float(source["desired_force_n"]),
    contact_threshold_n=float(source["contact_threshold_n"]),
    force_limit_n=float(source["force_limit_n"]),
    seed=int(source["seed"]),
    surface=str(source["surface"]),
    friction_coefficient=float(source["friction_coefficient"]),
    object_offset_x_m=float(source["terrain_offset_x_m"]),
    object_offset_y_m=float(source["terrain_offset_y_m"]),
    object_offset_z_m=float(source["terrain_offset_z_m"]),
    rebase_wrist_plan_at_dp_activation=False,
  )
  trace, metrics = run_track_d_closed_loop(
    checkpoint_path,
    teacher_trace_path,
    config,
    checkpoint_kind="FORMAL_DATASET_I",
  )
  checks = {
    "contact_continuity": metrics.contact_continuity >= config.minimum_contact_continuity,
    "zero_contact_gap": metrics.longest_zero_contact_gap_s <= config.maximum_zero_contact_gap_s,
    "tip_force": metrics.maximum_force_n < config.force_limit_n,
    "non_tip": metrics.non_tip_contact_frames <= config.maximum_non_tip_contact_frames,
    "authority_solver": metrics.authority_solver_failure_frames == 0,
    "hard_guard": metrics.hard_guard_frames == 0,
    "dp_executed": metrics.policy_replan_count > 0,
  }
  failed = tuple(name for name, passed in checks.items() if not passed)
  summary = {
    "stage": "FORMAL_DATASET_I_HELDOUT_CLOSED_LOOP",
    "dataset_class": "DATASET_I_RAW_VERIFIED",
    "dataset_scale": dataset_scale,
    "checkpoint": str(Path(checkpoint_path)),
    "config": asdict(config),
    "metrics": asdict(metrics),
    "heldout_gate": {
      "status": "PASS" if not failed else "FAIL",
      "blocking_reason": ("NONE",) if not failed else failed,
      "checks": checks,
    },
  }
  output = Path(output_directory)
  output.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(
    output / "closed_loop_trace.npz",
    **{definition.name: np.asarray(getattr(trace, definition.name)) for definition in fields(trace)},
  )
  (output / "closed_loop_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True),
    encoding="utf-8",
  )
  return summary


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--teacher-trace", type=Path, required=True)
  parser.add_argument("--source-config", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--dataset-scale", required=True)
  args = parser.parse_args()
  summary = run_formal_heldout(
    args.checkpoint,
    args.teacher_trace,
    args.source_config,
    args.output,
    dataset_scale=args.dataset_scale,
  )
  print(json.dumps(summary, indent=2))


if __name__ == "__main__":
  main()
