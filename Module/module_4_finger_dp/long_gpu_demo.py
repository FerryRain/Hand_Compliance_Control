"""Reproduce the long-trajectory CUDA Finger-DP control diagnostic.

This command deliberately keeps Dataset-D and Dataset-I provenance separate.
It collects complete 12 s physical E05-H-MCC demonstrations, rejects whole
episodes that do not maintain contact for at least 10 s, trains only on the
accepted TRAIN split, and evaluates CUDA Finger DP + Wrist MCC on a held-out
accepted EVAL trajectory.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from Module.module_4_finger_dp.long_trajectory_dataset import (
  LongTrajectoryRecord,
  LongTrajectorySpec,
  audit_long_trajectory,
  build_compliant_long_training_set,
  collect_long_trajectory,
  load_long_trace,
)
from Module.module_4_finger_dp.track_d_closed_loop import (
  d_gate_verdict,
  save_track_d_closed_loop,
)
from Module.module_4_finger_dp.track_d_train import (
  TrackDTrainingConfig,
  train_track_d_policy,
)
from Module.module_4_finger_dp.track_d_visual import render_track_d_review
from Module.module_4_finger_dp.whole_hand_dp_control import (
  WholeHandDPControlConfig,
  run_whole_hand_dp_control,
  save_whole_hand_dp_control,
)


DEFAULT_OUTPUT = Path("Module/generated/whole_hand_dp_long_v1")


def frozen_specs() -> tuple[LongTrajectorySpec, ...]:
  """Return frozen v1 collection/evaluation episodes."""

  return (
    LongTrajectorySpec(
      episode_id="train_seed31",
      split="TRAIN",
      seed=31,
      traversal_y_m=0.060,
      lateral_primary_amplitude_m=0.0080,
      lateral_secondary_amplitude_m=0.0030,
      friction_coefficient=0.90,
    ),
    LongTrajectorySpec(
      episode_id="train_seed37",
      split="TRAIN",
      seed=37,
      traversal_y_m=0.058,
      lateral_primary_amplitude_m=0.0075,
      lateral_secondary_amplitude_m=0.0027,
      friction_coefficient=0.95,
      force_noise_std_n=0.015,
      initial_joint_noise_std_rad=0.001,
    ),
    LongTrajectorySpec(
      episode_id="train_seed43",
      split="TRAIN",
      seed=43,
      traversal_y_m=0.062,
      lateral_primary_amplitude_m=0.0085,
      lateral_secondary_amplitude_m=0.0025,
      friction_coefficient=0.88,
      force_noise_std_n=0.020,
      initial_joint_noise_std_rad=0.0015,
    ),
    LongTrajectorySpec(
      episode_id="eval_seed47",
      split="EVAL",
      seed=47,
      traversal_y_m=0.060,
      lateral_primary_amplitude_m=0.0080,
      lateral_secondary_amplitude_m=0.0030,
      friction_coefficient=0.92,
      force_noise_std_n=0.010,
      initial_joint_noise_std_rad=0.001,
    ),
  )


def _parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description=(
      "Collect audited >=10 s contact trajectories, train CUDA Finger DP, "
      "and run Wrist-MCC + Finger-DP physics"
    ),
  )
  parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
  parser.add_argument("--updates", type=int, default=8000)
  parser.add_argument(
    "--reuse-data",
    action="store_true",
    help="reuse existing long trace NPZ files but rerun every compliance audit",
  )
  parser.add_argument(
    "--reuse-checkpoint",
    action="store_true",
    help="reuse the existing CUDA checkpoint after rebuilding/auditing data",
  )
  parser.add_argument(
    "--render-only",
    action="store_true",
    help="rebuild review assets from an existing completed run",
  )
  parser.add_argument(
    "--require-pass",
    action="store_true",
    help="return exit code 2 unless data gates and whole-hand readiness pass",
  )
  return parser


def _load_and_reaudit(
  spec: LongTrajectorySpec,
  data_directory: Path,
) -> LongTrajectoryRecord:
  trace_path = data_directory / f"{spec.episode_id}.npz"
  if not trace_path.is_file():
    raise FileNotFoundError(trace_path)
  metadata_path = data_directory / f"{spec.episode_id}.json"
  metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
  if metadata.get("spec") != asdict(spec):
    raise RuntimeError(
      f"{spec.episode_id} was collected with a stale trajectory specification"
    )
  trace = load_long_trace(trace_path)
  audit = audit_long_trajectory(trace, spec)
  return LongTrajectoryRecord(spec=spec, trace_path=trace_path, audit=audit)


def _collection_summary(
  output: Path,
  records: list[LongTrajectoryRecord],
) -> Path:
  path = output / "long_collection_summary.json"
  path.write_text(
    json.dumps(
      {
        "scope": "DATASET_D_LONG_DIAGNOSTIC",
        "formal_dataset_i": False,
        "minimum_maintained_contact_duration_s": 10.0,
        "whole_episode_fail_closed": True,
        "partial_safe_window_cherry_picking": False,
        "training_episode_ids": [
          record.spec.episode_id
          for record in records
          if record.audit.accepted_for_training
        ],
        "evaluation_episode_ids": [
          record.spec.episode_id
          for record in records
          if record.audit.classification == "ACCEPTED_EVAL"
        ],
        "rejected_episode_ids": [
          record.spec.episode_id
          for record in records
          if record.audit.classification == "REJECTED_DIAGNOSTIC"
        ],
        "episode_audits": [asdict(record.audit) for record in records],
      },
      indent=2,
      sort_keys=True,
    ),
    encoding="utf-8",
  )
  return path


def run(args: argparse.Namespace) -> dict[str, object]:
  output = args.output
  output.mkdir(parents=True, exist_ok=True)
  data_directory = output / "long_trajectories"
  data_directory.mkdir(parents=True, exist_ok=True)
  specs = frozen_specs()

  if args.render_only:
    paths = render_track_d_review(
      output,
      data_directory / "eval_seed47.npz",
      title="Whole-hand CUDA Finger DP + Wrist MCC",
    )
    summary = json.loads(
      (output / "whole_hand_dp_summary.json").read_text(encoding="utf-8")
    )
    return {
      "readiness": summary["verdict"],
      "review_paths": [str(path) for path in paths],
    }

  records: list[LongTrajectoryRecord] = []
  for spec in specs:
    record = (
      _load_and_reaudit(spec, data_directory)
      if args.reuse_data
      else collect_long_trajectory(spec, data_directory)
    )
    records.append(record)
  collection_path = _collection_summary(output, records)

  training_records = [record for record in records if record.spec.split == "TRAIN"]
  evaluation_records = [record for record in records if record.spec.split == "EVAL"]
  if not evaluation_records:
    raise RuntimeError("the frozen EVAL trajectory is missing")
  eval_record = evaluation_records[0]
  if eval_record.audit.classification != "ACCEPTED_EVAL":
    raise RuntimeError(
      "held-out long evaluation trajectory failed compliance gate: "
      f"{eval_record.audit.reasons}"
    )
  samples = build_compliant_long_training_set(
    training_records,
    output / "dataset_d_samples.npz",
  )
  if args.reuse_checkpoint:
    checkpoint_path = output / "track_d_overfit_checkpoint.pt"
    open_summary_path = output / "open_loop_summary.json"
    if not checkpoint_path.is_file() or not open_summary_path.is_file():
      raise FileNotFoundError(
        "--reuse-checkpoint requires an existing checkpoint and open-loop summary"
      )
    open_summary = json.loads(open_summary_path.read_text(encoding="utf-8"))
    open_loop_first_rmse = float(
      open_summary["metrics"]["first_command_rmse_rad"]
    )
  else:
    training = train_track_d_policy(
      samples,
      output,
      TrackDTrainingConfig(
        updates=args.updates,
        batch_size=128,
        device="cuda:0",
      ),
    )
    checkpoint_path = training.checkpoint_path
    open_loop_first_rmse = training.metrics.first_command_rmse_rad
  whole_config = WholeHandDPControlConfig(
    duration_s=11.5,
    initialization_s=1.0,
    desired_force_n=eval_record.spec.desired_force_n,
    seed=eval_record.spec.seed,
    friction_coefficient=eval_record.spec.friction_coefficient,
    initial_joint_noise_std_rad=eval_record.spec.initial_joint_noise_std_rad,
    torch_device="cuda:0",
  )
  result = run_whole_hand_dp_control(
    checkpoint_path,
    eval_record.trace_path,
    whole_config,
  )
  save_whole_hand_dp_control(output, result, whole_config)
  d_verdict = d_gate_verdict(
    causal_audit_passed=samples.audit.passed,
    open_loop_first_command_rmse_rad=open_loop_first_rmse,
    closed_loop=result.metrics,
    config=whole_config.runtime_config(),
  )
  save_track_d_closed_loop(
    output,
    result.trace,
    result.metrics,
    d_verdict,
    whole_config.runtime_config(),
  )
  review_paths = render_track_d_review(
    output,
    eval_record.trace_path,
    title="Whole-hand CUDA Finger DP + Wrist MCC",
  )
  accepted_train = sum(
    record.audit.accepted_for_training for record in training_records
  )
  all_data_gates_pass = (
    accepted_train == len(training_records)
    and eval_record.audit.classification == "ACCEPTED_EVAL"
  )
  overall_pass = (
    all_data_gates_pass
    and result.verdict.readiness_status == "MET"
  )
  return {
    "status": "PASS" if overall_pass else "FAIL",
    "scope": "DATASET_D_LONG_CONTROL_DIAGNOSTIC",
    "formal_dataset_i": False,
    "accepted_training_episodes": accepted_train,
    "total_training_episodes": len(training_records),
    "accepted_training_samples": samples.count,
    "collection_summary": str(collection_path),
    "checkpoint": str(checkpoint_path),
    "whole_hand_summary": str(output / "whole_hand_dp_summary.json"),
    "review_paths": [str(path) for path in review_paths],
  }


def main() -> int:
  args = _parser().parse_args()
  result = run(args)
  print(json.dumps(result, indent=2, sort_keys=True))
  if args.require_pass and result.get("status") != "PASS":
    return 2
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
