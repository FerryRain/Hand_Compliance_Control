"""Paired physical E05-H-MCC versus E05-H-DP evaluation.

Both cells use the same FR3+LEAP plant, initial state, wrist reference, Wrist
MCC, contact-force coordinator, limits, disturbance, and M03 force-safety
executor. The only experimental replacement after the unscored one-second
contact initialization is Finger MCC versus Finger DP + authority filter.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, fields, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from Module.fr3_leap import FullRobotModelConfig, build_full_robot
from Module.module_4_finger_dp.gpu_runtime import require_cuda
from Module.module_4_finger_dp.track_d_closed_loop import (
  TrackDClosedLoopConfig,
  TrackDClosedLoopMetrics,
  TrackDClosedLoopTrace,
  run_track_d_closed_loop,
  save_track_d_closed_loop,
)
from Module.module_4_whole_hand_mcc.benchmark import (
  COMMON_THRESHOLDS,
  H_THRESHOLDS,
  evaluate_episode_thresholds,
  formal_episode_configs,
)
from Module.module_4_whole_hand_mcc.runner import (
  E05MCCConfig,
  E05MCCTrace,
  run_e05_mcc,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "Module/generated/e05_h_mcc_vs_dp"
DEFAULT_CHECKPOINT = (
  REPO_ROOT
  / "Module/generated/finger_dp_formal_v1/training_d20/formal_finger_dp_checkpoint.pt"
)
PROTOCOL_PATH = REPO_ROOT / "Module/E05_DP_CURRENT_PROTOCOL.md"
POSE_STEP_TIME_S = 9.0


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1 << 20), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _save_trace(path: Path, trace: E05MCCTrace | TrackDClosedLoopTrace) -> Path:
  path.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(
    path,
    **{definition.name: np.asarray(getattr(trace, definition.name)) for definition in fields(trace)},
  )
  return path


def _recovery_latency(
  time_s: np.ndarray,
  condition: np.ndarray,
  event_time_s: float,
) -> float:
  start = int(np.searchsorted(time_s, event_time_s))
  indices = np.flatnonzero(condition[start:])
  if not len(indices):
    return float(time_s[-1] - event_time_s)
  return float(time_s[start + int(indices[0])] - event_time_s)


def _dp_metrics(
  trace: TrackDClosedLoopTrace,
  core: TrackDClosedLoopMetrics,
  config: TrackDClosedLoopConfig,
) -> dict[str, Any]:
  handles = build_full_robot(
    FullRobotModelConfig(
      surface="extreme",
      timestep_s=config.dt_s,
      gravity_m_s2=0.0,
      arm_kp=1800.0,
      arm_damping_ratio=0.9,
    )
  )
  mask = trace.time_s >= config.dp_activation_s
  contacts = trace.actual_contacts[mask]
  forces = trace.fingertip_forces_n[mask]
  any_contact = np.any(contacts, axis=1)
  contact_count = np.sum(contacts, axis=1)
  force_error = forces - config.desired_force_n
  arm_lower = trace.arm_q_rad[mask] - handles.arm_joint_ranges_rad[:, 0]
  arm_upper = handles.arm_joint_ranges_rad[:, 1] - trace.arm_q_rad[mask]
  finger_lower = trace.finger_q_rad[mask] - handles.hand_joint_ranges_rad[:, 0]
  finger_upper = handles.hand_joint_ranges_rad[:, 1] - trace.finger_q_rad[mask]
  palm_error = trace.palm_pose_world[mask, :3] - trace.commanded_palm_pose_world[mask, :3]
  actual_delta = np.diff(trace.palm_pose_world[mask, :3], axis=0)
  planned_delta = np.diff(trace.planned_palm_pose_world[mask, :3], axis=0)
  wrench_error = trace.desired_hand_wrench_world[mask] - trace.estimated_hand_wrench_world[mask]
  step_window = (trace.time_s >= POSE_STEP_TIME_S) & (
    trace.time_s < POSE_STEP_TIME_S + 1.0
  )
  force_settled = np.all(
    np.abs(trace.fingertip_forces_n - config.desired_force_n) <= 0.5,
    axis=1,
  ) & np.all(trace.actual_contacts, axis=1)
  replans = trace.policy_replan[mask]
  replan_indices = np.flatnonzero(mask)[replans]
  controller_latency = (
    trace.policy_latency_s[replan_indices]
    + trace.authority_latency_s[replan_indices]
  )
  safety_aborted = trace.guard_state[mask] == "ABORTED"
  result = {
    "cell": "E05-H-DP",
    "controller": "FINGER_DP_WITH_AUTHORITY_FILTER",
    "dp_evaluated": True,
    "surface": config.surface,
    "seed": config.seed,
    "duration_s": config.duration_s,
    "contact_continuity_probability": float(np.mean(any_contact)),
    "average_contact_count": float(np.mean(contact_count)),
    "minimum_contact_count": int(np.min(contact_count)),
    "per_finger_contact_probability": np.mean(contacts, axis=0).tolist(),
    "zero_contact_time_s": float(np.count_nonzero(~any_contact) * config.dt_s),
    "contact_loss_events": core.contact_loss_events,
    "force_rmse_n": float(np.sqrt(np.mean(force_error**2))),
    "force_mae_n": float(np.mean(np.abs(force_error))),
    "force_p95_n": float(np.percentile(forces, 95.0)),
    "max_tip_force_n": float(np.max(forces)),
    "force_violation_probability": float(np.mean(forces > config.force_limit_n)),
    "force_violation_time_s": float(
      np.count_nonzero(np.any(forces > config.force_limit_n, axis=1)) * config.dt_s
    ),
    "pose_step_peak_force_n": float(np.max(trace.fingertip_forces_n[step_window])),
    "any_contact_recovery_s": _recovery_latency(
      trace.time_s,
      np.any(trace.actual_contacts, axis=1),
      POSE_STEP_TIME_S,
    ),
    "four_contact_recovery_s": _recovery_latency(
      trace.time_s,
      np.all(trace.actual_contacts, axis=1),
      POSE_STEP_TIME_S,
    ),
    "force_settling_s": _recovery_latency(
      trace.time_s,
      force_settled,
      POSE_STEP_TIME_S,
    ),
    "actual_palm_path_length_m": float(np.sum(np.linalg.norm(actual_delta, axis=1))),
    "planned_palm_path_length_m": float(np.sum(np.linalg.norm(planned_delta, axis=1))),
    "traversal_y_m": float(
      trace.palm_pose_world[mask][-1, 1] - trace.palm_pose_world[mask][0, 1]
    ),
    "palm_position_tracking_rmse_m": float(np.sqrt(np.mean(palm_error**2))),
    "wrist_wrench_rmse_6d": float(np.sqrt(np.mean(wrench_error**2))),
    "wrist_force_z_rmse_n": float(np.sqrt(np.mean(wrench_error[:, 2] ** 2))),
    "max_wrist_compliance_translation_m": float(
      np.max(np.linalg.norm(trace.wrist_mcc_offset[mask, :3], axis=1))
    ),
    "max_abs_arm_external_torque_nm": float(
      np.max(np.abs(trace.arm_external_torque_nm[mask]))
    ),
    "minimum_arm_joint_margin_rad": float(np.min(np.minimum(arm_lower, arm_upper))),
    "minimum_finger_joint_margin_rad": float(
      np.min(np.minimum(finger_lower, finger_upper))
    ),
    "controller_latency_mean_s": float(np.mean(controller_latency)),
    "controller_latency_p95_s": float(np.percentile(controller_latency, 95.0)),
    "deadline_miss_probability": float(np.mean(controller_latency > config.policy_dt_s)),
    "policy_replan_count": core.policy_replan_count,
    "policy_latency_mean_s": core.policy_latency_mean_s,
    "policy_latency_p95_s": core.policy_latency_p95_s,
    "authority_intervention_probability": core.authority_intervention_probability,
    "authority_intervention_mean_rad": core.authority_intervention_mean_rad,
    "authority_solver_failure_frames": core.authority_solver_failure_frames,
    "authority_maximum_constraint_violation": core.authority_maximum_constraint_violation,
    "hard_guard_frames": core.hard_guard_frames,
    "soft_recovery_frames": core.soft_recovery_frames,
    "safety_aborted_frames": int(np.count_nonzero(safety_aborted)),
    "dp_active_probability": core.dp_active_probability,
    "opposition_rate": core.opposition_rate,
    "opposition_energy": core.opposition_energy,
    "opposition_valid_frames": core.opposition_valid_frames,
    "opposition_conflict_frames": core.opposition_conflict_frames,
    "finger_collective_normal_velocity_p95_m_s": (
      core.finger_collective_normal_velocity_p95_m_s
    ),
    "finger_collective_normal_max_abs_velocity_m_s": (
      core.finger_collective_normal_max_abs_velocity_m_s
    ),
    "wrist_collective_normal_velocity_p95_m_s": (
      core.wrist_collective_normal_velocity_p95_m_s
    ),
    "physics_step_latency_p95_s": float(
      np.percentile(trace.physics_step_latency_s, 95.0)
    ),
    "loop_latency_p95_s": float(np.percentile(trace.loop_latency_s, 95.0)),
    "measured_real_time_factor": float(
      config.dt_s / max(float(np.mean(trace.loop_latency_s)), 1e-12)
    ),
    "non_tip_contact_count": int(np.sum(trace.non_tip_contact_count[mask])),
    "shared_force_safety_enabled": True,
  }
  return result


def _reference_limit_observations(metrics: dict[str, Any]) -> dict[str, Any]:
  if metrics["cell"] == "E05-H-MCC":
    source_checks = evaluate_episode_thresholds(metrics)["checks"]
    thresholds = {
      "hard_guard_frames": ("==", 0.0),
      "safety_aborted_frames": ("==", 0.0),
    }
  else:
    base = dict(COMMON_THRESHOLDS)
    base["controller_latency_p95_s"] = ("<=", 0.020)
    base.update(
      {
        name: spec
        for name, spec in H_THRESHOLDS.items()
        if name != "coordinator_internal_leakage_p95_n"
      }
    )
    thresholds = {
      **base,
      "hard_guard_frames": ("==", 0.0),
      "safety_aborted_frames": ("==", 0.0),
      "authority_solver_failure_frames": ("==", 0.0),
      "authority_maximum_constraint_violation": ("<=", 1e-6),
      "opposition_energy": ("<=", 1e-5),
      "policy_replan_count": (">=", 1.0),
    }
    source_checks = {}
  observations: dict[str, Any] = {}
  for name, check in source_checks.items():
    observations[name] = {
      "value": float(check["value"]),
      "operator": check["operator"],
      "reference": float(check["threshold"]),
    }
  for name, (operator, threshold) in thresholds.items():
    value = float(metrics[name])
    observations[name] = {
      "value": value,
      "operator": operator,
      "reference": threshold,
    }
  peak_force = float(metrics["max_tip_force_n"])
  return {
    "observations": observations,
    "force_reference_limit_n": 8.0,
    "force_peak_excess_n": max(0.0, peak_force - 8.0),
  }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
  numeric_names = sorted(
    name
    for name, value in rows[0]["metrics"].items()
    if isinstance(value, (int, float)) and not isinstance(value, bool)
  )
  numeric: dict[str, Any] = {}
  for name in numeric_names:
    values = np.asarray([row["metrics"][name] for row in rows], dtype=np.float64)
    numeric[name] = {
      "mean": float(np.mean(values)),
      "min": float(np.min(values)),
      "max": float(np.max(values)),
    }
  return {
    "execution_status": "EVALUATED",
    "episodes_completed": len(rows),
    "worst_force_reference_excess_n": max(
      float(row["reference_limit_observations"]["force_peak_excess_n"])
      for row in rows
    ),
    "numeric_metrics": numeric,
  }


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
  flat_rows: list[dict[str, Any]] = []
  for row in rows:
    flat: dict[str, Any] = {
      "cell": row["cell"],
      "episode": row["episode"],
      "force_peak_excess_above_8_n": row["reference_limit_observations"]["force_peak_excess_n"],
    }
    for name, value in row["metrics"].items():
      flat[name] = value if np.isscalar(value) else json.dumps(value)
    flat_rows.append(flat)
  names = sorted({name for row in flat_rows for name in row})
  with path.open("w", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=names)
    writer.writeheader()
    writer.writerows(flat_rows)


def _initializer_alignment(
  baseline: E05MCCTrace,
  dp: TrackDClosedLoopTrace,
  activation_s: float,
) -> dict[str, float]:
  index = int(np.searchsorted(dp.time_s, activation_s)) - 1
  return {
    "finger_q_rmse_rad": float(
      np.sqrt(np.mean((baseline.finger_q_rad[index] - dp.finger_q_rad[index]) ** 2))
    ),
    "palm_position_error_m": float(
      np.linalg.norm(baseline.palm_pose_world[index, :3] - dp.palm_pose_world[index, :3])
    ),
    "force_rmse_n": float(
      np.sqrt(
        np.mean(
          (baseline.fingertip_forces_n[index] - dp.fingertip_forces_n[index]) ** 2
        )
      )
    ),
  }


def run_paired_e05(
  checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
  output_directory: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
  checkpoint = Path(checkpoint_path).resolve()
  if not checkpoint.is_file():
    raise FileNotFoundError(checkpoint)
  if not PROTOCOL_PATH.is_file():
    raise FileNotFoundError(PROTOCOL_PATH)
  device, cuda_info = require_cuda("cuda:0")
  output = Path(output_directory)
  output.mkdir(parents=True, exist_ok=True)
  rows: list[dict[str, Any]] = []
  alignments: dict[str, dict[str, float]] = {}

  for episode_name, frozen_config in formal_episode_configs("E05-H-MCC"):
    # Position-noise perturbations in the historical MCC-only evaluator can
    # begin with a deeply interpenetrating pad (>100 N before any controller
    # step). They are not a valid controller comparison initial condition.
    # Friction and force-observation perturbations remain paired and active.
    if episode_name == "noisy_pose":
      episode_name = "noisy_observation"
    baseline_config = replace(
      frozen_config,
      enforce_shared_force_safety=True,
      initial_joint_noise_std_rad=0.0,
    )
    episode_dir = output / episode_name
    episode_dir.mkdir(parents=True, exist_ok=True)
    baseline_trace, baseline_metrics = run_e05_mcc(baseline_config)
    baseline_path = _save_trace(episode_dir / "e05_h_mcc_trace.npz", baseline_trace)
    baseline_limits = _reference_limit_observations(baseline_metrics)
    rows.append(
      {
        "cell": "E05-H-MCC",
        "episode": episode_name,
        "config": asdict(baseline_config),
        "metrics": baseline_metrics,
        "reference_limit_observations": baseline_limits,
      }
    )

    dp_config = TrackDClosedLoopConfig(
      duration_s=baseline_config.duration_s,
      dp_activation_s=baseline_config.settling_time_s,
      dt_s=baseline_config.dt_s,
      policy_period_steps=baseline_config.wrist_update_period_steps,
      desired_force_n=baseline_config.desired_force_n,
      contact_threshold_n=baseline_config.contact_threshold_n,
      force_limit_n=baseline_config.force_limit_n,
      seed=baseline_config.seed,
      surface=baseline_config.surface,
      friction_coefficient=baseline_config.friction_coefficient,
      force_filter_alpha=baseline_config.force_filter_alpha,
      force_noise_std_n=baseline_config.force_noise_std_n,
      initial_joint_noise_std_rad=baseline_config.initial_joint_noise_std_rad,
      rebase_wrist_plan_at_dp_activation=False,
    )
    dp_trace, core_metrics = run_track_d_closed_loop(
      checkpoint,
      baseline_path,
      dp_config,
      checkpoint_kind="FORMAL_DATASET_I",
    )
    dp_metrics = _dp_metrics(dp_trace, core_metrics, dp_config)
    dp_limits = _reference_limit_observations(dp_metrics)
    _save_trace(episode_dir / "e05_h_dp_trace.npz", dp_trace)
    rows.append(
      {
        "cell": "E05-H-DP",
        "episode": episode_name,
        "config": asdict(dp_config),
        "metrics": dp_metrics,
        "reference_limit_observations": dp_limits,
      }
    )
    alignments[episode_name] = _initializer_alignment(
      baseline_trace,
      dp_trace,
      dp_config.dp_activation_s,
    )
    (episode_dir / "episode_summary.json").write_text(
      json.dumps(
        {
          "episode": episode_name,
          "common_initializer_unscored_until_s": dp_config.dp_activation_s,
          "initializer_alignment": alignments[episode_name],
          "baseline": rows[-2],
          "dp": rows[-1],
        },
        indent=2,
        sort_keys=True,
      ),
      encoding="utf-8",
    )

  cells = {
    cell: _aggregate([row for row in rows if row["cell"] == cell])
    for cell in ("E05-H-MCC", "E05-H-DP")
  }
  summary = {
    "experiment": "E05-H-MCC-VS-E05-H-DP",
    "execution_status": "EVALUATED",
    "evaluation_semantics": "DESCRIPTIVE_ONLY_NO_STRATEGY_PASS_FAIL",
    "reference_limits": {"fingertip_force_n": 8.0},
    "scope": ["E05-H-MCC", "E05-H-DP"],
    "primary_comparison": True,
    "finger_controller_replacement_only_after_s": 1.0,
    "shared_components": [
      "FR3_LEAP_PLANT",
      "INITIAL_STATE_AND_NOISE_SEED",
      "WRIST_REFERENCE",
      "WRIST_MCC",
      "CONTACT_FORCE_COORDINATOR",
      "M03_FORCE_SAFETY_EXECUTOR",
      "ACTUATOR_AND_JOINT_LIMITS",
      "DISTURBANCE_AND_EVALUATION_HORIZON",
    ],
    "replaced_component": "FINGER_MCC <-> FINGER_DP_PLUS_AUTHORITY_FILTER",
    "checkpoint": {
      "path": str(checkpoint.relative_to(REPO_ROOT)),
      "sha256": _sha256(checkpoint),
      "selection": "D20 selected on frozen validation closed-loop stability; D100 retained as scaling evidence",
      "cuda_only": True,
    },
    "protocol": {
      "path": str(PROTOCOL_PATH.relative_to(REPO_ROOT)),
      "sha256": _sha256(PROTOCOL_PATH),
    },
    "cuda_runtime": cuda_info.to_dict(),
    "torch_version": torch.__version__,
    "initializer_alignment": alignments,
    "episodes": rows,
    "cells": cells,
  }
  (output / "summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True),
    encoding="utf-8",
  )
  _write_csv(output / "episodes.csv", rows)
  (output / "README.md").write_text(
    "# E05-H-MCC vs E05-H-DP\n\n"
    "这是正式的三组配对 MuJoCo 物理评测。策略不设置 Pass/Fail 或 MET/NOT_MET；"
    "只报告连续性能和参考限制越界。\n\n"
    f"- E05-H-MCC worst peak 超过 8 N: `{cells['E05-H-MCC']['worst_force_reference_excess_n']:.3f} N`\n"
    f"- E05-H-DP worst peak 超过 8 N: `{cells['E05-H-DP']['worst_force_reference_excess_n']:.3f} N`\n"
    "- 前 1 s 为双方相同、且不计分的接触初始化；之后只替换 finger controller。\n"
    "- 两边共享 Wrist MCC、M03 hard guard、轨迹、扰动、初态和物理参数。\n"
    "- 逐 episode 原始 trace、reference-limit observations 与初始化对齐误差均保留。\n",
    encoding="utf-8",
  )
  return summary


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
  parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
  args = parser.parse_args()
  summary = run_paired_e05(args.checkpoint, args.output)
  print(json.dumps({"execution_status": summary["execution_status"], "cells": summary["cells"]}, indent=2))


if __name__ == "__main__":
  main()
