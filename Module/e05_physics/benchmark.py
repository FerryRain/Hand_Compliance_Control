"""Frozen MuJoCo evaluation for the E05 MCC baseline arm only."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from Module.e05_physics.extreme_surface import profile_characteristics
from Module.e05_physics.runner import PhysicsConfig, PhysicsTrace, run_scenario
from Module.e05_physics.scene import FINGERS, PAD_HALF_SIZE_M, Q_NOMINAL


PROTOCOL_PATH = Path(__file__).resolve().parents[1] / "E05_PHYSICS_PROTOCOL.md"
DEFAULT_OUTPUT_DIR = Path("Module/generated/e05_physics_v3")

MAINTENANCE_THRESHOLDS = {
  "contact_continuity_probability_min": 0.999,
  "average_contact_count_min": 3.90,
  "force_rmse_n_max": 0.30,
  "max_tip_force_n_max": 8.0,
  "zero_contact_time_s_max": 0.002,
  "non_tip_contact_count_max": 0,
  "minimum_joint_margin_rad_min": 0.05,
  "joint_limit_probability_max": 0.0,
  "thumb_contact_probability_min": 0.95,
  "contact_distal_head_clearance_min_m": 0.010,
}

HANDOVER_THRESHOLDS = {
  "ordered_contact_sets_required": True,
  "anchor_retention_min": 1.0,
  "make_recovery_time_s_max": 0.25,
  "zero_contact_time_s_max": 0.0,
  "final_set_retention_min": 1.0,
  "max_tip_force_n_max": 8.0,
  "non_tip_contact_count_max": 0,
  "minimum_joint_margin_rad_min": 0.05,
}

ROBUSTNESS_THRESHOLDS = {
  "episode_success_rate_min": 0.90,
  "contact_continuity_probability_min": 0.995,
  "average_contact_count_min": 3.50,
  "force_rmse_n_max": 0.35,
  "force_violation_probability_max": 0.005,
  "joint_limit_probability_max": 0.0,
  "non_tip_contact_count_max": 0,
}

EXTREME_THRESHOLDS = {
  "continuous_hand_contact_probability_min": 0.995,
  "continuous_average_contact_count_min": 3.50,
  "continuous_force_rmse_n_max": 0.50,
  "continuous_force_violation_probability_max": 0.005,
  "continuous_max_tip_force_n_max": 8.0,
  "continuous_thumb_contact_probability_min": 0.95,
  "continuous_contact_distal_head_clearance_min_m": 0.010,
  "continuous_relative_path_length_m_min": 0.30,
  "pose_step_any_contact_recovery_s_max": 0.10,
  "pose_step_all_finger_recovery_s_max": 0.25,
  "pose_step_longest_zero_contact_s_max": 0.15,
  "pose_step_force_settling_s_max": 0.75,
  "minimum_joint_margin_rad_min": 0.05,
  "non_tip_contact_count_max": 0,
}


def protocol_sha256() -> str:
  return hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest()


def _maintenance_thresholds_met(metrics: dict[str, Any]) -> bool:
  return bool(
    metrics["contact_continuity_probability"]
    >= MAINTENANCE_THRESHOLDS["contact_continuity_probability_min"]
    and metrics["average_contact_count"]
    >= MAINTENANCE_THRESHOLDS["average_contact_count_min"]
    and metrics["force_rmse_n"] <= MAINTENANCE_THRESHOLDS["force_rmse_n_max"]
    and metrics["max_tip_force_n"] <= MAINTENANCE_THRESHOLDS["max_tip_force_n_max"]
    and metrics["zero_contact_time_s"]
    <= MAINTENANCE_THRESHOLDS["zero_contact_time_s_max"]
    and metrics["non_tip_contact_count"]
    <= MAINTENANCE_THRESHOLDS["non_tip_contact_count_max"]
    and metrics["minimum_joint_margin_rad"]
    >= MAINTENANCE_THRESHOLDS["minimum_joint_margin_rad_min"]
    and metrics["joint_limit_probability"]
    <= MAINTENANCE_THRESHOLDS["joint_limit_probability_max"]
    and metrics["thumb_contact_probability"]
    >= MAINTENANCE_THRESHOLDS["thumb_contact_probability_min"]
    and metrics["contact_distal_head_clearance_min_m"]
    >= MAINTENANCE_THRESHOLDS["contact_distal_head_clearance_min_m"]
  )


def evaluate_contact_maintenance(
  config: PhysicsConfig | None = None,
) -> tuple[dict[str, Any], dict[str, PhysicsTrace]]:
  base = config or PhysicsConfig()
  scenarios: dict[str, Any] = {}
  traces: dict[str, PhysicsTrace] = {}
  for scenario in (
    "maintenance_translation",
    "maintenance_rotation",
    "maintenance_curved",
  ):
    scenario_config = PhysicsConfig(**{**asdict(base), "scenario": scenario})
    trace, metrics = run_scenario(scenario_config)
    metrics["thresholds_met"] = _maintenance_thresholds_met(metrics)
    metrics["scenario"] = scenario
    scenarios[scenario] = metrics
    traces[scenario] = trace
  return {
    "thresholds_met": all(item["thresholds_met"] for item in scenarios.values()),
    "thresholds": MAINTENANCE_THRESHOLDS,
    "scenarios": scenarios,
  }, traces


def _all_rows_equal(
  values: np.ndarray,
  expected: tuple[bool, bool, bool, bool],
) -> bool:
  return bool(np.all(values == np.asarray(expected, dtype=np.bool_)))


def evaluate_contact_handover(
  config: PhysicsConfig | None = None,
) -> tuple[dict[str, Any], PhysicsTrace]:
  base = config or PhysicsConfig()
  handover_config = PhysicsConfig(**{**asdict(base), "scenario": "handover"})
  trace, common = run_scenario(handover_config)
  break_time_s = handover_config.settling_time_s + 1.0
  make_time_s = handover_config.settling_time_s + 1.25
  start_mask = (trace.time_s >= handover_config.settling_time_s) & (
    trace.time_s < break_time_s
  )
  middle_mask = (trace.time_s >= break_time_s) & (trace.time_s < make_time_s)
  final_mask = trace.time_s >= handover_config.duration_s - 0.50
  post_make_indices = np.flatnonzero(
    (trace.time_s >= make_time_s) & trace.actual_contacts[:, 3]
  )
  make_recovery_time_s: float | None = None
  if post_make_indices.size:
    make_recovery_time_s = float(
      trace.time_s[int(post_make_indices[0])] - make_time_s
    )
  start_set_confirmed = _all_rows_equal(
    trace.actual_contacts[start_mask],
    (True, True, True, False),
  )
  middle_set_confirmed = _all_rows_equal(
    trace.actual_contacts[middle_mask],
    (True, True, False, False),
  )
  final_set_retention = float(
    np.mean(
      np.all(
        trace.actual_contacts[final_mask]
        == np.array([True, True, False, True]),
        axis=1,
      )
    )
  )
  evaluation_mask = trace.time_s >= handover_config.settling_time_s
  anchor_retention = float(
    np.mean(trace.actual_contacts[evaluation_mask, :2])
  )
  ordered_contact_sets = bool(
    start_set_confirmed
    and middle_set_confirmed
    and make_recovery_time_s is not None
    and final_set_retention == 1.0
  )
  result = {
    **common,
    "start_set_confirmed": start_set_confirmed,
    "middle_set_confirmed": middle_set_confirmed,
    "ordered_contact_sets": ordered_contact_sets,
    "anchor_retention": anchor_retention,
    "make_recovery_time_s": make_recovery_time_s,
    "final_set_retention": final_set_retention,
    "break_command_time_s": break_time_s,
    "make_command_time_s": make_time_s,
    "thresholds": HANDOVER_THRESHOLDS,
  }
  result["thresholds_met"] = bool(
    ordered_contact_sets
    and anchor_retention >= HANDOVER_THRESHOLDS["anchor_retention_min"]
    and make_recovery_time_s is not None
    and make_recovery_time_s <= HANDOVER_THRESHOLDS["make_recovery_time_s_max"]
    and common["zero_contact_time_s"]
    <= HANDOVER_THRESHOLDS["zero_contact_time_s_max"]
    and final_set_retention >= HANDOVER_THRESHOLDS["final_set_retention_min"]
    and common["max_tip_force_n"] <= HANDOVER_THRESHOLDS["max_tip_force_n_max"]
    and common["non_tip_contact_count"]
    <= HANDOVER_THRESHOLDS["non_tip_contact_count_max"]
    and common["minimum_joint_margin_rad"]
    >= HANDOVER_THRESHOLDS["minimum_joint_margin_rad_min"]
  )
  return result, trace


def _robustness_config(seed: int) -> PhysicsConfig:
  rng = np.random.default_rng(seed)
  return PhysicsConfig(
    scenario="maintenance_rotation" if seed % 2 == 0 else "maintenance_curved",
    seed=seed,
    friction_coefficient=float(rng.uniform(0.55, 1.05)),
    initial_joint_noise_std_rad=float(rng.uniform(0.0, 0.020)),
    force_noise_std_n=float(rng.uniform(0.0, 0.080)),
    surface_bias_m=float(rng.uniform(-0.00040, 0.00040)),
    wrist_error_amplitude_m=float(rng.uniform(0.0, 0.00050)),
    motion_scale=float(rng.uniform(0.90, 1.10)),
  )


def _robust_episode_thresholds_met(metrics: dict[str, Any]) -> bool:
  return bool(
    metrics["contact_continuity_probability"] >= 0.999
    and metrics["average_contact_count"] >= 3.50
    and metrics["max_tip_force_n"] <= 8.0
    and metrics["minimum_joint_margin_rad"] >= 0.05
    and metrics["joint_limit_probability"] == 0.0
    and metrics["non_tip_contact_count"] == 0
  )


def evaluate_control_robustness(
  episode_count: int = 24,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
  if episode_count != 24:
    raise ValueError("the frozen E05-PHY protocol requires exactly 24 episodes")
  rows: list[dict[str, Any]] = []
  for seed in range(episode_count):
    config = _robustness_config(seed)
    _, metrics = run_scenario(config)
    thresholds_met = _robust_episode_thresholds_met(metrics)
    rows.append({**asdict(config), **metrics, "thresholds_met": thresholds_met})

  aggregate = {
    "episode_count": episode_count,
    "episode_success_rate": float(
      np.mean([row["thresholds_met"] for row in rows])
    ),
    "contact_continuity_probability": float(
      np.mean([row["contact_continuity_probability"] for row in rows])
    ),
    "average_contact_count": float(
      np.mean([row["average_contact_count"] for row in rows])
    ),
    "force_rmse_n": float(
      np.sqrt(np.mean([row["force_rmse_n"] ** 2 for row in rows]))
    ),
    "force_violation_probability": float(
      np.mean([row["force_violation_probability"] for row in rows])
    ),
    "joint_limit_probability": float(
      np.mean([row["joint_limit_probability"] for row in rows])
    ),
    "non_tip_contact_count": int(
      np.sum([row["non_tip_contact_count"] for row in rows])
    ),
    "minimum_joint_margin_rad": float(
      np.min([row["minimum_joint_margin_rad"] for row in rows])
    ),
    "maximum_tip_force_n": float(np.max([row["max_tip_force_n"] for row in rows])),
  }
  aggregate["thresholds_met"] = bool(
    aggregate["episode_success_rate"]
    >= ROBUSTNESS_THRESHOLDS["episode_success_rate_min"]
    and aggregate["contact_continuity_probability"]
    >= ROBUSTNESS_THRESHOLDS["contact_continuity_probability_min"]
    and aggregate["average_contact_count"]
    >= ROBUSTNESS_THRESHOLDS["average_contact_count_min"]
    and aggregate["force_rmse_n"] <= ROBUSTNESS_THRESHOLDS["force_rmse_n_max"]
    and aggregate["force_violation_probability"]
    <= ROBUSTNESS_THRESHOLDS["force_violation_probability_max"]
    and aggregate["joint_limit_probability"]
    <= ROBUSTNESS_THRESHOLDS["joint_limit_probability_max"]
    and aggregate["non_tip_contact_count"]
    <= ROBUSTNESS_THRESHOLDS["non_tip_contact_count_max"]
  )
  return {
    "thresholds_met": aggregate["thresholds_met"],
    "thresholds": ROBUSTNESS_THRESHOLDS,
    "aggregate": aggregate,
  }, rows


def extreme_surface_config() -> PhysicsConfig:
  return PhysicsConfig(
    scenario="extreme_surface",
    duration_s=15.0,
    settling_time_s=0.75,
    pose_step_time_s=10.0,
    pose_step_m=0.004,
  )


def _first_sustained_time(
  trace: PhysicsTrace,
  condition: np.ndarray,
  start_index: int,
  duration_s: float,
) -> float | None:
  frame_count = max(1, int(round(duration_s / (trace.time_s[1] - trace.time_s[0]))))
  valid_windows = np.convolve(
    np.asarray(condition, dtype=np.int32),
    np.ones(frame_count, dtype=np.int32),
    mode="valid",
  )
  candidates = np.flatnonzero(valid_windows[start_index:] >= frame_count)
  if candidates.size == 0:
    return None
  window_start = start_index + int(candidates[0])
  return float(trace.time_s[window_start] - trace.time_s[start_index])


def _longest_false_duration(condition: np.ndarray, dt_s: float) -> float:
  longest = 0
  current = 0
  for value in np.asarray(condition, dtype=np.bool_):
    current = 0 if value else current + 1
    longest = max(longest, current)
  return float(longest * dt_s)


def evaluate_extreme_surface() -> tuple[dict[str, Any], PhysicsTrace]:
  config = extreme_surface_config()
  trace, common = run_scenario(config)
  continuous_mask = (trace.time_s >= config.settling_time_s) & (
    trace.time_s < config.pose_step_time_s
  )
  continuous_contacts = trace.actual_contacts[continuous_mask]
  continuous_forces = trace.fingertip_forces_n[continuous_mask]
  continuous_curvature = trace.surface_curvatures_inv_m[continuous_mask]
  curvature_bins: dict[str, Any] = {}
  for name, lower, upper in (
    ("low_0_10", 0.0, 10.0),
    ("high_10_40", 10.0, 40.0),
    ("extreme_ge_40", 40.0, np.inf),
  ):
    selection = (continuous_curvature >= lower) & (continuous_curvature < upper)
    selected_force = continuous_forces[selection]
    selected_contact = continuous_contacts[selection]
    curvature_bins[name] = {
      "sample_count": int(np.sum(selection)),
      "contact_retention": float(np.mean(selected_contact)),
      "force_rmse_n": float(
        np.sqrt(np.mean((selected_force - config.desired_force_n) ** 2))
      ),
      "max_tip_force_n": float(np.max(selected_force)),
    }

  continuous = {
    "hand_contact_probability": float(
      np.mean(np.any(continuous_contacts, axis=1))
    ),
    "average_contact_count": float(
      np.mean(np.sum(continuous_contacts, axis=1))
    ),
    "per_finger_contact_retention": [
      float(value) for value in np.mean(continuous_contacts, axis=0)
    ],
    "force_rmse_n": float(
      np.sqrt(np.mean((continuous_forces - config.desired_force_n) ** 2))
    ),
    "force_violation_probability": float(
      np.mean(continuous_forces > config.force_limit_n)
    ),
    "max_tip_force_n": float(np.max(continuous_forces)),
    "thumb_contact_probability": float(np.mean(continuous_contacts[:, 3])),
    "contact_distal_head_clearance_min_m": float(
      np.nanmin(trace.contact_head_clearances_m[continuous_mask])
    ),
    "relative_path_length_m": float(
      np.sum(
        np.linalg.norm(
          np.diff(trace.object_positions_m[continuous_mask], axis=0),
          axis=1,
        )
      )
    ),
    "duration_s": float(config.pose_step_time_s - config.settling_time_s),
    "curvature_bins": curvature_bins,
  }

  step_index = int(np.searchsorted(trace.time_s, config.pose_step_time_s))
  any_contact = np.any(trace.actual_contacts, axis=1)
  all_contacts = np.all(trace.actual_contacts, axis=1)
  per_finger_recovery = [
    _first_sustained_time(
      trace,
      trace.actual_contacts[:, finger_index],
      step_index,
      0.050,
    )
    for finger_index in range(4)
  ]
  settled_force = all_contacts & np.all(
    (trace.fingertip_forces_n >= 1.5)
    & (trace.fingertip_forces_n <= 2.5),
    axis=1,
  )
  one_second_after_step = (trace.time_s >= config.pose_step_time_s) & (
    trace.time_s < config.pose_step_time_s + 1.0
  )
  pose_step = {
    "time_s": config.pose_step_time_s,
    "magnitude_m": config.pose_step_m,
    "pre_step_contact_set": [
      int(index + 1)
      for index, active in enumerate(trace.actual_contacts[step_index - 1])
      if active
    ],
    "immediate_contact_set": [
      int(index + 1)
      for index, active in enumerate(trace.actual_contacts[step_index])
      if active
    ],
    "any_contact_recovery_s": _first_sustained_time(
      trace,
      any_contact,
      step_index,
      0.050,
    ),
    "per_finger_contact_recovery_s": per_finger_recovery,
    "all_finger_contact_recovery_s": _first_sustained_time(
      trace,
      all_contacts,
      step_index,
      0.050,
    ),
    "force_settling_s": _first_sustained_time(
      trace,
      settled_force,
      step_index,
      0.100,
    ),
    "longest_zero_contact_s": _longest_false_duration(
      any_contact[step_index:],
      config.dt_s,
    ),
    "max_tip_force_first_second_n": float(
      np.max(trace.fingertip_forces_n[one_second_after_step])
    ),
    "final_contact_set": [
      int(index + 1)
      for index, active in enumerate(trace.actual_contacts[-1])
      if active
    ],
  }
  result = {
    **common,
    "config": asdict(config),
    "profile": profile_characteristics(),
    "continuous_sweep": continuous,
    "pose_step_recovery": pose_step,
    "thresholds": EXTREME_THRESHOLDS,
  }
  recovery_values = (
    pose_step["any_contact_recovery_s"],
    pose_step["all_finger_contact_recovery_s"],
    pose_step["force_settling_s"],
  )
  result["thresholds_met"] = bool(
    all(value is not None for value in recovery_values)
    and continuous["hand_contact_probability"]
    >= EXTREME_THRESHOLDS["continuous_hand_contact_probability_min"]
    and continuous["average_contact_count"]
    >= EXTREME_THRESHOLDS["continuous_average_contact_count_min"]
    and continuous["force_rmse_n"]
    <= EXTREME_THRESHOLDS["continuous_force_rmse_n_max"]
    and continuous["force_violation_probability"]
    <= EXTREME_THRESHOLDS["continuous_force_violation_probability_max"]
    and continuous["max_tip_force_n"]
    <= EXTREME_THRESHOLDS["continuous_max_tip_force_n_max"]
    and continuous["thumb_contact_probability"]
    >= EXTREME_THRESHOLDS["continuous_thumb_contact_probability_min"]
    and continuous["contact_distal_head_clearance_min_m"]
    >= EXTREME_THRESHOLDS["continuous_contact_distal_head_clearance_min_m"]
    and continuous["relative_path_length_m"]
    >= EXTREME_THRESHOLDS["continuous_relative_path_length_m_min"]
    and pose_step["any_contact_recovery_s"]
    <= EXTREME_THRESHOLDS["pose_step_any_contact_recovery_s_max"]
    and pose_step["all_finger_contact_recovery_s"]
    <= EXTREME_THRESHOLDS["pose_step_all_finger_recovery_s_max"]
    and pose_step["longest_zero_contact_s"]
    <= EXTREME_THRESHOLDS["pose_step_longest_zero_contact_s_max"]
    and pose_step["force_settling_s"]
    <= EXTREME_THRESHOLDS["pose_step_force_settling_s_max"]
    and common["minimum_joint_margin_rad"]
    >= EXTREME_THRESHOLDS["minimum_joint_margin_rad_min"]
    and common["non_tip_contact_count"]
    <= EXTREME_THRESHOLDS["non_tip_contact_count_max"]
  )
  return result, trace


def run_physics_evaluation() -> tuple[
  dict[str, Any],
  list[dict[str, Any]],
  dict[str, PhysicsTrace],
]:
  nominal = PhysicsConfig()
  maintenance, traces = evaluate_contact_maintenance(nominal)
  handover, handover_trace = evaluate_contact_handover(nominal)
  traces["handover"] = handover_trace
  robustness, robustness_rows = evaluate_control_robustness()
  extreme_surface, extreme_trace = evaluate_extreme_surface()
  traces["extreme_surface"] = extreme_trace
  baseline_sanity_thresholds_met = bool(
    maintenance["thresholds_met"]
    and handover["thresholds_met"]
    and robustness["thresholds_met"]
  )
  all_thresholds_met = bool(
    baseline_sanity_thresholds_met and extreme_surface["thresholds_met"]
  )
  if all_thresholds_met:
    benchmark_verdict = "ALL_THRESHOLDS_MET"
  elif not baseline_sanity_thresholds_met:
    benchmark_verdict = "BASELINE_SANITY_THRESHOLDS_NOT_MET"
  else:
    benchmark_verdict = "EXTREME_THRESHOLDS_NOT_MET"
  summary = {
    "schema_version": "e05-physics-baseline-v3",
    "experiment": "E05-PHY-v3",
    "method": "explicit_fingertip_reference_plus_fingertip_mcc",
    "physics_engine": f"MuJoCo {mujoco.__version__}",
    "protocol": str(PROTOCOL_PATH),
    "protocol_sha256": protocol_sha256(),
    "nominal_config": asdict(nominal),
    "contact_maintenance": maintenance,
    "contact_handover": handover,
    "control_robustness": robustness,
    "extreme_surface_challenge": extreme_surface,
    "evaluation_status": "COMPLETED",
    "evaluation_completed": True,
    "benchmark_verdict": benchmark_verdict,
    "baseline_sanity_thresholds_met": baseline_sanity_thresholds_met,
    "extreme_challenge_thresholds_met": extreme_surface["thresholds_met"],
    "all_thresholds_met": all_thresholds_met,
    "evaluated_arms": ["mcc_baseline"],
    "excluded_arms": ["finger_dp"],
    "comparison_performed": False,
    "gate_g1_unlocked": False,
    "claims": {
      "uses_physics_engine": True,
      "physical_contact_force_source": "mujoco.mj_contactForce",
      "dynamic_wrist_evaluated": False,
      "relative_wrist_motion_equivalence": "fixed_palm_plus_inverse_mocap_object",
      "non_tip_collision_evaluated": False,
      "contact_proxy": "distal_belly_ellipsoid",
      "pad_full_size_m": [float(value) for value in 2.0 * PAD_HALF_SIZE_M],
      "pad_parent_bodies": [finger.tip_body for finger in FINGERS],
      "pad_fsr_reference_bodies": [finger.fsr_body for finger in FINGERS],
      "pad_distal_head_clearance_design_min_m": float(
        min(
          finger.pad_local_position_m[1]
          - PAD_HALF_SIZE_M[0]
          - finger.distal_head_y_m
          for finger in FINGERS
        )
      ),
      "thumb_nominal_joint_angles_rad": [
        float(value) for value in Q_NOMINAL[12:16]
      ],
      "extreme_episode_duration_s": extreme_surface_config().duration_s,
      "extreme_nominal_sweep_displacement_m": 0.480,
    },
  }
  return summary, robustness_rows, traces


def _write_trace_archive(path: Path, traces: dict[str, PhysicsTrace]) -> None:
  arrays: dict[str, np.ndarray] = {}
  for scenario, trace in traces.items():
    for item in fields(trace):
      arrays[f"{scenario}__{item.name}"] = np.asarray(getattr(trace, item.name))
  np.savez_compressed(path, **arrays)


def write_results(
  output_dir: Path,
  summary: dict[str, Any],
  robustness_rows: list[dict[str, Any]],
  traces: dict[str, PhysicsTrace],
) -> dict[str, Path]:
  output = Path(output_dir)
  output.mkdir(parents=True, exist_ok=True)
  summary_path = output / "summary.json"
  episodes_path = output / "robustness_episodes.csv"
  traces_path = output / "traces.npz"
  summary_path.write_text(
    json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
  )
  with episodes_path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(robustness_rows[0]))
    writer.writeheader()
    writer.writerows(robustness_rows)
  _write_trace_archive(traces_path, traces)
  return {
    "summary": summary_path,
    "episodes": episodes_path,
    "traces": traces_path,
  }
