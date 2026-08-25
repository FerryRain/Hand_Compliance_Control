"""Exp. 2: ordinary MCC plus three sources using the shared safe MCC stack."""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields, replace
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from Module.module_4_finger_dp.dpref_reference_sources import (
  DPRefReferenceSource,
  PassiveHoldReferenceSource,
  ReactiveHeuristicReferenceSource,
)
from Module.module_4_whole_hand_mcc.runner import (
  E05MCCConfig,
  E05MCCTrace,
  run_e05_mcc,
)
from Module.module_4_whole_hand_mcc.visual_demo import render_video


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = REPO_ROOT / "Module/generated/dpref_v1/training_i100/dpref_checkpoint.pt"
DEFAULT_OUTPUT = REPO_ROOT / "Module/generated/exp2_dpref_mcc"
SCHEMA_VERSION = "fr3-leap-exp2-four-strategy.v3"

BRANCHES = (
  "PLAIN_WHOLE_HAND_MCC",
  "PASSIVE_HOLD_MCC",
  "REACTIVE_HEURISTIC_MCC",
  "DPREF_MCC",
)


def exp2_configs(*, duration_s: float = 15.0) -> tuple[tuple[str, E05MCCConfig], ...]:
  """Paired conditions with identical initial q across all four strategies."""

  common = dict(
    mode="E05-H-MCC",
    duration_s=duration_s,
    pose_step_time_s=min(9.0, duration_s - 1.0),
    enforce_shared_force_safety=True,
    initial_joint_noise_std_rad=0.0,
  )
  return (
    ("nominal", E05MCCConfig(seed=7, **common)),
    (
      "low_friction",
      E05MCCConfig(
        seed=11,
        friction_coefficient=0.75,
        force_noise_std_n=0.03,
        **common,
      ),
    ),
    (
      "noisy_observation",
      E05MCCConfig(
        seed=19,
        friction_coefficient=1.05,
        force_noise_std_n=0.05,
        **common,
      ),
    ),
  )


def _trace_payload(trace: E05MCCTrace) -> dict[str, np.ndarray]:
  return {
    definition.name: np.asarray(getattr(trace, definition.name))
    for definition in fields(trace)
  }


def _source_factory(
  branch: str,
  checkpoint: Path,
  device: str,
) -> Callable[[], Any] | None:
  if branch == "PLAIN_WHOLE_HAND_MCC":
    return None
  if branch == "PASSIVE_HOLD_MCC":
    return PassiveHoldReferenceSource
  if branch == "REACTIVE_HEURISTIC_MCC":
    return ReactiveHeuristicReferenceSource
  if branch == "DPREF_MCC":
    return lambda: DPRefReferenceSource(str(checkpoint), device=device)
  raise ValueError(f"unknown Exp. 2 branch {branch}")


def _limit_observations(metrics: dict[str, Any]) -> dict[str, Any]:
  peak_force = float(metrics["max_tip_force_n"])
  return {
    "force_reference_limit_n": 8.0,
    "force_peak_excess_n": max(0.0, peak_force - 8.0),
    "force_violation_time_s": float(metrics["force_violation_time_s"]),
    "force_violation_max_consecutive_time_s": float(
      metrics["force_violation_max_consecutive_time_s"]
    ),
    "multi_pad_force_violation_time_s": float(
      metrics["multi_pad_force_violation_time_s"]
    ),
    "force_excess_impulse_n_s": float(metrics["force_excess_impulse_n_s"]),
    "force_above_20n_time_s": float(metrics["force_above_20n_time_s"]),
    "hard_guard_frames": int(metrics["hard_guard_frames"]),
    "abort_frames": int(metrics["safety_aborted_frames"]),
    "non_tip_contact_count": int(metrics["non_tip_contact_count"]),
    "palm_tracking_reference_m": 0.008,
    "palm_tracking_excess_m": max(
      0.0, float(metrics["palm_position_tracking_rmse_m"]) - 0.008
    ),
  }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
  keys = (
    "contact_continuity_probability",
    "average_contact_count",
    "contact_count_ge2_probability",
    "contact_count_ge3_probability",
    "four_contact_probability",
    "zero_contact_time_s",
    "force_rmse_n",
    "max_tip_force_n",
    "force_violation_time_s",
    "force_violation_max_consecutive_time_s",
    "multi_pad_force_violation_probability",
    "multi_pad_force_violation_time_s",
    "force_excess_impulse_n_s",
    "force_above_20n_time_s",
    "traversal_y_m",
    "positive_y_traversal_m",
    "supported_y_traversal_ge1_m",
    "supported_y_traversal_ge2_m",
    "contact_loss_events",
    "hard_guard_frames",
    "soft_recovery_frames",
    "reference_inference_latency_p95_s",
    "controller_latency_p95_s",
    "palm_position_tracking_rmse_m",
  )
  result: dict[str, Any] = {}
  for key in keys:
    values = np.asarray([row["metrics"][key] for row in rows], dtype=np.float64)
    result[key] = {
      "mean": float(np.mean(values)),
      "min": float(np.min(values)),
      "max": float(np.max(values)),
    }
  result["worst_force_reference_excess_n"] = max(
    float(row["limit_observations"]["force_peak_excess_n"]) for row in rows
  )
  per_finger = np.asarray(
    [row["metrics"]["per_finger_contact_probability"] for row in rows],
    dtype=np.float64,
  )
  result["per_finger_contact_probability"] = {
    "mean": np.mean(per_finger, axis=0).tolist(),
    "min": np.min(per_finger, axis=0).tolist(),
    "max": np.max(per_finger, axis=0).tolist(),
  }
  return result


def _comparison_observations(
  aggregates: dict[str, dict[str, Any]],
  training_summary: dict[str, Any],
) -> dict[str, Any]:
  plain = aggregates["PLAIN_WHOLE_HAND_MCC"]
  passive = aggregates["PASSIVE_HOLD_MCC"]
  reactive = aggregates["REACTIVE_HEURISTIC_MCC"]
  dpref = aggregates["DPREF_MCC"]
  best_supported = max(
    passive["supported_y_traversal_ge2_m"]["mean"],
    reactive["supported_y_traversal_ge2_m"]["mean"],
  )
  best_continuity = max(
    passive["contact_continuity_probability"]["mean"],
    reactive["contact_continuity_probability"]["mean"],
  )
  best_contacts = max(
    passive["average_contact_count"]["mean"],
    reactive["average_contact_count"]["mean"],
  )
  return {
    "plain_minus_passive_average_contacts": (
      plain["average_contact_count"]["mean"]
      - passive["average_contact_count"]["mean"]
    ),
    "plain_minus_passive_contact_continuity": (
      plain["contact_continuity_probability"]["mean"]
      - passive["contact_continuity_probability"]["mean"]
    ),
    "plain_minus_passive_supported_y_ge2_m": (
      plain["supported_y_traversal_ge2_m"]["mean"]
      - passive["supported_y_traversal_ge2_m"]["mean"]
    ),
    "best_passive_or_reactive_supported_y_ge2_m": best_supported,
    "dpref_supported_y_ge2_difference_m": (
      dpref["supported_y_traversal_ge2_m"]["mean"] - best_supported
    ),
    "dpref_contact_continuity_difference_from_best_control": (
      dpref["contact_continuity_probability"]["mean"] - best_continuity
    ),
    "dpref_average_contacts_difference_from_best_control": (
      dpref["average_contact_count"]["mean"] - best_contacts
    ),
    "role_coverage_limited": not bool(
      training_summary["role_coverage"]["handover_generalization_claim_allowed"]
    ),
    "handover_generalization_claim_allowed": bool(
      training_summary["role_coverage"]["handover_generalization_claim_allowed"]
    ),
  }


def run_exp2(
  checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
  output_directory: str | Path = DEFAULT_OUTPUT,
  *,
  device: str = "cuda:0",
  duration_s: float = 15.0,
  render: bool = True,
) -> dict[str, Any]:
  checkpoint = Path(checkpoint_path)
  training_summary_path = checkpoint.parent / "training_summary.json"
  training_summary = json.loads(training_summary_path.read_text(encoding="utf-8"))
  output = Path(output_directory)
  output.mkdir(parents=True, exist_ok=True)
  rows: list[dict[str, Any]] = []
  nominal_traces: dict[str, E05MCCTrace] = {}
  for branch in BRANCHES:
    factory = _source_factory(branch, checkpoint, device)
    for episode, config in exp2_configs(duration_s=duration_s):
      branch_config = (
        replace(config, enforce_shared_force_safety=False)
        if branch == "PLAIN_WHOLE_HAND_MCC"
        else config
      )
      source = None if factory is None else factory()
      print(f"[Exp.2] start {branch}/{episode}", flush=True)
      trace, metrics = run_e05_mcc(branch_config, reference_source=source)
      print(
        "[Exp.2] done "
        f"{branch}/{episode}: continuity={metrics['contact_continuity_probability']:.4f}, "
        f"avg_contacts={metrics['average_contact_count']:.3f}, "
        f"peak={metrics['max_tip_force_n']:.3f} N",
        flush=True,
      )
      row = {
        "branch": branch,
        "episode": episode,
        "config": asdict(branch_config),
        "metrics": metrics,
        "limit_observations": _limit_observations(metrics),
      }
      rows.append(row)
      np.savez_compressed(
        output / f"{branch.lower()}_{episode}_trace.npz",
        **_trace_payload(trace),
      )
      if episode == "nominal":
        nominal_traces[branch] = trace
  aggregates = {
    branch: _aggregate([row for row in rows if row["branch"] == branch])
    for branch in BRANCHES
  }
  observations = _comparison_observations(aggregates, training_summary)
  summary = {
    "schema_version": SCHEMA_VERSION,
    "experiment": "EXP2_PLAIN_PASSIVE_REACTIVE_DPREF",
    "execution_status": "EVALUATED",
    "evaluation_semantics": "DESCRIPTIVE_ONLY_NO_STRATEGY_PASS_FAIL",
    "reference_limits": {"fingertip_force_n": 8.0, "palm_tracking_rmse_m": 0.008},
    "comparison_contract": {
      "shared": [
        "wrist trajectory",
        "robot/object/initial states",
        "friction/noise condition",
        "desired fingertip force",
      ],
      "absolute_reference": (
        "PLAIN_WHOLE_HAND_MCC is the historical/basic analytical controller "
        "without the new role interpreter or shared force-safety wrapper"
      ),
      "fair_reference_source_subset": [
        "PASSIVE_HOLD_MCC",
        "REACTIVE_HEURISTIC_MCC",
        "DPREF_MCC",
      ],
      "subset_only_variable": "nominal finger reference and role-intention source",
    },
    "checkpoint": str(checkpoint),
    "checkpoint_role_coverage": training_summary["role_coverage"],
    "rows": rows,
    "aggregates": aggregates,
    "comparison_observations": observations,
  }
  (output / "summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True),
    encoding="utf-8",
  )
  lines = [
    "# Exp. 2 — ordinary MCC plus three shared-stack reference sources",
    "",
    "Execution: `EVALUATED`. Strategies receive no Pass/Fail or MET/NOT_MET verdict.",
    "",
    "Plain whole-hand MCC is an absolute basic-controller reference. Passive, Reactive and DPRef",
    "share the adjusted Wrist/Finger MCC, role interpreter and guard; only their reference source changes.",
    "",
    "| Branch | Contact continuity | Avg contacts | P(Nc>=3) | Four-contact | Supported Y >=2 (m) | Peak (N) | >8 N time (s) | Multi-pad >8 N (s) |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
  ]
  for branch in BRANCHES:
    value = aggregates[branch]
    lines.append(
      f"| {branch} | {value['contact_continuity_probability']['mean']:.4f} | "
      f"{value['average_contact_count']['mean']:.3f} | "
      f"{value['contact_count_ge3_probability']['mean']:.4f} | "
      f"{value['four_contact_probability']['mean']:.4f} | "
      f"{value['supported_y_traversal_ge2_m']['mean']:.4f} | "
      f"{value['max_tip_force_n']['max']:.3f} | "
      f"{value['force_violation_time_s']['mean']:.4f} | "
      f"{value['multi_pad_force_violation_time_s']['mean']:.4f} |"
    )
  lines.extend(
    (
      "",
      "Interpret Plain-vs-Passive as the cost/benefit of the new safe execution stack. Interpret",
      "Passive-vs-Reactive-vs-DPRef as the fair nominal-reference-source comparison.",
      "",
      "Reproduce (CUDA is mandatory for DPRef):",
      "",
      "```bash",
      "/home/ferry/data/Anaconda/envs/handcomp/bin/python -m "
      "Module.module_4_finger_dp.exp2_benchmark",
      "```",
    )
  )
  (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
  if render:
    for branch, trace in nominal_traces.items():
      render_video(
        trace,
        "E05-H-MCC",
        output / f"{branch.lower()}_video.mp4",
        output / f"{branch.lower()}_frame.png",
      )
  return summary


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
  parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--duration", type=float, default=15.0)
  parser.add_argument("--no-render", action="store_true")
  args = parser.parse_args()
  summary = run_exp2(
    args.checkpoint,
    args.output,
    device=args.device,
    duration_s=args.duration,
    render=not args.no_render,
  )
  print(json.dumps({
    "execution": summary["execution_status"],
    "semantics": summary["evaluation_semantics"],
    "comparison_observations": summary["comparison_observations"],
  }, indent=2))


if __name__ == "__main__":
  main()
