"""Reproducible shared-execution safety audit for G1a.

G1a deliberately does not score learned-reference value.  It only checks that
the shared Wrist MCC, coordinated Finger MCC, role interpreter, command
continuity limiter and runtime guard can be used unchanged by every Exp. 2
reference source without unsafe command jumps or force-limit violations.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import json
from pathlib import Path
from typing import Any

import numpy as np

from Module.module_4_whole_hand_mcc.runner import (
  E05MCCConfig,
  E05MCCTrace,
  run_e05_mcc,
)
from Module.module_4_whole_hand_mcc.visual_demo import render_video


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "Module/generated/g1a_shared_stack"
SCHEMA_VERSION = "fr3-leap-g1a-shared-stack.v1"


def g1a_configs(*, duration_s: float = 15.0) -> tuple[tuple[str, E05MCCConfig], ...]:
  common = dict(
    mode="E05-H-MCC",
    duration_s=duration_s,
    pose_step_time_s=min(9.0, duration_s - 1.0),
    enforce_shared_force_safety=True,
  )
  return (
    ("nominal", E05MCCConfig(seed=7, **common)),
    (
      "low_friction",
      E05MCCConfig(
        seed=11,
        friction_coefficient=0.75,
        force_noise_std_n=0.03,
        initial_joint_noise_std_rad=0.004,
        **common,
      ),
    ),
    (
      "noisy_pose",
      E05MCCConfig(
        seed=19,
        friction_coefficient=1.05,
        force_noise_std_n=0.05,
        initial_joint_noise_std_rad=0.006,
        **common,
      ),
    ),
  )


def _trace_payload(trace: E05MCCTrace) -> dict[str, np.ndarray]:
  return {
    definition.name: np.asarray(getattr(trace, definition.name))
    for definition in fields(trace)
  }


def _safety_checks(
  trace: E05MCCTrace,
  metrics: dict[str, Any],
  config: E05MCCConfig,
) -> dict[str, dict[str, Any]]:
  finger_step = float(np.max(np.abs(np.diff(trace.finger_command_rad, axis=0))))
  wrist_step = float(
    np.max(np.linalg.norm(np.diff(trace.commanded_palm_pose_world[:, :3], axis=0), axis=1))
  )
  # The orthogonal normal/tangent bounds imply this conservative Euclidean
  # bound.  The hard-release branch has its own explicitly larger bound.
  nominal_wrist_bound = float(
    np.hypot(
      config.shared_wrist_translation_step_m,
      config.shared_wrist_normal_step_m,
    )
  )
  hard_wrist_bound = float(
    np.hypot(
      config.shared_wrist_translation_step_m * 0.02,
      config.shared_hard_wrist_step_m,
    )
  )
  wrist_bound = max(nominal_wrist_bound, hard_wrist_bound)
  definitions = {
    "max_tip_force_n": (
      float(metrics["max_tip_force_n"]),
      "<=",
      float(config.force_limit_n),
    ),
    "force_violation_time_s": (
      float(metrics["force_violation_time_s"]),
      "==",
      0.0,
    ),
    "hard_guard_frames": (float(metrics["hard_guard_frames"]), "==", 0.0),
    "safety_aborted_frames": (
      float(metrics["safety_aborted_frames"]),
      "==",
      0.0,
    ),
    "non_tip_contact_count": (
      float(metrics["non_tip_contact_count"]),
      "==",
      0.0,
    ),
    "palm_position_tracking_rmse_m": (
      float(metrics["palm_position_tracking_rmse_m"]),
      "<=",
      0.008,
    ),
    "max_finger_command_step_rad": (
      finger_step,
      "<=",
      float(config.shared_active_finger_step_rad) + 1e-9,
    ),
    "max_wrist_translation_step_m": (
      wrist_step,
      "<=",
      wrist_bound + 1e-9,
    ),
  }
  checks: dict[str, dict[str, Any]] = {}
  for name, (value, operator, threshold) in definitions.items():
    met = value <= threshold if operator == "<=" else value == threshold
    checks[name] = {
      "value": value,
      "operator": operator,
      "threshold": threshold,
      "met": bool(met),
    }
  return checks


def run_g1a_audit(
  output_directory: str | Path = DEFAULT_OUTPUT,
  *,
  duration_s: float = 15.0,
  render: bool = True,
) -> dict[str, Any]:
  output = Path(output_directory)
  output.mkdir(parents=True, exist_ok=True)
  episodes: list[dict[str, Any]] = []
  nominal_trace: E05MCCTrace | None = None
  for name, config in g1a_configs(duration_s=duration_s):
    trace, metrics = run_e05_mcc(config)
    checks = _safety_checks(trace, metrics, config)
    verdict = "PASS" if all(item["met"] for item in checks.values()) else "FAIL"
    episodes.append(
      {
        "episode": name,
        "verdict": verdict,
        "config": asdict(config),
        "checks": checks,
        "diagnostic_performance": {
          key: metrics[key]
          for key in (
            "contact_continuity_probability",
            "average_contact_count",
            "zero_contact_time_s",
            "traversal_y_m",
            "force_rmse_n",
            "soft_recovery_frames",
          )
        },
      }
    )
    np.savez_compressed(output / f"{name}_trace.npz", **_trace_payload(trace))
    if name == "nominal":
      nominal_trace = trace

  verdict = "PASS" if all(row["verdict"] == "PASS" for row in episodes) else "FAIL"
  summary: dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "gate": "G1a",
    "verdict": verdict,
    "scope": "shared low-level execution safety only",
    "does_not_claim": [
      "DPRef learned-reference value",
      "Exp. 2 superiority",
      "active-planner readiness",
    ],
    "episodes": episodes,
  }
  (output / "summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True),
    encoding="utf-8",
  )
  lines = [
    "# G1a shared execution audit",
    "",
    f"Verdict: `{verdict}`.",
    "",
    "This gate checks only the shared MCC/interpreter/guard safety stack. ",
    "Contact richness and traversal remain diagnostics until G1b.",
    "",
    "| Episode | Safety | Max force (N) | Contact continuity | Avg. contacts | Traversal Y (m) |",
    "| --- | --- | ---: | ---: | ---: | ---: |",
  ]
  for row in episodes:
    diag = row["diagnostic_performance"]
    lines.append(
      "| {episode} | {verdict} | {force:.3f} | {continuity:.4f} | "
      "{contacts:.3f} | {traversal:.4f} |".format(
        episode=row["episode"],
        verdict=row["verdict"],
        force=row["checks"]["max_tip_force_n"]["value"],
        continuity=diag["contact_continuity_probability"],
        contacts=diag["average_contact_count"],
        traversal=diag["traversal_y_m"],
      )
    )
  lines.extend(
    (
      "",
      "Reproduce:",
      "",
      "```bash",
      "/home/ferry/data/Anaconda/envs/handcomp/bin/python -m "
      "Module.module_4_whole_hand_mcc.g1a_benchmark",
      "```",
    )
  )
  (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
  if render:
    if nominal_trace is None:
      raise AssertionError("nominal G1a trace was not generated")
    render_video(
      nominal_trace,
      "E05-H-MCC",
      output / "g1a_nominal_video.mp4",
      output / "g1a_nominal_frame.png",
    )
  return summary


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
  parser.add_argument("--duration", type=float, default=15.0)
  parser.add_argument("--no-render", action="store_true")
  args = parser.parse_args()
  summary = run_g1a_audit(
    args.output,
    duration_s=args.duration,
    render=not args.no_render,
  )
  print(json.dumps({"verdict": summary["verdict"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
  main()
