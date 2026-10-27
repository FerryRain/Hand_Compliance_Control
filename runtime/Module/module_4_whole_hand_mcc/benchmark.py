"""Formal MCC-only E05-F/H evaluator and provenance writer."""

from __future__ import annotations

import csv
from dataclasses import asdict, fields
import hashlib
import json
import platform
from pathlib import Path
import subprocess
from typing import Any, Iterable

import mujoco
import numpy as np

from Module.fr3_leap import FullRobotModelConfig, build_full_robot, export_model_xml, model_audit
from Module.module_4_whole_hand_mcc.runner import (
  E05MCCConfig,
  E05MCCTrace,
  run_e05_mcc,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPO_ROOT / "Module/E05_MCC_CURRENT_PROTOCOL.md"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "Module/generated/e05_mcc_current"
CODE_PATHS = (
  REPO_ROOT / "Module/e05_physics/extreme_surface.py",
  REPO_ROOT / "Module/fr3_leap/model.py",
  REPO_ROOT / "Module/common/full_robot_contracts.py",
  REPO_ROOT / "Module/module_1_oracle_surface_model/robot_geometry.py",
  REPO_ROOT / "Module/module_2_fingertip_mcc/full_robot.py",
  REPO_ROOT / "Module/module_3_runtime_guards/full_robot_guards.py",
  REPO_ROOT / "Module/module_4_whole_hand_mcc/coordinator.py",
  REPO_ROOT / "Module/module_4_whole_hand_mcc/wrist_mcc.py",
  REPO_ROOT / "Module/module_4_whole_hand_mcc/robot_control.py",
  REPO_ROOT / "Module/module_4_whole_hand_mcc/runner.py",
  Path(__file__).resolve(),
)

COMMON_THRESHOLDS: dict[str, tuple[str, float]] = {
  "contact_continuity_probability": (">=", 0.995),
  "average_contact_count": (">=", 3.0),
  "zero_contact_time_s": ("<=", 0.05),
  "force_rmse_n": ("<=", 1.0),
  "force_violation_probability": ("<=", 0.001),
  "max_tip_force_n": ("<=", 8.0),
  "four_contact_recovery_s": ("<=", 0.25),
  "force_settling_s": ("<=", 0.75),
  "traversal_y_m": (">=", 0.16),
  "palm_position_tracking_rmse_m": ("<=", 0.008),
  "minimum_arm_joint_margin_rad": (">=", 0.03),
  "minimum_finger_joint_margin_rad": (">=", 0.03),
  "controller_latency_p95_s": ("<=", 0.002),
  "deadline_miss_probability": ("<=", 0.01),
  "non_tip_contact_count": ("==", 0.0),
}
H_THRESHOLDS: dict[str, tuple[str, float]] = {
  "wrist_force_z_rmse_n": ("<=", 2.5),
  "max_wrist_compliance_translation_m": ("<=", 0.0121),
  "coordinator_internal_leakage_p95_n": ("<=", 0.05),
}


def formal_episode_configs(mode: str) -> tuple[tuple[str, E05MCCConfig], ...]:
  return (
    ("nominal", E05MCCConfig(mode=mode, seed=7)),
    (
      "low_friction",
      E05MCCConfig(
        mode=mode,
        seed=11,
        friction_coefficient=0.75,
        force_noise_std_n=0.03,
        initial_joint_noise_std_rad=0.004,
      ),
    ),
    (
      "noisy_pose",
      E05MCCConfig(
        mode=mode,
        seed=19,
        friction_coefficient=1.05,
        force_noise_std_n=0.05,
        initial_joint_noise_std_rad=0.006,
      ),
    ),
  )


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1 << 20), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _code_hash() -> str:
  digest = hashlib.sha256()
  for path in CODE_PATHS:
    digest.update(str(path.relative_to(REPO_ROOT)).encode("utf-8"))
    digest.update(path.read_bytes())
  return digest.hexdigest()


def _git_metadata() -> dict[str, Any]:
  try:
    commit = subprocess.run(
      ["git", "rev-parse", "HEAD"],
      cwd=REPO_ROOT,
      check=True,
      capture_output=True,
      text=True,
    ).stdout.strip()
    dirty = bool(
      subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
      ).stdout.strip()
    )
    return {"commit": commit, "worktree_dirty": dirty}
  except (OSError, subprocess.CalledProcessError):
    return {"commit": None, "worktree_dirty": None}


def _evaluate_threshold(value: float, operator: str, threshold: float) -> bool:
  if operator == ">=":
    return value >= threshold
  if operator == "<=":
    return value <= threshold
  if operator == "==":
    return value == threshold
  raise ValueError(f"unknown threshold operator: {operator}")


def evaluate_episode_thresholds(metrics: dict[str, Any]) -> dict[str, Any]:
  thresholds = dict(COMMON_THRESHOLDS)
  if metrics["cell"] == "E05-H-MCC":
    thresholds.update(H_THRESHOLDS)
  checks: dict[str, Any] = {}
  for name, (operator, threshold) in thresholds.items():
    value = float(metrics[name])
    checks[name] = {
      "value": value,
      "operator": operator,
      "threshold": threshold,
      "met": _evaluate_threshold(value, operator, threshold),
    }
  return {
    "performance_met": all(check["met"] for check in checks.values()),
    "checks": checks,
  }


def _aggregate(cell_rows: list[dict[str, Any]]) -> dict[str, Any]:
  numeric_names = sorted(
    name
    for name, value in cell_rows[0]["metrics"].items()
    if isinstance(value, (int, float)) and not isinstance(value, bool)
  )
  numeric: dict[str, Any] = {}
  for name in numeric_names:
    values = np.asarray([row["metrics"][name] for row in cell_rows], dtype=np.float64)
    numeric[name] = {
      "mean": float(np.mean(values)),
      "min": float(np.min(values)),
      "max": float(np.max(values)),
    }
  unmet = sorted(
    {
      name
      for row in cell_rows
      for name, check in row["thresholds"]["checks"].items()
      if not check["met"]
    }
  )
  return {
    "execution_status": "EVALUATED",
    "episodes_completed": len(cell_rows),
    "performance_verdict": "MET" if not unmet else "NOT_MET",
    "unmet_metrics": unmet,
    "numeric_metrics": numeric,
  }


def _trace_payload(prefix: str, trace: E05MCCTrace) -> dict[str, np.ndarray]:
  return {
    f"{prefix}__{definition.name}": np.asarray(getattr(trace, definition.name))
    for definition in fields(trace)
  }


def load_base_trace(path: str | Path, mode: str) -> E05MCCTrace:
  prefix = "F" if mode == "E05-F-MCC" else "H"
  with np.load(path, allow_pickle=False) as payload:
    kwargs = {
      definition.name: payload[f"{prefix}__{definition.name}"]
      for definition in fields(E05MCCTrace)
    }
  return E05MCCTrace(**kwargs)


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
  flat_rows: list[dict[str, Any]] = []
  for row in rows:
    flat: dict[str, Any] = {
      "cell": row["cell"],
      "episode": row["episode"],
      "seed": row["config"]["seed"],
      "performance_met": row["thresholds"]["performance_met"],
    }
    for name, value in row["metrics"].items():
      if isinstance(value, (str, int, float, bool)) or value is None:
        flat[name] = value
      else:
        flat[name] = json.dumps(value, sort_keys=True)
    flat_rows.append(flat)
  names = sorted({name for row in flat_rows for name in row})
  with path.open("w", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=names)
    writer.writeheader()
    writer.writerows(flat_rows)


def _readme(summary: dict[str, Any]) -> str:
  f = summary["cells"]["E05-F-MCC"]
  h = summary["cells"]["E05-H-MCC"]
  return f"""# E05 MCC-only FR3 results

This directory is generated by the frozen `E05-MCC-CURRENT` evaluator.
The execution status is `EVALUATED`; performance is reported independently.

| Cell | Execution | Performance | Unmet metrics |
| --- | --- | --- | --- |
| E05-F-MCC | {f['execution_status']} | {f['performance_verdict']} | {', '.join(f['unmet_metrics']) or 'none'} |
| E05-H-MCC | {h['execution_status']} | {h['performance_verdict']} | {', '.join(h['unmet_metrics']) or 'none'} |

DP is not implemented or evaluated by this frozen result. `base_traces.npz`
contains the two nominal MCC physics traces used by the visual demo.
`generated_fr3_leap.xml` is the exact 23-DoF model, with absolute mesh paths,
used for this local run.
"""


def run_formal_benchmark(
  output_dir: str | Path = DEFAULT_OUTPUT_DIR,
  *,
  episode_configs: dict[str, tuple[tuple[str, E05MCCConfig], ...]] | None = None,
) -> dict[str, Any]:
  if not PROTOCOL_PATH.is_file():
    raise FileNotFoundError(PROTOCOL_PATH)
  output = Path(output_dir)
  output.mkdir(parents=True, exist_ok=True)
  is_formal = episode_configs is None
  configurations = episode_configs or {
    mode: formal_episode_configs(mode) for mode in ("E05-F-MCC", "E05-H-MCC")
  }
  rows: list[dict[str, Any]] = []
  base_traces: dict[str, np.ndarray] = {}
  for mode in ("E05-F-MCC", "E05-H-MCC"):
    for episode_name, config in configurations[mode]:
      trace, metrics = run_e05_mcc(config)
      threshold_result = evaluate_episode_thresholds(metrics)
      rows.append(
        {
          "cell": mode,
          "episode": episode_name,
          "config": asdict(config),
          "metrics": metrics,
          "thresholds": threshold_result,
        }
      )
      if episode_name == "nominal":
        base_traces.update(_trace_payload("F" if mode == "E05-F-MCC" else "H", trace))

  cells = {
    mode: _aggregate([row for row in rows if row["cell"] == mode])
    for mode in ("E05-F-MCC", "E05-H-MCC")
  }
  if not is_formal:
    for cell in cells.values():
      cell["execution_status"] = "EVIDENCE_ONLY"
  handles = build_full_robot(FullRobotModelConfig(surface="extreme", gravity_m_s2=0.0, arm_kp=1800.0, arm_damping_ratio=0.9))
  audit = model_audit(handles)
  audit["plant_structure_valid"] = bool(
    audit["nq"] == audit["nv"] == audit["nu"] == 23
    and audit["all_pads_face_down"]
    and audit["object_mocap_id"] == -1
    and audit["mount_geometrically_closed"]
  )
  summary: dict[str, Any] = {
    "experiment": "E05-MCC-CURRENT" if is_formal else "E05-MCC-CURRENT-QUICK-SMOKE",
    "execution_status": "EVALUATED" if is_formal else "EVIDENCE_ONLY",
    "scope": ["E05-F-MCC", "E05-H-MCC"],
    "dp_implemented": False,
    "dp_evaluated": False,
    "protocol": {
      "path": str(PROTOCOL_PATH.relative_to(REPO_ROOT)),
      "sha256": _sha256(PROTOCOL_PATH),
      "code_bundle_sha256": _code_hash(),
    },
    "environment": {
      "python": platform.python_version(),
      "mujoco": mujoco.__version__,
      "numpy": np.__version__,
      "environment_name": "handcomp",
    },
    "git": _git_metadata(),
    "model_audit": audit,
    "episodes": rows,
    "cells": cells,
  }
  (output / "summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True),
    encoding="utf-8",
  )
  (output / "model_audit.json").write_text(
    json.dumps(audit, indent=2, sort_keys=True),
    encoding="utf-8",
  )
  _write_csv(output / "episodes.csv", rows)
  np.savez_compressed(output / "base_traces.npz", **base_traces)
  export_model_xml(
    output / "generated_fr3_leap.xml",
    FullRobotModelConfig(surface="extreme", gravity_m_s2=0.0, arm_kp=1800.0, arm_damping_ratio=0.9),
  )
  (output / "README.md").write_text(_readme(summary), encoding="utf-8")
  return summary
