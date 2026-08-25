"""Formal paired benchmark for frozen I02/I03 Bunny physics protocols."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import subprocess
from time import perf_counter
from typing import Any

import mujoco
import numpy as np
import trimesh

from Module.i01_bunny_physics.surface import canonical_bunny_heightfield
from Module.i02_i03_bunny_physics.core import EVALUATOR_VERSION, TRACE_SCHEMA_VERSION
from Module.i02_i03_bunny_physics.runner import (
  I02I03BunnyConfig,
  run_i02_i03_bunny,
  save_trace,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "Module/generated/i02_i03_bunny_physics"
SEEDS = (7, 11, 19)
CELLS = ("i02_long", "i02_short", "i03_beam", "i03_shadow")
SOURCE_FILES = tuple(
  REPO_ROOT / path
  for path in (
    "Module/I02_I03_BUNNY_PROTOCOL.md",
    "Module/fr3_leap/model.py",
    "Module/i01_bunny_physics/surface.py",
    "Module/i01_bunny_physics/runner.py",
    "Module/i02_i03_bunny_physics/core.py",
    "Module/i02_i03_bunny_physics/runner.py",
    "Module/i02_i03_bunny_physics/benchmark.py",
    "Module/module_2_fingertip_mcc/controller.py",
    "Module/module_2_fingertip_mcc/full_robot.py",
    "Module/module_3_runtime_guards/force_safety_executor.py",
    "Module/module_3_runtime_guards/command_continuity.py",
    "Module/module_6_prefix_executor/executor.py",
    "Module/module_7_contact_mode_graph/graph.py",
    "Module/module_8_cheap_cert/cheap_cert.py",
    "Module/module_9_continuous_optimize/optimizer.py",
    "Module/module_10_exact_prefix_audit/audit.py",
    "Module/module_11_lazy_beam_search/search.py",
    "Module/module_12_shadow_viability/shadow.py",
  )
)


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1 << 20), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _bootstrap_ci(values: list[float], *, seed: int = 7) -> list[float]:
  array = np.asarray(values, dtype=np.float64)
  random = np.random.default_rng(seed)
  samples = random.choice(array, size=(10000, len(array)), replace=True)
  means = np.mean(samples, axis=1)
  return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
  traversal = [float(row["supported_cumulative_traversal_m"]) for row in rows]
  continuity = [float(row["nonempty_contact_fraction"]) for row in rows]
  peak = [float(row["peak_valid_fingertip_force_n"]) for row in rows]
  errors = [
    float(row["final_reposition_terminal_error_m"])
    for row in rows
    if row["final_reposition_terminal_error_m"] is not None
  ]
  margins = [
    float(row["actual_terminal_joint_margin_rad"])
    for row in rows
    if row["actual_terminal_joint_margin_rad"] is not None
  ]
  return {
    "episode_count": len(rows),
    "common_task_pass_count": sum(bool(row["common_task_pass"]) for row in rows),
    "mechanism_pass_count": sum(bool(row["mechanism_pass"]) for row in rows),
    "execution_failure_count": sum(
      row["stop_reason"] is not None and row["stop_reason"] != "DEAD_END"
      for row in rows
    ),
    "dead_end_count": int(sum(int(row["dead_end_count"]) for row in rows)),
    "supported_cumulative_traversal_m": {
      "values": traversal,
      "median": float(np.median(traversal)),
      "mean_95pct_bootstrap_ci": _bootstrap_ci(traversal),
    },
    "nonempty_contact_fraction": {
      "values": continuity,
      "median": float(np.median(continuity)),
      "mean_95pct_bootstrap_ci": _bootstrap_ci(continuity),
    },
    "peak_valid_fingertip_force_n": {
      "values": peak,
      "maximum": float(np.max(peak)),
    },
    "final_reposition_terminal_error_m": {
      "values": errors,
      "median": float(np.median(errors)) if errors else None,
    },
    "actual_terminal_joint_margin_rad": {
      "values": margins,
      "minimum": float(np.min(margins)) if margins else None,
    },
    "maximum_all_contact_loss_gap_s": float(
      np.max([row["maximum_all_contact_loss_gap_s"] for row in rows])
    ),
    "certificate_count": int(sum(row["certificate_count"] for row in rows)),
    "micro_barrier_count": int(sum(row["micro_barrier_count"] for row in rows)),
    "authority_violation_count": int(
      sum(row["authority_violation_count"] for row in rows)
    ),
    "shadow_execution_authority_count": int(
      sum(row["shadow_execution_authority_count"] for row in rows)
    ),
    "prediction_suffix_command_count": int(
      sum(row["prediction_suffix_command_count"] for row in rows)
    ),
    "over_force_ticks": int(sum(row["over_force_ticks"] for row in rows)),
    "non_tip_contact_ticks": int(sum(row["non_tip_contact_ticks"] for row in rows)),
    "all_certificates_authentic": all(row["all_certificates_authentic"] for row in rows),
    "fresh_measured_root_evidence": all(row["fresh_measured_root_evidence"] for row in rows),
  }


def _machine_metadata() -> dict[str, Any]:
  cpu_model = platform.processor()
  try:
    for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
      if line.lower().startswith("model name"):
        cpu_model = line.split(":", maxsplit=1)[1].strip()
        break
  except OSError:
    pass
  return {
    "platform": platform.platform(),
    "machine": platform.machine(),
    "cpu_model": cpu_model or None,
    "logical_cpu_count": os.cpu_count(),
  }


def run_benchmark(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
  start_hashes = {str(path.relative_to(REPO_ROOT)): _sha256(path) for path in SOURCE_FILES}
  output = output_dir.resolve()
  output.mkdir(parents=True, exist_ok=True)
  bunny = canonical_bunny_heightfield()
  visual_mesh = bunny.export_visual_mesh(output / "canonical_bunny_side_laid.obj")
  rows: list[dict[str, Any]] = []
  combined: dict[str, np.ndarray] = {}
  xml: str | None = None
  started = perf_counter()
  for cell in CELLS:
    for seed in SEEDS:
      trace, metrics, episode_xml = run_i02_i03_bunny(
        I02I03BunnyConfig(
          cell=cell,
          seed=seed,
          visual_mesh_path=str(visual_mesh),
        )
      )
      if xml is None:
        xml = episode_xml
      stem = f"{cell}_seed_{seed}"
      save_trace(output / f"trace_{stem}.npz", trace)
      for name, array in trace.npz_payload().items():
        combined[f"{stem}__{name}"] = array
      metrics["trace_path"] = f"trace_{stem}.npz"
      rows.append(metrics)
      print(
        f"{cell:10s} seed={seed:2d} "
        f"supported={1000*metrics['supported_cumulative_traversal_m']:6.2f} mm "
        f"continuity={metrics['nonempty_contact_fraction']:.5f} "
        f"peak={metrics['peak_valid_fingertip_force_n']:.3f} N "
        f"pass={metrics['common_task_pass']} dead={metrics['dead_end_count']} "
        f"stop={metrics['stop_reason']}",
        flush=True,
      )
  np.savez_compressed(output / "traces.npz", **combined)
  assert xml is not None
  (output / "generated_fr3_leap_bunny.xml").write_text(xml, encoding="utf-8")

  fieldnames = [
    "module_id",
    "cell",
    "seed",
    "common_task_pass",
    "mechanism_pass",
    "supported_cumulative_traversal_m",
    "maximum_actual_path_coordinate_m",
    "nonempty_contact_fraction",
    "four_contact_fraction",
    "maximum_all_contact_loss_gap_s",
    "peak_valid_fingertip_force_n",
    "over_force_ticks",
    "non_tip_contact_ticks",
    "authority_violation_count",
    "shadow_execution_authority_count",
    "prediction_suffix_command_count",
    "handover_4_3_4_measured",
    "certificate_count",
    "micro_barrier_count",
    "reposition_certificate_count",
    "reposition_barrier_count",
    "final_reposition_xy_error_m",
    "final_reposition_terminal_error_m",
    "dead_end_count",
    "predicted_terminal_viability",
    "actual_terminal_viability",
    "actual_terminal_joint_margin_rad",
    "minimum_physical_joint_margin_rad",
    "controller_latency_p95_s",
    "physics_latency_p95_s",
    "optimizer_latency_p95_s",
    "audit_latency_p95_s",
    "search_latency_s",
    "predicted_shadow_latency_s",
    "actual_shadow_latency_s",
    "stop_reason",
    "trace_path",
  ]
  with (output / "episodes.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)

  by_cell = {
    cell: _aggregate([row for row in rows if row["cell"] == cell])
    for cell in CELLS
  }
  long = by_cell["i02_long"]
  short = by_cell["i02_short"]
  beam = by_cell["i03_beam"]
  shadow = by_cell["i03_shadow"]
  short_rows = [row for row in rows if row["cell"] == "i02_short"]
  shadow_rows = [row for row in rows if row["cell"] == "i03_shadow"]
  i02_reposition_evidence = all(
    (not row["common_task_pass"])
    or (
      row["reposition_certificate_count"] == 3
      and row["reposition_barrier_count"] == 3
      and row["fresh_measured_root_evidence"]
    )
    for row in short_rows
  )
  long_error = long["final_reposition_terminal_error_m"]["median"]
  short_error = short["final_reposition_terminal_error_m"]["median"]
  error_improvement = bool(
    long_error is not None
    and short_error is not None
    and short_error <= 0.80 * long_error + 0.00025
  )
  failure_improvement = (
    short["execution_failure_count"] < long["execution_failure_count"]
  )
  i02_met = bool(
    short["common_task_pass_count"] >= 2
    and short["mechanism_pass_count"] >= 2
    and i02_reposition_evidence
    and short["common_task_pass_count"] >= long["common_task_pass_count"]
    and short["supported_cumulative_traversal_m"]["median"]
    >= long["supported_cumulative_traversal_m"]["median"] - 0.005
    and (failure_improvement or error_improvement)
    and short["all_certificates_authentic"]
    and short["prediction_suffix_command_count"] == 0
    and short["authority_violation_count"] == 0
  )
  shadow_margin_ok = all(
    row["actual_terminal_joint_margin_rad"] is not None
    and row["actual_terminal_joint_margin_rad"] >= 0.025 - 1e-12
    and len(row["actual_successor_fingers"]) >= 1
    for row in shadow_rows
  )
  i03_dead_end_reduction = beam["dead_end_count"] - shadow["dead_end_count"]
  i03_traversal_advantage = (
    shadow["supported_cumulative_traversal_m"]["median"]
    - beam["supported_cumulative_traversal_m"]["median"]
  )
  i03_met = bool(
    shadow["common_task_pass_count"] >= 2
    and shadow["dead_end_count"] == 0
    and shadow["mechanism_pass_count"] >= 2
    and i03_dead_end_reduction >= 2
    and i03_traversal_advantage >= 0.030
    and shadow_margin_ok
    and shadow["shadow_execution_authority_count"] == 0
    and shadow["prediction_suffix_command_count"] == 0
    and shadow["authority_violation_count"] == 0
  )
  recommended_clean = all(
    row["over_force_ticks"] == 0
    and row["non_tip_contact_ticks"] == 0
    and row["authority_violation_count"] == 0
    and row["shadow_execution_authority_count"] == 0
    and row["prediction_suffix_command_count"] == 0
    and row["dead_end_count"] == 0
    for row in short_rows + shadow_rows
  )
  g3_go = bool(
    i02_met
    and i03_met
    and short["common_task_pass_count"] >= 2
    and shadow["common_task_pass_count"] >= 2
    and recommended_clean
    and short["all_certificates_authentic"]
    and shadow["all_certificates_authentic"]
    and short["fresh_measured_root_evidence"]
    and shadow["fresh_measured_root_evidence"]
  )

  end_hashes = {str(path.relative_to(REPO_ROOT)): _sha256(path) for path in SOURCE_FILES}
  changed = sorted(path for path in start_hashes if start_hashes[path] != end_hashes[path])
  if changed:
    raise RuntimeError("source changed during formal benchmark: " + ", ".join(changed))
  git_head = subprocess.run(
    ["git", "rev-parse", "HEAD"],
    cwd=REPO_ROOT,
    check=True,
    capture_output=True,
    text=True,
  ).stdout.strip()
  git_dirty = bool(
    subprocess.run(
      ["git", "status", "--porcelain"],
      cwd=REPO_ROOT,
      check=True,
      capture_output=True,
      text=True,
    ).stdout.strip()
  )
  summary: dict[str, Any] = {
    "module_ids": ["I02-PHY-BUNNY-v1", "I03-PHY-BUNNY-v1"],
    "status": {
      "i02": "EVALUATED / MET" if i02_met else "EVALUATED / NOT_MET",
      "i03": "EVALUATED / MET" if i03_met else "EVALUATED / NOT_MET",
      "gate_g3": "GO" if g3_go else "NO_GO",
    },
    "trace_schema_version": TRACE_SCHEMA_VERSION,
    "evaluator_version": EVALUATOR_VERSION,
    "protocol": "Module/I02_I03_BUNNY_PROTOCOL.md",
    "fair_comparison": {
      "paired_seeds": list(SEEDS),
      "cells": list(CELLS),
      "same_bunny_robot_path_mcc_guards_and_exact_mesh_evaluator": True,
      "i02_only_changed_factor": "one 12 mm prefix versus 3 x 4 mm fresh-root prefixes",
      "i03_only_changed_factor": "M12 terminal predicate disabled versus enabled",
      "dp_enabled": False,
      "gravity_m_s2": 0.0,
      "timestep_s": 0.002,
      "duration_s": 20.0,
      "planned_cumulative_traversal_m": 0.110,
      "desired_force_n": 2.0,
      "hard_force_limit_n": 8.0,
      "decision_profile_hand_kp": 22.0,
      "long_traversal_profile_hand_kp": 16.0,
    },
    "by_cell": by_cell,
    "i02_acceptance": {
      "met": i02_met,
      "short_reposition_evidence": i02_reposition_evidence,
      "failure_improvement": failure_improvement,
      "terminal_error_improvement": error_improvement,
      "long_median_terminal_error_m": long_error,
      "short_median_terminal_error_m": short_error,
      "median_supported_traversal_difference_short_minus_long_m": (
        short["supported_cumulative_traversal_m"]["median"]
        - long["supported_cumulative_traversal_m"]["median"]
      ),
    },
    "i03_acceptance": {
      "met": i03_met,
      "paired_dead_end_reduction": i03_dead_end_reduction,
      "median_supported_traversal_advantage_m": i03_traversal_advantage,
      "shadow_terminal_margin_and_successor_ok": shadow_margin_ok,
    },
    "gate_g3": {
      "decision": "GO" if g3_go else "NO_GO",
      "recommended_cells_clean": recommended_clean,
      "scope": "fixed known Bunny, Geometry Oracle + explicit MCC baseline only",
    },
    "episodes": rows,
    "geometry": {
      "source_path": str(bunny.source_path.relative_to(REPO_ROOT)),
      "source_sha256": bunny.source_sha256,
      "visual_mesh": visual_mesh.name,
      "extents_m": bunny.extents_m.tolist(),
      "hfield_shape": list(bunny.height_m.shape),
    },
    "runtime": {
      "wall_time_s": perf_counter() - started,
      "machine": _machine_metadata(),
      "versions": {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "mujoco": mujoco.__version__,
        "trimesh": trimesh.__version__,
        "scipy": importlib.metadata.version("scipy"),
      },
      "git_head": git_head,
      "git_worktree_dirty": git_dirty,
      "source_stability": "PASS",
      "source_sha256": start_hashes,
    },
  }
  (output / "summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True, allow_nan=False),
    encoding="utf-8",
  )
  return summary


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
  arguments = parser.parse_args()
  summary = run_benchmark(arguments.output)
  print(json.dumps(summary["status"], indent=2), flush=True)


if __name__ == "__main__":
  main()
