"""Run the frozen paired I01 Bunny MuJoCo experiment and save evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from typing import Any

import mujoco
import numpy as np
import trimesh

from Module.i01_bunny_physics.runner import (
  EVALUATOR_VERSION,
  TRACE_SCHEMA_VERSION,
  I01BunnyConfig,
  run_i01_bunny,
  save_trace,
)
from Module.i01_bunny_physics.surface import canonical_bunny_heightfield


DEFAULT_OUTPUT_DIR = Path("Module/generated/i01_bunny_physics")
SEEDS = (7, 11, 19)
CELLS = ("fixed", "variable")
SOURCE_FILES = (
  Path("Module/I01_BUNNY_PROTOCOL.md"),
  Path("Module/e05_physics/scene.py"),
  Path("Module/fr3_leap/model.py"),
  Path("Module/i01_bunny_physics/surface.py"),
  Path("Module/i01_bunny_physics/runner.py"),
  Path("Module/i01_bunny_physics/benchmark.py"),
  Path("Module/module_2_fingertip_mcc/controller.py"),
  Path("Module/module_2_fingertip_mcc/full_robot.py"),
  Path("Module/module_3_runtime_guards/force_safety_executor.py"),
  Path("Module/module_4_whole_hand_mcc/robot_control.py"),
  Path("Module/module_6_prefix_executor/executor.py"),
  Path("Module/module_7_contact_mode_graph/graph.py"),
  Path("Module/module_9_continuous_optimize/optimizer.py"),
  Path("Module/module_10_exact_prefix_audit/audit.py"),
)


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _bootstrap_ci(values: list[float], *, seed: int = 7) -> list[float]:
  array = np.asarray(values, dtype=np.float64)
  rng = np.random.default_rng(seed)
  samples = rng.choice(array, size=(10000, len(array)), replace=True)
  means = np.mean(samples, axis=1)
  return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def _aggregate_cell(rows: list[dict[str, Any]]) -> dict[str, Any]:
  progress = [float(row["actual_progress_m"]) for row in rows]
  continuity = [float(row["nonempty_contact_fraction"]) for row in rows]
  peak_force = [float(row["peak_valid_fingertip_force_n"]) for row in rows]
  return {
    "episode_count": len(rows),
    "primary_pass_count": sum(bool(row["primary_pass"]) for row in rows),
    "variable_mechanism_pass_count": sum(
      bool(row["variable_mechanism_pass"]) for row in rows
    ),
    "actual_progress_m": {
      "values": progress,
      "median": float(np.median(progress)),
      "mean_95pct_bootstrap_ci": _bootstrap_ci(progress),
    },
    "nonempty_contact_fraction": {
      "values": continuity,
      "median": float(np.median(continuity)),
      "mean_95pct_bootstrap_ci": _bootstrap_ci(continuity),
    },
    "peak_valid_fingertip_force_n": {
      "values": peak_force,
      "maximum": float(np.max(peak_force)),
      "mean_95pct_bootstrap_ci": _bootstrap_ci(peak_force),
    },
    "maximum_all_contact_loss_gap_s": float(
      np.max([row["maximum_all_contact_loss_gap_s"] for row in rows])
    ),
    "authority_violation_count": int(
      sum(row["authority_violation_count"] for row in rows)
    ),
    "certificate_count": int(sum(row["certificate_count"] for row in rows)),
    "micro_barrier_count": int(sum(row["micro_barrier_count"] for row in rows)),
    "controller_latency_p95_s_max": float(
      np.max([row["controller_latency_p95_s"] for row in rows])
    ),
    "physics_latency_p95_s_max": float(
      np.max([row["physics_latency_p95_s"] for row in rows])
    ),
  }


def run_benchmark(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
  source_sha256_start = {
    str(path): _sha256(path.resolve()) for path in SOURCE_FILES
  }
  output = output_dir.resolve()
  output.mkdir(parents=True, exist_ok=True)
  bunny = canonical_bunny_heightfield()
  visual_mesh = bunny.export_visual_mesh(output / "canonical_bunny_side_laid.obj")
  episode_rows: list[dict[str, Any]] = []
  combined_trace: dict[str, np.ndarray] = {}
  xml: str | None = None
  started = perf_counter()
  for cell in CELLS:
    for seed in SEEDS:
      config = I01BunnyConfig(
        cell=cell,
        seed=seed,
        visual_mesh_path=str(visual_mesh),
      )
      trace, metrics, episode_xml = run_i01_bunny(config)
      xml = episode_xml if xml is None else xml
      stem = f"{cell}_seed_{seed}"
      save_trace(output / f"trace_{stem}.npz", trace)
      for name, array in trace.npz_payload().items():
        combined_trace[f"{stem}__{name}"] = array
      metrics["trace_path"] = f"trace_{stem}.npz"
      episode_rows.append(metrics)
      print(
        f"{cell:8s} seed={seed:2d} progress={1000*metrics['actual_progress_m']:6.2f} mm "
        f"continuity={metrics['nonempty_contact_fraction']:.5f} "
        f"peak={metrics['peak_valid_fingertip_force_n']:.3f} N "
        f"pass={metrics['primary_pass']} stop={metrics['stop_reason']}",
        flush=True,
      )
  np.savez_compressed(output / "traces.npz", **combined_trace)
  assert xml is not None
  (output / "generated_fr3_leap_bunny.xml").write_text(xml, encoding="utf-8")

  fieldnames = [
    "cell",
    "seed",
    "primary_pass",
    "variable_mechanism_pass",
    "actual_progress_m",
    "planned_progress_m",
    "nonempty_contact_fraction",
    "four_contact_fraction",
    "maximum_all_contact_loss_gap_s",
    "peak_valid_fingertip_force_n",
    "over_force_ticks",
    "non_tip_contact_ticks",
    "authority_violation_count",
    "handover_4_3_4_measured",
    "certificate_count",
    "micro_barrier_count",
    "mesh_contact_residual_p95_m",
    "mesh_contact_residual_max_m",
    "mesh_rejected_contact_fraction",
    "controller_latency_p95_s",
    "physics_latency_p95_s",
    "audit_latency_p95_s",
    "stop_reason",
    "trace_path",
  ]
  with (output / "episodes.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(episode_rows)

  by_cell = {
    cell: _aggregate_cell([row for row in episode_rows if row["cell"] == cell])
    for cell in CELLS
  }
  fixed_median = by_cell["fixed"]["actual_progress_m"]["median"]
  variable_median = by_cell["variable"]["actual_progress_m"]["median"]
  variable_pass = by_cell["variable"]["primary_pass_count"] >= 2
  mechanism_pass = by_cell["variable"]["variable_mechanism_pass_count"] >= 2
  g2_go = bool(
    variable_pass
    and mechanism_pass
    and variable_median >= fixed_median + 0.010
    and by_cell["variable"]["authority_violation_count"] == 0
  )
  source_sha256_end = {
    str(path): _sha256(path.resolve()) for path in SOURCE_FILES
  }
  changed_sources = sorted(
    path for path, digest in source_sha256_start.items()
    if source_sha256_end[path] != digest
  )
  if changed_sources:
    raise RuntimeError(
      "I01 source changed during formal benchmark: " + ", ".join(changed_sources)
    )
  git_head = subprocess.run(
    ["git", "rev-parse", "HEAD"],
    check=True,
    capture_output=True,
    text=True,
  ).stdout.strip()
  git_dirty = bool(
    subprocess.run(
      ["git", "status", "--porcelain"],
      check=True,
      capture_output=True,
      text=True,
    ).stdout.strip()
  )
  summary: dict[str, Any] = {
    "module_id": "I01-PHY-BUNNY-v1",
    "status": "EVALUATED / MET" if g2_go else "EVALUATED / NOT_MET",
    "trace_schema_version": TRACE_SCHEMA_VERSION,
    "evaluator_version": EVALUATOR_VERSION,
    "protocol": "Module/I01_BUNNY_PROTOCOL.md",
    "question_answered": {
      "can_move_on_bunny_with_continuous_contact": variable_pass,
      "variable_mode_mechanism_valid": mechanism_pass,
      "gate_g2": "GO" if g2_go else "NO_GO",
    },
    "fair_comparison": {
      "paired_seeds": list(SEEDS),
      "same_robot_object_initialization_path_mcc_guards_and_evaluator": True,
      "only_changed_factor": "fixed |A|=4 versus certified variable nonempty mode",
      "gravity_m_s2": 0.0,
      "timestep_s": 0.002,
      "duration_s": 12.0,
      "acquisition_s": 3.0,
      "planned_traversal_m": 0.060,
      "desired_force_n": 2.0,
      "contact_threshold_n": 0.20,
      "force_limit_n": 8.0,
      "mesh_residual_limit_m": 0.0025,
      "dp_used": False,
    },
    "bunny": {
      "source_path": str(bunny.source_path),
      "source_sha256": bunny.source_sha256,
      "vertex_count": len(bunny.mesh.vertices),
      "face_count": len(bunny.mesh.faces),
      "canonical_extents_m": bunny.extents_m.tolist(),
      "hfield_shape": list(bunny.height_m.shape),
      "hfield_coverage_fraction": bunny.coverage_fraction,
      "collision_representation": "vertical-ray upper-envelope hfield",
      "visual_representation": "exact transformed triangle mesh",
    },
    "cells": by_cell,
    "g2": {
      "median_l_fixed_m": fixed_median,
      "median_l_variable_m": variable_median,
      "median_advantage_m": variable_median - fixed_median,
      "required_advantage_m": 0.010,
      "variable_primary_pass_required": "at least 2/3",
      "variable_mechanism_pass_required": "at least 2/3",
      "decision": "GO" if g2_go else "NO_GO",
    },
    "episodes": episode_rows,
    "provenance": {
      "wall_time_s": perf_counter() - started,
      "python": sys.version,
      "platform": platform.platform(),
      "mujoco": mujoco.__version__,
      "numpy": np.__version__,
      "trimesh": trimesh.__version__,
      "scipy": importlib.metadata.version("scipy"),
      "source_sha256": source_sha256_start,
      "source_stability_check": "PASS",
      "git_head": git_head,
      "git_worktree_dirty": git_dirty,
      "command": (
        "/home/ferry/data/Anaconda/envs/handcomp/bin/python "
        "-m Module.i01_bunny_physics.benchmark"
      ),
    },
    "artifacts": {
      "summary": "summary.json",
      "episodes": "episodes.csv",
      "combined_traces": "traces.npz",
      "model_xml": "generated_fr3_leap_bunny.xml",
      "visual_mesh": visual_mesh.name,
    },
  }
  (output / "summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True),
    encoding="utf-8",
  )
  (output / "README.md").write_text(
    "# I01 Bunny physics generated evidence\n\n"
    f"Status: `{summary['status']}`; G2: `{summary['g2']['decision']}`.\n\n"
    "This directory is generated by `python -m Module.i01_bunny_physics.benchmark`. "
    "Numerical results live in `summary.json` and `episodes.csv`; `traces.npz` "
    "contains the complete six paired physics traces.\n",
    encoding="utf-8",
  )
  return summary


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
  args = parser.parse_args()
  summary = run_benchmark(args.output)
  print(json.dumps({
    "status": summary["status"],
    "gate_g2": summary["g2"]["decision"],
    "fixed_median_mm": 1000.0 * summary["g2"]["median_l_fixed_m"],
    "variable_median_mm": 1000.0 * summary["g2"]["median_l_variable_m"],
  }, indent=2))
  if summary["status"] != "EVALUATED / MET":
    raise SystemExit(2)


if __name__ == "__main__":
  main()
