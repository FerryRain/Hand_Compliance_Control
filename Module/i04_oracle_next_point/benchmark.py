"""Run and persist the I04 full-Bunny Explicit MCC baseline episode."""

from __future__ import annotations

import argparse
from dataclasses import asdict
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

from Module.i01_bunny_physics.surface import canonical_bunny_heightfield
from Module.i04_oracle_next_point.planner import I04PlannerConfig
from Module.i04_oracle_next_point.runner import (
  I04BunnyConfig,
  run_i04_bunny,
  save_trace,
)


DEFAULT_OUTPUT_DIR = Path("Module/generated/i04_oracle_next_point")
SOURCE_FILES = (
  Path("Module/I04_ORACLE_NEXT_POINT_PROTOCOL.md"),
  Path("Module/fr3_leap/model.py"),
  Path("Module/i04_oracle_next_point/surface_graph.py"),
  Path("Module/i04_oracle_next_point/planner.py"),
  Path("Module/i04_oracle_next_point/runner.py"),
  Path("Module/module_6_prefix_executor/executor.py"),
  Path("Module/module_7_contact_mode_graph/graph.py"),
  Path("Module/module_8_cheap_cert/certifier.py"),
  Path("Module/module_9_continuous_optimize/optimizer.py"),
  Path("Module/module_10_exact_prefix_audit/audit.py"),
  Path("Module/module_11_lazy_beam_search/search.py"),
  Path("Module/module_12_shadow_viability/shadow.py"),
)


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _git_value(*args: str) -> str:
  result = subprocess.run(
    ["git", *args],
    check=False,
    capture_output=True,
    text=True,
  )
  return result.stdout.strip() if result.returncode == 0 else "UNAVAILABLE"


def run_benchmark(
  output_dir: Path = DEFAULT_OUTPUT_DIR,
  *,
  profile: str = "formal",
  maximum_duration_s: float | None = None,
  maximum_goals: int | None = None,
  seed: int = 7,
) -> dict[str, Any]:
  if profile not in {"smoke", "formal"}:
    raise ValueError("profile must be 'smoke' or 'formal'")
  output = output_dir.resolve()
  output.mkdir(parents=True, exist_ok=True)
  bunny = canonical_bunny_heightfield()
  mesh_path = bunny.export_visual_mesh(output / "canonical_bunny_side_laid.obj")
  duration = (
    float(maximum_duration_s)
    if maximum_duration_s is not None
    else (45.0 if profile == "smoke" else 1800.0)
  )
  goal_limit = maximum_goals
  if goal_limit is None and profile == "smoke":
    goal_limit = 5
  config = I04BunnyConfig(
    seed=seed,
    maximum_duration_s=duration,
    maximum_goals=goal_limit,
    visual_mesh_path=str(mesh_path),
  )
  planner_config = I04PlannerConfig()
  source_hashes = {
    str(path): _sha256(path.resolve())
    for path in SOURCE_FILES
    if path.is_file()
  }
  started = perf_counter()
  trace, metrics = run_i04_bunny(config, planner_config=planner_config)
  wall_time = perf_counter() - started
  save_trace(output / "trace.npz", trace)
  (output / "events.json").write_text(
    json.dumps(trace.events, indent=2, sort_keys=True, allow_nan=False),
    encoding="utf-8",
  )
  source_hashes_end = {
    str(path): _sha256(path.resolve())
    for path in SOURCE_FILES
    if path.is_file()
  }
  changed = sorted(
    path for path, digest in source_hashes.items()
    if source_hashes_end.get(path) != digest
  )
  if changed:
    raise RuntimeError(
      "I04 source changed during benchmark: " + ", ".join(changed)
    )
  summary: dict[str, Any] = {
    **metrics,
    "module_id": "I04-ORACLE-NEXT-POINT-EXPLICIT-MCC-v1",
    "profile": profile,
    "protocol": "Module/I04_ORACLE_NEXT_POINT_PROTOCOL.md",
    "complete_required_set": (
      metrics["stop_reason"] == "FULL_REQUIRED_SET_COMPLETED"
    ),
    "config": asdict(config),
    "planner_config": asdict(planner_config),
    "bunny": {
      "source_path": str(bunny.source_path),
      "source_sha256": bunny.source_sha256,
      "vertex_count": int(len(bunny.mesh.vertices)),
      "face_count": int(len(bunny.mesh.faces)),
      "canonical_extents_m": bunny.extents_m.tolist(),
      "physical_collision": "MuJoCo non-convex mesh SDF",
    },
    "provenance": {
      "wall_time_s": wall_time,
      "python": sys.version,
      "platform": platform.platform(),
      "mujoco": mujoco.__version__,
      "numpy": np.__version__,
      "scipy": importlib.metadata.version("scipy"),
      "source_sha256": source_hashes,
      "source_stability_check": "PASS",
      "git_head": _git_value("rev-parse", "HEAD"),
      "git_worktree_dirty": bool(_git_value("status", "--porcelain")),
      "command": (
        "/home/ferry/data/Anaconda/envs/handcomp/bin/python "
        "-m Module.i04_oracle_next_point.benchmark --profile " + profile
      ),
    },
    "artifacts": {
      "trace": "trace.npz",
      "events": "events.json",
      "summary": "summary.json",
      "visual_mesh": mesh_path.name,
    },
  }
  (output / "summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True, allow_nan=False),
    encoding="utf-8",
  )
  (output / "README.md").write_text(
    "# I04 generated evidence\n\n"
    f"Profile: `{profile}`; stop: `{summary['stop_reason']}`; "
    f"visited: `{summary['visited_goal_count']}/{summary['required_goal_count']}`.\n\n"
    "Replay this exact trace with:\n\n"
    "```bash\n"
    "/home/ferry/data/Anaconda/envs/handcomp/bin/python "
    "-m Module.i04_oracle_next_point.visual_demo --reuse --speed 12\n"
    "```\n",
    encoding="utf-8",
  )
  return summary


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
  parser.add_argument("--profile", choices=("smoke", "formal"), default="formal")
  parser.add_argument("--duration", type=float)
  parser.add_argument("--goals", type=int)
  parser.add_argument("--seed", type=int, default=7)
  args = parser.parse_args()
  summary = run_benchmark(
    args.output,
    profile=args.profile,
    maximum_duration_s=args.duration,
    maximum_goals=args.goals,
    seed=args.seed,
  )
  print(
    json.dumps(
      {
        "stop_reason": summary["stop_reason"],
        "visited_goal_count": summary["visited_goal_count"],
        "required_goal_count": summary["required_goal_count"],
        "coverage_fraction": summary["coverage_fraction"],
        "contact_continuity_fraction": summary["contact_continuity_fraction"],
        "output": str(args.output.resolve()),
      },
      indent=2,
    )
  )


if __name__ == "__main__":
  main()
