"""Visual force-tracking and curved-contact demo for Fingertip MCC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from Module.module_2_fingertip_mcc.benchmarks import (
  run_curved_surface,
  run_static_contact,
  run_tangential_sliding,
  trace_curved_surface,
  trace_static_contact,
  trace_tangential_sliding,
)
from Module.visualization import COLORS, get_pyplot, save_figure


def _add_force_panel(axis) -> None:
  for desired, color in zip(
    (1.0, 2.0, 3.0),
    (COLORS["green"], COLORS["blue"], COLORS["purple"]),
  ):
    trace = trace_static_contact(desired)
    metrics = run_static_contact(desired)
    axis.plot(
      trace.time_s,
      trace.measured_force_n,
      color=color,
      linewidth=2.0,
      label=f"target {desired:.0f} N  (RMSE {metrics.force_rmse_n:.3f} N)",
    )
    axis.axhline(desired, color=color, linestyle=":", linewidth=1.0, alpha=0.65)
  axis.axhspan(3.5, 3.75, color=COLORS["red"], alpha=0.10)
  axis.text(2.98, 3.53, "force limit", color=COLORS["red"], ha="right", va="bottom")
  axis.set_xlim(0.0, 3.0)
  axis.set_ylim(0.0, 3.75)
  axis.set_xlabel("time [s]")
  axis.set_ylabel("normal force [N]")
  axis.set_title("A  Static contact: force converges to each target", loc="left")
  axis.legend(loc="lower right", fontsize=8)


def _add_sliding_panel(axis) -> None:
  trace = trace_tangential_sliding()
  metrics = run_tangential_sliding()
  x_mm = 1000.0 * trace.planned_position_m[:, 0]
  axis.axhspan(-3.2, 0.0, color="#DDEFF7", alpha=0.8, label="object")
  axis.axhline(0.0, color=COLORS["navy"], linewidth=1.7, label="surface")
  axis.plot(
    x_mm,
    1000.0 * trace.planned_position_m[:, 2],
    color=COLORS["gray"],
    linestyle="--",
    linewidth=1.6,
    label="planner path",
  )
  axis.plot(
    x_mm,
    1000.0 * trace.commanded_position_m[:, 2],
    color=COLORS["pink"],
    linewidth=2.4,
    label="MCC command",
  )
  indices = np.linspace(100, len(trace.time_s) - 1, 7, dtype=int)
  axis.scatter(
    x_mm[indices],
    1000.0 * trace.commanded_position_m[indices, 2],
    color=COLORS["pink"],
    edgecolors="white",
    s=45,
    zorder=4,
  )
  axis.text(
    -58.0,
    -2.85,
    "MCC changes only the normal direction\nwhile the tangential path is preserved",
    color=COLORS["navy"],
    fontsize=9,
  )
  axis.text(
    57.0,
    -2.85,
    f"max tangential error\n{1e6 * metrics.max_tangential_error_m:.3f} μm",
    color=COLORS["green"],
    fontsize=9,
    ha="right",
  )
  axis.set_xlim(-65.0, 65.0)
  axis.set_ylim(-3.2, 1.1)
  axis.set_xlabel("tangential x [mm]")
  axis.set_ylabel("normal z [mm]")
  axis.set_title("B  Tangential sliding: plan + normal compliance", loc="left")
  axis.legend(loc="upper center", ncol=3, fontsize=8)


def _add_curved_path_panel(axis) -> None:
  trace = trace_curved_surface("sphere")
  angles = np.linspace(0.0, 2.0 * np.pi, 361)
  axis.fill(
    1000.0 * 0.10 * np.cos(angles),
    1000.0 * 0.10 * np.sin(angles),
    color="#DDEFF7",
    edgecolor=COLORS["blue"],
    linewidth=1.8,
    label="sphere surface",
  )
  axis.plot(
    1000.0 * trace.planned_position_m[:, 0],
    1000.0 * trace.planned_position_m[:, 1],
    color=COLORS["gray"],
    linestyle="--",
    linewidth=2.0,
    label="planned surface path",
  )
  axis.plot(
    1000.0 * trace.commanded_position_m[:, 0],
    1000.0 * trace.commanded_position_m[:, 1],
    color=COLORS["pink"],
    linewidth=2.5,
    label="MCC fingertip command",
  )
  indices = np.linspace(100, len(trace.time_s) - 1, 8, dtype=int)
  for index in indices:
    plan = trace.planned_position_m[index]
    command = trace.commanded_position_m[index]
    axis.plot(
      1000.0 * np.array([plan[0], command[0]]),
      1000.0 * np.array([plan[1], command[1]]),
      color=COLORS["orange"],
      linewidth=1.0,
    )
  axis.scatter(
    1000.0 * trace.commanded_position_m[indices, 0],
    1000.0 * trace.commanded_position_m[indices, 1],
    color=COLORS["pink"],
    edgecolors="white",
    s=38,
    zorder=5,
  )
  axis.set_aspect("equal")
  axis.set_xlim(60.0, 112.0)
  axis.set_ylim(-76.0, 76.0)
  axis.set_xlabel("x [mm]")
  axis.set_ylabel("y [mm]")
  axis.set_title("C  Curved surface: normal direction rotates", loc="left")
  axis.legend(loc="upper left", fontsize=8)


def _add_curved_force_panel(axis) -> None:
  traces = {
    "sphere": trace_curved_surface("sphere"),
    "cylinder": trace_curved_surface("cylinder"),
  }
  metrics = {
    surface: run_curved_surface(surface)
    for surface in ("sphere", "cylinder")
  }
  axis.axhline(2.0, color=COLORS["navy"], linestyle=":", linewidth=1.4, label="target 2 N")
  for surface, color in (("sphere", COLORS["pink"]), ("cylinder", COLORS["blue"])):
    trace = traces[surface]
    axis.plot(
      trace.time_s,
      trace.measured_force_n,
      color=color,
      linewidth=2.1,
      label=f"{surface} (RMSE {metrics[surface].force_rmse_n:.3f} N)",
    )
  axis.set_xlim(0.0, 3.0)
  axis.set_ylim(0.0, 2.45)
  axis.set_xlabel("time [s]")
  axis.set_ylabel("normal force [N]")
  axis.set_title("D  Force remains stable while the surface curves", loc="left")
  axis.legend(loc="lower right", fontsize=8)
  axis.text(
    0.08,
    2.25,
    "analytic contact stiffness: 1000 N/m\nno post-settling contact loss",
    fontsize=9,
    color=COLORS["gray"],
    va="top",
  )


def render_dashboard(output_path: Path) -> dict[str, Any]:
  plt = get_pyplot()
  figure, axes = plt.subplots(2, 2, figsize=(14.5, 9.0))
  _add_force_panel(axes[0, 0])
  _add_sliding_panel(axes[1, 0])
  _add_curved_path_panel(axes[0, 1])
  _add_curved_force_panel(axes[1, 1])
  figure.suptitle(
    "Module 2 — Fingertip MCC: the controller adds only a normal-force correction",
    fontsize=16,
    weight="bold",
    x=0.04,
    ha="left",
  )
  figure.subplots_adjust(top=0.91, left=0.07, right=0.97, bottom=0.08, hspace=0.33, wspace=0.23)
  save_figure(figure, output_path)
  plt.close(figure)
  return {
    "output": str(output_path),
    "static_rmse_n": {
      str(force): run_static_contact(force).force_rmse_n
      for force in (1.0, 2.0, 3.0)
    },
    "sphere_rmse_n": run_curved_surface("sphere").force_rmse_n,
    "cylinder_rmse_n": run_curved_surface("cylinder").force_rmse_n,
  }


def render_animation(output_path: Path, *, frame_count: int = 60) -> dict[str, Any]:
  if frame_count < 4:
    raise ValueError("frame_count must be at least 4")
  plt = get_pyplot()
  from matplotlib.animation import FuncAnimation, PillowWriter

  trace = trace_curved_surface("sphere")
  frame_indices = np.linspace(0, len(trace.time_s) - 1, frame_count, dtype=int)
  figure, (path_axis, force_axis) = plt.subplots(1, 2, figsize=(11.5, 5.2))
  angles = np.linspace(0.0, 2.0 * np.pi, 361)
  path_axis.fill(
    1000.0 * 0.10 * np.cos(angles),
    1000.0 * 0.10 * np.sin(angles),
    color="#DDEFF7",
    edgecolor=COLORS["blue"],
    linewidth=2.0,
  )
  path_axis.plot(
    1000.0 * trace.planned_position_m[:, 0],
    1000.0 * trace.planned_position_m[:, 1],
    color=COLORS["gray"],
    linestyle="--",
    linewidth=1.4,
    label="planner path",
  )
  trail, = path_axis.plot([], [], color=COLORS["pink"], linewidth=2.4, label="executed command")
  plan_marker, = path_axis.plot([], [], "o", color=COLORS["gray"], markersize=6)
  tip_marker, = path_axis.plot(
    [], [], "o", color=COLORS["pink"], markeredgecolor="white", markersize=11
  )
  correction, = path_axis.plot([], [], color=COLORS["orange"], linewidth=2.2)
  status = path_axis.text(
    0.03,
    0.03,
    "",
    transform=path_axis.transAxes,
    fontsize=10,
    bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.9, "edgecolor": "none"},
  )
  path_axis.set_aspect("equal")
  path_axis.set_xlim(58.0, 112.0)
  path_axis.set_ylim(-78.0, 78.0)
  path_axis.set_xlabel("x [mm]")
  path_axis.set_ylabel("y [mm]")
  path_axis.set_title("Fingertip moves along a sphere")
  path_axis.legend(loc="upper left", fontsize=8)

  force_axis.axhline(2.0, color=COLORS["navy"], linestyle=":", label="desired 2 N")
  force_line, = force_axis.plot([], [], color=COLORS["pink"], linewidth=2.5, label="measured force")
  cursor = force_axis.axvline(0.0, color=COLORS["orange"], linewidth=1.4)
  force_axis.set_xlim(0.0, 3.0)
  force_axis.set_ylim(0.0, 2.5)
  force_axis.set_xlabel("time [s]")
  force_axis.set_ylabel("normal force [N]")
  force_axis.set_title("Closed-loop normal-force response")
  force_axis.legend(loc="lower right")

  def update(frame: int):
    index = int(frame_indices[frame])
    planned = trace.planned_position_m[index]
    command = trace.commanded_position_m[index]
    trail.set_data(
      1000.0 * trace.commanded_position_m[: index + 1, 0],
      1000.0 * trace.commanded_position_m[: index + 1, 1],
    )
    plan_marker.set_data([1000.0 * planned[0]], [1000.0 * planned[1]])
    tip_marker.set_data([1000.0 * command[0]], [1000.0 * command[1]])
    correction.set_data(
      1000.0 * np.array([planned[0], command[0]]),
      1000.0 * np.array([planned[1], command[1]]),
    )
    force_line.set_data(trace.time_s[: index + 1], trace.measured_force_n[: index + 1])
    cursor.set_xdata([trace.time_s[index], trace.time_s[index]])
    status.set_text(
      f"t = {trace.time_s[index]:.2f} s\n"
      f"force = {trace.measured_force_n[index]:.2f} N\n"
      f"normal offset = {1000.0 * trace.offset_m[index]:.2f} mm"
    )
    return trail, plan_marker, tip_marker, correction, force_line, cursor, status

  animation = FuncAnimation(figure, update, frames=frame_count, interval=1000.0 / 15.0, blit=False)
  figure.suptitle("Module 2 — Curved-surface MCC animation", fontsize=15, weight="bold")
  figure.tight_layout()
  output_path = Path(output_path)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  animation.save(output_path, writer=PillowWriter(fps=15), dpi=110)
  plt.close(figure)
  return {"output": str(output_path), "frame_count": frame_count}


def render_visual_demo(
  dashboard_path: Path,
  animation_path: Path,
  *,
  animation_frames: int = 60,
) -> dict[str, Any]:
  dashboard = render_dashboard(dashboard_path)
  animation = render_animation(animation_path, frame_count=animation_frames)
  passed = (
    Path(dashboard_path).is_file()
    and Path(animation_path).is_file()
    and all(value <= 0.05 for value in dashboard["static_rmse_n"].values())
    and dashboard["sphere_rmse_n"] <= 0.06
    and dashboard["cylinder_rmse_n"] <= 0.06
  )
  return {"module": "M02", "passed": passed, "dashboard": dashboard, "animation": animation}


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--output",
    type=Path,
    default=Path("Module/generated/visual_demo/module_2_fingertip_mcc.png"),
  )
  parser.add_argument(
    "--animation",
    type=Path,
    default=Path("Module/generated/visual_demo/module_2_curved_surface.gif"),
  )
  parser.add_argument("--animation-frames", type=int, default=60)
  args = parser.parse_args()
  result = render_visual_demo(
    args.output,
    args.animation,
    animation_frames=args.animation_frames,
  )
  print(json.dumps(result, indent=2, sort_keys=True))
  if not result["passed"]:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
