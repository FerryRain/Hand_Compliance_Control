"""Visual timelines for blockage, joint-limit, force, and collision guards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from Module.module_3_runtime_guards import (
  GuardObservation,
  GuardReason,
  RuntimeGuardConfig,
  RuntimeGuards,
)
from Module.visualization import COLORS, get_pyplot, save_figure


def _observation(
  *,
  q=(0.0, 0.0),
  command=(0.1, 0.0),
  actual=(0.1, 0.0),
  forces=(0.0, 0.0, 0.0, 0.0),
  collision_distance=0.02,
) -> GuardObservation:
  return GuardObservation(
    q_rad=q,
    qd_command_rad_s=command,
    qd_actual_rad_s=actual,
    fingertip_forces_n=forces,
    contact_states=("FREE", "FREE", "FREE", "FREE"),
    min_self_collision_distance_m=collision_distance,
  )


def _blockage_trace(config: RuntimeGuardConfig, frame_count: int = 28) -> dict[str, Any]:
  guard = RuntimeGuards(config)
  time_s = np.arange(1, frame_count + 1, dtype=np.float64) * config.dt_s
  command_speed = np.full(frame_count, 0.1, dtype=np.float64)
  actual_speed = np.zeros(frame_count, dtype=np.float64)
  stall_duration = np.zeros(frame_count, dtype=np.float64)
  reasons: list[GuardReason] = []
  for index in range(frame_count):
    decision = guard.evaluate(_observation(actual=(0.0, 0.0)))
    stall_duration[index] = decision.evidence.stall_duration_s
    reasons.append(decision.reason)
  detection_indices = [
    index
    for index, reason in enumerate(reasons)
    if reason is GuardReason.SUSPECTED_OBJECT_BLOCKAGE
  ]
  detection_s = float(time_s[detection_indices[0]]) if detection_indices else None
  return {
    "time_s": time_s,
    "command_speed": command_speed,
    "actual_speed": actual_speed,
    "stall_duration": stall_duration,
    "reasons": reasons,
    "detection_s": detection_s,
  }


def _add_blockage_speed_panel(axis, trace: dict[str, Any]) -> None:
  time_s = trace["time_s"]
  detection_s = trace["detection_s"]
  axis.plot(
    time_s,
    trace["command_speed"],
    color=COLORS["blue"],
    linewidth=2.4,
    label="commanded speed",
  )
  axis.plot(
    time_s,
    trace["actual_speed"],
    color=COLORS["pink"],
    linewidth=2.4,
    label="actual progress",
  )
  if detection_s is not None:
    axis.axvspan(detection_s, time_s[-1], color=COLORS["red"], alpha=0.10)
    axis.axvline(detection_s, color=COLORS["red"], linestyle="--", linewidth=1.8)
    axis.text(
      detection_s + 0.008,
      0.054,
      f"BLOCKED at {detection_s:.2f} s",
      color=COLORS["red"],
      fontsize=9,
      weight="bold",
    )
  axis.set_xlim(float(time_s[0]), float(time_s[-1]))
  axis.set_ylim(-0.012, 0.118)
  axis.set_xlabel("time [s]")
  axis.set_ylabel("joint speed [rad/s]")
  axis.set_title("A  Suspected object blockage: command moves, joint does not", loc="left")
  axis.legend(loc="upper right", fontsize=8)


def _add_evidence_panel(axis, trace: dict[str, Any], config: RuntimeGuardConfig) -> None:
  time_s = trace["time_s"]
  detection_s = trace["detection_s"]
  axis.plot(
    time_s,
    trace["stall_duration"],
    color=COLORS["orange"],
    linewidth=2.7,
    label="accumulated stall evidence",
  )
  axis.axhline(
    config.stall_time_s,
    color=COLORS["red"],
    linestyle=":",
    linewidth=1.7,
    label=f"threshold {config.stall_time_s:.2f} s",
  )
  if detection_s is not None:
    axis.scatter(
      [detection_s],
      [config.stall_time_s],
      color=COLORS["red"],
      edgecolors="white",
      s=85,
      zorder=5,
    )
  axis.text(
    0.015,
    0.125,
    "tip force ≈ 0 N  →  blockage evidence\n(non-zero force would become NO_PROGRESS)",
    fontsize=9,
    color=COLORS["gray"],
  )
  axis.set_xlim(float(time_s[0]), float(time_s[-1]))
  axis.set_ylim(0.0, 0.19)
  axis.set_xlabel("time [s]")
  axis.set_ylabel("stall evidence [s]")
  axis.set_title("B  Evidence accumulates before the soft stop", loc="left")
  axis.legend(loc="lower right", fontsize=8)


def _add_immediate_guard_panel(axis, config: RuntimeGuardConfig) -> tuple[str, ...]:
  cases = (
    (
      "tip over-force",
      RuntimeGuards(config).evaluate(_observation(forces=(3.6, 0.0, 0.0, 0.0))),
      3.6 / config.max_tip_force_n,
      "3.60 N > 3.50 N",
    ),
    (
      "joint limit",
      RuntimeGuards(config).evaluate(
        _observation(q=(0.99, 0.0), command=(0.1, 0.0), actual=(0.0, 0.0))
      ),
      0.99 / (1.0 - config.joint_limit_margin_rad),
      "q=0.99 rad, moving outward",
    ),
    (
      "self collision",
      RuntimeGuards(config).evaluate(_observation(collision_distance=0.001)),
      config.min_self_collision_distance_m / 0.001,
      "1 mm < 5 mm clearance",
    ),
  )
  y = np.arange(len(cases))
  display_ratios = np.minimum([case[2] for case in cases], 1.25)
  axis.barh(y, display_ratios, color=(COLORS["pink"], COLORS["orange"], COLORS["purple"]), alpha=0.88)
  axis.axvline(1.0, color=COLORS["red"], linestyle="--", linewidth=1.8, label="stop boundary")
  for index, (_, decision, _, raw_text) in enumerate(cases):
    axis.text(0.03, index, raw_text, color="white", va="center", fontsize=9, weight="bold")
    axis.text(
      1.28,
      index,
      f"{decision.reason.value}\nHARD_STOP in {config.dt_s:.2f} s",
      color=COLORS["red"],
      va="center",
      fontsize=8,
      weight="bold",
    )
  axis.set_yticks(y, [case[0] for case in cases])
  axis.invert_yaxis()
  axis.set_xlim(0.0, 1.75)
  axis.set_xlabel("normalized risk (boundary = 1)")
  axis.set_title("C  Immediate guards: one frame to HARD_STOP", loc="left")
  axis.legend(loc="lower right", fontsize=8)
  return tuple(case[1].reason.value for case in cases)


def _add_logic_panel(axis) -> None:
  axis.set_axis_off()
  box = {"boxstyle": "round,pad=0.5", "edgecolor": "none", "alpha": 0.95}
  axis.text(
    0.10,
    0.78,
    "Observable signals\nq, q̇cmd, q̇actual\ntip force, self-clearance",
    transform=axis.transAxes,
    ha="center",
    va="center",
    fontsize=10,
    bbox={**box, "facecolor": "#DDEFF7"},
  )
  axis.annotate(
    "",
    xy=(0.42, 0.78),
    xytext=(0.25, 0.78),
    xycoords="axes fraction",
    arrowprops={"arrowstyle": "-|>", "color": COLORS["navy"], "linewidth": 2.0},
  )
  axis.text(
    0.55,
    0.78,
    "Deterministic priority\nforce → collision → joint\n→ accumulated stall",
    transform=axis.transAxes,
    ha="center",
    va="center",
    fontsize=10,
    bbox={**box, "facecolor": "#FFF1D6"},
  )
  axis.annotate(
    "",
    xy=(0.81, 0.78),
    xytext=(0.69, 0.78),
    xycoords="axes fraction",
    arrowprops={"arrowstyle": "-|>", "color": COLORS["navy"], "linewidth": 2.0},
  )
  axis.text(
    0.90,
    0.78,
    "Executor output\nRUNNING or BLOCKED\n+ reason + evidence",
    transform=axis.transAxes,
    ha="center",
    va="center",
    fontsize=10,
    bbox={**box, "facecolor": "#FADDE3"},
  )
  axis.text(
    0.5,
    0.30,
    "Important boundary\n"
    "SUSPECTED_OBJECT_BLOCKAGE does not invent an unknown collision point or normal.\n"
    "It reports only the measured command / progress / force evidence.",
    transform=axis.transAxes,
    ha="center",
    va="center",
    fontsize=10,
    color=COLORS["gray"],
    bbox={"boxstyle": "round,pad=0.6", "facecolor": "#F8FAFC", "edgecolor": "#CBD5E1"},
  )
  axis.set_title("D  What the guard does—and does not claim", loc="left")


def render_visual_demo(output_path: Path) -> dict[str, Any]:
  config = RuntimeGuardConfig(joint_lower_rad=[-1.0, -1.0], joint_upper_rad=[1.0, 1.0])
  trace = _blockage_trace(config)
  plt = get_pyplot()
  figure, axes = plt.subplots(2, 2, figsize=(14.5, 8.8))
  _add_blockage_speed_panel(axes[0, 0], trace)
  _add_evidence_panel(axes[1, 0], trace, config)
  reasons = _add_immediate_guard_panel(axes[0, 1], config)
  _add_logic_panel(axes[1, 1])
  figure.suptitle(
    "Module 3 — Runtime Guards: turn observable failure evidence into a safe stop",
    fontsize=16,
    weight="bold",
    x=0.04,
    ha="left",
  )
  figure.subplots_adjust(top=0.91, left=0.08, right=0.97, bottom=0.08, hspace=0.34, wspace=0.25)
  save_figure(figure, output_path)
  plt.close(figure)
  passed = (
    Path(output_path).is_file()
    and trace["detection_s"] is not None
    and 0.15 <= trace["detection_s"] <= 0.17
    and reasons == ("TIP_OVERFORCE", "JOINT_LIMIT", "SELF_COLLISION")
  )
  return {
    "module": "M03",
    "passed": passed,
    "output": str(output_path),
    "blockage_detection_s": trace["detection_s"],
    "immediate_reasons": reasons,
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--output",
    type=Path,
    default=Path("Module/generated/visual_demo/module_3_runtime_guards.png"),
  )
  args = parser.parse_args()
  result = render_visual_demo(args.output)
  print(json.dumps(result, indent=2, sort_keys=True))
  if not result["passed"]:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
