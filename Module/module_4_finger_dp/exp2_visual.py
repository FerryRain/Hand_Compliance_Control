"""Render final Exp. 2 traces without rerunning controllers."""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path

import numpy as np

from Module.module_4_whole_hand_mcc.runner import E05MCCTrace
from Module.module_4_whole_hand_mcc.visual_demo import render_video
from Module.visualization import get_pyplot, save_figure


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "Module/generated/exp2_dpref_mcc"
BRANCHES = (
  "PLAIN_WHOLE_HAND_MCC",
  "PASSIVE_HOLD_MCC",
  "REACTIVE_HEURISTIC_MCC",
  "DPREF_MCC",
)
COLORS = {
  "PLAIN_WHOLE_HAND_MCC": "#8B5CF6",
  "PASSIVE_HOLD_MCC": "#6C7A89",
  "REACTIVE_HEURISTIC_MCC": "#E09F3E",
  "DPREF_MCC": "#277DA1",
}


def _load_trace(path: Path) -> E05MCCTrace:
  with np.load(path, allow_pickle=False) as archive:
    return E05MCCTrace(
      **{definition.name: archive[definition.name] for definition in fields(E05MCCTrace)}
    )


def render_exp2_review(directory: str | Path = DEFAULT_INPUT) -> tuple[Path, ...]:
  root = Path(directory)
  summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
  traces = {
    branch: _load_trace(root / f"{branch.lower()}_nominal_trace.npz")
    for branch in BRANCHES
  }
  pyplot = get_pyplot()
  figure, axes = pyplot.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
  for branch, trace in traces.items():
    color = COLORS[branch]
    contact_count = np.sum(trace.actual_contacts, axis=1)
    axes[0, 0].plot(trace.time_s, contact_count, color=color, label=branch, linewidth=1.4)
    axes[0, 1].plot(
      trace.time_s,
      np.max(trace.fingertip_forces_n, axis=1),
      color=color,
      label=branch,
      linewidth=1.2,
    )
    axes[1, 0].plot(
      trace.time_s,
      1000.0 * (trace.palm_pose_world[:, 1] - trace.palm_pose_world[0, 1]),
      color=color,
      label=branch,
      linewidth=1.4,
    )
  axes[0, 0].set(title="Nominal episode: actual contact count", xlabel="time (s)", ylabel="contacts")
  axes[0, 0].set_ylim(-0.1, 4.2)
  axes[0, 0].legend(fontsize=8)
  axes[0, 1].axhline(
    8.0,
    color="#C1121F",
    linestyle="--",
    label="8 N diagnostic reference",
  )
  axes[0, 1].set(title="Maximum fingertip force", xlabel="time (s)", ylabel="force (N)")
  axes[0, 1].legend(fontsize=8)
  axes[1, 0].set(title="Palm traversal", xlabel="time (s)", ylabel="Y progress (mm)")
  axes[1, 0].legend(fontsize=8)

  labels = ["Continuity", "Avg contacts", "Supported Y>=2"]
  x = np.arange(len(labels))
  width = 0.19
  for index, branch in enumerate(BRANCHES):
    aggregate = summary["aggregates"][branch]
    values = [
      aggregate["contact_continuity_probability"]["mean"],
      aggregate["average_contact_count"]["mean"] / 4.0,
      aggregate["supported_y_traversal_ge2_m"]["mean"] / 0.18,
    ]
    axes[1, 1].bar(
      x + (index - 1.5) * width,
      values,
      width,
      color=COLORS[branch],
      label=branch,
    )
  axes[1, 1].set_xticks(x, labels)
  axes[1, 1].set_ylim(0.0, 1.05)
  axes[1, 1].set_ylabel("normalized score")
  axes[1, 1].set_title("Three-condition aggregate · descriptive comparison")
  axes[1, 1].legend(fontsize=8)
  figure.suptitle(
    "Exp. 2 · basic MCC reference + three sources on the same adjusted shared stack",
    fontsize=15,
  )
  dashboard = root / "exp2_comparison.png"
  save_figure(figure, dashboard)
  pyplot.close(figure)

  artifacts: list[Path] = [dashboard]
  labels_for_video = {
    "PLAIN_WHOLE_HAND_MCC": "Exp.2 Plain whole-hand MCC (basic analytical reference)",
    "PASSIVE_HOLD_MCC": "Exp.2 Passive-Hold + shared Wrist/Finger MCC",
    "REACTIVE_HEURISTIC_MCC": "Exp.2 Reactive-Heuristic + shared Wrist/Finger MCC",
    "DPREF_MCC": "Exp.2 DPRef/Role + shared Wrist/Finger MCC",
  }
  for branch, trace in traces.items():
    video = root / f"{branch.lower()}_video.mp4"
    frame = root / f"{branch.lower()}_frame.png"
    render_video(
      trace,
      "E05-H-MCC",
      video,
      frame,
      display_label=labels_for_video[branch],
    )
    artifacts.extend((video, frame))
  return tuple(artifacts)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
  args = parser.parse_args()
  paths = render_exp2_review(args.input)
  print("\n".join(str(path) for path in paths))


if __name__ == "__main__":
  main()
