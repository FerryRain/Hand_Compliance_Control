"""Static audit panels for DPRef relabeling and CUDA training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from Module.visualization import get_pyplot, save_figure


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = REPO_ROOT / "Module/generated/dpref_v1/relabelled_dataset_i"
DEFAULT_TRAINING = REPO_ROOT / "Module/generated/dpref_v1/training_i100"
ROLE_NAMES = ("KEEP", "RELEASE", "FREE", "MAKE")
ROLE_COLORS = ("#277DA1", "#D62828", "#6C757D", "#2A9D8F")


def render_dpref_audit(
  data_directory: str | Path = DEFAULT_DATA,
  training_directory: str | Path = DEFAULT_TRAINING,
) -> tuple[Path, Path]:
  data_root = Path(data_directory)
  training_root = Path(training_directory)
  label_summary = json.loads((data_root / "summary.json").read_text(encoding="utf-8"))
  train_summary = json.loads(
    (training_root / "training_summary.json").read_text(encoding="utf-8")
  )
  with np.load(training_root / "training_history.npz", allow_pickle=False) as history:
    updates = history["update"]
    total_loss = history["total"]
    diffusion_loss = history["diffusion"]
    role_loss = history["role"]
  with np.load(training_root / "validation_predictions.npz", allow_pickle=False) as prediction:
    target_offset = prediction["target_nominal_offsets_rad"]
    predicted_offset = prediction["predicted_nominal_offsets_rad"]
    target_role = prediction["target_role"]
    predicted_role = prediction["predicted_role"]
    role_valid = prediction["role_label_valid"]
    episode_id = prediction["episode_id"]

  pyplot = get_pyplot()
  figure, axes = pyplot.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
  pools = ("i20_train", "i100_train", "validation", "test")
  x = np.arange(len(pools))
  bottom = np.zeros(len(pools))
  for role, color in zip(ROLE_NAMES, ROLE_COLORS):
    values = np.array(
      [label_summary["splits"][pool]["label_counts"][role] for pool in pools],
      dtype=np.float64,
    )
    axes[0, 0].bar(x, values, bottom=bottom, label=role, color=color)
    bottom += values
  axes[0, 0].set_yscale("log")
  axes[0, 0].set_xticks(x, pools)
  axes[0, 0].set(title="Time-confirmed role labels (log scale)", ylabel="valid labels")
  axes[0, 0].legend(fontsize=8)

  axes[0, 1].plot(updates, total_loss, label="total", linewidth=1.5)
  axes[0, 1].plot(updates, diffusion_loss, label="diffusion", linewidth=1.2)
  axes[0, 1].plot(updates, role_loss, label="role CE", linewidth=1.2)
  axes[0, 1].set_yscale("log")
  axes[0, 1].set(title="I100 CUDA training", xlabel="update", ylabel="loss")
  axes[0, 1].legend()

  episode = np.unique(episode_id)[0]
  selected = np.flatnonzero(episode_id == episode)
  shown = selected[: min(300, len(selected))]
  axes[1, 0].plot(
    target_offset[shown, 0, 0],
    label="target q_nom offset F1/J1",
    linewidth=1.4,
  )
  axes[1, 0].plot(
    predicted_offset[shown, 0, 0],
    label="predicted",
    linewidth=1.2,
  )
  axes[1, 0].set(
    title=f"Object-disjoint validation · {episode}",
    xlabel="50 Hz sample",
    ylabel="first command offset (rad)",
  )
  axes[1, 0].legend(fontsize=8)

  confusion = np.zeros((4, 4), dtype=np.int64)
  for target, predicted in zip(target_role[role_valid], predicted_role[role_valid]):
    confusion[int(target), int(predicted)] += 1
  row_sum = np.maximum(confusion.sum(axis=1, keepdims=True), 1)
  normalized = confusion / row_sum
  image = axes[1, 1].imshow(normalized, vmin=0.0, vmax=1.0, cmap="Blues")
  for row in range(4):
    for column in range(4):
      value = "N/A" if confusion[row].sum() == 0 else f"{normalized[row, column]:.2f}"
      axes[1, 1].text(column, row, value, ha="center", va="center", fontsize=10)
  axes[1, 1].set_xticks(range(4), ROLE_NAMES)
  axes[1, 1].set_yticks(range(4), ROLE_NAMES)
  axes[1, 1].set(
    title="Validation role confusion (RELEASE absent)",
    xlabel="prediction",
    ylabel="time-confirmed target",
  )
  figure.colorbar(image, ax=axes[1, 1], fraction=0.046)
  figure.suptitle(
    "DPRef audit · continuous reference learned; complete role coverage not established",
    fontsize=15,
  )
  dashboard = training_root / "dpref_training_and_label_audit.png"
  save_figure(figure, dashboard)
  pyplot.close(figure)

  figure, axis = pyplot.subplots(figsize=(12, 5), constrained_layout=True)
  metrics = train_summary["validation_metrics"]
  role_accuracy = [
    metrics["role_per_class"][role]["accuracy"]
    for role in ROLE_NAMES
  ]
  plotted = [0.0 if value is None else value for value in role_accuracy]
  bars = axis.bar(ROLE_NAMES, plotted, color=ROLE_COLORS)
  for bar, value in zip(bars, role_accuracy):
    axis.text(
      bar.get_x() + bar.get_width() / 2,
      bar.get_height() + 0.025,
      "NO DATA" if value is None else f"{value:.1%}",
      ha="center",
    )
  axis.set_ylim(0.0, 1.12)
  axis.set_ylabel("validation accuracy")
  axis.set_title(
    "Role head coverage · KEEP/FREE strong, MAKE partial, RELEASE unvalidated"
  )
  coverage = training_root / "dpref_role_coverage.png"
  save_figure(figure, coverage)
  pyplot.close(figure)
  return dashboard, coverage


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
  parser.add_argument("--training", type=Path, default=DEFAULT_TRAINING)
  args = parser.parse_args()
  print("\n".join(str(path) for path in render_dpref_audit(args.data, args.training)))


if __name__ == "__main__":
  main()
