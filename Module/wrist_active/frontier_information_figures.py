"""Draw publication-size cumulative and early per-control-step information figures."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, StrMethodFormatter

from .face_regression_suite import read
from .frontier_information import ARMS
from .frontier_study import ROOT

COLORS = ("#087568", "#ad4f25")
INK, MUTED, GRID = "#172d2a", "#47605b", "#dce7e2"


def axis_style(ax):
  ax.spines[["right", "top"]].set_visible(False)
  ax.spines[["bottom", "left"]].set_color("#93a8a0")
  ax.tick_params(colors=MUTED, labelsize=10, length=3)
  ax.grid(axis="y", color=GRID, linewidth=0.8)
  ax.set_axisbelow(True)


def legend_handles():
  return [
    Line2D([], [], color=c, linewidth=2.6, linestyle=s, label=n)
    for c, s, n in zip(COLORS, ("-", "--"), ARMS.values(), strict=True)
  ]


def cumulative_figure(directory, data, paired, average, time):
  fig, axes = plt.subplots(2, 3, figsize=(12.4, 7.4), sharex=True, sharey=True)
  fig.subplots_adjust(
    left=0.083, right=0.975, bottom=0.113, top=0.80, hspace=0.40, wspace=0.17
  )
  fig.text(0.083, 0.949, "0–600 秒的累计信息增益", size=21, weight="bold", color=INK)
  fig.text(
    0.083,
    0.895,
    "五个物体各有一对同初态试次。浅色区对应前 60 秒，右下图为五物体等权平均。",
    size=11.5,
    color=MUTED,
  )
  fig.legend(
    handles=legend_handles(),
    loc="upper right",
    bbox_to_anchor=(0.98, 0.994),
    ncol=2,
    frameon=False,
  )
  labels = [v["name"] for v in data["objects"].values()] + ["五物体等权平均"]
  max_info = max(pair.sum(axis=1).max() for pair in paired)
  for ax, pair, name in zip(axes.flat, [*paired, average], labels, strict=True):
    axis_style(ax)
    ax.axvspan(0, data["default_early_s"], color="#eaf3ef", linewidth=0, zorder=0)
    for values, color, ls in zip(pair, COLORS, ("-", "--"), strict=True):
      ax.plot(time, np.cumsum(values), color=color, ls=ls, lw=2.1)
    ax.set_title(name, loc="left", weight="bold", size=13, color=INK)
    ax.set_xlim(0, time[-1])
    ax.set_ylim(0, np.ceil(max_info / 2000) * 2000)
    ax.set_xticks([0, 120, 240, 360, 480, 600])
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
  fig.supxlabel("仿真时间（秒）", y=0.026, size=12, color=INK)
  fig.supylabel("累计信息增益（nat）", x=0.010, size=12, color=INK)
  fig.savefig(directory / "cumulative-600s.png", dpi=190, facecolor="white")
  plt.close(fig)


def early_figure(directory, data, average):
  dt, seconds = data["dt_s"], data["default_early_s"]
  n, hz = round(seconds / dt), round(1 / dt)
  early = average[:, 1 : n + 1]
  x = np.arange(1, n + 1) * dt
  fig, axes = plt.subplots(
    3, 1, figsize=(12.2, 8.8), gridspec_kw={"height_ratios": [1, 1, 0.86]}
  )
  fig.subplots_adjust(left=0.099, right=0.969, top=0.823, bottom=0.096, hspace=0.62)
  fig.text(
    0.098, 0.951, f"前 {seconds} 秒的逐步信息增益", size=21, weight="bold", color=INK
  )
  fig.text(
    0.098,
    0.909,
    "每个点是五个物体在同一 10 ms 控制步的平均增益。没有新增触点格的步计为 0。",
    size=11.5,
    color=MUTED,
  )
  fig.text(
    0.098,
    0.870,
    "浅点为逐步值，深色阶梯线为每 1 秒内 100 个控制步的均值。两个方法共用纵轴。",
    size=10.5,
    color=MUTED,
  )
  top = max(0.2, np.ceil(early.max() * 4) / 4)
  for ax, values, (key, label), color in zip(
    axes[:2], early, ARMS.items(), COLORS, strict=True
  ):
    axis_style(ax)
    ax.scatter(
      x,
      values,
      s=6.0,
      alpha=0.24,
      facecolor=color,
      edgecolor="none",
      clip_on=True,
      zorder=2,
    )
    ax.stairs(
      values.reshape(seconds, hz).mean(axis=1),
      np.arange(seconds + 1),
      baseline=None,
      color=color,
      lw=1.9,
      zorder=3,
    )
    value = data["macro"][key][str(seconds)]["mean_nat_per_step"]
    ax.set_title(label, loc="left", weight="bold", color=color, size=13)
    ax.set_title(f"{n:,} 步均值   {value:.3f} nat/步", loc="right", color=INK, size=11)
    ax.set(xlim=(0, seconds), ylim=(-0.035 * top, top * 1.04), ylabel="nat / 步")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.set_xticks(np.linspace(0, seconds, 7))
  axes[1].set_xlabel("仿真时间（秒）")
  ax = axes[2]
  axis_style(ax)
  ticks = np.arange(1, hz + 1)
  for v, color, ls in zip(early, COLORS, ("-", "--"), strict=True):
    ax.plot(ticks, v[:hz], ls=ls, color=color, lw=1.1, marker="o", ms=2.5)
  ax.legend(
    handles=legend_handles(), frameon=False, ncol=2, loc="upper right", fontsize=9.5
  )
  ax.set_title("放大首秒的 100 个控制步", loc="left", size=13, weight="bold", color=INK)
  ax.set(
    xlim=(1, hz), ylim=(-0.03 * top, top * 1.04), ylabel="nat / 步", xlabel="控制步"
  )
  ax.set_xticks([1, 20, 40, 60, 80, 100])
  ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
  fig.savefig(directory / "per-step-first-60s.png", dpi=190, facecolor="white")
  plt.close(fig)


def draw(study):
  directory = study / "information"
  data = read(directory / "gain.json")
  n = round(data["duration_s"] / data["dt_s"])
  paired = []
  for item in data["objects"].values():
    pair = np.zeros((len(ARMS), n + 1))
    for j, key in enumerate(ARMS):
      r = item["arms"][key]
      pair[j, r["tick"]] = r["gain_nat"]
      np.testing.assert_allclose(pair[j].sum(), r["windows"]["600"]["total_nat"])
    paired.append(pair)
  average = np.mean(paired, axis=0)
  with plt.rc_context(
    {
      "font.family": ["Microsoft YaHei", "DejaVu Sans"],
      "axes.unicode_minus": False,
      "font.size": 11,
      "axes.labelcolor": MUTED,
      "text.color": INK,
    }
  ):
    cumulative_figure(directory, data, paired, average, np.arange(n + 1) * data["dt_s"])
    early_figure(directory, data, average)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--study", type=Path, default=ROOT / "validated")
  draw(parser.parse_args().study)
