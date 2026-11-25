"""Render Atlas comparison charts from the public, paired PARK scoring tables."""
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np


STYLE = {"PARK": ("#087568", "-"), "PalmReach-6": ("#3e669d", "-."), "AUDE": ("#9d4c26", "--")}
LABELS = {
    "coverage": ("合格触点覆盖了多少表面", "实触覆盖 C25（%）"),
    "f5": ("重建表面与真值的吻合程度", "重建 F@5（%）"),
    "cumulative": ("累计获得的实触信息", "累计信息（nat）"),
    "rate": ("每 10 秒获得的信息", "信息获取率（nat/s）"),
    "decisions": ("每次下发目标后的实得信息", "每次决策收益（nat）"),
}


def values(metric, records, ax, color):
    if metric in ("coverage", "f5"):
        key = "coverage" if metric == "coverage" else "reconstruction"
        series = [np.asarray(r[key], dtype=float) for r in records]
        xs = series[0][:, 0]
        if any(not np.array_equal(a[:, 0], xs) for a in series[1:]):
            raise ValueError("Different scoring times within the same cohort")
        return xs, np.mean([a[:, 1] for a in series], axis=0)
    if metric in ("cumulative", "rate"):
        cumulative = np.mean([r["cumulative_nat"] for r in records], axis=0)
        if len(cumulative) != 241:
            raise ValueError("Expected causal information samples from 0 to 240 s")
        if metric == "rate":
            return np.arange(10,241,10), np.diff(cumulative[::10])/10
        return np.arange(241), cumulative
    edges = np.arange(0,241,10)
    means = []
    for record in records:
        issued = np.asarray(record["decisions"], dtype=float)
        times, gains = issued[:,0], issued[:,2]
        if len(records) == 1:
            ax.scatter(times, gains, s=3, color=color, alpha=.17, linewidths=0)
        totals = np.histogram(times, bins=edges, weights=gains)[0]
        counts = np.histogram(times, bins=edges)[0]
        means.append(np.divide(totals, counts, out=np.full(len(totals),np.nan), where=counts>0))
    bins = np.asarray(means)
    counts = np.isfinite(bins).sum(axis=0)
    return edges[:-1]+5, np.divide(np.nansum(bins,axis=0), counts, out=np.full(len(counts),np.nan), where=counts>0)


def render_charts(directory, objects, curves, information):
    directory.mkdir(exist_ok=True, parents=True)
    scopes = {"mean": "五物体等权平均", **objects}
    tracks = {(r["object"], r["method"]): r for r in curves["series"] if r["seconds"] == 600}
    short = {(r["object"], r["method"]): r for r in information["series"] if r["seconds"] == 240}
    with plt.rc_context({"font.family": "Microsoft YaHei", "axes.unicode_minus": False, "font.size": 10}):
        for scope, title in scopes.items():
            selected = objects if scope == "mean" else [scope]
            for metric, (label, unit) in LABELS.items():
                long = metric in ("coverage", "f5")
                data = tracks if long else short
                fig, ax = plt.subplots(figsize=(6.8,3.75), dpi=150, layout="constrained")
                ax.set_title(title+" · "+label, loc="left", fontsize=12.5, weight="bold", pad=10)
                for method, (color, style) in STYLE.items():
                    series = [data[obj,method] for obj in selected]
                    xs, ys = values(metric, series, ax, color)
                    ax.plot(xs, ys, color=color, linestyle=style, label=method, linewidth=1.9,
                            marker="o" if metric == "f5" else None, markersize=3.5)
                if long:
                    ax.axvline(240, color="#87978c", linestyle=":", linewidth=.8, zorder=0)
                    ax.set(xlim=(0,600), ylim=(0,103), xticks=np.arange(0,601,120), yticks=(0,25,50,75,100))
                else:
                    ax.set(xlim=(0,240), xticks=np.arange(0,241,60))
                    ax.set_ylim(bottom=0)
                ax.set(xlabel="仿真时间（秒）", ylabel=unit)
                ax.grid(axis="y", color="#dce7df", linewidth=.7)
                ax.spines[["top","right"]].set_visible(False)
                ax.legend(loc="best", frameon=True, framealpha=.93, fontsize=9)
                fig.savefig(directory/f"{metric}-{scope}.png", facecolor="white")
                plt.close(fig)
