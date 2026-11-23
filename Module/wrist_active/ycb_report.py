"""Plot and report five-object native Active rollouts from the recorded coverage curves."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .face_regression_suite import read
from .ycb_fixtures import OBJECTS, STUDY

ARMS = ("wrist-dp", "native-aude")
LABELS = ("我们的腕部 Active", "完整 Aude")


def elapsed(value):
  return "600 秒内未达" if value is None else f"{value:.2f} s"


def graph(data, destination=STUDY, seconds=240):
  plt.rcParams.update(
    {
      "font.family": ["Microsoft YaHei", "DejaVu Sans"],
      "axes.unicode_minus": False,
      "axes.spines.top": False,
      "axes.spines.right": False,
      "font.size": 10,
    }
  )
  fig, axes = plt.subplots(
    2, 3, figsize=(12.2, 6.8), sharex=True, sharey=True, layout="constrained"
  )
  times = np.arange(seconds + 1)
  curves = {a: [] for a in ARMS}
  colors = ("#126a60", "#af5224")
  for (key, item), ax in zip(data["objects"].items(), axes.flat[:-1], strict=True):
    for a, name, color, style in zip(ARMS, LABELS, colors, ("-", "--"), strict=True):
      c = item["native"][a]["curve"]
      row = np.interp(times, c["s"], c["coverage"]) * 100
      curves[a].append(row)
      ax.plot(times, row, c=color, linestyle=style, lw=2, label=name)
    ax.set_title(OBJECTS[key][1], loc="left", fontweight="bold")
  ax = axes.flat[-1]
  for a, name, color, style in zip(ARMS, LABELS, colors, ("-", "--"), strict=True):
    ax.plot(times, np.mean(curves[a], axis=0), c=color, ls=style, lw=2.6, label=name)
  ax.set_title("五物体等权平均", loc="left", fontweight="bold")
  ax.legend(loc="lower right", frameon=False)
  for ax in axes.flat:
    ax.set(
      xlim=(0, seconds),
      ylim=(0, 100),
      xticks=np.linspace(0, seconds, 6 if seconds == 600 else 5),
      yticks=[0, 25, 50, 75, 100],
    )
    ax.grid(axis="y", color="#d9e3dd", lw=0.7)
    if seconds == 600:
      ax.axvline(240, color="#87958f", lw=0.8, ls=":", zorder=0)
    ax.tick_params(length=2, color="#aebbb5", labelcolor="#394a46")
    for spine in ("bottom", "left"):
      ax.spines[spine].set_color("#aebbb5")
  fig.supxlabel("仿真时间（秒）")
  fig.supylabel("整物体实触覆盖（%）")
  fig.savefig(destination / "coverage.png", dpi=170, facecolor="white")
  plt.close(fig)


def report():
  d = read(STUDY / "results.json")
  graph(d)
  m = d["macro"]
  lines = [
    "# 五个 YCB 物体上的主动触觉探索",
    "",
    "我们的 GPIS 规划器输出腕部位姿，冻结的 `single_all` 生成名义指令，MCC 根据触觉调整手指。对照为项目 Active-02-A 中完整的 Aude 腕指联合规划与 MCC。双方在同一物体上从相同的四指轻载状态启动，各有 600 仿真秒预算。主指标取前 240 秒。",
    "",
    "| 物体 | 系统 | 240 s 覆盖 | 前 240 s 平均覆盖 | 到 30% | 到 70% | 到 95% | 600 s 覆盖 |",
    "|---|---|---:|---:|---:|---:|---:|---:|",
  ]
  for obj, item in d["objects"].items():
    for a, label in zip(ARMS, LABELS, strict=True):
      v = item["native"][a]
      t = v["thresholds_s"]
      lines.append(
        f"| {OBJECTS[obj][1]} | {label} | {v['coverage']:.2%} | {v['mean_coverage']:.2%} | {elapsed(t['30'])} | {elapsed(t['70'])} | {elapsed(t['95'])} | {v['coverage_600']:.2%} |"
      )
  lines += [
    "",
    "## 五物体等权汇总",
    "",
    "| 系统 | 240 s 覆盖 | 时间平均覆盖 | 达到 30% | 到 30% 限时均值 | 达到 70% | 到 70% 限时均值 | 95% 达标 |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
  ]
  for a, label in zip(ARMS, LABELS, strict=True):
    v = m[a]
    t = v["thresholds"]
    lines.append(
      f"| {label} | {v['coverage']:.2%} | {v['mean_coverage']:.2%} | {t['30']['hit']}/5 | {t['30']['mean_restricted_s']:.2f} s | {t['70']['hit']}/5 | {t['70']['mean_restricted_s']:.2f} s | {t['95']['hit']}/5 |"
    )
  lines += [
    "",
    "实触覆盖是所有有效触点的 10 mm 表面测地邻域并集。有效指尖法向力至少为 0.2 N，全部五个分区载荷低于 8 N。时间平均覆盖为前 240 秒逐帧覆盖率的算术平均，反映表面被发现的早晚。限时均值把 600 秒仍未达标的试次按 600 秒计入，达标数单列。",
    "",
    "## 重建、接触、计算用时",
    "",
    "| 物体 | 系统 | 重建 F@5 mm | 轻载三指累计 | 实测信息量 | ≥8 N 累计 | 分区峰值 | 240 仿真秒墙钟 | 达到 30% 的墙钟 |",
    "|---|---|---:|---:|---:|---:|---:|---:|---:|",
  ]
  for obj, item in d["objects"].items():
    for a, label in zip(ARMS, LABELS, strict=True):
      v = item["native"][a]
      wall = v["wall_thresholds_s"]["30"]
      lines.append(
        f"| {OBJECTS[obj][1]} | {label} | {v['f5']:.2%} | {v['light3']:.2f} s | {v['nats']:,.1f} nat | {v['over8']:.2f} s | {v['peak_n']:.1f} N | {v['wall_s_240']:.2f} s | {elapsed(wall)} |"
      )
  lines += [
    "",
    "墙钟从首次请求在线规划计到该控制步的物理状态采集，包含手指指令计算、控制和 MuJoCo 推进。各组在同一机器上依次运行，每组计时一次。模型载入、最终轨迹压缩和离线重建另计。",
    "",
    f"在 600 秒时，五物体等权覆盖为我们 {m[ARMS[0]]['coverage_600']:.2%}、Aude {m[ARMS[1]]['coverage_600']:.2%}。咖啡罐上，我们在前 240 秒有 {d['objects']['coffee']['native'][ARMS[0]]['zero_measured_fingertip_s']:.2f} 秒没有指尖接触，Aude 为 {d['objects']['coffee']['native'][ARMS[1]]['zero_measured_fingertip_s']:.2f} 秒。",
    "",
    "每个物体一个起始抓姿和一个固定种子。芥末瓶用于选定腕部改动，梨在此前版本上已评估，本轮另外三种物体使用同一组冻结参数。漂白清洁剂瓶与咖啡罐复用了历史手指日志的第 0 帧，两组从已经接触的状态开始。结论适用于这些仿真试次。",
    "",
    "[逐秒覆盖曲线](coverage.png) · [原始指标、轨迹指纹与阈值](results.json) · [物体与初态协议](protocol.json)",
    "",
    "从实验仓库根目录运行 `python -m Module.wrist_active.ycb_native five` 可核对已保存的轨迹并生成同一套评分。`python -m Module.wrist_active.ycb_report` 生成本报告和覆盖曲线。",
    "",
  ]
  (STUDY / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
  report()
