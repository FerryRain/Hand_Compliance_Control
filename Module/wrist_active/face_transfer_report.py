"""Register the selected wrist law and audit matched 240 s physical rollouts."""

import argparse
import json
from pathlib import Path

import numpy as np

from .__main__ import RESULTS, sources
from .evaluate import bouts
from .experiment import ACTIVE2, digest, save_json
from .information_score import gp_prefix, measured_cells, wrist_travel
from .mcc_trials import read
from .sweep_experiment import OUT
from .sweep_suite import jobs

CASES = [(r["obj"], r["start"]) for r in jobs() if r["series"] == "runs" and r["arm"] == "sweep-dp"]
EXTRA_SEED = 20261125
ARMS = ("previous", "updated", "aude")
PHYSICAL = (
  "qpos",
  "palm_command",
  "published_preview",
  "finger_nominal",
  "points_m",
  "normals_outward",
  "pad_valid",
  "pad_force_n",
  "region_load_n",
)


def freeze(root):
  root.mkdir(parents=True, exist_ok=True)
  protocol = dict(
    source=sources(),
    cases=[list(key) for key in CASES],
    seconds=240,
    extra_seed=EXTRA_SEED,
    selector="Developed on meat-can/start-01. Start 02 and extra policy noise excluded from development.",
    primary="10 mm area-weighted geodesic coverage, 240 simulated seconds, all five regions <8 N",
    secondary="light one/two/three pads, signed F@5 mm, GP information, equal motion budget and loads",
    policy="same released single_all, same joint envelope and finger MCC",
    baseline="complete Active-2-A Aude+MCC. Archived reference trajectories and initial hashes checked.",
    reference_protocol=digest(OUT / "protocol.json"),
  )
  file = root / "protocol.json"
  if file.exists() and read(file) != protocol:
    raise ValueError("Selected controller or experiment protocol changed")
  if not file.exists():
    save_json(file, protocol)
  return protocol


def metrics(folder, quality_dir=None):
  quality_dir = quality_dir or folder
  receipt = read(folder / "run.json")
  quality = read(quality_dir / "scores.json")
  trace_hash = digest(folder / "trace.npz")
  if receipt.get("trace_sha256", trace_hash) != trace_hash or quality["spec"]["trace_sha256"] != trace_hash:
    raise ValueError(f"Trace and score mismatch in {folder}")
  snap = next(s for s in quality["snapshots"] if s["s"] == 240.0)
  with np.load(folder / "trace.npz") as f:
    q = f["qpos"][:24001]
    light = f["region_load_n"][1:24001].max(1) < 8.0
    count = f["pad_valid"][1:24001].sum(1)
    one = light & (count == 1)
    dist, angle = wrist_travel(q)
    points, ticks = measured_cells(f)
    if len(q) != 24001 or (ticks == 0).sum() != 4:
      raise ValueError("Incorrect duration or initial physical contacts")
  with np.load(quality_dir / "coverage.npz") as f:
    coverage = f["c10"][:24001]
  total, _ = gp_prefix(points, 0.025, 0.008)
  info = total[np.searchsorted(ticks, np.arange(24001), side="right")] - total[4]
  m = dict(
    coverage=snap["coverage"],
    f5=snap["signed"]["fscore"],
    information_nats=float(info[-1]),
    cells=int(len(points) - 4),
    light_one_s=float(one.sum() * 0.01),
    light_two_s=float((light & (count >= 2)).sum() * 0.01),
    light_three_s=snap["low3_s"],
    longest_light_one_s=float(bouts(one).max(initial=0.0)),
    over8_s=snap["over8_s"],
    over40_s=snap["over40_s"],
    peak_n=snap["peak_n"],
    distance_m=float(dist[-1]),
    rotation_rad=float(angle[-1]),
    trace_sha256=trace_hash,
    checkpoints={str(int(s["s"])): s["coverage"] for s in quality["snapshots"] if s["s"] <= 240},
    t70_s=quality["thresholds_s"]["70"] if coverage[-1] >= 0.70 else None,
  )
  return m, (dist, angle, coverage, info)


def audit_case(root, obj, start, *, extra=False):
  path = Path(obj) / start
  orig = (root / "repeat-original" if extra else OUT / "force-priority") / "runs" / path / "sweep-dp"
  new = (root / "repeat-updated" if extra else root) / "runs" / path / "sweep-dp"
  folders = {"previous": (orig, orig), "updated": (new, new)}
  if not extra:
    ref = OUT / "strong-baseline" / path / "aude-mcc"
    folders["aude"] = (
      (ACTIVE2 / "aude-mcc", OUT / "references" / path / "aude-mcc") if obj == "pear" else (ref, ref)
    )
  before, after = (read(p / "run.json") for p in (orig, new))
  for field in ("seed", "initial_sha256", "model_sha256", "initial_file_sha256", "robot_sha256"):
    if before[field] != after[field]:
      raise ValueError(f"{field} changed for {path}")
  for field in ("checkpoint_sha256", "sampler", "schedule", "execution_envelope_rad", "nominal_rate_rad_s"):
    if before["policy"][field] != after["policy"][field]:
      raise ValueError(f"Finger policy changed for {path}")
  if {k: after["signature"]["code"][k] for k in sources()} != sources():
    raise ValueError("Trajectory was run by another version")
  if "aude" in folders and obj != "pear":
    reference = read(folders["aude"][0] / "run.json")
    if any(reference[k] != after[k] for k in ("seed", "initial_sha256", "model_sha256")):
      raise ValueError("Unmatched intact Aude reference")
  if obj == "pear":
    with np.load(folders["aude"][0] / "trace.npz") as f, np.load(new / "trace.npz") as g:
      np.testing.assert_array_equal(f["qpos"][0], g["qpos"][0])
  rows = [json.loads(s) for s in (new / "plans.jsonl").read_text().splitlines()]
  face = [r for r in rows if r.get("face_tick") is not None]
  equal = []
  if not face:
    with np.load(orig / "trace.npz") as f, np.load(new / "trace.npz") as g:
      for field in PHYSICAL:
        np.testing.assert_array_equal(f[field][:24001], g[field], err_msg=f"{path}/{field}")
        equal.append(field)
  arms, curves = {}, {}
  for name, (source, quality) in folders.items():
    arms[name], curves[name] = metrics(source, quality)
  for name in folders:
    if name == "updated":
      continue
    pair = ("updated", name)
    d = min(curves[k][0][-1] for k in pair)
    a = min(curves[k][1][-1] for k in pair)
    for k in pair:
      dist, angle, cov, info = curves[k]
      end = int(min(np.searchsorted(dist, d, "right"), np.searchsorted(angle, a, "right")) - 1)
      arms[k].setdefault("matched_motion", {})[name] = dict(
        seconds=end * 0.01, coverage=float(cov[end]), nats=float(info[end]), distance_cap_m=d, angle_cap_rad=a
      )
  return dict(
    obj=obj,
    start=start,
    seed=after["seed"],
    initial_sha256=after["initial_sha256"],
    checkpoint_sha256=after["policy"]["checkpoint_sha256"],
    same_trace_fields=equal,
    plane_first_tick=face[0]["face_tick"] if face else None,
    arms=arms,
  )


def means(rows, names=ARMS):
  fields = (
    "coverage",
    "f5",
    "information_nats",
    "cells",
    "light_one_s",
    "light_two_s",
    "light_three_s",
    "over8_s",
    "over40_s",
    "distance_m",
  )
  return {a: {k: float(np.mean([r["arms"][a][k] for r in rows])) for k in fields} for a in names}


def write_report(root, data):
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  plt.rcParams.update(
    {"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False}
  )
  labels = {"previous": "修正前腕轨", "updated": "修正后腕轨", "aude": "完整 Aude＋MCC"}
  colors = {"previous": "#757e84", "updated": "#13756b", "aude": "#b76832"}
  fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
  for k in ARMS:
    place = {
      "previous": OUT / "force-priority/runs",
      "updated": root / "runs",
      "aude": OUT / "strong-baseline",
    }[k]
    trajectories = []
    for s in ("start-01", "start-02"):
      with np.load(
        place / "meat-can" / s / ("aude-mcc" if k == "aude" else "sweep-dp") / "coverage.npz"
      ) as d:
        trajectories.append(d["c10"][:24001:50])
    stack = 100 * np.stack(trajectories)
    seconds = np.arange(stack.shape[1]) * 0.5
    left.fill_between(seconds, stack.min(0), stack.max(0), color=colors[k], alpha=0.10, linewidth=0)
    left.plot(seconds, stack.mean(0), color=colors[k], label=labels[k], linewidth=2)
  left.set(
    xlabel="仿真时间（秒）",
    ylabel="10 mm 实触覆盖（%）",
    xlim=(0, 240),
    ylim=(0, 100),
    title="午餐肉罐两个起点的覆盖",
  )
  left.legend(frameon=False, loc="upper left", fontsize=9)
  columns = ("light_one_s", "light_two_s", "light_three_s")
  for i, k in enumerate(ARMS):
    rects = right.bar(
      np.arange(3) + (i - 1) * 0.23,
      [data["can"][k][s] for s in columns],
      width=0.22,
      color=colors[k],
      label=labels[k],
    )
    right.bar_label(rects, fmt="%.0f", fontsize=8, padding=2)
  right.set(
    xticks=np.arange(3),
    xticklabels=("恰好一指", "至少两指", "至少三指"),
    ylabel="240 秒内累计时长（秒）",
    title="五个分区均低于 8 N 的接触",
  )
  for ax in (left, right):
    ax.grid(axis="y", alpha=0.16)
    ax.spines[["top", "right"]].set_visible(False)
  fig.savefig(root / "can-coverage.png", dpi=175)
  plt.close(fig)
  lines = [
    "# 罐形物体的腕部主动探索",
    "",
    "在两个午餐肉罐起点和五个平滑曲面起点，每套控制均从对应的同一初态运行 240 仿真秒。两版腕规划器使用相同的 single_all 权重与 MCC，完整 Aude 使用其原生腕指控制与 MCC。梨另作回归。罐体起点 01 参与开发，起点 02 在冻结参数后运行。额外噪声种子 20261125 在起点 02 复核。",
    "",
    "规划器只读手部运动学、触觉和已知夹具中心。宽度至少 55 mm 的三个轻载实触点共同支持一个平面后，规划器将腕部路径对准指尖扫过的表面，按触觉载荷差调整手腕倾角，并用已观测表面引导下一段扫查。所有版本均使用同一实触覆盖评分和有符号 GPIS 重建。",
    "",
    "![午餐肉罐覆盖和轻载接触](can-coverage.png)",
    "",
    "## 午餐肉罐两个起点的 240 秒均值",
    "",
    "| 方法 | 实触覆盖 | F@5 mm | 仅一指 | ≥2 指 | ≥3 指 | GP 信息 | ≥8 N | >40 N | 实际腕平移 |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
  ]
  for k, m in data["can"].items():
    lines.append(
      f"| {labels[k]} | {m['coverage']:.1%} | {m['f5']:.1%} | {m['light_one_s']:.2f} s | {m['light_two_s']:.2f} s | {m['light_three_s']:.2f} s | {m['information_nats']:.1f} nat | {m['over8_s']:.2f} s | {m['over40_s']:.2f} s | {m['distance_m']:.2f} m |"
    )
  lines += [
    "",
    "覆盖采用实触点的 10 mm 测地邻域。指尖力至少 0.2 N，五个分区均低于 8 N。GP 分数采用 25 mm Matérn 3/2 核和 4.5 mm 体素内的首次实触。图中的阴影给出两个起点的范围。",
    "",
    "## 各起点 240 秒覆盖",
    "",
    "| 物体／起点 | 修正前 | 修正后 | 完整 Aude | 修正后轻载三指 | 是否触发平面扫查 |",
    "|---|---:|---:|---:|---:|---|",
  ]
  for r in data["cases"]:
    a = r["arms"]
    lines.append(
      f"| {r['obj']}／{r['start']} | {a['previous']['coverage']:.1%} | {a['updated']['coverage']:.1%} | {a['aude']['coverage']:.1%} | {a['updated']['light_three_s']:.2f} s | {'是' if r['plane_first_tick'] is not None else '否'} |"
    )
  t = data["transfer"]
  lines += [
    "",
    f"七个迁移起点等权平均覆盖为修正前 {t['previous']['coverage']:.1%}、修正后 {t['updated']['coverage']:.1%}、完整 Aude {t['aude']['coverage']:.1%}。",
    "",
    "## 相同腕部运动预算",
    "",
    "按每对轨迹实际累计平移和转角分别取较小终值，各自截到任一上限之前。下表比较修正后的腕轨与完整 Aude 的实触覆盖。",
    "",
    "| 起点 | 修正后 | 完整 Aude | 平移上限 | 旋转上限 |",
    "|---|---:|---:|---:|---:|",
  ]
  for r in data["cases"]:
    if r["obj"] == "meat-can":
      new, base = (r["arms"][k]["matched_motion"]["aude"] for k in ("updated", "aude"))
      lines.append(
        f"| {r['start']} | {new['coverage']:.1%} | {base['coverage']:.1%} | {new['distance_cap_m']:.2f} m | {new['angle_cap_rad']:.1f} rad |"
      )
  alt = data["extra_seed"]["arms"]
  lines += [
    "",
    "## 复核",
    "",
    f"额外种子下，同一罐体起点的修正前／后覆盖为 {alt['previous']['coverage']:.1%}／{alt['updated']['coverage']:.1%}，轻载三指累计为 {alt['previous']['light_three_s']:.2f}／{alt['updated']['light_three_s']:.2f} 秒。",
    "平滑物体和梨分别运行完整闭环，腕部指令、策略预览、手指指令、关节状态和触觉等九个字段均逐帧比对。各项结果与原始轨迹指纹可在结果文件中查阅。",
    "",
    "[逐组指标、载荷与原始轨迹指纹](results.json) · [试验协议](protocol.json)",
    "",
    "在实验仓库根目录运行 `Module/wrist_active/run_face_transfer.ps1` 可重做物理试次、计分和本报告。[其他实验的复验结果](regression/REPORT.md) · [含罐体配对回放的视频](wrist-face-transfer.mp4)。",
  ]
  paced = data["paced"]
  lines += [
    "",
    "## 150 秒实时时钟仿真",
    "",
    f"罐体起点 02 在 100 Hz 执行腕令、5 Hz 后台重规划。腕规划 p95 为 {paced['plan_p95_ms']:.2f} ms，p99 为 {paced['plan_p99_ms']:.2f} ms。15,000 帧中有 {paced['missed_10ms']} 帧在释放后超过 10 ms 才完成，{paced['late_wrist']} 次腕段延迟交接。该测量包含本机 MuJoCo 与冻结策略执行。",
    f"手指策略在 {paced['finger_installs']} 次动作块安装中有 {paced['late_finger']} 次未赶上预定的 50 ms 交接。等待期间继续使用已缓冲的手指动作，耗尽缓冲的控制帧数为 {paced['finger_underrun_ticks']}。整帧从释放至完成的 p99 为 {paced['finish_p99_ms']:.2f} ms。",
  ]
  (root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def report(root):
  protocol = freeze(root)
  archive = []
  for obj, start in CASES:
    archive.append(audit_case(root, obj, start))
    print("Scored", obj, start, flush=True)
  extra = audit_case(root, "meat-can", "start-02", extra=True)
  can = [r for r in archive if r["obj"] == "meat-can"]
  transfer = [r for r in archive if r["obj"] != "pear"]
  p = root / "paced/runs/meat-can/start-02/sweep-dp"
  receipt = read(p / "run.json")
  timed = read(p / "coverage-only.json")
  if (
    not receipt["signature"]["paced"]
    or receipt["steps"] != 15000
    or receipt["trace_sha256"] != timed["spec"]["trace_sha256"]
  ):
    raise ValueError("Incorrect paced trial")
  timing = timed["timing"]
  paced = dict(
    seconds=150,
    steps=receipt["steps"],
    trace_sha256=receipt["trace_sha256"],
    plan_p95_ms=timing["plan"]["p95"],
    plan_p99_ms=timing["plan"]["p99"],
    missed_10ms=timing["finish_from_release"]["over"],
    late_wrist=receipt["late_wrist_handoffs"],
    late_finger=receipt["missed_handoffs"],
    finger_installs=receipt["installs"],
    finger_underrun_ticks=receipt["underrun_ticks"],
    finish_p99_ms=timing["finish_from_release"]["p99"],
    control_p99_ms=timing["loop_service"]["p99"],
  )
  data = dict(
    protocol=protocol, cases=archive, can=means(can), transfer=means(transfer), extra_seed=extra, paced=paced
  )
  save_json(root / "results.json", data)
  write_report(root, data)
  return data


if __name__ == "__main__":
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("action", choices=("freeze", "report"))
  p.add_argument("--root", type=Path, default=RESULTS)
  args = p.parse_args()
  (freeze if args.action == "freeze" else report)(args.root)
