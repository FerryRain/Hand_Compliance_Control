"""Collect the new physical controls, recalculated information and I01-A comparisons."""

import argparse
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from ..i04_geometric_traversal.i01_a_report import score as i01_score
from . import sweep_verify
from .__main__ import RESULTS
from .experiment import seed_for
from .face_regression_suite import PEAR, TRANSFERS, matrix, read, write
from .face_wrist import FaceWorker, FaceWrist
from .guard_report import same_three_seconds
from .information_report import OBJECTS, check_pair, compare
from .information_score import gp_prefix, measured_cells
from .sweep_score import score as score_active


def trial_row(path):
  result = read(path / "scores.json")
  snap = next(s for s in result["snapshots"] if s["s"] == 240)
  endpoint = result["snapshots"][-1]
  return dict(
    coverage=snap["coverage"],
    f5=snap["signed"]["fscore"],
    light_two_s=snap["low2"] * snap["s"],
    light_three_s=snap["low3_s"],
    over8_s=snap["over8_s"],
    peak_n=snap["peak_n"],
    filters_all_regions_under_n=8,
    thresholds_s=result["thresholds_s"],
    trace_sha256=result["spec"]["trace_sha256"],
    endpoint=dict(
      seconds=endpoint["s"],
      coverage=endpoint["coverage"],
      f5=endpoint["signed"]["fscore"],
    ),
  )


def controlled(root):
  area = root / "regression/controlled"
  rows = []
  for case in (PEAR, *TRANSFERS):
    obj, start = case.split("/")
    seeds = (20261120, 20261122) if obj == "pear" else (seed_for(obj, start),)
    for seed in seeds:
      methods = (
        ["wrist-dp", "aude-dp", "wrist-mcc", "aude-mcc"]
        if (obj == "pear" and seed == 20261120)
        else ["wrist-dp", "aude-dp"]
      )
      checks = check_pair(area, obj, start, seed, methods)
      folder = area / "evaluation/matched" / case / f"seed-{seed}"
      values = {}
      for method in methods:
        info = read(folder / method / "information.json")
        snap = info["snapshots"]
        values[method] = dict(
          nats=snap["t240"]["nats"],
          cells=snap["t240"]["new_cells"],
          variance_drop=snap["t240"]["variance_reduction_fraction"],
          n1000=snap.get("n1000", {}).get("nats"),
          sensitivity={k: v["t240"] for k, v in info["sensitivity"].items()},
          trace_sha256=info["spec"]["trace_sha256"],
        )
      motion = compare(folder, "wrist-dp", "aude-dp")
      rows.append(
        dict(
          case=case,
          seed=seed,
          arms=values,
          motion=motion,
          paired_steps=checks["wrist-dp"]["steps"],
          paired_segments=checks["wrist-dp"]["segments"],
        )
      )
  held = [r for r in rows if r["case"] != PEAR]
  equal = [
    r
    for r in held
    if all(r["arms"][a]["n1000"] is not None for a in ("wrist-dp", "aude-dp"))
  ]
  return dict(
    rows=rows,
    transfer_nats={
      a: float(np.mean([r["arms"][a]["nats"] for r in held]))
      for a in ("wrist-dp", "aude-dp")
    },
    wins=sum(
      r["arms"]["wrist-dp"]["nats"] > r["arms"]["aude-dp"]["nats"] for r in held
    ),
    equal_1000_pairs=len(equal),
    equal_1000_wins=sum(
      r["arms"]["wrist-dp"]["n1000"] > r["arms"]["aude-dp"]["n1000"] for r in equal
    ),
    equal_1000_nats={
      a: float(np.mean([r["arms"][a]["n1000"] for r in equal])) if equal else None
      for a in ("wrist-dp", "aude-dp")
    },
    equal_motion_nats={
      a: float(np.mean([r["motion"]["arms"][a]["nats"] for r in held]))
      for a in ("wrist-dp", "aude-dp")
    },
    equal_motion_wins=sum(
      r["motion"]["arms"]["wrist-dp"]["nats"] > r["motion"]["arms"]["aude-dp"]["nats"]
      for r in held
    ),
  )


def plot_information(sub, info, main):
  import matplotlib

  matplotlib.use("Agg")
  from matplotlib import pyplot as plt

  plt.rcParams.update(
    {
      "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
      "axes.unicode_minus": False,
    }
  )
  native = {r["obj"] + "/" + r["start"]: r["arms"] for r in main["cases"]}
  held = [r for r in info["rows"] if r["case"] != PEAR]
  names = [OBJECTS[r["case"].split("/")[0]] + " " + r["case"][-2:] for r in held]
  y = np.arange(len(names))
  fig, (left, right) = plt.subplots(
    1, 2, figsize=(11.0, 4.6), layout="constrained", sharey=True
  )
  full_delta = [
    native[r["case"]]["updated"]["information_nats"]
    - native[r["case"]]["aude"]["information_nats"]
    for r in held
  ]
  wrist_delta = [
    r["arms"]["wrist-dp"]["nats"] - r["arms"]["aude-dp"]["nats"] for r in held
  ]
  left.scatter(
    wrist_delta, y - 0.14, label="同一手指策略和腕令限幅", color="#167769", s=44
  )
  left.scatter(
    full_delta, y + 0.14, label="各自完整系统", color="#b96935", marker="s", s=38
  )
  left.set(
    title="240 秒内实际采到的信息",
    xlabel="腕部 Active 减去 Aude（nat）",
    yticks=y,
    yticklabels=names,
  )
  left.invert_yaxis()
  left.legend(
    frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=9, ncol=2
  )
  for i, r in enumerate(held):
    ours, aude = (r["arms"][arm]["n1000"] for arm in ("wrist-dp", "aude-dp"))
    if ours is not None and aude is not None:
      right.scatter(ours - aude, i, color="#167769", s=44)
    else:
      right.text(
        0.88,
        i,
        "一侧未达 1,000 格",
        transform=right.get_yaxis_transform(),
        ha="right",
        va="center",
        fontsize=9,
        color="#636c70",
      )
  right.set(
    title="同一手指策略 · 前 1,000 个新格", xlabel="腕部 Active 减去 Aude（nat）"
  )
  for ax in (left, right):
    ax.axvline(0, color="#4b5354", lw=0.8)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
  fig.savefig(sub / "information-gain.png", dpi=175)
  plt.close(fig)


def i01_rows(root, tasks):
  groups = {}
  for t in tasks:
    d = t["details"]
    if d:
      after = Path(d["dest"])
      groups.setdefault(after.parent, []).append((Path(d["previous"]), after))
  rows = []
  for parent, pairs in groups.items():
    # On flat facets, a change in BLAS reduction order can break an almost
    # tied normal score differently. Use the archived single-thread setting.
    with threadpool_limits(limits=1, user_api="blas"):
      i01_score(parent)
    for previous, after in pairs:
      a, b = (read(base / "score.json") for base in (previous, after))

      def quality(r):
        return {
          k: r["quality"][k] for k in ("coverage_10mm", "coverage_25mm", "fscore_5mm")
        }

      rows.append(
        dict(
          case=after.relative_to(root / "regression/i01-a").as_posix(),
          reference=quality(a),
          recomputed=quality(b),
          identical_nominal_wrist=b["wrist_sha256"] == a["wrist_sha256"],
        )
      )
  return rows


def registration_sensitivity(root):
  from ..i04_geometric_traversal.contracts import ExperimentConfig
  from ..i04_geometric_traversal.fingertip_benchmark import read_contacts
  from ..i04_geometric_traversal.fingertip_replay import case_surface
  from ..i04_geometric_traversal.fingertip_surface_report import coverage_score
  from ..i04_geometric_traversal.fingertip_traversal import surface_graph
  from ..i04_geometric_traversal.oracle import MeasuredCoverage

  parent = root / "regression/i01-a/fair/runs/meat-can/start-02/shared"
  measures = {}
  for threads in (1, 3):
    with threadpool_limits(limits=threads, user_api="blas"):
      graph = surface_graph(case_surface(parent / "fixed-dp_r01"))
      for arm in ("fixed-grasp_r01", "fixed-mcc_r01"):
        case = parent / arm
        points, normals, steps, loads = read_contacts(case)
        states = read(case / "stopwatch.json")["states"]
        score, _ = coverage_score(
          MeasuredCoverage(graph, ExperimentConfig()),
          points,
          normals,
          steps,
          loads < 8,
          states,
          np.array([0, states]),
        )
        measures.setdefault(arm, {})[str(threads)] = dict(
          coverage=score["all_heads"]["area_fraction_by_radius_mm"]["10"],
          registered_rows=score["registered_contact_rows"],
        )
  return dict(
    reference_blas_threads=1,
    case="meat-can/start-02/shared",
    repeated_registration=measures,
  )


def can_replays(root):
  verified = {}
  for case in ("meat-can/start-01", "meat-can/start-02"):
    dest = root / "regression" / ("causal-" + case.replace("/", "-") + ".json")
    source = read(root / "runs" / case / "sweep-dp/run.json")
    if not dest.is_file() or read(dest)["trace_sha256"] != source["trace_sha256"]:
      with (
        patch.object(sweep_verify, "SweepWrist", FaceWrist),
        patch.object(sweep_verify, "WristWorker", FaceWorker),
      ):
        sweep_verify.replay(root, case, output=dest)
    result = read(dest)
    if result["packets"] != 24000 or result["object_geometries"] != 0:
      raise ValueError("Incomplete sensor-only can replay")
    verified[case] = result
  return verified


def collect(root):
  sub = root / "regression"
  tasks = matrix(root)
  receipts = []
  for t in tasks:
    receipts.append(read(sub / "receipts" / (t["group"] + "_" + t["tag"] + ".json")))
  native = {}
  for case in (PEAR, *TRANSFERS):
    tested = sub / "native/runs" / case / "sweep-mcc"
    raw = np.load(tested / "trace.npz", allow_pickle=False)
    with raw:
      pts, ticks = measured_cells(raw)
      light = np.max(raw["region_load_n"][1:24001], axis=1) < 8
      valid_count = raw["pad_valid"][1:24001].sum(axis=1)
      light_one_s = 0.01 * np.count_nonzero(light & (valid_count == 1))
    ig, _ = gp_prefix(pts, 0.025, 0.008)
    native[case] = dict(
      **trial_row(tested),
      nats=float(ig[-1] - ig[(ticks == 0).sum()]),
      light_one_s=float(light_one_s),
    )
  checks = {}
  for name, folder in [
    ("pear_long_dp", sub / "native/runs" / PEAR / "sweep-dp"),
    ("pear_long_mcc", sub / "native/runs" / PEAR / "sweep-mcc"),
    ("pear_second_noise", sub / "second-noise/runs" / PEAR / "sweep-dp"),
    ("pear_route_only", sub / "ablations/runs" / PEAR / "route-dp"),
    ("pear_without_recontact", sub / "ablations/runs" / PEAR / "no-recontact-dp"),
  ]:
    checks[name] = dict(
      **trial_row(folder),
      sim_seconds=read(folder / "run.json")["signature"]["seconds"],
      verified_physical_fields=read(folder / "regression.json")["fields"],
    )
  references = {}
  continuous_three = {}
  for case in TRANSFERS:
    folder = sub / "strong-baseline" / case / "aude-mcc"
    continuous_three[case] = {
      "updated": same_three_seconds(root / "runs" / case / "sweep-dp"),
      "pure": same_three_seconds(sub / "native/runs" / case / "sweep-mcc"),
      "aude": same_three_seconds(folder),
    }
    references[case] = dict(
      **trial_row(folder), identical_fields=read(folder / "regression.json")["fields"]
    )
  for method in ("aude-mcc", "aude-single-all-mcc"):
    folder = sub / "active-02-a" / method
    done = read(folder / "run.json")["seconds"]
    metrics = trial_row(folder)
    if metrics["endpoint"]["seconds"] < done:
      terminal = score_active(folder, folder / "terminal", "pear", "active-02", [done])[
        "snapshots"
      ][-1]
      metrics["endpoint"] = dict(
        seconds=done, coverage=terminal["coverage"], f5=terminal["signed"]["fscore"]
      )
    references[PEAR + "/" + method] = dict(
      **metrics,
      identical_fields=read(folder / "regression.json")["fields"],
      sim_seconds=done,
    )
  fixed = {
    case: read(root / "paired-wrist" / case / "result.json")
    for case in (PEAR, "ovoid/start-03", "meat-can/start-01", "meat-can/start-02")
  }
  timed = {}
  for mode in ("paced", "paced-dispatch"):
    trial = sub / "timing" / mode / PEAR / "sweep-dp"
    r = read(trial / "run.json")
    timing = read(trial / "scores.json")["timing"]
    timed[mode] = dict(
      steps=r["steps"],
      late_wrist=r["late_wrist_handoffs"],
      late_finger=r["missed_handoffs"],
      finger_installs=r["installs"],
      finger_underrun_ticks=r["underrun_ticks"],
      over_10ms=timing["finish_from_release"]["over"],
      finish_p99_ms=timing["finish_from_release"]["p99"],
      plan_p99_ms=timing["plan"]["p99"],
    )
  info = controlled(root)
  main = read(root / "results.json")
  plot_information(sub, info, main)
  fixed_policy = i01_rows(root, tasks)
  counts = dict(Counter(r["group"] for r in receipts))
  diffs = {
    key: max(abs(r["reference"][key] - r["recomputed"][key]) for r in fixed_policy)
    for key in ("coverage_10mm", "coverage_25mm", "fscore_5mm")
  }
  out = dict(
    registered_controller=read(root / "protocol.json")["source"]["face_wrist.py"],
    completed_jobs=counts,
    native_mcc=native,
    controlled_information=info,
    additional_pear=checks,
    fresh_aude=references,
    moving_same_three_s=continuous_three,
    fixed_wrist_pairs=fixed,
    pear_causal_replay=read(root / "causal-replay.json"),
    can_causal_replays=can_replays(root),
    pear_timing=timed,
    i01_a=fixed_policy,
    i01_a_max_abs_change=diffs,
    i01_a_registration=registration_sensitivity(root),
    raw_i01_equal=sum(
      r["verification"].get("head_observations_equal", False) for r in receipts
    ),
  )
  write(sub / "results.json", out)
  lines = [
    "# 腕部修正后的全套复验",
    "",
    "使用与主实验相同的冻结腕规划器。所有组都重新运行物理仿真，完整 Aude 基线以及两个种子的固定手指对照使用各自原来的预算。",
    "",
    "| 复验 | 实际运行 |",
    "|---|---:|",
    *[f"| {key} | {value} |" for key, value in counts.items()],
    "",
    f"七个迁移起点统一执行时，腕部 Active／Aude 的平均信息增益为 {info['transfer_nats']['wrist-dp']:.1f}／{info['transfer_nats']['aude-dp']:.1f} nat，腕部 Active 领先 {info['wins']}/7。双方均达到 1,000 个新触点格的 {info['equal_1000_pairs']} 个起点中，腕部 Active 领先 {info['equal_1000_wins']} 个。总平移与总转角取配对上限时，腕部 Active 领先 {info['equal_motion_wins']}/7。",
    "",
    "![固定时间与固定触点数的信息增益](information-gain.png)",
    "",
    "| 相同手指控制与腕令限幅 | 腕部 Active（nat） | Aude（nat） | 首千格 · Active | 首千格 · Aude |",
    "|---|---:|---:|---:|---:|",
    *[
      f"| {r['case']} · {r['seed']} | {r['arms']['wrist-dp']['nats']:.1f} | {r['arms']['aude-dp']['nats']:.1f} | "
      + " | ".join(
        "未达" if r["arms"][k]["n1000"] is None else f"{r['arms'][k]['n1000']:.1f}"
        for k in ("wrist-dp", "aude-dp")
      )
      + " |"
      for r in info["rows"]
    ],
    "",
    f"纯 MCC 的七组平均 240 秒覆盖为 {np.mean([native[c]['coverage'] for c in TRANSFERS]):.1%}，梨上达到 95% 覆盖耗时 {native[PEAR]['thresholds_s']['95']:.2f} 秒。",
    "",
    "| 腕部 Active＋纯 MCC | 240 秒覆盖 | F@5 mm | 轻载三指时长 | 累计信息（nat） |",
    "|---|---:|---:|---:|---:|",
    *[
      f"| {key} | {m['coverage']:.1%} | {m['f5']:.1%} | {m['light_three_s']:.2f} s | {m['nats']:.1f} |"
      for key, m in native.items()
    ],
    "",
    f"I01-A 两套固定腕轨试验共重跑 {len(fixed_policy)} 条。重新计算 10 mm、25 mm 覆盖和 5 mm F-score，最大绝对差分别为 {diffs['coverage_10mm']:.6g}、{diffs['coverage_25mm']:.6g} 和 {diffs['fscore_5mm']:.6g}。",
    "I01-A 计分沿用归档时的单线程 BLAS 设置。对同一罐体起点的原始触点另用三个 BLAS 线程复算，触点到网格的登记有微小差别。逐分支的登记行数与 10 mm 面积结果保存在 JSON 中。",
    "",
    "把与 single_all 配合产生的 240 秒腕令固定，再交给纯 MCC 重放。下表在相同腕令下统计轻载三指累计时长，腕部实际位姿仍会随受力变化。",
    "",
    "| 固定同一串腕令 | single_all＋MCC（秒） | 纯 MCC（秒） |",
    "|---|---:|---:|",
    *[
      f"| {case} | {240 * rec['dp_low3']:.2f} | {240 * rec['mcc_low3']:.2f} |"
      for case, rec in fixed.items()
    ],
    "",
    "两个罐体起点各将 24,000 帧历史触觉和手部关节状态输入只含手模型的规划器。逐帧核验腕令、策略预览以及每个 200 ms 腕段的规划决定。结果文件记录了两条轨迹的最大分量误差。",
    "",
    "[全部复验结果](results.json) · [主实验报告](../REPORT.md)",
  ]
  (sub / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--root", type=Path, default=RESULTS)
  opts = p.parse_args()
  torch.set_num_threads(1)
  collect(opts.root.resolve())
