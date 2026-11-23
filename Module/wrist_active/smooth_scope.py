"""Report all eligible curved-surface starts and complete their native Aude reference."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from itertools import combinations
from pathlib import Path
from statistics import mean

import numpy as np
import torch

from ..i04_geometric_traversal.fingertip_hand import DT as FIXED_DT
from . import information_score, sweep_aude_transfer
from .component_report import (
  ARM_NAMES,
  assert_paired,
  differences,
  figure,
  means,
  title,
  trial_values,
)
from .component_report import report as full_report
from .component_study import NEW_STARTS, OUT
from .evaluate import movement
from .experiment import digest
from .face_regression_suite import REPO, TRANSFERS, read, write
from .information_score import LENGTHS_M, NOISE_VARIANCE, gp_prefix, measured_cells
from .sweep import DT
from .sweep_score import score

ROOT = OUT / "smooth"
ORIGINAL = tuple(c for c in TRANSFERS if c.split("/")[0] != "meat-can")
ARMS = ("wrist-dp", "wrist-mcc", "native-aude")


def select():
  old = read(OUT / "protocol.json")
  if set(ORIGINAL) & set(NEW_STARTS) or len(ORIGINAL) != 5 or len(NEW_STARTS) != 6:
    raise ValueError("Unexpected curve cohort")
  spec = dict(
    scope="Post hoc geometric subgroup of the registered transfer study. Select all four curved meshes, omit both meat-can starts irrespective of arm and metric, keep the separate pear test out of the averages.",
    original=list(ORIGINAL),
    validation=list(NEW_STARTS),
    excluded=[c for c in TRANSFERS if c not in ORIGINAL],
    aggregation="Equal weight per paired start; show original five, new six and all eleven. Report every native system on every start.",
    budget_s=240,
    source_protocol_sha256=digest(OUT / "protocol.json"),
    baseline_sources=dict(
      runner=digest(Path(sweep_aude_transfer.__file__)),
      controller=digest(Path(sweep_aude_transfer.aude.__file__)),
      scorer=digest(Path(information_score.__file__).with_name("sweep_score.py")),
      information=digest(Path(information_score.__file__)),
    ),
    validation_fixtures=old["native_validation"]["fixtures"],
    endpoints=old["endpoints"],
  )
  target = ROOT / "selection.json"
  if target.exists():
    if read(target) != spec:
      raise ValueError("Curve subset or frozen inputs changed")
  else:
    write(target, spec)
  return spec


def run_reference(case):
  torch.set_num_threads(1)
  spec = read(ROOT / "selection.json")
  obj, start = case.split("/")
  path = sweep_aude_transfer.rollout(ROOT, obj, start, spec["budget_s"])
  run = read(path / "run.json")
  fixture = spec["validation_fixtures"][case]
  if (run["initial_sha256"], run["model_sha256"], run["seed"]) != (
    fixture["initial"],
    fixture["model"],
    fixture["seed"],
  ):
    raise ValueError(
      f"Aude and wrist-only systems saw different initial states on {case}"
    )
  if (
    run["active2_source"] != spec["baseline_sources"]["controller"]
    or run["code"] != spec["baseline_sources"]["runner"]
  ):
    raise ValueError("Unexpected Aude controller")
  quality = score(path, path, obj, start, [240.0])
  native = OUT / "native" / case / "sweep-dp"
  if quality["spec"]["mesh_sha256"] != fixture["mesh"]:
    raise ValueError("Different evaluation mesh")
  with np.load(path / "trace.npz") as tr, np.load(native / "trace.npz") as dp:
    for key in ("qpos", "pad_valid", "points_m", "region_load_n"):
      np.testing.assert_array_equal(tr[key][0], dp[key][0])
    if len(tr["pad_valid"]) != 24001 or not tr["pad_valid"][0].all():
      raise ValueError("Incorrect time or contact budget")
    pts, ticks = measured_cells(tr)
  initial = int((ticks == 0).sum())
  if initial != 4:
    raise ValueError("Unexpected initial GP support")
  cumulative, _ = gp_prefix(pts, LENGTHS_M[0], NOISE_VARIANCE)
  row = trial_values(path, path, float(cumulative[-1] - cumulative[initial]))
  write(path / "verified.json", row)
  return case


def native_groups(data):
  old = {c: data["native_transfer"][c] for c in ORIGINAL}
  fresh = {}
  for c in NEW_STARTS:
    rows = dict(data["native_validation"][c]["arms"])
    rows["native-aude"] = read(ROOT / "strong-baseline" / c / "aude-mcc/verified.json")
    assert_paired(rows)
    fresh[c] = {"arms": rows}
  return {"original": old, "validation": fresh, "all": {**old, **fresh}}


def common_contact(pads, loads, qpos):
  light = loads < 8
  three = light & (pads.sum(axis=1) >= 3)
  moving = movement(qpos)[1:]
  retained = np.zeros(len(pads), dtype=bool)
  for trio in combinations(range(4), 3):
    both = moving & light & pads[:, trio].all(axis=1)
    edges = np.flatnonzero(np.diff(np.r_[False, both, False])).reshape(-1, 2)
    for start, end in edges:
      if (end - start) * DT >= 1:
        retained[start:end] = True
  return dict(
    light3_rate=float(three.mean()), same_three_moving_rate=float(retained.mean())
  )


def matched_i01_window(cases):
  folder = OUT.parent / "i01_a_smooth_20261121"
  groups = read(folder / "metrics.json")["groups"]
  stride = round(DT / FIXED_DT)
  assert np.isclose(stride * FIXED_DT, DT)
  end = min(
    round(240 / DT),
    min(
      row["states"] // stride
      for g in groups
      for row in g["trials"]
      if row["method"] in ("fixed-dp", "fixed-mcc")
    ),
  )
  indices = np.arange(1, end + 1) * stride - 1
  by_case = {}
  for g in groups:
    case = g["object"] + "/" + g["start"]
    pair = {}
    for fingers in ("dp", "mcc"):
      row = next(v for v in g["trials"] if v["method"] == "fixed-" + fingers)
      fixed = folder / row["trial"]
      active = cases[case]["arms"]["wrist-" + fingers]
      assert digest(fixed / "initial.npz") == row["initial_sha256"]
      with np.load(fixed / "initial.npz") as t:
        if (
          sha256(t["integration_state"].tobytes()).hexdigest()
          != active["initial_sha256"]
        ):
          raise ValueError(
            "I01-A and Active do not share the same simulated initial state"
          )
      with np.load(REPO / active["source"] / "trace.npz") as t:
        q = t["qpos"][: end + 1, :7]
        active_measured = common_contact(
          t["pad_valid"][1 : end + 1], t["region_load_n"][1 : end + 1].max(axis=1), q
        )
      with np.load(fixed / "live_head_contacts.npz") as t:
        pads = np.zeros((row["states"], 4), dtype=bool)
        assert np.all(t["normal_force_n"] >= 0.2 - 1e-6)
        pads[t["step"] - 1, t["finger"]] = True
      with np.load(fixed / "moving_contact.npz") as t:
        np.testing.assert_array_equal(pads.sum(axis=1), t["head_count"])
        np.testing.assert_allclose(
          t["time_s"][indices], np.arange(1, end + 1) * DT, rtol=0, atol=1.0e-8
        )
        loads = t["region_load_n"][indices]
      recorded = []
      for part in sorted(fixed.glob("trace_*.npz")):
        with np.load(part) as t:
          recorded.append(t["qpos"][:, :7])
      positions = np.concatenate(recorded)
      if len(positions) != row["states"]:
        raise ValueError("Incomplete fixed-wrist measurements")
      pair["fixed-" + fingers] = common_contact(
        pads[indices], loads, np.vstack((q[0], positions[indices]))
      )
      pair["active-" + fingers] = active_measured
    by_case[case] = pair
  averages = {
    arm: {k: mean(v[arm][k] for v in by_case.values()) for k in val}
    for arm, val in pair.items()
  }
  return dict(
    seconds=end * DT,
    dt=DT,
    cases=by_case,
    means=averages,
    definition="Shared 100 Hz clock from the initial state through the shortest common full 10 ms tick. Cumulative light contact uses all frames. Identical movement mask and union of qualifying same-trio bouts for the separate continuous metric.",
  )


def summarize(data):
  groups = native_groups(data)
  planned = {c: data["matched_transfer"][c] for c in ORIGINAL}
  metric = {}
  for f in ("dp", "mcc"):
    first, second = "wrist-" + f, "aude-" + f
    arms = [r["arms"] for r in planned.values()]
    common = [r["common_motion"][f]["arms"] for r in planned.values()]
    cells = [
      (p[first]["fixed_1000"], p[second]["fixed_1000"])
      for p in arms
      if p[first]["fixed_1000"] is not None and p[second]["fixed_1000"] is not None
    ]
    metric[f] = dict(
      time=[mean(p[a]["nats"] for p in arms) for a in (first, second)],
      time_wins=sum(p[first]["nats"] > p[second]["nats"] for p in arms),
      travel=[mean(p[a]["nats"] for p in common) for a in (first, second)],
      travel_wins=sum(p[first]["nats"] > p[second]["nats"] for p in common),
      cells=[mean(row[i] for row in cells) for i in (0, 1)],
      cells_n=len(cells),
      cells_wins=sum(a > b for a, b in cells),
    )
  fixed = data["fixed_wrist_i01"]
  return dict(
    cohort=read(ROOT / "selection.json"),
    cases=groups["all"],
    means={k: means(v) for k, v in groups.items()},
    contrasts={k: differences(v, "wrist-dp", "wrist-mcc") for k, v in groups.items()},
    complete_comparison={
      k: differences(v, "wrist-dp", "native-aude") for k, v in groups.items()
    },
    matched_wrist=metric,
    fixed_wrist=fixed,
    common_i01=matched_i01_window(groups["all"]),
    pear=data["native_pear"],
  )


def write_report(d):
  window = d["common_i01"]
  fixed = d["fixed_wrist"]
  lines = [
    "# 平滑曲面上的腕部主动探索与手指接触",
    "",
    "椭球有 2 个起始抓姿，扁椭球、卵形体和超椭球各有 3 个。共 11 个抓姿，在均值中逐抓姿等权。四种曲面是在全物体试验完成后按几何选出的。梨另作独立同初态对照。[原全物体结果及午餐肉罐试验](ALL_OBJECTS.md)",
    "",
    "## I01-A 与 Active 在相同起点、相同计分窗口下的结果",
    "",
    f"三种曲面共有九个相同的仿真初态。两种试验都从初始状态取前 {window['seconds']:.2f} 秒，这是九个起点所有配对记录共有的最长整 10 ms 时间窗。每 10 ms 按四根指尖的实测接触与五区载荷计分。轻载要求每个分区低于 8 N，有效指尖力至少为 0.2 N。每个起点先计算时间占比，再等权平均。",
    "",
    "| 腕部条件 | 手指控制 | 轻载至少三指 | 腕动时同三指连续 ≥1 s |",
    "|---|---|---:|---:|",
  ]
  for source, label in [("fixed", "I01-A 共同腕令"), ("active", "腕部 Active 闭环")]:
    for f, name in [("dp", "single_all＋MCC"), ("mcc", "纯 MCC")]:
      x = window["means"][source + "-" + f]
      lines.append(
        f"| {label} | {name} | {x['light3_rate']:.2%} | {x['same_three_moving_rate']:.2%} |"
      )
  lines += [
    "",
    (
      f"I01-A 原定义取各自的扫查区间。在纯 MCC 恰有两指轻载的 {fixed['totals']['mcc_exact_two_s']:.2f} 秒里，"
      f"single_all 在对应时刻保住相同两指并至少再接触一指的比例为 {fixed['dp_retention_fraction']:.2%}。"
      "固定腕令下，两组看到相同的腕部指令。闭环 Active 根据各组触觉重新规划，因此两组的实际轨迹会分开。"
    ),
    "",
    "I01-A 原报告的连续三指指标允许参与的指尖更换。上表连续指标要求腕部移动，同时同一组三根指尖持续轻载。统一计分后，两个实验中 single_all 均提高了至少三指轻载的时间占比，均由纯 MCC 更久地连续维持同三根指尖。",
    "",
    "## 完整系统的曲面探索",
    "",
    "每套系统从相同的 11 个初始抓姿各运行 240 仿真秒。椭球有 2 个抓姿，扁椭球、卵形体和超椭球各有 3 个。我们的系统由 GPIS 规划手腕，由 single_all 调整手型并由 MCC 执行柔顺控制。完整 Aude 执行其腕指联合指令及自身反馈。",
    "",
    "| 系统 | 实触覆盖 | F@5 mm | 累计信息 | 轻载至少三指 | 同三指随腕连续 ≥1 s | 首达 70% 的限时均值 | 达到 70% | 载荷 ≥8 N |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
  ]
  systems = ("wrist-dp", "native-aude")
  for arm in systems:
    v = d["means"]["all"][arm]
    lines.append(
      f"| {ARM_NAMES[arm]} | {v['coverage']:.2%} | {v['f5']:.2%} | {v['nats']:,.1f} nat | {v['light3']:.2f} s | {v['continuous']:.2f} s | {v['rm70']:.2f} s | {v['t70_reached']}/11 | {v['over8']:.2f} s |"
    )
  lines += [
    "",
    "实触覆盖按有效触点的 10 mm 测地邻域计算。未达到覆盖阈值的试次按 240 秒计入限时均值。",
    "",
    "| 物体与抓姿 | 我们的覆盖 | 完整 Aude 覆盖 | 我们的轻载三指 | 完整 Aude 轻载三指 |",
    "|---|---:|---:|---:|---:|",
  ]
  for case, pair in sorted(d["cases"].items()):
    a, b = (pair["arms"][arm] for arm in systems)
    lines.append(
      f"| {title(case)} | {a['coverage']:.2%} | {b['coverage']:.2%} | {a['light3']:.2f} s | {b['light3']:.2f} s |"
    )
  link = ROOT / "wrist-comparison/comparison.json"
  if link.is_file():
    comparison = read(link)
    diag = comparison["native_diagnostics"]
    case = diag["largest_gap_case"]
    a, b = (diag["cases"][case][arm] for arm in systems)
    lines += [
      "",
      f"完整 Aude 的实触覆盖在 {diag['aude_coverage_wins']}/11 个抓姿领先。{title(case)} 的覆盖差距最大。我们的系统最后一个新触点格出现在 {a['last_new_cell_s']:.2f} 秒，最后一次实测指尖接触出现在 {a['last_measured_fingertip_s']:.2f} 秒。随后发起的 {a['recoveries_after_last_contact']} 次重接触均未采到新触点格。完整 Aude 在这一抓姿的最后 60 秒仍采到 {b['new_cells_by_60s'][-1]:,} 个新格。这一例占全部有符号覆盖差值之和的 {diag['share_of_sum_of_gaps']:.1%}，其余十例平均相差 {100 * diag['mean_remaining_gap']:.2f} 个百分点。",
      "",
      "![两套完整系统的实触覆盖随仿真时间变化](smooth/wrist-comparison/native-coverage.png)",
      "",
      "## 固定 single_all＋MCC 的腕规划器对比",
      "",
      "在上述全部 11 个抓姿，双方只更换腕目标来源，共用冻结手指策略、腕部执行器、运动限幅与仿真预算。Aude 在该对照中只提供腕目标。完整系统试验中，它也执行自己生成的手指指令。每个 4.5 mm 触点格只在首次轻载实触时计入。累计信息以自然对数计算。",
      "",
      "| 计分条件 | 我们的腕规划器 | Aude 腕目标 | 双方达到条件的抓姿 |",
      "|---|---:|---:|---:|",
    ]
    for key, label, unit in (
      ("nats", "240 s 累计信息", " nat"),
      ("common_motion_nats", "相同实际腕部平移和旋转上限", " nat"),
      ("first_1000", "前 1,000 个新触点格", " nat"),
      ("cells", "240 s 新触点格数", " 格"),
    ):
      row = comparison["summary"][key]
      a, b = (row["means"][arm] for arm in ("wrist-dp", "aude-dp"))
      lines.append(
        f"| {label} | {a:,.1f}{unit} | {b:,.1f}{unit} | {len(row['eligible'])}/11 |"
      )
    lines += [
      "",
      f"同样运行 240 秒时，我们的腕规划器在 {comparison['summary']['nats']['wins']}/11 个抓姿的累计信息更高。上表的前千格比较只计双方都达到千格的抓姿。两种计分分别衡量等时间和等采样数量下的实际信息。",
      "",
      "[逐抓姿的腕规划器评分、运动上限和完整系统触点诊断](smooth/wrist-comparison/comparison.json) · [补测设置](smooth/wrist-comparison/protocol.json)",
    ]
  else:
    old = d["matched_wrist"]["dp"]
    lines += [
      "",
      "## 固定 single_all＋MCC 的腕规划器对比",
      "",
      f"已有五个配对抓姿的 240 秒累计信息，腕部 Active 为 {old['time'][0]:,.1f} nat，Aude 腕目标为 {old['time'][1]:,.1f} nat。",
    ]
  x = d["contrasts"]["all"]
  lines += [
    "",
    "## 手指消融与早期四组数据",
    "",
    f"同一腕部 Active 上加入 single_all 后，{x['light3']['higher']}/11 个起点的轻载三指时长增加，平均多 {x['light3']['mean']:.2f} 秒。{x['coverage']['higher']}/11 个起点的覆盖增加。",
    "",
    "最早的五个抓姿是椭球 01、02 与其余三种物体的 03。随后补齐三种物体的 01、02。数据文件中保留这两个采集批次的均值以及早期五个抓姿的腕规划器与纯 MCC 交叉对照。",
    "",
    "[完整系统与手指消融数据](smooth-results.json) · [曲面抓姿与分组依据](smooth/selection.json) · [梨与原七组结果](ALL_OBJECTS.md)",
  ]
  (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("action", choices=("baselines", "report"))
  parser.add_argument("--workers", type=int, default=3)
  args = parser.parse_args()
  select()
  if args.action == "baselines":
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
      for case in pool.map(run_reference, NEW_STARTS):
        print("Verified complete Aude", case, flush=True)
  else:
    source = read(OUT / "results.json")
    result = summarize(source)
    full_report(source)
    write(OUT / "smooth-results.json", result)
    figure(
      {
        "native_transfer": {c: result["cases"][c] for c in ORIGINAL},
        "native_validation": {c: result["cases"][c] for c in NEW_STARTS},
      },
      "smooth-finger-effects.png",
    )
    write_report(result)


if __name__ == "__main__":
  main()
