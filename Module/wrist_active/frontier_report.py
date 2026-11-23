"""Score the two complete systems and compare targeted regressions to their recorded history."""

import numpy as np

from . import ycb_native
from .experiment import digest, seed_for
from .face_regression_suite import REPO, read, write
from .frontier_study import loads
from .ycb_fixtures import OBJECTS
from .ycb_fixtures import STUDY as PREVIOUS
from .ycb_report import ARMS, LABELS, graph


def windows(folder):
  snapshots = read(folder / "scores.json")["snapshots"]
  result = {}
  for end in (240, 600):
    snap = next(s for s in snapshots if s["s"] == end)
    result[str(end)] = dict(
      loads(folder, end),
      coverage=snap["coverage"],
      mean_coverage=snap["mean_coverage"],
      light3_s=snap["low3_s"],
      all_zone_peak_n=snap["peak_n"],
      all_zone_over8_s=snap["over8_s"],
      f5=snap["signed"]["fscore"],
      cells=snap["cells"],
    )
  with np.load(folder / "coverage.npz") as f:
    curve = f["c10"]
    final = result["600"]
    final["coverage_at_300"] = float(curve[30000])
    final["gain_last_300"] = float(curve[60000] - curve[30000])
    final["gain_last_60"] = float(curve[60000] - curve[54000])
  return result


def verify_control(folder, old):
  with np.load(folder / "trace.npz") as a, np.load(old / "trace.npz") as b:
    for key in (
      "qpos",
      "finger_nominal",
      "finger_command",
      "pad_valid",
      "points_m",
      "normals_outward",
      "region_load_n",
      "pad_force_n",
    ):
      np.testing.assert_array_equal(
        a[key], b[key], err_msg=f"Aude changed, field {key}"
      )


def score_release(study, spec, objects):
  old = read(PREVIOUS / "results.json")
  for obj in objects:
    start = OBJECTS[obj][2]
    ycb_native.audit_pair(obj, spec)
    arms = {}
    for arm, key in zip(("ours", "aude"), ARMS, strict=True):
      folder = ycb_native.destination(obj, arm)
      original = REPO / old["objects"][obj]["native"][key]["source"]
      clock = read(folder / "clock.json")
      if clock["clock_sha256"] != digest(folder / "native-clock.npz") or clock[
        "trace_sha256"
      ] != digest(folder / "trace.npz"):
        raise ValueError("Acquisition receipt does not match the trajectory")
      if arm == "ours":
        if read(folder / "run.json")["signature"]["code"] != spec["code"]:
          raise ValueError(
            "The five runs must share the registered wrist code and parameters"
          )
      else:
        verify_control(folder, original)
      row = ycb_native.summarize(obj, arm, spec, study)
      row["windows"] = vals = windows(folder)
      row.update(
        light3_600=vals["600"]["light3_s"],
        f5_600=vals["600"]["f5"],
        coverage_gain_240_600=row["coverage_600"] - row["coverage"],
      )
      with np.load(folder / "coverage.npz") as f:
        curve = f["c10"]
        last = np.flatnonzero(np.diff(curve) > 1e-12)
        row["last_coverage_increase_s"] = (
          (int(last[-1]) + 1) / 100 if len(last) else 0.0
        )
      arms[key] = row
    prior = REPO / old["objects"][obj]["native"][ARMS[0]]["source"]
    write(
      study / "pairs" / f"{obj}.json",
      dict(
        start=start,
        seed=seed_for(obj, start),
        initial_sha256=spec["fixtures"][f"{obj}/{start}"]["integration_state"],
        native=arms,
        previous_wrist=dict(
          windows=windows(prior), trace_sha256=digest(prior / "trace.npz")
        ),
      ),
    )
    print(
      obj,
      "600 s",
      *[round(100 * r["coverage_600"], 2) for r in arms.values()],
      flush=True,
    )
  if len(objects) != len(OBJECTS):
    return
  pairs = {obj: read(study / "pairs" / f"{obj}.json") for obj in OBJECTS}
  macro = {}
  for key in ARMS:
    rows = [p["native"][key] for p in pairs.values()]
    macro[key] = m = {
      name: float(np.mean([r[name] for r in rows]))
      for name in (
        "coverage",
        "coverage_600",
        "coverage_gain_240_600",
        "mean_coverage",
        "f5",
        "f5_600",
        "light3",
        "light3_600",
        "nats",
        "over8",
        "wall_s_240",
        "wall_s_600",
      )
    }
    m["thresholds"] = {}
    for t in spec["thresholds"]:
      values = [r["thresholds_s"][str(t)] for r in rows]
      m["thresholds"][str(t)] = dict(
        hit=sum(v is not None for v in values),
        mean_restricted_s=float(np.mean([600 if v is None else v for v in values])),
      )
  result = dict(
    objects=pairs,
    macro=macro,
    chosen="FrontierWrist",
    arms=ycb_native.NAMES,
    budget_s=600,
    primary_budget_s=600,
    snapshots_s=ycb_native.TIMES,
    primary=spec["primary"],
    limit_n=8,
    score_radius_mm=10,
    control_dt_s=0.01,
    scope=spec["scope"],
    baseline="New 600-second Aude acquisitions; recorded joint, force, touch and position arrays checked against the archived baseline.",
    clock_note="Rollouts ran batch-parallel on one host. For method comparisons, threshold times use simulation seconds.",
    checkpoint_sha256=spec["checkpoint"],
    release_sha256=spec["release_sha256"],
    protocol_sha256=digest(study / "protocol.json"),
  )
  write(study / "results.json", result)
  graph(result, study, 600)
  report(study, result)


def time_text(value):
  return "未达" if value is None else f"{value:.2f}"


def report(study, result):
  rows = [
    "# 五物体 600 秒腕部主动探索",
    "",
    "每个物体从原有轻载四指抓姿各运行两套完整系统。腕部 Active 的参数在五个物体上相同，手指由冻结的 single_all 和 MCC 控制。Aude 使用项目 Active-02-A 的腕指联合规划与 MCC。开发以咖啡罐与清洁剂瓶为主，另根据梨的回归结果修正了无法估计接触面时的接近速度。",
    "",
    "|物体|系统|240 s 覆盖|600 s 覆盖|240 至 600 s 增量（百分点）|到 70%（s）|到 95%（s）|",
    "|---|---|---:|---:|---:|---:|---:|",
  ]
  for obj, p in result["objects"].items():
    for key, name in zip(ARMS, LABELS, strict=True):
      r = p["native"][key]
      t = r["thresholds_s"]
      rows.append(
        f"|{OBJECTS[obj][1]}|{name}|{r['coverage']:.2%}|{r['coverage_600']:.2%}|{100 * r['coverage_gain_240_600']:+.2f}|{time_text(t['70'])}|{time_text(t['95'])}|"
      )
  for key, name in zip(ARMS, LABELS, strict=True):
    m = result["macro"][key]
    rows.append(
      f"|五物体等权平均|{name}|{m['coverage']:.2%}|{m['coverage_600']:.2%}|{100 * m['coverage_gain_240_600']:+.2f}|—|—|"
    )
  rows.extend(
    [
      "",
      "覆盖按有效指尖触点在整个参考表面上的 10 mm 测地邻域计算。指尖力至少 0.2 N，五个分区均须低于 8 N。参考网格只用于离线评估。阈值时间均为仿真秒。",
      "",
      "## 载荷、失触与建模",
      "",
      "|物体|系统|窗口（s）|无指尖接触（s）|掌部 ≥8 N（s）|掌部峰值（N）|任一区峰值（N）|轻载至少三指（s）|重建 F@5 mm|",
      "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
  )
  for obj, p in result["objects"].items():
    for name, data in [
      (LABELS[0], p["native"][ARMS[0]]),
      (LABELS[1], p["native"][ARMS[1]]),
      ("修改前腕部 Active", p["previous_wrist"]),
    ]:
      for end, v in data["windows"].items():
        rows.append(
          f"|{OBJECTS[obj][1]}|{name}|{end}|{v['zero_tip_s']:.2f}|{v['palm_over8_s']:.2f}|{v['palm_peak_n']:.2f}|{v['all_zone_peak_n']:.2f}|{v['light3_s']:.2f}|{v['f5']:.2%}|"
        )
  rows += [
    "",
    "无指尖接触按四个指尖均未测到有效头部接触的控制帧累计。掌部阈值只统计掌部通道，任一区峰值覆盖手掌与四根手指。重建使用同一有符号 GPIS 的 5 mm F-score。",
    "",
    "两系统均重新推进了 60,000 个 10 ms 控制步。逐物体核对模型、初态、随机种子和首次触觉观测，并与上一轮逐帧比对 Aude 的物理轨迹。原始记录保留墙钟时间。并行采集期间共享主机资源，方法间用时采用相同覆盖阈值的仿真时间。",
  ]
  bottle = result["objects"]["bleach"]
  can = result["objects"]["coffee"]
  before = bottle["previous_wrist"]["windows"]["600"]
  after = bottle["native"][ARMS[0]]["windows"]["600"]
  old_can = can["previous_wrist"]["windows"]["600"]
  new_can = can["native"][ARMS[0]]["windows"]["600"]
  rows.extend(
    [
      "",
      "## 两个开发问题的回归结果",
      "",
      f"咖啡罐全程无指尖接触从 {old_can['zero_tip_s']:.2f} s 降至 {new_can['zero_tip_s']:.2f} s。掌部峰值从 {old_can['palm_peak_n']:.1f} N 降至 {new_can['palm_peak_n']:.1f} N，掌部达到 8 N 的累计时间从 {old_can['palm_over8_s']:.2f} s 变为 {new_can['palm_over8_s']:.2f} s。",
      "",
      f"清洁剂瓶从第 300 秒到第 600 秒的覆盖增量从 {100 * before['gain_last_300']:.2f} 增至 {100 * after['gain_last_300']:.2f} 个百分点。全程至少一指轻载接触从 {before['light1_s']:.2f} s 增至 {after['light1_s']:.2f} s。",
      "",
      f"五物体协议和腕规划源码快照分别见同目录的 `protocol.json` 与 `wrist-source.py`。冻结的腕规划源码 SHA-256 为 `{result['release_sha256']}`。",
    ]
  )
  (study / "REPORT.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
