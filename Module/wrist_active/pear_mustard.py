"""Evaluate the released wrist and finger controllers on one paired grasp per YCB object."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import sha256
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch

from ..acitive_02 import active_a
from ..i04_geometric_traversal.fingertip_dp_worker import FIELD_WIDTHS
from ..i04_geometric_traversal.fingertip_hand import HEAD_FRACTION, FingertipHand
from . import (
  experiment,
  information_score,
  sweep_aude_transfer,
  sweep_pair,
  sweep_score,
)
from . import matched_information as matched
from . import sweep_experiment as runtime
from .__main__ import RESULTS, sources
from .component_report import assert_paired, trial_values
from .component_study import ARCHIVE, matched_path
from .component_study import OUT as COMPONENTS
from .experiment import ACTIVE2, digest, seed_for
from .face_regression_suite import REPO, read, write
from .face_wrist import FaceWorker, FaceWrist
from .history import GENERATED, ROBOT
from .information_report import check_pair, compare

ROOT = GENERATED / "gpis_wrist_pear_mustard_20261122"
FIXTURE = GENERATED / "i01_a_policy_wrist_20261120/mustard/fixed-dp_r01"
STARTS = {"pear": "active-02", "mustard": "start-01"}
NATIVE = ("wrist-dp", "wrist-mcc", "native-aude", "aude-single-all")
WRISTS = ("wrist-dp", "aude-dp")
SECONDS = 240


def register():
  old = read(ARCHIVE / "protocol.json")
  prior = read(COMPONENTS / "protocol.json")
  if sources() != old["current_wrist_sources"] or old["runner_sha256"] != digest(
    Path(matched.__file__)
  ):
    raise ValueError(
      "Wrist planner or common execution interface differs from the reference"
    )
  native = read(ACTIVE2 / "protocol.json")
  if digest(ACTIVE2 / "protocol.json") != old["frozen_baseline_protocol"]:
    raise ValueError("Different native Aude protocol")
  for name, fingerprint in native["source_sha256"].items():
    if digest(Path(active_a.__file__).with_name(name)) != fingerprint:
      raise ValueError(f"Aude implementation changed: {name}")
  fixtures = {}
  for obj, start in STARTS.items():
    folder, mesh = (
      (FIXTURE, FIXTURE / "surface.obj")
      if obj == "mustard"
      else experiment.scenario(obj, start)
    )
    with np.load(folder / "initial.npz") as data:
      state = data["integration_state"]
      fixtures[f"{obj}/{start}"] = {
        "model.mjb": digest(folder / "model.mjb"),
        "initial.npz": digest(folder / "initial.npz"),
        "surface.obj": digest(mesh),
        "integration_state": sha256(state.tobytes()).hexdigest(),
        "warm_observations": len(data["warm_policy_observations"])
        if "warm_policy_observations" in data
        else 0,
        "source": folder.relative_to(REPO).as_posix(),
      }
  spec = dict(
    old,
    cases={f"{obj}/{start}": seed_for(obj, start) for obj, start in STARTS.items()},
    fixtures=fixtures,
    checkpoint_sha256=prior["checkpoint"],
    planning_hand=digest(ROBOT),
    native_arms=list(NATIVE),
    matched_arms=list(WRISTS),
    budget_s=SECONDS,
    scope="Two named objects, one previously archived grasp per object. Pear is a held-out finger-policy object. Mustard occurred in finger-policy training and hand-history development. Report per object. Shape selection requested after earlier results.",
    tape="Replay the 240 s wrist commands from our single_all run with constant initial finger nominal and identical MCC. The recorded wrist policy saw single_all contacts.",
    endpoints=prior["endpoints"],
    parent_matched_protocol=digest(ARCHIVE / "protocol.json"),
    scorers={
      m.__name__: digest(Path(m.__file__))
      for m in (information_score, sweep_score, sweep_pair)
    },
  )
  file = ROOT / "protocol.json"
  if file.exists():
    if read(file) != spec:
      raise ValueError("Registered study settings differ")
  else:
    write(file, spec)
  return spec


def native_path(obj, arm):
  start = STARTS[obj]
  if arm in ("wrist-dp", "wrist-mcc"):
    home = RESULTS / "regression/native" if obj == "pear" else ROOT
    return (
      home / "runs" / obj / start / ("sweep-dp" if arm == "wrist-dp" else "sweep-mcc")
    )
  home = (
    RESULTS / "regression/active-02-a"
    if obj == "pear"
    else ROOT / "native-aude" / obj / start
  )
  return home / ("aude-mcc" if arm == "native-aude" else "aude-single-all-mcc")


def mustard_hand(model, *, object_name, **kwargs):
  # Bind the recorded plant body and the same physical fingertip definition used by the other arms.
  kwargs["min_head_fraction"] = HEAD_FRACTION
  hand = FingertipHand(
    model, object_name="mustard" if object_name == "pear" else object_name, **kwargs
  )
  hand.inward_in_palm[:] = (0.0, 0.0, -1.0)
  return hand


def native_aude(arm, spec):
  destination = native_path("mustard", arm)
  if (destination / "run.json").is_file():
    return destination
  destination.parent.mkdir(exist_ok=True, parents=True)
  with TemporaryDirectory(dir=destination.parent) as temp:
    stage = Path(temp)
    copyfile(FIXTURE / "model.mjb", stage / "model.mjb")
    copyfile(ROBOT, stage / "planning-hand.mjb")
    with np.load(FIXTURE / "initial.npz") as f:
      assert spec["fixtures"]["mustard/start-01"]["warm_observations"] == 0
      np.savez(
        stage / "initial.npz",
        integration_state=f["integration_state"],
        warm_policy_observations=np.empty((0, sum(FIELD_WIDTHS)), np.float32),
      )
    protocol = dict(
      read(ACTIVE2 / "protocol.json"),
      seed=spec["cases"]["mustard/start-01"],
      max_s=SECONDS,
    )
    # The mustard fixture starts without pre-roll. Feed frames from the first measured tick.
    with (
      patch.object(active_a, "FingertipHand", mustard_hand),
      patch.object(active_a, "HISTORY_TICKS", 0),
    ):
      receipt = active_a.run_arm((stage, destination.name, protocol))
    if receipt["seconds"] != SECONDS or receipt["stop"] != "time_budget":
      raise ValueError("Incomplete native Aude trial")
    result = stage / destination.name
    receipt.update(
      model_sha256=spec["fixtures"]["mustard/start-01"]["model.mjb"],
      trace_sha256=digest(result / "trace.npz"),
      protocol_sha256=digest(ROOT / "protocol.json"),
    )
    write(result / "run.json", receipt)
    result.replace(destination)
  return destination


def score_trial(obj, arm, family, source):
  start = STARTS[obj]
  seed = seed_for(obj, start) if family == "matched" else None
  dest = ROOT / "evaluation" / family / obj / start
  if seed is not None:
    dest = dest / f"seed-{seed}"
  dest = dest / arm
  quality = source if obj == "pear" else dest
  if obj == "mustard":
    sweep_score.score(source, dest, obj, start, list(information_score.TIMES))
  information_score.evaluate(
    ROOT, family, obj, start, arm, source, quality / "scores.json", seed
  )
  return dest, quality


def acquire(job):
  family, arm = job
  torch.set_num_threads(1)
  obj, start = "mustard", STARTS["mustard"]
  spec = read(ROOT / "protocol.json")
  seed = seed_for(obj, start)
  with (
    patch.object(
      experiment, "scenario", return_value=(FIXTURE, FIXTURE / "surface.obj")
    ),
    patch.object(
      sweep_score, "scenario", return_value=(FIXTURE, FIXTURE / "surface.obj")
    ),
  ):
    if family == "matched":
      with (
        patch.object(matched, "register", return_value=spec),
        patch.object(matched, "OUT", ROOT),
        patch.object(matched, "GuardedWrist", FaceWrist),
        patch.object(matched, "WristWorker", FaceWorker),
      ):
        matched.run(ROOT, arm, seed, obj, start)
      source = matched_path(ROOT, f"{obj}/{start}", arm)
    elif arm.startswith("wrist-"):
      source = native_path(obj, arm)
      code = dict(spec["current_wrist_sources"], controller="FaceWrist")
      with (
        patch.object(runtime, "SweepWrist", FaceWrist),
        patch.object(runtime, "WristWorker", FaceWorker),
        patch.object(runtime, "hashes", return_value=code),
      ):
        runtime.run(source, obj, start, source.name, SECONDS, seed=seed)
    else:
      source = native_aude(arm, spec)
    score_trial(obj, arm, family, source)
    if job == ("native", "wrist-dp"):
      sweep_pair.replay(ROOT, obj, start)
    if job == ("native", "native-aude"):
      reference = sweep_aude_transfer.rollout(ROOT / "baseline-check", obj, start, 1.0)
      with np.load(reference / "trace.npz") as x, np.load(source / "trace.npz") as y:
        for key in ("qpos", "finger_command", "points_m", "pad_valid", "region_load_n"):
          np.testing.assert_array_equal(x[key], y[key][: len(x[key])])
  return job


def run_all(workers):
  jobs = [
    ("native", "wrist-dp"),
    ("native", "native-aude"),
    ("native", "wrist-mcc"),
    ("matched", "aude-dp"),
    ("matched", "wrist-dp"),
    ("native", "aude-single-all"),
  ]
  with ProcessPoolExecutor(max_workers=workers) as pool:
    for result in as_completed([pool.submit(acquire, j) for j in jobs]):
      print("Completed", *result.result(), flush=True)


def audit_trial(obj, source, quality, information, arm, spec, first):
  fixture = spec["fixtures"][f"{obj}/{STARTS[obj]}"]
  r = read(source / "run.json")
  score = read(information / "information.json")
  snap = score["snapshots"]["t240"]
  row = trial_values(source, quality, snap["nats"])
  if (
    row["initial_sha256"] != fixture["integration_state"]
    or row["mesh_sha256"] != fixture["surface.obj"]
    or row["trace_sha256"] != score["spec"]["trace_sha256"]
    or score["spec"]["mesh_sha256"] != fixture["surface.obj"]
    or score["spec"]["evaluator_sha256"] != digest(Path(information_score.__file__))
    or score["initial_cells"] != 4
  ):
    raise ValueError(f"Fixture, GP scorer or physical trace changed in {source}")
  model = (
    r["model_sha256"] if "model_sha256" in r else digest(source.parent / "model.mjb")
  )
  if model != fixture["model.mjb"]:
    raise ValueError(f"Different simulation model: {source}")
  if arm in ("wrist-dp", "aude-dp", "aude-single-all"):
    if (
      r["policy"]["checkpoint_sha256"] != spec["checkpoint_sha256"]
      or r["policy"]["inference_seed"] != spec["cases"][f"{obj}/{STARTS[obj]}"]
    ):
      raise ValueError("Different finger-policy weights or sampling seed")
  with np.load(source / "trace.npz", allow_pickle=False) as tr:
    for key in first:
      np.testing.assert_array_equal(
        tr[key][0], first[key], err_msg=f"{obj}/{arm}/{key}"
      )
    if arm == "wrist-mcc":
      q = tr["finger_nominal"]
      np.testing.assert_array_equal(q, np.broadcast_to(q[0], q.shape))
      assert r["installs"] == 0 and r["policy"] is None
    if arm == "native-aude":
      np.testing.assert_array_equal(tr["finger_nominal"], tr["aude_joint_nominal"])
    p = tr["pad_valid"][:24001]
    assert len(p) == 24001 and p[0].all()
    row["zero_measured_fingertip_s"] = float((~p[1:].any(axis=1)).sum() * 0.01)
    row["last_measured_fingertip_s"] = float(np.flatnonzero(p.any(axis=1))[-1] * 0.01)
  np.testing.assert_allclose(
    [snap["coverage"], snap["light_three_s"], snap["fscore"]],
    [row["coverage"], row["light3"], row["f5"]],
  )
  with np.load(information / "information-curve.npz") as curve:
    row["last_new_cell_s"] = (
      float(curve["increment_tick"][-1] * 0.01)
      if len(curve["increment_tick"])
      else None
    )
  row.update(
    new_cells=snap["new_cells"],
    first_1000=score["snapshots"].get("n1000", {}).get("nats"),
    t70_within_240=row["t70"] if row["t70"] is not None and row["t70"] <= 240 else None,
    score_status=next(
      s["signed"]["status"]
      for s in read(quality / "scores.json")["snapshots"]
      if s["s"] == 240
    ),
  )
  return row


def tape_result(obj, native):
  root = RESULTS if obj == "pear" else ROOT
  source = root / "runs" / obj / STARTS[obj] / "sweep-dp"
  folder = root / "paired-wrist" / obj / STARTS[obj]
  result = read(folder / "result.json")
  if result["dp_trace_sha256"] != digest(source / "trace.npz") or result[
    "code"
  ] != digest(Path(sweep_pair.__file__)):
    raise ValueError("Unexpected wrist replay or controller")
  with (
    np.load(source / "trace.npz") as t,
    np.load(native_path(obj, "wrist-dp") / "trace.npz") as ref,
    np.load(folder / "mcc-replay.npz") as replay,
  ):
    wrist = t["palm_command"][:24001]
    for k in ("qpos", "palm_command", "pad_valid", "points_m", "region_load_n"):
      np.testing.assert_array_equal(t[k][:24001], ref[k][:24001])
    assert sha256(wrist.tobytes()).hexdigest() == result["wrist_sha256"]
    pads = replay["pad_valid"][1:24001]
    light = replay["region_load_n"][1:24001].max(1) < 8
    mcc_s = float(((pads.sum(1) >= 3) & light).sum() * 0.01)
    np.testing.assert_allclose(
      [result["dp_low3"] * 240, result["mcc_low3"] * 240], [native["light3"], mcc_s]
    )
  return dict(
    wrist_sha256=result["wrist_sha256"],
    dp_light3_s=native["light3"],
    mcc_light3_s=mcc_s,
    jointly_moving_s=result["jointly_moving_s"],
    mcc_exact2_s=result["mcc_exact2_s"],
    retained_plus_s=result["retained_plus_s"],
    retained_fraction=result["conditional"],
    source=folder.relative_to(REPO).as_posix(),
  )


def summarize():
  spec = register()
  objects = {}
  for obj, start in STARTS.items():
    native, wrists = {}, {}
    base = ARCHIVE if obj == "pear" else ROOT
    seed = seed_for(obj, start)
    with np.load(native_path(obj, "wrist-dp") / "trace.npz") as t:
      first = {
        key: t[key][0]
        for key in ("qpos", "points_m", "normals_outward", "pad_valid", "region_load_n")
      }
    assert first["pad_valid"].all() and first["region_load_n"].max() < 8
    for arm in NATIVE:
      source = native_path(obj, arm)
      if obj == "pear":
        dest, quality = score_trial(obj, arm, "native", source)
      else:
        dest = quality = ROOT / "evaluation/native" / obj / start / arm
      native[arm] = audit_trial(obj, source, quality, dest, arm, spec, first)
    check_pair(base, obj, start, seed, WRISTS)
    for arm in WRISTS:
      source = matched_path(base, f"{obj}/{start}", arm)
      dest = base / "evaluation" / source.relative_to(base)
      assert read(dest / "information.json")["spec"]["study_protocol_sha256"] == digest(
        base / "protocol.json"
      )
      wrists[arm] = audit_trial(obj, source, dest, dest, arm, spec, first)
    assert_paired({**native, **{"shared-" + a: v for a, v in wrists.items()}})
    common = compare(dest.parent, *WRISTS)
    for arm in WRISTS:
      wrists[arm]["common_motion_nats"] = common["arms"][arm]["nats"]
    objects[obj] = dict(
      start=start,
      seed=seed,
      native=native,
      matched=wrists,
      common_motion=common,
      same_wrist=tape_result(obj, native["wrist-dp"]),
    )
  result = dict(
    protocol_sha256=digest(ROOT / "protocol.json"),
    budget_s=SECONDS,
    unit="one matched starting grasp per object",
    object_split={
      "pear": "finger-policy held-out object",
      "mustard": "finger-policy training object",
    },
    objects=objects,
  )
  write(ROOT / "results.json", result)
  return result


def write_report(result):
  names = {
    "wrist-dp": "我们的腕部 Active＋single_all＋MCC",
    "wrist-mcc": "我们的腕部 Active＋纯 MCC",
    "native-aude": "完整 Aude＋MCC",
    "aude-single-all": "Aude 原生腕部＋single_all＋MCC",
  }
  rows = [
    "# 梨与芥末瓶上的腕部主动探索",
    "",
    "沿用冻结规划器及手指权重。每个物体取一个已有四指轻载初态，逐方法运行相同的 240 仿真秒。梨是手指策略的留出物体，芥末瓶曾用于手指策略训练和手型研究。主实验对象在查看早期曲面结果后选定，两个物体分别报告。",
    "",
  ]
  for obj, label in (("pear", "梨"), ("mustard", "芥末瓶")):
    d = result["objects"][obj]
    rows += [
      f"## {label}",
      "",
      "| 完整系统或手指消融 | 实触覆盖 | F@5 mm | 轻载至少三指 | 腕动时同三指连续 ≥1 s | 累计信息 | 任一区域 ≥8 N | 240 s 内首达 70% |",
      "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in NATIVE:
      x = d["native"][arm]
      t = "未达" if x["t70_within_240"] is None else f"{x['t70_within_240']:.2f} s"
      rows.append(
        f"| {names[arm]} | {x['coverage']:.2%} | {x['f5']:.2%} | {x['light3']:.2f} s | {x['continuous']:.2f} s | {x['nats']:,.1f} nat | {x['over8']:.2f} s | {t} |"
      )
    r = d["same_wrist"]
    rows += [
      "",
      f"给纯 MCC 逐帧重放本方 single_all 闭环生成的同一串腕部指令时，single_all 与纯 MCC 的轻载三指时长为 {r['dp_light3_s']:.2f} s 和 {r['mcc_light3_s']:.2f} s。这条腕轨由前者的触觉反馈形成。",
      "",
      "| 同一 single_all＋MCC 执行器上的腕规划器 | 240 s 累计信息 | 新触点格数 | 相同实际平移与转角上限 | 前 1,000 个新触点格 |",
      "|---|---:|---:|---:|---:|",
    ]
    for arm, label in zip(
      WRISTS, ("我们的 GPIS 腕规划器", "Aude 提供腕部位姿"), strict=True
    ):
      x = d["matched"][arm]
      fixed = "未达千格" if x["first_1000"] is None else f"{x['first_1000']:,.1f} nat"
      rows.append(
        f"| {label} | {x['nats']:,.1f} nat | {x['new_cells']:,} | {x['common_motion_nats']:,.1f} nat | {fixed} |"
      )
    rows.append("")
  rows += [
    "覆盖按有效触点的 10 mm 测地邻域并集计分，重建使用 5 mm 容差的有符号 GPIS F-score。轻载要求五个分区均低于 8 N，有效指尖力至少为 0.2 N。信息以最初四个轻载触点为条件，每个 4.5 mm 格只计首次新观测。",
    "",
    "[实验协议](protocol.json) · [逐物体数据、来源与轨迹校验](results.json)",
  ]
  (ROOT / "REPORT.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("action", choices=("run", "register", "report"))
  p.add_argument("--workers", type=int, default=4)
  args = p.parse_args()
  register()
  if args.action == "run":
    run_all(args.workers)
  if args.action in ("run", "report"):
    write_report(summarize())
