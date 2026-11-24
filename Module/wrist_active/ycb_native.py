"""Run and time the two complete Active systems against the same YCB fixtures."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from unittest.mock import patch

import numpy as np
import torch

from ..acitive_02 import active_a
from . import experiment, information_score, sweep_aude_transfer, sweep_score
from . import sweep_experiment as runtime
from .experiment import digest, seed_for
from .face_regression_suite import REPO, read, write
from .face_wrist import FaceWorker
from .pear_mustard import STARTS, native_path
from .sweep import DT
from .ycb_active import ROOT, register, register_footprints, release, run_wrist
from .ycb_fixtures import OBJECTS, STUDY, fixture, prepare

SECONDS = 600
TIMES = [60.0, 120.0, 240.0, float(SECONDS)]
ORDER = (("pear", "aude"), ("pear", "ours"), ("mustard", "ours"), ("mustard", "aude"))
NAMES = {"ours": "腕部 Active＋single_all＋MCC", "aude": "完整 Aude＋MCC"}


def destination(obj, arm):
  return ROOT / "native" / obj / arm


def inputs(obj):
  return fixture(obj)


class Clock:
  def __init__(self, obj, arm):
    self.label = f"{obj} {arm}"
    self.start = None
    self.post_step = [0.0]

  def begin(self):
    if self.start is None:
      self.start = perf_counter()

  def measured(self, f):
    def step(*args):
      value = f(*args)
      if self.start is not None:
        self.post_step.append(perf_counter() - self.start)
        tick = len(self.post_step) - 1
        if tick % 12000 == 0:
          print(self.label, f"{tick * DT:g} simulation seconds", flush=True)
      return value

    return step


def timed_trial(obj, arm, chosen, spec):
  start = OBJECTS[obj][2]
  target = destination(obj, arm)
  receipt = target / "clock.json"
  if receipt.exists():
    saved = read(receipt)
    if (
      saved["release"] != (chosen if arm == "ours" else "active-02-a")
      or saved["seconds"] != SECONDS
      or saved["trace_sha256"] != digest(target / "trace.npz")
      or saved["clock_sha256"] != digest(target / "native-clock.npz")
    ):
      raise ValueError(f"Existing run differs in {target}")
    return
  target.parent.mkdir(parents=True, exist_ok=True)
  timer = Clock(obj, arm)
  with TemporaryDirectory(dir=target.parent) as staging:
    stage = Path(staging)
    if arm == "ours":

      class TimedWorker(FaceWorker):
        def submit(self, *args):
          timer.begin()
          return super().submit(*args)

      with patch.object(runtime, "robot_state", timer.measured(runtime.robot_state)):
        trial = run_wrist(
          stage / "trial",
          obj,
          chosen,
          seconds=SECONDS,
          quality=False,
          worker_type=TimedWorker,
          start=start,
          fixture_paths=inputs(obj),
          fixture_spec=spec["fixtures"][f"{obj}/{start}"],
        )
    else:
      original = active_a.AudeExplorer.plan

      def plan(explorer, *args):
        timer.begin()
        return original(explorer, *args)

      with (
        patch.object(experiment, "scenario", return_value=inputs(obj)),
        patch.object(active_a.AudeExplorer, "plan", plan),
        patch.object(active_a, "robot_state", timer.measured(active_a.robot_state)),
      ):
        trial = sweep_aude_transfer.rollout(stage, obj, start, float(SECONDS))
    stamps = np.asarray(timer.post_step)
    if len(stamps) != round(SECONDS / DT) + 1 or not np.all(np.diff(stamps) > 0):
      raise ValueError("Physical ticks and native wall-clock ticks did not agree")
    np.savez_compressed(trial / "native-clock.npz", first_plan_to_post_step_s=stamps)
    receipt_data = dict(
      obj=obj,
      arm=arm,
      release=chosen if arm == "ours" else "active-02-a",
      seconds=SECONDS,
      trace_sha256=digest(trial / "trace.npz"),
      clock_sha256=digest(trial / "native-clock.npz"),
      first_240_wall_s=float(stamps[24000]),
      all_wall_s=float(stamps[-1]),
    )
    write(trial / "clock.json", receipt_data)
    trial.rename(target)
  print(
    obj,
    arm,
    "240 s acquisition wall time",
    receipt_data["first_240_wall_s"],
    flush=True,
  )


def audit_pair(obj, spec):
  start = OBJECTS[obj][2]
  ours, other = (destination(obj, a) for a in ("ours", "aude"))
  fi = spec["fixtures"][obj + "/" + start]
  for folder in (ours, other):
    r = read(folder / "run.json")
    if (
      r["initial_sha256"] != fi["integration_state"]
      or r["model_sha256"] != fi["model.mjb"]
      or r["seed"] != seed_for(obj, start)
      or r["trace_sha256"] != digest(folder / "trace.npz")
    ):
      raise ValueError("Unequal starting conditions or trace hash")
  with (
    np.load(ours / "trace.npz") as x,
    np.load(other / "trace.npz") as y,
  ):
    for name in ("qpos", "points_m", "normals_outward", "pad_valid", "region_load_n"):
      np.testing.assert_array_equal(x[name][0], y[name][0])
    if obj in STARTS:
      with np.load(native_path(obj, "native-aude") / "trace.npz") as previous:
        for name in (
          "qpos",
          "points_m",
          "normals_outward",
          "pad_valid",
          "region_load_n",
          "finger_nominal",
          "finger_command",
        ):
          np.testing.assert_array_equal(y[name][:24001], previous[name][:24001])
    np.testing.assert_array_equal(y["finger_nominal"], y["aude_joint_nominal"])
    if not np.all(x["pad_valid"][0]) or len(x["qpos"]) != len(y["qpos"]):
      raise ValueError("Unequal physical test window")


def summarize(obj, arm, spec, study):
  folder = destination(obj, arm)
  start = OBJECTS[obj][2]
  with patch.object(sweep_score, "scenario", return_value=inputs(obj)):
    score = sweep_score.score(folder, folder, obj, start, TIMES)
    if (
      score["spec"]["mesh_sha256"] != spec["fixtures"][f"{obj}/{start}"]["surface.obj"]
    ):
      raise ValueError("Different offline reference mesh")
    information_score.evaluate(
      study, "native", obj, start, arm, folder, folder / "scores.json"
    )
  info = read(study / "evaluation/native" / obj / start / arm / "information.json")[
    "snapshots"
  ]["t240"]
  with (
    np.load(folder / "coverage.npz") as coverage,
    np.load(folder / "native-clock.npz") as timing,
  ):
    curve = coverage["c10"]
    clock = timing["first_plan_to_post_step_s"]
    if len(curve) != len(clock) or not np.isfinite(curve).all():
      raise ValueError("Coverage and time differ")
    thresholds, clocks = {}, {}
    for target in spec["thresholds"]:
      first = int(np.searchsorted(curve, target / 100.0))
      thresholds[str(target)] = first * DT if first < len(curve) else None
      clocks[str(target)] = float(clock[first]) if first < len(curve) else None
    # Display the exact first tick at each threshold and one sample per simulation second for plotting.
    ticks = np.unique(
      np.r_[
        np.arange(0, len(curve), 100),
        [round(v / DT) for v in thresholds.values() if v is not None],
      ]
    ).astype(int)
    series = {"s": (ticks * DT).round(2).tolist(), "coverage": curve[ticks].tolist()}
  snap = next(v for v in score["snapshots"] if v["s"] == 240.0)
  with np.load(folder / "trace.npz") as trajectory:
    pads = trajectory["pad_valid"][1:24001]
    loads = trajectory["region_load_n"][1:24001].max(axis=1)
    zero_pad_s = float((~pads.any(axis=1)).sum() * DT)
    no_light_pad_s = float((~(pads.any(axis=1) & (loads < 8.0))).sum() * DT)
  curve_path = study / "evaluation/native" / obj / start / arm / "information-curve.npz"
  with np.load(curve_path) as gains:
    first = gains["increment_tick"]
    last_new_cell_s = float(first[-1] * DT) if len(first) else None
  row = dict(
    coverage=snap["coverage"],
    mean_coverage=snap["mean_coverage"],
    light3=snap["low3_s"],
    f5=snap["signed"]["fscore"],
    over8=snap["over8_s"],
    peak_n=snap["peak_n"],
    zero_measured_fingertip_s=zero_pad_s,
    no_light_fingertip_s=no_light_pad_s,
    last_new_cell_s=last_new_cell_s,
    new_cells=info["new_cells"],
    nats=info["nats"],
    coverage_600=score["snapshots"][-1]["coverage"],
    thresholds_s=thresholds,
    wall_thresholds_s=clocks,
    wall_s_240=float(clock[24000]),
    wall_s_600=float(clock[-1]),
    trace_sha256=score["spec"]["trace_sha256"],
    scored_at=folder.relative_to(REPO).as_posix(),
    source=folder.relative_to(REPO).as_posix(),
    curve=series,
  )
  if "timing" in score:
    row["timing"] = score["timing"]
  return row


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "action", choices=("record", "score", "all", "five", "five-record", "five-score")
  )
  args = parser.parse_args()
  torch.set_num_threads(1)
  spec = register()
  register_footprints()
  chosen = release()
  five = args.action.startswith("five")
  study = STUDY if five else ROOT
  starts = {obj: row[2] for obj, row in OBJECTS.items()} if five else STARTS
  if five:
    spec = prepare()
  order = list(ORDER)
  if five:
    for i, obj in enumerate(("cracker", "bleach", "coffee")):
      order.extend(
        (obj, arm) for arm in (("aude", "ours") if i % 2 == 0 else ("ours", "aude"))
      )
  if args.action in ("record", "all", "five", "five-record"):
    for obj, arm in order:
      timed_trial(obj, arm, chosen, spec)
  if args.action in ("score", "all", "five", "five-score"):
    result = dict(
      chosen=chosen,
      arms=NAMES,
      budget_s=SECONDS,
      primary_budget_s=240,
      snapshots_s=TIMES,
      primary=spec["primary"],
      release_sha256=digest(ROOT / "release.json"),
      score_radius_mm=10,
      limit_n=8,
      control_dt_s=DT,
      checkpoint_sha256=spec["checkpoint"],
      time_origin="First call to the native planner through acquired physics tick, including finger policy, control and simulation. Excludes setup, trace compression and offline scoring. Unpaced, serial order, one wall-clock sample per system and object.",
      scope=spec.get(
        "scope",
        "One original grasp per object. Revision selected on mustard. Both starts have previously been evaluated.",
      ),
      protocol_sha256=digest(study / "protocol.json"),
      objects={},
    )
    for obj, start in starts.items():
      audit_pair(obj, spec)
      arms = {a: summarize(obj, a, spec, study) for a in NAMES}
      result["objects"][obj] = dict(
        start=start,
        seed=seed_for(obj, start),
        initial_sha256=spec["fixtures"][f"{obj}/{start}"]["integration_state"],
        native={"wrist-dp": arms["ours"], "native-aude": arms["aude"]},
      )
      print(
        obj,
        {
          a: (v["coverage"], v["thresholds_s"]["30"], v["mean_coverage"])
          for a, v in arms.items()
        },
        flush=True,
      )
    if five:
      result["macro"] = {}
      for arm in ("wrist-dp", "native-aude"):
        rows = [item["native"][arm] for item in result["objects"].values()]
        macro = {
          k: float(np.mean([r[k] for r in rows]))
          for k in (
            "coverage",
            "mean_coverage",
            "light3",
            "f5",
            "nats",
            "coverage_600",
            "wall_s_240",
            "over8",
          )
        }
        macro["thresholds"] = {}
        for threshold in spec["thresholds"]:
          values = [r["thresholds_s"][str(threshold)] for r in rows]
          macro["thresholds"][str(threshold)] = {
            "n": len(rows),
            "hit": sum(v is not None for v in values),
            "mean_restricted_s": float(
              np.mean([SECONDS if v is None else v for v in values])
            ),
          }
        result["macro"][arm] = macro
    write(study / "results.json", result)


if __name__ == "__main__":
  main()
