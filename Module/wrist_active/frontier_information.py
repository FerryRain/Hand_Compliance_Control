"""Score information gained on every physical control tick of the paired YCB release."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import information_score as evaluator
from .experiment import digest
from .face_regression_suite import REPO, read
from .frontier_study import ROOT
from .ycb_fixtures import OBJECTS

ARMS = {"wrist-dp": "腕部 Active", "native-aude": "完整 Aude"}
PERIODS = (1, 10, 30, 60, 120, 240, 600)
DEFAULT_EARLY = 60


def window_stats(step, dt):
  result = {}
  for seconds in PERIODS:
    count = round(seconds / dt)
    values = step[1 : count + 1]
    result[str(seconds)] = {
      "steps": count,
      "total_nat": float(values.sum()),
      "mean_nat_per_step": float(values.mean()),
    }
  return result


def score_run(study, obj, start, arm, row, receipt):
  seconds, dt = receipt["budget_s"], receipt["control_dt_s"]
  end = round(seconds / dt)
  kind = {"wrist-dp": "ours", "native-aude": "aude"}[arm]
  previous = study / "evaluation" / "native" / obj / start / kind
  archive = read(previous / "information.json")
  spec = archive["spec"]
  raw = REPO / row["source"] / "trace.npz"
  if not (
    digest(raw) == row["trace_sha256"] == spec["trace_sha256"]
    and receipt["protocol_sha256"] == spec["study_protocol_sha256"]
    and digest(Path(evaluator.__file__)) == spec["evaluator_sha256"]
    and (archive["obj"], archive["start"], archive["arm"]) == (obj, start, kind)
    and spec["primary_length_m"] == evaluator.LENGTHS_M[0]
    and spec["gp_noise_variance"] == evaluator.NOISE_VARIANCE
  ):
    raise ValueError(f"Different trajectory or GP scoring protocol for {obj}/{arm}")
  # Keep the exact registered 240 s contact filter, extending only its horizon.
  with (
    np.load(raw, allow_pickle=False) as trace,
    patch.object(evaluator, "SECONDS", seconds),
  ):
    if len(trace["pad_valid"]) != end + 1:
      raise ValueError(f"Incomplete physical trajectory for {obj}/{arm}")
    np.testing.assert_array_equal(trace["points_m"][0], archive["initial_points"])
    points, ticks = evaluator.measured_cells(trace)
  initial = int(np.searchsorted(ticks, 0, side="right"))
  if (
    initial != 4
    or initial != archive["initial_cells"]
    or len(ticks) != row["windows"]["600"]["cells"]
  ):
    raise ValueError(f"Incorrect initial or final contact count for {obj}/{arm}")
  prefix, _ = evaluator.gp_prefix(
    points, spec["primary_length_m"], spec["gp_noise_variance"]
  )
  step = np.bincount(
    ticks[initial:], weights=np.diff(prefix)[initial:], minlength=end + 1
  )
  if step[0] != 0 or not np.isfinite(step).all() or np.min(step) < 0:
    raise ValueError(f"Invalid per-step information for {obj}/{arm}")
  total = np.cumsum(step)
  np.testing.assert_allclose(
    total[-1], prefix[-1] - prefix[initial], atol=1e-8, rtol=1e-11
  )
  # Test every archived tick, including no-contact ticks and simultaneous arrivals.
  with np.load(previous / "information-curve.npz", allow_pickle=False) as old:
    old_total = old["cumulative_nats"]
    if len(old_total) != round(240 / dt) + 1:
      raise ValueError("Unexpected reference horizon")
    np.testing.assert_allclose(
      total[: len(old_total)], old_total, atol=2e-8, rtol=1e-11
    )
    old_step = np.bincount(
      old["increment_tick"], weights=old["increment_nats"], minlength=len(old_total)
    )
    np.testing.assert_allclose(step[: len(old_step)], old_step, atol=2e-8, rtol=1e-11)
  np.testing.assert_allclose(total[round(240 / dt)], row["nats"], atol=2e-8, rtol=1e-11)
  nonzero = np.flatnonzero(step)
  return step, {
    "trace_sha256": row["trace_sha256"],
    "initial_cells": initial,
    "new_cells_600": len(points) - initial,
    "tick": nonzero.tolist(),
    "gain_nat": step[nonzero].tolist(),
    "windows": window_stats(step, dt),
  }


def score(study):
  receipt = read(study / "results.json")
  dt, seconds = receipt["control_dt_s"], receipt["budget_s"]
  if dt != evaluator.DT or seconds != 600 or set(receipt["objects"]) != set(OBJECTS):
    raise ValueError("Expected the five-object 600 s paired release at 100 Hz")
  if digest(study / "protocol.json") != receipt["protocol_sha256"]:
    raise ValueError("Registered physical protocol changed")
  output = {
    "study_sha256": digest(study / "results.json"),
    "protocol_sha256": receipt["protocol_sha256"],
    "evaluator_sha256": digest(Path(evaluator.__file__)),
    "dt_s": dt,
    "duration_s": seconds,
    "default_early_s": DEFAULT_EARLY,
    "kernel": "unit-variance Matern 3/2",
    "length_m": evaluator.LENGTHS_M[0],
    "noise_variance": evaluator.NOISE_VARIANCE,
    "voxel_m": evaluator.VOXEL_M,
    "region_limit_n": evaluator.LOAD_LIMIT,
    "fingertip_min_n": 0.2,
    "encoding": "Increment in nat at each listed physical tick. All omitted ticks, including tick 0, have zero gain. The common initial four cells are conditioned on. Multiple new cells at a tick are summed.",
    "macro_definition": "Arithmetic mean of the five paired object traces at the same control tick.",
    "arms": ARMS,
    "objects": {},
    "macro": {},
  }
  sums = {key: np.zeros(round(seconds / dt) + 1) for key in ARMS}
  for obj, item in receipt["objects"].items():
    output["objects"][obj] = pair = {
      "name": OBJECTS[obj][1],
      "start": item["start"],
      "arms": {},
    }
    for arm in ARMS:
      step, series = score_run(
        study, obj, item["start"], arm, item["native"][arm], receipt
      )
      pair["arms"][arm] = series
      sums[arm] += step
      print(
        f"{obj}/{arm} · 60 s {series['windows']['60']['mean_nat_per_step']:.5f} nat/step · 600 s {step.sum():.2f} nat",
        flush=True,
      )
  for arm, values in sums.items():
    output["macro"][arm] = window_stats(values / len(receipt["objects"]), dt)
  directory = study / "information"
  directory.mkdir(exist_ok=True)
  path = directory / "gain.json"
  staged = path.with_suffix(".json.tmp")
  staged.write_text(
    json.dumps(output, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
    encoding="utf-8",
  )
  staged.replace(path)
  return output


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--study", type=Path, default=ROOT / "validated")
  score(parser.parse_args().study)
