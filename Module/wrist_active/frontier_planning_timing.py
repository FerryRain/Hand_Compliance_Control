"""Measure the two released complete systems serially, without changing their commands."""

import argparse
import json
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter, perf_counter_ns
from unittest.mock import patch

import numpy as np
import torch

from ..acitive_02 import active_a as aude
from . import evaluate, frontier_study, sweep_aude_transfer, ycb_native
from . import sweep_experiment as runtime
from .experiment import digest
from .face_regression_suite import REPO, read, write
from .frontier_wrist import FrontierWorker
from .sweep import DT, PLAN_TICKS
from .ycb_fixtures import OBJECTS

STUDY = frontier_study.ROOT / "validated"
OUT = STUDY / "planning-timing"
SAMPLES = 60000


class AudeTimers:
  """Time both planner returns; assign the outer proposal only when recomputed."""

  def __init__(self):
    self.ticks = []
    self.local_ms = []
    self.proposal_ms = []
    self.pending = None

  def attach(self, stack):
    propose = aude.AudeExplorer.plan
    local = aude.ActiveSurfacePlanner.plan

    def timed_proposal(*args):
      if self.pending is not None:
        raise ValueError("Unconsumed joint proposal")
      began = perf_counter_ns()
      result = propose(*args)
      self.pending = (perf_counter_ns() - began) * 1e-6
      return result

    def timed_local(planner, live, proposal, touch, time_s):
      tick = round(time_s / DT)
      if (tick % aude.PLAN_TICKS == 0) != (self.pending is not None):
        raise ValueError("Unexpected global/local plan ordering")
      began = perf_counter_ns()
      result = local(planner, live, proposal, touch, time_s)
      self.local_ms.append((perf_counter_ns() - began) * 1e-6)
      self.ticks.append(tick)
      self.proposal_ms.append(self.pending or 0.0)
      self.pending = None
      return result

    stack.enter_context(patch.object(aude.AudeExplorer, "plan", timed_proposal))
    stack.enter_context(patch.object(aude.ActiveSurfacePlanner, "plan", timed_local))


def final_mesh(trace, path):
  """Time retrieval of measured touches to final signed mesh export, without truth scoring."""
  began = perf_counter()
  with np.load(trace, allow_pickle=False) as data:
    valid = data["pad_valid"] & (data["region_load_n"].max(axis=1) < 8)[:, None]
    points = data["points_m"][valid]
    normals = data["normals_outward"][valid]
  ids = evaluate.ridge.voxel_representatives(points)
  if len(ids) > evaluate.ridge.CONTACT_BUDGET:
    ids = ids[
      evaluate.ridge.farthest_indices(points[ids], evaluate.ridge.CONTACT_BUDGET)
    ]
  mesh, _ = evaluate.signed_gpis(points[ids], normals[ids])
  mesh.export(path)
  return perf_counter() - began


def verified_replay(reference, replay):
  with (
    np.load(reference, allow_pickle=False) as a,
    np.load(replay, allow_pickle=False) as b,
  ):
    if set(a.files) != set(b.files) or len(b["qpos"]) != SAMPLES + 1:
      raise ValueError("Incomplete or incompatible native replay")
    fields = sorted(set(a.files) - {"wall_s"})
    for key in fields:
      np.testing.assert_array_equal(a[key], b[key], err_msg=f"Trace mismatch: {key}")
  return fields


def record(obj, arm, results, spec):
  key = {"ours": "wrist-dp", "aude": "native-aude"}[arm]
  native = results["objects"][obj]["native"][key]
  original = REPO / native["source"]
  release = native["trace_sha256"]
  output = OUT / obj / arm
  marker = output / "measurement.json"
  signature = {
    "protocol": results["protocol_sha256"],
    "trace": release,
    "timing_program": digest(Path(__file__)),
    "mode": "serial-unpaced",
  }
  if marker.exists():
    saved = read(marker)
    if saved["signature"] != signature or saved["samples_sha256"] != digest(
      output / "events.npz"
    ):
      raise ValueError(f"Different measurement already exists in {output}")
    return
  if digest(original / "trace.npz") != release:
    raise ValueError(f"Registered trace mismatch in {original}")
  output.mkdir(parents=True, exist_ok=True)
  print(
    f"Starting {obj}/{arm}: 600 simulated seconds, one rollout at a time", flush=True
  )
  stamps = {}
  fixture_owner = runtime if arm == "ours" else sweep_aude_transfer
  fixture = fixture_owner.fixture
  begin = ycb_native.Clock.begin

  def start_fixture(*args, **kwargs):
    stamps["fixture"] = perf_counter()
    return fixture(*args, **kwargs)

  def start_clock(clock):
    begin(clock)
    stamps["planning"] = clock.start

  times = AudeTimers()
  with TemporaryDirectory(dir=output) as tmp:
    stage = Path(tmp) / "replay"
    with ExitStack() as stack:
      stack.enter_context(patch.object(fixture_owner, "fixture", start_fixture))
      stack.enter_context(patch.object(ycb_native.Clock, "begin", start_clock))
      stack.enter_context(patch.object(ycb_native, "destination", return_value=stage))
      stack.enter_context(patch.object(ycb_native, "FaceWorker", FrontierWorker))
      stack.enter_context(
        patch.object(
          ycb_native, "run_wrist", partial(frontier_study.run_wrist, spec=spec)
        )
      )
      if arm == "aude":
        times.attach(stack)
      ycb_native.timed_trial(obj, arm, spec["code"]["frontier_source"], spec)

    trace = stage / "trace.npz"
    matched = verified_replay(original / "trace.npz", trace)
    if arm == "ours":
      plans = [
        json.loads(line) for line in (stage / "plans.jsonl").read_bytes().splitlines()
      ]
      ticks = np.array([p["apply_tick"] for p in plans])
      local_ms = np.array([p["planning_ms"] for p in plans])
      with np.load(stage / "control-timing.npz") as timing:
        np.testing.assert_array_equal(local_ms, timing["plan_ms"])
      proposal_ms = np.zeros_like(local_ms)
      period = PLAN_TICKS
    else:
      ticks = np.asarray(times.ticks)
      local_ms = np.asarray(times.local_ms)
      proposal_ms = np.asarray(times.proposal_ms)
      period = aude.REPLAN_TICKS
      np.testing.assert_array_equal(
        ticks[proposal_ms > 0], np.arange(0, SAMPLES, aude.PLAN_TICKS)
      )
    np.testing.assert_array_equal(ticks, np.arange(0, SAMPLES, period))
    if not (
      np.isfinite(local_ms).all()
      and np.all(local_ms > 0)
      and np.isfinite(proposal_ms).all()
    ):
      raise ValueError("Invalid planner duration")
    with np.load(stage / "native-clock.npz") as clock:
      elapsed = clock["first_plan_to_post_step_s"]
    if (
      len(elapsed) != SAMPLES + 1 or elapsed[0] != 0 or not np.all(np.diff(elapsed) > 0)
    ):
      raise ValueError("Invalid full-system elapsed time")
    mesh_path = Path(tmp) / "final.ply"
    mesh_s = final_mesh(trace, mesh_path)
    mesh_hash = digest(mesh_path)
    if mesh_hash != digest(original / "signed-600s.ply"):
      raise ValueError(f"Final signed mesh differs for {obj}/{arm}")
    np.savez_compressed(
      Path(tmp) / "events.npz",
      apply_tick=ticks,
      local_ms=local_ms,
      proposal_ms=proposal_ms,
      online_elapsed_s=elapsed,
    )
    (Path(tmp) / "events.npz").replace(output / "events.npz")
    setup = stamps["planning"] - stamps["fixture"]
    if setup < 0:
      raise ValueError("Negative setup time")
    result = dict(
      signature=signature,
      obj=obj,
      arm=key,
      matched_frames=len(elapsed),
      matched_fields=matched,
      final_mesh_sha256=mesh_hash,
      init_s=setup,
      online_240_s=float(elapsed[24000]),
      online_600_s=float(elapsed[-1]),
      final_mesh_s=mesh_s,
      init_online_final_mesh_s=setup + elapsed[-1] + mesh_s,
      samples_sha256=digest(output / "events.npz"),
    )
    write(marker, result)
    print(
      f"Verified {obj}/{arm}, {len(ticks)} plans, {elapsed[-1]:.2f}s online, "
      f"{result['init_online_final_mesh_s']:.2f}s to mesh",
      flush=True,
    )


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("objects", nargs="*")
  args = parser.parse_args()
  selected = args.objects or list(OBJECTS)
  if set(selected) - OBJECTS.keys():
    parser.error("Unknown object")
  output = read(STUDY / "results.json")
  protocol = frontier_study.register(STUDY)
  if output["control_dt_s"] != DT or output["budget_s"] != SAMPLES * DT:
    raise ValueError("Unexpected physical budget")
  if digest(STUDY / "protocol.json") != output["protocol_sha256"]:
    raise ValueError("Protocol digest mismatch")
  torch.set_num_threads(1)
  for index, obj in enumerate(OBJECTS):
    if obj in selected:
      for arm in ("aude", "ours") if index % 2 == 0 else ("ours", "aude"):
        record(obj, arm, output, protocol)
