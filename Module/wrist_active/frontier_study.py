"""Acquire paired, complete-system five-object experiments and audit their inputs."""

import argparse
from functools import partial
from pathlib import Path
from shutil import copyfile
from unittest.mock import patch

import numpy as np
import torch

from . import experiment, frontier_wrist, sweep_score, ycb_native
from . import sweep_experiment as runtime
from .experiment import digest, seed_for
from .face_regression_suite import REPO, read, write
from .frontier_wrist import FrontierWorker, FrontierWrist
from .ycb_fixtures import OBJECTS, fixture
from .ycb_fixtures import STUDY as PREVIOUS

ROOT = runtime.GENERATED / "ycb_frontier_active_20261123"


def register(study):
  source = read(PREVIOUS / "protocol.json")
  previous = read(PREVIOUS / "results.json")
  code = dict(
    read(
      REPO / previous["objects"]["pear"]["native"]["wrist-dp"]["source"] / "run.json"
    )["signature"]["code"]
  )
  code.update(
    controller="FrontierWrist",
    config="shared",
    frontier_source=digest(Path(frontier_wrist.__file__)),
  )
  for filename, expected in code.items():
    if (
      filename.endswith(".py")
      and digest(Path(__file__).with_name(filename)) != expected
    ):
      raise ValueError(f"Underlying controller dependency changed: {filename}")
  for filename, key in [
    ("contact_center.py", "revision"),
    ("footprint_wrist.py", "footprint_source"),
  ]:
    if digest(Path(__file__).with_name(filename)) != code[key]:
      raise ValueError(f"Underlying wrist implementation changed: {filename}")
  spec = dict(
    source,
    code=code,
    controller="FrontierWrist",
    primary_budget_s=600,
    release_sha256=code["frontier_source"],
    previous_protocol_sha256=digest(PREVIOUS / "protocol.json"),
    primary="600 simulation seconds, equal-weight mean of five whole-surface 10 mm measured coverage fractions.",
    comparison="Complete wrist Active + single_all + MCC and complete Active-02 Aude + MCC.",
    development="Coffee can and bleach bottle guided development. A pear regression informed the unsupported-footprint speed safeguard.",
    scope="One paired original grasp per object. Five objects share identical planner parameters and finger weights; these starts were previously evaluated.",
  )
  path = study / "protocol.json"
  if path.exists() and read(path) != spec:
    raise ValueError(
      "Controller or inputs changed after registration; use a new experiment directory"
    )
  if not path.exists():
    write(path, spec)
  snapshot = study / "wrist-source.py"
  if not snapshot.exists():
    copyfile(frontier_wrist.__file__, snapshot)
  if digest(snapshot) != code["frontier_source"]:
    raise ValueError("Archived wrist source differs from its registration")
  return spec


def run_wrist(
  folder,
  obj,
  chosen,
  *,
  seconds,
  worker_type,
  start,
  fixture_paths,
  fixture_spec,
  **kwargs,
):
  spec = kwargs["spec"]
  with (
    patch.object(experiment, "scenario", return_value=fixture_paths),
    patch.object(runtime, "SweepWrist", FrontierWrist),
    patch.object(runtime, "WristWorker", worker_type),
    patch.object(runtime, "hashes", return_value=spec["code"]),
  ):
    run = runtime.run(
      folder, obj, start, "sweep-dp", seconds, seed=seed_for(obj, start)
    )
  if (
    run["initial_sha256"],
    run["model_sha256"],
    run["offline_mesh_sha256"],
    run["policy"]["checkpoint_sha256"],
  ) != (
    fixture_spec["integration_state"],
    fixture_spec["model.mjb"],
    fixture_spec["surface.obj"],
    spec["checkpoint"],
  ):
    raise ValueError("Model, start or finger weights differ")
  return folder


def loads(folder, end):
  with np.load(folder / "trace.npz") as tr:
    regions = tr["region_load_n"][1 : 100 * end + 1]
    palm = regions[:, 4]
    count = tr["pad_valid"][1 : 100 * end + 1].sum(axis=1)
    empty = count == 0
    light = regions.max(axis=1) < 8
    clear = np.flatnonzero(palm < 8)
    edges = np.flatnonzero(np.diff(np.r_[False, empty, False])).reshape(-1, 2)
    return dict(
      palm_peak_n=float(palm.max()),
      palm_over8_s=float((palm >= 8).sum() / 100),
      zero_tip_s=float(empty.sum() / 100),
      light1_s=float((light & (count >= 1)).sum() / 100),
      light2_s=float((light & (count >= 2)).sum() / 100),
      longest_zero_tip_s=float(np.diff(edges, axis=1).max(initial=0) / 100),
      final_palm_over8_s=float(
        (len(palm) - 1 - clear[-1]) / 100 if len(clear) else end
      ),
    )


def pilot_score(folder, obj):
  with patch.object(sweep_score, "scenario", return_value=fixture(obj)):
    score = sweep_score.score(
      folder, folder, obj, OBJECTS[obj][2], [240.0, 600.0], fast=True
    )
  print(
    obj,
    [
      (s["coverage"], s["low3_s"], loads(folder, int(s["s"])))
      for s in score["snapshots"]
    ],
    flush=True,
  )


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("stage")
  parser.add_argument(
    "action",
    choices=["register", "pilot", "record", "score", "record-aude", "record-ours"],
  )
  parser.add_argument("objects", nargs="*")
  args = parser.parse_args()
  if not args.stage.replace("-", "").isalnum():
    raise ValueError("Invalid stage")
  study = ROOT / args.stage
  study.mkdir(parents=True, exist_ok=True)
  spec = read(study / "protocol.json") if args.action == "score" else register(study)
  if args.action == "register":
    return
  objects = args.objects or list(OBJECTS)
  if not set(objects) <= OBJECTS.keys():
    raise ValueError(objects)

  def dest(obj, arm):
    if arm == "aude":
      return ROOT / "aude-native" / obj / arm
    return study / "native" / obj / arm

  torch.set_num_threads(1)
  with (
    patch.object(ycb_native, "destination", dest),
    patch.object(ycb_native, "FaceWorker", FrontierWorker),
    patch.object(ycb_native, "run_wrist", partial(run_wrist, spec=spec)),
  ):
    if args.action == "score":
      from .frontier_report import score_release

      score_release(study, spec, objects)
      return
    for i, obj in enumerate(objects):
      arms = (
        ["ours"]
        if args.action == "pilot"
        else (["aude", "ours"] if i % 2 == 0 else ["ours", "aude"])
      )
      if args.action.startswith("record-"):
        arms = [args.action.removeprefix("record-")]
      for arm in arms:
        ycb_native.timed_trial(obj, arm, spec["code"]["frontier_source"], spec)
      if args.action == "pilot":
        pilot_score(dest(obj, "ours"), obj)


if __name__ == "__main__":
  main()
