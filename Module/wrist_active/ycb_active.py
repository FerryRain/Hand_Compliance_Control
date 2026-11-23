"""Compare the deployed wrist-only and complete Aude stacks on pear and mustard."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from unittest.mock import patch

import torch

from . import contact_center, experiment, footprint_wrist, sweep_score
from . import sweep_experiment as runtime
from .__main__ import sources
from .contact_center import ContactCenterWrist
from .experiment import digest, seed_for
from .face_regression_suite import read, write
from .face_wrist import FaceWorker, FaceWrist
from .footprint_wrist import FootprintWrist
from .pear_mustard import FIXTURE, STARTS
from .pear_mustard import ROOT as BASE

ROOT = runtime.GENERATED / "ycb_active_efficiency_20261122"
CONFIGS = {
  "center": dict(center=True, free_depth=False),
  "depth": dict(center=False, free_depth=True),
  "center-depth": dict(center=True, free_depth=True),
}
FOOTPRINTS = {
  "footprint": dict(center=False, free_depth=True),
  "center-footprint": dict(center=True, free_depth=True),
}


def register_footprints():
  spec = dict(
    configurations=FOOTPRINTS,
    source=digest(Path(footprint_wrist.__file__)),
    prior=digest(ROOT / "protocol.json"),
    hypothesis="Four light fingertip contacts whose normals and heights agree define a usable footprint on a gently curved patch. Initialize height and attitude from it before scanning.",
    selection="Use the same mustard mean measured coverage endpoint for all development variants and the original controller.",
    test="After freezing the controller, execute complete native systems only on mustard and pear; report 240 s coverage and restricted times to all registered thresholds through 600 s.",
  )
  f = ROOT / "footprint-protocol.json"
  if f.exists() and read(f) != spec:
    raise ValueError("Footprint protocol changed")
  if not f.exists():
    write(f, spec)
  return spec


def register():
  old = read(BASE / "protocol.json")
  spec = dict(
    objects=STARTS,
    seconds=240,
    development="mustard",
    configurations=CONFIGS,
    selection="Maximize 0-240 s mean 10 mm measured coverage on mustard. Break ties using final coverage, then lower overload duration. Keep the original controller if none improves mean coverage.",
    primary="Mean measured surface coverage over the 240 s rollout and time to 30% of the whole object's surface",
    thresholds=[10, 20, 30, 40, 50, 70, 90, 95],
    comparison="Full native systems, original time and force limits, same physical starting state within object; same scoring for both.",
    baseline_sha256=digest(BASE / "protocol.json"),
    prior_wrist_sources=sources(),
    revision_source=digest(Path(contact_center.__file__)),
    fixtures=old["fixtures"],
    checkpoint=old["checkpoint_sha256"],
  )
  path = ROOT / "protocol.json"
  if path.exists() and read(path) != spec:
    raise ValueError("Registered acquisition settings have changed")
  if not path.exists():
    write(path, spec)
  return spec


def run_wrist(
  folder,
  obj,
  config,
  *,
  seed=None,
  seconds=240,
  quality=True,
  worker_type=FaceWorker,
  start=None,
  fixture_paths=None,
  fixture_spec=None,
):
  torch.set_num_threads(1)
  start = start or STARTS[obj]
  seed = seed_for(obj, start) if seed is None else seed
  mesh = fixture_paths or (
    (FIXTURE, FIXTURE / "surface.obj")
    if obj == "mustard"
    else experiment.scenario(obj, start)
  )
  spec = read(ROOT / "protocol.json")
  code = dict(
    spec["prior_wrist_sources"],
    controller="ContactCenterWrist",
    revision=spec["revision_source"],
    config=config,
  )
  if config in FOOTPRINTS:
    extra = read(ROOT / "footprint-protocol.json")
    code.update(controller="FootprintWrist", footprint_source=extra["source"])
    controller = partial(FootprintWrist, **extra["configurations"][config])
  elif config == "original":
    controller = FaceWrist
    code.update(controller="FaceWrist")
  else:
    controller = partial(ContactCenterWrist, **spec["configurations"][config])
  with (
    patch.object(experiment, "scenario", return_value=mesh),
    patch.object(sweep_score, "scenario", return_value=mesh),
    patch.object(runtime, "SweepWrist", controller),
    patch.object(runtime, "WristWorker", worker_type),
    patch.object(runtime, "hashes", return_value=code),
  ):
    r = runtime.run(folder, obj, start, "sweep-dp", seconds, seed=seed)
    fi = fixture_spec or spec["fixtures"][f"{obj}/{start}"]
    if (
      r["initial_sha256"],
      r["model_sha256"],
      r["offline_mesh_sha256"],
      r["policy"]["checkpoint_sha256"],
    ) != (
      fi["integration_state"],
      fi["model.mjb"],
      fi["surface.obj"],
      spec["checkpoint"],
    ):
      raise ValueError("Controller evaluation saw different inputs or finger weights")
    if quality:
      sweep_score.score(folder, folder, obj, start, [60.0, 120.0, 240.0], fast=True)
  return folder


def develop(config):
  folder = ROOT / "development" / config
  run_wrist(folder, "mustard", config)
  snap = read(folder / "coverage-only.json")["snapshots"][-1]
  print(
    config,
    {k: snap[k] for k in ("coverage", "mean_coverage", "low3_s", "over8_s")},
    flush=True,
  )
  return config


def select():
  values = {}
  old = read(BASE / "results.json")["objects"]["mustard"]["native"]["wrist-dp"]
  values["original"] = [old["mean_coverage"], old["coverage"], -old["over8"]]
  for config in CONFIGS:
    q = read(ROOT / "development" / config / "coverage-only.json")["snapshots"][-1]
    values[config] = [q["mean_coverage"], q["coverage"], -q["over8_s"]]
  winner = max(values, key=values.get)
  result = dict(
    chosen=winner,
    selection_values=values,
    protocol_sha256=digest(ROOT / "protocol.json"),
  )
  f = ROOT / "selection.json"
  if f.exists() and read(f) != result:
    raise ValueError("Frozen wrist selection differs")
  if not f.exists():
    write(f, result)
  return winner


def release():
  select()
  rows = dict(read(ROOT / "selection.json")["selection_values"])
  for name in FOOTPRINTS:
    s = read(ROOT / "development" / name / "coverage-only.json")["snapshots"][-1]
    rows[name] = [s["mean_coverage"], s["coverage"], -s["over8_s"]]
  chosen = max(rows, key=rows.get)
  result = dict(
    chosen=chosen,
    development=rows,
    original_protocol=digest(ROOT / "protocol.json"),
    footprint_protocol=digest(ROOT / "footprint-protocol.json"),
  )
  file = ROOT / "release.json"
  if file.exists() and read(file) != result:
    raise ValueError("Release selection differs")
  if not file.exists():
    write(file, result)
  return chosen


if __name__ == "__main__":
  cli = argparse.ArgumentParser(description=__doc__)
  cli.add_argument("action", choices=("develop", "select", "footprints", "release"))
  args = cli.parse_args()
  register()
  if args.action in ("footprints", "release"):
    register_footprints()
  if args.action in ("develop", "footprints"):
    candidates = CONFIGS if args.action == "develop" else FOOTPRINTS
    with ProcessPoolExecutor(max_workers=len(candidates)) as pool:
      for f in as_completed([pool.submit(develop, x) for x in candidates]):
        f.result()
  print(
    "Frozen selection",
    release() if args.action in ("footprints", "release") else select(),
    flush=True,
  )
