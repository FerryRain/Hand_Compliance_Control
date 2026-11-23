"""Rerun the wrist-dependent studies and both sides of the published comparisons."""

import argparse
from pathlib import Path
from shutil import copyfile
from unittest.mock import patch

import numpy as np
import torch

from ..acitive_02 import active_a
from . import matched_information as matched
from . import sweep_verify
from .__main__ import RESULTS, run, sources
from .experiment import ACTIVE2, digest, save_json, seed_for
from .face_wrist import FaceWorker, FaceWrist
from .information_score import TIMES, evaluate
from .mcc_trials import RESULTS as OLD_INFORMATION
from .mcc_trials import read
from .sweep_aude_transfer import rollout
from .sweep_experiment import OUT
from .sweep_pair import replay as replay_tape
from .sweep_score import score


def identical(first, second, fields):
  with np.load(first / "trace.npz") as a, np.load(second / "trace.npz") as b:
    for field in fields:
      np.testing.assert_array_equal(a[field], b[field], err_msg=f"{second}/{field}")
  return list(fields)


def register(root):
  folder = root / "regression/controlled"
  folder.mkdir(parents=True, exist_ok=True)
  previous = matched.register(OLD_INFORMATION / "transfer-extension")
  spec = dict(
    previous,
    current_wrist_sources=sources(),
    acquisition_adapter=digest(Path(__file__)),
    current_wrist_protocol_sha256=digest(root / "protocol.json"),
  )
  path = folder / "protocol.json"
  if path.exists():
    if read(path) != spec:
      raise ValueError("The controlled study code changed")
  else:
    save_json(path, spec)
  return folder, spec


def controlled(root, obj, start, method, seed):
  sub, spec = register(root)
  with (
    patch.object(matched, "register", return_value=spec),
    patch.object(matched, "GuardedWrist", FaceWrist),
    patch.object(matched, "WristWorker", FaceWorker),
  ):
    matched.run(sub, method, seed, obj, start)


def score_controlled(root, obj, start, method, seed):
  sub, _ = register(root)
  suffix = Path(obj) / start / f"seed-{seed}" / method
  source = sub / "matched" / suffix
  dest = sub / "evaluation/matched" / suffix
  score(source, dest, obj, start, list(TIMES))
  evaluate(sub, "matched", obj, start, method, source, dest / "scores.json", seed)


def native(root, obj, start, method):
  sub = root / "regression"
  if method == "sweep-mcc":
    seconds = 1129.32 if obj == "pear" else 240.0
    folder = sub / "native"
    seed = None
  elif method == "long-dp":
    seconds, folder, method, seed = 1129.32, sub / "native", "sweep-dp", None
  elif method == "seed-dp":
    seconds, folder, method, seed = 600.0, sub / "second-noise", "sweep-dp", matched.PEAR_SEEDS[1]
  else:
    seconds, folder, seed = 240.0, sub / "ablations", None
  run(folder, obj, start, seconds, arm=method, seed=seed)
  trial = folder / "runs" / obj / start / method
  if seconds > 600:
    score(trial, trial, obj, start, [60.0, 120.0, 240.0, 600.0, seconds])
  if obj == "pear" or (method == "sweep-mcc" and obj != "meat-can"):
    if folder == sub / "second-noise":
      previous = OUT / "force-priority/new-seed" / obj / start / method
    elif method == "sweep-mcc":
      previous = OLD_INFORMATION / "runs" / obj / start / method
    else:
      previous = OUT / "force-priority/runs" / obj / start / method
    verified = identical(
      previous,
      trial,
      (
        "qpos",
        "finger_nominal",
        "palm_command",
        "published_preview",
        "pad_valid",
        "points_m",
        "region_load_n",
      ),
    )
    save_json(trial / "regression.json", dict(reference=digest(previous / "trace.npz"), fields=verified))
  if method == "sweep-mcc":
    receipt = read(trial / "run.json")
    assert receipt["installs"] == 0 and receipt["policy"] is None
    with np.load(trial / "trace.npz") as f:
      q = f["finger_nominal"]
      np.testing.assert_array_equal(q, np.broadcast_to(q[0], q.shape))


def aude_reference(root, obj, start, method):
  sub = root / "regression"
  if obj == "pear":
    folder = sub / "active-02-a"
    folder.mkdir(parents=True, exist_ok=True)
    protocol = read(ACTIVE2 / "protocol.json")
    for name in ("model.mjb", "planning-hand.mjb", "initial.npz", "protocol.json"):
      if not (folder / name).exists():
        copyfile(ACTIVE2 / name, folder / name)
      if digest(folder / name) != digest(ACTIVE2 / name):
        raise ValueError("Active-02-A fixture changed")
    trial = folder / method
    if not (trial / "run.json").is_file():
      active_a.run_arm((folder, method, protocol))
    previous = ACTIVE2 / method
    times = [60.0, 120.0, 240.0, 600.0, 1129.32]
  else:
    trial = rollout(sub, obj, start, 240.0)
    previous = OUT / "strong-baseline" / obj / start / "aude-mcc"
    times = list(TIMES)
  fields = identical(
    previous, trial, ("qpos", "finger_command", "points_m", "normals_outward", "pad_valid", "region_load_n")
  )
  save_json(trial / "regression.json", dict(reference=digest(previous / "trace.npz"), fields=fields))
  score(trial, trial, obj, start, times)


def fixed_tapes(root):
  for obj, start in [
    ("pear", "active-02"),
    ("ovoid", "start-03"),
    ("meat-can", "start-01"),
    ("meat-can", "start-02"),
  ]:
    replay_tape(root, obj, start)
  with (
    patch.object(sweep_verify, "SweepWrist", FaceWrist),
    patch.object(sweep_verify, "WristWorker", FaceWorker),
  ):
    sweep_verify.replay(root)


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("action", choices=("register", "controlled", "score-controlled", "native", "aude", "tapes"))
  p.add_argument("--root", type=Path, default=RESULTS)
  p.add_argument("--case", default="pear/active-02", choices=[f"{o}/{s}" for o, s in matched.CASES])
  p.add_argument("--method", default="wrist-dp")
  p.add_argument("--seed", type=int)
  a = p.parse_args()
  torch.set_num_threads(1)
  obj, start = a.case.split("/")
  seed = seed_for(obj, start) if a.seed is None else a.seed
  if a.action == "register":
    register(a.root)
  elif a.action == "tapes":
    fixed_tapes(a.root)
  elif a.action == "native":
    native(a.root, obj, start, a.method)
  elif a.action == "aude":
    aude_reference(a.root, obj, start, a.method)
  else:
    if a.method not in matched.METHODS:
      p.error("Choose a registered controlled arm")
    (controlled if a.action == "controlled" else score_controlled)(a.root, obj, start, a.method, seed)


if __name__ == "__main__":
  main()
