"""Complete the wrist/finger factorial and validate the frozen system on six further starts."""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from . import matched_information as matched
from . import sweep_experiment as runtime
from .__main__ import RESULTS, sources
from .experiment import digest, seed_for
from .face_regression_suite import PEAR, REPO, TRANSFERS, read, write
from .face_wrist import FaceWorker, FaceWrist
from .guard_report import same_three_seconds
from .information_score import TIMES, evaluate, gp_prefix, measured_cells
from .sweep_score import score

OUT = runtime.GENERATED / "gpis_wrist_components_20261122"
ARCHIVE = RESULTS / "regression/controlled"
NEW_STARTS = tuple(f"{obj}/start-{i:02}" for obj in runtime.SMOOTH for i in (1, 2))
NATIVE_METHODS = ("sweep-dp", "sweep-mcc")


def study_jobs():
  # Interleave native validation and the missing cells; each job has a distinct trace and log.
  for idx, case in enumerate(TRANSFERS):
    yield from (("matched", case, method) for method in ("wrist-mcc", "aude-mcc"))
    if idx < len(NEW_STARTS):
      yield from (("native", NEW_STARTS[idx], method) for method in NATIVE_METHODS)


def fixture_files(case):
  obj, start = case.split("/")
  folder = (
    runtime.GENERATED
    / "i01_a_smooth_20261121/runs"
    / obj
    / start
    / "shared/fixed-dp_r01"
  )
  with np.load(folder / "initial.npz", allow_pickle=False) as tr:
    initial = sha256(tr["integration_state"].tobytes()).hexdigest()
  return dict(
    initial=initial,
    model=digest(folder / "model.mjb"),
    mesh=digest(folder / "surface.obj"),
    initial_file=digest(folder / "initial.npz"),
    seed=seed_for(obj, start),
  )


def freeze():
  old = read(ARCHIVE / "protocol.json")
  if old["current_wrist_sources"] != sources() or old["runner_sha256"] != digest(
    Path(matched.__file__)
  ):
    raise ValueError("Archived factorial and loaded controller differ")
  spec = dict(
    seconds=240,
    control_hz=100,
    original_transfer=list(TRANSFERS),
    separate_pear=PEAR,
    original_factorial=dict(
      wrists=["wrist", "aude"],
      fingers=["dp", "mcc"],
      seed="original seed for each start",
      interface="Both sources drive the same wrist actuator at the same step limits. Aude supplies only its wrist proposal; native Aude+MCC is a separate reference.",
      inherited_protocol=digest(ARCHIVE / "protocol.json"),
    ),
    native_validation=dict(
      fixtures={case: fixture_files(case) for case in NEW_STARTS},
      rationale="All remaining A/B starts on the three I01-A smooth meshes. These starts are new to the active rollout comparison; they were used for earlier fixed-wrist I01-A tests.",
      comparison="Identical FaceWrist with single_all plus ForceFeedback or fixed initial nominal angles plus the same ForceFeedback; each replans from its own tactile stream.",
    ),
    endpoints=dict(
      wrist="Realized conditional GP information (nat) after 240 s at the same finger controller.",
      finger="Cumulative light >=3 measured fingertips in 240 s at a fixed wrist-planner implementation.",
      system="10 mm geodesic area coverage at 240 s.",
      secondary=[
        "signed GPIS F-score at 5 mm",
        "time to 70% and 90% coverage, censored at 240 s",
        "area under the 0-to-240 s coverage curve divided by 240 s",
        "moving same three light fingertips sustained for >=1 s",
        "time at or above 8 N and peak N",
      ],
      definitions="At least 0.2 N per qualifying distal fingertip and all five loads under 8 N for light contact; original scorers and mesh.",
    ),
    aggregation="Pair within each start. Report all seven original transfers equally weighted, the pear separately, and the new six starts as a separate cohort. Also report signs and the four-cell interaction. Frames are not independent replicates.",
    sources=sources(),
    acquisition_driver=digest(Path(__file__)),
    reference_protocol=digest(RESULTS / "protocol.json"),
    checkpoint=read(RESULTS / "results.json")["cases"][0]["checkpoint_sha256"],
  )
  file = OUT / "protocol.json"
  if file.is_file() and read(file) != spec:
    raise ValueError("Existing study has a different specification")
  if not file.is_file():
    write(file, spec)
  copy = OUT / "controlled/protocol.json"
  copy.parent.mkdir(parents=True, exist_ok=True)
  if not copy.is_file():
    copy.write_bytes((ARCHIVE / "protocol.json").read_bytes())
  if digest(copy) != spec["original_factorial"]["inherited_protocol"]:
    raise ValueError(
      "Matched interface protocol must be byte-identical to the archived arms"
    )
  return spec


def matched_path(root, case, method):
  obj, start = case.split("/")
  return root / "matched" / obj / start / f"seed-{seed_for(obj, start)}" / method


def run_matched(case, method):
  obj, start = case.split("/")
  seed = seed_for(obj, start)
  sub = OUT / "controlled"
  with (
    patch.object(matched, "register", return_value=read(sub / "protocol.json")),
    patch.object(matched, "GuardedWrist", FaceWrist),
    patch.object(matched, "WristWorker", FaceWorker),
  ):
    matched.run(sub, method, seed, obj, start)
  source = matched_path(sub, case, method)
  dest = sub / "evaluation" / source.relative_to(sub)
  quality = score(source, dest, obj, start, list(TIMES))
  evaluate(sub, "matched", obj, start, method, source, dest / "scores.json", seed)
  r, archived = (
    read(source / "run.json"),
    read(matched_path(ARCHIVE, case, "wrist-dp") / "run.json"),
  )
  for key in (
    "initial_sha256",
    "model_sha256",
    "protocol_sha256",
    "steps",
    "segments",
    "seed",
    "maximum_radius_m",
    "finger_controller",
  ):
    if r[key] != archived[key]:
      raise ValueError(f"Mismatched {key} in {source}")
  if r["steps"] != 24000 or r["segments"] != 1200 or r["policy"] is not None:
    raise ValueError("Unexpected paired budget or finger policy")
  with np.load(source / "trace.npz") as tr:
    q = tr["finger_nominal"]
    np.testing.assert_array_equal(q, np.broadcast_to(q[0], q.shape))
  return dict(trace_sha256=r["trace_sha256"], scored=quality["snapshots"][-1]["s"])


def run_native(spec, case, method):
  obj, start = case.split("/")
  frozen = spec["native_validation"]["fixtures"][case]
  if fixture_files(case) != frozen:
    raise ValueError(f"Validation fixture changed for {case}")
  folder = OUT / "native" / case / method
  code = dict(sources(), controller="FaceWrist")
  with (
    patch.object(runtime, "SweepWrist", FaceWrist),
    patch.object(runtime, "WristWorker", FaceWorker),
    patch.object(runtime, "hashes", return_value=code),
  ):
    r = runtime.run(folder, obj, start, method, 240.0, seed=frozen["seed"])
  if (
    r["initial_sha256"] != frozen["initial"]
    or r["model_sha256"] != frozen["model"]
    or r["offline_mesh_sha256"] != frozen["mesh"]
    or r["steps"] != 24000
  ):
    raise ValueError(f"Validation inputs changed in {folder}")
  quality = score(folder, folder, obj, start, [240.0])
  with np.load(folder / "trace.npz") as tr:
    if not np.all(tr["pad_valid"][0]) or tr["region_load_n"][0].max() >= 8:
      raise ValueError("Not the registered light four-pad initial grasp")
    if method == "sweep-mcc":
      if r["installs"] or r["policy"] is not None:
        raise ValueError("Unexpected finger policy in the control")
      q = tr["finger_nominal"]
      np.testing.assert_array_equal(q, np.broadcast_to(q[0], q.shape))
    elif r["policy"]["checkpoint_sha256"] != spec["checkpoint"] or r["installs"] <= 0:
      raise ValueError("Unexpected DP checkpoint or execution")
    pts, ticks = measured_cells(tr)
  info, _ = gp_prefix(pts, matched.LENGTHS_M[0], matched.NOISE_VARIANCE)
  facts = dict(
    trace_sha256=r["trace_sha256"],
    matched_initial_sha256=frozen["initial"],
    new_cells=int(np.sum(ticks > 0)),
    nats=float(info[-1] - info[(ticks == 0).sum()]),
    moving_same_three_s=same_three_seconds(folder),
    scored=quality["snapshots"][-1]["s"],
  )
  write(folder / "verification.json", facts)
  return facts


def trial(kind, case, method):
  torch.set_num_threads(1)
  if (kind, case, method) not in set(study_jobs()):
    raise ValueError("Trial outside the frozen design")
  spec = read(OUT / "protocol.json")
  if (
    spec["sources"] != sources() or digest(Path(__file__)) != spec["acquisition_driver"]
  ):
    raise ValueError("Frozen acquisition source changed")
  result = (
    run_matched(case, method) if kind == "matched" else run_native(spec, case, method)
  )
  write(OUT / "receipts" / kind / case / (method + ".json"), result)


def dispatch(job):
  kind, case, method = job
  log = OUT / "logs" / kind / case / (method + ".log")
  log.parent.mkdir(parents=True, exist_ok=True)
  env = dict(
    os.environ,
    OPENBLAS_NUM_THREADS="1",
    OMP_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    PYTHONUTF8="1",
  )
  with log.open("w", encoding="utf-8") as stream:
    result = subprocess.run(
      [
        sys.executable,
        "-u",
        "-m",
        "Module.wrist_active.component_study",
        "trial",
        kind,
        case,
        method,
      ],
      cwd=REPO,
      env=env,
      stdout=stream,
      stderr=subprocess.STDOUT,
    )
  if result.returncode:
    raise RuntimeError(
      f"{job} failed. {log}\n{log.read_text(encoding='utf-8')[-1500:]}"
    )
  return job


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("action", choices=("freeze", "all", "trial"))
  p.add_argument("args", nargs="*")
  p.add_argument("--workers", type=int, default=4)
  opt = p.parse_args()
  if opt.action == "trial":
    trial(*opt.args)
  else:
    freeze()
    if opt.action == "all":
      with ThreadPoolExecutor(max_workers=opt.workers) as pool:
        tasks = [pool.submit(dispatch, job) for job in study_jobs()]
        errors = []
        for f in as_completed(tasks):
          try:
            print("completed", *f.result(), flush=True)
          except Exception as exc:
            print(str(exc), flush=True)
            errors.append(str(exc))
        if errors:
          raise RuntimeError("\n".join(errors))


if __name__ == "__main__":
  main()
