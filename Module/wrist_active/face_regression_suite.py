"""Run the retained comparison matrix from immutable starts into a new result tree."""

import argparse
import json
import subprocess
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from hashlib import sha256
from pathlib import Path
from time import perf_counter_ns, sleep

REPO = Path(__file__).resolve().parents[2]
GENERATED = REPO / "Module/generated"
DEFAULT = GENERATED / "gpis_wrist_face_transfer_20261121"
SUBMODULE = "Module.wrist_active.face_regression"
TRANSFERS = (
  "ellipsoid/start-01",
  "ellipsoid/start-02",
  "meat-can/start-01",
  "meat-can/start-02",
  "oblate/start-03",
  "ovoid/start-03",
  "superellipsoid/start-03",
)
PEAR = "pear/active-02"


def read(path):
  return json.loads(path.read_bytes())


def write(path, value):
  path.parent.mkdir(parents=True, exist_ok=True)
  temp = path.with_suffix(".tmp")
  temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
  temp.replace(path)


def command(module, *args):
  return [sys.executable, "-u", "-m", module, *map(str, args)]


def matrix(root):
  tasks = []

  def add(group, tag, commands, details=None):
    tasks.append(dict(group=group, tag=tag.replace("/", "_"), commands=commands, details=details))

  def regression(action, case, method, *args):
    return command(SUBMODULE, action, "--root", root, "--case", case, "--method", method, *args)

  for method in ("long-dp", "sweep-mcc", "seed-dp", "route-dp", "no-recontact-dp"):
    add("native", "pear_" + method, [regression("native", PEAR, method)])
  for case in TRANSFERS:
    add("native", case + "_pure", [regression("native", case, "sweep-mcc")])
  for method in ("aude-mcc", "aude-single-all-mcc"):
    add("aude", "pear_" + method, [regression("aude", PEAR, method)])
  for case in TRANSFERS:
    add("aude", case + "_aude", [regression("aude", case, "aude-mcc")])
  for case in (PEAR, *TRANSFERS):
    variants = [("wrist-dp", None), ("aude-dp", None)]
    if case == PEAR:
      variants.extend(
        [("wrist-mcc", None), ("aude-mcc", None), ("wrist-dp", 20261122), ("aude-dp", 20261122)]
      )
    for method, seed in variants:
      args = ("--seed", seed) if seed else ()
      add(
        "controlled",
        case + "_" + method + "_" + str(seed or "default"),
        [regression("controlled", case, method, *args), regression("score-controlled", case, method, *args)],
      )
  add("tapes", "paired_tapes_and_causal_replay", [command(SUBMODULE, "tapes", "--root", root)])
  for group in ("fair", "smooth"):
    source = GENERATED / f"i01_a_{group}_20261121"
    archived = sorted((source / "runs").rglob("stopwatch.json"))
    if len(archived) != {"fair": 24, "smooth": 27}[group]:
      raise ValueError(f"Incomplete original I01-A {group} matrix")
    for path in archived:
      relative = path.parent.relative_to(source / "runs")
      obj, start, regime = relative.parts[:3]
      spec = read(path)
      arm = spec["branch"]
      plan = source / "plans" / obj / start / ("shared" if regime == "shared" else arm)
      dest = root / "regression/i01-a" / group / "runs" / relative
      cmd = command(
        "Module.i04_geometric_traversal.i01_a",
        "trial",
        "--branch",
        arm,
        "--output",
        dest,
        "--prepared",
        plan,
        "--seed",
        spec["seed"],
        "--calibration",
        GENERATED / "i01_a_20261120/wrist_calibration.json",
      )
      add(
        "i01-" + group,
        str(relative).replace("\\", "_"),
        [cmd],
        dict(previous=str(path.parent), dest=str(dest), plan=str(plan)),
      )
  return tasks


def verify_i01(task, start, stop):
  import numpy as np

  d = task["details"]
  before, after = Path(d["previous"]), Path(d["dest"])
  base = read(before / "stopwatch.json")
  fresh = read(after / "task_written.json")
  for key in ("branch", "seed", "states", "executed_s", "wrist_sha256", "nominal_wrist_sha256"):
    if base[key] != fresh[key]:
      raise ValueError(f"{task['tag']} differs at {key}")
  chunks = sorted(before.glob("trace_*.npz"))
  if not chunks or [p.name for p in chunks] != [p.name for p in sorted(after.glob("trace_*.npz"))]:
    raise ValueError("Incomplete physics trace")
  for a in chunks + [before / "live_head_contacts.npz", before / "r_gpis_inputs.npz"]:
    b = after / a.name
    if sha256(a.read_bytes()).digest() == sha256(b.read_bytes()).digest():
      continue
    with np.load(a, allow_pickle=False) as f, np.load(b, allow_pickle=False) as g:
      if f.files != g.files:
        raise ValueError(f"Trace fields differ in {b}")
      for key in f.files:
        np.testing.assert_array_equal(f[key], g[key], err_msg=f"{b}/{key}")
  if start < fresh["written_ns"] < stop:
    write(
      after / "stopwatch.json",
      dict(
        fresh,
        repeat=1,
        seconds_to_written=(fresh["written_ns"] - start) / 1e9,
        seconds_through_process_exit=(stop - start) / 1e9,
        cpu_affinity=None,
      ),
    )
  return dict(
    states=fresh["states"],
    seconds=fresh["executed_s"],
    chunks_equal=len(chunks),
    wrist_sha256=fresh["wrist_sha256"],
    head_observations_equal=True,
    gp_inputs_equal=True,
  )


def execute(root, task):
  tag = task["group"] + "_" + task["tag"]
  memo = root / "regression/receipts" / (tag + ".json")
  if memo.is_file():
    return
  logs = root / "regression/logs"
  logs.mkdir(parents=True, exist_ok=True)
  start = perf_counter_ns()
  wall = logs / (tag + "-wall.json")
  ran = False
  with (logs / (tag + ".log")).open("ab") as log:
    for cmd in task["commands"]:
      if task["details"] and (Path(task["details"]["dest"]) / "task_written.json").is_file():
        continue
      run = subprocess.run(
        cmd,
        cwd=REPO,
        stdout=log,
        stderr=subprocess.STDOUT,
        check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
      )
      if run.returncode:
        raise RuntimeError(f"{tag} failed ({run.returncode}). See {logs / (tag + '.log')}")
      ran = True
  stop = perf_counter_ns()
  if task["details"]:
    if ran:
      write(wall, dict(start=start, stop=stop))
    elif wall.is_file():
      span = read(wall)
      start, stop = span["start"], span["stop"]
  verified = verify_i01(task, start, stop) if task["details"] else {}
  write(
    memo, dict(group=task["group"], tag=task["tag"], elapsed_s=(stop - start) / 1e9, verification=verified)
  )
  print("Completed", tag, flush=True)


def run_batch(root, group, workers):
  import psutil

  if group in ("all", "aude"):
    folder = root / "regression/active-02-a"
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("model.mjb", "planning-hand.mjb", "initial.npz", "protocol.json"):
      src = (GENERATED / "active-02-a_20261121" / name).read_bytes()
      dest = folder / name
      if not dest.is_file():
        dest.write_bytes(src)
      if dest.read_bytes() != src:
        raise ValueError(f"Pear reference differs at {name}")
  tasks = [t for t in matrix(root) if (group == "all" or t["group"] == group)]
  seen = root / "regression/receipts"
  tasks = [t for t in tasks if not (seen / (t["group"] + "_" + t["tag"] + ".json")).is_file()]
  with ThreadPoolExecutor(max_workers=workers) as pool:
    pending = set()
    failures = []
    while tasks or pending:
      while (
        tasks and len(pending) < workers and (not pending or psutil.virtual_memory().available > 5.5 * 2**30)
      ):
        task = tasks.pop(0)
        pending.add(pool.submit(execute, root, task))
        sleep(3)
      if pending:
        done, pending = wait(pending, timeout=6, return_when=FIRST_COMPLETED)
        for job in done:
          try:
            job.result()
          except Exception as error:
            failures.append(str(error))
            print(str(error), flush=True)
    if failures:
      raise RuntimeError("\n".join(failures))


def pace_comparison(root):
  from unittest.mock import patch

  import torch

  from . import sweep_dispatch
  from . import sweep_experiment as runtime
  from .__main__ import sources
  from .face_wrist import FaceWorker, FaceWrist
  from .sweep_paced import timer_resolution
  from .sweep_score import score

  torch.set_num_threads(1)
  sub = root / "regression/timing"
  for mode in ("paced", "paced-dispatch"):
    trial = sub / mode / "pear/active-02/sweep-dp"
    clock = sweep_dispatch.FrameRelease()
    law = dict(sources(), controller="FaceWrist", timing_mode=mode)
    with (
      patch.object(runtime, "SweepWrist", FaceWrist),
      patch.object(runtime, "hashes", return_value=law),
      patch.object(sweep_dispatch.wrist_worker, "WristWorker", FaceWorker),
      patch.object(runtime, "WristWorker", FaceWorker),
      sweep_dispatch.phased_dispatch(clock) if mode == "paced-dispatch" else timer_resolution(),
    ):
      rec = runtime.run(trial, "pear", "active-02", "sweep-dp", 120.0, paced=True)
    score(trial, trial, "pear", "active-02", [60.0, 120.0])
    if mode == "paced-dispatch" and clock.count:
      write(trial / "dispatch.json", dict(applied_after_frame=clock.count, trace_sha256=rec["trace_sha256"]))
    if mode == "paced-dispatch" and not (trial / "dispatch.json").exists():
      raise ValueError("Missing post-frame scheduling receipt")


if __name__ == "__main__":
  cli = argparse.ArgumentParser(description=__doc__)
  cli.add_argument("--root", type=Path, default=DEFAULT)
  cli.add_argument(
    "--group",
    choices=("native", "aude", "controlled", "tapes", "timing", "i01-fair", "i01-smooth", "all"),
    default="all",
  )
  cli.add_argument("--workers", type=int, default=3)
  conf = cli.parse_args()
  if conf.group == "timing":
    pace_comparison(conf.root.resolve())
  else:
    run_batch(conf.root.resolve(), conf.group, conf.workers)
