"""Complete the two fixed-finger wrist-planner arms on the same eleven smooth starts."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from unittest.mock import patch

import numpy as np
import torch

from . import information_score
from . import matched_information as matched
from .__main__ import sources
from .component_study import ARCHIVE, NEW_STARTS, fixture_files, matched_path
from .experiment import ACTIVE2, digest, seed_for
from .face_regression_suite import REPO, read, write
from .face_wrist import FaceWorker, FaceWrist
from .information_report import check_pair, compare
from .smooth_scope import ORIGINAL, ROOT, select
from .sweep import DT
from .sweep_score import score

EXTENSION = ROOT / "wrist-comparison"
ARMS = ("wrist-dp", "aude-dp")


def register():
  scope = select()
  inherited = read(ARCHIVE / "protocol.json")
  if inherited["current_wrist_sources"] != sources() or inherited[
    "runner_sha256"
  ] != digest(Path(matched.__file__)):
    raise ValueError("Acquisition differs from the five original pairs")
  baseline = ACTIVE2 / "protocol.json"
  if digest(baseline) != inherited["frozen_baseline_protocol"]:
    raise ValueError("Aude protocol changed")
  for name, fingerprint in read(baseline)["source_sha256"].items():
    if digest(Path(matched.aude.__file__).with_name(name)) != fingerprint:
      raise ValueError(f"Aude source differs: {name}")
  fixtures = {case: fixture_files(case) for case in NEW_STARTS}
  if fixtures != scope["validation_fixtures"]:
    raise ValueError("Initial states do not match the complete systems")
  checkpoint = read(matched_path(ARCHIVE, ORIGINAL[0], ARMS[0]) / "run.json")["policy"][
    "checkpoint_sha256"
  ]
  spec = dict(
    inherited,
    cases={c: f["seed"] for c, f in fixtures.items()},
    fixtures={
      c: {"model.mjb": f["model"], "initial.npz": f["initial_file"]}
      for c, f in fixtures.items()
    },
    arms=list(ARMS),
    checkpoint_sha256=checkpoint,
    parent_protocol_sha256=digest(ARCHIVE / "protocol.json"),
    scope_sha256=digest(ROOT / "selection.json"),
  )
  target = EXTENSION / "protocol.json"
  if target.exists():
    if read(target) != spec:
      raise ValueError("Stored wrist comparison differs")
  else:
    write(target, spec)


def acquire(case, arm):
  torch.set_num_threads(1)
  obj, start = case.split("/")
  seed = seed_for(obj, start)
  spec = read(EXTENSION / "protocol.json")
  if arm not in spec["arms"] or spec["cases"][case] != seed:
    raise ValueError("Unregistered planner or start")
  with (
    patch.object(matched, "register", return_value=spec),
    patch.object(matched, "OUT", EXTENSION),
    patch.object(matched, "GuardedWrist", FaceWrist),
    patch.object(matched, "WristWorker", FaceWorker),
  ):
    matched.run(EXTENSION, arm, seed, obj, start)
  source = matched_path(EXTENSION, case, arm)
  receipt = read(source / "run.json")
  fixture = fixture_files(case)
  if (
    receipt["initial_sha256"],
    receipt["model_sha256"],
    receipt["policy"]["checkpoint_sha256"],
  ) != (fixture["initial"], fixture["model"], spec["checkpoint_sha256"]):
    raise ValueError(f"Initial state, model or single_all weights changed in {source}")
  dest = EXTENSION / "evaluation" / source.relative_to(EXTENSION)
  quality = score(source, dest, obj, start, list(information_score.TIMES))
  if quality["spec"]["mesh_sha256"] != fixture["mesh"]:
    raise ValueError(f"Different evaluation surface in {case}")
  information_score.evaluate(
    EXTENSION, "matched", obj, start, arm, source, dest / "scores.json", seed
  )
  return case, arm


def native_contacts(native):
  """Re-read physical observations and measured-area traces for the two complete systems."""
  cases, gaps = {}, {}
  native_arms = ("wrist-dp", "native-aude")
  for case, item in sorted(native.items()):
    rows = item["arms"]
    cases[case] = {}
    for arm in native_arms:
      row = rows[arm]
      folder = REPO / row["source"]
      if digest(folder / "trace.npz") != row["trace_sha256"]:
        raise ValueError(f"Unexpected complete-system trace in {case}/{arm}")
      with (
        np.load(folder / "trace.npz") as tr,
        np.load(REPO / row["scored_at"] / "coverage.npz") as cov,
      ):
        _, ticks = information_score.measured_cells(tr)
        cutoff = round(240 / DT)
        stages = [round(s / DT) for s in (0, 60, 120, 180, 240)]
        covered = cov["c10"][stages]
        np.testing.assert_allclose(covered[-1], row["coverage"], rtol=0, atol=1.0e-10)
        pads = tr["pad_valid"][1 : cutoff + 1]
        if len(pads) != cutoff or (ticks == 0).sum() != 4:
          raise ValueError(f"Incomplete physical window in {case}/{arm}")
        contact_ticks = np.flatnonzero(pads.any(axis=1)) + 1
        last_contact = float(contact_ticks[-1] * DT) if len(contact_ticks) else 0.0
        cases[case][arm] = dict(
          new_cells=int((ticks > 0).sum()),
          last_new_cell_s=float(ticks[-1] * DT),
          last_measured_fingertip_s=last_contact,
          no_fingertip_s=float((~pads.any(axis=1)).sum() * DT),
          coverage_by_60s=covered.tolist(),
          new_cells_by_60s=np.diff(
            np.searchsorted(ticks, stages, side="right")
          ).tolist(),
        )
      events_path = REPO / row["scored_at"] / "recoveries.json"
      if arm == "wrist-dp" and events_path.is_file():
        events = [e for e in read(events_path) if last_contact < e["start_s"] <= 240]
        cases[case][arm]["recoveries_after_last_contact"] = len(events)
        cases[case][arm]["recoveries_after_last_contact_with_new_cells"] = sum(
          e["new_cells"] > 0 for e in events
        )
    gaps[case] = rows["native-aude"]["coverage"] - rows["wrist-dp"]["coverage"]
  largest = max(gaps, key=gaps.get)
  return dict(
    cases=cases,
    largest_gap_case=largest,
    largest_gap=gaps[largest],
    share_of_sum_of_gaps=gaps[largest] / sum(gaps.values()),
    mean_remaining_gap=mean(v for c, v in gaps.items() if c != largest),
    aude_coverage_wins=sum(i > 0 for i in gaps.values()),
    mean_new_cells={
      a: mean(row[a]["new_cells"] for row in cases.values()) for a in native_arms
    },
  )


def coverage_figure(cases, diagnostics):
  import matplotlib

  matplotlib.use("Agg")
  from matplotlib import pyplot as plt

  from .component_report import title

  detail = diagnostics["largest_gap_case"]
  times = np.arange(0, 240 + DT, DT)[::50]
  plt.rcParams.update(
    {"font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"], "axes.unicode_minus": False}
  )
  fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
  for arm, label, color, style in [
    ("wrist-dp", "我们的腕部 Active＋single_all", "#086765", "-"),
    ("native-aude", "完整 Aude", "#98512c", "--"),
  ]:
    curves = {}
    for case, pair in cases.items():
      path = REPO / pair["arms"][arm]["scored_at"] / "coverage.npz"
      with np.load(path) as data:
        np.testing.assert_allclose(data["time_s"][:24001:50], times, atol=1e-9, rtol=0)
        curves[case] = data["c10"][:24001:50] * 100
    for ax, series in zip(
      axes, (np.mean(list(curves.values()), axis=0), curves[detail]), strict=True
    ):
      ax.plot(times, series, color=color, linestyle=style, linewidth=2, label=label)
  last = diagnostics["cases"][detail]["wrist-dp"]["last_new_cell_s"]
  axes[1].axvline(last, color="#52646a", linewidth=0.9, linestyle=":")
  axes[1].text(
    last + 4, 16, f"{last:.2f} s\n最后一个新触点格", fontsize=9, color="#354b51"
  )
  for ax, label in zip(
    axes,
    ("四种物体 · 11 个抓姿等权平均", title(detail) + " · 覆盖差距最大的抓姿"),
    strict=True,
  ):
    ax.set(
      xlim=(0, 240),
      ylim=(0, 101),
      xticks=[0, 60, 120, 180, 240],
      title=label,
      xlabel="仿真时间（秒）",
    )
    ax.grid(axis="y", color="#dbe2df", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
  axes[0].set_ylabel("实触覆盖（%）")
  axes[0].legend(loc="lower right", frameon=False, fontsize=9)
  fig.tight_layout()
  fig.savefig(EXTENSION / "native-coverage.png", dpi=160)
  plt.close(fig)


def wrist_execution_difference(native_folder, common_folder):
  """Locate the first different wrist command in the two execution interfaces."""
  receipt = read(common_folder / "run.json")
  with (
    np.load(native_folder / "trace.npz") as a,
    np.load(common_folder / "trace.npz") as b,
  ):
    common = b["palm_command"]
    native = a["palm_command"][: len(common)]
  np.testing.assert_array_equal(native[0], common[0])
  changed = np.flatnonzero(np.max(np.abs(native - common), axis=1) > 1.0e-8)
  details = dict(
    clipped_common_steps=receipt["clipped_wrist_steps"], first_different_s=None
  )
  if len(changed):
    k = int(changed[0])
    previous, target = native[k - 1 : k + 1]
    bounded, clipped = matched.bounded(previous, target, receipt["maximum_radius_m"], 1)
    details.update(
      first_different_s=k * DT,
      native_translation_step_mm=float(
        1000 * np.linalg.norm(target[:3] - previous[:3])
      ),
      common_translation_step_mm=float(
        1000 * np.linalg.norm(common[k, :3] - common[k - 1, :3])
      ),
      bounded_native_reproduces_common=bool(
        clipped and np.allclose(bounded, common[k], atol=1.0e-10, rtol=0)
      ),
    )
  return details


def report():
  cases = {}
  native = read(ROOT.parent / "smooth-results.json")["cases"]
  checkpoint = read(EXTENSION / "protocol.json")["checkpoint_sha256"]
  for case in sorted((*ORIGINAL, *NEW_STARTS)):
    obj, start = case.split("/")
    base = ARCHIVE if case in ORIGINAL else EXTENSION
    receipts = check_pair(base, obj, start, seed_for(obj, start), ARMS)
    reference = native[case]["arms"]["wrist-dp"]
    arms = {}
    for arm in ARMS:
      folder = matched_path(base, case, arm)
      dest = base / "evaluation" / folder.relative_to(base)
      info = read(dest / "information.json")
      spec = info["spec"]
      if (
        spec["trace_sha256"] != digest(folder / "trace.npz")
        or spec["study_protocol_sha256"] != digest(base / "protocol.json")
        or spec["evaluator_sha256"] != digest(Path(information_score.__file__))
        or spec["mesh_sha256"] != reference["mesh_sha256"]
        or receipts[arm]["initial_sha256"] != reference["initial_sha256"]
        or receipts[arm]["checkpoint_sha256"] != checkpoint
      ):
        raise ValueError(
          f"Physical start, checkpoint or scoring differs for {case}/{arm}"
        )
      snap = info["snapshots"]
      arms[arm] = dict(
        nats=snap["t240"]["nats"],
        coverage=snap["t240"]["coverage"],
        cells=snap["t240"]["new_cells"],
        first_1000=snap.get("n1000", {}).get("nats"),
        source=folder.relative_to(REPO).as_posix(),
        trace_sha256=spec["trace_sha256"],
      )
    common = compare(dest.parent, *ARMS)
    for arm in ARMS:
      arms[arm]["common_motion_nats"] = common["arms"][arm]["nats"]
    cases[case] = dict(
      arms=arms,
      common_motion=common,
      our_execution=wrist_execution_difference(
        REPO / reference["source"], matched_path(base, case, ARMS[0])
      ),
    )
  overview = {}
  for key in ("nats", "common_motion_nats", "first_1000", "cells"):
    eligible = {
      c: val["arms"]
      for c, val in cases.items()
      if all(val["arms"][a][key] is not None for a in ARMS)
    }
    overview[key] = dict(
      means={a: mean(pair[a][key] for pair in eligible.values()) for a in ARMS},
      eligible=list(eligible),
      wins=sum(pair[ARMS[0]][key] > pair[ARMS[1]][key] for pair in eligible.values()),
    )
  for arm in ARMS:
    old = mean(cases[c]["arms"][arm]["nats"] for c in ORIGINAL)
    historical = read(ROOT.parent / "smooth-results.json")["matched_wrist"]["dp"][
      "time"
    ][ARMS.index(arm)]
    np.testing.assert_allclose(old, historical, atol=1e-8, rtol=0)
  native_evidence = native_contacts(native)
  coverage_figure(native, native_evidence)
  write(
    EXTENSION / "comparison.json",
    dict(
      scope="Eleven paired smooth-surface grasp starts, equally weighted. Matched planners use frozen single_all and the same force controller. Native diagnostics use each complete system's own physical trace.",
      inherited_five=list(ORIGINAL),
      new_six=list(NEW_STARTS),
      cases=cases,
      summary=overview,
      native_diagnostics=native_evidence,
      checkpoint_sha256=checkpoint,
      extension_protocol_sha256=digest(EXTENSION / "protocol.json"),
    ),
  )
  return overview


if __name__ == "__main__":
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("action", choices=("run", "report"))
  p.add_argument("--workers", type=int, default=3)
  args = p.parse_args()
  register()
  if args.action == "run":
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
      futures = [pool.submit(acquire, c, a) for c in NEW_STARTS for a in ARMS]
      for future in as_completed(futures):
        print("Verified", *future.result(), flush=True)
  print(report(), flush=True)
