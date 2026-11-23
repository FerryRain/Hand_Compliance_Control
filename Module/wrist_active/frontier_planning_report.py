"""Relate each applied command to measured GP information and its online planning cost."""

import csv
import json
from io import TextIOWrapper
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from . import frontier_planning_timing as timing
from .experiment import digest
from .face_regression_suite import REPO, read, write
from .frontier_information import ARMS

HORIZONS = (60, 240, 600)
SHARED_TICKS = timing.aude.REPLAN_TICKS
TOTALS = (
  "online_240_s",
  "online_600_s",
  "online_ms_per_tick",
)


def distribution(values):
  return dict(
    n=len(values),
    mean=float(np.mean(values)),
    p50=float(np.median(values)),
    p95=float(np.percentile(values, 95)),
  )


def intervals(prefix, applied, end):
  start = applied[applied < end]
  stop = np.r_[start[1:], end]
  gain = prefix[stop] - prefix[start]
  np.testing.assert_allclose(gain.sum(), prefix[end], atol=2e-8)
  full = start + SHARED_TICKS <= end
  shared = prefix[start[full] + SHARED_TICKS] - prefix[start[full]]
  return (
    dict(
      plans=len(start),
      full_shared=len(shared),
      total_nat=float(prefix[end]),
      nat_per_tick=float(prefix[end] / end),
      nat_per_plan=float(gain.mean()),
      nat_per_shared=float(shared.mean()),
    ),
    gain,
    shared,
    stop,
  )


def timings(local, proposal, local_joint, global_ik):
  all_ms = local + proposal
  high = proposal > 0
  result = {"all": distribution(all_ms), "local": distribution(local)}
  if high.any():
    result.update(
      proposal=distribution(proposal[high]),
      with_proposal=distribution(all_ms[high]),
      without_proposal=distribution(all_ms[~high]),
      local_with_joint_projection=distribution(local[local_joint]),
      outer_with_ik=distribution(proposal[global_ik]),
      full_update_with_ik=distribution(all_ms[global_ik]),
    )
  return result


def rows_for(obj, arm, s, planning, piece, cols, csv_writer):
  route = timing.OUT / obj / ("ours" if arm == "wrist-dp" else "aude")
  receipt = read(route / "measurement.json")
  original = REPO / planning["source"]
  if not (
    s["trace_sha256"] == planning["trace_sha256"] == receipt["signature"]["trace"]
    and digest(route / "events.npz") == receipt["samples_sha256"]
    and receipt["matched_frames"] == timing.SAMPLES + 1
    and digest(original / "signed-600s.ply") == receipt["final_mesh_sha256"]
  ):
    raise ValueError(f"Measurement and outcome have different inputs for {obj}/{arm}")
  with np.load(route / "events.npz") as archive:
    applied = archive["apply_tick"]
    local = archive["local_ms"]
    proposal = archive["proposal_ms"]
    clock = archive["online_elapsed_s"]
  expected_period = timing.PLAN_TICKS if arm == "wrist-dp" else timing.aude.REPLAN_TICKS
  np.testing.assert_array_equal(applied, np.arange(0, timing.SAMPLES, expected_period))
  local_joint = np.zeros(len(applied), dtype=bool)
  global_ik = np.zeros_like(local_joint)
  if arm == "wrist-dp":
    log = original / "plans.jsonl"
    if digest(log) != read(original / "run.json")["plan_sha256"]:
      raise ValueError("Archived applied-plan log was modified")
    np.testing.assert_array_equal(
      applied,
      [json.loads(line)["apply_tick"] for line in log.read_bytes().splitlines()],
    )
  else:
    # These are the native early-return conditions in each planning function.
    with np.load(original / "trace.npz", allow_pickle=False) as trace:
      local_joint = (
        trace["region_load_n"][applied].max(axis=1) <= timing.aude.SURFACE_LOAD_N
      )
      global_ik = (proposal > 0) & trace["pad_valid"][applied].any(axis=1)
    np.testing.assert_array_equal(
      applied[proposal > 0], np.arange(0, timing.SAMPLES, timing.aude.PLAN_TICKS)
    )
  step = np.zeros(timing.SAMPLES + 1)
  step[s["tick"]] = s["gain_nat"]
  prefix = np.cumsum(step)
  windows = {}
  for h in HORIZONS:
    end = round(h / timing.DT)
    w, gain, shared, stop = intervals(prefix, applied, end)
    np.testing.assert_allclose(w["total_nat"], s["windows"][str(h)]["total_nat"])
    if arm == "native-aude":
      count = int(np.count_nonzero((applied < end) & (proposal > 0)))
      w["outer_proposals"] = count
      w["nat_per_outer_proposal_interval"] = float(prefix[end] / count)
    windows[str(h)] = w
  for i, t in enumerate(applied):
    csv_writer.writerow(
      [
        obj,
        arm,
        int(t),
        f"{t * timing.DT:.2f}",
        f"{stop[i] * timing.DT:.2f}",
        int(stop[i] - t),
        int(local_joint[i]),
        int(global_ik[i]),
        float(local[i]),
        float(proposal[i]),
        float(local[i] + proposal[i]),
        float(gain[i]),
        float(shared[i]) if i < len(shared) else "",
        float(clock[t]),
        float(clock[stop[i]]),
      ]
    )
  cols[arm].append((local, proposal, local_joint, global_ik))
  meta = {key: receipt[key] for key in TOTALS if key != "online_ms_per_tick"}
  meta["online_ms_per_tick"] = 1000 * clock[-1] / timing.SAMPLES
  frame_ms = np.diff(clock) * 1000
  np.testing.assert_allclose(
    [clock[24000], clock[-1]], [meta["online_240_s"], meta["online_600_s"]]
  )
  piece[arm] = dict(
    windows=windows,
    clock=meta,
    duration_ms=timings(local, proposal, local_joint, global_ik),
    control_interval_ms=distribution(frame_ms),
    matched_frames=receipt["matched_frames"],
    matched_fields=receipt["matched_fields"],
    trace_sha256=s["trace_sha256"],
    mesh_sha256=receipt["final_mesh_sha256"],
    measurement_sha256=digest(route / "measurement.json"),
  )
  return frame_ms, step[1:]


def report():
  study, out = timing.STUDY, timing.OUT
  experiment = read(study / "results.json")
  info_file = study / "information/gain.json"
  info = read(info_file)
  if (
    info["study_sha256"] != digest(study / "results.json") or info["dt_s"] != timing.DT
  ):
    raise ValueError("Information scores and benchmark differ")
  cols = {a: [] for a in ARMS}
  frames = {}
  result = dict(
    protocol_sha256=experiment["protocol_sha256"],
    scores_sha256=digest(info_file),
    measurement_program_sha256=digest(Path(timing.__file__)),
    control_dt_s=timing.DT,
    shared_followup_s=SHARED_TICKS * timing.DT,
    timing_protocol="One 600 s unpaced rollout at a time, alternating the first system in each pair. The full-system exploration time starts at the first planning request and stops at the last acquired physical control frame. It includes online tactile mapping and local GPIS, target planning, finger control, interpolation, feedback control and simulated motion. Actual wall time counts concurrent work once. Aude outer and local planning calls have separate timers. System preparation precedes the first request. Final visualization and reference scoring follow exploration.",
    device_note="The local GPIS uses a double-precision GPyTorch Matérn kernel on CPU tensors followed by NumPy and SciPy computation. The wrist system runs the frozen single_all finger policy on CUDA.",
    gain_protocol="Sum GP increments for (apply_tick, next_apply_tick], with every zero and the clipped terminal interval in the native mean. Fixed followup averages the first 18 ticks after each applied target with a complete observation window. Outer proposals last 90 ticks and typically span five local segments.",
    aggregation="Information and system times use equal-object means. Duration statistics pool the relevant calls across all five objects. Each object contributes the same number of native target updates for its method. Conditional solver counts depend on measured contact and load.",
    objects={},
    mean={},
  )
  temp = out / "plan-events.tmp"
  with (
    ZipFile(temp, "w", ZIP_DEFLATED, compresslevel=6) as packed,
    packed.open("plan-events.csv", "w") as buf,
    TextIOWrapper(buf, encoding="utf-8", newline="") as dest,
  ):
    writer = csv.writer(dest)
    writer.writerow(
      [
        "object",
        "system",
        "apply_tick",
        "start_sim_s",
        "end_sim_s",
        "control_steps",
        "local_joint_projection",
        "global_joint_ik",
        "local_planning_ms",
        "global_proposal_ms",
        "total_planning_ms",
        "realized_gain_nat",
        "first_180ms_gain_nat",
        "start_online_wall_s",
        "end_online_wall_s",
      ]
    )
    for obj, example in info["objects"].items():
      block = {}
      frames[obj] = {}
      result["objects"][obj] = dict(
        name=example["name"], start=example["start"], arms=block
      )
      for arm in ARMS:
        costs = read(
          out / obj / ("ours" if arm == "wrist-dp" else "aude") / "measurement.json"
        )
        if not (
          costs["signature"]["protocol"] == result["protocol_sha256"]
          and costs["signature"]["timing_program"]
          == result["measurement_program_sha256"]
        ):
          raise ValueError("Mixed timing protocols")
        frames[obj][arm] = rows_for(
          obj,
          arm,
          example["arms"][arm],
          experiment["objects"][obj]["native"][arm],
          block,
          cols,
          writer,
        )
  with (
    ZipFile(temp, "a", ZIP_DEFLATED, compresslevel=6) as packed,
    packed.open("control-steps.csv", "w") as buf,
    TextIOWrapper(buf, encoding="utf-8", newline="") as dest,
  ):
    writer = csv.writer(dest)
    writer.writerow(
      [
        "object",
        "control_step",
        "sim_s",
        "wrist_online_interval_ms",
        "aude_online_interval_ms",
        "wrist_gain_nat",
        "aude_gain_nat",
      ]
    )
    for obj, pair in frames.items():
      wrist_ms, wrist_gain = pair["wrist-dp"]
      aude_ms, aude_gain = pair["native-aude"]
      for tick, values in enumerate(
        zip(wrist_ms, aude_ms, wrist_gain, aude_gain, strict=True), 1
      ):
        writer.writerow([obj, tick, f"{tick * timing.DT:.2f}", *map(float, values)])
  temp.replace(out / "plan-events.zip")
  result["events_sha256"] = digest(out / "plan-events.zip")
  for arm in ARMS:
    pieces = [o["arms"][arm] for o in result["objects"].values()]
    pooled = [np.concatenate(a) for a in zip(*cols[arm], strict=True)]
    win = {}
    for s in map(str, HORIZONS):
      entries = [p["windows"][s] for p in pieces]
      counts = {"plans", "full_shared", "outer_proposals"} & entries[0].keys()
      for name in counts:
        assert len({w[name] for w in entries}) == 1
      win[s] = {
        k: int(entries[0][k])
        if k in counts
        else float(np.mean([w[k] for w in entries]))
        for k in entries[0]
      }
    result["mean"][arm] = dict(
      windows=win,
      duration_ms=timings(*pooled),
      control_interval_ms=distribution(
        np.concatenate([pair[arm][0] for pair in frames.values()])
      ),
      clock={k: float(np.mean([p["clock"][k] for p in pieces])) for k in TOTALS},
    )
  write(out / "planning.json", result)
  return result


def figure(data):
  from .frontier_information_figures import COLORS, INK, MUTED, plt

  pairs = [item["arms"] for item in data["objects"].values()] + [data["mean"]]
  labels = [item["name"] for item in data["objects"].values()] + ["五物体平均"]
  y = np.arange(len(labels))
  with plt.rc_context(
    {
      "font.family": ["Microsoft YaHei", "DejaVu Sans"],
      "axes.unicode_minus": False,
      "font.size": 11,
    }
  ):
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(12.5, 6.6))
    fig.subplots_adjust(left=0.138, right=0.977, top=0.77, bottom=0.17, wspace=0.13)
    fig.text(
      0.055,
      0.934,
      "一次目标更新的收益与计算耗时",
      fontsize=21,
      weight="bold",
      color=INK,
    )
    fig.text(
      0.055,
      0.877,
      "五个物体各从同一初态执行两套完整系统，统计整个 600 仿真秒。",
      fontsize=11.5,
      color=MUTED,
    )
    for panel, ax in enumerate(axes):
      ax.set_axisbelow(True)
      ax.grid(axis="x", color="#dbe5e1", linewidth=0.8)
      ax.spines[["top", "right", "left"]].set_visible(False)
      ax.spines["bottom"].set_color("#a2b4ac")
      ax.tick_params(axis="both", length=0, colors=MUTED)
      ax.axhspan(len(y) - 1.47, len(y) - 0.53, color="#edf4f0", zorder=0)
      max_value = 0.0
      for j, (arm, label) in enumerate(ARMS.items()):
        vals = np.array(
          [
            p[arm]["windows"]["600"]["nat_per_shared"]
            if panel == 0
            else p[arm]["duration_ms"]["all"]["mean"]
            for p in pairs
          ]
        )
        max_value = max(max_value, float(np.max(vals)))
        bars = ax.barh(
          y + (j - 0.5) * 0.33, vals, height=0.27, color=COLORS[j], label=label
        )
        ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=9, color=INK)
      ax.set_xlim(0, max_value * 1.22)
      ax.set_ylim(len(y) - 0.50, -0.50)
      ax.set_yticks(y, labels=labels)
      ax.set_title(
        ("目标生效后前 180 ms 的实触增益", "每次更新目标的平均计算耗时")[panel],
        fontsize=13,
        weight="bold",
        loc="left",
        color=INK,
        pad=9,
      )
      ax.set_xlabel(
        ("信息增益（nat / 次）", "规划耗时（ms / 次）")[panel], color=MUTED, labelpad=6
      )
    fig.legend(
      *axes[0].get_legend_handles_labels(),
      loc="lower right",
      bbox_to_anchor=(0.985, 0.851),
      frameon=False,
      ncol=2,
      fontsize=11,
    )
    fig.text(
      0.055,
      0.043,
      "Aude 每 180 ms 更新局部腕指目标，每五次另算一次全局提案，右图已把这部分耗时计入当次。",
      fontsize=10,
      color=MUTED,
    )
    fig.savefig(timing.OUT / "per-plan-comparison.png", dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
  figure(report())
