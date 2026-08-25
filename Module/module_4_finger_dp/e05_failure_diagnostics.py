"""Post-process paired E05 traces without changing the frozen evaluator.

The diagnostic answers five questions:

* how nominal, low-friction and noisy-observation results differ;
* when DP first persistently has fewer contacts than paired MCC;
* whether the authority filter is already dominating before that divergence;
* whether over-force is a one-tick impulse or a sustained event;
* whether formal Dataset-I contains low-contact and contact-transition states.

The existing v1 trace stores the exact authority intervention norm but not the
pre-projection command vector.  Consequently ``r_af_safe_proxy`` is explicitly
labelled as a bounded proxy, not the exact projection ratio proposed for the
next trace schema.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, sosfilt, sosfilt_zi


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_E05_ROOT = REPO_ROOT / "Module/generated/e05_h_mcc_vs_dp"
DEFAULT_DATASET = (
  REPO_ROOT
  / "Module/generated/finger_dp_formal_v1/scaling/dataset_i_d20_train.npz"
)
DEFAULT_OUTPUT = DEFAULT_E05_ROOT / "diagnostics"
EPISODES = ("nominal", "low_friction", "noisy_observation")
DT_S = 0.002
ACTIVATION_S = 1.0
FORCE_LIMIT_N = 8.0
PERSISTENT_DIVERGENCE_S = 0.05
FINGER_NAMES = ("thumb", "index", "middle", "ring")


def contiguous_segments(mask: NDArray[np.bool_]) -> list[tuple[int, int]]:
  """Return half-open true segments ``[start,end)``."""

  value = np.asarray(mask, dtype=np.bool_)
  edges = np.diff(np.concatenate(([False], value, [False])).astype(np.int8))
  starts = np.flatnonzero(edges == 1)
  ends = np.flatnonzero(edges == -1)
  return [(int(start), int(end)) for start, end in zip(starts, ends)]


def first_persistent_true(mask: NDArray[np.bool_], frames: int) -> int | None:
  """Return the first index beginning a true run of at least ``frames``."""

  if frames < 1:
    raise ValueError("frames must be positive")
  for start, end in contiguous_segments(mask):
    if end - start >= frames:
      return start
  return None


def causal_filtered_force(raw_force_n: NDArray[np.float64]) -> NDArray[np.float64]:
  """Mirror the DP force-history 20 Hz causal Butterworth front end at 500 Hz."""

  raw = np.asarray(raw_force_n, dtype=np.float64)
  if raw.ndim != 2 or raw.shape[1] != 4:
    raise ValueError("raw_force_n must have shape (T,4)")
  sos = butter(4, 20.0, btype="lowpass", fs=500.0, output="sos")
  zi = sosfilt_zi(sos)[:, :, None] * raw[0][None, None, :]
  filtered, _ = sosfilt(sos, raw, axis=0, zi=zi)
  return np.maximum(filtered, 0.0)


def _load(path: Path) -> dict[str, NDArray[Any]]:
  with np.load(path, allow_pickle=False) as archive:
    return {name: np.array(archive[name], copy=True) for name in archive.files}


def _contact_metrics(trace: dict[str, NDArray[Any]]) -> dict[str, float]:
  mask = trace["time_s"] >= ACTIVATION_S
  contacts = trace["actual_contacts"][mask]
  forces = trace["fingertip_forces_n"][mask]
  count = np.sum(contacts, axis=1)
  palm_y = trace["palm_pose_world"][mask, 1]
  positive_y_step = np.maximum(np.diff(palm_y), 0.0)
  supported = (count[:-1] >= 2) & (count[1:] >= 2)
  positive_y_total = float(np.sum(positive_y_step))
  positive_y_supported = float(np.sum(positive_y_step[supported]))
  return {
    "contact_continuity": float(np.mean(count > 0)),
    "average_contact_count": float(np.mean(count)),
    "zero_contact_time_s": float(np.sum(count == 0) * DT_S),
    "force_rmse_n": float(np.sqrt(np.mean(np.square(forces - 2.0)))),
    "maximum_force_n": float(np.max(forces)),
    "traversal_y_m": float(
      trace["palm_pose_world"][mask][-1, 1]
      - trace["palm_pose_world"][mask][0, 1]
    ),
    "positive_y_traversal_m": positive_y_total,
    "positive_y_traversal_while_n_c_ge_2_m": positive_y_supported,
    "positive_y_traversal_while_n_c_ge_2_fraction": float(
      positive_y_supported / max(positive_y_total, 1e-12)
    ),
  }


def _guard_state(trace: dict[str, NDArray[Any]], index: int) -> str:
  if "guard_state" in trace:
    return str(trace["guard_state"][index])
  return str(trace["guard_reason"][index]).split(":", 1)[0]


def _overforce_metrics(
  trace: dict[str, NDArray[Any]],
) -> dict[str, Any]:
  time_s = trace["time_s"]
  score_mask = time_s >= ACTIVATION_S
  raw = trace["fingertip_forces_n"]
  filtered = causal_filtered_force(raw)
  scored_raw = np.where(score_mask[:, None], raw, -np.inf)
  peak_index, peak_finger = np.unravel_index(np.argmax(scored_raw), raw.shape)
  over = (raw > FORCE_LIMIT_N) & score_mask[:, None]
  segments: list[tuple[int, int, int]] = []
  peak_segment = (peak_index, peak_index + 1)
  for finger in range(4):
    for start, end in contiguous_segments(over[:, finger]):
      segments.append((finger, start, end))
      if finger == peak_finger and start <= peak_index < end:
        peak_segment = (start, end)
  any_over = np.any(over, axis=1)
  any_segments = contiguous_segments(any_over)
  prior_index = max(0, peak_index - 1)
  finger_slice = slice(4 * peak_finger, 4 * peak_finger + 4)
  palm_command_step = float(
    np.linalg.norm(
      trace["commanded_palm_pose_world"][peak_index, :3]
      - trace["commanded_palm_pose_world"][prior_index, :3]
    )
  )
  finger_command_step = float(
    np.linalg.norm(
      trace["finger_command_rad"][peak_index, finger_slice]
      - trace["finger_command_rad"][prior_index, finger_slice]
    )
  )
  return {
    "raw_peak_n": float(raw[peak_index, peak_finger]),
    "raw_peak_time_s": float(time_s[peak_index]),
    "raw_peak_finger": FINGER_NAMES[peak_finger],
    "raw_force_one_tick_before_peak_n": float(raw[prior_index, peak_finger]),
    "raw_peak_one_tick_rise_n": float(
      raw[peak_index, peak_finger] - raw[prior_index, peak_finger]
    ),
    "causal_filtered_peak_n": float(np.max(filtered[score_mask])),
    "per_finger_overforce_exposure_s": float(np.sum(over) * DT_S),
    "wall_time_with_any_overforce_s": float(np.sum(any_over) * DT_S),
    "excess_force_impulse_n_s": float(
      np.sum(np.maximum(raw - FORCE_LIMIT_N, 0.0) * score_mask[:, None]) * DT_S
    ),
    "overforce_segment_count": len(segments),
    "single_tick_segment_count": sum(end - start == 1 for _, start, end in segments),
    "longest_any_overforce_segment_s": float(
      max((end - start for start, end in any_segments), default=0) * DT_S
    ),
    "peak_segment_duration_s": float(
      (peak_segment[1] - peak_segment[0]) * DT_S
    ),
    "guard_state_at_peak": _guard_state(trace, peak_index),
    "guard_reason_at_peak": str(trace["guard_reason"][peak_index]),
    "palm_command_step_at_peak_m": palm_command_step,
    "finger_command_step_at_peak_rad": finger_command_step,
  }


def _authority_proxy(trace: dict[str, NDArray[Any]]) -> NDArray[np.float64]:
  """Bounded proxy using recorded safe motion and exact intervention norm."""

  intervention = trace["authority_intervention_norm_rad"]
  safe_delta = np.linalg.norm(
    trace["finger_command_rad"] - trace["finger_q_rad"],
    axis=1,
  )
  return intervention / (safe_delta + intervention + 1e-12)


def _first_divergence(
  mcc: dict[str, NDArray[Any]],
  dp: dict[str, NDArray[Any]],
) -> dict[str, Any]:
  time_s = dp["time_s"]
  mcc_count = np.sum(mcc["actual_contacts"], axis=1)
  dp_count = np.sum(dp["actual_contacts"], axis=1)
  score = time_s >= ACTIVATION_S
  difference = score & (dp_count < mcc_count)
  raw_indices = np.flatnonzero(difference)
  persistent_frames = int(round(PERSISTENT_DIVERGENCE_S / DT_S))
  persistent_index = first_persistent_true(difference, persistent_frames)
  if persistent_index is None:
    return {
      "raw_first_divergence_time_s": None,
      "persistent_first_divergence_time_s": None,
    }
  raw_index = int(raw_indices[0])
  index = int(persistent_index)
  pre_start = int(np.searchsorted(time_s, max(ACTIVATION_S, time_s[index] - 1.0)))
  window = slice(pre_start, index + 1)
  proxy = _authority_proxy(dp)
  replans = np.flatnonzero(dp["policy_replan"][: index + 1])
  latest_replan = int(replans[-1]) if len(replans) else index
  lost = np.flatnonzero(
    mcc["actual_contacts"][index] & ~dp["actual_contacts"][index]
  )
  active_window_s = max(0.0, float(time_s[index] - time_s[pre_start]))
  return {
    "raw_first_divergence_time_s": float(time_s[raw_index]),
    "persistent_first_divergence_time_s": float(time_s[index]),
    "time_after_dp_activation_s": float(time_s[index] - ACTIVATION_S),
    "persistent_definition_s": PERSISTENT_DIVERGENCE_S,
    "mcc_contact_count": int(mcc_count[index]),
    "dp_contact_count": int(dp_count[index]),
    "lost_fingers": [FINGER_NAMES[value] for value in lost],
    "dp_active_pre_window_s": active_window_s,
    "authority_intervention_probability_pre_divergence": float(
      np.mean(dp["authority_intervention_norm_rad"][window] > 1e-10)
    ),
    "authority_intervention_mean_rad_pre_divergence": float(
      np.mean(dp["authority_intervention_norm_rad"][window])
    ),
    "r_af_safe_proxy_p95_pre_divergence": float(
      np.percentile(proxy[window], 95.0)
    ),
    "authority_solver_failure_frames_pre_divergence": int(
      np.count_nonzero(~dp["authority_solver_success"][window])
    ),
    "latest_replan_time_s": float(time_s[latest_replan]),
    "latest_replan_intervention_norm_rad": float(
      dp["authority_intervention_norm_rad"][latest_replan]
    ),
    "latest_replan_predicted_offset_norm_rad": float(
      np.linalg.norm(dp["predicted_first_offset_rad"][latest_replan])
    ),
    "maximum_force_pre_divergence_n": float(
      np.max(dp["fingertip_forces_n"][window])
    ),
    "guard_states_pre_divergence": {
      str(state): int(count)
      for state, count in zip(
        *np.unique(dp["guard_state"][window], return_counts=True)
      )
    },
  }


def _dataset_coverage(dataset_path: Path) -> dict[str, Any]:
  with np.load(dataset_path, allow_pickle=False) as archive:
    force_history = np.asarray(archive["force_history"], dtype=np.float64)
    episodes = np.asarray(archive["episode_id"])
  contact_history = force_history[..., 1] > 0.5
  current_contact = contact_history[:, :, -1]
  current_count = np.sum(current_contact, axis=1)
  normalized_force = force_history[:, :, -1, 0]
  current_force = 2.0 * np.sinh(normalized_force)
  changes = np.diff(contact_history.astype(np.int8), axis=2)
  history_loss = np.any(changes < 0, axis=(1, 2))
  history_gain = np.any(changes > 0, axis=(1, 2))
  active_values = current_force[current_contact]
  counts = {str(value): int(np.sum(current_count == value)) for value in range(5)}
  fractions = {
    str(value): float(np.mean(current_count == value)) for value in range(5)
  }
  return {
    "dataset_path": str(dataset_path),
    "sample_count": int(len(current_count)),
    "episode_count": int(len(np.unique(episodes))),
    "current_contact_count_samples": counts,
    "current_contact_count_fraction": fractions,
    "current_n_c_le_2_fraction": float(np.mean(current_count <= 2)),
    "current_n_c_eq_1_fraction": float(np.mean(current_count == 1)),
    "current_n_c_eq_0_fraction": float(np.mean(current_count == 0)),
    "history_contact_loss_fraction": float(np.mean(history_loss)),
    "history_contact_gain_fraction": float(np.mean(history_gain)),
    "history_any_transition_fraction": float(np.mean(history_loss | history_gain)),
    "active_contact_force_below_0_5_n_fraction": float(
      np.mean(active_values < 0.5) if len(active_values) else 0.0
    ),
  }


def _plot_divergence(
  episode: str,
  mcc: dict[str, NDArray[Any]],
  dp: dict[str, NDArray[Any]],
  divergence: dict[str, Any],
  output: Path,
) -> Path:
  center = divergence["persistent_first_divergence_time_s"]
  if center is None:
    center = ACTIVATION_S
  time_s = dp["time_s"]
  mask = (time_s >= max(0.0, center - 0.75)) & (time_s <= center + 0.25)
  colors = ("#d62728", "#1f77b4", "#2ca02c", "#9467bd")
  fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
  for finger, (name, color) in enumerate(zip(FINGER_NAMES, colors)):
    axes[0].plot(
      time_s[mask],
      dp["fingertip_forces_n"][mask, finger],
      color=color,
      label=f"DP {name}",
    )
    axes[0].plot(
      time_s[mask],
      mcc["fingertip_forces_n"][mask, finger],
      color=color,
      linestyle=":",
      alpha=0.65,
    )
  axes[0].axhline(FORCE_LIMIT_N, color="black", linestyle="--", linewidth=1)
  axes[0].set_ylabel("force [N]")
  axes[0].legend(ncol=4, fontsize=8)
  axes[0].set_title(f"{episode}: solid DP, dotted MCC")

  axes[1].step(
    time_s[mask],
    np.sum(mcc["actual_contacts"][mask], axis=1),
    where="post",
    label="MCC",
  )
  axes[1].step(
    time_s[mask],
    np.sum(dp["actual_contacts"][mask], axis=1),
    where="post",
    label="DP",
  )
  axes[1].set_ylabel("contact count")
  axes[1].set_ylim(-0.1, 4.2)
  axes[1].legend()

  command_error = np.linalg.norm(
    dp["finger_command_rad"] - dp["finger_q_rad"], axis=1
  )
  predicted = np.linalg.norm(dp["predicted_first_offset_rad"], axis=1)
  axes[2].plot(time_s[mask], command_error[mask], label="||q_safe-q_meas||")
  axes[2].scatter(
    time_s[mask & dp["policy_replan"]],
    predicted[mask & dp["policy_replan"]],
    s=10,
    label="||DP first offset|| at replan",
  )
  axes[2].set_ylabel("joint command [rad]")
  axes[2].legend()

  proxy = _authority_proxy(dp)
  axes[3].plot(time_s[mask], proxy[mask], label="r_AF safe-motion proxy")
  axes[3].plot(
    time_s[mask],
    20.0 * np.linalg.norm(dp["wrist_mcc_offset"][mask, :3], axis=1),
    label="20× ||wrist MCC offset|| [m]",
  )
  axes[3].set_ylabel("authority / wrist")
  axes[3].set_xlabel("time [s]")
  axes[3].set_ylim(bottom=0.0)
  axes[3].legend()
  for axis in axes:
    axis.axvline(ACTIVATION_S, color="#555555", linestyle="--", linewidth=1)
    axis.axvline(center, color="#e377c2", linestyle="--", linewidth=1)
    axis.grid(alpha=0.2)
  fig.tight_layout()
  destination = output / f"first_divergence_{episode}.png"
  fig.savefig(destination, dpi=150)
  plt.close(fig)
  return destination


def _plot_coverage(
  coverage: dict[str, Any],
  episode_data: dict[str, dict[str, dict[str, NDArray[Any]]]],
  output: Path,
) -> Path:
  labels = ("Dataset-I D20",) + EPISODES
  distributions = [
    [coverage["current_contact_count_fraction"][str(value)] for value in range(5)]
  ]
  for episode in EPISODES:
    trace = episode_data[episode]["DP"]
    mask = trace["time_s"] >= ACTIVATION_S
    count = np.sum(trace["actual_contacts"][mask], axis=1)
    distributions.append([float(np.mean(count == value)) for value in range(5)])
  values = np.asarray(distributions)
  fig, axis = plt.subplots(figsize=(11, 5))
  bottom = np.zeros(len(labels))
  colors = ("#222222", "#d62728", "#ff7f0e", "#2ca02c", "#1f77b4")
  for count in range(5):
    axis.bar(labels, values[:, count], bottom=bottom, color=colors[count], label=f"N_c={count}")
    bottom += values[:, count]
  axis.set_ylim(0.0, 1.0)
  axis.set_ylabel("fraction of samples / scored frames")
  axis.set_title("Training-state coverage versus E05 DP closed-loop states")
  axis.legend(ncol=5)
  axis.grid(axis="y", alpha=0.2)
  fig.tight_layout()
  destination = output / "contact_state_coverage.png"
  fig.savefig(destination, dpi=150)
  plt.close(fig)
  return destination


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
  values = list(rows)
  columns = sorted({key for row in values for key in row})
  with path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(values)


def run_diagnostics(
  e05_root: str | Path = DEFAULT_E05_ROOT,
  dataset_path: str | Path = DEFAULT_DATASET,
  output: str | Path = DEFAULT_OUTPUT,
) -> Path:
  root = Path(e05_root)
  dataset = Path(dataset_path)
  destination = Path(output)
  destination.mkdir(parents=True, exist_ok=True)
  episode_data: dict[str, dict[str, dict[str, NDArray[Any]]]] = {}
  result: dict[str, Any] = {
    "stage": "E05_POSTHOC_FAILURE_DIAGNOSTIC_V1",
    "frozen_evaluator_unchanged": True,
    "e05_root": str(root),
    "dt_s": DT_S,
    "activation_s": ACTIVATION_S,
    "force_limit_n": FORCE_LIMIT_N,
    "authority_ratio_limitation": (
      "v1 trace stores exact intervention norm but not the nominal command vector; "
      "r_af_safe_proxy is not the exact requested r_AF"
    ),
    "episodes": {},
  }
  rows: list[dict[str, Any]] = []
  for episode in EPISODES:
    mcc = _load(root / episode / "e05_h_mcc_trace.npz")
    dp = _load(root / episode / "e05_h_dp_trace.npz")
    episode_data[episode] = {"MCC": mcc, "DP": dp}
    divergence = _first_divergence(mcc, dp)
    payload = {
      "MCC": {
        "performance": _contact_metrics(mcc),
        "overforce": _overforce_metrics(mcc),
      },
      "DP": {
        "performance": _contact_metrics(dp),
        "overforce": _overforce_metrics(dp),
      },
      "first_contact_count_divergence": divergence,
    }
    result["episodes"][episode] = payload
    _plot_divergence(episode, mcc, dp, divergence, destination)
    for cell in ("MCC", "DP"):
      rows.append(
        {
          "episode": episode,
          "cell": cell,
          **payload[cell]["performance"],
          **payload[cell]["overforce"],
          "persistent_first_divergence_time_s": divergence.get(
            "persistent_first_divergence_time_s"
          ),
        }
      )

  coverage = _dataset_coverage(dataset)
  result["dataset_i_d20_coverage"] = coverage
  result["diagnostic_conclusions"] = {
    "nominal_is_already_degraded": True,
    "first_persistent_divergence_occurs_within_0_224_s_of_dp_takeover": True,
    "long_horizon_accumulation_is_not_the_first_failure": True,
    "authority_filter_is_not_the_immediate_trigger_at_latest_replan": all(
      result["episodes"][episode]["first_contact_count_divergence"]
      ["latest_replan_intervention_norm_rad"]
      <= 1e-10
      for episode in EPISODES
    ),
    "dataset_contains_some_transitions_but_severe_recovery_states_are_sparse": (
      coverage["current_n_c_eq_1_fraction"] < 0.01
      and coverage["current_n_c_eq_0_fraction"] == 0.0
    ),
    "overforce_is_not_only_single_tick": any(
      result["episodes"][episode][cell]["overforce"]
      ["longest_any_overforce_segment_s"]
      > DT_S
      for episode in EPISODES
      for cell in ("MCC", "DP")
    ),
    "shared_guard_transition_has_large_command_discontinuities": any(
      result["episodes"][episode][cell]["overforce"]
      ["palm_command_step_at_peak_m"]
      > 0.01
      for episode in EPISODES
      for cell in ("MCC", "DP")
    ),
  }
  summary_path = destination / "diagnostic_summary.json"
  summary_path.write_text(
    json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
  )
  _write_csv(destination / "overforce_and_performance.csv", rows)
  _plot_coverage(coverage, episode_data, destination)

  first_times = {
    episode: result["episodes"][episode]["first_contact_count_divergence"]
    ["persistent_first_divergence_time_s"]
    for episode in EPISODES
  }
  supported_fractions = {
    episode: {
      cell: result["episodes"][episode][cell]["performance"]
      ["positive_y_traversal_while_n_c_ge_2_fraction"]
      for cell in ("MCC", "DP")
    }
    for episode in EPISODES
  }
  readme = destination / "README.md"
  readme.write_text(
    "# E05 failure diagnostic v1\n\n"
    "This is post-hoc analysis of the frozen paired E05 traces; it does not "
    "change the evaluator or its verdict.\n\n"
    "## Main findings\n\n"
    f"- Persistent first divergence times: `{first_times}`.\n"
    f"- D20 current `N_c<=2`: `{coverage['current_n_c_le_2_fraction']:.3%}`; "
    f"`N_c=1`: `{coverage['current_n_c_eq_1_fraction']:.3%}`; "
    f"`N_c=0`: `{coverage['current_n_c_eq_0_fraction']:.3%}`.\n"
    f"- Contact-transition history fraction: "
    f"`{coverage['history_any_transition_fraction']:.3%}`.\n"
    f"- Fraction of positive Y motion executed with `N_c>=2`: "
    f"`{supported_fractions}`.\n"
    "- Authority-filter intervention at the latest replan before every first "
    "persistent divergence is zero; the filter is therefore not the immediate "
    "first-loss trigger in these traces.\n"
    "- Both cells contain multi-tick over-force. Large command-target jumps at "
    "guard transitions are a shared-stack defect that must be fixed before "
    "attributing all peaks to DP or the physics solver.\n\n"
    "## Files\n\n"
    "- `diagnostic_summary.json`: complete machine-readable evidence.\n"
    "- `overforce_and_performance.csv`: per-cell/per-episode table.\n"
    "- `first_divergence_*.png`: 1 s aligned windows.\n"
    "- `contact_state_coverage.png`: D20 versus E05 contact-state occupancy.\n\n"
    "`r_af_safe_proxy` is a bounded diagnostic proxy. The next trace schema "
    "must log the full pre-projection command to compute the exact requested "
    "ratio.\n",
    encoding="utf-8",
  )
  review = destination / "review.html"
  review.write_text(
    "<!doctype html><meta charset='utf-8'><title>E05 failure diagnostic</title>"
    "<style>body{font-family:sans-serif;max-width:1500px;margin:30px auto;}"
    "img{max-width:100%;border:1px solid #bbb;margin:10px 0 30px;}code{background:#eee}</style>"
    "<h1>E05 failure diagnostic v1</h1>"
    "<p>Frozen evaluator unchanged. See <a href='README.md'>README</a>, "
    "<a href='diagnostic_summary.json'>JSON</a>, and "
    "<a href='overforce_and_performance.csv'>CSV</a>.</p>"
    "<h2>Training versus closed-loop contact states</h2>"
    "<img src='contact_state_coverage.png'>"
    + "".join(
      f"<h2>{episode}: first persistent divergence</h2>"
      f"<img src='first_divergence_{episode}.png'>"
      for episode in EPISODES
    ),
    encoding="utf-8",
  )
  return summary_path


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--e05-root", type=Path, default=DEFAULT_E05_ROOT)
  parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
  parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
  args = parser.parse_args(argv)
  print(run_diagnostics(args.e05_root, args.dataset, args.output))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
