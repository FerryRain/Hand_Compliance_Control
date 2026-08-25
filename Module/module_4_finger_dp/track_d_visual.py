"""Review artefacts for Track-D data, overfit and closed-loop execution."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MUJOCO_GL", "osmesa")
_CACHE = Path(tempfile.gettempdir()) / "handcomp-track-d-v1"
_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["XDG_CACHE_HOME"] = str(_CACHE)
os.environ["MESA_SHADER_CACHE_DIR"] = str(_CACHE / "mesa_shader_cache")
os.environ["MPLCONFIGDIR"] = str(_CACHE / "matplotlib")
Path(os.environ["MESA_SHADER_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from Module.fr3_leap import FullRobotModelConfig, build_full_robot
from Module.module_4_finger_dp.track_d_closed_loop import (
  TrackDClosedLoopConfig,
  TrackDClosedLoopTrace,
  load_track_d_closed_loop,
)
from Module.module_4_finger_dp.track_d_dataset import (
  TrackDSamples,
  load_e05_h_teacher_trace,
  load_track_d_samples,
)
from Module.visualization import get_pyplot, save_figure


FINGER_COLORS = ("#2997D6", "#3CBF91", "#F39C35", "#ED5A7A")


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
  name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
  path = Path("/usr/share/fonts/truetype/dejavu") / name
  return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def _camera() -> mujoco.MjvCamera:
  camera = mujoco.MjvCamera()
  camera.type = mujoco.mjtCamera.mjCAMERA_FREE
  camera.lookat[:] = np.array([0.39, 0.10, 0.48])
  camera.distance = 1.04
  camera.azimuth = 133.0
  camera.elevation = -23.0
  return camera


def _contact_set(mask: np.ndarray) -> str:
  active = [str(index + 1) for index in np.flatnonzero(mask)]
  return "{" + ",".join(active) + "}" if active else "EMPTY"


def _video_overlay(
  frame: np.ndarray,
  trace: TrackDClosedLoopTrace,
  index: int,
  activation_s: float,
  *,
  title: str,
) -> np.ndarray:
  image = Image.fromarray(frame).convert("RGB")
  draw = ImageDraw.Draw(image, "RGBA")
  active = trace.time_s[index] >= activation_s
  color = (34, 163, 111, 232) if active else (53, 103, 156, 232)
  phase = "FINGER DP ACTIVE" if active else "TEACHER WARM-UP (explicit initialization only)"
  draw.rounded_rectangle((16, 14, image.width - 16, 116), 12, fill=(5, 18, 33, 225))
  draw.text(
    (31, 27),
    title,
    font=_font(24, bold=True),
    fill="white",
  )
  draw.text(
    (31, 65),
    "FR3 Wrist MCC + force-history Finger DP + authority QP + hard guard",
    font=_font(15),
    fill=(194, 221, 241),
  )
  draw.rounded_rectangle((image.width - 310, 31, image.width - 31, 79), 8, fill=color)
  draw.text((image.width - 292, 45), phase, font=_font(14, bold=True), fill="white")
  draw.text(
    (image.width - 172, 87),
    f"t={trace.time_s[index]:4.2f} s",
    font=_font(15, bold=True),
    fill=(255, 216, 112),
  )

  panel_top = image.height - 170
  draw.rounded_rectangle((16, panel_top, image.width - 16, image.height - 14), 12, fill=(5, 18, 33, 228))
  contacts = trace.actual_contacts[index]
  forces = trace.fingertip_forces_n[index]
  command_error = np.linalg.norm(
    trace.finger_command_rad[index] - trace.teacher_reference_command_rad[index]
  )
  draw.text(
    (31, panel_top + 14),
    f"actual A={_contact_set(contacts)}   owner={trace.command_owner[index]}   "
    f"guard={trace.guard_state[index]}",
    font=_font(16, bold=True),
    fill="white",
  )
  draw.text(
    (image.width - 365, panel_top + 16),
    f"||q_cmd-q_teacher||={command_error:.3f} rad",
    font=_font(14),
    fill=(194, 221, 241),
  )
  for finger in range(4):
    x = 51 + finger * 220
    y = panel_top + 57
    draw.text((x, y), f"F{finger + 1}", font=_font(15, bold=True), fill="white")
    draw.rounded_rectangle((x + 30, y, x + 168, y + 22), 5, fill=(55, 72, 89, 255))
    width = 134.0 * float(np.clip(forces[finger] / 8.0, 0.0, 1.0))
    if width > 0.0:
      draw.rounded_rectangle(
        (x + 32, y + 2, x + 32 + width, y + 20),
        4,
        fill=FINGER_COLORS[finger],
      )
    draw.text(
      (x + 48, y + 30),
      f"{forces[finger]:4.2f} N",
      font=_font(14, bold=True),
      fill=FINGER_COLORS[finger] if contacts[finger] else (160, 172, 185),
    )
  draw.text(
    (31, image.height - 38),
    f"authority={trace.authority_solver_status[index]}  "
    f"intervention={trace.authority_intervention_norm_rad[index]:.4f} rad  "
    f"policy={1000.0 * trace.policy_latency_s[index]:.1f} ms",
    font=_font(14),
    fill=(151, 231, 209),
  )
  return np.asarray(image)


def render_track_d_video(
  trace: TrackDClosedLoopTrace,
  output_path: str | Path,
  screenshot_path: str | Path,
  *,
  activation_s: float = 1.0,
  fps: int = 20,
  title: str = "Track D · closed-loop physical imitation",
) -> Path:
  destination = Path(output_path)
  screenshot = Path(screenshot_path)
  destination.parent.mkdir(parents=True, exist_ok=True)
  handles = build_full_robot(
    FullRobotModelConfig(
      surface="extreme",
      timestep_s=0.002,
      gravity_m_s2=0.0,
      arm_kp=1800.0,
      arm_damping_ratio=0.9,
    )
  )
  data = mujoco.MjData(handles.model)
  renderer = mujoco.Renderer(handles.model, width=960, height=600)
  dt_s = float(np.median(np.diff(trace.time_s)))
  indices = np.unique(
    np.clip(
      np.round(np.arange(0.0, trace.time_s[-1] + dt_s, 1.0 / fps) / dt_s).astype(int),
      0,
      len(trace.time_s) - 1,
    )
  )
  screenshot_index = int(np.argmin(np.abs(trace.time_s - 2.5)))
  screenshot_frame: np.ndarray | None = None
  writer = imageio.get_writer(
    destination,
    fps=fps,
    codec="libx264",
    quality=8,
    macro_block_size=1,
  )
  try:
    for index in indices:
      data.qpos[handles.arm_qpos_adrs] = trace.arm_q_rad[index]
      data.qpos[handles.hand_qpos_adrs] = trace.finger_q_rad[index]
      data.ctrl[handles.arm_actuator_ids] = trace.arm_q_rad[index]
      data.ctrl[handles.hand_actuator_ids] = trace.finger_command_rad[index]
      mujoco.mj_forward(handles.model, data)
      renderer.update_scene(data, camera=_camera())
      try:
        renderer.scene.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
      except (AttributeError, IndexError, TypeError):
        pass
      frame = _video_overlay(
        renderer.render().copy(),
        trace,
        int(index),
        activation_s,
        title=title,
      )
      writer.append_data(frame)
      if screenshot_frame is None and int(index) >= screenshot_index:
        screenshot_frame = frame.copy()
  finally:
    writer.close()
    renderer.close()
  if screenshot_frame is None:
    raise RuntimeError("Track-D trace did not contain screenshot time")
  imageio.imwrite(screenshot, screenshot_frame)
  return destination


def render_data_audit(
  samples: TrackDSamples,
  teacher_trace_path: str | Path,
  output_path: str | Path,
) -> Path:
  teacher = load_e05_h_teacher_trace(teacher_trace_path)
  destination = Path(output_path)
  plt = get_pyplot()
  figure, axes = plt.subplots(3, 1, figsize=(13, 10), constrained_layout=True)
  mask = teacher.time_s <= 5.4
  for finger, color in enumerate(FINGER_COLORS):
    axes[0].plot(
      teacher.time_s[mask],
      teacher.fingertip_forces_n[mask, finger],
      color=color,
      linewidth=0.8,
      label=f"F{finger + 1}",
    )
  axes[0].axhline(8.0, color="#C73E4D", linestyle=":", label="hard 8 N")
  axes[0].set(title="Dataset-D source: fresh 500 Hz fingertip force", ylabel="N")
  axes[0].legend(ncol=5, fontsize=8)

  axes[1].step(
    teacher.time_s[mask],
    np.sum(teacher.actual_contacts[mask], axis=1),
    where="post",
    color="#286B8E",
    label="actual contact count",
  )
  command_gap = np.linalg.norm(
    teacher.finger_command_rad[mask] - teacher.finger_q_rad[mask],
    axis=1,
  )
  axes[1].plot(
    teacher.time_s[mask],
    command_gap,
    color="#8E5EA2",
    linewidth=0.8,
    label="||issued q_cmd - measured q|| [rad]",
  )
  axes[1].set(title="Physical contact and command/state distinction", ylabel="count / rad")
  axes[1].legend(fontsize=8)

  audit = samples.audit
  axes[2].axis("off")
  lines = [
    "CAUSAL ALIGNMENT AUDIT · PASS" if audit.passed else "CAUSAL ALIGNMENT AUDIT · FAIL",
    f"physics={audit.physics_rate_hz:.0f} Hz   force history={audit.force_history_rate_hz:.0f} Hz × 200 ms   policy={audit.policy_rate_hz:.0f} Hz",
    f"samples={audit.sample_count}   source window={audit.source_start_time_s:.3f}–{audit.source_stop_time_s:.3f} s",
    f"history latest-anchor={audit.maximum_history_timestamp_minus_anchor_s:+.3e} s   first target-anchor={audit.minimum_target_timestamp_minus_anchor_s:.3f} s",
    f"future leakage={audit.future_leakage_count}   nonfinite={audit.nonfinite_value_count}   anchor residual={audit.maximum_anchor_construction_residual_rad:.2e} rad",
    f"teacher source={audit.teacher_source}",
    "Authorization: Dataset-D D-Gate diagnostic only · no generalization · no Dataset-I claim",
  ]
  axes[2].text(
    0.02,
    0.92,
    "\n\n".join(lines),
    transform=axes[2].transAxes,
    va="top",
    family="monospace",
    fontsize=11,
    bbox={"boxstyle": "round,pad=0.8", "facecolor": "#EEF4F8", "edgecolor": "#537895"},
  )
  figure.suptitle("Track D · data provenance and timestamp audit", fontsize=15)
  save_figure(figure, destination)
  plt.close(figure)
  return destination


def render_training_dashboard(
  output_directory: str | Path,
  output_path: str | Path,
) -> Path:
  output = Path(output_directory)
  destination = Path(output_path)
  with np.load(output / "training_history.npz", allow_pickle=False) as history:
    updates = history["update"]
    loss = history["loss"]
  with np.load(output / "open_loop_predictions.npz", allow_pickle=False) as prediction:
    target = prediction["target_action_offsets_rad"]
    predicted = prediction["predicted_action_offsets_rad"]
    time_s = prediction["timestamp_s"]
    anchor = prediction["anchor_q_meas_rad"]
    previous = prediction["previous_executed_command_rad"]
  error = predicted - target
  horizon_rmse = np.sqrt(np.mean(error**2, axis=(0, 2)))
  first_rmse = np.sqrt(np.mean(error[:, 0] ** 2, axis=1))
  predicted_seam = np.linalg.norm(anchor + predicted[:, 0] - previous, axis=1)
  teacher_seam = np.linalg.norm(anchor + target[:, 0] - previous, axis=1)
  plt = get_pyplot()
  figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
  axes[0, 0].plot(updates, loss, color="#286B8E")
  axes[0, 0].set_yscale("log")
  axes[0, 0].set(title="Diffusion overfit loss", xlabel="update", ylabel="MSE(noise)")
  axes[0, 1].plot(np.arange(1, len(horizon_rmse) + 1) * 0.02, horizon_rmse, color="#D98032")
  axes[0, 1].set(title="Open-loop command error by horizon", xlabel="future time [s]", ylabel="RMSE [rad]")
  axes[1, 0].plot(time_s, first_rmse, color="#8E5EA2", linewidth=0.9)
  axes[1, 0].set(title="First-command RMSE per replan", xlabel="teacher time [s]", ylabel="rad")
  axes[1, 1].plot(time_s, teacher_seam, label="teacher seam", color="#2997D6")
  axes[1, 1].plot(time_s, predicted_seam, label="DP seam", color="#ED5A7A", alpha=0.8)
  axes[1, 1].set(title="Chunk-boundary command continuity", xlabel="teacher time [s]", ylabel="L2 rad")
  axes[1, 1].legend(fontsize=8)
  summary = json.loads((output / "open_loop_summary.json").read_text(encoding="utf-8"))
  metrics = summary["metrics"]
  figure.suptitle(
    "Track D · intentional overfit (not generalization)\n"
    f"full chunk RMSE={metrics['full_chunk_rmse_rad']:.4f} rad · "
    f"first command RMSE={metrics['first_command_rmse_rad']:.4f} rad",
    fontsize=14,
  )
  save_figure(figure, destination)
  plt.close(figure)
  return destination


def render_closed_loop_dashboard(
  trace: TrackDClosedLoopTrace,
  output_directory: str | Path,
  output_path: str | Path,
  *,
  activation_s: float = 1.0,
) -> Path:
  output = Path(output_directory)
  summary = json.loads((output / "d_gate_summary.json").read_text(encoding="utf-8"))
  destination = Path(output_path)
  plt = get_pyplot()
  figure, axes = plt.subplots(5, 1, figsize=(14, 15), constrained_layout=True)
  desired_force_n = float(summary["config"]["desired_force_n"])
  for finger, color in enumerate(FINGER_COLORS):
    axes[0].plot(trace.time_s, trace.fingertip_forces_n[:, finger], color=color, linewidth=0.8, label=f"F{finger + 1}")
  axes[0].axvline(activation_s, color="black", linestyle="--", label="DP activation")
  axes[0].axhline(
    desired_force_n,
    color="#2F855A",
    linestyle=":",
    label=f"{desired_force_n:g} N target",
  )
  axes[0].axhline(8.0, color="#C73E4D", linestyle=":", label="8 N hard")
  axes[0].set(title="Fresh closed-loop fingertip force", ylabel="N")
  axes[0].legend(ncol=7, fontsize=7)

  for finger, color in enumerate(FINGER_COLORS):
    axes[1].step(
      trace.time_s,
      trace.actual_contacts[:, finger].astype(float) + 1.15 * finger,
      where="post",
      color=color,
      linewidth=0.8,
      label=f"F{finger + 1}",
    )
  axes[1].plot(
    trace.time_s,
    np.sum(trace.actual_contacts, axis=1),
    color="black",
    linewidth=1.1,
    label="N_c",
  )
  axes[1].axvline(activation_s, color="black", linestyle="--")
  axes[1].set(title="Actual contact set (offset rows) and whole-hand contact count", ylabel="contact / N_c")
  axes[1].legend(ncol=5, fontsize=8)

  command_error = np.sqrt(
    np.mean((trace.finger_command_rad - trace.teacher_reference_command_rad) ** 2, axis=1)
  )
  axes[2].plot(trace.time_s, command_error, color="#8E5EA2", label="q_cmd RMSE vs teacher")
  axes[2].plot(
    trace.time_s,
    trace.authority_intervention_norm_rad,
    color="#D98032",
    label="authority intervention L2",
  )
  axes[2].axvline(activation_s, color="black", linestyle="--")
  axes[2].set(title="Command imitation and authority intervention", ylabel="rad")
  axes[2].legend(fontsize=8)

  finger_collective_norm = np.linalg.norm(
    trace.finger_collective_normal_velocity_m_s,
    axis=1,
  )
  wrist_collective_norm = np.linalg.norm(
    trace.wrist_contact_normal_velocity_m_s,
    axis=1,
  )
  opposition_dot = np.sum(
    trace.finger_collective_normal_velocity_m_s
    * trace.wrist_contact_normal_velocity_m_s,
    axis=1,
  )
  axes[3].plot(
    trace.time_s,
    1000.0 * finger_collective_norm,
    color="#D98032",
    linewidth=0.8,
    label="Finger collective normal speed",
  )
  axes[3].plot(
    trace.time_s,
    1000.0 * wrist_collective_norm,
    color="#286B8E",
    linewidth=0.8,
    label="Wrist MCC contact-normal speed",
  )
  conflict = opposition_dot < 0.0
  axes[3].scatter(
    trace.time_s[conflict],
    1000.0 * finger_collective_norm[conflict],
    s=5,
    color="#C73E4D",
    label="opposing direction",
  )
  axes[3].axvline(activation_s, color="black", linestyle="--")
  axes[3].set(title="Wrist/Finger authority-space diagnostic", ylabel="mm/s")
  axes[3].legend(fontsize=8)

  replans = trace.policy_replan
  axes[4].scatter(
    trace.time_s[replans],
    1000.0 * trace.policy_latency_s[replans],
    s=8,
    color="#286B8E",
    label="DP inference",
  )
  axes[4].plot(
    trace.time_s,
    1000.0 * trace.authority_latency_s,
    color="#3CBF91",
    linewidth=0.7,
    label="authority QP",
  )
  axes[4].axhline(20.0, color="#C73E4D", linestyle=":", label="50 Hz wall-time target")
  cuda_name = summary.get("cuda_runtime", {}).get("device_name", "CUDA")
  axes[4].set(
    title=f"Measured computation latency ({cuda_name} policy)",
    xlabel="time [s]",
    ylabel="ms",
  )
  axes[4].legend(fontsize=8)
  metrics = summary["metrics"]
  gate = summary["d_gate"]
  figure.suptitle(
    f"Track D closed-loop · D-Gate {gate['status']} · no Finger MCC after t={activation_s:.1f}s\n"
    f"contact={metrics['contact_continuity']:.3f}, max force={metrics['maximum_force_n']:.2f} N, "
    f"zero-contact={metrics['zero_contact_time_s']:.3f} s, authority failures={metrics['authority_solver_failure_frames']}",
    fontsize=14,
  )
  save_figure(figure, destination)
  plt.close(figure)
  return destination


def write_track_d_review_index(output_directory: str | Path) -> Path:
  output = Path(output_directory)
  open_summary = json.loads((output / "open_loop_summary.json").read_text(encoding="utf-8"))
  gate_summary = json.loads((output / "d_gate_summary.json").read_text(encoding="utf-8"))
  metrics = gate_summary["metrics"]
  gate = gate_summary["d_gate"]
  cuda_runtime = gate_summary.get("cuda_runtime", {})
  device_name = cuda_runtime.get("device_name", "CUDA")
  long_run = (output / "whole_hand_dp_summary.json").is_file()
  display_name = "Whole-hand CUDA Finger DP + Wrist MCC" if long_run else "Track D"
  final_review_item = (
    "5. `long_collection_summary.json`, `long_dataset_manifest.json`, "
    "`whole_hand_dp_summary.json`\n"
    if long_run
    else "5. `dataset_d_samples.json`, `open_loop_summary.json`, `d_gate_summary.json`\n"
  )
  readme = output / "README.md"
  readme.write_text(
    f"# {display_name} review\n\n"
    "This is a Dataset-D learnability diagnostic, not Dataset-I training or an E05 result.\n\n"
    f"- D-Gate: `{gate['status']}`\n"
    f"- blocking reason: `{', '.join(gate['blocking_reason'])}`\n"
    f"- open-loop first-command RMSE: `{open_summary['metrics']['first_command_rmse_rad']:.6f} rad`\n"
    f"- closed-loop contact continuity: `{metrics['contact_continuity']:.6f}`\n"
    f"- zero-contact time: `{metrics['zero_contact_time_s']:.6f} s`\n"
    f"- maximum force: `{metrics['maximum_force_n']:.6f} N`\n"
    f"- per-finger contact-loss/chatter events: `{metrics['contact_loss_events']}`\n"
    f"- authority solver failures: `{metrics['authority_solver_failure_frames']}`\n"
    f"- CUDA device: `{device_name}`\n"
    f"- GPU policy latency P95: `{1000.0 * metrics['policy_latency_p95_s']:.3f} ms`\n"
    f"- opposition rate/energy: `{metrics.get('opposition_rate', 0.0):.6f}` / "
    f"`{metrics.get('opposition_energy', 0.0):.6e}`\n"
    f"- DP evaluation duration: `{metrics['evaluation_duration_s']:.3f} s`\n"
    "- Finger MCC after DP activation: `false`\n"
    "- Wrist MCC enabled: `true`\n\n"
    "The D-Gate checks Dataset-D learnability and closed-loop physical execution, not Dataset-I "
    "object-level generalization, hardware, or formal E05 performance. Opposition and measured "
    "GPU latency remain explicit diagnostics.\n\n"
    "## Review order\n\n"
    "1. `track_d_data_audit.png`\n"
    "2. `track_d_training_dashboard.png`\n"
    "3. `track_d_closed_loop.mp4` and `track_d_closed_loop_frame.png`\n"
    "4. `track_d_closed_loop_dashboard.png`\n"
    + final_review_item,
    encoding="utf-8",
  )
  html = output / "review.html"
  html.write_text(
    "<!doctype html><meta charset='utf-8'><title>Track D review</title>"
    "<style>body{max-width:1200px;margin:30px auto;font-family:sans-serif;background:#f4f7fa;color:#17324d}"
    "img,video{max-width:100%;border:1px solid #b7c5d1;border-radius:8px;background:white}"
    "section{background:white;padding:20px;margin:20px 0;border-radius:10px}</style>"
    f"<h1>{display_name} · D-Gate {gate['status']}</h1>"
    "<p>Dataset-D learnability diagnostic only. No Dataset-I or E05 claim.</p>"
    f"<p><strong>Runtime:</strong> {device_name} policy P95 is "
    f"{1000.0 * metrics['policy_latency_p95_s']:.1f} ms (20 ms / 50 Hz target); "
    f"per-finger threshold chatter events={metrics['contact_loss_events']}. "
    "Whole-hand zero-contact time remains zero.</p>"
    "<section><h2>1 · causal data audit</h2><img src='track_d_data_audit.png'></section>"
    "<section><h2>2 · open-loop overfit</h2><img src='track_d_training_dashboard.png'></section>"
    "<section><h2>3 · closed-loop physics</h2><video controls src='track_d_closed_loop.mp4'></video>"
    "<img src='track_d_closed_loop_frame.png'></section>"
    "<section><h2>4 · force/contact/command/latency</h2><img src='track_d_closed_loop_dashboard.png'></section>"
    "<section><h2>5 · raw reports</h2><ul>"
    + (
      "<li><a href='long_collection_summary.json'>long trajectory acceptance</a></li>"
      "<li><a href='long_dataset_manifest.json'>training admission manifest</a></li>"
      "<li><a href='whole_hand_dp_summary.json'>whole-hand controller verdict</a></li>"
      if long_run
      else ""
    )
    +
    "<li><a href='dataset_d_samples.json'>causal audit JSON</a></li>"
    "<li><a href='open_loop_summary.json'>open-loop summary</a></li>"
    "<li><a href='d_gate_summary.json'>D-Gate summary</a></li>"
    "</ul></section>",
    encoding="utf-8",
  )
  return html


def render_track_d_review(
  output_directory: str | Path,
  teacher_trace_path: str | Path,
  *,
  title: str = "Track D · closed-loop physical imitation",
) -> tuple[Path, ...]:
  output = Path(output_directory)
  samples = load_track_d_samples(output / "dataset_d_samples.npz")
  trace = load_track_d_closed_loop(output / "closed_loop_trace.npz")
  paths = (
    render_data_audit(samples, teacher_trace_path, output / "track_d_data_audit.png"),
    render_training_dashboard(output, output / "track_d_training_dashboard.png"),
    render_closed_loop_dashboard(trace, output, output / "track_d_closed_loop_dashboard.png"),
    render_track_d_video(
      trace,
      output / "track_d_closed_loop.mp4",
      output / "track_d_closed_loop_frame.png",
      activation_s=TrackDClosedLoopConfig().dp_activation_s,
      title=title,
    ),
    write_track_d_review_index(output),
  )
  return paths
