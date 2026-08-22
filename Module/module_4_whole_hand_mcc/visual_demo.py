"""Render formal FR3+Leap MCC traces into videos and a compact dashboard."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any

os.environ.setdefault("MUJOCO_GL", "osmesa")
cache = Path(tempfile.gettempdir()) / "handcomp-fr3-mesa"
cache.mkdir(parents=True, exist_ok=True)
os.environ["XDG_CACHE_HOME"] = str(cache)
os.environ["MPLCONFIGDIR"] = str(cache / "matplotlib")

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from Module.e05_physics.extreme_surface import profile_characteristics
from Module.fr3_leap import FullRobotModelConfig, build_full_robot
from Module.module_4_whole_hand_mcc.benchmark import (
  DEFAULT_OUTPUT_DIR,
  load_base_trace,
)
from Module.module_4_whole_hand_mcc.runner import E05MCCConfig, E05MCCTrace
from Module.visualization import get_pyplot, save_figure


DEFAULT_VISUAL_DIR = Path("Module/generated/visual_demo")
FINGER_COLORS = ("#2997D6", "#3CBF91", "#F39C35", "#ED5A7A")


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
  name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
  path = Path("/usr/share/fonts/truetype/dejavu") / name
  return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def _camera(
  lookat: tuple[float, float, float],
  *,
  distance: float,
  azimuth: float,
  elevation: float,
) -> mujoco.MjvCamera:
  camera = mujoco.MjvCamera()
  camera.type = mujoco.mjtCamera.mjCAMERA_FREE
  camera.lookat[:] = np.asarray(lookat)
  camera.distance = distance
  camera.azimuth = azimuth
  camera.elevation = elevation
  return camera


def _contact_set(mask: np.ndarray) -> str:
  members = [str(index + 1) for index, active in enumerate(mask) if active]
  return "{" + ",".join(members) + "}" if members else "EMPTY"


def _overlay(
  frame: np.ndarray,
  inset: np.ndarray,
  trace: E05MCCTrace,
  index: int,
  mode: str,
) -> np.ndarray:
  image = Image.fromarray(frame).convert("RGBA")
  inset_image = Image.fromarray(inset).convert("RGB")
  inset_width = 360
  inset_height = int(inset_image.height * inset_width / inset_image.width)
  inset_image = inset_image.resize((inset_width, inset_height), Image.Resampling.LANCZOS)
  image.alpha_composite(inset_image.convert("RGBA"), (image.width - inset_width - 24, 126))
  draw = ImageDraw.Draw(image)
  draw.rounded_rectangle((18, 14, image.width - 18, 112), 13, fill=(8, 22, 38, 218))
  title = "E05-F-MCC  |  prescribed FR3 wrist + full local Finger MCC" if mode == "E05-F-MCC" else "E05-H-MCC  |  FR3 Wrist MCC + resultant/internal coordinator"
  draw.text((34, 28), title, font=_font(22, bold=True), fill="white")
  draw.text(
    (34, 65),
    "23-DoF FR3+Leap  |  fixed world hfield  |  physical fingertip-belly contacts",
    font=_font(16),
    fill=(202, 224, 241, 255),
  )
  draw.text(
    (image.width - 185, 31),
    f"t = {trace.time_s[index]:4.1f} s",
    font=_font(20, bold=True),
    fill=(255, 215, 105, 255),
  )
  draw.rectangle(
    (image.width - inset_width - 26, 124, image.width - 22, 126 + inset_height + 4),
    outline=(235, 242, 249, 255),
    width=2,
  )
  draw.text(
    (image.width - inset_width - 18, 132),
    "CONTACT CLOSE-UP",
    font=_font(13, bold=True),
    fill=(255, 255, 255, 255),
    stroke_width=2,
    stroke_fill=(10, 25, 40, 255),
  )

  panel_top = image.height - 168
  draw.rounded_rectangle((18, panel_top, image.width - 18, image.height - 16), 13, fill=(8, 22, 38, 225))
  contacts = trace.actual_contacts[index]
  forces = trace.fingertip_forces_n[index]
  y_progress = 1000.0 * (
    trace.palm_pose_world[index, 1] - trace.palm_pose_world[0, 1]
  )
  draw.text(
    (34, panel_top + 14),
    f"actual A = {_contact_set(contacts)}    palm Y progress = {y_progress:5.1f} mm",
    font=_font(19, bold=True),
    fill="white",
  )
  desired_z = trace.desired_hand_wrench_world[index, 2]
  measured_z = trace.estimated_hand_wrench_world[index, 2]
  offset_z = 1000.0 * trace.wrist_compliance_offset[index, 2]
  draw.text(
    (image.width - 430, panel_top + 17),
    f"wrist Fz des/meas {desired_z:4.1f}/{measured_z:4.1f} N   dz {offset_z:+5.1f} mm",
    font=_font(15),
    fill=(202, 224, 241, 255),
  )
  bar_y = panel_top + 55
  for finger in range(4):
    x = 84 + finger * 207
    draw.text((x - 48, bar_y + 2), f"F{finger + 1}", font=_font(15, bold=True), fill="white")
    draw.rounded_rectangle((x, bar_y, x + 128, bar_y + 22), 5, fill=(62, 77, 93, 255))
    width = 124 * float(np.clip(forces[finger] / 8.0, 0.0, 1.0))
    if width > 0:
      draw.rounded_rectangle((x + 2, bar_y + 2, x + 2 + width, bar_y + 20), 3, fill=FINGER_COLORS[finger])
    draw.text(
      (x, bar_y + 29),
      f"{forces[finger]:4.2f} N",
      font=_font(14),
      fill=FINGER_COLORS[finger] if contacts[finger] else (180, 190, 201, 255),
    )
  footer = "Wrist MCC OFF · object fixed" if mode == "E05-F-MCC" else f"Wrist MCC ON · rank {int(trace.coordinator_rank[index])} · leakage {trace.coordinator_internal_leakage_n[index]:.3f} N"
  draw.text((34, image.height - 38), footer, font=_font(15, bold=True), fill=(151, 231, 209, 255))
  if trace.disturbance_active[index] and trace.time_s[index] < 9.5:
    draw.rounded_rectangle((28, 126, 275, 166), 8, fill=(207, 68, 86, 238))
    draw.text((43, 136), "WRIST STEP: +4 mm AWAY", font=_font(16, bold=True), fill="white")
  return np.asarray(image.convert("RGB"))


def render_video(
  trace: E05MCCTrace,
  mode: str,
  output_path: Path,
  screenshot_path: Path,
  *,
  fps: int = 12,
) -> Path:
  handles = build_full_robot(
    FullRobotModelConfig(
      surface="extreme",
      gravity_m_s2=0.0,
      arm_kp=1800.0,
      arm_damping_ratio=0.9,
    )
  )
  data = mujoco.MjData(handles.model)
  renderer = mujoco.Renderer(handles.model, width=960, height=540)
  close_renderer = mujoco.Renderer(handles.model, width=480, height=300)
  full_camera = _camera((0.34, 0.10, 0.43), distance=1.25, azimuth=133.0, elevation=-22.0)
  close_camera = _camera((0.53, 0.14, 0.49), distance=0.43, azimuth=130.0, elevation=-28.0)
  frame_times = np.arange(0.0, float(trace.time_s[-1]), 1.0 / fps)
  indices = np.unique(
    np.clip(np.round(frame_times / 0.002).astype(int), 0, len(trace.time_s) - 1)
  )
  screenshot_index = int(np.argmin(np.abs(trace.time_s - 8.2)))
  screenshot_frame: np.ndarray | None = None
  output_path.parent.mkdir(parents=True, exist_ok=True)
  writer = imageio.get_writer(
    output_path,
    fps=fps,
    codec="libx264",
    quality=8,
    macro_block_size=1,
  )
  try:
    for index in indices:
      data.qpos[handles.arm_qpos_adrs] = trace.arm_q_rad[index]
      data.qpos[handles.hand_qpos_adrs] = trace.finger_q_rad[index]
      data.ctrl[handles.arm_actuator_ids] = trace.arm_command_rad[index]
      data.ctrl[handles.hand_actuator_ids] = trace.finger_command_rad[index]
      mujoco.mj_forward(handles.model, data)
      renderer.update_scene(data, camera=full_camera)
      close_renderer.update_scene(data, camera=close_camera)
      try:
        renderer.scene.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
        close_renderer.scene.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
      except (AttributeError, IndexError, TypeError):
        pass
      frame = _overlay(
        renderer.render().copy(),
        close_renderer.render().copy(),
        trace,
        int(index),
        mode,
      )
      writer.append_data(frame)
      if screenshot_frame is None and int(index) >= screenshot_index:
        screenshot_frame = frame.copy()
  finally:
    writer.close()
    renderer.close()
    close_renderer.close()
  if screenshot_frame is None:
    raise RuntimeError("formal trace did not contain the screenshot target")
  imageio.imwrite(screenshot_path, screenshot_frame)
  return output_path


def render_dashboard(
  f_trace: E05MCCTrace,
  h_trace: E05MCCTrace,
  summary: dict[str, Any],
  output_path: Path,
) -> Path:
  plt = get_pyplot()
  figure, axes = plt.subplots(3, 2, figsize=(15, 12), constrained_layout=True)
  for trace, label, color in ((f_trace, "E05-F-MCC", "#277DA1"), (h_trace, "E05-H-MCC", "#43AA8B")):
    axes[0, 0].plot(trace.time_s, np.mean(trace.fingertip_forces_n, axis=1), label=label, color=color, linewidth=1.2)
    axes[0, 1].plot(trace.time_s, np.sum(trace.actual_contacts, axis=1), label=label, color=color, linewidth=1.0)
    axes[1, 0].plot(1000 * (trace.palm_pose_world[:, 1] - trace.palm_pose_world[0, 1]), 1000 * (trace.palm_pose_world[:, 0] - trace.palm_pose_world[0, 0]), label=label, color=color)
  axes[0, 0].axhline(2.0, color="black", linestyle="--", linewidth=1, label="2 N target")
  axes[0, 0].axvline(9.0, color="#D1495B", linestyle=":")
  axes[0, 0].set(title="Mean fingertip force", xlabel="time [s]", ylabel="force [N]")
  axes[0, 1].axvline(9.0, color="#D1495B", linestyle=":")
  axes[0, 1].set(title="Measured contact count", xlabel="time [s]", ylabel="count", ylim=(-0.1, 4.3))
  axes[1, 0].set(title="Actual two-dimensional palm path", xlabel="Y progress [mm]", ylabel="X displacement [mm]")
  axes[1, 1].plot(h_trace.time_s, h_trace.desired_hand_wrench_world[:, 2], label="desired reaction Fz", color="#F8961E")
  axes[1, 1].plot(h_trace.time_s, h_trace.estimated_hand_wrench_world[:, 2], label="FR3 joint-torque estimate", color="#43AA8B", linewidth=1)
  axes[1, 1].plot(h_trace.time_s, 1000 * h_trace.wrist_compliance_offset[:, 2], label="wrist dz [mm]", color="#577590", alpha=.8)
  axes[1, 1].axvline(9.0, color="#D1495B", linestyle=":")
  axes[1, 1].set(title="E05-H resultant wrist branch", xlabel="time [s]", ylabel="N or mm")
  axes[2, 0].plot(h_trace.time_s, np.max(h_trace.surface_curvature_inv_m, axis=1), color="#9B5DE5", label="max local curvature")
  axes[2, 0].axvline(9.0, color="#D1495B", linestyle=":", label="+4 mm away step")
  axes[2, 0].set(title="Continuous varying curvature and disturbance", xlabel="time [s]", ylabel="curvature [1/m]")

  metric_names = ("contact_continuity_probability", "force_rmse_n", "max_tip_force_n", "traversal_y_m")
  f_nominal = next(row["metrics"] for row in summary["episodes"] if row["cell"] == "E05-F-MCC" and row["episode"] == "nominal")
  h_nominal = next(row["metrics"] for row in summary["episodes"] if row["cell"] == "E05-H-MCC" and row["episode"] == "nominal")
  axes[2, 1].axis("off")
  rows = [
    ["metric", "F-MCC", "H-MCC"],
    ["contact continuity", f"{100*f_nominal[metric_names[0]]:.3f}%", f"{100*h_nominal[metric_names[0]]:.3f}%"],
    ["force RMSE", f"{f_nominal[metric_names[1]]:.3f} N", f"{h_nominal[metric_names[1]]:.3f} N"],
    ["peak force", f"{f_nominal[metric_names[2]]:.2f} N", f"{h_nominal[metric_names[2]]:.2f} N"],
    ["Y traversal", f"{1000*f_nominal[metric_names[3]]:.1f} mm", f"{1000*h_nominal[metric_names[3]]:.1f} mm"],
    ["performance", summary["cells"]["E05-F-MCC"]["performance_verdict"], summary["cells"]["E05-H-MCC"]["performance_verdict"]],
  ]
  table = axes[2, 1].table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="center")
  table.auto_set_font_size(False)
  table.set_fontsize(11)
  table.scale(1, 1.8)
  axes[2, 1].set_title("Formal nominal trace (evaluation != pass/fail)", pad=18)
  for axis in axes.ravel()[:5]:
    axis.grid(alpha=.2)
    axis.legend(fontsize=8, loc="best")
  characteristics = profile_characteristics()
  figure.suptitle(
    f"FR3 + Leap Hand MCC-only E05 | fixed object | 23 DoF | min curvature radius {1000*characteristics['minimum_curvature_radius_m']:.1f} mm",
    fontsize=17,
    fontweight="bold",
  )
  save_figure(figure, output_path)
  plt.close(figure)
  return output_path


def _index_html(summary: dict[str, Any]) -> str:
  f = summary["cells"]["E05-F-MCC"]
  h = summary["cells"]["E05-H-MCC"]
  return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FR3 + Leap MCC Visual Demo</title>
<style>body{{margin:0;background:#f3f6f9;color:#17324d;font-family:system-ui,sans-serif}}main{{width:min(1120px,94vw);margin:36px auto 70px}}section{{background:white;border:1px solid #dbe4ec;border-radius:16px;padding:20px;margin:18px 0}}video,img{{width:100%;border-radius:11px;border:1px solid #dbe4ec}}code{{background:#edf2f6;padding:2px 6px;border-radius:5px}}.badge{{display:inline-block;background:#dff7ef;color:#146c5b;padding:5px 10px;border-radius:999px;font-weight:700}}p{{line-height:1.65;color:#5b6e80}}</style></head>
<body><main><span class="badge">formal trace replay · no DP</span><h1>FR3 + Leap Hand：MCC-only E05</h1>
<p>两个视频都来自同一冻结对象、trajectory 和 nominal seed。物体固定在 world；画面中的 palm 位移由 7-DoF FR3 真实执行。右上 inset 显示四个真实 fingertip-body belly pads。</p>
<section><h2>E05-F-MCC</h2><p>规定式 FR3 wrist tracking；四个 Finger MCC 使用完整 local force error。执行状态 <code>{f['execution_status']}</code>，性能 <code>{f['performance_verdict']}</code>。</p><video controls preload="metadata" src="fr3_leap_e05_f_mcc.mp4"></video></section>
<section><h2>E05-H-MCC</h2><p>同一 nominal trajectory；FR3 Wrist MCC 调 resultant wrench，Finger MCC 只调 internal/differential error。执行状态 <code>{h['execution_status']}</code>，性能 <code>{h['performance_verdict']}</code>。</p><video controls preload="metadata" src="fr3_leap_e05_h_mcc.mp4"></video></section>
<section><h2>数值与模型审计</h2><img src="fr3_leap_mcc_dashboard.png"><img src="fr3_leap_model_audit.png"></section>
<p>边界：本页只评测 MCC。DP 未实现、未运行、未产生指标；gravity 在冻结协议中关闭，以隔离 contact control，不能把本结果外推为 gravity-on 或硬件结果。</p></main></body></html>"""


def run_visual_demo(
  result_dir: Path = DEFAULT_OUTPUT_DIR,
  output_dir: Path = DEFAULT_VISUAL_DIR,
) -> dict[str, Any]:
  summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
  trace_path = result_dir / "base_traces.npz"
  f_trace = load_base_trace(trace_path, "E05-F-MCC")
  h_trace = load_base_trace(trace_path, "E05-H-MCC")
  output_dir.mkdir(parents=True, exist_ok=True)
  f_video = render_video(
    f_trace,
    "E05-F-MCC",
    output_dir / "fr3_leap_e05_f_mcc.mp4",
    output_dir / "fr3_leap_model_audit.png",
  )
  h_video = render_video(
    h_trace,
    "E05-H-MCC",
    output_dir / "fr3_leap_e05_h_mcc.mp4",
    output_dir / "fr3_leap_h_mcc_frame.png",
  )
  dashboard = render_dashboard(
    f_trace,
    h_trace,
    summary,
    output_dir / "fr3_leap_mcc_dashboard.png",
  )
  page = output_dir / "fr3_leap_mcc.html"
  page.write_text(_index_html(summary), encoding="utf-8")
  result = {
    "source_experiment": summary["experiment"],
    "source_protocol_sha256": summary["protocol"]["sha256"],
    "dp_evaluated": False,
    "videos": [str(f_video), str(h_video)],
    "dashboard": str(dashboard),
    "page": str(page),
  }
  (output_dir / "fr3_leap_visual_summary.json").write_text(
    json.dumps(result, indent=2, sort_keys=True),
    encoding="utf-8",
  )
  return result


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--result-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
  parser.add_argument("--output-dir", type=Path, default=DEFAULT_VISUAL_DIR)
  args = parser.parse_args()
  print(json.dumps(run_visual_demo(args.result_dir, args.output_dir), indent=2))


if __name__ == "__main__":
  main()
