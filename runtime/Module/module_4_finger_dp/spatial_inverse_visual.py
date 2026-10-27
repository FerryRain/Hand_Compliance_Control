"""Visual evidence for the real forward/spatial-inverse physical pair."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

os.environ.setdefault("MUJOCO_GL", "osmesa")
_CACHE = Path(tempfile.gettempdir()) / "handcomp-spatial-inverse-v1"
_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["XDG_CACHE_HOME"] = str(_CACHE)
os.environ["MESA_SHADER_CACHE_DIR"] = str(_CACHE / "mesa_shader_cache")
os.environ["MPLCONFIGDIR"] = str(_CACHE / "matplotlib")
Path(os.environ["MESA_SHADER_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from Module.e05_physics.scene import build_scene
from Module.fr3_leap import FullRobotModelConfig, build_full_robot
from Module.module_4_finger_dp.spatial_inverse_data import (
  SpatialInverseAudit,
  SpatialInversePhysicalPair,
)
from Module.visualization import get_pyplot, save_figure


FINGER_COLORS = ("#2997D6", "#3CBF91", "#F39C35", "#ED5A7A")


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
  name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
  path = Path("/usr/share/fonts/truetype/dejavu") / name
  return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def _camera(
  lookat: tuple[float, float, float],
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
  active = [str(index + 1) for index in np.flatnonzero(mask)]
  return "{" + ",".join(active) + "}" if active else "EMPTY"


def _phase_panel(
  frame: np.ndarray,
  *,
  title: str,
  subtitle: str,
  force: np.ndarray,
  contact: np.ndarray,
) -> Image.Image:
  image = Image.fromarray(frame).convert("RGB")
  draw = ImageDraw.Draw(image, "RGBA")
  draw.rounded_rectangle((12, 10, image.width - 12, 75), 10, fill=(5, 18, 33, 220))
  draw.text((26, 18), title, font=_font(20, bold=True), fill="white")
  draw.text((26, 48), subtitle, font=_font(13), fill=(194, 221, 241))
  top = image.height - 84
  draw.rounded_rectangle((12, top, image.width - 12, image.height - 10), 10, fill=(5, 18, 33, 224))
  draw.text(
    (24, top + 9),
    f"actual A={_contact_set(contact)}",
    font=_font(15, bold=True),
    fill="white",
  )
  for finger in range(4):
    x = 28 + finger * 147
    draw.text(
      (x, top + 39),
      f"F{finger + 1} {force[finger]:4.2f}N",
      font=_font(14, bold=True),
      fill=FINGER_COLORS[finger] if contact[finger] else (154, 166, 180),
    )
  return image


def render_spatial_inverse_video(
  pair: SpatialInversePhysicalPair,
  audit: SpatialInverseAudit,
  output_path: str | Path,
  *,
  fps: int = 20,
) -> Path:
  """Render source physics and replay physics side by side."""

  if fps < 1:
    raise ValueError("fps must be positive")
  destination = Path(output_path)
  destination.parent.mkdir(parents=True, exist_ok=True)
  source_handles = build_scene("extreme", timestep_s=0.002)
  source_data = mujoco.MjData(source_handles.model)
  replay_handles = build_full_robot(
    FullRobotModelConfig(
      surface="extreme",
      timestep_s=0.002,
      gravity_m_s2=0.0,
      arm_kp=8000.0,
      arm_damping_ratio=0.95,
    )
  )
  replay_data = mujoco.MjData(replay_handles.model)
  source_renderer = mujoco.Renderer(source_handles.model, width=640, height=430)
  replay_renderer = mujoco.Renderer(replay_handles.model, width=640, height=430)
  source_camera = _camera((-0.025, 0.135, 0.025), 0.62, 137.0, -25.0)
  replay_camera = _camera((0.39, 0.10, 0.48), 1.03, 133.0, -23.0)
  dt_s = float(np.median(np.diff(pair.replay.time_s)))
  indices = np.unique(
    np.clip(
      np.round(np.arange(0.0, pair.replay.time_s[-1] + dt_s, 1.0 / fps) / dt_s).astype(int),
      0,
      pair.replay.length - 1,
    )
  )
  writer = imageio.get_writer(
    destination,
    fps=fps,
    codec="libx264",
    quality=8,
    macro_block_size=1,
  )
  try:
    for index in indices:
      source_data.qpos[source_handles.joint_qpos_adrs] = pair.forward.q_f_meas_rad[index]
      source_data.ctrl[:] = pair.forward.q_f_command_rad[index]
      source_data.mocap_pos[source_handles.object_mocap_id] = pair.forward.object_pose_world[index, :3]
      source_data.mocap_quat[source_handles.object_mocap_id] = pair.forward.object_pose_world[index, 3:]
      mujoco.mj_forward(source_handles.model, source_data)
      source_renderer.update_scene(source_data, camera=source_camera)
      source_frame = source_renderer.render().copy()

      replay_data.qpos[replay_handles.arm_qpos_adrs] = pair.replay.arm_q_meas_rad[index]
      replay_data.qpos[replay_handles.hand_qpos_adrs] = pair.replay.q_f_meas_rad[index]
      replay_data.ctrl[replay_handles.arm_actuator_ids] = pair.replay.arm_command_rad[index]
      replay_data.ctrl[replay_handles.hand_actuator_ids] = pair.replay.q_f_command_rad[index]
      mujoco.mj_forward(replay_handles.model, replay_data)
      replay_renderer.update_scene(replay_data, camera=replay_camera)
      replay_frame = replay_renderer.render().copy()

      left = _phase_panel(
        source_frame,
        title="A · Forward physical collection",
        subtitle="moving object + measured force/contact + recorded q_cmd",
        force=pair.forward.contact_force_n[index],
        contact=pair.forward.contact_mask[index],
      )
      right = _phase_panel(
        replay_frame,
        title="B · Spatial inverse physical replay",
        subtitle="fixed object + FR3 motion + SAME-t q_cmd; no finger repair",
        force=pair.replay.contact_force_n[index],
        contact=pair.replay.contact_mask[index],
      )
      canvas = Image.new("RGB", (1280, 560), (7, 17, 29))
      canvas.paste(left, (0, 94))
      canvas.paste(right, (640, 94))
      draw = ImageDraw.Draw(canvas, "RGBA")
      draw.text(
        (24, 16),
        "Forward → spatial inverse → fresh physical replay",
        font=_font(25, bold=True),
        fill="white",
      )
      verdict = "RAW REPLAY GATE PASSED" if audit.accepted else "RAW REPLAY GATE FAILED"
      verdict_color = (39, 174, 119, 235) if audit.accepted else (203, 65, 82, 235)
      draw.rounded_rectangle((972, 15, 1258, 57), 8, fill=verdict_color)
      draw.text((990, 27), verdict, font=_font(15, bold=True), fill="white")
      draw.text(
        (24, 57),
        f"t={pair.replay.time_s[index]:4.2f}s  mode=SPATIAL_ONLY  time=SAME_ORDER  "
        f"max |q_cmd^R-q_cmd^F|={pair.maximum_finger_command_mapping_residual_rad:.1e} rad",
        font=_font(15),
        fill=(187, 214, 234),
      )
      draw.line((640, 94, 640, 524), fill=(126, 158, 184, 180), width=2)
      writer.append_data(np.asarray(canvas))
  finally:
    writer.close()
    source_renderer.close()
    replay_renderer.close()
  return destination


def render_spatial_inverse_dashboard(
  pair: SpatialInversePhysicalPair,
  audit: SpatialInverseAudit,
  output_path: str | Path,
) -> Path:
  destination = Path(output_path)
  plt = get_pyplot()
  figure, axes = plt.subplots(3, 1, figsize=(13, 10), constrained_layout=True)
  for finger, color in enumerate(FINGER_COLORS):
    axes[0].plot(
      pair.forward.time_s,
      pair.forward.contact_force_n[:, finger],
      color=color,
      linewidth=0.9,
      label=f"F{finger + 1} forward",
    )
    axes[0].plot(
      pair.replay.time_s,
      pair.replay.contact_force_n[:, finger],
      color=color,
      linewidth=0.8,
      linestyle="--",
      alpha=0.75,
      label=f"F{finger + 1} replay",
    )
  axes[0].axhline(8.0, color="#D1495B", linestyle=":", label="8 N hard gate")
  axes[0].set(title="Fresh physical forces (solid=forward, dashed=replay)", ylabel="N")
  axes[0].legend(ncol=5, fontsize=7)

  axes[1].step(
    pair.forward.time_s,
    np.sum(pair.forward.contact_mask, axis=1),
    where="post",
    color="#277DA1",
    label="forward actual contact count",
  )
  axes[1].step(
    pair.replay.time_s,
    np.sum(pair.replay.contact_mask, axis=1),
    where="post",
    color="#D98032",
    label="replay actual contact count",
  )
  palm_error_mm = 1000.0 * np.linalg.norm(
    pair.replay.palm_pose_real_world[:, :3] - pair.replay.palm_pose_plan_world[:, :3],
    axis=1,
  )
  axes[1].plot(pair.replay.time_s, palm_error_mm, color="#8E5EA2", label="palm tracking error [mm]")
  axes[1].set(title="Contact preservation and FR3 tracking", ylabel="count / mm")
  axes[1].legend(fontsize=8)

  forward_object_delta_mm = 1000.0 * (
    pair.forward.object_pose_world[:, 1] - pair.forward.object_pose_world[0, 1]
  )
  replay_palm_delta_mm = 1000.0 * (
    pair.replay.palm_pose_plan_world[:, 1] - pair.replay.palm_pose_plan_world[0, 1]
  )
  axes[2].plot(
    pair.forward.time_s,
    forward_object_delta_mm,
    color="#4C78A8",
    label="forward object delta-y [mm]",
  )
  axes[2].plot(
    pair.replay.time_s,
    replay_palm_delta_mm,
    color="#F58518",
    linestyle="--",
    label="inverse replay palm delta-y [mm]",
  )
  axes[2].axhline(
    pair.maximum_finger_command_mapping_residual_rad,
    color="#54A24B",
    label="max q_cmd mapping residual [rad]",
  )
  axes[2].set(title="Spatial role inversion (opposite world motion, same finger command time)", xlabel="time [s]")
  axes[2].legend(fontsize=8)
  status = "RAW REPLAY PASSED" if audit.accepted else "RAW REPLAY FAILED: " + ", ".join(audit.reasons)
  figure.suptitle(
    f"Dataset-D inverse-pipeline diagnostic · {status}\n"
    "Not a DP evaluation; replay force/contact are fresh measurements",
    fontsize=14,
  )
  save_figure(figure, destination)
  plt.close(figure)
  return destination
