"""Render videos, screenshots, dashboard, and HTML for frozen E05-PHY traces."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import tempfile
from dataclasses import fields
from pathlib import Path
from typing import Any

import imageio.v2 as imageio

os.environ.setdefault("MUJOCO_GL", "osmesa")
cache_directory = Path(tempfile.gettempdir()) / "handcomp-mesa"
cache_directory.mkdir(parents=True, exist_ok=True)
os.environ["XDG_CACHE_HOME"] = str(cache_directory)

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from Module.e05_physics.benchmark import (
  DEFAULT_OUTPUT_DIR,
  extreme_surface_config,
  run_physics_evaluation,
  write_results,
)
from Module.e05_physics.extreme_surface import (
  X_HALF_M,
  Y_HALF_M,
  height_full_derivatives,
  maximum_principal_curvature,
)
from Module.e05_physics.runner import PhysicsConfig, PhysicsTrace
from Module.e05_physics.scene import build_scene
from Module.visualization import COLORS, get_pyplot, save_figure


SCENARIO_TITLES = {
  "maintenance_translation": "5A / Plane tangential translation (40 mm)",
  "maintenance_rotation": "5A / Plane wrist-relative rotation (5 deg)",
  "maintenance_curved": "5A / Large sphere surface following (25 mm)",
  "extreme_surface": "5D / 15 s, 480 mm S-scan on a 2D multi-scale surface + 4 mm step",
  "handover": "5B / Contact handover {1,2,3} -> {1,2} -> {1,2,4}",
}
FINGER_COLORS = ("#2997D6", "#3CBF91", "#F39C35", "#ED5A7A")


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
  name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
  path = Path("/usr/share/fonts/truetype/dejavu") / name
  if path.is_file():
    return ImageFont.truetype(str(path), size=size)
  return ImageFont.load_default()


def _contact_set(mask: np.ndarray) -> str:
  members = [str(index + 1) for index, active in enumerate(mask) if active]
  return "{" + ",".join(members) + "}" if members else "EMPTY"


def _overlay_frame(
  frame: np.ndarray,
  scenario: str,
  timestamp_s: float,
  desired_contacts: np.ndarray,
  actual_contacts: np.ndarray,
  forces_n: np.ndarray,
  curvature_inv_m: np.ndarray,
  disturbance_active: bool,
) -> np.ndarray:
  image = Image.fromarray(frame).convert("RGBA")
  overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
  draw = ImageDraw.Draw(overlay)
  width, height = image.size
  header_bottom = 128 if scenario == "extreme_surface" else 104
  draw.rounded_rectangle((18, 16, width - 18, header_bottom), 14, fill=(10, 24, 40, 210))
  draw.text((36, 30), "E05-PHY-v3  |  mesh-registered fingertip-belly contacts", font=_font(23, bold=True), fill="white")
  draw.text((36, 66), SCENARIO_TITLES[scenario], font=_font(18), fill=(205, 224, 240, 255))
  draw.text(
    (width - 205, 34),
    f"t = {timestamp_s:4.2f} s",
    font=_font(20, bold=True),
    fill=(255, 214, 102, 255),
  )
  if scenario == "extreme_surface":
    maximum_curvature = float(np.max(curvature_inv_m))
    radius_mm = 1000.0 / maximum_curvature if maximum_curvature > 1e-9 else float("inf")
    draw.text(
      (36, 94),
      f"max local curvature {maximum_curvature:5.1f} 1/m  |  radius {radius_mm:4.1f} mm",
      font=_font(16),
      fill=(205, 224, 240, 255),
    )
    if disturbance_active:
      draw.rounded_rectangle(
        (width - 270, 82, width - 34, 118),
        8,
        fill=(209, 73, 91, 235),
      )
      draw.text(
        (width - 254, 90),
        "POSE STEP: -4 mm",
        font=_font(17, bold=True),
        fill="white",
      )

  panel_top = height - 166
  draw.rounded_rectangle((18, panel_top, width - 18, height - 18), 14, fill=(10, 24, 40, 218))
  draw.text(
    (34, panel_top + 14),
    f"actual contact set  {_contact_set(actual_contacts)}",
    font=_font(20, bold=True),
    fill="white",
  )
  draw.text(
    (width - 325, panel_top + 16),
    "force source: mj_contactForce",
    font=_font(15),
    fill=(205, 224, 240, 255),
  )
  bar_left = 84
  bar_width = 132
  bar_top = panel_top + 55
  for finger_index in range(4):
    x = bar_left + finger_index * 205
    color = FINGER_COLORS[finger_index]
    draw.text((x - 48, bar_top + 2), f"F{finger_index + 1}", font=_font(16, bold=True), fill="white")
    draw.rounded_rectangle(
      (x, bar_top, x + bar_width, bar_top + 22),
      5,
      fill=(65, 78, 92, 255),
      outline=(255, 255, 255, 230) if desired_contacts[finger_index] else (100, 112, 124, 220),
      width=2,
    )
    fraction = float(np.clip(forces_n[finger_index] / 8.0, 0.0, 1.0))
    if fraction > 0.0:
      fill_color = color if actual_contacts[finger_index] else "#8C98A4"
      draw.rounded_rectangle(
        (x + 2, bar_top + 2, x + 2 + (bar_width - 4) * fraction, bar_top + 20),
        3,
        fill=fill_color,
      )
    draw.text(
      (x, bar_top + 28),
      f"{forces_n[finger_index]:4.2f} N",
      font=_font(15),
      fill=color if actual_contacts[finger_index] else (190, 198, 206, 255),
    )
  draw.text(
    (34, height - 38),
    "Fixed palm + inverse mocap object motion  |  FINGER DP NOT EVALUATED",
    font=_font(15, bold=True),
    fill=(255, 173, 190, 255),
  )
  return np.asarray(Image.alpha_composite(image, overlay).convert("RGB"))


def _camera(
  *,
  distance: float = 0.39,
  azimuth: float = 132.0,
  elevation: float = -30.0,
  lookat: tuple[float, float, float] = (-0.025, -0.025, 0.025),
) -> mujoco.MjvCamera:
  camera = mujoco.MjvCamera()
  camera.type = mujoco.mjtCamera.mjCAMERA_FREE
  camera.distance = distance
  camera.azimuth = azimuth
  camera.elevation = elevation
  camera.lookat[:] = np.array(lookat, dtype=np.float64)
  return camera


def render_trace_video(
  trace: PhysicsTrace,
  config: PhysicsConfig,
  output_path: Path,
  screenshot_dir: Path,
  *,
  fps: int = 30,
  width: int = 960,
  height: int = 720,
) -> dict[str, Path]:
  if config.scenario == "maintenance_curved":
    shape = "sphere"
  elif config.scenario == "extreme_surface":
    shape = "extreme"
  else:
    shape = "plane"
  handles = build_scene(shape, timestep_s=config.dt_s)
  data = mujoco.MjData(handles.model)
  renderer = mujoco.Renderer(handles.model, width=width, height=height)
  camera = _camera()
  frame_times = np.arange(0.0, config.duration_s, 1.0 / fps)
  frame_indices = np.unique(
    np.clip(np.round(frame_times / config.dt_s).astype(int), 0, len(trace.time_s) - 1)
  )
  screenshot_targets = {
    "start": config.settling_time_s,
    "mid": 0.5 * (config.settling_time_s + config.duration_s),
    "end": config.duration_s - config.dt_s,
  }
  if config.scenario == "extreme_surface":
    screenshot_targets.update(
      {
        "pre_step": config.pose_step_time_s - 0.05,
        "post_step": config.pose_step_time_s + 0.03,
        "recovered": config.pose_step_time_s + 0.12,
        "force_settled": config.pose_step_time_s + 2.05,
      }
    )
  screenshot_indices = {
    name: int(frame_indices[np.argmin(np.abs(trace.time_s[frame_indices] - target))])
    for name, target in screenshot_targets.items()
  }
  frames_out: list[np.ndarray] = []
  screenshots: dict[str, Path] = {}
  screenshot_dir.mkdir(parents=True, exist_ok=True)
  try:
    for index in frame_indices:
      data.qpos[handles.joint_qpos_adrs] = trace.joint_positions_rad[index]
      data.ctrl[:] = trace.joint_commands_rad[index]
      data.mocap_pos[handles.object_mocap_id] = trace.object_positions_m[index]
      data.mocap_quat[handles.object_mocap_id] = trace.object_quaternions[index]
      mujoco.mj_forward(handles.model, data)
      renderer.update_scene(data, camera=camera)
      try:
        renderer.scene.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
      except (AttributeError, IndexError, TypeError):
        pass
      frame = _overlay_frame(
        renderer.render().copy(),
        config.scenario,
        float(trace.time_s[index]),
        trace.desired_contacts[index],
        trace.actual_contacts[index],
        trace.fingertip_forces_n[index],
        trace.surface_curvatures_inv_m[index],
        bool(trace.disturbance_active[index]),
      )
      frames_out.append(frame)
      for name, target_index in screenshot_indices.items():
        if name not in screenshots and int(index) == target_index:
          path = screenshot_dir / f"{config.scenario}_{name}.png"
          imageio.imwrite(path, frame)
          screenshots[name] = path
  finally:
    renderer.close()
  output_path.parent.mkdir(parents=True, exist_ok=True)
  imageio.mimwrite(
    output_path,
    frames_out,
    fps=fps,
    codec="libx264",
    quality=8,
    macro_block_size=1,
  )
  if len(screenshots) != len(screenshot_targets):
    raise RuntimeError(f"failed to capture all screenshots for {config.scenario}")
  return screenshots


def _caption_view(frame: np.ndarray, title: str, subtitle: str) -> np.ndarray:
  image = Image.fromarray(frame).convert("RGBA")
  overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
  draw = ImageDraw.Draw(overlay)
  draw.rounded_rectangle(
    (14, 14, image.width - 14, 82),
    12,
    fill=(10, 24, 40, 214),
  )
  draw.text((30, 24), title, font=_font(19, bold=True), fill="white")
  draw.text((30, 53), subtitle, font=_font(13), fill=(205, 224, 240, 255))
  return np.asarray(Image.alpha_composite(image, overlay).convert("RGB"))


def render_pad_geometry_audit(
  summary: dict[str, Any],
  trace: PhysicsTrace,
  output_path: Path,
) -> Path:
  """Render three explicit views proving pad parent, position, and thumb pose."""

  handles = build_scene("plane", timestep_s=0.002)
  data = mujoco.MjData(handles.model)
  index = int(np.searchsorted(trace.time_s, 1.25))
  data.qpos[handles.joint_qpos_adrs] = trace.joint_positions_rad[index]
  data.ctrl[:] = trace.joint_commands_rad[index]
  data.mocap_pos[handles.object_mocap_id] = trace.object_positions_m[index]
  data.mocap_quat[handles.object_mocap_id] = trace.object_quaternions[index]
  mujoco.mj_forward(handles.model, data)
  renderer = mujoco.Renderer(handles.model, width=600, height=450)
  cameras = (
    (
      "A  UNDERSIDE / OBJECT HIDDEN",
      "colored ellipsoids are fixed to the physical fingertip bodies",
      _camera(
        distance=0.31,
        azimuth=92.0,
        elevation=16.0,
        lookat=(-0.005, -0.015, 0.015),
      ),
      0.0,
    ),
    (
      "B  LONG-FINGER SIDE VIEW",
      "pads are proximal to the rounded heads; colored rods show outward axes",
      _camera(
        distance=0.30,
        azimuth=176.0,
        elevation=-2.0,
        lookat=(-0.035, -0.035, 0.010),
      ),
      0.42,
    ),
    (
      "C  THUMB CONTACT VIEW",
      "thumb joints rotate its physical belly downward onto the plane",
      _camera(
        distance=0.22,
        azimuth=135.0,
        elevation=-4.0,
        lookat=(0.085, 0.070, 0.008),
      ),
      0.42,
    ),
  )
  original_alpha = float(handles.model.geom_rgba[handles.object_geom_id, 3])
  views: list[np.ndarray] = []
  try:
    for title, subtitle, camera, object_alpha in cameras:
      handles.model.geom_rgba[handles.object_geom_id, 3] = object_alpha
      renderer.update_scene(data, camera=camera)
      try:
        renderer.scene.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
      except (AttributeError, IndexError, TypeError):
        pass
      views.append(_caption_view(renderer.render().copy(), title, subtitle))
  finally:
    handles.model.geom_rgba[handles.object_geom_id, 3] = original_alpha
    renderer.close()

  canvas = Image.new("RGB", (1840, 710), (238, 244, 248))
  for view_index, view in enumerate(views):
    canvas.paste(Image.fromarray(view), (20 + 610 * view_index, 20))
  draw = ImageDraw.Draw(canvas)
  maintenance = summary["contact_maintenance"]["scenarios"]["maintenance_translation"]
  claims = summary["claims"]
  draw.rounded_rectangle((20, 490, 1820, 690), 16, fill=(255, 255, 255))
  draw.text(
    (44, 512),
    "E05-PHY-v3 GEOMETRY AUDIT — CONTACT CANNOT COME FROM THE ROUNDED TIP HEAD",
    font=_font(25, bold=True),
    fill=(23, 50, 77),
  )
  draw.text(
    (44, 558),
    "pad parent bodies: " + ", ".join(claims["pad_parent_bodies"]),
    font=_font(17),
    fill=(44, 65, 84),
  )
  draw.text(
    (44, 590),
    (
      f"design distal-head gap >= {1000 * claims['pad_distal_head_clearance_design_min_m']:.1f} mm"
      f"   |   measured contact-to-head clearance >= "
      f"{1000 * maintenance['contact_distal_head_clearance_min_m']:.1f} mm"
    ),
    font=_font(17, bold=True),
    fill=(25, 119, 92),
  )
  draw.text(
    (44, 622),
    (
      "thumb q[12:16] = ["
      + ", ".join(f"{value:.3f}" for value in claims["thumb_nominal_joint_angles_rad"])
      + f"] rad   |   physical thumb contact = {100 * maintenance['thumb_contact_probability']:.1f}%"
    ),
    font=_font(17),
    fill=(209, 73, 91),
  )
  draw.text(
    (44, 654),
    "Only the four colored ellipsoids have collision bits; all original hand/tip meshes are visual-only.",
    font=_font(15),
    fill=(91, 107, 122),
  )
  output_path.parent.mkdir(parents=True, exist_ok=True)
  canvas.save(output_path)
  return output_path


def _load_trace_archive(path: Path) -> dict[str, PhysicsTrace]:
  result: dict[str, PhysicsTrace] = {}
  with np.load(path) as archive:
    scenarios = sorted({name.split("__", 1)[0] for name in archive.files})
    for scenario in scenarios:
      values = {
        item.name: archive[f"{scenario}__{item.name}"]
        for item in fields(PhysicsTrace)
      }
      result[scenario] = PhysicsTrace(**values)
  return result


def _load_rows(path: Path) -> list[dict[str, Any]]:
  with path.open(newline="", encoding="utf-8") as stream:
    return [dict(row) for row in csv.DictReader(stream)]


def render_dashboard(
  summary: dict[str, Any],
  rows: list[dict[str, Any]],
  traces: dict[str, PhysicsTrace],
  output_path: Path,
) -> Path:
  plt = get_pyplot()
  figure, axes = plt.subplots(2, 2, figsize=(15.2, 9.2))

  rotation = traces["maintenance_rotation"]
  mask = rotation.time_s >= 0.75
  for finger_index, color in enumerate(FINGER_COLORS):
    axes[0, 0].plot(
      rotation.time_s[mask],
      rotation.fingertip_forces_n[mask, finger_index],
      color=color,
      linewidth=1.6,
      label=f"finger {finger_index + 1}",
    )
  axes[0, 0].axhline(2.0, color=COLORS["navy"], linestyle=":", label="target 2 N")
  axes[0, 0].axhline(8.0, color=COLORS["red"], linestyle="--", label="limit 8 N")
  axes[0, 0].set_ylim(0.0, 8.4)
  axes[0, 0].set_xlabel("time [s]")
  axes[0, 0].set_ylabel("MuJoCo normal force [N]")
  axes[0, 0].set_title("A  5A physical contact maintenance", loc="left")
  axes[0, 0].legend(fontsize=8, ncol=2)

  handover = traces["handover"]
  axes[1, 0].imshow(
    handover.actual_contacts.T.astype(float),
    origin="lower",
    aspect="auto",
    interpolation="nearest",
    extent=(0.0, 3.0, 0.5, 4.5),
    cmap="Blues",
    vmin=0.0,
    vmax=1.0,
  )
  handover_result = summary["contact_handover"]
  break_time = handover_result["break_command_time_s"]
  make_time = handover_result["make_command_time_s"]
  confirm_time = make_time + handover_result["make_recovery_time_s"]
  axes[1, 0].axvline(break_time, color=COLORS["orange"], linestyle="--", label="BREAK F3")
  axes[1, 0].axvline(make_time, color=COLORS["pink"], linestyle="--", label="MAKE F4")
  axes[1, 0].axvline(confirm_time, color=COLORS["green"], linestyle=":", label="F4 confirmed")
  axes[1, 0].set_yticks([1, 2, 3, 4], ["finger 1", "finger 2", "finger 3", "finger 4"])
  axes[1, 0].set_xlabel("time [s]  (blue = physical contact >= 0.20 N)")
  axes[1, 0].set_title("B  5B physical contact handover", loc="left")
  axes[1, 0].legend(fontsize=8, loc="lower right")

  for scenario, color, marker in (
    ("maintenance_rotation", COLORS["blue"], "o"),
    ("maintenance_curved", COLORS["pink"], "s"),
  ):
    selected = [row for row in rows if row["scenario"] == scenario]
    axes[0, 1].scatter(
      [float(row["force_rmse_n"]) for row in selected],
      [float(row["max_tip_force_n"]) for row in selected],
      color=color,
      marker=marker,
      s=42,
      alpha=0.75,
      label=scenario.replace("maintenance_", ""),
    )
  axes[0, 1].axvline(0.35, color=COLORS["red"], linestyle="--", label="RMSE limit")
  axes[0, 1].axhline(8.0, color=COLORS["red"], linestyle=":", label="force limit")
  axes[0, 1].set_xlim(0.0, 0.40)
  axes[0, 1].set_ylim(0.0, 8.4)
  axes[0, 1].set_xlabel("episode physical force RMSE [N]")
  axes[0, 1].set_ylabel("episode maximum force [N]")
  axes[0, 1].set_title("C  5C frozen 24-episode MuJoCo sweep", loc="left")
  axes[0, 1].legend(fontsize=8)

  axes[1, 1].set_axis_off()
  maintenance = summary["contact_maintenance"]["scenarios"]
  robustness = summary["control_robustness"]["aggregate"]
  values = (
    (0.17, "5A CONTINUITY", f"{100 * min(v['contact_continuity_probability'] for v in maintenance.values()):.1f}%", "three physical scenes", "#DDEFF7"),
    (0.50, "5B MAKE", f"{1000 * handover_result['make_recovery_time_s']:.0f} ms", "zero-contact 0 ms", "#DFF7EF"),
    (0.83, "5C SUCCESS", f"{100 * robustness['episode_success_rate']:.1f}%", "24 MuJoCo episodes", "#FFF1D6"),
  )
  for x, title, value, subtitle, color in values:
    axes[1, 1].text(
      x,
      0.72,
      f"{title}\n{value}\n{subtitle}",
      transform=axes[1, 1].transAxes,
      ha="center",
      va="center",
      fontsize=10,
      linespacing=1.5,
      bbox={"boxstyle": "round,pad=0.7", "facecolor": color, "edgecolor": "none"},
    )
  axes[1, 1].text(
    0.5,
    0.32,
    "FINGER DP: NOT EVALUATED\nNo comparison claim is made.",
    transform=axes[1, 1].transAxes,
    ha="center",
    va="center",
    fontsize=12,
    weight="bold",
    color=COLORS["red"],
    bbox={"boxstyle": "round,pad=0.7", "facecolor": "#FCE8EC", "edgecolor": "#F4B8C3"},
  )
  axes[1, 1].text(
    0.5,
    0.08,
    "MuJoCo 3.6.0 | fixed palm + inverse mocap object\n4 mesh-registered belly pads only; no dynamic wrist or non-tip collision claim",
    transform=axes[1, 1].transAxes,
    ha="center",
    fontsize=9,
    color=COLORS["gray"],
  )
  axes[1, 1].set_title("D  Result boundary", loc="left")
  figure.suptitle(
    "E05-PHY-v3 — MCC baseline sanity checks with mesh-registered belly pads",
    fontsize=17,
    weight="bold",
    x=0.04,
    ha="left",
  )
  figure.subplots_adjust(top=0.91, left=0.08, right=0.97, bottom=0.08, hspace=0.34, wspace=0.24)
  save_figure(figure, output_path)
  plt.close(figure)
  return output_path


def render_extreme_dashboard(
  summary: dict[str, Any],
  trace: PhysicsTrace,
  output_path: Path,
) -> Path:
  result = summary["extreme_surface_challenge"]
  continuous = result["continuous_sweep"]
  recovery = result["pose_step_recovery"]
  config = result["config"]
  plt = get_pyplot()
  figure, axes = plt.subplots(2, 2, figsize=(15.2, 9.2))

  x = np.linspace(-X_HALF_M, X_HALF_M, 241)
  y = np.linspace(-Y_HALF_M, Y_HALF_M, 321)
  grid_x, grid_y = np.meshgrid(x, y)
  height, dx, dy, dxx, dxy, dyy = height_full_derivatives(grid_x, grid_y)
  curvature = maximum_principal_curvature(dx, dy, dxx, dxy, dyy)
  surface_plot = axes[0, 0].contourf(
    grid_x * 1000.0,
    grid_y * 1000.0,
    height * 1000.0,
    levels=28,
    cmap="coolwarm",
  )
  axes[0, 0].contour(
    grid_x * 1000.0,
    grid_y * 1000.0,
    curvature,
    levels=[10.0, 40.0, 60.0],
    colors=["white", "#521945", "#151515"],
    linewidths=[0.7, 1.0, 1.2],
  )
  local_paths = trace.fingertip_positions_m - trace.object_positions_m[:, None, :]
  for finger_index, color in enumerate(FINGER_COLORS):
    axes[0, 0].plot(
      local_paths[:, finger_index, 0] * 1000.0,
      local_paths[:, finger_index, 1] * 1000.0,
      color=color,
      linewidth=1.4,
      label=f"F{finger_index + 1}",
    )
  figure.colorbar(surface_plot, ax=axes[0, 0], label="height [mm]", fraction=0.046)
  axes[0, 0].set_xlabel("surface x [mm]")
  axes[0, 0].set_ylabel("surface y [mm]")
  axes[0, 0].set_aspect("equal", adjustable="box")
  axes[0, 0].legend(fontsize=7, ncol=4, loc="lower right")
  axes[0, 0].set_title("A  2D multi-scale surface and four S-scan tracks", loc="left")

  for finger_index, color in enumerate(FINGER_COLORS):
    axes[0, 1].plot(
      trace.time_s,
      trace.fingertip_forces_n[:, finger_index],
      color=color,
      linewidth=1.2,
      label=f"finger {finger_index + 1}",
    )
  axes[0, 1].axhline(2.0, color=COLORS["navy"], linestyle=":", label="target 2 N")
  axes[0, 1].axhline(8.0, color=COLORS["red"], linestyle="--", label="force limit 8 N")
  axes[0, 1].axvline(config["pose_step_time_s"], color=COLORS["purple"], linestyle="-.", label="-4 mm pose step")
  axes[0, 1].set_xlim(config["settling_time_s"], config["duration_s"])
  full_max_force = float(np.max(trace.fingertip_forces_n))
  axes[0, 1].set_yscale("symlog", linthresh=8.0, linscale=1.0)
  axes[0, 1].set_ylim(0.0, max(10.0, full_max_force * 1.08))
  axes[0, 1].set_yticks([0.0, 2.0, 8.0, 20.0, 50.0, 100.0])
  axes[0, 1].set_yticklabels(["0", "2", "8", "20", "50", "100"])
  axes[0, 1].set_xlabel("time [s]")
  axes[0, 1].set_ylabel("MuJoCo normal force [N]")
  axes[0, 1].set_title("B  Physical force: continuous sweep and pose step", loc="left")
  axes[0, 1].legend(fontsize=7, ncol=2)

  axes[1, 0].imshow(
    trace.actual_contacts.T.astype(float),
    origin="lower",
    aspect="auto",
    interpolation="nearest",
    extent=(0.0, config["duration_s"], 0.5, 4.5),
    cmap="Blues",
    vmin=0.0,
    vmax=1.0,
  )
  step_time = config["pose_step_time_s"]
  any_recovery = step_time + recovery["any_contact_recovery_s"]
  all_recovery = step_time + recovery["all_finger_contact_recovery_s"]
  axes[1, 0].axvline(step_time, color=COLORS["red"], linestyle="--", label="pose step")
  axes[1, 0].axvline(any_recovery, color=COLORS["orange"], linestyle=":", label="any contact recovered")
  axes[1, 0].axvline(all_recovery, color=COLORS["green"], linestyle="-.", label="all fingers recovered")
  axes[1, 0].set_yticks([1, 2, 3, 4], ["finger 1", "finger 2", "finger 3", "finger 4"])
  axes[1, 0].set_xlabel("time [s]  (blue = physical pad contact >= 0.20 N)")
  axes[1, 0].set_title("C  Contact loss and recovery after abrupt change", loc="left")
  axes[1, 0].legend(fontsize=7, loc="lower left")

  axes[1, 1].set_axis_off()
  status_color = COLORS["green"] if result["thresholds_met"] else COLORS["red"]
  status_text = (
    "ALL THRESHOLDS MET"
    if result["thresholds_met"]
    else "THRESHOLDS NOT MET"
  )
  boxes = (
    (0.15, "CONTINUOUS", f"{100 * continuous['hand_contact_probability']:.1f}%", f"avg {continuous['average_contact_count']:.2f} pads", "#DDEFF7"),
    (
      0.50,
      "FORCE PEAKS",
      f"{continuous['max_tip_force_n']:.1f}/{recovery['max_tip_force_first_second_n']:.1f} N",
      "sweep / post-step",
      "#FCE8EC",
    ),
    (0.85, "ALL RECOVERY", f"{1000 * recovery['all_finger_contact_recovery_s']:.0f} ms", f"settled {recovery['force_settling_s']:.2f} s", "#FFF1D6"),
  )
  for x, title, value, subtitle, color in boxes:
    axes[1, 1].text(
      x,
      0.72,
      f"{title}\n{value}\n{subtitle}",
      transform=axes[1, 1].transAxes,
      ha="center",
      va="center",
      fontsize=9,
      linespacing=1.5,
      bbox={"boxstyle": "round,pad=0.55", "facecolor": color, "edgecolor": "none"},
    )
  axes[1, 1].text(
    0.5,
    0.34,
    f"EXTREME-SURFACE METRICS: {status_text}",
    transform=axes[1, 1].transAxes,
    ha="center",
    va="center",
    fontsize=14,
    weight="bold",
    color=status_color,
    bbox={"boxstyle": "round,pad=0.7", "facecolor": "white", "edgecolor": status_color},
  )
  axes[1, 1].text(
    0.5,
    0.13,
    "Verdict follows every frozen safety/recovery limit; quick contact recovery cannot override force violations.\n"
    "All object contacts are restricted to mesh-registered fingertip-belly pads.",
    transform=axes[1, 1].transAxes,
    ha="center",
    va="center",
    fontsize=10,
    color=COLORS["gray"],
  )
  axes[1, 1].set_title("D  Measured threshold verdict", loc="left")
  figure.suptitle(
    "E05-PHY-v3 — 15 s mesh-registered pad traversal on a 2D multi-scale surface",
    fontsize=17,
    weight="bold",
    x=0.04,
    ha="left",
  )
  figure.subplots_adjust(top=0.91, left=0.08, right=0.92, bottom=0.08, hspace=0.34, wspace=0.30)
  save_figure(figure, output_path)
  plt.close(figure)
  return output_path


def _write_index(output_dir: Path, summary: dict[str, Any]) -> Path:
  scenarios = list(SCENARIO_TITLES)
  maintenance = summary["contact_maintenance"]["scenarios"]
  handover = summary["contact_handover"]
  extreme = summary["extreme_surface_challenge"]
  cards: list[str] = []
  for scenario in scenarios:
    if scenario == "handover":
      metric = (
        f"MAKE {1000 * handover['make_recovery_time_s']:.0f} ms; "
        f"max force {handover['max_tip_force_n']:.2f} N"
      )
    elif scenario == "extreme_surface":
      continuous = extreme["continuous_sweep"]
      recovery = extreme["pose_step_recovery"]
      metric = (
        f"thresholds {'MET' if extreme['thresholds_met'] else 'NOT MET'}; "
        f"pre-step S-path {continuous['relative_path_length_m']:.3f} m; "
        f"thumb contact {100 * continuous['thumb_contact_probability']:.1f}%; "
        f"force peaks {continuous['max_tip_force_n']:.1f}/{recovery['max_tip_force_first_second_n']:.1f} N; "
        f"all-finger recovery {1000 * recovery['all_finger_contact_recovery_s']:.0f} ms"
      )
    else:
      item = maintenance[scenario]
      metric = (
        f"continuity {100 * item['contact_continuity_probability']:.1f}%; "
        f"force RMSE {item['force_rmse_n']:.3f} N"
      )
    extra_shots = ""
    if scenario == "extreme_surface":
      extra_shots = f"""
          <img src="screenshots/{scenario}_pre_step.png" alt="pre step">
          <img src="screenshots/{scenario}_post_step.png" alt="post step">
          <img src="screenshots/{scenario}_recovered.png" alt="recovered">
          <img src="screenshots/{scenario}_force_settled.png" alt="force settled">
      """
    cards.append(
      f"""
      <article>
        <h2>{html.escape(SCENARIO_TITLES[scenario])}</h2>
        <p>{html.escape(metric)}</p>
        <video controls preload="metadata" poster="screenshots/{scenario}_mid.png">
          <source src="videos/{scenario}.mp4" type="video/mp4">
        </video>
        <div class="shots">
          <img src="screenshots/{scenario}_start.png" alt="start">
          <img src="screenshots/{scenario}_mid.png" alt="mid">
          <img src="screenshots/{scenario}_end.png" alt="end">
          {extra_shots}
        </div>
      </article>
      """
    )
  document = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>E05-PHY-v3 MuJoCo baseline</title>
<style>
body {{ margin: 0; background: #eef3f7; color: #17324d; font: 16px/1.55 system-ui, sans-serif; }}
main {{ max-width: 1180px; margin: auto; padding: 36px 24px 64px; }}
.hero, article {{ background: white; border-radius: 18px; padding: 24px; margin-bottom: 24px; box-shadow: 0 8px 30px #17324d18; }}
h1 {{ margin: 0 0 8px; }} h2 {{ margin-top: 0; font-size: 20px; }}
.boundary {{ border-left: 5px solid #d1495b; background: #fce8ec; padding: 12px 16px; border-radius: 8px; }}
video, .dashboard {{ display: block; width: 100%; border-radius: 12px; background: #0a1828; }}
.shots {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; margin-top: 10px; }}
.shots img {{ width: 100%; border-radius: 8px; }} code {{ background: #eef3f7; padding: 2px 5px; }}
</style>
</head>
<body><main>
<section class="hero">
  <h1>E05-PHY-v3：真实 fingertip-body 指腹与长程二维复杂曲面</h1>
  <p>视频来自正式 trace 的状态重放；彩色薄椭球直接固定在真实 fingertip body 的腹侧，并与圆形 tip head 保留可测间隔。只有这些 pad proxy 能与物体碰撞，力值来自 <code>mj_contactForce</code>。</p>
  <p class="boundary"><strong>评测状态：</strong><strong>{summary['evaluation_status']}</strong>；基础 sanity thresholds <strong>{'MET' if summary['baseline_sanity_thresholds_met'] else 'NOT MET'}</strong>；extreme-surface thresholds <strong>{'MET' if extreme['thresholds_met'] else 'NOT MET'}</strong>。指标判定只评价 baseline 表现，不改变评测已完成的状态。</p>
  <p class="boundary"><strong>边界：</strong>掌部固定，物体作等效反向 mocap 运动；验证 finger/contact dynamics，未验证机械臂 wrist dynamics。只评测 MCC，Finger DP 未运行。</p>
  <img class="dashboard" src="pad_geometry_audit.png" alt="mesh-registered pad geometry audit">
  <img class="dashboard" src="e05_physics_dashboard.png" alt="E05 physics dashboard">
  <img class="dashboard" src="e05_extreme_dashboard.png" alt="E05 extreme-surface dashboard" style="margin-top:16px">
</section>
{''.join(cards)}
</main></body></html>
"""
  index_path = output_dir / "index.html"
  index_path.write_text(document, encoding="utf-8")
  return index_path


def run_visual_demo(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
  output_dir = Path(output_dir)
  summary_path = output_dir / "summary.json"
  episodes_path = output_dir / "robustness_episodes.csv"
  traces_path = output_dir / "traces.npz"
  if not (summary_path.is_file() and episodes_path.is_file() and traces_path.is_file()):
    summary, rows, traces = run_physics_evaluation()
    write_results(output_dir, summary, rows, traces)
  else:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = _load_rows(episodes_path)
    traces = _load_trace_archive(traces_path)
  video_paths: list[Path] = []
  screenshot_paths: list[Path] = []
  for scenario, trace in traces.items():
    config = (
      extreme_surface_config()
      if scenario == "extreme_surface"
      else PhysicsConfig(scenario=scenario)
    )
    video_path = output_dir / "videos" / f"{scenario}.mp4"
    screenshots = render_trace_video(
      trace,
      config,
      video_path,
      output_dir / "screenshots",
    )
    video_paths.append(video_path)
    screenshot_paths.extend(screenshots.values())
  dashboard = render_dashboard(
    summary,
    rows,
    traces,
    output_dir / "e05_physics_dashboard.png",
  )
  extreme_dashboard = render_extreme_dashboard(
    summary,
    traces["extreme_surface"],
    output_dir / "e05_extreme_dashboard.png",
  )
  pad_geometry_audit = render_pad_geometry_audit(
    summary,
    traces["maintenance_translation"],
    output_dir / "pad_geometry_audit.png",
  )
  index = _write_index(output_dir, summary)
  return {
    "evaluation_status": summary["evaluation_status"],
    "evaluation_completed": summary["evaluation_completed"],
    "benchmark_verdict": summary["benchmark_verdict"],
    "all_thresholds_met": summary["all_thresholds_met"],
    "rendered": True,
    "dashboard": str(dashboard),
    "extreme_dashboard": str(extreme_dashboard),
    "pad_geometry_audit": str(pad_geometry_audit),
    "index": str(index),
    "videos": [str(path) for path in sorted(video_paths)],
    "screenshots": [str(path) for path in sorted(screenshot_paths)],
    "finger_dp_evaluated": False,
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
  args = parser.parse_args()
  print(json.dumps(run_visual_demo(args.output_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
  main()
