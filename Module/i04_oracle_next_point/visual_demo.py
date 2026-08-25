"""Replay a saved I04 trace with physical and planner-state annotations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("IMAGEIO_FFMPEG_EXE", "/usr/bin/ffmpeg")
cache = Path(tempfile.gettempdir()) / "handcomp-i04-mesa"
cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(cache))
os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from Module.fr3_leap import FullRobotModelConfig, build_full_robot
from Module.i01_bunny_physics.surface import canonical_bunny_heightfield
from Module.i04_oracle_next_point.benchmark import DEFAULT_OUTPUT_DIR
from Module.visualization import get_pyplot, save_figure


FINGER_COLORS = ("#2997D6", "#3CBF91", "#F39C35", "#ED5A7A")
GOAL_RGBA = np.asarray([0.96, 0.20, 0.65, 1.0], dtype=np.float32)
BRIDGE_RGBA = np.asarray([0.32, 0.88, 0.45, 1.0], dtype=np.float32)
TRAIL_RGBA = np.asarray([0.25, 0.78, 1.0, 0.82], dtype=np.float32)
ANCHOR_RGBA = np.asarray([0.28, 0.64, 1.0, 0.85], dtype=np.float32)


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
  name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
  path = Path("/usr/share/fonts/truetype/dejavu") / name
  return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def _camera() -> mujoco.MjvCamera:
  camera = mujoco.MjvCamera()
  camera.type = mujoco.mjtCamera.mjCAMERA_FREE
  camera.lookat[:] = [0.55, -0.055, 0.49]
  camera.distance = 0.68
  camera.azimuth = 132.0
  camera.elevation = -24.0
  return camera


def _load_trace(path: Path) -> dict[str, np.ndarray]:
  with np.load(path, allow_pickle=False) as archive:
    return {name: np.array(archive[name], copy=True) for name in archive.files}


def _contact_set(mask: np.ndarray) -> str:
  members = [str(index + 1) for index, active in enumerate(mask) if active]
  return "{" + ",".join(members) + "}" if members else "EMPTY"


def _certificate_plans(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
  return {
    str(event["certificate_id"]): event
    for event in events
    if event.get("event") == "CERTIFIED_PREFIX_COMMITTED"
    and event.get("certificate_id")
  }


def _event_timeline(events: list[dict[str, Any]]) -> tuple[np.ndarray, list[dict[str, Any]]]:
  visible = {
    "ORACLE_GOAL_PUBLISHED",
    "CERTIFIED_PREFIX_COMMITTED",
    "REPLAN_REJECTED",
    "GEODESIC_STAGNATION_BREAK_REQUESTED",
    "FRESH_MEASURED_MICRO_BARRIER",
    "GOAL_ARRIVED",
    "M06_SAFE_HOLD_REPLAN",
  }
  selected = sorted(
    (
      event
      for event in events
      if event.get("event") in visible and "time_s" in event
    ),
    key=lambda event: float(event["time_s"]),
  )
  return (
    np.asarray([float(event["time_s"]) for event in selected]),
    selected,
  )


def _event_label(event: dict[str, Any] | None) -> str:
  if event is None:
    return "planner event: acquisition"
  kind = str(event.get("event", ""))
  if kind == "ORACLE_GOAL_PUBLISHED":
    return f"Oracle publishes G{event.get('goal_id')} (finger ID: none)"
  if kind == "CERTIFIED_PREFIX_COMMITTED":
    primitive = event.get("selected_primitive", "-")
    finger = event.get("selected_finger")
    suffix = f"({finger})" if finger else ""
    return f"M11 selects {primitive}{suffix} · M10 certifies → M06 executes"
  if kind == "REPLAN_REJECTED":
    reason = str(event.get("reason", "unknown")).replace("\n", " ")
    return "replan rejected · " + reason[:105]
  if kind == "GEODESIC_STAGNATION_BREAK_REQUESTED":
    return (
      f"geodesic stagnation F{event.get('finger')} · request certified BREAK"
    )
  if kind == "FRESH_MEASURED_MICRO_BARRIER":
    contacts = "{" + ",".join(map(str, event.get("actual_contact_set", []))) + "}"
    return f"fresh measured barrier · A_real={contacts} · replan next tick"
  if kind == "GOAL_ARRIVED":
    fingers = ",".join(map(str, event.get("arrival_fingers", [])))
    return f"ARRIVE G{event.get('goal_id')} by F{fingers}"
  if kind == "M06_SAFE_HOLD_REPLAN":
    return f"M06 SAFE_HOLD · {event.get('reason', 'unknown')}"
  return kind


def _certificate_start_indices(certificate_ids: np.ndarray) -> np.ndarray:
  starts = np.zeros(len(certificate_ids), dtype=np.int32)
  start = 0
  previous = ""
  for index, certificate in enumerate(certificate_ids.astype(str)):
    if index == 0 or certificate != previous:
      start = index
      previous = certificate
    starts[index] = start
  return starts


def _whole_hand_roles(
  contacts: np.ndarray,
  primitive: str,
  selected_finger: int,
  plan: dict[str, Any] | None,
) -> dict[str, str]:
  if plan is not None and isinstance(plan.get("whole_hand_roles"), dict):
    return {
      str(finger): str(role)
      for finger, role in plan["whole_hand_roles"].items()
    }
  roles = {
    str(index + 1): ("ANCHOR" if active else "FREE")
    for index, active in enumerate(contacts)
  }
  if selected_finger:
    selected_roles = {
      "SLIDE": "EXPLORER",
      "MAKE": "REPLACEMENT_MAKE",
      "REPOSITION": "REPLACEMENT_STAGING",
      "BREAK": "RELEASE",
    }
    if primitive in selected_roles:
      roles[str(selected_finger)] = selected_roles[primitive]
  return roles


def _append_sphere(
  scene: mujoco.MjvScene,
  position: np.ndarray,
  radius: float,
  rgba: np.ndarray,
) -> None:
  if scene.ngeom >= len(scene.geoms):
    return
  geom = scene.geoms[scene.ngeom]
  mujoco.mjv_initGeom(
    geom,
    mujoco.mjtGeom.mjGEOM_SPHERE,
    np.full(3, radius, dtype=np.float64),
    np.asarray(position, dtype=np.float64),
    np.eye(3, dtype=np.float64).ravel(),
    rgba,
  )
  scene.ngeom += 1


def _append_connector(
  scene: mujoco.MjvScene,
  start: np.ndarray,
  end: np.ndarray,
  rgba: np.ndarray,
  *,
  arrow: bool,
) -> None:
  if scene.ngeom >= len(scene.geoms):
    return
  geom = scene.geoms[scene.ngeom]
  mujoco.mjv_initGeom(
    geom,
    mujoco.mjtGeom.mjGEOM_LINE,
    np.zeros(3, dtype=np.float64),
    np.zeros(3, dtype=np.float64),
    np.eye(3, dtype=np.float64).ravel(),
    rgba,
  )
  mujoco.mjv_connector(
    geom,
    mujoco.mjtGeom.mjGEOM_ARROW if arrow else mujoco.mjtGeom.mjGEOM_LINE,
    0.0018 if arrow else 3.0,
    np.asarray(start, dtype=np.float64),
    np.asarray(end, dtype=np.float64),
  )
  scene.ngeom += 1


def _annotate_scene(
  scene: mujoco.MjvScene,
  handles: Any,
  data: mujoco.MjData,
  trace: dict[str, np.ndarray],
  index: int,
  bunny_vertices_m: np.ndarray,
) -> None:
  goal_vertex = int(trace["goal_vertex"][index])
  bridge_vertex = int(trace["bridge_target_vertex"][index])
  goal = None
  bridge = None
  if 0 <= goal_vertex < len(bunny_vertices_m):
    goal = handles.object_position_m + bunny_vertices_m[goal_vertex]
    _append_sphere(scene, goal, 0.0060, GOAL_RGBA)
  if 0 <= bridge_vertex < len(bunny_vertices_m):
    bridge = handles.object_position_m + bunny_vertices_m[bridge_vertex]
    _append_sphere(scene, bridge, 0.0042, BRIDGE_RGBA)

  contacts = trace["contact_active"][index]
  for finger in np.flatnonzero(contacts):
    _append_sphere(
      scene,
      trace["contact_positions_world_m"][index, finger],
      0.0025,
      ANCHOR_RGBA,
    )

  primitive = str(trace["primitive"][index])
  selected = int(trace["selected_finger"][index])
  if selected > 0:
    selected_index = selected - 1
    start = np.asarray(
      trace["contact_positions_world_m"][index, selected_index]
      if contacts[selected_index]
      else data.site_xpos[handles.tip_site_ids[selected_index]],
      dtype=np.float64,
    )
    if primitive == "BREAK":
      normal = np.asarray(
        trace["contact_normals_world"][index, selected_index],
        dtype=np.float64,
      )
      normal /= max(float(np.linalg.norm(normal)), 1e-12)
      _append_connector(scene, start, start + 0.018 * normal, BRIDGE_RGBA, arrow=True)
    elif bridge is not None and primitive in {"SLIDE", "MAKE", "REPOSITION"}:
      _append_connector(scene, start, bridge, BRIDGE_RGBA, arrow=True)
  elif primitive == "WRIST_ADJUST" and bridge is not None:
    active = np.flatnonzero(contacts)
    if len(active):
      root_index = min(
        active,
        key=lambda finger: float(
          np.linalg.norm(
            trace["contact_positions_world_m"][index, finger] - bridge
          )
        ),
      )
      root = trace["contact_positions_world_m"][index, root_index]
      normal = trace["contact_normals_world"][index, root_index]
      direction = bridge - root
      direction -= float(np.dot(direction, normal)) * normal
      norm = float(np.linalg.norm(direction))
      if norm > 1e-12:
        start = np.asarray(data.site_xpos[handles.palm_site_id])
        _append_connector(
          scene,
          start,
          start + 0.035 * direction / norm,
          BRIDGE_RGBA,
          arrow=True,
        )

  trail_start = max(0, index - 100)
  trail_indices = np.arange(trail_start, index + 1, 5, dtype=np.int32)
  if len(trail_indices) >= 2:
    if selected > 0:
      points = trace["fingertip_positions_world_m"][trail_indices, selected - 1]
    elif primitive == "WRIST_ADJUST":
      points = trace["palm_pose_world"][trail_indices, :3]
    else:
      points = np.empty((0, 3), dtype=np.float64)
    for start, end in zip(points[:-1], points[1:]):
      _append_connector(scene, start, end, TRAIL_RGBA, arrow=False)


def _overlay(
  frame: np.ndarray,
  trace: dict[str, np.ndarray],
  index: int,
  summary: dict[str, Any],
  speed: float,
  plan: dict[str, Any] | None,
  latest_event: dict[str, Any] | None,
  certificate_start: int,
) -> np.ndarray:
  image = Image.fromarray(frame).convert("RGBA")
  draw = ImageDraw.Draw(image)
  draw.rounded_rectangle((18, 14, image.width - 18, 116), 14, fill=(7, 20, 35, 230))
  draw.text((34, 24), "I04 · ORACLE NEXT-POINT WHOLE-HAND TRAVERSAL", font=_font(22, bold=True), fill=(81, 220, 167, 255))
  draw.text(
    (34, 61),
    "MuJoCo Bunny SDF | Explicit MCC | M01-M12 | DPRef OFF | GPIS OFF",
    font=_font(16),
    fill=(207, 225, 241, 255),
  )
  draw.text(
    (image.width - 250, 28),
    f"t={float(trace['time_s'][index]):6.1f}s  x{speed:g}",
    font=_font(19, bold=True),
    fill="white",
  )

  top = image.height - 218
  draw.rounded_rectangle((18, top, image.width - 18, image.height - 16), 14, fill=(7, 20, 35, 234))
  contacts = trace["contact_active"][index]
  goal = int(trace["goal_id"][index])
  bridge = int(trace["bridge_target_vertex"][index])
  primitive = str(trace["primitive"][index])
  selected = int(trace["selected_finger"][index])
  coverage = 100.0 * float(trace["coverage_fraction"][index])
  draw.text(
    (34, top + 13),
    f"real A={_contact_set(contacts)}  goal={goal if goal >= 0 else '-'}  "
    f"primitive={primitive}  finger={selected if selected else '-'}",
    font=_font(18, bold=True),
    fill="white",
  )
  draw.text(
    (image.width - 285, top + 15),
    f"coverage {coverage:6.2f}%",
    font=_font(17, bold=True),
    fill=(81, 220, 167, 255),
  )
  roles = _whole_hand_roles(contacts, primitive, selected, plan)
  role_codes = {
    "ANCHOR": "A",
    "FREE": "F",
    "EXPLORER": "E",
    "REPLACEMENT_MAKE": "M",
    "REPLACEMENT_STAGING": "S",
    "RELEASE": "R",
  }
  role_text = "  ".join(
    f"F{finger}:{role_codes.get(roles.get(str(finger), 'FREE'), '?')}"
    for finger in range(1, 5)
  )
  certificate = str(trace["certificate_id"][index])
  planned_step_m = (
    float(plan.get("committed_participant_displacement_m", 0.0))
    if plan is not None
    else 0.0
  )
  actual_step_m = 0.0
  if certificate != "NONE" and certificate_start <= index:
    if primitive == "WRIST_ADJUST":
      actual_step_m = float(
        np.linalg.norm(
          trace["palm_pose_world"][index, :3]
          - trace["palm_pose_world"][certificate_start, :3]
        )
      )
    elif selected > 0:
      actual_step_m = float(
        np.linalg.norm(
          trace["fingertip_positions_world_m"][index, selected - 1]
          - trace["fingertip_positions_world_m"][certificate_start, selected - 1]
        )
      )
  draw.text(
    (34, top + 45),
    f"roles {role_text}  bridge-v={bridge if bridge >= 0 else '-'}  "
    f"plan/actual Δ={1000*planned_step_m:4.1f}/{1000*actual_step_m:4.1f} mm",
    font=_font(15, bold=True),
    fill=(189, 215, 234, 255),
  )
  draw.text(
    (34, top + 73),
    _event_label(latest_event),
    font=_font(14),
    fill=(111, 211, 255, 255),
  )
  forces = trace["fingertip_forces_n"][index]
  bar_y = top + 105
  for finger in range(4):
    x = 82 + 207 * finger
    draw.text((x - 46, bar_y + 2), f"F{finger + 1}", font=_font(15, bold=True), fill="white")
    draw.rounded_rectangle((x, bar_y, x + 128, bar_y + 22), 5, fill=(61, 76, 93, 255))
    width = 124.0 * float(np.clip(forces[finger] / 8.0, 0.0, 1.0))
    if width > 0.0:
      draw.rounded_rectangle((x + 2, bar_y + 2, x + 2 + width, bar_y + 20), 3, fill=FINGER_COLORS[finger])
    draw.text((x, bar_y + 30), f"{forces[finger]:4.2f} N", font=_font(14), fill=FINGER_COLORS[finger])
  draw.text(
    (34, image.height - 42),
    "MAGENTA final goal (never executed directly) · GREEN local plan/bridge · CYAN actual 1 s trail",
    font=_font(14, bold=True),
    fill=(147, 190, 223, 255),
  )
  return np.asarray(image.convert("RGB"))


def _render_video(
  output: Path,
  trace: dict[str, np.ndarray],
  summary: dict[str, Any],
  *,
  speed: float,
  fps: int,
  codec: str,
  events: list[dict[str, Any]],
) -> tuple[Path, Path]:
  mesh = (output / "canonical_bunny_side_laid.obj").resolve()
  handles = build_full_robot(
    FullRobotModelConfig(
      surface="bunny",
      gravity_m_s2=0.0,
      arm_kp=1800.0,
      arm_damping_ratio=0.9,
      hand_kp=60.0,
      hand_damping_ratio=1.5,
      bunny_visual_mesh_path=str(mesh),
      bunny_collision_mode="sdf",
    )
  )
  data = mujoco.MjData(handles.model)
  renderer = mujoco.Renderer(handles.model, width=960, height=540)
  path = output / f"i04_bunny_replay_x{speed:g}_{codec}.mp4"
  screenshot = output / "i04_bunny_replay_frame.png"
  if codec == "h264_nvenc":
    writer = imageio.get_writer(
      path,
      fps=fps,
      codec=codec,
      macro_block_size=1,
      output_params=[
        # The workstation FFmpeg exposes the pre-P1..P7 NVENC preset names.
        # ``hq`` is the compatible high-quality preset on that API.
        "-preset", "hq",
        "-rc", "vbr",
        "-cq", "19",
        "-b:v", "0",
        "-pix_fmt", "yuv420p",
      ],
    )
  else:
    writer = imageio.get_writer(
      path,
      fps=fps,
      codec="libx264",
      quality=8,
      macro_block_size=1,
    )
  times = trace["time_s"]
  source_times = np.arange(float(times[0]), float(times[-1]) + 1e-9, speed / fps)
  indices = np.searchsorted(times, source_times, side="left")
  indices = np.clip(indices, 0, len(times) - 1)
  camera = _camera()
  bunny_vertices_m = np.asarray(
    canonical_bunny_heightfield().mesh.vertices,
    dtype=np.float64,
  )
  plans = _certificate_plans(events)
  event_times, timeline = _event_timeline(events)
  certificate_starts = _certificate_start_indices(trace["certificate_id"])
  last_frame: np.ndarray | None = None
  try:
    for index in indices:
      data.qpos[handles.arm_qpos_adrs] = trace["arm_q_rad"][index]
      data.qpos[handles.hand_qpos_adrs] = trace["finger_q_rad"][index]
      mujoco.mj_forward(handles.model, data)
      renderer.update_scene(data, camera=camera)
      try:
        renderer.scene.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
      except (AttributeError, IndexError, TypeError):
        pass
      _annotate_scene(
        renderer.scene,
        handles,
        data,
        trace,
        int(index),
        bunny_vertices_m,
      )
      certificate = str(trace["certificate_id"][index])
      plan = plans.get(certificate)
      event_index = int(
        np.searchsorted(event_times, float(times[index]), side="right") - 1
      )
      latest_event = timeline[event_index] if event_index >= 0 else None
      last_frame = _overlay(
        renderer.render().copy(),
        trace,
        int(index),
        summary,
        speed,
        plan,
        latest_event,
        int(certificate_starts[index]),
      )
      writer.append_data(last_frame)
  finally:
    writer.close()
    renderer.close()
  if last_frame is None:
    raise RuntimeError("trace contains no replay frames")
  imageio.imwrite(screenshot, last_frame)
  return path, screenshot


def _dashboard(output: Path, trace: dict[str, np.ndarray], summary: dict[str, Any]) -> Path:
  plt = get_pyplot()
  figure, axes = plt.subplots(2, 2, figsize=(15, 9), dpi=150, facecolor="#071423")
  for axis in axes.ravel():
    axis.set_facecolor("#0d2033")
    axis.grid(color="#294055", alpha=0.65, linewidth=0.7)
    axis.tick_params(colors="#b9cedf")
    for spine in axis.spines.values():
      spine.set_color("#294055")
  time = trace["time_s"]
  axes[0, 0].plot(time, 100.0 * trace["coverage_fraction"], color="#51dca7")
  axes[0, 0].set(title="Required Bunny goal coverage", xlabel="time [s]", ylabel="coverage [%]")
  for finger in range(4):
    axes[0, 1].plot(time, trace["fingertip_forces_n"][:, finger], color=FINGER_COLORS[finger], linewidth=0.9, label=f"F{finger+1}")
  axes[0, 1].axhline(8.0, color="#ff5e6c", linestyle="--", label="8 N hard limit")
  axes[0, 1].set(title="Physical fingertip force", xlabel="time [s]", ylabel="force [N]")
  axes[0, 1].legend(facecolor="#0d2033", labelcolor="white", ncol=3, fontsize=8)
  axes[1, 0].step(time, np.sum(trace["contact_active"], axis=1), where="post", color="#6db7ff")
  axes[1, 0].set(title="Measured contact count", xlabel="time [s]", ylabel="|A_real|", ylim=(-0.1, 4.4), yticks=(0, 1, 2, 3, 4))
  axes[1, 1].plot(time, 1000.0 * trace["controller_latency_s"], color="#f39c35", label="controller")
  axes[1, 1].plot(time, 1000.0 * trace["physics_latency_s"], color="#ed5a7a", label="physics")
  axes[1, 1].set(title="Per logged tick latency", xlabel="time [s]", ylabel="latency [ms]")
  axes[1, 1].legend(facecolor="#0d2033", labelcolor="white")
  for axis in axes.ravel():
    axis.title.set_color("white")
    axis.xaxis.label.set_color("#b9cedf")
    axis.yaxis.label.set_color("#b9cedf")
  figure.suptitle(
    f"I04 Explicit MCC · {summary['visited_goal_count']}/{summary['required_goal_count']} goals · "
    f"contact {100*summary['contact_continuity_fraction']:.3f}% · {summary['stop_reason']}",
    color="white",
    fontsize=16,
    weight="bold",
  )
  figure.tight_layout(rect=(0, 0, 1, 0.95))
  path = output / "i04_bunny_dashboard.png"
  save_figure(figure, path)
  plt.close(figure)
  return path


def _write_html(output: Path, summary: dict[str, Any], video: Path, dashboard: Path) -> Path:
  html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>I04 Bunny traversal</title><style>body{{margin:0;background:#071423;color:#eaf4fb;font-family:Arial,sans-serif}}main{{max-width:1120px;margin:auto;padding:34px}}p{{color:#b8cede;line-height:1.65}}.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}.card{{background:#0d2033;border:1px solid #294055;border-radius:14px;padding:17px}}.k{{color:#92aec3;font-size:12px}}.v{{font-size:23px;font-weight:800;margin-top:7px}}img,video{{width:100%;border-radius:14px;border:1px solid #294055;margin-top:20px}}code{{color:#8fd5ff}}</style></head><body><main><h1>I04 · Oracle Next-Point Whole-Hand Contact Traversal</h1><p>Oracle 只发布 Bunny surface goal，不发布 finger ID。Explicit MCC baseline 从每个 MuJoCo 实测 barrier 重新选择 finger、contact primitive 和局部 mesh-geodesic bridge；DPRef 与 GPIS 均关闭。紫色球是最终目标，绿色球与箭头是当前局部规划，青色轨迹是最近 1 秒实际运动。</p><div class="grid"><div class="card"><div class="k">GOALS</div><div class="v">{summary['visited_goal_count']}/{summary['required_goal_count']}</div></div><div class="card"><div class="k">CONTACT</div><div class="v">{100*summary['contact_continuity_fraction']:.3f}%</div></div><div class="card"><div class="k">MAX GAP</div><div class="v">{1000*summary['maximum_contact_gap_s']:.1f} ms</div></div><div class="card"><div class="k">PEAK FORCE</div><div class="v">{summary['peak_fingertip_force_n']:.2f} N</div></div></div><video controls preload="metadata" src="{video.name}"></video><img src="{dashboard.name}" alt="I04 dashboard"><h2>复现该 GPU 回放</h2><p><code>MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg /home/ferry/data/Anaconda/envs/handcomp/bin/python -m Module.i04_oracle_next_point.visual_demo --output {output} --reuse --speed 1 --fps 24 --codec h264_nvenc</code></p></main></body></html>"""
  path = output / "index.html"
  path.write_text(html, encoding="utf-8")
  return path


def run_visual(
  output_dir: Path = DEFAULT_OUTPUT_DIR,
  *,
  speed: float = 12.0,
  fps: int = 24,
  codec: str = "libx264",
  reuse: bool = False,
) -> dict[str, Any]:
  if speed <= 0.0 or fps < 1:
    raise ValueError("speed and fps must be positive")
  if codec not in {"libx264", "h264_nvenc"}:
    raise ValueError("codec must be libx264 or h264_nvenc")
  output = output_dir.resolve()
  summary_path = output / "summary.json"
  trace_path = output / "trace.npz"
  if not summary_path.is_file() or not trace_path.is_file():
    raise FileNotFoundError("run Module.i04_oracle_next_point.benchmark first")
  summary = json.loads(summary_path.read_text(encoding="utf-8"))
  trace = _load_trace(trace_path)
  events_path = output / "events.json"
  events = (
    json.loads(events_path.read_text(encoding="utf-8"))
    if events_path.is_file()
    else []
  )
  video, screenshot = _render_video(
    output,
    trace,
    summary,
    speed=speed,
    fps=fps,
    codec=codec,
    events=events,
  )
  dashboard = _dashboard(output, trace, summary)
  index = _write_html(output, summary, video, dashboard)
  result = {
    "reuse_requested": reuse,
    "source_summary_sha256": __import__("hashlib").sha256(summary_path.read_bytes()).hexdigest(),
    "speed": speed,
    "fps": fps,
    "codec": codec,
    "mujoco_gl": os.environ.get("MUJOCO_GL", "UNSET"),
    "mujoco_egl_device_id": os.environ.get("MUJOCO_EGL_DEVICE_ID", "UNSET"),
    "artifacts": {
      "video": video.name,
      "screenshot": screenshot.name,
      "dashboard": dashboard.name,
      "index": index.name,
    },
  }
  (output / "visual_summary.json").write_text(
    json.dumps(result, indent=2, sort_keys=True),
    encoding="utf-8",
  )
  return result


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
  parser.add_argument("--reuse", action="store_true")
  parser.add_argument("--speed", type=float, default=12.0)
  parser.add_argument("--fps", type=int, default=24)
  parser.add_argument(
    "--codec",
    choices=("libx264", "h264_nvenc"),
    default="libx264",
  )
  args = parser.parse_args()
  print(
    json.dumps(
      run_visual(
        args.output,
        speed=args.speed,
        fps=args.fps,
        codec=args.codec,
        reuse=args.reuse,
      ),
      indent=2,
    )
  )


if __name__ == "__main__":
  main()
