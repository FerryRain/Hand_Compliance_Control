"""Render the frozen I01 Bunny traces into videos, dashboard and HTML."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any

os.environ.setdefault("MUJOCO_GL", "osmesa")
cache = Path(tempfile.gettempdir()) / "handcomp-i01-bunny-mesa"
cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(cache))
os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from Module.fr3_leap import FullRobotModelConfig, build_full_robot
from Module.i01_bunny_physics.benchmark import DEFAULT_OUTPUT_DIR
from Module.visualization import get_pyplot, save_figure


FINGER_COLORS = ("#2997D6", "#3CBF91", "#F39C35", "#ED5A7A")


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
  name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
  path = Path("/usr/share/fonts/truetype/dejavu") / name
  return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def _camera() -> mujoco.MjvCamera:
  camera = mujoco.MjvCamera()
  camera.type = mujoco.mjtCamera.mjCAMERA_FREE
  camera.lookat[:] = [0.55, -0.055, 0.50]
  camera.distance = 0.72
  camera.azimuth = 135.0
  camera.elevation = -27.0
  return camera


def _contact_set(mask: np.ndarray) -> str:
  members = [str(index + 1) for index, active in enumerate(mask) if active]
  return "{" + ",".join(members) + "}" if members else "EMPTY"


def _load_trace(path: Path) -> dict[str, np.ndarray]:
  with np.load(path, allow_pickle=False) as archive:
    return {name: np.array(archive[name], copy=True) for name in archive.files}


def _overlay(
  frame: np.ndarray,
  trace: dict[str, np.ndarray],
  index: int,
  cell: str,
) -> np.ndarray:
  image = Image.fromarray(frame).convert("RGBA")
  draw = ImageDraw.Draw(image)
  variable = cell == "variable"
  color = (74, 214, 160, 255) if variable else (255, 174, 66, 255)
  title = (
    "I01-B  VARIABLE CONTACT MODE"
    if variable
    else "I01-A  FIXED FOUR-CONTACT BASELINE"
  )
  draw.rounded_rectangle((18, 14, image.width - 18, 112), 14, fill=(7, 20, 35, 226))
  draw.text((34, 26), title, font=_font(23, bold=True), fill=color)
  draw.text(
    (34, 64),
    "FR3 + LEAP  |  fixed Bunny  |  MuJoCo contact  |  MCC only  |  DP OFF",
    font=_font(16),
    fill=(207, 225, 241, 255),
  )
  draw.text(
    (image.width - 180, 30),
    f"t={float(trace['time_s'][index]):4.1f} s",
    font=_font(20, bold=True),
    fill="white",
  )

  panel_top = image.height - 174
  draw.rounded_rectangle(
    (18, panel_top, image.width - 18, image.height - 16),
    14,
    fill=(7, 20, 35, 232),
  )
  contacts = trace["mesh_valid_contacts"][index]
  forces = trace["fingertip_forces_n"][index]
  progress_mm = 1000.0 * float(trace["actual_progress_m"][index])
  phase = str(trace["transaction_phase"][index])
  state = str(trace["transaction_state"][index])
  guard = str(trace["guard_reason"][index])
  draw.text(
    (34, panel_top + 14),
    f"measured A = {_contact_set(contacts)}    actual progress = {progress_mm:5.1f} mm",
    font=_font(19, bold=True),
    fill="white",
  )
  status = f"phase {phase} / {state}" if variable else f"guard {guard}"
  draw.text(
    (image.width - 340, panel_top + 17),
    status,
    font=_font(15, bold=True),
    fill=color,
  )
  bar_y = panel_top + 58
  for finger in range(4):
    x = 82 + 207 * finger
    draw.text((x - 46, bar_y + 2), f"F{finger + 1}", font=_font(15, bold=True), fill="white")
    draw.rounded_rectangle((x, bar_y, x + 128, bar_y + 22), 5, fill=(61, 76, 93, 255))
    width = 124.0 * float(np.clip(forces[finger] / 8.0, 0.0, 1.0))
    if width > 0.0:
      draw.rounded_rectangle(
        (x + 2, bar_y + 2, x + 2 + width, bar_y + 20),
        3,
        fill=FINGER_COLORS[finger],
      )
    draw.text(
      (x, bar_y + 30),
      f"{forces[finger]:4.2f} N",
      font=_font(14),
      fill=FINGER_COLORS[finger] if contacts[finger] else (168, 181, 195, 255),
    )
  footer = (
    "M10 certificate -> M06 prefix -> measured micro barrier"
    if variable
    else "motion stops after >40 ms violation of |A|=4"
  )
  draw.text((34, image.height - 40), footer, font=_font(15, bold=True), fill=color)
  return np.asarray(image.convert("RGB"))


def _render_pair(output: Path) -> dict[str, str]:
  traces = {
    cell: _load_trace(output / f"trace_{cell}_seed_7.npz")
    for cell in ("fixed", "variable")
  }
  mesh = (output / "canonical_bunny_side_laid.obj").resolve()
  handles = {
    cell: build_full_robot(
      FullRobotModelConfig(
        surface="bunny",
        gravity_m_s2=0.0,
        arm_kp=1800.0,
        arm_damping_ratio=0.9,
        object_offset_x_m=0.002,
        object_offset_y_m=-0.005,
        object_offset_z_m=-0.003,
        bunny_visual_mesh_path=str(mesh),
      )
    )
    for cell in ("fixed", "variable")
  }
  data = {cell: mujoco.MjData(handles[cell].model) for cell in handles}
  renderers = {
    cell: mujoco.Renderer(handles[cell].model, width=960, height=540)
    for cell in handles
  }
  camera = _camera()
  fps = 12
  frame_times = np.arange(0.0, 12.0, 1.0 / fps)
  indices = np.unique(np.clip(np.round(frame_times / 0.002).astype(int), 0, 5999))
  paths = {
    "fixed_video": output / "i01_fixed_seed7.mp4",
    "variable_video": output / "i01_variable_seed7.mp4",
    "side_by_side_video": output / "i01_fixed_vs_variable.mp4",
  }
  writers = {
    name: imageio.get_writer(
      path,
      fps=fps,
      codec="libx264",
      quality=8,
      macro_block_size=1,
    )
    for name, path in paths.items()
  }
  screenshots: dict[str, np.ndarray] = {}
  try:
    for index in indices:
      frames: dict[str, np.ndarray] = {}
      for cell in ("fixed", "variable"):
        h = handles[cell]
        d = data[cell]
        trace = traces[cell]
        d.qpos[h.arm_qpos_adrs] = trace["arm_q_rad"][index]
        d.qpos[h.hand_qpos_adrs] = trace["finger_q_rad"][index]
        mujoco.mj_forward(h.model, d)
        renderers[cell].update_scene(d, camera=camera)
        try:
          renderers[cell].scene.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
        except (AttributeError, IndexError, TypeError):
          pass
        frames[cell] = _overlay(renderers[cell].render().copy(), trace, int(index), cell)
        writers[f"{cell}_video"].append_data(frames[cell])
      side = np.concatenate(
        (
          np.asarray(Image.fromarray(frames["fixed"]).resize((640, 360))),
          np.asarray(Image.fromarray(frames["variable"]).resize((640, 360))),
        ),
        axis=1,
      )
      writers["side_by_side_video"].append_data(side)
      timestamp = float(traces["variable"]["time_s"][index])
      if "handover_frame" not in screenshots and timestamp >= 5.0:
        screenshots["handover_frame"] = side.copy()
      if "result_frame" not in screenshots and timestamp >= 11.8:
        screenshots["result_frame"] = side.copy()
  finally:
    for writer in writers.values():
      writer.close()
    for renderer in renderers.values():
      renderer.close()
  result: dict[str, str] = {name: path.name for name, path in paths.items()}
  for name, frame in screenshots.items():
    path = output / f"{name}.png"
    imageio.imwrite(path, frame)
    result[name] = path.name
  return result


def _dashboard(output: Path, summary: dict[str, Any]) -> Path:
  plt = get_pyplot()
  plt.rcParams["savefig.facecolor"] = "#071423"
  fixed = _load_trace(output / "trace_fixed_seed_7.npz")
  variable = _load_trace(output / "trace_variable_seed_7.npz")
  figure = plt.figure(figsize=(16, 10), dpi=150, facecolor="#071423")
  grid = figure.add_gridspec(3, 2, height_ratios=(0.62, 1.0, 1.0), hspace=0.36, wspace=0.23)
  title = figure.add_subplot(grid[0, :])
  title.set_facecolor("#071423")
  title.axis("off")
  fixed_mm = 1000.0 * summary["g2"]["median_l_fixed_m"]
  variable_mm = 1000.0 * summary["g2"]["median_l_variable_m"]
  advantage = 1000.0 * summary["g2"]["median_advantage_m"]
  title.text(0.0, 0.82, "I01 · PHYSICAL CONTACT-MODE TRAVERSAL ON BUNNY", color="white", fontsize=23, weight="bold")
  title.text(0.0, 0.50, "Exact Bunny visual mesh · Bunny-derived collision upper envelope · MuJoCo forces · MCC only", color="#afc8dc", fontsize=13)
  cards = [
    ("FIXED MEDIAN", f"{fixed_mm:.2f} mm", "#ffad42"),
    ("VARIABLE MEDIAN", f"{variable_mm:.2f} mm", "#4ad6a0"),
    ("FEASIBLE-SET GAIN", f"+{advantage:.2f} mm", "#6db7ff"),
    ("G2 DECISION", summary["g2"]["decision"], "#4ad6a0"),
  ]
  for index, (label, value, color) in enumerate(cards):
    x = 0.01 + 0.247 * index
    title.text(x, 0.16, label, color="#9fb9ce", fontsize=10, weight="bold")
    title.text(x, -0.04, value, color=color, fontsize=19, weight="bold")

  axes = [figure.add_subplot(grid[1, 0]), figure.add_subplot(grid[1, 1]), figure.add_subplot(grid[2, 0]), figure.add_subplot(grid[2, 1])]
  for axis in axes:
    axis.set_facecolor("#0d2033")
    axis.grid(color="#294055", alpha=0.65, linewidth=0.7)
    axis.tick_params(colors="#b9cedf")
    for spine in axis.spines.values():
      spine.set_color("#294055")

  axes[0].plot(fixed["time_s"], 1000 * fixed["actual_progress_m"], color="#ffad42", label="fixed")
  axes[0].plot(variable["time_s"], 1000 * variable["actual_progress_m"], color="#4ad6a0", label="variable")
  axes[0].axhline(50, color="#6db7ff", linestyle="--", linewidth=1.1, label="50 mm pass")
  axes[0].set(title="Actual palm progress", xlabel="time [s]", ylabel="progress [mm]")
  axes[0].legend(facecolor="#0d2033", labelcolor="white", loc="upper left")

  for finger in range(4):
    axes[1].plot(variable["time_s"], variable["fingertip_forces_n"][:, finger], color=FINGER_COLORS[finger], linewidth=1.0, label=f"F{finger+1}")
  axes[1].axhline(8.0, color="#ff5e6c", linestyle="--", label="8 N hard limit")
  axes[1].set(title="Variable-mode physical fingertip force", xlabel="time [s]", ylabel="force [N]", ylim=(-0.2, 8.6))
  axes[1].legend(facecolor="#0d2033", labelcolor="white", ncol=3, fontsize=8)

  axes[2].step(fixed["time_s"], np.sum(fixed["mesh_valid_contacts"], axis=1), where="post", color="#ffad42", label="fixed")
  axes[2].step(variable["time_s"], np.sum(variable["mesh_valid_contacts"], axis=1), where="post", color="#4ad6a0", label="variable")
  axes[2].axvspan(4.5, 5.64, color="#6db7ff", alpha=0.12, label="certified handover")
  axes[2].set(title="Measured Bunny contact count", xlabel="time [s]", ylabel="|A_actual|", ylim=(-0.1, 4.4), yticks=(0, 1, 2, 3, 4))
  axes[2].legend(facecolor="#0d2033", labelcolor="white", fontsize=8)

  for x, cell, color in ((0, "fixed", "#ffad42"), (1, "variable", "#4ad6a0")):
    values = 1000 * np.asarray(summary["cells"][cell]["actual_progress_m"]["values"])
    axes[3].scatter(np.full(3, x) + [-0.04, 0.0, 0.04], values, s=60, color=color, edgecolor="white", linewidth=0.7)
    axes[3].plot([x - 0.16, x + 0.16], [np.median(values)] * 2, color="white", linewidth=2.0)
  axes[3].axhline(50, color="#6db7ff", linestyle="--", linewidth=1.1)
  axes[3].set(title="Paired-seed feasible traversal", ylabel="actual progress [mm]", xticks=(0, 1), xticklabels=("fixed |A|=4", "variable |A|≥1"), xlim=(-0.4, 1.4))
  for axis in axes:
    axis.title.set_color("white")
    axis.xaxis.label.set_color("#b9cedf")
    axis.yaxis.label.set_color("#b9cedf")
  path = output / "i01_bunny_dashboard.png"
  save_figure(figure, path)
  plt.close(figure)
  return path


def _write_html(output: Path, summary: dict[str, Any], artifacts: dict[str, str]) -> Path:
  fixed = summary["cells"]["fixed"]
  variable = summary["cells"]["variable"]
  html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>I01 Bunny Physics Review</title>
<style>
body{{margin:0;background:#071423;color:#eaf4fb;font-family:Inter,Arial,sans-serif}}main{{max-width:1180px;margin:auto;padding:34px}}h1{{font-size:36px;margin:0 0 8px}}h2{{margin-top:34px}}p{{color:#b8cede;line-height:1.65}}.tag{{display:inline-block;background:#15344d;color:#59dda8;padding:7px 12px;border-radius:999px;font-weight:700}}.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:25px 0}}.card{{background:#0d2033;border:1px solid #294055;border-radius:15px;padding:18px}}.k{{color:#92aec3;font-size:12px;font-weight:700}}.v{{font-size:24px;font-weight:800;margin-top:8px}}img,video{{width:100%;border-radius:14px;border:1px solid #294055;background:#000}}table{{width:100%;border-collapse:collapse;background:#0d2033}}th,td{{padding:12px;border-bottom:1px solid #294055;text-align:left}}th{{color:#91b4cc}}code{{color:#8fd5ff}}.two{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}@media(max-width:850px){{.grid{{grid-template-columns:1fr 1fr}}.two{{grid-template-columns:1fr}}}}
</style></head><body><main>
<span class="tag">{summary['status']} · G2 {summary['g2']['decision']}</span><h1>I01：Bunny 上的真实连续接触移动</h1>
<p>目的：回答 FR3+LEAP 在 MuJoCo 中能否沿 Bunny 表面移动并保持真实接触，以及允许经证书约束的 contact-mode handover 是否能突破固定四指的局部可行域。完整 Bunny 三角网格负责视觉与残差审计；同一网格派生的 181×181 upper-envelope hfield 负责碰撞。</p>
<div class="grid"><div class="card"><div class="k">FIXED MEDIAN</div><div class="v">{1000*summary['g2']['median_l_fixed_m']:.2f} mm</div></div><div class="card"><div class="k">VARIABLE MEDIAN</div><div class="v">{1000*summary['g2']['median_l_variable_m']:.2f} mm</div></div><div class="card"><div class="k">MEDIAN GAIN</div><div class="v">+{1000*summary['g2']['median_advantage_m']:.2f} mm</div></div><div class="card"><div class="k">VARIABLE WORST PEAK</div><div class="v">{variable['peak_valid_fingertip_force_n']['maximum']:.3f} N</div></div></div>
<img src="{artifacts['dashboard']}" alt="I01 dashboard"><h2>同步物理视频</h2><video controls preload="metadata" src="{artifacts['side_by_side_video']}"></video>
<h2>目的、效果与性能</h2><table><thead><tr><th>Cell</th><th>目的</th><th>效果</th><th>性能</th></tr></thead><tbody><tr><td>I01-A fixed</td><td>强制 |A|=4，测局部可行边界</td><td>3/3 在约 20 mm 持续破坏四指模式并按规则停止；仍未全失联</td><td>median {1000*fixed['actual_progress_m']['median']:.2f} mm；continuity {100*fixed['nonempty_contact_fraction']['median']:.3f}%</td></tr><tr><td>I01-B variable</td><td>非空 mode + M10/M06 4→3→4 handover</td><td>3/3 完成 60 mm 计划的约 58.43 mm 实际移动</td><td>median {1000*variable['actual_progress_m']['median']:.2f} mm；continuity {100*variable['nonempty_contact_fraction']['median']:.3f}%；peak {variable['peak_valid_fingertip_force_n']['maximum']:.3f} N</td></tr></tbody></table>
<h2>执行权限与结果边界</h2><p>Variable 三个 episode 合计 {variable['certificate_count']} 个 M10 certificate、{variable['micro_barrier_count']} 个真实 micro barrier、{variable['authority_violation_count']} 个 authority violation。DP 未使用。结果是 gravity-off、固定 Bunny 的 control-isolation 仿真结论，不是硬件或完整非凸 mesh collision 结论；计分接触必须离完整 mesh ≤2.5 mm。</p>
<div class="two"><video controls preload="metadata" src="{artifacts['fixed_video']}"></video><video controls preload="metadata" src="{artifacts['variable_video']}"></video></div>
<h2>复现</h2><p><code>/home/ferry/data/Anaconda/envs/handcomp/bin/python -m Module.i01_bunny_physics.benchmark</code><br><code>MUJOCO_GL=osmesa /home/ferry/data/Anaconda/envs/handcomp/bin/python -m Module.i01_bunny_physics.visual_demo --reuse</code></p>
</main></body></html>"""
  path = output / "index.html"
  path.write_text(html, encoding="utf-8")
  return path


def run_visual(output: Path = DEFAULT_OUTPUT_DIR, *, reuse: bool = False) -> dict[str, Any]:
  output = output.resolve()
  summary_path = output / "summary.json"
  if not summary_path.is_file():
    raise FileNotFoundError("run Module.i01_bunny_physics.benchmark first")
  summary = json.loads(summary_path.read_text(encoding="utf-8"))
  artifacts = _render_pair(output)
  artifacts["dashboard"] = _dashboard(output, summary).name
  artifacts["index"] = _write_html(output, summary, artifacts).name
  result = {
    "module_id": summary["module_id"],
    "source_summary_sha256": __import__("hashlib").sha256(summary_path.read_bytes()).hexdigest(),
    "metrics_recomputed": False,
    "reuse_requested": reuse,
    "artifacts": artifacts,
  }
  (output / "visual_summary.json").write_text(
    json.dumps(result, indent=2, sort_keys=True),
    encoding="utf-8",
  )
  return result


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
  parser.add_argument("--reuse", action="store_true", help="declare that frozen benchmark traces are reused")
  args = parser.parse_args()
  print(json.dumps(run_visual(args.output, reuse=args.reuse), indent=2))


if __name__ == "__main__":
  main()
