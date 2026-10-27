"""Render frozen I02/I03 traces into review videos, dashboard and HTML."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

os.environ.setdefault("MUJOCO_GL", "osmesa")
_CACHE = Path(tempfile.gettempdir()) / "handcomp-i02-i03-bunny-mesa"
_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE))
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE / "matplotlib"))

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from Module.fr3_leap import FullRobotModelConfig, build_full_robot
from Module.i02_i03_bunny_physics.benchmark import DEFAULT_OUTPUT_DIR
from Module.visualization import get_pyplot, save_figure


COLORS = {
  "i02_long": "#ffad42",
  "i02_short": "#4ad6a0",
  "i03_beam": "#ed5a7a",
  "i03_shadow": "#6db7ff",
}
FINGER_COLORS = ("#2997D6", "#3CBF91", "#F39C35", "#ED5A7A")


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
  filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
  path = Path("/usr/share/fonts/truetype/dejavu") / filename
  return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def _camera() -> mujoco.MjvCamera:
  camera = mujoco.MjvCamera()
  camera.type = mujoco.mjtCamera.mjCAMERA_FREE
  camera.lookat[:] = [0.55, -0.055, 0.50]
  camera.distance = 0.72
  camera.azimuth = 135.0
  camera.elevation = -27.0
  return camera


def _load_trace(output: Path, cell: str) -> dict[str, np.ndarray]:
  with np.load(output / f"trace_{cell}_seed_7.npz", allow_pickle=False) as archive:
    return {name: np.array(archive[name], copy=True) for name in archive.files}


def _contact_set(mask: np.ndarray) -> str:
  members = [str(index + 1) for index, active in enumerate(mask) if active]
  return "{" + ",".join(members) + "}" if members else "EMPTY"


def _overlay(
  frame: np.ndarray,
  trace: dict[str, np.ndarray],
  index: int,
  cell: str,
) -> np.ndarray:
  image = Image.fromarray(frame).convert("RGBA")
  draw = ImageDraw.Draw(image)
  color = COLORS[cell]
  title = {
    "i02_long": "I02-LONG · ONE 12 mm PREFIX",
    "i02_short": "I02-SHORT · 3 × 4 mm FRESH-ROOT PREFIXES",
    "i03_beam": "I03-BEAM · TERMINAL FILTER OFF",
    "i03_shadow": "I03-SHADOW · M12 FILTER ON",
  }[cell]
  draw.rounded_rectangle((18, 14, image.width - 18, 112), 14, fill=(7, 20, 35, 230))
  draw.text((34, 26), title, font=_font(22, bold=True), fill=color)
  draw.text(
    (34, 64),
    "FR3 + LEAP | fixed Bunny | Geometry Oracle + MCC | DP OFF",
    font=_font(16),
    fill=(207, 225, 241, 255),
  )
  draw.text(
    (image.width - 180, 30),
    f"t={float(trace['time_s'][index]):4.1f} s",
    font=_font(20, bold=True),
    fill="white",
  )
  panel_top = image.height - 178
  draw.rounded_rectangle(
    (18, panel_top, image.width - 18, image.height - 16),
    14,
    fill=(7, 20, 35, 235),
  )
  contacts = trace["mesh_valid_contacts"][index]
  forces = trace["fingertip_forces_n"][index]
  coordinate = 1000.0 * float(trace["actual_path_coordinate_m"][index])
  phase = str(trace["transaction_phase"][index])
  viability = str(trace["terminal_viability"][index])
  draw.text(
    (34, panel_top + 13),
    f"measured A = {_contact_set(contacts)}   path coordinate = {coordinate:5.1f} mm",
    font=_font(18, bold=True),
    fill="white",
  )
  draw.text(
    (image.width - 390, panel_top + 16),
    f"{phase} | {viability}",
    font=_font(14, bold=True),
    fill=color,
  )
  bar_y = panel_top + 57
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
    "Only M10-certified edge 0 reaches M06; every new prefix starts at a measured barrier"
    if cell != "i03_beam"
    else "Actual M12 diagnosis: terminal has no cheap safe continuation → DEAD_END hold"
  )
  draw.text((34, image.height - 40), footer, font=_font(14, bold=True), fill=color)
  return np.asarray(image.convert("RGB"))


def _render_comparison(
  output: Path,
  left_cell: str,
  right_cell: str,
  name: str,
) -> tuple[Path, Path]:
  cells = (left_cell, right_cell)
  traces = {cell: _load_trace(output, cell) for cell in cells}
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
    for cell in cells
  }
  data = {cell: mujoco.MjData(handles[cell].model) for cell in cells}
  renderers = {
    cell: mujoco.Renderer(handles[cell].model, width=960, height=540)
    for cell in cells
  }
  video = output / f"{name}.mp4"
  screenshot = output / f"{name}_keyframe.png"
  writer = imageio.get_writer(
    video,
    fps=10,
    codec="libx264",
    quality=8,
    macro_block_size=1,
  )
  camera = _camera()
  frame_times = np.arange(0.0, 20.0, 0.1)
  indices = np.unique(np.clip(np.round(frame_times / 0.002).astype(int), 0, 9999))
  keyframe: np.ndarray | None = None
  try:
    for index in indices:
      frames: list[np.ndarray] = []
      for cell in cells:
        h = handles[cell]
        d = data[cell]
        trace = traces[cell]
        d.qpos[h.arm_qpos_adrs] = trace["arm_q_rad"][index]
        d.qpos[h.hand_qpos_adrs] = trace["finger_q_rad"][index]
        mujoco.mj_forward(h.model, d)
        renderers[cell].update_scene(d, camera=camera)
        frames.append(_overlay(renderers[cell].render().copy(), trace, int(index), cell))
      side = np.concatenate(
        tuple(np.asarray(Image.fromarray(frame).resize((640, 360))) for frame in frames),
        axis=1,
      )
      writer.append_data(side)
      timestamp = float(traces[right_cell]["time_s"][index])
      target_time = 4.6 if name.startswith("i03") else 5.5
      if keyframe is None and timestamp >= target_time:
        keyframe = side.copy()
  finally:
    writer.close()
    for renderer in renderers.values():
      renderer.close()
  assert keyframe is not None
  imageio.imwrite(screenshot, keyframe)
  return video, screenshot


def _dashboard(output: Path, summary: dict[str, Any]) -> Path:
  plt = get_pyplot()
  plt.rcParams["savefig.facecolor"] = "#071423"
  figure = plt.figure(figsize=(16, 10), dpi=150, facecolor="#071423")
  grid = figure.add_gridspec(3, 2, height_ratios=(0.64, 1.0, 1.0), hspace=0.36, wspace=0.23)
  title = figure.add_subplot(grid[0, :])
  title.set_facecolor("#071423")
  title.axis("off")
  title.text(0.0, 0.82, "I02 / I03 · PHYSICAL PREFIX AND TERMINAL-VIABILITY EVALUATION", color="white", fontsize=22, weight="bold")
  title.text(0.0, 0.50, "Exact Bunny evidence · paired seeds 7/11/19 · MCC only · frozen acceptance", color="#afc8dc", fontsize=13)
  cards = [
    ("I02", summary["status"]["i02"].split(" / ")[-1], "#ffad42"),
    ("I02 ERROR", f"{1000*summary['i02_acceptance']['short_median_terminal_error_m']:.3f} mm", "#4ad6a0"),
    ("I03", summary["status"]["i03"].split(" / ")[-1], "#6db7ff"),
    ("G3", summary["status"]["gate_g3"], "#ed5a7a" if summary["status"]["gate_g3"] != "GO" else "#4ad6a0"),
  ]
  for index, (label, value, color) in enumerate(cards):
    x = 0.01 + 0.247 * index
    title.text(x, 0.16, label, color="#9fb9ce", fontsize=10, weight="bold")
    title.text(x, -0.04, value, color=color, fontsize=18, weight="bold")

  axes = [figure.add_subplot(grid[1, 0]), figure.add_subplot(grid[1, 1]), figure.add_subplot(grid[2, 0]), figure.add_subplot(grid[2, 1])]
  for axis in axes:
    axis.set_facecolor("#0d2033")
    axis.grid(color="#294055", alpha=0.65, linewidth=0.7)
    axis.tick_params(colors="#b9cedf")
    for spine in axis.spines.values():
      spine.set_color("#294055")

  cells = ("i02_long", "i02_short", "i03_beam", "i03_shadow")
  for x, cell in enumerate(cells):
    values = 1000 * np.asarray(summary["by_cell"][cell]["supported_cumulative_traversal_m"]["values"])
    axes[0].scatter(np.full(3, x) + [-0.06, 0.0, 0.06], values, s=55, color=COLORS[cell], edgecolor="white", linewidth=0.7)
    axes[0].plot([x - 0.18, x + 0.18], [np.median(values)] * 2, color="white", linewidth=2)
  axes[0].axhline(100, color="#4ad6a0", linestyle="--", linewidth=1.1)
  axes[0].set(title="Supported cumulative Bunny traversal", ylabel="distance [mm]", xticks=range(4), xticklabels=("LONG", "SHORT", "BEAM", "SHADOW"))

  long_errors = 1000 * np.asarray(summary["by_cell"]["i02_long"]["final_reposition_terminal_error_m"]["values"])
  short_errors = 1000 * np.asarray(summary["by_cell"]["i02_short"]["final_reposition_terminal_error_m"]["values"])
  axes[1].scatter(np.arange(3) - 0.08, long_errors, color=COLORS["i02_long"], s=60, label="LONG")
  axes[1].scatter(np.arange(3) + 0.08, short_errors, color=COLORS["i02_short"], s=60, label="SHORT")
  threshold = 1000 * (0.8 * np.median(long_errors / 1000) + 0.00025)
  axes[1].axhline(threshold, color="#ed5a7a", linestyle="--", label=f"frozen threshold {threshold:.3f} mm")
  axes[1].set(title="I02 final prefix prediction error", xlabel="paired seed", ylabel="error [mm]", xticks=range(3), xticklabels=(7, 11, 19))
  axes[1].legend(facecolor="#0d2033", labelcolor="white", fontsize=8)

  beam = _load_trace(output, "i03_beam")
  shadow = _load_trace(output, "i03_shadow")
  axes[2].plot(beam["time_s"], 1000 * beam["actual_path_coordinate_m"], color=COLORS["i03_beam"], label="Beam")
  axes[2].plot(shadow["time_s"], 1000 * shadow["actual_path_coordinate_m"], color=COLORS["i03_shadow"], label="Beam + ShadowSucc")
  axes[2].axvline(4.1, color="#afc8dc", linestyle=":", label="decision")
  axes[2].set(title="I03 physical path coordinate (seed 7)", xlabel="time [s]", ylabel="coordinate [mm]")
  axes[2].legend(facecolor="#0d2033", labelcolor="white", fontsize=8)

  for finger in range(4):
    axes[3].plot(shadow["time_s"], shadow["fingertip_forces_n"][:, finger], color=FINGER_COLORS[finger], linewidth=0.9, label=f"F{finger+1}")
  axes[3].axhline(8.0, color="#ed5a7a", linestyle="--", label="8 N hard limit")
  axes[3].set(title="I03-Shadow physical fingertip force", xlabel="time [s]", ylabel="force [N]", ylim=(-0.2, 8.6))
  axes[3].legend(facecolor="#0d2033", labelcolor="white", ncol=3, fontsize=8)
  for axis in axes:
    axis.title.set_color("white")
    axis.xaxis.label.set_color("#b9cedf")
    axis.yaxis.label.set_color("#b9cedf")
  path = output / "i02_i03_bunny_dashboard.png"
  save_figure(figure, path)
  plt.close(figure)
  return path


def _write_html(output: Path, summary: dict[str, Any], artifacts: dict[str, str]) -> Path:
  i02 = summary["i02_acceptance"]
  i03 = summary["i03_acceptance"]
  cells = summary["by_cell"]
  html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>I02/I03 Bunny Physics Review</title>
<style>body{{margin:0;background:#071423;color:#eaf4fb;font-family:Inter,Arial,sans-serif}}main{{max-width:1180px;margin:auto;padding:34px}}h1{{font-size:36px;margin:0 0 8px}}h2{{margin-top:34px}}p{{color:#b8cede;line-height:1.65}}.tag{{display:inline-block;background:#15344d;color:#59dda8;padding:7px 12px;border-radius:999px;font-weight:700}}.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:25px 0}}.card{{background:#0d2033;border:1px solid #294055;border-radius:15px;padding:18px}}.k{{color:#92aec3;font-size:12px;font-weight:700}}.v{{font-size:23px;font-weight:800;margin-top:8px}}img,video{{width:100%;border-radius:14px;border:1px solid #294055;background:#000}}table{{width:100%;border-collapse:collapse;background:#0d2033}}th,td{{padding:12px;border-bottom:1px solid #294055;text-align:left;vertical-align:top}}th{{color:#91b4cc}}code{{color:#8fd5ff}}.warn{{color:#ffbd69}}.ok{{color:#59dda8}}@media(max-width:850px){{.grid{{grid-template-columns:1fr 1fr}}}}</style></head><body><main>
<span class="tag">I02 {summary['status']['i02']} · I03 {summary['status']['i03']} · G3 {summary['status']['gate_g3']}</span><h1>I02 / I03：Bunny 物理评测</h1><p>统一使用固定 Bunny、FR3+LEAP、Geometry Oracle、显式 MCC 与 M10→M06 执行链；DP 关闭。所有 acceptance 都由冻结 evaluator 从保存 trace 计算，可视化不重算结论。</p>
<div class="grid"><div class="card"><div class="k">I02 LONG ERROR</div><div class="v">{1000*i02['long_median_terminal_error_m']:.3f} mm</div></div><div class="card"><div class="k">I02 SHORT ERROR</div><div class="v">{1000*i02['short_median_terminal_error_m']:.3f} mm</div></div><div class="card"><div class="k">I03 DEAD ENDS</div><div class="v">3 → 0</div></div><div class="card"><div class="k">I03 TRAVERSAL GAIN</div><div class="v">+{1000*i03['median_supported_traversal_advantage_m']:.2f} mm</div></div></div>
<img src="{artifacts['dashboard']}" alt="I02 I03 dashboard">
<h2>I02：短前缀重规划</h2><p><b>目的：</b>比较一次 12 mm committed prefix 与 3×4 mm、每段从真实 micro barrier 重新线性化的策略。<br><b>效果：</b>LONG 与 SHORT 都是 3/3 完成、没有安全或权限违规；SHORT 每条恰有 3 个 REPOSITION certificate 和 3 个 fresh barrier。<br><b>性能：</b>LONG/SHORT 支持距离中位数分别为 {1000*cells['i02_long']['supported_cumulative_traversal_m']['median']:.2f}/{1000*cells['i02_short']['supported_cumulative_traversal_m']['median']:.2f} mm。SHORT 误差只从 {1000*i02['long_median_terminal_error_m']:.3f} 降到 {1000*i02['short_median_terminal_error_m']:.3f} mm，没有达到冻结阈值，因此 <span class="warn">I02=NOT_MET</span>，不能声称短前缀显著提升稳健性。</p><video controls preload="metadata" src="{artifacts['i02_video']}"></video>
<h2>I03：终端可行性过滤</h2><p><b>目的：</b>只改变 M12 terminal predicate，验证它能否拒绝“当前边合法、但终点没有 continuation”的候选。<br><b>效果：</b>普通 Beam 3/3 选择 SLIDE(3)，真实 barrier 后都诊断 NONVIABLE 并安全保持；ShadowSucc 3/3 选择 SLIDE(1)，真实余量约 0.048 rad，均有 successor 并完成换指。<br><b>性能：</b>dead end 3→0；支持距离中位数 {1000*cells['i03_beam']['supported_cumulative_traversal_m']['median']:.2f}→{1000*cells['i03_shadow']['supported_cumulative_traversal_m']['median']:.2f} mm，优势 {1000*i03['median_supported_traversal_advantage_m']:.2f} mm；<span class="ok">I03=MET</span>。</p><video controls preload="metadata" src="{artifacts['i03_video']}"></video>
<h2>执行权限与边界</h2><table><thead><tr><th>模块</th><th>权限/证据</th><th>结论</th></tr></thead><tbody><tr><td>M09/M10/M06</td><td>每个执行前缀均从实测根优化、由 M10 扫掠审计签证、由 M06 执行至 barrier；suffix command=0</td><td>执行链有效</td></tr><tr><td>M11</td><td>只把 edge 0 暴露给 audit；预测 suffix 无执行权限</td><td>普通 Beam 会落入已冻结 dead end</td></tr><tr><td>M12</td><td>仅返回 viability/successors，execution_authority=false</td><td>过滤有效，但不能自己下发命令</td></tr><tr><td>G3</td><td>I02 与 I03 必须同时 MET</td><td class="warn">NO_GO（由 I02 阻断）</td></tr></tbody></table>
<h2>复现</h2><p><code>/home/ferry/data/Anaconda/envs/handcomp/bin/python -m Module.i02_i03_bunny_physics.benchmark</code><br><code>MUJOCO_GL=osmesa /home/ferry/data/Anaconda/envs/handcomp/bin/python -m Module.i02_i03_bunny_physics.visual_demo --reuse</code></p>
</main></body></html>"""
  path = output / "index.html"
  path.write_text(html, encoding="utf-8")
  return path


def run_visual(output: Path = DEFAULT_OUTPUT_DIR, *, reuse: bool = False) -> dict[str, Any]:
  output = output.resolve()
  summary_path = output / "summary.json"
  if not summary_path.is_file():
    raise FileNotFoundError("run Module.i02_i03_bunny_physics.benchmark first")
  summary = json.loads(summary_path.read_text(encoding="utf-8"))
  i02_video, i02_frame = _render_comparison(output, "i02_long", "i02_short", "i02_long_vs_short")
  i03_video, i03_frame = _render_comparison(output, "i03_beam", "i03_shadow", "i03_beam_vs_shadow")
  artifacts = {
    "i02_video": i02_video.name,
    "i02_keyframe": i02_frame.name,
    "i03_video": i03_video.name,
    "i03_keyframe": i03_frame.name,
    "dashboard": _dashboard(output, summary).name,
  }
  artifacts["index"] = _write_html(output, summary, artifacts).name
  result = {
    "module_ids": summary["module_ids"],
    "source_summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
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
  parser.add_argument("--reuse", action="store_true")
  arguments = parser.parse_args()
  print(json.dumps(run_visual(arguments.output, reuse=arguments.reuse), indent=2))


if __name__ == "__main__":
  main()
