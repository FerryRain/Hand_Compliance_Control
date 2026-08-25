"""Render the M06--M12 MCC-baseline benchmark in the existing gallery style."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import numpy as np

from Module.m06_m12_benchmark import DEFAULT_OUTPUT, run_benchmark
from Module.module_1_oracle_surface_model import MeshScalePolicy, MeshSurface
from Module.module_1_oracle_surface_model.mesh_demo import DEFAULT_BUNNY
from Module.module_7_contact_mode_graph import ContactModeGraph, PrimitiveKind
from Module.visualization import COLORS, get_pyplot, save_figure


def _ms(seconds: float) -> str:
  return f"{1000.0 * float(seconds):.3f} ms"


def _us(seconds: float) -> str:
  return f"{1e6 * float(seconds):.2f} µs"


def _render_m06(output: Path, trace: Any, summary: dict[str, Any]) -> None:
  plt = get_pyplot()
  figure, axes = plt.subplots(3, 1, figsize=(13.5, 8.5), sharex=True)
  time_s = trace["m06_time_s"]
  contacts = trace["m06_contacts"].astype(bool)
  forces = trace["m06_forces_n"]
  correction_mm = 1000.0 * np.linalg.norm(
    trace["m06_commanded_positions_m"] - trace["m06_nominal_positions_m"],
    axis=2,
  )
  colors = (COLORS["blue"], COLORS["pink"], COLORS["green"], COLORS["orange"])
  axes[0].imshow(
    contacts.T.astype(float),
    aspect="auto",
    interpolation="nearest",
    cmap="Blues",
    vmin=0.0,
    vmax=1.0,
    extent=(float(time_s[0]), float(time_s[-1]), 4.5, 0.5),
  )
  for finger in range(4):
    axes[1].plot(time_s, forces[:, finger], color=colors[finger], linewidth=1.8)
    axes[2].plot(
      time_s,
      correction_mm[:, finger],
      color=colors[finger],
      linewidth=1.8,
      label=f"finger {finger + 1}",
    )
  axes[0].set_yticks([1, 2, 3, 4])
  axes[0].set_yticklabels(["F1", "F2", "F3", "F4"])
  axes[0].set_title("A  Measured A_actual never becomes empty", loc="left")
  axes[0].text(
    0.995,
    0.05,
    "dark = CONTACT · light = FREE",
    transform=axes[0].transAxes,
    ha="right",
    va="bottom",
    color=COLORS["gray"],
    fontsize=9,
  )
  axes[1].axhline(2.0, color=COLORS["gray"], linestyle="--", linewidth=1.2)
  axes[1].set_ylabel("normal force [N]")
  axes[1].set_title("B  Existing Fingertip MCC remains the anchor-force layer", loc="left")
  axes[2].set_ylabel("MCC correction [mm]")
  axes[2].set_xlabel("transaction time [s]")
  axes[2].set_title("C  Only the certificate-bound prefix reaches the command path", loc="left")
  axes[2].legend(ncol=4, loc="upper left", fontsize=8)
  metric = summary["metrics"]
  figure.suptitle(
    "M06 — Transactional Prefix Executor",
    x=0.05,
    ha="left",
    fontsize=17,
    weight="bold",
  )
  figure.text(
    0.05,
    0.015,
    f"5/5 semantic scenarios · authority violations = 0 · step P95 = {_ms(metric['step_latency_p95_s'])}",
    color=COLORS["gray"],
  )
  figure.tight_layout(rect=(0.0, 0.04, 1.0, 0.95))
  save_figure(figure, output)
  plt.close(figure)


def _render_m07(output: Path, summary: dict[str, Any]) -> None:
  plt = get_pyplot()
  figure, axis = plt.subplots(figsize=(12.5, 8.5))
  graph = ContactModeGraph()
  positions: dict[int, tuple[float, float]] = {}
  for cardinality in range(1, 5):
    modes = [mode for mode in graph.modes if len(mode.contacts) == cardinality]
    xs = np.linspace(-1.0, 1.0, len(modes) + 2)[1:-1]
    for x, mode in zip(xs, modes):
      positions[mode.mask] = (float(x), float(cardinality))
  for mode in graph.modes:
    for edge in graph.edges_from(mode):
      if edge.primitive.kind not in {PrimitiveKind.MAKE, PrimitiveKind.BREAK}:
        continue
      source = positions[edge.source.mask]
      target = positions[edge.target.mask]
      color = COLORS["green"] if edge.primitive.kind is PrimitiveKind.MAKE else COLORS["orange"]
      axis.annotate(
        "",
        xy=target,
        xytext=source,
        arrowprops={
          "arrowstyle": "->",
          "color": color,
          "alpha": 0.24,
          "linewidth": 0.9,
          "shrinkA": 15,
          "shrinkB": 15,
        },
      )
  cardinality_colors = {
    1: COLORS["pink"],
    2: COLORS["orange"],
    3: COLORS["cyan"],
    4: COLORS["blue"],
  }
  for mode in graph.modes:
    x, y = positions[mode.mask]
    axis.scatter(
      x,
      y,
      s=780,
      c=cardinality_colors[len(mode.contacts)],
      edgecolors="white",
      linewidths=2.0,
      zorder=5,
    )
    axis.text(
      x,
      y,
      "{" + ",".join(map(str, sorted(mode.contacts))) + "}",
      ha="center",
      va="center",
      color="white",
      weight="bold",
      fontsize=9,
      zorder=6,
    )
  axis.set_xlim(-1.18, 1.18)
  axis.set_ylim(0.55, 4.45)
  axis.set_yticks([1, 2, 3, 4])
  axis.set_yticklabels(["1 contact", "2 contacts", "3 contacts", "4 contacts"])
  axis.set_xticks([])
  axis.set_title("MAKE (green) and BREAK (orange) topology edges; local actions remain in-mode", loc="left")
  metric = summary["metrics"]
  figure.suptitle(
    "M07 — ContactModeGraph: all 15 nonempty four-finger modes",
    x=0.05,
    ha="left",
    fontsize=17,
    weight="bold",
  )
  figure.text(
    0.05,
    0.025,
    f"{metric['legal_edge_count']} legal primitive edges · deterministic enumeration · legality P95 = {_us(metric['legality_latency_p95_s'])}",
    color=COLORS["gray"],
  )
  figure.tight_layout(rect=(0.0, 0.05, 1.0, 0.94))
  save_figure(figure, output)
  plt.close(figure)


def _render_m08(output: Path, summary: dict[str, Any]) -> None:
  plt = get_pyplot()
  figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.4))
  confusion = summary["effect"]["confusion_matrix"]
  matrix = np.array(
    [[confusion["TP"], confusion["FN"]], [confusion["FP"], confusion["TN"]]],
    dtype=np.int64,
  )
  image = axes[0].imshow(matrix, cmap="Blues")
  for row in range(2):
    for column in range(2):
      axes[0].text(
        column,
        row,
        f"{matrix[row, column]:,}",
        ha="center",
        va="center",
        fontsize=17,
        weight="bold",
        color="white" if matrix[row, column] > matrix.max() * 0.5 else COLORS["navy"],
      )
  axes[0].set_xticks([0, 1], ["survive", "reject"])
  axes[0].set_yticks([0, 1], ["exact feasible", "exact infeasible"])
  axes[0].set_title("A  4,096-candidate confusion matrix", loc="left")
  figure.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)
  metric = summary["metrics"]
  values = [1e6 * metric["screen_latency_p50_s"], 1e6 * metric["screen_latency_p95_s"]]
  axes[1].bar(["P50", "P95"], values, color=[COLORS["cyan"], COLORS["blue"]])
  axes[1].axhline(500.0, color=COLORS["orange"], linestyle="--", label="0.50 ms gate")
  axes[1].set_yscale("log")
  axes[1].set_ylim(1.0, 1000.0)
  axes[1].set_ylabel("latency [µs / candidate]")
  axes[1].set_title("B  Cheap screening latency", loc="left")
  axes[1].legend()
  axes[1].text(
    0.03,
    0.91,
    f"False-negative rate = {100.0 * metric['false_negative_rate']:.2f}%\nFalse positives are allowed by design",
    transform=axes[1].transAxes,
    va="top",
    bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "edgecolor": "#dbe4ec"},
  )
  figure.suptitle(
    "M08 — CheapCert: low-FN pruning, never execution authority",
    x=0.04,
    ha="left",
    fontsize=17,
    weight="bold",
  )
  figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
  save_figure(figure, output)
  plt.close(figure)


def _render_bunny(axis) -> None:
  from mpl_toolkits.mplot3d.art3d import Poly3DCollection

  surface = MeshSurface.from_file(
    DEFAULT_BUNNY,
    source_up_axis="y",
    scale_policy=MeshScalePolicy(),
  )
  mesh = surface.mesh
  light = np.array([0.35, -0.45, 1.0])
  light /= np.linalg.norm(light)
  intensity = 0.32 + 0.68 * np.clip(mesh.face_normals @ light, 0.0, 1.0)
  base = np.array([0.25, 0.60, 0.83])
  colors = np.column_stack(
    [intensity[:, None] * base[None, :], np.full(len(intensity), 0.96)]
  )
  axis.add_collection3d(
    Poly3DCollection(mesh.triangles, facecolors=colors, edgecolor="none")
  )
  samples = surface.sample_surface(96, np.random.default_rng(7))
  selected = samples[np.argsort(samples[:, 2])[-4:]]
  axis.scatter(
    selected[:, 0],
    selected[:, 1],
    selected[:, 2],
    c=COLORS["pink"],
    s=65,
    edgecolors="white",
    depthshade=False,
  )
  bounds = surface.bounds
  center = np.mean(bounds, axis=0)
  radius = 0.57 * float(np.max(surface.extents))
  axis.set_xlim(center[0] - radius, center[0] + radius)
  axis.set_ylim(center[1] - radius, center[1] + radius)
  axis.set_zlim(0.0, 2.0 * radius)
  axis.set_box_aspect((1, 1, 1))
  axis.view_init(elev=18, azim=-58)
  axis.set_title("A  Bunny model is used as the visual object", loc="left")
  axis.set_xlabel("x [m]")
  axis.set_ylabel("y [m]")
  axis.set_zlabel("z [m]")


def _render_m09(output: Path, trace: Any, summary: dict[str, Any]) -> None:
  plt = get_pyplot()
  figure = plt.figure(figsize=(15.0, 7.3))
  bunny = figure.add_subplot(1, 2, 1, projection="3d")
  paths = figure.add_subplot(1, 2, 2, projection="3d")
  _render_bunny(bunny)
  colors = {
    "slide": COLORS["blue"],
    "reposition": COLORS["purple"],
    "make": COLORS["green"],
    "break": COLORS["orange"],
  }
  selected = {"slide": 0, "reposition": 2, "make": 2, "break": 0}
  for kind, finger in selected.items():
    key = f"m09_{kind}_tips_m"
    trajectory = trace[key][:, finger, :]
    paths.plot(
      1000.0 * trajectory[:, 0],
      1000.0 * trajectory[:, 1],
      1000.0 * trajectory[:, 2],
      color=colors[kind],
      linewidth=3.0,
      marker="o",
      markersize=3.5,
      label=kind.upper(),
    )
  wrist = trace["m09_wrist_adjust_wrist_m"]
  wrist_relative = wrist - wrist[0] + np.array([0.0, 0.0, 0.018])
  paths.plot(
    1000.0 * wrist_relative[:, 0],
    1000.0 * wrist_relative[:, 1],
    1000.0 * wrist_relative[:, 2],
    color=COLORS["pink"],
    linewidth=3.0,
    marker="s",
    markersize=3.5,
    label="WRIST_ADJUST",
  )
  xx, yy = np.meshgrid(np.linspace(-55, 55, 2), np.linspace(-38, 45, 2))
  paths.plot_surface(xx, yy, np.zeros_like(xx), color="#DDEFF7", alpha=0.5)
  paths.set_xlabel("x [mm]")
  paths.set_ylabel("y [mm]")
  paths.set_zlabel("z [mm]")
  paths.set_title("B  Numeric protocol uses analytic plane + linearized backend", loc="left")
  paths.view_init(elev=25, azim=-58)
  paths.legend(loc="upper right", fontsize=8)
  metric = summary["metrics"]
  figure.suptitle(
    "M09 — ContinuousOptimize: five smooth constrained primitive trajectories",
    x=0.035,
    ha="left",
    fontsize=17,
    weight="bold",
  )
  figure.text(
    0.04,
    0.02,
    (
      f"160 cases · success {100 * metric['optimizer_success_rate']:.1f}% · "
      f"target P95 {1000 * metric['terminal_target_error_p95_m']:.4f} mm · "
      f"solve P95 {_ms(metric['solve_latency_p95_s'])}"
    ),
    color=COLORS["gray"],
  )
  figure.tight_layout(rect=(0.0, 0.05, 1.0, 0.93))
  save_figure(figure, output)
  plt.close(figure)


def _render_m10(output: Path, trace: Any, summary: dict[str, Any]) -> None:
  plt = get_pyplot()
  figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.3))
  alpha = trace["m10_collision_alpha"]
  clearance_mm = 1000.0 * trace["m10_collision_clearance_m"]
  axes[0].plot(alpha, clearance_mm, color=COLORS["blue"], linewidth=2.5)
  axes[0].fill_between(
    alpha,
    clearance_mm,
    0.0,
    where=clearance_mm < 0.0,
    color=COLORS["red"],
    alpha=0.35,
  )
  axes[0].axhline(0.0, color=COLORS["navy"], linewidth=1.2)
  axes[0].scatter([0.0, 1.0], [clearance_mm[0], clearance_mm[-1]], c=COLORS["green"], s=60)
  axes[0].set_xlabel("swept-prefix phase")
  axes[0].set_ylabel("link clearance [mm]")
  axes[0].set_title("A  Safe endpoints, colliding midpoint → REJECT", loc="left")
  adversarial = summary["effect"]["adversarial_rejections"]
  labels = [name.replace("_", "\n") for name in adversarial]
  values = [int(value) for value in adversarial.values()]
  axes[1].barh(labels, values, color=COLORS["green"])
  axes[1].set_xlim(0.0, 1.1)
  axes[1].set_xticks([0, 1], ["miss", "rejected"])
  axes[1].set_title("B  Six authority/safety adversaries", loc="left")
  metric = summary["metrics"]
  figure.suptitle(
    "M10 — ExactPrefixAudit: the sole ExecutionCertificate issuer",
    x=0.04,
    ha="left",
    fontsize=17,
    weight="bold",
  )
  figure.text(
    0.04,
    0.015,
    f"6/6 adversaries rejected · {metric['positive_swept_samples']} swept samples · audit P95 = {_ms(metric['audit_latency_p95_s'])}",
    color=COLORS["gray"],
  )
  figure.tight_layout(rect=(0.0, 0.04, 1.0, 0.91))
  save_figure(figure, output)
  plt.close(figure)


def _render_m11(output: Path, summary: dict[str, Any]) -> None:
  plt = get_pyplot()
  figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))
  comparison = summary["effect"]["comparison"]
  horizons = ["H2", "H3"]
  x = np.arange(2)
  width = 0.36
  beam_nodes = [comparison[h]["beam_optimized_edges"] for h in horizons]
  exhaustive_nodes = [comparison[h]["exhaustive_optimized_edges"] for h in horizons]
  axes[0].bar(x - width / 2, beam_nodes, width, label="beam", color=COLORS["blue"])
  axes[0].bar(x + width / 2, exhaustive_nodes, width, label="exhaustive", color=COLORS["orange"])
  axes[0].set_xticks(x, horizons)
  axes[0].set_ylabel("optimized edges")
  axes[0].set_title("A  Lazy optimization work", loc="left")
  axes[0].legend()
  latency_ms = [1000.0 * comparison[h]["beam_latency_p95_s"] for h in horizons]
  axes[1].bar(horizons, latency_ms, color=[COLORS["cyan"], COLORS["purple"]])
  axes[1].set_ylabel("beam search P95 [ms]")
  axes[1].set_title("B  Search latency and retained optimum", loc="left")
  for index, horizon in enumerate(horizons):
    sequence = " → ".join(comparison[horizon]["best_sequence"])
    axes[1].text(
      index,
      latency_ms[index] + max(latency_ms) * 0.04,
      f"gap={comparison[horizon]['score_gap']:.1e}\n{sequence}",
      ha="center",
      va="bottom",
      fontsize=8,
    )
  axes[1].set_ylim(0.0, max(latency_ms) * 1.42)
  figure.suptitle(
    "M11 — Lazy Beam Search: diverse sequence search, only edge zero can be audited",
    x=0.04,
    ha="left",
    fontsize=17,
    weight="bold",
  )
  figure.text(
    0.04,
    0.015,
    "H=2/3 optimal-sequence retention = 100% · score gap = 0 · all later edges remain prediction suffixes",
    color=COLORS["gray"],
  )
  figure.tight_layout(rect=(0.0, 0.04, 1.0, 0.91))
  save_figure(figure, output)
  plt.close(figure)


def _render_m12(output: Path, summary: dict[str, Any]) -> None:
  plt = get_pyplot()
  figure, axes = plt.subplots(1, 2, figsize=(13.2, 5.2))
  for axis, viable in zip(axes, (True, False)):
    axis.scatter([0.0], [0.0], s=950, c=COLORS["pink"], edgecolors="white", linewidths=2)
    axis.text(0.0, 0.0, "{1}", ha="center", va="center", color="white", weight="bold")
    for index, finger in enumerate((2, 3, 4)):
      angle = np.deg2rad(25 + 65 * index)
      target = np.array([np.cos(angle), np.sin(angle)])
      color = COLORS["green"] if viable else COLORS["gray"]
      axis.annotate(
        "",
        xy=target,
        xytext=(0.12 * target[0], 0.12 * target[1]),
        arrowprops={"arrowstyle": "->", "color": color, "linewidth": 2.2 if viable else 1.2},
      )
      axis.scatter([target[0]], [target[1]], s=330, c=color, alpha=1.0 if viable else 0.35)
      axis.text(target[0], target[1], f"F{finger}", ha="center", va="center", fontsize=8, color="white", weight="bold")
      axis.text(target[0], target[1] + 0.13, f"MAKE({finger})", ha="center", va="bottom", fontsize=8, color=color)
      if not viable:
        axis.text(target[0], target[1] - 0.16, "joint/collision/reach reject", ha="center", fontsize=7, color=COLORS["red"])
    axis.set_aspect("equal")
    axis.set_xlim(-1.25, 1.25)
    axis.set_ylim(-0.35, 1.35)
    axis.axis("off")
    axis.set_title(
      "A  VIABLE: inactive MAKE survives"
      if viable
      else "B  NONVIABLE: no inactive MAKE survives",
      loc="left",
    )
  metric = summary["metrics"]
  figure.suptitle(
    "M12 — Shadow Terminal Viability: detect a dead end before committing",
    x=0.04,
    ha="left",
    fontsize=17,
    weight="bold",
  )
  figure.text(
    0.04,
    0.02,
    f"1,024 states · distinct viable successor fingers = 3 · viability P95 = {_ms(metric['viability_latency_p95_s'])} · no certificate authority",
    color=COLORS["gray"],
  )
  figure.tight_layout(rect=(0.0, 0.05, 1.0, 0.90))
  save_figure(figure, output)
  plt.close(figure)


def _module_rows(summary: dict[str, Any]) -> str:
  modules = summary["modules"]
  concise = {
    "M06": lambda m: f"5/5 场景，0 authority violation；step P95 {_ms(m['metrics']['step_latency_p95_s'])}",
    "M07": lambda m: f"15 modes / {m['metrics']['legal_edge_count']} legal edges；legality P95 {_us(m['metrics']['legality_latency_p95_s'])}",
    "M08": lambda m: f"4,096 candidates；FN {100*m['metrics']['false_negative_rate']:.2f}%；screen P95 {_us(m['metrics']['screen_latency_p95_s'])}",
    "M09": lambda m: f"160 cases；success {100*m['metrics']['optimizer_success_rate']:.1f}%；solve P95 {_ms(m['metrics']['solve_latency_p95_s'])}",
    "M10": lambda m: f"6/6 adversaries；{m['metrics']['positive_swept_samples']} swept samples；audit P95 {_ms(m['metrics']['audit_latency_p95_s'])}",
    "M11": lambda m: f"H2/H3 optimum retention 100%；beam width {m['metrics']['beam_width']}；suffix prediction-only",
    "M12": lambda m: f"1,024 states；3 distinct MAKE fingers；P95 {_ms(m['metrics']['viability_latency_p95_s'])}",
  }
  purpose_cn = {
    "M06": "把证书绑定的短前缀原子执行到 micro-barrier，并返回真实 snapshot。",
    "M07": "定义 15 个非空接触集合及 MAKE/BREAK/SLIDE 等合法转换。",
    "M08": "在昂贵优化前低漏检地筛掉明显不可行 edge。",
    "M09": "为每种 primitive 构造平滑、受 trust/anchor/joint/collision 约束的轨迹。",
    "M10": "检查完整 swept prefix；它是唯一可以签发执行证书的模块。",
    "M11": "在 H=2/3 上保留有进展且 mode 多样的预测 sequence。",
    "M12": "在 commit 前检查 terminal state 是否至少还有一个便宜安全后继。",
  }
  rows = []
  for module_id in modules:
    module = modules[module_id]
    rows.append(
      "<tr>"
      f"<td><strong>{module_id}</strong></td>"
      f"<td>{html.escape(purpose_cn[module_id])}</td>"
      f"<td>{html.escape(concise[module_id](module))}</td>"
      f"<td><span class='ok'>{module['performance_verdict']}</span></td>"
      "</tr>"
    )
  return "\n".join(rows)


def _gallery_html(summary: dict[str, Any]) -> str:
  integration = summary["integration_smoke"]
  machine = summary["provenance"]["machine"]
  return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>M06–M12 Oracle + MCC Baseline Visual Demo</title>
  <style>
    :root {{ color-scheme:light; --ink:#17324d; --muted:#64748b; --line:#dbe4ec; --ok:#146c5b; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:Inter,system-ui,-apple-system,"Segoe UI",sans-serif; color:var(--ink); background:#f5f8fb; }}
    main {{ width:min(1200px,94vw); margin:42px auto 72px; }}
    h1 {{ margin:0 0 8px; font-size:clamp(30px,4vw,48px); letter-spacing:-.03em; }}
    .lead {{ margin:0 0 24px; color:var(--muted); font-size:18px; line-height:1.65; }}
    .flow {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:0 0 24px; }}
    .flow div {{ padding:16px 18px; border-radius:14px; background:white; border:1px solid var(--line); font-weight:700; }}
    .flow small {{ display:block; margin-top:5px; color:var(--muted); font-weight:400; line-height:1.45; }}
    section {{ margin:18px 0; padding:22px; border-radius:18px; background:white; border:1px solid var(--line); box-shadow:0 12px 30px rgba(23,50,77,.06); }}
    h2 {{ margin:0 0 7px; font-size:25px; }}
    p {{ color:var(--muted); line-height:1.65; }}
    img {{ width:100%; height:auto; display:block; margin-top:14px; border-radius:12px; border:1px solid var(--line); }}
    table {{ width:100%; border-collapse:collapse; font-size:14px; }}
    th,td {{ padding:12px 10px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; line-height:1.5; }}
    th {{ color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.05em; }}
    .badge,.ok {{ display:inline-block; padding:4px 9px; border-radius:999px; color:var(--ok); background:#dff7ef; font-weight:700; font-size:13px; }}
    .boundary {{ padding:16px 18px; border-left:4px solid #f8961e; background:#fff8eb; color:#6f4a12; line-height:1.65; }}
    code {{ padding:2px 6px; border-radius:5px; background:#eef3f7; }}
    @media(max-width:820px) {{ .flow {{ grid-template-columns:1fr; }} section {{ padding:14px; overflow:auto; }} }}
  </style>
</head>
<body><main>
  <span class="badge">handcomp · seed 7 · benchmark-backed visualization</span>
  <h1>M06–M12：MCC baseline 如何从预测走到安全短前缀执行</h1>
  <p class="lead">所有图片直接读取同一次 benchmark 的 <code>summary.json</code> 与 <code>traces.npz</code>。数值验收使用 analytic plane；Stanford Bunny 只作为复杂物体展示，不替代几何验收。</p>
  <div class="flow">
    <div>Predict<small>M07 graph → M08 cheap screen → M09 continuous construction</small></div>
    <div>Search<small>M11 diverse beam → M12 terminal continuation check</small></div>
    <div>Certify<small>M10 audits every swept sample and binds one certificate</small></div>
    <div>Execute<small>M06 runs only Pi_commit → barrier → measured snapshot</small></div>
  </div>
  <section>
    <h2>目的、效果与实测性能总览</h2>
    <table><thead><tr><th>模块</th><th>目的</th><th>效果 / 性能</th><th>模块协议</th></tr></thead><tbody>
      {_module_rows(summary)}
    </tbody></table>
    <p>本机 timing：{html.escape(str(machine['cpu_model']))}，{machine['logical_cpu_count']} logical CPUs；各模块计时边界记录在 <code>summary.json</code>，不包含未声明的 physics/I/O。</p>
  </section>
  <section><h2>M06 · Transactional Prefix Executor</h2><p>先完成的 participant 进入 hold；所有 participant terminal 后还需一个新的真实 observation 才关闭 barrier。Anchor contacts 继续走现有 Fingertip MCC。</p><img src="module_6_prefix_executor.png" alt="M06 executor trace"></section>
  <section><h2>M07 · ContactModeGraph</h2><p>四指共 15 个非空 mode。图只画改变 topology 的 MAKE/BREAK；SLIDE、REPOSITION 与 WRIST_ADJUST 仍由同一 deterministic legality API 管理。</p><img src="module_7_contact_mode_graph.png" alt="M07 graph"></section>
  <section><h2>M08 · CheapCert</h2><p>CheapCert 的设计目标是低 false negative：允许一部分 false positive 留给 optimizer/audit，自己永不签证书。</p><img src="module_8_cheap_cert.png" alt="M08 confusion and timing"></section>
  <section><h2>M09 · ContinuousOptimize</h2><p>Bunny 说明复杂物体展示接口；右侧是正式模块协议使用的 plane/linearized-backend 五类轨迹。后者不能冒充 FR3 nonlinear IK 或 MuJoCo exact collision。</p><img src="module_9_continuous_optimize.png" alt="M09 bunny and primitive paths"></section>
  <section><h2>M10 · ExactPrefixAudit</h2><p>Endpoint 都安全仍不足够：中途 collision/joint/trust/anchor/model-version/commit-legality 任一不满足都拒绝。只有正例拿到 <code>ExecutionCertificate</code>。</p><img src="module_10_exact_prefix_audit.png" alt="M10 swept audit"></section>
  <section><h2>M11 · Lazy Beam Search</h2><p>固定 beam width 8、per-mode quota 2；H=2/3 都保留 exhaustive optimum。只有 edge zero 保持 audit-candidate 身份，其余 prefix 强制标为 prediction suffix。</p><img src="module_11_lazy_beam_search.png" alt="M11 search comparison"></section>
  <section><h2>M12 · Shadow Terminal Viability</h2><p>Singleton mode 必须至少找到一个 inactive finger 的 cheap-feasible MAKE。ShadowSucc 只回答“后面还有没有路”，没有 execution authority。</p><img src="module_12_shadow_viability.png" alt="M12 viability"></section>
  <p class="boundary"><strong>结果边界：</strong>集成 smoke 完成={str(integration['completed']).lower()}，contact continuity={100*integration['contact_continuity']:.1f}%，执行 {html.escape(integration['committed_primitive'])}，但它仍是 module-validation fixture。G1 保持 No-Go；未使用 DP，未运行正式 I01/G2/G3，也不声称 gravity-on、完整 MuJoCo traversal 或硬件性能。</p>
</main></body></html>"""


def _generated_readme(summary: dict[str, Any]) -> str:
  return f"""# M06–M12 Oracle + MCC baseline module validation

Execution status: `{summary['execution_status']}`
Module-protocol performance: `{summary['performance_verdict']}`

Open `index.html` for the module-by-module purpose, effect, and measured
performance. Numeric acceptance uses the analytic plane and the deterministic
linearized validation backend. Stanford Bunny is visualization only.

Reproduce from the repository root:

```bash
PY=/home/ferry/data/Anaconda/envs/handcomp/bin/python
$PY -m Module.m06_m12_benchmark
$PY -m Module.m06_m12_visual_demo --reuse-benchmark
$PY -m unittest Module.tests.test_m06_m12_planner -v
```

This result does not change G1, does not use Finger DP, and is not a formal
I01/G2/G3 result.
"""


def run_visual_demo(
  output_dir: str | Path = DEFAULT_OUTPUT,
  *,
  refresh_benchmark: bool = True,
) -> dict[str, Any]:
  output = Path(output_dir)
  output.mkdir(parents=True, exist_ok=True)
  summary_path = output / "summary.json"
  trace_path = output / "traces.npz"
  if refresh_benchmark or not summary_path.is_file() or not trace_path.is_file():
    summary = run_benchmark(output)
  else:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
  with np.load(trace_path, allow_pickle=False) as trace:
    artifacts = {
      "M06": output / "module_6_prefix_executor.png",
      "M07": output / "module_7_contact_mode_graph.png",
      "M08": output / "module_8_cheap_cert.png",
      "M09": output / "module_9_continuous_optimize.png",
      "M10": output / "module_10_exact_prefix_audit.png",
      "M11": output / "module_11_lazy_beam_search.png",
      "M12": output / "module_12_shadow_viability.png",
    }
    _render_m06(artifacts["M06"], trace, summary["modules"]["M06"])
    _render_m07(artifacts["M07"], summary["modules"]["M07"])
    _render_m08(artifacts["M08"], summary["modules"]["M08"])
    _render_m09(artifacts["M09"], trace, summary["modules"]["M09"])
    _render_m10(artifacts["M10"], trace, summary["modules"]["M10"])
    _render_m11(artifacts["M11"], summary["modules"]["M11"])
    _render_m12(artifacts["M12"], summary["modules"]["M12"])
  gallery = output / "index.html"
  gallery.write_text(_gallery_html(summary), encoding="utf-8")
  (output / "README.md").write_text(_generated_readme(summary), encoding="utf-8")
  result = {
    "demo": "M06_M12_ORACLE_MCC_BASELINE_VISUAL_GALLERY_V1",
    "passed": bool(
      summary["performance_verdict"] == "MET"
      and all(path.is_file() and path.stat().st_size > 10_000 for path in artifacts.values())
      and gallery.is_file()
    ),
    "gallery": str(gallery),
    "summary": str(summary_path),
    "artifacts": {module: str(path) for module, path in artifacts.items()},
    "g1_changed": False,
    "dp_used": False,
  }
  (output / "visual_summary.json").write_text(
    json.dumps(result, indent=2, sort_keys=True),
    encoding="utf-8",
  )
  return result


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
  parser.add_argument(
    "--reuse-benchmark",
    action="store_true",
    help="reuse summary.json/traces.npz when they already exist",
  )
  args = parser.parse_args()
  result = run_visual_demo(
    args.output,
    refresh_benchmark=not args.reuse_benchmark,
  )
  print(json.dumps(result, indent=2, sort_keys=True))
  if not result["passed"]:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
