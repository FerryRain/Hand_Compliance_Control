"""Build the descriptive Exp. 1 + Exp. 2 E05 review bundle.

The source experiment outputs retain their original provenance.  This review
normalizes only numeric metrics and reports reference-limit exceedances; it
does not assign pass/fail or MET/NOT_MET labels to any strategy.
"""

from __future__ import annotations

import argparse
import csv
from html import escape
import json
from pathlib import Path
import shutil
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXP1 = REPO_ROOT / "Module/generated/e05_h_mcc_vs_dp"
DEFAULT_EXP2 = REPO_ROOT / "Module/generated/exp2_dpref_mcc"
DEFAULT_OUTPUT = REPO_ROOT / "Module/generated/e05_exp1_exp2_review"
FORCE_REFERENCE_LIMIT_N = 8.0


def _load(path: Path) -> dict[str, Any]:
  return json.loads(path.read_text(encoding="utf-8"))


def _stat(metrics: dict[str, Any], name: str, statistic: str = "mean") -> float:
  value = metrics[name]
  if isinstance(value, dict):
    value = value[statistic]
  return float(value)


def _stat_optional(
  metrics: dict[str, Any],
  name: str,
  statistic: str = "mean",
) -> float | None:
  if name not in metrics:
    return None
  return _stat(metrics, name, statistic)


def _limit_observation(peak_force_n: float, violation_time_s: float, hard_guard_frames: float) -> dict[str, Any]:
  return {
    "reference_limit_n": FORCE_REFERENCE_LIMIT_N,
    "peak_force_n": peak_force_n,
    "excess_n": max(0.0, peak_force_n - FORCE_REFERENCE_LIMIT_N),
    "peak_to_limit_ratio": peak_force_n / FORCE_REFERENCE_LIMIT_N,
    "mean_force_above_limit_time_s": violation_time_s,
    "mean_hard_guard_frames": hard_guard_frames,
  }


def _exp1_strategy(name: str, source_name: str, metrics: dict[str, Any]) -> dict[str, Any]:
  peak = _stat(metrics, "max_tip_force_n", "max")
  return {
    "experiment": "Exp.1",
    "strategy": name,
    "source_cell": source_name,
    "contact_continuity": _stat(metrics, "contact_continuity_probability"),
    "average_contacts": _stat(metrics, "average_contact_count"),
    "contact_ge2": None,
    "contact_ge3": None,
    "four_contact": None,
    "force_rmse_n": _stat(metrics, "force_rmse_n"),
    "worst_peak_force_n": peak,
    "force_reference_time_s": _stat(metrics, "force_violation_time_s"),
    "force_reference_max_consecutive_s": None,
    "multi_pad_force_reference_time_s": None,
    "force_excess_impulse_n_s": None,
    "mean_y_traversal_m": _stat(metrics, "traversal_y_m"),
    "supported_y_ge2_m": None,
    "zero_contact_time_s": _stat(metrics, "zero_contact_time_s"),
    "controller_latency_p95_s": _stat(metrics, "controller_latency_p95_s"),
    "wrist_force_z_rmse_n": _stat(metrics, "wrist_force_z_rmse_n"),
    "limit_observation": _limit_observation(
      peak,
      _stat(metrics, "force_violation_time_s"),
      _stat(metrics, "hard_guard_frames"),
    ),
  }


def _exp2_strategy(name: str, source_name: str, metrics: dict[str, Any]) -> dict[str, Any]:
  peak = _stat(metrics, "max_tip_force_n", "max")
  return {
    "experiment": "Exp.2",
    "strategy": name,
    "source_cell": source_name,
    "contact_continuity": _stat(metrics, "contact_continuity_probability"),
    "average_contacts": _stat(metrics, "average_contact_count"),
    "contact_ge2": _stat(metrics, "contact_count_ge2_probability"),
    "contact_ge3": _stat(metrics, "contact_count_ge3_probability"),
    "four_contact": _stat(metrics, "four_contact_probability"),
    "force_rmse_n": _stat(metrics, "force_rmse_n"),
    "worst_peak_force_n": peak,
    "force_reference_time_s": _stat(metrics, "force_violation_time_s"),
    "force_reference_max_consecutive_s": (
      _stat_optional(metrics, "force_violation_max_consecutive_time_s") or 0.0
    ),
    "multi_pad_force_reference_time_s": (
      _stat_optional(metrics, "multi_pad_force_violation_time_s") or 0.0
    ),
    "force_excess_impulse_n_s": (
      _stat_optional(metrics, "force_excess_impulse_n_s") or 0.0
    ),
    "mean_y_traversal_m": _stat(metrics, "traversal_y_m"),
    "supported_y_ge2_m": _stat(metrics, "supported_y_traversal_ge2_m"),
    "zero_contact_time_s": _stat(metrics, "zero_contact_time_s"),
    "controller_latency_p95_s": _stat(metrics, "reference_inference_latency_p95_s"),
    "wrist_force_z_rmse_n": None,
    "limit_observation": _limit_observation(
      peak,
      _stat(metrics, "force_violation_time_s"),
      _stat(metrics, "hard_guard_frames"),
    ),
  }


def _copy_assets(exp1: Path, exp2: Path, output: Path) -> dict[str, str]:
  assets = {
    "exp1_dashboard.png": exp1 / "e05_h_mcc_vs_dp_dashboard.png",
    "exp1_mcc.mp4": exp1 / "e05_h_mcc_nominal.mp4",
    "exp1_dp_direct.mp4": exp1 / "e05_h_dp_nominal.mp4",
    "exp1_side_by_side.mp4": exp1 / "e05_h_mcc_vs_dp_side_by_side.mp4",
    "exp2_dashboard.png": exp2 / "exp2_comparison.png",
    "exp2_plain.mp4": exp2 / "plain_whole_hand_mcc_video.mp4",
    "exp2_passive.mp4": exp2 / "passive_hold_mcc_video.mp4",
    "exp2_reactive.mp4": exp2 / "reactive_heuristic_mcc_video.mp4",
    "exp2_dpref.mp4": exp2 / "dpref_mcc_video.mp4",
  }
  for target_name, source in assets.items():
    if not source.is_file():
      raise FileNotFoundError(source)
    shutil.copy2(source, output / target_name)
  return {key: key for key in assets}


def _format_force_observation(strategy: dict[str, Any]) -> str:
  observation = strategy["limit_observation"]
  if observation["excess_n"] > 0.0:
    text = (
      f"峰值 {observation['peak_force_n']:.2f} N，超过 8 N 诊断参考线 "
      f"{observation['excess_n']:.2f} N（{observation['peak_to_limit_ratio']:.2f}×）"
    )
  else:
    text = f"峰值 {observation['peak_force_n']:.2f} N，未观察到超过 8 N"
  if strategy.get("force_reference_max_consecutive_s") is not None:
    text += (
      f"；平均 >8 N 时间 {strategy['force_reference_time_s']:.4f} s，"
      f"最长连续 {strategy['force_reference_max_consecutive_s']:.4f} s，"
      f"多指同时 >8 N {strategy['multi_pad_force_reference_time_s']:.4f} s"
    )
  return text


def _strategy_rows(
  strategies: list[dict[str, Any]],
  include_supported: bool,
  include_multicontact: bool = False,
  include_force_occupancy: bool = False,
) -> str:
  rows: list[str] = []
  for item in strategies:
    supported = (
      f"{1000.0 * item['supported_y_ge2_m']:.2f}"
      if include_supported and item["supported_y_ge2_m"] is not None
      else "—"
    )
    multicontact = ""
    if include_multicontact:
      multicontact = (
        f"<td>{100.0 * item['contact_ge2']:.2f}%</td>"
        f"<td>{100.0 * item['contact_ge3']:.2f}%</td>"
        f"<td>{100.0 * item['four_contact']:.2f}%</td>"
      )
    force_occupancy = ""
    if include_force_occupancy:
      force_occupancy = (
        f"<td>{item['force_reference_time_s']:.4f}</td>"
        f"<td>{item['multi_pad_force_reference_time_s']:.4f}</td>"
      )
    rows.append(
      "<tr>"
      f"<td><strong>{escape(item['strategy'])}</strong></td>"
      f"<td>{100.0 * item['contact_continuity']:.2f}%</td>"
      f"<td>{item['average_contacts']:.3f}</td>"
      f"{multicontact}"
      f"<td>{item['force_rmse_n']:.3f}</td>"
      f"<td>{item['worst_peak_force_n']:.2f}</td>"
      f"{force_occupancy}"
      f"<td>{1000.0 * item['mean_y_traversal_m']:.2f}</td>"
      f"<td>{supported}</td>"
      f"<td>{1000.0 * item['controller_latency_p95_s']:.2f}</td>"
      "</tr>"
    )
  return "\n".join(rows)


def _analysis_cards(analysis: list[dict[str, str]]) -> str:
  cards: list[str] = []
  for item in analysis:
    cards.append(
      "<article class='card'>"
      f"<h4>{escape(item['strategy'])}</h4>"
      f"<p><b>优势：</b>{escape(item['strengths'])}</p>"
      f"<p><b>代价/限制：</b>{escape(item['limitations'])}</p>"
      "</article>"
    )
  return "\n".join(cards)


def _build_html(summary: dict[str, Any], assets: dict[str, str]) -> str:
  exp1 = summary["experiments"]["exp1"]["strategies"]
  exp2 = summary["experiments"]["exp2"]["strategies"]
  limit_cards = "\n".join(
    f"<li><b>{escape(item['strategy'])}</b>：{escape(_format_force_observation(item))}</li>"
    for item in exp1 + exp2
  )
  return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>E05 Exp.1 + Exp.2 策略性能</title>
  <style>
    :root{{--bg:#07111f;--panel:#0d1d30;--panel2:#122841;--text:#ecf4ff;--muted:#a8bdd2;
      --line:#2a435d;--cyan:#52d3d8;--orange:#ffb454;--red:#ff7b72;--green:#75d6a1}}
    *{{box-sizing:border-box}} body{{margin:0;background:linear-gradient(160deg,#06101d,#0a1a2b 55%,#101c2a);
      color:var(--text);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}}
    main{{max-width:1280px;margin:auto;padding:28px}} h1{{font-size:32px;margin:0 0 8px}} h2{{margin-top:34px}}
    h3{{margin:8px 0}} .muted{{color:var(--muted)}} .notice{{border-left:4px solid var(--orange);
      background:#192235;padding:14px 18px;border-radius:8px;margin:18px 0}}
    .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px}}
    .card,.panel{{background:rgba(13,29,48,.95);border:1px solid var(--line);border-radius:14px;padding:18px}}
    .card h4{{margin:0 0 8px;color:var(--cyan);font-size:18px}} .panel{{margin:16px 0}}
    table{{border-collapse:collapse;width:100%;min-width:900px}} th,td{{padding:9px 10px;border-bottom:1px solid var(--line);
      text-align:right}} th:first-child,td:first-child{{text-align:left}} .table-wrap{{overflow:auto}}
    img{{width:100%;border-radius:10px;border:1px solid var(--line)}} video{{width:100%;background:#000;border-radius:10px}}
    .videos{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px}}
    .limit-list li{{margin:6px 0}} a{{color:#7fdcff}} code{{color:#b9e8ff}}
    .tag{{display:inline-block;border:1px solid var(--line);background:var(--panel2);border-radius:999px;
      padding:3px 10px;margin-right:6px;color:var(--muted)}}
  </style>
</head>
<body><main>
  <h1>E05 Exp.1 + Exp.2 策略性能总览</h1>
  <p class="muted">固定 wrist trajectory 的 controller/reference-source evaluation。只报告数值、
  相对优劣和参考限制越界，不给策略设置 Pass/Fail 或 MET/NOT_MET。</p>
  <div class="notice"><b>可比性边界：</b>Exp.1 内部两策略可比。Exp.2 的 Plain MCC 是基础绝对参考；
  Passive、Reactive、DPRef 三者共享同一调整后 MCC/role/guard 栈，只有这三者可用于严格的
  reference-source 归因。Exp.1 与 Exp.2 不能直接做因果比较。</div>

  <section><h2>Exp.1：低层 Finger MCC vs. DP-direct</h2>
    <p><span class="tag">同一旧版 Wrist MCC</span><span class="tag">只替换 finger low-level controller</span></p>
    <div class="panel table-wrap"><table><thead><tr><th>策略</th><th>Contact continuity</th>
      <th>平均 contacts</th><th>Force RMSE (N)</th><th>Worst peak (N)</th><th>Y traversal (mm)</th>
      <th>Supported Y≥2 (mm)</th><th>Controller P95 (ms)</th></tr></thead><tbody>
      {_strategy_rows(exp1, False)}</tbody></table></div>
    <div class="grid">{_analysis_cards(summary['experiments']['exp1']['analysis'])}</div>
    <div class="panel"><img src="{assets['exp1_dashboard.png']}" alt="Exp.1 dashboard"></div>
    <div class="videos">
      <div class="panel"><h3>MCC</h3><video controls preload="metadata" src="{assets['exp1_mcc.mp4']}"></video></div>
      <div class="panel"><h3>DP-direct</h3><video controls preload="metadata" src="{assets['exp1_dp_direct.mp4']}"></video></div>
      <div class="panel"><h3>同步对比</h3><video controls preload="metadata" src="{assets['exp1_side_by_side.mp4']}"></video></div>
    </div>
  </section>

  <section><h2>Exp.2：普通 MCC 绝对参考 + 三种 shared-stack reference source</h2>
    <p><span class="tag">Plain whole-hand MCC</span><span class="tag">Passive-Hold</span>
    <span class="tag">Reactive-Heuristic</span><span class="tag">DPRef/Role</span></p>
    <div class="panel table-wrap"><table><thead><tr><th>策略</th><th>Contact continuity</th>
      <th>平均 contacts</th><th>P(Nc≥2)</th><th>P(Nc≥3)</th><th>P(Nc=4)</th><th>Force RMSE (N)</th><th>Worst peak (N)</th>
      <th>&gt;8 N time (s)</th><th>multi-pad &gt;8 N (s)</th><th>Y traversal (mm)</th>
      <th>Supported Y≥2 (mm)</th><th>Reference P95 (ms)</th></tr></thead><tbody>
      {_strategy_rows(exp2, True, True, True)}</tbody></table></div>
    <div class="grid">{_analysis_cards(summary['experiments']['exp2']['analysis'])}</div>
    <div class="panel"><img src="{assets['exp2_dashboard.png']}" alt="Exp.2 dashboard"></div>
    <div class="videos">
      <div class="panel"><h3>Plain whole-hand MCC</h3><video controls preload="metadata" src="{assets['exp2_plain.mp4']}"></video></div>
      <div class="panel"><h3>Passive-Hold + MCC</h3><video controls preload="metadata" src="{assets['exp2_passive.mp4']}"></video></div>
      <div class="panel"><h3>Reactive-Heuristic + MCC</h3><video controls preload="metadata" src="{assets['exp2_reactive.mp4']}"></video></div>
      <div class="panel"><h3>DPRef + MCC</h3><video controls preload="metadata" src="{assets['exp2_dpref.mp4']}"></video></div>
    </div>
  </section>

  <section><h2>仿真力诊断（8 N 统一参考线）</h2><div class="panel"><ul class="limit-list">{limit_cards}</ul>
    <p class="muted">MuJoCo 接触力不作为真实硬件力标定。这里重点观察持续高力、多指同时高力和
    明显 penetration；单个瞬时峰值不决定策略优劣。</p></div></section>

  <section><h2>整体解读</h2><div class="panel"><ul>
    {''.join(f"<li>{escape(item)}</li>" for item in summary['overall_analysis'])}
  </ul></div></section>

  <section><h2>Exp.3 的位置</h2><div class="panel"><p>Exp.3 不属于固定 wrist trajectory 的 E05。
  它已安排为 <code>I06 / Exp.3</code>，位于 I05 完整 GPIS 主实验之后，使用 active planner 在线
  选择 wrist trajectory，并隔离 explicit finger/contact-mode planning 与 wrist-only+DPRef 的差异。</p></div></section>

  <p class="muted">机器数据：<a href="summary.json">summary.json</a> ·
  <a href="metrics.csv">metrics.csv</a> · 生成说明：<a href="README.md">README.md</a></p>
</main></body></html>"""


def _difference_text(value: float, *, scale: float = 1.0, suffix: str = "") -> str:
  sign = "+" if value >= 0.0 else ""
  return f"{sign}{scale * value:.3f}{suffix}"


def _exp2_analysis(strategies: list[dict[str, Any]]) -> list[dict[str, str]]:
  by_source = {item["source_cell"]: item for item in strategies}
  plain = by_source["PLAIN_WHOLE_HAND_MCC"]
  passive = by_source["PASSIVE_HOLD_MCC"]
  reactive = by_source["REACTIVE_HEURISTIC_MCC"]
  dpref = by_source["DPREF_MCC"]
  analytical = (passive, reactive)
  best_continuity = max(item["contact_continuity"] for item in analytical)
  best_contacts = max(item["average_contacts"] for item in analytical)
  best_ge2 = max(item["contact_ge2"] for item in analytical)
  best_ge3 = max(item["contact_ge3"] for item in analytical)
  best_four = max(item["four_contact"] for item in analytical)
  best_supported = max(item["supported_y_ge2_m"] for item in analytical)
  return [
    {
      "strategy": plain["strategy"],
      "strengths": (
        f"基础解析控制的平均接触数为 {plain['average_contacts']:.3f}，"
        f"P(Nc≥3)={100.0 * plain['contact_ge3']:.2f}%，可作为绝对接触保持参考。"
      ),
      "limitations": (
        f"未使用新 role/force-safety wrapper；{_format_force_observation(plain)}。"
        "因此它是绝对参考，不能参与只替换 reference source 的因果归因。"
      ),
    },
    {
      "strategy": passive["strategy"],
      "strengths": (
        f"共享执行栈下无需预测模型；相对 Plain，平均 >8 N 占用时间 "
        f"{_difference_text(passive['force_reference_time_s'] - plain['force_reference_time_s'], suffix=' s')}，"
        "且没有两指同时 >8 N。"
      ),
      "limitations": (
        f"相对 Plain 平均接触数 {_difference_text(passive['average_contacts'] - plain['average_contacts'])}，"
        f"supported Y≥2 {_difference_text(passive['supported_y_ge2_m'] - plain['supported_y_ge2_m'], scale=1000.0, suffix=' mm')}；"
        "这量化了新执行栈的接触保持代价。"
      ),
    },
    {
      "strategy": reactive["strategy"],
      "strengths": (
        f"相对 Passive，平均接触数 {_difference_text(reactive['average_contacts'] - passive['average_contacts'])}、"
        f"P(Nc≥3) {_difference_text(reactive['contact_ge3'] - passive['contact_ge3'], scale=100.0, suffix=' pp')}、"
        f"Y traversal {_difference_text(reactive['mean_y_traversal_m'] - passive['mean_y_traversal_m'], scale=1000.0, suffix=' mm')}。"
      ),
      "limitations": (
        f"相对 Passive，supported Y≥2 "
        f"{_difference_text(reactive['supported_y_ge2_m'] - passive['supported_y_ge2_m'], scale=1000.0, suffix=' mm')}；"
        f"{_format_force_observation(reactive)}。"
      ),
    },
    {
      "strategy": dpref["strategy"],
      "strengths": (
        "相对每项最佳解析 reference source，"
        f"continuity {_difference_text(dpref['contact_continuity'] - best_continuity, scale=100.0, suffix=' pp')}、"
        f"平均接触数 {_difference_text(dpref['average_contacts'] - best_contacts)}、"
        f"P(Nc≥2) {_difference_text(dpref['contact_ge2'] - best_ge2, scale=100.0, suffix=' pp')}、"
        f"P(Nc≥3) {_difference_text(dpref['contact_ge3'] - best_ge3, scale=100.0, suffix=' pp')}、"
        f"supported Y≥2 {_difference_text(dpref['supported_y_ge2_m'] - best_supported, scale=1000.0, suffix=' mm')}；"
        "三条件均无 >8 N 记录。"
      ),
      "limitations": (
        f"四指同时接触率相对最佳解析源 "
        f"{_difference_text(dpref['four_contact'] - best_four, scale=100.0, suffix=' pp')}；"
        "载荷更偏向前三指，且 checkpoint 的 RELEASE/MAKE 覆盖仍限制 handover 结论。"
      ),
    },
  ]


def build_review(
  exp1_directory: str | Path = DEFAULT_EXP1,
  exp2_directory: str | Path = DEFAULT_EXP2,
  output_directory: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
  exp1_dir = Path(exp1_directory)
  exp2_dir = Path(exp2_directory)
  output = Path(output_directory)
  output.mkdir(parents=True, exist_ok=True)
  exp1_source = _load(exp1_dir / "summary.json")
  exp2_source = _load(exp2_dir / "summary.json")

  exp1 = [
    _exp1_strategy("Whole-hand MCC", "E05-H-MCC", exp1_source["cells"]["E05-H-MCC"]["numeric_metrics"]),
    _exp1_strategy("DP-direct", "E05-H-DP", exp1_source["cells"]["E05-H-DP"]["numeric_metrics"]),
  ]
  exp2 = [
    _exp2_strategy("Plain whole-hand MCC", "PLAIN_WHOLE_HAND_MCC", exp2_source["aggregates"]["PLAIN_WHOLE_HAND_MCC"]),
    _exp2_strategy("Passive-Hold + MCC", "PASSIVE_HOLD_MCC", exp2_source["aggregates"]["PASSIVE_HOLD_MCC"]),
    _exp2_strategy("Reactive-Heuristic + MCC", "REACTIVE_HEURISTIC_MCC", exp2_source["aggregates"]["REACTIVE_HEURISTIC_MCC"]),
    _exp2_strategy("DPRef/Role + MCC", "DPREF_MCC", exp2_source["aggregates"]["DPREF_MCC"]),
  ]

  summary = {
    "schema_version": "fr3-leap-e05-exp1-exp2-descriptive-review.v2",
    "evaluation_semantics": "DESCRIPTIVE_ONLY_NO_STRATEGY_PASS_FAIL",
    "cross_experiment_comparison_allowed": False,
    "cross_experiment_reason": "shared MCC/guard stack changed between Exp.1 and Exp.2",
    "reference_limits": {"fingertip_force_n": FORCE_REFERENCE_LIMIT_N},
    "experiments": {
      "exp1": {
        "question": "Can DP-direct replace low-level Finger MCC?",
        "comparison_scope": "within Exp.1 only",
        "strategies": exp1,
        "analysis": [
          {
            "strategy": "Whole-hand MCC",
            "strengths": "contact continuity、平均接触数、force RMSE、traversal 和延迟均优于 DP-direct。",
            "limitations": "worst peak 81.35 N，超过 8 N 参考限制 73.35 N；仍有明显 zero-contact 与 guard intervention。",
          },
          {
            "strategy": "DP-direct",
            "strengths": "12.00 ms P95 仍低于 20 ms policy 周期，wrist-force RMSE 低于 MCC。",
            "limitations": "continuity 下降 20.62 pp、平均少 1.436 个 contact、worst peak 103.02 N；authority projection/solver intervention 明显。",
          },
        ],
      },
      "exp2": {
        "question": "How do plain MCC and three shared-stack reference sources trade contact, motion and force?",
        "comparison_scope": (
          "Plain is an absolute reference; causal reference-source attribution is restricted "
          "to Passive/Reactive/DPRef"
        ),
        "strategies": exp2,
        "analysis": _exp2_analysis(exp2),
      },
    },
    "overall_analysis": [
      "Exp.1 保留为低层 DP-direct 与 Finger MCC 的历史诊断，不作为 Exp.2 新架构的安全基准。",
      (
        f"Exp.2 的 Plain MCC 仍给出最高平均接触数 {exp2[0]['average_contacts']:.3f} 和 "
        f"P(Nc≥3)={100.0 * exp2[0]['contact_ge3']:.2f}%，但平均 >8 N 占用 "
        f"{exp2[0]['force_reference_time_s']:.3f} s；它是绝对参考，不是同栈策略。"
      ),
      (
        f"在严格共享执行栈的三策略中，DPRef continuity={100.0 * exp2[3]['contact_continuity']:.2f}%、"
        f"平均接触数={exp2[3]['average_contacts']:.3f}、supported Y≥2="
        f"{1000.0 * exp2[3]['supported_y_ge2_m']:.1f} mm，三项均优于 Passive/Reactive。"
      ),
      "所有策略只报告性能；MuJoCo 力用于持续/多指高力诊断，单个瞬时峰值不设置策略级 Pass/Fail。",
      "DPRef 的验证标签缺少 RELEASE 且 MAKE 很少，因此当前结果不能支持 handover 泛化声明。",
    ],
    "source_artifacts": {
      "exp1_summary": str(exp1_dir / "summary.json"),
      "exp2_summary": str(exp2_dir / "summary.json"),
    },
  }

  assets = _copy_assets(exp1_dir, exp2_dir, output)
  (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
  with (output / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
    fieldnames = [
      "experiment", "strategy", "contact_continuity", "average_contacts", "force_rmse_n",
      "contact_ge2", "contact_ge3", "four_contact", "worst_peak_force_n", "force_limit_excess_n", "mean_y_traversal_m", "supported_y_ge2_m",
      "force_reference_time_s", "force_reference_max_consecutive_s",
      "multi_pad_force_reference_time_s", "force_excess_impulse_n_s",
      "zero_contact_time_s", "controller_latency_p95_s",
    ]
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for item in exp1 + exp2:
      writer.writerow({
        **{key: item[key] for key in fieldnames if key in item},
        "force_limit_excess_n": item["limit_observation"]["excess_n"],
      })
  (output / "index.html").write_text(_build_html(summary, assets), encoding="utf-8")
  readme = """# E05 Exp.1 + Exp.2 统一审阅

本目录把两组固定 wrist trajectory 实验放在同一页面。Exp.2 的 Plain 是绝对参考，
Passive/Reactive/DPRef 才构成严格共享执行栈的 reference-source comparison。
策略没有 Pass/Fail；MuJoCo force 只作为持续/多指高力与 penetration 诊断，8 N 是统一参考线。

- `index.html`：统一网页；
- `index_preview.png`：本次网页渲染检查截图（若已生成）；
- `summary.json`：去除方法级 verdict 后的统一机器数据；
- `metrics.csv`：六种策略的扁平指标；
- `exp1_*`：Exp.1 dashboard 和视频副本；
- `exp2_*`：Exp.2 dashboard 和视频副本。

复现：

```bash
/home/ferry/data/Anaconda/envs/handcomp/bin/python -m Module.e05_strategy_review
```

源数据仍保留在 `generated/e05_h_mcc_vs_dp/` 与 `generated/exp2_dpref_mcc/`，不因本汇总被
重写。Exp.1 和 Exp.2 之间 shared MCC/guard 版本不同，不允许跨实验做策略排名。
"""
  (output / "README.md").write_text(readme, encoding="utf-8")
  return summary


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--exp1", type=Path, default=DEFAULT_EXP1)
  parser.add_argument("--exp2", type=Path, default=DEFAULT_EXP2)
  parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
  args = parser.parse_args()
  summary = build_review(args.exp1, args.exp2, args.output)
  print(json.dumps({
    "output": str(args.output / "index.html"),
    "semantics": summary["evaluation_semantics"],
    "strategy_count": sum(len(value["strategies"]) for value in summary["experiments"].values()),
  }, indent=2))


if __name__ == "__main__":
  main()
