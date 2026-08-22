"""Generate the M0–M4 and MCC-only E05 visual gallery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from Module.module_1_oracle_surface_model.visual_demo import (
  render_visual_demo as render_module_1,
)
from Module.module_2_fingertip_mcc.visual_demo import (
  render_visual_demo as render_module_2,
)
from Module.module_3_runtime_guards.visual_demo import (
  render_visual_demo as render_module_3,
)


def _gallery_html() -> str:
  return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>M0–M4 + E05 MCC Visual Demo</title>
  <style>
    :root { color-scheme: light; --ink:#17324d; --muted:#64748b; --line:#dbe4ec; }
    * { box-sizing: border-box; }
    body { margin:0; font-family:Inter,system-ui,-apple-system,"Segoe UI",sans-serif; color:var(--ink); background:#f5f8fb; }
    main { width:min(1180px,94vw); margin:42px auto 72px; }
    h1 { margin:0 0 8px; font-size:clamp(30px,4vw,48px); letter-spacing:-0.03em; }
    .lead { margin:0 0 28px; color:var(--muted); font-size:18px; line-height:1.7; }
    .flow { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:0 0 28px; }
    .flow div { padding:16px 18px; border-radius:14px; background:white; border:1px solid var(--line); font-weight:700; }
    .flow small { display:block; margin-top:5px; color:var(--muted); font-weight:400; line-height:1.5; }
    section { margin:18px 0; padding:22px; border-radius:18px; background:white; border:1px solid var(--line); box-shadow:0 12px 30px rgba(23,50,77,.06); }
    h2 { margin:0 0 6px; font-size:25px; }
    p { color:var(--muted); line-height:1.65; }
    img, video { width:100%; height:auto; display:block; margin-top:14px; border-radius:12px; border:1px solid var(--line); }
    .badge { display:inline-block; padding:4px 9px; border-radius:999px; color:#146c5b; background:#dff7ef; font-weight:700; font-size:13px; }
    .boundary { padding:16px 18px; border-left:4px solid #f8961e; background:#fff8eb; color:#6f4a12; line-height:1.65; }
    code { padding:2px 6px; border-radius:5px; background:#eef3f7; }
    @media (max-width:800px) { .flow { grid-template-columns:1fr; } section { padding:14px; } }
  </style>
</head>
<body>
<main>
  <span class="badge">handcomp · deterministic visual reproduction</span>
  <h1>M0–M4 与 MCC-only E05 到底做了什么？</h1>
  <p class="lead">解析模块、23-DoF 物理机器人与正式 E05 trace 放在同一页。视频中的 FR3 真正在移动，物体固定在 world；没有 DP 结果。</p>
  <div class="flow">
    <div>0 · FR3+Leap Contract<small>7 arm + 16 finger 的分组状态、wrench frame/sign、传感器有效性和日志。</small></div>
    <div>1 · Geometry Oracle<small>告诉系统表面在哪里、法向是什么、link 还有多少 clearance。</small></div>
    <div>2 · Fingertip MCC<small>沿 planner 的切向轨迹运动，只在法向上补偿接触力。</small></div>
    <div>3 · Runtime Guards<small>把真实可观测的过力、限位、碰撞与停滞证据转换成安全停止。</small></div>
  </div>

  <section>
    <h2>M0–M3 · FR3 + Leap Hand 适配</h2>
    <p>模型严格为 <code>nq=nv=nu=23</code>。右上 close-up 是四个真实 fingertip-body 指腹 pad；surface 固定，整只手由 FR3 执行 2D traversal。</p>
    <img src="fr3_leap_model_audit.png" alt="FR3 Leap model and fingertip belly audit">
  </section>

  <section>
    <h2>Module 1 · Oracle SurfaceModel</h2>
    <p>粉色点是可供 MAKE 使用的接触候选，橙色箭头是物体外法向；右侧分别展示 point projection 与 capsule clearance。</p>
    <img src="module_1_surface_model.png" alt="Module 1 geometry visualization">
  </section>

  <section>
    <h2>Module 2 · Fingertip MCC</h2>
    <p>上图展示静态力跟踪、平面滑动和曲面接触；动画展示 fingertip 在球面运动时，法向如何旋转而 2 N 接触力仍保持稳定。</p>
    <img src="module_2_fingertip_mcc.png" alt="Module 2 MCC plots">
    <img src="module_2_curved_surface.gif" alt="Module 2 curved-surface animation">
  </section>

  <section>
    <h2>Module 3 · Runtime Guards</h2>
    <p>时间线直观显示 commanded motion 与 actual progress 分离后，stall evidence 累积并在 0.15 s 触发；过力、关节限位和已知自碰撞则一帧触发。</p>
    <img src="module_3_runtime_guards.png" alt="Module 3 guard timelines">
  </section>

  <section>
    <h2>M04 / E05-F-MCC · 规定式 FR3 Wrist</h2>
    <p>Wrist MCC 关闭；FR3 跟踪同一 nominal palm trajectory，四个 Finger MCC 使用完整 local force error。</p>
    <video controls preload="metadata" src="fr3_leap_e05_f_mcc.mp4"></video>
  </section>

  <section>
    <h2>M04 / E05-H-MCC · 协调整手 MCC</h2>
    <p>FR3 joint-torque wrench estimate 驱动 resultant Wrist MCC；Finger MCC 只接 internal/differential error。视频来自正式 nominal trace。</p>
    <video controls preload="metadata" src="fr3_leap_e05_h_mcc.mp4"></video>
    <img src="fr3_leap_mcc_dashboard.png" alt="Formal MCC-only E05 dashboard">
  </section>

  <p class="boundary"><strong>当前边界：</strong>M0–M4/E05 视频是 MuJoCo 动力学证据，不是硬件结果；正式协议关闭 gravity 以隔离 contact control。E05 只评测 MCC，DP 未实现、未运行、未填写指标。实验完整执行记为 EVALUATED，性能由 MET/NOT_MET 单独表达。</p>
</main>
</body>
</html>
"""


def run_visual_demo(
  output_dir: Path = Path("Module/generated/visual_demo"),
  *,
  animation_frames: int = 60,
) -> dict[str, Any]:
  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  module_1 = render_module_1(output_dir / "module_1_surface_model.png")
  module_2 = render_module_2(
    output_dir / "module_2_fingertip_mcc.png",
    output_dir / "module_2_curved_surface.gif",
    animation_frames=animation_frames,
  )
  module_3 = render_module_3(output_dir / "module_3_runtime_guards.png")
  fr3_paths = (
    output_dir / "fr3_leap_model_audit.png",
    output_dir / "fr3_leap_e05_f_mcc.mp4",
    output_dir / "fr3_leap_e05_h_mcc.mp4",
    output_dir / "fr3_leap_mcc_dashboard.png",
  )
  fr3_ready = all(path.is_file() and path.stat().st_size > 0 for path in fr3_paths)
  gallery = output_dir / "index.html"
  gallery.write_text(_gallery_html(), encoding="utf-8")
  result = {
    "demo": "M0_TO_M4_E05_MCC_VISUAL_GALLERY",
    "passed": bool(module_1["passed"] and module_2["passed"] and module_3["passed"] and fr3_ready),
    "gallery": str(gallery),
    "modules": {"M01": module_1, "M02": module_2, "M03": module_3, "M0_M04_FR3": {"passed": fr3_ready, "artifacts": [str(path) for path in fr3_paths]}},
  }
  (output_dir / "summary.json").write_text(
    json.dumps(result, indent=2, sort_keys=True),
    encoding="utf-8",
  )
  return result


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path("Module/generated/visual_demo"),
  )
  parser.add_argument("--animation-frames", type=int, default=60)
  args = parser.parse_args()
  result = run_visual_demo(args.output_dir, animation_frames=args.animation_frames)
  print(json.dumps(result, indent=2, sort_keys=True))
  if not result["passed"]:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
