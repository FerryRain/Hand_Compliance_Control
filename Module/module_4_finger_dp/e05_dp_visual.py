"""Visual review bundle for the paired E05-H-MCC / E05-H-DP result."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

os.environ["MUJOCO_GL"] = "osmesa"
_CACHE = Path(tempfile.gettempdir()) / "handcomp-e05-dp-review"
_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["XDG_CACHE_HOME"] = str(_CACHE)
os.environ["MESA_SHADER_CACHE_DIR"] = str(_CACHE / "mesa_shader_cache")
os.environ["MPLCONFIGDIR"] = str(_CACHE / "matplotlib")
Path(os.environ["MESA_SHADER_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np

from Module.module_4_finger_dp.track_d_closed_loop import load_track_d_closed_loop
from Module.module_4_finger_dp.track_d_dataset import load_e05_h_teacher_trace
from Module.module_4_finger_dp.track_d_visual import render_track_d_video
from Module.module_4_whole_hand_mcc.visual_demo import render_video as render_mcc_video
from Module.visualization import get_pyplot, save_figure


FINGER_COLORS = ("#2997D6", "#3CBF91", "#F39C35", "#ED5A7A")


def _nominal_metrics(summary: dict, cell: str) -> dict:
  return next(
    row["metrics"]
    for row in summary["episodes"]
    if row["cell"] == cell and row["episode"] == "nominal"
  )


def render_paired_dashboard(output_directory: str | Path) -> Path:
  output = Path(output_directory)
  summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
  mcc = load_e05_h_teacher_trace(output / "nominal/e05_h_mcc_trace.npz")
  dp = load_track_d_closed_loop(output / "nominal/e05_h_dp_trace.npz")
  destination = output / "e05_h_mcc_vs_dp_dashboard.png"
  plt = get_pyplot()
  figure, axes = plt.subplots(4, 2, figsize=(16, 16), constrained_layout=True)
  for finger, color in enumerate(FINGER_COLORS):
    axes[0, 0].plot(mcc.time_s, mcc.fingertip_forces_n[:, finger], color=color, linewidth=.65, label=f"F{finger + 1}")
    axes[0, 1].plot(dp.time_s, dp.fingertip_forces_n[:, finger], color=color, linewidth=.65, label=f"F{finger + 1}")
  for axis, title in zip(axes[0], ("E05-H-MCC fingertip force", "E05-H-DP fingertip force")):
    axis.axhline(2.0, color="black", linestyle="--", linewidth=.9, label="2 N target")
    axis.axhline(8.0, color="#C73E4D", linestyle=":", linewidth=1.0, label="8 N hard")
    axis.axvline(1.0, color="#286B8E", linestyle=":", label="scored interval")
    axis.axvline(9.0, color="#8E5EA2", linestyle=":", label="+4 mm wrist step")
    axis.set(title=title, xlabel="time [s]", ylabel="N", ylim=(-.2, 16.0))
    axis.legend(ncol=4, fontsize=7)

  axes[1, 0].plot(mcc.time_s, np.sum(mcc.actual_contacts, axis=1), color="#286B8E")
  axes[1, 1].plot(dp.time_s, np.sum(dp.actual_contacts, axis=1), color="#D98032")
  for axis, title in zip(axes[1], ("MCC measured contact count", "DP measured contact count")):
    axis.axvline(1.0, color="#286B8E", linestyle=":")
    axis.axvline(9.0, color="#8E5EA2", linestyle=":")
    axis.set(title=title, xlabel="time [s]", ylabel="N_c", ylim=(-.2, 4.2))

  axes[2, 0].plot(mcc.time_s, mcc.desired_hand_wrench_world[:, 2], color="#2F855A", label="desired Fz")
  axes[2, 0].plot(mcc.time_s, mcc.estimated_hand_wrench_world[:, 2], color="#286B8E", linewidth=.8, label="estimated Fz")
  axes[2, 1].plot(dp.time_s, dp.desired_hand_wrench_world[:, 2], color="#2F855A", label="desired Fz")
  axes[2, 1].plot(dp.time_s, dp.estimated_hand_wrench_world[:, 2], color="#D98032", linewidth=.8, label="estimated Fz")
  for axis, title in zip(axes[2], ("Shared Wrist MCC in MCC cell", "Shared Wrist MCC in DP cell")):
    axis.axvline(9.0, color="#8E5EA2", linestyle=":")
    axis.set(title=title, xlabel="time [s]", ylabel="N")
    axis.legend(fontsize=8)

  mcc_nominal = _nominal_metrics(summary, "E05-H-MCC")
  dp_nominal = _nominal_metrics(summary, "E05-H-DP")
  axes[3, 0].axis("off")
  rows = [
    ["contact continuity", f"{100*mcc_nominal['contact_continuity_probability']:.1f}%", f"{100*dp_nominal['contact_continuity_probability']:.1f}%"],
    ["average contacts", f"{mcc_nominal['average_contact_count']:.2f}", f"{dp_nominal['average_contact_count']:.2f}"],
    ["force RMSE", f"{mcc_nominal['force_rmse_n']:.2f} N", f"{dp_nominal['force_rmse_n']:.2f} N"],
    ["peak force", f"{mcc_nominal['max_tip_force_n']:.1f} N", f"{dp_nominal['max_tip_force_n']:.1f} N"],
    ["Y traversal", f"{1000*mcc_nominal['traversal_y_m']:.1f} mm", f"{1000*dp_nominal['traversal_y_m']:.1f} mm"],
    ["hard-guard frames", str(mcc_nominal['hard_guard_frames']), str(dp_nominal['hard_guard_frames'])],
    ["peak excess above 8 N", f"{max(0.0, mcc_nominal['max_tip_force_n']-8.0):.1f} N", f"{max(0.0, dp_nominal['max_tip_force_n']-8.0):.1f} N"],
  ]
  table = axes[3, 0].table(
    cellText=rows,
    colLabels=["nominal metric", "H-MCC", "H-DP"],
    loc="center",
    cellLoc="center",
  )
  table.auto_set_font_size(False)
  table.set_fontsize(10)
  table.scale(1.0, 1.65)
  axes[3, 0].set_title("Nominal paired result · descriptive metrics")

  replans = dp.policy_replan
  axes[3, 1].scatter(
    dp.time_s[replans],
    1000.0 * dp.policy_latency_s[replans],
    s=6,
    color="#286B8E",
    label="CUDA diffusion",
  )
  axes[3, 1].plot(
    dp.time_s,
    1000.0 * dp.authority_latency_s,
    color="#3CBF91",
    linewidth=.6,
    label="authority QP",
  )
  axes[3, 1].axhline(20.0, color="#C73E4D", linestyle=":", label="50 Hz budget")
  axes[3, 1].set(title="Measured DP computation", xlabel="time [s]", ylabel="ms")
  axes[3, 1].legend(fontsize=8)
  for axis in axes.ravel()[:7]:
    axis.grid(alpha=.18)
  figure.suptitle(
    "Formal paired E05 · same FR3 Wrist MCC and M03 guard · only finger controller replaced",
    fontsize=16,
    fontweight="bold",
  )
  save_figure(figure, destination)
  plt.close(figure)
  return destination


def _side_by_side(mcc_path: Path, dp_path: Path, destination: Path) -> Path:
  command = [
    "ffmpeg", "-y", "-loglevel", "error",
    "-i", str(mcc_path), "-i", str(dp_path),
    "-filter_complex",
    "[0:v]pad=960:600:0:30:color=black[m];[m][1:v]hstack=inputs=2[v]",
    "-map", "[v]", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
    str(destination),
  ]
  subprocess.run(command, check=True)
  return destination


def write_review(output_directory: str | Path) -> tuple[Path, ...]:
  output = Path(output_directory)
  summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
  mcc = load_e05_h_teacher_trace(output / "nominal/e05_h_mcc_trace.npz")
  dp = load_track_d_closed_loop(output / "nominal/e05_h_dp_trace.npz")
  mcc_video = render_mcc_video(
    mcc,
    "E05-H-MCC",
    output / "e05_h_mcc_nominal.mp4",
    output / "e05_h_mcc_nominal_frame.png",
    fps=12,
  )
  dp_video = render_track_d_video(
    dp,
    output / "e05_h_dp_nominal.mp4",
    output / "e05_h_dp_nominal_frame.png",
    activation_s=1.0,
    fps=12,
    title="Formal E05-H-DP · Dataset-I D20 checkpoint",
  )
  paired = _side_by_side(mcc_video, dp_video, output / "e05_h_mcc_vs_dp_side_by_side.mp4")
  dashboard = render_paired_dashboard(output)
  mcc_cell = summary["cells"]["E05-H-MCC"]
  dp_cell = summary["cells"]["E05-H-DP"]
  mcc_excess = max(0.0, float(mcc_cell["numeric_metrics"]["max_tip_force_n"]["max"]) - 8.0)
  dp_excess = max(0.0, float(dp_cell["numeric_metrics"]["max_tip_force_n"]["max"]) - 8.0)
  review = output / "review.html"
  review.write_text(
    "<!doctype html><meta charset='utf-8'><title>E05 H-MCC vs H-DP</title>"
    "<style>body{max-width:1280px;margin:30px auto;font-family:system-ui;background:#f4f7fa;color:#17324d}"
    "section{background:white;padding:20px;margin:20px 0;border-radius:12px}video,img{width:100%;border:1px solid #ccd7e1;border-radius:8px}"
    "code{background:#edf2f6;padding:2px 6px;border-radius:4px}</style>"
    "<h1>E05-H-MCC vs E05-H-DP · formal paired physics</h1>"
    "<p><code>EVALUATED</code> means all three 15 s cells completed. Strategies receive no Pass/Fail or MET/NOT_MET verdict.</p>"
    f"<p>Worst peak excess above the 8 N reference: H-MCC <strong>{mcc_excess:.2f} N</strong>; "
    f"H-DP <strong>{dp_excess:.2f} N</strong>. "
    "The first second is the same unscored contact initializer. Wrist MCC and the M03 force-safety executor are shared.</p>"
    "<section><h2>1 · synchronized nominal comparison</h2><video controls src='e05_h_mcc_vs_dp_side_by_side.mp4'></video></section>"
    "<section><h2>2 · numerical dashboard</h2><img src='e05_h_mcc_vs_dp_dashboard.png'></section>"
    "<section><h2>3 · individual views</h2><video controls src='e05_h_mcc_nominal.mp4'></video>"
    "<video controls src='e05_h_dp_nominal.mp4'></video></section>"
    "<section><h2>4 · raw evidence</h2><ul><li><a href='summary.json'>summary.json</a></li>"
    "<li><a href='episodes.csv'>episodes.csv</a></li><li><a href='nominal/episode_summary.json'>nominal paired summary</a></li>"
    "<li><a href='low_friction/episode_summary.json'>low-friction paired summary</a></li>"
    "<li><a href='noisy_observation/episode_summary.json'>noisy-observation paired summary</a></li></ul></section>",
    encoding="utf-8",
  )
  return mcc_video, dp_video, paired, dashboard, review


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, default=Path("Module/generated/e05_h_mcc_vs_dp"))
  args = parser.parse_args()
  for path in write_review(args.output):
    print(path)


if __name__ == "__main__":
  main()
