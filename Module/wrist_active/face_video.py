"""Add the held-out can trajectory and refreshed end card to the paired experiment film."""

import argparse
import subprocess
from contextlib import ExitStack
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw

from . import sweep_video as film
from .__main__ import RESULTS
from .experiment import digest, save_json
from .sweep_experiment import OUT


def chapters_of(movie, ffmpeg):
  metadata = subprocess.check_output(
    [ffmpeg, "-v", "error", "-i", str(movie), "-f", "ffmetadata", "-"]
  )
  chapters = []
  for part in metadata.decode("utf-8").split("[CHAPTER]")[1:]:
    item = dict(line.split("=", 1) for line in part.strip().splitlines() if "=" in line)
    chapters.append(
      (round(int(item["START"]) * Fraction(item["TIMEBASE"]) * film.FPS), item["title"])
    )
  if not chapters or chapters[-1][1] != "结果汇总":
    raise ValueError(
      "Expected the published pear and ellipsoid film with its original end card"
    )
  return chapters


def end_card(results):
  im = Image.new("RGB", film.SIZE, film.BG)
  pen = ImageDraw.Draw(im)
  film.label(pen, (48, 56), "修正腕轨后的迁移结果", 36, bold=True)
  film.label(pen, (49, 104), "MuJoCo 闭环仿真  ·  每个起点 240 秒", 20, film.MUTED)
  pen.line((48, 139, 1232, 139), fill=film.HAIR)
  film.label(pen, (49, 176), "实测指标", 22, film.MUTED)
  for x, title, color in zip(
    (855, 1206), ("手腕 Active＋single_all", "完整 Aude＋MCC"), film.ACCENTS, strict=True
  ):
    film.label(pen, (x, 176), title, 22, color, bold=True, anchor="rm")
  can, transfer = results["can"], results["transfer"]
  for i, (name, group, metric, spec) in enumerate(
    (
      ("罐体两起点 · 实触覆盖", can, "coverage", ".1%"),
      ("罐体两起点 · 轻载三指累计（秒）", can, "light_three_s", ".2f"),
      ("罐体两起点 · 分区载荷 ≥8 N（秒）", can, "over8_s", ".2f"),
      ("七个迁移起点 · 平均实触覆盖", transfer, "coverage", ".2%"),
    )
  ):
    y = 244 + 74 * i
    film.label(pen, (50, y), name, 23)
    for x, key, color in zip((855, 1206), ("updated", "aude"), film.ACCENTS, strict=True):
      film.label(
        pen,
        (x, y),
        format(group[key][metric], spec),
        31,
        color,
        mono=True,
        bold=True,
        anchor="rm",
      )
    pen.line((49, y + 38, 1232, y + 38), fill=film.HAIR)
  film.label(pen, (50, 557), "起点 01 用于开发。视频展示冻结参数后运行的起点 02。", 20)
  film.label(
    pen,
    (50, 612),
    "修正腕轨在两个罐体起点的实际平移更长。匹配总平移和转角后，双方各领先一个起点。",
    19,
    film.MUTED,
  )
  film.label(
    pen,
    (50, 664),
    "实触覆盖按 10 mm 测地邻域计。轻载要求五个分区均低于 8 N。",
    17,
    film.MUTED,
  )
  return im


def render(root):
  source = OUT / "force-priority/wrist-active-result.mp4"
  data = film.read(root / "results.json")
  ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
  chapters = chapters_of(source, ffmpeg)
  start, _ = chapters.pop()
  case = next(
    r for r in data["cases"] if r["obj"] == "meat-can" and r["start"] == "start-02"
  )
  fresh = root / "runs/meat-can/start-02/sweep-dp"
  reference = OUT / "strong-baseline/meat-can/start-02/aude-mcc"
  for folder, key in ((fresh, "updated"), (reference, "aude")):
    if digest(folder / "trace.npz") != case["arms"][key]["trace_sha256"]:
      raise ValueError("Recorded physics differs from the reported can test")
  for item in data["cases"]:
    if item["obj"] in ("pear", "ellipsoid") and len(item["same_trace_fields"]) != 9:
      raise ValueError("The earlier film includes a trajectory that changed")
  chapters.append((start, "午餐肉罐的配对探索"))
  destination = root / "wrist-face-transfer.mp4"
  with TemporaryDirectory(dir=root, prefix="face-film-") as tmp, ExitStack() as stack:
    work = Path(tmp)
    cut = film.Replay(
      stack, OUT, "meat-can", "start-02", first_run=fresh, second_run=reference
    )
    title = "午餐肉罐的主动探索"
    subhead = "起点 02 · 同一四指接触初态，修正腕轨在本起点运行前已冻结。"
    note = "绿点表示轻载接触。下方实触点与重建只使用画面时刻以前的记录。"
    tail = 0
    with imageio.get_writer(
      work / "tail.mp4",
      fps=film.FPS,
      codec="libx264",
      quality=None,
      macro_block_size=2,
      ffmpeg_params=[
        "-threads",
        "2",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
      ],
    ) as writer:
      for tick in film.span(0, 24000, 12):
        writer.append_data(
          np.asarray(film.draw_frame(cut, tick, title, subhead, "12×", note))
        )
        tail += 1
      last = np.asarray(film.draw_frame(cut, 24000, title, subhead, "定格", note))
      for _ in range(3 * film.FPS):
        writer.append_data(last)
        tail += 1
      chapters.append((start + tail, "修正后的迁移结果"))
      card = np.asarray(end_card(data))
      for _ in range(7 * film.FPS):
        writer.append_data(card)
        tail += 1
    total = start + tail
    metadata = [";FFMETADATA1", "title=GPIS 手腕主动探索与 Aude 配对复验"]
    for (begin, name), (finish, _) in zip(chapters, chapters[1:] + [(total, "")], strict=True):
      metadata.extend(
        (
          "[CHAPTER]",
          f"TIMEBASE=1/{film.FPS}",
          f"START={begin}",
          f"END={finish}",
          f"title={name}",
        )
      )
    (work / "chapters.ffmeta").write_text("\n".join(metadata) + "\n", encoding="utf-8")
    subprocess.run(
      [
        ffmpeg,
        "-v",
        "error",
        "-i",
        str(source),
        "-i",
        str(work / "tail.mp4"),
        "-f",
        "ffmetadata",
        "-i",
        str(work / "chapters.ffmeta"),
        "-filter_complex",
        f"[0:v]trim=end_frame={start},setpts=PTS-STARTPTS[a];[a][1:v]concat=n=2:v=1:a=0[v]",
        "-map",
        "[v]",
        "-map_chapters",
        "2",
        "-map_metadata",
        "2",
        "-c:v",
        "libx264",
        "-threads",
        "2",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(work / "film.mp4"),
      ],
      check=True,
    )
    frames, _ = imageio_ffmpeg.count_frames_and_secs(str(work / "film.mp4"))
    if frames != total:
      raise ValueError(f"Expected {total} frames and encoded {frames}")
    (work / "film.mp4").replace(destination)
  save_json(
    root / "film.json",
    dict(frames=frames, fps=film.FPS, chapters=chapters, sha256=digest(destination)),
  )
  print(
    f"{frames} frames, {frames / film.FPS:.2f} seconds; can chapter begins at {start / film.FPS:.3f}",
    flush=True,
  )


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--root", type=Path, default=RESULTS)
  render(parser.parse_args().root)
