"""Render the paired YCB trials directly from the scored physical trajectories."""

import argparse
import subprocess
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np

from . import sweep_video as film
from .experiment import digest
from .face_regression_suite import REPO, read, write
from .pear_mustard import ROOT
from .ycb_fixtures import OBJECTS, fixture


def render(root=ROOT, seconds=240, speed=12):
  result = read(root / "results.json")
  if seconds not in (240, 600) or speed <= 0:
    raise ValueError("Use a scored horizon and a positive replay speed")
  end = seconds * 100
  names = list(result["objects"])
  stem = "five-ycb" if len(names) == 5 else "pear-mustard"
  frames, chapters = 0, []
  with TemporaryDirectory(dir=root) as temporary:
    work = Path(temporary)
    with imageio.get_writer(
      work / "frames.mp4",
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
      for obj in names:
        _, label, start = OBJECTS[obj]
        arms = result["objects"][obj]["native"]
        ours, other = (REPO / arms[a]["source"] for a in ("wrist-dp", "native-aude"))
        scores = {
          REPO / arms[a]["source"]: REPO / arms[a]["scored_at"]
          for a in ("wrist-dp", "native-aude")
        }
        inputs = fixture(obj)
        with ExitStack() as stack:
          pair = film.Replay(
            stack,
            root,
            obj,
            start,
            first_run=ours,
            second_run=other,
            first_scores=scores[ours],
            second_scores=scores[other],
            fixture_paths=inputs,
          )
          for a, source in zip(pair.arms, ("wrist-dp", "native-aude"), strict=True):
            coverage_key = "coverage_600" if seconds == 600 else "coverage"
            contact_key = "light3_600" if seconds == 600 else "light3"
            np.testing.assert_allclose(
              [a.coverage[end], a.three[end]],
              [arms[source][coverage_key], arms[source][contact_key]],
            )
          chapters.append((frames, label))
          title = f"{label} · 两套完整系统的主动探索"
          subtitle = f"同一四指起始抓姿，{seconds} 仿真秒。绿色指尖表示当前轻载接触。"
          note = "左、右分别根据各自的实测触点更新覆盖与重建。"
          for tick in film.span(0, end, speed):
            pic = film.draw_frame(pair, tick, title, subtitle, f"{speed}×", note)
            if obj == "pear" and tick == 12000:
              pic.save(work / "poster.webp", quality=88)
            writer.append_data(np.asarray(pic))
            frames += 1
          final = np.asarray(film.draw_frame(pair, end, title, subtitle, "定格", note))
          for _ in range(2 * film.FPS):
            writer.append_data(final)
            frames += 1
          print(f"{label} verified and rendered", flush=True)
    metadata = [";FFMETADATA1", "title=YCB 物体上的腕部主动探索"]
    for (begin, name), (chapter_end, _) in zip(
      chapters, chapters[1:] + [(frames, "")], strict=True
    ):
      metadata += [
        "[CHAPTER]",
        f"TIMEBASE=1/{film.FPS}",
        f"START={begin}",
        f"END={chapter_end}",
        f"title={name}",
      ]
    (work / "chapters.txt").write_text("\n".join(metadata) + "\n", encoding="utf-8")
    clip = work / "paired.mp4"
    subprocess.run(
      [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-v",
        "error",
        "-i",
        str(work / "frames.mp4"),
        "-f",
        "ffmetadata",
        "-i",
        str(work / "chapters.txt"),
        "-map",
        "0:v:0",
        "-map_metadata",
        "1",
        "-map_chapters",
        "1",
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        str(clip),
      ],
      check=True,
    )
    count, duration = imageio_ffmpeg.count_frames_and_secs(str(clip))
    if count != frames or abs(duration - frames / film.FPS) > 0.05:
      raise ValueError("Incomplete paired video")
    clip.replace(root / f"{stem}.mp4")
    (work / "poster.webp").replace(root / f"{stem}.webp")
    write(
      root / "video.json",
      dict(
        frames=count,
        seconds=duration,
        simulation_seconds_per_object=seconds,
        playback_speed=speed,
        chapters_s={n: t / film.FPS for t, n in chapters},
        trace_sha256={
          obj: {a: d["native"][a]["trace_sha256"] for a in ("wrist-dp", "native-aude")}
          for obj, d in result["objects"].items()
        },
        video_sha256=digest(root / f"{stem}.mp4"),
      ),
    )


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--root", type=Path, default=ROOT)
  parser.add_argument("--seconds", type=int, choices=(240, 600), default=240)
  parser.add_argument("--speed", type=int, default=12)
  args = parser.parse_args()
  render(args.root, args.seconds, args.speed)
