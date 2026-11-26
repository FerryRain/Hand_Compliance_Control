"""Render the Active-3 five-object plate and paired pear replays from scored trials."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import imageio.v2 as imageio
import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw
import torch
import trimesh

from Module.active_1_b import film_support as film
from Module.active_1_b.video import FPS, SIZE, VIEW, HandView, draw, surface_view
from Module.palmreach6.__main__ import FIXTURES
from Module.wrist_active.evaluate import Reference, reconstruct
from PARK.core.common import ROOT, digest, read, write


OBJECTS = {"pear": "梨", "mustard": "芥末瓶", "cracker": "饼干盒", "bleach": "漂白剂瓶", "coffee": "咖啡罐"}
CONDITIONS = {"planned": "AUDE 规划手指", "single_all": "single_all 主动调指", "compliance": "仅底层柔顺"}
EYES = ((.56, -1., .62), (-.56, 1., .45))
SECONDS, HZ, SPEED, INTERVAL = 240, 100, 4, 15


def scored_mesh(run):
    scores = read(run / "score.json")
    if (scores["object"], scores["method"]) != (run.parent.name, run.name):
        raise ValueError(f"Wrong object or finger condition in {run}")
    path = run / f"signed-{SECONDS}s.ply"
    if digest(path) != scores["endpoint_sha256"]:
        raise ValueError(f"Different scored endpoint in {run}")
    return scores, path


def plate(root, fixtures):
    width, row_height, left, col_width = 1280, 158, 95, 290
    canvas = Image.new("RGB", (width, 83 + row_height * len(OBJECTS)), film.PANEL)
    pen = ImageDraw.Draw(canvas)
    film.label(pen, (16, 26), "Active-3　五物体最终触觉重建", 26, bold=True)
    film.label(pen, (width-16, 26), "240 秒 · 同一物体共用视角和比例", 16, anchor="rm", color=film.MUTED)
    for j, title in enumerate(("离线真值", *CONDITIONS.values())):
        film.label(pen, (left+(j+.5)*col_width, 65), title, 19, bold=True, anchor="mm")
    records = {}
    for i, (obj, name) in enumerate(OBJECTS.items()):
        case = fixtures / obj
        truth_hash = digest(case / "surface.obj")
        truth = trimesh.load_mesh(case / "surface.obj", process=False)
        row = {"surface_sha256": truth_hash, "conditions": {}}
        y = 82 + row_height * i
        pen.line((12, y, width-12, y), fill=film.HAIR)
        film.label(pen, (12, y+69), name, 16, bold=True)
        for j, condition in enumerate((None, *CONDITIONS)):
            x = int(left + (j+.5)*col_width)
            mesh = truth
            if condition:
                run = root / "runs" / obj / condition
                scores, path = scored_mesh(run)
                receipt = read(run / "run.json")
                if receipt["offline_mesh_sha256"] != truth_hash or receipt["trace_sha256"] != scores["trace_sha256"]:
                    raise ValueError(f"Different fixture or trace in {run}")
                mesh = trimesh.load_mesh(path, process=False)
                row["conditions"][condition] = {key: scores[key] for key in ("trace_sha256", "endpoint_sha256", "c25", "f5")}
                film.label(pen, (x, y+139), f"C25 {scores['c25']:.2%}    F@5 {scores['f5']:.2%}", 15, anchor="mm")
            for n, eye in enumerate(EYES):
                tile = surface_view(truth.vertices, mesh, eye).resize((109, 119), Image.Resampling.LANCZOS)
                canvas.paste(tile, (x-114+n*121, y+4))
        records[obj] = row
    output = root / "site-overlay/images/active-2/five-objects.webp"
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=93, method=6)
    write(root / "media-evidence/five-objects.json",
          dict(image_sha256=digest(output), size_px=canvas.size, seconds=SECONDS, objects=records))


def checkpoints(root, ref, mode, scores, endpoint, points, normals, qualified):
    folder = root / "media-evidence" / "pear" / mode
    folder.mkdir(parents=True, exist_ok=True)
    ticks, fingers = np.nonzero(qualified)
    xyz, directions = points[ticks, fingers], normals[ticks, fingers]
    snapshots = []
    for seconds in range(0, SECONDS + 1, INTERVAL):
        count = int(np.searchsorted(ticks, seconds * HZ, side="right"))
        if seconds == SECONDS:
            path, f5 = endpoint, scores["f5"]
        else:
            path = folder / f"signed-{seconds}s.ply"
            quality = reconstruct(xyz[:count], directions[:count], ref, path, "signed", signed_device="cuda")
            f5 = quality["fscore"] if quality["status"] == "ok" else None
            if f5 is None:
                path = None
        snapshots.append(dict(s=seconds, samples=count, mesh=path.relative_to(root).as_posix() if path else None,
                              mesh_sha256=digest(path) if path else None, f5=f5))
    return snapshots


def movie(root, fixtures, mode):
    case, run = fixtures / "pear", root / "runs/pear" / mode
    scores, endpoint = scored_mesh(run)
    receipt = read(run / "run.json")
    spec = read(fixtures / "fixtures.json")["objects"]["pear"]
    if not (digest(run / "trace.npz") == receipt["trace_sha256"] == scores["trace_sha256"]
            and digest(run / "coverage.npz") == scores["coverage_sha256"]
            and digest(case / "model.mjb") == receipt["model_sha256"] == scores["model_sha256"]
            and digest(case / "surface.obj") == receipt["offline_mesh_sha256"] == scores["reference_sha256"]
            and receipt["initial_sha256"] == spec["integration_state"] == scores["initial_sha256"]
            and receipt["seed"] == spec["seed"] == scores["seed"] and receipt["steps"] == SECONDS * HZ):
        raise ValueError(f"Replay data differ from the scored pear trial in {run}")
    with np.load(run / "trace.npz", allow_pickle=False) as tr, np.load(run / "coverage.npz", allow_pickle=False) as curves:
        q, points, normals = tr["qpos"], tr["points_m"], tr["normals_outward"]
        live = (tr["pad_force_n"] > 0) & (np.linalg.norm(normals, axis=2) > .5)
        qualified = tr["pad_valid"] & (tr["region_load_n"].max(1) < 8.)[:, None]
        curve = curves["c25"]
        np.testing.assert_allclose(curves["time_s"], np.arange(SECONDS*HZ+1)/HZ, atol=1e-10, rtol=0)
        np.testing.assert_allclose(curve[-1], scores["c25"], atol=1e-10, rtol=0)
    if q.shape != (SECONDS*HZ+1, 23) or curve.shape != (len(q),) or not np.isfinite(q).all():
        raise ValueError(f"Incomplete pear trajectory in {run}")
    ref = Reference(case / "surface.obj")
    snaps = checkpoints(root, ref, mode, scores, endpoint, points, normals, qualified)
    vertices = trimesh.load_mesh(case / "surface.obj", process=False).vertices
    views = []
    for snap in snaps:
        mesh = trimesh.load_mesh(root / snap["mesh"], process=False) if snap["mesh"] else None
        views.append([surface_view(vertices, mesh, eye) for eye in EYES])
    model = film.read_model(case / "model.mjb")
    scene = HandView(model, vertices, q[0])
    with np.load(case / "initial.npz", allow_pickle=False) as initial:
        mujoco.mj_setState(model, scene.data, initial["integration_state"], mujoco.mjtState.mjSTATE_INTEGRATION)
        np.testing.assert_allclose(q[0], scene.data.qpos, rtol=0, atol=1e-12)
    overlay = root / "site-overlay"
    movie_path = overlay / f"media/active-2/pear-{mode}-4x.mp4"
    poster_path = overlay / f"images/active-2/pear-{mode}.webp"
    movie_path.parent.mkdir(parents=True, exist_ok=True)
    poster_path.parent.mkdir(parents=True, exist_ok=True)
    step = HZ*SPEED // FPS
    ticks = range(0, SECONDS*HZ+1, step)
    if ticks[-1] != SECONDS*HZ or step*FPS != HZ*SPEED:
        raise ValueError("Replay frame cadence misses the scored endpoint")
    with TemporaryDirectory(dir=root) as tmp:
        video = Path(tmp) / movie_path.name
        poster = Path(tmp) / poster_path.name
        with mujoco.Renderer(model, height=VIEW[1], width=VIEW[0]) as renderer, imageio.get_writer(
                video, fps=FPS, codec="libx264", quality=8, macro_block_size=1, pixelformat="yuv420p",
                output_params=["-movflags", "+faststart", "-threads", "2"]) as writer:
            for i, tick in enumerate(ticks):
                j = tick // (INTERVAL * HZ)
                physical = scene.render(renderer, q[tick], live[tick], qualified[tick], points[tick])
                frame = draw(tick, SECONDS, SPEED, "梨　/　"+CONDITIONS[mode], physical, views[j],
                             snaps[j], curve[tick], live[tick], qualified[tick])
                writer.append_data(np.asarray(frame))
                if i == len(ticks)//2:
                    frame.save(poster, quality=91)
            # Display the endpoint for one second, including the last sampled frame.
            for _ in range(FPS-1):
                writer.append_data(np.asarray(frame))
        count, duration = imageio_ffmpeg.count_frames_and_secs(str(video))
        if count != len(ticks)+FPS-1 or abs(duration-count/FPS) > .011:
            raise ValueError(f"Encoded timeline differs at {video}: {count} frames, {duration} seconds")
        with imageio.get_reader(video) as reader:
            video_meta = reader.get_meta_data()
            if video_meta["fps"] != FPS or tuple(video_meta["size"]) != SIZE:
                raise ValueError("Unexpected video dimensions or frame rate")
            for i, reference in ((len(ticks)//2, Image.open(poster)), (count-1, frame)):
                difference = np.abs(reader.get_data(i).astype(np.int16)-np.asarray(reference))
                if difference.mean() > 5:
                    raise ValueError(f"Decoded frame {i} differs from the synchronized source")
        video.replace(movie_path)
        poster.replace(poster_path)
    write(root / "media-evidence/pear" / mode / "video.json", dict(
        object="pear", condition=mode, seed=receipt["seed"], initial_sha256=receipt["initial_sha256"],
        model_sha256=receipt["model_sha256"], trace_sha256=scores["trace_sha256"], surface_sha256=receipt["offline_mesh_sha256"],
        coverage_sha256=digest(run / "coverage.npz"), endpoint_sha256=scores["endpoint_sha256"], endpoint_c25=scores["c25"],
        endpoint_f5=scores["f5"], checkpoints=snaps, speed=SPEED, source_hz=HZ, frame_stride_ticks=step,
        motion_tick_range=[ticks[0], ticks[-1]], fps=FPS, frame_count=count, duration_s=count/FPS,
        endpoint_display_s=1, poster_tick=ticks[len(ticks)//2], size_px=SIZE, movie_sha256=digest(movie_path),
        poster_sha256=digest(poster_path)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset", choices=("plate", *CONDITIONS))
    parser.add_argument("--root", type=Path, default=ROOT / "outputs/active-3-paired")
    parser.add_argument("--fixtures", type=Path, default=FIXTURES)
    args = parser.parse_args()
    torch.set_num_threads(1)
    root, fixtures = args.root.resolve(), args.fixtures.resolve()
    if args.asset == "plate":
        plate(root, fixtures)
    else:
        movie(root, fixtures, args.asset)


if __name__ == "__main__":
    main()
