"""Render synchronized measured exploration with the same camera for each pair of fingerall runs."""

import argparse
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

import imageio.v2 as imageio
import imageio_ffmpeg
import mujoco
import numpy as np
import trimesh
from PARK.identity import digest, read
from PIL import Image, ImageDraw

from ..active_1_b import film_support as film
from ..active_1_b.video import HandView, view_axes
from ..i04_geometric_traversal.pad_contact_visual import camera_from_direction, marker
from . import report

FPS, SPEED = 20, 4
WIDTH, HEIGHT = 1600, 632
TILE = (276, 180)
TICKS = range(0, report.TICKS + 1, report.HZ * SPEED // FPS)
ROWS = (113, 354)


def recording(folder, obj, method, fixture):
    score = read(folder / "score.json")
    if (score["object"] != obj or score["method"] != method or
            score["initial_sha256"] != fixture["integration_state"] or score["model_sha256"] != fixture["model.mjb"] or
            score["reference_sha256"] != fixture["surface.obj"] or
            digest(folder / "trace.npz") != score["trace_sha256"] or
            digest(folder / "coverage.npz") != score["coverage_sha256"]):
        raise ValueError(f"Replay inputs do not match the scored {obj}/{method}")
    with np.load(folder / "trace.npz", allow_pickle=False) as tr:
        valid, _ = report.qualified(tr)
        active = (tr["pad_force_n"] > 0) & (np.linalg.norm(tr["normals_outward"], axis=2) > .5)
        qpos, points = tr["qpos"], tr["points_m"]
    with np.load(folder / "coverage.npz", allow_pickle=False) as coverage:
        curve = coverage["c25"]
        np.testing.assert_allclose(coverage["time_s"], np.arange(report.TICKS + 1) / report.HZ, rtol=0, atol=1e-10)
    np.testing.assert_allclose(curve[-1], score["c25"], rtol=0, atol=1e-10)
    if qpos.shape != (report.TICKS + 1, 23):
        raise ValueError("Incomplete motion trace")
    return qpos, points, active, valid, curve


def set_shared_camera(scene, recordings):
    initial = recordings[0][0][0]
    radial = initial[:3] - scene.center
    radial /= np.linalg.norm(radial)
    side = np.cross([0., 0., 1.], radial)
    if np.linalg.norm(side) < .05:
        side = np.cross([1., 0., 0.], radial)
    axes = view_axes(.92 * radial + side / np.linalg.norm(side))
    reference = scene.reference @ axes
    low, high = reference.min(axis=0), reference.max(axis=0)
    for qpos, *_ in recordings:
        for q in qpos[TICKS.start::TICKS.step]:
            scene.data.qpos[:] = q
            mujoco.mj_kinematics(scene.model, scene.data)
            rotations = scene.data.geom_xmat[scene.ids].reshape(-1, 3, 3)
            hand = scene.data.geom_xpos[scene.ids, None, :] + np.einsum("gij,gkj->gki", rotations, scene.corners)
            bounds = hand.reshape(-1, 3) @ axes
            low, high = np.minimum(low, bounds.min(axis=0)), np.maximum(high, bounds.max(axis=0))
    half = (high - low) / 2
    tangent = np.tan(np.radians(scene.model.vis.global_.fovy / 2))
    distance = half[2] + 1.07 * max(half[1], half[0] * TILE[1] / TILE[0]) / tangent
    camera_from_direction(scene.camera, axes[:, 2], axes @ ((high + low) / 2), max(.28, distance))


def hand_frame(scene, renderer, recording, tick):
    qpos, points, active, valid, _ = recording
    scene.data.qpos[:] = qpos[tick]
    mujoco.mj_kinematics(scene.model, scene.data)
    mujoco.mj_camlight(scene.model, scene.data)
    colors = [(0.26, .99, .66, 1.) if good else (1., .43, .25, 1.) if hit else (.36, .63, .74, 1.)
              for good, hit in zip(valid[tick], active[tick], strict=True)]
    for pad, color in zip(scene.pads, colors, strict=True):
        scene.model.geom_rgba[pad] = color
    renderer.update_scene(scene.data, camera=scene.camera, scene_option=scene.options)
    for idx in np.flatnonzero(active[tick]):
        marker(renderer.scene, points[tick, idx], .0045, colors[idx])
    return Image.fromarray(renderer.render())


def render(original, new, fixtures, output):
    columns = []
    template = Image.new("RGB", (WIDTH, HEIGHT), film.BG)
    static = ImageDraw.Draw(template)
    film.label(static, (22, 26), "完整 fingerall 的五物体同步探索", 28, bold=True)
    film.label(static, (22, 58), "AUDE 腕部规划  ·  single_all + MCC 手指执行  ·  4 倍速", 18, film.MUTED)
    for top, title in zip(ROWS, ("腕部不限距", "掌轴限距"), strict=True):
        film.label(static, (19, top + 85), title, 21, bold=True)
    static.line((18, 339, WIDTH - 18, 339), fill=film.HAIR)
    film.label(static, (20, 604), "绿色为合格指腹触点，橙色为其余受力触点。每列共用固定相机。C25 为当前实触覆盖率。", 17, film.MUTED)
    manifest = read(fixtures / "fixtures.json")["objects"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack, TemporaryDirectory(dir=output.parent) as temp:
        for i, (obj, label) in enumerate(report.OBJECTS.items()):
            spec, case = manifest[obj], fixtures / obj
            if any(digest(case / name) != spec[name] for name in ("model.mjb", "initial.npz", "surface.obj")):
                raise ValueError(f"Different paired fixture for {obj}")
            pair = [recording(root / obj / arm, obj, arm, spec)
                    for root, arm in ((original, "aude_mcc"), (new, "aude_mcc_height"))]
            model = film.read_model(case / "model.mjb")
            scene = HandView(model, trimesh.load_mesh(case / "surface.obj", process=False).vertices, pair[0][0][0])
            with np.load(case / "initial.npz", allow_pickle=False) as initial:
                mujoco.mj_setState(model, scene.data, initial["integration_state"], mujoco.mjtState.mjSTATE_INTEGRATION)
            for qpos, *_ in pair:
                np.testing.assert_allclose(qpos[0], scene.data.qpos, rtol=0, atol=1e-12)
            set_shared_camera(scene, pair)
            renderer = stack.enter_context(mujoco.Renderer(model, height=TILE[1], width=TILE[0]))
            left = 164 + 286 * i
            film.label(static, (left + TILE[0] // 2, 88), label, 20, bold=True, anchor="mm")
            columns.append((left, scene, renderer, pair))
        movie = Path(temp) / "exploration.mp4"
        with imageio.get_writer(movie, fps=FPS, codec="libx264", quality=8, macro_block_size=1, pixelformat="yuv420p",
                                output_params=["-movflags", "+faststart", "-threads", "2"]) as writer:
            for tick in TICKS:
                canvas = template.copy()
                pen = ImageDraw.Draw(canvas)
                film.label(pen, (1578, 32), f"{tick/report.HZ:05.1f} / 240.0 s", 25, mono=True, anchor="rm")
                for left, scene, renderer, pair in columns:
                    for top, rec in zip(ROWS, pair, strict=True):
                        canvas.paste(hand_frame(scene, renderer, rec, tick), (left, top))
                        film.label(pen, (left + 2, top + TILE[1] + 17), f"C25 {rec[4][tick]:.2%}", 18, mono=True)
                        film.label(pen, (left + TILE[0] - 2, top + TILE[1] + 17),
                                   f"{rec[3][tick].sum()} 指接触", 16, film.MUTED, anchor="rm")
                writer.append_data(np.asarray(canvas))
                if tick in (0, report.TICKS):
                    canvas.save(Path(temp) / f"check-{tick}.png")
            for _ in range(FPS - 1):
                writer.append_data(np.asarray(canvas))
        frames, duration = imageio_ffmpeg.count_frames_and_secs(str(movie))
        if frames != len(TICKS) + FPS - 1 or abs(duration - frames/FPS) > 1e-6:
            raise ValueError("Replay has an incomplete timeline")
        with imageio.get_reader(movie) as reader:
            for tick, index in ((0, 0), (report.TICKS, frames - 1)):
                decoded = reader.get_data(index).astype(np.int16)
                reference = np.asarray(Image.open(Path(temp) / f"check-{tick}.png"))
                if np.abs(decoded - reference).mean() > 5:
                    raise ValueError(f"Encoded replay differs from its measured source at {tick}")
        movie.replace(output)
    print(f"{output}: {frames} frames, {duration:.1f} s", flush=True)


def main():
    root = report.ROOT_OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, default=root / "extended/runs")
    parser.add_argument("--new", type=Path, default=root / "fingerall/runs")
    parser.add_argument("--fixtures", type=Path, default=report.FIXTURES)
    parser.add_argument("--out", type=Path, default=root / "fingerall/report/exploration.mp4")
    args = parser.parse_args()
    render(args.control, args.new, args.fixtures, args.out)


if __name__ == "__main__":
    main()
