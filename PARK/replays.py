"""Synchronized three-way replay of recorded MuJoCo states and causal scored meshes."""
from contextlib import closing
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import imageio.v2 as imageio
import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import maximum_filter
import trimesh

from Module.palmreach6.__main__ import FIXTURES
from Module.wrist_active.history import read_model
from PARK.comparison import METHODS
from PARK.core.common import digest, read, write
from PARK.replay_video import Track, face, load_view

ROOT = Path(__file__).resolve().parents[1]
NAMES = dict(coffee="咖啡罐", cracker="饼干盒", mustard="芥末瓶", bleach="漂白剂瓶", pear="梨")
COLORS = ("#63ddbd", "#8cb9ff", "#ffc077")
INK, MUTED, BG, LINE = "#f2f4f5", "#b4c1c6", "#132029", "#415361"
SIZE, MINI, FPS = (1536, 752), (229, 147), 20
XS = (24, 528, 1032)


class SurfaceView:
    def __init__(self, mesh, basis):
        self.basis = basis
        plane = np.asarray(mesh.vertices) @ basis[:, :2]
        span = np.maximum(np.ptp(plane, axis=0), .001)
        self.center = .5 * (plane.max(0) + plane.min(0))
        self.scale = min((np.array(MINI)-24)/span)
        self.reference = self.image(mesh, (82, 96, 105))

    def pixels(self, points):
        xy = (np.asarray(points) @ self.basis[:, :2] - self.center) * self.scale
        return np.rint(np.array(MINI)/2 + xy*[1,-1]).astype(int)

    def image(self, mesh, rgb):
        points, faces = trimesh.sample.sample_surface(mesh, 65000, seed=71)
        px = self.pixels(points)
        ok = ((px >= 0) & (px < MINI)).all(1)
        dist = points @ self.basis[:, 2]
        order = np.flatnonzero(ok)[np.argsort(dist[ok], kind="stable")]
        light = np.clip(np.abs(mesh.face_normals[faces] @ (self.basis @ np.array([-.36,.48,.8]))), 0, 1)
        shade = np.zeros((MINI[1], MINI[0]), np.uint8)
        shade[px[order,1], px[order,0]] = np.rint(122 + 128*light[order]).astype(np.uint8)
        shade = maximum_filter(shade, size=2)
        colors = (.00392 * shade[...,None] * np.array(rgb)).astype(np.uint8)
        colors[shade == 0] = (22, 36, 45)
        return Image.fromarray(colors)


def label(draw, xy, text, size, color=INK, bold=False, anchor="lt"):
    draw.text(xy, text, font=face(size, bold), fill=color, anchor=anchor)


def header(obj, seconds, seed, speed):
    edge = SIZE[0]-25
    page = Image.new("RGB", SIZE, BG)
    draw = ImageDraw.Draw(page)
    label(draw, (24, 17), f"{NAMES[obj]}  /  触觉探索", 37, bold=True)
    label(draw, (edge, 32), f"种子 {seed}     ·     同步回放 {speed}×", 22, MUTED, anchor="rt")
    for x, method, color in zip(XS, METHODS, COLORS):
        draw.line((x,76,x+478,76), fill=color, width=3)
        label(draw,(x,90),method,28,color,True)
        label(draw,(x,447),"累计合格触点",20,MUTED)
        label(draw,(x+249,447),"已评分重建",20,MUTED)
    draw.line((24,686,edge,686),fill=LINE)
    label(draw,(24,705),"腕位 □     绿色为合格接触，橙色为超载时的接触",18,MUTED)
    label(draw,(527,705),"左栏触点图   ◇ 全局选区    ＋ 已下发目标",18,INK)
    label(draw,(1511,705),"C25 每秒实测   ·   F@5 按对应网格的评分时刻更新",18,MUTED,anchor="rt")
    return page


class GoalOverlay:
    def __init__(self, native, surface_view, receipt):
        log = native / "global-plans.jsonl"
        if digest(log) != receipt["telemetry"][log.name]:
            raise ValueError(f"Unverified planner targets at {log}")
        with log.open(encoding="utf-8") as stream:
            events = [json.loads(line) for line in stream]
        self.ticks = [r["apply_tick"] for r in events]
        self.states = []
        active = None
        for row in events:
            if row["reason"] == "new_target":
                active = surface_view.pixels([row["point"]])[0]
            elif row.get("target_invalidated"):
                active = None
            frontier = row.get("global_frontier_m")
            frontier = surface_view.pixels([frontier])[0] if frontier is not None else None
            self.states.append((active, frontier))

    def draw(self, image, tick):
        event = np.searchsorted(self.ticks, tick, side="right") - 1
        if event < 0:
            return
        draw = ImageDraw.Draw(image)
        active, frontier = self.states[event]
        for point, symbol in ((frontier, "diamond"), (active, "cross")):
            if point is None or not ((point >= 0) & (point < MINI)).all():
                continue
            u,v = map(int, point)
            if symbol == "diamond":
                shape = [(u,v-7),(u+7,v),(u,v+7),(u-7,v),(u,v-7)]
                draw.line(shape, fill="#0e1b23", width=5)
                draw.line(shape, fill="#ffe681", width=2)
            else:
                for line in ([(u-5,v),(u+5,v)],[(u,v-5),(u,v+5)]):
                    draw.line(line, fill="#0e1b23", width=5)
                    draw.line(line, fill="white", width=2)


def replay(root, obj, seconds, view_class, repeat=0, *, destination=None):
    if seconds not in (240, 600):
        raise ValueError("Choose a scored 240 or 600 second budget")
    seed = read(FIXTURES / "fixtures.json")["objects"][obj]["seed"] + repeat
    key = f"seed-{seed}"
    history = ROOT / "outputs/PARK/experiments" / f"{seconds}s" / obj
    acquisition = root / "trials" / f"{seconds}s" / obj / key
    native = acquisition / "native"
    receipt = read(native / "adapter.json")
    signature = receipt["signature"]
    if (signature["method"], signature["object"], signature["seconds"], signature["seed"]) != (
            "PARK", obj, seconds, seed):
        raise ValueError(f"Incorrect PARK acquisition at {native}")
    folders = (acquisition / "score", history / "palmreach-6" / key, history / "aude" / key)
    tracks = [Track(folder, obj, seed, seconds, method) for folder, method in zip(folders, METHODS, strict=True)]
    for track in tracks[1:]:
        np.testing.assert_array_equal(track.q[0], tracks[0].q[0])
        if (track.initial,track.model) != (tracks[0].initial,tracks[0].model):
            raise ValueError("Unpaired video inputs")
    if destination is None:
        destination = root / "comparisons" / f"{seconds}s" / f"repeat-{repeat+1}" / f"{obj}.mp4"
    dest = Path(destination)
    output = dest.parent
    output.mkdir(parents=True, exist_ok=True)
    meta = dest.with_suffix(".json")
    step = 50 if seconds == 240 else 100
    speed = FPS * step // 100
    ticks = np.arange(0, seconds*100+1, step)
    signature = dict(object=obj, seed=seed, budget_s=seconds, fps=FPS, frame_stride_ticks=step,
                     source_sha256=digest(Path(__file__)), traces=[t.sha for t in tracks],
                     methods=list(METHODS), frames=len(ticks)+FPS, size=list(SIZE),
                     global_logs_sha256=[receipt["telemetry"]["global-plans.jsonl"]])
    if dest.is_file() and meta.is_file() and read(meta) == dict(signature=signature, sha256=digest(dest)):
        return dest
    truth = trimesh.load_mesh(FIXTURES/obj/"surface.obj", process=False)
    model = read_model(FIXTURES/obj/"model.mjb")
    view = view_class(model,truth,tracks,sample_ticks=ticks)
    mini = SurfaceView(truth,view.basis)
    overlay = GoalOverlay(native, mini, receipt)
    clouds, meshes = [], []
    for track, folder, color in zip(tracks, folders, COLORS):
        t, f = np.nonzero(track.light)
        points = track.points[t, f]
        _, first = np.unique(np.floor(points/.0045).astype(int), axis=0, return_index=True)
        first.sort()
        clouds.append((t[first], mini.pixels(points[first])))
        rgb = tuple(bytes.fromhex(color[1:]))
        meshes.append({s: mini.image(trimesh.load_mesh(folder/f"signed-{s}s.ply",process=False), rgb)
                       for s in track.f5 if (folder/f"signed-{s}s.ply").is_file()})

    fixed = header(obj, seconds, seed, speed)
    contact_images = [mini.reference.copy() for _ in tracks]
    shown = [0]*len(tracks)
    with TemporaryDirectory(dir=output) as temp:
        partial = Path(temp) / "video.mp4"
        with (closing(mujoco.Renderer(model,height=390,width=604)) as renderer,
              imageio.get_writer(str(partial),format="FFMPEG",mode="I",fps=FPS,codec="libx264",
                                 quality=7,pixelformat="yuv420p",macro_block_size=1,
                                 output_params=["-movflags","+faststart","-threads","2"]) as writer):
            for tick in ticks:
                sec = tick/100
                frame = fixed.copy()
                draw = ImageDraw.Draw(frame)
                label(draw,(SIZE[0]//2,28),f"{sec:g} / {seconds} s",31,anchor="mt")
                for j,(x,track,color) in enumerate(zip(XS,tracks,COLORS)):
                    photo = view.image(renderer,track,tick).resize((480,310),Image.Resampling.LANCZOS)
                    frame.paste(photo,(x,128))
                    label(draw,(x+478,97),f"合格指尖 {track.light[tick].sum()} / 4",17,MUTED,anchor="rt")
                    times,px = clouds[j]
                    upto = int(np.searchsorted(times,tick,"right"))
                    pen = ImageDraw.Draw(contact_images[j])
                    for u,v in px[shown[j]:upto]:
                        pen.ellipse((u-1,v-1,u+1,v+1),fill=color)
                    shown[j] = upto
                    cloud = contact_images[j]
                    if j == 0:
                        cloud = cloud.copy()
                        overlay.draw(cloud, tick)
                    frame.paste(cloud,(x,476))
                    # Display the most recent scored mesh at or before the current physical frame.
                    last = max((s for s in track.f5 if s <= sec), default=None)
                    if last in meshes[j]:
                        frame.paste(meshes[j][last],(x+249,476))
                    else:
                        label(draw,(x+364,545),"等待评分网格",17,MUTED,anchor="mm")
                    label(draw,(x,634),f"C25  {100*track.c25[tick//100]:.1f}%",24,color,True)
                    label(draw,(x+249,634),f"F@5  {100*track.f5[last]:.1f}%" if last is not None else "F@5  —",24,color,True)
                    if last is not None:
                        label(draw,(x+478,451),f"{last} s",16,MUTED,anchor="rt")
                    end = int(sec//10)*10
                    if end:
                        delta = 100*(track.c25[end]-track.c25[end-10])
                        label(draw,(x,665),f"最近完整 10 s 窗口   C25 {delta:+.2f} 个百分点",15,
                              "#ffc077" if delta < .1 else MUTED)
                array = np.asarray(frame)
                writer.append_data(array)
                if tick == seconds*100:
                    for _ in range(FPS):
                        writer.append_data(array)
        # Decode the end frame before publishing a clip.
        with imageio.get_reader(str(partial),format="FFMPEG") as reader:
            if reader.count_frames() != signature["frames"] or reader.get_data(signature["frames"]-1).shape[:2] != SIZE[::-1]:
                raise ValueError("Invalid encoded replay dimensions or frame count")
        partial.replace(dest)
        write(meta, dict(signature=signature,sha256=digest(dest)))
    print(f"Verified {dest.name} ({seconds}s; {signature['frames']} frames)",flush=True)
    return dest


def render(root, seconds, obj=None, repeat=0):
    if seconds not in (240,600):
        raise ValueError("Choose a scored 240 or 600 second budget")
    view_class = load_view()
    if obj is not None:
        return replay(root,obj,seconds,view_class,repeat)
    movies = [replay(root,name,seconds,view_class,repeat) for name in NAMES]
    join_movies(movies, movies[0].parent/"five-objects.mp4")


def join_movies(movies, destination):
    output = destination.parent
    if any(p.parent != output for p in movies):
        raise ValueError("The comparison chapters must share an output directory")
    with TemporaryDirectory(dir=output) as temp:
        temp = Path(temp)
        listing = temp/"chapters.txt"
        listing.write_text("".join(f"file '../{p.name}'\n" for p in movies), encoding="ascii")
        joined = temp/destination.name
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),"-hide_banner","-loglevel","error","-f","concat",
                        "-safe","0","-i",str(listing),"-c","copy","-movflags","+faststart",str(joined)],check=True)
        with imageio.get_reader(str(joined), format="FFMPEG") as reader:
            frames = sum(read(p.with_suffix(".json"))["signature"]["frames"] for p in movies)
            if reader.count_frames() != frames:
                raise ValueError("Joined replay lost frames")
        joined.replace(destination)
