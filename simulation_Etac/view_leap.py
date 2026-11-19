"""Visualize live ETac deformations on the palm-facing side of the MuJoCo LEAP Hand."""

import argparse
from functools import lru_cache
import os
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import blind_grasping_assets
from .check_leap import INDENT_M, RETRACT_M, SurfaceProbe, load_model
from .mujoco_leap import MujocoLeapETac, NODE_SPACING_M


TARGETS = (
    ("掌心面", 0, (0, 0, -1), 0.008),
    ("食指近节", 2, (0, 0, -1), 0.005),
    ("中指远节", 7, (0, 0, -1), 0.004),
    ("拇指远节侧面", 15, (0, -1, 0), 0.004),
)
TIMES = np.array([0.0, 1.10, 2.10, 4.10, 5.20, 6.20])
PHASES = ("靠近", "接触", "滑动", "分离", "离开")
SLIDE = np.array([-1, -1, -1, 1, 1, -1])
GAP = np.array([RETRACT_M, -INDENT_M, -INDENT_M, -INDENT_M, RETRACT_M, RETRACT_M])
PANEL_WIDTH, PANEL_HEIGHT = 376, 694
DISPLAY_GAIN = 8
DISPLAY_LIMIT_M = 0.012
MARGIN = 18
IDENTITY = np.eye(3).ravel()


class LeapDemo:
    def __init__(self):
        self.model = load_model(probe=True)
        self.data = mujoco.MjData(self.model)
        self.reader = MujocoLeapETac(self.model, blind_grasping_assets() / "propagation-ckpt")
        self.probe = SurfaceProbe(self.model, self.data, self.reader)
        self.targets = tuple(self.probe.find_contact(link, (normal,), slide=travel)
                             for _, link, normal, travel in TARGETS)
        self.model.geom_rgba[self.reader.object_id] = [0.23, 0.56, 0.98, 0.24]
        self.model.geom_rgba[self.reader.geom_ids, 3] = 0.22
        self.model.vis.headlight.ambient[:] = 0.25
        self.model.vis.headlight.diffuse[:] = 0.51
        self.model.light_diffuse[:] *= 0.65
        self.model.vis.quality.numslices = 6
        self.model.vis.quality.numstacks = 3

        self.display_points = self.probe.points + 0.00008 * self.probe.normals
        self.elapsed = 1.65
        self.step(0)
        indented, _ = self.reader.deformation(self.data)
        if not indented.any() or not self.probe.contacts():
            raise RuntimeError("The initial probe pose did not produce physical and tactile contact.")

    def place_probe(self):
        cycle, seconds = divmod(self.elapsed, TIMES[-1])
        route = int(cycle) % len(self.targets)
        segment = int(np.searchsorted(TIMES, seconds, side="right") - 1)
        duration = TIMES[segment + 1] - TIMES[segment]
        progress = (seconds - TIMES[segment]) / duration
        interpolation = progress**2 * (3 - 2 * progress)
        rate = 6 * progress * (1 - progress) / duration
        travel = TARGETS[route][3]
        slide_change = travel * (SLIDE[segment + 1] - SLIDE[segment])
        gap_change = GAP[segment + 1] - GAP[segment]
        self.probe.place(
            self.targets[route],
            slide=travel * SLIDE[segment] + interpolation * slide_change,
            gap=GAP[segment] + interpolation * gap_change,
            tangent_speed=rate * slide_change,
            normal_speed=rate * gap_change,
        )
        return route, PHASES[segment]

    def step(self, count):
        for _ in range(count):
            self.place_probe()
            mujoco.mj_step(self.model, self.data)
            self.elapsed += self.model.opt.timestep
        state = self.place_probe()
        mujoco.mj_forward(self.model, self.data)
        return state

    def sample(self):
        indented, deformation = self.reader.deformation(self.data)
        shifted = deformation.cpu().numpy() * DISPLAY_GAIN
        magnitude = np.linalg.norm(shifted, axis=-1)
        if not np.isfinite(magnitude).all():
            raise RuntimeError("ETac produced a non-finite displacement.")
        shifted /= np.maximum(1, magnitude[:, None] / DISPLAY_LIMIT_M)
        strength = np.sqrt(np.clip(magnitude / DISPLAY_LIMIT_M, 0, 1))
        return indented.cpu().numpy(), shifted, strength


def set_camera(cam, target):
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.041, -0.020, 0.064]
    cam.azimuth = 92
    cam.elevation = -50 if target.normal[2] > 0 else 51
    if target.normal[1] < 0:
        cam.elevation = 37
    cam.distance = 0.350


class DeformationCloud:
    """Cache the dense reference cloud and update the displaced nodes every frame."""

    def __init__(self, scene, points):
        self.scene = scene
        self.points = points
        self.start = scene.ngeom
        self.end = self.start + len(points)
        self.previous = np.zeros(len(points), dtype=bool)
        self.idle_color = np.array([0.33, 0.52, 0.57, 1], dtype=np.float32)
        self.idle_radius = 0.00021
        for i, point in enumerate(points):
            geom = scene.geoms[self.start + i]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, [self.idle_radius, 0, 0],
                                point, IDENTITY, self.idle_color)
            geom.emission = 0.32
        scene.ngeom = self.end

    def update(self, indented, shifted, strength):
        moving = indented | (strength > 0.025)
        for index in np.flatnonzero(moving | self.previous):
            geom = self.scene.geoms[self.start + index]
            geom.pos[:] = self.points[index] + shifted[index]
            if not moving[index]:
                geom.rgba[:] = self.idle_color
                geom.size[0] = self.idle_radius
                continue
            lit = strength[index]
            if indented[index]:
                geom.rgba[:] = [1, 0.38 + 0.39 * lit, 0.12 + 0.20 * lit, 1]
            else:
                geom.rgba[:] = [0.10 + 0.15 * lit, 0.53 + 0.39 * lit, 0.62 + 0.34 * lit, 1]
            geom.size[0] = 0.00029 + 0.00016 * indented[index] + 0.00022 * lit
        self.previous = moving

        start = self.end
        arrows = np.flatnonzero(strength > 0.19)
        # Use a bounded set of vectors so the full deformed cloud remains legible.
        arrows = arrows[:: max(1, (len(arrows) + 47) // 48)]
        for index in arrows:
            origin = self.points[index]
            endpoint = origin + shifted[index]
            arrow = self.scene.geoms[start]
            color = [1.0, 0.62, 0.28, 0.84] if indented[index] else [0.22, 0.82, 0.80, 0.75]
            mujoco.mjv_initGeom(arrow, mujoco.mjtGeom.mjGEOM_ARROW, [0.00016, 0, 0], origin, IDENTITY, color)
            mujoco.mjv_connector(arrow, mujoco.mjtGeom.mjGEOM_ARROW, 0.00017, origin, endpoint)
            arrow.emission = 0.55
            start += 1
        self.scene.ngeom = start


@lru_cache(maxsize=None)
def font(size, bold=False):
    name = "msyhbd.ttc" if bold else "msyh.ttc"
    return ImageFont.truetype(str(Path(os.environ["WINDIR"]) / "Fonts" / name), size)


def closeup(demo, route, indented, shifted, strength):
    target = demo.targets[route]
    nodes = demo.reader.link_slices[target.link]
    frame = np.column_stack((target.tangent, np.cross(target.normal, target.tangent), target.normal))
    local = (demo.probe.points[nodes] - target.point) @ frame
    # Rotate the 3D shape towards the viewer so the normal displacement stays visible.
    projection = np.array([[0.866, -0.50, 0], [0.23, 0.398, -0.888]])
    visible = (np.abs(local) < [0.020, 0.018, 0.006]).all(axis=1)
    visible &= (demo.probe.normals[nodes] @ target.normal > 0.12) | indented[nodes]
    start = local[visible] @ projection.T
    moved = (local[visible] + shifted[nodes][visible] @ frame) @ projection.T
    active = indented[nodes][visible]
    responding = (strength[nodes][visible] > 0.043) | active
    image = Image.new("RGB", (PANEL_WIDTH - 40, 208), "#0a151e")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width - 1, image.height - 1), outline="#2b4354")
    origin = np.array([image.width / 2, 78])
    start = start * 5700 + origin
    moved = moved * 5700 + origin
    for x, y in start:
        draw.ellipse((x - 0.8, y - 0.8, x + 0.8, y + 0.8), fill="#66828d")
    ids = np.flatnonzero(responding)
    stride = max(1, (len(ids) + 29) // 30)
    for index in ids[::stride]:
        before, after, source = start[index], moved[index], active[index]
        color = "#ffad68" if source else "#59cfca"
        draw.line((tuple(before), tuple(after)), fill=color, width=1)
        delta = after - before
        length = np.linalg.norm(delta)
        if length > 7:
            axis = delta / length
            tail = after - 4.5 * axis
            side = np.array([-axis[1], axis[0]]) * 2.0
            draw.polygon((tuple(after), tuple(tail + side), tuple(tail - side)), fill=color)
    for (x, y), source in zip(moved[responding], active[responding]):
        color = "#ffad68" if source else "#59cfca"
        radius = 1.45 if source else 1.06
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    return image


def make_panel(demo, route, phase, indented, shifted, strength):
    counts = [int(indented[nodes].sum()) for nodes in demo.reader.link_slices]
    response = [bool((strength[nodes] > 0.025).any()) for nodes in demo.reader.link_slices]
    touching = bool(demo.probe.contacts())
    panel = Image.new("RGB", (PANEL_WIDTH, PANEL_HEIGHT), "#101e2a")
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, PANEL_WIDTH - 1, PANEL_HEIGHT - 1), outline="#3a5365")
    draw.rectangle((1, 1, PANEL_WIDTH - 2, 3), fill="#58c9c3")
    draw.text((20, 16), "LEAP Hand  /  ETac", font=font(14, True), fill="#9bdcd9")
    draw.text((19, 40), "掌心侧三维形变", font=font(27, True), fill="#f5f8fc")
    draw.rounded_rectangle((289, 47, 354, 76), radius=8,
                           fill="#71402a" if touching else "#293f51")
    draw.text((321, 61), phase, anchor="mm", font=font(15, True), fill="#fdf3e7")
    draw.text((20, 88), f"{1000 * NODE_SPACING_M:g} mm 间距  ·  {indented.size:,} 节点",
              font=font(15), fill="#b1c5d4")
    draw.text((20, 126), TARGETS[route][0], font=font(18, True), fill="#f4f8fc")
    status = "有接触" if touching else "无接触"
    draw.text((354, 144), f"MuJoCo {status}", font=font(14), anchor="rb",
              fill="#ffbf87" if touching else "#9cb4c4")
    draw.line((20, 158, 354, 158), fill="#304655")

    draw.text((20, 172), "局部点云", font=font(15, True), fill="#dbe9f3")
    draw.text((354, 189), f"位移放大 {DISPLAY_GAIN} 倍", font=font(13), anchor="rb", fill="#b1c5d4")
    panel.paste(closeup(demo, route, indented, shifted, strength), (20, 201))

    palm_color = "#a9532d" if counts[0] else "#233948"
    draw.rounded_rectangle((20, 421, 354, 456), radius=8, fill=palm_color)
    draw.text((34, 439), "掌部", font=font(16, True), anchor="lm", fill="#eff7fa")
    draw.text((338, 439), f"{counts[0]} 压入", font=font(15), anchor="rm", fill="#f4f8fc")

    sections = ("根节", "近节", "远节", "指尖")
    for col, name in enumerate(sections):
        draw.text((110 + col * 68, 475), name, font=font(13), anchor="mm", fill="#a9bfcd")
    for row, name in enumerate(("食指", "中指", "无名指", "拇指")):
        top = 489 + row * 35
        draw.text((21, top + 15), name, font=font(15, True), anchor="lm", fill="#e6eff4")
        for col in range(4):
            link = 1 + row * 4 + col
            left = 80 + col * 68
            fill = "#a9532d" if counts[link] else "#20616a" if response[link] else "#233948"
            draw.rounded_rectangle((left, top, left + 61, top + 27), radius=6, fill=fill)
            draw.text((left + 30, top + 14), str(counts[link]), font=font(14, True),
                      anchor="mm", fill="#eef6fa" if counts[link] else "#9bb3c2")

    for x, color, label in ((20, "#ff9b4f", "压入点"), (122, "#61d6cc", "传播点"), (224, "#78949e", "参考点")):
        draw.ellipse((x, 636, x + 9, 645), fill=color)
        draw.text((x + 16, 630), label, font=font(13), fill="#e4eef5")
    draw.line((20, 657, 354, 657), fill="#304655")
    draw.text((20, 668), "空格键暂停或继续  ·  拖动旋转视角", font=font(13), fill="#a1b5c5")
    return panel


def save_snapshot(demo, path):
    route, phase = demo.step(0)
    camera, option = mujoco.MjvCamera(), mujoco.MjvOption()
    set_camera(camera, demo.targets[route])
    option.sitegroup[:] = 0
    indented, shifted, strength = demo.sample()
    max_geom = len(demo.display_points) + demo.model.ngeom + 150
    with mujoco.Renderer(demo.model, width=1320, height=780, max_geom=max_geom) as renderer:
        renderer.update_scene(demo.data, camera=camera, scene_option=option)
        cloud = DeformationCloud(renderer.scene, demo.display_points)
        cloud.update(indented, shifted, strength)
        image = Image.fromarray(renderer.render())
    panel = make_panel(demo, route, phase, indented, shifted, strength)
    image.paste(panel, (image.width - panel.width - MARGIN, MARGIN))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def show(demo):
    paused = False

    def on_key(key):
        nonlocal paused
        if key == 32:
            paused = not paused

    steps_per_frame = 22
    frame_interval = steps_per_frame * demo.model.opt.timestep
    with mujoco.viewer.launch_passive(demo.model, demo.data, key_callback=on_key,
                                      show_left_ui=False, show_right_ui=False) as viewer:
        with viewer.lock():
            cloud = DeformationCloud(viewer.user_scn, demo.display_points)
        previous_route = -1
        start = time.perf_counter()
        while viewer.is_running():
            tick = time.perf_counter()
            steps = 0 if paused or tick - start < 2.2 else steps_per_frame
            route, phase = demo.step(steps)
            indented, shifted, strength = demo.sample()
            with viewer.lock():
                if route != previous_route:
                    set_camera(viewer.cam, demo.targets[route])
                    viewer.opt.sitegroup[:] = 0
                cloud.update(indented, shifted, strength)
            viewport = viewer.viewport
            scale = min(1.0, (viewport.height - 2 * MARGIN) / PANEL_HEIGHT,
                        0.40 * viewport.width / PANEL_WIDTH) if viewport else 0
            if scale >= 0.52:
                panel = make_panel(demo, route, phase, indented, shifted, strength)
                if scale < 1:
                    panel = panel.resize((int(PANEL_WIDTH * scale), int(PANEL_HEIGHT * scale)),
                                         Image.Resampling.BILINEAR)
                rect = mujoco.MjrRect(viewport.width - panel.width - MARGIN,
                                      viewport.height - panel.height - MARGIN, panel.width, panel.height)
                viewer.set_images((rect, np.asarray(panel)))
            else:
                viewer.clear_images()
            viewer.sync()
            if previous_route < 0:
                print(f"LEAP 3D deformation viewer running with {len(indented):,} tactile nodes.", flush=True)
            previous_route = route
            time.sleep(max(0, frame_interval - (time.perf_counter() - tick)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, help="Save the initial whole-hand contact frame as a PNG.")
    args = parser.parse_args()
    demo = LeapDemo()
    if args.snapshot:
        save_snapshot(demo, args.snapshot)
    show(demo)


if __name__ == "__main__":
    main()
