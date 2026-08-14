"""Visualize the config-driven contact-object catalog in MuJoCo."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import mujoco.viewer
import numpy as np

from object_catalog import (
    add_object_body,
    list_object_ids,
    load_object_config,
    object_local_aabb,
)


def _gallery_spec(object_ids: list[str], columns: int) -> tuple[mujoco.MjSpec, list[str]]:
    spec = mujoco.MjSpec()
    spec.option.timestep = 0.01
    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 1080
    spec.worldbody.add_light(
        name="key_light",
        pos=(0.0, -1.0, 2.5),
        dir=(0.0, 0.4, -1.0),
        diffuse=(0.9, 0.9, 0.9),
    )
    floor = spec.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=(5.0, 5.0, 0.1),
        rgba=(0.12, 0.14, 0.18, 1.0),
    )
    floor.contype = 0
    floor.conaffinity = 0

    spacing = 0.55
    rows = int(math.ceil(len(object_ids) / columns))
    body_names: list[str] = []
    for index, object_id in enumerate(object_ids):
        config = load_object_config(object_id)
        row, column = divmod(index, columns)
        x = (column - 0.5 * (min(columns, len(object_ids)) - 1)) * spacing
        y = (0.5 * (rows - 1) - row) * spacing
        lower, upper = object_local_aabb(config)
        z = float(0.05 + 0.5 * (upper[2] - lower[2]) - 0.5 * (
            upper[2] + lower[2]
        ))
        body_name = f"gallery_{object_id}"
        add_object_body(
            spec,
            config,
            body_name=body_name,
            pos=(x, y, z),
            mocap=True,
        )
        body_names.append(body_name)
        print(
            f"[{index:02d}] {object_id:<18} family={config.family:<13} "
            f"size={(upper - lower).round(3).tolist()}m pos={[round(x, 3), round(y, 3), round(z, 3)]}"
        )
    return spec, body_names


def _camera(columns: int) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (0.0, 0.0, 0.12)
    camera.distance = max(1.4, 0.55 * max(columns, 3))
    camera.azimuth = 125.0
    camera.elevation = -22.0
    return camera


def main() -> None:
    available = list_object_ids()
    parser = argparse.ArgumentParser(
        description="Display one object or the complete contact-object gallery."
    )
    parser.add_argument(
        "--object",
        dest="objects",
        action="append",
        default=None,
        help="Object id to show; repeat the option for several objects. Default: all.",
    )
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument(
        "--viewer", choices=("native", "headless", "image"), default="native"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("mcc_finger_compliance_control/outputs/object_gallery.png"),
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--spin-speed-deg-s", type=float, default=12.0)
    parser.add_argument(
        "--duration-s",
        type=float,
        default=0.0,
        help="Automatically close after this duration; zero waits for the window.",
    )
    args = parser.parse_args()

    object_ids = list(available if args.objects is None else args.objects)
    unknown = sorted(set(object_ids) - set(available))
    if unknown:
        raise ValueError(f"Unknown objects {unknown}; available={list(available)}")
    if not object_ids:
        raise ValueError("At least one object must be selected")
    if args.columns <= 0:
        raise ValueError("--columns must be positive")

    spec, body_names = _gallery_spec(object_ids, args.columns)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    print(
        f"[INFO] compiled {len(object_ids)} objects, "
        f"{model.ngeom - 1} object geoms, viewer={args.viewer}"
    )
    if args.viewer == "headless":
        for _ in range(10):
            mujoco.mj_step(model, data)
        print("[INFO] headless geometry validation passed")
        return
    camera = _camera(args.columns)
    if args.viewer == "image":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with mujoco.Renderer(model, height=args.height, width=args.width) as renderer:
            renderer.update_scene(data, camera=camera)
            imageio.imwrite(args.output, renderer.render())
        print(f"[INFO] wrote gallery image: {args.output}")
        return

    mocap_ids = []
    initial_quats = []
    for body_name in body_names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        mocap_id = int(model.body_mocapid[body_id])
        if mocap_id < 0:
            raise RuntimeError(f"Gallery body {body_name!r} is not mocap")
        mocap_ids.append(mocap_id)
        initial_quats.append(data.mocap_quat[mocap_id].copy())

    start = time.monotonic()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = camera.lookat
        viewer.cam.distance = camera.distance
        viewer.cam.azimuth = camera.azimuth
        viewer.cam.elevation = camera.elevation
        while viewer.is_running():
            now = time.monotonic()
            elapsed = now - start
            if args.duration_s > 0.0 and elapsed >= args.duration_s:
                break
            angle = math.radians(args.spin_speed_deg_s) * elapsed
            spin = np.asarray(
                (math.cos(0.5 * angle), 0.0, 0.0, math.sin(0.5 * angle))
            )
            for mocap_id, initial in zip(
                mocap_ids, initial_quats, strict=True
            ):
                mujoco.mju_mulQuat(data.mocap_quat[mocap_id], spin, initial)
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(0.01)


if __name__ == "__main__":
    main()
