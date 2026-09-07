"""Display several palm plans together in the fixed-object inverse space.

The object is placed at the origin and ``palm_pose_object`` is rendered as a
coloured polyline.  This viewer intentionally runs no controller or physics;
it is only a geometric check of the planner output.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py
import mujoco
import mujoco.viewer
import numpy as np

from object_catalog import add_object_body, load_object_config


HAND_XML = Path(
    "src/mjlab/asset_zoo/robots/xarm6_leap_hand/leap_hand_tactile.xml"
)
HAND_JOINT_NAMES = (
    "1", "0", "2", "3",
    "5", "4", "6", "7",
    "9", "8", "10", "11",
    "12", "13", "14", "15",
)

COLORS = (
    (0.95, 0.12, 0.10, 1.0),  # red
    (0.10, 0.85, 0.25, 1.0),  # green
    (0.10, 0.35, 1.00, 1.0),  # blue
    (0.85, 0.15, 0.95, 1.0),  # magenta
    (1.00, 0.65, 0.05, 1.0),  # orange
    (0.05, 0.85, 0.90, 1.0),  # cyan
    (1.00, 0.92, 0.10, 1.0),  # yellow
    (0.52, 0.20, 1.00, 1.0),  # violet
    (0.95, 0.95, 0.95, 1.0),  # white
)
COLOR_NAMES = (
    "red", "green", "blue", "magenta", "orange", "cyan", "yellow",
    "violet", "white",
)

# The first ten points follow the collision perimeter of ``palm_lower``.
# Rendering this at each trajectory start shows both the initial palm
# position and its orientation without loading four complete robot models.
PALM_OUTLINE_LOCAL = np.asarray(
    (
        (-0.100095, -0.027242, -0.0347224),
        (-0.100095, -0.054761, -0.0347224),
        (-0.093899, -0.080485, -0.0347224),
        (-0.071635, -0.093574, -0.0347224),
        (-0.044283, -0.096601, -0.0347224),
        (-0.036095, -0.078225, -0.0347224),
        (-0.036095, 0.004332, -0.0347224),
        (-0.042189, 0.025758, -0.0347224),
        (-0.065295, 0.015398, -0.0347224),
        (-0.082695, -0.005922, -0.0347224),
    ),
    dtype=np.float64,
)
PALM_CONTROL_POINT_LOCAL = np.asarray(
    (-0.0559703, -0.04142053, -0.0340008), dtype=np.float64
)


def _load_plans(
    paths: list[Path],
) -> tuple[
    str, float, list[np.ndarray], list[np.ndarray], list[np.ndarray]
]:
    object_id: str | None = None
    object_scale: float | None = None
    trajectories: list[np.ndarray] = []
    initial_poses: list[np.ndarray] = []
    initial_q: list[np.ndarray] = []
    for path in paths:
        with h5py.File(path, "r") as file:
            current_id = str(file.attrs["object_id"])
            current_scale = float(file.attrs.get("object_scale", 1.0))
            poses = np.asarray(file["palm_pose_object"][:, 0], dtype=np.float64)
            control_local = np.asarray(
                file.attrs.get(
                    "planner_palm_control_point_local",
                    PALM_CONTROL_POINT_LOCAL,
                ),
                dtype=np.float64,
            )
            rotations = np.stack(
                [_quat_wxyz_to_matrix(quaternion) for quaternion in poses[:, 3:7]],
                axis=0,
            )
            positions = poses[:, :3] + np.einsum(
                "nij,j->ni", rotations, control_local
            )
            q0 = np.asarray(file["q_hand"][0, 0], dtype=np.float64)
        if object_id is None:
            object_id = current_id
            object_scale = current_scale
        elif current_id != object_id or not np.isclose(current_scale, object_scale):
            raise ValueError(
                "All plans must use the same object and scale; "
                f"expected {object_id}@{object_scale}, got {current_id}@{current_scale}"
            )
        if len(positions) < 2:
            raise ValueError(f"Plan has fewer than two poses: {path}")
        trajectories.append(positions)
        initial_poses.append(poses[0])
        initial_q.append(q0)
    assert object_id is not None and object_scale is not None
    return object_id, object_scale, trajectories, initial_poses, initial_q


def _quat_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    quaternion_norm = max(float(np.linalg.norm((w, x, y, z))), 1.0e-12)
    w, x, y, z = np.asarray((w, x, y, z)) / quaternion_norm
    return np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def _fixed_hand_spec(color: tuple[float, float, float, float]) -> mujoco.MjSpec:
    hand = mujoco.MjSpec.from_file(str(HAND_XML))
    # Attached specs lose their source directory, so mesh paths must be made
    # absolute before merging them into the inverse viewer scene.
    for mesh in hand.meshes:
        if mesh.file:
            mesh.file = str((HAND_XML.parent / mesh.file).resolve())
    for key in list(hand.keys):
        hand.delete(key)
    palm_joint = hand.joint("palm_base")
    if palm_joint is not None:
        hand.delete(palm_joint)
    palm = hand.body("palm_lower")
    if palm is None:
        raise ValueError(f"palm_lower is missing from {HAND_XML}")
    palm.pos[:] = (0.0, 0.0, 0.0)
    palm.quat[:] = (1.0, 0.0, 0.0, 0.0)
    palm.alt.type = mujoco.mjtOrientation.mjORIENTATION_QUAT
    tint = np.asarray(color[:3], dtype=np.float64)
    for geom in hand.geoms:
        original = np.asarray(geom.rgba[:3], dtype=np.float64)
        geom.rgba[:3] = 0.65 * original + 0.35 * tint
        geom.rgba[3] = 0.72
        geom.contype = 0
        geom.conaffinity = 0
    return hand


def _scene_spec(
    object_id: str,
    object_scale: float,
    initial_poses: list[np.ndarray],
    show_hand_models: bool,
) -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    spec.option.gravity[:] = (0.0, 0.0, 0.0)
    spec.worldbody.add_light(
        name="key_light",
        pos=(0.4, -0.6, 1.0),
        dir=(-0.3, 0.4, -1.0),
        diffuse=(0.9, 0.9, 0.9),
    )
    add_object_body(
        spec,
        load_object_config(object_id),
        body_name="inverse_object",
        pos=(0.0, 0.0, 0.0),
        quat=(1.0, 0.0, 0.0, 0.0),
        mocap=False,
        scale=object_scale,
    )
    if show_hand_models:
        for index, pose in enumerate(initial_poses):
            frame = spec.worldbody.add_frame(name=f"hand_{index}_frame")
            frame.pos[:] = pose[:3]
            frame.quat[:] = pose[3:7]
            spec.attach(
                _fixed_hand_spec(COLORS[index % len(COLORS)]),
                prefix=f"hand_{index}/",
                frame=frame,
            )
    return spec


def _set_initial_hand_q(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    initial_q: list[np.ndarray],
) -> None:
    for hand_index, q_hand in enumerate(initial_q):
        if len(q_hand) != len(HAND_JOINT_NAMES):
            raise ValueError(
                f"Expected 16 hand joints, got {len(q_hand)} for hand {hand_index}"
            )
        for name, value in zip(HAND_JOINT_NAMES, q_hand, strict=True):
            full_name = f"hand_{hand_index}/{name}"
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, full_name
            )
            if joint_id < 0:
                raise ValueError(f"Attached hand joint not found: {full_name}")
            data.qpos[int(model.jnt_qposadr[joint_id])] = float(value)


def _add_cylinder(
    scene: mujoco.MjvScene,
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
    color: tuple[float, float, float, float],
) -> None:
    if scene.ngeom >= scene.maxgeom:
        raise RuntimeError(
            f"Viewer user scene is full ({scene.maxgeom} geoms); increase --stride"
        )
    geom = scene.geoms[scene.ngeom]
    scene.ngeom += 1
    geom.category = mujoco.mjtCatBit.mjCAT_DECOR
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_CYLINDER,
        np.zeros(3),
        np.zeros(3),
        np.zeros(9),
        np.asarray(color, dtype=np.float32),
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_CYLINDER,
        radius,
        start,
        end,
    )


def _add_sphere(
    scene: mujoco.MjvScene,
    center: np.ndarray,
    radius: float,
    color: tuple[float, float, float, float],
) -> None:
    if scene.ngeom >= scene.maxgeom:
        raise RuntimeError(
            f"Viewer user scene is full ({scene.maxgeom} geoms); increase --stride"
        )
    geom = scene.geoms[scene.ngeom]
    scene.ngeom += 1
    geom.category = mujoco.mjtCatBit.mjCAT_DECOR
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.asarray((radius, 0.0, 0.0)),
        center,
        np.eye(3).ravel(),
        np.asarray(color, dtype=np.float32),
    )


def _draw_paths(
    scene: mujoco.MjvScene,
    paths: list[np.ndarray],
    initial_poses: list[np.ndarray],
    stride: int,
    radius: float,
    show_initial_palms: bool,
) -> None:
    scene.ngeom = 0
    for index, path in enumerate(paths):
        color = COLORS[index % len(COLORS)]
        sample = path[::stride]
        if not np.allclose(sample[-1], path[-1]):
            sample = np.vstack((sample, path[-1]))
        for start, end in zip(sample[:-1], sample[1:], strict=True):
            _add_cylinder(scene, start, end, radius, color)
        # A larger start marker and smaller endpoint make direction visible.
        _add_sphere(scene, path[0], 2.5 * radius, color)
        _add_sphere(scene, path[-1], 1.6 * radius, color)
        if show_initial_palms:
            pose = initial_poses[index]
            rotation = _quat_wxyz_to_matrix(pose[3:7])
            outline = pose[:3] + (rotation @ PALM_OUTLINE_LOCAL.T).T
            closed_outline = np.vstack((outline, outline[0]))
            for start, end in zip(
                closed_outline[:-1], closed_outline[1:], strict=True
            ):
                _add_cylinder(scene, start, end, 1.25 * radius, color)
            # Palm -Z is the surface-facing normal used by the planner.
            control_center = pose[:3] + rotation @ PALM_CONTROL_POINT_LOCAL
            normal_end = control_center + rotation @ np.asarray(
                (0.0, 0.0, -0.055)
            )
            _add_cylinder(
                scene, control_center, normal_end, 1.7 * radius, color
            )
            _add_sphere(scene, normal_end, 2.1 * radius, color)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show coloured palm trajectories in the fixed-object inverse frame."
    )
    parser.add_argument("plans", nargs="+", type=Path)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--line-radius-m", type=float, default=0.0018)
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--camera-elevation", type=float, default=-18.0)
    parser.add_argument(
        "--no-initial-palms",
        action="store_true",
        help="Hide the coloured palm outlines and facing-normal markers.",
    )
    parser.add_argument(
        "--no-hand-models",
        action="store_true",
        help="Do not load the four fixed-base Leap Hand models at path starts.",
    )
    args = parser.parse_args()
    if args.stride < 1:
        raise ValueError("--stride must be at least one")
    if args.line_radius_m <= 0.0:
        raise ValueError("--line-radius-m must be positive")

    (
        object_id,
        object_scale,
        trajectories,
        initial_poses,
        initial_q,
    ) = _load_plans(args.plans)
    show_hand_models = not args.no_hand_models
    model = _scene_spec(
        object_id, object_scale, initial_poses, show_hand_models
    ).compile()
    data = mujoco.MjData(model)
    if show_hand_models:
        _set_initial_hand_q(model, data, initial_q)
    mujoco.mj_forward(model, data)

    all_points = np.concatenate(trajectories, axis=0)
    lower = all_points.min(axis=0)
    upper = all_points.max(axis=0)
    center = 0.5 * (lower + upper)
    span = max(float(np.linalg.norm(upper - lower)), 0.25)
    print(f"[INFO] inverse frame: object={object_id}, scale={object_scale:.4f}")
    for index, (path, trajectory) in enumerate(zip(args.plans, trajectories, strict=True)):
        color_name = COLOR_NAMES[index % len(COLORS)]
        length = float(np.linalg.norm(np.diff(trajectory, axis=0), axis=1).sum())
        print(f"[PATH {index + 1}] {color_name:<7} length={length*1000:.1f} mm  {path}")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = center
        viewer.cam.distance = 1.25 * span
        viewer.cam.azimuth = args.camera_azimuth
        viewer.cam.elevation = args.camera_elevation
        _draw_paths(
            viewer.user_scn,
            trajectories,
            initial_poses,
            args.stride,
            args.line_radius_m,
            not args.no_initial_palms,
        )
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.02)


if __name__ == "__main__":
    main()
