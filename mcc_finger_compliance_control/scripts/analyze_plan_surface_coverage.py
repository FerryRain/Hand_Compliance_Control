"""Summarize geometric diversity of sparse fingertip-contact planners.

The report deliberately does not score force magnitude.  These plans are
teachers for hand qpos; force is only used by the physical replay to establish
contact.  Coverage is measured from object-frame material points and normals.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np


REGION_NAMES = ("bottom", "lower_body", "upper_body", "shoulder_neck", "cap")
REGION_EDGES = np.asarray((0.0, 0.10, 0.60, 0.85, 0.93, 1.000001))


def _mesh_vertices(path: Path) -> np.ndarray:
    vertices: list[list[float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("v "):
                vertices.append([float(value) for value in line.split()[1:4]])
    if not vertices:
        raise ValueError(f"No OBJ vertices found in {path}")
    return np.asarray(vertices, dtype=np.float64)


def _normal_angle_degrees(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1.0e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1.0e-12)
    return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=-1), -1.0, 1.0)))


def analyze(path: Path, z_min: float, z_max: float) -> dict[str, object]:
    with h5py.File(path, "r") as file:
        required = (
            "grasp_keyframe_contact_point_object",
            "grasp_keyframe_normal_object",
            "grasp_keyframe_valid",
        )
        missing = [name for name in required if name not in file]
        if missing:
            raise KeyError(f"{path} lacks sparse keyframes: {missing}")
        valid = np.asarray(file["grasp_keyframe_valid"], dtype=bool).reshape(-1)
        points = np.asarray(
            file["grasp_keyframe_contact_point_object"], dtype=np.float64
        )[valid]
        normals = np.asarray(
            file["grasp_keyframe_normal_object"], dtype=np.float64
        )[valid]
        plane_normal = np.asarray(
            file.attrs.get("planner_ellipse_plane_normal_object", (np.nan,) * 3),
            dtype=np.float64,
        )
        seed = int(file.attrs.get("planner_seed", -1))
        azimuth = float(file.attrs.get("planner_ellipse_azimuth_deg", np.nan))

    if len(points) < 2:
        raise ValueError(f"{path} has fewer than two valid sparse keyframes")
    z_fraction = np.clip((points[..., 2] - z_min) / (z_max - z_min), 0.0, 1.0)
    region_index = np.clip(
        np.searchsorted(REGION_EDGES, z_fraction, side="right") - 1,
        0,
        len(REGION_NAMES) - 1,
    )
    region_fraction = np.asarray(
        [(region_index == index).mean() for index in range(len(REGION_NAMES))]
    )
    step_distance = np.linalg.norm(np.diff(points, axis=0), axis=-1)
    normal_step = _normal_angle_degrees(normals[1:], normals[:-1])
    path_per_tip = step_distance.sum(axis=0)
    normal_change_per_tip = normal_step.sum(axis=0)
    occupied = [
        REGION_NAMES[index]
        for index, fraction in enumerate(region_fraction)
        if fraction >= 0.02
    ]
    return {
        "plan": path.name,
        "seed": seed,
        "azimuth_deg": azimuth,
        "valid_ratio": float(valid.mean()),
        "valid_knots": int(valid.sum()),
        "z_min_m": float(points[..., 2].min()),
        "z_max_m": float(points[..., 2].max()),
        "mean_tip_path_m": float(path_per_tip.mean()),
        "min_tip_path_m": float(path_per_tip.min()),
        "mean_normal_change_deg": float(normal_change_per_tip.mean()),
        "normal_step_p95_deg": float(np.percentile(normal_step, 95)),
        "occupied_regions": "+".join(occupied),
        **{
            f"region_{name}_ratio": float(region_fraction[index])
            for index, name in enumerate(REGION_NAMES)
        },
        "plane_normal_x": float(plane_normal[0]),
        "plane_normal_y": float(plane_normal[1]),
        "plane_normal_z": float(plane_normal[2]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("plans", nargs="+", type=Path)
    parser.add_argument(
        "--mesh",
        type=Path,
        default=Path(
            "assets_external/ycb/collision/006_mustard_bottle/"
            "vhacd_256_scaled2p8_objstage/visual_scaled.obj"
        ),
    )
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()

    vertices = _mesh_vertices(args.mesh)
    z_min, z_max = vertices[:, 2].min(), vertices[:, 2].max()
    rows = [analyze(path, z_min, z_max) for path in args.plans]
    print(
        "plan valid path_mm normal_deg regions "
        "[bottom lower_body upper_body shoulder_neck cap]"
    )
    for row in rows:
        ratios = " ".join(
            f"{100.0 * float(row[f'region_{name}_ratio']):4.0f}%"
            for name in REGION_NAMES
        )
        print(
            f"{row['plan']} {100.0 * float(row['valid_ratio']):5.1f}% "
            f"{1000.0 * float(row['mean_tip_path_m']):7.1f} "
            f"{float(row['mean_normal_change_deg']):8.1f} "
            f"{row['occupied_regions']} [{ratios}]"
        )
    normals = np.asarray(
        [
            (row["plane_normal_x"], row["plane_normal_y"], row["plane_normal_z"])
            for row in rows
        ],
        dtype=np.float64,
    )
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-12)
    if len(normals) > 1:
        pairwise = np.degrees(
            np.arccos(np.clip(normals @ normals.T, -1.0, 1.0))
        )
        upper = pairwise[np.triu_indices(len(normals), 1)]
        print(
            "plane-normal pairwise separation: "
            f"min={upper.min():.1f}deg median={np.median(upper):.1f}deg "
            f"max={upper.max():.1f}deg"
        )
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[SUCCESS] wrote {args.csv}")


if __name__ == "__main__":
    main()
