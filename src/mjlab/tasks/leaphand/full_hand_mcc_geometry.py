"""Object-surface helpers used by the full-hand MCC demo."""

from __future__ import annotations

import numpy as np


def capsule_project(
    points_world: np.ndarray,
    center_world: np.ndarray,
    rotation_world_from_object: np.ndarray,
    radius: float,
    half_height: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Project points to a Z-axis capsule and return outward normals."""

    if radius <= 0.0 or half_height < 0.0:
        raise ValueError("Capsule radius must be positive and half-height non-negative")
    points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    center = np.asarray(center_world, dtype=np.float64).reshape(3)
    rotation = np.asarray(rotation_world_from_object, dtype=np.float64).reshape(3, 3)
    local = (rotation.T @ (points - center).T).T
    axis_z = np.clip(local[:, 2], -half_height, half_height)
    axis_points = np.zeros_like(local)
    axis_points[:, 2] = axis_z
    radial = local - axis_points
    radial_norm = np.linalg.norm(radial, axis=1, keepdims=True)
    fallback = np.zeros_like(radial)
    fallback[:, 0] = 1.0
    normals_local = np.where(
        radial_norm > 1.0e-9,
        radial / np.maximum(radial_norm, 1.0e-9),
        fallback,
    )
    surface_local = axis_points + radius * normals_local
    surface_world = center + (rotation @ surface_local.T).T
    normals_world = (rotation @ normals_local.T).T
    return surface_world.astype(np.float32), normals_world.astype(np.float32)


def rotate_about_capsule_axis(
    points_world: np.ndarray,
    center_world: np.ndarray,
    rotation_world_from_object: np.ndarray,
    angle: float,
) -> np.ndarray:
    """Move points tangentially around the capsule's local Z axis."""

    points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    center = np.asarray(center_world, dtype=np.float64).reshape(3)
    rotation = np.asarray(rotation_world_from_object, dtype=np.float64).reshape(3, 3)
    local = (rotation.T @ (points - center).T).T
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation_z = np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
    )
    moved_local = (rotation_z @ local.T).T
    return center + (rotation @ moved_local.T).T
