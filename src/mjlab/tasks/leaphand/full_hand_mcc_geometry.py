"""Object-surface helpers used by the full-hand MCC demo."""

from __future__ import annotations

from functools import lru_cache

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


def capsule_meridian_coordinates(
    points_world: np.ndarray,
    center_world: np.ndarray,
    rotation_world_from_object: np.ndarray,
    radius: float,
    half_height: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return longitudinal arc length and azimuth for capsule surface points.

    Arc length is measured from the lower pole, continues over the lower
    hemisphere and cylinder, and ends at the upper pole.  Inputs are projected
    first so small numerical or contact-site offsets do not corrupt the
    parameterization.
    """

    surface_world, _ = capsule_project(
        points_world,
        center_world,
        rotation_world_from_object,
        radius,
        half_height,
    )
    center = np.asarray(center_world, dtype=np.float64).reshape(3)
    rotation = np.asarray(rotation_world_from_object, dtype=np.float64).reshape(3, 3)
    local = (rotation.T @ (surface_world - center).T).T
    radial = np.linalg.norm(local[:, :2], axis=1)
    azimuth = np.arctan2(local[:, 1], local[:, 0])
    lower_join = 0.5 * np.pi * radius
    upper_join = lower_join + 2.0 * half_height

    arc = np.empty(local.shape[0], dtype=np.float64)
    lower = local[:, 2] < -half_height
    upper = local[:, 2] > half_height
    middle = ~(lower | upper)
    arc[lower] = radius * np.arctan2(
        radial[lower],
        np.maximum(-(local[lower, 2] + half_height), 0.0),
    )
    arc[middle] = lower_join + local[middle, 2] + half_height
    arc[upper] = upper_join + radius * np.arctan2(
        local[upper, 2] - half_height,
        radial[upper],
    )
    return arc, azimuth


def capsule_meridian_targets(
    arc_length: np.ndarray,
    azimuth: np.ndarray,
    center_world: np.ndarray,
    rotation_world_from_object: np.ndarray,
    radius: float,
    half_height: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate capsule surface points, normals, and contact frames.

    The returned frame has columns ``(normal, azimuth tangent, meridian
    tangent)``.  It transports the calibrated site-to-contact offset while a
    contact moves from one end of the capsule to the other.
    """

    arc = np.asarray(arc_length, dtype=np.float64).reshape(-1)
    phi = np.asarray(azimuth, dtype=np.float64).reshape(-1)
    if arc.shape != phi.shape:
        raise ValueError("arc_length and azimuth must have the same shape")
    total_length = np.pi * radius + 2.0 * half_height
    if np.any(arc < 0.0) or np.any(arc > total_length):
        raise ValueError(
            f"arc_length must be within [0, {total_length:.6f}]"
        )

    lower_join = 0.5 * np.pi * radius
    upper_join = lower_join + 2.0 * half_height
    er = np.stack((np.cos(phi), np.sin(phi), np.zeros_like(phi)), axis=1)
    ephi = np.stack((-np.sin(phi), np.cos(phi), np.zeros_like(phi)), axis=1)
    ez = np.zeros_like(er)
    ez[:, 2] = 1.0

    point_local = np.zeros_like(er)
    normal_local = np.zeros_like(er)
    tangent_local = np.zeros_like(er)
    lower = arc < lower_join
    upper = arc > upper_join
    middle = ~(lower | upper)

    alpha = arc[lower] / radius
    point_local[lower] = (
        radius * np.sin(alpha)[:, None] * er[lower]
        + (-half_height - radius * np.cos(alpha))[:, None] * ez[lower]
    )
    normal_local[lower] = (
        np.sin(alpha)[:, None] * er[lower]
        - np.cos(alpha)[:, None] * ez[lower]
    )
    tangent_local[lower] = (
        np.cos(alpha)[:, None] * er[lower]
        + np.sin(alpha)[:, None] * ez[lower]
    )

    point_local[middle] = (
        radius * er[middle]
        + (-half_height + arc[middle] - lower_join)[:, None] * ez[middle]
    )
    normal_local[middle] = er[middle]
    tangent_local[middle] = ez[middle]

    beta = (arc[upper] - upper_join) / radius
    point_local[upper] = (
        radius * np.cos(beta)[:, None] * er[upper]
        + (half_height + radius * np.sin(beta))[:, None] * ez[upper]
    )
    normal_local[upper] = (
        np.cos(beta)[:, None] * er[upper]
        + np.sin(beta)[:, None] * ez[upper]
    )
    tangent_local[upper] = (
        -np.sin(beta)[:, None] * er[upper]
        + np.cos(beta)[:, None] * ez[upper]
    )

    rotation = np.asarray(rotation_world_from_object, dtype=np.float64).reshape(3, 3)
    center = np.asarray(center_world, dtype=np.float64).reshape(3)
    points_world = center + (rotation @ point_local.T).T
    normals_world = (rotation @ normal_local.T).T
    azimuth_world = (rotation @ ephi.T).T
    meridian_world = (rotation @ tangent_local.T).T
    frames_world = np.stack(
        (normals_world, azimuth_world, meridian_world), axis=-1
    )
    return (
        points_world.astype(np.float32),
        normals_world.astype(np.float32),
        frames_world.astype(np.float32),
    )


def capsule_meridian_curvature(
    arc_length: np.ndarray,
    radius: float,
    half_height: float,
) -> np.ndarray:
    """Return meridian curvature (1/m): zero on the cylinder, 1/r on caps."""

    arc = np.asarray(arc_length, dtype=np.float64)
    lower_join = 0.5 * np.pi * radius
    upper_join = lower_join + 2.0 * half_height
    return np.where(
        (arc < lower_join) | (arc > upper_join),
        1.0 / radius,
        0.0,
    )


@lru_cache(maxsize=32)
def _ellipsoid_meridian_lookup(
    radial_radius: float,
    axial_radius: float,
    sample_count: int = 16385,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a dense lower-pole-to-upper-pole arc-length lookup."""

    if radial_radius <= 0.0 or axial_radius <= 0.0:
        raise ValueError("Ellipsoid radii must be positive")
    theta = np.linspace(0.0, np.pi, sample_count, dtype=np.float64)
    speed = np.sqrt(
        radial_radius**2 * np.cos(theta) ** 2
        + axial_radius**2 * np.sin(theta) ** 2
    )
    delta = np.diff(theta)
    arc = np.zeros_like(theta)
    arc[1:] = np.cumsum(0.5 * (speed[:-1] + speed[1:]) * delta)
    theta.setflags(write=False)
    arc.setflags(write=False)
    return theta, arc


def ellipsoid_meridian_total_length(
    radial_radius: float,
    axial_radius: float,
) -> float:
    """Return the pole-to-pole meridian length of a spheroid."""

    _, arc = _ellipsoid_meridian_lookup(
        float(radial_radius),
        float(axial_radius),
    )
    return float(arc[-1])


def ellipsoid_project(
    points_world: np.ndarray,
    center_world: np.ndarray,
    rotation_world_from_object: np.ndarray,
    radial_radius: float,
    axial_radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Project points to a Z-axis spheroid and return its true normals.

    The projection preserves azimuth and the angle in normalized ellipsoid
    coordinates.  This is single-valued everywhere except the poles and is
    consistent with the meridian planner below.
    """

    if radial_radius <= 0.0 or axial_radius <= 0.0:
        raise ValueError("Ellipsoid radii must be positive")
    points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    center = np.asarray(center_world, dtype=np.float64).reshape(3)
    rotation = np.asarray(rotation_world_from_object, dtype=np.float64).reshape(3, 3)
    local = (rotation.T @ (points - center).T).T
    radial = np.linalg.norm(local[:, :2], axis=1)
    azimuth = np.arctan2(local[:, 1], local[:, 0])
    theta = np.arctan2(
        radial / radial_radius,
        -local[:, 2] / axial_radius,
    )
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    er = np.stack(
        (np.cos(azimuth), np.sin(azimuth), np.zeros_like(azimuth)),
        axis=1,
    )
    ez = np.zeros_like(er)
    ez[:, 2] = 1.0
    surface_local = (
        radial_radius * sin_theta[:, None] * er
        - axial_radius * cos_theta[:, None] * ez
    )
    normal_local = (
        (sin_theta / radial_radius)[:, None] * er
        - (cos_theta / axial_radius)[:, None] * ez
    )
    normal_local /= np.maximum(
        np.linalg.norm(normal_local, axis=1, keepdims=True),
        1.0e-12,
    )
    surface_world = center + (rotation @ surface_local.T).T
    normals_world = (rotation @ normal_local.T).T
    return surface_world.astype(np.float32), normals_world.astype(np.float32)


def ellipsoid_meridian_coordinates(
    points_world: np.ndarray,
    center_world: np.ndarray,
    rotation_world_from_object: np.ndarray,
    radial_radius: float,
    axial_radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return lower-pole arc length and azimuth on a Z-axis spheroid."""

    surface_world, _ = ellipsoid_project(
        points_world,
        center_world,
        rotation_world_from_object,
        radial_radius,
        axial_radius,
    )
    center = np.asarray(center_world, dtype=np.float64).reshape(3)
    rotation = np.asarray(rotation_world_from_object, dtype=np.float64).reshape(3, 3)
    local = (rotation.T @ (surface_world - center).T).T
    radial = np.linalg.norm(local[:, :2], axis=1)
    theta = np.arctan2(
        radial / radial_radius,
        -local[:, 2] / axial_radius,
    )
    lookup_theta, lookup_arc = _ellipsoid_meridian_lookup(
        float(radial_radius),
        float(axial_radius),
    )
    arc = np.interp(theta, lookup_theta, lookup_arc)
    azimuth = np.arctan2(local[:, 1], local[:, 0])
    return arc, azimuth


def ellipsoid_meridian_targets(
    arc_length: np.ndarray,
    azimuth: np.ndarray,
    center_world: np.ndarray,
    rotation_world_from_object: np.ndarray,
    radial_radius: float,
    axial_radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate spheroid points, normals, and transported contact frames."""

    arc = np.asarray(arc_length, dtype=np.float64).reshape(-1)
    phi = np.asarray(azimuth, dtype=np.float64).reshape(-1)
    if arc.shape != phi.shape:
        raise ValueError("arc_length and azimuth must have the same shape")
    lookup_theta, lookup_arc = _ellipsoid_meridian_lookup(
        float(radial_radius),
        float(axial_radius),
    )
    total_length = float(lookup_arc[-1])
    if np.any(arc < 0.0) or np.any(arc > total_length):
        raise ValueError(
            f"arc_length must be within [0, {total_length:.6f}]"
        )
    theta = np.interp(arc, lookup_arc, lookup_theta)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    er = np.stack((np.cos(phi), np.sin(phi), np.zeros_like(phi)), axis=1)
    ephi = np.stack((-np.sin(phi), np.cos(phi), np.zeros_like(phi)), axis=1)
    ez = np.zeros_like(er)
    ez[:, 2] = 1.0
    point_local = (
        radial_radius * sin_theta[:, None] * er
        - axial_radius * cos_theta[:, None] * ez
    )
    normal_local = (
        (sin_theta / radial_radius)[:, None] * er
        - (cos_theta / axial_radius)[:, None] * ez
    )
    normal_local /= np.maximum(
        np.linalg.norm(normal_local, axis=1, keepdims=True),
        1.0e-12,
    )
    tangent_local = (
        radial_radius * cos_theta[:, None] * er
        + axial_radius * sin_theta[:, None] * ez
    )
    tangent_local /= np.maximum(
        np.linalg.norm(tangent_local, axis=1, keepdims=True),
        1.0e-12,
    )
    rotation = np.asarray(rotation_world_from_object, dtype=np.float64).reshape(3, 3)
    center = np.asarray(center_world, dtype=np.float64).reshape(3)
    points_world = center + (rotation @ point_local.T).T
    normals_world = (rotation @ normal_local.T).T
    azimuth_world = (rotation @ ephi.T).T
    meridian_world = (rotation @ tangent_local.T).T
    frames_world = np.stack(
        (normals_world, azimuth_world, meridian_world),
        axis=-1,
    )
    return (
        points_world.astype(np.float32),
        normals_world.astype(np.float32),
        frames_world.astype(np.float32),
    )


def ellipsoid_meridian_curvature(
    arc_length: np.ndarray,
    radial_radius: float,
    axial_radius: float,
) -> np.ndarray:
    """Return ellipse meridian curvature (1/m) at each arc position."""

    arc = np.asarray(arc_length, dtype=np.float64)
    lookup_theta, lookup_arc = _ellipsoid_meridian_lookup(
        float(radial_radius),
        float(axial_radius),
    )
    theta = np.interp(arc, lookup_arc, lookup_theta)
    speed_squared = (
        radial_radius**2 * np.cos(theta) ** 2
        + axial_radius**2 * np.sin(theta) ** 2
    )
    return (
        radial_radius
        * axial_radius
        / np.maximum(speed_squared, 1.0e-18) ** 1.5
    )
