"""Causal local-surface Gaussian-process features for fingertip histories.

Each fingertip gets an independent height-field GP in the tangent frame of
its latest reliable contact.  Queries use a deterministic equal-area disk so
that the representation is reproducible in training and deployment.  The GP
uses past contacts only; future contacts are never part of the feature.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


GP_POINT_FEATURE_DIM = 10


@dataclass(frozen=True)
class GPManifoldConfig:
    history_steps: int = 16
    query_count: int = 8
    query_radius: float = 0.006
    length_scale: float = 0.008
    signal_std: float = 0.004
    noise_std: float = 0.0005

    def validate(self) -> None:
        if self.history_steps <= 0 or self.query_count <= 0:
            raise ValueError("history_steps and query_count must be positive")
        for name in ("query_radius", "length_scale", "signal_std", "noise_std"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")


def equal_area_disk_queries(count: int, radius: float) -> np.ndarray:
    """Return deterministic approximately uniform 2-D disk queries."""
    if count <= 0 or radius <= 0.0:
        raise ValueError("count and radius must be positive")
    if count == 1:
        return np.zeros((1, 2), dtype=np.float64)
    points = np.zeros((count, 2), dtype=np.float64)
    index = np.arange(count - 1, dtype=np.float64)
    radial = radius * np.sqrt((index + 0.5) / (count - 1))
    angle = index * (np.pi * (3.0 - np.sqrt(5.0)))
    points[1:, 0] = radial * np.cos(angle)
    points[1:, 1] = radial * np.sin(angle)
    return points


def tangent_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a stable right-handed tangent basis for one surface normal."""
    n = np.asarray(normal, dtype=np.float64)
    norm = float(np.linalg.norm(n))
    if norm < 1.0e-8:
        raise ValueError("Cannot build a tangent frame from a zero normal")
    n = n / norm
    reference = np.zeros(3, dtype=np.float64)
    reference[int(np.argmin(np.abs(n)))] = 1.0
    tangent_u = np.cross(reference, n)
    tangent_u /= np.linalg.norm(tangent_u)
    tangent_v = np.cross(n, tangent_u)
    return tangent_u, tangent_v, n


def _fit_and_query_height_gp(
    xy: np.ndarray,
    height: np.ndarray,
    query_xy: np.ndarray,
    config: GPManifoldConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return posterior mean/std, mean gradient and local support."""
    length_sq = config.length_scale**2
    signal_var = config.signal_std**2
    difference = xy[:, None, :] - xy[None, :, :]
    kernel = signal_var * np.exp(
        -0.5 * np.sum(difference * difference, axis=-1) / length_sq
    )
    kernel.flat[:: len(xy) + 1] += config.noise_std**2 + 1.0e-10
    alpha = np.linalg.solve(kernel, height)

    query_difference = query_xy[:, None, :] - xy[None, :, :]
    cross_kernel = signal_var * np.exp(
        -0.5 * np.sum(query_difference * query_difference, axis=-1) / length_sq
    )
    mean = cross_kernel @ alpha
    solved = np.linalg.solve(kernel, cross_kernel.T)
    variance = np.maximum(signal_var - np.sum(cross_kernel * solved.T, axis=1), 0.0)
    std = np.sqrt(variance + config.noise_std**2)
    # d k(q, x) / d q = k(q, x) * (x - q) / l^2
    gradient = np.einsum(
        "qn,qnd,n->qd",
        cross_kernel,
        -query_difference / length_sq,
        alpha,
    )
    support = np.max(cross_kernel / signal_var, axis=1)
    return mean, std, gradient, support


def local_gp_point_features(
    contact_positions: np.ndarray,
    contact_normals: np.ndarray,
    contact_mask: np.ndarray,
    config: GPManifoldConfig = GPManifoldConfig(),
) -> np.ndarray:
    """Create one fingertip's GP point set in a common coordinate frame.

    Inputs are a causal history ``[H, 3]`` expressed in the *current* palm
    frame.  Output point features are

    ``[position(3), normal(3), mu, delta, support, valid]``.
    """
    config.validate()
    positions = np.asarray(contact_positions, dtype=np.float64)
    normals = np.asarray(contact_normals, dtype=np.float64)
    valid = np.asarray(contact_mask, dtype=bool).reshape(-1)
    if positions.shape != normals.shape or positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("contact positions/normals must both have shape [H, 3]")
    if len(valid) != len(positions):
        raise ValueError("contact_mask length does not match contact history")
    output = np.zeros((config.query_count, GP_POINT_FEATURE_DIM), dtype=np.float32)
    indices = np.flatnonzero(valid)
    if not len(indices):
        return output

    anchor_index = int(indices[-1])
    anchor = positions[anchor_index]
    tangent_u, tangent_v, normal = tangent_basis(normals[anchor_index])
    relative = positions[indices] - anchor
    xy = np.stack((relative @ tangent_u, relative @ tangent_v), axis=-1)
    height = relative @ normal
    query_xy = equal_area_disk_queries(config.query_count, config.query_radius)
    mean, std, gradient, support = _fit_and_query_height_gp(
        xy, height, query_xy, config
    )
    query_position = (
        anchor[None]
        + query_xy[:, :1] * tangent_u[None]
        + query_xy[:, 1:] * tangent_v[None]
        + mean[:, None] * normal[None]
    )
    local_normal = np.concatenate((-gradient, np.ones((len(gradient), 1))), axis=1)
    local_normal /= np.maximum(np.linalg.norm(local_normal, axis=1, keepdims=True), 1.0e-8)
    query_normal = (
        local_normal[:, :1] * tangent_u[None]
        + local_normal[:, 1:2] * tangent_v[None]
        + local_normal[:, 2:] * normal[None]
    )
    output[:, :3] = query_position.astype(np.float32)
    output[:, 3:6] = query_normal.astype(np.float32)
    output[:, 6] = mean.astype(np.float32)
    output[:, 7] = std.astype(np.float32)
    output[:, 8] = support.astype(np.float32)
    output[:, 9] = 1.0
    return output
