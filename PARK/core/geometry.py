"""Contact-supported zero crossings, exclusion-radius clustering and wrist geometry."""
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .common import dispersed, pack, rot, unit


@dataclass
class Surface:
    points: np.ndarray
    normals: np.ndarray
    variance: np.ndarray
    regions: np.ndarray


def spaced(points, priority, spacing, limit):
    order = np.argsort(-priority, kind="stable")
    chosen = []
    while len(order) and len(chosen) < limit:
        idx = order[0]
        chosen.append(idx)
        rest = order[1:]
        order = rest[np.square(points[rest] - points[idx]).sum(1) >= spacing**2]
    return np.asarray(chosen, dtype=int)


def uncertain_surface(gp, points, normals, focus, cfg):
    near = np.flatnonzero(np.linalg.norm(points - focus, axis=1) <= cfg.local_radius_m)
    # Sample both the local patch and a spatially distributed set of anchors.
    local = near[dispersed(points[near], max(1, cfg.support_sites//2))]
    ids = np.unique(np.r_[local, dispersed(points, cfg.support_sites)])
    p, n = points[ids], normals[ids]
    u = unit(np.cross(n, np.eye(3)[np.argmin(abs(n), axis=1)]))
    v = np.cross(n, u)
    angle = 2*np.pi*np.arange(cfg.azimuths)/cfg.azimuths
    rays = np.cos(angle)[None, :, None]*u[:, None] + np.sin(angle)[None, :, None]*v[:, None]
    tangent = p[:, None, None] + rays[:, :, None]*np.asarray(cfg.rings_m)[None, None, :, None]
    xyz = tangent.reshape(-1, 3)
    hints = np.broadcast_to(n[:, None, None], tangent.shape).reshape(-1, 3)
    offset = cfg.bracket_m*hints
    # A signed crossing brackets a zero on a tactile-supported normal ray.
    ends = gp.query(np.vstack((xyz-offset, xyz+offset)))[0]
    lower, upper = np.split(ends, 2)
    crossed = (lower < 0) & (upper > 0)
    if not crossed.any():
        return None
    location = (-lower[crossed]/(upper[crossed]-lower[crossed]))*2-1
    x = xyz[crossed] + location[:, None]*offset[crossed]
    hints = hints[crossed]
    mean, _, gradient = gp.query(x)
    slope = (gradient*hints).sum(1)
    correction = np.clip(mean/np.maximum(slope, .1), -cfg.bracket_m, cfg.bracket_m)
    x = x - correction[:, None]*hints
    mean, variance, gradient = gp.query(x)
    outward = unit(gradient)
    keep = ((abs(mean) < cfg.surface_band_m) & (variance > cfg.variance_m2)
            & ((outward*hints).sum(1) >= cfg.normal_agreement)
            & (np.linalg.norm(x - xyz[crossed], axis=1) <= cfg.bracket_m))
    if not keep.any():
        return None
    x, outward, variance = x[keep], outward[keep], variance[keep]
    keep = spaced(x, variance, cfg.surface_spacing_m, cfg.surface_count)
    x, outward, variance = x[keep], outward[keep], variance[keep]
    regions = spaced(x, variance, cfg.target_spacing_m, cfg.target_count)
    return Surface(x, outward, variance, regions)


def anchors(plate, pads, fraction):
    # The contact depth and lateral extent come from the current hand geometry.
    # All scoring arms share these whole-palm anchors, including uncertainty-only.
    center = .5*(plate + pads.mean(0))
    center[2] = np.median(pads[:, 2])
    half = np.ptp(pads[:, :2], axis=0)*fraction
    stencil = np.array([[0, 0], [-1, -1], [-1, 1], [1, -1], [1, 1]])
    return center + np.column_stack((stencil*half, np.zeros(len(stencil))))


def candidates(surface, actual, origin, references, cfg):
    frames, owners = [], []
    current = rot(actual)
    x_current = current.as_matrix()[:, 0]
    for j in surface.regions:
        normal = surface.normals[j]
        x = x_current - (x_current @ normal)*normal
        if np.linalg.norm(x) < 1e-6:
            x = np.cross(normal, np.eye(3)[abs(normal).argmin()])
        x = unit(x)
        face = Rotation.from_matrix(np.column_stack((x, np.cross(normal, x), normal)))
        for roll in cfg.roll_rad:
            attitude = Rotation.from_rotvec(roll*normal)*face
            if ((current.inv()*attitude).magnitude() > cfg.max_turn_rad
                    or (rot(origin).inv()*attitude).magnitude() > cfg.attitude_limit_rad):
                continue
            xyz = surface.points[j] - attitude.apply(references)
            valid = ((np.linalg.norm(xyz - actual[:3], axis=1) <= cfg.max_step_m)
                     & (np.linalg.norm(xyz - origin[:3], axis=1) <= cfg.workspace_radius_m))
            for pos in xyz[valid]:
                frames.append(pack(pos, attitude))
                owners.append(j)
    return np.array(frames).reshape(-1, 7), np.array(owners, int)


class Clearance:
    def __init__(self, points, normals, hull, actual, cfg):
        self.points, self.normals, self.hull, self.cfg = points, normals, hull, cfg
        self.tree = cKDTree(points)
        self.minimum = np.minimum(cfg.clearance_m, self.gaps(actual[None])[0])

    def gaps(self, poses):
        world = np.einsum("cij,hj->chi", rot(poses).as_matrix(), self.hull) + poses[:, None, :3]
        distance, idx = self.tree.query(world, distance_upper_bound=self.cfg.collision_radius_m)
        near = np.isfinite(distance)
        delta = world[near] - self.points[idx[near]]
        gaps = np.einsum("ij,ij->i", delta, self.normals[idx[near]])
        lateral2 = np.square(delta).sum(1) - gaps**2
        distance[near] = np.where(lateral2 <= self.cfg.collision_lateral_m**2, gaps, np.inf)
        return distance

    def valid(self, poses):
        # An already touching link may separate. It may not penetrate further.
        return (self.gaps(poses) >= self.minimum - 1e-7).all(1)
