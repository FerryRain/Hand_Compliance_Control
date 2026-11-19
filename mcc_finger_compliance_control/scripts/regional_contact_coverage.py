"""Separate planned palm coverage from measured, valid fingertip coverage.

Reuse the existing five height bands and twelve azimuth sectors. Coordinates
come from the source mesh PCA; these are geometric bins, not semantic labels
such as bottle shoulder. The frame is frozen in the collection protocol.
"""
from __future__ import annotations

import h5py
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

HEIGHT_EDGES = np.array([0., .2, .4, .7, .9, 1.])
AZIMUTH_SECTORS = 12
HEIGHT_BANDS = 5
MIN_REGION_FRAMES = 50


class SurfaceRegions:
    def __init__(self, vertices):
        self.vertices = np.asarray(vertices, float)
        self.center = self.vertices.mean(0)
        _, _, axes = np.linalg.svd(self.vertices-self.center, full_matrices=False)
        if axes[0, np.abs(axes[0]).argmax()] < 0:
            axes[0] *= -1
        # Fix transverse signs too, making repeated region numbering stable.
        if axes[1, np.abs(axes[1]).argmax()] < 0:
            axes[1] *= -1
        axes[2] = np.cross(axes[0], axes[1])
        self.basis = np.column_stack((axes[1], axes[2], axes[0]))
        coordinates = (self.vertices-self.center) @ self.basis
        self.zmin, self.zmax = coordinates[:, 2].min(), coordinates[:, 2].max()
        if self.zmax-self.zmin <= 1.e-8:
            raise ValueError('Degenerate source mesh')
        self.tree = cKDTree(self.vertices)

    def description(self):
        return dict(height_edges=HEIGHT_EDGES.tolist(), azimuth_sector_count=AZIMUTH_SECTORS,
            center_object_m=self.center.tolist(), basis_columns_object=self.basis.tolist(),
            axial_min_max_m=[float(self.zmin), float(self.zmax)],
            pole_radius_excluded_from_azimuth_m=.005, minimum_distinct_frames=MIN_REGION_FRAMES,
            semantics='Source-mesh PCA bins; planned palm and actual fingertip coverage reported separately')

    def indices(self, points):
        points = np.asarray(points, float)
        finite = np.isfinite(points).all(-1)
        safe = np.where(finite[..., None], points, self.center)
        _, vertex = self.tree.query(safe.reshape(-1, 3))
        local = ((self.vertices[vertex]-self.center) @ self.basis).reshape(points.shape)
        height = (local[..., 2]-self.zmin)/(self.zmax-self.zmin)
        band = np.clip(np.searchsorted(HEIGHT_EDGES, height, side='right')-1, 0, 4)
        azimuth = np.mod(np.arctan2(local[..., 1], local[..., 0]), 2*np.pi)
        sector = np.minimum((azimuth/(2*np.pi)*AZIMUTH_SECTORS).astype(int), 11)
        off_pole = np.linalg.norm(local[..., :2], axis=-1) > .005
        return band, sector, finite, off_pole

    def coverage(self, points, valid):
        points = np.asarray(points)
        if points.ndim == 2:
            points = points[:, None, :]
        valid = np.asarray(valid, bool).reshape(points.shape[:-1])
        band, sector, finite, off_pole = self.indices(points)
        valid = valid & finite
        height_frames = np.array([(valid & (band == b)).any(1).sum() for b in range(5)])
        cell_frames = np.array([[(valid & off_pole & (band == b) & (sector == s)).any(1).sum()
            for s in range(AZIMUTH_SECTORS)] for b in range(5)])
        return dict(height_distinct_frames=height_frames.tolist(),
            cell_distinct_frames=cell_frames.tolist(),
            visited_height_bands=np.flatnonzero(height_frames >= MIN_REGION_FRAMES).tolist(),
            visited_cells=np.argwhere(cell_frames >= MIN_REGION_FRAMES).tolist(),
            dominant_height_band=int(np.argmax(height_frames)) if height_frames.any() else None)

    def planned(self, reference):
        from generate_manifold_palm_plan import PALM_CENTER_LOCAL, pose_to_rt
        with h5py.File(reference) as h:
            position, rotation = pose_to_rt(h['palm_pose_object'][:, 0])
        points = position + np.einsum('nij,j->ni', rotation, PALM_CENTER_LOCAL)
        return self.coverage(points, np.ones(len(points), bool))

    def measured(self, collected):
        with h5py.File(collected) as h:
            points = np.asarray(h['fingertip_contact_pos_world'])[:, 0]
            poses = np.asarray(h['object_pose_world'])[:, 0]
            strict = np.asarray(h['true_fingertip_contact_valid'])[:, 0] > .5
            found = np.asarray(h['fingertip_collision_found'])[:, 0] > .5
            if len(points) != len(poses):
                raise ValueError('Contact/object timing mismatch')
            if not np.isfinite(poses).all() or np.any(np.linalg.norm(poses[:, 3:], axis=-1) < .5):
                raise ValueError('Invalid object pose for measured coverage')
            rotation = Rotation.from_quat(poses[:, [4, 5, 6, 3]]).as_matrix()
            local = np.einsum('tji,tfj->tfi', rotation, points-poses[:, None, :3])
        return dict(strict=self.coverage(local, strict), found=self.coverage(local, found))


def regional_priority(coverage, height_counts, cell_counts):
    """Prefer references visiting regions with fewer accepted trajectories."""
    bands = coverage.get('visited_height_bands', [])
    cells = coverage.get('visited_cells', [])
    h = min((height_counts[b] for b in bands), default=float('inf'))
    c = min((cell_counts[b][s] for b, s in cells), default=float('inf'))
    return (h, c)
