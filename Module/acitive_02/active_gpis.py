"""Local signed GPIS fitted to this trial's physical tactile observations."""

import gpytorch
import numpy as np
import torch
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial import cKDTree

from .aude import unit
from .surface_scan import SURFACE_LOAD_N


VOXEL_M = .0045
SIGN_OFFSET_M = .005
NEIGHBORS = 24
TRUST_RADIUS_M = .048
NOISE_VARIANCE = .008


class LocalGPIS:
    """Oriented near-surface GP with the tangent plane as its local mean."""

    def __init__(self, points, normals, kernel):
        self.anchor, self.normal = points[0], normals[0]
        self.train = np.vstack((points, points - SIGN_OFFSET_M * normals, points + SIGN_OFFSET_M * normals))
        self.length = float(np.clip(.017 + .46 * np.linalg.norm(points - points[0], axis=1).max(), .018, .039))
        self.rate = np.sqrt(3.) / self.length
        self.labels = np.repeat([0., -SIGN_OFFSET_M, SIGN_OFFSET_M], len(points))
        residual = self.labels - (self.train - self.anchor) @ self.normal
        with torch.inference_mode():
            kernel.lengthscale = self.length
            train = torch.from_numpy(self.train)
            covariance = kernel(train, train).to_dense().numpy()
        covariance.flat[::len(covariance) + 1] += NOISE_VARIANCE
        self.factor = cho_factor(covariance, overwrite_a=True, check_finite=False)
        self.weights = cho_solve(self.factor, residual, check_finite=False)

    def predict(self, points, *, variance=False):
        query = np.atleast_2d(points)
        difference = query[:, None, :] - self.train
        scaled = self.rate * np.linalg.norm(difference, axis=2)
        exponent = np.exp(-scaled)
        cross = (1. + scaled) * exponent
        value = (query - self.anchor) @ self.normal + cross @ self.weights
        gradient = self.normal - self.rate**2 * np.einsum('ij,ijk->ik', exponent * self.weights, difference)
        if variance:
            solved = cho_solve(self.factor, cross.T, check_finite=False).T
            return np.clip(1. - np.sum(cross * solved, axis=1), 0., 1.)
        return value, gradient


class ObservedSurface:
    def __init__(self):
        self.samples = {}
        self.points = np.empty((0, 3))
        self.normals = np.empty((0, 3))
        self.tree = None
        self.kernel = gpytorch.kernels.MaternKernel(nu=1.5).double()
        self.patches = {}
        self.last_observation_tick = None

    def add(self, touch, tick):
        if touch.loads.max() > SURFACE_LOAD_N:
            return
        for point, normal in zip(touch.points[touch.measured_heads], touch.normals[touch.measured_heads]):
            cell = tuple(np.floor(point / VOXEL_M).astype(int))
            if cell not in self.samples:
                self.samples[cell] = np.r_[point, normal]
                self.last_observation_tick = tick

    def refresh(self):
        if len(self.samples) != len(self.points):
            samples = np.asarray(list(self.samples.values()))
            self.points, self.normals = samples[:, :3], samples[:, 3:]
            self.tree = cKDTree(self.points)
            self.patches.clear()

    def patch(self, point, *, preferred_normal=None, radius=TRUST_RADIUS_M):
        if self.tree is None:
            return None
        distance, indices = self.tree.query(point, k=min(2 * NEIGHBORS, len(self.points)))
        distance, indices = np.atleast_1d(distance), np.atleast_1d(indices)
        if preferred_normal is not None:
            allowed = self.normals[indices] @ preferred_normal > .18
            distance, indices = distance[allowed], indices[allowed]
        if not len(indices) or distance[0] > radius:
            return None
        anchor = indices[0]
        use = (distance <= radius + .015) & (self.normals[indices] @ self.normals[anchor] > .44)
        selected = tuple(indices[use][:NEIGHBORS])
        if selected not in self.patches:
            self.patches[selected] = LocalGPIS(self.points[list(selected)], self.normals[list(selected)], self.kernel)
        return self.patches[selected]

    def gap(self, point, patch, contact=None):
        value, gradient = patch.predict(point)
        normal = unit(gradient[0])
        correction = 0.
        if contact is not None:
            correction = float(patch.predict(contact)[0][0])
        distance = (float(value[0]) - correction) / np.clip(np.linalg.norm(gradient[0]), .45, 2.)
        return distance, normal
