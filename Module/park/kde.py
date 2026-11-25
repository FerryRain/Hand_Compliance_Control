"""Four episode-balanced isotropic Gaussian KDEs and integrated spherical mass."""
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.signal import fftconvolve
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
from scipy.stats import ncx2

from .settings import FINGERS


def density_log(points, centers, weights, bandwidth):
    points = np.asarray(points).reshape(-1, 3)
    constant = -3 * np.log(bandwidth * np.sqrt(2 * np.pi))
    return np.concatenate([logsumexp(np.log(weights)[None, :] - .5 * cdist(block, centers, "sqeuclidean")
                                     / bandwidth**2, axis=1) + constant
                           for block in np.array_split(points, max(1, int(np.ceil(len(points) / 256))))])


def exact_mass(points, centers, weights, h, radius):
    distance2 = cdist(np.atleast_2d(points), centers, "sqeuclidean") / h**2
    mass = np.zeros_like(distance2)
    near = distance2 < (radius / h + 9.)**2
    mass[near] = ncx2.cdf((radius / h)**2, 3, distance2[near])
    return mass @ weights


class FingerKDE:
    def __init__(self, centers, weights, bandwidth, radius, spacing, origin, grid):
        self.centers, self.weights = centers, weights
        self.bandwidth, self.radius, self.spacing = float(bandwidth), float(radius), float(spacing)
        self.origin, self.grid = origin, grid

    @classmethod
    def fit(cls, centers, weights, bandwidth, radius, spacing):
        # Linear binning and an FFT of the Gaussian ball integral amortise the numerical
        # integration across all wrist queries. Padding includes the integration tail.
        span = int(np.ceil((radius + 8 * bandwidth) / spacing)) + 2
        origin = (np.floor(centers.min(0) / spacing) - span) * spacing
        positions = (centers - origin) / spacing
        shape = np.ceil(positions.max(0)).astype(int) + span + 2
        histogram = np.zeros(shape)
        lower = np.floor(positions).astype(int)
        fraction = positions - lower
        for corner in np.ndindex(2, 2, 2):
            coef = np.prod(np.where(corner, fraction, 1 - fraction), axis=1)
            np.add.at(histogram, tuple((lower + corner).T), weights * coef)
        axes = np.arange(-span, span + 1) * (spacing / bandwidth)
        delta2 = axes[:, None, None]**2 + axes[None, :, None]**2 + axes[None, None, :]**2
        radii, inverse = np.unique(delta2, return_inverse=True)
        kernel = ncx2.cdf((radius / bandwidth)**2, 3, radii)[inverse].reshape(delta2.shape)
        grid = np.clip(fftconvolve(histogram, kernel, mode="same"), 0, 1).astype("float32")
        return cls(centers, weights, bandwidth, radius, spacing, origin, grid)

    def mass(self, points):
        coords = (np.asarray(points).reshape(-1, 3) - self.origin) / self.spacing
        return map_coordinates(self.grid, coords.T, order=1, mode="constant", cval=0.,
                               prefilter=False).reshape(np.shape(points)[:-1])

    def log_density(self, points):
        return density_log(points, self.centers, self.weights, self.bandwidth)

    def save(self, path):
        np.savez_compressed(path, **vars(self))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as d:
            return cls(**dict(d))


class HandKDE:
    def __init__(self, models):
        self.models = models
        # Physical finger reference offsets from the same palm frame used at deployment.
        self.offsets = np.stack([m.weights @ m.centers for m in models])

    @classmethod
    def load(cls, directory):
        return cls([FingerKDE.load(Path(directory) / (f + ".npz")) for f in FINGERS])

    def scores(self, points):
        individual = np.stack([m.mass(points) for m in self.models], axis=-1)
        return individual, 1 - np.prod(1 - individual, axis=-1), individual.max(axis=-1)

