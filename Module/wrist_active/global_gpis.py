"""A bounded, object-wide signed GP posterior for online tactile acquisition.

The domain comes entirely from this trial's oriented tactile observations. All
wrist actions in a decision share the same conditioning data and query set.
"""
import numpy as np
from scipy.linalg import cho_solve, cholesky, solve_triangular
from scipy.spatial.distance import cdist

from ..acitive_02.active_gpis import NOISE_VARIANCE, SIGN_OFFSET_M


LENGTH = .028
RATE = np.sqrt(3.) / LENGTH
SITE_BUDGET = 48
QUERY_BUDGET = 112
QUERY_JITTER = 1e-7
STROKE_M = .024
ANGLES = np.arange(8) * (np.pi / 4)


def kernel(left, right):
    scaled = RATE * cdist(left, right)
    return (1.0 + scaled) * np.exp(-scaled)


def representatives(points, budget):
    """Deterministic farthest-point cover, independent of the current wrist."""
    if len(points) <= budget:
        return np.arange(len(points))
    chosen = np.empty(budget, dtype=int)
    dist = np.full(len(points), np.inf)
    i = np.argmin(np.linalg.norm(points - points.mean(axis=0), axis=1))
    for j in range(budget):
        chosen[j] = i
        d = points - points[i]
        np.minimum(dist, np.einsum("ij,ij->i", d, d), out=dist)
        dist[i] = -1.
        i = np.argmax(dist)
    return chosen


def tangents(normals):
    axis = np.eye(3)[np.argmin(np.abs(normals), axis=1)]
    first = np.cross(normals, axis)
    first /= np.maximum(np.linalg.norm(first, axis=1, keepdims=True), 1e-9)
    second = np.cross(normals, first)
    return first[:, None] * np.cos(ANGLES)[None, :, None] + second[:, None] * np.sin(ANGLES)[None, :, None]


class GlobalGPIS:
    def __init__(self):
        self.count = 0
        self.last_fit_tick = -1
        self.version = 0

    def update(self, surface, tick):
        n = len(surface.points)
        if n == self.count or (self.count >= 12 and tick - self.last_fit_tick < 80 and n - self.count < 48):
            return False
        ids = representatives(surface.points, SITE_BUDGET)
        anchors, normals = surface.points[ids], surface.normals[ids]
        offset = SIGN_OFFSET_M * normals
        self.train = np.concatenate([anchors, anchors + offset, anchors - offset])
        values = np.repeat([0., SIGN_OFFSET_M, -SIGN_OFFSET_M], len(ids))
        k = kernel(self.train, self.train)
        k.flat[::len(k) + 1] += NOISE_VARIANCE
        self.factor = cholesky(k, lower=True, check_finite=False)
        self.alpha = cho_solve((self.factor, True), values, check_finite=False)

        # Q includes measured bands and one short outward extension along the
        # local tangent. Extend only with a supported signed zero crossing.
        dirs = tangents(normals)
        ring = anchors[:, None] + STROKE_M * dirs
        q, supported = self.project(ring, normals)
        all_q = np.vstack((anchors, q[supported]))
        self.query = all_q[representatives(all_q, QUERY_BUDGET)]
        self.query_white = self.whiten(self.query)
        qq = kernel(self.query, self.query) - self.query_white.T @ self.query_white
        qq.flat[::len(qq) + 1] += QUERY_JITTER
        self.query_factor = cholesky(qq, lower=True, check_finite=False)
        self.count, self.last_fit_tick = n, tick
        self.version += 1
        return True

    def project(self, proposals, normals):
        """Locally project tangent proposals onto the object-wide signed mean."""
        x = proposals.reshape(-1, 3)
        rho = RATE * cdist(x, self.train)
        exp = np.exp(-rho)
        mean = ((1. + rho) * exp) @ self.alpha
        terms = exp * self.alpha
        grad = RATE**2 * (terms @ self.train - terms.sum(axis=1)[:, None] * x)
        normal = np.broadcast_to(normals[:, None], proposals.shape).reshape(-1, 3)
        slope = np.einsum("ij,ij->i", grad, normal)
        gap = mean / np.maximum(slope, .08)
        keep = (slope > .10) & (np.abs(gap) < .012)
        return (x - normal * gap[:, None].clip(-.012, .012)).reshape(proposals.shape), keep.reshape(proposals.shape[:-1])

    def whiten(self, x):
        return solve_triangular(self.factor, kernel(self.train, x), lower=True, check_finite=False)

    def factors(self, x):
        dw = self.whiten(x)
        cross = kernel(self.query, x) - self.query_white.T @ dw
        qw = solve_triangular(self.query_factor, cross, lower=True, check_finite=False)
        return dw, qw

    def regional_gain(self, x):
        """I(f(Q); one hypothetical observation at x | observed touches)."""
        dw, qw = self.factors(x)
        total = 1. + NOISE_VARIANCE - np.einsum("ij,ij->j", dw, dw)
        reduction = np.einsum("ij,ij->j", qw, qw)
        return .5 * np.log(total / np.maximum(total - reduction, NOISE_VARIANCE))

    def observation_covariances(self, x):
        """Latent covariances at C, before and after also conditioning on Q."""
        dw, qw = self.factors(x)
        prior = kernel(x, x) - dw.T @ dw
        return prior, prior - qw.T @ qw
