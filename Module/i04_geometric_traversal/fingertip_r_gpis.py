"""Offline R-GPIS contact reconstruction, following Zhao and Xiao's reconstruction GP."""
from time import perf_counter

import gpytorch
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from skimage.measure import marching_cubes
import torch
import trimesh


REFERENCE = ('https://github.com/FerryRain/Autonomous-Exploration-for-Shape-Reconstruction'
             '/blob/afd1bb93a1e900b9415941545ad60c374c223077/utils/HE_GPIS.py')
VOXEL_M = .006
CONTACT_BUDGET = 1000
COORDINATE_UNIT_M = .1
DUMMY_GRID = 10
REMOVE_NEIGHBORS = 4
BOX_MARGIN_FRACTION = .2
TRAINING_ITERATIONS = 40
LEARNING_RATE = .1
GRID_STEP_M = .004
MIN_RESPONSE = .5
CONTACT_SUPPORT_M = .025
NORMAL_NEIGHBORS = 4
QUERY_BATCH = 2048


class InverseMultiquadric(gpytorch.kernels.Kernel):
    has_lengthscale = True

    def forward(self, x1, x2, diag=False, **params):
        distance_squared = ((x1 - x2).square().sum(dim=-1) if diag else
                            torch.cdist(x1, x2).square())
        return (distance_squared + self.lengthscale.squeeze().square()).rsqrt()


class ReconstructionGP(gpytorch.models.ExactGP):
    def __init__(self, points, labels, likelihood):
        super().__init__(points, labels, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(InverseMultiquadric())

    def forward(self, points):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(points), self.covar_module(points))


def voxel_representatives(points):
    """Keep an actual measured point nearest each 6 mm voxel's contact centroid."""
    _, cell, counts = np.unique(np.floor(points / VOXEL_M).astype(np.int32), axis=0,
                                return_inverse=True, return_counts=True)
    centers = np.column_stack([np.bincount(cell, weights=axis) / counts for axis in points.T])
    distances = np.square(points - centers[cell]).sum(axis=1)
    order = np.lexsort((distances, cell))
    first_in_cell = np.r_[0, np.cumsum(counts)[:-1]]
    return order[first_in_cell]


def farthest_indices(points, count):
    if len(points) <= count:
        return np.arange(len(points))
    distances = np.full(len(points), np.inf)
    current = int(np.argmin(np.square(points - points.mean(axis=0)).sum(axis=1)))
    selected = []
    for _ in range(count):
        selected.append(current)
        distances = np.minimum(distances, np.square(points - points[current]).sum(axis=1))
        current = int(np.argmax(distances))
    return np.asarray(selected)


def bounds_for_contacts(points):
    bounds = np.array([points.min(axis=0), points.max(axis=0)])
    margin = np.maximum(2 * GRID_STEP_M, BOX_MARGIN_FRACTION * np.ptp(bounds, axis=0))
    return bounds + np.array([-margin, margin])


def training_data(contacts, bounds):
    axes = [np.linspace(lo, hi, DUMMY_GRID) for lo, hi in bounds.T]
    dummy = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
    keep = np.ones(len(dummy), dtype=bool)
    _, nearby = cKDTree(dummy).query(contacts, k=REMOVE_NEIGHBORS)
    keep[nearby] = False
    return np.vstack((dummy[keep], contacts)), np.r_[np.zeros(keep.sum()), np.ones(len(contacts))]


def normal_field(tree, normals, query):
    distance, neighbors = tree.query(query, k=NORMAL_NEIGHBORS)
    nearby = normals[neighbors]
    same_side = np.einsum('nkj,nj->nk', nearby, nearby[:, 0]) >= .3
    weights = same_side / (distance + GRID_STEP_M)**2
    direction = np.sum(nearby * weights[:, :, None], axis=1)
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    return distance[:, 0], direction


def posterior_ridge(field, contacts, normals, origin, spacing):
    """Extract peaks in the unsigned posterior along measured contact normals."""
    tree = cKDTree(contacts)
    gradients = np.asarray(np.gradient(field, *spacing)).reshape(3, -1).T
    derivative = np.empty(field.size, dtype=np.float32)
    supported = np.empty(field.size, dtype=bool)
    for start in range(0, field.size, 12000):
        end = min(start + 12000, field.size)
        index = np.stack(np.unravel_index(np.arange(start, end), field.shape), axis=1)
        distance, direction = normal_field(tree, normals, origin + index * spacing)
        derivative[start:end] = np.einsum('ij,ij->i', gradients[start:end], direction)
        supported[start:end] = (distance <= CONTACT_SUPPORT_M + GRID_STEP_M) & (
            field.ravel()[start:end] >= MIN_RESPONSE)

    derivative = derivative.reshape(field.shape)
    vertices, faces, _, _ = marching_cubes(derivative, level=0., spacing=spacing,
                                           mask=supported.reshape(field.shape), allow_degenerate=False)
    mesh = trimesh.Trimesh(vertices=vertices + origin, faces=faces, process=False)
    centers = mesh.triangles_center
    distance, direction = normal_field(tree, normals, centers)
    index = (centers - origin) / spacing
    offset = direction * GRID_STEP_M / spacing
    # Discard minima and any triangles beyond the observed support of the ridge.
    plus = map_coordinates(derivative, (index + offset).T, order=1, mode='nearest')
    minus = map_coordinates(derivative, (index - offset).T, order=1, mode='nearest')
    response = map_coordinates(field, index.T, order=1, mode='nearest')
    mesh.update_faces((plus < minus) & (distance <= CONTACT_SUPPORT_M) & (response >= MIN_RESPONSE))
    mesh.remove_unreferenced_vertices()
    if not len(mesh.faces):
        raise ValueError('R-GPIS found no supported posterior maxima')
    return mesh


def reconstruct(contacts, normals, training_bounds, evaluation_bounds):
    """Fit the exact +1/0 R-GPIS model and mesh the posterior ridge."""
    started = perf_counter()
    if len(contacts) < 4:
        raise ValueError('At least four distinct contact points are required for R-GPIS')
    train_xyz, labels = training_data(contacts, training_bounds)
    center = evaluation_bounds.mean(axis=0)
    train_xyz = (train_xyz - center) / COORDINATE_UNIT_M
    train_x, train_y = torch.from_numpy(train_xyz), torch.from_numpy(labels)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().double()
    model = ReconstructionGP(train_x, train_y, likelihood).double()
    objective = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    with gpytorch.settings.max_cholesky_size(len(train_xyz) + 1):
        for _ in range(TRAINING_ITERATIONS):
            optimizer.zero_grad()
            loss = -objective(model(train_x), train_y)
            loss.backward()
            optimizer.step()
    fit_s = perf_counter() - started
    lengthscale = model.covar_module.base_kernel.lengthscale.item()
    amplitude = model.covar_module.outputscale.item()
    noise = likelihood.noise.item()
    mean = model.mean_module.constant.item()
    # Exact posterior mean queried in batches. No query-query covariance is needed.
    cov = amplitude / np.sqrt(cdist(train_xyz, train_xyz, 'sqeuclidean') + lengthscale**2)
    cov.flat[::len(cov) + 1] += noise
    weights = cho_solve(cho_factor(cov), labels - mean)
    shape = np.ceil(np.ptp(evaluation_bounds, axis=0) / GRID_STEP_M).astype(int) + 1
    spacing = np.ptp(evaluation_bounds, axis=0) / (shape - 1)
    mean_field = np.empty(np.prod(shape), dtype=np.float32)
    origin = evaluation_bounds[0]
    for start in range(0, len(mean_field), QUERY_BATCH):
        end = min(start + QUERY_BATCH, len(mean_field))
        index = np.stack(np.unravel_index(np.arange(start, end), shape), axis=1)
        query = (origin + index * spacing - center) / COORDINATE_UNIT_M
        distance = cdist(query, train_xyz, 'sqeuclidean')
        distance += lengthscale**2
        np.sqrt(distance, out=distance)
        np.reciprocal(distance, out=distance)
        mean_field[start:end] = mean + amplitude * (distance @ weights)
    mesh = posterior_ridge(mean_field.reshape(shape), contacts, normals, origin, spacing)
    return mesh, dict(contact_points=len(contacts), dummy_points=int(np.sum(labels == 0)),
        fit_s=fit_s, extract_s=perf_counter() - started - fit_s,
        fitted_lengthscale_m=lengthscale * COORDINATE_UNIT_M, output_scale=amplitude,
        gaussian_noise=noise, constant_mean=mean, grid_shape=shape.tolist(),
        posterior_range=[float(mean_field.min()), float(mean_field.max())],
        faces=len(mesh.faces), watertight=bool(mesh.is_watertight))
