"""An append-only signed GPIS and its supported online zero-level mesh."""
from dataclasses import dataclass
from time import perf_counter

import gpytorch
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes
import torch
from trimesh.graph import face_adjacency

from ..acitive_02.active_gpis import NOISE_VARIANCE, SIGN_OFFSET_M, VOXEL_M
from ..i04_geometric_traversal.fingertip_r_gpis import VOXEL_M as RIDGE_VOXEL_M
from .global_gpis import LENGTH, QUERY_BUDGET, QUERY_JITTER, representatives


GRID = .008
BAND = .024
FREQUENCIES = 192
MESH_MAX_TICKS = 100
MESH_MAX_NEW_CELLS = 48


@dataclass(frozen=True)
class ContactBand:
    """Normalized inverse-multiquadric response in local normal distance.

    This is the normalized kernel profile about one contact, not the original
    3-D reconstruction GP's posterior mean or a contact probability.
    """
    level: float = .95
    scale_m: float = RIDGE_VOXEL_M

    def __post_init__(self):
        if not 0. < self.level < 1. or not np.isfinite(self.scale_m) or self.scale_m <= 0.:
            raise ValueError("Expected a response level in (0,1) and a positive distance scale")

    @property
    def half_width(self):
        return self.scale_m * np.sqrt(self.level ** -2 - 1.)

    def response(self, distance):
        return 1. / np.sqrt(1. + (np.asarray(distance) / self.scale_m) ** 2)


class MaternFeatures(gpytorch.kernels.RFFKernel):
    """GPyTorch sine/cosine features with the Matérn-3/2 spectral measure."""
    def __init__(self):
        super().__init__(num_samples=FREQUENCIES)
        rng = np.random.default_rng(2071)
        # Multivariate Student t with 3 df: one scale draw per 3-D frequency.
        omega = rng.normal(size=(3, FREQUENCIES)) / np.sqrt(rng.chisquare(3, FREQUENCIES) / 3.)
        self.register_buffer("randn_weights", torch.from_numpy(omega))
        self.double()
        self.lengthscale = LENGTH
        self.requires_grad_(False)

    def features(self, xyz):
        return self._featurize(xyz, normalize=True)


@dataclass
class MeshGoal:
    face: int
    center: np.ndarray
    normal: np.ndarray
    waypoint: np.ndarray
    waypoint_normal: np.ndarray
    distance: float
    rate: float
    response: float


class PlanningMesh:
    def __init__(self, vertices, faces, surface):
        self.vertices, self.faces = vertices, faces
        self.nominees = np.empty(0, dtype=int)
        self.centers = vertices[faces].mean(axis=1)
        if not len(faces):
            return
        self.tree = cKDTree(self.centers)
        delta = vertices[faces[:, 1:]] - vertices[faces[:, :1]]
        cross = np.cross(delta[:, 0], delta[:, 1])
        self.normals = cross / np.maximum(np.linalg.norm(cross, axis=1, keepdims=True), 1e-12)
        adjacency = face_adjacency(faces=faces)
        i, j = adjacency.T
        keep = (self.normals[i] * self.normals[j]).sum(axis=1) > .15
        i, j = i[keep], j[keep]
        length = np.linalg.norm(self.centers[i] - self.centers[j], axis=1)
        self.graph = coo_matrix((length, (i, j)), shape=(len(faces), len(faces))).tocsr()
        self.contact_distance = surface.tree.query(self.centers)[0]
        boundary = np.flatnonzero(self.contact_distance >= VOXEL_M)
        self.nominees = boundary[representatives(self.centers[boundary], 144)]

    @classmethod
    def reconstruct(cls, surface, predict_distance, level_m):
        points = surface.points
        origin = GRID * np.floor((points.min(axis=0) - BAND - 2 * GRID) / GRID)
        shape = np.ceil((points.max(axis=0) + BAND + 2 * GRID - origin) / GRID).astype(int) + 1
        occupied = np.zeros(shape, dtype=bool)
        occupied[tuple(np.rint((points - origin) / GRID).astype(int).T)] = True
        distance = distance_transform_edt(~occupied) * GRID
        index = np.column_stack(np.where(distance <= BAND + 2 * GRID))
        field = np.zeros(shape, dtype=np.float32)
        field[tuple(index.T)] = predict_distance(origin + GRID * index)
        mask = distance <= BAND
        def empty():
            return cls(np.empty((0, 3)), np.empty((0, 3), dtype=int), surface)
        if not (field[mask].min() < level_m < field[mask].max()):
            return empty()
        try:
            vertices, faces, _, _ = marching_cubes(field, level=level_m, spacing=(GRID,) * 3,
                                                   mask=mask, allow_degenerate=False)
        except RuntimeError as exc:
            if "No surface found" in str(exc):
                return empty()
            raise
        vertices = vertices.astype(float) + origin
        tri = vertices[faces]
        centers = tri.mean(axis=1)
        dist, nearest = surface.tree.query(centers)
        normals = surface.normals[nearest]
        face_n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        area2 = np.linalg.norm(face_n, axis=1)
        agreement = np.sum(face_n * normals, axis=1) / np.maximum(area2, 1e-12)
        keep = (dist <= BAND) & (np.abs(agreement) > .5) & (area2 > 1e-12)
        faces, centers, normals, agreement = faces[keep].copy(), centers[keep], normals[keep], agreement[keep]
        if not len(faces):
            return empty()
        # Reject sign-reversed or flat extrapolations even if marching cubes
        # could create a triangle there. Leave unsupported space open.
        offset = SIGN_OFFSET_M * normals
        plus, minus = np.split(predict_distance(np.vstack((centers + offset, centers - offset))), 2)
        faces[agreement < 0] = faces[agreement < 0, ::-1]
        faces = faces[(plus - minus) > .10 * (2 * SIGN_OFFSET_M)]
        ids, inverse = np.unique(faces, return_inverse=True)
        return cls(vertices[ids], inverse.reshape(-1, 3), surface)

    def routes(self, point, normal):
        if not len(self.nominees):
            return []
        distance, faces = self.tree.query(point, k=min(24, len(self.faces)))
        distance, faces = np.atleast_1d(distance), np.atleast_1d(faces)
        near = np.flatnonzero((distance <= BAND) & (self.normals[faces] @ normal > .35))
        if not len(near):
            return []
        source = int(faces[near[0]])
        travel, prev = dijkstra(self.graph, directed=False, indices=source, return_predecessors=True)
        path = travel[self.nominees]
        rate = np.where((path > GRID) & (path < .32), self.value / (.2 + path / .04), 0.)
        goals = []
        for _ in range(3):
            k = int(rate.argmax())
            if rate[k] <= 0:
                break
            target = int(self.nominees[k])
            waypoint = target
            while prev[waypoint] >= 0 and travel[prev[waypoint]] > GRID:
                waypoint = prev[waypoint]
            goals.append(MeshGoal(target, self.goal_points[k], self.normals[target],
                                  self.centers[waypoint], self.normals[waypoint], float(path[k]),
                                  float(rate[k]), float(self.goal_response[k])))
            rate[np.linalg.norm(self.centers[self.nominees] - self.centers[target], axis=1) < BAND] = 0.
        return goals


class StreamingGPIS:
    """Exact sequential conditioning in a fixed finite-feature approximation."""
    def __init__(self, device="cpu", band=None):
        self.device = torch.device(device)
        self.band = band if band is not None else ContactBand()
        self.kernel = MaternFeatures().to(self.device).eval()
        self.precision = torch.eye(2 * FREQUENCIES, dtype=torch.float64, device=self.device)
        self.rhs = torch.zeros_like(self.precision[:, :1])
        self.count = self.version = 0
        self.last_fit_tick = -1
        self.mesh_count = self.mesh_version = 0
        self.mesh_tick = -1
        self.query = np.empty((0, 3))
        self.mesh = None
        self.timing = {}

    def features(self, x):
        return self.kernel.features(torch.as_tensor(x, dtype=torch.float64, device=self.device))

    @torch.inference_mode()
    def metric_distance(self, xyz, alpha=None):
        """First-order metric signed distance from the same posterior used for the mesh."""
        alpha = self.alpha if alpha is None else alpha
        weights = alpha.flatten()
        omega = self.kernel.randn_weights / self.kernel.lengthscale.T
        output = []
        for i in range(0, len(xyz), 8192):
            phi = self.features(xyz[i:i+8192])
            cos, sin = phi.split(FREQUENCIES, dim=1)
            grad = (cos * weights[FREQUENCIES:] - sin * weights[:FREQUENCIES]) @ omega.T
            value = (phi @ weights) / torch.linalg.vector_norm(grad, dim=1).clamp_min(.1)
            output.append(value.cpu().numpy())
        return np.concatenate(output) if output else np.empty(0)

    def sync(self):
        if self.device.type == "cuda":
            torch.cuda.current_stream(self.device).synchronize()

    def condition(self, points, normals):
        precision, rhs = self.precision.clone(), self.rhs.clone()
        for k in range(0, len(points), 2048):
            p = points[k:k+2048]
            offset = SIGN_OFFSET_M * normals[k:k+2048]
            phi = self.features(np.vstack((p, p+offset, p-offset)))
            y = torch.as_tensor(np.repeat([0., SIGN_OFFSET_M, -SIGN_OFFSET_M], len(p)),
                                device=self.device, dtype=torch.float64)[:, None]
            precision.addmm_(phi.T, phi, alpha=1. / NOISE_VARIANCE)
            rhs.addmm_(phi.T, y, alpha=1. / NOISE_VARIANCE)
        chol = torch.linalg.cholesky(precision)
        alpha = torch.cholesky_solve(rhs, chol)
        return precision, rhs, chol, alpha

    @torch.inference_mode()
    def update(self, surface, tick):
        n = len(surface.points)
        new = n - self.count
        if new < 0:
            raise ValueError("The tactile map must be append-only")
        due = n > self.mesh_count and (tick-self.mesh_tick >= MESH_MAX_TICKS or n-self.mesh_count >= MESH_MAX_NEW_CELLS)
        self.timing = dict(new_cells=new, mesh_rebuilt=False, condition_ms=0., mesh_ms=0., query_cache_ms=0.)
        if not new and not due:
            return False
        started = perf_counter()
        points, normals = surface.points[self.count:], surface.normals[self.count:]
        if new:
            precision, rhs, chol, alpha = self.condition(points, normals)
            self.sync()
            self.timing["condition_ms"] = 1000*(perf_counter()-started)
        else:
            precision, rhs, chol, alpha = self.precision, self.rhs, self.chol, self.alpha
        level_m = self.band.half_width
        fitted = perf_counter()
        mesh = self.mesh
        rebuild = mesh is None or not len(mesh.faces) or due
        if not rebuild and new:
            distances, nearby = mesh.tree.query(points)
            aligned = np.einsum('ij,ij->i', normals, mesh.normals[nearby])
            rebuild = bool(np.any((distances > 1.25*GRID) | (aligned < .5)))
        if rebuild:
            mesh = PlanningMesh.reconstruct(surface, lambda x:self.metric_distance(x, alpha), level_m)
            domain = mesh.centers if len(mesh.faces) else surface.points
            query = domain[representatives(domain, QUERY_BUDGET)]
        else:
            query = self.query
        meshed = perf_counter()
        white = torch.linalg.solve_triangular(chol, self.features(query).T, upper=False)
        qq = white.T @ white
        qq.diagonal().add_(QUERY_JITTER)
        qchol = torch.linalg.cholesky(qq)
        conditional = torch.linalg.solve_triangular(qchol, white.T, upper=False)
        if len(mesh.nominees):
            sites = mesh.centers[mesh.nominees]
            outward = mesh.normals[mesh.nominees]
            offset = .5 * GRID * outward
            plus, center, minus = np.split(self.metric_distance(np.vstack((sites+offset, sites, sites-offset)), alpha), 3)
            slope = (plus-minus) / GRID
            # Sub-grid refinement matters for a 2--3 mm contact band on an
            # 8 mm mesh. Aim just inside the threshold and verify directly.
            desired = level_m - min(.00005, .025*level_m)
            shift = np.clip((desired-center) / np.maximum(slope, .1), -.5*GRID, .5*GRID)
            targets = sites + shift[:,None]*outward
            actual = self.metric_distance(targets, alpha)
            response = self.band.response(actual)
            supported = ((plus > level_m) & (minus < level_m) & (slope > .1)
                         & (np.abs(actual-level_m) < .0003) & (response >= self.band.level) & (actual >= 0.))
            w = torch.linalg.solve_triangular(chol, self.features(targets).T, upper=False)
            prior = NOISE_VARIANCE + (w * w).sum(dim=0)
            reduction = ((conditional @ w) ** 2).sum(dim=0)
            information = (.5 * torch.log(prior / (prior-reduction).clamp_min(NOISE_VARIANCE))).cpu().numpy()
            distance = surface.tree.query(targets)[0]
            mesh.goal_points, mesh.goal_response = targets, response
            mesh.value = information * np.clip(distance / BAND, 0., 1.) * (distance <= BAND) * supported
        self.sync()
        self.timing.update(mesh_rebuilt=rebuild, mesh_ms=1000*(meshed-fitted),
                           query_cache_ms=1000*(perf_counter()-meshed))
        self.precision, self.rhs, self.chol, self.alpha = precision, rhs, chol, alpha
        self.conditional, self.query, self.mesh = conditional, query, mesh
        if new:
            self.count, self.last_fit_tick = n, tick
            self.version += 1
        if rebuild:
            self.mesh_count, self.mesh_tick = n, tick
            self.mesh_version += 1
        return True

    @torch.inference_mode()
    def observation_covariances(self, sites):
        white = torch.linalg.solve_triangular(self.chol, self.features(sites).T, upper=False)
        common = self.conditional @ white
        prior = white.T @ white
        return prior.cpu().numpy(), (prior - common.T @ common).cpu().numpy()
