"""LEAP port of the tactile acquisition and joint wrist/finger planning of Khadivar et al."""

from dataclasses import dataclass
from itertools import combinations

import gpytorch
import mujoco
import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from ..i04_geometric_traversal.fingertip_plan import Posture
from ..i04_geometric_traversal.physics import quaternion, rotation

UPSTREAM = 'https://github.com/epfl-lasa/object_exploration/tree/d89a6941fec699768e64fb6acc07439ab822e59f'
CONTACT_VOXEL_M = .004
FREE_VOXEL_M = .006
GLOBAL_LENGTH_M = .08
LOCAL_LENGTH_RANGE_M = (.20, .04)
GP_AMPLITUDE = np.exp(2.)
GP_NOISE = .3
MAX_TURN_RAD = np.pi / 6
TARGET_PRELOAD_M = .0008
REACHABILITY_SAMPLES = 10_000
DEFAULT_STEP_M = .018


def unit(vector):
    return vector / max(np.linalg.norm(vector), 1e-12)


def tangent(vector, normal):
    return vector - (vector @ normal) * normal


def state_posture(hand, data):
    return Posture(data.qpos[:3].copy(), rotation(data.qpos[3:7]), data.qpos[hand.q].copy())


class ScalarGP(gpytorch.models.ExactGP):
    """Fixed published noise, constant mean and squared exponential amplitude."""

    def __init__(self):
        samples = torch.zeros(1, 3, dtype=torch.float64)
        super().__init__(samples, samples[:, 0], gpytorch.likelihoods.GaussianLikelihood().double())
        self.mean_module = gpytorch.means.ConstantMean().double()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()).double()
        self.mean_module.constant = 1.
        self.covar_module.outputscale = GP_AMPLITUDE
        self.likelihood.noise = GP_NOISE**2
        self.requires_grad_(False).eval()

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))

    def variance_gradient(self, points, labels, center, length):
        self.set_train_data(inputs=torch.from_numpy(points), targets=torch.from_numpy(labels), strict=False)
        self.covar_module.base_kernel.lengthscale = length
        query = torch.tensor(center[None], dtype=torch.float64, requires_grad=True)
        with gpytorch.settings.fast_pred_var(False), gpytorch.settings.debug(False):
            variance = self(query).variance
            return torch.autograd.grad(variance.sum(), query)[0].detach().numpy()[0]


class ContactGP:
    """Squared exponential GP with mean 1, surface label 0 and free-space label 1."""

    def __init__(self):
        self.surface = {}
        self.free = {}

    def observe(self, points, normals, valid, centers):
        for point in points[valid]:
            self.surface.setdefault(tuple(np.floor(point / CONTACT_VOXEL_M).astype(int)), point.copy())
        # A pad center is a free-space witness when it is separated from the
        # object or lies on the free side of its measured contact plane.
        outside = ~valid | (np.einsum('ij,ij->i', centers - points, normals) > .003)
        for point in centers[outside]:
            self.free.setdefault(tuple(np.floor(point / FREE_VOXEL_M).astype(int)), point.copy())

    def training(self):
        points = np.asarray([*self.surface.values(), *self.free.values()]).reshape(-1, 3)
        labels = np.r_[np.zeros(len(self.surface)), np.ones(len(self.free))]
        return points, labels


class ExplorationDirections:
    """Adaptive local length scale and tangential GP variance gradient, per finger."""

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.model = ScalarGP()
        self.gamma = np.zeros(4)
        self.connectivity = np.zeros(4)
        self.previous = np.zeros((4, 3))
        self.tangent_length = np.full(4, LOCAL_LENGTH_RANGE_M[0])

    def directions(self, gp, anchors, normals, moving, valid, stuck):
        points, labels = gp.training()
        directions = np.zeros((4, 3))
        for finger, (center, normal) in enumerate(zip(anchors, normals)):
            squared = np.sum((points - center)**2, axis=1)
            indices = np.argsort(squared, kind='stable')[:100]
            neighbors = points[indices]
            spread = np.linalg.norm(np.std(neighbors[:20], axis=0))
            desired_gamma = min(1., (spread / .1)**6)
            connected = np.exp(-squared / (2 * GLOBAL_LENGTH_M**2)).mean() * GP_AMPLITUDE
            smoothed = .8 * self.connectivity[finger] + .2 * connected
            gain = 10. * max(0., smoothed - self.connectivity[finger])
            self.connectivity[finger] = smoothed
            self.gamma[finger] = np.clip(self.gamma[finger] + gain * (desired_gamma - self.gamma[finger]), 0., 1.)
            length = np.interp(self.gamma[finger], [0., 1.], LOCAL_LENGTH_RANGE_M)
            self.tangent_length[finger] = length
            gradient = self.model.variance_gradient(neighbors, labels[indices], center, length)
            direction = tangent(gradient, normal)
            if np.linalg.norm(direction) < 1e-6 or (valid[finger] and stuck[finger]):
                direction = tangent(self.rng.normal(size=3), normal)
            direction = unit(direction)
            prior = moving[finger] if np.linalg.norm(moving[finger]) > .001 else self.previous[finger]
            prior = unit(tangent(prior, normal))
            if np.linalg.norm(prior) > .9 and not stuck[finger]:
                cosine = np.clip(prior @ direction, -1., 1.)
                if cosine < np.cos(MAX_TURN_RAD):
                    across = direction - cosine * prior
                    if np.linalg.norm(across) < 1e-9:
                        across = np.cross(normal, prior)
                    direction = np.cos(MAX_TURN_RAD) * prior + np.sin(MAX_TURN_RAD) * unit(across)
            directions[finger] = direction
        self.previous = directions.copy()
        return directions


@dataclass(frozen=True)
class Reachability:
    planes: np.ndarray
    vertices: np.ndarray
    scale: float

    def eta(self, point):
        # ConvexHull uses the sign convention n*x+b <= 0 in the interior.
        gap = np.min(-self.planes[:, :3] @ point - self.planes[:, 3])
        radius = np.sqrt(np.max(np.sum((self.vertices - point)**2, axis=1)))
        return max(gap, 0.) / max(radius, 1e-9)


class JointIK:
    """Solve a single wrist pose and all 16 finger joints against virtual targets."""

    def __init__(self, hand, *, limits=None, solver=None):
        self.hand = hand
        self.solver = solver
        self.model = hand.model
        self.data = mujoco.MjData(hand.model)
        self.jacobian = np.empty((3, hand.model.nv))
        self.limits = hand.limits + np.array([.012, -.012]) if limits is None else limits
        self.scales = np.r_[np.full(3, .030), np.full(3, .25), np.full(16, .24)]
        self.joint_weights = np.tile(np.sqrt([1.1, 1.5, 1.2, 1.3]), 4)
        self.joint_weights[-4:-2] = np.sqrt([1.5, 1.1])
        self.contact_weights = np.sqrt(10. * np.array([2., 2., 2., 1.]) * 4 / 7)
        self.flexion_indices = [0, 2, 3, 4, 6, 7, 8, 10, 11, 14, 15]
        pads = hand.pads.tolist()
        penultimate = [self.model.geom(name).id for name in ('dip_geom', 'dip_2_geom', 'dip_3_geom')]
        self.pairs = [*combinations(pads, 2), *zip(penultimate[:-1], penultimate[1:])]
        self.reach = self.build_reachability()

    def set_pose(self, position, orientation, joints):
        self.data.qpos[:3] = position
        self.data.qpos[3:7] = quaternion(orientation)
        self.data.qpos[self.hand.q] = joints
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

    def manipulability(self, finger, point):
        hand = self.hand
        mujoco.mj_jac(self.model, self.data, self.jacobian, None, point, hand.tip_bodies[finger])
        block = self.jacobian[:, hand.dof[4 * finger:4 * finger + 4]]
        return np.sqrt(max(np.linalg.det(block @ block.T), 0.))

    def build_reachability(self):
        maps = []
        rng = np.random.default_rng(5401)
        for finger in range(4):
            sl = slice(finger * 4, finger * 4 + 4)
            low, high = self.limits[sl].T
            configurations = rng.uniform(low, high, (REACHABILITY_SAMPLES, 4))
            points = np.empty((len(configurations), 3))
            mu = np.empty(len(configurations))
            q = self.limits.mean(axis=1)
            zero = np.zeros(3)
            identity = Rotation.identity()
            for index, configuration in enumerate(configurations):
                q[sl] = configuration
                self.set_pose(zero, identity, q)
                points[index] = self.data.geom_xpos[self.hand.pads[finger]]
                mu[index] = self.manipulability(finger, points[index])
            hull = ConvexHull(points)
            maps.append(Reachability(hull.equations, points[hull.vertices], max(np.percentile(mu, 99.5), 1e-9)))
        return maps

    def fit(self, start, offsets, targets):
        base_q = np.clip(start.joints, *self.limits.T)
        lower = np.maximum(-1., (self.limits[:, 0] - base_q) / self.scales[6:])
        upper = np.minimum(1., (self.limits[:, 1] - base_q) / self.scales[6:])
        wrist_bounds = np.ones(6)
        # Use the fixture origin, as with the workspace ROI center in the
        # upstream MuJoCo path. No object surface is queried during this solve.
        current_facing = start.orientation.apply([0., 0., -1.]) @ unit(-start.position)
        approach_floor = min(np.cos(np.deg2rad(20.)), current_facing - .03)

        def pose(variables):
            delta = variables * self.scales
            return Posture(start.position + delta[:3], Rotation.from_rotvec(delta[3:6]) * start.orientation,
                           base_q + delta[6:])

        def residual(variables):
            command = pose(variables)
            self.set_pose(command.position, command.orientation, command.joints)
            body_axes = self.data.xmat[self.hand.tip_bodies].reshape(4, 3, 3)
            tips = self.data.xpos[self.hand.tip_bodies] + np.einsum('ijk,ik->ij', body_axes, offsets)
            position_error = ((tips - targets) * self.contact_weights[:, None]).ravel() * 1000
            centers = self.data.geom_xpos[self.hand.pads]
            local = command.orientation.inv().apply(centers - command.position)
            eta = np.array([geometry.eta(point) for geometry, point in zip(self.reach, local)])
            mu = np.array([self.manipulability(finger, point) / self.reach[finger].scale
                           for finger, point in enumerate(centers)])
            natural = command.joints.copy()
            natural[self.flexion_indices] -= self.limits[self.flexion_indices, 0]
            facing = command.orientation.apply([0., 0., -1.]) @ unit(-command.position)
            gaps = self.gaps()
            return np.r_[position_error, np.sqrt(2.5) * self.joint_weights * natural,
                         1. / np.sqrt(eta + .03), 1. / np.sqrt(mu + .03),
                         5. * variables[:3], 5. * variables[3:6], 2. * variables[6:],
                         80. * max(0., approach_floor - facing), 3000. * np.maximum(.002 - gaps, 0.)]

        solve = self.solver or least_squares
        solution = solve(residual, np.zeros(22), bounds=(np.r_[-wrist_bounds, lower],
                                 np.r_[wrist_bounds, upper]), max_nfev=12, ftol=.003, xtol=.003)
        metrics = residual(solution.x)
        error = np.linalg.norm(metrics[:12].reshape(4, 3) / self.contact_weights[:, None], axis=1)
        return pose(solution.x), float(error.max()), float(self.gaps().min())

    def gaps(self):
        return np.array([mujoco.mj_geomDistance(self.model, self.data, a, b, .025, None) for a, b in self.pairs])

@dataclass
class Plan:
    posture: Posture
    mode: str
    targets: np.ndarray
    direction: np.ndarray
    valid: np.ndarray
    error_mm: float | None
    self_gap_m: float | None


class AudeExplorer:
    def __init__(self, hand, ik, seed, step_m):
        self.hand = hand
        self.ik = ik
        self.step_m = step_m
        self.gp = ContactGP()
        self.directions = ExplorationDirections(seed)
        self.last_points = np.zeros((4, 3))
        self.last_normals = np.zeros((4, 3))
        self.offsets = np.zeros((4, 3))
        self.previous = None
        self.anchor_history = []
        self.safe = None
        self.loss_plans = 0

    def observe(self, data, touch):
        valid = touch.measured_heads
        centers = data.geom_xpos[self.hand.pads]
        self.gp.observe(touch.points, touch.normals, valid, centers)
        self.last_points[valid] = touch.points[valid]
        self.last_normals[valid] = touch.normals[valid]
        bases = data.xpos[self.hand.tip_bodies]
        axes = data.xmat[self.hand.tip_bodies].reshape(4, 3, 3)
        self.offsets[valid] = np.einsum('ijk,ij->ik', axes[valid], (touch.points - bases)[valid])

    def plan(self, data, touch):
        start = state_posture(self.hand, data)
        valid = touch.measured_heads
        anchors = self.last_points.copy()
        missing = np.linalg.norm(self.last_normals, axis=1) < .5
        anchors[missing] = data.geom_xpos[self.hand.pads[missing]]
        normals = self.last_normals.copy()
        normals[missing] = start.orientation.apply([0., 0., 1.])
        if valid.sum() >= 3 and touch.loads.max() < 8.:
            self.safe = start
        if not valid.any():
            self.loss_plans += 1
            probing_depth = min(.008, .001 * (self.loss_plans - 1))
            goal = (self.safe or start).lift(-unit(normals.sum(axis=0)), probing_depth)
            return Plan(goal, 'recontact', anchors, np.zeros((4, 3)), valid, None, None)
        self.loss_plans = 0

        moving = anchors - self.previous if self.previous is not None else np.zeros_like(anchors)
        stuck = np.zeros(4, dtype=bool)
        if len(self.anchor_history) == 5:
            stored = np.asarray(self.anchor_history)
            stuck = np.max(np.linalg.norm(stored - anchors, axis=2), axis=0) < .005
            self.anchor_history.pop(0)
        self.anchor_history.append(anchors.copy())
        self.previous = anchors.copy()
        directions = self.directions.directions(self.gp, anchors, normals, moving, valid, stuck)
        ease = 1. if valid.all() else .65
        # In the tactile-only path the goal is virtual and tangent to the
        # measured normal. A small preload compensates plant tracking lag.
        for factor in (1., .5, .25):
            targets = anchors + ((self.step_m * ease * factor) * directions * valid[:, None]
                                 - TARGET_PRELOAD_M * normals * (~missing)[:, None])
            wrist_fingers, error, gap = self.ik.fit(start, self.offsets, targets)
            if error <= 13. and gap >= -.001:
                return Plan(wrist_fingers, 'explore', targets, directions, valid, error, gap)
        return Plan(self.safe or start, 'recontact', anchors, directions, valid, error, gap)
