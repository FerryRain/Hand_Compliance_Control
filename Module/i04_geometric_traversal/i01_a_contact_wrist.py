"""Fit only wrist height and tilt to multiple distal pads on a full-surface guide."""
from hashlib import sha256

import numpy as np
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.transform import Rotation

from .contracts import ExperimentConfig
from .fingertip_hand import FINGERS, ROOT
from .i01_a_wrist import as_wxyz
from .oracle import MeasuredCoverage, align_vector


LEVER_M = .065
PRELOAD_M = .00065
POLICY_REFERENCE_WEIGHT = .75
START_BLEND_M = .26


def policy_reference(directory):
    """Estimate one operating hand shape from two objects outside the benchmark."""
    poses, source = [], []
    for name in ('ycb_002_master_chef_can', 'ycb_021_bleach_cleanser'):
        path = directory / f'{name}.npz'
        with np.load(path, allow_pickle=False) as data:
            force = np.linalg.norm(data['input_tip_force_world'][:, 0], axis=-1)
            found = data['input_tip_found'][:, 0].astype(bool)
            valid = data['observed'][:, 0].astype(bool) & (data['reset_phase'][:, 0] == 0)
            valid &= ((found & (force >= .2)).sum(axis=1) >= 3) & (force.max(axis=1) < 8.)
            if not valid.any():
                raise ValueError(f'No measured low-load, three-tip calibration samples in {path}')
            poses.append(np.median(data['input_q_live'][:, 0][valid], axis=0))
        source.append(dict(object=name, file=path.relative_to(ROOT).as_posix(),
                           sha256=sha256(path.read_bytes()).hexdigest(), samples=int(valid.sum())))
    return np.mean(poses, axis=0), source


def limit_length(vector, maximum):
    return vector * min(1., maximum / max(np.linalg.norm(vector), 1.e-12))


def connect_guide(graph, adjacency, start, guide, spacing):
    """Connect the initial central pad along the mesh, then visit the original guide verbatim."""
    start_vertex = graph.nearest_vertex(start)[0]
    goal = graph.nearest_vertex(guide[0])[0]
    _, predecessors = dijkstra(adjacency, directed=False, indices=start_vertex, return_predecessors=True)
    vertices = [int(goal)]
    while vertices[-1] != start_vertex:
        parent = int(predecessors[vertices[-1]])
        if parent < 0:
            raise ValueError("The initial fingertip cannot reach the selected guide on the mesh")
        vertices.append(parent)
    polyline = np.vstack((start, graph.vertices_m[vertices[::-1]], guide[0]))
    polyline = polyline[np.r_[True, np.linalg.norm(np.diff(polyline, axis=0), axis=1) > 1.e-8]]
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(polyline, axis=0), axis=1))]
    if arc[-1] < 1.e-7:
        return guide, 0
    stations = np.linspace(0., arc[-1], int(np.ceil(arc[-1] / spacing)) + 1)[:-1]
    prefix = np.column_stack([np.interp(stations, arc, axis) for axis in polyline.T])
    return np.vstack((prefix, guide)), len(prefix)


class WristContactFit:
    def __init__(self, footprint, grasp, initial, supports):
        self.footprint = footprint
        position, attitude, _ = grasp
        self.points_local = attitude.inv().apply(initial.points - position)
        self.normals_local = attitude.inv().apply(initial.normals)
        spacing = np.linalg.norm(self.points_local[:, None] - self.points_local, axis=2)
        self.anchor = int(np.sort(spacing, axis=1)[:, 1:3].sum(axis=1).argmin())
        self.supports = supports
        self.height = 0.
        self.pad_queries = 0

    def linearize(self, target, surface_normal, attitude, height):
        origin = target - attitude.apply(self.points_local[self.anchor])
        pivot = target + height * surface_normal
        contacts = self.footprint.collisions(origin + height * surface_normal, as_wxyz(attitude))
        hand, data = self.footprint.h, self.footprint.data
        points = pivot + attitude.apply(self.points_local - self.points_local[self.anchor])
        outward = np.tile(surface_normal, (4, 1))
        gaps, fractions = np.empty(4), np.zeros(4)
        for f in range(4):
            gaps[f], witness = hand.nearest_pad_surface(data, f)
            if witness is not None:
                points[f], outward[f] = witness
                fractions[f] = hand.fraction(data, f, witness[0])
        self.pad_queries += 4

        distal = fractions >= hand.min_head_fraction
        cost = np.maximum(gaps, 0.) + .035 * np.maximum(.80 - fractions, 0.)
        weights = .28 * np.exp(-np.maximum(gaps, 0.) / .008) * distal
        weights[np.argsort(cost)[:self.supports]] = 1.
        weights[gaps < -PRELOAD_M] = 1.8
        desired = np.where(distal, -PRELOAD_M, .0002)
        residuals = gaps - desired
        # Bound penetration of the remaining links when a sharp corner or edge
        # makes the closest pair compete with the palm or the finger connectors.
        other = [c for c in contacts if c[3] not in hand.pad_index and c[0] < -.00055]
        if other:
            depth, point, normal, _ = min(other, key=lambda row: row[0])
            points = np.vstack((points, point))
            outward = np.vstack((outward, normal))
            residuals = np.r_[residuals, depth + .00055]
            weights = np.r_[weights, 2.5]
        axes = attitude.as_matrix()[:, :2]
        jacobian = np.column_stack((outward @ surface_normal, np.cross(points - pivot, outward) @ axes / LEVER_M))
        return weights[:, None] * jacobian, weights * residuals

    def advance(self, target, surface_normal, previous, length):
        # All rotations for a station share the previous palm's two tilt axes.
        # This preserves continuous transport without a free spin about the palm.
        tilt_limit = min(.19, max(.015, length / .043))
        desired = align_vector(previous.apply(self.normals_local[self.anchor]), surface_normal).as_rotvec()
        preferred = limit_length(previous.inv().apply(desired)[:2], tilt_limit)
        tilt = preferred.copy()
        height = self.height
        regularizer = np.diag([.08, .24, .24])
        for _ in range(7):
            attitude = previous * Rotation.from_rotvec(np.r_[tilt, 0.])
            jacobian, residuals = self.linearize(target, surface_normal, attitude, height)
            bias = regularizer @ np.r_[0., (tilt - preferred) * LEVER_M]
            delta = np.linalg.lstsq(np.vstack((jacobian, regularizer)),
                                    -np.r_[residuals, bias], rcond=None)[0]
            next_height = np.clip(height + np.clip(delta[0], -.007, .007), -.025, .060)
            next_tilt = limit_length(tilt + limit_length(delta[1:] / LEVER_M, .09), tilt_limit)
            difference = abs(next_height - height) + LEVER_M * np.linalg.norm(next_tilt - tilt)
            height, tilt = next_height, next_tilt
            if difference < .00009:
                break
        attitude = previous * Rotation.from_rotvec(np.r_[tilt, 0.])
        origin = target - attitude.apply(self.points_local[self.anchor])
        position, points, normals, valid = self.footprint.settle(origin, attitude, surface_normal, PRELOAD_M, height)
        self.height = float((position - origin) @ surface_normal)
        return position, attitude, points, normals, valid


def fit_contact_guide(graph, adjacency, footprint, grasp, initial, reference, spacing, supports=2,
                      operating_footprint=None, *, startup_supports=2):
    fitter = WristContactFit(footprint, grasp, initial, supports)
    start_fit = None
    if operating_footprint is not None:
        start_fit = WristContactFit(footprint, grasp, initial, startup_supports)
        fitter.points_local += operating_footprint.heads - footprint.heads
        fitter.footprint = operating_footprint
    guide, prefix = connect_guide(graph, adjacency, initial.points[fitter.anchor], reference, spacing)
    distances, nearest = graph._tree.query(guide, k=24)
    weights = np.exp(-(distances / .018)**2)
    normals = np.einsum('nk,nkj->nj', weights, graph.normals[nearest])
    magnitude = np.linalg.norm(normals, axis=1)
    normals = np.where((magnitude > 1.e-8)[:, None], normals / np.maximum(magnitude[:, None], 1.e-8),
                       graph.normals[nearest[:, 0]])
    position, attitude, _ = grasp
    positions, rotations, counts = [position], [attitude], [int(initial.measured_heads.sum())]
    seed_attitude = attitude
    covered = MeasuredCoverage(graph, ExperimentConfig(coverage_radius_m=.023))
    covered.update(initial.points, initial.normals, initial.measured_heads)
    lengths = np.linalg.norm(np.diff(guide, axis=0), axis=1)
    arc = np.r_[0., np.cumsum(lengths)]
    for i, length in enumerate(lengths, start=1):
        position, attitude, points, outward, valid = fitter.advance(guide[i], normals[i], attitude, length)
        issued = attitude
        if start_fit is not None and arc[i] < START_BLEND_M:
            # Interpolate only wrist poses. Each of the two planning models
            # keeps all sixteen joint angles fixed. The live fingers run their controller.
            seed_position, seed_attitude, _, _, _ = start_fit.advance(guide[i], normals[i], seed_attitude, length)
            fraction = arc[i] / START_BLEND_M
            blend = fraction * fraction * (3. - 2. * fraction)
            position = seed_position + blend * (position - seed_position)
            issued = seed_attitude * Rotation.from_rotvec(blend * (seed_attitude.inv() * attitude).as_rotvec())
            probe = footprint if blend < .5 else operating_footprint
            points, outward, valid = probe.head_contacts(probe.collisions(position, as_wxyz(issued)))
        positions.append(position)
        rotations.append(issued)
        counts.append(int(valid.sum()))
        covered.update(points, outward, valid)
    info = dict(variables_optimized_per_knot=3, guide_reference_finger=FINGERS[fitter.anchor],
                guide_prefix_knots=prefix, guide_prefix_m=float(np.linalg.norm(np.diff(guide[:prefix + 1], axis=0), axis=1).sum()),
                shared_guide_sha256=sha256(np.ascontiguousarray(reference, dtype='<f8').tobytes()).hexdigest(),
                native_pad_queries=fitter.pad_queries, rigid_preload_m=PRELOAD_M, target_supports=supports)
    if start_fit is not None:
        info.update(native_pad_queries=fitter.pad_queries + start_fit.pad_queries,
                    startup_blend_m=START_BLEND_M, operating_footprint_weight=POLICY_REFERENCE_WEIGHT)
    return np.asarray(positions), Rotation.concatenate(rotations), guide, covered, np.asarray(counts), info
