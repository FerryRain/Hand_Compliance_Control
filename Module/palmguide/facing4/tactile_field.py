"""Select tactile surface regions from the posterior and move a wrist toward them."""

import argparse
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.transform import Rotation

from ..acitive_02.active_gpis import NOISE_VARIANCE, VOXEL_M
from ..acitive_02.aude import tangent, unit
from ..acitive_02.surface_scan import align
from . import contact_first, informative_run, informative_wrist, multirate_wrist, online_tactile_mesh
from .contact_first import signed_gap
from .experiment import digest, seed_for
from .history import normal_spread
from .informative_actions import FRACTIONS, HISTORY, HORIZON, MASK_THRESHOLDS, IndexedHandHistory, joint_gain
from .multirate_wrist import GLOBAL_TICKS, SPEED_M_S
from .online_tactile_mesh import BAND, GRID
from .ycb_fixtures import OBJECTS


@dataclass
class Region:
    finger: int
    point: np.ndarray
    normal: np.ndarray
    path: np.ndarray
    distance: float
    source: str


def mesh_regions(model, pads, ids, normals):
    mesh = model.mesh
    if not len(mesh.nominees) or not np.any(mesh.value > 0):
        return []
    result = []
    for finger in ids:
        pad, n = pads[finger], normals[finger]
        near, faces = mesh.tree.query(pad, k=min(24, len(mesh.centers)))
        near, faces = np.atleast_1d(near), np.atleast_1d(faces)
        supported = faces[(near <= BAND) & (mesh.normals[faces] @ n > .35)]
        if not len(supported):
            continue
        source = int(supported[0])
        distance, previous = dijkstra(mesh.graph, directed=False, indices=source, return_predecessors=True)
        travel = distance[mesh.nominees]
        candidates = np.flatnonzero((travel > GRID) & (travel <= .32) & (mesh.value > 0))
        rate = mesh.value[candidates] / (HORIZON + travel[candidates] / SPEED_M_S)
        retained = []
        for j in candidates[np.argsort(-rate, kind='stable')]:
            face = int(mesh.nominees[j])
            end_n = mesh.normals[face]
            end = mesh.goal_points[j] - model.band.half_width * end_n
            if np.linalg.norm(pad-end) < .010 or any(np.linalg.norm(end-p) < .016 for p in retained):
                continue
            chain = [face]
            while chain[-1] != source:
                chain.append(int(previous[chain[-1]]))
            chain = chain[::-1]
            path = mesh.centers[chain] - model.band.half_width * mesh.normals[chain]
            result.append(Region(int(finger), end, end_n, np.vstack((path, end)), float(travel[j]), 'mesh_frontier'))
            retained.append(end)
            if len(retained) == 3:
                break
    return result


def gradient_regions(model, surface, pads, ids, normals, rng):
    # The coordinates differentiate the continuous posterior. They are never candidate moves.
    epsilon = .0015
    offsets = epsilon * np.eye(3)
    queries = np.concatenate((pads[ids, None]+offsets, pads[ids, None]-offsets), axis=1)
    with torch.inference_mode():
        features = model.features(queries.reshape(-1, 3))
        white = torch.linalg.solve_triangular(model.chol, features.T, upper=False)
        variances = white.square().sum(0).cpu().numpy().reshape(-1, 6)
    gradients = (variances[:, :3] - variances[:, 3:]) / (2*epsilon)
    result = []
    for finger, gradient in zip(ids, gradients, strict=True):
        pad, n = pads[finger], normals[finger]
        heading = tangent(gradient, n)
        kind = 'posterior_gradient'
        if np.linalg.norm(heading) < 1.e-6:
            # At an exactly symmetric contact the data supply no preferred tangent.
            heading = tangent(rng.normal(size=3), n)
            kind = 'isotropic_bootstrap'
        patch = surface.patch(pad, preferred_normal=n)
        if patch is None or np.linalg.norm(heading) < 1.e-9:
            continue
        # Extrapolate a local chart up to two mesh cells from real support.
        # A band of recent observations can support a longer move. At a fresh
        # edge the wrist commits only to the nearest trustworthy part of it.
        reach = SPEED_M_S * HORIZON
        progress = np.linspace(0., reach, 1 + int(np.ceil(reach / (.5*GRID))))
        bearing = unit(heading)
        distance, nearest = surface.tree.query(pad + progress[:, None]*bearing)
        outside = np.flatnonzero((distance > 2*GRID) | (surface.normals[nearest] @ n < .45))
        if len(outside):
            j = int(outside[0])
            if j == 0:
                continue
            reach = progress[j-1]
            if distance[j] > 2*GRID and surface.normals[nearest[j]] @ n >= .45:
                reach += (progress[j]-reach)*(2*GRID-distance[j-1])/(distance[j]-distance[j-1])
        if reach <= GRID:
            continue
        query = pad + reach * bearing
        gap, end_n = signed_gap(patch, query, patch.predict(pad)[0][0])
        if abs(gap) <= .010 and end_n @ n > .45:
            goal = query - gap * end_n
            result.append(Region(int(finger), goal, end_n, goal[None],
                                 float(np.linalg.norm(goal-pad)), kind))
    return result


def rank_regions(regions, state, model, history, *, paired=False):
    """Approximate the terminal footprint from empirical joint finger responses."""
    p = state
    pads, hits, normals = p['pads'], p['hits'], p['normals']
    wrist, orient, surface = p['wrist_position'], p['wrist_rotation'], p['surface']
    owner = np.array([r.finger for r in regions])
    end = np.array([r.point for r in regions])
    n = np.array([r.normal for r in regions])
    travel = np.array([r.distance for r in regions])
    turn = np.array([align(normals[f], nn).as_rotvec() for f, nn in zip(owner, n, strict=True)])
    arm = wrist - pads[owner]
    end_w = end + Rotation.from_rotvec(turn).apply(arm)
    # Consult the bank only for the last half-second. Longer transfers are charged
    # in the denominator and followed incrementally on the observed mesh.
    last = np.minimum(1., HORIZON*SPEED_M_S/np.maximum(travel, 1.e-9))
    start_w = end_w + last[:, None]*(wrist-end_w)
    start_rot = Rotation.from_rotvec((1.-last[:, None])*turn) * orient
    actions = np.column_stack((start_rot.inv().apply(end_w-start_w), start_rot.inv().apply(last[:, None]*turn)))
    delta, low, weights = history.predict(p['q'], hits, normal_spread(normals, hits), actions)
    fractions = 1. - last[:, None]*(1.-FRACTIONS)
    rot = (Rotation.from_rotvec((fractions[..., None]*turn[:, None]).reshape(-1, 3)) * orient).as_matrix()
    rot = rot.reshape(len(regions), len(FRACTIONS), 3, 3)
    wrist_knots = wrist + fractions[..., None]*(end_w-wrist)[:, None]
    local = orient.inv().apply(pads-wrist)
    proposed = wrist_knots[:, None, :, None] + np.einsum('atij,ahtfj->ahtfi', rot, local+delta)
    # Each empirical response moves all four fingers together. Project near a
    # measured tangent plane and reject remote/unsupported apparent contacts.
    d, nearest = surface.tree.query(proposed)
    support_n = surface.normals[nearest]
    gap = np.sum((proposed-surface.points[nearest])*support_n, axis=-1)
    projected = proposed-gap[..., None]*support_n
    probability = low * np.exp(-.5*((gap+.0007)/.007)**2) * (d <= 1.6*BAND)
    gain, _ = joint_gain(projected, probability, weights, surface, model)
    contact = probability[..., None, :] > MASK_THRESHOLDS[:, :4]
    quorum = contact.sum(-1) >= 3
    p3 = np.sum(weights*quorum.mean(axis=(2,3)), axis=1)
    duration = HORIZON + travel/SPEED_M_S
    legal = np.linalg.norm(end_w-p['workspace_center'], axis=1) <= .45
    vertices = p['palm_vertices']
    future_palm = wrist_knots[:, :, None] + np.einsum('atij,vj->atvi', rot, orient.inv().apply(vertices-wrist))
    legal &= contact_first.palm_clear(surface, vertices, future_palm)
    arrived = probability[np.arange(len(regions)), :, -1, owner]
    owner_support = np.sum(weights*arrived, axis=1)
    if paired:
        # The same pair must survive both prediction times in a joint hand response.
        kept = contact.all(axis=2).sum(axis=-1) >= 2
        pair_support = np.sum(weights * kept.mean(axis=-1), axis=1)
        return gain, p3, duration, legal, owner_support, pair_support
    return gain, p3, duration, legal, owner_support


def unobserved_side(surface, pads, ids, normals, headings):
    """Soft fraction of the nearby observed footprint lying behind each proposed bearing."""
    numerator = np.ones(len(headings))
    denominator = 2.
    for finger in ids:
        idx = surface.tree.query_ball_point(pads[finger], 2*BAND)
        relative = surface.points[idx]-pads[finger]
        n = normals[finger]
        offset = relative-(relative@n)[:,None]*n
        radii = np.linalg.norm(offset, axis=1)
        weight = np.exp(-.5*(radii/BAND)**2)*(radii>VOXEL_M)*np.clip(surface.normals[idx]@n,0.,1.)**2
        shoulder = headings-(headings@n)[:,None]*n
        shoulder /= np.maximum(np.linalg.norm(shoulder,axis=1,keepdims=True),1.e-9)
        cosine = shoulder@(offset/np.maximum(radii[:,None],1.e-9)).T
        numerator += .5*(1.-cosine)@weight
        denominator += weight.sum()
    return numerator/denominator


class PosteriorGuide:
    def __init__(self, model, history, seed):
        self.model, self.history = model, history
        self.rng = np.random.default_rng(seed)
        self.region = None
        self.direction = self.normal = None
        self.batch, self.last_tick, self.cursor = 0, -GLOBAL_TICKS, 0
        self.refresh = True
        self.lost_since = self.owner_lost_since = None
        self.state = None

    def update(self, surface, tick, pads, ids, normal, wrist_rotation):
        ids = np.asarray(ids, dtype=int)
        event = None
        if not len(ids):
            if self.lost_since is None:
                self.lost_since = tick
            if self.state['observed_phase'] in ('withdraw', 'approach') or tick-self.lost_since >= 15:
                self.region = self.direction = None
                self.refresh = True
        else:
            self.lost_since = None
        if len(ids) and self.region is not None:
            owner = self.region.finger
            if owner in ids:
                self.owner_lost_since = None
            elif self.owner_lost_since is None:
                self.owner_lost_since = tick
            if (self.owner_lost_since is not None and tick-self.owner_lost_since >= 30
                    or owner in ids and np.linalg.norm(pads[owner]-self.region.point) < .006):
                self.refresh = True
        elapsed = tick-self.last_tick
        if surface.tree is not None and (elapsed >= GLOBAL_TICKS or (len(ids) and self.refresh and elapsed >= 25)):
            old = self.model.count
            self.model.update(surface, tick)
            self.batch += 1
            self.last_tick = tick
            self.region = None
            self.cursor = 0
            self.owner_lost_since = None
            self.refresh = not len(ids)
            event = dict(batch=self.batch, observed_tick=tick, old_cells=old, new_cells=self.model.count-old,
                         total_cells=self.model.count, mesh_version=self.model.mesh_version,
                         mesh_faces=len(self.model.mesh.faces), direction_chosen=False)
            if len(ids):
                self.choose(surface, pads, ids, normal, event)
                self.refresh = self.region is None
        self.normal, self.direction = normal.copy(), None
        if self.region is not None and len(ids):
            pad = pads[self.region.finger]
            path = self.region.path
            # A cursor on this posterior's connected mesh path, rebuilt on replan.
            while self.cursor < len(path)-1 and np.linalg.norm(path[self.cursor]-pad) < .012:
                self.cursor += 1
            heading = tangent(path[self.cursor]-pad, normal)
            if np.linalg.norm(heading) > .0003:
                self.direction = unit(heading)
            else:
                self.refresh = True
            if event is not None and self.direction is not None:
                event.update(direction_chosen=True, direction=self.direction.tolist(), normal=normal.tolist())
        return event

    def choose(self, surface, pads, ids, normal, event):
        normals = self.state['normals']
        regions = mesh_regions(self.model, pads, ids, normals)
        regions += gradient_regions(self.model, surface, pads, ids, normals, self.rng)
        if not regions:
            return
        gain, p3, duration, legal, owner_support = rank_regions(regions, self.state, self.model, self.history)
        # The target must lead away from currently measured pads in the shared tangent.
        first = [next((pt for pt in r.path if np.linalg.norm(pt-pads[r.finger]) >= .012),r.point)
                 for r in regions]
        headings = np.array([tangent(pt-pads[r.finger],normal) for pt,r in zip(first,regions)])
        size = np.linalg.norm(headings,axis=1)
        legal &= size > .002
        novelty = unobserved_side(surface,pads,ids,normals,headings)
        # At a new contact the empirical bank may predict no samples. Keep the
        # observed supporting finger as a single, explicit fallback hypothesis.
        score = gain*owner_support*novelty/duration
        fallback = not np.any(legal & (score > 1.e-9))
        if fallback and np.any(legal):
            sites = np.array([r.point for r in regions])
            prior, conditional = self.model.observation_covariances(sites)
            gain = .5*np.log((NOISE_VARIANCE+prior.diagonal())/(NOISE_VARIANCE+conditional.diagonal()))
            score = gain * np.clip(surface.tree.query(sites)[0]/BAND, 0, 1) * novelty / duration
        if getattr(self, 'recovery_observation', None) is not None:
            score *= self.after_loss_weights(regions)
        valid = np.flatnonzero(legal & (score > 0))
        if not len(valid):
            return
        best = valid[score[valid] >= .98*score[valid].max()]
        k = int(best[np.lexsort((-score[best], -p3[best]))[0]])
        self.region = r = regions[k]
        event.update(source=r.source, target_m=r.point.tolist(), target_normal=r.normal.tolist(), target_finger=r.finger,
                     pivot_m=pads[r.finger].tolist(), mesh_path_m=r.distance, candidates=len(regions),
                     acquisition_nat=float(gain[k]), selected_rate_nat_s=float(score[k]), predicted_light3=float(p3[k]),
                     predicted_owner_support=float(owner_support[k]), unobserved_side_weight=float(novelty[k]),
                     single_support_fallback=fallback)


class FieldFollower(multirate_wrist.SurfaceFollower):
    def __init__(self, device, seed):
        super().__init__(device, .95, 1.)
        self.guide = PosteriorGuide(self.global_model, IndexedHandHistory(HISTORY), seed)

    def select(self, **state):
        self.guide.state = state
        # Recompute the live bearing to the chosen point on the mesh at every
        # local step; preserve local_normal for support selection at edges.
        self.local_direction = None
        return super().select(**state)


class FieldWrist(multirate_wrist.MultirateWrist):
    def __init__(self, hand, data, touch, *, device, seed, **options):
        informative_wrist.InformativeWrist.__init__(self, hand, data, touch, **options)
        self.selector = FieldFollower(device, seed)
        self.flow = None


def run(folder, obj, seconds=240, device='cuda', prior_path=None, *, wrist_cls=FieldWrist,
        worker_cls=multirate_wrist.MultirateWorker, source_files=(), release='tactile-field'):
    code = informative_run.source_hashes()
    for module in (contact_first, multirate_wrist, online_tactile_mesh):
        path = Path(module.__file__)
        code[path.name] = digest(path)
    code[Path(__file__).name] = digest(Path(__file__))
    for path in map(Path, source_files):
        code[path.name] = digest(path)
    seed = seed_for(obj, OBJECTS[obj][2])
    constructor = partial(wrist_cls, device=device, seed=seed)
    worker = worker_cls
    if prior_path is not None:
        with np.load(prior_path, allow_pickle=False) as prior:
            points, normals = prior['points_m'], prior['normals']
        code['intervention-history.npz'] = digest(prior_path)

        class InterventionWorker(worker):
            def __init__(self, servo, paced):
                super().__init__(servo, paced)
                for point, n in zip(points, normals, strict=True):
                    cell = tuple(np.floor(point/VOXEL_M).astype(int))
                    self.map.samples.setdefault(cell, np.r_[point,n])

        worker = InterventionWorker
    with patch.multiple(informative_run, InformativeWrist=constructor, InformativeWorker=worker,
                        RELEASE=f'{release}-{device}', source_hashes=lambda:code):
        return informative_run.run(folder, obj, seconds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--object', choices=OBJECTS, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seconds', type=int, default=240)
    parser.add_argument('--device', choices=('cpu','cuda'), default='cuda')
    parser.add_argument('--prior', type=Path, help='Matched-history intervention runs only')
    args = parser.parse_args()
    print(run(args.out, args.object, args.seconds, args.device, args.prior), flush=True)
