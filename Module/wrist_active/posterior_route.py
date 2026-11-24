"""Choose a curved sweep using only the tactile posterior and a provisional convex shape prior."""
from itertools import combinations

import numpy as np

from ..acitive_02.aude import tangent, unit
from .multirate_wrist import GLOBAL_TICKS
from .online_tactile_mesh import BAND
from .strip_information import batch_information


def normalized(v):
    return v/np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


class TactileEnvelope:
    """Observed supporting planes complete a hand-scale prior where no contact exists yet."""
    def __init__(self, pads, normal, rotation):
        self.up = normal
        self.side = unit(max((tangent(x, normal) for x in rotation.as_matrix().T), key=np.linalg.norm))
        self.radius = np.clip(.46*np.max(np.linalg.norm(pads[:, None]-pads, axis=-1)), .045, .105)
        self.center = pads.mean(0)-self.radius*normal
        i = np.arange(78)+.5
        z = 1.-2*i/len(i)
        a = i*np.pi*(3.-np.sqrt(5.))
        self.directions = rotation.apply(np.column_stack((np.sqrt(1-z*z)*np.cos(a), np.sqrt(1-z*z)*np.sin(a), z)))

    def fit(self, surface):
        points, normals = surface.points, surface.normals
        probes = normals[np.linspace(0, len(points)-1, min(450, len(points)), dtype=int)]
        lateral = probes[abs(probes@self.up) < .5]
        if len(lateral) >= 8:
            test = normalized(lateral-(lateral@self.up)[:, None]*self.up)
            self.side = test[np.argmax((abs(test@test.T) > .96).sum(1))]
        axes = np.stack((self.side, np.cross(self.up, self.side), self.up), axis=1)
        pos, ns = points@axes, probes@axes
        lo, hi = pos.min(0), pos.max(0)
        half = np.maximum(.5*(hi-lo), self.radius)
        plus, minus = (ns > .75).sum(0) >= 2, (ns < -.75).sum(0) >= 2
        middle = np.where(plus & ~minus, hi-half, np.where(minus & ~plus, lo+half, (lo+hi)*.5))
        self.center += .5*(axes@middle-self.center)
        # Add actual measured plane orientations independent of the object's identity.
        unique = []
        for n in probes:
            if not unique or np.max(np.asarray(unique)@n) < .95:
                unique.append(n)
        self.planes = np.vstack((self.directions, unique))
        measured = (points-self.center)@self.planes.T
        alignment = (probes@self.planes.T).max(0)
        prior = self.radius*(.90+.14*(1.-np.clip(alignment, 0., 1.)))
        # Only a measured aligned normal removes the prior padding on that side.
        self.offset = np.maximum(.022, np.where(alignment>.96, measured.max(0),
                                                np.maximum(measured.max(0)+.002, prior)))

    def surface(self, ray):
        ray = normalized(ray)
        cosine = ray@self.planes.T
        radius = (self.offset/np.maximum(cosine, 1e-8)).min(-1)
        point = ray*radius[..., None]
        clearance = self.offset-point@self.planes.T
        weights = np.exp(-np.maximum(clearance, 0.)/.009)
        return self.center+point, normalized(weights@self.planes)


class PosteriorRoute:
    def __init__(self, model, pads, normal, rotation):
        self.model = model
        self.body = TactileEnvelope(pads, normal, rotation)
        # Use the closest physical triplet to locate the palm-side finger row.
        self.row = list(min(combinations(range(len(pads)), 3),
                            key=lambda ids: np.linalg.norm(np.ptp(pads[list(ids)], axis=0))))
        self.previous = None
        self.last_tick = -GLOBAL_TICKS
        self.batch = self.index = 0
        self.path = self.normals = None
        self.failed = []
        self.last_count = 0

    def update(self, surface, tick, pads, normal, rotation, light):
        if tick-self.last_tick < GLOBAL_TICKS or surface.tree is None:
            return None
        old = self.model.count
        self.model.update(surface, tick)
        self.last_tick, self.batch = tick, self.batch+1
        event = dict(batch=self.batch, observed_tick=tick, old_cells=old, new_cells=self.model.count-old,
                     total_cells=self.model.count, mesh_version=self.model.mesh_version,
                     mesh_faces=len(self.model.mesh.faces), direction_chosen=False)
        if not light:
            return event
        self.body.fit(surface)
        pivot = pads[self.row].mean(0)
        start = unit(pivot-self.body.center)
        axis = tangent(self.previous, start) if self.previous is not None else np.zeros(3)
        if np.linalg.norm(axis) < .25:
            axis = max((tangent(x, start) for x in rotation.as_matrix().T), key=np.linalg.norm)
        axis = unit(axis)
        angles = np.arange(20)*2*np.pi/20
        directions = np.cos(angles[:, None])*axis+np.sin(angles[:, None])*np.cross(start, axis)
        arc = np.linspace(0., 2.15, 32)
        rays = np.cos(arc[None, :, None])*start+np.sin(arc[None, :, None])*directions[:, None]
        paths, ns = self.body.surface(rays)
        stops = [3, 6, 10, 15, 21, 29]
        anchors, normals = paths[:, stops], ns[:, stops]
        # Parallel transport the observed finger footprint about each candidate strip.
        start_n = ns[:, :1]
        cross = np.cross(start_n, normals)
        def rotate(arm):
            return arm+np.cross(cross, arm)+np.cross(cross, np.cross(cross, arm))/np.maximum(1.+(start_n*normals).sum(-1, keepdims=True), .05)
        sites = np.stack([anchors+rotate(pad-pivot) for pad in pads], axis=2)
        points, local = self.body.surface(sites-self.body.center)
        distance = surface.tree.query(points.reshape(-1, 3))[0].reshape(points.shape[:-1])
        raw = points.reshape(-1, 3)
        gp_normals = local.reshape(-1, 3)
        near = distance.ravel() < BAND
        prob = np.full(len(raw), .58)
        if near.any():
            delta = .004*gp_normals[near]
            low, high = np.split(self.model.metric_distance(np.vstack((raw[near]-delta, raw[near]+delta))), 2)
            slope = (high-low)/.008
            offset = -.5*(low+high)/np.maximum(slope, .15)
            good = (slope>.22) & (abs(offset)<.012)
            idx = np.flatnonzero(near)[good]
            raw[idx] += offset[good, None]*gp_normals[idx]
            prob[idx] = .95
        clearance = np.linalg.norm(sites-points, axis=-1)
        quality = np.exp(-np.maximum(clearance-.006, 0.)/.045)
        new = np.clip(surface.tree.query(raw)[0].reshape(quality.shape)/.014, 0., 1.)
        gain = batch_information(self.model, raw.reshape(len(paths), -1, 3),
                                 (prob.reshape(quality.shape)*new**2*quality).reshape(len(paths), -1))
        length = np.linalg.norm(np.diff(paths, axis=1), axis=2).sum(1)
        turn = np.zeros(len(paths)) if self.previous is None else np.arccos(np.clip(directions@axis, -1., 1.))
        score = gain/(.75+length/.072+.28*turn)
        if self.previous is not None and self.model.count-self.last_count < 15:
            self.failed = (self.failed+[(self.start_point.copy(), self.previous.copy(), tick)])[-12:]
        for old_pos, old_d, at in self.failed:
            weight = np.exp(-np.linalg.norm(pivot-old_pos)/.050-(tick-at)*.01/32.)
            score *= 1.-.85*weight*np.maximum(0., directions@old_d)**2
        pick = int(np.argmax(score))
        self.path, self.normals, self.index = paths[pick], ns[pick], 0
        self.start_point, self.previous, self.last_count = pivot.copy(), directions[pick].copy(), self.model.count
        event.update(direction_chosen=True, source='posterior_curved_strip', pivot_m=pivot.tolist(),
                     direction=self.previous.tolist(), normal=normal.tolist(), acquisition_nat=float(gain[pick]),
                     predicted_strip_m=float(length[pick]), target_m=self.path[-1].tolist(),
                     estimated_center_m=self.body.center.tolist())
        return event

    def waypoint(self, pivot, lead):
        hi = min(len(self.path), self.index+7)
        self.index += int(np.argmin(np.linalg.norm(self.path[self.index:hi]-pivot, axis=1)))
        i = self.index
        while i+1 < len(self.path) and np.linalg.norm(self.path[i]-self.path[self.index]) < lead:
            i += 1
        return self.path[i], self.normals[i]
