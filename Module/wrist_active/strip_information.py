"""Choose a sustained wrist heading by expected tactile strip information per second."""
from collections import deque

import numpy as np
import torch

from ..acitive_02.active_gpis import NOISE_VARIANCE
from ..acitive_02.aude import tangent, unit
from .multirate_wrist import GLOBAL_TICKS, transport
from .online_tactile_mesh import BAND


def batch_information(model, sites, reliability):
    """Mutual information of noisy future pad locations, conditional on the full tactile history."""
    candidates, observations, _ = sites.shape
    with torch.inference_mode():
        phi = model.features(sites.reshape(-1, 3))
        white = torch.linalg.solve_triangular(model.chol, phi.T, upper=False).T
        white = white.reshape(candidates, observations, -1)
        weights = torch.as_tensor(reliability, device=white.device, dtype=white.dtype)
        features = white * (weights / NOISE_VARIANCE).sqrt()[:, :, None]
        covariance = features @ features.transpose(1, 2)
        covariance.diagonal(dim1=1, dim2=2).add_(1.)
        return torch.linalg.cholesky(covariance).diagonal(dim1=1, dim2=2).log().sum(1).cpu().numpy()


class StripGuide:
    """Use only observed contact normals, their cumulative GP posterior and past outcomes."""
    def __init__(self, model):
        self.model = model
        self.direction = self.normal = None
        self.batch = 0
        self.last_tick = -GLOBAL_TICKS
        self.last_start = None
        self.last_count = 0
        self.failed = deque(maxlen=16)

    def update(self, surface, tick, pads, ids, normal, rotation, pad_normals=None):
        if tick-self.last_tick < GLOBAL_TICKS or surface.tree is None:
            return None
        old = self.model.count
        self.model.update(surface, tick)
        count = len(surface.points)
        assert self.model.count == count
        self.batch += 1
        self.last_tick = tick
        event = dict(batch=self.batch, observed_tick=tick, old_cells=old, new_cells=count-old,
                     total_cells=count, mesh_version=self.model.mesh_version,
                     mesh_faces=len(self.model.mesh.faces), direction_chosen=False)
        if not len(ids):
            return event
        tip = pads[ids]
        ns = surface.normals[surface.tree.query(tip)[1]] if pad_normals is None else pad_normals[ids]
        center = tip.mean(0)
        prev = None if self.direction is None else transport(self.direction, self.normal, normal)
        if self.last_start is not None and count-self.last_count < 10:
            self.failed.append((self.last_start.copy(), self.direction.copy(), tick))
        basis = prev if prev is not None and np.linalg.norm(prev)>.8 else unit(
            max((tangent(a, normal) for a in rotation.as_matrix().T), key=np.linalg.norm))
        stride = np.arange(16)*np.pi/8
        directions = list(np.cos(stride[:, None])*basis + np.sin(stride[:, None])*np.cross(normal,basis))
        route_dirs, route_rates = [], []
        for pos,n in zip(tip,ns,strict=True):
            for route in self.model.mesh.routes(pos,n):
                chord = tangent(route.waypoint-pos,normal)
                if np.linalg.norm(chord)>.002:
                    route_dirs.append(unit(chord))
                    route_rates.append(route.rate)
        directions = np.array(directions+route_dirs)
        # The reach of one command is measured in centimetres. Each strip has
        # the measured pad spacing; its later stations are less certain on open boundaries.
        horizon = .105 if len(ids)>=2 else .052
        distances = horizon*np.array([.10,.23,.39,.57,.78,1.])
        along = directions[:,None,:]-(directions@ns.T)[:,:,None]*ns[None,:,:]
        sites = tip[None,:,None,:]+along[:,:,None,:]*distances[None,None,:,None]
        flat = sites.reshape(-1,3)
        distance,nearest = surface.tree.query(flat)
        nearest_normal = surface.normals[nearest]
        local_normals = np.broadcast_to(ns[None,:,None,:],sites.shape).reshape(-1,3)
        agreement = np.einsum('ij,ij->i',nearest_normal,local_normals)
        # Project supported sites to the GP isosurface. Farther out we test a
        # short tangent extrapolation and attenuate its chance of an observation.
        offset = .004*local_normals
        lo,hi = np.split(self.model.metric_distance(np.vstack((flat-offset,flat+offset))),2)
        slope = (hi-lo)/.008
        correction = -.5*(hi+lo)/np.maximum(slope,.15)
        trusted = (distance < 1.3*BAND) & (agreement>.3) & (slope>.15) & (abs(correction)<.014)
        projected = flat + np.where(trusted,correction,0.)[:,None]*local_normals
        dist = surface.tree.query(projected)[0].reshape(sites.shape[:-1])
        expected_contact = np.where(trusted,1.,.58*np.exp(-np.maximum(distance-BAND,0.)/.060))
        expected_contact *= np.where((distance<.015)&(agreement<.10),.25,1.)
        expected_contact = expected_contact.reshape(dist.shape)
        # A path retracing the same track carries little new area even when a
        # sub-voxel shift would make its contact lattice indices different.
        novelty = np.clip(dist/.012,0.,1.)
        reliability = expected_contact * novelty**2 * np.linalg.norm(along,axis=2)[:,:,None]
        gain = batch_information(self.model,projected.reshape(len(directions),-1,3),
                                 reliability.reshape(len(directions),-1))
        # Estimate the union of parallel 10-mm-radius tip tracks. This corrects
        # a direction that would put several pads one behind another.
        across = np.cross(normal,directions)
        offsets = np.sort((tip-center)@across.T,axis=0)
        width = .020+np.minimum(np.diff(offsets,axis=0),.020).sum(0)
        turn = np.zeros(len(gain)) if prev is None else np.arccos(np.clip(directions@prev,-1,1))
        score = gain*width/(len(tip)*.020)/(1.+.12*turn)
        if route_rates and np.max(score)>0:
            follow = np.maximum(0.,directions@np.array(route_dirs).T)**4
            score *= 1.+.18*(follow*np.array(route_rates)/max(route_rates)).max(1)
        for start,heading,at in self.failed:
            near = np.exp(-np.linalg.norm(center-start)/.040-(tick-at)*.01/30.)
            score *= 1.-.75*near*np.maximum(directions@heading,0.)**3
        pick = int(np.argmax(score))
        self.direction, self.normal = directions[pick].copy(), normal.copy()
        self.last_start, self.last_count = center.copy(), count
        event.update(direction_chosen=True,source='conditional_strip_information',direction=self.direction.tolist(),
                     normal=normal.tolist(),pivot_m=center.tolist(),acquisition_nat=float(gain[pick]),
                     predicted_strip_m=float(horizon),effective_width_m=float(width[pick]),
                     mesh_routes=len(route_dirs))
        return event
