"""Rank reachable wrist poses by marginal information, footprint overlap and physical residence."""
from copy import copy
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from Module.wrist_active.palm_progress_recontact import ProgressWorker
from park3.common import DT, rot
from park3.deployment import Worker as GlobalWorker
from park3.geometry import Clearance, anchors, candidates, spaced, uncertain_surface
from park3.planner import ms, scale
from planner_adapter import PalmWrist, PalmWorker


class Residence:
    """Decaying seconds of actual contact and unsuccessful pursuit, per oriented surface cell."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.cells = {}
        self.seen = set()
        self.last_discovery = 0
        self.previous = None

    def key(self, p, n):
        axis = int(np.argmax(abs(n)))
        return (*np.floor(p/self.cfg.memory_cell_m).astype(int), axis, int(n[axis] > 0))

    def add(self, p, n, tick, seconds, channel):
        key = self.key(p, n)
        if key not in self.cells:
            self.cells[key] = np.r_[p, n, 0., 0., tick]
        cell = self.cells[key]
        cell[6:8] *= np.exp2(-(tick-cell[8])*DT/self.cfg.memory_half_life_s)
        cell[6+channel] += seconds
        cell[8] = tick
        return key

    def observe(self, touch, wrist, target, tick):
        cfg = self.cfg
        if tick % cfg.sample_ticks:
            return
        duration = cfg.sample_ticks*DT if tick else 0.
        valid = (touch.measured_heads & (touch.forces >= .2) & np.isfinite(touch.points).all(1)
                 & (abs(np.linalg.norm(touch.normals,axis=1)-1) < .01))
        if np.isfinite(touch.loads).all() and max(touch.loads) < 8 and valid.any():
            contacts = list(zip(touch.points[valid], touch.normals[valid]))
            self.previous = (wrist.copy(), contacts)
            for p,n in contacts:
                key = self.add(p,n,tick,duration/len(contacts),0)
                if key not in self.seen:
                    self.seen.add(key)
                    self.last_discovery = tick
        elif self.previous is not None and np.linalg.norm(wrist-self.previous[0]) < cfg.memory_cell_m:
            contacts = self.previous[1]
            for p,n in contacts:
                self.add(p,n,tick,duration/len(contacts),0)
        if target is not None and (tick-self.last_discovery)*DT >= cfg.unproductive_s:
            self.add(np.asarray(target["point"]),np.asarray(target["normal"]),tick,duration,1)

    def snapshot(self, tick):
        cells = np.array(list(self.cells.values())).reshape(-1,9)
        cells[:,6:8] *= np.exp2(-(tick-cells[:,8,None])*DT/self.cfg.memory_half_life_s)
        return cells


def surface_cost(surface, points, normals, memory, cfg):
    xyz, ns = surface.points, surface.normals
    near = cKDTree(points).query_ball_point(xyz, 3*cfg.overlap_radius_m)
    overlap = np.zeros(len(xyz))
    for i, neighbors in enumerate(near):
        ids = np.asarray(neighbors, dtype=int)
        ids = ids[(normals[ids] @ ns[i]) > .5]
        if len(ids):
            distance2 = np.square(points[ids]-xyz[i]).sum(1).min()
            overlap[i] = np.exp(-.5*distance2/cfg.overlap_radius_m**2)
    influence = np.exp(-.5*cdist(xyz,memory[:,:3],"sqeuclidean")/cfg.dwell_radius_m**2)
    influence *= np.clip((ns @ memory[:,3:6].T - .25)/.55, 0, 1)
    seconds = influence @ memory[:,6:8]
    penalties = 1 - np.exp(-np.square(seconds/[cfg.dwell_scale_s,cfg.failed_scale_s]))
    return overlap, seconds, penalties


def global_frontier(surface, priority, actual_contact, cfg):
    """Pick the highest-utility region in the sensed component and guide the first feasible hop."""
    ix = spaced(surface.points,priority,cfg.region_spacing_m,len(priority))
    xyz, ns = surface.points[ix], surface.normals[ix]
    edges = cdist(xyz,xyz)
    edges[(edges > cfg.graph_edge_m) | (ns @ ns.T < .15)] = 0.
    start = np.linalg.norm(xyz-actual_contact,axis=1).argmin()
    distance = dijkstra(edges,directed=False,indices=start)
    best = int(np.argmax(np.where(np.isfinite(distance),priority[ix],-np.inf)))
    to_goal = dijkstra(edges,directed=False,indices=best)
    progress = np.clip((distance[best]-to_goal)/cfg.graph_edge_m,-1,1)
    return ix, int(ix[best]), progress, float(distance[best])


class FrontierPlanner:
    """Use PARK-3's posterior, composite probe, 6-DoF candidates and checked trajectories."""
    def __init__(self, original, cfg):
        self.base, self.cfg, self.variant = original, cfg, original.variant
        self.sampling = copy(original.cfg)
        self.sampling.support_sites, self.sampling.surface_count = cfg.support_sites, cfg.surface_count
        self.memory = np.empty((0,9))

    def plan(self,tick,splice,path,actual,pads,hull,points,normals,incumbent):
        began = perf_counter()
        base, cfg = self.base, self.cfg
        original = base.cfg
        row = dict(tick=tick,splice_tick=splice,cells=len(points),variant="D",revision="v2")
        def finish(reason,tail=None):
            row.update(reason=reason,planning_ms=ms(began))
            return row,tail
        if not len(points):
            return finish("awaiting_contact")
        started = perf_counter()
        row["update_mode"] = base.gp.condition(points,normals)
        row["gpis_update_ms"] = ms(started)
        started = perf_counter()
        refs = anchors(base.plate,pads,original.anchor_width_fraction)
        contact = actual[:3]+rot(actual).apply(refs[0])
        surface = uncertain_surface(base.gp,points,normals,contact,self.sampling)
        row["gpis_query_ms"] = ms(started)
        if surface is None:
            return finish("no_supported_crossing")
        started = perf_counter()
        overlap, seconds, sat = surface_cost(surface,points,normals,self.memory,cfg)
        noise = base.gp.scale**2 * base.gp.cfg["noise"]
        # A scalar Gaussian observation gives 1/2 log(1 + posterior variance / observation noise).
        info = .5*np.log1p(surface.variance/noise)
        fresh_info = info*(1-overlap)
        neighborhood = np.exp(-.5*cdist(surface.points,surface.points,"sqeuclidean")/cfg.dwell_radius_m**2)
        neighborhood *= np.maximum(surface.normals @ surface.normals.T,0)
        patch_info = (neighborhood @ fresh_info) / neighborhood.sum(1)
        priority = (scale(patch_info) - cfg.overlap_weight*overlap
                    - cfg.dwell_weight*sat[:,0] - cfg.failed_weight*sat[:,1])
        regions, global_goal, progress, distance = global_frontier(surface,priority,contact,cfg)
        surface.regions = regions
        poses, owners = candidates(surface,actual,base.origin,refs,original)
        active = incumbent is not None and splice < path.end
        if active:
            poses = np.vstack((poses,path.at(path.end)))
            owners = np.r_[owners, regions[np.linalg.norm(surface.points[regions]-incumbent["point"],axis=1).argmin()]]
        if not len(poses):
            return finish("outside_workspace")
        clearance = Clearance(points,normals,hull,actual,original)
        legal = clearance.valid(poses)
        has_incumbent = active and legal[-1]
        poses,owners = poses[legal],owners[legal]
        row.update(candidate_ms=ms(started),feasible=len(poses),regions_count=len(regions),
                   global_frontier_m=surface.points[global_goal].tolist(),global_route_distance_m=distance)
        if not len(poses):
            return finish("no_clear_endpoint")
        started = perf_counter()
        attitudes = rot(poses)
        matrices = attitudes.as_matrix()
        local = np.einsum("cpi,cij->cpj",surface.points[None]-poses[:,None,:3],matrices)
        k = perf_counter()
        footprint = base.probe.fingers(local).max(axis=-1)
        row["kde_ms"] = ms(k)
        mass = footprint.sum(1)
        # A zero-mass pose must not appear novel just because it cannot sense anything.
        normalized = footprint / np.maximum(mass[:,None],1e-12)
        gain = footprint @ fresh_info
        redundancy = np.where(mass>1e-12, normalized @ overlap, 1.)
        dwell = np.maximum(sat[owners,0], normalized @ sat[:,0])
        failed = np.maximum(sat[owners,1], normalized @ sat[:,1])
        routing = np.zeros(len(surface.points))
        routing[regions] = progress
        motion = (np.linalg.norm(poses[:,:3]-actual[:3],axis=1)/original.translation_scale_m)**2
        motion += original.gamma*((rot(actual).inv()*attitudes).magnitude()/original.rotation_scale_rad)**2
        alignment = ((1+np.clip(matrices[:,:,2] @ surface.normals.T,-1,1))*.5) @ info / info.sum()
        value = (scale(gain) - cfg.overlap_weight*redundancy - cfg.dwell_weight*dwell
                 - cfg.failed_weight*failed + cfg.global_progress_weight*routing[owners]
                 + original.orientation_weight*alignment - original.motion_weight*scale(motion))
        for j in np.argsort(-value,kind="stable"):
            keep = has_incumbent and (j == len(value)-1 or value[j] <= value[-1]+original.switch_margin)
            if keep:
                j = len(value)-1
            owner = owners[j]
            row.update(marginal_info=float(gain[j]),footprint_overlap=float(redundancy[j]),
                       residence_s=float(seconds[owner,0]),unproductive_pursuit_s=float(seconds[owner,1]),
                       dwell_penalty=float(cfg.dwell_weight*dwell[j]),failed_penalty=float(cfg.failed_weight*failed[j]),
                       objective=float(value[j]),route_progress=float(routing[owner]),evaluation_ms=ms(started))
            if keep:
                return finish("keep_current_target")
            tail = path.revised(splice,poses[j],original)
            if tail is None or np.linalg.norm(tail.samples[:,:3]-base.origin[:3],axis=1).max() > original.workspace_radius_m:
                continue
            if not all(clearance.valid(block).all() for block in np.array_split(tail.samples,1+len(tail.samples)//96)):
                continue
            row.update(selected=int(j),target=poses[j].tolist(),end_tick=tail.end,
                       point=surface.points[owner].tolist(),normal=surface.normals[owner].tolist(),
                       variance=float(surface.variance[owner]),trigger="geometry" if active else "segment_end",
                       evaluation_ms=ms(started))
            return finish("new_target",tail)
        row["evaluation_ms"] = ms(started)
        return finish("no_clear_trajectory")


class WatchedWorker(GlobalWorker):
    def __init__(self,wrist,paced):
        super().__init__(wrist,paced)
        self.memory = Residence(wrist.planner.cfg)

    def observe(self,data,touch,tick):
        super().observe(data,touch,tick)
        self.memory.observe(touch,data.qpos[:3],self.target,tick)

    def submit(self,data,touch,tick,lead):
        self.wrist.planner.memory = self.memory.snapshot(tick)
        super().submit(data,touch,tick,lead)

    def update(self,data,touch,tick,length):
        row = super().update(data,touch,tick,length)
        if (self.pending is None and (tick-self.latest)*DT >= self.memory.cfg.watchdog_s
                and tick+self.lead < length):
            self.submit(data,touch,tick,self.lead)
        return row


class FrontierWrist(PalmWrist):
    def __init__(self,*args,frontier,**kwargs):
        super().__init__(*args,**kwargs)
        self.global_wrist.planner = FrontierPlanner(self.global_wrist.planner,SimpleNamespace(**frontier))


class FrontierWorker(PalmWorker):
    def __init__(self,servo,paced):
        ProgressWorker.__init__(self,servo,paced)
        self.global_worker = WatchedWorker(servo.global_wrist,paced)
        self.reference = None
