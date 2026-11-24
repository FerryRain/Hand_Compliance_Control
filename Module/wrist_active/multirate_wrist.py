"""Hold an information-selected heading while a faster tactile servo follows the surface."""
from time import perf_counter

import numpy as np
from scipy.spatial.transform import Rotation

from ..acitive_02.active_gpis import NOISE_VARIANCE, VOXEL_M
from ..acitive_02.aude import tangent, unit
from ..acitive_02.surface_scan import limit
from ..i04_geometric_traversal.i01_a_contact_wrist import LEVER_M, PRELOAD_M
from .contact_first import MAX_INWARD_M, signed_gap
from .informative_wrist import InformativeWorker, InformativeWrist
from .online_tactile_mesh import BAND, ContactBand, StreamingGPIS
from .sweep import DT, LOAD_LIMIT, PLAN_TICKS


LOCAL_TICKS = 5
GLOBAL_TICKS = 400
SPEED_M_S = .045
LOCAL_S = LOCAL_TICKS * DT


def transport(direction, old, new):
    """Minimal rotation between two tangent planes, including a sharp edge."""
    c = np.cross(old, new)
    dot = np.clip(old @ new, -1., 1.)
    if dot > -.9999:
        direction = direction + np.cross(c, direction) + np.cross(c, np.cross(c, direction)) / (1. + dot)
    return unit(tangent(direction, new))


def contact_frame(pads, normals, ids, previous=None):
    ns = normals[ids]
    # Choose an observed side if an edge has opposing normals. A group of pads
    # on the same face supplies a better tangent than averaging opposing faces.
    agreement = ns @ ns.T
    votes = (agreement > .35).sum(1).astype(float)
    if previous is not None:
        votes += .35 * (ns @ previous)
    same = ids[agreement[votes.argmax()] > .35]
    normal = unit(normals[same].mean(axis=0))
    return same, pads[same].mean(axis=0), normal


class GlobalHeading:
    def __init__(self, device, band):
        self.model = StreamingGPIS(device, band)
        self.direction = self.normal = None
        self.last_tick = -GLOBAL_TICKS
        self.batch = 0
        self.last_supported_pivot = None
        self.previous_count = 0

    def update(self, surface, tick, pads, ids, normal, wrist_rotation):
        if tick - self.last_tick < GLOBAL_TICKS or surface.tree is None:
            return None
        # Append just the suffix to the previous sufficient statistics. Both
        # the mesh and candidate novelty use the entire cumulative tactile map.
        old_count = self.model.count
        self.model.update(surface, tick)
        assert self.model.count == len(surface.points)
        self.batch += 1
        self.last_tick = tick
        event = dict(batch=self.batch, observed_tick=tick, old_cells=old_count,
                     new_cells=self.model.count-old_count, total_cells=self.model.count,
                     mesh_version=self.model.mesh_version, mesh_faces=len(self.model.mesh.faces),
                     direction_chosen=False)
        if not len(ids):
            return event
        first = (transport(self.direction, self.normal, normal) if self.direction is not None else
                 unit(max((tangent(axis, normal) for axis in wrist_rotation.as_matrix().T), key=np.linalg.norm)))
        angles = np.arange(12) * (np.pi/6)
        directions = list(np.cos(angles[:,None])*first + np.sin(angles[:,None])*np.cross(normal, first))
        routes = []
        for pad in pads[ids]:
            routes.extend((pad, route) for route in self.model.mesh.routes(pad, normal))
        route_dirs, rates = [], []
        for pad, route in routes:
            vector = tangent(route.waypoint - pad, normal)
            if np.linalg.norm(vector) > .002:
                route_dirs.append(unit(vector))
                rates.append(route.rate)
        directions = np.array([*directions, *route_dirs])
        # Test strips ahead of the actual contact footprint. Count information
        # only near observations with a compatible, signed GPIS crossing.
        stops = np.array([.007, .014, .022, .031])
        sites = pads[ids][None,:,None,:] + directions[:,None,None,:]*stops[None,None,:,None]
        pts = sites.reshape(-1, 3)
        distance, near = surface.tree.query(pts)
        normals = surface.normals[near]
        delta = .004 * normals
        low, high = np.split(self.model.metric_distance(np.vstack((pts-delta, pts+delta))), 2)
        derivative = (high-low)/.008
        correction = -.5*(low+high)/np.maximum(derivative,.12)
        projected = pts + np.clip(correction,-.010,.010)[:,None]*normals
        dist = surface.tree.query(projected)[0]
        valid = ((distance <= 1.35*BAND) & (derivative > .15) & (np.abs(correction) <= .010)
                 & (normals @ normal > .35) & (dist <= 1.35*BAND))
        per = len(ids) * len(stops)
        gain = np.zeros(len(directions))
        valid_ids = np.flatnonzero(valid)
        if len(valid_ids):
            prior, conditional = self.model.observation_covariances(projected[valid])
            for idx in np.unique(valid_ids//per):
                which = np.flatnonzero(valid_ids//per == idx)
                sub = np.ix_(which, which)
                noise = NOISE_VARIANCE*np.eye(len(which))
                sign_a, log_a = np.linalg.slogdet(prior[sub]+noise)
                sign_b, log_b = np.linalg.slogdet(conditional[sub]+noise)
                if min(sign_a, sign_b) <= 0:
                    raise ValueError('Invalid directional GP covariance')
                novel = np.clip(dist[valid_ids[which]] / (.6*BAND), 0, 1)
                gain[idx] = max(0., .5*(log_a-log_b)) * novel.mean()
        score = gain / max(gain.max(), 1e-12)
        if rates:
            alignment = np.maximum(0., directions @ np.asarray(route_dirs).T)**4
            score += .35*(alignment*(np.asarray(rates)/max(rates))).max(1)
        if not np.any(score):
            # No supported zero crossing yet: probe a short observed tangent,
            # comparing spacing to every previous tactile cell.
            score = np.minimum(distance.reshape(len(directions),len(ids),-1)[:,:,:2], BAND).mean((1,2))/BAND
            source = 'tactile_bootstrap'
        else:
            source = 'global_gpis'
        pivot = pads[ids].mean(0)
        if self.direction is not None:
            stalled = (self.last_supported_pivot is not None and self.model.count-self.previous_count < 8 and
                       np.linalg.norm(pivot-self.last_supported_pivot) < .012)
            score += (-.45 if stalled else .12) * (directions @ first)
        pick = int(np.argmax(score))
        self.direction, self.normal = directions[pick].copy(), normal.copy()
        self.previous_count, self.last_supported_pivot = self.model.count, pivot.copy()
        event.update(direction_chosen=True, source=source, direction=self.direction.tolist(), normal=normal.tolist(),
                     pivot_m=pivot.tolist(), acquisition_nat=float(gain[pick]), mesh_routes=len(route_dirs))
        return event


class SurfaceFollower:
    def __init__(self, device, level, force):
        self.band = ContactBand(level)
        self.guide = GlobalHeading(device, self.band)
        self.global_model = self.guide.model
        self.force = force
        self.tip_forces = np.zeros(4)
        self.local_direction = self.local_normal = None

    def select(self, *, surface, q, hits, normals, wrist_position, wrist_rotation,
               pads, palm_vertices, loads, tick, workspace_center, observed_phase):
        ids = np.flatnonzero(hits)
        if len(ids):
            ids, pivot, normal = contact_frame(pads, normals, ids, self.local_normal)
        else:
            normal = self.local_normal if self.local_normal is not None else wrist_rotation.apply([0.,0.,1.])
        light = observed_phase == 'scan' and np.max(loads) < LOAD_LIMIT
        began = perf_counter()
        event = self.guide.update(surface, tick, pads, ids if light else [], normal, wrist_rotation)
        global_ms = 1000*(perf_counter()-began) if event is not None else 0.
        h = self.guide
        info = dict(batch=h.batch, global_event=event, global_ms=global_ms, local_period_s=LOCAL_S,
                    gpis_device=str(h.model.device), contact_iso=self.band.level,
                    anchor_ids=[], tangent_speed_m_s=0., height_speed_m_s=0., angular_velocity=[0.,0.,0.])
        position, rotation, direction = wrist_position, wrist_rotation, np.zeros(3)
        if not light or not len(ids) or h.direction is None:
            info['mode'] = 'withdraw' if observed_phase=='withdraw' or max(loads)>=LOAD_LIMIT else 'recontact'
        else:
            if (event is not None and event['direction_chosen']) or self.local_direction is None:
                heading = transport(h.direction,h.normal,normal)
            else:
                heading = transport(self.local_direction,self.local_normal,normal)
            self.local_direction, self.local_normal = heading.copy(), normal.copy()
            speed = SPEED_M_S * (1. if len(ids) >= 2 else .55)
            shift = heading * speed * LOCAL_S
            axes = wrist_rotation.as_matrix()[:, :2]
            rows, gaps, bend = [], [], np.zeros(3)
            for i in range(len(pads)):
                measured = i in ids
                if hits[i] and not measured:
                    continue
                patch = surface.patch(pads[i], preferred_normal=normals[i] if measured else normal)
                if patch is None:
                    continue
                start, ns = signed_gap(patch, pads[i])
                if not measured and (ns @ normal < .55 or abs(start) > .009):
                    continue
                base = patch.predict(pads[i])[0][0] if measured else 0.
                gap, n = signed_gap(patch, pads[i]+shift, base)
                preload = np.clip(PRELOAD_M*(self.force-self.tip_forces[i]),0.,MAX_INWARD_M) if measured and loads[-1]<.35 and max(loads)<4. else 0.
                # Missing fingers close a fraction of their measured gap. The
                # measured pads dominate the height and two tilt coordinates.
                w = 1. if measured else .32
                rows.append(w*np.r_[n@normal, np.cross(pads[i]-pivot,n)@axes/LEVER_M])
                gaps.append(w*(gap+preload if measured else .35*gap))
                if measured:
                    bend += np.cross(ns,n)/len(ids)
            if rows:
                j = np.asarray(rows)
                reg = np.diag([.10,.25,.25])**2
                preferred = np.r_[0., LEVER_M*(bend@axes)]
                fit = np.linalg.solve(j.T@j+reg, reg@preferred-j.T@np.asarray(gaps))
                # Admittance in the underlying 100 Hz loop also responds to load.
                dh = float(np.clip(fit[0], -min(.00065,self.band.half_width), .001))
                turn = limit(axes@fit[1:]/LEVER_M, .055)
                rotation = Rotation.from_rotvec(turn)*wrist_rotation
                position = pivot+shift+dh*normal+Rotation.from_rotvec(turn).apply(wrist_position-pivot)
                direction = heading
                info.update(mode='explore', anchor_ids=ids.tolist(), tangent_speed_m_s=speed,
                            height_speed_m_s=dh/LOCAL_S, angular_velocity=(turn/LOCAL_S).tolist())
            else:
                info['mode'] = 'stabilize'
        return dict(position=position, rotation=rotation, normal=normal, direction=direction, diagnostics=info)


class MultirateWrist(InformativeWrist):
    def __init__(self, hand, data, touch, *, device, level, target_force, **options):
        super().__init__(hand, data, touch, **options)
        self.selector = SurfaceFollower(device, level, target_force)
        self.flow = None

    def observe(self, data, touch, tick):
        super().observe(data,touch,tick)
        if not self.unloading and max(touch.loads) >= LOAD_LIMIT:
            hits = touch.measured_heads
            i = int(np.argmax(touch.loads))
            self.relief = touch.normals[i] if i<len(hits) and hits[i] else self.normal
            self.unloading, self.rescue, self.clear_ticks, self.good_ticks = True, None, 0, 0
            self.escape_count += 1

    def command(self, touch):
        ids = np.flatnonzero(touch.measured_heads)
        if self.flow is not None and self.plan_mode=='explore' and self.phase()=='scan' and len(ids):
            row, frame_normal = self.flow
            _, pivot, n = contact_frame(touch.points,touch.normals,ids,frame_normal)
            heading = transport(np.asarray(row['selected_direction']), frame_normal,n)
            # Rebase the velocity on this packet. Normal correction and tilt can
            # change without invalidating tangential motion along the held heading.
            remain = max(1,self.apply_tick+PLAN_TICKS-self.tick)*DT
            omega = np.asarray(row['angular_velocity'])
            rot = Rotation.from_rotvec(DT*omega)
            height = row['height_speed_m_s']
            if max(touch.loads) >= 4. or touch.loads[-1] >= .35:
                height = max(0.,height)
            lever = self.measured_position-pivot
            velocity = row['tangent_speed_m_s']*heading + height*n + (rot.apply(lever)-lever)/DT
            self.goal = self.position + remain*velocity
            self.goal_attitude = Rotation.from_rotvec(remain*omega)*self.attitude
            self.normal = n
        else:
            self.goal, self.goal_attitude = self.position.copy(), self.attitude
        return super().command(touch)


class MultirateWorker(InformativeWorker):
    lead = 1

    def submit(self, data, touch, tick, lead):
        self.servo.selector.tip_forces = touch.forces.copy()
        super().submit(data,touch,tick,lead)

    def install(self, tick, *, initial=False):
        row = super().install(tick,initial=initial)
        if row is not None:
            self.servo.flow = (row,self.servo.normal.copy()) if row['mode']=='explore' else None
        return row

    def update(self, data, touch, tick, length):
        self.observe(data,touch,tick)
        row = self.install(tick)
        if tick%LOCAL_TICKS==LOCAL_TICKS-self.lead and tick+self.lead<length and self.pending is None:
            self.submit(data,touch,tick,self.lead)
        return row
