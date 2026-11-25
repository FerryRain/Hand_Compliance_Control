"""Palm exploration with an optional six-way search and online mesh-driven recontact."""

import argparse
import json
from collections import deque
from dataclasses import dataclass
from functools import partial
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from scipy.spatial.transform import Rotation

from ..acitive_02.surface_scan import align, limit
from . import entry_frontier, paired_support, palm_acquisition, palm_exploration as palm
from . import palm_surface, palm_unload, policy_reach, tactile_field, transition_compute
from .contact_first import palm_clear
from .contact_transition import ContactPose
from .palm_surface import OUT, normalize
from .sweep import DT, LOAD_LIMIT
from .tactile_reentry import Sample


WEAK_TICKS = 75
RETRY_TICKS = 75
MAX_PATCHES = 6


@dataclass(frozen=True)
class RecoveryInput:
    weak_ticks: int
    station: ContactPose | None
    material_sites: np.ndarray
    failed: tuple


def patch_seeds(mesh, points, reference, live_center, normal, failed):
    """Nominate a bounded set of distinct, observed surface patches for the whole palm."""
    distance = np.linalg.norm(points - live_center, axis=1)
    normals = mesh.normals
    valid = ((distance <= .115) & (normals @ normal > .50) & (mesh.contact_distance <= .018))
    for missed in failed:
        valid &= np.linalg.norm(points - missed, axis=1) > .020
    ranked = np.argsort(np.where(valid, distance + .7 * mesh.contact_distance, np.inf), kind="stable")
    seeds = [reference] if valid[reference] else []
    for idx in ranked:
        if not valid[idx] or len(seeds) == MAX_PATCHES:
            break
        if not seeds or np.linalg.norm(points[seeds] - points[idx], axis=1).min() >= .018:
            seeds.append(int(idx))
    return seeds


def clear_approach(state, rotation, target, outward):
    """Check the outward waypoint and terminal palm path against observed contact planes."""
    start, attitude = state["wrist_position"], state["wrist_rotation"]
    way = target + .010 * outward
    fractions = np.array([.25, .5, .75, 1.])
    positions = np.vstack((start + fractions[:, None] * (way - start), (way + target) / 2, target))
    if np.max(np.linalg.norm(positions - state["workspace_center"], axis=1)) > .447:
        return False
    delta = (rotation * attitude.inv()).as_rotvec()
    angles = Rotation.from_rotvec(np.r_[fractions, 1., 1.][:, None] * delta)
    relative = state["palm_vertices"] - start
    plates = positions[:, None] + np.einsum("aij,vj->avi", angles.as_matrix(), relative)
    return bool(palm_clear(state["surface"], state["palm_vertices"], plates[None])[0])


def mesh_target(state, model, patches, packet):
    mesh, station = model.mesh, packet.station
    audit = dict(weak_ticks=packet.weak_ticks, template_tick=station.tick if station is not None else None,
                 candidates=0, feasible=0, mesh_version=int(model.mesh_version),
                 mesh_faces=len(mesh.faces) if mesh is not None else 0)
    if mesh is None or not len(mesh.faces) or station is None:
        return None, audit
    points = mesh.centers - model.band.half_width * mesh.normals
    cloud_center = station.points.mean(axis=0)
    outward = normalize(station.normals.sum(axis=0))
    if np.linalg.norm(outward) < .5:
        outward = station.attitude.apply(OUT)
    dist = np.linalg.norm(points - cloud_center, axis=1)
    source_ok = (mesh.normals @ outward > .40) & (dist < .080) & (mesh.contact_distance <= .018)
    if not source_ok.any():
        return None, audit
    reference = int(np.argmin(np.where(source_ok, dist, np.inf)))
    source = points[reference]
    current, attitude = state["wrist_position"], state["wrist_rotation"]
    local = attitude.inv().apply(packet.material_sites - current)
    live_center = packet.material_sites.mean(axis=0) + source - cloud_center
    seeds = patch_seeds(mesh, points, reference, live_center, mesh.normals[reference], packet.failed)
    audit["candidates"] = len(seeds)
    best, best_merit = None, -np.inf
    for face in seeds:
        normal = mesh.normals[face]
        turn = align(mesh.normals[reference], normal)
        pose = turn * station.attitude
        angle = (pose * attitude.inv()).magnitude()
        if angle > 1.12:
            continue
        center = points[face] + turn.apply(cloud_center - source)
        levers = pose.apply(local)
        wrist = center - levers.mean(axis=0)
        pads = wrist + levers
        patch = patches.query(model, state["surface"], pads, turn.apply(station.normals), .030)
        valid = patch.mesh & (np.sum(patch.normals * turn.apply(station.normals), axis=1) > .5)
        if valid.sum() < 2:
            continue
        # Fit one wrist translation to the simultaneous signed pad gaps on the mesh.
        # Fingers remain under single_all. Measured-cloud fallback patches cannot vote.
        n = patch.normals[valid]
        gap = np.sum((pads[valid] - patch.points[valid]) * n, axis=1)
        correction = limit(np.linalg.solve(n.T @ n + .18 * np.eye(3), -n.T @ gap), .014)
        wrist += correction
        delta = wrist + levers - patch.points
        gaps = np.sum(delta * patch.normals, axis=1)
        lateral = np.linalg.norm(delta - gaps[:, None] * patch.normals, axis=1)
        actual_distance = state["surface"].tree.query(patch.points)[0]
        fits = valid & (abs(gaps) <= .008) & (lateral <= .021) & (actual_distance <= .022)
        travel = np.linalg.norm(wrist - current)
        if fits.sum() < 2 or travel > .125 or not clear_approach(state, pose, wrist, normal):
            continue
        quality = np.exp(-.5 * ((gaps[fits] / .006)**2 + (lateral[fits] / .016)**2
                                + (actual_distance[fits] / .022)**2))
        cost = travel / .075 + angle
        merit = float(quality.sum() / (.50 + cost))
        audit["feasible"] += 1
        if merit > best_merit:
            best_merit = merit
            best = dict(face=int(face), patch_m=points[face].tolist(), outward=normal.tolist(),
                        wrist_m=wrist.tolist(), wrist_xyzw=pose.as_quat().tolist(), merit=merit,
                        expected_pads=station.fingers[fits].tolist(), sites_m=patch.points[fits].tolist(),
                        normals=patch.normals[fits].tolist(), travel_m=float(travel),
                        deadline_ticks=round(np.clip(1.1 + 1.5 * cost, 1.2, 3.5) / DT))
    return best, audit


class PalmGuide(palm.PalmDirectionGuide):
    def __init__(self, previous, reach, *, compact):
        super().__init__(previous, reach)
        self.compact = compact
        self.recovery_input = None
        self.active_target = None

    def evaluate(self, state, pivot, normal, headings, speed, scale):
        if self.compact:
            headings = np.concatenate((headings[:1], headings[1::2]))
        self.moving_candidates += len(headings) - 1
        return super().evaluate(state, pivot, normal, headings, speed, scale)

    def choose(self, state, ids, normal, tick, event):
        self.moving_candidates = 0
        super().choose(state, ids, normal, tick, event)
        event["candidates"] = self.moving_candidates

    def update(self, surface, tick, pads, ids, normal, wrist_rotation):
        if self.active_target is not None:
            event = super().update(surface, tick, pads, [], normal, wrist_rotation)
            goal = self.active_target
            if "palm_surface_m" in goal:
                wrist = np.asarray(goal["wrist_m"])
                pose = Rotation.from_quat(goal["wrist_xyzw"])
                n = np.asarray(goal["palm_surface_normal"])
                region = palm.PalmMove(wrist + pose.apply(self.site), self.site, n)
                self.region = region
                self.palm_goal = palm.Landing(region, wrist, pose, np.asarray(goal["palm_surface_m"]), n,
                                              1., self.site, goal["palm_height_m"])
            return event
        packet = self.recovery_input
        if packet is None or surface.tree is None:
            return super().update(surface, tick, pads, ids, normal, wrist_rotation)
        began = perf_counter()
        old = self.model.count
        self.model.update(surface, tick)
        target, audit = mesh_target(self.state, self.model, self.patches, packet)
        if target is not None:
            pose = Rotation.from_quat(target["wrist_xyzw"])
            center = np.asarray(target["wrist_m"]) + pose.apply(self.site)
            n = np.asarray(target["outward"])
            projected = center - n * ((center - target["patch_m"]) @ n)
            patch = self.patches.query(self.model, surface, projected, n, .082)
            height = float((center - patch.points[0]) @ pose.apply(OUT))
            if patch.mesh[0] and .018 <= height <= .18:
                target.update(palm_surface_m=patch.points[0].tolist(), palm_surface_normal=patch.normals[0].tolist(),
                              palm_height_m=height)
        audit["query_ms"] = 1000 * (perf_counter() - began)
        if target is None:
            event = super().update(surface, tick, pads, ids, normal, wrist_rotation)
            if event is not None:
                event["mesh_recontact_query"] = audit
                return event
        else:
            self.region = self.palm_goal = self.direction = None
            self.holding = False
            self.unsuccessful_hold = None
        self.batch += 1
        self.last_tick = tick
        return dict(batch=self.batch, observed_tick=tick, old_cells=old, total_cells=self.model.count,
                    new_cells=self.model.count - old, mesh_version=self.model.mesh_version,
                    mesh_faces=audit["mesh_faces"], direction_chosen=False, mesh_recontact_query=audit,
                    source="mesh_recontact" if target else "mesh_recontact_unavailable", mesh_target=target)


class MeshWrist(palm.PalmReachWrist):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weak_ticks = self.retry_after = 0
        self.mesh_attempt = None
        self.mesh_finished = []
        self.failed_patches = deque(maxlen=8)
        self.free_reference_frames = 0

    def observe(self, data, touch, tick):
        before_partial = self.partial_returns
        waypoints = tuple(self.path) if self.mesh_attempt is not None else ()
        super().observe(data, touch, tick)
        weak = touch.loads.max() < LOAD_LIMIT and touch.measured_heads.sum() < 2
        self.weak_ticks = self.weak_ticks + 1 if weak else 0
        if self.mesh_attempt is None:
            return
        issued, anchor, goal = self.mesh_attempt
        if self.unloading or touch.loads.max() >= LOAD_LIMIT:
            reason = "load_relief"
        elif np.count_nonzero(self.dwell >= 8) >= 2:
            reason = "pair"
        elif tick - issued >= goal["deadline_ticks"]:
            reason = "timeout"
        elif self.anchor is not anchor:
            if self.anchor is None and self.partial_returns > before_partial:
                # One pad on new surface completes the ordinary local return. The global
                # mesh target still asks for a pair. Continue the same bounded approach.
                self.anchor = anchor
                self.path = deque(waypoints)
                self.began_return = tick
                self.start_cells = len(self.surface.samples)
                return
            reason = "local_return_ended"
        else:
            return
        if self.anchor is anchor:
            self.leave_return(tick)
        if reason != "pair":
            self.failed_patches.append((tick, np.asarray(goal["patch_m"])))
        self.mesh_finished.append(dict(issued_tick=issued, end_tick=tick, reason=reason,
                                       free_reference_frames=self.free_reference_frames))
        self.mesh_attempt = None
        self.retry_after = tick + RETRY_TICKS

    def command(self, touch):
        if (self.mesh_attempt is None or self.transition is None or touch.measured_heads.any()
                or self.unloading or touch.loads.max() >= LOAD_LIMIT or touch.loads[-1] > .65):
            return super().command(touch)
        # With zero pads there is no measured pivot. Feed the existing pose servo
        # the selected mesh wrist reference and rotate about the wrist in free space.
        # Actual pad contact immediately restores its ordinary tactile tilt/height input.
        _, _, target = self.mesh_attempt
        ordinary = self.transition
        self.transition = dict(ordinary, palm_reference_valid=True, palm_target_xyzw=target["wrist_xyzw"],
                               palm_patch_m=target.get("palm_surface_m", target["patch_m"]),
                               palm_patch_outward=target.get("palm_surface_normal", target["outward"]),
                               palm_pivot_m=self.measured_position, palm_trim_velocity=[0., 0., 0.])
        self.free_reference_frames += 1
        try:
            return super().command(touch)
        finally:
            self.transition = ordinary

    def can_recontact(self, touch, tick):
        return (self.weak_ticks >= WEAK_TICKS and tick >= self.retry_after and self.mesh_attempt is None
                and self.anchor is None and not self.unloading and self.release_velocity is None
                and touch.loads.max() < 4.5 and touch.loads[-1] < .45 and touch.measured_heads.sum() < 2)


class MeshWorker(palm_unload.UnloadWorker):
    def submit(self, data, touch, tick, lead):
        servo = self.servo
        servo.selector.guide.active_target = servo.mesh_attempt[2] if servo.mesh_attempt is not None else None
        packet = None
        if servo.can_recontact(touch, tick):
            def cost(s):
                return (np.linalg.norm(s.position - servo.measured_position)
                        + .04 * (s.attitude * servo.live_attitude.inv()).magnitude() - .012 * len(s.fingers))
            station = min(servo.stations, key=cost, default=None)
            material = np.empty((0, 3))
            if station is not None:
                ids = station.fingers
                material = servo.origins[ids] + np.einsum("fij,fj->fi", servo.matrices[ids], station.material)
            failed = tuple(point for stamp, point in servo.failed_patches if tick - stamp < 500)
            packet = RecoveryInput(servo.weak_ticks, station, material, failed)
        servo.selector.guide.recovery_input = packet
        return super().submit(data, touch, tick, lead)

    def install(self, tick, *, initial=False):
        row = super().install(tick, initial=initial)
        if row is None:
            return None
        servo = self.servo
        if servo.mesh_finished:
            row["mesh_recontact_ended"] = servo.mesh_finished
            servo.mesh_finished = []
        event = row["global_event"] or {}
        if "mesh_recontact_query" not in event:
            return row
        target = event.get("mesh_target")
        ready = servo.can_recontact(servo.touch, tick)
        servo.retry_after = tick + RETRY_TICKS
        row["mesh_recontact_installed"] = bool(target is not None and ready)
        if not row["mesh_recontact_installed"]:
            return row
        position = np.asarray(target["wrist_m"])
        normal = np.asarray(target["outward"])
        attitude = Rotation.from_quat(target["wrist_xyzw"])
        anchor = Sample(tick, position, attitude, normal, len(target["expected_pads"]))
        # The native return executor adds a further 3 mm to a transfer waypoint.
        servo.path = deque([Sample(tick, position + .007 * normal, attitude, normal, 0)])
        servo.anchor = anchor
        servo.at_station = servo.began_return = servo.last_progress = tick
        servo.last_error = np.inf
        servo.start_cells = len(servo.surface.samples)
        servo.rescue = servo.flow = None
        servo.reentries += 1
        servo.mesh_attempt = (tick, anchor, target)
        servo.free_reference_frames = 0
        return row


def run(folder, obj, seconds=90, device="cuda", *, compact=True, recontact=True):
    if not compact and not recontact:
        return palm.run(folder, obj, seconds, device)
    reach = policy_reach.PolicyReach()
    if reach.metadata["checkpoint"] != json.loads(policy_reach.PROTOCOL.read_bytes())["checkpoint"]:
        raise ValueError("Reach model and finger policy do not match")
    guide = partial(PalmGuide, compact=compact)
    cls = MeshWrist if recontact else palm.PalmReachWrist
    worker = MeshWorker if recontact else palm_unload.UnloadWorker
    sources = {Path(sys.modules[c.__module__].__file__) for c in (*cls.__mro__, *worker.__mro__)
               if c.__module__.startswith("Module.wrist_active.")}
    for module in (palm, palm_unload, policy_reach, entry_frontier, paired_support, palm_acquisition,
                   palm_surface, transition_compute):
        sources.add(Path(module.__file__))
    sources.update((Path(__file__), policy_reach.MODEL))
    name = f"palm-{6 if compact else 12}-direction-{'mesh' if recontact else 'local'}-recontact"
    if recontact:
        name += "-reference"
    return tactile_field.run(folder, obj, seconds, device, wrist_cls=partial(cls, reach=reach, guide_cls=guide),
                             worker_cls=worker, source_files=sorted(sources), release=name)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--object", choices=palm.OBJECTS, required=True)
    args.add_argument("--out", type=Path, required=True)
    args.add_argument("--seconds", type=int, default=90)
    args.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args.add_argument("--directions", type=int, choices=(6, 12), default=6)
    args.add_argument("--local-recontact-only", action="store_true")
    options = args.parse_args()
    print(run(options.out, options.object, options.seconds, options.device, compact=options.directions == 6,
              recontact=not options.local_recontact_only), flush=True)
