"""Select a wrist pose that preserves measured support before ranking information.

All lengths are metres. The planner has access to the hand, past tactile
observations and the frozen hand-response library. Its planning mesh is tactile.
"""
from dataclasses import dataclass
from itertools import combinations
from time import perf_counter

import numpy as np
from scipy.spatial.transform import Rotation

from ..acitive_02.active_gpis import TRUST_RADIUS_M, VOXEL_M
from ..acitive_02.aude import tangent, unit
from ..acitive_02.surface_scan import limit
from ..i04_geometric_traversal.physics import rotation as wrist_attitude
from .history import KNOTS, normal_spread
from .informative_actions import EXECUTE, HORIZON, MASK_THRESHOLDS, InformativeActions, joint_gain
from .informative_wrist import InformativeWorker, InformativeWrist
from .online_tactile_mesh import ContactBand, MeshGoal, StreamingGPIS
from .planner import ANGLES, PIVOTS, STEP_M
from .sweep import LOAD_LIMIT
from ..i04_geometric_traversal.i01_a_contact_wrist import LEVER_M, PRELOAD_M


# Two checkpoints inside the command actually issued to the servo.
FRACTIONS = np.array([.5, 1.])
GAP_LIMIT = VOXEL_M / 2
SETTLE_TICKS = 8  # Consecutive 100 Hz light-contact measurements.
MAX_INWARD_M = .0015
TARGET_FORCE_N = 1.


@dataclass
class Candidate:
    position: np.ndarray
    turn: np.ndarray
    anchors: tuple
    recruit: int | None
    shift: np.ndarray
    route: MeshGoal | None = None


def signed_gap(patch, positions, reference=0.):
    shape = positions.shape
    value, gradient = patch.predict(positions.reshape(-1, 3))
    magnitude = np.linalg.norm(gradient, axis=-1)
    gap = (value - reference) / np.maximum(magnitude, .5)
    normal = gradient / np.maximum(magnitude[:, None], 1e-10)
    return gap.reshape(shape[:-1]), normal.reshape(shape)


def palm_clear(surface, vertices, proposed):
    distance, idx = surface.tree.query(proposed)
    normals = surface.normals[idx]
    gap = np.sum((proposed - surface.points[idx]) * normals, axis=-1)
    # Compare the original and proposed vertex to the SAME measured plane.
    # An already close vertex may move away or slide along the plane.
    start = vertices - surface.points[idx]
    before = np.sum(start * normals, axis=-1)
    near_start = np.linalg.norm(start, axis=-1) < .025
    floor = np.where(near_start, np.minimum(.001, before) - .0002, .001)
    return ~np.any((distance < .020) & (gap < floor), axis=(1, 2))


class ContactFirstActions:
    def __init__(self, history, device="cpu", contact_iso=.95, target_force_n=TARGET_FORCE_N):
        if not 0. < target_force_n < LOAD_LIMIT:
            raise ValueError("The fingertip force target must be positive and below the light-load limit")
        self.history = history
        self.target_force = target_force_n
        self.band = ContactBand(contact_iso)
        self.global_model = StreamingGPIS(device, self.band)
        self.heading = None
        self.settled = np.zeros(4, dtype=bool)
        self.tip_forces = None

    def pair_strokes(self, patches, pads, normals, pair, wrist, rotation, remote, preload):
        a, b = pair
        pivot = (pads[a] + pads[b]) / 2
        normal = unit(normals[a] + normals[b])
        if np.linalg.norm(normal) < .5:
            normal = normals[a]
        heading = tangent(self.heading, normal) if self.heading is not None else np.zeros(3)
        if np.linalg.norm(heading) < .1:
            heading = max((tangent(axis, normal) for axis in rotation.as_matrix().T), key=np.linalg.norm)
        heading = unit(heading)
        directions = list(np.cos(ANGLES)[:, None] * heading + np.sin(ANGLES)[:, None] * np.cross(normal, heading))
        routes = [None] * len(directions)
        for route in remote:
            # The mesh is the OUTER response isosurface. Move the tactile
            # footprint toward the center of the band before fitting preload.
            chord = route.waypoint - self.band.half_width * route.waypoint_normal - pivot
            along = tangent(chord, normal)
            if np.linalg.norm(along) > .001:
                # Preserve the reconstructed height at the next mesh waypoint.
                directions.append(chord / max(STEP_M, np.linalg.norm(chord)))
                routes.append(route)
        # Try a shorter stroke too: curvature or an edge can invalidate the full step.
        shifts = (np.asarray(directions)[:, None] * np.array([STEP_M, STEP_M / 2])[None, :, None]).reshape(-1, 3)
        bend = np.zeros_like(shifts)
        for finger in pair:
            patch = patches[finger]
            _, start_n = signed_gap(patch, pads[finger])
            _, end_n = signed_gap(patch, pads[finger] + shifts)
            bend += np.cross(start_n, end_n) / len(pair)
        bend *= np.minimum(1., .08 / np.maximum(np.linalg.norm(bend, axis=1), 1e-10))[:, None]
        line = unit(pads[b] - pads[a])
        turns = (bend[:, None] + PIVOTS[None, :, None] * line).reshape(-1, 3)
        shifts = np.repeat(shifts, len(PIVOTS), axis=0)
        turns_world = Rotation.from_rotvec(turns)
        # I01's rigid footprint fit: normal height and two palm tilt angles.
        # Use measured local signed gaps in place of the reference collision mesh.
        axes = rotation.as_matrix()[:, :2]
        jacobians, residuals = [], []
        for finger in pair:
            patch = patches[finger]
            lever = turns_world.apply(pads[finger] - pivot)
            at = pivot + shifts + lever
            gap, outward = signed_gap(patch, at, patch.predict(pads[finger])[0][0])
            jacobians.append(np.column_stack((outward @ normal, np.cross(lever, outward) @ axes / LEVER_M)))
            residuals.append(gap + preload[finger])
        j = np.stack(jacobians, axis=1)
        jt = j.transpose(0, 2, 1)
        correction = np.linalg.solve(jt @ j + np.diag(np.square([.08, .24, .24])),
                                     -jt @ np.stack(residuals, axis=1)[:, :, None])[:, :, 0]
        height = correction[:, 0].clip(-GAP_LIMIT, GAP_LIMIT)
        tilt = correction[:, 1:] @ axes.T / LEVER_M
        tilt *= np.minimum(1., .045 / np.maximum(np.linalg.norm(tilt, axis=1), 1e-10))[:, None]
        fitted = Rotation.from_rotvec(tilt) * turns_world
        positions = pivot + shifts + height[:, None] * normal + fitted.apply(wrist - pivot)
        return [Candidate(pos, turn, pair, None, shift, routes[i // (2*len(PIVOTS))])
                for i, (pos, turn, shift) in enumerate(zip(positions, fitted.as_rotvec(), shifts))]

    @staticmethod
    def acquire_strokes(surface, patches, pads, normals, anchor, wrist, preload):
        candidates = []
        pivot = pads[anchor]
        for finger, pad in enumerate(pads):
            if finger == anchor:
                continue
            patch = patches[finger]
            if patch is None:
                # Probe toward an actual observation. A missing patch never counts as contact.
                correction = surface.points[surface.tree.query(pad)[1]] - pad
            else:
                gap, normal = signed_gap(patch, pad)
                correction = (-preload[finger] - gap) * normal
            arm = pad - pivot
            turn = limit(np.cross(arm, correction) / max(arm @ arm, 1e-8), .08)
            slide = limit(tangent(correction, normals[anchor]), STEP_M / 2)
            for shift, rot in ((np.zeros(3), turn), (slide, turn / 2), (slide, np.zeros(3))):
                position = pivot + shift + Rotation.from_rotvec(rot).apply(wrist - pivot)
                candidates.append(Candidate(position, rot, (anchor,), finger, shift))
        return candidates

    def select(self, *, surface, q, hits, normals, wrist_position, wrist_rotation,
               pads, palm_vertices, loads, tick, workspace_center=None, observed_phase="scan", tip_forces=None):
        hits = np.asarray(hits, dtype=bool)
        ids = np.flatnonzero(hits)
        wrist = np.asarray(wrist_position)
        normal = unit(np.sum(normals[hits], axis=0))
        if np.linalg.norm(normal) < .5:
            normal = normals[ids[0]] if len(ids) else unit(wrist - np.mean(pads, axis=0))
        if not len(ids) and surface.tree is not None:
            normal = surface.normals[surface.tree.query(np.mean(pads, axis=0))[1]]
        if np.linalg.norm(normal) < .5:
            normal = wrist_rotation.apply([0., 0., 1.])
        model = self.global_model
        model.update(surface, tick)
        mesh_info = dict(gpis_version=model.version, mesh_version=model.mesh_version, mesh_tick=model.mesh_tick,
                         mesh_faces=len(model.mesh.faces) if model.mesh else 0,
                         contact_iso=self.band.level, band_half_width_m=float(self.band.half_width),
                         gpis_device=str(model.device), map_timing=dict(model.timing))

        def pause(mode, reason, count=0):
            result = InformativeActions.recontact(wrist, wrist_rotation, normal, loads, count)
            if mode == "stabilize":
                result["position"] = wrist.copy()
            elif mode == "withdraw":
                result["position"] = wrist + .005 * normal
            result["diagnostics"].update(mode=mode, reason=reason, anchors=[], recruit=None, **mesh_info)
            return result

        if observed_phase == "withdraw" or np.max(loads) >= LOAD_LIMIT:
            return pause("withdraw", "measured load")
        if not len(ids) or observed_phase != "scan" or surface.tree is None:
            return pause("recontact", "no available tactile support")
        anchors_now = np.flatnonzero(hits & self.settled)
        if len(anchors_now) < min(len(ids), 2):
            return pause("stabilize", "waiting for consecutive light-contact measurements")

        patches = [surface.patch(pad, preferred_normal=normals[i] if hits[i] else None)
                   for i, pad in enumerate(pads)]
        tip_forces = self.tip_forces if tip_forces is None else tip_forces
        preload = np.zeros(4)
        if tip_forces is not None and max(loads) < 4. and loads[-1] < .35:
            preload = np.clip(PRELOAD_M * (self.target_force - np.asarray(tip_forces)), 0.,
                              min(MAX_INWARD_M, self.band.half_width))
        absolute_start = np.array([signed_gap(patch, pads[i])[0] if patch is not None else 0.
                                   for i,patch in enumerate(patches)])
        preload = np.minimum(preload, np.maximum(0., MAX_INWARD_M + absolute_start))
        candidates = []
        if len(ids) == 1:
            if patches[ids[0]] is not None:
                candidates = self.acquire_strokes(surface, patches, pads, normals, ids[0], wrist, preload)
        else:
            for pair in combinations(anchors_now, 2):
                if all(patches[i] is not None for i in pair):
                    center = pads[list(pair)].mean(axis=0)
                    outward = unit(normals[list(pair)].sum(axis=0))
                    remote = model.mesh.routes(center, outward) if model.mesh else []
                    candidates.extend(self.pair_strokes(patches, pads, normals, pair,
                                                        wrist, wrist_rotation, remote, preload))
        if not candidates:
            return pause("stabilize", "missing anchor patches")

        turns = np.array([c.turn for c in candidates])
        displacement = np.array([c.position - wrist for c in candidates])
        xyz = wrist + FRACTIONS[None, :, None] * displacement[:, None]
        relative = Rotation.from_rotvec((turns[:, None] * FRACTIONS[None, :, None]).reshape(-1, 3))
        matrix = relative.as_matrix().reshape(-1, 2, 3, 3)
        rigid = xyz[:, :, None] + np.einsum("ctij,fj->ctfi", matrix, pads - wrist)
        hull = xyz[:, :, None] + np.einsum("ctij,pj->ctpi", matrix, palm_vertices - wrist)
        safe = palm_clear(surface, palm_vertices, hull)
        safe &= np.linalg.norm(displacement, axis=1) <= .040
        if workspace_center is not None:
            safe &= np.linalg.norm(xyz - workspace_center, axis=-1).max(axis=1) <= .45

        gap = np.full(rigid.shape[:-1], np.inf)
        outward = np.zeros_like(rigid)
        observed = np.zeros_like(gap, dtype=bool)
        start_error = np.zeros(4)
        for i, patch in enumerate(patches):
            if patch is None:
                start_error[i] = surface.tree.query(pads[i])[0]
                continue
            value = patch.predict(pads[i])[0][0] if hits[i] else 0.
            start_error[i] = abs(signed_gap(patch, pads[i], value)[0] + preload[i])
            gap[:, :, i], outward[:, :, i] = signed_gap(patch, rigid[:, :, i], value)
            projected = rigid[:, :, i] - gap[:, :, i, None] * outward[:, :, i]
            distance, idx = surface.tree.query(projected)
            aligned = np.sum(outward[:, :, i] * surface.normals[idx], axis=-1) > .35
            observed[:, :, i] = (distance < TRUST_RADIUS_M) & aligned

        anchor_ids = np.array([c.anchors for c in candidates])
        rows = np.arange(len(candidates))[:, None]
        residual = np.abs(gap).max(axis=1)[rows, anchor_ids].max(axis=1)
        safe &= observed.all(axis=1)[rows, anchor_ids].all(axis=1) & (self.band.response(residual) >= self.band.level)
        # Bound inward travel in millimetres, independently of the iso level.
        # Retain the measured zero offset where an anchor already touches.
        absolute = gap + np.where(hits, absolute_start, 0.)
        floor = np.minimum(-MAX_INWARD_M, absolute_start)
        safe &= np.all((absolute >= floor[None,None,:] - 1e-7) | ~observed, axis=(1, 2))
        valid = np.flatnonzero(safe)
        if not len(valid):
            return pause("stabilize", "no wrist pose preserves the measured support", len(candidates))

        # Interpolate the first 250 ms hand-response knot to the 100 and 200 ms
        # checkpoints. The history indexes the wrist displacement over 500 ms.
        inv = wrist_rotation.inv()
        action = np.c_[inv.apply(displacement[valid]), inv.apply(turns[valid])] * (HORIZON / EXECUTE)
        delta, light, weights = self.history.predict(q, hits, normal_spread(normals, hits), action)
        alpha = FRACTIONS * EXECUTE / KNOTS[0]
        world = matrix[valid] @ wrist_rotation.as_matrix()
        motion = np.einsum("ctij,chfj->chtfi", world, delta[:, :, 0]) * alpha[None, None, :, None, None]
        distance = gap[valid, None] + np.sum(motion * outward[valid, None], axis=-1)
        prior = hits + alpha[None, None, :, None] * (light[:, :, :1] - hits)
        response = self.band.response(distance)
        prob = prior * response * (response >= self.band.level)
        # Unsupported points use their finite rigid locations as placeholders;
        # they have zero probability and never enter the information update.
        projection = rigid[valid, None] + motion - np.where(np.isfinite(distance), distance, 0.)[..., None] * outward[valid, None]
        nearest, index = surface.tree.query(projection)
        aligned = (surface.normals[index] * outward[valid, None]).sum(axis=-1) > .35
        prob *= observed[valid, None] & (nearest < TRUST_RADIUS_M) & aligned
        # Keep a joint hand-response example and each stratified mask fixed at
        # both checkpoints, so the supported fingers must have the same IDs.
        persistent = prob.min(axis=2)[:, :, None, :] > MASK_THRESHOLDS[:, :4]
        anchors = anchor_ids[valid]
        kept = np.take_along_axis(persistent, anchors[:, None, None, :], axis=-1).all(axis=-1)
        support = np.sum(weights * kept.mean(axis=-1), axis=1)
        # This is an empirical ranking score, not a calibrated success probability.
        # If support is weak, the robot can keep sensing at the present pose.
        credible = support >= .35
        if not credible.any():
            return pause("stabilize", "insufficient predicted anchor retention", len(candidates))

        gain = np.zeros(len(valid))
        new_cells = np.zeros(len(valid))
        if len(ids) == 1:
            fingers = np.array([candidates[i].recruit for i in valid])
            modeled = np.array([patches[i] is not None for i in fingers])
            credible &= ~modeled | observed[valid, -1, fingers]
            final_error = np.abs(gap[valid, -1, fingers] + preload[fingers])
            unknown = ~modeled
            if unknown.any():
                final_error[unknown] = surface.tree.query(rigid[valid[unknown], -1, fingers[unknown]])[0]
            improvement = start_error[fingers] - final_error
            credible &= improvement > .0001
            if not credible.any():
                return pause("stabilize", "no safe progress toward a second pad", len(candidates))
            near = modeled & (self.band.response(gap[valid,-1,fingers]) >= self.band.level)
            merit = improvement / np.maximum(start_error[fingers], self.band.half_width)
            # First obtain a locally supported second contact, then maximize progress.
            choices = np.flatnonzero(credible)
            rank = np.lexsort((support[choices], merit[choices], modeled[choices], near[choices]))
            best = choices[rank[-1]]
            mode = "acquire_pair" if modeled[best] else "probe_pair"
        else:
            choices = np.flatnonzero(credible)
            # Only mesh goals admitting a safe, supported wrist prefix compete.
            # The local directions remain available at bootstrap or on a disconnected patch.
            guided = [i for i in choices if candidates[valid[i]].route is not None]
            if guided:
                best_route = max(candidates[valid[i]].route.rate for i in guided)
                choices = np.array([i for i in guided if candidates[valid[i]].route.rate >= .95 * best_route])
            gain[choices], new_cells[choices] = joint_gain(projection[choices], prob[choices], weights[choices],
                                                           surface, self.global_model)
            near = choices[gain[choices] >= .98 * gain[choices].max()]
            follow = np.zeros(len(near)) if self.heading is None else np.array([
                unit(candidates[valid[i]].shift) @ self.heading for i in near])
            kept_three = kept[near] & (persistent[near].sum(axis=-1) >= 3)
            third = np.sum(weights[near] * kept_three.mean(axis=-1), axis=1)
            rank = np.lexsort((follow, gain[near], support[near], third))
            best = near[rank[-1]]
            mode = "explore"

        chosen = candidates[valid[best]]
        direction = unit(chosen.shift)
        anchor_normal = unit(normals[list(chosen.anchors)].sum(axis=0))
        if np.linalg.norm(anchor_normal) > .5:
            normal = anchor_normal
        if mode == "explore":
            self.heading = direction.copy()
        info = dict(mode=mode, candidate_count=len(candidates), feasible_count=int(credible.sum()),
                    anchors=list(map(int, chosen.anchors)), recruit=chosen.recruit,
                    max_anchor_gap_m=float(residual[valid[best]]), support_score=float(support[best]),
                    min_anchor_response=float(self.band.response(residual[valid[best]])),
                    requested_preload_m={int(i):float(preload[i]) for i in (
                        chosen.anchors if chosen.recruit is None else (chosen.recruit,)) if preload[i]>0},
                    gain_nat=float(gain[best]), expected_new_cells=float(new_cells[best]),
                    frontier=chosen.route is not None, prefix_s=EXECUTE, **mesh_info)
        if chosen.route is not None:
            route = chosen.route
            info.update(mesh_target_face=route.face, mesh_target_m=route.center.tolist(),
                        mesh_face_center_m=model.mesh.centers[route.face].tolist(), mesh_target_response=route.response,
                        mesh_target_normal=route.normal.tolist(), mesh_waypoint_m=route.waypoint.tolist(),
                        mesh_travel_m=route.distance, mesh_region_score=route.rate)
        if len(ids) == 1:
            info.update(start_error_m=float(start_error[fingers[best]]), end_error_m=float(final_error[best]))
        else:
            masks = prob[best, :, :, None, :] > MASK_THRESHOLDS[:, :4]
            light_three = weights[best] @ (masks.sum(axis=-1) >= 3).mean(axis=(1, 2))
            info["winner"] = dict(direction=direction.tolist(), light3=float(light_three),
                                  frontier=info["frontier"], gain_nat=info["gain_nat"],
                                  rate=info["gain_nat"] / EXECUTE, duration_s=EXECUTE,
                                  expected_new_cells=info["expected_new_cells"])
        return dict(position=chosen.position, rotation=Rotation.from_rotvec(chosen.turn) * wrist_rotation,
                    normal=normal, direction=direction, diagnostics=info)


class ContactFirstWrist(InformativeWrist):
    def __init__(self, hand, data, touch, *, gpis_device="cpu", contact_iso=.95,
                 target_force_n=TARGET_FORCE_N, **options):
        super().__init__(hand, data, touch, **options)
        self.selector = ContactFirstActions(self.selector.history, device=gpis_device, contact_iso=contact_iso,
                                           target_force_n=target_force_n)
        self.command_anchors = ()
        self.recruit = None
        self.preloaded = ()
        self.light_ticks = np.zeros(4, dtype=int)
        self.last_packet = -1
        self.latest_touch = touch
        self.measured_attitude = self.attitude

    def observe(self, data, touch, tick):
        super().observe(data, touch, tick)
        self.latest_touch = touch
        self.measured_attitude = wrist_attitude(data.qpos[3:7])
        if tick != self.last_packet:
            if tick != self.last_packet + 1:
                self.light_ticks.fill(0)
            light = touch.measured_heads & (touch.loads.max() < LOAD_LIMIT) & (not self.unloading)
            self.light_ticks = np.where(light, np.minimum(self.light_ticks + 1, SETTLE_TICKS), 0)
            self.last_packet = tick
        if not self.unloading and touch.loads.max() >= LOAD_LIMIT:
            hits = touch.measured_heads
            region = int(np.argmax(touch.loads))
            self.relief = touch.normals[region] if region < len(hits) and hits[region] else self.normal
            self.unloading = True
            self.clear_ticks = self.good_ticks = 0
            self.rescue = None
            self.escape_count += 1

    def hold_pose(self):
        self.goal, self.goal_attitude = self.position.copy(), self.attitude
        self.plan_mode = "stabilize"
        self.command_anchors, self.recruit, self.preloaded = (), None, ()

    def application_error(self, anchors, recruit):
        if self.phase() != "scan" or any(self.light_ticks[i] < SETTLE_TICKS for i in anchors):
            return "support changed before application"
        if recruit is not None and self.latest_touch.measured_heads[recruit]:
            return "second pad already measured before application"
        # The asynchronous plan can be five tactile packets old. Check its
        # endpoint against the CURRENT support planes before releasing the wrist.
        ids = list(anchors)
        points = self.latest_touch.points[ids]
        turn = self.goal_attitude * self.measured_attitude.inv()
        projected = self.goal + turn.apply(points - self.measured_position)
        depth = np.sum((projected - points) * self.latest_touch.normals[ids], axis=-1)
        if np.max(np.abs(depth)) > self.selector.band.half_width or np.min(depth) < -MAX_INWARD_M:
            return "wrist target no longer fits measured contact planes"
        return None

    def command(self, touch):
        hits = touch.measured_heads
        lost = any(not hits[i] for i in self.command_anchors)
        acquired = self.recruit is not None and hits[self.recruit]
        loaded = bool(self.preloaded) and (max(touch.loads) >= 4. or touch.loads[-1] >= .35 or
                                           any(touch.forces[i] >= self.selector.target_force for i in self.preloaded))
        if lost or acquired or self.unloading or loaded:
            self.hold_pose()
        now, ahead = super().command(touch)
        return (now, now.copy()) if self.plan_mode == "stabilize" else (now, ahead)


class ContactFirstWorker(InformativeWorker):
    def submit(self, data, touch, tick, lead):
        # The single worker accepts one request after installing the previous
        # result. Give that request a separate mask, never the live counter.
        self.servo.selector.settled = self.servo.light_ticks >= SETTLE_TICKS
        self.servo.selector.tip_forces = touch.forces.copy()
        super().submit(data, touch, tick, lead)

    def install(self, tick, *, initial=False):
        row = super().install(tick, initial=initial)
        if row is not None:
            began = perf_counter()
            servo = self.servo
            anchors = tuple(row["anchors"])
            error = servo.application_error(anchors, row["recruit"]) if anchors else None
            if error:
                servo.hold_pose()
                row.update(proposed_mode=row["mode"], mode="stabilize", reason=error,
                           anchors=[], recruit=None, requested_preload_m={}, gain_nat=0., expected_new_cells=0., winner=None,
                           selected_direction=[0., 0., 0.], wrist_target_m=servo.goal.tolist(),
                           wrist_target_xyzw=servo.goal_attitude.as_quat().tolist())
            else:
                servo.command_anchors = anchors
                servo.recruit = row["recruit"]
                servo.preloaded = tuple(row.get("requested_preload_m", {}))
            row["application_check_ms"] = 1000 * (perf_counter() - began)
            row["planning_ms"] += row["application_check_ms"]
        return row
