"""Continue the wrist and all four fingertips along the measured contact surface."""
from collections import Counter
from itertools import islice
import json
from time import perf_counter

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.transform import Rotation, RotationSpline
import trimesh

from .fingertip_hand import DT
from .fingertip_plan import JointPatchPlanner, Patch, Posture
from .fingertip_traversal import Leg, Traversal, latency_ms, write_json
from .oracle import align_vector
from .physics import quaternion

WAYPOINT_M = .006
STROKE_M = .13


class SurfaceStroke:
    """Time-parameterize one fitted contact route, with no dwell at interior knots."""
    phase = 'slide'
    tracks_surface = True

    def __init__(self, patches):
        self.patches = patches
        postures = [patch.posture for patch in patches]
        self.start, self.end = postures[0], postures[-1]
        positions = np.stack([pose.position for pose in postures])
        rotations = Rotation.concatenate([pose.orientation for pose in postures])
        joints = np.stack([pose.joints for pose in postures])
        motion = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        turn = (rotations[:-1].inv() * rotations[1:]).magnitude()
        articulation = abs(np.diff(joints, axis=0)).max(axis=1)
        duration = np.maximum.reduce([motion / .045, turn / .65, articulation / .65,
                                      np.full(len(motion), .018)])
        self.times = np.r_[0., np.cumsum(duration)]
        self.travel = np.array([motion.sum(), turn.sum(), articulation.sum()])
        self.guide_length_m = sum(np.linalg.norm(right.points[right.anchor] - left.points[right.anchor])
                                  for left, right in zip(patches, patches[1:]))
        self.position_curve = PchipInterpolator(self.times, positions)
        self.rotation_curve = RotationSpline(self.times, rotations)
        self.joint_curve = PchipInterpolator(self.times, joints)
        self.normal_curve = PchipInterpolator(self.times, np.stack([p.normals for p in patches]))

    def distance(self):
        return self.travel

    def ticks(self):
        return int(np.ceil(1.5 * self.times[-1] / DT))

    def sample_times(self, count):
        u = np.arange(1, count + 1) / count
        return self.times[-1] * (u * u * (3. - 2. * u))

    def samples(self, count):
        times = self.sample_times(count)
        return (self.position_curve(times), self.rotation_curve(times).as_quat()[:, [3, 0, 1, 2]],
                self.joint_curve(times))

    def sample_normals(self, count, default):
        normals = self.normal_curve(self.sample_times(count))
        return normals / np.linalg.norm(normals, axis=2, keepdims=True)


class ContinuousJointPlanner(JointPatchPlanner):
    def __init__(self, hand, graph):
        super().__init__(hand, graph)
        self.solve_ms = []

    def solve_patch(self, start, points, normals, anchor, vertex, *, previous=None):
        started = perf_counter()
        result = super().solve_patch(start, points, normals, anchor, vertex, previous=previous)
        self.solve_ms.append((perf_counter() - started) * 1000.)
        return result

    def project(self, points):
        projected, _, triangles = trimesh.proximity.closest_point(self.graph.mesh, points)
        face_normals = self.graph.mesh.face_normals[triangles]
        distance, near = self.graph._tree.query(projected, k=12)
        normals = self.graph.normals[near]
        same_side = np.einsum('nkj,nj->nk', normals, face_normals) > .4
        weights = same_side / np.maximum(.002, distance)
        smooth = (normals * weights[..., None]).sum(axis=1) + 20. * face_normals
        return projected, smooth / np.linalg.norm(smooth, axis=1, keepdims=True)

    def root_formation(self, posture, touch):
        self.set_pose(posture)
        up = posture.orientation.apply(-self.h.inward_in_palm)
        reference = np.where(touch.eligible[:, None], touch.normals, up)
        raw = np.array([self.support(f, n) for f, n in enumerate(reference)])
        points, normals = self.project(np.where(touch.eligible[:, None], touch.points, raw))
        return points, normals

    def mesh_route(self, point, source, target, parents, nearby_goals):
        vertices = self.graph.reconstruct_path(source, target, parents)
        xyz = self.graph.vertices_m
        goals = [target]
        length = np.linalg.norm(np.diff(xyz[vertices], axis=0), axis=1).sum()
        # Fold nearby unvisited goals into the same forward stroke. Reversals
        # and large turns close this stroke so the hand can turn in contact.
        for _ in range(2):
            end = vertices[-1]
            previous = vertices[max(0, len(vertices) - 4)]
            direction = xyz[end] - xyz[previous]
            if length > STROKE_M - .024 or np.linalg.norm(direction) < .005:
                break
            distance, predecessors = self.graph.distances_from(end, return_predecessors=True)
            offset = xyz[nearby_goals] - xyz[end]
            forward = offset @ (direction / np.linalg.norm(direction))
            aligned = forward > .45 * np.linalg.norm(offset, axis=1)
            face = self.graph.normals[nearby_goals] @ self.graph.normals[end] > .65
            travel = distance[nearby_goals]
            options = nearby_goals[aligned & face & (travel > .018) & (travel <= STROKE_M - length)]
            if not len(options):
                break
            for candidate in options[np.argsort(distance[options])]:
                extension = self.graph.reconstruct_path(end, candidate, predecessors)[1:]
                if not np.intersect1d(vertices, extension).size:
                    vertices = np.r_[vertices, extension]
                    length += distance[candidate]
                    goals.append(int(candidate))
                    break
            else:
                break
        line = xyz[vertices].copy()
        line[0] = point
        if len(line) > 2:
            line[1:-1] = .25 * line[:-2] + .5 * line[1:-1] + .25 * line[2:]
        distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))]
        horizon = min(STROKE_M, distance[-1])
        count = max(2, int(np.ceil(horizon / WAYPOINT_M)) + 1)
        stations = np.linspace(0., horizon, count)[1:]
        guide = self.project(np.column_stack([np.interp(stations, distance, axis) for axis in line.T]))
        return guide, goals

    def continue_patch(self, previous, point, normal, anchor, vertex, *, turn=None):
        before = previous.points[anchor]
        if turn is None:
            turn = align_vector(previous.normals[anchor], normal)
            delta = turn.as_rotvec()
            angle = np.linalg.norm(delta)
            if angle > .14:
                turn = Rotation.from_rotvec(.14 * delta / angle)
        root = previous.posture
        transported = Posture(point + turn.apply(root.position - before),
                              turn * root.orientation, root.joints)
        points, normals = self.project(point + turn.apply(previous.points - before))
        points[anchor], normals[anchor] = point, normal
        return self.solve_patch(transported, points, normals, anchor, vertex, previous=root)

    def stroke(self, root, points, normals, guide, anchor, vertex):
        patches = [Patch(root, points, normals, np.zeros(4), anchor, vertex)]
        for point, normal in zip(*guide):
            previous = patches[-1]
            if np.linalg.norm(point - previous.points[anchor]) < .001:
                continue
            patch = self.continue_patch(previous, point, normal, anchor, vertex)
            if patch is None:
                break
            motion, turn, joints = Leg(previous.posture, patch.posture, 'slide').distance()
            if motion > .026 or turn > .20 or joints > .125:
                break
            patches.append(patch)
        return patches


class ContinuousTraversal(Traversal):
    """Follow local groups of surface waypoints before considering relocation."""
    def __init__(self, output, seconds, *, collect_contacts=False, hand=None, graph=None):
        super().__init__(output, seconds, planner_type=ContinuousJointPlanner,
                         collect_contacts=collect_contacts, hand=hand, graph=graph)
        self.local_failures = set()
        self.pivots = set()
        self.stale_routes = 0
        self.plan_ms = []
        self.preview_ms = []
        self.execution_s = 0.
        manifest_path = self.out / 'manifest.json'
        manifest = json.loads(manifest_path.read_bytes())
        manifest['planning'] = dict(mode='continuous_surface', waypoint_spacing_m=WAYPOINT_M,
                                    stroke_horizon_m=STROKE_M, order=['surface_stroke', 'contact_pivot',
                                    'withdraw_transfer_approach'])
        write_json(manifest_path, manifest)

    def surface_candidates(self, points, normals, touch):
        anchors = np.flatnonzero(touch.eligible)
        if not len(anchors):
            return []
        graph = self.graph
        roots, _ = graph.oriented_nearest_vertices(points[anchors], normals[anchors])
        distances, parents = dijkstra(graph.adjacency, directed=False, indices=roots,
                                      return_predecessors=True)
        unseen = ~self.metrics.coverage.visited_vertices & (self.attempts < 25)
        targets = graph.required_vertices[unseen[graph.required_vertices]]
        if not len(targets):
            targets = np.flatnonzero(unseen)
        scores = distances[:, targets] + .045 * self.attempts[targets]
        rows, cols = np.unravel_index(np.argsort(scores, axis=None), scores.shape)
        return [(int(anchors[r]), int(roots[r]), int(targets[c]), parents[r])
                for r, c in zip(rows, cols) if distances[r, targets[c]] > .004]

    def commit(self, route, patch, prediction, vertex, mode, waypoints, route_goals=()):
        first_step = self.metrics.steps
        before = self.metrics.coverage.visited_vertices.sum()
        reason = self.execute(prediction)
        gained = int(self.metrics.coverage.visited_vertices.sum() - before)
        self.events.append(dict(start_step=first_step, end_step=self.metrics.steps, vertex=vertex,
            target_site=patch.vertex, anchor=patch.anchor, phases=[item.phase for item in route], mode=mode,
            wrist_xyz=patch.posture.position.tolist(),
            wrist_wxyz=quaternion(patch.posture.orientation).tolist(),
            finger_targets_rad=patch.posture.joints.tolist(), surface_targets_m=patch.points.tolist(),
            surface_normals=patch.normals.tolist(), planned_gaps_m=patch.gaps.tolist(),
            waypoints=waypoints, route_goals=list(route_goals), new_vertices=gained))
        self.current, self.last_normals = patch.posture, patch.normals
        self.stale_routes = 0 if gained else self.stale_routes + 1
        if not self.metrics.coverage.visited_vertices[vertex]:
            self.attempts[vertex] += int(not gained)
        return reason

    def preview_stroke(self, patches, vertex, mode, route_goals=()):
        # A useful prefix can run if the whole stroke reaches a sharp corner.
        for size in dict.fromkeys((len(patches), (len(patches) + 1) // 2)):
            if size < 3:
                continue
            stroke = SurfaceStroke(patches[:size])
            route = [stroke]
            if not self.geometry_clear(route):
                self.rejected += 1
                continue
            patch = patches[size - 1]
            prediction = self.forecast(route, patch.normals)
            if prediction is None:
                self.rejected += 1
                continue
            guide = [part.points[part.anchor].tolist() for part in patches[:size]]
            return self.commit(route, patch, prediction, vertex, mode, guide, route_goals), True
        return None, False

    def try_pivot(self, points, normals, touch):
        if touch.eligible.sum() < 2:
            return None, False
        anchor = int(np.argmax(np.where(touch.eligible, touch.forces, -np.inf)))
        vertex, _ = self.graph.nearest_vertex(points[anchor])
        region = tuple(np.rint(points[anchor] / .025).astype(int))
        for degrees in (16., -16., 28., -28.):
            key = region + (degrees,)
            if key in self.pivots:
                continue
            self.pivots.add(key)
            patches = [Patch(self.current, points, normals, np.zeros(4), anchor, vertex)]
            turn = Rotation.from_rotvec(np.deg2rad(degrees / 5.) * normals[anchor])
            for _ in range(5):
                patch = self.planner.continue_patch(patches[-1], points[anchor], normals[anchor],
                                                    anchor, vertex, turn=turn)
                if patch is None:
                    break
                patches.append(patch)
            if len(patches) == 6:
                result, accepted = self.preview_stroke(patches, vertex, 'contact_pivot')
                if accepted:
                    return result, True
        return None, False

    def attempt(self, vertex):
        started = perf_counter()
        native_before = self.execution_s
        result = self.plan_from_contact(vertex)
        self.plan_ms.append((perf_counter() - started - (self.execution_s - native_before)) * 1000.)
        return result

    def plan_from_contact(self, vertex):
        touch = self.h.measure(self.data)
        points, normals = self.planner.root_formation(self.current, touch)
        if self.stale_routes < 5:
            tried = 0
            nearby_goals = self.graph.required_vertices[~self.metrics.coverage.visited_goals]
            for anchor, source, target, parents in self.surface_candidates(points, normals, touch):
                key = (source, target, anchor)
                if key in self.local_failures:
                    continue
                tried += 1
                guide, goals = self.planner.mesh_route(points[anchor], source, target, parents, nearby_goals)
                patches = self.planner.stroke(self.current, points, normals, guide, anchor, target)
                result, accepted = self.preview_stroke(patches, target, 'surface_stroke', goals)
                if accepted:
                    return result
                self.local_failures.add(key)
                if tried >= 9:
                    break
            result, accepted = self.try_pivot(points, normals, touch)
            if accepted:
                return result

        count = int(self.attempts[vertex])
        self.attempts[vertex] += 1
        site = vertex if count < 5 else self.alternate_site(vertex, count)
        for patch in islice(self.planner.candidates(site, self.current, count), 6):
            for route in islice(self.routes(patch), 1, None):
                if not self.geometry_clear(route):
                    self.rejected += 1
                    continue
                route.append(Leg(patch.posture, patch.posture, 'hold'))
                prediction = self.forecast(route, patch.normals)
                if prediction is None:
                    self.rejected += 1
                    continue
                return self.commit(route, patch, prediction, vertex, 'clearance_transfer', [])
        return None

    def forecast(self, route, normals):
        started = perf_counter()
        result = super().forecast(route, normals)
        self.preview_ms.append((perf_counter() - started) * 1000.)
        return result

    def execute(self, forecast):
        started = perf_counter()
        result = super().execute(forecast)
        self.execution_s += perf_counter() - started
        return result

    def run(self):
        started = perf_counter()
        summary = super().run()
        modes = Counter(event['mode'] for event in self.events)
        summary.update(planning='continuous_surface', executed_routes_by_mode=dict(modes),
                       lifts=modes['clearance_transfer'],
                       surface_waypoints=sum(len(event['waypoints']) for event in self.events))
        summary['timing'] = dict(active_wall_s=perf_counter() - started, execution_s=self.execution_s,
            per_route_request_including_preview=latency_ms(self.plan_ms),
            per_joint_patch_solve=latency_ms(self.planner.solve_ms),
            per_native_preview=latency_ms(self.preview_ms))
        np.savez_compressed(self.out / 'planning_timing.npz', route_request_ms=self.plan_ms,
                            joint_patch_solve_ms=self.planner.solve_ms, native_preview_ms=self.preview_ms)
        write_json(self.out / 'summary.json', summary)
        return summary
