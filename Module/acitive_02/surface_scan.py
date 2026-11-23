"""Tactile-only wrist guidance for revisiting the unseen sides of a fixed object."""

from dataclasses import dataclass

import numpy as np
import mujoco
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.transform import Rotation

from .aude import unit
from ..i04_geometric_traversal.physics import quaternion, rotation

OUTSIDE_RADIUS_M = .45
SWEEP_SPEED_M_S = .115
ROW_SPACING_RAD = .051
POINT_SPACING_RAD = .045
GAP_M = .005
SURFACE_LOAD_N = 12.
PAD_GAP_M = -.00028


def sweep_directions(axis=0, phase=0.0):
    """Wrap a latitude raster around the fixture origin, with no shape prior."""
    rows = int(np.ceil(np.pi / ROW_SPACING_RAD))
    directions = [[-1., 0., 0.]]
    for row, phi in enumerate(np.linspace(-np.pi / 2, np.pi / 2, rows + 1)[1:-1]):
        count = max(5, int(np.ceil(2. * np.pi * np.cos(phi) / POINT_SPACING_RAD)))
        theta = phase + np.linspace(0., 2. * np.pi, count + 1)
        if row % 2:
            theta = theta[::-1]
        directions.extend(np.stack((np.repeat(np.sin(phi), len(theta)),
                                    np.cos(phi) * np.cos(theta), np.cos(phi) * np.sin(theta)), axis=1))
    directions.append([1., 0., 0.])
    return np.roll(np.asarray(directions), axis, axis=1)


def align(a, b):
    cross = np.cross(a, b)
    size = np.linalg.norm(cross)
    dot = float(np.clip(a @ b, -1., 1.))
    if size < 1e-10:
        if dot > 0.:
            return Rotation.identity()
        cross = np.cross(a, np.eye(3)[np.argmin(abs(a))])
        return Rotation.from_rotvec(np.pi * unit(cross))
    return Rotation.from_rotvec(cross * (np.arctan2(size, dot) / size))


def limit(vector, length):
    return vector * min(1., length / max(np.linalg.norm(vector), 1e-12))


def slew(reference, position, orientation, dt, speed=.25, angular=1.7):
    previous = rotation(reference[3:])
    displacement = limit(position - reference[:3], speed * dt)
    turn = (orientation * previous.inv()).as_rotvec()
    orientation = Rotation.from_rotvec(limit(turn, angular * dt)) * previous
    return np.r_[reference[:3] + displacement, quaternion(orientation)]


class TactileSurface:
    """Sparse live points and local planes. The fixture origin is known."""

    def __init__(self):
        self.samples = {}
        self.tree = None
        self.fitted_count = 0
        self.points = np.empty((0, 3))
        self.normals = np.empty((0, 3))
        self.radii = np.empty(0)

    def add(self, touch):
        if touch.loads.max() > SURFACE_LOAD_N:
            return
        for point, normal in zip(touch.points[touch.measured_heads], touch.normals[touch.measured_heads]):
            if np.linalg.norm(point) > .025:
                cell = tuple(np.floor(point / .003).astype(int))
                self.samples.setdefault(cell, np.r_[point, normal])

    def refresh(self, force=False):
        if not self.samples or len(self.samples) == self.fitted_count:
            return
        if self.tree is not None and not force and len(self.samples) < self.fitted_count + 16:
            return
        samples = np.asarray(list(self.samples.values()))
        self.points, self.normals = samples[:, :3], samples[:, 3:]
        self.radii = np.linalg.norm(self.points, axis=1)
        self.tree = cKDTree(self.points / self.radii[:, None])
        self.fitted_count = len(samples)

    def estimate(self, direction, default=.13):
        if self.tree is None:
            return default, direction
        separation, index = self.tree.query(direction, k=min(6, self.fitted_count))
        index, separation = np.atleast_1d(index), np.atleast_1d(separation)
        if separation[0] > .24:
            return default, direction
        weights = 1. / (separation + .018)**2
        radii = self.radii[index]
        normals = self.normals[index]
        cosines = normals @ direction
        local_planes = np.einsum('ij,ij->i', normals, self.points[index]) / np.maximum(cosines, .35)
        prediction = np.where(cosines >= .35, np.clip(local_planes, radii - .028, radii + .028), radii)
        radial = float(np.average(prediction, weights=weights))
        normal = unit(np.average(normals, axis=0, weights=weights) + .27 * direction)
        if normal @ direction < .4:
            normal = unit(normal + direction)
        return radial, normal

    def uncovered(self, directions, distance=GAP_M):
        self.refresh(force=True)
        if self.tree is None:
            return np.ones(len(directions), dtype=bool)
        gaps, closest = self.tree.query(directions)
        return gaps * self.radii[closest] > distance


class PadSupport:
    """Use only the known robot geometry to put a fingertip ahead of its links."""

    def __init__(self, hand):
        self.hand = hand
        self.vertices = {}
        for identifier in hand.pads:
            mesh = int(hand.model.geom_dataid[identifier])
            start = int(hand.model.mesh_vertadr[mesh])
            count = int(hand.model.mesh_vertnum[mesh])
            vertices = hand.model.mesh_vert[start:start + count].astype(float)
            self.vertices[int(identifier)] = vertices[ConvexHull(vertices).vertices]

    def local_points(self, data):
        hand = self.hand
        inv = data.xmat[hand.palm].reshape(3, 3)
        root = data.xpos[hand.palm]
        for geom, vertices in self.vertices.items():
            body_axes = data.geom_xmat[geom].reshape(3, 3)
            yield (vertices @ body_axes.T + (data.geom_xpos[geom] - root)) @ inv

    def point(self, data, direction, preferred):
        candidates = []
        for finger, points in enumerate(self.local_points(data)):
            depth = points @ direction
            offset = points[depth >= depth.max() - .00045].mean(axis=0)
            preference = .012 if finger == preferred else 0.
            candidates.append((float(offset @ direction + preference), finger, offset))
        _, finger, offset = max(candidates, key=lambda x: x[0])
        return finger, offset


@dataclass(frozen=True)
class ScanCommand:
    wrist: np.ndarray
    phase: str
    direction: np.ndarray
    has_contact: bool


class NearFieldWristGuard:
    """Local, one-step MuJoCo collision preview for the shared wrist executor."""

    def __init__(self, hand):
        self.hand = hand
        self.test = mujoco.MjData(hand.model)

    def correct(self, proposed, data, normal, offset, dt):
        hand = self.hand
        test = self.test
        test.qpos[:] = data.qpos
        # Preview only the next slew-limited pose, including the present wrist
        # admittance. Do not query the object away from the robot's local pose.
        test.qpos[:3] = proposed[:3] + offset[:3]
        orientation = Rotation.from_rotvec(offset[3:].copy()) * rotation(proposed[3:])
        test.qpos[3:7] = quaternion(orientation)
        mujoco.mj_kinematics(hand.model, test)
        mujoco.mj_comPos(hand.model, test)
        mujoco.mj_collision(hand.model, test)
        shift = 0.
        for contact in test.contact:
            one, two = int(contact.geom1), int(contact.geom2)
            if one in hand.objects:
                other, outward = two, contact.frame[:3]
            elif two in hand.objects:
                other, outward = one, -contact.frame[:3]
            else:
                continue
            limit_gap = PAD_GAP_M if other in hand.pad_index else .0018
            into = float(outward @ normal)
            if into > .22 and contact.dist < limit_gap:
                shift = max(shift, (limit_gap - float(contact.dist)) / into)
        bounded_shift = min(1.2 * dt, shift)
        corrected = proposed.copy()
        corrected[:3] += bounded_shift * normal
        return corrected, bounded_shift


class SurfaceScanner:
    """Traverse fixture directions and servo probe depth from measured contacts."""

    def __init__(self, hand, dt, directions, local_axis=(.22, -.57, -.79), preferred=2):
        self.hand = hand
        self.dt = dt
        self.directions = np.asarray(directions)
        self.preferred = preferred
        self.axis = unit(np.asarray(local_axis))
        self.support = PadSupport(hand)
        self.guard = NearFieldWristGuard(hand)
        self.surface = TactileSurface()
        self.index = 0
        self.phase = 'withdraw'
        self.at_point = 0
        self.depth = .25
        self.approach_ticks = 0
        self.normal = self.directions[0]
        self.direction = self.directions[0]
        self.estimate = .13
        self.reference = None
        self.sampled = 0
        self.skipped = 0
        self.travel_direction = None
        self.yield_ticks = 0

    @property
    def done(self):
        return self.index >= len(self.directions)

    def start(self, reference):
        self.reference = reference.copy()
        self.travel_direction = unit(reference[:3])

    def _refresh(self, target):
        self.surface.refresh()
        self.estimate, self.normal = self.surface.estimate(target, self.estimate)

    def _transfer(self, data, target):
        previous = rotation(self.reference[3:])
        if self.phase == 'withdraw':
            goal = OUTSIDE_RADIUS_M * self.travel_direction
            wrist = slew(self.reference, goal, previous, self.dt, speed=.35)
            if np.linalg.norm(wrist[:3] - goal) < .002:
                self.phase = 'transfer'
            return wrist
        turn = align(self.travel_direction, target).as_rotvec()
        self.travel_direction = Rotation.from_rotvec(limit(turn, .9 * self.dt)).apply(self.travel_direction)
        outward = self.travel_direction
        want = align(-previous.apply(self.axis), outward) * previous
        wrist = slew(self.reference, OUTSIDE_RADIUS_M * outward, want, self.dt, speed=.46)
        angle_error = np.linalg.norm(turn)
        orient_error = (-rotation(wrist[3:]).apply(self.axis)) @ target
        if angle_error < .016 and orient_error > .998:
            self.phase = 'approach'
            _, offset = self.support.point(data, self.axis, self.preferred)
            self.depth = float((wrist[:3] + rotation(wrist[3:]).apply(offset)) @ target)
            self._refresh(target)
        return wrist

    def command(self, data, touch, wrist_offset):
        target = self.directions[min(self.index, len(self.directions) - 1)]
        current_phase = self.phase
        if self.phase in ('withdraw', 'transfer'):
            self.reference = self._transfer(data, target)
            return ScanCommand(self.reference, current_phase, target, False)
        if self.at_point == 0 or self.phase == 'approach':
            self._refresh(target)
        previous = rotation(self.reference[3:])
        desired = align(-previous.apply(self.axis), self.normal) * previous
        finger, offset = self.support.point(data, self.axis, self.preferred)
        # The current contact itself supplies the radial depth. A matched
        # force loop explores a few millimeters farther if the pad opens.
        points = touch.points
        depth = points @ target
        across = np.linalg.norm(points - depth[:, None] * target, axis=1)
        near = touch.measured_heads & (depth > 0.) & (across < .014)
        has_contact = bool(near.any()) and touch.loads.max() <= SURFACE_LOAD_N
        if has_contact:
            touched = int(finger) if near[finger] else int(np.argmin(np.where(near, across, np.inf)))
            force = touch.forces[touched]
            contact_normal = touch.normals[touched]
            slope = float(contact_normal @ target)
            load_depth = ((contact_normal @ points[touched]) / slope if slope > .43 else depth[touched])
            preload = np.clip(.00032 * (1.1 - force), -.0015, .0005)
            correction = .21 * (load_depth - preload - self.depth)
            self.depth += float(np.clip(correction, -.0012, .0012))
        elif self.phase == 'approach':
            self.depth -= (.15 if self.depth > self.estimate + .02 else .055) * self.dt
        else:
            self.depth -= (.060 if self.depth > self.estimate + .016 else .030) * self.dt
        # Bounds come from the broad fixture workspace and the nearest tactile
        # estimate. They do not use object collision geometry.
        self.depth = max(.045, self.estimate - .019, self.depth)
        if touch.loads.max() > 14.:
            self.yield_ticks = 12
        if self.yield_ticks > 0:
            # Rebase against the existing wrist correction so the nominal
            # path does not keep pushing deeper while compliance unloads.
            self.depth = max(self.depth, self.estimate + .0015)
            self.yield_ticks -= 1
        selected_orientation = Rotation.from_rotvec(limit((desired * previous.inv()).as_rotvec(),
                                                           2.8 * self.dt)) * previous
        position = self.depth * target - selected_orientation.apply(offset)
        proposed = slew(self.reference, position, selected_orientation, self.dt, speed=.22, angular=2.8)
        if self.yield_ticks:
            proposed[:3] += .00085 * self.normal
        self.reference, shift = self.guard.correct(proposed, data, self.normal, wrist_offset, self.dt)
        if shift:
            self.depth += min(.0014, .42 * shift)
        if self.phase == 'approach':
            self.approach_ticks += 1
            if has_contact or self.approach_ticks * self.dt >= 4.5:
                self.phase = 'scan'
                self.at_point = 0
        else:
            self.at_point += 1
            previous_target = self.directions[max(0, self.index - 1)]
            distance = max(.003, np.linalg.norm(target - previous_target) * max(self.estimate, .09))
            standard = max(3, int(np.ceil(distance / (SWEEP_SPEED_M_S * self.dt))))
            if (has_contact and self.at_point >= standard) or self.at_point >= standard + 7:
                self.sampled += int(has_contact)
                self.skipped += int(not has_contact)
                self.index += 1
                self.at_point = 0
        return ScanCommand(self.reference, current_phase, target, has_contact)
