"""Joint wrist/finger patch planning for the hand-only I04-c experiment."""
from __future__ import annotations

from dataclasses import dataclass
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

from .belly_bounded_step import bounded_belly_step
from .fingertip_hand import reference_grasp
from .oracle import align_vector
from .physics import quaternion, rotation


@dataclass
class Posture:
    position: np.ndarray
    orientation: Rotation
    joints: np.ndarray

    def lift(self, normal, distance):
        return Posture(self.position + distance * normal, self.orientation, self.joints)


@dataclass
class Patch:
    posture: Posture
    points: np.ndarray
    normals: np.ndarray
    gaps: np.ndarray
    anchor: int
    vertex: int


class JointPatchPlanner:
    def __init__(self, hand, graph):
        self.h, self.model, self.graph = hand, hand.model, graph
        self.seed = Posture(*reference_grasp())
        s = self.seed
        self.data = hand.initialize(s.position, s.orientation, s.joints)
        self.solver_steps = 0
        self.head_vertices = []
        self.reference_points = []
        for f, g in enumerate(hand.pads):
            mid = self.model.geom_dataid[g]
            start, size = self.model.mesh_vertadr[mid], self.model.mesh_vertnum[mid]
            verts = self.model.mesh_vert[start:start + size].astype(float)
            world = self.data.geom_xpos[g] + verts @ self.data.geom_xmat[g].reshape(3, 3).T
            self.head_vertices.append(verts[[hand.fraction(self.data, f, x) > .82 for x in world]])
            point = self.support(f, np.array([0., 0., 1.]))
            self.reference_points.append(s.orientation.inv().apply(point - s.position))
        self.eye = np.eye(22)

    def set_pose(self, posture):
        d, m = self.data, self.model
        d.qpos[:3] = posture.position
        d.qpos[3:7] = quaternion(posture.orientation)
        d.qpos[self.h.q] = posture.joints
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)

    def support(self, finger, normal):
        g = self.h.pads[finger]
        rot = self.data.geom_xmat[g].reshape(3, 3)
        verts = self.head_vertices[finger]
        # Average essentially coplanar support vertices for stable tangential IK.
        heights = verts @ (normal @ rot)
        return self.data.geom_xpos[g] + rot @ verts[heights < heights.min() + .0001].mean(axis=0)

    def nearest_part(self, f):
        return self.h.nearest_pad_surface(self.data, f)

    def jac(self, point, body):
        pos, ang = np.zeros((3, 22)), np.zeros((3, 22))
        mujoco.mj_jac(self.model, self.data, pos, ang, point, body)
        return pos, ang

    def buffered(self, maximum_depth=.0028):
        if not self.h.posture(self.data)[1]:
            return False
        mujoco.mj_collision(self.model, self.data)
        for c in self.data.contact:
            a, b = int(c.geom1), int(c.geom2)
            if (a in self.h.objects) == (b in self.h.objects):
                continue
            if c.dist < -maximum_depth:
                return False
        return True

    def fit(self, vertex, anchor, orientation):
        graph, h, d = self.graph, self.h, self.data
        goal = graph.vertices_m[vertex]
        start = Posture(goal - orientation.apply(self.reference_points[anchor]),
                        orientation, self.seed.joints.copy())
        self.set_pose(start)
        raw = start.position + orientation.apply(self.reference_points)
        points, _, tri = trimesh.proximity.closest_point(graph.mesh, raw)
        normals = graph.mesh.face_normals[tri].copy()
        points[anchor], normals[anchor] = goal, graph.normals[vertex]
        return self.solve_patch(start, points, normals, anchor, vertex)

    def solve_patch(self, start, points, normals, anchor, vertex, *, previous=None):
        """Fit a contact formation, optionally continuing the last accepted posture."""
        h, d = self.h, self.data
        self.set_pose(start)
        following = previous is not None
        goal = points[anchor]
        # All four position and orientation tasks share these 22 columns at
        # every iteration. The guide finger adds a tangential coverage target.
        lower = np.r_[start.position - .035, [-1.] * 3,
                      np.maximum(h.limits[:, 0] + .02, self.seed.joints - .305)]
        upper = np.r_[start.position + .035, [1.] * 3,
                      np.minimum(h.limits[:, 1] - .02, self.seed.joints + .305)]
        roll_weight = .046 if following else .033
        weights = np.r_[[.25] * 3, [roll_weight] * 3, [.019] * 16]
        preload = .00052 if following else .0009
        for iteration in range(13):
            pose = rotation(d.qpos[3:7])
            wrist_log = (start.orientation.inv() * pose).as_rotvec()
            errors = [weights * np.r_[d.qpos[:3] - start.position, wrist_log,
                                      d.qpos[h.q] - self.seed.joints]]
            rows = [weights[:, None] * self.eye]
            if following:
                errors.append(.030 * (d.qpos[h.q] - previous.joints))
                rows.append(.030 * self.eye[6:])
            for f, normal in enumerate(normals):
                body = h.tip_bodies[f]
                point = self.support(f, normal)
                jp, jr = self.jac(point, body)
                axis = d.xmat[body].reshape(3, 3) @ h.head_axes[f]
                tangent = .85 if f == anchor else .17
                metric = tangent * np.eye(3) + (1 - tangent) * np.outer(normal, normal)
                weight = 1.8 if f == anchor else .8
                errors.append(weight * metric @ (point - points[f] + min(.0008, preload) * normal))
                rows.append(weight * metric @ jp)
                errors.append(.012 * (axis + normal))
                rows.append(.012 * np.cross(jr.T, axis).T)
                if iteration >= 3:
                    gap, witness = self.nearest_part(f)
                    if witness is not None and gap < .02:
                        physical, outward = witness
                        top = h.fraction(d, f, physical)
                        target = -preload if top >= .83 else .004
                        j, _ = self.jac(physical, body)
                        errors.append(np.array([3. * (gap - target)]))
                        rows.append(3. * outward[None, :] @ j)
            prior = np.r_[d.qpos[:3], np.zeros(3), d.qpos[h.q]]
            delta, diag = bounded_belly_step(np.vstack(rows), np.concatenate(errors), prior,
                lower, upper, gain=.8, damping=.002,
                maximum_joint_step=np.r_[[.007] * 3, [.10] * 3, [.12] * 16])
            self.solver_steps += 1
            if not diag['feasible']:
                return None
            mujoco.mj_integratePos(self.model, d.qpos, delta, 1.)
            mujoco.mj_kinematics(self.model, d)
            mujoco.mj_comPos(self.model, d)
            if iteration >= 4 and abs(delta).max() < .0004:
                break
        if not self.buffered():
            return None
        gaps, witnesses = zip(*(self.nearest_part(f) for f in range(4)))
        max_gap = .00065 if following else -.00015
        max_error = .009 if following else .014
        if (witnesses[anchor] is None or not -.0028 <= gaps[anchor] <= max_gap
                or np.linalg.norm(witnesses[anchor][0] - goal) > max_error):
            return None
        if following and h.fraction(d, anchor, witnesses[anchor][0]) < .78:
            return None
        end = Posture(d.qpos[:3].copy(), rotation(d.qpos[3:7]), d.qpos[h.q].copy())
        return Patch(end, points, normals, np.array(gaps), anchor, int(vertex))

    def candidates(self, vertex, current, attempt):
        n = self.graph.normals[vertex]
        # Parallel transport the current grasp around the object. Alternate
        # rolls/guide fingers keep the same four simultaneous tasks at edges.
        up = self.seed.orientation.inv().apply([0., 0., 1.])
        base = align_vector(current.orientation.apply(up), n) * current.orientation
        target = self.graph.vertices_m[vertex]
        tips = current.position + current.orientation.apply(self.reference_points)
        order = np.argsort(np.linalg.norm(tips - target, axis=1))
        for yaw in (0., np.pi / 2, -np.pi / 2, np.pi):
            orient = Rotation.from_rotvec(n * (yaw + attempt * np.pi / 5)) * base
            for f in order:
                patch = self.fit(vertex, int(f), orient)
                if patch is not None:
                    yield patch
