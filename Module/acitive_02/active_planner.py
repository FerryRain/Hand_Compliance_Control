"""Plan Aude-style wrist/finger motions against a tactile GPIS and a known hand."""

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .active_gpis import ObservedSurface, TRUST_RADIUS_M
from .aude import state_posture, tangent, unit
from .cooperative import MAX_ROUTE_STEP_RAD, ROUTE_REACH_RAD, trajectory
from .surface_scan import PadSupport, SURFACE_LOAD_N, align, limit
from ..i04_geometric_traversal.fingertip_plan import Posture
from ..i04_geometric_traversal.physics import quaternion, rotation


LONG_MIDDLE = np.array([2, 6, 10])
LONG_DISTAL = np.array([3, 7, 11])
PRELOAD_M = -.00075
REPLAN_TICKS = 18
SERVO_TICKS = 4
CONTACT_HOLD_TICKS = 16


class GraspEnvelope:
    """Keep the long fingers curved and the thumb in the starting opposition sector."""

    def __init__(self, reference, physical):
        self.reference = reference.copy()
        radius = np.deg2rad([34, 17, 37, 38] * 3 + [32, 30, 37, 44])
        lower = np.maximum(physical[:, 0] + .060, reference - radius)
        upper = np.minimum(physical[:, 1] - .060, reference + radius)
        lower[LONG_MIDDLE] = np.maximum(lower[LONG_MIDDLE], np.deg2rad(8.))
        lower[LONG_DISTAL] = np.maximum(lower[LONG_DISTAL], np.deg2rad(14.))
        upper[LONG_DISTAL] = np.minimum(upper[LONG_DISTAL], np.deg2rad(74.))
        lower[12:16] = np.maximum(lower[12:16], np.deg2rad([-8., 52., 6., 8.]))
        self.limits = np.column_stack((lower, upper))
        if not np.allclose(self.bound(reference), reference, atol=1.e-9, rtol=0.):
            raise ValueError('The initial measured grasp must lie inside the hand-shape envelope')

    def bound(self, joints):
        joints = np.clip(joints, *self.limits.T)
        middle = joints[..., LONG_MIDDLE]
        distal = joints[..., LONG_DISTAL]
        lower = np.maximum(self.limits[LONG_DISTAL, 0], .35 * middle)
        upper = np.minimum(self.limits[LONG_DISTAL, 1], .60 + middle)
        joints[..., LONG_DISTAL] = np.clip(distal, lower, upper)
        return joints


class TactileProjection:
    def __init__(self, ik, surface, grasp):
        self.ik, self.surface, self.grasp = ik, surface, grasp
        self.hand = ik.hand
        self.data = ik.data
        self.jac = np.empty((3, ik.model.nv))
        self.pad_mesh = PadSupport(self.hand).vertices
        self.anchor_points = np.zeros((4, 3))
        self.anchor_normals = np.zeros((4, 3))
        self.anchor_offsets = np.zeros((4, 3))
        self.anchor_ticks = np.full(4, -1000)
        self.tick = 0
        geom = self.hand.palm_geom
        model = self.ik.model
        mesh = model.geom_dataid[geom]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        vertices = model.mesh_vert[start:start + count].astype(float)
        # A small set of extreme samples spans the palm hull for learned clearance.
        directions = np.array([[x, y, z] for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)])
        self.palm_mesh = vertices[np.unique(np.argmax(vertices @ directions.T, axis=0))]

    def world_vertices(self, geom, vertices):
        d = self.data
        return d.geom_xpos[geom] + vertices @ d.geom_xmat[geom].reshape(3, 3).T

    def support_point(self, finger, normal):
        geom = self.hand.pads[finger]
        points = self.world_vertices(geom, self.pad_mesh[int(geom)])
        heights = points @ normal
        return points[heights <= heights.min() + .0005].mean(axis=0)

    def normal_jacobian(self, point, normal, body, width):
        mujoco.mj_jac(self.ik.model, self.data, self.jac, None, point, body)
        return normal @ self.jac[:, :width]

    def observe(self, data, touch, tick):
        self.tick = tick
        if touch.loads.max() > SURFACE_LOAD_N:
            return
        valid = touch.measured_heads
        pads = self.hand.pads[valid]
        self.anchor_points[valid] = touch.points[valid]
        self.anchor_normals[valid] = touch.normals[valid]
        self.anchor_offsets[valid] = np.einsum('ij,ijk->ik', touch.points[valid] - data.geom_xpos[pads],
                                             data.geom_xmat[pads].reshape(-1, 3, 3))
        self.anchor_ticks[valid] = tick

    def anchored_point(self, finger):
        geom = self.hand.pads[finger]
        return self.data.geom_xpos[geom] + self.data.geom_xmat[geom].reshape(3, 3) @ self.anchor_offsets[finger]

    def supports(self, normal):
        result = []
        for finger in range(4):
            recent = self.tick - self.anchor_ticks[finger] <= CONTACT_HOLD_TICKS
            anchor = self.anchor_points[finger]
            if recent and np.linalg.norm(self.anchored_point(finger) - anchor) < .024:
                patch = self.surface.patch(anchor, preferred_normal=self.anchor_normals[finger])
                if patch is not None:
                    result.append((finger, patch, anchor))
                    continue
            point = self.support_point(finger, normal)
            patch = self.surface.patch(point, preferred_normal=normal, radius=.066)
            if patch is None:
                continue
            result.append((finger, patch, None))
        return result

    def solve(self, guide, normal, *, joint, iterations, search_depth):
        ik, hand, data = self.ik, self.hand, self.data
        q = self.grasp.bound(guide.joints) if joint else guide.joints
        ik.set_pose(guide.position, guide.orientation, q)
        width = 22 if joint else 6
        diagonal = np.r_[[1.0] * 3, [.043] * 3, [.027] * 16][:width]
        regularizer = diagonal[:, None] * np.eye(width)
        if joint:
            regularizer[:3, :3] = 3.3 * np.eye(3) - 2.55 * np.outer(normal, normal)
        supports = self.supports(normal)
        palm = hand.palm_geom
        candidates = self.world_vertices(palm, self.palm_mesh)
        palm_point = candidates[np.argmin(candidates @ normal)]
        palm_patch = self.surface.patch(palm_point, preferred_normal=normal, radius=.036)
        for _ in range(iterations):
            differences = np.r_[data.qpos[:3] - guide.position,
                                (rotation(data.qpos[3:7]) * guide.orientation.inv()).as_rotvec(),
                                data.qpos[hand.q] - q]
            rows = [regularizer]
            residuals = [regularizer @ differences[:width]]
            probes, exploratory = [], []
            for finger, patch, anchor in supports:
                point = self.anchored_point(finger) if anchor is not None else self.support_point(finger, patch.normal)
                if np.linalg.norm(point - patch.anchor) > TRUST_RADIUS_M + .028:
                    continue
                distance, outward = self.surface.gap(point, patch, anchor)
                if outward @ patch.normal > .40:
                    row = (distance, finger, point, outward)
                    (probes if anchor is not None else exploratory).append(row)
            exploratory.sort(key=lambda row: abs(row[0]))
            for rank, (distance, finger, point, outward) in enumerate(probes + exploratory):
                anchored = rank < len(probes)
                weight = 4.5 if anchored else (1.85 if rank < 3 else .38)
                gap = PRELOAD_M if anchored else PRELOAD_M - search_depth
                row = self.normal_jacobian(point, outward, hand.tip_bodies[finger], width)
                rows.append(weight * row[None, :])
                residuals.append(np.array([weight * np.clip(distance - gap, -.012, .012)]))
            if palm_patch is not None:
                vertices = self.world_vertices(palm, self.palm_mesh)
                values, gradients = palm_patch.predict(vertices)
                local = np.linalg.norm(vertices - palm_patch.anchor, axis=1) < .038
                clearances = np.where(local, values, np.inf)
                index = int(clearances.argmin())
                if clearances[index] < .004:
                    outward = unit(gradients[index])
                    row = self.normal_jacobian(vertices[index], outward, hand.palm, width)
                    rows.append(5. * row[None, :])
                    residuals.append(np.array([5. * (clearances[index] - .004)]))
            delta = np.linalg.lstsq(np.vstack(rows), -np.concatenate(residuals), rcond=None)[0]
            increment = np.zeros(ik.model.nv)
            increment[:3] = limit(delta[:3], .0055)
            increment[3:6] = limit(delta[3:6], .075 if joint else .045)
            if joint:
                increment[hand.dof] = np.clip(delta[6:], -.090, .090)
            mujoco.mj_integratePos(ik.model, data.qpos, increment, 1.)
            if joint:
                data.qpos[hand.q] = self.grasp.bound(data.qpos[hand.q])
            mujoco.mj_kinematics(ik.model, data)
            mujoco.mj_comPos(ik.model, data)
            if np.linalg.norm(delta) < .00035:
                break
        result = state_posture(hand, data)
        if joint and ik.gaps().min() < -.001:
            ik.set_pose(result.position, result.orientation, q)
            result.joints = q
        return result

    def pivot(self, live, touch, normal):
        """Tilt around a measured support to send one nearby idle pad toward the GPIS."""
        self.ik.set_pose(live.position, live.orientation, live.joints)
        hits = touch.measured_heads
        anchor = touch.points[hits].mean(axis=0)
        outward = unit(touch.normals[hits].sum(axis=0) + .22 * normal)
        nearby = self.surface.patch(anchor, preferred_normal=outward)
        choices = []
        for finger in np.flatnonzero(~hits):
            point = self.support_point(finger, outward)
            lever = point - anchor
            length = np.linalg.norm(lever)
            if length < .012:
                continue
            field = self.surface.patch(point, preferred_normal=outward, radius=.078) or nearby
            gap, direction = self.surface.gap(point, field) if field else (lever @ outward, outward)
            axis = np.cross(direction, lever)
            score = abs(gap) + .10 * length
            choices.append((score, axis))
        if not choices:
            return live.position, live.orientation
        _, axis = min(choices, key=lambda item: item[0])
        turn = Rotation.from_rotvec(.052 * unit(axis))
        return anchor + turn.apply(live.position - anchor), turn * live.orientation


class ActiveSurfacePlanner:
    def __init__(self, ik, grasp, initial, passes, dt):
        if ik.hand.objects:
            raise ValueError('The planning model must contain only robot geometry')
        self.surface = ObservedSurface()
        self.projection = TactileProjection(ik, self.surface, grasp)
        self.grasp = grasp
        self.plan_dt = REPLAN_TICKS * dt
        self.route = trajectory(0, initial.position)
        self.route_direction = unit(initial.position)
        self.route_attitude = initial.orientation
        self.route_index = 0
        self.stage = 0
        self.passes = passes
        self.finished = []
        self.missing_time = 0.
        self.quorum_recovery_time = 0.
        self.depth_search = 0.
        self.last_normal = unit(initial.position)
        self.last_point = None
        self.servo = np.zeros(6)
        self.novelty = 0.

    def observe(self, data, touch, tick):
        self.surface.add(touch, tick)
        self.projection.observe(data, touch, tick)

    @property
    def done(self):
        return self.stage >= self.passes

    def waypoint(self, position, time_s):
        while not self.done:
            while self.route_index < len(self.route) and position @ self.route[self.route_index] > np.cos(ROUTE_REACH_RAD):
                self.route_index += 1
            if self.route_index < len(self.route):
                return self.route[self.route_index]
            self.finished.append(dict(pass_number=self.stage + 1, time_s=time_s, directions_total=len(self.route)))
            self.stage += 1
            self.route = trajectory(self.stage % 3, position)
            self.route_index = 0
        return position

    def exploration_turn(self, radial, to_waypoint, live, aude):
        base = limit(to_waypoint, MAX_ROUTE_STEP_RAD)
        self.novelty = 0.
        if self.last_point is None or np.linalg.norm(base) < .025:
            return base
        field = self.surface.patch(self.last_point, preferred_normal=self.last_normal)
        if field is None:
            return base
        angles = Rotation.from_rotvec(np.array([-.46, 0., .46])[:, None] * radial).apply(base)
        query = Rotation.from_rotvec(angles).apply(self.last_point)
        gp_information = field.predict(query, variance=True)
        route_progress = (angles @ base) / (base @ base)
        direction = tangent(aude.position - live.position, radial)
        agreement = np.cross(angles, radial) @ unit(direction) / np.linalg.norm(base)
        index = int(np.argmax(route_progress + .44 * gp_information + .12 * agreement))
        self.novelty = float(gp_information[index])
        return angles[index]

    def plan(self, live, aude, touch, time_s):
        self.surface.refresh()
        radial = unit(live.position)
        valid = touch.measured_heads
        load = touch.loads.max()
        if valid.any() and load <= SURFACE_LOAD_N:
            observed = unit(touch.normals[valid].sum(axis=0))
            self.last_normal = unit(.48 * self.last_normal + .52 * observed)
            self.last_point = touch.points[valid].mean(axis=0)
        normal = unit(.72 * radial + .28 * self.last_normal)
        supported = int(valid.sum()) >= 2 and load <= SURFACE_LOAD_N
        self.missing_time = 0. if valid.any() else self.missing_time + self.plan_dt
        self.quorum_recovery_time = 0. if supported else self.quorum_recovery_time + self.plan_dt
        self.depth_search = min(.013, .0025 + .0045 * self.quorum_recovery_time)
        goal = self.waypoint(self.route_direction, time_s)
        error = align(self.route_direction, goal).as_rotvec()
        angle = self.exploration_turn(self.route_direction, error, live, aude)
        cap = MAX_ROUTE_STEP_RAD if supported else (.045 if valid.any() else .032)
        if load > SURFACE_LOAD_N or radial @ self.route_direction < np.cos(.18):
            cap = 0.
        if .18 < self.missing_time % 1.6 < .63:
            cap = 0.
        angle = limit(angle, cap)
        turn = Rotation.from_rotvec(angle)
        self.route_direction = turn.apply(self.route_direction)
        self.route_attitude = turn * self.route_attitude
        desired = self.route_direction * np.linalg.norm(live.position)
        position = live.position + limit(desired - live.position, .014)
        relative = (turn * live.orientation * self.route_attitude.inv()).as_rotvec()
        orientation = Rotation.from_rotvec(limit(relative, .74)) * self.route_attitude
        if supported:
            position += .19 * limit(tangent(aude.position - live.position, radial), .007)
        if load > SURFACE_LOAD_N:
            position = live.position + normal * .0026
            orientation = live.orientation
        elif not valid.any():
            position -= normal * min(.0046, .0011 + .55 * self.depth_search)
        elif int(valid.sum()) == 1 and .25 < self.quorum_recovery_time % 2.7 < 1.46:
            pivot_position, pivot_orientation = self.projection.pivot(live, touch, normal)
            correction = (pivot_orientation * live.orientation.inv()).as_rotvec()
            orientation = Rotation.from_rotvec(.72 * correction) * orientation
            position += .67 * (pivot_position - live.position)
        shape = self.grasp.bound(.74 * live.joints + .21 * aude.joints + .05 * self.grasp.reference)
        guide = Posture(position, orientation, shape)
        mode = 'multi_slide' if supported else 'multi_recontact'
        if load > SURFACE_LOAD_N:
            return guide, mode, normal
        return self.projection.solve(guide, normal, joint=True, iterations=4, search_depth=self.depth_search), mode, normal

    def stabilize(self, raw, joints, touch, normal, wrist, update):
        orientation = rotation(raw[3:])
        if update:
            compliant = wrist.forecast(raw)
            if touch.loads.max() > SURFACE_LOAD_N:
                adjustment = np.r_[.0018 * normal, np.zeros(3)]
            else:
                guide = Posture(compliant[:3], rotation(compliant[3:]), joints)
                support = self.projection.solve(guide, normal, joint=False, iterations=2, search_depth=self.depth_search)
                adjustment = np.r_[limit(support.position - guide.position, .006),
                                   limit((support.orientation * guide.orientation.inv()).as_rotvec(), .070)]
            self.servo = .55 * self.servo + .45 * adjustment
        command = raw.copy()
        command[:3] += self.servo[:3]
        command[3:] = quaternion(Rotation.from_rotvec(self.servo[3:]) * orientation)
        return command
