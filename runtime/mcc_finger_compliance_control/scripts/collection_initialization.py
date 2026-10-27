"""Measured-palm, true-hull contact acquisition before teacher motion.

Only initialization uses this controller. It retains a continuous actuator
target instead of switching a lost finger back to a fixed offline posture.
"""
from __future__ import annotations

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from surface_mcc_finger import HAND_JOINT_NAMES
from true_tip_waypoint_optimizer import SplitHullGeometry, TrueTipWaypointController


def relative_palm_pose(palm_world, oracle):
    palm = np.asarray(palm_world, float)
    rotation = Rotation.from_quat(palm[[4, 5, 6, 3]]).as_matrix()
    obj_rotation = oracle.rotation_world_from_object
    relative = Rotation.from_matrix(obj_rotation.T @ rotation).as_quat()
    return np.r_[obj_rotation.T @ (palm[:3]-oracle.center_world), relative[[3, 0, 1, 2]]]


def bounded_normal_step(row, distance, limit=.012):
    row = np.asarray(row, float)
    result = float(distance)*row/(float(row@row)+1.e-8)
    magnitude = float(np.abs(result).max())
    if magnitude > limit:
        result *= limit/magnitude
    return result


def bounded_cartesian_normal_step(jacobian, normal, distance, limit=.012):
    """Minimum joint change for normal motion with zero Cartesian tangent.

    A scalar normal-Jacobian inverse alone leaves tangential motion free.
    Solve all three Cartesian coordinates, then scale the whole vector so
    joint limiting does not rotate the requested displacement.
    """
    jacobian = np.asarray(jacobian, float)
    displacement = float(distance)*np.asarray(normal, float)
    delta = np.linalg.pinv(jacobian, rcond=1e-4) @ displacement
    # Do not replace an infeasible normal motion with a sideways search.
    if np.linalg.norm(jacobian @ delta-displacement) > max(1e-8, .01*abs(distance)):
        return np.zeros(jacobian.shape[1])
    magnitude = float(np.abs(delta).max())
    if magnitude > limit:
        delta *= limit/magnitude
    return delta


class StableContactInitializer:
    def __init__(self, controller, object_config, scale, seed_q):
        self.controller = controller
        self.geometry = SplitHullGeometry(object_config, scale)
        self.solver = TrueTipWaypointController(controller, object_config, scale,
            non_tip_clearance_m=.002, max_cross_finger_penetration_m=.0001,
            preferred_non_tip_clearance_m=.005)
        # Both queries must refer to the exact same geometry/model.
        self.solver.geometry = self.geometry
        self.solver.measured_seed_mode = True
        # Reuse the normalized reachability/manipulability soft costs from
        # our Aude/Khadivar-style planner. True-hull clearance remains an
        # independent feasibility check, not a posture similarity score.
        self.solver.kinematic_health_weight = 1.
        self.dofs = np.asarray([self.geometry.model.jnt_dofadr[
            mujoco.mj_name2id(self.geometry.model, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in HAND_JOINT_NAMES])
        self.seed_q = np.asarray(seed_q, float).copy()
        self.nominal_q = self.seed_q.copy()
        self.command = None
        self.contacted = np.zeros(4, bool)
        self.contact_q = np.zeros(16)
        self.loss_streak = np.zeros(4, int)
        self.tangent_path_m = np.zeros(4)
        self.last_contact_local = np.zeros((4, 3))
        self.previous_tip_points_object = None
        self.previous_loaded = np.zeros(4, bool)
        self.predicted_tangent_step_m = np.zeros(4)
        self.calls = 0
        self.last_solve_pose = None
        self.solve_count = 0
        self.solved_geometry_valid = False
        self.solve_health = {}
        self.ready = False
        self.streak = 0
        self.debug = {}

    def _solve(self, pose):
        # The wrapper's oracle receives object-frame witness points. A small
        # adapter returns normals in that same frame for its diagnostics.
        class LocalNormals:
            def normals_at_world(self, points):
                # Normals are only used in the returned diagnostic; use the
                # actual source-mesh oracle installed by update().
                world = self.outer_oracle.center_world + (
                    self.outer_oracle.rotation_world_from_object @ np.asarray(points).T).T
                normals = self.outer_oracle.normals_at_world(world)
                return (self.outer_oracle.rotation_world_from_object.T @ normals.T).T
        local = LocalNormals()
        local.outer_oracle = self.oracle
        q, info = self.solver.solve_geometry_contact_posture(local, pose,
            self.nominal_q, self.seed_q, preload_m=.0003, posture_gain=.18)
        self.nominal_q = np.asarray(q, float)
        self.last_solve_pose = pose.copy()
        self.solved_geometry_valid = bool(info['valid'])
        self.solve_health = {key: np.asarray(info[key]).tolist() for key in
            ('isotropic_reachability', 'manipulability')}
        self.solve_count += 1

    def update(self, q_live, palm_world, oracle, force_world, found,
               contact_points_world, connector_force, minimum_force=.10,
               settle_frames=50):
        c = self.controller
        q = np.asarray(q_live, float)
        self.oracle = oracle
        pose = relative_palm_pose(palm_world, oracle)
        self.calls += 1
        if self.command is None:
            self.command = q.copy()
        moved = (self.last_solve_pose is None or
                 np.linalg.norm(pose[:3]-self.last_solve_pose[:3]) > .001)
        # Re-solve only during acquisition, after the measured palm moves.
        # Every solve uses the current relative pose, including frame zero.
        if self.last_solve_pose is None or (moved and self.calls % 20 == 0):
            self._solve(pose)
        g = self.geometry.evaluate(q, pose, self_collision=True, non_tip_clearance=True)
        point_world = oracle.center_world + (
            oracle.rotation_world_from_object @ g['object_points'].T).T
        normals_world = oracle.normals_at_world(point_world)
        palm_rotation = Rotation.from_quat(np.asarray(palm_world)[[4, 5, 6, 3]]).as_matrix()
        normals_palm = (palm_rotation.T @ normals_world.T).T
        force = np.linalg.norm(force_world, axis=-1)
        # Ignore stale observations from the object pose before re-anchoring.
        loaded = (np.asarray(found, bool) & np.isfinite(force) & (force >= minimum_force)
                  & (g['distance'] <= .004) & (g['distance'] >= -.012))
        self.loss_streak = np.where(loaded, 0, self.loss_streak+1)
        setpoint = np.asarray(c.force_setpoint)
        target = self.command.copy()
        self.geometry.position(q, pose)
        mujoco.mj_comPos(self.geometry.model, self.geometry.data)
        tip_points_object = np.zeros((4, 3))
        for f in range(4):
            block = slice(4*f, 4*f+4)
            body = int(self.geometry.model.geom_bodyid[self.geometry.pads[f]])
            body_rotation = self.geometry.data.xmat[body].reshape(3, 3)
            body_position = self.geometry.data.xpos[body]
            if loaded[f] and not self.contacted[f]:
                self.contacted[f] = True
                self.contact_q[block] = q[block]
                point = palm_rotation.T @ (np.asarray(contact_points_world)[f]-np.asarray(palm_world)[:3])
                self.last_contact_local[f] = body_rotation.T @ (point-body_position)
            point = (body_position+body_rotation @ self.last_contact_local[f]
                     if self.contacted[f] else g['hand_points_palm'][f])
            pose_rotation = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
            tip_points_object[f] = pose[:3]+pose_rotation @ point
            jac = np.zeros((3, self.geometry.model.nv))
            mujoco.mj_jac(self.geometry.model, self.geometry.data, jac, None,
                point, body)
            finger_jacobian = jac[:, self.dofs[block]]
            if loaded[f]:
                error = force[f]-setpoint[f]
                distance = np.clip(error*.00002, -.00008, .00020)
                if abs(error) < .25*setpoint[f]:
                    distance = 0.
            elif self.contacted[f] and self.loss_streak[f] <= 5:
                # A short sensor/contact flicker keeps the existing grasp;
                # it must not restart a full search or return to offline q0.
                distance = 0.
            else:
                distance = -min(.00015, max(float(g['distance'][f])+.0003, .00002))
            if not self.contacted[f] and g['distance'][f] > .004:
                # Approach the optimized initial grasp, rather than wandering
                # tangentially in an unconstrained normal search far away.
                delta = np.clip(self.nominal_q[block]-self.command[block], -.012, .012)
            else:
                delta = bounded_cartesian_normal_step(finger_jacobian, normals_palm[f], distance)
            # Once acquired, there is no posture/null-space/tangent search.
            if self.contacted[f]:
                normal_motion = finger_jacobian @ delta
                tangent = normal_motion-normals_palm[f]*(normals_palm[f] @ normal_motion)
                self.predicted_tangent_step_m[f] = np.linalg.norm(tangent)
            target[block] += delta
            if self.contacted[f]:
                target[block] = np.clip(target[block], self.contact_q[block]-.10,
                                        self.contact_q[block]+.10)
        if self.previous_tip_points_object is not None:
            displacement = tip_points_object-self.previous_tip_points_object
            normal_object = (oracle.rotation_world_from_object.T @ normals_world.T).T
            tangent = displacement-normal_object*np.sum(displacement*normal_object, axis=-1, keepdims=True)
            self.tangent_path_m += np.where(loaded & self.previous_loaded,
                                            np.linalg.norm(tangent, axis=-1), 0.)
        self.previous_tip_points_object = tip_points_object
        self.previous_loaded = loaded.copy()
        # Bound command lag as well as slew. This prevents accumulating an
        # unreachable target behind an existing physical collision.
        target = q + np.clip(target-q, -.05, .05)
        target = self.command + np.clip(target-self.command, -.012, .012)
        target = np.clip(target, np.maximum(c.lower+.01, self.seed_q-.90),
                         np.minimum(c.upper-.01, self.seed_q+.90))
        # Retain/improve non-tip clearance and self collision throughout
        # acquisition, including an initially colliding physical state.
        for fraction in (1., .5, .25, .125, 0.):
            trial = q + fraction*(target-q)
            check = self.geometry.evaluate(trial, pose, self_collision=True, non_tip_clearance=True)
            if (check['non_tip_distance'].min() >= min(.001, g['non_tip_distance'].min())-1.e-6
                    and check['self_penetration'] <= max(.0001, g['self_penetration'])+1.e-6):
                target = trial
                break
        self.command = target.copy()
        c.previous_command = target.copy()
        connector = np.linalg.norm(connector_force, axis=-1)
        valid = bool(loaded.all() and np.all(force <= 2.5*setpoint)
            and np.all(connector < .05) and g['non_tip_distance'].min() >= -.0001
            and g['self_penetration'] <= .0002)
        self.streak = self.streak+1 if valid else 0
        self.ready = self.streak >= settle_frames
        self.debug = dict(calls=self.calls, solves=self.solve_count,
            solved_geometry_valid=self.solved_geometry_valid,
            solve_health=self.solve_health,
            contacted=self.contacted.tolist(), loss_streak=self.loss_streak.tolist(),
            loaded_material_tangent_path_mm=(self.tangent_path_m*1000).tolist(),
            predicted_tangent_step_mm=(self.predicted_tangent_step_m*1000).tolist(),
            held_grasp_deviation_rad=[float(np.abs(q[4*f:4*f+4]-self.contact_q[4*f:4*f+4]).max())
                                      if self.contacted[f] else None for f in range(4)],
            pad_distance_m=g['distance'].tolist(), force_n=force.tolist(),
            connector_force_n=connector.tolist(), streak=self.streak,
            non_tip_min_m=float(g['non_tip_distance'].min()),
            self_penetration_m=float(g['self_penetration']), ready=self.ready)
        return target, self.nominal_q.copy(), self.ready
