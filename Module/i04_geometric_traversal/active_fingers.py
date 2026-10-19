"""Bounded geometric searches for individually missing fingertip contacts.

This is a B-cell geometric planner: it uses the fixed object's known surface
to propose fingertip-center targets.  It never creates sensor contacts, awards
coverage, changes the wrist, or writes physical joint state.  Apply the existing
measured-contact MCC and whole-robot collision/force guards after this layer.

Search bias persists after an actual recontact so a newly flexed finger does
not immediately return to its old nominal posture.  Call ``rebase(new_nominal)``
whenever a fresh trajectory already includes the measured hand posture.  This
avoids applying that flexion twice.  Intentional air moves use enabled=False,
which clears the bias and returns their provided nominal reference unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import trimesh

from Module.e05_physics.scene import FINGERS
from .full_surface import pad_support_vector


@dataclass(frozen=True)
class ActiveFingerSearchConfig:
    dt_s: float = .002
    speed_m_s: float = .004
    maximum_search_distance_m: float = .040
    maximum_search_time_s: float = 12.
    maximum_joint_bias_rad: float = .50
    maximum_search_joint_speed_rad_s: float = .50
    target_refresh_s: float = .050
    loss_debounce_s: float = .060
    maximum_target_distance_m: float = .060
    target_clearance_m: float = -.0005
    force_pause_n: float = 8.
    joint_margin_rad: float = .015
    jacobian_damping_m: float = .001

    def __post_init__(self):
        positive = ('dt_s', 'speed_m_s', 'maximum_search_distance_m',
                    'maximum_search_time_s', 'maximum_joint_bias_rad',
                    'maximum_search_joint_speed_rad_s', 'target_refresh_s',
                    'maximum_target_distance_m', 'force_pause_n', 'jacobian_damping_m')
        for name in positive:
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if not 0 <= self.loss_debounce_s < self.maximum_search_time_s:
            raise ValueError('loss_debounce_s must be nonnegative and shorter than search time')
        if not np.isfinite(self.joint_margin_rad) or self.joint_margin_rad < 0:
            raise ValueError('joint_margin_rad must be finite and nonnegative')
        if not np.isfinite(self.target_clearance_m) or not -.001 <= self.target_clearance_m <= .002:
            raise ValueError('target_clearance_m must stay within [-1, 2] mm')


class ActiveFingerSearch:
    """Four independent finger searches with persistent, bounded hand bias.

    ``update`` returns ``(q_reference[23], diagnostics)``.  Diagnostics use
    JSON-compatible lists, with one state/target/distance per physical finger
    and sixteen hand joint biases/deltas.  ``target_valid`` distinguishes real
    planning targets from zero-filled unused slots.  ``seeking`` means an actual
    nonzero *command increment*, never an observed contact or successful search.
    """
    def __init__(self, h, graph, dt=.002, *, config=None):
        self.h, self.graph = h, graph
        self.config = config or ActiveFingerSearchConfig(dt_s=dt)
        ranges = np.asarray(h.hand_joint_ranges_rad)
        self.lower = ranges[:, 0] + self.config.joint_margin_rad
        self.upper = ranges[:, 1] - self.config.joint_margin_rad
        if np.any(self.lower >= self.upper):
            raise ValueError('Joint margins leave no finger joint range')
        self.columns = [np.asarray([int(name) for name in finger.joint_names]) for finger in FINGERS]
        self.dofs = np.r_[h.arm_dof_adrs, h.hand_dof_adrs]
        self.reset()

    def reset(self):
        self.bias = np.zeros(16)
        self.search_distance = np.zeros(4)
        self.loss_age = np.zeros(4)
        self.was_active = np.zeros(4, dtype=bool)
        self.targets = np.zeros((4, 3))
        self.target_normals = np.zeros((4, 3))
        self.target_valid = np.zeros(4, dtype=bool)
        self.target_age = np.inf
        self.refresh_count = 0
        self.initialized = False

    def rebase(self, nominal_q):
        nominal_q = np.asarray(nominal_q, dtype=float)
        if nominal_q.shape != (23,) or not np.all(np.isfinite(nominal_q)):
            raise ValueError('A finite 23-joint nominal reference is required')
        self.reset()
        self.initialized = True

    def _refresh_targets(self, data):
        centers = data.geom_xpos[self.h.tip_geom_ids].copy()
        points, _, faces = trimesh.proximity.closest_point(
            self.graph.mesh, centers - self.h.object_position_m)
        surface_normals = np.asarray(self.graph.mesh.face_normals)[faces]
        for finger, geom in enumerate(self.h.tip_geom_ids):
            normal = surface_normals[finger]
            length = np.linalg.norm(normal)
            if not np.all(np.isfinite(points[finger])) or length < .5:
                self.target_valid[finger] = False
                continue
            normal = normal / length
            frame = data.geom_xmat[int(geom)].reshape(3, 3)
            support = pad_support_vector(self.h.model, int(geom), frame, normal)
            target = (self.h.object_position_m + points[finger] + support
                      + self.config.target_clearance_m * normal)
            self.targets[finger] = target
            self.target_normals[finger] = normal
            self.target_valid[finger] = bool(np.all(np.isfinite(target)) and
                np.linalg.norm(target - centers[finger]) <= self.config.maximum_target_distance_m)
        self.target_age = 0.
        self.refresh_count += 1

    def _finger_jacobian(self, data, finger):
        jp, jr = np.zeros((3, self.h.model.nv)), np.zeros((3, self.h.model.nv))
        mujoco.mj_jacGeom(self.h.model, data, jp, jr, int(self.h.tip_geom_ids[finger]))
        return jp[:, self.dofs][:, self.columns[finger] + 7]

    def update(self, data, nominal_q, forces, normals, active, enabled=True):
        nominal = np.asarray(nominal_q, dtype=float)
        force = np.asarray(forces, dtype=float)
        actual = np.asarray(active, dtype=bool)
        sensor_normals = np.asarray(normals, dtype=float)
        if (nominal.shape != (23,) or force.shape != (4,) or actual.shape != (4,)
                or sensor_normals.shape != (4, 3) or not np.all(np.isfinite(nominal))
                or not np.all(np.isfinite(force)) or np.any(force < 0)):
            raise ValueError('Expected finite nominal[23], nonnegative forces[4], normals[4,3], active[4]')
        cfg = self.config
        result = nominal.copy()
        seeking, recontacted = np.zeros(4, dtype=bool), np.zeros(4, dtype=bool)
        increments = np.zeros(16)
        state = ['DISABLED'] * 4
        force_pause = bool(np.any(force >= cfg.force_pause_n))
        if not enabled:
            self.reset()
        else:
            if not self.initialized:
                self.rebase(nominal)
            self.target_age += cfg.dt_s
            if np.any(~actual) and not force_pause and self.target_age >= cfg.target_refresh_s:
                self._refresh_targets(data)
            for finger, columns in enumerate(self.columns):
                if actual[finger]:
                    recontacted[finger] = not self.was_active[finger] and self.search_distance[finger] > 0.
                    self.loss_age[finger] = 0.
                    state[finger] = 'CONTACT_HOLD'
                else:
                    if self.was_active[finger]:
                        self.loss_age[finger] = 0.
                        self.search_distance[finger] = 0.
                    self.loss_age[finger] += cfg.dt_s
                    if force_pause:
                        state[finger] = 'FORCE_PAUSE'
                    elif self.loss_age[finger] + 1e-12 < cfg.loss_debounce_s:
                        state[finger] = 'DEBOUNCE'
                    elif not self.target_valid[finger]:
                        state[finger] = 'OUT_OF_REACH'
                    elif (self.search_distance[finger] >= cfg.maximum_search_distance_m - 1e-12
                          or self.loss_age[finger] >= cfg.maximum_search_time_s):
                        state[finger] = 'SEARCH_LIMIT'
                    else:
                        error = self.targets[finger] - data.geom_xpos[int(self.h.tip_geom_ids[finger])]
                        distance = np.linalg.norm(error)
                        if distance <= .0001:
                            state[finger] = 'GEOMETRIC_TARGET_NO_CONTACT'
                        else:
                            distance_step = min(distance, cfg.speed_m_s * cfg.dt_s,
                                cfg.maximum_search_distance_m - self.search_distance[finger])
                            motion = error * (distance_step / distance)
                            jac = self._finger_jacobian(data, finger)
                            dq = jac.T @ np.linalg.solve(jac @ jac.T +
                                cfg.jacobian_damping_m ** 2 * np.eye(3), motion)
                            joint_step = cfg.maximum_search_joint_speed_rad_s * cfg.dt_s
                            dq *= min(1., joint_step / max(float(np.max(abs(dq))), 1e-15))
                            previous = self.bias[columns].copy()
                            proposed = np.clip(previous + dq, -cfg.maximum_joint_bias_rad,
                                               cfg.maximum_joint_bias_rad)
                            proposed = np.clip(nominal[columns + 7] + proposed,
                                               self.lower[columns], self.upper[columns]) - nominal[columns + 7]
                            delta = proposed - previous
                            # Count the commanded Cartesian search, including
                            # partial Jacobian progress; sensor movement remains
                            # independently measured by the experiment ledger.
                            traveled = float(np.linalg.norm(jac @ delta))
                            scale = min(1., distance_step / max(traveled, 1e-15),
                                        joint_step / max(float(np.max(abs(delta))), 1e-15))
                            delta *= scale
                            self.bias[columns] += delta
                            self.search_distance[finger] += float(np.linalg.norm(jac @ delta))
                            increments[columns] = delta
                            seeking[finger] = bool(np.max(abs(delta)) > 1e-12)
                            state[finger] = 'SEEK' if seeking[finger] else 'JOINT_LIMIT'
                self.was_active[finger] = actual[finger]
            result[7:] = np.clip(nominal[7:] + self.bias, self.lower, self.upper)
            # Store any nominal/joint-limit reconciliation so CONTACT_HOLD
            # diagnostics continue to describe exactly the returned bias.
            self.bias[:] = result[7:] - nominal[7:]
        diagnostics = dict(enabled=bool(enabled), force_pause=force_pause, state=state,
            seeking=seeking.tolist(), recontacted=recontacted.tolist(),
            measured_active=actual.tolist(), target_valid=self.target_valid.tolist(),
            target_positions_world_m=self.targets.tolist(), target_normals_world=self.target_normals.tolist(),
            search_distance_m=self.search_distance.tolist(), loss_age_s=self.loss_age.tolist(),
            joint_bias_rad=self.bias.tolist(), search_delta_rad=increments.tolist(),
            target_refresh_count=self.refresh_count,
            target_authority='GT_GEOMETRY_ONLY_NO_CONTACT_OR_COVERAGE')
        return result, diagnostics
