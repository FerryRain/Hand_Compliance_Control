"""Static fixed-wrist, one-finger reconfiguration on the original surface.

This independent planner is not integrated into a frozen experiment. It uses
only scratch FK/collision queries. A returned path is geometric evidence, not
a successful recovery: the native caller must retain force/collision guards,
stop inward motion at actual selected-finger contact, and transition to MCC.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import mujoco
import numpy as np
import trimesh
from scipy.optimize import least_squares

from .contracts import ExperimentConfig
from .full_surface import JointPath, pad_support_vector
from .physics import joint_state, pose, set_joint_state
from .sliding_safety import SlidingSafetyPlanner, arm_self_contacts


def ellipsoid_support_derivative(frame, sizes, normal, angular_jacobian):
    """d(R diag(a^2) R^T n / sqrt(n^T R diag(a^2) R^T n))/dq.

    ``angular_jacobian[:,j]`` is the world angular velocity for unit qdot_j.
    The surface normal stays fixed; changing pad orientation changes its
    center-to-support vector. Ignoring this term fits the wrong pad center.
    """
    frame, sizes, normal = np.asarray(frame), np.asarray(sizes), np.asarray(normal)
    angular = np.asarray(angular_jacobian)
    shape = (frame * (sizes * sizes)[None, :]) @ frame.T
    radius = float(np.sqrt(normal @ shape @ normal))
    if radius <= 1e-12:
        raise ValueError('Ellipsoid support radius must be positive')
    vector = shape @ normal
    derivatives = []
    for axis in angular.T:
        x, y, z = axis
        skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
        change = skew @ shape - shape @ skew
        derivatives.append(change @ normal / radius - vector * (normal @ change @ normal) / (2 * radius**3))
    return np.column_stack(derivatives)


@dataclass(frozen=True)
class FingerReconfigurationConfig:
    target_clearance_m: float = -.0003
    target_residual_m: float = .0007
    joint_speed_rad_s: float = .15
    dense_joint_step_rad: float = .004
    minimum_duration_s: float = .5
    maximum_ik_evaluations: int = 180
    random_starts: int = 4
    seed: int = 7
    force_pause_n: float = 8.

    def __post_init__(self):
        if not -.0005 <= self.target_clearance_m <= .002:
            raise ValueError('Target clearance must remain within [-0.5, 2] mm')
        for key in ('target_residual_m', 'joint_speed_rad_s', 'dense_joint_step_rad', 'minimum_duration_s', 'force_pause_n'):
            if not np.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f'{key} must be finite and positive')
        if self.maximum_ik_evaluations < 1 or self.random_starts < 0:
            raise ValueError('IK evaluation/start counts are invalid')


@dataclass
class FingerReconfigurationPath(JointPath):
    anchor_finger: int
    selected_finger: int


class FingerReconfigurationPlanner:
    """Plan only the selected four joints; wrist/other fingers stay measured.

    ``plan(data, selected_finger, active, forces, normals)`` mirrors the small
    wrist-pivot planner API. The target is the original mesh point nearest the
    measured pad center, with a fixed mesh normal and orientation-dependent
    ellipsoid support. Full original joint ranges plus the existing margin are
    allowed, without relaxing collision rules or the runtime search gate.
    """
    def __init__(self, h, graph, config=None):
        self.h, self.graph = h, graph
        self.config = config or FingerReconfigurationConfig()
        self.planner = SlidingSafetyPlanner(h, ExperimentConfig())
        self.scratch = mujoco.MjData(h.model)

    def _dense_edge(self, root, candidate, selected_columns):
        cfg = self.config
        if not self.planner.edge_free(root, candidate):
            return False, dict(reason='STRICT_COLLISION_EDGE', checked_states=0)
        count = max(3, int(np.ceil(np.max(abs(candidate-root)) / cfg.dense_joint_step_rad)) + 1)
        previous = root
        for fraction in np.linspace(0., 1., count)[1:]:
            trial = root + fraction * (candidate - root)
            # Preserve every unselected degree of freedom bit-for-bit.
            unselected = np.ones(23, bool); unselected[selected_columns] = False
            trial[unselected] = root[unselected]
            if not self.planner.edge_free(previous, trial):
                return False, dict(reason='DENSE_STRICT_COLLISION_EDGE', checked_states=count,
                                   failed_fraction=float(fraction))
            previous = trial
        return True, dict(reason='DENSE_STRICT_EDGE_PASSED', checked_states=count,
                          maximum_joint_spacing_rad=float(np.max(abs(candidate-root)) / (count-1)))

    def plan(self, data, selected_finger, active, forces, normals):
        started = perf_counter()
        cfg, h = self.config, self.h
        active, forces, normals = np.asarray(active, bool), np.asarray(forces, float), np.asarray(normals, float)
        if (selected_finger not in range(4) or active.shape != (4,) or forces.shape != (4,)
                or normals.shape != (4, 3) or not np.isfinite(forces).all()
                or not np.isfinite(normals).all() or np.any(forces < 0)):
            raise ValueError('Expected selected finger and finite actual four-finger observations')
        diag = dict(selected_finger=int(selected_finger), candidates=[],
                    authority='STATIC_GEOMETRY_ONLY_NO_CONTACT_OR_COVERAGE',
                    physical_recontact_tested=False, wrist_and_unselected_fingers_fixed=True)
        if active[selected_finger]:
            return None, dict(diag, reason='SELECTED_FINGER_ALREADY_ACTIVE')
        if np.any(forces >= cfg.force_pause_n) or arm_self_contacts(h, data):
            return None, dict(diag, reason='MEASURED_FORCE_OR_SELF_CONTACT')
        anchors = np.flatnonzero(active & (forces >= .2) & (forces < cfg.force_pause_n))
        if not len(anchors):
            return None, dict(diag, reason='NO_MEASURED_SAFE_ANCHOR')
        anchor = int(anchors[np.argmax(forces[anchors])])
        root, root_pose = joint_state(h, data), pose(h, data)
        geom = int(h.tip_geom_ids[selected_finger])
        if h.model.geom_type[geom] not in (mujoco.mjtGeom.mjGEOM_ELLIPSOID, mujoco.mjtGeom.mjGEOM_SPHERE):
            return None, dict(diag, reason='PAD_SUPPORT_TYPE_UNSUPPORTED')
        nearest, _, faces = trimesh.proximity.closest_point(self.graph.mesh,
            (data.geom_xpos[geom] - h.object_position_m)[None])
        normal = np.asarray(self.graph.mesh.face_normals[int(faces[0])]).copy()
        normal /= np.linalg.norm(normal)
        surface_point = h.object_position_m + nearest[0]
        target_support = surface_point + cfg.target_clearance_m * normal
        columns = np.arange(7 + 4*selected_finger, 11 + 4*selected_finger)
        dofs = np.r_[h.arm_dof_adrs, h.hand_dof_adrs][columns]
        lo, hi = self.planner.lower[columns], self.planner.upper[columns]
        def update(q_selected):
            q = root.copy(); q[columns] = q_selected
            set_joint_state(h, self.scratch, q)
        def residual(q_selected):
            update(q_selected)
            frame = self.scratch.geom_xmat[geom].reshape(3, 3)
            support = pad_support_vector(h.model, geom, frame, normal)
            return self.scratch.geom_xpos[geom] - support - target_support
        def jacobian(q_selected):
            update(q_selected)
            jp, jr = np.zeros((3, h.model.nv)), np.zeros((3, h.model.nv))
            mujoco.mj_jacGeom(h.model, self.scratch, jp, jr, geom)
            derivative = np.zeros((3, 4))
            if h.model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
                derivative = ellipsoid_support_derivative(self.scratch.geom_xmat[geom].reshape(3, 3),
                    h.model.geom_size[geom], normal, jr[:, dofs])
            return jp[:, dofs] - derivative
        baseline = float(np.linalg.norm(residual(root[columns])))
        diag.update(anchor_finger=anchor, initial_support_gap_m=baseline,
                    target_surface_point_world_m=surface_point.tolist(), target_outward_normal_world=normal.tolist(),
                    target_clearance_m=cfg.target_clearance_m, unchanged_joint_indices=np.flatnonzero(~np.isin(np.arange(23), columns)).tolist())
        rng = np.random.default_rng(cfg.seed)
        starts = [np.clip(root[columns], lo+1e-9, hi-1e-9), (lo+hi)/2, lo+.25*(hi-lo), lo+.75*(hi-lo)]
        starts.extend(lo+rng.uniform(.05, .95, (cfg.random_starts, 4))*(hi-lo))
        geometries = []
        for index, start in enumerate(starts):
            result = least_squares(residual, start, jac=jacobian, bounds=(lo, hi),
                                   max_nfev=cfg.maximum_ik_evaluations, ftol=1e-11, xtol=1e-11, gtol=1e-11)
            candidate = root.copy(); candidate[columns] = result.x
            error = float(np.linalg.norm(result.fun))
            meta = dict(start_index=index, target_residual_m=error, ik_evaluations=int(result.nfev),
                        selected_joint_delta_rad=(result.x-root[columns]).tolist())
            if error > cfg.target_residual_m:
                diag['candidates'].append(dict(meta, reason='DYNAMIC_SUPPORT_IK_RESIDUAL'))
                continue
            if any(np.max(abs(candidate-old)) < 1e-4 for old, _ in geometries):
                diag['candidates'].append(dict(meta, reason='DUPLICATE_GEOMETRIC_CANDIDATE'))
                continue
            geometries.append((candidate, meta))
        geometries.sort(key=lambda item: np.max(abs(item[0]-root)))
        for candidate, meta in geometries:
            valid, edge = self._dense_edge(root, candidate, columns)
            if not valid:
                diag['candidates'].append(dict(meta, **edge))
                continue
            duration = max(cfg.minimum_duration_s, 1.875*float(np.max(abs(candidate-root)))/cfg.joint_speed_rad_s)
            set_joint_state(h, self.scratch, candidate)
            path = FingerReconfigurationPath(np.stack([root, candidate]), np.array([duration]),
                root_pose.copy(), pose(h, self.scratch), perf_counter()-started, anchor, int(selected_finger))
            selected = dict(meta, **edge, duration_s=duration,
                            maximum_quintic_joint_speed_rad_s=1.875*float(np.max(abs(candidate-root)))/duration,
                            final_support_gap_m=float(np.linalg.norm(residual(candidate[columns]))))
            diag['candidates'].append(dict(selected, reason='GEOMETRIC_RECONFIGURATION_SELECTED'))
            return path, dict(diag, reason='GEOMETRIC_RECONFIGURATION_SELECTED', selected=selected,
                              latency_s=perf_counter()-started)
        return None, dict(diag, reason='NO_SAFE_FIXED_WRIST_RECONFIGURATION', latency_s=perf_counter()-started)
