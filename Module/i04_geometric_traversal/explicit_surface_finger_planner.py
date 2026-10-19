"""Independent fixed-wrist finger planner for an explicit nearby mesh point.

The frozen nearest-point planner remains unchanged. This candidate preserves
its joint, dynamic ellipsoid-support, residual, path and speed rules while
allowing a point on a known original-mesh triangle within 40 mm of the measured
pad's nearest surface point. It never changes plant state or grants coverage.
"""
from time import perf_counter

import mujoco
import numpy as np
import trimesh
from scipy.optimize import least_squares

from .finger_reconfiguration_planner import (
    FingerReconfigurationPlanner, FingerReconfigurationPath, ellipsoid_support_derivative,
)
from .full_surface import pad_support_vector
from .physics import joint_state, pose, set_joint_state
from .sliding_safety import arm_self_contacts


def validate_mesh_face_point(mesh, point_local_m, face_id):
    point = np.asarray(point_local_m, dtype=float)
    if (point.shape != (3,) or not np.all(np.isfinite(point)) or not isinstance(face_id, (int, np.integer))
            or face_id not in range(len(mesh.faces))):
        raise ValueError('Expected a finite point and original mesh face index')
    triangle = np.asarray(mesh.triangles[int(face_id)])
    uv = np.linalg.lstsq(np.column_stack((triangle[1]-triangle[0], triangle[2]-triangle[0])),
                        point-triangle[0], rcond=None)[0]
    barycentric = np.r_[1.-uv.sum(), uv]
    projected = barycentric@triangle
    if np.linalg.norm(projected-point) > 1e-10 or np.min(barycentric) < -1e-9:
        raise ValueError('The target must lie on the specified original mesh triangle')
    normal = np.asarray(mesh.face_normals[int(face_id)], dtype=float).copy()
    if not np.all(np.isfinite(normal)) or np.linalg.norm(normal) < .5:
        raise ValueError('The target triangle needs a valid original mesh normal')
    return point.copy(), normal/np.linalg.norm(normal), barycentric


class ExplicitSurfaceFingerPlanner(FingerReconfigurationPlanner):
    def __init__(self, h, graph, config=None, *, target_point_local_m, target_face_id,
                 maximum_target_offset_m=.040):
        super().__init__(h, graph, config)
        if not np.isfinite(maximum_target_offset_m) or not 0. < maximum_target_offset_m <= .040:
            raise ValueError('Explicit surface candidates stay within 40 mm of the nearest point')
        self.point_local, self.target_normal, self.barycentric = validate_mesh_face_point(
            graph.mesh, target_point_local_m, target_face_id)
        self.target_face_id = int(target_face_id)
        self.maximum_target_offset_m = float(maximum_target_offset_m)

    def plan(self, data, selected_finger, active, forces, normals):
        started = perf_counter()
        cfg, h = self.config, self.h
        active, forces, normals = np.asarray(active, bool), np.asarray(forces, float), np.asarray(normals, float)
        if (selected_finger not in range(4) or active.shape != (4,) or forces.shape != (4,)
                or normals.shape != (4, 3) or not np.isfinite(forces).all()
                or not np.isfinite(normals).all() or np.any(forces < 0)):
            raise ValueError('Expected selected finger and finite actual four-finger observations')
        diag = dict(selected_finger=int(selected_finger), candidates=[],
            authority='STATIC_GEOMETRY_ONLY_NO_CONTACT_OR_COVERAGE', physical_recontact_tested=False,
            wrist_and_unselected_fingers_fixed=True, target_policy='EXPLICIT_NEARBY_ORIGINAL_MESH_FACE',
            target_mesh_face_id=self.target_face_id, target_face_barycentric=self.barycentric.tolist())
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
        nearest, _, _ = trimesh.proximity.closest_point(self.graph.mesh,
            (data.geom_xpos[geom]-h.object_position_m)[None])
        offset = float(np.linalg.norm(self.point_local-nearest[0]))
        diag['target_offset_from_measured_nearest_m'] = offset
        if offset > self.maximum_target_offset_m+1e-12:
            return None, dict(diag, reason='EXPLICIT_TARGET_OUTSIDE_NEARBY_SURFACE_RADIUS')
        normal = self.target_normal
        surface_point = h.object_position_m+self.point_local
        target_support = surface_point+cfg.target_clearance_m*normal
        columns = np.arange(7+4*selected_finger, 11+4*selected_finger)
        dofs = np.r_[h.arm_dof_adrs, h.hand_dof_adrs][columns]
        lo, hi = self.planner.lower[columns], self.planner.upper[columns]
        def update(q_selected):
            q = root.copy(); q[columns] = q_selected
            set_joint_state(h, self.scratch, q)
        def residual(q_selected):
            update(q_selected)
            frame = self.scratch.geom_xmat[geom].reshape(3, 3)
            support = pad_support_vector(h.model, geom, frame, normal)
            return self.scratch.geom_xpos[geom]-support-target_support
        def jacobian(q_selected):
            update(q_selected)
            jp, jr = np.zeros((3, h.model.nv)), np.zeros((3, h.model.nv))
            mujoco.mj_jacGeom(h.model, self.scratch, jp, jr, geom)
            derivative = np.zeros((3, 4))
            if h.model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
                derivative = ellipsoid_support_derivative(self.scratch.geom_xmat[geom].reshape(3, 3),
                    h.model.geom_size[geom], normal, jr[:, dofs])
            return jp[:, dofs]-derivative
        baseline = float(np.linalg.norm(residual(root[columns])))
        diag.update(anchor_finger=anchor, initial_support_gap_m=baseline,
            target_surface_point_world_m=surface_point.tolist(), target_outward_normal_world=normal.tolist(),
            target_clearance_m=cfg.target_clearance_m,
            unchanged_joint_indices=np.flatnonzero(~np.isin(np.arange(23), columns)).tolist())
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
                diag['candidates'].append(dict(meta, reason='DYNAMIC_SUPPORT_IK_RESIDUAL')); continue
            if any(np.max(abs(candidate-old)) < 1e-4 for old, _ in geometries):
                diag['candidates'].append(dict(meta, reason='DUPLICATE_GEOMETRIC_CANDIDATE')); continue
            geometries.append((candidate, meta))
        geometries.sort(key=lambda item:np.max(abs(item[0]-root)))
        for candidate, meta in geometries:
            valid, edge = self._dense_edge(root, candidate, columns)
            if not valid:
                diag['candidates'].append(dict(meta, **edge)); continue
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
