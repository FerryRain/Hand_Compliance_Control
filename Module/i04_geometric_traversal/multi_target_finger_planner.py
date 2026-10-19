"""Bounded nearby-surface alternatives for a fixed-wrist single-finger path.

The frozen nearest-point planner remains the fast first attempt. Only if it
cannot find a path are a bounded, spatially dispersed set of original-mesh
face centers tried through ExplicitSurfaceFingerPlanner. Every accepted path
retains its dynamic pad-support residual, native joint margins, dense strict
collision checks and speed limits. The shortest duration among the attempted
safe neighbor paths is returned; this is neither a global optimum nor proof
that an unselected surface target is unreachable. No physical recovery or
coverage is inferred from a geometric path.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import mujoco
import numpy as np
from scipy.optimize import lsq_linear
import trimesh

from .explicit_surface_finger_planner import ExplicitSurfaceFingerPlanner
from .finger_reconfiguration_planner import FingerReconfigurationPlanner, FingerReconfigurationConfig
from .full_surface import pad_support_vector
from .physics import joint_state


@dataclass(frozen=True)
class MultiTargetFingerConfig:
    maximum_targets: int = 32  # Includes the original nearest-point attempt.
    maximum_surface_offset_m: float = .040
    minimum_candidate_spacing_m: float = .004
    linear_priority_fraction: float = .5

    def __post_init__(self):
        if not isinstance(self.maximum_targets, int) or not 1 <= self.maximum_targets <= 128:
            raise ValueError('Expected 1..128 bounded total target attempts')
        if (not np.isfinite(self.maximum_surface_offset_m)
                or not 0. < self.maximum_surface_offset_m <= .040):
            raise ValueError('Surface alternatives must stay within 40 mm')
        if (not np.isfinite(self.minimum_candidate_spacing_m)
                or not 0. < self.minimum_candidate_spacing_m <= self.maximum_surface_offset_m):
            raise ValueError('Candidate spacing must be positive and within the neighborhood')
        if (not np.isfinite(self.linear_priority_fraction)
                or not 0. <= self.linear_priority_fraction <= 1.):
            raise ValueError('Linear priority fraction must be in [0, 1]')


def dispersed_candidate_indices(points, nearest, scores, count, spacing, linear_fraction=.5):
    """Mix well-spaced low linear residuals with farthest spatial coverage.

    Linear fits only prioritize finite samples; they never certify feasibility.
    The farthest-point half preserves directions that a local linearization
    ranks poorly. All comparisons use physical surface-point distances.
    """
    points, nearest, scores = np.asarray(points, float), np.asarray(nearest, float), np.asarray(scores, float)
    if points.shape != (len(scores), 3) or nearest.shape != (3,) or not all(
            np.isfinite(x).all() for x in (points, nearest, scores)):
        raise ValueError('Expected finite surface points, reference point and scores')
    if count <= 0 or not len(points):
        return []
    distance = np.linalg.norm(points-nearest, axis=1)
    chosen = []
    minimum_distance = distance.copy()
    # Numerical noise among approximately zero residuals must not determine
    # which mesh triangle is favored. Distance and stable index break ties.
    order = np.lexsort((np.arange(len(points)), distance, np.round(scores, 8)))
    linear_quota = min(count, int(np.ceil(count*linear_fraction)))
    for index in order:
        if len(chosen) >= linear_quota:
            break
        if minimum_distance[index] < spacing-1e-12:
            continue
        chosen.append(int(index))
        minimum_distance = np.minimum(minimum_distance, np.linalg.norm(points-points[index], axis=1))
    while len(chosen) < count:
        index = int(np.argmax(minimum_distance))
        if minimum_distance[index] < spacing-1e-12:
            break
        chosen.append(index)
        minimum_distance = np.minimum(minimum_distance, np.linalg.norm(points-points[index], axis=1))
    return chosen


class MultiTargetFingerPlanner:
    """Same plan(data, selected, active, forces, normals) interface as before.

    A nearest-point success returns immediately, preserving the old fast path.
    A fallback considers at most maximum_targets-1 real nearby face centers;
    among their accepted paths, duration selects the result. Wrist and all
    unselected finger joints remain exactly at their measured root values.
    This module is independent and is not installed into a frozen runner.
    """
    def __init__(self, h, graph, config=None, *, finger_config=None):
        self.h, self.graph = h, graph
        self.config = config or MultiTargetFingerConfig()
        self.finger_config = finger_config or FingerReconfigurationConfig()
        self.nearest_planner = FingerReconfigurationPlanner(h, graph, self.finger_config)
        self.planner = self.nearest_planner.planner

    def candidate_targets(self, data, selected_finger):
        """Return bounded explicit mesh targets and diagnostic sampling details."""
        h, cfg = self.h, self.config
        geom = int(h.tip_geom_ids[selected_finger])
        pad_center = data.geom_xpos[geom]
        closest, _, original_faces = trimesh.proximity.closest_point(
            self.graph.mesh, (pad_center-h.object_position_m)[None])
        nearest = closest[0]
        centers = np.asarray(self.graph.mesh.triangles_center)
        distance = np.linalg.norm(centers-nearest, axis=1)
        faces = np.flatnonzero(distance <= cfg.maximum_surface_offset_m+1e-12)
        points = centers[faces]
        root = joint_state(h, data)
        columns = np.arange(7+4*selected_finger, 11+4*selected_finger)
        dofs = np.r_[h.arm_dof_adrs, h.hand_dof_adrs][columns]
        jp, jr = np.zeros((3, h.model.nv)), np.zeros((3, h.model.nv))
        mujoco.mj_jacGeom(h.model, data, jp, jr, geom)
        jac = jp[:, dofs]
        lower = self.planner.lower[columns]-root[columns]
        upper = self.planner.upper[columns]-root[columns]
        frame = data.geom_xmat[geom].reshape(3, 3)
        scores = []
        for point, face in zip(points, faces):
            normal = np.asarray(self.graph.mesh.face_normals[int(face)])
            support = pad_support_vector(h.model, geom, frame, normal)
            error = (h.object_position_m+point+support
                     +self.finger_config.target_clearance_m*normal-pad_center)
            fit = lsq_linear(jac, error, bounds=(lower, upper), method='bvls', tol=1e-10)
            scores.append(float(np.linalg.norm(jac@fit.x-error)))
        chosen = dispersed_candidate_indices(points, nearest, np.asarray(scores),
            cfg.maximum_targets-1, cfg.minimum_candidate_spacing_m, cfg.linear_priority_fraction)
        targets = [dict(face_id=int(faces[index]), point_local_m=points[index].tolist(),
                        distance_from_original_nearest_m=float(distance[faces[index]]),
                        linear_ordering_residual_m=float(scores[index])) for index in chosen]
        return targets, dict(nearest_point_local_m=nearest.tolist(), nearest_face_id=int(original_faces[0]),
            nearby_face_count=len(faces), selected_neighbor_count=len(targets),
            maximum_surface_offset_m=cfg.maximum_surface_offset_m,
            minimum_candidate_spacing_m=cfg.minimum_candidate_spacing_m,
            selection='Bounded linear-priority plus farthest-spatial original face centers',
            linear_model_scope='Center-Jacobian ordering only; acceptance always full nonlinear dynamic support')

    def plan(self, data, selected_finger, active, forces, normals):
        started = perf_counter()
        path, original = self.nearest_planner.plan(data, selected_finger, active, forces, normals)
        diag = dict(authority='STATIC_GEOMETRY_ONLY_NO_CONTACT_OR_COVERAGE',
            selected_finger=int(selected_finger), physical_recontact_tested=False,
            wrist_and_unselected_fingers_fixed=True, no_global_unreachability_or_optimality_claim=True,
            maximum_target_attempts=self.config.maximum_targets, nearest_attempt=original,
            neighbor_attempts=[], attempted_target_count=1)
        if path is not None:
            return path, dict(diag, reason='ORIGINAL_NEAREST_SAFE_PATH',
                selected_target_policy='ORIGINAL_NEAREST', latency_s=perf_counter()-started)
        if original['reason'] != 'NO_SAFE_FIXED_WRIST_RECONFIGURATION':
            return None, dict(diag, reason=original['reason'], latency_s=perf_counter()-started)
        targets, sampling = self.candidate_targets(data, selected_finger)
        diag['sampling'] = sampling
        accepted = []
        for order, target in enumerate(targets):
            adapter = ExplicitSurfaceFingerPlanner(self.h, self.graph, self.finger_config,
                target_point_local_m=target['point_local_m'], target_face_id=target['face_id'],
                maximum_target_offset_m=self.config.maximum_surface_offset_m)
            candidate, trial = adapter.plan(data, selected_finger, active, forces, normals)
            record = dict(order=order, **target, accepted=candidate is not None, planner=trial)
            if candidate is not None:
                record['duration_s'] = float(candidate.duration_s)
                accepted.append((candidate.duration_s, order, candidate, record))
            diag['neighbor_attempts'].append(record)
        diag['attempted_target_count'] += len(targets)
        diag['safe_neighbor_count'] = len(accepted)
        if not accepted:
            return None, dict(diag, reason='NO_SAFE_PATH_AMONG_BOUNDED_SURFACE_TARGETS',
                              latency_s=perf_counter()-started)
        _, _, result, selected = min(accepted, key=lambda item: item[:2])
        return result, dict(diag, reason='SAFE_NEARBY_SURFACE_PATH_SELECTED',
            selected_target_policy='SHORTEST_SAFE_ATTEMPTED_NEIGHBOR', selected=selected,
            latency_s=perf_counter()-started)
