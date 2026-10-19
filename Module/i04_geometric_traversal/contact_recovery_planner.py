"""Independent, bounded wrist-pivot proposal for a missing finger.

This module is not integrated with the frozen whole-surface experiment. It
queries geometry on scratch data only. Native execution, measured support,
force guards, MCC, finger search and recovery confirmation belong to its caller.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import mujoco
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from .contracts import ExperimentConfig, WristTarget
from .full_surface import JointPath, pad_support_vector
from .physics import joint_state, pose, quaternion, rotation, set_joint_state
from .sliding_safety import SlidingSafetyPlanner, arm_self_contacts


@dataclass(frozen=True)
class ContactRecoveryConfig:
    angles_deg: tuple = (3., 6.)
    angular_waypoint_deg: float = 1.5
    minimum_gap_improvement_m: float = .001
    anchor_support_drift_m: float = .0008
    anchor_center_drift_m: float = .001
    secondary_anchor_drift_m: float = .002
    maximum_anchor_inward_m: float = .0002
    wrist_speed_m_s: float = .004
    rotation_speed_rad_s: float = .035
    joint_speed_rad_s: float = .15
    force_pause_n: float = 8.
    position_tolerance_m: float = .00025
    rotation_tolerance_rad: float = .003

    def __post_init__(self):
        if not self.angles_deg or any(not 0. < x <= 6. for x in self.angles_deg):
            raise ValueError('Recovery angles must be positive and at most 6 degrees')
        for key in self.__dataclass_fields__:
            if key != 'angles_deg' and (not np.isfinite(getattr(self, key)) or getattr(self, key) <= 0.):
                raise ValueError(f'{key} must be finite and positive')


@dataclass
class AnchoredRecoveryPath(JointPath):
    anchor_finger: int
    selected_finger: int


class ContactRecoveryPlanner:
    """Propose a small pivot about an actually loaded pad's support point.

    ``plan(data, selected_finger, active, forces, normals)`` returns a path or
    None and JSON-compatible diagnostics. It preserves the measured hand shape
    throughout the wrist path. A geometric anchor is never a claimed contact.
    """
    def __init__(self, h, graph, config=None):
        self.h, self.graph = h, graph
        self.config = config or ContactRecoveryConfig()
        self.planner = SlidingSafetyPlanner(h, ExperimentConfig())
        self.scratch = mujoco.MjData(h.model)

    def _support(self, data, finger, normal):
        geom = int(self.h.tip_geom_ids[finger])
        return data.geom_xpos[geom] - pad_support_vector(self.h.model, geom,
            data.geom_xmat[geom].reshape(3, 3), normal)

    def _gap(self, data, finger):
        geom = int(self.h.tip_geom_ids[finger])
        center = data.geom_xpos[geom]
        point, _, face = trimesh.proximity.closest_point(self.graph.mesh,
            (center-self.h.object_position_m)[None])
        normal = np.asarray(self.graph.mesh.face_normals)[int(face[0])]
        normal = normal / np.linalg.norm(normal)
        support = pad_support_vector(self.h.model, geom, data.geom_xmat[geom].reshape(3, 3), normal)
        target = self.h.object_position_m + point[0] + support - .0005*normal
        return float(np.linalg.norm(target-center))

    def plan(self, data, selected_finger, active, forces, normals):
        started = perf_counter()
        cfg = self.config
        active = np.asarray(active, dtype=bool)
        forces, normals = np.asarray(forces), np.asarray(normals)
        if (selected_finger not in range(4) or active.shape != (4,) or forces.shape != (4,)
                or normals.shape != (4, 3) or not np.all(np.isfinite(forces))
                or not np.all(np.isfinite(normals)) or np.any(forces < 0.)):
            raise ValueError('Expected a physical finger and finite four-finger measurements')
        diag = dict(selected_finger=int(selected_finger), candidates=[],
            authority='GT_GEOMETRY_ONLY_NO_CONTACT_OR_COVERAGE', physical_recontact_tested=False)
        if active[selected_finger]:
            return None, dict(diag, reason='SELECTED_FINGER_ALREADY_ACTIVE')
        if np.any(forces >= cfg.force_pause_n) or arm_self_contacts(self.h, data):
            return None, dict(diag, reason='MEASURED_FORCE_OR_SELF_CONTACT')
        anchors = np.flatnonzero(active & (forces > .2) & (forces < cfg.force_pause_n)
                                & (np.linalg.norm(normals, axis=1) > .5))
        if not len(anchors):
            return None, dict(diag, reason='NO_MEASURED_SAFE_ANCHOR')
        root, root_pose = joint_state(self.h, data), pose(self.h, data)
        root_rotation = rotation(root_pose[3:])
        centers = data.geom_xpos[self.h.tip_geom_ids].copy()
        baseline_gap = self._gap(data, selected_finger)
        diag['initial_target_gap_m'] = baseline_gap
        choices = []
        # Try every real anchor; no simulated contact is promoted to support.
        for anchor in anchors:
            normal = normals[anchor]/np.linalg.norm(normals[anchor])
            seed = np.eye(3)[int(np.argmin(abs(normal)))]
            tangent = np.cross(normal, seed); tangent /= np.linalg.norm(tangent)
            axes = (tangent, np.cross(normal, tangent), normal)
            support_points = {int(i):self._support(data, int(i), normals[i]/np.linalg.norm(normals[i]))
                              for i in anchors}
            pivot = support_points[int(anchor)]
            geom = int(self.h.tip_geom_ids[anchor])
            root_frame = data.geom_xmat[geom].reshape(3, 3).copy()
            for axis_index, axis in enumerate(axes):
                for degrees in cfg.angles_deg:
                    for sign in (-1., 1.):
                        angle = np.deg2rad(degrees)*sign
                        meta = dict(anchor_finger=int(anchor), axis_index=axis_index,
                                    axis_world=axis.tolist(), angle_rad=float(angle))
                        waypoints = [root.copy()]
                        durations = []
                        failure = None
                        max_drift = max_inward = max_secondary = 0.
                        previous_pose = root_pose.copy()
                        fractions = np.linspace(0., 1., int(np.ceil(degrees/cfg.angular_waypoint_deg))+1)[1:]
                        for fraction in fractions:
                            delta = Rotation.from_rotvec(axis*angle*fraction)
                            desired_rotation = delta*root_rotation
                            desired_center = pivot + pad_support_vector(self.h.model, geom,
                                delta.as_matrix()@root_frame, normal)
                            desired_position = desired_center - delta.apply(centers[anchor]-root_pose[:3])
                            target = WristTarget(np.r_[desired_position, quaternion(desired_rotation)])
                            q, ep, er, _ = self.planner.solve(waypoints[-1], target, iterations=65)
                            # Wrist IK clips its fixed full vector. Keep the measured
                            # fingers exactly; later search/MCC is the caller's job.
                            q[7:] = root[7:]
                            if ep > cfg.position_tolerance_m or er > cfg.rotation_tolerance_rad:
                                failure = 'WRIST_IK_RESIDUAL'; break
                            previous = waypoints[-1]
                            if not self.planner.edge_free(previous, q):
                                failure = 'STRICT_COLLISION_EDGE'; break
                            steps = max(5, int(np.ceil(np.max(abs(q-previous))/.005))+1)
                            for u in np.linspace(0., 1., steps):
                                trial = previous+u*(q-previous)
                                set_joint_state(self.h, self.scratch, trial)
                                support_delta = self._support(self.scratch, int(anchor), normal)-pivot
                                drift = float(np.linalg.norm(support_delta))
                                inward = -float(normal@support_delta)
                                center_drift = float(np.linalg.norm(self.scratch.geom_xpos[geom]-centers[anchor]))
                                max_drift, max_inward = max(max_drift, drift), max(max_inward, inward)
                                if (drift > cfg.anchor_support_drift_m or inward > cfg.maximum_anchor_inward_m
                                        or center_drift > cfg.anchor_center_drift_m):
                                    failure = 'PRIMARY_ANCHOR_GEOMETRY'; break
                                for other in anchors:
                                    if other == anchor: continue
                                    other_normal = normals[other]/np.linalg.norm(normals[other])
                                    other_delta = self._support(self.scratch, int(other), other_normal)-support_points[int(other)]
                                    secondary = float(np.linalg.norm(other_delta))
                                    max_secondary = max(max_secondary, secondary)
                                    if secondary > cfg.secondary_anchor_drift_m or -float(other_normal@other_delta) > cfg.maximum_anchor_inward_m:
                                        failure = 'SECONDARY_ANCHOR_GEOMETRY'; break
                                if failure: break
                            if failure: break
                            duration = max(.2,
                                1.875*np.linalg.norm(target.pose_world[:3]-previous_pose[:3])/cfg.wrist_speed_m_s,
                                1.875*(rotation(target.pose_world[3:])*rotation(previous_pose[3:]).inv()).magnitude()/cfg.rotation_speed_rad_s,
                                1.875*np.max(abs(q-previous))/cfg.joint_speed_rad_s)
                            waypoints.append(q); durations.append(float(duration))
                            previous_pose = target.pose_world.copy()
                        meta.update(maximum_anchor_support_drift_m=max_drift,
                                    maximum_anchor_inward_m=max_inward,
                                    maximum_secondary_support_drift_m=max_secondary)
                        if failure:
                            diag['candidates'].append(dict(meta, reason=failure)); continue
                        set_joint_state(self.h, self.scratch, waypoints[-1])
                        gap = self._gap(self.scratch, selected_finger)
                        improvement = baseline_gap-gap
                        meta.update(final_target_gap_m=gap, improvement_m=improvement,
                                    duration_s=float(sum(durations)))
                        if improvement < cfg.minimum_gap_improvement_m:
                            diag['candidates'].append(dict(meta, reason='INSUFFICIENT_GAP_IMPROVEMENT')); continue
                        path = AnchoredRecoveryPath(np.array(waypoints), np.array(durations), root_pose.copy(),
                            pose(self.h, self.scratch), perf_counter()-started, int(anchor), int(selected_finger))
                        diag['candidates'].append(dict(meta, reason='GEOMETRIC_CANDIDATE'))
                        choices.append((improvement, -path.duration_s, path, meta))
        if not choices:
            return None, dict(diag, reason='NO_SAFE_IMPROVING_PIVOT', latency_s=perf_counter()-started)
        _, _, path, meta = max(choices, key=lambda item:item[:2])
        return path, dict(diag, reason='GEOMETRIC_PIVOT_SELECTED', selected=meta,
                          latency_s=perf_counter()-started)
