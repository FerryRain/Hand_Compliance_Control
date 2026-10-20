"""Shared geometric IK, trajectory planning, and hardware-only integrity."""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
import mujoco
import numpy as np

from .contracts import WristTarget, WholeHandTarget, array_digest
from .physics import joint_state, pose, position_jacobian, quaternion, rotation, set_joint_state, tip_positions


@dataclass(frozen=True)
class GeometricPlan:
    root_q: np.ndarray
    goal_q: np.ndarray
    duration_s: float
    root_pose: np.ndarray
    goal_pose: np.ndarray
    latency_s: float
    wrist_error_m: float
    wrist_error_rad: float
    fingertip_error_m: float
    digest: str

    def sample(self, elapsed_s):
        u = float(np.clip(elapsed_s / self.duration_s, 0, 1))
        s = u * u * u * (10 - 15 * u + 6 * u * u)
        return self.root_q + s * (self.goal_q - self.root_q)


class SharedGeometricTrajectoryPlanner:
    """Both cells use this IK and time parameterization; B adds tip objectives."""
    def __init__(self, handles, config):
        self.h = handles
        self.config = config
        self.scratch = mujoco.MjData(handles.model)
        ranges = np.vstack((handles.arm_joint_ranges_rad, handles.hand_joint_ranges_rad))
        self.lower = ranges[:, 0] + config.joint_margin_rad
        self.upper = ranges[:, 1] - config.joint_margin_rad

    def solve(self, q_root, target, *, iterations=100):
        if not isinstance(target, (WristTarget, WholeHandTarget)):
            raise TypeError('Only versioned geometric targets are allowed')
        full = isinstance(target, WholeHandTarget)
        wrist = target.wrist if full else target
        desired_rotation = rotation(wrist.pose_world[3:])
        q = np.clip(np.asarray(q_root).copy(), self.lower, self.upper)
        active = np.arange(23) if full else np.arange(7)
        for _ in range(iterations):
            set_joint_state(self.h, self.scratch, q)
            current = pose(self.h, self.scratch)
            ep = wrist.pose_world[:3] - current[:3]
            er = (desired_rotation * rotation(current[3:]).inv()).as_rotvec()
            jp, jr = position_jacobian(self.h, self.scratch, self.h.palm_site_id)
            errors = [ep, 0.10 * er]
            jacobians = [jp, 0.10 * jr]
            tip_error = np.empty((0, 3))
            if full:
                tip_error = target.fingertip_positions_world_m - tip_positions(self.h, self.scratch)
                for i, site in enumerate(self.h.tip_site_ids):
                    jt, _ = position_jacobian(self.h, self.scratch, site)
                    errors.append(tip_error[i])
                    jacobians.append(jt)
            if (np.linalg.norm(ep) < 0.0003 and np.linalg.norm(er) < 0.005
                    and (not full or np.max(np.linalg.norm(tip_error, axis=1)) < 0.0004)):
                break
            j = np.vstack(jacobians)[:, active]
            e = np.concatenate(errors)
            damping = 0.003
            inverse = j.T @ np.linalg.solve(j @ j.T + damping**2 * np.eye(len(e)), np.eye(len(e)))
            delta = inverse @ e
            # Posture regularization acts in the geometric null space only.
            null = np.eye(len(active)) - inverse @ j
            delta += 0.015 * null @ (np.asarray(q_root)[active] - q[active])
            q[active] = np.clip(q[active] + np.clip(0.65 * delta, -0.06, 0.06),
                               self.lower[active], self.upper[active])
        set_joint_state(self.h, self.scratch, q)
        current = pose(self.h, self.scratch)
        position_error = float(np.linalg.norm(current[:3] - wrist.pose_world[:3]))
        orientation_error = float((rotation(current[3:]) * desired_rotation.inv()).magnitude())
        fingertip_error = (float(np.max(np.linalg.norm(tip_positions(self.h, self.scratch)
                          - target.fingertip_positions_world_m, axis=1))) if full else 0.0)
        return q, position_error, orientation_error, fingertip_error

    def collision_reason(self, q):
        set_joint_state(self.h, self.scratch, q)
        tips = set(int(g) for g in self.h.tip_geom_ids)
        object_geoms = set(int(g) for g in getattr(self.h, 'object_geom_ids', [self.h.object_geom_id]))
        for index in range(self.scratch.ncon):
            c = self.scratch.contact[index]
            a, b = int(c.geom1), int(c.geom2)
            if object_geoms.intersection((a, b)):
                other = b if a in object_geoms else a
                if other not in tips and c.dist < -0.0005:
                    return 'NON_TIP_OBJECT_COLLISION'
            elif c.dist < -0.002:
                # The fixture has no ground; contacts among robot links are
                # physical self collisions, apart from same/adjacent bodies.
                ba, bb = int(self.h.model.geom_bodyid[a]), int(self.h.model.geom_bodyid[b])
                parents = self.h.model.body_parentid
                if ba != bb and parents[ba] != bb and parents[bb] != ba:
                    return 'SELF_COLLISION'
        return None

    def plan(self, measured_data, target):
        start = perf_counter()
        q_root = joint_state(self.h, measured_data)
        root_pose = pose(self.h, measured_data)
        q, ep, er, ef = self.solve(q_root, target)
        config = self.config
        if ep > config.goal_position_tolerance_m or er > config.goal_rotation_tolerance_rad:
            return None, {'reason': 'WRIST_IK_RESIDUAL', 'position_m': ep, 'rotation_rad': er}
        if isinstance(target, WholeHandTarget) and ef > config.fingertip_tolerance_m:
            return None, {'reason': 'FINGERTIP_IK_RESIDUAL', 'position_m': ef}
        wrist = target.wrist if isinstance(target, WholeHandTarget) else target
        distance = np.linalg.norm(wrist.pose_world[:3] - root_pose[:3])
        angle = (rotation(wrist.pose_world[3:]) * rotation(root_pose[3:]).inv()).magnitude()
        # Quintic smoothstep max derivative = 1.875, not 1.
        duration = max(config.prefix_s, 1.875 * float(distance) / config.wrist_speed_m_s,
                       1.875 * float(angle) / config.wrist_rotation_speed_rad_s,
                       1.875 * float(np.max(np.abs(q - q_root))) / config.trajectory_joint_speed_rad_s)
        for u in np.linspace(0, 1, 21):
            s = u**3 * (10 - 15*u + 6*u*u)
            reason = self.collision_reason(q_root + s * (q - q_root))
            if reason:
                return None, {'reason': reason, 'path_fraction': float(u)}
        plan = GeometricPlan(q_root.copy(), q.copy(), duration, root_pose,
                             wrist.pose_world.copy(), perf_counter() - start, ep, er, ef,
                             array_digest(np.vstack((q_root, q))))
        return plan, None


class GenericTrajectoryExecutor:
    """Short prefixes with fresh measured feedback; no contact task veto."""
    def __init__(self, config):
        self.config = config
        self.plan = None
        self.elapsed = 0.0
        self.last_timestamp = None
        self.prefix_boundaries = 0
        self.contact_task_vetoes = 0

    def commit(self, plan, measured_q, timestamp):
        if not np.allclose(plan.root_q, measured_q, atol=1e-10, rtol=0):
            raise ValueError('STALE_ROOT')
        self.plan, self.elapsed = plan, 0.0
        self.last_timestamp = timestamp

    def step(self, timestamp, measured_q, *, advance=True):
        if self.last_timestamp is not None and timestamp <= self.last_timestamp:
            raise ValueError('Measured timestamps must advance')
        self.last_timestamp = timestamp
        if self.plan is None:
            return np.asarray(measured_q).copy()
        old_prefix = int(self.elapsed / self.config.prefix_s)
        if advance:
            self.elapsed += self.config.dt_s
        if int(self.elapsed / self.config.prefix_s) > old_prefix:
            self.prefix_boundaries += 1
        return self.plan.sample(self.elapsed)

    @property
    def finished(self):
        return self.plan is not None and self.elapsed >= self.plan.duration_s
