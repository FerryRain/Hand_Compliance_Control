"""Basic 23-DoF kinematics and actual MuJoCo contact measurements."""
from __future__ import annotations

from pathlib import Path
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from Module.fr3_leap import ARM_HOME_Q, HAND_NATURAL_Q, FullRobotModelConfig, build_full_robot
from Module.e05_physics.scene import FINGERS
from Module.module_2_fingertip_mcc import FingertipMCC, MCCConfig


def pose(handles, data):
    quaternion = np.zeros(4)
    mujoco.mju_mat2Quat(quaternion, data.site_xmat[handles.palm_site_id])
    return np.r_[data.site_xpos[handles.palm_site_id], quaternion]


def rotation(quaternion):
    return Rotation.from_quat(np.asarray(quaternion)[[1, 2, 3, 0]])


def quaternion(rot):
    return rot.as_quat()[[3, 0, 1, 2]]


def joint_state(handles, data):
    return np.r_[data.qpos[handles.arm_qpos_adrs], data.qpos[handles.hand_qpos_adrs]]


def set_joint_state(handles, data, q):
    data.qpos[handles.arm_qpos_adrs] = q[:7]
    data.qpos[handles.hand_qpos_adrs] = q[7:]
    mujoco.mj_forward(handles.model, data)


def set_control(handles, data, q):
    data.ctrl[handles.arm_actuator_ids] = q[:7]
    data.ctrl[handles.hand_actuator_ids] = q[7:]


def position_jacobian(handles, data, site):
    jp, jr = np.zeros((3, handles.model.nv)), np.zeros((3, handles.model.nv))
    mujoco.mj_jacSite(handles.model, data, jp, jr, int(site))
    order = np.r_[handles.arm_dof_adrs, handles.hand_dof_adrs]
    return jp[:, order], jr[:, order]


def tip_positions(handles, data):
    return data.site_xpos[handles.tip_site_ids].copy()


def pad_support(handles, data, finger, normal):
    geom = int(handles.tip_geom_ids[finger])
    local_normal = data.geom_xmat[geom].reshape(3, 3).T @ normal
    return float(np.abs(local_normal) @ handles.model.geom_size[geom])


def contact_measurement(handles, data, threshold):
    forces = np.zeros(4)
    points = np.zeros((4, 3))
    normals = np.zeros((4, 3))
    valid = np.zeros(4, dtype=bool)
    lookup = {int(g): i for i, g in enumerate(handles.tip_geom_ids)}
    wrench = np.zeros(6)
    unsafe = []
    penetration = 0.0
    object_geoms = set(int(g) for g in getattr(handles, 'object_geom_ids', [handles.object_geom_id]))
    for index in range(data.ncon):
        contact = data.contact[index]
        a, b = int(contact.geom1), int(contact.geom2)
        mujoco.mj_contactForce(handles.model, data, index, wrench)
        magnitude = max(0.0, float(wrench[0]))
        if not object_geoms.intersection((a, b)):
            ba, bb = int(handles.model.geom_bodyid[a]), int(handles.model.geom_bodyid[b])
            parents = handles.model.body_parentid
            if (contact.dist < -0.002 and magnitude > 0.5 and ba != bb
                    and parents[ba] != bb and parents[bb] != ba):
                names = [mujoco.mj_id2name(handles.model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in (a, b)]
                unsafe.append('SELF_COLLISION:' + '/'.join(str(n) for n in names))
            continue
        other = b if a in object_geoms else a
        penetration = max(penetration, -float(contact.dist))
        if other not in lookup:
            if magnitude > 0.05:
                unsafe.append(mujoco.mj_id2name(handles.model, mujoco.mjtObj.mjOBJ_GEOM, other))
            continue
        finger = lookup[other]
        normal = np.asarray(contact.frame[:3]).copy()
        # MuJoCo normal points geom1 -> geom2. Outward object normal points
        # from object into the pad, independent of contact ordering.
        if b in object_geoms:
            normal *= -1
        forces[finger] += magnitude
        points[finger] += magnitude * np.asarray(contact.pos)
        normals[finger] += magnitude * normal
    for finger in range(4):
        if forces[finger] >= threshold and np.linalg.norm(normals[finger]) > 1e-10:
            points[finger] /= forces[finger]
            normals[finger] /= np.linalg.norm(normals[finger])
            valid[finger] = True
    return forces, points, normals, valid, unsafe, penetration


class BasicPassiveMCC:
    """Only measured-contact normal corrections; free fingers do not search."""
    def __init__(self, handles, dt_s):
        self.handles = handles
        self.controllers = [FingertipMCC(MCCConfig(
            virtual_mass=0.08, damping=14.0, stiffness=25.0, dt_s=dt_s,
            max_offset_m=0.010, max_velocity_m_s=0.030,
            max_acceleration_m_s2=20.0)) for _ in range(4)]
        self.was_active = np.zeros(4, dtype=bool)

    def command(self, data, nominal_q, forces, normals, active, desired_force):
        result = np.asarray(nominal_q).copy()
        h = self.handles
        for finger in range(4):
            if not active[finger]:
                self.controllers[finger].reset()
                self.was_active[finger] = False
                continue
            if not self.was_active[finger]:
                self.controllers[finger].reset()
            self.was_active[finger] = True
            command = self.controllers[finger].step(
                [0, 0, 0], -normals[finger], desired_force, float(forces[finger]))
            jp, _ = position_jacobian(h, data, h.tip_site_ids[finger])
            columns = [int(name) for name in FINGERS[finger].joint_names]
            jf = jp[:, np.asarray(columns) + 7]
            cartesian = -normals[finger] * command.offset_m
            dq = jf.T @ np.linalg.solve(jf @ jf.T + 1e-6 * np.eye(3), cartesian)
            result[np.asarray(columns) + 7] += np.clip(dq, -0.18, 0.18)
        return result


def make_plant(mesh_path, dt_s):
    handles = build_full_robot(FullRobotModelConfig(
        surface='bunny', timestep_s=dt_s, gravity_m_s2=0.0,
        arm_kp=1800.0, arm_damping_ratio=0.9,
        hand_kp=60.0, hand_damping_ratio=1.5,
        bunny_visual_mesh_path=str(Path(mesh_path).resolve()), bunny_collision_mode='sdf'))
    # The old isolation fixture omitted arm/object collision masks. New I04
    # must actually observe such collisions rather than silently allow them.
    handles.model.geom_conaffinity[handles.object_geom_id] |= 4
    return handles
