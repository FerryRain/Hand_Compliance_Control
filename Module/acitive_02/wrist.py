"""Apply FR3-style Cartesian wrist compliance to the arm-free pear plant."""

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from ..i04_geometric_traversal.fingertip_hand import DT
from ..i04_geometric_traversal.physics import quaternion, rotation
from ..module_4_whole_hand_mcc.wrist_mcc import WristMCC, WristMCCConfig

WRIST_WELD_TIME_S = .022
PAD_LOAD_FREE_N = 2.
FORCE_CAP_N = 75.
TORQUE_CAP_NM = 5.
FILTER_ALPHA = .28
PALM_PROXIMITY_M = .012
PALM_LOOKAHEAD_S = .12
PALM_CLEARANCE_M = .006
PROXIMITY_STIFFNESS_N_M = 2000.


def wrist_config():
    return WristMCCConfig(
        virtual_mass=(1.1, 1.1, 1.1, .14, .14, .14),
        damping=(42., 42., 42., 3.2, 3.2, 3.2),
        stiffness=(240., 240., 240., 19., 19., 19.),
        dt_s=DT,
        max_abs_offset=(.040, .040, .040, .14, .14, .14),
        max_abs_velocity=(.12, .12, .12, .45, .45, .45),
        max_abs_acceleration=(4., 4., 4., 8., 8., 8.),
    )


class WristFeedback:
    """Use object-on-hand load at the palm origin to yield around the planner reference."""

    def __init__(self, *, palm_proximity=True):
        self.palm_proximity = palm_proximity
        self.controller = WristMCC(wrist_config())
        self.wrench = np.zeros(6)
        self.wrench_buffer = np.zeros(6)
        self.selection = np.eye(6)
        self.desired = np.zeros(6)
        self.offset = np.zeros(6)
        self.proximity_force_n = 0.

    def measure(self, hand, data):
        group_wrench = np.zeros((5, 6))
        pad_load = np.zeros(4)
        origin = data.xpos[hand.palm]
        palm_twist = None
        proximity = np.zeros(6)
        self.proximity_force_n = 0.
        for index, contact in enumerate(data.contact):
            left, right = int(contact.geom1), int(contact.geom2)
            if left in hand.objects:
                geometry, direction = right, contact.frame[:3]
            elif right in hand.objects:
                geometry, direction = left, -contact.frame[:3]
            else:
                continue
            offset = contact.pos - origin
            if self.palm_proximity and geometry == hand.palm_geom:
                if palm_twist is None:
                    palm_twist = np.empty(6)
                    mujoco.mj_objectVelocity(hand.model, data, mujoco.mjtObj.mjOBJ_BODY, hand.palm, palm_twist, 0)
                speed = direction @ (palm_twist[3:] + np.cross(palm_twist[:3], offset))
                forecast = contact.dist + PALM_LOOKAHEAD_S * min(0., speed)
                magnitude = min(45., PROXIMITY_STIFFNESS_N_M * max(0., PALM_CLEARANCE_M - forecast))
                if magnitude > self.proximity_force_n:
                    self.proximity_force_n = magnitude
                    proximity[:3] = magnitude * direction
                    proximity[3:] = np.cross(offset, proximity[:3])
            if contact.dist > 0.:
                continue
            mujoco.mj_contactForce(hand.model, data, index, self.wrench_buffer)
            normal_load = max(0., float(self.wrench_buffer[0]))
            force = normal_load * direction
            moment = np.cross(offset, force)
            finger = hand.pad_index.get(geometry)
            group = 4 if finger is None else finger
            group_wrench[group, :3] += force
            group_wrench[group, 3:] += moment
            if finger is not None:
                pad_load[finger] += normal_load
        # The finger loops own normal preload. Their excess and all rigid-link
        # contacts contribute to the resultant wrist wrench.
        scale = np.maximum(pad_load - PAD_LOAD_FREE_N, 0.) / np.maximum(pad_load, 1e-9)
        wrench = proximity + group_wrench[4] + (scale[:, None] * group_wrench[:4]).sum(axis=0)
        for axes, limit in ((slice(0, 3), FORCE_CAP_N), (slice(3, 6), TORQUE_CAP_NM)):
            magnitude = np.linalg.norm(wrench[axes])
            wrench[axes] *= min(1., limit / max(magnitude, 1e-9))
        return wrench

    def forecast(self, nominal):
        position = nominal[:3] + self.offset[:3]
        orientation = Rotation.from_rotvec(self.offset[3:].copy()) * rotation(nominal[3:])
        return np.r_[position, quaternion(orientation)]

    def command(self, hand, data, nominal):
        load = self.measure(hand, data)
        self.wrench += FILTER_ALPHA * (load - self.wrench)
        result = self.controller.step(nominal, self.desired, self.wrench, self.selection)
        self.offset = result.offset
        return result.pose_command
