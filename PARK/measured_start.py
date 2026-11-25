"""Reconnect global planning to the measured wrist when the old splice loses clearance."""
from types import SimpleNamespace
from time import perf_counter

import numpy as np
from scipy.spatial.transform import Rotation

from Module.wrist_active.palm_progress_recontact import ProgressWorker
from Module.wrist_active.sweep import PREVIEW_S
from PARK.frontier_planner import FrontierPlanner, WatchedWorker
from PARK.core.common import DT, PREVIEW, pack, rot
from PARK.core.geometry import Clearance
from PARK.core.trajectory import Path, extrema, limits
from PARK.planner_adapter import PalmWorker, PalmWrist


def twist(first, last, dt):
    return np.stack(((last[:3]-first[:3])/dt, (rot(last)*rot(first).inv()).as_rotvec()/dt))


def cap(vectors, maximum):
    lengths = np.linalg.norm(vectors, axis=1)
    return vectors * np.minimum(1., maximum/np.maximum(lengths, 1e-12))[:, None]


def measured_prefix(tick, splice, actual, previous, published, clearance, origin, cfg):
    """Forecast only the committed handoff interval, anchored at this measured pose."""
    held = Path(tick, actual.copy())
    steps = splice-tick
    if steps <= 0 or previous is None or published is None:
        return held
    speed = np.array([cfg.linear_speed_m_s, cfg.angular_speed_rad_s])
    accel = np.array([cfg.linear_accel_m_s2, cfg.angular_accel_rad_s2])
    velocity = cap(twist(previous, actual, DT), speed)
    preview_velocity = cap(twist(*published, PREVIEW_S), speed)
    positions = [actual[:3].copy()]
    attitudes = [rot(actual)]
    for step in range(1, steps+1):
        desired = preview_velocity if step*DT <= PREVIEW_S else np.zeros((2,3))
        velocity += cap(desired-velocity, accel*DT)
        # The new nominal tail can start at rest after the buffered preview.
        velocity = cap(velocity, np.minimum(speed, accel*(steps-step)*DT))
        positions.append(positions[-1]+velocity[0]*DT)
        attitudes.append(Rotation.from_rotvec(velocity[1]*DT)*attitudes[-1])
    samples = pack(np.asarray(positions), Rotation.concatenate(attitudes))
    samples[0] = actual
    if (np.any(extrema(samples) > limits(cfg)*(1+1e-8)) or
            np.any(np.linalg.norm(samples[:,:3]-origin[:3],axis=1) > cfg.workspace_radius_m) or
            not clearance.valid(samples).all()):
        return held
    return Path(tick, samples)


class MeasuredStartPlanner(FrontierPlanner):
    previous = published = None

    def plan(self, tick, splice, path, actual, pads, hull, points, normals, incumbent):
        began = perf_counter()
        inputs = (actual, pads, hull, points, normals)
        clearance = Clearance(points,normals,hull,actual,self.base.cfg) if len(points) else None
        nominal_start = path.at(splice)
        reset = clearance is not None and not clearance.valid(nominal_start[None])[0]
        if reset:
            prefix = measured_prefix(tick, splice, actual, self.previous, self.published, clearance,
                                     self.base.origin, self.base.cfg)
            # A new solve must also revalidate the old endpoint if it occurs among the candidates.
            row, tail = super().plan(tick,splice,prefix,*inputs,None)
            if prefix.end > tick and tail is None and row["reason"] == "no_clear_trajectory":
                prefix = Path(tick,actual.copy())
                row, tail = super().plan(tick,splice,prefix,*inputs,None)
            row.update(anchor_pose=actual.tolist(), old_start_pose=nominal_start.tolist(),
                       splice_pose=prefix.at(splice).tolist(), measured_preview=prefix.end>tick)
            if tail is not None:
                # Only the endpoint is consumed by PalmReach's native actuator controller. Install
                # the checked measured prefix in the global reference, never the rejected old one.
                due = splice-PREVIEW if tick else 0
                tail = Path(due,np.vstack((prefix.at(np.arange(due,splice)),tail.samples)))
        else:
            row, tail = super().plan(tick,splice,path,*inputs,incumbent)
        row.update(replanned_from_measured=bool(reset),planning_ms=1000*(perf_counter()-began))
        return row, tail


class MeasuredStartGlobalWorker(WatchedWorker):
    previous = measured = None

    def observe(self, data, touch, tick):
        self.previous, self.measured = self.measured, data.qpos[:7].copy()
        super().observe(data,touch,tick)

    def submit(self, data, touch, tick, lead):
        # A single outstanding solve owns these immutable snapshots until installation.
        self.wrist.planner.previous = self.previous
        self.wrist.planner.published = self.wrist.published
        super().submit(data,touch,tick,lead)

    def install(self, tick, initial=False):
        row = super().install(tick,initial)
        if row and row["replanned_from_measured"] and row["reason"] != "new_target":
            self.target = None
            self.wrist.path = Path(tick,self.measured.copy())
            row["target_invalidated"] = True
        return row


class MeasuredStartWrist(PalmWrist):
    def __init__(self,*args,frontier,**kwargs):
        super().__init__(*args,**kwargs)
        wrist = self.global_wrist
        wrist.planner = MeasuredStartPlanner(wrist.planner,SimpleNamespace(**frontier))
        wrist.published = None

    def command(self,touch):
        command,preview = super().command(touch)
        self.global_wrist.published = (command.copy(),preview.copy())
        return command,preview


class MeasuredStartWorker(PalmWorker):
    def __init__(self,servo,paced):
        ProgressWorker.__init__(self,servo,paced)
        self.global_worker = MeasuredStartGlobalWorker(servo.global_wrist,paced)
        self.reference = None

    def receive(self,row):
        if row and row.get("target_invalidated"):
            self.reference = None
        super().receive(row)
