"""Apply the published AUDE palm-axis limit before the single_all wrist forecast."""

from shutil import copyfile

import numpy as np

from PARK.identity import digest, read
from ..acitive_02 import active_a, route_free
from .height import AXIS, CALIBRATION, WORKSPACE_M, project_wrist


class PalmDistance:
    def __init__(self, ticks):
        calibration = read(CALIBRATION)
        self.lower, self.upper = calibration["lower_m"], calibration["upper_m"]
        if not 0 < self.lower < self.upper or not np.array_equal(AXIS, calibration["axis_palm"]):
            raise ValueError("Invalid palm-axis training calibration")
        self.anchor = None
        self.anchor_tick = -1
        self.measured = np.full(ticks + 1, np.nan)
        self.reference_ticks = np.full(ticks + 1, -1, dtype=np.int64)
        self.corrections = []

    def signature(self):
        return dict(height_calibration_sha256=digest(CALIBRATION), height_bounds_m=[self.lower, self.upper],
                    axis_palm=AXIS.tolist())

    def observe(self, tick, touch, wrist):
        live = touch.measured_heads
        if live.any() and touch.loads.max() < 8.:
            self.anchor = touch.points[live].copy()
            self.anchor_tick = tick
        if self.anchor is None:
            raise ValueError("A measured contact is required at the paired start")
        axis = active_a.rotation(wrist[3:7]).apply(AXIS)
        self.reference_ticks[tick] = self.anchor_tick
        self.measured[tick] = np.median((self.anchor - wrist[:3]) @ axis)

    def issue(self, executor, proposal, tick):
        raw = executor.origin + route_free.limit(proposal.position - executor.origin, WORKSPACE_M)
        axis = proposal.orientation.apply(AXIS)
        raw_wrist = np.r_[raw, active_a.quaternion(proposal.orientation)]
        projected, before = project_wrist(raw, executor.origin, axis, self.anchor, self.lower, self.upper)
        if not self.lower <= before <= self.upper:
            proposal = active_a.Posture(projected, proposal.orientation, proposal.joints)
        executor.set_wrist(proposal)
        issued = executor.target_wrist
        after = float(np.median((self.anchor - issued[:3]) @ axis))
        if not self.lower - 1e-10 <= after <= self.upper + 1e-10:
            raise ValueError("Issued wrist target is outside the calibrated palm-axis band")
        correction = float(np.linalg.norm(issued[:3] - raw))
        self.corrections.append(correction)
        return dict(height_plan_tick=tick, height_anchor_tick=self.anchor_tick,
                    height_anchor_points_m=self.anchor.tolist(), raw_wrist=raw_wrist.tolist(),
                    raw_joints_rad=proposal.joints.tolist(), height_raw_m=before,
                    height_issued_m=after, height_correction_m=correction)

    def add_trace(self, trace, stage):
        if not np.isfinite(self.measured).all() or np.any(self.reference_ticks < 0):
            raise ValueError("Incomplete measured wrist height trace")
        trace.update(height_measured_m=self.measured, height_reference_tick=self.reference_ticks)
        copyfile(CALIBRATION, stage / CALIBRATION.name)

    def summary(self):
        height = self.measured[1:]
        corrections = np.asarray(self.corrections)
        return dict(plans=len(corrections), changed=int((corrections > 1e-10).sum()),
                    max_correction_m=float(corrections.max()),
                    measured_in_band_ratio=float(((height >= self.lower) & (height <= self.upper)).mean()))
