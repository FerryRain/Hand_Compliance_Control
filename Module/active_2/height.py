"""Add a training-calibrated palm-to-contact height limit to the native AUDE wrist goal."""

from functools import partial
import json
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from PARK.identity import digest, read
from ..acitive_02 import active_a, author_control, route_free


CALIBRATION = Path(__file__).with_name("height_calibration.json")
AXIS = np.array([0., 0., -1.])
WORKSPACE_M = .45


def project_wrist(raw, origin, axis, points, lower, upper):
    """Project the palm-axis height into the band, keeping the original AUDE workspace."""
    surface_axis = float(np.median((points - origin) @ axis))
    delta = raw - origin
    axial = float(delta @ axis)
    height = surface_axis - axial
    if lower <= height <= upper:
        return raw, height
    low = max(lower, surface_axis - WORKSPACE_M)
    high = min(upper, surface_axis + WORKSPACE_M)
    if low > high:
        raise ValueError("The observed surface and candidate orientation admit no wrist goal in the AUDE workspace")
    target_axis = surface_axis - np.clip(height, low, high)
    tangent = delta - axial * axis
    radius = np.sqrt(max(0., WORKSPACE_M**2 - target_axis**2))
    # Inside the original 45 cm workspace this changes only the axial component.
    tangent = route_free.limit(tangent, radius)
    return origin + tangent + target_axis * axis, height


class HeightExecutor(author_control.AuthorExecutor):
    def __init__(self, *args, band, audit):
        super().__init__(*args)
        self.band = band
        self.audit = audit
        self.anchor = None
        self.anchor_tick = -1
        self.tick = -1

    def sample(self, touch, mode, clock):
        self.tick += 1
        valid = touch.measured_heads
        if valid.any() and touch.loads.max() < 8.:
            self.anchor = touch.points[valid].copy()
            self.anchor_tick = self.tick
        if self.anchor is None:
            raise ValueError("The paired initial state must supply a measured contact patch")
        wrist = self.data.qpos[:7]
        axis = active_a.rotation(wrist[3:]).apply(AXIS)
        self.audit["measured"].append(float(np.median((self.anchor - wrist[:3]) @ axis)))
        self.audit["anchor_ticks"].append(self.anchor_tick)
        return super().sample(touch, mode, clock)

    def set_goal(self, posture):
        if self.anchor is None:
            # The native executor sets its initial goal before the first tactile sample.
            return super().set_goal(posture)
        raw = self.origin + route_free.limit(posture.position - self.origin, WORKSPACE_M)
        axis = posture.orientation.apply(AXIS)
        position, raw_height = project_wrist(raw, self.origin, axis, self.anchor, *self.band)
        super().set_goal(active_a.Posture(position, posture.orientation, posture.joints))
        issued_height = float(np.median((self.anchor - self.target_wrist[:3]) @ axis))
        if not self.band[0] - 1e-10 <= issued_height <= self.band[1] + 1e-10:
            raise ValueError("Issued goal violates the palm-axis height band")
        self.audit["plans"].append(dict(
            height_plan_tick=self.tick, height_anchor_tick=self.anchor_tick, height_anchor_points_m=self.anchor.tolist(),
            raw_wrist=np.r_[raw, active_a.quaternion(posture.orientation)].tolist(),
            raw_joints_rad=self.grasp.bound(posture.joints).tolist(), height_raw_m=raw_height,
            height_issued_m=issued_height, height_correction_m=float(np.linalg.norm(position-raw))))


def run(obj, fixtures, out, seconds):
    calibration = read(CALIBRATION)
    band = (calibration["lower_m"], calibration["upper_m"])
    if not 0 < band[0] < band[1] or not np.array_equal(calibration["axis_palm"], AXIS):
        raise ValueError("Invalid training calibration")
    native_code = {p.name: digest(p) for p in (Path(__file__), Path(author_control.__file__),
                   Path(author_control.aude.__file__), Path(route_free.__file__))}
    spec = read(fixtures / "fixtures.json")["objects"][obj]
    extra = dict(condition="aude_height", start=spec["start"], height_calibration_sha256=digest(CALIBRATION),
                 height_bounds_m=list(band), axis_palm=AXIS.tolist(), execute_planned_fingers=True, code=native_code)
    out = Path(out).resolve()
    if out.exists():
        saved = read(out / "run.json")
        expected = dict(**extra, seconds=seconds, seed=spec["seed"], object=obj, initial=spec["integration_state"])
        if (any(saved["signature"].get(k) != v for k, v in expected.items())
                or any(digest(out / f) != h for f, h in saved["hashes"].items())):
            raise ValueError(f"A different trial exists at {out}")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out.parent) as temp:
        stage = Path(temp) / "run"
        audit = dict(plans=[], measured=[], anchor_ticks=[])
        with patch.object(author_control, "AuthorExecutor", partial(HeightExecutor, band=band, audit=audit)):
            author_control.run(obj, fixtures, stage, seconds, planner_feedback="physical")
        receipt = read(stage / "run.json")
        plans = [json.loads(s) for s in (stage / "plans.jsonl").read_text(encoding="utf-8").splitlines()]
        with (stage / "plans.jsonl").open("w", encoding="utf-8") as f:
            for plan, event in zip(plans, audit["plans"], strict=True):
                if plan["tick"] != event["height_plan_tick"] or plan["joints"] != event["raw_joints_rad"]:
                    raise ValueError("Height constraint changed the finger command or planning clock")
                f.write(json.dumps({**plan, **event}, separators=(",", ":"), allow_nan=False) + "\n")
        trace = stage / "trace.npz"
        with np.load(trace, allow_pickle=False) as f:
            values = dict(f)
        if len(audit["measured"]) != seconds*100+1:
            raise ValueError("Incomplete wrist height trace")
        np.savez_compressed(trace, **values, height_measured_m=audit["measured"], height_reference_tick=audit["anchor_ticks"])
        copyfile(CALIBRATION, stage / CALIBRATION.name)
        receipt["signature"].update(extra)
        receipt["hashes"] = {f: digest(stage / f) for f in ("trace.npz", "plans.jsonl", CALIBRATION.name)}
        receipt["trace_sha256"] = receipt["hashes"]["trace.npz"]
        receipt["initial_file_sha256"] = spec["initial.npz"]
        receipt["robot_sha256"] = digest(fixtures / "hand.mjb")
        diffs = np.array([p["height_correction_m"] for p in audit["plans"]])
        measured = np.asarray(audit["measured"][1:])
        receipt["height_summary"] = dict(plans=len(plans), changed=int((diffs > 1e-10).sum()),
            max_correction_m=float(diffs.max()), measured_in_band_ratio=float(((measured >= band[0]) & (measured <= band[1])).mean()))
        active_a.write_json(stage / "run.json", receipt)
        stage.rename(out)
