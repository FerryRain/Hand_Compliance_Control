"""Recover AUDE's palm-axis constraint from qualified measurements and issued wrist goals."""

import io
from pathlib import Path

import numpy as np
from PARK.identity import digest, read

from . import report as base
from .verify import checked_bytes

CALIBRATION = Path(__file__).with_name("height_calibration.json")
RATIOS = ("measured_all_in_band_fraction", "measured_fresh_in_band_fraction")


def latest_qualified(valid):
    return np.maximum.accumulate(np.where(valid.any(axis=1), np.arange(len(valid)), -1))


def world_axis(pose, axis):
    """Rotate a palm-frame unit vector using measured or planned MuJoCo w,x,y,z."""
    q = np.asarray(pose)[..., 3:7]
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    np.testing.assert_allclose(norm, 1., atol=1e-6, rtol=0)
    q = q / norm
    v = q[..., 1:]
    return axis + 2 * np.cross(v, np.cross(v, axis) + q[..., :1] * axis)


def measured_height(trace, valid, axis, lower, upper, latest, *, audit_log):
    """Check the actual wrist height on every frame and distinguish fresh and retained contacts."""
    wrist = trace["qpos"][:, :7]
    points = trace["points_m"][latest]
    distances = np.einsum("tfc,tc->tf", points - wrist[:, None, :3], world_axis(wrist, axis))
    heights = np.nanmedian(np.where(valid[latest], distances, np.nan), axis=1)
    if audit_log:
        np.testing.assert_array_equal(trace["height_reference_tick"], latest)
        recorded = trace["height_measured_m"]
        np.testing.assert_allclose(heights, recorded, rtol=0, atol=1e-10)
        heights = recorded
    live = valid[1:].any(axis=1)
    heights = heights[1:]
    in_band = (heights >= lower) & (heights <= upper)
    return dict(measured_touch_frames=int(live.sum()), measured_height_min_m=float(heights.min()),
                measured_height_max_m=float(heights.max()), measured_height_median_m=float(np.median(heights)),
                measured_all_in_band_fraction=float(in_band.mean()),
                measured_fresh_in_band_fraction=float(in_band[live].mean()) if live.any() else None)


def add_baseline_height(score, folder):
    """Measure the same palm-axis statistic on an unconstrained recorded wrist trajectory."""
    calibration = read(CALIBRATION)
    with np.load(io.BytesIO(checked_bytes(folder / "trace.npz", score["trace_sha256"])),
                 allow_pickle=False) as trace:
        valid, _ = base.qualified(trace)
        if not valid[0].any():
            raise ValueError("The paired start has no contact anchor")
        measurements = measured_height(trace, valid, np.asarray(calibration["axis_palm"]),
                                       calibration["lower_m"], calibration["upper_m"], latest_qualified(valid),
                                       audit_log=False)
    return {**score, **measurements, "height_calibration_sha256": digest(CALIBRATION)}


def audit(run, trace, plans, plan_index, *, execute_planned_fingers=True):
    calibration = read(CALIBRATION)
    fingerprint = digest(CALIBRATION)
    sig = run["signature"]
    if not sig["height_calibration_sha256"] == run["hashes"]["height_calibration.json"] == fingerprint:
        raise ValueError("The height limit differs from the frozen training-record calibration")
    lower, upper, target = (calibration[k] for k in ("lower_m", "upper_m", "target_m"))
    axis = np.asarray(calibration["axis_palm"])
    np.testing.assert_array_equal(axis, [0, 0, -1])
    np.testing.assert_array_equal(sig["axis_palm"], axis)
    np.testing.assert_allclose(sig["height_bounds_m"], [lower, upper], rtol=0, atol=1e-13)
    if not lower < target < upper:
        raise ValueError("Invalid calibrated palm-axis distance bounds")
    ticks = np.asarray([p["tick"] for p in plans])
    valid, _ = base.qualified(trace)
    latest_frame = latest_qualified(valid)
    latest = latest_frame[ticks]
    if (latest < 0).any():
        raise ValueError("A height command has no observed physical contact")
    np.testing.assert_array_equal(latest, [p["height_anchor_tick"] for p in plans])
    np.testing.assert_array_equal(ticks, [p["height_plan_tick"] for p in plans])
    raw = np.asarray([p["raw_wrist"] for p in plans])
    issued = np.asarray([p["wrist"] for p in plans])
    joint_key = "joints" if execute_planned_fingers else "aude_joints_unexecuted_rad"
    joint_targets = np.asarray([p[joint_key] for p in plans])
    np.testing.assert_allclose(issued[:, 3:], raw[:, 3:], rtol=0, atol=1e-10)
    np.testing.assert_allclose(joint_targets, [p["raw_joints_rad"] for p in plans], rtol=0, atol=1e-10)
    if execute_planned_fingers:
        np.testing.assert_allclose(trace["goal_joints_rad"][1:], joint_targets[plan_index], rtol=0, atol=1e-10)
    directions = world_axis(raw, axis)
    contacts = trace["points_m"]
    raw_height = np.empty(len(ticks))
    issued_height = np.empty(len(ticks))
    surface_axis = np.empty(len(ticks))
    origin = trace["qpos"][0, :3]
    for i, (p, at) in enumerate(zip(plans, latest, strict=True)):
        anchor = contacts[at, valid[at]]
        np.testing.assert_allclose(anchor, p["height_anchor_points_m"], rtol=0, atol=1e-12)
        raw_height[i] = np.median((anchor - raw[i, :3]) @ directions[i])
        issued_height[i] = np.median((anchor - issued[i, :3]) @ directions[i])
        surface_axis[i] = np.median((anchor - origin) @ directions[i])
    radius = .45
    low, high = np.maximum(lower, surface_axis - radius), np.minimum(upper, surface_axis + radius)
    if np.any(low > high):
        raise ValueError("No common intersection of AUDE's wrist workspace and the training interval")
    bound = np.clip(raw_height, low, high)
    target_along_axis = surface_axis - bound
    offset = raw[:, :3] - origin
    tangent = offset - np.einsum("ij,ij->i", offset, directions)[:, None] * directions
    tangent_radius = np.sqrt(np.maximum(0, radius**2 - target_along_axis**2))
    scale = np.minimum(1., tangent_radius / np.maximum(1e-15, np.linalg.norm(tangent, axis=1)))
    expected = origin + target_along_axis[:, None] * directions + scale[:, None] * tangent
    corrected = np.linalg.norm(issued[:, :3] - raw[:, :3], axis=1)
    np.testing.assert_allclose(raw_height, [p["height_raw_m"] for p in plans], atol=1e-10, rtol=0)
    np.testing.assert_allclose(issued_height, [p["height_issued_m"] for p in plans], atol=1e-10, rtol=0)
    np.testing.assert_allclose(issued[:, :3], expected, atol=1e-10, rtol=0)
    np.testing.assert_allclose(corrected, [p["height_correction_m"] for p in plans], atol=1e-10, rtol=0)
    np.testing.assert_allclose(issued_height, bound, rtol=0, atol=1e-10)
    actual = measured_height(trace, valid, axis, lower, upper, latest_frame, audit_log=True)
    summary = run["height_summary"]
    count = int((corrected > 1e-10).sum())
    if summary["plans"] != len(plans) or summary["changed"] != count:
        raise ValueError("Mismatch in the number of projected AUDE wrist goals")
    np.testing.assert_allclose([summary["max_correction_m"], summary["measured_in_band_ratio"]],
                               [corrected.max(), actual["measured_all_in_band_fraction"]], rtol=0, atol=1e-10)
    result = dict(planning_cycles=len(plans), height_calibration_sha256=fingerprint,
                height_lower_m=float(lower), height_upper_m=float(upper), training_median_m=float(target),
                height_clipped_low=int(((corrected > 1e-10) & (raw_height < lower)).sum()),
                height_clipped_high=int(((corrected > 1e-10) & (raw_height > upper)).sum()),
                height_clipped_cycles=count, height_max_correction_m=float(corrected.max()),
                max_anchor_age_s=float((ticks-latest).max()/base.HZ), **actual)
    if execute_planned_fingers:
        result.update(executed_fingers="AUDE planned fingers with direct joint torque", planning_fingers_executed=True)
    return result
