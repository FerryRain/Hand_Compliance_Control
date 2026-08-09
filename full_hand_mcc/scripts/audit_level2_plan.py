#!/usr/bin/env python3
"""Audit a saved Baseline-2 Level-2 plan without starting a GPU environment.

The archive is always opened with ``allow_pickle=False``.  NumPy-only schema,
contact-schedule, bridge, and upper-cap checks run first.  The MuJoCo CPU
reachability model is imported lazily only for the pad-angle and joint-limit
audit; this module never creates a MJLab environment or simulation app.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence

import numpy as np


FINGER_NAMES = ("if_tip", "mf_tip", "rf_tip", "th_tip")
REQUIRED_FORWARD_FINGERS = 3
FORWARD_PROGRESS_RATIO = 0.10
DEFAULT_WINDOW_FRAMES = 20
DEFAULT_HZ = 100.0
MAX_PLAN_JOINT_STEP_RAD = 0.03
DIAGNOSTIC_PAD_LIMIT_DEG = 50.0
LEVEL2_PLANNER_PAD_LIMIT_DEG = 40.0
RUNTIME_PAD_LIMIT_DEG = 45.0
FK_POSITION_TOLERANCE_M = 1.0e-4
NUMERICAL_TOLERANCE = 1.0e-8
LEVEL2_RADIUS_M = 0.10
LEVEL2_HALF_HEIGHT_M = 0.17
LEVEL2_ROUTE_LENGTH_M = 0.48
LEVEL2_TERMINAL_PROGRESS_TOLERANCE_M = 0.004


class AuditInputError(ValueError):
    """Raised when an input archive cannot be audited safely."""


def load_npz_no_pickle(path: Path) -> dict[str, np.ndarray]:
    """Load every field eagerly and reject archives requiring pickle."""

    archive_path = Path(path)
    if not archive_path.is_file():
        raise AuditInputError(f"NPZ file does not exist: {archive_path}")
    values: dict[str, np.ndarray] = {}
    try:
        with np.load(archive_path, allow_pickle=False) as archive:
            for key in archive.files:
                try:
                    value = np.asarray(archive[key])
                except ValueError as exc:
                    raise AuditInputError(
                        f"Field {key!r} cannot be loaded with allow_pickle=False"
                    ) from exc
                if value.dtype.hasobject:
                    raise AuditInputError(
                        f"Field {key!r} has forbidden object dtype"
                    )
                values[key] = value.copy()
    except (OSError, ValueError) as exc:
        if isinstance(exc, AuditInputError):
            raise
        raise AuditInputError(f"Failed to read NPZ file {archive_path}: {exc}") from exc
    return values


def _scalar(values: Mapping[str, np.ndarray], key: str) -> Any:
    value = np.asarray(values[key])
    if value.shape != ():
        raise AuditInputError(f"Field {key!r} must be scalar, got {value.shape}")
    return value.item()


def _finite(array: np.ndarray) -> bool:
    if np.issubdtype(np.asarray(array).dtype, np.complexfloating):
        return False
    try:
        numeric = np.asarray(array, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(np.all(np.isfinite(numeric)))


def _real_numeric(array: np.ndarray) -> bool:
    dtype = np.asarray(array).dtype
    return bool(
        np.issubdtype(dtype, np.integer)
        or np.issubdtype(dtype, np.floating)
    )


def _shape_error(
    errors: list[str],
    values: Mapping[str, np.ndarray],
    key: str,
    expected: tuple[int | None, ...],
) -> None:
    if key not in values:
        return
    actual = np.asarray(values[key]).shape
    matches = len(actual) == len(expected) and all(
        wanted is None or got == wanted
        for got, wanted in zip(actual, expected, strict=True)
    )
    if not matches:
        errors.append(f"{key}: expected shape {expected}, got {actual}")


def reconstruct_frame_target_distance(
    frame_count: int,
    coarse_distance_m: np.ndarray,
) -> np.ndarray:
    """Rebuild the exact linspace used for adaptive-plan interpolation."""

    coarse = np.asarray(coarse_distance_m, dtype=np.float64).reshape(-1)
    return np.linspace(0.0, float(coarse[-1]), frame_count + 1)[1:]


def derive_frame_bridge_masks(
    frame_target_distance_m: np.ndarray,
    coarse_distance_m: np.ndarray,
    coarse_static_mask: np.ndarray,
    coarse_recovery_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map coarse right-endpoint bridge labels to interpolation frames."""

    target = np.asarray(frame_target_distance_m, dtype=np.float64).reshape(-1)
    coarse = np.asarray(coarse_distance_m, dtype=np.float64).reshape(-1)
    left = np.searchsorted(coarse, target, side="right") - 1
    left = np.clip(left, 0, coarse.size - 2)
    static = np.asarray(coarse_static_mask, dtype=np.bool_).reshape(-1)[left + 1]
    recovery = np.asarray(coarse_recovery_mask, dtype=np.bool_).reshape(-1)[
        left + 1
    ]
    return static, recovery


def capsule_meridian_arcs_local(
    points_local_m: np.ndarray,
    radius_m: float,
    half_height_m: float,
) -> np.ndarray:
    """Return bottom-to-top capsule meridian arcs for local surface points."""

    points = np.asarray(points_local_m, dtype=np.float64)
    radial = np.linalg.norm(points[..., :2], axis=-1)
    z = points[..., 2]
    arc = np.empty_like(radial)
    lower_cap = z < -half_height_m
    upper_cap = z > half_height_m
    cylinder = ~(lower_cap | upper_cap)
    lower_join = 0.5 * np.pi * radius_m
    upper_join = lower_join + 2.0 * half_height_m
    arc[lower_cap] = radius_m * np.arctan2(
        radial[lower_cap],
        np.maximum(-(z[lower_cap] + half_height_m), 0.0),
    )
    arc[cylinder] = lower_join + z[cylinder] + half_height_m
    arc[upper_cap] = upper_join + radius_m * np.arctan2(
        z[upper_cap] - half_height_m,
        radial[upper_cap],
    )
    return arc


def validate_schema(
    plan: Mapping[str, np.ndarray],
    grasp: Mapping[str, np.ndarray],
    *,
    window_frames: int,
) -> dict[str, Any]:
    """Validate the fields and cross-field invariants needed by this audit."""

    required_plan = {
        "surface_points_m",
        "kinematic_points_m",
        "joint_positions_rad",
        "progress_m",
        "progress_residual_m",
        "normal_error_m",
        "start_surface_local_m",
        "end_surface_local_m",
        "scheduled_contact_mask",
        "scheduled_contact_count",
        "recovery_bridge_mask",
        "min_planner_contact_fingers",
        "mpc_coarse_distance_m",
        "mpc_coarse_static_feasibility_bridge",
        "mpc_coarse_recovery_bridge",
        "coarse_joint_positions_rad",
        "coarse_progress_m",
        "mpc_recovery_bridge_min_contact_fingers",
        "final_contact_recovery_frames",
        "transient_contact_finger",
        "transient_contact_start_m",
        "transient_contact_end_m",
        "object_shape",
        "object_radius_m",
        "object_half_height_m",
        "max_joint_step_rad",
        "axial_distance_m",
        "axial_direction",
    }
    required_grasp = {
        "joint_position_rad",
        "object_center_m",
        "object_rotation",
    }
    errors: list[str] = []
    missing_plan = sorted(required_plan - set(plan))
    missing_grasp = sorted(required_grasp - set(grasp))
    if missing_plan:
        errors.append(f"plan missing fields: {missing_plan}")
    if missing_grasp:
        errors.append(f"initial grasp missing fields: {missing_grasp}")
    if errors:
        return {"passed": False, "errors": errors}

    surface = np.asarray(plan["surface_points_m"])
    frame_count = int(surface.shape[0]) if surface.ndim else 0
    _shape_error(errors, plan, "surface_points_m", (None, 5, 3))
    _shape_error(errors, plan, "kinematic_points_m", (frame_count, 5, 3))
    _shape_error(errors, plan, "joint_positions_rad", (frame_count, 23))
    _shape_error(errors, plan, "progress_m", (frame_count, 5))
    _shape_error(errors, plan, "progress_residual_m", (frame_count, 5))
    _shape_error(errors, plan, "normal_error_m", (frame_count, 5))
    _shape_error(errors, plan, "scheduled_contact_mask", (frame_count, 4))
    _shape_error(errors, plan, "scheduled_contact_count", (frame_count,))
    _shape_error(errors, plan, "recovery_bridge_mask", (frame_count,))
    _shape_error(errors, plan, "axial_distance_m", (frame_count,))
    _shape_error(errors, plan, "start_surface_local_m", (5, 3))
    _shape_error(errors, plan, "end_surface_local_m", (5, 3))
    _shape_error(errors, grasp, "joint_position_rad", (23,))
    _shape_error(errors, grasp, "object_center_m", (3,))
    _shape_error(errors, grasp, "object_rotation", (3, 3))

    if frame_count <= window_frames:
        errors.append(
            f"plan has {frame_count} frames; requires more than "
            f"window_frames={window_frames}"
        )

    numeric_plan_fields = (
        "surface_points_m",
        "kinematic_points_m",
        "joint_positions_rad",
        "progress_m",
        "progress_residual_m",
        "normal_error_m",
        "start_surface_local_m",
        "end_surface_local_m",
        "scheduled_contact_count",
        "mpc_coarse_distance_m",
        "coarse_joint_positions_rad",
        "coarse_progress_m",
        "object_radius_m",
        "object_half_height_m",
        "max_joint_step_rad",
        "axial_distance_m",
        "axial_direction",
        "transient_contact_finger",
        "transient_contact_start_m",
        "transient_contact_end_m",
        "min_planner_contact_fingers",
        "mpc_recovery_bridge_min_contact_fingers",
        "final_contact_recovery_frames",
    )
    for key in numeric_plan_fields:
        if key in plan and not _real_numeric(plan[key]):
            errors.append(f"{key} must have a numeric dtype")
    for key in (
        "scheduled_contact_mask",
        "recovery_bridge_mask",
        "mpc_coarse_static_feasibility_bridge",
        "mpc_coarse_recovery_bridge",
    ):
        if key in plan and np.asarray(plan[key]).dtype != np.bool_:
            errors.append(f"{key} must have boolean dtype")
    if "scheduled_contact_count" in plan and not np.issubdtype(
        np.asarray(plan["scheduled_contact_count"]).dtype, np.integer
    ):
        errors.append("scheduled_contact_count must have an integer dtype")

    coarse = np.asarray(plan["mpc_coarse_distance_m"])
    coarse_static = np.asarray(
        plan["mpc_coarse_static_feasibility_bridge"]
    )
    coarse_recovery = np.asarray(plan["mpc_coarse_recovery_bridge"])
    coarse_is_numeric = _real_numeric(coarse)
    if coarse.ndim != 1 or coarse.size < 2:
        errors.append("mpc_coarse_distance_m must be a 1-D array of length >= 2")
    elif coarse_is_numeric:
        if not _finite(coarse) or np.any(np.diff(coarse) <= 0.0):
            errors.append("mpc_coarse_distance_m must be finite and increasing")
        if abs(float(coarse[0])) > NUMERICAL_TOLERANCE:
            errors.append("mpc_coarse_distance_m must start at zero")
    if coarse_static.shape != coarse.shape:
        errors.append("coarse static bridge mask shape differs from coarse distance")
    if coarse_recovery.shape != coarse.shape:
        errors.append("coarse recovery bridge mask shape differs from coarse distance")
    _shape_error(
        errors,
        plan,
        "coarse_joint_positions_rad",
        (int(coarse.size), 23),
    )
    _shape_error(
        errors,
        plan,
        "coarse_progress_m",
        (int(coarse.size), 5),
    )
    if (
        coarse_static.shape == coarse_recovery.shape
        and np.any(np.asarray(coarse_static, dtype=bool) & ~np.asarray(coarse_recovery, dtype=bool))
    ):
        errors.append("every coarse static bridge must also be marked recovery")

    finite_plan_fields = numeric_plan_fields
    for key in finite_plan_fields:
        if key in plan and not _finite(plan[key]):
            errors.append(f"{key} contains NaN or Inf")
    for key in ("progress_residual_m", "normal_error_m"):
        if (
            key in plan
            and _real_numeric(plan[key])
            and np.any(np.asarray(plan[key]) < -NUMERICAL_TOLERANCE)
        ):
            errors.append(f"{key} must be non-negative")
    for key in ("joint_position_rad", "object_center_m", "object_rotation"):
        if key in grasp:
            if not _real_numeric(grasp[key]):
                errors.append(f"initial grasp {key} must have a numeric dtype")
            elif not _finite(grasp[key]):
                errors.append(f"initial grasp {key} contains NaN or Inf")

    try:
        shape = str(_scalar(plan, "object_shape"))
        if shape != "capsule":
            errors.append(f"Level-2 audit requires object_shape='capsule', got {shape!r}")
        radius = float(_scalar(plan, "object_radius_m"))
        half_height = float(_scalar(plan, "object_half_height_m"))
        if radius <= 0.0 or half_height < 0.0:
            errors.append("capsule radius must be positive and half-height non-negative")
        raw_transient_finger = _scalar(plan, "transient_contact_finger")
        transient_finger = int(raw_transient_finger)
        if float(raw_transient_finger) != transient_finger:
            errors.append("transient_contact_finger must be an integer")
        if transient_finger not in range(4):
            errors.append("transient_contact_finger must be in [0, 3]")
        raw_minimum_contacts = _scalar(plan, "min_planner_contact_fingers")
        minimum_contacts = int(raw_minimum_contacts)
        if float(raw_minimum_contacts) != minimum_contacts:
            errors.append("min_planner_contact_fingers must be an integer")
        if minimum_contacts not in range(1, 5):
            errors.append("min_planner_contact_fingers must be in [1, 4]")
        raw_recovery_minimum = _scalar(
            plan, "mpc_recovery_bridge_min_contact_fingers"
        )
        recovery_minimum_contacts = int(raw_recovery_minimum)
        if float(raw_recovery_minimum) != recovery_minimum_contacts:
            errors.append(
                "mpc_recovery_bridge_min_contact_fingers must be an integer"
            )
        if recovery_minimum_contacts not in range(1, 5):
            errors.append(
                "mpc_recovery_bridge_min_contact_fingers must be in [1, 4]"
            )
        raw_axial_direction = _scalar(plan, "axial_direction")
        axial_direction = int(raw_axial_direction)
        if float(raw_axial_direction) != axial_direction:
            errors.append("axial_direction must be an integer")
        if axial_direction not in (-1, 1):
            errors.append("axial_direction must be -1 or +1")
        saved_maximum_joint_step = float(
            _scalar(plan, "max_joint_step_rad")
        )
        if saved_maximum_joint_step < 0.0:
            errors.append("max_joint_step_rad must be non-negative")
        raw_final_contact_frames = _scalar(
            plan, "final_contact_recovery_frames"
        )
        final_contact_frames = int(raw_final_contact_frames)
        if float(raw_final_contact_frames) != final_contact_frames:
            errors.append("final_contact_recovery_frames must be an integer")
        if final_contact_frames <= 0 or final_contact_frames > frame_count:
            errors.append(
                "final_contact_recovery_frames must be in [1, frame_count]"
            )
        transient_start = float(_scalar(plan, "transient_contact_start_m"))
        transient_end = float(_scalar(plan, "transient_contact_end_m"))
        if transient_end < transient_start:
            errors.append("transient contact end precedes its start")
    except (AuditInputError, TypeError, ValueError) as exc:
        errors.append(str(exc))

    try:
        rotation = np.asarray(grasp["object_rotation"], dtype=np.float64)
    except (TypeError, ValueError):
        errors.append("initial grasp object_rotation must have a numeric dtype")
    else:
        if rotation.shape == (3, 3) and _finite(rotation):
            if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-6):
                errors.append("initial grasp object_rotation is not orthonormal")
            if not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-6):
                errors.append("initial grasp object_rotation is not a proper rotation")

    if not errors:
        scheduled_mask = np.asarray(plan["scheduled_contact_mask"], dtype=bool)
        scheduled_count = np.asarray(plan["scheduled_contact_count"], dtype=int)
        expected_count = np.count_nonzero(scheduled_mask, axis=1)
        if not np.array_equal(scheduled_count, expected_count):
            errors.append("scheduled_contact_count does not match scheduled_contact_mask")
        target_distance = reconstruct_frame_target_distance(frame_count, coarse)
        static, recovery = derive_frame_bridge_masks(
            target_distance,
            coarse,
            coarse_static,
            coarse_recovery,
        )
        saved_recovery = np.asarray(plan["recovery_bridge_mask"], dtype=bool)
        if not np.array_equal(saved_recovery, recovery):
            mismatch = int(np.count_nonzero(saved_recovery != recovery))
            errors.append(
                "recovery_bridge_mask disagrees with coarse right-endpoint "
                f"labels on {mismatch} frames"
            )
        if np.any(static & ~recovery):
            errors.append("derived static bridge frames are not recovery frames")

    return {
        "passed": not errors,
        "errors": errors,
        "frame_count": frame_count,
        "joint_count": 23,
    }


def _contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    active = np.asarray(mask, dtype=bool).reshape(-1)
    padded = np.concatenate(([False], active, [False])).astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def find_unmarked_low_motion_windows(
    progress_m: np.ndarray,
    kinematic_points_m: np.ndarray,
    frame_target_distance_m: np.ndarray,
    marked_bridge_mask: np.ndarray,
    axial_distance_m: np.ndarray,
    *,
    window_frames: int = DEFAULT_WINDOW_FRAMES,
    forward_progress_ratio: float = FORWARD_PROGRESS_RATIO,
) -> list[dict[str, Any]]:
    """Find overlapping windows with fewer than three progressing tips."""

    progress = np.asarray(progress_m, dtype=np.float64)
    points = np.asarray(kinematic_points_m, dtype=np.float64)
    target = np.asarray(frame_target_distance_m, dtype=np.float64)
    marked = np.asarray(marked_bridge_mask, dtype=bool)
    axial = np.asarray(axial_distance_m, dtype=np.float64)
    raw: list[dict[str, Any]] = []
    for start in range(0, progress.shape[0] - window_frames):
        end = start + window_frames
        if np.any(marked[start : end + 1]):
            continue
        route_delta = float(target[end] - target[start])
        if route_delta <= NUMERICAL_TOLERANCE:
            continue
        required = forward_progress_ratio * route_delta
        tip_delta = progress[end, 1:] - progress[start, 1:]
        forward = tip_delta >= required - 1.0e-12
        if int(np.count_nonzero(forward)) >= REQUIRED_FORWARD_FINGERS:
            continue
        cartesian = np.linalg.norm(
            points[end, 1:] - points[start, 1:], axis=1
        )
        raw.append(
            {
                "start": start,
                "end": end,
                "forward_finger_count": int(np.count_nonzero(forward)),
                "forward_finger_required": REQUIRED_FORWARD_FINGERS,
                "forward_mask": forward.tolist(),
                "route_delta_m": route_delta,
                "required_tip_progress_m": required,
                "tip_progress_delta_m": tip_delta.tolist(),
                "tip_cartesian_delta_m": cartesian.tolist(),
                "target_distance_start_m": float(target[start]),
                "target_distance_end_m": float(target[end]),
                "axial_distance_start_m": float(axial[start]),
                "axial_distance_end_m": float(axial[end]),
            }
        )
    if not raw:
        return []

    groups: list[list[dict[str, Any]]] = [[raw[0]]]
    for item in raw[1:]:
        if int(item["start"]) <= int(groups[-1][-1]["end"]) + 1:
            groups[-1].append(item)
        else:
            groups.append([item])

    summaries: list[dict[str, Any]] = []
    for group in groups:
        worst = min(
            group,
            key=lambda item: (
                int(item["forward_finger_count"]),
                min(item["tip_progress_delta_m"]),
            ),
        )
        summaries.append(
            {
                "frame_start": int(group[0]["start"]),
                "frame_end": int(group[-1]["end"]),
                "overlapping_window_count": len(group),
                "worst_window": worst,
            }
        )
    return summaries


def audit_nominal_three_of_four_runs(
    scheduled_contact_mask: np.ndarray,
    scheduled_contact_count: np.ndarray,
    recovery_mask: np.ndarray,
    frame_target_distance_m: np.ndarray,
    *,
    transient_finger: int,
    transient_start_m: float,
    transient_end_m: float,
    transient_enabled: bool = True,
    minimum_run_frames: int = DEFAULT_WINDOW_FRAMES,
    hz: float = DEFAULT_HZ,
) -> dict[str, Any]:
    """Report long nominal 3/4 runs and distinguish configured relaxations."""

    mask = np.asarray(scheduled_contact_mask, dtype=bool)
    count = np.asarray(scheduled_contact_count, dtype=int)
    recovery = np.asarray(recovery_mask, dtype=bool)
    target = np.asarray(frame_target_distance_m, dtype=np.float64)
    transient_region = (
        (target > transient_start_m) & (target < transient_end_m)
    )
    by_finger: dict[str, list[dict[str, Any]]] = {}
    unexpected: list[dict[str, Any]] = []
    for finger, name in enumerate(FINGER_NAMES):
        dropped = (count == 3) & ~mask[:, finger]
        runs: list[dict[str, Any]] = []
        for start, end in _contiguous_runs(dropped):
            frames = end - start + 1
            if frames < minimum_run_frames:
                continue
            configured = (
                transient_region
                if transient_enabled and finger == transient_finger
                else np.zeros_like(recovery)
            )
            allowed = recovery | configured
            fully_allowed = bool(np.all(allowed[start : end + 1]))
            if bool(np.all(recovery[start : end + 1])):
                classification = "recovery"
            elif bool(np.all(configured[start : end + 1])):
                classification = "configured_transient"
            elif fully_allowed:
                classification = "mixed_allowed"
            else:
                classification = "unexpected"
            record = {
                "finger": name,
                "finger_index": finger,
                "frame_start": start,
                "frame_end": end,
                "frames": frames,
                "duration_s": frames / hz,
                "target_distance_start_m": float(target[start]),
                "target_distance_end_m": float(target[end]),
                "recovery_frame_count": int(
                    np.count_nonzero(recovery[start : end + 1])
                ),
                "classification": classification,
            }
            runs.append(record)
            if not fully_allowed:
                unexpected.append(record)
        by_finger[name] = runs
    return {
        "minimum_reported_run_frames": minimum_run_frames,
        "per_finger": by_finger,
        "unexpected_long_runs": unexpected,
        "unexpected_long_run_count": len(unexpected),
    }


def audit_nominal_support_policy(
    scheduled_contact_mask: np.ndarray,
    scheduled_contact_count: np.ndarray,
    recovery_mask: np.ndarray,
    frame_target_distance_m: np.ndarray,
    *,
    transient_finger: int,
    transient_start_m: float,
    transient_end_m: float,
    minimum_planner_contacts: int,
    minimum_recovery_contacts: int,
) -> dict[str, Any]:
    """Require 4/4 ordinarily and bounded support in explicit exceptions."""

    mask = np.asarray(scheduled_contact_mask, dtype=bool)
    count = np.asarray(scheduled_contact_count, dtype=int)
    recovery = np.asarray(recovery_mask, dtype=bool)
    target = np.asarray(frame_target_distance_m, dtype=np.float64)
    transient = (
        minimum_planner_contacts < 4
        and transient_end_m > transient_start_m
        and (target > transient_start_m)
        & (target < transient_end_m)
    )
    valid = np.ones(target.size, dtype=bool)
    ordinary = ~recovery & ~transient
    valid[ordinary] = count[ordinary] == 4

    transient_only = transient & ~recovery
    missing_non_transient = ~mask.copy()
    missing_non_transient[:, transient_finger] = False
    valid[transient_only] = (
        (count[transient_only] >= minimum_planner_contacts)
        & ~np.any(missing_non_transient[transient_only], axis=1)
    )
    valid[recovery] = count[recovery] >= minimum_recovery_contacts

    violations: list[dict[str, Any]] = []
    for start, end in _contiguous_runs(~valid):
        if np.all(recovery[start : end + 1]):
            region = "recovery"
        elif np.all(transient_only[start : end + 1]):
            region = "configured_transient"
        elif np.all(ordinary[start : end + 1]):
            region = "ordinary"
        else:
            region = "mixed"
        missing = np.any(~mask[start : end + 1], axis=0)
        violations.append(
            {
                "frame_start": start,
                "frame_end": end,
                "frames": end - start + 1,
                "target_distance_start_m": float(target[start]),
                "target_distance_end_m": float(target[end]),
                "minimum_scheduled_contacts": int(
                    np.min(count[start : end + 1])
                ),
                "missing_fingers": [
                    name
                    for name, is_missing in zip(
                        FINGER_NAMES, missing.tolist(), strict=True
                    )
                    if is_missing
                ],
                "region": region,
            }
        )
    return {
        "passed": not violations,
        "violation_frame_count": int(np.count_nonzero(~valid)),
        "violation_regions": violations,
        "minimum_planner_contacts": minimum_planner_contacts,
        "minimum_recovery_contacts": minimum_recovery_contacts,
    }


def audit_terminal_nominal_support(
    scheduled_contact_count: np.ndarray,
    *,
    configured_frames: int,
    mode: str,
    hz: float,
) -> dict[str, Any]:
    """Check saved terminal 4/4 policy and Acceptance's 0.50 s window."""

    count = np.asarray(scheduled_contact_count, dtype=int).reshape(-1)
    required_frames = (
        int(np.ceil(0.20 * hz))
        if mode == "Diagnostic"
        else int(np.ceil(0.50 * hz))
    )
    enough_frames = count.size >= required_frames
    terminal_count = count[-required_frames:] if enough_frames else count
    checks = {
        "saved_configuration_covers_mode_requirement": (
            configured_frames >= required_frames
        ),
        "plan_has_required_terminal_frames": enough_frames,
        "terminal_nominal_support_is_four_of_four": bool(
            enough_frames and np.all(terminal_count == 4)
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "mode": mode,
        "configured_frames": configured_frames,
        "required_frames": required_frames,
        "required_duration_s": required_frames / hz,
        "minimum_terminal_scheduled_contacts": (
            int(np.min(terminal_count)) if terminal_count.size else None
        ),
    }


def _create_cpu_kinematics() -> tuple[Any, tuple[str, ...], Callable[..., Any]]:
    from mjlab.tasks.leaphand.full_hand_mcc_geometry import capsule_project
    from mjlab.tasks.leaphand.leaphand_full_hand_mcc_env_cfg import (
        ARM_JOINT_NAMES,
        HAND_QPOS_NAMES,
        FivePointReachabilitySolver,
    )

    return (
        FivePointReachabilitySolver(),
        tuple(ARM_JOINT_NAMES + HAND_QPOS_NAMES),
        capsule_project,
    )


def _load_cpu_kinematics() -> tuple[Any, tuple[str, ...], Callable[..., Any]]:
    """Create the CPU backend while keeping stdout valid JSON for the CLI."""

    import_output = io.StringIO()
    with redirect_stdout(import_output):
        backend = _create_cpu_kinematics()
    noise = import_output.getvalue()
    if noise:
        print(noise, end="", file=sys.stderr)
    return backend


def audit_kinematics(
    plan: Mapping[str, np.ndarray],
    grasp: Mapping[str, np.ndarray],
    *,
    mode: str,
    solver: Any | None = None,
    joint_names: Sequence[str] | None = None,
    projection_fn: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Recompute all-frame FK, pad angle, and signed joint-limit margin."""

    if solver is None or joint_names is None or projection_fn is None:
        solver, loaded_names, projection_fn = _load_cpu_kinematics()
        joint_names = loaded_names
    q_plan = np.asarray(plan["joint_positions_rad"], dtype=np.float64)
    saved_points = np.asarray(plan["kinematic_points_m"], dtype=np.float64)
    saved_surface = np.asarray(plan["surface_points_m"], dtype=np.float64)
    axial_distance = np.asarray(plan["axial_distance_m"], dtype=np.float64)
    target_distance = reconstruct_frame_target_distance(
        q_plan.shape[0], plan["mpc_coarse_distance_m"]
    )
    center = np.asarray(grasp["object_center_m"], dtype=np.float64).reshape(3)
    rotation = np.asarray(grasp["object_rotation"], dtype=np.float64).reshape(3, 3)
    radius = float(_scalar(plan, "object_radius_m"))
    half_height = float(_scalar(plan, "object_half_height_m"))

    def evaluate_state(
        q: np.ndarray,
        *,
        label: str,
        index: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        points = np.asarray(solver.forward_points(q), dtype=np.float64)
        if points.shape != (5, 3) or not np.all(np.isfinite(points)):
            raise AuditInputError(
                "Reachability forward_points returned invalid data at "
                f"{label} {index}: shape={points.shape}"
            )
        projected_surface, outward = projection_fn(
            points,
            center,
            rotation,
            radius,
            half_height,
        )
        projected_surface = np.asarray(projected_surface, dtype=np.float64)
        if projected_surface.shape != (5, 3) or not np.all(
            np.isfinite(projected_surface)
        ):
            raise AuditInputError(
                "Capsule projection returned invalid surface points at "
                f"{label} {index}: shape={projected_surface.shape}"
            )
        outward = np.asarray(outward, dtype=np.float64)
        if outward.shape != (5, 3) or not np.all(np.isfinite(outward)):
            raise AuditInputError(
                "Capsule projection returned invalid normals at "
                f"{label} {index}: shape={outward.shape}"
            )
        if not np.allclose(
            np.linalg.norm(outward, axis=1), 1.0, atol=1.0e-5
        ):
            raise AuditInputError(
                "Capsule projection returned non-unit normals at "
                f"{label} {index}"
            )
        pad_normals = np.asarray(
            solver.fingertip_pad_normals(q), dtype=np.float64
        )
        if pad_normals.shape != (4, 3) or not np.all(np.isfinite(pad_normals)):
            raise AuditInputError(
                "Reachability fingertip_pad_normals returned invalid data at "
                f"{label} {index}: shape={pad_normals.shape}"
            )
        if not np.allclose(
            np.linalg.norm(pad_normals, axis=1), 1.0, atol=1.0e-5
        ):
            raise AuditInputError(
                "Reachability fingertip_pad_normals returned non-unit "
                f"vectors at {label} {index}"
            )
        alignment = np.einsum("ij,ij->i", pad_normals, -outward[1:])
        if not np.all(np.isfinite(alignment)):
            raise AuditInputError(
                f"Pad alignment is non-finite at {label} {index}"
            )
        angles = np.degrees(np.arccos(np.clip(alignment, -1.0, 1.0)))
        return points, projected_surface, angles

    max_frame_angle = -np.inf
    max_angle_frame = -1
    max_frame_angle_finger = -1
    max_fk_error = -np.inf
    max_fk_error_frame = -1
    max_surface_error = -np.inf
    max_surface_error_frame = -1
    for frame, q in enumerate(q_plan):
        points, projected_surface, angles = evaluate_state(
            q, label="frame", index=frame
        )
        fk_error = float(
            np.max(np.linalg.norm(points - saved_points[frame], axis=1))
        )
        if fk_error > max_fk_error:
            max_fk_error = fk_error
            max_fk_error_frame = frame
        surface_error = float(
            np.max(
                np.linalg.norm(
                    projected_surface - saved_surface[frame], axis=1
                )
            )
        )
        if surface_error > max_surface_error:
            max_surface_error = surface_error
            max_surface_error_frame = frame
        finger = int(np.argmax(angles))
        if float(angles[finger]) > max_frame_angle:
            max_frame_angle = float(angles[finger])
            max_angle_frame = frame
            max_frame_angle_finger = finger

    coarse_q = np.asarray(
        plan["coarse_joint_positions_rad"], dtype=np.float64
    )
    coarse_surface = np.empty((coarse_q.shape[0], 5, 3), dtype=np.float64)
    max_coarse_angle = -np.inf
    max_angle_keyframe = -1
    max_coarse_angle_finger = -1
    for keyframe, q in enumerate(coarse_q):
        _, projected_surface, angles = evaluate_state(
            q, label="keyframe", index=keyframe
        )
        coarse_surface[keyframe] = projected_surface
        finger = int(np.argmax(angles))
        if float(angles[finger]) > max_coarse_angle:
            max_coarse_angle = float(angles[finger])
            max_angle_keyframe = keyframe
            max_coarse_angle_finger = finger

    if max_coarse_angle > max_frame_angle:
        max_angle = max_coarse_angle
        max_angle_source = "keyframe"
        max_angle_index = max_angle_keyframe
        max_angle_finger = max_coarse_angle_finger
        max_angle_axial_distance = float(
            np.min(
                np.asarray(plan["coarse_progress_m"], dtype=np.float64)[
                    max_angle_keyframe, 1:
                ]
            )
        )
        max_angle_target_distance = float(
            np.asarray(plan["mpc_coarse_distance_m"], dtype=np.float64)[
                max_angle_keyframe
            ]
        )
    else:
        max_angle = max_frame_angle
        max_angle_source = "interpolated_frame"
        max_angle_index = max_angle_frame
        max_angle_finger = max_frame_angle_finger
        max_angle_axial_distance = float(axial_distance[max_angle_frame])
        max_angle_target_distance = float(target_distance[max_angle_frame])

    coarse_local_surface = (coarse_surface - center) @ rotation
    coarse_arc = capsule_meridian_arcs_local(
        coarse_local_surface[:, 1:], radius, half_height
    )
    coarse_progress = np.asarray(
        plan["coarse_progress_m"], dtype=np.float64
    )[:, 1:]
    axial_direction = int(_scalar(plan, "axial_direction"))
    coarse_reference_arc = (
        coarse_arc[0] - axial_direction * coarse_progress[0]
    )
    recomputed_coarse_progress = axial_direction * (
        coarse_arc - coarse_reference_arc
    )
    coarse_progress_geometry_error = float(
        np.max(np.abs(recomputed_coarse_progress - coarse_progress))
    )
    recomputed_coarse_backtracking_step = float(
        max(
            0.0,
            -np.min(np.diff(recomputed_coarse_progress, axis=0)),
        )
    )

    lower = np.asarray(solver.lower, dtype=np.float64).reshape(23)
    upper = np.asarray(solver.upper, dtype=np.float64).reshape(23)
    if np.any(np.isnan(lower)) or np.any(np.isnan(upper)):
        raise AuditInputError("Reachability joint limits contain NaN")
    finite_pairs = np.isfinite(lower) & np.isfinite(upper)
    if np.any(lower[finite_pairs] >= upper[finite_pairs]):
        raise AuditInputError("Reachability joint limits are not ordered")
    frame_margins = np.minimum(q_plan - lower, upper - q_plan)
    coarse_margins = np.minimum(coarse_q - lower, upper - coarse_q)
    frame_minimum_flat = int(np.argmin(frame_margins))
    margin_frame, frame_margin_joint = np.unravel_index(
        frame_minimum_flat, frame_margins.shape
    )
    coarse_minimum_flat = int(np.argmin(coarse_margins))
    margin_keyframe, coarse_margin_joint = np.unravel_index(
        coarse_minimum_flat, coarse_margins.shape
    )
    frame_minimum_margin = float(
        frame_margins[margin_frame, frame_margin_joint]
    )
    coarse_minimum_margin = float(
        coarse_margins[margin_keyframe, coarse_margin_joint]
    )
    if coarse_minimum_margin < frame_minimum_margin:
        minimum_margin = coarse_minimum_margin
        margin_source = "keyframe"
        margin_index = int(margin_keyframe)
        margin_joint = int(coarse_margin_joint)
        margin_q = coarse_q[margin_keyframe]
        margin_axial_distance = float(
            np.min(coarse_progress[margin_keyframe])
        )
        margin_target_distance = float(
            np.asarray(plan["mpc_coarse_distance_m"], dtype=np.float64)[
                margin_keyframe
            ]
        )
    else:
        minimum_margin = frame_minimum_margin
        margin_source = "interpolated_frame"
        margin_index = int(margin_frame)
        margin_joint = int(frame_margin_joint)
        margin_q = q_plan[margin_frame]
        margin_axial_distance = float(axial_distance[margin_frame])
        margin_target_distance = float(target_distance[margin_frame])
    if not np.isfinite(minimum_margin):
        raise AuditInputError(
            "Reachability model did not provide a finite joint-limit margin"
        )
    names = tuple(joint_names)
    joint_name = (
        names[margin_joint]
        if margin_joint < len(names)
        else f"joint_{margin_joint}"
    )

    planner_seed_q = np.asarray(
        plan["coarse_joint_positions_rad"], dtype=np.float64
    )[0].reshape(23)
    q_with_seed = np.vstack((planner_seed_q[None], q_plan))
    step_matrix = np.abs(np.diff(q_with_seed, axis=0))
    max_step_flat = int(np.argmax(step_matrix))
    step_frame, step_joint = np.unravel_index(max_step_flat, step_matrix.shape)
    maximum_joint_step = float(step_matrix[step_frame, step_joint])
    saved_maximum_joint_step = float(_scalar(plan, "max_joint_step_rad"))
    pad_limit = (
        DIAGNOSTIC_PAD_LIMIT_DEG
        if mode == "Diagnostic"
        else LEVEL2_PLANNER_PAD_LIMIT_DEG
    )
    checks = {
        "pad_angle_within_mode_limit": max_angle <= pad_limit + 1.0e-9,
        "joint_limits_satisfied": minimum_margin >= -NUMERICAL_TOLERANCE,
        "maximum_joint_step_satisfied": (
            maximum_joint_step <= MAX_PLAN_JOINT_STEP_RAD + NUMERICAL_TOLERANCE
        ),
        "saved_joint_step_consistent": abs(
            maximum_joint_step - saved_maximum_joint_step
        ) <= 1.0e-6,
        "forward_kinematics_consistent": (
            max_fk_error <= FK_POSITION_TOLERANCE_M
        ),
        "projected_surface_consistent": (
            max_surface_error <= FK_POSITION_TOLERANCE_M
        ),
        "coarse_progress_geometry_consistent": (
            coarse_progress_geometry_error <= FK_POSITION_TOLERANCE_M
        ),
        "recomputed_keyframe_backtracking_within_0_2mm": (
            recomputed_coarse_backtracking_step
            <= 0.0002 + NUMERICAL_TOLERANCE
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "pad_angle": {
            "maximum_deg": max_angle,
            "source": max_angle_source,
            "index": max_angle_index,
            "finger": FINGER_NAMES[max_angle_finger],
            "axial_distance_m": max_angle_axial_distance,
            "target_distance_m": max_angle_target_distance,
            "maximum_interpolated_frame_deg": max_frame_angle,
            "maximum_interpolated_frame": max_angle_frame,
            "maximum_keyframe_deg": max_coarse_angle,
            "maximum_keyframe": max_angle_keyframe,
            "mode_limit_deg": pad_limit,
            "level2_planner_limit_deg": LEVEL2_PLANNER_PAD_LIMIT_DEG,
            "within_level2_planner_limit": (
                max_angle <= LEVEL2_PLANNER_PAD_LIMIT_DEG + 1.0e-9
            ),
            "runtime_reference_limit_deg": RUNTIME_PAD_LIMIT_DEG,
        },
        "joint_limit": {
            "minimum_margin_rad": minimum_margin,
            "minimum_margin_deg": float(np.degrees(minimum_margin)),
            "source": margin_source,
            "index": margin_index,
            "joint_index": int(margin_joint),
            "joint_name": joint_name,
            "axial_distance_m": margin_axial_distance,
            "target_distance_m": margin_target_distance,
            "position_rad": float(margin_q[margin_joint]),
            "lower_rad": float(lower[margin_joint]),
            "upper_rad": float(upper[margin_joint]),
            "minimum_interpolated_frame_margin_rad": frame_minimum_margin,
            "minimum_keyframe_margin_rad": coarse_minimum_margin,
        },
        "joint_step": {
            "maximum_rad": maximum_joint_step,
            "saved_maximum_rad": saved_maximum_joint_step,
            "limit_rad": MAX_PLAN_JOINT_STEP_RAD,
            "transition_index": int(step_frame),
            "joint_index": int(step_joint),
            "seed_source": "coarse_joint_positions_rad[0]",
        },
        "forward_kinematics": {
            "maximum_saved_point_error_m": max_fk_error,
            "tolerance_m": FK_POSITION_TOLERANCE_M,
            "frame": max_fk_error_frame,
        },
        "projected_surface": {
            "maximum_saved_surface_error_m": max_surface_error,
            "tolerance_m": FK_POSITION_TOLERANCE_M,
            "frame": max_surface_error_frame,
        },
        "coarse_progress_geometry": {
            "maximum_error_m": coarse_progress_geometry_error,
            "tolerance_m": FK_POSITION_TOLERANCE_M,
            "maximum_recomputed_backtracking_step_m": (
                recomputed_coarse_backtracking_step
            ),
            "backtracking_limit_m": 0.0002,
        },
    }


def audit_level2_geometry(
    plan: Mapping[str, np.ndarray],
    grasp: Mapping[str, np.ndarray],
    frame_target_distance_m: np.ndarray,
    recovery_mask: np.ndarray,
) -> dict[str, Any]:
    """Audit the canonical capsule seam, upper-cap, top, and terminal tail."""

    surface = np.asarray(plan["surface_points_m"], dtype=np.float64)
    center = np.asarray(grasp["object_center_m"], dtype=np.float64).reshape(3)
    rotation = np.asarray(grasp["object_rotation"], dtype=np.float64).reshape(3, 3)
    local = (surface - center) @ rotation
    tip_z = local[:, 1:, 2]
    radius = float(_scalar(plan, "object_radius_m"))
    half_height = float(_scalar(plan, "object_half_height_m"))
    seam_tolerance = 1.0e-6

    crossed: list[bool] = []
    crossing_frames: list[int | None] = []
    for finger in range(4):
        indices = np.flatnonzero(
            tip_z[:, finger] >= half_height - seam_tolerance
        )
        first_crossing = int(indices[0]) if indices.size else None
        started_below = bool(
            tip_z[0, finger] < half_height - seam_tolerance
        )
        lower_before_crossing = bool(
            first_crossing is not None
            and np.any(
                tip_z[: first_crossing + 1, finger]
                < half_height - seam_tolerance
            )
        )
        crossed.append(started_below and lower_before_crossing)
        crossing_frames.append(
            first_crossing if started_below and lower_before_crossing else None
        )

    terminal_z = tip_z[-1]
    terminal_centroid_z = float(np.mean(terminal_z))
    terminal_max_z = float(np.max(terminal_z))
    target = np.asarray(frame_target_distance_m, dtype=np.float64)
    tail_start_m = float(target[-1] - 0.020)
    tail = target >= tail_start_m - NUMERICAL_TOLERANCE
    tail_recovery_frames = int(
        np.count_nonzero(np.asarray(recovery_mask, dtype=bool)[tail])
    )

    saved_start = np.asarray(plan["start_surface_local_m"], dtype=np.float64)
    saved_end = np.asarray(plan["end_surface_local_m"], dtype=np.float64)
    endpoint_local_error = float(
        max(
            np.max(np.abs(local[0] - saved_start)),
            np.max(np.abs(local[-1] - saved_end)),
        )
    )
    route_length = float(np.asarray(frame_target_distance_m)[-1])
    axial_direction = int(_scalar(plan, "axial_direction"))
    terminal_progress = np.asarray(
        plan["progress_m"], dtype=np.float64
    )[-1, 1:]
    terminal_residual = np.asarray(
        plan["progress_residual_m"], dtype=np.float64
    )[-1, 1:]
    expected_terminal_residual = np.abs(
        terminal_progress - LEVEL2_ROUTE_LENGTH_M
    )
    terminal_residual_consistency_error = float(
        np.max(np.abs(terminal_residual - expected_terminal_residual))
    )
    arc = capsule_meridian_arcs_local(
        local[:, 1:],
        radius,
        half_height,
    )
    saved_progress = np.asarray(plan["progress_m"], dtype=np.float64)[:, 1:]
    progress_reference_arc = arc[0] - axial_direction * saved_progress[0]
    geometry_progress = axial_direction * (arc - progress_reference_arc)
    progress_geometry_error = float(
        np.max(np.abs(geometry_progress - saved_progress))
    )
    maximum_interpolated_backtracking_step = float(
        max(0.0, -np.min(np.diff(geometry_progress, axis=0)))
    )
    coarse_progress = np.asarray(
        plan["coarse_progress_m"], dtype=np.float64
    )[:, 1:]
    maximum_keyframe_backtracking_step = float(
        max(0.0, -np.min(np.diff(coarse_progress, axis=0)))
    )
    checks = {
        "canonical_capsule_radius": bool(
            np.isclose(radius, LEVEL2_RADIUS_M, atol=1.0e-9)
        ),
        "canonical_capsule_half_height": bool(
            np.isclose(half_height, LEVEL2_HALF_HEIGHT_M, atol=1.0e-9)
        ),
        "canonical_route_length": bool(
            np.isclose(route_length, LEVEL2_ROUTE_LENGTH_M, atol=1.0e-9)
        ),
        "positive_bottom_to_top_direction": axial_direction == 1,
        "saved_progress_matches_surface_geometry": (
            progress_geometry_error <= FK_POSITION_TOLERANCE_M
        ),
        "object_pose_matches_saved_local_endpoints": (
            endpoint_local_error <= FK_POSITION_TOLERANCE_M
        ),
        "all_tips_cross_upper_seam": all(crossed),
        "terminal_all_tips_on_upper_cap": bool(
            np.all(terminal_z >= half_height - seam_tolerance)
        ),
        "terminal_centroid_reaches_top_region": (
            terminal_centroid_z >= half_height + 0.50 * radius - seam_tolerance
        ),
        "at_least_one_tip_reaches_high_top_region": (
            terminal_max_z >= half_height + 0.75 * radius - seam_tolerance
        ),
        "final_20mm_has_no_recovery": tail_recovery_frames == 0,
        "terminal_planned_tip_progress_reaches_route": bool(
            np.all(
                np.abs(terminal_progress - LEVEL2_ROUTE_LENGTH_M)
                <= LEVEL2_TERMINAL_PROGRESS_TOLERANCE_M
                + NUMERICAL_TOLERANCE
            )
        ),
        "terminal_planned_tip_progress_residual_within_4mm": bool(
            np.all(
                terminal_residual
                <= LEVEL2_TERMINAL_PROGRESS_TOLERANCE_M
                + NUMERICAL_TOLERANCE
            )
        ),
        "terminal_progress_residual_is_consistent": (
            terminal_residual_consistency_error <= 1.0e-5
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "endpoint_local_pose_error_m": endpoint_local_error,
        "maximum_progress_geometry_error_m": progress_geometry_error,
        "maximum_saved_keyframe_backtracking_step_m": (
            maximum_keyframe_backtracking_step
        ),
        "maximum_interpolated_backtracking_step_m": (
            maximum_interpolated_backtracking_step
        ),
        "upper_seam_z_m": half_height,
        "tip_crossed_upper_seam": dict(zip(FINGER_NAMES, crossed, strict=True)),
        "tip_first_crossing_frame": dict(
            zip(FINGER_NAMES, crossing_frames, strict=True)
        ),
        "terminal_tip_z_m": dict(
            zip(FINGER_NAMES, terminal_z.tolist(), strict=True)
        ),
        "terminal_centroid_z_m": terminal_centroid_z,
        "terminal_centroid_required_z_m": half_height + 0.50 * radius,
        "terminal_maximum_z_m": terminal_max_z,
        "terminal_maximum_required_z_m": half_height + 0.75 * radius,
        "final_tail_start_target_distance_m": tail_start_m,
        "final_tail_frame_count": int(np.count_nonzero(tail)),
        "final_tail_recovery_frame_count": tail_recovery_frames,
        "terminal_planned_tip_progress_m": terminal_progress.tolist(),
        "terminal_planned_tip_progress_residual_m": terminal_residual.tolist(),
        "terminal_progress_residual_consistency_error_m": (
            terminal_residual_consistency_error
        ),
    }


def audit_loaded_plan(
    plan: Mapping[str, np.ndarray],
    grasp: Mapping[str, np.ndarray],
    *,
    mode: str = "Diagnostic",
    window_frames: int = DEFAULT_WINDOW_FRAMES,
    hz: float = DEFAULT_HZ,
    solver: Any | None = None,
    joint_names: Sequence[str] | None = None,
    projection_fn: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Run all plan audits and return one JSON-serializable report."""

    schema = validate_schema(plan, grasp, window_frames=window_frames)
    report: dict[str, Any] = {
        "schema_version": 1,
        "mode": mode,
        "conclusions": {
            "SCHEMA": "PASS" if schema["passed"] else "FAIL",
            "DIAGNOSTIC": "NOT_RUN",
            "LEVEL2_GEOMETRY": "NOT_RUN",
        },
        "schema": schema,
    }
    if not schema["passed"]:
        return report

    frame_count = int(schema["frame_count"])
    coarse = np.asarray(plan["mpc_coarse_distance_m"], dtype=np.float64)
    target_distance = reconstruct_frame_target_distance(frame_count, coarse)
    static_mask, derived_recovery = derive_frame_bridge_masks(
        target_distance,
        coarse,
        plan["mpc_coarse_static_feasibility_bridge"],
        plan["mpc_coarse_recovery_bridge"],
    )
    saved_recovery = np.asarray(plan["recovery_bridge_mask"], dtype=bool)
    marked = saved_recovery | static_mask
    low_motion = find_unmarked_low_motion_windows(
        plan["progress_m"],
        plan["kinematic_points_m"],
        target_distance,
        marked,
        plan["axial_distance_m"],
        window_frames=window_frames,
    )
    nominal = audit_nominal_three_of_four_runs(
        plan["scheduled_contact_mask"],
        plan["scheduled_contact_count"],
        saved_recovery,
        target_distance,
        transient_finger=int(_scalar(plan, "transient_contact_finger")),
        transient_start_m=float(_scalar(plan, "transient_contact_start_m")),
        transient_end_m=float(_scalar(plan, "transient_contact_end_m")),
        transient_enabled=(
            int(_scalar(plan, "min_planner_contact_fingers")) < 4
        ),
        minimum_run_frames=window_frames,
        hz=hz,
    )
    nominal_support = audit_nominal_support_policy(
        plan["scheduled_contact_mask"],
        plan["scheduled_contact_count"],
        saved_recovery,
        target_distance,
        transient_finger=int(_scalar(plan, "transient_contact_finger")),
        transient_start_m=float(_scalar(plan, "transient_contact_start_m")),
        transient_end_m=float(_scalar(plan, "transient_contact_end_m")),
        minimum_planner_contacts=int(
            _scalar(plan, "min_planner_contact_fingers")
        ),
        minimum_recovery_contacts=int(
            _scalar(plan, "mpc_recovery_bridge_min_contact_fingers")
        ),
    )
    terminal_support = audit_terminal_nominal_support(
        plan["scheduled_contact_count"],
        configured_frames=int(
            _scalar(plan, "final_contact_recovery_frames")
        ),
        mode=mode,
        hz=hz,
    )
    geometry = audit_level2_geometry(
        plan,
        grasp,
        target_distance,
        saved_recovery,
    )
    report["bridge_masks"] = {
        "derived_static_frame_count": int(np.count_nonzero(static_mask)),
        "derived_recovery_frame_count": int(
            np.count_nonzero(derived_recovery)
        ),
        "saved_recovery_frame_count": int(np.count_nonzero(saved_recovery)),
    }
    report["unmarked_low_motion"] = {
        "window_frames": window_frames,
        "duration_s": window_frames / hz,
        "forward_progress_ratio": FORWARD_PROGRESS_RATIO,
        "required_forward_fingers": REQUIRED_FORWARD_FINGERS,
        "region_count": len(low_motion),
        "regions": low_motion,
    }
    report["nominal_three_of_four"] = nominal
    report["nominal_support_policy"] = nominal_support
    report["terminal_nominal_support"] = terminal_support
    report["level2_geometry"] = geometry

    try:
        kinematics = audit_kinematics(
            plan,
            grasp,
            mode=mode,
            solver=solver,
            joint_names=joint_names,
            projection_fn=projection_fn,
        )
    except Exception as exc:  # Preserve NumPy-only results if CPU model fails.
        report["kinematics"] = {
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
            "cpu_only": True,
            "mjlab_environment_started": False,
        }
        report["conclusions"]["DIAGNOSTIC"] = "ERROR"
    else:
        diagnostic_checks = {
            "no_unmarked_low_motion_regions": len(low_motion) == 0,
            "no_unexpected_long_nominal_three_of_four_runs": (
                nominal["unexpected_long_run_count"] == 0
            ),
            "nominal_support_policy": bool(nominal_support["passed"]),
            "terminal_nominal_support": bool(terminal_support["passed"]),
            "kinematics": bool(kinematics["passed"]),
        }
        report["kinematics"] = kinematics
        report["diagnostic_checks"] = diagnostic_checks
        report["conclusions"]["DIAGNOSTIC"] = (
            "PASS" if all(diagnostic_checks.values()) else "FAIL"
        )

    report["conclusions"]["LEVEL2_GEOMETRY"] = (
        "PASS" if geometry["passed"] else "FAIL"
    )
    return report


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot encode {type(value).__name__} as JSON")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit a saved Baseline-2 Level-2 plan using NumPy and the "
            "MuJoCo CPU reachability model; no GPU environment is launched. "
            "The audit is frozen at 100 Hz with a 20-frame motion window."
        )
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--initial-grasp", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("Diagnostic", "Acceptance"),
        default="Diagnostic",
    )
    parser.add_argument("--json-indent", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        plan = load_npz_no_pickle(args.plan)
        grasp = load_npz_no_pickle(args.initial_grasp)
        report = audit_loaded_plan(
            plan,
            grasp,
            mode=args.mode,
            window_frames=DEFAULT_WINDOW_FRAMES,
            hz=DEFAULT_HZ,
        )
    except AuditInputError as exc:
        report = {
            "schema_version": 1,
            "mode": args.mode,
            "conclusions": {
                "SCHEMA": "FAIL",
                "DIAGNOSTIC": "NOT_RUN",
                "LEVEL2_GEOMETRY": "NOT_RUN",
            },
            "schema": {"passed": False, "errors": [str(exc)]},
        }
    except Exception as exc:
        report = {
            "schema_version": 1,
            "mode": args.mode,
            "conclusions": {
                "SCHEMA": "ERROR",
                "DIAGNOSTIC": "NOT_RUN",
                "LEVEL2_GEOMETRY": "NOT_RUN",
            },
            "error": f"Unexpected audit error: {type(exc).__name__}: {exc}",
        }
    report["inputs"] = {
        "plan": str(args.plan.resolve()),
        "initial_grasp": str(args.initial_grasp.resolve()),
    }
    try:
        payload = json.dumps(
            report,
            default=_json_default,
            ensure_ascii=False,
            indent=args.json_indent,
            sort_keys=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        report = {
            "schema_version": 1,
            "mode": args.mode,
            "conclusions": {
                "SCHEMA": "FAIL",
                "DIAGNOSTIC": "ERROR",
                "LEVEL2_GEOMETRY": "NOT_RUN",
            },
            "error": f"Non-JSON audit result: {type(exc).__name__}: {exc}",
            "inputs": report.get("inputs", {}),
        }
        payload = json.dumps(
            report,
            ensure_ascii=False,
            indent=args.json_indent,
            sort_keys=True,
            allow_nan=False,
        )
    print(payload)
    statuses = report["conclusions"].values()
    if all(status == "PASS" for status in statuses):
        return 0
    if report["conclusions"]["SCHEMA"] == "FAIL" or "ERROR" in statuses:
        return 2
    return 1


if __name__ == "__main__":
    sys.exit(main())
