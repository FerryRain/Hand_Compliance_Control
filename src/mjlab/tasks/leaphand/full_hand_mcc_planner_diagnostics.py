"""Pure-NumPy diagnostics for the Baseline-2 surface planner."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

import numpy as np


BRIDGE_CONDITION_NAMES = (
    "progress",
    "normal",
    "tangent",
    "monotonic",
    "palm",
    "collision",
    "joint",
    "motion",
    "budget",
)

# A moving bridge is a physical continuity mechanism, not a contact-policy
# relaxation.  Its forward-motion quorum remains three regardless of whether
# the current contact schedule requires two or all four nominal contacts.
MOVING_BRIDGE_FORWARD_FINGER_COUNT = 3


@dataclass(frozen=True)
class RejectedMovingBridgeCandidate:
    """State paired with one moving-bridge rejection record."""

    q_rad: np.ndarray
    points_m: np.ndarray
    arcs_m: np.ndarray
    desired_arcs_m: np.ndarray


def evaluate_moving_bridge_motion(
    *,
    max_joint_motion_rad: float,
    tip_motion_m: np.ndarray,
    minimum_tip_motion_m: float,
    active_fingers: np.ndarray,
) -> tuple[bool, int]:
    """Check genuine moving-bridge motion with a fixed three-tip quorum."""

    tip_motion = np.asarray(tip_motion_m, dtype=np.float64)
    active = np.asarray(active_fingers, dtype=np.bool_)
    progressing_count = int(
        np.count_nonzero(
            tip_motion >= minimum_tip_motion_m - 1.0e-12
        )
    )
    motion_ok = bool(
        max_joint_motion_rad > 1.0e-6
        and progressing_count >= MOVING_BRIDGE_FORWARD_FINGER_COUNT
        and (
            not np.any(active)
            or np.all(
                tip_motion[active]
                >= minimum_tip_motion_m - 1.0e-12
            )
        )
    )
    return motion_ok, progressing_count


def evaluate_bridge_conditions(
    *,
    progress_error_m: np.ndarray,
    progress_limit_m: float,
    normal_ok: bool,
    tangential_error_m: np.ndarray,
    tangential_limit_m: np.ndarray,
    monotonic_error_m: np.ndarray,
    monotonic_limit_m: float,
    palm_error_m: float,
    palm_limit_m: float,
    collision_ok: bool,
    joint_ok: bool,
    motion_ok: bool,
    budget_ok: bool,
) -> dict[str, bool]:
    """Evaluate the nine named conditions used by a moving bridge."""

    return {
        "progress": bool(
            float(np.max(progress_error_m)) <= progress_limit_m
        ),
        "normal": bool(normal_ok),
        "tangent": bool(
            np.all(
                np.abs(tangential_error_m)
                <= np.asarray(tangential_limit_m, dtype=np.float64)
            )
        ),
        "monotonic": bool(
            float(np.max(monotonic_error_m)) <= monotonic_limit_m
        ),
        "palm": bool(palm_error_m <= palm_limit_m),
        "collision": bool(collision_ok),
        "joint": bool(joint_ok),
        "motion": bool(motion_ok),
        "budget": bool(budget_ok),
    }


def _json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    return value


def make_bridge_rejection_record(
    *,
    keyframe: int,
    keyframe_count: int,
    distance_m: float,
    fallback: str,
    strict_conditions: Mapping[str, bool],
    recovery_conditions: Mapping[str, bool],
    metrics: Mapping[str, object],
) -> dict[str, object]:
    """Create a stable, JSON/NPZ-safe bridge rejection record."""

    expected = set(BRIDGE_CONDITION_NAMES)
    for label, conditions in (
        ("strict", strict_conditions),
        ("recovery", recovery_conditions),
    ):
        missing = expected - set(conditions)
        extra = set(conditions) - expected
        if missing or extra:
            raise ValueError(
                f"{label} bridge conditions mismatch: "
                f"missing={sorted(missing)} extra={sorted(extra)}"
            )
    return {
        "schema_version": 1,
        "keyframe": int(keyframe),
        "keyframe_count": int(keyframe_count),
        "distance_m": float(distance_m),
        "fallback": str(fallback),
        "strict": {
            name: bool(strict_conditions[name])
            for name in BRIDGE_CONDITION_NAMES
        },
        "recovery": {
            name: bool(recovery_conditions[name])
            for name in BRIDGE_CONDITION_NAMES
        },
        "metrics": _json_ready(metrics),
    }


def format_bridge_rejection_record(record: Mapping[str, object]) -> str:
    """Serialize a bridge record into one deterministic log line."""

    return json.dumps(
        _json_ready(record),
        sort_keys=True,
        separators=(",", ":"),
    )


def build_bridge_rejection_metrics(
    *,
    bridge_interval_m: float,
    bridge_interval_limit_m: float,
    progress_error_m: np.ndarray,
    strict_progress_limit_m: float,
    recovery_progress_limit_m: float,
    normal_error_m: np.ndarray,
    strict_normal_tolerance_m: np.ndarray,
    strict_contact_mask: np.ndarray,
    strict_contact_count_required: int,
    recovery_normal_tolerance_m: np.ndarray,
    recovery_contact_mask: np.ndarray,
    recovery_contact_count_required: int,
    tangential_error_m: np.ndarray,
    tangential_tolerance_m: np.ndarray,
    monotonic_error_m: np.ndarray,
    monotonic_limit_m: float,
    palm_error_m: float,
    palm_limit_m: float,
    collision_mode: str,
    arm_clearance_m: float,
    arm_clearance_limit_m: float,
    hand_clearance_m: float,
    hand_clearance_limit_m: float,
    self_collision_count: int,
    pad_alignment: float,
    pad_alignment_limit: float,
    joint_min_margin_rad: float,
    max_joint_motion_rad: float,
    tip_motion_m: np.ndarray,
    minimum_tip_motion_m: float,
    bridge_active_fingers: np.ndarray,
    progressing_finger_count: int,
    progressing_finger_count_required: int,
    recovery_dwell_m: float,
    recovery_dwell_limit_m: float,
    recovery_total_m: float,
    recovery_total_limit_m: float,
    distance_m: float,
    recovery_terminal_cutoff_m: float,
    solver_cost: float,
    solver_nfev: int,
) -> dict[str, object]:
    """Build values, limits, and signed margins for a rejected bridge."""

    progress_error = np.asarray(progress_error_m, dtype=np.float64)
    normal_error = np.asarray(normal_error_m, dtype=np.float64)
    strict_normal_tolerance = np.asarray(
        strict_normal_tolerance_m, dtype=np.float64
    )
    recovery_normal_tolerance = np.asarray(
        recovery_normal_tolerance_m, dtype=np.float64
    )
    tangential_error = np.abs(
        np.asarray(tangential_error_m, dtype=np.float64)
    )
    tangential_tolerance = np.asarray(
        tangential_tolerance_m, dtype=np.float64
    )
    monotonic_error = np.asarray(monotonic_error_m, dtype=np.float64)
    tip_motion = np.asarray(tip_motion_m, dtype=np.float64)
    active_fingers = np.asarray(bridge_active_fingers, dtype=np.bool_)
    progress_max_error_m = float(progress_error.max())
    monotonic_max_error_m = float(monotonic_error.max())
    strict_normal_margin = strict_normal_tolerance - normal_error
    recovery_normal_margin = recovery_normal_tolerance - normal_error
    tangential_margin = tangential_tolerance - tangential_error
    strict_contact_count = int(np.count_nonzero(strict_contact_mask))
    recovery_contact_count = int(np.count_nonzero(recovery_contact_mask))
    active_tip_motion_margin = (
        tip_motion[active_fingers] - minimum_tip_motion_m
    )
    metrics: dict[str, object] = {
        "bridge_interval_m": bridge_interval_m,
        "bridge_interval_limit_m": bridge_interval_limit_m,
        "bridge_interval_margin_m": (
            bridge_interval_limit_m - bridge_interval_m
        ),
        "progress_error_m": progress_error,
        "progress_max_error_m": progress_max_error_m,
        "strict_progress_limit_m": strict_progress_limit_m,
        "strict_progress_margin_m": (
            strict_progress_limit_m - progress_max_error_m
        ),
        "recovery_progress_limit_m": recovery_progress_limit_m,
        "recovery_progress_margin_m": (
            recovery_progress_limit_m - progress_max_error_m
        ),
        "normal_error_m": normal_error,
        "strict_normal_tolerance_m": strict_normal_tolerance,
        "strict_normal_margin_m": strict_normal_margin,
        "strict_normal_min_margin_m": float(strict_normal_margin.min()),
        "strict_contact_count": strict_contact_count,
        "strict_contact_count_required": strict_contact_count_required,
        "strict_contact_count_margin": (
            strict_contact_count - strict_contact_count_required
        ),
        "recovery_normal_tolerance_m": recovery_normal_tolerance,
        "recovery_normal_margin_m": recovery_normal_margin,
        "recovery_normal_min_margin_m": float(
            recovery_normal_margin.min()
        ),
        "recovery_contact_count": recovery_contact_count,
        "recovery_contact_count_required": (
            recovery_contact_count_required
        ),
        "recovery_contact_count_margin": (
            recovery_contact_count - recovery_contact_count_required
        ),
        "tangential_error_m": tangential_error,
        "tangential_tolerance_m": tangential_tolerance,
        "tangential_margin_m": tangential_margin,
        "tangential_min_margin_m": float(tangential_margin.min()),
        "monotonic_error_m": monotonic_error,
        "monotonic_max_error_m": monotonic_max_error_m,
        "monotonic_limit_m": monotonic_limit_m,
        "monotonic_margin_m": monotonic_limit_m - monotonic_max_error_m,
        "palm_error_m": palm_error_m,
        "palm_limit_m": palm_limit_m,
        "palm_margin_m": palm_limit_m - palm_error_m,
        "collision_mode": collision_mode,
        "arm_clearance_limit_m": arm_clearance_limit_m,
        "hand_clearance_limit_m": hand_clearance_limit_m,
        "self_collision_count": self_collision_count,
        "self_collision_limit": 0,
        "self_collision_margin": -self_collision_count,
        "pad_alignment": pad_alignment,
        "pad_alignment_limit": pad_alignment_limit,
        "pad_alignment_margin": pad_alignment - pad_alignment_limit,
        "joint_min_margin_rad": joint_min_margin_rad,
        "joint_margin_limit_rad": 0.0,
        "max_joint_motion_rad": max_joint_motion_rad,
        "minimum_joint_motion_rad": 1.0e-6,
        "joint_motion_margin_rad": max_joint_motion_rad - 1.0e-6,
        "tip_motion_m": tip_motion,
        "minimum_tip_motion_m": minimum_tip_motion_m,
        "bridge_active_fingers": active_fingers,
        "active_tip_motion_margin_m": active_tip_motion_margin,
        "active_tip_motion_min_margin_m": (
            float(active_tip_motion_margin.min())
            if active_tip_motion_margin.size
            else None
        ),
        "progressing_finger_count": progressing_finger_count,
        "progressing_finger_count_required": (
            progressing_finger_count_required
        ),
        "progressing_finger_count_margin": (
            progressing_finger_count - progressing_finger_count_required
        ),
        "recovery_dwell_m": recovery_dwell_m,
        "recovery_dwell_limit_m": recovery_dwell_limit_m,
        "recovery_dwell_margin_m": (
            recovery_dwell_limit_m - recovery_dwell_m
        ),
        "recovery_total_m": recovery_total_m,
        "recovery_total_limit_m": recovery_total_limit_m,
        "recovery_total_margin_m": (
            recovery_total_limit_m - recovery_total_m
        ),
        "recovery_terminal_cutoff_m": recovery_terminal_cutoff_m,
        "recovery_terminal_margin_m": (
            recovery_terminal_cutoff_m - distance_m
        ),
        "solver_cost": solver_cost,
        "solver_nfev": solver_nfev,
    }
    if collision_mode == "full_robot":
        metrics.update(
            {
                "arm_clearance_m": arm_clearance_m,
                "arm_clearance_margin_m": (
                    arm_clearance_m - arm_clearance_limit_m
                ),
                "hand_clearance_m": hand_clearance_m,
                "hand_clearance_margin_m": (
                    hand_clearance_m - hand_clearance_limit_m
                ),
            }
        )
    return metrics


def build_candidate_failure_metrics(
    *,
    progress_error_m: np.ndarray,
    progress_limit_m: float,
    normal_error_m: np.ndarray,
    normal_tolerance_m: np.ndarray,
    contact_mask: np.ndarray,
    contact_count_required: int,
    tangential_error_m: np.ndarray,
    tangential_tolerance_m: np.ndarray,
    monotonic_error_m: np.ndarray,
    monotonic_limit_m: float,
    palm_error_m: float,
    palm_limit_m: float,
    palm_error_world_m: np.ndarray,
    palm_error_local_m: np.ndarray,
    collision_mode: str,
    arm_clearance_m: float,
    arm_clearance_limit_m: float,
    arm_nearest_geometry: str,
    hand_clearance_m: float,
    hand_clearance_limit_m: float,
    hand_nearest_geometry: str,
    self_collision_count: int,
    pad_alignment: float,
    pad_alignment_limit: float,
    joint_min_margin_rad: float,
    solver_cost: float,
    solver_nfev: int,
) -> dict[str, object]:
    """Build flat, pickle-free metrics for a rejected coarse candidate."""

    progress_error = np.asarray(progress_error_m, dtype=np.float64)
    normal_error = np.asarray(normal_error_m, dtype=np.float64)
    normal_tolerance = np.asarray(normal_tolerance_m, dtype=np.float64)
    tangential_error = np.abs(
        np.asarray(tangential_error_m, dtype=np.float64)
    )
    tangential_tolerance = np.asarray(
        tangential_tolerance_m, dtype=np.float64
    )
    monotonic_error = np.asarray(monotonic_error_m, dtype=np.float64)
    progress_max_error_m = float(progress_error.max())
    monotonic_max_error_m = float(monotonic_error.max())
    normal_margin = normal_tolerance - normal_error
    tangential_margin = tangential_tolerance - tangential_error
    contact_count = int(np.count_nonzero(contact_mask))
    collision_ok = bool(
        collision_mode != "full_robot"
        or (
            arm_clearance_m >= arm_clearance_limit_m
            and hand_clearance_m >= hand_clearance_limit_m
            and self_collision_count == 0
            and pad_alignment >= pad_alignment_limit
        )
    )
    return {
        "progress_error_m": progress_error,
        "progress_max_error_m": progress_max_error_m,
        "progress_limit_m": progress_limit_m,
        "progress_margin_m": progress_limit_m - progress_max_error_m,
        "normal_error_m": normal_error,
        "normal_tolerance_m": normal_tolerance,
        "normal_margin_m": normal_margin,
        "normal_min_margin_m": float(normal_margin.min()),
        "contact_count": contact_count,
        "contact_count_required": contact_count_required,
        "contact_count_margin": contact_count - contact_count_required,
        "tangential_error_m": tangential_error,
        "tangential_tolerance_m": tangential_tolerance,
        "tangential_margin_m": tangential_margin,
        "tangential_min_margin_m": float(tangential_margin.min()),
        "monotonic_error_m": monotonic_error,
        "monotonic_max_error_m": monotonic_max_error_m,
        "monotonic_limit_m": monotonic_limit_m,
        "monotonic_margin_m": monotonic_limit_m - monotonic_max_error_m,
        "palm_error_m": palm_error_m,
        "palm_limit_m": palm_limit_m,
        "palm_margin_m": palm_limit_m - palm_error_m,
        "palm_error_world_m": np.asarray(
            palm_error_world_m, dtype=np.float64
        ),
        "palm_error_local_m": np.asarray(
            palm_error_local_m, dtype=np.float64
        ),
        "collision_mode": collision_mode,
        "arm_clearance_m": arm_clearance_m,
        "arm_clearance_limit_m": arm_clearance_limit_m,
        "arm_clearance_margin_m": arm_clearance_m - arm_clearance_limit_m,
        "arm_nearest_geometry": arm_nearest_geometry,
        "hand_clearance_m": hand_clearance_m,
        "hand_clearance_limit_m": hand_clearance_limit_m,
        "hand_clearance_margin_m": hand_clearance_m - hand_clearance_limit_m,
        "hand_nearest_geometry": hand_nearest_geometry,
        "self_collision_count": self_collision_count,
        "self_collision_limit": 0,
        "self_collision_margin": -self_collision_count,
        "pad_alignment": pad_alignment,
        "pad_alignment_limit": pad_alignment_limit,
        "pad_alignment_margin": pad_alignment - pad_alignment_limit,
        "joint_min_margin_rad": joint_min_margin_rad,
        "joint_margin_limit_rad": 0.0,
        "condition_progress_ok": progress_max_error_m <= progress_limit_m,
        "condition_normal_ok": bool(
            contact_count >= contact_count_required
            and np.all(normal_error <= normal_tolerance)
        ),
        "condition_tangent_ok": bool(
            np.all(tangential_error <= tangential_tolerance)
        ),
        "condition_monotonic_ok": (
            monotonic_max_error_m <= monotonic_limit_m
        ),
        "condition_palm_ok": palm_error_m <= palm_limit_m,
        "condition_collision_ok": collision_ok,
        "condition_joint_ok": joint_min_margin_rad >= -1.0e-12,
        "solver_cost": solver_cost,
        "solver_nfev": solver_nfev,
    }


def build_palm_guide_multistart_specs(
    normal: np.ndarray,
    azimuth: np.ndarray,
    meridian: np.ndarray,
    max_drift_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return bounded layered palm offsets and paired FR3 redundancy seeds.

    The Cartesian offsets span three radii inside the guide drift guard.  The
    paired arm perturbations seed different 7-DoF FR3 branches; they are only
    initial guesses and never relax the final joint, collision, pad, or palm
    drift checks.
    """

    if max_drift_m <= 0.0:
        raise ValueError("max_drift_m must be positive")

    basis = []
    for vector in (normal, azimuth, meridian):
        value = np.asarray(vector, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(value))
        if norm <= 1.0e-12:
            raise ValueError("Palm guide basis vectors must be nonzero")
        basis.append(value / norm)
    n, a, m = basis

    directions = (
        # Inner layer: local axes around the previous arm branch.
        (0.20, -n),
        (0.20, n),
        (0.20, a),
        (0.20, -a),
        (0.20, m),
        (0.20, -m),
        # Middle layer: coupled normal/tangent motion changes elbow branch.
        (0.50, -n + a),
        (0.50, -n - a),
        (0.50, -n + m),
        (0.50, -n - m),
        # Outer layer remains strictly inside the guide drift guard.
        (0.85, a + m),
        (0.85, a - m),
        (0.85, -a + m),
        (0.85, -a - m),
    )
    offsets = np.stack(
        [
            fraction
            * max_drift_m
            * direction
            / max(float(np.linalg.norm(direction)), 1.0e-12)
            for fraction, direction in directions
        ]
    )

    # Deterministic perturbations on FR3 joints 1, 3, and 7.  Their amplitude
    # is small enough to remain a seed rather than an implicit relaxed target.
    arm_patterns = np.zeros((len(directions), 7), dtype=np.float64)
    seed_patterns = (
        (6, 1.0),
        (6, -1.0),
        (2, 1.0),
        (2, -1.0),
        (0, 1.0),
        (0, -1.0),
    )
    for index, (fraction, _) in enumerate(directions):
        joint, sign = seed_patterns[index % len(seed_patterns)]
        arm_patterns[index, joint] = sign * (0.04 + 0.08 * fraction)
    return offsets, arm_patterns


def save_mpc_failure_prefix(
    path: Path,
    *,
    reason: str,
    keyframe: int,
    keyframe_count: int,
    failure_distance_m: float,
    last_feasible_distance_m: np.ndarray,
    last_feasible_q_rad: np.ndarray,
    last_feasible_points_m: np.ndarray,
    last_feasible_arcs_m: np.ndarray,
    final_best_desired_arcs_m: np.ndarray,
    final_best_q_rad: np.ndarray,
    final_best_points_m: np.ndarray,
    final_best_arcs_m: np.ndarray,
    rephase_offset_m: np.ndarray,
    budget_values: Mapping[str, object],
    failure_metrics: Mapping[str, object],
    bridge_record: Mapping[str, object] | None = None,
    rejected_moving_bridge: RejectedMovingBridgeCandidate | None = None,
) -> Path:
    """Persist an exhausted coarse-shooting prefix without pickles."""

    if (bridge_record is None) != (rejected_moving_bridge is None):
        raise ValueError(
            "bridge_record and rejected_moving_bridge must be provided "
            "together so their metrics and state cannot be mispaired"
        )

    requested_output = Path(path)
    if requested_output.suffix.lower() != ".npz":
        requested_output = requested_output.with_suffix(".npz")
    requested_output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "schema_version": np.asarray(2, dtype=np.int32),
        "reason": np.asarray(str(reason)),
        "keyframe": np.asarray(keyframe, dtype=np.int32),
        "keyframe_count": np.asarray(keyframe_count, dtype=np.int32),
        "failure_distance_m": np.asarray(failure_distance_m, dtype=np.float64),
        "last_feasible_distance_m": np.asarray(
            last_feasible_distance_m, dtype=np.float64
        ),
        "last_feasible_coarse_q_rad": np.asarray(
            last_feasible_q_rad, dtype=np.float64
        ),
        "last_feasible_coarse_points_m": np.asarray(
            last_feasible_points_m, dtype=np.float64
        ),
        "last_feasible_coarse_arcs_m": np.asarray(
            last_feasible_arcs_m, dtype=np.float64
        ),
        "failure_final_best_desired_arcs_m": np.asarray(
            final_best_desired_arcs_m, dtype=np.float64
        ),
        "failure_final_best_q_rad": np.asarray(
            final_best_q_rad, dtype=np.float64
        ),
        "failure_final_best_points_m": np.asarray(
            final_best_points_m, dtype=np.float64
        ),
        "failure_final_best_arcs_m": np.asarray(
            final_best_arcs_m, dtype=np.float64
        ),
        "rephase_offset_m": np.asarray(rephase_offset_m, dtype=np.float64),
    }
    for prefix, values in (
        ("budget", budget_values),
        ("metric", failure_metrics),
    ):
        for name, value in values.items():
            payload[f"{prefix}_{name}"] = np.asarray(value)
    if bridge_record is not None:
        assert rejected_moving_bridge is not None
        payload["bridge_rejection_json"] = np.asarray(
            format_bridge_rejection_record(bridge_record)
        )
        payload["bridge_rejected_q_rad"] = np.asarray(
            rejected_moving_bridge.q_rad,
            dtype=np.float64,
        )
        payload["bridge_rejected_points_m"] = np.asarray(
            rejected_moving_bridge.points_m,
            dtype=np.float64,
        )
        payload["bridge_rejected_arcs_m"] = np.asarray(
            rejected_moving_bridge.arcs_m,
            dtype=np.float64,
        )
        payload["bridge_rejected_desired_arcs_m"] = np.asarray(
            rejected_moving_bridge.desired_arcs_m,
            dtype=np.float64,
        )
        strict = bridge_record["strict"]
        recovery = bridge_record["recovery"]
        assert isinstance(strict, Mapping)
        assert isinstance(recovery, Mapping)
        payload["bridge_condition_names"] = np.asarray(
            BRIDGE_CONDITION_NAMES
        )
        payload["bridge_strict_conditions"] = np.asarray(
            [bool(strict[name]) for name in BRIDGE_CONDITION_NAMES],
            dtype=np.bool_,
        )
        payload["bridge_recovery_conditions"] = np.asarray(
            [bool(recovery[name]) for name in BRIDGE_CONDITION_NAMES],
            dtype=np.bool_,
        )
    for name, value in payload.items():
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise TypeError(
                f"Failure-prefix field {name!r} has object dtype and cannot "
                "be replayed with allow_pickle=False"
            )
        payload[name] = array

    # Opening with ``xb`` is the no-overwrite guarantee.  A preceding
    # exists() check alone would race when several deterministic seeds fail
    # concurrently and choose the same suffix.
    suffix_index = 0
    while True:
        output = (
            requested_output
            if suffix_index == 0
            else requested_output.with_name(
                f"{requested_output.stem}_{suffix_index:03d}"
                f"{requested_output.suffix}"
            )
        )
        created_output = False
        try:
            with output.open("xb") as output_file:
                created_output = True
                np.savez_compressed(output_file, **payload)
            return output
        except FileExistsError:
            suffix_index += 1
        except Exception:
            if created_output:
                output.unlink(missing_ok=True)
            raise
