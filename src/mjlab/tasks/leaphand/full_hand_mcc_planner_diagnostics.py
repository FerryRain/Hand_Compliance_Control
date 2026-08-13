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

# This is the frozen Level-2 plan-audit definition.  Keep the online
# pre-dynamics guard and the standalone auditor on this single NumPy
# implementation so a plan cannot pass one interpretation and fail the other.
LOW_MOTION_REQUIRED_FORWARD_FINGERS = 3
LOW_MOTION_FORWARD_PROGRESS_RATIO = 0.10
LOW_MOTION_DEFAULT_WINDOW_FRAMES = 20
LOW_MOTION_NUMERICAL_TOLERANCE = 1.0e-8


def smooth_pad_alignment_residual(
    pad_alignment: np.ndarray,
    *,
    target_alignment: float,
    tau: float,
) -> np.ndarray:
    """Return a stable softplus penalty below a preferred pad alignment."""

    alignment = np.asarray(pad_alignment, dtype=np.float64)
    if not np.all(np.isfinite(alignment)):
        raise ValueError("pad_alignment must be finite")
    if not np.isfinite(target_alignment) or not -1.0 <= target_alignment <= 1.0:
        raise ValueError("target_alignment must be finite and in [-1, 1]")
    if not np.isfinite(tau) or tau <= 0.0:
        raise ValueError("tau must be finite and positive")
    return tau * np.logaddexp(
        0.0,
        (float(target_alignment) - alignment) / tau,
    )


def positive_self_clearance_residual(
    protected_clearance_m: np.ndarray,
    *,
    target_clearance_m: float,
) -> np.ndarray:
    """Return a one-sided residual below a positive self-clearance target."""

    clearance = np.asarray(protected_clearance_m, dtype=np.float64)
    if clearance.ndim != 1 or clearance.size == 0:
        raise ValueError("protected_clearance_m must be a non-empty vector")
    if not np.all(np.isfinite(clearance)):
        raise ValueError("protected_clearance_m must be finite")
    if not np.isfinite(target_clearance_m) or target_clearance_m < 0.0:
        raise ValueError("target_clearance_m must be finite and non-negative")
    return np.maximum(float(target_clearance_m) - clearance, 0.0)


def central_difference_clearance_gradient(
    plus_clearance_m: np.ndarray,
    minus_clearance_m: np.ndarray,
    sample_span_rad: np.ndarray,
) -> np.ndarray:
    """Recover a deterministic central-FD distance gradient from samples."""

    plus = np.asarray(plus_clearance_m, dtype=np.float64)
    minus = np.asarray(minus_clearance_m, dtype=np.float64)
    span = np.asarray(sample_span_rad, dtype=np.float64)
    if plus.ndim != 1 or plus.shape != minus.shape or plus.shape != span.shape:
        raise ValueError("central-difference inputs must be equal-sized vectors")
    if plus.size == 0 or not np.all(np.isfinite(plus)):
        raise ValueError("plus_clearance_m must be finite and non-empty")
    if not np.all(np.isfinite(minus)) or not np.all(np.isfinite(span)):
        raise ValueError("central-difference inputs must be finite")
    if np.any(span <= 0.0):
        raise ValueError("sample_span_rad must be positive")
    return (plus - minus) / span


def self_separation_ascent_seeds(
    q_rad: np.ndarray,
    clearance_gradient_m_per_rad: np.ndarray,
    lower_rad: np.ndarray,
    upper_rad: np.ndarray,
    *,
    maximum_step_rad: float,
) -> tuple[np.ndarray, ...]:
    """Build 0.4x/1.0x normalized ascent seeds for a protected pair."""

    q = np.asarray(q_rad, dtype=np.float64)
    gradient = np.asarray(clearance_gradient_m_per_rad, dtype=np.float64)
    lower = np.asarray(lower_rad, dtype=np.float64)
    upper = np.asarray(upper_rad, dtype=np.float64)
    if q.ndim != 1 or q.shape != gradient.shape:
        raise ValueError("q_rad and clearance gradient must be equal vectors")
    if lower.shape != q.shape or upper.shape != q.shape:
        raise ValueError("joint bounds must match q_rad")
    if not all(
        np.all(np.isfinite(value))
        for value in (q, gradient, lower, upper)
    ):
        raise ValueError("self-separation inputs must be finite")
    if np.any(lower > upper):
        raise ValueError("lower_rad cannot exceed upper_rad")
    if not np.isfinite(maximum_step_rad) or maximum_step_rad <= 0.0:
        raise ValueError("maximum_step_rad must be finite and positive")
    gradient_norm = float(np.linalg.norm(gradient))
    if gradient_norm <= 1.0e-15:
        return ()
    direction = gradient / gradient_norm
    seeds: list[np.ndarray] = []
    for fraction in (0.4, 1.0):
        seed = np.clip(
            q + fraction * float(maximum_step_rad) * direction,
            lower,
            upper,
        )
        if np.allclose(seed, q, atol=1.0e-14, rtol=0.0):
            continue
        if any(
            np.allclose(seed, old, atol=1.0e-14, rtol=0.0)
            for old in seeds
        ):
            continue
        seeds.append(seed)
    return tuple(seeds)


def deduplicated_bridge_multistart_seeds(
    previous_q_rad: np.ndarray,
    separation_seeds_rad: tuple[np.ndarray, ...],
    *,
    atol_rad: float = 1.0e-14,
) -> tuple[np.ndarray, ...]:
    """Keep the predecessor first and append every distinct separation seed.

    A protected-self nudge is only an additional basin probe.  It must never
    replace the last accepted state, because doing so silently discards the
    locally continuous branch that the feasibility bridge exists to recover.
    """

    previous = np.asarray(previous_q_rad, dtype=np.float64)
    if previous.ndim != 1 or previous.size == 0:
        raise ValueError("previous_q_rad must be a non-empty vector")
    if not np.all(np.isfinite(previous)):
        raise ValueError("previous_q_rad must be finite")
    if not np.isfinite(atol_rad) or atol_rad < 0.0:
        raise ValueError("atol_rad must be finite and non-negative")
    seeds: list[np.ndarray] = [previous.copy()]
    for candidate_raw in separation_seeds_rad:
        candidate = np.asarray(candidate_raw, dtype=np.float64)
        if candidate.shape != previous.shape:
            raise ValueError("all bridge seeds must match previous_q_rad")
        if not np.all(np.isfinite(candidate)):
            raise ValueError("all bridge seeds must be finite")
        if any(
            np.allclose(candidate, old, atol=atol_rad, rtol=0.0)
            for old in seeds
        ):
            continue
        seeds.append(candidate.copy())
    return tuple(seeds)


def moving_bridge_tip_geometry_residual(
    tip_clearance_m: np.ndarray,
    target_clearance_m: np.ndarray,
    *,
    inner_cap_m: float,
    target_weight: float,
    target_scale: float,
    inner_weight: float,
) -> np.ndarray:
    """Return four target and four one-sided inner-tip bridge residuals."""

    clearance = np.asarray(tip_clearance_m, dtype=np.float64)
    target = np.asarray(target_clearance_m, dtype=np.float64)
    if clearance.shape != (4,) or target.shape != (4,):
        raise ValueError("tip clearance and target must have shape (4,)")
    scalar_values = np.asarray(
        [inner_cap_m, target_weight, target_scale, inner_weight],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(clearance)) or not np.all(np.isfinite(target)):
        raise ValueError("tip clearance and target must be finite")
    if not np.all(np.isfinite(scalar_values)):
        raise ValueError("tip residual parameters must be finite")
    if target_weight < 0.0 or target_scale < 0.0 or inner_weight < 0.0:
        raise ValueError("tip residual weights and scale must be non-negative")
    return np.concatenate(
        (
            float(target_scale)
            * float(target_weight)
            * (clearance - target),
            float(inner_weight)
            * np.minimum(clearance - float(inner_cap_m), 0.0),
        )
    )


def moving_bridge_candidate_rank(
    *,
    strict_hard_feasible: bool,
    recovery_hard_feasible: bool,
    collision_hard_feasible: bool,
    failed_condition_count: int,
    minimum_tip_clearance_m: float,
    tip_inner_cap_m: float,
    minimum_protected_self_clearance_m: float,
    soft_self_clearance_target_m: float,
    minimum_pad_alignment: float,
    soft_pad_alignment: float,
    task_error_score: float,
    continuity_error: float,
    solver_cost: float,
) -> tuple[float, ...]:
    """Rank independently audited bridge solves without seed privilege.

    Strict feasibility precedes recovery feasibility.  Within either class,
    the existing segment collision bundle remains safe before physical-tip
    inner buffer, protected-self buffer, pad posture, failed-gate count, task
    error, continuity, and cost are ordered lexicographically.  This prevents
    an all-rejected, deeply colliding candidate with one failed gate from
    replacing a collision-safe rejection snapshot with two task failures.
    """

    scalar_values = np.asarray(
        [
            minimum_tip_clearance_m,
            tip_inner_cap_m,
            minimum_protected_self_clearance_m,
            soft_self_clearance_target_m,
            minimum_pad_alignment,
            soft_pad_alignment,
            task_error_score,
            continuity_error,
            solver_cost,
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(scalar_values)):
        raise ValueError("moving bridge rank inputs must be finite")
    if failed_condition_count < 0:
        raise ValueError("failed_condition_count cannot be negative")
    if soft_self_clearance_target_m < 0.0:
        raise ValueError("soft self-clearance target cannot be negative")
    feasibility_class = (
        0.0
        if strict_hard_feasible
        else 1.0
        if recovery_hard_feasible
        else 2.0
    )
    failed_gate_rank = (
        float(failed_condition_count)
        if feasibility_class == 2.0
        else 0.0
    )
    return (
        feasibility_class,
        float(not collision_hard_feasible),
        max(float(tip_inner_cap_m) - float(minimum_tip_clearance_m), 0.0),
        max(
            float(soft_self_clearance_target_m)
            - float(minimum_protected_self_clearance_m),
            0.0,
        ),
        max(float(soft_pad_alignment) - float(minimum_pad_alignment), 0.0),
        failed_gate_rank,
        float(task_error_score),
        float(continuity_error),
        float(solver_cost),
    )


def orientation_aware_candidate_rank(
    *,
    hard_feasible: bool,
    hard_violation_score: float,
    minimum_pad_alignment: float,
    hard_pad_alignment: float,
    soft_pad_alignment: float,
    task_error_score: float,
    continuity_error: float,
    solver_cost: float,
    minimum_protected_self_clearance_m: float,
    soft_self_clearance_target_m: float,
) -> tuple[float, ...]:
    """Build the common lexicographic rank for ordinary MPC candidates.

    Hard feasibility always wins.  Among hard-feasible states, positive
    protected self-clearance is preferred before the soft pad cone.  Once
    both soft bands pass, task tracking, continuity, and solver cost regain
    priority.
    """

    scalars = np.asarray(
        [
            hard_violation_score,
            minimum_pad_alignment,
            hard_pad_alignment,
            soft_pad_alignment,
            task_error_score,
            continuity_error,
            solver_cost,
            minimum_protected_self_clearance_m,
            soft_self_clearance_target_m,
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(scalars)):
        raise ValueError("candidate rank inputs must be finite")
    if soft_pad_alignment < hard_pad_alignment:
        raise ValueError(
            "soft_pad_alignment must describe a tighter cone than the hard "
            "alignment"
        )
    if soft_self_clearance_target_m < 0.0:
        raise ValueError("soft self-clearance target must be non-negative")
    hard_pad_deficit = max(
        float(hard_pad_alignment) - float(minimum_pad_alignment),
        0.0,
    )
    effective_hard_feasible = bool(
        hard_feasible and hard_pad_deficit <= 1.0e-12
    )
    soft_pad_deficit = max(
        float(soft_pad_alignment) - float(minimum_pad_alignment),
        0.0,
    )
    outside_soft_cone = soft_pad_deficit > 1.0e-12
    soft_self_deficit = max(
        float(soft_self_clearance_target_m)
        - float(minimum_protected_self_clearance_m),
        0.0,
    )
    outside_soft_self_clearance = soft_self_deficit > 1.0e-12
    return (
        float(not effective_hard_feasible),
        (
            0.0
            if effective_hard_feasible
            else max(float(hard_violation_score), 0.0)
            + hard_pad_deficit
        ),
        float(outside_soft_self_clearance),
        soft_self_deficit if outside_soft_self_clearance else 0.0,
        float(outside_soft_cone),
        soft_pad_deficit if outside_soft_cone else 0.0,
        float(task_error_score),
        float(continuity_error),
        float(solver_cost),
    )


def segment_tip_clearance_status(
    tip_clearance_m: np.ndarray,
    *,
    maximum_penetration_m: float,
) -> tuple[bool, float, tuple[int, int]]:
    """Audit all four physical tip geoms over sampled segment states."""

    clearances = np.asarray(tip_clearance_m, dtype=np.float64)
    if clearances.ndim != 2 or clearances.shape[1] != 4:
        raise ValueError("tip_clearance_m must have shape (samples, 4)")
    if clearances.shape[0] == 0 or not np.all(np.isfinite(clearances)):
        raise ValueError("tip_clearance_m must contain finite samples")
    if not np.isfinite(maximum_penetration_m) or maximum_penetration_m < 0.0:
        raise ValueError(
            "maximum_penetration_m must be finite and non-negative"
        )
    flat_index = int(np.argmin(clearances))
    sample_index, finger_index = np.unravel_index(
        flat_index,
        clearances.shape,
    )
    minimum = float(clearances[sample_index, finger_index])
    return (
        minimum >= -float(maximum_penetration_m) - 1.0e-12,
        minimum,
        (int(sample_index), int(finger_index)),
    )


@dataclass(frozen=True)
class RejectedMovingBridgeCandidate:
    """State paired with one moving-bridge rejection record."""

    q_rad: np.ndarray
    points_m: np.ndarray
    arcs_m: np.ndarray
    desired_arcs_m: np.ndarray


def find_unmarked_low_motion_windows(
    progress_m: np.ndarray,
    kinematic_points_m: np.ndarray,
    frame_target_distance_m: np.ndarray,
    marked_bridge_mask: np.ndarray,
    axial_distance_m: np.ndarray,
    *,
    window_frames: int = LOW_MOTION_DEFAULT_WINDOW_FRAMES,
    forward_progress_ratio: float = LOW_MOTION_FORWARD_PROGRESS_RATIO,
) -> list[dict[str, object]]:
    """Find frozen-window stalls not covered by an explicit bridge mask."""

    progress = np.asarray(progress_m, dtype=np.float64)
    points = np.asarray(kinematic_points_m, dtype=np.float64)
    target = np.asarray(frame_target_distance_m, dtype=np.float64)
    marked = np.asarray(marked_bridge_mask, dtype=bool)
    axial = np.asarray(axial_distance_m, dtype=np.float64)
    if window_frames <= 0:
        raise ValueError("window_frames must be positive")
    if not np.isfinite(forward_progress_ratio) or forward_progress_ratio < 0.0:
        raise ValueError("forward_progress_ratio must be finite and non-negative")
    if progress.ndim != 2 or progress.shape[1] != 5:
        raise ValueError("progress_m must have shape (frames, 5)")
    frame_count = progress.shape[0]
    if points.shape != (frame_count, 5, 3):
        raise ValueError("kinematic_points_m must have shape (frames, 5, 3)")
    for name, values in (
        ("frame_target_distance_m", target),
        ("marked_bridge_mask", marked),
        ("axial_distance_m", axial),
    ):
        if values.shape != (frame_count,):
            raise ValueError(f"{name} must have shape (frames,)")
    if not (
        np.all(np.isfinite(progress))
        and np.all(np.isfinite(points))
        and np.all(np.isfinite(target))
        and np.all(np.isfinite(axial))
    ):
        raise ValueError("low-motion audit inputs must be finite")

    raw: list[dict[str, object]] = []
    for start in range(0, frame_count - window_frames):
        end = start + window_frames
        if np.any(marked[start : end + 1]):
            continue
        route_delta = float(target[end] - target[start])
        if route_delta <= LOW_MOTION_NUMERICAL_TOLERANCE:
            continue
        required = forward_progress_ratio * route_delta
        tip_delta = progress[end, 1:] - progress[start, 1:]
        forward = tip_delta >= required - 1.0e-12
        if (
            int(np.count_nonzero(forward))
            >= LOW_MOTION_REQUIRED_FORWARD_FINGERS
        ):
            continue
        cartesian = np.linalg.norm(
            points[end, 1:] - points[start, 1:], axis=1
        )
        raw.append(
            {
                "start": start,
                "end": end,
                "forward_finger_count": int(np.count_nonzero(forward)),
                "forward_finger_required": (
                    LOW_MOTION_REQUIRED_FORWARD_FINGERS
                ),
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

    groups: list[list[dict[str, object]]] = [[raw[0]]]
    for item in raw[1:]:
        if int(item["start"]) <= int(groups[-1][-1]["end"]) + 1:
            groups[-1].append(item)
        else:
            groups.append([item])

    summaries: list[dict[str, object]] = []
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
                "first_window": group[0],
                "worst_window": worst,
            }
        )
    return summaries


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


def bounded_incremental_arc_targets(
    *,
    current_arc_m: np.ndarray,
    desired_arc_m: np.ndarray,
    direction: float,
    interval_m: float,
) -> np.ndarray:
    """Advance locally without overshooting a possibly rephased hard target."""

    if not np.isfinite(direction) or direction not in (-1.0, 1.0):
        raise ValueError("direction must be -1 or +1")
    if not np.isfinite(interval_m) or interval_m < 0.0:
        raise ValueError("interval_m must be finite and non-negative")
    current = np.asarray(current_arc_m, dtype=np.float64)
    desired = np.asarray(desired_arc_m, dtype=np.float64)
    if current.shape != desired.shape:
        raise ValueError("current_arc_m and desired_arc_m must have equal shape")
    if not np.all(np.isfinite(current)) or not np.all(np.isfinite(desired)):
        raise ValueError("Arc targets must be finite")
    remaining_forward = np.maximum(direction * (desired - current), 0.0)
    advance = np.minimum(remaining_forward, float(interval_m))
    return current + direction * advance


def progress_aware_arc_targets(
    *,
    current_arc_m: np.ndarray,
    desired_arc_m: np.ndarray,
    direction: float,
    nominal_advance_m: float,
    hard_progress_limit_m: float,
    interior_guard_m: float,
) -> np.ndarray:
    """Advance into a strict progress band without moving backwards."""

    if not np.isfinite(direction) or direction not in (-1.0, 1.0):
        raise ValueError("direction must be -1 or +1")
    scalars = np.asarray(
        [nominal_advance_m, hard_progress_limit_m, interior_guard_m],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(scalars)) or np.any(scalars < 0.0):
        raise ValueError("progress target distances must be finite and non-negative")
    if interior_guard_m > hard_progress_limit_m + 1.0e-15:
        raise ValueError("interior_guard_m cannot exceed hard_progress_limit_m")
    current = np.asarray(current_arc_m, dtype=np.float64)
    desired = np.asarray(desired_arc_m, dtype=np.float64)
    if current.shape != desired.shape:
        raise ValueError("current_arc_m and desired_arc_m must have equal shape")
    if current.ndim != 1 or current.size == 0:
        raise ValueError("arc targets must be non-empty vectors")
    if not np.all(np.isfinite(current)) or not np.all(np.isfinite(desired)):
        raise ValueError("arc targets must be finite")

    signed_current = float(direction) * current
    signed_desired = float(direction) * desired
    signed_nominal = signed_current + np.minimum(
        np.maximum(signed_desired - signed_current, 0.0),
        float(nominal_advance_m),
    )
    inner_limit = max(
        float(hard_progress_limit_m) - float(interior_guard_m),
        0.0,
    )
    signed_target = np.maximum(
        signed_nominal,
        signed_desired - inner_limit,
    )
    signed_target = np.minimum(signed_target, signed_desired + inner_limit)
    # Never request backwards meridian motion. The unchanged hard audit will
    # reject an already-ahead state if no forward-only feasible state exists.
    signed_target = np.maximum(signed_target, signed_current)
    return float(direction) * signed_target


def strict_suffix_task_hinge_residual(
    *,
    progress_error_m: np.ndarray,
    progress_limit_m: float,
    normal_error_m: np.ndarray,
    normal_tolerance_m: np.ndarray,
    tangent_error_m: np.ndarray,
    tangent_tolerance_m: np.ndarray,
    monotonic_error_m: np.ndarray,
    monotonic_tolerance_m: float,
    tip_motion_m: np.ndarray,
    minimum_tip_motion_m: float,
    interior_guard_m: float,
    weight: float,
    minimum_progressing_fingers: int = MOVING_BRIDGE_FORWARD_FINGER_COUNT,
) -> np.ndarray:
    """Shape a suffix toward the unchanged task gates with an inner margin.

    The four task arrays are kept separate so a least-squares solve cannot
    trade a large violation on one finger against a small aggregate cost.  A
    motion residual is emitted only for the best ``minimum_progressing_fingers``
    deficits, matching the planner's 3-of-4 moving contract while permitting a
    different finger to pause briefly at each node.
    """

    arrays = {
        "progress_error_m": progress_error_m,
        "normal_error_m": normal_error_m,
        "normal_tolerance_m": normal_tolerance_m,
        "tangent_error_m": tangent_error_m,
        "tangent_tolerance_m": tangent_tolerance_m,
        "monotonic_error_m": monotonic_error_m,
        "tip_motion_m": tip_motion_m,
    }
    normalized: dict[str, np.ndarray] = {}
    for name, value in arrays.items():
        array = np.asarray(value, dtype=np.float64)
        if array.shape != (4,):
            raise ValueError(f"{name} must have shape (4,)")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite")
        normalized[name] = array

    scalars = np.asarray(
        (
            progress_limit_m,
            monotonic_tolerance_m,
            minimum_tip_motion_m,
            interior_guard_m,
            weight,
        ),
        dtype=np.float64,
    )
    if not np.all(np.isfinite(scalars)) or np.any(scalars < 0.0):
        raise ValueError(
            "strict suffix limits, guard, and weight must be non-negative"
        )
    if minimum_progressing_fingers < 0 or minimum_progressing_fingers > 4:
        raise ValueError("minimum_progressing_fingers must lie in [0, 4]")

    guard = float(interior_guard_m)
    progress_inner = max(float(progress_limit_m) - guard, 0.0)
    normal_inner = np.maximum(normalized["normal_tolerance_m"] - guard, 0.0)
    tangent_inner = np.maximum(normalized["tangent_tolerance_m"] - guard, 0.0)
    monotonic_inner = max(float(monotonic_tolerance_m) - guard, 0.0)
    residual_rows = [
        np.maximum(normalized["progress_error_m"] - progress_inner, 0.0),
        np.maximum(normalized["normal_error_m"] - normal_inner, 0.0),
        np.maximum(normalized["tangent_error_m"] - tangent_inner, 0.0),
        np.maximum(normalized["monotonic_error_m"] - monotonic_inner, 0.0),
    ]
    if minimum_progressing_fingers:
        motion_deficit = np.maximum(
            float(minimum_tip_motion_m) - normalized["tip_motion_m"],
            0.0,
        )
        residual_rows.append(
            np.sort(motion_deficit)[:minimum_progressing_fingers]
        )
    return float(weight) * np.concatenate(residual_rows)


def terminal_contact_start_distance(
    axial_travel_m: float,
    frame_count: int,
    final_contact_frames: int,
) -> float:
    """Return the first published route sample requiring terminal 4/4."""

    if not np.isfinite(axial_travel_m) or axial_travel_m <= 0.0:
        raise ValueError("axial_travel_m must be finite and positive")
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if final_contact_frames <= 0 or final_contact_frames > frame_count:
        raise ValueError("final_contact_frames must lie in [1, frame_count]")
    first_one_based_frame = frame_count - final_contact_frames + 1
    return (
        float(first_one_based_frame)
        * float(axial_travel_m)
        / float(frame_count)
    )


def terminal_contact_sample_mask(
    route_distance_m: np.ndarray,
    *,
    terminal_start_m: float,
) -> np.ndarray:
    """Mark published samples at or beyond the exact terminal boundary."""

    route = np.asarray(route_distance_m, dtype=np.float64)
    if route.ndim != 1 or route.size == 0:
        raise ValueError("route_distance_m must be a non-empty vector")
    if not np.all(np.isfinite(route)) or not np.isfinite(terminal_start_m):
        raise ValueError("terminal contact inputs must be finite")
    return route >= float(terminal_start_m) - 1.0e-12


def build_receding_horizon_distances(
    *,
    first_distance_m: float,
    nominal_step_m: float,
    horizon_nodes: int,
    route_end_m: float,
    terminal_start_m: float | None,
) -> np.ndarray:
    """Build an increasing H-node suffix grid with exact sentinels."""

    scalars = np.asarray(
        [first_distance_m, nominal_step_m, route_end_m],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(scalars)):
        raise ValueError("horizon distances must be finite")
    if first_distance_m < 0.0 or nominal_step_m <= 0.0:
        raise ValueError("first distance must be non-negative and step positive")
    if route_end_m < first_distance_m - 1.0e-12:
        raise ValueError("route_end_m cannot precede first_distance_m")
    if horizon_nodes <= 0:
        raise ValueError("horizon_nodes must be positive")
    if terminal_start_m is not None and not np.isfinite(terminal_start_m):
        raise ValueError("terminal_start_m must be finite when provided")

    uniform_end = min(
        float(first_distance_m)
        + float(nominal_step_m) * float(horizon_nodes - 1),
        float(route_end_m),
    )
    witness_end = uniform_end
    # Once a receding window enters the terminal-tail lookahead span it must
    # witness the exact first 4/4 sample.  A short nominal H-step reach alone
    # can otherwise stop a few micrometres before the terminal contract and
    # certify a suffix which cannot be continued.
    terminal_tail_span = 0.0
    if terminal_start_m is not None:
        terminal_tail_span = max(
            float(route_end_m) - float(terminal_start_m),
            0.0,
        )
    lookahead_reach = float(first_distance_m) + max(
        float(nominal_step_m) * float(horizon_nodes + 1),
        terminal_tail_span,
    )
    for sentinel in (terminal_start_m, route_end_m):
        if sentinel is None:
            continue
        sentinel_value = float(sentinel)
        if (
            sentinel_value > first_distance_m + 1.0e-12
            and sentinel_value <= lookahead_reach + 1.0e-12
        ):
            witness_end = max(witness_end, sentinel_value)
    witness_end = min(witness_end, float(route_end_m))
    if horizon_nodes == 1:
        return np.asarray([float(first_distance_m)], dtype=np.float64)
    distances = np.linspace(
        float(first_distance_m),
        witness_end,
        horizon_nodes,
        dtype=np.float64,
    )
    if np.any(np.diff(distances) <= 0.0):
        distances = np.unique(distances)
    return distances


def damped_task_nullspace_directions(
    task_jacobian: np.ndarray,
    joint_directions: np.ndarray,
    *,
    damping: float = 1.0e-6,
) -> np.ndarray:
    """Project generic joint-space directions into a damped task nullspace.

    Rows of ``joint_directions`` are projected with
    ``I - J.T @ inv(J @ J.T + damping * I) @ J`` and normalized by their
    infinity norm.  The helper deliberately knows nothing about joint names;
    callers may feed every near-limit inward direction without hard-coding a
    particular finger or distal joint.
    """

    jacobian = np.asarray(task_jacobian, dtype=np.float64)
    directions = np.asarray(joint_directions, dtype=np.float64)
    if jacobian.ndim != 2 or jacobian.shape[1] == 0:
        raise ValueError("task_jacobian must be a non-empty matrix")
    if directions.ndim == 1:
        directions = directions[None, :]
    if directions.ndim != 2 or directions.shape[1] != jacobian.shape[1]:
        raise ValueError("joint_directions must have one column per joint")
    if not np.all(np.isfinite(jacobian)) or not np.all(np.isfinite(directions)):
        raise ValueError("nullspace projection inputs must be finite")
    if not np.isfinite(damping) or damping <= 0.0:
        raise ValueError("damping must be finite and positive")

    gram = jacobian @ jacobian.T
    regularized = gram + float(damping) * np.eye(
        gram.shape[0], dtype=np.float64
    )
    task_components = np.linalg.solve(
        regularized,
        jacobian @ directions.T,
    )
    projected = directions.T - jacobian.T @ task_components
    projected = projected.T
    infinity_norm = np.max(np.abs(projected), axis=1)
    valid = infinity_norm > 1.0e-12
    projected[valid] /= infinity_norm[valid, None]
    projected[~valid] = 0.0
    return projected


def smoothstep_joint_interpolation(
    knot_distance_m: np.ndarray,
    knot_q_rad: np.ndarray,
    sample_distance_m: np.ndarray,
) -> np.ndarray:
    """Interpolate joint knots exactly like the adaptive plan publisher."""

    distance = np.asarray(knot_distance_m, dtype=np.float64)
    q = np.asarray(knot_q_rad, dtype=np.float64)
    sample = np.asarray(sample_distance_m, dtype=np.float64)
    if distance.ndim != 1 or distance.size < 2:
        raise ValueError("knot_distance_m must contain at least two knots")
    if q.ndim != 2 or q.shape[0] != distance.size:
        raise ValueError("knot_q_rad must have one row per distance knot")
    if sample.ndim != 1:
        raise ValueError("sample_distance_m must be a vector")
    if not all(np.all(np.isfinite(v)) for v in (distance, q, sample)):
        raise ValueError("interpolation inputs must be finite")
    if np.any(np.diff(distance) <= 0.0):
        raise ValueError("knot distances must be strictly increasing")
    if np.any(sample < distance[0] - 1.0e-12) or np.any(
        sample > distance[-1] + 1.0e-12
    ):
        raise ValueError("sample distances must lie inside the knot range")

    out = np.empty((sample.size, q.shape[1]), dtype=np.float64)
    for index, value in enumerate(sample):
        left = int(np.searchsorted(distance, value, side="right") - 1)
        left = min(max(left, 0), distance.size - 2)
        interval = distance[left + 1] - distance[left]
        blend = float(np.clip((value - distance[left]) / interval, 0.0, 1.0))
        blend = blend * blend * (3.0 - 2.0 * blend)
        out[index] = (1.0 - blend) * q[left] + blend * q[left + 1]
    return out


def horizon_joint_margin_residual(
    q_rad: np.ndarray,
    lower_rad: np.ndarray,
    upper_rad: np.ndarray,
    *,
    minimum_margin_rad: float,
    weight: float,
) -> np.ndarray:
    """Return per-node, per-joint one-sided interior-margin residuals."""

    q = np.asarray(q_rad, dtype=np.float64)
    lower = np.asarray(lower_rad, dtype=np.float64)
    upper = np.asarray(upper_rad, dtype=np.float64)
    if q.ndim != 2 or lower.shape != (q.shape[1],) or upper.shape != lower.shape:
        raise ValueError("joint margin shapes are inconsistent")
    if not all(np.all(np.isfinite(v)) for v in (q, lower, upper)):
        raise ValueError("joint margin inputs must be finite")
    if np.any(lower > upper):
        raise ValueError("lower_rad cannot exceed upper_rad")
    if (
        not np.isfinite(minimum_margin_rad)
        or minimum_margin_rad < 0.0
        or not np.isfinite(weight)
        or weight < 0.0
    ):
        raise ValueError("joint margin and weight must be finite and non-negative")
    margin = np.minimum(q - lower[None, :], upper[None, :] - q)
    return float(weight) * np.maximum(float(minimum_margin_rad) - margin, 0.0).ravel()


def horizon_joint_step_residual(
    q_rad: np.ndarray,
    anchor_q_rad: np.ndarray,
    *,
    maximum_step_rad: float,
    interior_guard_rad: float,
    weight: float,
) -> np.ndarray:
    """Return one-sided residuals for every anchor/node transition."""

    q = np.asarray(q_rad, dtype=np.float64)
    anchor = np.asarray(anchor_q_rad, dtype=np.float64)
    if q.ndim != 2 or anchor.shape != (q.shape[1],):
        raise ValueError("joint step shapes are inconsistent")
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(anchor)):
        raise ValueError("joint step inputs must be finite")
    scalars = np.asarray(
        [maximum_step_rad, interior_guard_rad, weight], dtype=np.float64
    )
    if not np.all(np.isfinite(scalars)) or np.any(scalars < 0.0):
        raise ValueError("joint step parameters must be finite and non-negative")
    if interior_guard_rad > maximum_step_rad + 1.0e-15:
        raise ValueError("joint-step guard cannot exceed maximum step")
    transitions = np.vstack((q[0] - anchor, np.diff(q, axis=0)))
    inner_limit = float(maximum_step_rad) - float(interior_guard_rad)
    return float(weight) * np.maximum(np.abs(transitions) - inner_limit, 0.0).ravel()


def bounded_moving_bridge_trust_radius(
    configured_radius_rad: float,
    maximum_plan_step_rad: float,
) -> float:
    """Cap a local bridge solve by the plan's formal joint-step limit."""

    if (
        not np.isfinite(configured_radius_rad)
        or not np.isfinite(maximum_plan_step_rad)
        or configured_radius_rad <= 0.0
        or maximum_plan_step_rad <= 0.0
    ):
        raise ValueError("Bridge and plan joint radii must be positive")
    return min(float(configured_radius_rad), float(maximum_plan_step_rad))


def moving_bridge_local_residual(
    *,
    arc_m: np.ndarray,
    target_arc_m: np.ndarray,
    standoff_m: np.ndarray,
    anchor_standoff_m: np.ndarray,
    azimuth_rad: np.ndarray,
    anchor_azimuth_rad: np.ndarray,
    q_rad: np.ndarray,
    anchor_q_rad: np.ndarray,
    capsule_radius_m: float,
    task_weight: float,
    joint_regularization: float = 1.0e-4,
) -> np.ndarray:
    """Return the anchor-local four-tip bridge task and weak joint prior.

    The first twelve entries are exactly four meridian, four normal, and four
    wrapped circumferential errors.  No palm, global route, or posture target
    is permitted in this local branch-preserving solve.
    """

    if not np.isfinite(capsule_radius_m) or capsule_radius_m <= 0.0:
        raise ValueError("capsule_radius_m must be positive")
    if not np.isfinite(task_weight) or task_weight <= 0.0:
        raise ValueError("task_weight must be positive")
    if not np.isfinite(joint_regularization) or joint_regularization < 0.0:
        raise ValueError("joint_regularization cannot be negative")

    tip_values = {
        "arc_m": arc_m,
        "target_arc_m": target_arc_m,
        "standoff_m": standoff_m,
        "anchor_standoff_m": anchor_standoff_m,
        "azimuth_rad": azimuth_rad,
        "anchor_azimuth_rad": anchor_azimuth_rad,
    }
    normalized: dict[str, np.ndarray] = {}
    for name, value in tip_values.items():
        array = np.asarray(value, dtype=np.float64)
        if array.shape != (4,):
            raise ValueError(f"{name} must have shape (4,)")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite")
        normalized[name] = array

    q = np.asarray(q_rad, dtype=np.float64).reshape(-1)
    anchor_q = np.asarray(anchor_q_rad, dtype=np.float64).reshape(-1)
    if q.shape != anchor_q.shape:
        raise ValueError("q_rad and anchor_q_rad must have equal shape")
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(anchor_q)):
        raise ValueError("Joint vectors must be finite")
    wrapped_azimuth_error = (
        normalized["azimuth_rad"]
        - normalized["anchor_azimuth_rad"]
        + np.pi
    ) % (2.0 * np.pi) - np.pi
    return np.concatenate(
        (
            task_weight
            * (normalized["arc_m"] - normalized["target_arc_m"]),
            task_weight
            * (
                normalized["standoff_m"]
                - normalized["anchor_standoff_m"]
            ),
            task_weight * capsule_radius_m * wrapped_azimuth_error,
            joint_regularization * (q - anchor_q),
        )
    )


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
    tip_clearance_m: float,
    tip_clearance_limit_m: float,
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
    self_collision_sample_occurrence_count: int | None = None,
    minimum_protected_self_clearance_m: float = np.inf,
    protected_self_clearance_target_m: float = 0.0,
    minimum_protected_self_pair_name: str = "",
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
    self_collision_occurrences = (
        int(self_collision_count)
        if self_collision_sample_occurrence_count is None
        else int(self_collision_sample_occurrence_count)
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
        "tip_clearance_limit_m": tip_clearance_limit_m,
        "self_collision_count": self_collision_count,
        "self_collision_unique_pair_count": self_collision_count,
        "self_collision_sample_occurrence_count": self_collision_occurrences,
        "self_collision_limit": 0,
        "self_collision_margin": -self_collision_count,
        "protected_self_clearance_m": minimum_protected_self_clearance_m,
        "protected_self_clearance_target_m": (
            protected_self_clearance_target_m
        ),
        "protected_self_clearance_margin_m": (
            minimum_protected_self_clearance_m
            - protected_self_clearance_target_m
        ),
        "protected_self_nearest_pair": minimum_protected_self_pair_name,
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
                "tip_clearance_m": tip_clearance_m,
                "tip_clearance_margin_m": (
                    tip_clearance_m - tip_clearance_limit_m
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
    tip_clearance_m: float,
    tip_clearance_limit_m: float,
    tip_nearest_geometry: str,
    self_collision_count: int,
    pad_alignment: float,
    pad_alignment_limit: float,
    joint_min_margin_rad: float,
    solver_cost: float,
    solver_nfev: int,
    self_collision_sample_occurrence_count: int | None = None,
    minimum_protected_self_clearance_m: float = np.inf,
    protected_self_clearance_target_m: float = 0.0,
    minimum_protected_self_pair_name: str = "",
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
    self_collision_occurrences = (
        int(self_collision_count)
        if self_collision_sample_occurrence_count is None
        else int(self_collision_sample_occurrence_count)
    )
    collision_ok = bool(
        collision_mode != "full_robot"
        or (
            arm_clearance_m >= arm_clearance_limit_m
            and hand_clearance_m >= hand_clearance_limit_m
            and tip_clearance_m >= tip_clearance_limit_m
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
        "tip_clearance_m": tip_clearance_m,
        "tip_clearance_limit_m": tip_clearance_limit_m,
        "tip_clearance_margin_m": tip_clearance_m - tip_clearance_limit_m,
        "tip_nearest_geometry": tip_nearest_geometry,
        "self_collision_count": self_collision_count,
        "self_collision_unique_pair_count": self_collision_count,
        "self_collision_sample_occurrence_count": self_collision_occurrences,
        "self_collision_limit": 0,
        "self_collision_margin": -self_collision_count,
        "protected_self_clearance_m": minimum_protected_self_clearance_m,
        "protected_self_clearance_target_m": (
            protected_self_clearance_target_m
        ),
        "protected_self_clearance_margin_m": (
            minimum_protected_self_clearance_m
            - protected_self_clearance_target_m
        ),
        "protected_self_nearest_pair": minimum_protected_self_pair_name,
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


def save_npz_no_overwrite(
    path: Path,
    payload: Mapping[str, object],
    *,
    field_label: str = "NPZ",
) -> Path:
    """Save a pickle-free NPZ with exclusive creation and stable suffixes."""

    requested_output = Path(path)
    if requested_output.suffix.lower() != ".npz":
        requested_output = requested_output.with_suffix(".npz")
    requested_output.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for name, value in payload.items():
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise TypeError(
                f"{field_label} field {name!r} has object dtype and cannot "
                "be replayed with allow_pickle=False"
            )
        arrays[name] = array

    # Opening with ``xb`` is the no-overwrite guarantee.  A preceding
    # exists() check alone would race when deterministic runs share a path.
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
                np.savez_compressed(output_file, **arrays)
            return output
        except FileExistsError:
            suffix_index += 1
        except Exception:
            if created_output:
                output.unlink(missing_ok=True)
            raise


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
    return save_npz_no_overwrite(
        path,
        payload,
        field_label="Failure-prefix",
    )
