"""Continuous first-edge optimizer with a deterministic kinematic backend.

The optimizer is backend-agnostic at its boundary.  The included linearized
hand model is an explicit module-validation fixture; FR3 nonlinear IK and exact
MuJoCo collision remain responsibilities of later physical integration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from time import perf_counter_ns
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from Module.module_1_oracle_surface_model import OracleSurfaceModel
from Module.module_6_prefix_executor import (
  PlannedPrefix,
  PrefixSample,
  PrefixSource,
  TransactionType,
)
from Module.module_7_contact_mode_graph import (
  ContactMode,
  ContactModeGraph,
  ContactPrimitive,
  PrimitiveKind,
)


NUM_FINGERS = 4


def _array(
  value: ArrayLike,
  name: str,
  *,
  shape: tuple[int, ...] | None = None,
  ndim: int | None = None,
) -> NDArray[np.float64]:
  array = np.asarray(value, dtype=np.float64)
  if shape is not None and array.shape != shape:
    raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
  if ndim is not None and array.ndim != ndim:
    raise ValueError(f"{name} must have ndim={ndim}, got {array.ndim}")
  if not np.all(np.isfinite(array)):
    raise ValueError(f"{name} must contain finite values")
  result = np.array(array, copy=True)
  result.setflags(write=False)
  return result


@dataclass(frozen=True, slots=True)
class PlannerState:
  joint_positions_rad: ArrayLike
  wrist_position_m: ArrayLike
  fingertip_positions_m: ArrayLike
  actual_contact_set: frozenset[int] | set[int] | tuple[int, ...]
  surface_model_version: str
  contact_set_authority: str = "MEASURED"

  def __post_init__(self) -> None:
    q = _array(self.joint_positions_rad, "joint_positions_rad", ndim=1)
    wrist = _array(self.wrist_position_m, "wrist_position_m", shape=(3,))
    tips = _array(
      self.fingertip_positions_m,
      "fingertip_positions_m",
      shape=(NUM_FINGERS, 3),
    )
    contacts = frozenset(int(finger) for finger in self.actual_contact_set)
    if not contacts or any(finger < 1 or finger > NUM_FINGERS for finger in contacts):
      raise ValueError("actual_contact_set must be a nonempty valid mode")
    if not self.surface_model_version:
      raise ValueError("surface_model_version must be nonempty")
    if self.contact_set_authority not in {"MEASURED", "PREDICTED"}:
      raise ValueError("contact_set_authority must be MEASURED or PREDICTED")
    object.__setattr__(self, "joint_positions_rad", q)
    object.__setattr__(self, "wrist_position_m", wrist)
    object.__setattr__(self, "fingertip_positions_m", tips)
    object.__setattr__(self, "actual_contact_set", contacts)

  @property
  def mode(self) -> ContactMode:
    return ContactMode(self.actual_contact_set)

  @property
  def contact_set_is_measured(self) -> bool:
    return self.contact_set_authority == "MEASURED"


@dataclass(frozen=True, slots=True)
class LinearizedHandKinematics:
  """Local tip Jacobians around a fixed reference hand state."""

  reference_q_rad: ArrayLike
  joint_lower_rad: ArrayLike
  joint_upper_rad: ArrayLike
  reference_wrist_position_m: ArrayLike
  reference_fingertip_positions_m: ArrayLike
  fingertip_jacobians_m_per_rad: ArrayLike
  minimum_tip_separation_m: float = 0.012

  def __post_init__(self) -> None:
    q_ref = _array(self.reference_q_rad, "reference_q_rad", ndim=1)
    lower = _array(self.joint_lower_rad, "joint_lower_rad", shape=q_ref.shape)
    upper = _array(self.joint_upper_rad, "joint_upper_rad", shape=q_ref.shape)
    if np.any(lower >= upper) or np.any(q_ref < lower) or np.any(q_ref > upper):
      raise ValueError("invalid joint limits or reference state")
    wrist = _array(
      self.reference_wrist_position_m,
      "reference_wrist_position_m",
      shape=(3,),
    )
    tips = _array(
      self.reference_fingertip_positions_m,
      "reference_fingertip_positions_m",
      shape=(NUM_FINGERS, 3),
    )
    jacobians = _array(
      self.fingertip_jacobians_m_per_rad,
      "fingertip_jacobians_m_per_rad",
      shape=(NUM_FINGERS, 3, len(q_ref)),
    )
    if self.minimum_tip_separation_m <= 0.0:
      raise ValueError("minimum_tip_separation_m must be positive")
    object.__setattr__(self, "reference_q_rad", q_ref)
    object.__setattr__(self, "joint_lower_rad", lower)
    object.__setattr__(self, "joint_upper_rad", upper)
    object.__setattr__(self, "reference_wrist_position_m", wrist)
    object.__setattr__(self, "reference_fingertip_positions_m", tips)
    object.__setattr__(self, "fingertip_jacobians_m_per_rad", jacobians)

  @classmethod
  def canonical(
    cls,
    fingertip_positions_m: ArrayLike,
    *,
    wrist_position_m: ArrayLike = (0.0, 0.0, 0.12),
    joint_span_rad: float = 2.0,
    local_gain_m_per_rad: float = 0.04,
  ) -> LinearizedHandKinematics:
    tips = _array(
      fingertip_positions_m,
      "fingertip_positions_m",
      shape=(NUM_FINGERS, 3),
    )
    if joint_span_rad <= 0.0 or local_gain_m_per_rad <= 0.0:
      raise ValueError("canonical kinematic scales must be positive")
    dimension = NUM_FINGERS * 3
    jacobians = np.zeros((NUM_FINGERS, 3, dimension), dtype=np.float64)
    for finger in range(NUM_FINGERS):
      start = 3 * finger
      jacobians[finger, :, start : start + 3] = (
        local_gain_m_per_rad * np.eye(3)
      )
    return cls(
      reference_q_rad=np.zeros(dimension),
      joint_lower_rad=-joint_span_rad * np.ones(dimension),
      joint_upper_rad=joint_span_rad * np.ones(dimension),
      reference_wrist_position_m=wrist_position_m,
      reference_fingertip_positions_m=tips,
      fingertip_jacobians_m_per_rad=jacobians,
    )

  @property
  def joint_dimension(self) -> int:
    return len(self.reference_q_rad)

  def forward(
    self,
    q_rad: ArrayLike,
    wrist_position_m: ArrayLike,
  ) -> NDArray[np.float64]:
    q = _array(q_rad, "q_rad", shape=self.reference_q_rad.shape)
    wrist = _array(wrist_position_m, "wrist_position_m", shape=(3,))
    delta_q = q - self.reference_q_rad
    tip_delta = np.einsum(
      "fij,j->fi",
      self.fingertip_jacobians_m_per_rad,
      delta_q,
    )
    return (
      self.reference_fingertip_positions_m
      + (wrist - self.reference_wrist_position_m)[None, :]
      + tip_delta
    )

  def solve(
    self,
    desired_fingertip_positions_m: ArrayLike,
    wrist_position_m: ArrayLike,
    seed_q_rad: ArrayLike,
    controlled_fingers: tuple[int, ...],
    *,
    damping: float,
    iterations: int,
  ) -> tuple[NDArray[np.float64], float]:
    desired = _array(
      desired_fingertip_positions_m,
      "desired_fingertip_positions_m",
      shape=(NUM_FINGERS, 3),
    )
    wrist = _array(wrist_position_m, "wrist_position_m", shape=(3,))
    q = np.array(_array(seed_q_rad, "seed_q_rad", shape=self.reference_q_rad.shape), copy=True)
    fingers = tuple(sorted(set(int(finger) for finger in controlled_fingers)))
    if not fingers:
      return q, 0.0
    indices = np.asarray([finger - 1 for finger in fingers], dtype=np.int64)
    jacobian = self.fingertip_jacobians_m_per_rad[indices].reshape(-1, self.joint_dimension)
    regularized = jacobian @ jacobian.T + damping**2 * np.eye(len(indices) * 3)
    for _ in range(iterations):
      current = self.forward(q, wrist)
      error = (desired[indices] - current[indices]).reshape(-1)
      if float(np.max(np.linalg.norm(error.reshape(-1, 3), axis=1))) <= 1e-7:
        break
      delta = jacobian.T @ np.linalg.solve(regularized, error)
      q = np.clip(q + delta, self.joint_lower_rad, self.joint_upper_rad)
    current = self.forward(q, wrist)
    residual = float(
      np.max(np.linalg.norm(desired[indices] - current[indices], axis=1))
    )
    return q, residual

  def joint_margin(self, q_rad: ArrayLike) -> float:
    q = _array(q_rad, "q_rad", shape=self.reference_q_rad.shape)
    return float(np.min(np.minimum(q - self.joint_lower_rad, self.joint_upper_rad - q)))

  def self_collision_clearance(self, fingertip_positions_m: ArrayLike) -> float:
    tips = _array(
      fingertip_positions_m,
      "fingertip_positions_m",
      shape=(NUM_FINGERS, 3),
    )
    distances = [
      float(np.linalg.norm(tips[left] - tips[right]))
      for left in range(NUM_FINGERS)
      for right in range(left + 1, NUM_FINGERS)
    ]
    return min(distances) - self.minimum_tip_separation_m


class OptimizationStatus(str, Enum):
  SUCCESS = "SUCCESS"
  MAKE_PROGRESS = "MAKE_PROGRESS"
  INFEASIBLE = "INFEASIBLE"


@dataclass(frozen=True, slots=True)
class OptimizationConfig:
  waypoint_count: int = 9
  max_commit_displacement_m: float = 0.015
  target_tolerance_m: float = 0.00075
  anchor_tolerance_m: float = 0.00075
  minimum_collision_clearance_m: float = 0.0
  minimum_joint_margin_rad: float = 0.0
  reach_radius_m: float = 0.085
  release_distance_m: float = 0.006
  free_surface_clearance_m: float = 0.004
  nominal_speed_m_s: float = 0.05
  damping: float = 1e-4
  ik_iterations: int = 8

  def __post_init__(self) -> None:
    if self.waypoint_count < 2:
      raise ValueError("waypoint_count must be >=2")
    for name in (
      "max_commit_displacement_m",
      "target_tolerance_m",
      "anchor_tolerance_m",
      "reach_radius_m",
      "release_distance_m",
      "free_surface_clearance_m",
      "nominal_speed_m_s",
      "damping",
    ):
      if not np.isfinite(float(getattr(self, name))) or float(getattr(self, name)) <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    if self.minimum_collision_clearance_m < 0.0 or self.minimum_joint_margin_rad < 0.0:
      raise ValueError("minimum margins must be non-negative")
    if self.ik_iterations < 1:
      raise ValueError("ik_iterations must be positive")


@dataclass(frozen=True, slots=True)
class OptimizationRequest:
  state: PlannerState
  primitive: ContactPrimitive
  target_position_m: ArrayLike | None = None
  target_wrist_position_m: ArrayLike | None = None
  prefix_id: str = "edge"
  progress_gain_m: float = 0.0
  metadata: Mapping[str, float] = field(default_factory=dict)

  def __post_init__(self) -> None:
    if not self.prefix_id:
      raise ValueError("prefix_id must be nonempty")
    target = None
    if self.target_position_m is not None:
      target = _array(self.target_position_m, "target_position_m", shape=(3,))
    wrist = None
    if self.target_wrist_position_m is not None:
      wrist = _array(
        self.target_wrist_position_m,
        "target_wrist_position_m",
        shape=(3,),
      )
    if self.primitive.kind is PrimitiveKind.WRIST_ADJUST:
      if wrist is None or target is not None:
        raise ValueError("WRIST_ADJUST requires only target_wrist_position_m")
    elif target is None or wrist is not None:
      raise ValueError("finger primitives require only target_position_m")
    if not np.isfinite(self.progress_gain_m):
      raise ValueError("progress_gain_m must be finite")
    metadata = {str(name): float(value) for name, value in self.metadata.items()}
    if any(not np.isfinite(value) for value in metadata.values()):
      raise ValueError("metadata must be finite")
    object.__setattr__(self, "target_position_m", target)
    object.__setattr__(self, "target_wrist_position_m", wrist)
    object.__setattr__(self, "metadata", metadata)


@dataclass(frozen=True, slots=True)
class OptimizationResult:
  status: OptimizationStatus
  prefix: PlannedPrefix | None
  reasons: tuple[str, ...]
  solve_time_s: float
  final_target_error_m: float
  minimum_clearance_m: float
  minimum_joint_margin_rad: float
  anchor_margin_m: float
  reach_margin_m: float
  trust_region_use: float
  requested_progress_m: float
  achieved_progress_m: float

  @property
  def feasible(self) -> bool:
    return self.status in {OptimizationStatus.SUCCESS, OptimizationStatus.MAKE_PROGRESS}


class ContinuousOptimizer:
  """Construct one smooth edge while preserving measured root anchors."""

  def __init__(
    self,
    graph: ContactModeGraph,
    surface_model: OracleSurfaceModel,
    kinematics: LinearizedHandKinematics,
    config: OptimizationConfig | None = None,
  ) -> None:
    self.graph = graph
    self.surface_model = surface_model
    self.kinematics = kinematics
    self.config = config or OptimizationConfig()

  def optimize(self, request: OptimizationRequest) -> OptimizationResult:
    started = perf_counter_ns()
    state = request.state
    primitive = request.primitive
    if state.surface_model_version != self.surface_model.version:
      return self._infeasible(started, "STALE_SURFACE_MODEL_VERSION")
    legality = self.graph.predict_legal(state.mode, primitive)
    if not legality.legal:
      return self._infeasible(started, legality.reason)
    if state.joint_positions_rad.shape != self.kinematics.reference_q_rad.shape:
      return self._infeasible(started, "JOINT_DIMENSION_MISMATCH")

    terminal_wrist = np.array(state.wrist_position_m, copy=True)
    desired_terminal_tips = np.array(state.fingertip_positions_m, copy=True)
    finger = primitive.finger_id
    target_requested: NDArray[np.float64]
    target_terminal: NDArray[np.float64]
    make_progress = False
    if primitive.kind is PrimitiveKind.WRIST_ADJUST:
      assert request.target_wrist_position_m is not None
      target_requested = np.array(request.target_wrist_position_m, copy=True)
      displacement = target_requested - state.wrist_position_m
      terminal_wrist = state.wrist_position_m + self._clip_vector(displacement)
      target_terminal = terminal_wrist
      selected = ()
      anchors = tuple(sorted(state.actual_contact_set))
      controlled = anchors
    else:
      assert finger is not None and request.target_position_m is not None
      selected = (finger,)
      anchors = tuple(sorted(state.actual_contact_set - {finger}))
      controlled = tuple(sorted(set(selected) | set(anchors)))
      physical_pad_center_target = bool(
        request.metadata.get("physical_pad_center_target", 0.0) > 0.5
      )
      if physical_pad_center_target:
        if primitive.kind not in {
          PrimitiveKind.SLIDE,
          PrimitiveKind.MAKE,
          PrimitiveKind.REPOSITION,
        }:
          return self._infeasible(started, "PAD_CENTER_TARGET_KIND_MISMATCH")
        # The caller has already converted a mesh surface point into the
        # physical collision-pad center using its oriented support radius and
        # primitive-specific preload/clearance.  Reprojecting that site center
        # onto the zero-distance surface would demand one extra pad radius of
        # impossible penetration.  The default point-fingertip semantics below
        # remain unchanged for every caller that does not opt in.
        target_requested = np.array(request.target_position_m, copy=True)
      else:
        requested_query = self.surface_model.query_surface(request.target_position_m)
        if primitive.kind in {PrimitiveKind.SLIDE, PrimitiveKind.MAKE}:
          target_requested = np.array(requested_query.point, copy=True)
        elif primitive.kind is PrimitiveKind.REPOSITION:
          signed = requested_query.signed_distance
          if signed < self.config.free_surface_clearance_m:
            target_requested = (
              requested_query.point
              + self.config.free_surface_clearance_m * requested_query.normal
            )
          else:
            target_requested = np.array(request.target_position_m, copy=True)
        elif primitive.kind is PrimitiveKind.BREAK:
          current_query = self.surface_model.query_surface(
            state.fingertip_positions_m[finger - 1]
          )
          target_requested = (
            state.fingertip_positions_m[finger - 1]
            + self.config.release_distance_m * current_query.normal
          )
        else:
          raise AssertionError("unhandled primitive")
      start = state.fingertip_positions_m[finger - 1]
      displacement = target_requested - start
      clipped = self._clip_vector(displacement)
      target_terminal = start + clipped
      desired_terminal_tips[finger - 1] = target_terminal
      make_progress = (
        primitive.kind is PrimitiveKind.MAKE
        and float(np.linalg.norm(target_terminal - target_requested))
        > self.config.target_tolerance_m
      )

    movement = (
      float(np.linalg.norm(terminal_wrist - state.wrist_position_m))
      if primitive.kind is PrimitiveKind.WRIST_ADJUST
      else float(
        np.linalg.norm(
          target_terminal - state.fingertip_positions_m[primitive.finger_id - 1]  # type: ignore[operator]
        )
      )
    )
    duration_s = max(0.05, movement / self.config.nominal_speed_m_s)
    times = np.linspace(0.0, duration_s, self.config.waypoint_count)
    q = np.array(state.joint_positions_rad, copy=True)
    samples: list[PrefixSample] = []
    minimum_clearance = float("inf")
    minimum_joint_margin = float("inf")
    maximum_anchor_error = 0.0
    maximum_solve_residual = 0.0
    for index, timestamp in enumerate(times):
      alpha_linear = index / (self.config.waypoint_count - 1)
      alpha = alpha_linear * alpha_linear * (3.0 - 2.0 * alpha_linear)
      wrist = (
        state.wrist_position_m
        + alpha * (terminal_wrist - state.wrist_position_m)
      )
      desired_tips = np.array(state.fingertip_positions_m, copy=True)
      if primitive.kind is not PrimitiveKind.WRIST_ADJUST:
        assert finger is not None
        desired_tips[finger - 1] = (
          state.fingertip_positions_m[finger - 1]
          + alpha
          * (target_terminal - state.fingertip_positions_m[finger - 1])
        )
      q, residual = self.kinematics.solve(
        desired_tips,
        wrist,
        q,
        controlled,
        damping=self.config.damping,
        iterations=self.config.ik_iterations,
      )
      tips = self.kinematics.forward(q, wrist)
      maximum_solve_residual = max(maximum_solve_residual, residual)
      if anchors:
        anchor_indices = np.asarray([anchor - 1 for anchor in anchors])
        maximum_anchor_error = max(
          maximum_anchor_error,
          float(
            np.max(
              np.linalg.norm(
                tips[anchor_indices] - state.fingertip_positions_m[anchor_indices],
                axis=1,
              )
            )
          ),
        )
      minimum_clearance = min(
        minimum_clearance,
        self.kinematics.self_collision_clearance(tips),
      )
      minimum_joint_margin = min(
        minimum_joint_margin,
        self.kinematics.joint_margin(q),
      )
      samples.append(PrefixSample(float(timestamp), wrist, tips, q))

    terminal_tips = samples[-1].fingertip_positions_m
    terminal_target_actual = (
      samples[-1].wrist_position_m
      if primitive.kind is PrimitiveKind.WRIST_ADJUST
      else terminal_tips[primitive.finger_id - 1]  # type: ignore[operator]
    )
    final_target_error = float(np.linalg.norm(terminal_target_actual - target_terminal))
    local_displacements = np.linalg.norm(
      terminal_tips
      - samples[-1].wrist_position_m[None, :]
      - (
        state.fingertip_positions_m - state.wrist_position_m[None, :]
      ),
      axis=1,
    )
    reach_margin = self.config.reach_radius_m - float(np.max(local_displacements))
    trust_use = movement / self.config.max_commit_displacement_m
    anchor_margin = self.config.anchor_tolerance_m - maximum_anchor_error
    reasons: list[str] = []
    if final_target_error > self.config.target_tolerance_m:
      reasons.append("TARGET_ERROR")
    if maximum_solve_residual > self.config.target_tolerance_m:
      reasons.append("IK_RESIDUAL")
    if anchor_margin < -1e-12:
      reasons.append("ANCHOR_PRESERVATION")
    if minimum_clearance < self.config.minimum_collision_clearance_m - 1e-12:
      reasons.append("SELF_COLLISION")
    if minimum_joint_margin < self.config.minimum_joint_margin_rad - 1e-12:
      reasons.append("JOINT_LIMIT")
    if reach_margin < -1e-12:
      reasons.append("UNREACHABLE")
    if trust_use > 1.0 + 1e-10:
      reasons.append("TRUST_REGION")

    status = (
      OptimizationStatus.INFEASIBLE
      if reasons
      else OptimizationStatus.MAKE_PROGRESS
      if make_progress
      else OptimizationStatus.SUCCESS
    )
    prefix: PlannedPrefix | None = None
    if status is not OptimizationStatus.INFEASIBLE:
      expected_terminal = self.graph.apply_predictive(state.mode, primitive).contacts
      topology_changes = primitive.topology_change_count
      if make_progress:
        expected_terminal = state.actual_contact_set
        topology_changes = 0
      prefix = PlannedPrefix(
        prefix_id=request.prefix_id,
        transaction_type=(
          TransactionType.WRIST_ADJUST
          if primitive.kind is PrimitiveKind.WRIST_ADJUST
          else TransactionType.FINGER_RECONFIGURE
        ),
        primitive_kind=primitive.kind.value,
        finger_id=primitive.finger_id,
        surface_model_version=state.surface_model_version,
        root_contact_set=state.actual_contact_set,
        expected_terminal_contact_set=expected_terminal,
        samples=tuple(samples),
        participating_fingers=selected,
        anchor_fingers=anchors,
        topology_change_count=topology_changes,
        source=PrefixSource.OPTIMIZER_COMMIT_CANDIDATE,
        metadata={
          "make_progress": make_progress,
          "requested_progress_m": float(request.progress_gain_m),
          "target_x_m": float(target_requested[0]),
          "target_y_m": float(target_requested[1]),
          "target_z_m": float(target_requested[2]),
          **request.metadata,
        },
      )
    solve_time_s = (perf_counter_ns() - started) * 1e-9
    achieved = min(float(request.progress_gain_m), movement)
    return OptimizationResult(
      status=status,
      prefix=prefix,
      reasons=tuple(reasons),
      solve_time_s=solve_time_s,
      final_target_error_m=final_target_error,
      minimum_clearance_m=minimum_clearance,
      minimum_joint_margin_rad=minimum_joint_margin,
      anchor_margin_m=anchor_margin,
      reach_margin_m=reach_margin,
      trust_region_use=trust_use,
      requested_progress_m=float(request.progress_gain_m),
      achieved_progress_m=achieved,
    )

  def _clip_vector(self, displacement: NDArray[np.float64]) -> NDArray[np.float64]:
    length = float(np.linalg.norm(displacement))
    if length <= self.config.max_commit_displacement_m:
      return np.array(displacement, copy=True)
    return displacement * (self.config.max_commit_displacement_m / length)

  def _infeasible(self, started_ns: int, reason: str) -> OptimizationResult:
    return OptimizationResult(
      status=OptimizationStatus.INFEASIBLE,
      prefix=None,
      reasons=(reason,),
      solve_time_s=(perf_counter_ns() - started_ns) * 1e-9,
      final_target_error_m=float("inf"),
      minimum_clearance_m=float("-inf"),
      minimum_joint_margin_rad=float("-inf"),
      anchor_margin_m=float("-inf"),
      reach_margin_m=float("-inf"),
      trust_region_use=float("inf"),
      requested_progress_m=0.0,
      achieved_progress_m=0.0,
    )
