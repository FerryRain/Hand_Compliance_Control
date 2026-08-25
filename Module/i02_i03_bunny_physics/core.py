"""Frozen planning primitives shared by the I02/I03 Bunny physics runner."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from Module.i01_bunny_physics.runner import (
  _link_clearance_function,
  _local_kinematics,
  _logical_q,
  _smoothstep,
  _world_mesh_oracle,
)
from Module.i01_bunny_physics.surface import BunnyHeightField
from Module.module_1_oracle_surface_model import OracleSurfaceModel
from Module.module_1_oracle_surface_model.geometry import (
  AnalyticShape,
  SurfaceProjection,
)
from Module.module_6_prefix_executor import PlannedPrefix
from Module.module_7_contact_mode_graph import (
  CommitContext,
  ContactModeGraph,
  ContactPrimitive,
  PrimitiveKind,
)
from Module.module_8_cheap_cert import (
  CheapCert,
  CheapCertConfig,
  CheapCertInput,
)
from Module.module_9_continuous_optimize import (
  ContinuousOptimizer,
  LinearizedHandKinematics,
  OptimizationConfig,
  OptimizationRequest,
  PlannerState,
)
from Module.module_10_exact_prefix_audit import (
  AuditConfig,
  AuditEnvironment,
  AuditRequest,
  ExactPrefixAuditor,
)
from Module.module_11_lazy_beam_search import (
  BeamSearchConfig,
  LazyBeamSearch,
  PlanningCandidate,
  SearchResult,
)
from Module.module_12_shadow_viability import (
  ShadowResult,
  ShadowViabilityEvaluator,
)


SURFACE_MODEL_VERSION = "oracle-bunny-pad-center.i02-i03.v1"
TRACE_SCHEMA_VERSION = "i02-i03-bunny-trace.v1"
EVALUATOR_VERSION = "i02-i03-bunny-evaluator.v1"
VALID_CELLS = ("i02_long", "i02_short", "i03_beam", "i03_shadow")
I02_FINGER = 3
I03_HANDOVER_FINGER = 4


@dataclass(frozen=True, slots=True)
class I02I03BunnyConfig:
  cell: str
  seed: int = 7
  duration_s: float = 20.0
  acquisition_s: float = 3.0
  dt_s: float = 0.002
  desired_force_n: float = 2.0
  contact_threshold_n: float = 0.20
  force_limit_n: float = 8.0
  mesh_residual_limit_m: float = 0.0025
  initial_arm_noise_rad: float = 0.0005
  initial_hand_noise_rad: float = 0.0010
  object_offset_x_m: float = 0.002
  object_offset_y_m: float = -0.005
  object_offset_z_m: float = -0.003
  reposition_total_m: float = 0.012
  short_reposition_segments: int = 3
  minimum_audit_joint_margin_rad: float = 0.010
  terminal_viability_joint_reserve_rad: float = 0.025
  visual_mesh_path: str | None = None

  def __post_init__(self) -> None:
    if self.cell not in VALID_CELLS:
      raise ValueError(f"cell must be one of {VALID_CELLS}")
    if not np.isclose(self.duration_s, 20.0):
      raise ValueError("I02/I03 formal duration is frozen at 20.0 s")
    if not np.isclose(self.acquisition_s, 3.0):
      raise ValueError("I02/I03 acquisition is frozen at 3.0 s")
    if not np.isclose(self.dt_s, 0.002):
      raise ValueError("I02/I03 timestep is frozen at 0.002 s")
    if self.short_reposition_segments != 3:
      raise ValueError("I02 SHORT is frozen to three reposition segments")
    for name in (
      "desired_force_n",
      "contact_threshold_n",
      "force_limit_n",
      "mesh_residual_limit_m",
      "reposition_total_m",
      "minimum_audit_joint_margin_rad",
      "terminal_viability_joint_reserve_rad",
    ):
      value = float(getattr(self, name))
      if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    if self.force_limit_n <= self.desired_force_n:
      raise ValueError("force_limit_n must exceed desired_force_n")
    if self.minimum_audit_joint_margin_rad >= self.terminal_viability_joint_reserve_rad:
      raise ValueError("terminal reserve must be stricter than the execution audit margin")

  @property
  def module_id(self) -> str:
    return "I02-PHY-BUNNY-v1" if self.cell.startswith("i02") else "I03-PHY-BUNNY-v1"


def planned_path_coordinate_m(timestamp_s: float) -> float:
  """Frozen 0 -> 10 -> 60 -> 10 mm path with a planning plateau."""

  t = float(timestamp_s)
  if t <= 3.0:
    return 0.0
  if t <= 4.0:
    return 0.010 * _smoothstep(t - 3.0)
  if t <= 7.0:
    return 0.010
  if t <= 12.0:
    return 0.010 + 0.050 * _smoothstep((t - 7.0) / 5.0)
  if t <= 17.0:
    return 0.060 - 0.050 * _smoothstep((t - 12.0) / 5.0)
  return 0.010


def planned_cumulative_distance_m(timestamp_s: float) -> float:
  """Cumulative length of the frozen path, capped at 110 mm."""

  t = float(timestamp_s)
  if t <= 3.0:
    return 0.0
  if t <= 4.0:
    return planned_path_coordinate_m(t)
  if t <= 7.0:
    return 0.010
  if t <= 12.0:
    return planned_path_coordinate_m(t)
  if t <= 17.0:
    return 0.060 + (0.060 - planned_path_coordinate_m(t))
  return 0.110


class BunnyPadCenterShape(AnalyticShape):
  """Upper-envelope planning surface for the physical belly-pad center.

  MuJoCo collision and final scoring remain unchanged.  This surface only
  gives M09 a Cartesian center target roughly one pad half-thickness outside
  the Bunny, avoiding an unphysical tip-site target inside the triangle mesh.
  """

  def __init__(
    self,
    bunny: BunnyHeightField,
    object_position_m: ArrayLike,
    *,
    pad_center_offset_m: float = 0.002,
  ) -> None:
    self.bunny = bunny
    self.object_position_m = np.asarray(object_position_m, dtype=np.float64).copy()
    self.pad_center_offset_m = float(pad_center_offset_m)
    if self.object_position_m.shape != (3,) or not np.all(np.isfinite(self.object_position_m)):
      raise ValueError("object_position_m must be a finite 3-vector")
    if self.pad_center_offset_m <= 0.0:
      raise ValueError("pad_center_offset_m must be positive")

  def query(self, point: ArrayLike) -> SurfaceProjection:
    query = np.asarray(point, dtype=np.float64)
    if query.shape != (3,) or not np.all(np.isfinite(query)):
      raise ValueError("point must be a finite 3-vector")
    local_xy = query[:2] - self.object_position_m[:2]
    height, normal, valid = self.bunny.query(float(local_xy[0]), float(local_xy[1]))
    if not valid:
      raise ValueError("planning query left the Bunny upper-envelope silhouette")
    surface = self.object_position_m + np.array([local_xy[0], local_xy[1], height])
    center = surface + self.pad_center_offset_m * normal
    return SurfaceProjection(
      point=center,
      normal=np.asarray(normal, dtype=np.float64),
      signed_distance=float(np.dot(query - center, normal)),
    )

  def sample_surface(
    self,
    count: int,
    rng: np.random.Generator,
  ) -> NDArray[np.float64]:
    if count <= 0:
      raise ValueError("count must be positive")
    valid = np.argwhere(self.bunny.valid_mask)
    indices = rng.choice(len(valid), size=count, replace=len(valid) < count)
    selected = valid[indices]
    points = np.zeros((count, 3), dtype=np.float64)
    for row, (iy, ix) in enumerate(selected):
      local_xy = np.array([self.bunny.x_m[ix], self.bunny.y_m[iy]])
      height, normal, _ = self.bunny.query(float(local_xy[0]), float(local_xy[1]))
      points[row] = (
        self.object_position_m
        + np.array([local_xy[0], local_xy[1], height])
        + self.pad_center_offset_m * normal
      )
    return points


@dataclass(slots=True)
class PlannerBundle:
  graph: ContactModeGraph
  planning_surface: OracleSurfaceModel
  kinematics: Any
  state: PlannerState
  optimizer: ContinuousOptimizer
  cheap_cert: CheapCert
  shadow: ShadowViabilityEvaluator


class _JointCenteredKinematics(LinearizedHandKinematics):
  """I02 redundant IK backend with a task-nullspace joint-center bias."""

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
    desired = np.asarray(desired_fingertip_positions_m, dtype=np.float64)
    wrist = np.asarray(wrist_position_m, dtype=np.float64)
    q = np.asarray(seed_q_rad, dtype=np.float64).copy()
    fingers = tuple(sorted(set(int(finger) for finger in controlled_fingers)))
    if not fingers:
      return q, 0.0
    indices = np.asarray([finger - 1 for finger in fingers], dtype=np.int64)
    jacobian = self.fingertip_jacobians_m_per_rad[indices].reshape(
      -1,
      self.joint_dimension,
    )
    gram = jacobian @ jacobian.T + damping**2 * np.eye(len(indices) * 3)
    inverse = np.linalg.solve(gram, np.eye(len(indices) * 3))
    pseudoinverse = jacobian.T @ inverse
    projector = np.eye(self.joint_dimension) - pseudoinverse @ jacobian
    active_columns = np.linalg.norm(jacobian, axis=0) > 1e-12
    midpoint = 0.5 * (self.joint_lower_rad + self.joint_upper_rad)
    for _ in range(iterations):
      current = self.forward(q, wrist)
      error = (desired[indices] - current[indices]).reshape(-1)
      if float(np.max(np.linalg.norm(error.reshape(-1, 3), axis=1))) <= 1e-7:
        break
      centering = np.zeros_like(q)
      centering[active_columns] = 0.12 * (
        midpoint[active_columns] - q[active_columns]
      )
      delta = pseudoinverse @ error + projector @ centering
      q = np.clip(q + delta, self.joint_lower_rad, self.joint_upper_rad)
    current = self.forward(q, wrist)
    residual = float(
      np.max(np.linalg.norm(desired[indices] - current[indices], axis=1))
    )
    return q, residual


def make_planner_bundle(
  handles: Any,
  data: Any,
  bunny: BunnyHeightField,
  actual_contact_set: frozenset[int],
  config: I02I03BunnyConfig,
) -> PlannerBundle:
  if not actual_contact_set:
    raise ValueError("planner root requires a measured nonempty contact set")
  graph = ContactModeGraph()
  kinematics = _local_kinematics(handles, data)
  if config.cell.startswith("i02"):
    kinematics = _JointCenteredKinematics(
      reference_q_rad=kinematics.reference_q_rad,
      joint_lower_rad=kinematics.joint_lower_rad,
      joint_upper_rad=kinematics.joint_upper_rad,
      reference_wrist_position_m=kinematics.reference_wrist_position_m,
      reference_fingertip_positions_m=kinematics.reference_fingertip_positions_m,
      fingertip_jacobians_m_per_rad=kinematics.fingertip_jacobians_m_per_rad,
      minimum_tip_separation_m=kinematics.minimum_tip_separation_m,
    )
  state = PlannerState(
    joint_positions_rad=_logical_q(handles, data),
    wrist_position_m=data.site_xpos[handles.palm_site_id],
    fingertip_positions_m=data.site_xpos[handles.tip_site_ids],
    actual_contact_set=actual_contact_set,
    surface_model_version=SURFACE_MODEL_VERSION,
  )
  planning_surface = OracleSurfaceModel(
    BunnyPadCenterShape(bunny, handles.object_position_m),
    version=SURFACE_MODEL_VERSION,
  )
  optimizer = ContinuousOptimizer(
    graph,
    planning_surface,
    kinematics,
    OptimizationConfig(
      waypoint_count=7,
      max_commit_displacement_m=0.015,
      target_tolerance_m=0.00075,
      anchor_tolerance_m=0.0015,
      minimum_collision_clearance_m=0.0,
      minimum_joint_margin_rad=config.minimum_audit_joint_margin_rad,
      reach_radius_m=0.085,
      release_distance_m=0.006,
      free_surface_clearance_m=(0.004 if config.cell.startswith("i02") else 0.0032),
      nominal_speed_m_s=0.04,
      damping=2e-4,
      ik_iterations=12,
    ),
  )
  cheap = CheapCert(
    graph,
    CheapCertConfig(reject_joint_below_rad=0.0),
  )
  return PlannerBundle(
    graph=graph,
    planning_surface=planning_surface,
    kinematics=kinematics,
    state=state,
    optimizer=optimizer,
    cheap_cert=cheap,
    shadow=ShadowViabilityEvaluator(graph, cheap),
  )


def optimize_prefix(
  bundle: PlannerBundle,
  primitive: ContactPrimitive,
  target_position_m: ArrayLike,
  *,
  prefix_id: str,
  progress_gain_m: float,
) -> tuple[PlannedPrefix, dict[str, Any]]:
  result = bundle.optimizer.optimize(
    OptimizationRequest(
      state=bundle.state,
      primitive=primitive,
      target_position_m=target_position_m,
      prefix_id=prefix_id,
      progress_gain_m=progress_gain_m,
    )
  )
  if not result.feasible or result.prefix is None:
    raise RuntimeError(
      "M09 rejected prefix: "
      + ",".join(result.reasons)
      + f" (target_error={result.final_target_error_m:.6g},"
      + f" joint_margin={result.minimum_joint_margin_rad:.6g},"
      + f" trust_use={result.trust_region_use:.6g})"
    )
  return result.prefix, {
    "optimizer_latency_s": result.solve_time_s,
    "optimizer_status": result.status.value,
    "optimizer_target_error_m": result.final_target_error_m,
    "optimizer_minimum_joint_margin_rad": result.minimum_joint_margin_rad,
    "optimizer_minimum_clearance_m": result.minimum_clearance_m,
    "optimizer_anchor_margin_m": result.anchor_margin_m,
    "optimizer_reach_margin_m": result.reach_margin_m,
    "optimizer_trust_region_use": result.trust_region_use,
    "optimizer_achieved_progress_m": result.achieved_progress_m,
  }


def audit_prefix(
  handles: Any,
  data: Any,
  bunny: BunnyHeightField,
  config: I02I03BunnyConfig,
  bundle: PlannerBundle,
  prefix: PlannedPrefix,
  *,
  timestamp_s: float,
  replacement_confirmation_s: dict[int, float],
) -> tuple[Any, dict[str, Any]]:
  exact = _world_mesh_oracle(bunny, handles.object_position_m)
  exact = OracleSurfaceModel(exact.shape, version=SURFACE_MODEL_VERSION)
  auditor = ExactPrefixAuditor(
    bundle.graph,
    AuditEnvironment(
      exact,
      bundle.kinematics,
      link_clearance_fn=_link_clearance_function(handles, bunny),
    ),
    AuditConfig(
      audit_version="exact-prefix-audit.i02-i03-bunny.v1",
      subdivisions_per_segment=9,
      minimum_link_clearance_m=0.0,
      minimum_joint_margin_rad=config.minimum_audit_joint_margin_rad,
      anchor_tolerance_m=0.0015,
      kinematic_consistency_tolerance_m=1e-8,
      max_commit_displacement_m=0.015,
    ),
  )
  result = auditor.audit(
    AuditRequest(
      prefix=prefix,
      current_state=bundle.state,
      commit_context=CommitContext(
        actual_contact_set=bundle.state.actual_contact_set,
        replacement_confirmation_s=replacement_confirmation_s,
        minimum_confirmation_s=0.05,
      ),
      issued_at_s=timestamp_s,
    )
  )
  if not result.certified or result.certificate is None:
    raise RuntimeError("M10 audit rejected prefix: " + ",".join(result.reasons))
  return result.certificate, {
    "audit_latency_s": result.latency_s,
    "audit_swept_samples": result.swept_samples,
    "audit_minimum_joint_margin_rad": result.minimum_joint_margin_rad,
    "audit_minimum_self_clearance_m": result.minimum_self_collision_clearance_m,
    "audit_minimum_link_clearance_m": result.minimum_link_clearance_m,
    "audit_maximum_anchor_error_m": result.maximum_anchor_error_m,
    "audit_maximum_trust_displacement_m": result.maximum_trust_displacement_m,
    "audit_maximum_kinematic_error_m": result.maximum_kinematic_error_m,
    "certificate_id": result.certificate.certificate_id,
  }


I03_SLIDE_X_M = {1: -0.0030, 2: -0.0025, 3: 0.0040, 4: 0.0020}


def make_i03_candidate_factory(
  bundle: PlannerBundle,
  config: I02I03BunnyConfig,
) -> Callable[[PlannerState, ContactPrimitive, int], PlanningCandidate | None]:
  """Frozen physical candidate set used identically by both I03 cells."""

  def factory(
    state: PlannerState,
    primitive: ContactPrimitive,
    depth: int,
  ) -> PlanningCandidate | None:
    if primitive.kind not in {PrimitiveKind.SLIDE, PrimitiveKind.BREAK}:
      return None
    assert primitive.finger_id is not None
    finger = primitive.finger_id
    if primitive.kind is PrimitiveKind.SLIDE:
      displacement = I03_SLIDE_X_M[finger]
      target = np.array(state.fingertip_positions_m[finger - 1], copy=True)
      target[0] += displacement
      progress = abs(displacement)
    else:
      target = np.array(state.fingertip_positions_m[finger - 1], copy=True)
      progress = 0.001
    current_margin = bundle.kinematics.joint_margin(state.joint_positions_rad)
    cheap_input = CheapCertInput(
      mode=state.mode,
      primitive=primitive,
      surface_model_version=SURFACE_MODEL_VERSION,
      anchor_margin_m=0.001,
      joint_margin_rad=(
        current_margin - config.terminal_viability_joint_reserve_rad
      ),
      collision_margin_m=0.005,
      reach_margin_m=0.030,
      uncertainty_margin=1.0,
      trust_margin_m=0.015 - progress,
      metadata={"depth": float(depth)},
    )
    request = OptimizationRequest(
      state=state,
      primitive=primitive,
      target_position_m=target,
      prefix_id=f"i03-d{depth}-m{state.mode.mask}-{primitive.key.lower()}",
      progress_gain_m=progress,
      metadata={"terminal_reserve_rad": config.terminal_viability_joint_reserve_rad},
    )
    return PlanningCandidate(
      cheap_input=cheap_input,
      optimization_request=request,
      motion_cost=0.1 * progress,
      risk_cost=0.0,
    )

  return factory


def search_i03_prefix(
  bundle: PlannerBundle,
  config: I02I03BunnyConfig,
  *,
  use_shadow: bool,
) -> tuple[SearchResult, ShadowResult, dict[str, Any]]:
  factory = make_i03_candidate_factory(bundle, config)
  terminal_predicate = bundle.shadow.predicate(factory) if use_shadow else None

  class _RecordingOptimizer:
    """Transparent timing tap; it does not alter the optimizer result."""

    def __init__(self, optimizer: ContinuousOptimizer) -> None:
      self.optimizer = optimizer
      self.solve_time_by_prefix: dict[str, float] = {}
      self.terminal_margin_by_prefix: dict[str, float] = {}

    def optimize(self, request: OptimizationRequest) -> Any:
      optimized = self.optimizer.optimize(request)
      self.solve_time_by_prefix[request.prefix_id] = float(optimized.solve_time_s)
      if optimized.prefix is not None:
        self.terminal_margin_by_prefix[request.prefix_id] = (
          self.optimizer.kinematics.joint_margin(
            optimized.prefix.samples[-1].joint_positions_rad
          )
        )
      return optimized

  recording_optimizer = _RecordingOptimizer(bundle.optimizer)
  started = perf_counter()
  result = LazyBeamSearch(
    bundle.graph,
    bundle.cheap_cert,
    recording_optimizer,  # type: ignore[arg-type]
    BeamSearchConfig(horizon=1, beam_width=8, per_mode_quota=2),
  ).search(
    bundle.state,
    factory,
    terminal_viability=terminal_predicate,
  )
  measured_latency = perf_counter() - started
  if not result.found or result.best_node is None or result.committed_prefix_candidate is None:
    raise RuntimeError("M11 found no I03 committed edge")
  terminal_shadow = bundle.shadow.evaluate(result.best_node.state, factory)
  selected_prefix_id = result.committed_prefix_candidate.prefix_id
  evidence = {
    "search_latency_s": result.latency_s,
    "search_wall_latency_s": measured_latency,
    "expanded_nodes": result.expanded_nodes,
    "enumerated_edges": result.enumerated_edges,
    "cheap_survivors": result.cheap_survivors,
    "optimized_edges": result.optimized_edges,
    "retained_nodes_per_depth": list(result.retained_nodes_per_depth),
    "distinct_modes_per_depth": list(result.distinct_modes_per_depth),
    "selected_sequence": list(result.best_node.sequence_key),
    "selected_score": result.best_node.score,
    "selected_optimizer_latency_s": recording_optimizer.solve_time_by_prefix.get(
      selected_prefix_id,
      0.0,
    ),
    "candidate_terminal_joint_margin_rad": dict(
      sorted(recording_optimizer.terminal_margin_by_prefix.items())
    ),
    "selected_terminal_joint_margin_rad": bundle.kinematics.joint_margin(
      result.best_node.state.joint_positions_rad
    ),
    "predicted_terminal_viability": terminal_shadow.status.value,
    "predicted_successor_fingers": list(terminal_shadow.distinct_successor_fingers),
    "predicted_shadow_latency_s": terminal_shadow.latency_s,
    "prediction_suffix_count": len(result.prediction_suffix),
    "shadow_filter_enabled": use_shadow,
  }
  return result, terminal_shadow, evidence


def evaluate_actual_shadow(
  bundle: PlannerBundle,
  config: I02I03BunnyConfig,
) -> tuple[ShadowResult, dict[str, Any]]:
  factory = make_i03_candidate_factory(bundle, config)
  result = bundle.shadow.evaluate(bundle.state, factory)
  return result, {
    "actual_terminal_viability": result.status.value,
    "actual_successor_fingers": list(result.distinct_successor_fingers),
    "actual_successor_count": len(result.successors),
    "actual_shadow_latency_s": result.latency_s,
    "actual_shadow_execution_authority": result.execution_authority,
    "actual_terminal_joint_margin_rad": bundle.kinematics.joint_margin(
      bundle.state.joint_positions_rad
    ),
  }
