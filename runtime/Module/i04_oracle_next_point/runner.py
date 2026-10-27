"""Full MuJoCo I04 Explicit-MCC traversal of the non-convex Bunny SDF."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import mujoco
import numpy as np
from numpy.typing import NDArray

from Module.e05_physics.scene import FINGERS
from Module.fr3_leap import (
  ARM_HOME_Q,
  HAND_NATURAL_Q,
  FullRobotModelConfig,
  build_full_robot,
)
from Module.i01_bunny_physics.runner import (
  _finger_ik,
  _logical_q,
  _pad_support_radius,
  _quaternion_from_matrix,
  _signed_compression_jacobian,
)
from Module.i01_bunny_physics.surface import canonical_bunny_heightfield
from Module.i04_oracle_next_point.planner import (
  SURFACE_MODEL_VERSION,
  ExplicitI04Planner,
  I04PlannerConfig,
  I04PlanResult,
)
from Module.i04_oracle_next_point.surface_graph import (
  BunnySurfaceGraph,
  CoverageLedger,
  GoalSelection,
  SurfaceGoal,
)
from Module.module_2_fingertip_mcc import (
  FingertipMCC,
  FullRobotFingertipMCC,
  MCCConfig,
)
from Module.module_3_runtime_guards import (
  CommandContinuityConfig,
  CommandContinuityLimiter,
  ForceSafetyConfig,
  ForceSafetyExecutor,
)
from Module.module_6_prefix_executor import (
  ExecutorConfig,
  ExecutorObservation,
  MCCBaselineAdapter,
  TransactionState,
  TransactionalPrefixExecutor,
)


TRACE_SCHEMA_VERSION = "i04-full-bunny-trace.v1"
EVALUATOR_VERSION = "i04-full-bunny-evaluator.v1"


@dataclass(frozen=True, slots=True)
class I04BunnyConfig:
  seed: int = 7
  dt_s: float = 0.002
  acquisition_s: float = 3.0
  maximum_duration_s: float = 600.0
  coverage_radius_m: float = 0.025
  arrival_tolerance_m: float = 0.012
  normal_tolerance_rad: float = np.deg2rad(55.0)
  bridge_step_m: float = 0.003
  desired_force_n: float = 2.0
  contact_threshold_n: float = 0.20
  force_limit_n: float = 8.0
  contact_loss_debounce_s: float = 0.010
  contact_confirmation_s: float = 0.040
  root_stabilization_force_n: float = 0.75
  recovery_force_n: float = 2.50
  safe_hold_recovery_s: float = 0.30
  failed_finger_cooldown_s: float = 8.0
  stagnation_progress_m: float = 0.001
  stagnation_prefix_limit: int = 8
  log_stride: int = 5
  maximum_goals: int | None = None
  initial_arm_noise_rad: float = 0.0003
  initial_hand_noise_rad: float = 0.0008
  visual_mesh_path: str | None = None

  def __post_init__(self) -> None:
    if not np.isclose(self.dt_s, 0.002):
      raise ValueError("I04 freezes the MuJoCo timestep at 0.002 s")
    for name in (
      "acquisition_s",
      "maximum_duration_s",
      "coverage_radius_m",
      "arrival_tolerance_m",
      "normal_tolerance_rad",
      "bridge_step_m",
      "desired_force_n",
      "contact_threshold_n",
      "force_limit_n",
      "contact_loss_debounce_s",
      "contact_confirmation_s",
      "root_stabilization_force_n",
      "recovery_force_n",
      "safe_hold_recovery_s",
      "failed_finger_cooldown_s",
      "stagnation_progress_m",
    ):
      if not np.isfinite(float(getattr(self, name))) or float(getattr(self, name)) <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    if self.maximum_duration_s <= self.acquisition_s:
      raise ValueError("maximum_duration_s must exceed acquisition_s")
    if self.force_limit_n <= self.desired_force_n:
      raise ValueError("force_limit_n must exceed desired_force_n")
    if self.recovery_force_n <= self.desired_force_n:
      raise ValueError("recovery_force_n must exceed desired_force_n")
    if self.recovery_force_n >= self.force_limit_n:
      raise ValueError("recovery_force_n must remain below force_limit_n")
    if self.log_stride < 1:
      raise ValueError("log_stride must be positive")
    if self.stagnation_prefix_limit < 2:
      raise ValueError("stagnation_prefix_limit must be >=2")
    if self.maximum_goals is not None and self.maximum_goals < 1:
      raise ValueError("maximum_goals must be positive when provided")


@dataclass(slots=True)
class I04BunnyTrace:
  config: I04BunnyConfig
  time_s: NDArray[np.float64]
  arm_q_rad: NDArray[np.float64]
  arm_command_rad: NDArray[np.float64]
  finger_q_rad: NDArray[np.float64]
  finger_command_rad: NDArray[np.float64]
  palm_pose_world: NDArray[np.float64]
  fingertip_positions_world_m: NDArray[np.float64]
  contact_positions_world_m: NDArray[np.float64]
  contact_normals_world: NDArray[np.float64]
  fingertip_forces_n: NDArray[np.float64]
  contact_active: NDArray[np.bool_]
  goal_id: NDArray[np.int32]
  goal_vertex: NDArray[np.int32]
  bridge_target_vertex: NDArray[np.int32]
  selected_finger: NDArray[np.int8]
  coverage_fraction: NDArray[np.float64]
  covered_area_fraction: NDArray[np.float64]
  transaction_state: NDArray[np.str_]
  primitive: NDArray[np.str_]
  certificate_id: NDArray[np.str_]
  guard_reason: NDArray[np.str_]
  controller_latency_s: NDArray[np.float64]
  physics_latency_s: NDArray[np.float64]
  events: list[dict[str, Any]]

  def npz_payload(self) -> dict[str, NDArray[Any]]:
    return {
      "time_s": self.time_s,
      "arm_q_rad": self.arm_q_rad,
      "arm_command_rad": self.arm_command_rad,
      "finger_q_rad": self.finger_q_rad,
      "finger_command_rad": self.finger_command_rad,
      "palm_pose_world": self.palm_pose_world,
      "fingertip_positions_world_m": self.fingertip_positions_world_m,
      "contact_positions_world_m": self.contact_positions_world_m,
      "contact_normals_world": self.contact_normals_world,
      "fingertip_forces_n": self.fingertip_forces_n,
      "contact_active": self.contact_active,
      "goal_id": self.goal_id,
      "goal_vertex": self.goal_vertex,
      "bridge_target_vertex": self.bridge_target_vertex,
      "selected_finger": self.selected_finger,
      "coverage_fraction": self.coverage_fraction,
      "covered_area_fraction": self.covered_area_fraction,
      "transaction_state": self.transaction_state,
      "primitive": self.primitive,
      "certificate_id": self.certificate_id,
      "guard_reason": self.guard_reason,
      "controller_latency_s": self.controller_latency_s,
      "physics_latency_s": self.physics_latency_s,
    }


def _contact_state(
  handles: Any,
  data: mujoco.MjData,
  graph: BunnySurfaceGraph,
) -> tuple[
  NDArray[np.float64],
  NDArray[np.float64],
  NDArray[np.float64],
  int,
]:
  forces = np.zeros(4, dtype=np.float64)
  weighted_positions = np.zeros((4, 3), dtype=np.float64)
  weighted_normals = np.zeros((4, 3), dtype=np.float64)
  lookup = {int(geom): index for index, geom in enumerate(handles.tip_geom_ids)}
  contact_force = np.zeros(6, dtype=np.float64)
  non_tip = 0
  for contact_index in range(data.ncon):
    contact = data.contact[contact_index]
    geom_1 = int(contact.geom1)
    geom_2 = int(contact.geom2)
    if handles.object_geom_id not in (geom_1, geom_2):
      continue
    other = geom_2 if geom_1 == handles.object_geom_id else geom_1
    finger = lookup.get(other)
    if finger is None:
      non_tip += 1
      continue
    mujoco.mj_contactForce(handles.model, data, contact_index, contact_force)
    force = abs(float(contact_force[0]))
    normal = np.asarray(contact.frame[:3], dtype=np.float64)
    if geom_2 == handles.object_geom_id:
      normal = -normal
    forces[finger] += force
    weighted_positions[finger] += force * np.asarray(contact.pos)
    weighted_normals[finger] += force * normal
  positions = np.array(data.site_xpos[handles.tip_site_ids], copy=True)
  active = forces > 0.0
  positions[active] = weighted_positions[active] / forces[active, None]
  query_normals = np.tile([0.0, 0.0, 1.0], (4, 1))
  query_normals[active] = weighted_normals[active] / np.linalg.norm(
    weighted_normals[active],
    axis=1,
    keepdims=True,
  )
  vertices, _ = graph.oriented_nearest_vertices(
    positions - handles.object_position_m,
    query_normals,
  )
  normals = np.array(graph.normals[vertices], dtype=np.float64, copy=True)
  normals /= np.linalg.norm(normals, axis=1, keepdims=True)
  return forces, positions, normals, non_tip


def _surface_centers(
  handles: Any,
  data: mujoco.MjData,
  graph: BunnySurfaceGraph,
  reference_normals_world: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
  tips = np.array(data.site_xpos[handles.tip_site_ids], copy=True)
  if reference_normals_world is None:
    vertices, _ = graph.nearest_vertices(tips - handles.object_position_m)
  else:
    vertices, _ = graph.oriented_nearest_vertices(
      tips - handles.object_position_m,
      reference_normals_world,
    )
  points = handles.object_position_m + graph.vertices_m[vertices]
  normals = np.array(graph.normals[vertices], copy=True)
  normals /= np.linalg.norm(normals, axis=1, keepdims=True)
  centers = np.stack(
    [
      points[finger]
      + _pad_support_radius(handles, data, finger, normals[finger])
      * normals[finger]
      for finger in range(4)
    ]
  )
  return centers, normals


def _goal_bridge_target(
  graph: BunnySurfaceGraph,
  goal: SurfaceGoal,
  contact_points_local_m: NDArray[np.float64],
  contact_normals_local: NDArray[np.float64],
  maximum_step_m: float,
) -> tuple[int, float, int]:
  contact_vertices, _ = graph.oriented_nearest_vertices(
    contact_points_local_m,
    contact_normals_local,
  )
  best: tuple[float, int, NDArray[np.int32]] | None = None
  for root in contact_vertices:
    distances, predecessors = graph.distances_from(
      int(root),
      return_predecessors=True,
    )
    path = graph.reconstruct_path(int(root), goal.vertex_index, predecessors)
    candidate = (float(distances[goal.vertex_index]), int(root), path)
    if best is None or candidate[0] < best[0]:
      best = candidate
  assert best is not None
  distance, root, full_path = best
  decimated = graph.decimate_path(full_path, maximum_step_m=maximum_step_m)
  target = int(decimated[1] if len(decimated) > 1 else decimated[0])
  return target, distance, root


def run_i04_bunny(
  config: I04BunnyConfig,
  *,
  planner_config: I04PlannerConfig | None = None,
) -> tuple[I04BunnyTrace, dict[str, Any]]:
  """Run one state-rooted Explicit MCC baseline episode."""

  bunny = canonical_bunny_heightfield()
  visual_mesh_path = config.visual_mesh_path
  if visual_mesh_path is None:
    raise ValueError("visual_mesh_path is required for I04 SDF collision")
  handles = build_full_robot(
    FullRobotModelConfig(
      surface="bunny",
      timestep_s=config.dt_s,
      gravity_m_s2=0.0,
      arm_kp=1800.0,
      arm_damping_ratio=0.9,
      hand_kp=60.0,
      hand_damping_ratio=1.5,
      bunny_visual_mesh_path=visual_mesh_path,
      bunny_collision_mode="sdf",
    )
  )
  data = mujoco.MjData(handles.model)
  rng = np.random.default_rng(config.seed)
  arm_initial = np.clip(
    ARM_HOME_Q + rng.normal(0.0, config.initial_arm_noise_rad, 7),
    handles.arm_joint_ranges_rad[:, 0] + 0.04,
    handles.arm_joint_ranges_rad[:, 1] - 0.04,
  )
  hand_initial = np.clip(
    HAND_NATURAL_Q + rng.normal(0.0, config.initial_hand_noise_rad, 16),
    handles.hand_joint_ranges_rad[:, 0] + 0.05,
    handles.hand_joint_ranges_rad[:, 1] - 0.05,
  )
  data.qpos[handles.arm_qpos_adrs] = arm_initial
  data.qpos[handles.hand_qpos_adrs] = hand_initial
  data.ctrl[handles.arm_actuator_ids] = arm_initial
  data.ctrl[handles.hand_actuator_ids] = hand_initial
  mujoco.mj_forward(handles.model, data)

  graph = BunnySurfaceGraph(
    bunny.mesh,
    coverage_radius_m=config.coverage_radius_m,
    arrival_tolerance_m=config.arrival_tolerance_m,
    normal_tolerance_rad=config.normal_tolerance_rad,
  )
  ledger = CoverageLedger(graph)
  planner = ExplicitI04Planner(handles, bunny, graph, planner_config)
  full_mcc = FullRobotFingertipMCC(
    tuple(
      FingertipMCC(
        MCCConfig(
          virtual_mass=0.08,
          damping=14.0,
          stiffness=25.0,
          dt_s=config.dt_s,
          max_offset_m=0.020,
          max_velocity_m_s=0.08,
          max_acceleration_m_s2=30.0,
        )
      )
      for _ in range(4)
    )
  )
  executor = TransactionalPrefixExecutor(
    ExecutorConfig(
      # M10 bounds Cartesian/joint interpolation disagreement to 2 mm.  A
      # progress-only prefix may close just outside that bound after 90% of
      # its certified duration; contact-changing MAKE still requires a real,
      # persistent collision independent of this tolerance.
      completion_tolerance_m=0.00225,
      wrist_completion_tolerance_m=0.0010,
      wrist_joint_completion_tolerance_rad=0.0020,
      wrist_joint_dimension=7,
      default_timeout_s=2.0,
      desired_anchor_force_n=config.desired_force_n,
      root_state_tolerance=1e-9,
      make_contact_is_terminal=True,
      minimum_execution_fraction=0.90,
    ),
    mcc_adapter=MCCBaselineAdapter(full_mcc),
  )
  force_safety = ForceSafetyExecutor(
    ForceSafetyConfig(
      joint_lower_rad=handles.hand_joint_ranges_rad[:, 0],
      joint_upper_rad=handles.hand_joint_ranges_rad[:, 1],
      dt_s=config.dt_s,
      soft_force_n=6.5,
      hard_force_n=config.force_limit_n,
      recover_force_n=2.5,
      rapid_loading_rate_n_s=5000.0,
      rapid_loading_min_force_n=7.0,
    )
  )
  continuity = CommandContinuityLimiter(
    CommandContinuityConfig(
      max_finger_step_rad=0.0025,
      max_wrist_translation_step_m=0.00012,
      max_wrist_rotation_step_rad=0.012,
    )
  )
  initial_pose = np.concatenate(
    (
      data.site_xpos[handles.palm_site_id],
      _quaternion_from_matrix(data.site_xmat[handles.palm_site_id]),
    )
  )
  continuity.reset(
    finger_command_rad=data.ctrl[handles.hand_actuator_ids],
    wrist_pose_world=initial_pose,
  )

  measured_forces = np.zeros(4, dtype=np.float64)
  measured_positions = np.array(data.site_xpos[handles.tip_site_ids], copy=True)
  measured_normals = np.tile([0.0, 0.0, 1.0], (4, 1))
  contact_active = np.zeros(4, dtype=np.bool_)
  # The planner/M06 owns the tangential nominal reference.  MCC may add only
  # a normal-force correction and HOLD must continue the last nominal rather
  # than reprojecting active pads to a new mesh point on every physics tick.
  hold_nominal_positions = np.array(
    data.site_xpos[handles.tip_site_ids],
    dtype=np.float64,
    copy=True,
  )
  contact_confirmed_s = np.zeros(4, dtype=np.float64)
  last_nonempty_set = frozenset[int]()
  empty_run = 0
  maximum_empty_run = 0
  contact_supported_steps = 0
  traversal_steps = 0
  peak_force_n = 0.0
  squared_force_error = 0.0
  force_error_samples = 0
  non_tip_contact_steps = 0
  make_before_break_violations = 0
  previous_contact_set = frozenset[int]()
  previous_finger_command = np.array(
    data.ctrl[handles.hand_actuator_ids],
    copy=True,
  )
  contact_mode_changes = 0
  plan: I04PlanResult | None = None
  current_goal: SurfaceGoal | None = None
  current_selection: GoalSelection | None = None
  current_bridge_target = -1
  transaction_sequence = 0
  committed_prefix_count = 0
  committed_wrist_adjust_count = 0
  safe_hold_replan_count = 0
  replan_rejected_count = 0
  next_planning_time_s = config.acquisition_s
  stop_reason: str | None = None
  failure_counts: dict[int, int] = {}
  assignment_counts = np.zeros(4, dtype=np.int64)
  completed_finger_prefixes_since_wrist = 0
  excluded_finger_until_s: dict[int, float] = {}
  recovery_until_s = config.acquisition_s
  root_stabilizing = False
  force_wrist_recovery = False
  slide_route_best_m: dict[tuple[int, int], float] = {}
  slide_stagnation_count: dict[tuple[int, int], int] = {}
  pending_workspace_break_finger: int | None = None
  planner_latencies: list[float] = []
  audit_latencies: list[float] = []
  search_latencies: list[float] = []
  events: list[dict[str, Any]] = []

  logs: dict[str, list[Any]] = {
    "time_s": [],
    "arm_q_rad": [],
    "arm_command_rad": [],
    "finger_q_rad": [],
    "finger_command_rad": [],
    "palm_pose_world": [],
    "fingertip_positions_world_m": [],
    "contact_positions_world_m": [],
    "contact_normals_world": [],
    "fingertip_forces_n": [],
    "contact_active": [],
    "goal_id": [],
    "goal_vertex": [],
    "bridge_target_vertex": [],
    "selected_finger": [],
    "coverage_fraction": [],
    "covered_area_fraction": [],
    "transaction_state": [],
    "primitive": [],
    "certificate_id": [],
    "guard_reason": [],
    "controller_latency_s": [],
    "physics_latency_s": [],
  }

  def measured_set(*, debounced: bool = False) -> frozenset[int]:
    current = frozenset(int(index + 1) for index in np.flatnonzero(contact_active))
    if (
      debounced
      and not current
      and empty_run * config.dt_s <= config.contact_loss_debounce_s + 1e-12
    ):
      return last_nonempty_set
    return current

  def confirmed_set() -> frozenset[int]:
    """Measured contacts that persisted for the frozen confirmation window."""

    return frozenset(
      int(index + 1)
      for index in np.flatnonzero(
        contact_active
        & (contact_confirmed_s + 1e-12 >= config.contact_confirmation_s)
      )
    )

  def observation(timestamp: float) -> ExecutorObservation:
    return ExecutorObservation(
      timestamp_s=timestamp,
      surface_model_version=SURFACE_MODEL_VERSION,
      wrist_position_m=data.site_xpos[handles.palm_site_id],
      fingertip_positions_m=data.site_xpos[handles.tip_site_ids],
      joint_positions_rad=_logical_q(handles, data),
      fingertip_forces_n=measured_forces,
      outward_normals=measured_normals,
      actual_contact_set=confirmed_set(),
    )

  def root_is_stable() -> bool:
    """Require a measured, force-bearing root before a new transaction.

    With only one or two contacts every member is a last-contact anchor, so a
    weak pad may not be treated as execution support merely because MuJoCo
    produced one threshold-crossing contact sample.  With three or four
    contacts, two independently confirmed load-bearing pads are sufficient;
    the remaining measured contacts are still retained in the audited mode.
    """

    actual = confirmed_set()
    if not actual:
      return False
    stable = {
      finger
      for finger in actual
      if contact_confirmed_s[finger - 1] + 1e-12
      >= config.contact_confirmation_s
      and measured_forces[finger - 1] + 1e-12
      >= config.root_stabilization_force_n
    }
    required = len(actual) if len(actual) <= 2 else 2
    return len(stable) >= required

  def choose_goal(timestamp: float) -> None:
    nonlocal current_goal, current_selection, pending_workspace_break_finger
    slide_route_best_m.clear()
    slide_stagnation_count.clear()
    pending_workspace_break_finger = None
    active = np.asarray(
      [finger - 1 for finger in sorted(confirmed_set())],
      dtype=np.int32,
    )
    if len(active) == 0:
      raise RuntimeError("cannot choose I04 goal without a measured contact")
    points = measured_positions[active] - handles.object_position_m
    palm_normal = np.array(
      data.site_xmat[handles.palm_site_id],
      copy=True,
    ).reshape(3, 3)[:, 2]

    def quick_score(goal_id: int, distance_m: float) -> float | None:
      normal = graph.normals[int(graph.required_vertices[goal_id])]
      normal_cost = float(
        np.arccos(np.clip(np.dot(palm_normal, normal), -1.0, 1.0))
      )
      failures = failure_counts.get(goal_id, 0)
      # Difficult goals may be postponed but never removed from the fixed
      # required set.  The capped penalty lets the current state choose an
      # easier bridge and guarantees the goal remains selectable later.
      return (
        0.012 * normal_cost
        + 0.025 * min(failures, 6)
        + 0.02 * distance_m
      )

    selection = ledger.select_from_measured_contacts(
      points,
      contact_normals_local=measured_normals[active],
      feasibility_score=quick_score,
      maximum_candidates=24,
    )
    current_selection = selection
    current_goal = selection.goal
    events.append(
      {
        "event": "ORACLE_GOAL_PUBLISHED",
        "time_s": timestamp,
        "goal_id": current_goal.goal_id,
        "goal_vertex": current_goal.vertex_index,
        "goal_position_local_m": current_goal.position_local_m.tolist(),
        "goal_normal_local": current_goal.normal_local.tolist(),
        "outgoing_tangent_local": current_goal.outgoing_tangent_local.tolist(),
        "oracle_target_finger": None,
        "root_vertex": selection.root_vertex,
        "bridge_length_m": selection.bridge_length_m,
        "remaining_goal_count": len(ledger.remaining_goal_ids),
        "measured_root_contact_set": sorted(confirmed_set()),
      }
    )

  def check_goal_arrival(timestamp: float) -> bool:
    nonlocal current_goal, current_selection
    if current_goal is None:
      return False
    active = np.asarray(
      [finger - 1 for finger in sorted(confirmed_set())],
      dtype=np.int32,
    )
    if len(active) == 0:
      return False
    fingers = ledger.arrival_fingers(
      current_goal,
      measured_positions[active] - handles.object_position_m,
      measured_normals[active],
    )
    if not fingers:
      return False
    actual_fingers = tuple(int(active[index] + 1) for index in (finger - 1 for finger in fingers))
    ledger.mark_arrived(current_goal.goal_id)
    events.append(
      {
        "event": "GOAL_ARRIVED",
        "time_s": timestamp,
        "goal_id": current_goal.goal_id,
        "goal_vertex": current_goal.vertex_index,
        "arrival_fingers": list(actual_fingers),
        "coverage_fraction": ledger.completion_fraction,
        "covered_area_fraction": ledger.covered_area_fraction,
        "measured_contact_set": sorted(confirmed_set()),
      }
    )
    current_goal = None
    current_selection = None
    return True

  maximum_steps = int(round(config.maximum_duration_s / config.dt_s))
  acquisition_steps = int(round(config.acquisition_s / config.dt_s))
  for step in range(maximum_steps):
    timestamp = float(data.time)
    controller_started = perf_counter()
    centers, surface_normals = _surface_centers(
      handles,
      data,
      graph,
      measured_normals,
    )
    planned_arm = np.array(data.ctrl[handles.arm_actuator_ids], copy=True)
    cartesian_commands: NDArray[np.float64]
    primitive_label = "ACQUIRE" if step < acquisition_steps else "HOLD"
    certificate = "NONE"
    selected_finger = 0
    ik_outward_normals = np.array(surface_normals, copy=True)

    if step < acquisition_steps:
      hold_nominal_positions = np.array(centers, copy=True)
      output = full_mcc.step(
        centers,
        -surface_normals,
        config.desired_force_n - measured_forces,
        np.ones(4, dtype=np.bool_),
      )
      cartesian_commands = np.stack(
        [command.position_command for command in output.commands]
      )
    else:
      if (
        stop_reason is None
        and confirmed_set()
        and root_is_stable()
        and plan is None
        and timestamp + 1e-12 >= recovery_until_s
        and timestamp + 1e-12 >= next_planning_time_s
      ):
        if check_goal_arrival(timestamp):
          if ledger.complete:
            stop_reason = "FULL_REQUIRED_SET_COMPLETED"
          elif (
            config.maximum_goals is not None
            and len(ledger.visit_order) >= config.maximum_goals
          ):
            stop_reason = "DEVELOPMENT_GOAL_LIMIT_REACHED"
        if stop_reason is None and current_goal is None:
          choose_goal(timestamp)
          # A newly published goal may already be occupied by another real
          # fingertip.  It is a valid no-finger-ID ARRIVE, not a zero-motion
          # planner shortcut, and must be consumed before commanding a move
          # that could unload that contact.
          if check_goal_arrival(timestamp):
            if ledger.complete:
              stop_reason = "FULL_REQUIRED_SET_COMPLETED"
            elif (
              config.maximum_goals is not None
              and len(ledger.visit_order) >= config.maximum_goals
            ):
              stop_reason = "DEVELOPMENT_GOAL_LIMIT_REACHED"
        if stop_reason is None and current_goal is not None:
          active = np.asarray(
            [finger - 1 for finger in sorted(confirmed_set())],
            dtype=np.int32,
          )
          request_wrist_adjust = False
          actual = confirmed_set()
          forced_break_request = (
            pending_workspace_break_finger
            if pending_workspace_break_finger in actual
            and len(actual) >= 3
            else None
          )
          if (
            pending_workspace_break_finger is not None
            and forced_break_request is None
          ):
            pending_workspace_break_finger = None
          try:
            current_bridge_target, bridge_remaining, root_vertex = _goal_bridge_target(
              graph,
              current_goal,
              measured_positions[active] - handles.object_position_m,
              measured_normals[active],
              config.bridge_step_m,
            )
            transaction_sequence += 1
            request_wrist_adjust = (
              (
                force_wrist_recovery
                or completed_finger_prefixes_since_wrist
                >= planner.config.wrist_adjust_interval_prefixes
              )
              and len(actual) >= 2
              and forced_break_request is None
            )
            planned = planner.plan_prefix(
              data,
              actual,
              current_goal.vertex_index,
              contact_positions_world_m=measured_positions,
              contact_normals_world=measured_normals,
              maximum_surface_step_m=config.bridge_step_m,
              force_wrist_adjust=request_wrist_adjust,
              forced_break_finger=forced_break_request,
              excluded_fingers=(
                frozenset(
                  finger
                  for finger, until_s in excluded_finger_until_s.items()
                  if timestamp < until_s
                )
              ),
              timestamp_s=timestamp,
              replacement_confirmation_s={
                int(index + 1): float(contact_confirmed_s[index])
                for index in np.flatnonzero(contact_active)
              },
              sequence=transaction_sequence,
            )
            # The Oracle publishes only the final surface goal.  The explicit
            # planner has now chosen both the realizing finger and that
            # finger's state-rooted geodesic micro-bridge.
            current_bridge_target = planned.target_vertex
            # Keep the acquired MCC integrator state across the certified
            # nominal-reference handoff.  CommandContinuityLimiter bounds the
            # small frame change; resetting here produced an avoidable one-tick
            # preload drop on every micro-prefix.
            executor.commit(
              planned.prefix,
              planned.certificate,
              observation(timestamp),
              # The certified Cartesian prefix is tracked through actuator
              # dynamics and MCC, so execution time is deliberately longer
              # than the nominal knot duration.  M06 still owns the timeout
              # and falls back to SAFE_HOLD if tracking does not converge.
              timeout_s=(
                max(2.0, planned.prefix.duration_s + 1.2)
                if planned.selected_primitive == "MAKE"
                else max(4.0, planned.prefix.duration_s + 3.0)
              ),
            )
            plan = planned
            committed_prefix_count += 1
            committed_wrist_adjust_count += int(
              planned.selected_primitive == "WRIST_ADJUST"
            )
            if planned.selected_primitive == "WRIST_ADJUST":
              force_wrist_recovery = False
            elif planned.selected_primitive == "BREAK":
              pending_workspace_break_finger = None
            elif (
              planned.selected_primitive == "SLIDE"
              and planned.selected_finger is not None
              and planned.evidence.get("selected_finger_route_remaining_m")
              is not None
            ):
              route_key = (
                current_goal.goal_id,
                planned.selected_finger,
              )
              route_remaining = float(
                planned.evidence["selected_finger_route_remaining_m"]
              )
              previous_best = slide_route_best_m.get(route_key, float("inf"))
              if (
                route_remaining
                <= previous_best - config.stagnation_progress_m
              ):
                slide_route_best_m[route_key] = route_remaining
                slide_stagnation_count[route_key] = 0
              else:
                slide_route_best_m[route_key] = min(
                  previous_best,
                  route_remaining,
                )
                slide_stagnation_count[route_key] = (
                  slide_stagnation_count.get(route_key, 0) + 1
                )
              if (
                slide_stagnation_count[route_key]
                >= config.stagnation_prefix_limit
                and len(actual) >= 3
              ):
                pending_workspace_break_finger = planned.selected_finger
                events.append(
                  {
                    "event": "GEODESIC_STAGNATION_BREAK_REQUESTED",
                    "time_s": timestamp,
                    "goal_id": current_goal.goal_id,
                    "finger": planned.selected_finger,
                    "root_contact_set": sorted(actual),
                    "best_route_remaining_m": slide_route_best_m[route_key],
                    "current_route_remaining_m": route_remaining,
                    "stagnant_prefix_count": slide_stagnation_count[route_key],
                  }
                )
            if planned.selected_finger is not None:
              assignment_counts[planned.selected_finger - 1] += 1
            planner_latencies.append(float(planned.evidence["planning_wall_latency_s"]))
            audit_latencies.append(float(planned.evidence["m10_latency_s"]))
            search_latencies.append(float(planned.evidence["m11_wall_latency_s"]))
            events.append(
              {
                "event": "CERTIFIED_PREFIX_COMMITTED",
                "time_s": timestamp,
                "goal_id": current_goal.goal_id,
                "bridge_target_vertex": current_bridge_target,
                "bridge_remaining_m": bridge_remaining,
                "root_vertex": root_vertex,
                "root_contact_set": sorted(actual),
                "oracle_target_finger": None,
                "planner_selected_finger": planned.selected_finger,
                "prefix_id": planned.prefix.prefix_id,
                "prefix_source": planned.prefix.source.value,
                "execution_authority": "M10_CERTIFICATE_VIA_M06",
                **planned.evidence,
              }
            )
          except (RuntimeError, ValueError, PermissionError) as error:
            replan_rejected_count += 1
            goal_id = current_goal.goal_id
            # WRIST_ADJUST is a workspace-management attempt, not evidence
            # that the required surface goal is infeasible.  If its M08-M12
            # search or M10 audit rejects, fall back to a finger edge from the
            # same retained goal at the next fresh measured state.
            if forced_break_request is not None:
              pending_workspace_break_finger = None
              force_wrist_recovery = len(actual) >= 2
              failure_count = failure_counts.get(goal_id, 0)
            elif request_wrist_adjust:
              completed_finger_prefixes_since_wrist = 0
              force_wrist_recovery = False
              failure_count = failure_counts.get(goal_id, 0)
            else:
              failure_counts[goal_id] = failure_counts.get(goal_id, 0) + 1
              failure_count = failure_counts[goal_id]
              if len(actual) >= 2:
                force_wrist_recovery = True
            events.append(
              {
                "event": "REPLAN_REJECTED",
                "time_s": timestamp,
                "goal_id": goal_id,
                "bridge_target_vertex": current_bridge_target,
                "reason": str(error),
                "failure_count": failure_count,
                "required_goal_retained": True,
                "wrist_adjust_fallback_to_finger": request_wrist_adjust,
                "forced_break_finger": forced_break_request,
                "root_contact_set": sorted(actual),
              }
            )
            if (
              forced_break_request is None
              and not request_wrist_adjust
              and failure_counts[goal_id] >= 3
            ):
              current_goal = None
              current_selection = None
            next_planning_time_s = timestamp + (
              0.05
              if request_wrist_adjust or forced_break_request is not None
              else 0.20
            )

      if plan is not None and stop_reason is None:
        try:
          command = executor.step(observation(timestamp))
          primitive_label = plan.selected_primitive
          certificate = command.certificate_id
          selected_finger = plan.selected_finger or 0
          if plan.selected_finger is not None:
            target_normal = plan.prefix.metadata.get("target_normal")
            if target_normal is not None:
              normal = np.asarray(target_normal, dtype=np.float64)
              normal /= np.linalg.norm(normal)
              ik_outward_normals[plan.selected_finger - 1] = normal
          if plan.selected_primitive == "WRIST_ADJUST":
            planned_arm = np.array(
              command.target_joint_positions_rad[:7],
              copy=True,
            )
          else:
            # A finger transaction certifies zero new arm motion.  Keep the
            # existing finite-stiffness arm setpoint (the terminal command of
            # the preceding certified WRIST transaction or the initial safe
            # hold) instead of replacing it with the loaded measured q.  The
            # latter silently removes the holding torque: in regression 7 a
            # 1.6 mrad setpoint reset moved the palm 0.51 mm and unloaded all
            # three contacts immediately after a valid SLIDE commit.
            planned_arm = np.array(
              data.ctrl[handles.arm_actuator_ids],
              copy=True,
            )
          cartesian_commands = np.array(
            command.commanded_fingertip_positions_m,
            copy=True,
          )
          hold_nominal_positions = np.array(
            command.nominal_fingertip_positions_m,
            copy=True,
          )
          if command.safe_hold:
            safe_hold_replan_count += 1
            terminal = plan.prefix.samples[-1]
            participant_error_m = (
              float(
                np.linalg.norm(
                  data.site_xpos[handles.palm_site_id]
                  - terminal.wrist_position_m
                )
              )
              if plan.selected_finger is None
              else float(
                np.linalg.norm(
                  data.site_xpos[handles.tip_site_ids[plan.selected_finger - 1]]
                  - terminal.fingertip_positions_m[plan.selected_finger - 1]
                )
              )
            )
            events.append(
              {
                "event": "M06_SAFE_HOLD_REPLAN",
                "time_s": timestamp,
                "reason": command.safety_reason,
                "prefix_id": plan.prefix.prefix_id,
                "selected_primitive": plan.selected_primitive,
                "selected_finger": plan.selected_finger,
                "root_contact_set": sorted(plan.prefix.root_contact_set),
                "actual_contact_set": sorted(confirmed_set()),
                "fingertip_forces_n": measured_forces.tolist(),
                "contact_confirmed_s": contact_confirmed_s.tolist(),
                "participant_terminal_error_m": participant_error_m,
              }
            )
            if plan.selected_primitive == "WRIST_ADJUST":
              completed_finger_prefixes_since_wrist = 0
            elif plan.selected_finger is not None:
              excluded_finger_until_s[plan.selected_finger] = (
                timestamp + config.failed_finger_cooldown_s
              )
            plan = None
            recovery_until_s = max(
              recovery_until_s,
              timestamp + config.safe_hold_recovery_s,
            )
            next_planning_time_s = recovery_until_s
          snapshot = executor.consume_barrier_snapshot()
          if snapshot is not None:
            completed_primitive = plan.selected_primitive
            completed_make = (
              snapshot.transaction_state is TransactionState.DONE
              and completed_primitive == "MAKE"
              and not bool(plan.prefix.metadata.get("make_progress", False))
            )
            events.append(
              {
                "event": "FRESH_MEASURED_MICRO_BARRIER",
                "time_s": timestamp,
                "transaction_id": snapshot.transaction_id,
                "certificate_id": snapshot.certificate_id,
                "transaction_state": snapshot.transaction_state.value,
                "actual_contact_set": sorted(snapshot.actual_contact_set),
                "prediction_suffix_executed": False,
              }
            )
            if snapshot.transaction_state is TransactionState.DONE:
              if completed_primitive == "WRIST_ADJUST":
                completed_finger_prefixes_since_wrist = 0
              elif completed_primitive == "BREAK":
                completed_finger_prefixes_since_wrist = 0
                force_wrist_recovery = True
              else:
                completed_finger_prefixes_since_wrist += 1
              if plan.selected_finger is not None:
                excluded_finger_until_s.pop(plan.selected_finger, None)
            plan = None
            # A successful micro-barrier is already the fresh state required
            # by I02.  Replan on the next physics tick so the current explorer
            # does not unload during an artificial open-loop hold.
            if completed_make:
              recovery_until_s = max(
                recovery_until_s,
                timestamp + config.safe_hold_recovery_s,
              )
            next_planning_time_s = (
              recovery_until_s
              if completed_make
              else timestamp + config.dt_s
            )
            # ARRIVE is evaluated from the same fresh measured barrier before
            # a short hold/replan cooldown can let a lightly loaded explorer
            # pad leave the neighborhood.
            if check_goal_arrival(timestamp):
              if ledger.complete:
                stop_reason = "FULL_REQUIRED_SET_COMPLETED"
              elif (
                config.maximum_goals is not None
                and len(ledger.visit_order) >= config.maximum_goals
              ):
                stop_reason = "DEVELOPMENT_GOAL_LIMIT_REACHED"
        except (RuntimeError, ValueError, PermissionError) as error:
          events.append(
            {
              "event": "M06_EXECUTION_ERROR",
              "time_s": timestamp,
              "reason": str(error),
            }
          )
          plan = None
          next_planning_time_s = timestamp + 0.10
          cartesian_commands = np.array(
            data.site_xpos[handles.tip_site_ids],
            copy=True,
          )
      else:
        stable_now = root_is_stable()
        if stable_now != (not root_stabilizing):
          root_stabilizing = not stable_now
          events.append(
            {
              "event": (
                "CONTACT_ROOT_STABILIZING"
                if root_stabilizing
                else "CONTACT_ROOT_STABLE"
              ),
              "time_s": timestamp,
              "actual_contact_set": sorted(confirmed_set()),
              "fingertip_forces_n": measured_forces.tolist(),
            }
          )
        hold_force_n = (
          config.recovery_force_n
          if root_stabilizing or timestamp < recovery_until_s
          else config.desired_force_n
        )
        # Preserve the terminal tangential reference of the last audited
        # transaction.  Inactive/free fingers are rebased to their current
        # measured sites so an old staging target cannot move them during a
        # HOLD; active contacts retain planner intent and MCC alone changes
        # their normal offset.
        hold_nominal_positions[~contact_active] = np.asarray(
          data.site_xpos[handles.tip_site_ids],
        )[~contact_active]
        output = full_mcc.step(
          hold_nominal_positions,
          -surface_normals,
          hold_force_n - measured_forces,
          contact_active,
        )
        cartesian_commands = np.stack(
          [command.position_command for command in output.commands]
        )

    safety = force_safety.step(
      fingertip_force_n=measured_forces,
      force_valid_mask=np.ones(4, dtype=np.bool_),
      history_ready=step >= acquisition_steps,
      current_q_rad=data.qpos[handles.hand_qpos_adrs],
      signed_compression_jacobian=_signed_compression_jacobian(
        handles,
        data,
        surface_normals,
      ),
    )
    current_arm = np.array(data.qpos[handles.arm_qpos_adrs], copy=True)
    issued_arm = current_arm + safety.wrist_velocity_scale * (
      planned_arm - current_arm
    )
    if step < acquisition_steps:
      issued_arm = arm_initial
    data.ctrl[handles.arm_actuator_ids] = np.clip(
      issued_arm,
      handles.arm_joint_ranges_rad[:, 0] + 0.018,
      handles.arm_joint_ranges_rad[:, 1] - 0.018,
    )

    nominal_finger_command = np.array(data.ctrl[handles.hand_actuator_ids], copy=True)
    for finger in range(4):
      joint_command = _finger_ik(
        handles,
        data,
        finger,
        cartesian_commands[finger],
        -ik_outward_normals[finger],
        gain=0.16,
        posture_gain=0.0,
      )
      actuator_indices = np.array(
        [int(name) for name in FINGERS[finger].joint_names],
        dtype=np.int32,
      )
      nominal_finger_command[actuator_indices] = joint_command
    current_finger_q = np.array(data.qpos[handles.hand_qpos_adrs], copy=True)
    if safety.override_delta_rad is not None:
      issued_finger = current_finger_q + safety.override_delta_rad
    elif safety.finger_authority_scale <= 0.0:
      issued_finger = previous_finger_command.copy()
    else:
      issued_finger = current_finger_q + safety.finger_authority_scale * (
        nominal_finger_command - current_finger_q
      )
    if step < acquisition_steps:
      issued_finger = nominal_finger_command
    issued_finger = continuity.limit_finger(issued_finger)
    issued_finger = np.clip(
      issued_finger,
      handles.hand_joint_ranges_rad[:, 0] + 0.02,
      handles.hand_joint_ranges_rad[:, 1] - 0.02,
    )
    data.ctrl[handles.hand_actuator_ids] = issued_finger
    previous_finger_command = issued_finger.copy()
    controller_latency = perf_counter() - controller_started

    physics_started = perf_counter()
    mujoco.mj_step(handles.model, data)
    # ``mj_step`` integrates qpos at the tail of the step, while derived site
    # transforms/contact geometry can still describe the preceding forward
    # pass.  I02/M10 require one atomic measured barrier, so refresh all
    # kinematics and contacts at the newly integrated qpos before exposing the
    # observation to the planner/auditor.
    mujoco.mj_forward(handles.model, data)
    physics_latency = perf_counter() - physics_started
    measured_forces, measured_positions, measured_normals, non_tip = _contact_state(
      handles,
      data,
      graph,
    )
    contact_active = np.where(
      contact_active,
      measured_forces >= 0.5 * config.contact_threshold_n,
      measured_forces >= config.contact_threshold_n,
    )
    contact_confirmed_s = np.where(
      contact_active,
      contact_confirmed_s + config.dt_s,
      0.0,
    )
    current_contact_set = measured_set()
    if current_contact_set:
      last_nonempty_set = current_contact_set
    if step >= acquisition_steps:
      traversal_steps += 1
      if current_contact_set:
        contact_supported_steps += 1
        empty_run = 0
      else:
        empty_run += 1
        maximum_empty_run = max(maximum_empty_run, empty_run)
      if current_contact_set != previous_contact_set:
        contact_mode_changes += 1
        if previous_contact_set and not current_contact_set:
          make_before_break_violations += 1
        previous_contact_set = current_contact_set
      active_indices = np.flatnonzero(contact_active)
      if len(active_indices):
        errors = measured_forces[active_indices] - config.desired_force_n
        squared_force_error += float(np.sum(errors**2))
        force_error_samples += len(active_indices)
      peak_force_n = max(peak_force_n, float(np.max(measured_forces)))
      non_tip_contact_steps += int(non_tip > 0)
      if empty_run * config.dt_s > config.contact_loss_debounce_s and stop_reason is None:
        stop_reason = "LAST_CONTACT_LOST"
      if safety.terminate_episode and stop_reason is None:
        stop_reason = safety.reason
      if not np.all(np.isfinite(data.qpos)) and stop_reason is None:
        stop_reason = "NONFINITE_STATE"

    if step % config.log_stride == 0 or stop_reason is not None:
      logs["time_s"].append(float(data.time))
      logs["arm_q_rad"].append(np.array(data.qpos[handles.arm_qpos_adrs], copy=True))
      logs["arm_command_rad"].append(np.array(data.ctrl[handles.arm_actuator_ids], copy=True))
      logs["finger_q_rad"].append(np.array(data.qpos[handles.hand_qpos_adrs], copy=True))
      logs["finger_command_rad"].append(np.array(data.ctrl[handles.hand_actuator_ids], copy=True))
      logs["palm_pose_world"].append(
        np.concatenate(
          (
            data.site_xpos[handles.palm_site_id],
            _quaternion_from_matrix(data.site_xmat[handles.palm_site_id]),
          )
        )
      )
      logs["fingertip_positions_world_m"].append(
        np.array(data.site_xpos[handles.tip_site_ids], copy=True)
      )
      logs["contact_positions_world_m"].append(measured_positions.copy())
      logs["contact_normals_world"].append(measured_normals.copy())
      logs["fingertip_forces_n"].append(measured_forces.copy())
      logs["contact_active"].append(contact_active.copy())
      logs["goal_id"].append(-1 if current_goal is None else current_goal.goal_id)
      logs["goal_vertex"].append(-1 if current_goal is None else current_goal.vertex_index)
      logs["bridge_target_vertex"].append(current_bridge_target)
      logs["selected_finger"].append(selected_finger)
      logs["coverage_fraction"].append(ledger.completion_fraction)
      logs["covered_area_fraction"].append(ledger.covered_area_fraction)
      logs["transaction_state"].append(executor.transaction_state.value)
      logs["primitive"].append(primitive_label)
      logs["certificate_id"].append(certificate)
      logs["guard_reason"].append(stop_reason or safety.reason)
      logs["controller_latency_s"].append(controller_latency)
      logs["physics_latency_s"].append(physics_latency)

    if stop_reason is not None:
      events.append(
        {
          "event": "EPISODE_STOP",
          "time_s": float(data.time),
          "reason": stop_reason,
          "visited_goal_count": len(ledger.visit_order),
          "required_goal_count": graph.required_goal_count,
          "coverage_fraction": ledger.completion_fraction,
          "covered_area_fraction": ledger.covered_area_fraction,
        }
      )
      break

  if stop_reason is None:
    stop_reason = "MAXIMUM_DURATION_REACHED"
  trace = I04BunnyTrace(
    config=config,
    time_s=np.asarray(logs["time_s"], dtype=np.float64),
    arm_q_rad=np.asarray(logs["arm_q_rad"], dtype=np.float64),
    arm_command_rad=np.asarray(logs["arm_command_rad"], dtype=np.float64),
    finger_q_rad=np.asarray(logs["finger_q_rad"], dtype=np.float64),
    finger_command_rad=np.asarray(logs["finger_command_rad"], dtype=np.float64),
    palm_pose_world=np.asarray(logs["palm_pose_world"], dtype=np.float64),
    fingertip_positions_world_m=np.asarray(
      logs["fingertip_positions_world_m"], dtype=np.float64
    ),
    contact_positions_world_m=np.asarray(
      logs["contact_positions_world_m"], dtype=np.float64
    ),
    contact_normals_world=np.asarray(logs["contact_normals_world"], dtype=np.float64),
    fingertip_forces_n=np.asarray(logs["fingertip_forces_n"], dtype=np.float64),
    contact_active=np.asarray(logs["contact_active"], dtype=np.bool_),
    goal_id=np.asarray(logs["goal_id"], dtype=np.int32),
    goal_vertex=np.asarray(logs["goal_vertex"], dtype=np.int32),
    bridge_target_vertex=np.asarray(logs["bridge_target_vertex"], dtype=np.int32),
    selected_finger=np.asarray(logs["selected_finger"], dtype=np.int8),
    coverage_fraction=np.asarray(logs["coverage_fraction"], dtype=np.float64),
    covered_area_fraction=np.asarray(logs["covered_area_fraction"], dtype=np.float64),
    transaction_state=np.asarray(logs["transaction_state"], dtype="U24"),
    primitive=np.asarray(logs["primitive"], dtype="U24"),
    certificate_id=np.asarray(logs["certificate_id"], dtype="U64"),
    guard_reason=np.asarray(logs["guard_reason"], dtype="U96"),
    controller_latency_s=np.asarray(logs["controller_latency_s"], dtype=np.float64),
    physics_latency_s=np.asarray(logs["physics_latency_s"], dtype=np.float64),
    events=events,
  )
  force_rmse = (
    float(np.sqrt(squared_force_error / force_error_samples))
    if force_error_samples
    else float("nan")
  )
  summary = {
    "schema_version": TRACE_SCHEMA_VERSION,
    "evaluator_version": EVALUATOR_VERSION,
    "surface_model_version": SURFACE_MODEL_VERSION,
    "method": "EXPLICIT_MCC_BASELINE",
    "dpref_enabled": False,
    "gpis_enabled": False,
    "collision_geometry": "MUJOCO_MESH_SDF_NONCONVEX",
    "seed": config.seed,
    "stop_reason": stop_reason,
    "simulated_duration_s": float(data.time),
    "traversal_duration_s": max(0.0, float(data.time) - config.acquisition_s),
    "required_goal_count": graph.required_goal_count,
    "visited_goal_count": len(ledger.visit_order),
    "visited_goal_ids": list(ledger.visit_order),
    "coverage_fraction": ledger.completion_fraction,
    "covered_area_fraction": ledger.covered_area_fraction,
    "mesh_area_m2": graph.total_area_m2,
    "coverage_radius_m": graph.realized_cover_radius_m,
    "arrival_geodesic_tolerance_m": config.arrival_tolerance_m,
    "arrival_normal_tolerance_rad": config.normal_tolerance_rad,
    "contact_continuity_fraction": (
      contact_supported_steps / traversal_steps if traversal_steps else 0.0
    ),
    "maximum_contact_gap_s": maximum_empty_run * config.dt_s,
    "force_rmse_n": force_rmse,
    "peak_fingertip_force_n": peak_force_n,
    "contact_mode_change_count": contact_mode_changes,
    "make_before_break_violation_count": make_before_break_violations,
    "non_tip_contact_fraction": (
      non_tip_contact_steps / traversal_steps if traversal_steps else 0.0
    ),
    "prefix_count": committed_prefix_count,
    "planning_attempt_count": transaction_sequence,
    "committed_wrist_adjust_count": committed_wrist_adjust_count,
    "safe_hold_replan_count": safe_hold_replan_count,
    "replan_rejected_count": replan_rejected_count,
    "planner_selected_finger_counts": assignment_counts.tolist(),
    "planner_latency_s": {
      "mean": float(np.mean(planner_latencies)) if planner_latencies else 0.0,
      "p95": float(np.percentile(planner_latencies, 95)) if planner_latencies else 0.0,
      "max": float(np.max(planner_latencies)) if planner_latencies else 0.0,
    },
    "m10_audit_latency_s": {
      "mean": float(np.mean(audit_latencies)) if audit_latencies else 0.0,
      "p95": float(np.percentile(audit_latencies, 95)) if audit_latencies else 0.0,
      "max": float(np.max(audit_latencies)) if audit_latencies else 0.0,
    },
    "m11_search_latency_s": {
      "mean": float(np.mean(search_latencies)) if search_latencies else 0.0,
      "p95": float(np.percentile(search_latencies, 95)) if search_latencies else 0.0,
      "max": float(np.max(search_latencies)) if search_latencies else 0.0,
    },
    "prediction_suffix_execution_count": 0,
    "execution_authority": "M10_CERTIFICATE_ONLY_THROUGH_M06",
    "oracle_outputs_finger_id": False,
    "route_root_authority": "MEASURED_CONTACT_AT_EVERY_MICRO_BARRIER",
  }
  return trace, summary


def save_trace(path: str | Path, trace: I04BunnyTrace) -> Path:
  output = Path(path)
  output.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(output, **trace.npz_payload())
  return output


__all__ = [
  "EVALUATOR_VERSION",
  "I04BunnyConfig",
  "I04BunnyTrace",
  "TRACE_SCHEMA_VERSION",
  "run_i04_bunny",
  "save_trace",
]
