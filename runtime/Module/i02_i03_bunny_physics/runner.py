"""Frozen MuJoCo runner for I02/I03 on Bunny (Geometry Oracle + MCC only).

The runner deliberately keeps planning and execution authority separate:
M11/M12 may rank or reject candidates, but only an M10-certified edge-zero
prefix is ever handed to M06.  Every subsequent prefix is rebuilt from a
fresh measured micro-barrier state.
"""

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
  _contact_state,
  _finger_ik,
  _logical_q,
  _max_true_run,
  _quaternion_from_matrix,
  _signed_compression_jacobian,
  _surface_targets,
)
from Module.i01_bunny_physics.surface import (
  BunnyHeightField,
  canonical_bunny_heightfield,
)
from Module.i02_i03_bunny_physics.core import (
  EVALUATOR_VERSION,
  I02_FINGER,
  I03_HANDOVER_FINGER,
  SURFACE_MODEL_VERSION,
  TRACE_SCHEMA_VERSION,
  I02I03BunnyConfig,
  audit_prefix,
  evaluate_actual_shadow,
  make_planner_bundle,
  optimize_prefix,
  planned_cumulative_distance_m,
  planned_path_coordinate_m,
  search_i03_prefix,
)
from Module.module_2_fingertip_mcc import (
  FingertipMCC,
  FullRobotFingertipMCC,
  MCCConfig,
)
from Module.module_3_runtime_guards import (
  CommandContinuityLimiter,
  ForceSafetyConfig,
  ForceSafetyExecutor,
)
from Module.module_4_whole_hand_mcc.robot_control import PalmPoseIK, PalmPoseIKConfig
from Module.module_6_prefix_executor import (
  ExecutorConfig,
  ExecutorObservation,
  MCCBaselineAdapter,
  PlannedPrefix,
  TransactionState,
  TransactionalPrefixExecutor,
)
from Module.module_7_contact_mode_graph import ContactPrimitive, PrimitiveKind


@dataclass(slots=True)
class I02I03BunnyTrace:
  """Complete physical trace; JSON events hold certificate/search evidence."""

  config: I02I03BunnyConfig
  time_s: NDArray[np.float64]
  arm_q_rad: NDArray[np.float64]
  arm_dq_rad_s: NDArray[np.float64]
  arm_command_rad: NDArray[np.float64]
  finger_q_rad: NDArray[np.float64]
  finger_dq_rad_s: NDArray[np.float64]
  finger_command_rad: NDArray[np.float64]
  palm_pose_world: NDArray[np.float64]
  planned_palm_pose_world: NDArray[np.float64]
  fingertip_positions_world_m: NDArray[np.float64]
  contact_positions_world_m: NDArray[np.float64]
  fingertip_forces_n: NDArray[np.float64]
  hfield_valid_contacts: NDArray[np.bool_]
  mesh_valid_contacts: NDArray[np.bool_]
  mesh_residual_m: NDArray[np.float64]
  planned_path_coordinate_m: NDArray[np.float64]
  planned_cumulative_distance_m: NDArray[np.float64]
  actual_path_coordinate_m: NDArray[np.float64]
  actual_cumulative_distance_m: NDArray[np.float64]
  controller_latency_s: NDArray[np.float64]
  physics_latency_s: NDArray[np.float64]
  optimizer_latency_s: NDArray[np.float64]
  search_latency_s: NDArray[np.float64]
  shadow_latency_s: NDArray[np.float64]
  audit_latency_s: NDArray[np.float64]
  minimum_joint_margin_rad: NDArray[np.float64]
  non_tip_contact_count: NDArray[np.int32]
  guard_reason: NDArray[np.str_]
  transaction_phase: NDArray[np.str_]
  transaction_state: NDArray[np.str_]
  certificate_id: NDArray[np.str_]
  selected_sequence: NDArray[np.str_]
  terminal_viability: NDArray[np.str_]
  authority_violation: NDArray[np.bool_]
  shadow_execution_authority: NDArray[np.bool_]
  suffix_command_count: NDArray[np.int32]
  events: list[dict[str, Any]]

  def npz_payload(self) -> dict[str, NDArray[Any]]:
    return {
      "time_s": self.time_s,
      "arm_q_rad": self.arm_q_rad,
      "arm_dq_rad_s": self.arm_dq_rad_s,
      "arm_command_rad": self.arm_command_rad,
      "finger_q_rad": self.finger_q_rad,
      "finger_dq_rad_s": self.finger_dq_rad_s,
      "finger_command_rad": self.finger_command_rad,
      "palm_pose_world": self.palm_pose_world,
      "planned_palm_pose_world": self.planned_palm_pose_world,
      "fingertip_positions_world_m": self.fingertip_positions_world_m,
      "contact_positions_world_m": self.contact_positions_world_m,
      "fingertip_forces_n": self.fingertip_forces_n,
      "hfield_valid_contacts": self.hfield_valid_contacts,
      "mesh_valid_contacts": self.mesh_valid_contacts,
      "mesh_residual_m": self.mesh_residual_m,
      "planned_path_coordinate_m": self.planned_path_coordinate_m,
      "planned_cumulative_distance_m": self.planned_cumulative_distance_m,
      "actual_path_coordinate_m": self.actual_path_coordinate_m,
      "actual_cumulative_distance_m": self.actual_cumulative_distance_m,
      "controller_latency_s": self.controller_latency_s,
      "physics_latency_s": self.physics_latency_s,
      "optimizer_latency_s": self.optimizer_latency_s,
      "search_latency_s": self.search_latency_s,
      "shadow_latency_s": self.shadow_latency_s,
      "audit_latency_s": self.audit_latency_s,
      "minimum_joint_margin_rad": self.minimum_joint_margin_rad,
      "non_tip_contact_count": self.non_tip_contact_count,
      "guard_reason": self.guard_reason,
      "transaction_phase": self.transaction_phase,
      "transaction_state": self.transaction_state,
      "certificate_id": self.certificate_id,
      "selected_sequence": self.selected_sequence,
      "terminal_viability": self.terminal_viability,
      "authority_violation": self.authority_violation,
      "shadow_execution_authority": self.shadow_execution_authority,
      "suffix_command_count": self.suffix_command_count,
    }


def _joint_margin(handles: Any, data: mujoco.MjData) -> float:
  q = _logical_q(handles, data)
  lower = np.concatenate(
    (handles.arm_joint_ranges_rad[:, 0], handles.hand_joint_ranges_rad[:, 0])
  )
  upper = np.concatenate(
    (handles.arm_joint_ranges_rad[:, 1], handles.hand_joint_ranges_rad[:, 1])
  )
  return float(np.min(np.minimum(q - lower, upper - q)))


def _stable_mode_evidence(
  contacts: NDArray[np.bool_],
  time_s: NDArray[np.float64],
  dt_s: float,
) -> tuple[list[dict[str, Any]], bool]:
  masks = np.sum(
    contacts.astype(np.int32) * np.array([1, 2, 4, 8], dtype=np.int32)[None, :],
    axis=1,
  )
  minimum_run = int(round(0.040 / dt_s))
  runs: list[dict[str, Any]] = []
  start = 0
  for index in range(1, len(masks) + 1):
    if index == len(masks) or masks[index] != masks[start]:
      if index - start >= minimum_run:
        runs.append(
          {
            "mask": int(masks[start]),
            "start_s": float(time_s[start]),
            "duration_s": float((index - start) * dt_s),
          }
        )
      start = index
  stable = [int(run["mask"]) for run in runs]
  handover = any(
    stable[left] == 15
    and stable[middle].bit_count() == 3
    and stable[right] == 15
    for left in range(len(stable))
    for middle in range(left + 1, len(stable))
    for right in range(middle + 1, len(stable))
  )
  return runs, handover


def _supported_segment_traversal(
  trace: I02I03BunnyTrace,
  nonempty: NDArray[np.bool_],
) -> tuple[float, list[dict[str, float]]]:
  """Noise-robust signed segment displacement weighted by exact support.

  Endpoint medians suppress MuJoCo servo jitter.  Weighting each scheduled
  monotone segment by its exact nonempty-contact fraction makes unsupported
  motion contribute zero without summing high-frequency pose noise.
  """

  coordinate = trace.actual_path_coordinate_m
  segments = ((3.0, 4.0, 1.0), (7.0, 12.0, 1.0), (12.0, 17.0, -1.0))
  evidence: list[dict[str, float]] = []
  total = 0.0
  for start_s, stop_s, direction in segments:
    start_window = (trace.time_s >= start_s - 0.050) & (trace.time_s <= start_s)
    stop_window = (trace.time_s >= stop_s - 0.050) & (trace.time_s <= stop_s)
    motion_window = (trace.time_s >= start_s) & (trace.time_s <= stop_s)
    start_value = float(np.median(coordinate[start_window]))
    stop_value = float(np.median(coordinate[stop_window]))
    displacement = max(0.0, direction * (stop_value - start_value))
    support_fraction = float(np.mean(nonempty[motion_window]))
    supported = displacement * support_fraction
    total += supported
    evidence.append(
      {
        "start_s": start_s,
        "stop_s": stop_s,
        "signed_actual_displacement_m": displacement,
        "exact_support_fraction": support_fraction,
        "supported_displacement_m": supported,
      }
    )
  return total, evidence


def run_i02_i03_bunny(
  config: I02I03BunnyConfig,
) -> tuple[I02I03BunnyTrace, dict[str, Any], str]:
  """Run one deterministic formal I02/I03 physical episode."""

  bunny = canonical_bunny_heightfield()
  handles = build_full_robot(
    FullRobotModelConfig(
      surface="bunny",
      timestep_s=config.dt_s,
      gravity_m_s2=0.0,
      arm_kp=1800.0,
      arm_damping_ratio=0.9,
      object_offset_x_m=config.object_offset_x_m,
      object_offset_y_m=config.object_offset_y_m,
      object_offset_z_m=config.object_offset_z_m,
      bunny_visual_mesh_path=config.visual_mesh_path,
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

  initial_pose = np.concatenate(
    (
      data.site_xpos[handles.palm_site_id].copy(),
      _quaternion_from_matrix(data.site_xmat[handles.palm_site_id]),
    )
  )
  command_continuity = CommandContinuityLimiter()
  command_continuity.reset(
    finger_command_rad=data.ctrl[handles.hand_actuator_ids],
    wrist_pose_world=initial_pose,
  )
  contact_anchor_xy = np.array(data.site_xpos[handles.tip_site_ids, :2], copy=True)
  _, _, initial_valid = _surface_targets(
    handles,
    data,
    bunny,
    contact_anchor_xy,
  )
  if not np.all(initial_valid):
    raise RuntimeError("initial fingertip footprint is outside Bunny")
  initial_surface_mean = float(
    np.mean(
      [
        bunny.query(*(xy - handles.object_position_m[:2]))[0]
        + handles.object_position_m[2]
        for xy in contact_anchor_xy
      ]
    )
  )

  pose_ik = PalmPoseIK(
    handles,
    PalmPoseIKConfig(gain=0.32, damping=0.018, max_joint_step_rad=0.02),
  )
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
  traversal_mcc = FullRobotFingertipMCC(
    tuple(
      FingertipMCC(
        MCCConfig(
          virtual_mass=0.05,
          damping=10.0,
          stiffness=15.0,
          dt_s=config.dt_s,
          max_offset_m=0.020,
          max_velocity_m_s=0.08,
          max_acceleration_m_s2=30.0,
        )
      )
      for _ in range(4)
    )
  )
  force_safety = ForceSafetyExecutor(
    ForceSafetyConfig(
      joint_lower_rad=handles.hand_joint_ranges_rad[:, 0],
      joint_upper_rad=handles.hand_joint_ranges_rad[:, 1],
      dt_s=config.dt_s,
      soft_force_n=(6.0 if config.cell.startswith("i02") else 7.0),
      hard_force_n=config.force_limit_n,
      recover_force_n=2.5,
      rapid_loading_rate_n_s=5000.0,
      rapid_loading_min_force_n=7.0,
    )
  )
  traversal_force_safety = ForceSafetyExecutor(
    ForceSafetyConfig(
      joint_lower_rad=handles.hand_joint_ranges_rad[:, 0],
      joint_upper_rad=handles.hand_joint_ranges_rad[:, 1],
      dt_s=config.dt_s,
      soft_force_n=7.0,
      hard_force_n=config.force_limit_n,
      recover_force_n=2.5,
      rapid_loading_rate_n_s=5000.0,
      rapid_loading_min_force_n=7.0,
    )
  )
  executor = TransactionalPrefixExecutor(
    ExecutorConfig(
      # I03 deliberately scores a locally-linear terminal prediction on a
      # nonlinear plant; its 4 mm physical ball matches the I01 belly-pad
      # convention while I02 keeps a strict 1 mm error-sensitive barrier.
      completion_tolerance_m=(0.0015 if config.cell.startswith("i02") else 0.0040),
      default_timeout_s=1.5,
      desired_anchor_force_n=config.desired_force_n,
      root_state_tolerance=1e-9,
      make_contact_is_terminal=True,
    ),
    mcc_adapter=MCCBaselineAdapter(full_mcc),
  )

  count = int(round(config.duration_s / config.dt_s))
  time_s = np.arange(count, dtype=np.float64) * config.dt_s
  arm_q = np.zeros((count, 7))
  arm_dq = np.zeros((count, 7))
  arm_command = np.zeros((count, 7))
  finger_q = np.zeros((count, 16))
  finger_dq = np.zeros((count, 16))
  finger_command = np.zeros((count, 16))
  palm = np.zeros((count, 7))
  planned_palm = np.zeros((count, 7))
  tips = np.zeros((count, 4, 3))
  contact_positions = np.zeros((count, 4, 3))
  forces_log = np.zeros((count, 4))
  hfield_contacts = np.zeros((count, 4), dtype=np.bool_)
  planned_coordinate = np.zeros(count)
  planned_cumulative = np.zeros(count)
  actual_coordinate = np.zeros(count)
  actual_cumulative = np.zeros(count)
  controller_latency = np.zeros(count)
  physics_latency = np.zeros(count)
  optimizer_latency = np.zeros(count)
  search_latency = np.zeros(count)
  shadow_latency = np.zeros(count)
  audit_latency = np.zeros(count)
  joint_margin = np.zeros(count)
  non_tip = np.zeros(count, dtype=np.int32)
  guard_reason = np.full(count, "NONE", dtype="U128")
  transaction_phase = np.full(count, "NORMAL", dtype="U40")
  transaction_state = np.full(count, TransactionState.IDLE.value, dtype="U24")
  certificate_id = np.full(count, "NONE", dtype="U64")
  selected_sequence = np.full(count, "NONE", dtype="U96")
  terminal_viability = np.full(count, "NOT_EVALUATED", dtype="U24")
  authority_violation = np.zeros(count, dtype=np.bool_)
  shadow_execution_authority = np.zeros(count, dtype=np.bool_)
  suffix_command_count = np.zeros(count, dtype=np.int32)
  events: list[dict[str, Any]] = []

  measured_forces = np.zeros(4)
  measured_positions = np.zeros((4, 3))
  measured_hfield_valid = np.zeros(4, dtype=np.bool_)
  contact_active = np.zeros(4, dtype=np.bool_)
  contact_confirmed_s = np.zeros(4)
  previous_finger_command = np.array(data.ctrl[handles.hand_actuator_ids], copy=True)
  empty_run = 0
  stop_reason: str | None = None
  stopped_coordinate = 0.0
  phase = "NORMAL"
  current_finger: int | None = None
  current_prefix: PlannedPrefix | None = None
  current_optimizer_latency = 0.0
  current_audit_latency = 0.0
  current_search_latency = 0.0
  current_shadow_latency = 0.0
  current_certificate_id = "NONE"
  current_commit_timestamp = 0.0
  current_selected_sequence = "NONE"
  current_terminal_viability = "NOT_EVALUATED"
  pending_action: tuple[str, PrimitiveKind, int, NDArray[np.float64], float] | None = None
  decision_started = False
  mechanism_complete = False
  dead_end = False
  transaction_sequence = 0
  last_barrier_certificate = "INITIAL_MEASURED_ROOT"
  reposition_origin_xy: NDArray[np.float64] | None = None
  reposition_goal_xy: NDArray[np.float64] | None = None
  short_segment_index = 0
  traversal_compliance_started = False
  last_nonempty_contact_set = frozenset({1, 2, 3, 4})

  def measured_set() -> frozenset[int]:
    return frozenset(int(index + 1) for index in np.flatnonzero(contact_active))

  def observation(timestamp: float, normals: NDArray[np.float64]) -> ExecutorObservation:
    contacts = measured_set()
    if not contacts and empty_run * config.dt_s <= 0.010 + 1e-12:
      # Protocol-level contact authority is a 10 ms debounced measurement,
      # never a predicted set.  This prevents one MuJoCo contact-solver tick
      # from bypassing the frozen all-contact-loss criterion.
      contacts = last_nonempty_contact_set
    if (
      phase.endswith("MAKE")
      and current_finger is not None
      and contact_confirmed_s[current_finger - 1] < 0.040
    ):
      contacts = contacts - {current_finger}
    return ExecutorObservation(
      timestamp_s=timestamp,
      surface_model_version=SURFACE_MODEL_VERSION,
      wrist_position_m=data.site_xpos[handles.palm_site_id],
      fingertip_positions_m=data.site_xpos[handles.tip_site_ids],
      joint_positions_rad=_logical_q(handles, data),
      fingertip_forces_n=measured_forces,
      outward_normals=normals,
      actual_contact_set=contacts,
    )

  def confirmations() -> dict[int, float]:
    return {
      int(index + 1): float(contact_confirmed_s[index])
      for index in np.flatnonzero(contact_active)
    }

  def commit_certified(
    *,
    action_label: str,
    primitive: ContactPrimitive,
    prefix: PlannedPrefix,
    certificate: Any,
    evidence: dict[str, Any],
    timestamp: float,
    normals: NDArray[np.float64],
    optimizer_time_s: float,
  ) -> None:
    nonlocal phase, current_finger, current_prefix
    nonlocal current_optimizer_latency, current_audit_latency
    nonlocal current_certificate_id, current_commit_timestamp
    full_mcc.reset()
    executor.commit(prefix, certificate, observation(timestamp, normals))
    phase = action_label
    current_finger = primitive.finger_id
    current_prefix = prefix
    current_optimizer_latency = float(optimizer_time_s)
    current_audit_latency = float(evidence["audit_latency_s"])
    current_certificate_id = str(evidence["certificate_id"])
    current_commit_timestamp = timestamp
    events.append(
      {
        "event": "CERTIFIED_COMMIT",
        "time_s": timestamp,
        "phase": action_label,
        "primitive": primitive.key,
        "prefix_id": prefix.prefix_id,
        "root_authority": "MEASURED_MICRO_BARRIER",
        "fresh_root_after_certificate": last_barrier_certificate,
        "root_contact_set": sorted(prefix.root_contact_set),
        "expected_terminal_contact_set": sorted(prefix.expected_terminal_contact_set),
        "prefix_source": prefix.source.value,
        "execution_authority": "M10_EXACT_PREFIX_AUDIT",
        "prediction_suffix_command_count": 0,
        "terminal_fingertip_position_m": (
          prefix.samples[-1].fingertip_positions_m[primitive.finger_id - 1].tolist()
          if primitive.finger_id is not None
          else None
        ),
        **evidence,
      }
    )

  def optimize_audit_commit(
    action_label: str,
    kind: PrimitiveKind,
    finger: int,
    target: NDArray[np.float64],
    progress_gain: float,
    timestamp: float,
    normals: NDArray[np.float64],
  ) -> None:
    nonlocal transaction_sequence
    actual = measured_set()
    if not actual:
      raise RuntimeError("cannot plan from an empty measured root")
    transaction_sequence += 1
    primitive = ContactPrimitive(kind, finger)
    bundle = make_planner_bundle(handles, data, bunny, actual, config)
    prefix, optimizer_evidence = optimize_prefix(
      bundle,
      primitive,
      target,
      prefix_id=f"{config.cell}-{transaction_sequence:02d}-{kind.value.lower()}-{finger}",
      progress_gain_m=progress_gain,
    )
    certificate, audit_evidence = audit_prefix(
      handles,
      data,
      bunny,
      config,
      bundle,
      prefix,
      timestamp_s=timestamp,
      replacement_confirmation_s=confirmations(),
    )
    commit_certified(
      action_label=action_label,
      primitive=primitive,
      prefix=prefix,
      certificate=certificate,
      evidence={**optimizer_evidence, **audit_evidence},
      timestamp=timestamp,
      normals=normals,
      optimizer_time_s=float(optimizer_evidence["optimizer_latency_s"]),
    )

  for step, timestamp_value in enumerate(time_s):
    timestamp = float(timestamp_value)
    started = perf_counter()
    if timestamp >= 7.0 and not traversal_compliance_started:
      # The decision/transaction fixture uses the unchanged I01 actuator
      # plant.  Once the planning plateau closes, the shared long-traversal
      # safety profile lowers only the hand position-servo stiffness.  This is
      # identical in all four cells and leaves M11/M12 comparisons untouched.
      hand_ids = handles.hand_actuator_ids
      handles.model.actuator_gainprm[hand_ids, 0] = 16.0
      handles.model.actuator_biasprm[hand_ids, 1] = -16.0
      handles.model.actuator_biasprm[hand_ids, 2] *= np.sqrt(16.0 / 22.0)
      for source, destination in zip(full_mcc.controllers, traversal_mcc.controllers):
        destination.reset(
          offset_m=source.state.offset_m,
          velocity_m_s=source.state.velocity_m_s,
        )
      # FullRobotFingertipMCC resets on active-mask transitions.  Preserve the
      # measured-contact ownership during this parameter-only handoff.
      traversal_mcc._active[:] = full_mcc.active_mask  # noqa: SLF001
      traversal_compliance_started = True
      events.append(
        {
          "event": "TRAVERSAL_COMPLIANCE_PROFILE",
          "time_s": timestamp,
          "hand_kp": 16.0,
          "hand_damping_ratio": 1.5,
          "mcc_virtual_mass": 0.05,
          "mcc_damping": 10.0,
          "mcc_stiffness": 15.0,
          "comparison_shared": True,
        }
      )
    coordinate = planned_path_coordinate_m(timestamp)
    if stop_reason is not None:
      coordinate = stopped_coordinate
    target_xy = contact_anchor_xy.copy()
    target_xy[:, 0] -= coordinate
    centers, normals, target_valid = _surface_targets(handles, data, bunny, target_xy)
    if not np.all(target_valid) and stop_reason is None:
      stop_reason = "TARGET_LEFT_BUNNY"
      stopped_coordinate = coordinate
    surface_mean = float(
      np.mean(
        [
          bunny.query(*(xy - handles.object_position_m[:2]))[0]
          + handles.object_position_m[2]
          for xy in target_xy
        ]
      )
    )
    planned_pose = initial_pose.copy()
    planned_pose[0] -= coordinate
    planned_pose[2] += surface_mean - initial_surface_mean

    if pending_action is not None and stop_reason is None:
      label, kind, finger, target, gain = pending_action
      pending_action = None
      try:
        optimize_audit_commit(
          label,
          kind,
          finger,
          target,
          gain,
          timestamp,
          normals,
        )
      except (RuntimeError, PermissionError, ValueError) as error:
        stop_reason = f"AUDIT_OR_COMMIT_REJECTED:{error}"
        stopped_coordinate = coordinate

    if (
      not decision_started
      and stop_reason is None
      and timestamp >= 4.10
      and measured_set() == frozenset({1, 2, 3, 4})
      and float(np.min(contact_confirmed_s)) >= 0.05
    ):
      decision_started = True
      try:
        if config.cell.startswith("i02"):
          reposition_origin_xy = np.array(
            data.site_xpos[handles.tip_site_ids[I02_FINGER - 1], :2],
            copy=True,
          )
          reposition_goal_xy = reposition_origin_xy + np.array(
            [config.reposition_total_m, 0.0]
          )
          optimize_audit_commit(
            "I02_BREAK",
            PrimitiveKind.BREAK,
            I02_FINGER,
            np.array(data.site_xpos[handles.tip_site_ids[I02_FINGER - 1]], copy=True),
            0.0,
            timestamp,
            normals,
          )
        else:
          bundle = make_planner_bundle(
            handles,
            data,
            bunny,
            measured_set(),
            config,
          )
          use_shadow = config.cell == "i03_shadow"
          search_result, predicted_shadow, search_evidence = search_i03_prefix(
            bundle,
            config,
            use_shadow=use_shadow,
          )
          prefix = search_result.committed_prefix_candidate
          assert prefix is not None
          primitive = ContactPrimitive(prefix.primitive_kind, prefix.finger_id)
          certificate, audit_evidence = audit_prefix(
            handles,
            data,
            bunny,
            config,
            bundle,
            prefix,
            timestamp_s=timestamp,
            replacement_confirmation_s=confirmations(),
          )
          current_search_latency = float(search_evidence["search_latency_s"])
          current_shadow_latency = float(search_evidence["predicted_shadow_latency_s"])
          current_selected_sequence = " -> ".join(search_evidence["selected_sequence"])
          current_terminal_viability = str(
            search_evidence["predicted_terminal_viability"]
          )
          events.append(
            {
              "event": "I03_SEARCH_DECISION",
              "time_s": timestamp,
              "cell": config.cell,
              "shadow_execution_authority": predicted_shadow.execution_authority,
              **search_evidence,
            }
          )
          commit_certified(
            action_label="I03_SELECTED_EDGE",
            primitive=primitive,
            prefix=prefix,
            certificate=certificate,
            evidence=audit_evidence,
            timestamp=timestamp,
            normals=normals,
            optimizer_time_s=float(
              search_evidence["selected_optimizer_latency_s"]
            ),
          )
      except (RuntimeError, PermissionError, ValueError) as error:
        stop_reason = f"PLANNING_OR_COMMIT_REJECTED:{error}"
        stopped_coordinate = coordinate

    current_pose = np.concatenate(
      (
        data.site_xpos[handles.palm_site_id].copy(),
        _quaternion_from_matrix(data.site_xmat[handles.palm_site_id]),
      )
    )
    transaction_active = phase != "NORMAL" and current_prefix is not None
    active_force_safety = force_safety
    safety = active_force_safety.step(
      fingertip_force_n=measured_forces,
      force_valid_mask=np.ones(4, dtype=np.bool_),
      history_ready=True,
      current_q_rad=data.qpos[handles.hand_qpos_adrs],
      signed_compression_jacobian=_signed_compression_jacobian(handles, data, normals),
    )
    issued_pose = planned_pose.copy()
    if safety.wrist_velocity_scale < 1.0:
      issued_pose[:3] = current_pose[:3] + safety.wrist_velocity_scale * (
        planned_pose[:3] - current_pose[:3]
      )
    if stop_reason is not None:
      issued_pose = current_pose
    data.ctrl[handles.arm_actuator_ids] = pose_ik.solve(data, issued_pose)

    if transaction_active and stop_reason is None:
      try:
        command = executor.step(observation(timestamp, normals))
        cartesian_commands = command.commanded_fingertip_positions_m
        transaction_state[step] = command.transaction_state.value
        certificate_id[step] = command.certificate_id
        if command.safe_hold:
          stop_reason = command.safety_reason or "EXECUTOR_SAFE_HOLD"
          stopped_coordinate = coordinate
      except (RuntimeError, PermissionError, ValueError) as error:
        authority_violation[step] = True
        stop_reason = f"EXECUTOR_ERROR:{error}"
        stopped_coordinate = coordinate
        cartesian_commands = np.array(data.site_xpos[handles.tip_site_ids], copy=True)
    else:
      if stop_reason is not None and stop_reason != "DEAD_END":
        cartesian_commands = np.array(data.site_xpos[handles.tip_site_ids], copy=True)
      else:
        force_errors = config.desired_force_n - measured_forces
        active_mcc = traversal_mcc if traversal_compliance_started else full_mcc
        output = active_mcc.step(
          centers,
          -normals,
          force_errors,
          np.ones(4, dtype=np.bool_),
        )
        cartesian_commands = np.stack(
          [command.position_command for command in output.commands]
        )

    nominal_finger_command = np.array(data.ctrl[handles.hand_actuator_ids], copy=True)
    for finger in range(4):
      joint_command = _finger_ik(
        handles,
        data,
        finger,
        cartesian_commands[finger],
        -normals[finger],
        damping=(0.006 if transaction_active and current_finger == finger + 1 else 0.012),
        gain=(
          0.50
          if transaction_active and current_finger == finger + 1
          else 0.18
        ),
        orientation_weight=(
          0.003
          if transaction_active and current_finger == finger + 1
          else 0.012
        ),
      )
      actuator_indices = np.array(
        [int(name) for name in FINGERS[finger].joint_names],
        dtype=np.int32,
      )
      if (
        transaction_active
        and phase == "I03_SELECTED_EDGE"
        and current_prefix is not None
        and current_finger == finger + 1
      ):
        elapsed = max(0.0, timestamp - current_commit_timestamp)
        if elapsed >= current_prefix.duration_s:
          planned_q = current_prefix.samples[-1].joint_positions_rad
        else:
          planned_q = current_prefix.samples[-1].joint_positions_rad
          for left, right in zip(
            current_prefix.samples,
            current_prefix.samples[1:],
          ):
            if elapsed <= right.time_s:
              alpha = (elapsed - left.time_s) / (right.time_s - left.time_s)
              planned_q = (
                (1.0 - alpha) * left.joint_positions_rad
                + alpha * right.joint_positions_rad
              )
              break
        planned_finger_q = planned_q[7 + actuator_indices]
        # Edge zero already contains the audited joint path.  Blend it with
        # the Cartesian MCC correction rather than resolving the same slide
        # through a second, potentially different nonlinear IK branch.
        joint_command = 0.90 * planned_finger_q + 0.10 * joint_command
      nominal_finger_command[actuator_indices] = joint_command
    current_finger_q = np.array(data.qpos[handles.hand_qpos_adrs], copy=True)
    if safety.override_delta_rad is not None:
      issued_finger_command = current_finger_q + safety.override_delta_rad
    elif stop_reason is not None and stop_reason != "DEAD_END":
      issued_finger_command = previous_finger_command.copy()
    elif safety.finger_authority_scale <= 0.0:
      issued_finger_command = previous_finger_command.copy()
    else:
      issued_finger_command = current_finger_q + safety.finger_authority_scale * (
        nominal_finger_command - current_finger_q
      )
    issued_finger_command = np.clip(
      issued_finger_command,
      handles.hand_joint_ranges_rad[:, 0] + 0.02,
      handles.hand_joint_ranges_rad[:, 1] - 0.02,
    )
    # Bound the command edge caused by M03 ownership changes.  This is most
    # important on the moving Bunny path, where an otherwise safe release
    # delta can excite a one-tick contact impulse through stiff actuators.
    issued_finger_command = command_continuity.limit_finger(
      issued_finger_command,
      maximum_step_rad=1.0,
    )
    data.ctrl[handles.hand_actuator_ids] = issued_finger_command
    previous_finger_command = issued_finger_command.copy()
    controller_latency[step] = perf_counter() - started
    physics_started = perf_counter()
    mujoco.mj_step(handles.model, data)
    physics_latency[step] = perf_counter() - physics_started

    measured_forces, measured_positions, measured_hfield_valid, non_tip_count = (
      _contact_state(handles, data, bunny)
    )
    contact_active = np.where(
      contact_active,
      (measured_forces >= 0.5 * config.contact_threshold_n) & measured_hfield_valid,
      (measured_forces >= config.contact_threshold_n) & measured_hfield_valid,
    )
    contact_confirmed_s = np.where(
      contact_active,
      contact_confirmed_s + config.dt_s,
      0.0,
    )
    if measured_set():
      last_nonempty_contact_set = measured_set()
    if timestamp >= config.acquisition_s:
      empty_run = empty_run + 1 if not np.any(contact_active) else 0
      if empty_run * config.dt_s > 0.010 and stop_reason is None:
        stop_reason = "LAST_CONTACT_LOST"
        stopped_coordinate = coordinate
    if np.max(measured_forces) > config.force_limit_n and stop_reason is None:
      stop_reason = "OVER_FORCE"
      stopped_coordinate = coordinate
    if non_tip_count > 0 and stop_reason is None:
      stop_reason = "NON_TIP_BUNNY_COLLISION"
      stopped_coordinate = coordinate
    if not np.all(np.isfinite(data.qpos)) and stop_reason is None:
      stop_reason = "NONFINITE_STATE"
      stopped_coordinate = coordinate
    if decision_started and not mechanism_complete and timestamp > 7.0:
      if stop_reason is None and not dead_end:
        stop_reason = "PLANNING_PLATEAU_OVERRUN"
        stopped_coordinate = coordinate

    if transaction_active and stop_reason is None:
      snapshot = executor.consume_barrier_snapshot()
      if snapshot is not None:
        assert current_prefix is not None
        assert current_finger is not None
        terminal_tip = current_prefix.samples[-1].fingertip_positions_m[
          current_finger - 1
        ]
        measured_tip = np.array(
          data.site_xpos[handles.tip_site_ids[current_finger - 1]],
          copy=True,
        )
        terminal_error = float(np.linalg.norm(measured_tip - terminal_tip))
        last_barrier_certificate = snapshot.certificate_id
        events.append(
          {
            "event": "MICRO_BARRIER",
            "time_s": timestamp,
            "phase": phase,
            "actual_contact_set": sorted(snapshot.actual_contact_set),
            "transaction_state": snapshot.transaction_state.value,
            "certificate_id": snapshot.certificate_id,
            "measured_root_authority": True,
            "terminal_prediction_error_m": terminal_error,
            "measured_terminal_fingertip_position_m": measured_tip.tolist(),
            "actual_joint_margin_rad": _joint_margin(handles, data),
          }
        )
        completed_phase = phase
        completed_finger = current_finger
        phase = "NORMAL"
        current_prefix = None
        current_finger = None

        if completed_phase == "I02_BREAK":
          assert reposition_origin_xy is not None
          assert reposition_goal_xy is not None
          short_segment_index = 1 if config.cell == "i02_short" else 0
          fraction = 1.0 / config.short_reposition_segments if short_segment_index else 1.0
          target = measured_tip.copy()
          target[:2] = reposition_origin_xy + fraction * (
            reposition_goal_xy - reposition_origin_xy
          )
          # REPOSITION remains free: M09 projects this request to its frozen
          # 4 mm clearance shell.  Asking below the shell avoids carrying the
          # full BREAK release height into the later MAKE Jacobian.
          target[2] -= 0.006
          pending_action = (
            "I02_REPOSITION_1",
            PrimitiveKind.REPOSITION,
            I02_FINGER,
            target,
            config.reposition_total_m * fraction,
          )
        elif completed_phase.startswith("I02_REPOSITION"):
          assert reposition_origin_xy is not None
          assert reposition_goal_xy is not None
          if config.cell == "i02_short" and short_segment_index < 3:
            short_segment_index += 1
            fraction = short_segment_index / config.short_reposition_segments
            target = measured_tip.copy()
            target[:2] = reposition_origin_xy + fraction * (
              reposition_goal_xy - reposition_origin_xy
            )
            target[2] -= 0.006
            pending_action = (
              f"I02_REPOSITION_{short_segment_index}",
              PrimitiveKind.REPOSITION,
              I02_FINGER,
              target,
              config.reposition_total_m / config.short_reposition_segments,
            )
          else:
            events.append(
              {
                "event": "I02_FINAL_REPOSITION",
                "time_s": timestamp,
                "goal_xy_m": reposition_goal_xy.tolist(),
                "measured_xy_m": measured_tip[:2].tolist(),
                "final_reposition_xy_error_m": float(
                  np.linalg.norm(measured_tip[:2] - reposition_goal_xy)
                ),
                "final_reposition_terminal_error_m": terminal_error,
              }
            )
            make_target = measured_tip.copy()
            make_target[:2] = reposition_goal_xy
            pending_action = (
              "I02_MAKE",
              PrimitiveKind.MAKE,
              I02_FINGER,
              make_target,
              0.0,
            )
        elif completed_phase == "I02_MAKE":
          assert reposition_goal_xy is not None
          contact_anchor_xy[I02_FINGER - 1] = reposition_goal_xy + np.array(
            [coordinate, 0.0]
          )
          mechanism_complete = True
          events.append(
            {
              "event": "I02_HANDOVER_COMPLETE",
              "time_s": timestamp,
              "actual_contact_set": sorted(measured_set()),
            }
          )
        elif completed_phase == "I03_SELECTED_EDGE":
          contact_anchor_xy[completed_finger - 1] = measured_tip[:2] + np.array(
            [coordinate, 0.0]
          )
          bundle = make_planner_bundle(
            handles,
            data,
            bunny,
            measured_set(),
            config,
          )
          actual_shadow, actual_evidence = evaluate_actual_shadow(bundle, config)
          current_shadow_latency = float(actual_evidence["actual_shadow_latency_s"])
          current_terminal_viability = str(
            actual_evidence["actual_terminal_viability"]
          )
          events.append(
            {
              "event": "I03_ACTUAL_SHADOW_DIAGNOSIS",
              "time_s": timestamp,
              **actual_evidence,
            }
          )
          if actual_shadow.execution_authority:
            authority_violation[step] = True
            stop_reason = "SHADOW_AUTHORITY_VIOLATION"
            stopped_coordinate = coordinate
          elif not actual_shadow.viable:
            dead_end = True
            stop_reason = "DEAD_END"
            stopped_coordinate = coordinate
            events.append(
              {
                "event": "DEAD_END",
                "time_s": timestamp,
                "reason": actual_shadow.reason,
              }
            )
          else:
            target = np.array(
              data.site_xpos[handles.tip_site_ids[I03_HANDOVER_FINGER - 1]],
              copy=True,
            )
            pending_action = (
              "I03_BREAK",
              PrimitiveKind.BREAK,
              I03_HANDOVER_FINGER,
              target,
              0.0,
            )
        elif completed_phase == "I03_BREAK":
          target = measured_tip.copy()
          target[0] -= 0.001
          pending_action = (
            "I03_REPOSITION",
            PrimitiveKind.REPOSITION,
            I03_HANDOVER_FINGER,
            target,
            0.001,
          )
        elif completed_phase == "I03_REPOSITION":
          target = measured_tip.copy()
          pending_action = (
            "I03_MAKE",
            PrimitiveKind.MAKE,
            I03_HANDOVER_FINGER,
            target,
            0.0,
          )
        elif completed_phase == "I03_MAKE":
          measured_xy = np.array(
            data.site_xpos[handles.tip_site_ids[I03_HANDOVER_FINGER - 1], :2],
            copy=True,
          )
          contact_anchor_xy[I03_HANDOVER_FINGER - 1] = measured_xy + np.array(
            [coordinate, 0.0]
          )
          mechanism_complete = True
          events.append(
            {
              "event": "I03_HANDOVER_COMPLETE",
              "time_s": timestamp,
              "actual_contact_set": sorted(measured_set()),
            }
          )

    arm_q[step] = data.qpos[handles.arm_qpos_adrs]
    arm_dq[step] = data.qvel[handles.arm_dof_adrs]
    arm_command[step] = data.ctrl[handles.arm_actuator_ids]
    finger_q[step] = data.qpos[handles.hand_qpos_adrs]
    finger_dq[step] = data.qvel[handles.hand_dof_adrs]
    finger_command[step] = data.ctrl[handles.hand_actuator_ids]
    palm[step] = np.concatenate(
      (
        data.site_xpos[handles.palm_site_id],
        _quaternion_from_matrix(data.site_xmat[handles.palm_site_id]),
      )
    )
    planned_palm[step] = planned_pose
    tips[step] = data.site_xpos[handles.tip_site_ids]
    contact_positions[step] = measured_positions
    forces_log[step] = measured_forces
    hfield_contacts[step] = contact_active
    planned_coordinate[step] = coordinate
    planned_cumulative[step] = (
      planned_cumulative_distance_m(timestamp)
      if stop_reason is None
      else planned_cumulative[step - 1] if step else 0.0
    )
    actual_coordinate[step] = initial_pose[0] - palm[step, 0]
    if step:
      actual_cumulative[step] = actual_cumulative[step - 1] + abs(
        actual_coordinate[step] - actual_coordinate[step - 1]
      )
    joint_margin[step] = _joint_margin(handles, data)
    non_tip[step] = non_tip_count
    guard_reason[step] = stop_reason or safety.state.value
    transaction_phase[step] = phase
    if phase != "NORMAL":
      transaction_state[step] = executor.transaction_state.value
      certificate_id[step] = current_certificate_id
      optimizer_latency[step] = current_optimizer_latency
      audit_latency[step] = current_audit_latency
    search_latency[step] = current_search_latency
    shadow_latency[step] = current_shadow_latency
    selected_sequence[step] = current_selected_sequence
    terminal_viability[step] = current_terminal_viability

  trace = I02I03BunnyTrace(
    config=config,
    time_s=time_s,
    arm_q_rad=arm_q,
    arm_dq_rad_s=arm_dq,
    arm_command_rad=arm_command,
    finger_q_rad=finger_q,
    finger_dq_rad_s=finger_dq,
    finger_command_rad=finger_command,
    palm_pose_world=palm,
    planned_palm_pose_world=planned_palm,
    fingertip_positions_world_m=tips,
    contact_positions_world_m=contact_positions,
    fingertip_forces_n=forces_log,
    hfield_valid_contacts=hfield_contacts,
    mesh_valid_contacts=np.zeros_like(hfield_contacts),
    mesh_residual_m=np.full((count, 4), np.nan),
    planned_path_coordinate_m=planned_coordinate,
    planned_cumulative_distance_m=planned_cumulative,
    actual_path_coordinate_m=actual_coordinate,
    actual_cumulative_distance_m=actual_cumulative,
    controller_latency_s=controller_latency,
    physics_latency_s=physics_latency,
    optimizer_latency_s=optimizer_latency,
    search_latency_s=search_latency,
    shadow_latency_s=shadow_latency,
    audit_latency_s=audit_latency,
    minimum_joint_margin_rad=joint_margin,
    non_tip_contact_count=non_tip,
    guard_reason=guard_reason,
    transaction_phase=transaction_phase,
    transaction_state=transaction_state,
    certificate_id=certificate_id,
    selected_sequence=selected_sequence,
    terminal_viability=terminal_viability,
    authority_violation=authority_violation,
    shadow_execution_authority=shadow_execution_authority,
    suffix_command_count=suffix_command_count,
    events=events,
  )
  metrics = evaluate_i02_i03_trace(trace, bunny, handles.object_position_m)
  return trace, metrics, handles.xml


def evaluate_i02_i03_trace(
  trace: I02I03BunnyTrace,
  bunny: BunnyHeightField,
  object_position_m: NDArray[np.float64],
) -> dict[str, Any]:
  """Apply the frozen exact-mesh filter and per-episode criteria."""

  strict_force = trace.fingertip_forces_n >= trace.config.contact_threshold_n
  candidates = strict_force & trace.hfield_valid_contacts
  indices = np.argwhere(candidates)
  if len(indices):
    points_world = trace.contact_positions_world_m[indices[:, 0], indices[:, 1]]
    residuals = bunny.mesh_residuals(points_world - object_position_m)
    trace.mesh_residual_m[indices[:, 0], indices[:, 1]] = residuals
    trace.mesh_valid_contacts[indices[:, 0], indices[:, 1]] = (
      residuals <= trace.config.mesh_residual_limit_m
    )
  evaluation = trace.time_s >= trace.config.acquisition_s
  exact_contacts = trace.mesh_valid_contacts
  contacts = exact_contacts[evaluation]
  nonempty_eval = np.any(contacts, axis=1)
  nonempty_full = np.any(exact_contacts, axis=1)
  empty_gap_s = _max_true_run(~nonempty_eval) * trace.config.dt_s
  supported_cumulative, segment_evidence = _supported_segment_traversal(
    trace,
    nonempty_full,
  )
  valid_forces = np.where(exact_contacts, trace.fingertip_forces_n, 0.0)
  peak_force = float(np.max(valid_forces[evaluation]))
  residual_values = trace.mesh_residual_m[np.isfinite(trace.mesh_residual_m)]
  stable_runs, handover = _stable_mode_evidence(
    contacts,
    trace.time_s[evaluation],
    trace.config.dt_s,
  )
  commits = [event for event in trace.events if event["event"] == "CERTIFIED_COMMIT"]
  barriers = [event for event in trace.events if event["event"] == "MICRO_BARRIER"]
  reposition_commits = [
    event for event in commits if "REPOSITION" in str(event.get("phase", ""))
  ]
  reposition_barriers = [
    event for event in barriers if "REPOSITION" in str(event.get("phase", ""))
  ]
  final_reposition = next(
    (
      event
      for event in reversed(trace.events)
      if event["event"] == "I02_FINAL_REPOSITION"
    ),
    None,
  )
  shadow_diagnosis = next(
    (
      event
      for event in reversed(trace.events)
      if event["event"] == "I03_ACTUAL_SHADOW_DIAGNOSIS"
    ),
    None,
  )
  search_decision = next(
    (event for event in trace.events if event["event"] == "I03_SEARCH_DECISION"),
    None,
  )
  stop_reason = next(
    (
      str(reason)
      for reason in trace.guard_reason[evaluation]
      if reason
      not in {
        "NONE",
        "INITIALIZE",
        "BUFFER_FILL",
        "ACTIVE",
        "NORMAL",
        "SOFT_RECOVERY",
        "REENTRY_RAMP",
        "BUFFER_RESET",
        "HARD_RELEASE",
      }
    ),
    None,
  )
  common_task_pass = bool(
    supported_cumulative >= 0.100
    and float(np.mean(nonempty_eval)) >= 0.990
    and empty_gap_s <= 0.010 + 1e-12
    and peak_force <= trace.config.force_limit_n + 1e-12
    and int(np.count_nonzero(valid_forces[evaluation] > trace.config.force_limit_n)) == 0
    and int(np.count_nonzero(trace.non_tip_contact_count[evaluation])) == 0
    and not np.any(trace.authority_violation)
    and not np.any(trace.shadow_execution_authority)
    and int(np.sum(trace.suffix_command_count)) == 0
    and stop_reason is None
  )
  mechanism_pass = bool(
    handover
    and any(
      event["event"] in {"I02_HANDOVER_COMPLETE", "I03_HANDOVER_COMPLETE"}
      for event in trace.events
    )
  )
  terminal_errors = [
    float(event["terminal_prediction_error_m"])
    for event in barriers
    if "terminal_prediction_error_m" in event
  ]
  fresh_root_evidence = all(
    str(event.get("fresh_root_after_certificate", ""))
    in {"INITIAL_MEASURED_ROOT"} | {str(barrier["certificate_id"]) for barrier in barriers}
    for event in commits
  )
  authentic_certificates = all(
    str(event.get("certificate_id", "")).startswith("cert-")
    and event.get("execution_authority") == "M10_EXACT_PREFIX_AUDIT"
    and event.get("prefix_source") == "OPTIMIZER_COMMIT_CANDIDATE"
    for event in commits
  )
  return {
    "trace_schema_version": TRACE_SCHEMA_VERSION,
    "evaluator_version": EVALUATOR_VERSION,
    "module_id": trace.config.module_id,
    "cell": trace.config.cell,
    "seed": trace.config.seed,
    "common_task_pass": common_task_pass,
    "mechanism_pass": mechanism_pass,
    "supported_cumulative_traversal_m": supported_cumulative,
    "scheduled_segment_evidence": segment_evidence,
    "maximum_actual_path_coordinate_m": float(
      np.max(trace.actual_path_coordinate_m[evaluation])
    ),
    "raw_actual_total_variation_m": float(trace.actual_cumulative_distance_m[-1]),
    "nonempty_contact_fraction": float(np.mean(nonempty_eval)),
    "four_contact_fraction": float(np.mean(np.all(contacts, axis=1))),
    "per_finger_contact_fraction": np.mean(contacts, axis=0).tolist(),
    "maximum_all_contact_loss_gap_s": empty_gap_s,
    "peak_valid_fingertip_force_n": peak_force,
    "over_force_ticks": int(
      np.count_nonzero(valid_forces[evaluation] > trace.config.force_limit_n)
    ),
    "non_tip_contact_ticks": int(
      np.count_nonzero(trace.non_tip_contact_count[evaluation])
    ),
    "authority_violation_count": int(np.count_nonzero(trace.authority_violation)),
    "shadow_execution_authority_count": int(
      np.count_nonzero(trace.shadow_execution_authority)
    ),
    "prediction_suffix_command_count": int(np.sum(trace.suffix_command_count)),
    "handover_4_3_4_measured": handover,
    "certificate_count": len(commits),
    "micro_barrier_count": len(barriers),
    "reposition_certificate_count": len(reposition_commits),
    "reposition_barrier_count": len(reposition_barriers),
    "fresh_measured_root_evidence": fresh_root_evidence,
    "all_certificates_authentic": authentic_certificates,
    "terminal_prediction_error_m": terminal_errors,
    "final_reposition_xy_error_m": (
      float(final_reposition["final_reposition_xy_error_m"])
      if final_reposition is not None
      else None
    ),
    "final_reposition_terminal_error_m": (
      float(final_reposition["final_reposition_terminal_error_m"])
      if final_reposition is not None
      else None
    ),
    "dead_end_count": int(
      sum(event["event"] == "DEAD_END" for event in trace.events)
    ),
    "predicted_terminal_viability": (
      str(search_decision["predicted_terminal_viability"])
      if search_decision is not None
      else None
    ),
    "actual_terminal_viability": (
      str(shadow_diagnosis["actual_terminal_viability"])
      if shadow_diagnosis is not None
      else None
    ),
    "viability_agreement": (
      bool(
        search_decision["predicted_terminal_viability"]
        == shadow_diagnosis["actual_terminal_viability"]
      )
      if search_decision is not None and shadow_diagnosis is not None
      else None
    ),
    "selected_sequence": (
      list(search_decision["selected_sequence"])
      if search_decision is not None
      else []
    ),
    "actual_terminal_joint_margin_rad": (
      float(shadow_diagnosis["actual_terminal_joint_margin_rad"])
      if shadow_diagnosis is not None
      else None
    ),
    "actual_successor_fingers": (
      list(shadow_diagnosis["actual_successor_fingers"])
      if shadow_diagnosis is not None
      else []
    ),
    "minimum_physical_joint_margin_rad": float(
      np.min(trace.minimum_joint_margin_rad[evaluation])
    ),
    "mesh_contact_residual_p95_m": (
      float(np.percentile(residual_values, 95.0)) if len(residual_values) else None
    ),
    "mesh_contact_residual_max_m": (
      float(np.max(residual_values)) if len(residual_values) else None
    ),
    "mesh_rejected_contact_fraction": float(
      np.count_nonzero(candidates & ~trace.mesh_valid_contacts)
      / max(np.count_nonzero(candidates), 1)
    ),
    "controller_latency_p95_s": float(
      np.percentile(trace.controller_latency_s, 95.0)
    ),
    "physics_latency_p95_s": float(np.percentile(trace.physics_latency_s, 95.0)),
    "optimizer_latency_p95_s": (
      float(np.percentile(trace.optimizer_latency_s[trace.optimizer_latency_s > 0.0], 95.0))
      if np.any(trace.optimizer_latency_s > 0.0)
      else None
    ),
    "audit_latency_p95_s": (
      float(np.percentile(trace.audit_latency_s[trace.audit_latency_s > 0.0], 95.0))
      if np.any(trace.audit_latency_s > 0.0)
      else None
    ),
    "search_latency_s": (
      float(search_decision["search_latency_s"])
      if search_decision is not None
      else None
    ),
    "predicted_shadow_latency_s": (
      float(search_decision["predicted_shadow_latency_s"])
      if search_decision is not None
      else None
    ),
    "actual_shadow_latency_s": (
      float(shadow_diagnosis["actual_shadow_latency_s"])
      if shadow_diagnosis is not None
      else None
    ),
    "stop_reason": stop_reason,
    "stable_mode_runs": stable_runs,
    "events": trace.events,
  }


def save_trace(path: str | Path, trace: I02I03BunnyTrace) -> Path:
  output = Path(path)
  output.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(output, **trace.npz_payload())
  return output
