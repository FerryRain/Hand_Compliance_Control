"""MuJoCo runner for I01 fixed/variable contact traversal on Bunny."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from numpy.typing import NDArray

from Module.e05_physics.scene import FINGERS, PAD_HALF_SIZE_M
from Module.fr3_leap import (
  ARM_HOME_Q,
  HAND_NATURAL_Q,
  FullRobotModelConfig,
  build_full_robot,
)
from Module.i01_bunny_physics.surface import BunnyHeightField, canonical_bunny_heightfield
from Module.module_1_oracle_surface_model import (
  FullRobotGeometryAdapter,
  MeshScalePolicy,
  MeshSurface,
  OracleSurfaceModel,
)
from Module.module_2_fingertip_mcc import FingertipMCC, FullRobotFingertipMCC, MCCConfig
from Module.module_3_runtime_guards import ForceSafetyConfig, ForceSafetyExecutor
from Module.module_4_whole_hand_mcc.robot_control import PalmPoseIK, PalmPoseIKConfig
from Module.module_6_prefix_executor import (
  ExecutorConfig,
  ExecutorObservation,
  MCCBaselineAdapter,
  PlannedPrefix,
  PrefixSample,
  TransactionState,
  TransactionType,
  TransactionalPrefixExecutor,
)
from Module.module_7_contact_mode_graph import (
  CommitContext,
  ContactModeGraph,
  ContactPrimitive,
  PrimitiveKind,
)
from Module.module_9_continuous_optimize import LinearizedHandKinematics, PlannerState
from Module.module_10_exact_prefix_audit import (
  AuditConfig,
  AuditEnvironment,
  AuditRequest,
  ExactPrefixAuditor,
)


SURFACE_MODEL_VERSION = "oracle-bunny-upper-envelope.v1"
TRACE_SCHEMA_VERSION = "i01-bunny-trace.v1"
EVALUATOR_VERSION = "i01-bunny-evaluator.v1"
VALID_CELLS = ("fixed", "variable")
HANDOVER_FINGER = 4


def _quaternion_from_matrix(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
  """Convert a rotation matrix using MuJoCo's scalar-first convention."""

  quaternion = np.zeros(4, dtype=np.float64)
  mujoco.mju_mat2Quat(quaternion, np.asarray(matrix, dtype=np.float64).reshape(9))
  return quaternion


def _smoothstep(value: float) -> float:
  clipped = float(np.clip(value, 0.0, 1.0))
  return clipped * clipped * (3.0 - 2.0 * clipped)


def _pad_support_radius(
  handles: Any,
  data: mujoco.MjData,
  finger_index: int,
  direction_world: NDArray[np.float64],
) -> float:
  rotation = data.geom_xmat[int(handles.tip_geom_ids[finger_index])].reshape(3, 3)
  local_direction = rotation.T @ direction_world
  return float(np.linalg.norm(PAD_HALF_SIZE_M * local_direction))


def _finger_ik(
  handles: Any,
  data: mujoco.MjData,
  finger_index: int,
  target_position: NDArray[np.float64],
  target_pad_normal: NDArray[np.float64],
  *,
  damping: float = 0.012,
  gain: float = 0.18,
  orientation_weight: float = 0.012,
  posture_gain: float = 0.01,
) -> NDArray[np.float64]:
  """Frozen resolved-rate fingertip IK used by both I01 cells."""

  jac_position = np.zeros((3, handles.model.nv), dtype=np.float64)
  jac_rotation = np.zeros((3, handles.model.nv), dtype=np.float64)
  site_id = int(handles.tip_site_ids[finger_index])
  mujoco.mj_jacSite(handles.model, data, jac_position, jac_rotation, site_id)
  dofs = handles.finger_dof_adrs[finger_index]
  qpos = handles.finger_qpos_adrs[finger_index]
  current_position = data.site_xpos[site_id]
  current_normal = data.site_xmat[site_id].reshape(3, 3)[:, 2]
  cross = np.cross(current_normal, target_pad_normal)
  cross_length = float(np.linalg.norm(cross))
  if cross_length > 1e-9:
    axis = cross / cross_length
    angle = float(
      np.arctan2(
        cross_length,
        np.clip(np.dot(current_normal, target_pad_normal), -1.0, 1.0),
      )
    )
  else:
    axis = np.array([1.0, 0.0, 0.0])
    angle = 0.0
  position_jacobian = jac_position[:, dofs]
  rotation_jacobian = jac_rotation[:, dofs]
  jacobian = np.vstack(
    (position_jacobian, orientation_weight * (axis @ rotation_jacobian)[None, :])
  )
  error = np.concatenate(
    (target_position - current_position, [orientation_weight * angle])
  )
  regularized = jacobian @ jacobian.T + damping**2 * np.eye(4)
  delta = jacobian.T @ np.linalg.solve(regularized, error)
  current_q = np.array(data.qpos[qpos], dtype=np.float64)
  nominal_indices = np.array(
    [int(name) for name in FINGERS[finger_index].joint_names],
    dtype=np.int32,
  )
  command = current_q + gain * delta + posture_gain * (
    HAND_NATURAL_Q[nominal_indices] - current_q
  )
  command = np.clip(command, current_q - 0.012, current_q + 0.012)
  lower = handles.hand_joint_ranges_rad[nominal_indices, 0] + 0.05
  upper = handles.hand_joint_ranges_rad[nominal_indices, 1] - 0.05
  return np.clip(command, lower, upper)


def _signed_compression_jacobian(
  handles: Any,
  data: mujoco.MjData,
  outward_normals_world: NDArray[np.float64],
) -> NDArray[np.float64]:
  """Return J_s with positive row motion defined as increased compression."""

  result = np.zeros((4, 16), dtype=np.float64)
  jacobian_position = np.zeros((3, handles.model.nv), dtype=np.float64)
  for finger, site_id in enumerate(handles.tip_site_ids):
    jacobian_position[:] = 0.0
    mujoco.mj_jacSite(
      handles.model,
      data,
      jacobian_position,
      None,
      int(site_id),
    )
    dofs = handles.finger_dof_adrs[finger]
    joint_indices = np.arange(4 * finger, 4 * finger + 4)
    outward_row = outward_normals_world[finger] @ jacobian_position[:, dofs]
    result[finger, joint_indices] = -outward_row
  return result


@dataclass(frozen=True, slots=True)
class I01BunnyConfig:
  cell: str = "fixed"
  seed: int = 7
  duration_s: float = 12.0
  acquisition_s: float = 3.0
  dt_s: float = 0.002
  traversal_m: float = 0.060
  desired_force_n: float = 2.0
  contact_threshold_n: float = 0.20
  force_limit_n: float = 8.0
  mesh_residual_limit_m: float = 0.0025
  initial_arm_noise_rad: float = 0.0005
  initial_hand_noise_rad: float = 0.0010
  object_offset_x_m: float = 0.002
  object_offset_y_m: float = -0.005
  object_offset_z_m: float = -0.003
  visual_mesh_path: str | None = None

  def __post_init__(self) -> None:
    if self.cell not in VALID_CELLS:
      raise ValueError(f"cell must be one of {VALID_CELLS}")
    if self.duration_s <= self.acquisition_s or self.acquisition_s <= 0.0:
      raise ValueError("duration_s must exceed positive acquisition_s")
    if not np.isclose(self.dt_s, 0.002):
      raise ValueError("I01-PHY-BUNNY-v1 freezes dt_s at 0.002")
    for name in (
      "traversal_m",
      "desired_force_n",
      "contact_threshold_n",
      "force_limit_n",
      "mesh_residual_limit_m",
    ):
      if not np.isfinite(float(getattr(self, name))) or float(getattr(self, name)) <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    if self.force_limit_n <= self.desired_force_n:
      raise ValueError("force_limit_n must exceed desired_force_n")


@dataclass(slots=True)
class I01BunnyTrace:
  config: I01BunnyConfig
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
  planned_progress_m: NDArray[np.float64]
  actual_progress_m: NDArray[np.float64]
  controller_latency_s: NDArray[np.float64]
  physics_latency_s: NDArray[np.float64]
  non_tip_contact_count: NDArray[np.int32]
  guard_reason: NDArray[np.str_]
  transaction_phase: NDArray[np.str_]
  transaction_state: NDArray[np.str_]
  certificate_id: NDArray[np.str_]
  audit_latency_s: NDArray[np.float64]
  authority_violation: NDArray[np.bool_]
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
      "planned_progress_m": self.planned_progress_m,
      "actual_progress_m": self.actual_progress_m,
      "controller_latency_s": self.controller_latency_s,
      "physics_latency_s": self.physics_latency_s,
      "non_tip_contact_count": self.non_tip_contact_count,
      "guard_reason": self.guard_reason,
      "transaction_phase": self.transaction_phase,
      "transaction_state": self.transaction_state,
      "certificate_id": self.certificate_id,
      "audit_latency_s": self.audit_latency_s,
      "authority_violation": self.authority_violation,
    }


def _planned_progress(config: I01BunnyConfig, timestamp_s: float) -> float:
  """Same 10 mm plateau path for both cells; variable uses it for handover."""

  tau = float(np.clip(timestamp_s - config.acquisition_s, 0.0, 9.0))
  if tau <= 1.0:
    return 0.010 * _smoothstep(tau / 1.0)
  if tau <= 4.5:
    return 0.010
  return 0.010 + (config.traversal_m - 0.010) * _smoothstep((tau - 4.5) / 4.5)


def _logical_q(handles: Any, data: mujoco.MjData) -> NDArray[np.float64]:
  return np.concatenate(
    (
      np.asarray(data.qpos[handles.arm_qpos_adrs], dtype=np.float64),
      np.asarray(data.qpos[handles.hand_qpos_adrs], dtype=np.float64),
    )
  )


def _contact_state(
  handles: Any,
  data: mujoco.MjData,
  bunny: BunnyHeightField,
) -> tuple[
  NDArray[np.float64],
  NDArray[np.float64],
  NDArray[np.bool_],
  int,
]:
  forces = np.zeros(4, dtype=np.float64)
  weighted_positions = np.zeros((4, 3), dtype=np.float64)
  lookup = {int(geom_id): index for index, geom_id in enumerate(handles.tip_geom_ids)}
  contact_force = np.zeros(6, dtype=np.float64)
  non_tip = 0
  for contact_index in range(data.ncon):
    contact = data.contact[contact_index]
    geom_1 = int(contact.geom1)
    geom_2 = int(contact.geom2)
    if handles.object_geom_id not in (geom_1, geom_2):
      continue
    other = geom_2 if geom_1 == handles.object_geom_id else geom_1
    finger_index = lookup.get(other)
    if finger_index is None:
      non_tip += 1
      continue
    mujoco.mj_contactForce(handles.model, data, contact_index, contact_force)
    normal_force = abs(float(contact_force[0]))
    forces[finger_index] += normal_force
    weighted_positions[finger_index] += normal_force * np.asarray(contact.pos)
  positions = np.zeros((4, 3), dtype=np.float64)
  valid_force = forces > 0.0
  positions[valid_force] = weighted_positions[valid_force] / forces[valid_force, None]
  hfield_valid = np.zeros(4, dtype=np.bool_)
  for finger in np.flatnonzero(valid_force):
    local = positions[finger] - handles.object_position_m
    _, _, hfield_valid[finger] = bunny.query(float(local[0]), float(local[1]))
  return forces, positions, hfield_valid, non_tip


def _surface_targets(
  handles: Any,
  data: mujoco.MjData,
  bunny: BunnyHeightField,
  target_xy_world: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
  centers = np.zeros((4, 3), dtype=np.float64)
  normals = np.zeros((4, 3), dtype=np.float64)
  valid = np.zeros(4, dtype=np.bool_)
  for finger, xy in enumerate(target_xy_world):
    local_xy = xy - handles.object_position_m[:2]
    height, normal, is_valid = bunny.query(float(local_xy[0]), float(local_xy[1]))
    surface = handles.object_position_m + np.array([local_xy[0], local_xy[1], height])
    centers[finger] = surface + _pad_support_radius(handles, data, finger, normal) * normal
    normals[finger] = normal
    valid[finger] = is_valid
  return centers, normals, valid


def _local_kinematics(handles: Any, data: mujoco.MjData) -> LinearizedHandKinematics:
  q = _logical_q(handles, data)
  tips = np.array(data.site_xpos[handles.tip_site_ids], dtype=np.float64, copy=True)
  dofs = np.concatenate((handles.arm_dof_adrs, handles.hand_dof_adrs))
  jacobians = np.zeros((4, 3, len(q)), dtype=np.float64)
  full_jacobian = np.zeros((3, handles.model.nv), dtype=np.float64)
  for finger, site_id in enumerate(handles.tip_site_ids):
    full_jacobian[:] = 0.0
    mujoco.mj_jacSite(handles.model, data, full_jacobian, None, int(site_id))
    jacobians[finger] = full_jacobian[:, dofs]
  # Finger transactions must not borrow arm motion.  Separate LEAP branches
  # already make unrelated finger columns zero; make the arm restriction
  # explicit for the audit model.
  jacobians[:, :, :7] = 0.0
  return LinearizedHandKinematics(
    reference_q_rad=q,
    joint_lower_rad=np.concatenate(
      (handles.arm_joint_ranges_rad[:, 0], handles.hand_joint_ranges_rad[:, 0])
    ),
    joint_upper_rad=np.concatenate(
      (handles.arm_joint_ranges_rad[:, 1], handles.hand_joint_ranges_rad[:, 1])
    ),
    reference_wrist_position_m=data.site_xpos[handles.palm_site_id],
    reference_fingertip_positions_m=tips,
    fingertip_jacobians_m_per_rad=jacobians,
  )


def _world_mesh_oracle(
  bunny: BunnyHeightField,
  object_position_m: NDArray[np.float64],
) -> OracleSurfaceModel:
  mesh = bunny.mesh.copy()
  mesh.apply_translation(object_position_m)
  shape = MeshSurface(
    mesh,
    source_path=bunny.source_path,
    source_up_axis="y",
    scale_policy=MeshScalePolicy(0.30, 0.18),
    scale_factor=1.0,
  )
  return OracleSurfaceModel(shape, version=SURFACE_MODEL_VERSION)


def _link_clearance_function(
  handles: Any,
  bunny: BunnyHeightField,
) -> Any:
  scratch = mujoco.MjData(handles.model)
  geometry = FullRobotGeometryAdapter(handles)
  alphas = np.linspace(0.0, 1.0, 9)
  cache: dict[bytes, float] = {}

  def clearance(
    q: NDArray[np.float64],
    _wrist: NDArray[np.float64],
    _tips: NDArray[np.float64],
  ) -> float:
    # Finger-only prefixes freeze all seven arm joints, so every swept sample
    # has identical arm capsules.  Cache on exact arm bytes; this preserves the
    # exact mesh query while avoiding 49 identical proximity computations.
    key = np.round(np.asarray(q[:7], dtype=np.float64), decimals=12).tobytes()
    if key in cache:
      return cache[key]
    scratch.qpos[handles.arm_qpos_adrs] = q[:7]
    scratch.qpos[handles.hand_qpos_adrs] = q[7:]
    mujoco.mj_forward(handles.model, scratch)
    capsules = geometry.world_capsules(scratch)
    samples: list[NDArray[np.float64]] = []
    radii: list[NDArray[np.float64]] = []
    for capsule in capsules:
      samples.append(
        capsule.start[None, :]
        + alphas[:, None] * (capsule.end - capsule.start)[None, :]
      )
      radii.append(np.full(len(alphas), capsule.radius, dtype=np.float64))
    points = np.concatenate(samples, axis=0)
    radius = np.concatenate(radii)
    exact_distance = bunny.mesh_residuals(points - handles.object_position_m)
    result = float(np.min(exact_distance - radius))
    cache[key] = result
    return result

  return clearance


def _certified_finger_prefix(
  *,
  handles: Any,
  data: mujoco.MjData,
  bunny: BunnyHeightField,
  actual_contact_set: frozenset[int],
  primitive_kind: PrimitiveKind,
  finger_id: int,
  target_position_m: NDArray[np.float64],
  timestamp_s: float,
  replacement_confirmation_s: dict[int, float],
  sequence: int,
) -> tuple[PlannedPrefix, Any, dict[str, Any]]:
  graph = ContactModeGraph()
  primitive = ContactPrimitive(primitive_kind, finger_id)
  kinematics = _local_kinematics(handles, data)
  root_q = _logical_q(handles, data)
  root_tips = np.array(data.site_xpos[handles.tip_site_ids], dtype=np.float64, copy=True)
  desired = root_tips.copy()
  desired[finger_id - 1] = target_position_m
  terminal_q, target_error = kinematics.solve(
    desired,
    data.site_xpos[handles.palm_site_id],
    root_q,
    (finger_id,),
    damping=2e-4,
    iterations=12,
  )
  if target_error > 0.00075:
    raise RuntimeError(f"local prefix IK target error {target_error:.6f} m")
  duration = 0.45 if primitive_kind is not PrimitiveKind.MAKE else 0.70
  samples: list[PrefixSample] = []
  for alpha in np.linspace(0.0, 1.0, 7):
    q = (1.0 - alpha) * root_q + alpha * terminal_q
    tips = kinematics.forward(q, data.site_xpos[handles.palm_site_id])
    samples.append(
      PrefixSample(
        time_s=float(alpha * duration),
        wrist_position_m=data.site_xpos[handles.palm_site_id],
        fingertip_positions_m=tips,
        joint_positions_rad=q,
      )
    )
  expected = graph.apply_predictive(
    kinematics_state_mode(actual_contact_set),
    primitive,
  ).contacts
  prefix = PlannedPrefix(
    prefix_id=f"i01-{sequence:03d}-{primitive.key.lower().replace('(', '-').replace(')', '')}",
    transaction_type=TransactionType.FINGER_RECONFIGURE,
    primitive_kind=primitive.kind.value,
    surface_model_version=SURFACE_MODEL_VERSION,
    root_contact_set=actual_contact_set,
    expected_terminal_contact_set=expected,
    samples=tuple(samples),
    participating_fingers=(finger_id,),
    anchor_fingers=tuple(sorted(actual_contact_set - {finger_id})),
    finger_id=finger_id,
    topology_change_count=primitive.topology_change_count,
    metadata={"target_error_m": target_error, "backend": "live_mujoco_jacobian"},
  )
  state = PlannerState(
    joint_positions_rad=root_q,
    wrist_position_m=data.site_xpos[handles.palm_site_id],
    fingertip_positions_m=root_tips,
    actual_contact_set=actual_contact_set,
    surface_model_version=SURFACE_MODEL_VERSION,
  )
  environment = AuditEnvironment(
    surface_model=_world_mesh_oracle(bunny, handles.object_position_m),
    kinematics=kinematics,
    link_clearance_fn=_link_clearance_function(handles, bunny),
  )
  auditor = ExactPrefixAuditor(
    graph,
    environment,
    AuditConfig(
      audit_version="exact-prefix-audit.i01-bunny.v1",
      subdivisions_per_segment=9,
      minimum_link_clearance_m=0.0,
      minimum_joint_margin_rad=0.01,
      anchor_tolerance_m=0.0015,
      kinematic_consistency_tolerance_m=1e-8,
      max_commit_displacement_m=0.015,
    ),
  )
  context = CommitContext(
    actual_contact_set=actual_contact_set,
    replacement_confirmation_s=replacement_confirmation_s,
    minimum_confirmation_s=0.05,
  )
  result = auditor.audit(
    AuditRequest(
      prefix=prefix,
      current_state=state,
      commit_context=context,
      issued_at_s=timestamp_s,
    )
  )
  if not result.certified or result.certificate is None:
    raise RuntimeError("M10 audit rejected prefix: " + ",".join(result.reasons))
  evidence = {
    "prefix_id": prefix.prefix_id,
    "primitive": primitive.key,
    "certificate_id": result.certificate.certificate_id,
    "audit_latency_s": result.latency_s,
    "swept_samples": result.swept_samples,
    "minimum_joint_margin_rad": result.minimum_joint_margin_rad,
    "minimum_self_collision_clearance_m": result.minimum_self_collision_clearance_m,
    "minimum_link_clearance_m": result.minimum_link_clearance_m,
    "maximum_anchor_error_m": result.maximum_anchor_error_m,
    "maximum_trust_displacement_m": result.maximum_trust_displacement_m,
    "maximum_kinematic_error_m": result.maximum_kinematic_error_m,
    "root_fingertip_position_m": root_tips[finger_id - 1].tolist(),
    "terminal_fingertip_position_m": prefix.samples[-1].fingertip_positions_m[
      finger_id - 1
    ].tolist(),
  }
  return prefix, result.certificate, evidence


def kinematics_state_mode(contacts: frozenset[int]) -> Any:
  # Kept as a tiny helper to make the predictive/real-state boundary visible.
  from Module.module_7_contact_mode_graph import ContactMode

  return ContactMode(contacts)


def _max_true_run(mask: NDArray[np.bool_]) -> int:
  best = 0
  current = 0
  for value in mask:
    current = current + 1 if bool(value) else 0
    best = max(best, current)
  return best


def run_i01_bunny(
  config: I01BunnyConfig,
) -> tuple[I01BunnyTrace, dict[str, Any], str]:
  """Run one deterministic physical episode and evaluate the frozen metrics."""

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
  arm_initial = ARM_HOME_Q + rng.normal(0.0, config.initial_arm_noise_rad, 7)
  hand_initial = HAND_NATURAL_Q + rng.normal(0.0, config.initial_hand_noise_rad, 16)
  arm_initial = np.clip(
    arm_initial,
    handles.arm_joint_ranges_rad[:, 0] + 0.04,
    handles.arm_joint_ranges_rad[:, 1] - 0.04,
  )
  hand_initial = np.clip(
    hand_initial,
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
  anchor_xy = np.array(data.site_xpos[handles.tip_site_ids, :2], copy=True)
  initial_centers, _, target_valid = _surface_targets(
    handles,
    data,
    bunny,
    anchor_xy,
  )
  if not np.all(target_valid):
    raise RuntimeError("initial fingertip footprint is outside the Bunny silhouette")
  initial_surface_mean = float(
    np.mean(
      [
        bunny.query(*(xy - handles.object_position_m[:2]))[0]
        + handles.object_position_m[2]
        for xy in anchor_xy
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
  force_safety = ForceSafetyExecutor(
    ForceSafetyConfig(
      joint_lower_rad=handles.hand_joint_ranges_rad[:, 0],
      joint_upper_rad=handles.hand_joint_ranges_rad[:, 1],
      dt_s=config.dt_s,
      soft_force_n=6.0,
      hard_force_n=config.force_limit_n,
      recover_force_n=2.5,
    )
  )
  executor = TransactionalPrefixExecutor(
    ExecutorConfig(
      # Topology completion is still force-confirmed.  The 5 mm Cartesian
      # tolerance accounts for the physical belly-pad/site offset and local
      # nonlinear IK error; M10's committed displacement remains bounded by
      # the stricter 15 mm trust region.
      completion_tolerance_m=0.0050,
      default_timeout_s=1.4,
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
  planned_progress = np.zeros(count)
  actual_progress = np.zeros(count)
  controller_latency = np.zeros(count)
  physics_latency = np.zeros(count)
  non_tip = np.zeros(count, dtype=np.int32)
  guard_reason = np.full(count, "NONE", dtype="U64")
  transaction_phase = np.full(count, "NORMAL", dtype="U24")
  transaction_state = np.full(count, TransactionState.IDLE.value, dtype="U24")
  certificate_id = np.full(count, "NONE", dtype="U64")
  audit_latency = np.zeros(count)
  authority_violation = np.zeros(count, dtype=np.bool_)
  events: list[dict[str, Any]] = []

  measured_forces = np.zeros(4)
  measured_positions = np.zeros((4, 3))
  measured_hfield_valid = np.zeros(4, dtype=np.bool_)
  contact_active = np.zeros(4, dtype=np.bool_)
  contact_confirmed_s = np.zeros(4)
  previous_finger_command = np.array(data.ctrl[handles.hand_actuator_ids], copy=True)
  fixed_loss_run = 0
  empty_run = 0
  stop_reason: str | None = None
  stop_progress = config.traversal_m
  phase = "NORMAL"
  pending_next_phase: str | None = None
  transaction_sequence = 0
  current_audit_latency = 0.0
  handover_complete = False

  def measured_set() -> frozenset[int]:
    return frozenset(int(index + 1) for index in np.flatnonzero(contact_active))

  def observation(timestamp: float, normals: NDArray[np.float64]) -> ExecutorObservation:
    contacts = measured_set()
    # MAKE authority changes topology only after 40 ms of measured force
    # confirmation.  Until then the executor continues the already-certified
    # prefix and the predicted target cannot promote itself into A_actual.
    if phase == "MAKE" and contact_confirmed_s[HANDOVER_FINGER - 1] < 0.040:
      contacts = contacts - {HANDOVER_FINGER}
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

  def commit_phase(
    next_phase: str,
    timestamp: float,
    centers: NDArray[np.float64],
    normals: NDArray[np.float64],
  ) -> None:
    nonlocal transaction_sequence, current_audit_latency, phase
    primitive_by_phase = {
      "BREAK": PrimitiveKind.BREAK,
      "REPOSITION": PrimitiveKind.REPOSITION,
      "MAKE": PrimitiveKind.MAKE,
    }
    primitive = primitive_by_phase[next_phase]
    current_tip = np.array(
      data.site_xpos[handles.tip_site_ids[HANDOVER_FINGER - 1]],
      dtype=np.float64,
      copy=True,
    )
    target = centers[HANDOVER_FINGER - 1].copy()
    if next_phase == "BREAK":
      # The committed trust region is 15 mm.  Start from the measured tip so
      # BREAK is a bounded 7 mm release even when pad support geometry makes
      # the nominal touching center differ slightly from the site center.
      target = current_tip + 0.007 * normals[HANDOVER_FINGER - 1]
    elif next_phase == "REPOSITION":
      # BREAK has already placed the free pad in clearance.  Reposition is a
      # short, tangential one-millimetre prefix at that clearance; using the
      # future touching center here would mix REPOSITION and MAKE semantics.
      tangent = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
      tangent -= np.dot(tangent, normals[HANDOVER_FINGER - 1]) * normals[
        HANDOVER_FINGER - 1
      ]
      tangent /= np.linalg.norm(tangent)
      target = current_tip + 0.001 * tangent
    else:
      # A free MAKE finger has no MCC force authority until force confirms
      # contact.  A bounded 3 mm preload compensates the measured nonlinear
      # thumb IK residual; M03 takes precedence immediately after contact.
      target -= 0.0030 * normals[HANDOVER_FINGER - 1]
    delta = target - current_tip
    delta_norm = float(np.linalg.norm(delta))
    if delta_norm > 0.012:
      target = current_tip + 0.012 * delta / delta_norm
    actual = measured_set()
    if not actual:
      raise RuntimeError("cannot commit from an empty measured contact set")
    transaction_sequence += 1
    confirmations = {
      int(index + 1): float(contact_confirmed_s[index])
      for index in np.flatnonzero(contact_active)
    }
    prefix, certificate, audit = _certified_finger_prefix(
      handles=handles,
      data=data,
      bunny=bunny,
      actual_contact_set=actual,
      primitive_kind=primitive,
      finger_id=HANDOVER_FINGER,
      target_position_m=target,
      timestamp_s=timestamp,
      replacement_confirmation_s=confirmations,
      sequence=transaction_sequence,
    )
    # The normal-traversal MCC offsets are expressed around surface touching
    # centers, while M06 anchors its prefix at the measured root tip positions.
    # Carrying the former offset into the latter frame would double-apply the
    # compression on the first transaction tick.  Reset is an explicit frame
    # transition, not a hidden command or predicted-contact substitution.
    full_mcc.reset()
    executor.commit(prefix, certificate, observation(timestamp, normals))
    current_audit_latency = float(audit["audit_latency_s"])
    phase = next_phase
    events.append({"event": "CERTIFIED_COMMIT", "time_s": timestamp, **audit})

  for step, timestamp in enumerate(time_s):
    started = perf_counter()
    progress = _planned_progress(config, float(timestamp))
    if stop_reason is not None:
      progress = stop_progress
    target_xy = anchor_xy.copy()
    target_xy[:, 0] -= progress
    centers, normals, target_valid = _surface_targets(handles, data, bunny, target_xy)
    if not np.all(target_valid) and stop_reason is None:
      stop_reason = "TARGET_LEFT_BUNNY"
      stop_progress = progress
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
    planned_pose[0] -= progress
    planned_pose[2] += surface_mean - initial_surface_mean

    if config.cell == "variable" and stop_reason is None:
      tau = float(timestamp - config.acquisition_s)
      if (
        phase == "NORMAL"
        and not handover_complete
        and tau >= 1.5
        and measured_set() == frozenset({1, 2, 3, 4})
        and float(np.min(contact_confirmed_s)) >= 0.05
      ):
        try:
          commit_phase("BREAK", float(timestamp), centers, normals)
        except (RuntimeError, PermissionError, ValueError) as error:
          stop_reason = f"AUDIT_OR_COMMIT_REJECTED:{error}"
          stop_progress = progress
      elif pending_next_phase is not None:
        try:
          commit_phase(pending_next_phase, float(timestamp), centers, normals)
          pending_next_phase = None
        except (RuntimeError, PermissionError, ValueError) as error:
          stop_reason = f"AUDIT_OR_COMMIT_REJECTED:{error}"
          stop_progress = progress

    current_pose = np.concatenate(
      (
        data.site_xpos[handles.palm_site_id].copy(),
        _quaternion_from_matrix(data.site_xmat[handles.palm_site_id]),
      )
    )
    safety = force_safety.step(
      fingertip_force_n=measured_forces,
      # The MuJoCo force channel is valid even when a finger is not touching;
      # zero force is a valid measurement, not a sensor dropout.
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

    if phase in {"BREAK", "REPOSITION", "MAKE"} and stop_reason is None:
      try:
        command = executor.step(observation(float(timestamp), normals))
        cartesian_commands = command.commanded_fingertip_positions_m
        transaction_state[step] = command.transaction_state.value
        certificate_id[step] = command.certificate_id
        if command.safe_hold:
          stop_reason = command.safety_reason or "EXECUTOR_SAFE_HOLD"
          stop_progress = progress
      except (RuntimeError, PermissionError, ValueError) as error:
        authority_violation[step] = True
        stop_reason = f"EXECUTOR_ERROR:{error}"
        stop_progress = progress
        cartesian_commands = np.array(data.site_xpos[handles.tip_site_ids], copy=True)
    else:
      force_errors = config.desired_force_n - measured_forces
      active_request = np.ones(4, dtype=np.bool_)
      output = full_mcc.step(centers, -normals, force_errors, active_request)
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
        gain=(
          0.34
          if phase in {"BREAK", "REPOSITION", "MAKE"}
          and finger == HANDOVER_FINGER - 1
          else 0.18
        ),
      )
      actuator_indices = np.array(
        [int(name) for name in FINGERS[finger].joint_names],
        dtype=np.int32,
      )
      nominal_finger_command[actuator_indices] = joint_command
    current_finger_q = np.array(data.qpos[handles.hand_qpos_adrs], copy=True)
    if safety.override_delta_rad is not None:
      issued_finger_command = current_finger_q + safety.override_delta_rad
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
    data.ctrl[handles.hand_actuator_ids] = issued_finger_command
    previous_finger_command = issued_finger_command.copy()
    controller_latency[step] = perf_counter() - started
    physics_started = perf_counter()
    mujoco.mj_step(handles.model, data)
    physics_latency[step] = perf_counter() - physics_started

    measured_forces, measured_positions, measured_hfield_valid, non_tip_count = _contact_state(
      handles,
      data,
      bunny,
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
    if timestamp >= config.acquisition_s:
      empty_run = empty_run + 1 if not np.any(contact_active) else 0
      if config.cell == "fixed":
        fixed_loss_run = fixed_loss_run + 1 if not np.all(contact_active) else 0
        if fixed_loss_run * config.dt_s > 0.040 and stop_reason is None:
          stop_reason = "FIXED_MODE_LOST"
          stop_progress = progress
          events.append(
            {"event": "FIXED_MODE_LOST", "time_s": float(timestamp), "progress_m": progress}
          )
      elif empty_run * config.dt_s > 0.010 and stop_reason is None:
        stop_reason = "LAST_CONTACT_LOST"
        stop_progress = progress
    if np.max(measured_forces) > config.force_limit_n and stop_reason is None:
      stop_reason = "OVER_FORCE"
      stop_progress = progress
    if non_tip_count > 0 and stop_reason is None:
      stop_reason = "NON_TIP_BUNNY_COLLISION"
      stop_progress = progress
    if not np.all(np.isfinite(data.qpos)) and stop_reason is None:
      stop_reason = "NONFINITE_STATE"
      stop_progress = progress

    if phase in {"BREAK", "REPOSITION", "MAKE"} and stop_reason is None:
      snapshot = executor.consume_barrier_snapshot()
      if snapshot is not None:
        events.append(
          {
            "event": "MICRO_BARRIER",
            "time_s": float(timestamp),
            "phase": phase,
            "actual_contact_set": sorted(snapshot.actual_contact_set),
            "transaction_state": snapshot.transaction_state.value,
            "certificate_id": snapshot.certificate_id,
          }
        )
        if phase == "BREAK":
          pending_next_phase = "REPOSITION"
        elif phase == "REPOSITION":
          pending_next_phase = "MAKE"
        else:
          handover_complete = True
          phase = "NORMAL"

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
    planned_progress[step] = progress
    actual_progress[step] = max(0.0, initial_pose[0] - palm[step, 0])
    non_tip[step] = non_tip_count
    guard_reason[step] = stop_reason or safety.state.value
    transaction_phase[step] = phase
    if phase in {"BREAK", "REPOSITION", "MAKE"}:
      transaction_state[step] = executor.transaction_state.value
      if executor.current_transaction_id is not None and certificate_id[step] == "NONE":
        # The executor already authenticated this certificate on commit; the
        # last event carries the immutable certificate id for provenance.
        certificate_id[step] = str(events[-1].get("certificate_id", "NONE"))
      audit_latency[step] = current_audit_latency

  trace = I01BunnyTrace(
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
    planned_progress_m=planned_progress,
    actual_progress_m=actual_progress,
    controller_latency_s=controller_latency,
    physics_latency_s=physics_latency,
    non_tip_contact_count=non_tip,
    guard_reason=guard_reason,
    transaction_phase=transaction_phase,
    transaction_state=transaction_state,
    certificate_id=certificate_id,
    audit_latency_s=audit_latency,
    authority_violation=authority_violation,
    events=events,
  )
  metrics = evaluate_i01_trace(trace, bunny, handles.object_position_m)
  return trace, metrics, handles.xml


def evaluate_i01_trace(
  trace: I01BunnyTrace,
  bunny: BunnyHeightField,
  object_position_m: NDArray[np.float64],
) -> dict[str, Any]:
  """Apply the frozen exact-mesh contact filter and I01 pass criteria."""

  strict_force = trace.fingertip_forces_n >= trace.config.contact_threshold_n
  candidates = strict_force & trace.hfield_valid_contacts
  indices = np.argwhere(candidates)
  if len(indices):
    points_world = trace.contact_positions_world_m[indices[:, 0], indices[:, 1]]
    residuals = bunny.mesh_residuals(points_world - object_position_m)
    trace.mesh_residual_m[indices[:, 0], indices[:, 1]] = residuals
    valid = residuals <= trace.config.mesh_residual_limit_m
    trace.mesh_valid_contacts[indices[:, 0], indices[:, 1]] = valid
  evaluation = trace.time_s >= trace.config.acquisition_s
  contacts = trace.mesh_valid_contacts[evaluation]
  nonempty = np.any(contacts, axis=1)
  four = np.all(contacts, axis=1)
  empty_gap_s = _max_true_run(~nonempty) * trace.config.dt_s
  progress = trace.actual_progress_m[evaluation]
  valid_forces = np.where(trace.mesh_valid_contacts, trace.fingertip_forces_n, 0.0)
  peak_force = float(np.max(valid_forces[evaluation]))
  over_force_ticks = int(np.count_nonzero(valid_forces[evaluation] > trace.config.force_limit_n))
  residual_values = trace.mesh_residual_m[np.isfinite(trace.mesh_residual_m)]
  phase = trace.transaction_phase[evaluation]
  measured_masks = np.sum(
    contacts.astype(np.int32) * np.array([1, 2, 4, 8], dtype=np.int32)[None, :],
    axis=1,
  )
  # Reject contact-solver flicker as a topology event.  A reported mode must
  # persist for the protocol's 40 ms confirmation duration.
  stable_runs: list[tuple[int, int, int]] = []
  run_start = 0
  for index in range(1, len(measured_masks) + 1):
    if index == len(measured_masks) or measured_masks[index] != measured_masks[run_start]:
      if index - run_start >= int(round(0.040 / trace.config.dt_s)):
        stable_runs.append((int(measured_masks[run_start]), run_start, index))
      run_start = index
  stable_modes = [run[0] for run in stable_runs]
  transitions = [
    (left, right)
    for left, right in zip(stable_modes, stable_modes[1:])
    if left != right
  ]
  handover_4_3_4 = any(
    stable_modes[first] == 15
    and stable_modes[middle].bit_count() == 3
    and stable_modes[last] == 15
    for first in range(len(stable_modes))
    for middle in range(first + 1, len(stable_modes))
    for last in range(middle + 1, len(stable_modes))
  )
  primary_pass = bool(
    float(np.max(progress)) >= 0.050
    and float(np.mean(nonempty)) >= 0.990
    and empty_gap_s <= 0.010 + 1e-12
    and peak_force <= trace.config.force_limit_n + 1e-12
    and over_force_ticks == 0
    and int(np.sum(trace.non_tip_contact_count[evaluation])) == 0
    and not np.any(trace.authority_violation[evaluation])
    and not np.any(np.char.startswith(trace.guard_reason[evaluation], "NONFINITE"))
  )
  variable_mechanism_pass = bool(
    trace.config.cell == "variable"
    and handover_4_3_4
    and any(event["event"] == "CERTIFIED_COMMIT" for event in trace.events)
    and sum(event["event"] == "MICRO_BARRIER" for event in trace.events) >= 3
    and not np.any(trace.authority_violation)
  )
  return {
    "trace_schema_version": TRACE_SCHEMA_VERSION,
    "evaluator_version": EVALUATOR_VERSION,
    "cell": trace.config.cell,
    "seed": trace.config.seed,
    "primary_pass": primary_pass,
    "variable_mechanism_pass": variable_mechanism_pass,
    "actual_progress_m": float(np.max(progress)),
    "planned_progress_m": float(np.max(trace.planned_progress_m[evaluation])),
    "nonempty_contact_fraction": float(np.mean(nonempty)),
    "four_contact_fraction": float(np.mean(four)),
    "per_finger_contact_fraction": np.mean(contacts, axis=0).tolist(),
    "maximum_all_contact_loss_gap_s": empty_gap_s,
    "peak_valid_fingertip_force_n": peak_force,
    "over_force_ticks": over_force_ticks,
    "non_tip_contact_ticks": int(np.sum(trace.non_tip_contact_count[evaluation] > 0)),
    "authority_violation_count": int(np.count_nonzero(trace.authority_violation)),
    "handover_4_3_4_measured": handover_4_3_4,
    "certificate_count": int(sum(event["event"] == "CERTIFIED_COMMIT" for event in trace.events)),
    "micro_barrier_count": int(sum(event["event"] == "MICRO_BARRIER" for event in trace.events)),
    "mesh_contact_residual_p95_m": (
      float(np.percentile(residual_values, 95.0)) if len(residual_values) else None
    ),
    "mesh_contact_residual_max_m": (
      float(np.max(residual_values)) if len(residual_values) else None
    ),
    "mesh_rejected_contact_fraction": (
      float(
        np.count_nonzero(candidates & ~trace.mesh_valid_contacts)
        / max(np.count_nonzero(candidates), 1)
      )
      if np.any(candidates)
      else 0.0
    ),
    "controller_latency_p95_s": float(np.percentile(trace.controller_latency_s, 95.0)),
    "physics_latency_p95_s": float(np.percentile(trace.physics_latency_s, 95.0)),
    "audit_latency_p95_s": (
      float(np.percentile(trace.audit_latency_s[trace.audit_latency_s > 0.0], 95.0))
      if np.any(trace.audit_latency_s > 0.0)
      else None
    ),
    "stop_reason": next(
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
    ),
    "measured_mode_transitions": [list(edge) for edge in transitions],
    "stable_mode_runs": [
      {
        "mask": mask,
        "start_s": float(trace.time_s[np.flatnonzero(evaluation)[0] + start]),
        "duration_s": float((stop - start) * trace.config.dt_s),
      }
      for mask, start, stop in stable_runs
    ],
    "events": trace.events,
    "transaction_phase_coverage": {
      str(name): int(np.count_nonzero(phase == name))
      for name in np.unique(phase)
    },
  }


def save_trace(path: str | Path, trace: I01BunnyTrace) -> Path:
  output = Path(path)
  output.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(output, **trace.npz_payload())
  return output
