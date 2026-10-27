"""Physics runner for the MCC-only E05-F and E05-H cells."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import mujoco
import numpy as np
from numpy.typing import NDArray

from Module.e05_physics.extreme_surface import query_surface as query_extreme_surface
from Module.e05_physics.scene import FINGERS, PAD_HALF_SIZE_M
from Module.fr3_leap import (
  ARM_HOME_Q,
  HAND_NATURAL_Q,
  FullRobotModelConfig,
  build_full_robot,
)
from Module.module_2_fingertip_mcc import FingertipMCC, MCCConfig
from Module.module_3_runtime_guards import (
  CommandContinuityConfig,
  CommandContinuityLimiter,
  ForceSafetyConfig,
  ForceSafetyExecutor,
  ForceSafetyState,
  FullRobotGuardConfig,
  FullRobotGuardObservation,
  FullRobotRuntimeGuards,
)
from Module.module_4_whole_hand_mcc.coordinator import (
  ContactForceCoordinator,
  CoordinatorConfig,
)
from Module.module_4_whole_hand_mcc.robot_control import (
  JointTorqueWrenchEstimator,
  PalmPoseIK,
  PalmPoseIKConfig,
)
from Module.module_4_whole_hand_mcc.reference_interpreter import (
  ContactRole,
  ContactRoleInterpreter,
  RoleInterpreterConfig,
)
from Module.module_4_whole_hand_mcc.wrist_mcc import (
  WristMCC,
  WristMCCCommand,
  WristMCCConfig,
)


VALID_MODES = ("E05-F-MCC", "E05-H-MCC")


@dataclass(frozen=True, slots=True)
class E05MCCConfig:
  mode: str = "E05-F-MCC"
  surface: str = "extreme"
  duration_s: float = 15.0
  settling_time_s: float = 1.0
  dt_s: float = 0.002
  desired_force_n: float = 2.0
  contact_threshold_n: float = 0.20
  force_limit_n: float = 8.0
  traversal_y_m: float = 0.18
  lateral_primary_amplitude_m: float = 0.018
  lateral_secondary_amplitude_m: float = 0.007
  pose_step_time_s: float = 9.0
  pose_step_m: float = 0.004
  wrist_update_period_steps: int = 10
  force_filter_alpha: float = 0.20
  friction_coefficient: float = 0.90
  force_noise_std_n: float = 0.0
  initial_joint_noise_std_rad: float = 0.0
  wrist_surface_following: bool = False
  enforce_shared_force_safety: bool = False
  shared_active_finger_step_rad: float = 0.001
  shared_make_finger_step_rad: float = 0.00020
  shared_make_preload_m: float = 0.002
  shared_make_acquisition_force_n: float = 0.35
  shared_nominal_tangent_step_m: float = 0.00003
  shared_wrist_translation_step_m: float = 0.00008
  shared_wrist_normal_step_m: float = 0.00001
  shared_wrist_mcc_max_velocity_m_s: float = 0.005
  shared_wrist_mcc_max_acceleration_m_s2: float = 0.25
  shared_hard_wrist_release_target_m: float = 0.0007
  shared_hard_wrist_step_m: float = 0.0003
  shared_hard_finger_step_rad: float = 0.0005
  shared_soft_wrist_release_target_m: float = 0.0002
  shared_soft_wrist_step_m: float = 0.00004
  shared_hand_kp: float = 22.0
  shared_arm_kp: float = 1200.0
  shared_contact_solref_s: float = 0.028
  shared_hand_torque_limit_nm: float = 0.7
  shared_rapid_loading_rate_n_s: float = 2000.0
  shared_rapid_loading_min_force_n: float = 5.0
  shared_soft_force_n: float = 8.0
  shared_recover_force_n: float = 5.0
  shared_hard_force_n: float = 20.0
  shared_soft_finger_authority_scale: float = 0.70
  shared_soft_wrist_velocity_scale: float = 0.70
  shared_soft_release_gain: float = 0.0
  shared_minimum_plan_contacts: int = 1
  shared_guard_stable_time_s: float = 0.02
  shared_guard_reentry_ramp_time_s: float = 0.04
  role_confirm_time_s: float = 0.030
  role_force_ramp_time_s: float = 0.20
  role_make_force_threshold_n: float = 0.05
  role_break_force_threshold_n: float = 0.02
  role_make_confirmation_motion_scale: float = 0.50
  seed: int = 7

  def __post_init__(self) -> None:
    if self.mode not in VALID_MODES:
      raise ValueError(f"mode must be one of {VALID_MODES}")
    if self.surface not in {"plane", "sphere", "extreme"}:
      raise ValueError("surface must be plane, sphere, or extreme")
    positive = {
      "duration_s": self.duration_s,
      "dt_s": self.dt_s,
      "desired_force_n": self.desired_force_n,
      "contact_threshold_n": self.contact_threshold_n,
      "force_limit_n": self.force_limit_n,
      "traversal_y_m": self.traversal_y_m,
      "wrist_update_period_steps": self.wrist_update_period_steps,
      "force_filter_alpha": self.force_filter_alpha,
      "friction_coefficient": self.friction_coefficient,
      "shared_active_finger_step_rad": self.shared_active_finger_step_rad,
      "shared_make_finger_step_rad": self.shared_make_finger_step_rad,
      "shared_make_acquisition_force_n": self.shared_make_acquisition_force_n,
      "shared_nominal_tangent_step_m": self.shared_nominal_tangent_step_m,
      "shared_wrist_translation_step_m": self.shared_wrist_translation_step_m,
      "shared_wrist_normal_step_m": self.shared_wrist_normal_step_m,
      "shared_wrist_mcc_max_velocity_m_s": self.shared_wrist_mcc_max_velocity_m_s,
      "shared_wrist_mcc_max_acceleration_m_s2": self.shared_wrist_mcc_max_acceleration_m_s2,
      "shared_hard_wrist_release_target_m": self.shared_hard_wrist_release_target_m,
      "shared_hard_wrist_step_m": self.shared_hard_wrist_step_m,
      "shared_hard_finger_step_rad": self.shared_hard_finger_step_rad,
      "shared_soft_wrist_release_target_m": self.shared_soft_wrist_release_target_m,
      "shared_soft_wrist_step_m": self.shared_soft_wrist_step_m,
      "shared_hand_kp": self.shared_hand_kp,
      "shared_arm_kp": self.shared_arm_kp,
      "shared_contact_solref_s": self.shared_contact_solref_s,
      "shared_hand_torque_limit_nm": self.shared_hand_torque_limit_nm,
      "shared_rapid_loading_rate_n_s": self.shared_rapid_loading_rate_n_s,
      "shared_rapid_loading_min_force_n": self.shared_rapid_loading_min_force_n,
      "shared_soft_force_n": self.shared_soft_force_n,
      "shared_recover_force_n": self.shared_recover_force_n,
      "shared_hard_force_n": self.shared_hard_force_n,
      "shared_guard_stable_time_s": self.shared_guard_stable_time_s,
      "shared_guard_reentry_ramp_time_s": self.shared_guard_reentry_ramp_time_s,
      "role_confirm_time_s": self.role_confirm_time_s,
      "role_force_ramp_time_s": self.role_force_ramp_time_s,
      "role_make_force_threshold_n": self.role_make_force_threshold_n,
      "role_break_force_threshold_n": self.role_break_force_threshold_n,
    }
    for name, value in positive.items():
      if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    if self.settling_time_s < 0.0 or self.settling_time_s >= self.duration_s:
      raise ValueError("settling_time_s must be in [0,duration_s)")
    if self.pose_step_time_s <= self.settling_time_s or self.pose_step_time_s >= self.duration_s:
      raise ValueError("pose_step_time_s must be inside the evaluation interval")
    if self.wrist_surface_following and self.surface != "extreme":
      raise ValueError("v1 wrist surface following is defined for the extreme surface")
    if self.role_break_force_threshold_n >= self.role_make_force_threshold_n:
      raise ValueError("role break force threshold must be below make threshold")
    if not 0.0 < self.role_make_confirmation_motion_scale <= 1.0:
      raise ValueError("role_make_confirmation_motion_scale must be in (0,1]")
    if not self.shared_recover_force_n < self.shared_soft_force_n < self.shared_hard_force_n:
      raise ValueError(
        "shared force thresholds must satisfy recover < soft < hard"
      )
    for name in (
      "shared_soft_finger_authority_scale",
      "shared_soft_wrist_velocity_scale",
    ):
      value = float(getattr(self, name))
      if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be in (0,1)")
    if not 0.0 <= self.shared_soft_release_gain <= 1.0:
      raise ValueError("shared_soft_release_gain must be in [0,1]")
    if not 1 <= int(self.shared_minimum_plan_contacts) <= 4:
      raise ValueError("shared_minimum_plan_contacts must be in [1,4]")
    if not 0.0 < self.force_filter_alpha <= 1.0:
      raise ValueError("force_filter_alpha must be in (0,1]")
    for name in (
      "lateral_primary_amplitude_m",
      "lateral_secondary_amplitude_m",
      "pose_step_m",
      "shared_make_preload_m",
      "force_noise_std_n",
      "initial_joint_noise_std_rad",
    ):
      value = getattr(self, name)
      if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class E05MCCTrace:
  time_s: NDArray[np.float64]
  arm_q_rad: NDArray[np.float64]
  arm_dq_rad_s: NDArray[np.float64]
  arm_command_rad: NDArray[np.float64]
  finger_q_rad: NDArray[np.float64]
  finger_dq_rad_s: NDArray[np.float64]
  finger_command_rad: NDArray[np.float64]
  palm_pose_world: NDArray[np.float64]
  planned_palm_pose_world: NDArray[np.float64]
  commanded_palm_pose_world: NDArray[np.float64]
  fingertip_positions_world_m: NDArray[np.float64]
  pad_normals_world: NDArray[np.float64]
  contact_positions_world_m: NDArray[np.float64]
  contact_normals_world: NDArray[np.float64]
  fingertip_forces_n: NDArray[np.float64]
  actual_contacts: NDArray[np.bool_]
  desired_hand_wrench_world: NDArray[np.float64]
  estimated_hand_wrench_world: NDArray[np.float64]
  contact_hand_wrench_world: NDArray[np.float64]
  arm_external_torque_nm: NDArray[np.float64]
  wrist_compliance_offset: NDArray[np.float64]
  finger_compliance_offsets_m: NDArray[np.float64]
  coordinator_rank: NDArray[np.int32]
  coordinator_condition: NDArray[np.float64]
  coordinator_internal_leakage_n: NDArray[np.float64]
  surface_curvature_inv_m: NDArray[np.float64]
  disturbance_active: NDArray[np.bool_]
  controller_latency_s: NDArray[np.float64]
  physics_step_latency_s: NDArray[np.float64]
  loop_latency_s: NDArray[np.float64]
  guard_reason: NDArray[np.str_]
  non_tip_contact_count: NDArray[np.int32]


def _quaternion_from_matrix(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
  quaternion = np.zeros(4, dtype=np.float64)
  mujoco.mju_mat2Quat(quaternion, np.asarray(matrix, dtype=np.float64).reshape(9))
  return quaternion


def _smoothstep(value: float) -> float:
  clipped = float(np.clip(value, 0.0, 1.0))
  return clipped * clipped * (3.0 - 2.0 * clipped)


def _planned_palm_pose(
  initial_pose: NDArray[np.float64],
  config: E05MCCConfig,
  timestamp_s: float,
  *,
  object_position_m: NDArray[np.float64] | None = None,
  fingertip_xy_offsets_m: NDArray[np.float64] | None = None,
  initial_mean_surface_height_m: float | None = None,
) -> NDArray[np.float64]:
  pose = initial_pose.copy()
  motion_time = max(0.0, timestamp_s - config.settling_time_s)
  motion_duration = max(config.duration_s - config.settling_time_s, config.dt_s)
  progress = _smoothstep(motion_time / motion_duration)
  pose[0] -= (
    config.lateral_primary_amplitude_m * np.sin(2.0 * np.pi * progress)
    + config.lateral_secondary_amplitude_m * np.sin(6.0 * np.pi * progress)
  )
  pose[1] += config.traversal_y_m * progress
  if config.wrist_surface_following:
    if (
      object_position_m is None
      or fingertip_xy_offsets_m is None
      or initial_mean_surface_height_m is None
    ):
      raise ValueError("surface-following wrist plan requires frozen geometry anchors")
    heights = []
    for xy_offset in np.asarray(fingertip_xy_offsets_m, dtype=np.float64):
      sample_xy = pose[:2] + xy_offset
      local_xy = sample_xy - object_position_m[:2]
      height, _, _ = query_extreme_surface(float(local_xy[0]), float(local_xy[1]))
      heights.append(float(object_position_m[2] + height))
    pose[2] += float(np.mean(heights) - initial_mean_surface_height_m)
  if timestamp_s >= config.pose_step_time_s:
    # Moving the wrist away from the fixed surface is the exact counterpart of
    # the old inverse-mocap object -Z step.
    pose[2] += config.pose_step_m
  return pose


def _surface_reference(
  surface: str,
  object_position: NDArray[np.float64],
  point_world: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
  if surface == "plane":
    normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    top_z = float(object_position[2] + 0.01)
    return np.array([point_world[0], point_world[1], top_z]), normal, 0.0
  if surface == "sphere":
    radial = point_world - object_position
    length = float(np.linalg.norm(radial))
    normal = radial / length
    return object_position + 0.45 * normal, normal, 1.0 / 0.45
  local = point_world - object_position
  height, normal, curvature = query_extreme_surface(float(local[0]), float(local[1]))
  return object_position + np.array([local[0], local[1], height]), normal, curvature


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
) -> NDArray[np.float64]:
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
  command = current_q + gain * delta + 0.01 * (
    HAND_NATURAL_Q[nominal_indices] - current_q
  )
  # A resolved-rate target can jump when the hfield normal changes rapidly.
  # Rate-limit the commanded joint displacement before actuator/joint clipping
  # so a narrow ridge cannot become a one-frame impact command.
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


def _fingertip_jacobian_world(
  handles: Any,
  data: mujoco.MjData,
) -> NDArray[np.float64]:
  """Return four 3x16 finger-only fingertip Jacobians in world axes."""

  result = np.zeros((4, 3, 16), dtype=np.float64)
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
    columns = slice(4 * finger, 4 * finger + 4)
    result[finger, :, columns] = jacobian_position[
      :, handles.finger_dof_adrs[finger]
    ]
  return result


def _contact_state(
  handles: Any,
  data: mujoco.MjData,
) -> tuple[
  NDArray[np.float64],
  NDArray[np.float64],
  NDArray[np.float64],
  int,
]:
  forces = np.zeros(4, dtype=np.float64)
  force_vectors = np.zeros((4, 3), dtype=np.float64)
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
    world_force = contact.frame.reshape(3, 3).T @ contact_force[:3]
    # Choose object-on-hand sign deterministically from the analytic outward
    # normal rather than relying on geom1/geom2 ordering.
    _, outward, _ = _surface_reference(
      handles.config.surface,
      handles.object_position_m,
      np.asarray(contact.pos),
    )
    if float(np.dot(world_force, outward)) < 0.0:
      world_force = -world_force
    forces[finger_index] += normal_force
    force_vectors[finger_index] += world_force
    weighted_positions[finger_index] += normal_force * np.asarray(contact.pos)
  positions = np.zeros((4, 3), dtype=np.float64)
  valid = forces > 0.0
  positions[valid] = weighted_positions[valid] / forces[valid, None]
  return forces, force_vectors, positions, non_tip


def _contact_wrench(
  force_vectors: NDArray[np.float64],
  contact_positions: NDArray[np.float64],
  valid: NDArray[np.bool_],
  reference: NDArray[np.float64],
) -> NDArray[np.float64]:
  wrench = np.zeros(6, dtype=np.float64)
  for index in np.flatnonzero(valid):
    wrench[:3] += force_vectors[index]
    wrench[3:] += np.cross(contact_positions[index] - reference, force_vectors[index])
  return wrench


def _initialize_data(handles: Any, config: E05MCCConfig) -> mujoco.MjData:
  rng = np.random.default_rng(config.seed)
  data = mujoco.MjData(handles.model)
  arm_q = ARM_HOME_Q + rng.normal(0.0, config.initial_joint_noise_std_rad, 7)
  finger_q = HAND_NATURAL_Q + rng.normal(
    0.0,
    config.initial_joint_noise_std_rad,
    16,
  )
  arm_q = np.clip(
    arm_q,
    handles.arm_joint_ranges_rad[:, 0] + 0.04,
    handles.arm_joint_ranges_rad[:, 1] - 0.04,
  )
  finger_q = np.clip(
    finger_q,
    handles.hand_joint_ranges_rad[:, 0] + 0.05,
    handles.hand_joint_ranges_rad[:, 1] - 0.05,
  )
  data.qpos[handles.arm_qpos_adrs] = arm_q
  data.qpos[handles.hand_qpos_adrs] = finger_q
  data.ctrl[handles.arm_actuator_ids] = arm_q
  data.ctrl[handles.hand_actuator_ids] = finger_q
  mujoco.mj_forward(handles.model, data)
  return data


def run_e05_mcc(
  config: E05MCCConfig = E05MCCConfig(),
  *,
  reference_source: Any | None = None,
) -> tuple[E05MCCTrace, dict[str, Any]]:
  handles = build_full_robot(
    FullRobotModelConfig(
      surface=config.surface,
      timestep_s=config.dt_s,
      gravity_m_s2=0.0,
      arm_kp=(config.shared_arm_kp if config.enforce_shared_force_safety else 1800.0),
      arm_damping_ratio=0.9,
      hand_kp=(config.shared_hand_kp if config.enforce_shared_force_safety else 22.0),
      hand_actuator_force_limit_nm=(
        config.shared_hand_torque_limit_nm
        if config.enforce_shared_force_safety
        else None
      ),
    )
  )
  if config.enforce_shared_force_safety:
    contact_geom_ids = np.concatenate(
      (handles.tip_geom_ids, np.array([handles.object_geom_id], dtype=np.int32))
    )
    handles.model.geom_solref[contact_geom_ids, 0] = config.shared_contact_solref_s
  handles.model.geom_friction[handles.object_geom_id, 0] = config.friction_coefficient
  handles.model.geom_friction[handles.tip_geom_ids, 0] = config.friction_coefficient
  data = _initialize_data(handles, config)
  nominal_data = mujoco.MjData(handles.model)
  rng = np.random.default_rng(config.seed + 103)
  initial_pose = np.concatenate(
    (
      data.site_xpos[handles.palm_site_id].copy(),
      _quaternion_from_matrix(data.site_xmat[handles.palm_site_id]),
    )
  )
  pose_ik = PalmPoseIK(
    handles,
    PalmPoseIKConfig(gain=0.32, damping=0.018, max_joint_step_rad=0.02),
  )
  estimator = JointTorqueWrenchEstimator(handles)
  coordinator = ContactForceCoordinator(
    CoordinatorConfig(transition_blend_steps=25, damping=1e-8)
  )
  wrist_mcc = WristMCC(
    WristMCCConfig(
      virtual_mass=(6.0, 6.0, 6.0, 0.3, 0.3, 0.3),
      damping=(150.0, 150.0, 150.0, 8.0, 8.0, 8.0),
      stiffness=(150.0, 150.0, 150.0, 35.0, 35.0, 35.0),
      dt_s=config.dt_s * config.wrist_update_period_steps,
      max_abs_offset=(0.012, 0.012, 0.012, 0.08, 0.08, 0.08),
      max_abs_velocity=(
        *(
          (config.shared_wrist_mcc_max_velocity_m_s,) * 3
          if config.enforce_shared_force_safety
          else (0.04, 0.04, 0.04)
        ),
        0.3,
        0.3,
        0.3,
      ),
      max_abs_acceleration=(
        *(
          (config.shared_wrist_mcc_max_acceleration_m_s2,) * 3
          if config.enforce_shared_force_safety
          else (0.8, 0.8, 0.8)
        ),
        5.0,
        5.0,
        5.0,
      ),
    )
  )
  finger_controllers = tuple(
    FingertipMCC(
      MCCConfig(
        virtual_mass=0.12,
        damping=20.0,
        stiffness=40.0,
        dt_s=config.dt_s,
        max_offset_m=0.010,
        max_velocity_m_s=0.015,
        max_acceleration_m_s2=2.0,
      )
    )
    for _ in range(4)
  )
  # MAKE/recontact has a separate low-force Cartesian admittance state.  It
  # closes the small pre-contact gap faster than a pure joint-space crawl, but
  # its low target and bounded offset prevent the hidden 2 N penetration target
  # used by the historical controller.
  make_controllers = tuple(
    FingertipMCC(
      MCCConfig(
        virtual_mass=0.12,
        damping=20.0,
        stiffness=40.0,
        dt_s=config.dt_s,
        max_offset_m=0.008,
        max_velocity_m_s=0.010,
        max_acceleration_m_s2=1.5,
      )
    )
    for _ in range(4)
  )
  guards = FullRobotRuntimeGuards(
    FullRobotGuardConfig(
      arm_joint_lower_rad=handles.arm_joint_ranges_rad[:, 0],
      arm_joint_upper_rad=handles.arm_joint_ranges_rad[:, 1],
      finger_joint_lower_rad=handles.hand_joint_ranges_rad[:, 0],
      finger_joint_upper_rad=handles.hand_joint_ranges_rad[:, 1],
      dt_s=config.dt_s,
      max_tip_force_n=config.force_limit_n,
    )
  )
  shared_force_safety = (
    ForceSafetyExecutor(
      ForceSafetyConfig(
        joint_lower_rad=handles.hand_joint_ranges_rad[:, 0],
        joint_upper_rad=handles.hand_joint_ranges_rad[:, 1],
        dt_s=config.dt_s,
        soft_force_n=config.shared_soft_force_n,
        hard_force_n=config.shared_hard_force_n,
        recover_force_n=config.shared_recover_force_n,
        stable_time_s=config.shared_guard_stable_time_s,
        reentry_ramp_time_s=config.shared_guard_reentry_ramp_time_s,
        soft_finger_authority_scale=config.shared_soft_finger_authority_scale,
        soft_wrist_velocity_scale=config.shared_soft_wrist_velocity_scale,
        soft_release_gain=config.shared_soft_release_gain,
        rapid_loading_rate_n_s=config.shared_rapid_loading_rate_n_s,
        rapid_loading_min_force_n=config.shared_rapid_loading_min_force_n,
      )
    )
    if config.enforce_shared_force_safety
    else None
  )

  count = int(round(config.duration_s / config.dt_s))
  time_s = np.arange(count, dtype=np.float64) * config.dt_s
  arm_q = np.zeros((count, 7))
  arm_dq = np.zeros((count, 7))
  arm_cmd = np.zeros((count, 7))
  finger_q = np.zeros((count, 16))
  finger_dq = np.zeros((count, 16))
  finger_cmd = np.zeros((count, 16))
  palm = np.zeros((count, 7))
  planned_palm = np.zeros((count, 7))
  commanded_palm = np.zeros((count, 7))
  tips = np.zeros((count, 4, 3))
  pad_normals = np.zeros((count, 4, 3))
  contact_positions = np.zeros((count, 4, 3))
  contact_normals = np.zeros((count, 4, 3))
  forces_log = np.zeros((count, 4))
  actual_contacts = np.zeros((count, 4), dtype=np.bool_)
  desired_wrench_log = np.zeros((count, 6))
  estimated_wrench_log = np.zeros((count, 6))
  contact_wrench_log = np.zeros((count, 6))
  torque_log = np.zeros((count, 7))
  wrist_offset_log = np.zeros((count, 6))
  finger_offset_log = np.zeros((count, 4))
  coordinator_rank = np.zeros(count, dtype=np.int32)
  coordinator_condition = np.full(count, np.nan)
  coordinator_leakage = np.zeros(count)
  curvature_log = np.zeros((count, 4))
  disturbance = np.zeros(count, dtype=np.bool_)
  latency = np.zeros(count)
  physics_latency = np.zeros(count)
  loop_latency = np.zeros(count)
  guard_reason = np.full(count, "NONE", dtype="U40")
  non_tip = np.zeros(count, dtype=np.int32)
  role_log = np.full((count, 4), int(ContactRole.KEEP), dtype=np.int64)
  requested_role_log = np.full((count, 4), int(ContactRole.KEEP), dtype=np.int64)
  role_veto_count = 0
  reference_inference_latencies: list[float] = []
  reference_inference_count = 0

  measured_forces = np.zeros(4)
  measured_force_vectors = np.zeros((4, 3))
  measured_positions = np.zeros((4, 3))
  contact_active = np.zeros(4, dtype=np.bool_)
  filtered_wrench = np.zeros(6)
  desired_wrench = np.zeros(6)
  current_wrist_command = initial_pose.copy()
  wrist_command_state: WristMCCCommand | None = None
  previous_arm_control = np.array(data.ctrl[handles.arm_actuator_ids], copy=True)
  previous_finger_control = np.array(data.ctrl[handles.hand_actuator_ids], copy=True)
  previous_nominal_command = previous_finger_control.copy()
  previous_mcc_correction = np.zeros(16, dtype=np.float64)
  nominal_reference_chunk = np.repeat(
    previous_nominal_command[None, :],
    20,
    axis=0,
  )
  nominal_free_authority_mask = np.zeros(4, dtype=np.bool_)
  nominal_tangent_authority_mask = np.zeros(4, dtype=np.bool_)
  if reference_source is not None:
    if not config.enforce_shared_force_safety:
      raise ValueError("Exp. 2 reference sources require the shared G1a safety stack")
    reference_source.reset()
  command_continuity = (
    CommandContinuityLimiter(
      CommandContinuityConfig(
        max_finger_step_rad=config.shared_active_finger_step_rad,
        max_wrist_translation_step_m=config.shared_wrist_translation_step_m,
      )
    )
    if shared_force_safety is not None
    else None
  )
  role_interpreter = (
    ContactRoleInterpreter(
      RoleInterpreterConfig(
        dt_s=config.dt_s,
        make_force_threshold_n=config.role_make_force_threshold_n,
        break_force_threshold_n=config.role_break_force_threshold_n,
        make_confirm_time_s=config.role_confirm_time_s,
        break_confirm_time_s=config.role_confirm_time_s,
        load_ramp_time_s=config.role_force_ramp_time_s,
        make_confirmation_motion_scale=config.role_make_confirmation_motion_scale,
      )
    )
    if shared_force_safety is not None
    else None
  )
  if command_continuity is not None:
    command_continuity.reset(
      finger_command_rad=previous_finger_control,
      wrist_pose_world=initial_pose,
    )
  fingertip_xy_offsets = (
    data.site_xpos[handles.tip_site_ids, :2] - initial_pose[None, :2]
  )
  initial_surface_heights: list[float] = []
  for xy_offset in fingertip_xy_offsets:
    sample_xy = initial_pose[:2] + xy_offset
    local_xy = sample_xy - handles.object_position_m[:2]
    height, _, _ = query_extreme_surface(float(local_xy[0]), float(local_xy[1]))
    initial_surface_heights.append(float(handles.object_position_m[2] + height))
  initial_mean_surface_height = float(np.mean(initial_surface_heights))
  executed_plan_time_s = 0.0
  plan_clock_scale = 1.0 if role_interpreter is None else 0.0

  for step, timestamp_s in enumerate(time_s):
    tic = perf_counter()
    if step > 0 and role_interpreter is not None:
      executed_plan_time_s += config.dt_s * plan_clock_scale
    planning_timestamp_s = (
      float(timestamp_s) if role_interpreter is None else executed_plan_time_s
    )
    planned_pose = _planned_palm_pose(
      initial_pose,
      config,
      planning_timestamp_s,
      object_position_m=handles.object_position_m,
      fingertip_xy_offsets_m=fingertip_xy_offsets,
      initial_mean_surface_height_m=initial_mean_surface_height,
    )
    surface_points = np.zeros((4, 3))
    normals = np.zeros((4, 3))
    curvatures = np.zeros(4)
    touching_centers = np.zeros((4, 3))
    for index in range(4):
      current_tip = np.array(data.site_xpos[int(handles.tip_site_ids[index])], copy=True)
      surface_point, normal, curvature = _surface_reference(
        config.surface,
        handles.object_position_m,
        current_tip,
      )
      surface_points[index] = surface_point
      normals[index] = normal
      curvatures[index] = curvature
      touching_centers[index] = surface_point + _pad_support_radius(
        handles,
        data,
        index,
        normal,
      ) * normal

    current_pose_for_reference = np.concatenate(
      (
        data.site_xpos[handles.palm_site_id].copy(),
        _quaternion_from_matrix(data.site_xmat[handles.palm_site_id]),
      )
    )
    current_finger_q_for_reference = np.array(
      data.qpos[handles.hand_qpos_adrs],
      copy=True,
    )
    requested_roles = np.full(4, int(ContactRole.KEEP), dtype=np.int64)
    if reference_source is not None:
      from Module.module_4_finger_dp.dpref_reference_sources import (
        ReferenceSourceContext,
      )

      policy_dt_s = config.dt_s * config.wrist_update_period_steps
      future_plan_poses = np.stack(
        [
          _planned_palm_pose(
            initial_pose,
            config,
            planning_timestamp_s + policy_dt_s * (horizon + 1),
            object_position_m=handles.object_position_m,
            fingertip_xy_offsets_m=fingertip_xy_offsets,
            initial_mean_surface_height_m=initial_mean_surface_height,
          )
          for horizon in range(20)
        ]
      )
      reference_output = reference_source.step(
        ReferenceSourceContext(
          step=step,
          timestamp_s=float(timestamp_s),
          dt_s=config.dt_s,
          policy_period_steps=config.wrist_update_period_steps,
          desired_force_n=config.desired_force_n,
          finger_q_rad=current_finger_q_for_reference,
          finger_dq_rad_s=np.array(data.qvel[handles.hand_dof_adrs], copy=True),
          fingertip_force_n=measured_forces.copy(),
          actual_contact_mask=contact_active.copy(),
          fingertip_positions_world_m=np.array(
            data.site_xpos[handles.tip_site_ids],
            copy=True,
          ),
          contact_positions_world_m=measured_positions.copy(),
          contact_normals_world=normals.copy(),
          palm_pose_world=current_pose_for_reference,
          future_plan_poses_world=future_plan_poses,
          object_position_world_m=handles.object_position_m.copy(),
          wrist_mcc_offset_world=wrist_mcc.state.offset,
          wrist_mcc_velocity_world=wrist_mcc.state.velocity,
          fingertip_jacobian_world=_fingertip_jacobian_world(handles, data),
          previous_nominal_command_rad=previous_nominal_command.copy(),
          previous_mcc_correction_rad=previous_mcc_correction.copy(),
        )
      )
      nominal_reference_chunk = np.asarray(
        reference_output.nominal_command_chunk_rad,
        dtype=np.float64,
      )
      if nominal_reference_chunk.shape != (20, 16):
        raise ValueError("reference source nominal chunk must have shape (20,16)")
      requested_roles = np.asarray(
        reference_output.requested_roles,
        dtype=np.int64,
      ).copy()
      if requested_roles.shape != (4,):
        raise ValueError("reference source roles must have shape (4,)")
      nominal_free_authority_mask = np.asarray(
        reference_output.nominal_free_authority_mask,
        dtype=np.bool_,
      )
      if nominal_free_authority_mask.shape != (4,):
        raise ValueError("nominal free-authority mask must have shape (4,)")
      nominal_tangent_authority_mask = np.asarray(
        reference_output.nominal_tangent_authority_mask,
        dtype=np.bool_,
      )
      if nominal_tangent_authority_mask.shape != (4,):
        raise ValueError("nominal tangent-authority mask must have shape (4,)")
      if reference_output.inference_executed:
        reference_inference_count += 1
        reference_inference_latencies.append(
          float(reference_output.inference_latency_s)
        )

      # A categorical head proposes an intention; current SurfaceModel
      # kinematics veto a MAKE/RELEASE whose nominal chunk moves in the wrong
      # signed normal direction.  The role FSM remains the only authority that
      # can approve a surviving request.
      nominal_data.qpos[:] = data.qpos
      nominal_data.qpos[handles.hand_qpos_adrs] = np.clip(
        nominal_reference_chunk[-1],
        handles.hand_joint_ranges_rad[:, 0] + 0.02,
        handles.hand_joint_ranges_rad[:, 1] - 0.02,
      )
      mujoco.mj_forward(handles.model, nominal_data)
      wrist_delta = future_plan_poses[-1, :3] - current_pose_for_reference[:3]
      for index in range(4):
        motion = (
          nominal_data.site_xpos[handles.tip_site_ids[index]]
          - data.site_xpos[handles.tip_site_ids[index]]
          + wrist_delta
        )
        signed_outward = float(np.dot(motion, normals[index]))
        role = ContactRole(int(requested_roles[index]))
        if role is ContactRole.MAKE and signed_outward >= -0.00002:
          requested_roles[index] = int(
            ContactRole.KEEP if contact_active[index] else ContactRole.FREE
          )
          role_veto_count += 1
        elif role is ContactRole.RELEASE and signed_outward <= 0.00002:
          requested_roles[index] = int(
            ContactRole.KEEP if contact_active[index] else ContactRole.FREE
          )
          role_veto_count += 1

    if role_interpreter is None:
      active = contact_active.copy()
      interpreted_desired_force = np.full(4, config.desired_force_n)
      executable_roles = np.where(
        active,
        int(ContactRole.KEEP),
        int(ContactRole.MAKE),
      ).astype(np.int64)
      # Preserve the frozen historical MCC-only protocol when the new shared
      # interpreter/safety stack is not explicitly enabled.
      full_reference_authority = np.zeros(4, dtype=np.bool_)
      reference_motion_scale = np.ones(4, dtype=np.float64)
    else:
      # Passive shared-stack source: keep every established anchor and recover
      # any lost contact. Exp. 2 swaps only this request/reference source.
      role_output = role_interpreter.step(
        requested_roles=requested_roles,
        measured_force_n=measured_forces,
        target_force_n=np.full(4, config.desired_force_n),
      )
      active = role_output.mcc_enabled.copy()
      interpreted_desired_force = role_output.desired_force_n.copy()
      executable_roles = role_output.roles.copy()
      full_reference_authority = role_output.full_reference_authority.copy()
      reference_motion_scale = role_output.reference_motion_scale.copy()

    safety_output = None
    if shared_force_safety is not None:
      safety_output = shared_force_safety.step(
        fingertip_force_n=measured_forces,
        force_valid_mask=np.ones(4, dtype=np.bool_),
        history_ready=True,
        current_q_rad=data.qpos[handles.hand_qpos_adrs],
        signed_compression_jacobian=_signed_compression_jacobian(
          handles,
          data,
          normals,
        ),
      )

    estimate = estimator.estimate(data)
    filtered_wrench = (
      (1.0 - config.force_filter_alpha) * filtered_wrench
      + config.force_filter_alpha * estimate.wrench_world
    )
    internal_error = np.zeros(4)
    active_positions = measured_positions.copy()
    active_positions[measured_forces <= 0.0] = surface_points[measured_forces <= 0.0]
    if np.any(active):
      coordinated = coordinator.step(
        active_positions,
        normals,
        interpreted_desired_force,
        measured_forces,
        active,
        data.site_xpos[handles.palm_site_id],
      )
      desired_wrench = coordinated.desired_hand_wrench_world.copy()
      internal_error = coordinated.internal_force_error_n.copy()
      coordinator_rank[step] = coordinated.rank
      coordinator_condition[step] = coordinated.condition_number
      coordinator_leakage[step] = coordinated.internal_wrench_leakage_norm
    else:
      desired_wrench[:] = 0.0

    current_pose = np.concatenate(
      (
        data.site_xpos[handles.palm_site_id].copy(),
        _quaternion_from_matrix(data.site_xmat[handles.palm_site_id]),
      )
    )
    wrist_authority_scale = (
      1.0 if safety_output is None else safety_output.wrist_velocity_scale
    )
    if role_interpreter is not None:
      # Whole-hand tangential exploration never advances without one physical
      # anchor. Wrist MCC nevertheless retains collective-normal recovery
      # authority around this frozen plan reference.
      supported = bool(
        np.count_nonzero(
          active & (measured_forces >= 0.5 * config.contact_threshold_n)
        )
        >= config.shared_minimum_plan_contacts
      )
      plan_clock_scale = (
        wrist_authority_scale if supported else 0.0
      )
    wrist_authority_available = wrist_authority_scale > 0.0
    if (
      config.mode == "E05-H-MCC"
      and step % config.wrist_update_period_steps == 0
      and wrist_authority_available
    ):
      selection = np.zeros((6, 6), dtype=np.float64)
      if np.any(active):
        collective_normal = np.mean(normals[active], axis=0)
        collective_normal /= np.linalg.norm(collective_normal)
        selection[:3, :3] = np.outer(collective_normal, collective_normal)
      wrist_command_state = wrist_mcc.step(
        planned_pose,
        desired_wrench,
        filtered_wrench,
        selection,
      )
      current_wrist_command = wrist_command_state.pose_command.copy()
    elif config.mode == "E05-H-MCC" and not wrist_authority_available:
      # Do not let an unissued Wrist-MCC state integrate behind SAFE_HOLD and
      # reappear as a large latent target when authority returns.
      wrist_mcc.reset()
      wrist_command_state = None
      current_wrist_command = planned_pose.copy()
    elif config.mode == "E05-F-MCC":
      current_wrist_command = planned_pose.copy()
    if safety_output is not None and safety_output.affected_fingers:
      command_normal = np.mean(
        normals[np.asarray(safety_output.affected_fingers, dtype=np.int32)],
        axis=0,
      )
    elif np.any(active):
      command_normal = np.mean(normals[active], axis=0)
    else:
      command_normal = np.mean(normals, axis=0)
    command_normal /= np.linalg.norm(command_normal)
    issued_wrist_command = current_wrist_command.copy()
    if wrist_authority_scale < 1.0:
      scale = wrist_authority_scale
      issued_wrist_command[:3] = current_pose[:3] + scale * (
        current_wrist_command[:3] - current_pose[:3]
      )
      target_quaternion = current_wrist_command[3:].copy()
      if np.dot(current_pose[3:], target_quaternion) < 0.0:
        target_quaternion *= -1.0
      blended = (1.0 - scale) * current_pose[3:] + scale * target_quaternion
      issued_wrist_command[3:] = blended / np.linalg.norm(blended)
    if safety_output is not None and safety_output.state in {
      ForceSafetyState.SOFT_RECOVERY,
      ForceSafetyState.HARD_RELEASE,
    } and safety_output.affected_fingers:
      # Surface normals point object->hand, so +normal is an unambiguous
      # collective decompression direction. Holding the arm target alone is
      # insufficient because the high-gain position servo can keep loading the
      # pad while the finger release is still taking effect.
      issued_wrist_command[:3] = (
        current_pose[:3]
        + (
          config.shared_hard_wrist_release_target_m
          if safety_output.state is ForceSafetyState.HARD_RELEASE
          else config.shared_soft_wrist_release_target_m
        )
        * command_normal
      )
      issued_wrist_command[3:] = current_pose[3:]
    if command_continuity is not None:
      bounded_wrist_scale = max(wrist_authority_scale, 0.02)
      hard_releasing = (
        safety_output is not None
        and safety_output.state is ForceSafetyState.HARD_RELEASE
      )
      issued_wrist_command = command_continuity.limit_wrist(
        issued_wrist_command,
        normal_direction_world=command_normal,
        max_normal_step_m=(
          config.shared_hard_wrist_step_m
          if hard_releasing
          else (
            config.shared_soft_wrist_step_m
            if safety_output is not None
            and safety_output.state is ForceSafetyState.SOFT_RECOVERY
            and safety_output.affected_fingers
            else config.shared_wrist_normal_step_m * bounded_wrist_scale
          )
        ),
        max_tangent_step_m=(
          config.shared_wrist_translation_step_m
          * (0.02 if hard_releasing else bounded_wrist_scale)
        ),
      )
    arm_control = pose_ik.solve(data, issued_wrist_command)
    data.ctrl[handles.arm_actuator_ids] = arm_control

    noisy_forces = np.maximum(
      0.0,
      measured_forces + rng.normal(0.0, config.force_noise_std_n, 4),
    )
    current_finger_q = np.array(data.qpos[handles.hand_qpos_adrs], copy=True)
    nominal_joint_reference = current_finger_q.copy()
    nominal_tip_positions = np.array(data.site_xpos[handles.tip_site_ids], copy=True)
    if reference_source is not None:
      nominal_joint_reference = np.clip(
        nominal_reference_chunk[0],
        handles.hand_joint_ranges_rad[:, 0] + 0.02,
        handles.hand_joint_ranges_rad[:, 1] - 0.02,
      )
      nominal_data.qpos[:] = data.qpos
      nominal_data.qpos[handles.hand_qpos_adrs] = nominal_joint_reference
      mujoco.mj_forward(handles.model, nominal_data)
      nominal_tip_positions = np.array(
        nominal_data.site_xpos[handles.tip_site_ids],
        copy=True,
      )
    finger_authority_available = (
      safety_output is None or safety_output.finger_authority_scale > 0.0
    )
    nominal_finger_command = current_finger_q.copy()
    if finger_authority_available:
      for index, controller in enumerate(finger_controllers):
        if full_reference_authority[index]:
          # MAKE/recontact uses the separate low-force acquisition admittance.
          # It never integrates the full 2 N KEEP target before contact exists.
          controller.reset()
          if reference_motion_scale[index] <= 0.0:
            make_controllers[index].reset()
            actuator_indices = np.array(
              [int(name) for name in FINGERS[index].joint_names],
              dtype=np.int32,
            )
            nominal_finger_command[actuator_indices] = current_finger_q[
              actuator_indices
            ]
            continue
          if reference_source is not None and nominal_free_authority_mask[index]:
            make_controllers[index].reset()
            actuator_indices = np.array(
              [int(name) for name in FINGERS[index].joint_names],
              dtype=np.int32,
            )
            nominal_finger_command[actuator_indices] = nominal_joint_reference[
              actuator_indices
            ]
            continue
          if ContactRole(int(executable_roles[index])) in {
            ContactRole.MAKE,
            ContactRole.KEEP,
          }:
            acquisition = make_controllers[index].step(
              touching_centers[index],
              -normals[index],
              max(
                config.role_make_force_threshold_n,
                config.shared_make_acquisition_force_n
                * float(reference_motion_scale[index]),
              ),
              float(noisy_forces[index]),
            )
            position_command = acquisition.position_command
          else:
            make_controllers[index].reset()
            position_command = touching_centers[index]
        else:
          make_state = make_controllers[index].state
          if abs(make_state.offset_m) > 1e-12 or abs(make_state.velocity_m_s) > 1e-12:
            # Preserve the physical preload across MAKE/recontact -> KEEP.
            # Resetting KEEP MCC at zero created a retreat command exactly at
            # contact confirmation and was the dominant contact-flicker bug.
            controller.reset(
              offset_m=float(
                np.clip(
                  make_state.offset_m,
                  -controller.config.max_offset_m,
                  controller.config.max_offset_m,
                )
              ),
              velocity_m_s=float(
                np.clip(
                  make_state.velocity_m_s,
                  -controller.config.max_velocity_m_s,
                  controller.config.max_velocity_m_s,
                )
              ),
            )
          make_controllers[index].reset()
          full_error = interpreted_desired_force[index] - noisy_forces[index]
          if config.mode == "E05-H-MCC" and active[index]:
            force_error = internal_error[index]
          else:
            force_error = full_error
          planned_position = touching_centers[index]
          if reference_source is not None and nominal_tangent_authority_mask[index]:
            tangential_projector = np.eye(3) - np.outer(normals[index], normals[index])
            tangential_delta = tangential_projector @ (
              nominal_tip_positions[index]
              - data.site_xpos[handles.tip_site_ids[index]]
            )
            tangent_norm = float(np.linalg.norm(tangential_delta))
            if tangent_norm > config.shared_nominal_tangent_step_m:
              tangential_delta *= config.shared_nominal_tangent_step_m / tangent_norm
            planned_position = (
              touching_centers[index]
              + float(role_output.tangential_scale[index])
              * tangential_delta
            )
          command = controller.step_force_error(
            planned_position,
            -normals[index],
            float(force_error),
          )
          position_command = command.position_command
        joint_command = _finger_ik(
          handles,
          data,
          index,
          position_command,
          -normals[index],
        )
        actuator_indices = np.array(
          [int(name) for name in FINGERS[index].joint_names],
          dtype=np.int32,
        )
        nominal_finger_command[actuator_indices] = joint_command
    else:
      for controller in finger_controllers:
        controller.reset()

    if safety_output is not None:
      if safety_output.override_delta_rad is not None:
        issued_finger_command = current_finger_q + safety_output.override_delta_rad
      elif safety_output.finger_authority_scale <= 0.0:
        issued_finger_command = current_finger_q.copy()
      else:
        issued_finger_command = current_finger_q + safety_output.finger_authority_scale * (
          nominal_finger_command - current_finger_q
        )
      issued_finger_command = np.clip(
        issued_finger_command,
        handles.hand_joint_ranges_rad[:, 0] + 0.02,
        handles.hand_joint_ranges_rad[:, 1] - 0.02,
      )
    else:
      issued_finger_command = nominal_finger_command
    if command_continuity is not None:
      maximum_finger_step = np.full(16, config.shared_active_finger_step_rad)
      for index in range(4):
        if full_reference_authority[index]:
          maximum_finger_step[4 * index : 4 * index + 4] = (
            config.shared_make_finger_step_rad
          )
      if (
        safety_output is not None
        and safety_output.state is ForceSafetyState.HARD_RELEASE
      ):
        for index in safety_output.affected_fingers:
          maximum_finger_step[4 * index : 4 * index + 4] = (
            config.shared_hard_finger_step_rad
          )
      elif safety_output is not None and safety_output.finger_authority_scale < 1.0:
        maximum_finger_step *= max(safety_output.finger_authority_scale, 0.02)
      issued_finger_command = command_continuity.limit_finger(
        issued_finger_command,
        maximum_step_rad=maximum_finger_step,
      )
    data.ctrl[handles.hand_actuator_ids] = issued_finger_command
    previous_nominal_command[:] = nominal_joint_reference
    previous_mcc_correction[:] = issued_finger_command - nominal_joint_reference

    latency[step] = perf_counter() - tic
    physics_tic = perf_counter()
    mujoco.mj_step(handles.model, data)
    physics_latency[step] = perf_counter() - physics_tic
    measured_forces, measured_force_vectors, measured_positions, non_tip_count = _contact_state(
      handles,
      data,
    )
    # Measured-force hysteresis prevents a 0.20 N threshold crossing from
    # creating hundreds of fictitious MAKE/BREAK events on one continuous pad.
    contact_active = np.where(
      contact_active,
      measured_forces >= 0.5 * config.contact_threshold_n,
      measured_forces >= config.contact_threshold_n,
    )
    contact_mask = contact_active.copy()
    measured_normals = normals.copy()
    physical_wrench = _contact_wrench(
      measured_force_vectors,
      measured_positions,
      measured_forces > 0.0,
      data.site_xpos[handles.palm_site_id],
    )

    current_arm_control = np.array(data.ctrl[handles.arm_actuator_ids], copy=True)
    current_finger_control = np.array(data.ctrl[handles.hand_actuator_ids], copy=True)
    arm_qd_command = (current_arm_control - previous_arm_control) / config.dt_s
    finger_qd_command = (current_finger_control - previous_finger_control) / config.dt_s
    previous_arm_control[:] = current_arm_control
    previous_finger_control[:] = current_finger_control
    arm_force_range = handles.model.actuator_forcerange[handles.arm_actuator_ids]
    hand_force_range = handles.model.actuator_forcerange[handles.hand_actuator_ids]
    arm_limited = handles.model.actuator_forcelimited[handles.arm_actuator_ids].astype(bool)
    hand_limited = handles.model.actuator_forcelimited[handles.hand_actuator_ids].astype(bool)
    arm_saturated = arm_limited & (
      np.isclose(data.actuator_force[handles.arm_actuator_ids], arm_force_range[:, 0], atol=1e-3)
      | np.isclose(data.actuator_force[handles.arm_actuator_ids], arm_force_range[:, 1], atol=1e-3)
    )
    hand_saturated = hand_limited & (
      np.isclose(data.actuator_force[handles.hand_actuator_ids], hand_force_range[:, 0], atol=1e-3)
      | np.isclose(data.actuator_force[handles.hand_actuator_ids], hand_force_range[:, 1], atol=1e-3)
    )
    wrist_offset = wrist_mcc.state.offset if config.mode == "E05-H-MCC" else np.zeros(6)
    finger_offsets = np.array([controller.state.offset_m for controller in finger_controllers])
    decision = guards.evaluate(
      FullRobotGuardObservation(
        arm_q_rad=data.qpos[handles.arm_qpos_adrs],
        arm_qd_command_rad_s=arm_qd_command,
        arm_qd_actual_rad_s=data.qvel[handles.arm_dof_adrs],
        finger_q_rad=data.qpos[handles.hand_qpos_adrs],
        finger_qd_command_rad_s=finger_qd_command,
        finger_qd_actual_rad_s=data.qvel[handles.hand_dof_adrs],
        fingertip_forces_n=measured_forces,
        wrist_wrench=filtered_wrench,
        arm_external_torque_nm=estimate.joint_external_torque_nm,
        wrist_compliance_offset=wrist_offset,
        finger_compliance_offsets_m=finger_offsets,
        arm_actuator_saturated=arm_saturated,
        finger_actuator_saturated=hand_saturated,
        sensor_validity={"joint_state": True, "tip_force": True, "wrist_wrench": estimate.jacobian_rank == 6},
      )
    )

    current_palm_quaternion = _quaternion_from_matrix(
      data.site_xmat[handles.palm_site_id]
    )
    arm_q[step] = data.qpos[handles.arm_qpos_adrs]
    arm_dq[step] = data.qvel[handles.arm_dof_adrs]
    arm_cmd[step] = current_arm_control
    finger_q[step] = data.qpos[handles.hand_qpos_adrs]
    finger_dq[step] = data.qvel[handles.hand_dof_adrs]
    finger_cmd[step] = current_finger_control
    palm[step] = np.concatenate((data.site_xpos[handles.palm_site_id], current_palm_quaternion))
    planned_palm[step] = planned_pose
    commanded_palm[step] = issued_wrist_command
    tips[step] = data.site_xpos[handles.tip_site_ids]
    pad_normals[step] = np.stack(
      [data.site_xmat[int(site)].reshape(3, 3)[:, 2] for site in handles.tip_site_ids]
    )
    contact_positions[step] = measured_positions
    contact_normals[step] = measured_normals
    forces_log[step] = measured_forces
    actual_contacts[step] = contact_mask
    desired_wrench_log[step] = desired_wrench
    estimated_wrench_log[step] = filtered_wrench
    contact_wrench_log[step] = physical_wrench
    torque_log[step] = estimate.joint_external_torque_nm
    wrist_offset_log[step] = wrist_offset
    finger_offset_log[step] = finger_offsets
    curvature_log[step] = curvatures
    disturbance[step] = planning_timestamp_s >= config.pose_step_time_s
    if safety_output is None:
      guard_reason[step] = decision.reason.value
    else:
      guard_reason[step] = f"{safety_output.state.value}:{safety_output.reason}"
    non_tip[step] = non_tip_count
    role_log[step] = executable_roles
    requested_role_log[step] = requested_roles
    loop_latency[step] = perf_counter() - tic

  trace = E05MCCTrace(
    time_s=time_s,
    arm_q_rad=arm_q,
    arm_dq_rad_s=arm_dq,
    arm_command_rad=arm_cmd,
    finger_q_rad=finger_q,
    finger_dq_rad_s=finger_dq,
    finger_command_rad=finger_cmd,
    palm_pose_world=palm,
    planned_palm_pose_world=planned_palm,
    commanded_palm_pose_world=commanded_palm,
    fingertip_positions_world_m=tips,
    pad_normals_world=pad_normals,
    contact_positions_world_m=contact_positions,
    contact_normals_world=contact_normals,
    fingertip_forces_n=forces_log,
    actual_contacts=actual_contacts,
    desired_hand_wrench_world=desired_wrench_log,
    estimated_hand_wrench_world=estimated_wrench_log,
    contact_hand_wrench_world=contact_wrench_log,
    arm_external_torque_nm=torque_log,
    wrist_compliance_offset=wrist_offset_log,
    finger_compliance_offsets_m=finger_offset_log,
    coordinator_rank=coordinator_rank,
    coordinator_condition=coordinator_condition,
    coordinator_internal_leakage_n=coordinator_leakage,
    surface_curvature_inv_m=curvature_log,
    disturbance_active=disturbance,
    controller_latency_s=latency,
    physics_step_latency_s=physics_latency,
    loop_latency_s=loop_latency,
    guard_reason=guard_reason,
    non_tip_contact_count=non_tip,
  )
  metrics = trace_metrics(trace, config, handles)
  role_names = tuple(role.name for role in ContactRole)
  metrics["role_frame_counts"] = {
    role_names[role]: int(np.count_nonzero(role_log == role))
    for role in range(len(role_names))
  }
  metrics["confirmed_keep_probability"] = float(
    np.mean(role_log == int(ContactRole.KEEP))
  )
  metrics["reference_source"] = (
    (
      "PASSIVE_SHARED_DEFAULT"
      if config.enforce_shared_force_safety
      else "PLAIN_WHOLE_HAND_MCC"
    )
    if reference_source is None
    else str(getattr(reference_source, "name", type(reference_source).__name__))
  )
  metrics["requested_role_frame_counts"] = {
    role_names[role]: int(np.count_nonzero(requested_role_log == role))
    for role in range(len(role_names))
  }
  metrics["role_geometry_veto_count"] = int(role_veto_count)
  metrics["reference_inference_count"] = int(reference_inference_count)
  metrics["reference_inference_latency_mean_s"] = (
    float(np.mean(reference_inference_latencies))
    if reference_inference_latencies
    else 0.0
  )
  metrics["reference_inference_latency_p95_s"] = (
    float(np.percentile(reference_inference_latencies, 95.0))
    if reference_inference_latencies
    else 0.0
  )
  return trace, metrics


def _recovery_latency(
  time_s: NDArray[np.float64],
  condition: NDArray[np.bool_],
  event_time_s: float,
) -> float:
  start = int(np.searchsorted(time_s, event_time_s))
  indices = np.flatnonzero(condition[start:])
  if not len(indices):
    return float(time_s[-1] - event_time_s)
  return float(time_s[start + indices[0]] - event_time_s)


def _maximum_true_duration(signal: NDArray[np.bool_], dt_s: float) -> float:
  padded = np.pad(np.asarray(signal, dtype=np.int8), (1, 1))
  edges = np.diff(padded)
  starts = np.flatnonzero(edges == 1)
  stops = np.flatnonzero(edges == -1)
  if not len(starts):
    return 0.0
  return float(np.max(stops - starts) * dt_s)


def trace_metrics(trace: E05MCCTrace, config: E05MCCConfig, handles: Any) -> dict[str, Any]:
  mask = trace.time_s >= config.settling_time_s
  contacts = trace.actual_contacts[mask]
  forces = trace.fingertip_forces_n[mask]
  any_contact = np.any(contacts, axis=1)
  contact_count = np.sum(contacts, axis=1)
  force_error = forces - config.desired_force_n
  losses = 0
  for index in range(4):
    signal = contacts[:, index]
    losses += int(np.sum(signal[:-1] & ~signal[1:]))
  arm_lower = trace.arm_q_rad[mask] - handles.arm_joint_ranges_rad[:, 0]
  arm_upper = handles.arm_joint_ranges_rad[:, 1] - trace.arm_q_rad[mask]
  finger_lower = trace.finger_q_rad[mask] - handles.hand_joint_ranges_rad[:, 0]
  finger_upper = handles.hand_joint_ranges_rad[:, 1] - trace.finger_q_rad[mask]
  palm_position_error = trace.palm_pose_world[mask, :3] - trace.commanded_palm_pose_world[mask, :3]
  actual_delta = np.diff(trace.palm_pose_world[mask, :3], axis=0)
  planned_delta = np.diff(trace.planned_palm_pose_world[mask, :3], axis=0)
  selected_wrench_error = trace.desired_hand_wrench_world[mask] - trace.estimated_hand_wrench_world[mask]
  finite_condition = trace.coordinator_condition[mask]
  finite_condition = finite_condition[np.isfinite(finite_condition)]
  disturbance_indices = np.flatnonzero(trace.disturbance_active)
  disturbance_time_s = (
    float(trace.time_s[disturbance_indices[0]])
    if len(disturbance_indices)
    else float(trace.time_s[-1])
  )
  step_window = (trace.time_s >= disturbance_time_s) & (
    trace.time_s < disturbance_time_s + 1.0
  )
  after_step = trace.time_s >= disturbance_time_s
  force_settled = np.all(
    np.abs(trace.fingertip_forces_n - config.desired_force_n) <= 0.5,
    axis=1,
  ) & np.all(trace.actual_contacts, axis=1)
  force_settling_latency = _recovery_latency(
    trace.time_s,
    force_settled,
    disturbance_time_s,
  )
  guard_values, guard_counts = np.unique(trace.guard_reason[mask], return_counts=True)
  guard_histogram = {
    str(name): int(count) for name, count in zip(guard_values, guard_counts)
  }
  positive_y_step = np.maximum(actual_delta[:, 1], 0.0)
  supported_two_contacts = (contact_count[:-1] >= 2) & (contact_count[1:] >= 2)
  supported_one_contact = any_contact[:-1] & any_contact[1:]
  above_force_reference = forces > config.force_limit_n
  any_above_force_reference = np.any(above_force_reference, axis=1)
  multi_pad_above_force_reference = np.sum(above_force_reference, axis=1) >= 2
  return {
    "cell": config.mode,
    "controller": "MCC",
    "dp_evaluated": False,
    "surface": config.surface,
    "seed": config.seed,
    "duration_s": config.duration_s,
    "contact_continuity_probability": float(np.mean(any_contact)),
    "average_contact_count": float(np.mean(contact_count)),
    "contact_count_ge2_probability": float(np.mean(contact_count >= 2)),
    "contact_count_ge3_probability": float(np.mean(contact_count >= 3)),
    "four_contact_probability": float(np.mean(contact_count == 4)),
    "minimum_contact_count": int(np.min(contact_count)),
    "per_finger_contact_probability": np.mean(contacts, axis=0).tolist(),
    "zero_contact_time_s": float(np.sum(~any_contact) * config.dt_s),
    "contact_loss_events": losses,
    "force_rmse_n": float(np.sqrt(np.mean(force_error**2))),
    "force_mae_n": float(np.mean(np.abs(force_error))),
    "force_p95_n": float(np.percentile(forces, 95.0)),
    "max_tip_force_n": float(np.max(forces)),
    "force_violation_probability": float(np.mean(forces > config.force_limit_n)),
    "force_violation_time_s": float(np.sum(any_above_force_reference) * config.dt_s),
    "force_violation_max_consecutive_time_s": _maximum_true_duration(
      any_above_force_reference,
      config.dt_s,
    ),
    "multi_pad_force_violation_probability": float(
      np.mean(multi_pad_above_force_reference)
    ),
    "multi_pad_force_violation_time_s": float(
      np.sum(multi_pad_above_force_reference) * config.dt_s
    ),
    "force_excess_impulse_n_s": float(
      np.sum(np.maximum(forces - config.force_limit_n, 0.0)) * config.dt_s
    ),
    "force_above_20n_time_s": float(
      np.sum(np.any(forces > 20.0, axis=1)) * config.dt_s
    ),
    "pose_step_peak_force_n": float(np.max(trace.fingertip_forces_n[step_window])),
    "any_contact_recovery_s": _recovery_latency(
      trace.time_s,
      np.any(trace.actual_contacts, axis=1),
      disturbance_time_s,
    ),
    "four_contact_recovery_s": _recovery_latency(
      trace.time_s,
      np.all(trace.actual_contacts, axis=1),
      disturbance_time_s,
    ),
    "force_settling_s": force_settling_latency,
    "actual_palm_path_length_m": float(np.sum(np.linalg.norm(actual_delta, axis=1))),
    "planned_palm_path_length_m": float(np.sum(np.linalg.norm(planned_delta, axis=1))),
    "traversal_y_m": float(trace.palm_pose_world[mask][-1, 1] - trace.palm_pose_world[mask][0, 1]),
    "positive_y_traversal_m": float(np.sum(positive_y_step)),
    "supported_y_traversal_ge1_m": float(
      np.sum(positive_y_step[supported_one_contact])
    ),
    "supported_y_traversal_ge2_m": float(
      np.sum(positive_y_step[supported_two_contacts])
    ),
    "palm_position_tracking_rmse_m": float(np.sqrt(np.mean(palm_position_error**2))),
    "wrist_wrench_rmse_6d": float(np.sqrt(np.mean(selected_wrench_error**2))),
    "wrist_force_z_rmse_n": float(np.sqrt(np.mean(selected_wrench_error[:, 2] ** 2))),
    "max_wrist_compliance_translation_m": float(np.max(np.linalg.norm(trace.wrist_compliance_offset[mask, :3], axis=1))),
    "max_abs_arm_external_torque_nm": float(np.max(np.abs(trace.arm_external_torque_nm[mask]))),
    "minimum_arm_joint_margin_rad": float(np.min(np.minimum(arm_lower, arm_upper))),
    "minimum_finger_joint_margin_rad": float(np.min(np.minimum(finger_lower, finger_upper))),
    "coordinator_rank_min": int(np.min(trace.coordinator_rank[after_step])),
    "coordinator_condition_p95": float(np.percentile(finite_condition, 95.0)) if len(finite_condition) else None,
    "coordinator_internal_leakage_p95_n": float(np.percentile(trace.coordinator_internal_leakage_n[mask], 95.0)),
    "controller_latency_mean_s": float(np.mean(trace.controller_latency_s)),
    "controller_latency_p95_s": float(np.percentile(trace.controller_latency_s, 95.0)),
    "deadline_miss_probability": float(np.mean(trace.controller_latency_s > config.dt_s)),
    "physics_step_latency_p95_s": float(np.percentile(trace.physics_step_latency_s, 95.0)),
    "loop_latency_p95_s": float(np.percentile(trace.loop_latency_s, 95.0)),
    "measured_real_time_factor": float(
      config.dt_s / max(float(np.mean(trace.loop_latency_s)), 1e-12)
    ),
    "non_tip_contact_count": int(np.sum(trace.non_tip_contact_count[mask])),
    "guard_histogram": guard_histogram,
    "shared_force_safety_enabled": config.enforce_shared_force_safety,
    "hard_guard_frames": int(
      np.count_nonzero(np.char.startswith(trace.guard_reason[mask], ForceSafetyState.HARD_RELEASE.value))
    ),
    "soft_recovery_frames": int(
      np.count_nonzero(np.char.startswith(trace.guard_reason[mask], ForceSafetyState.SOFT_RECOVERY.value))
    ),
    "safety_aborted_frames": int(
      np.count_nonzero(np.char.startswith(trace.guard_reason[mask], ForceSafetyState.ABORTED.value))
    ),
  }
