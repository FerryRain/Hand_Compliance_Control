"""Physics runner for the MCC-only E05-F and E05-H cells."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import mujoco
import numpy as np
from numpy.typing import NDArray

from Module.e05_physics.extreme_surface import query_surface as query_extreme_surface
from Module.e05_physics.scene import FINGERS, PAD_HALF_SIZE_M, Q_NOMINAL
from Module.fr3_leap import ARM_HOME_Q, FullRobotModelConfig, build_full_robot
from Module.module_2_fingertip_mcc import FingertipMCC, MCCConfig
from Module.module_3_runtime_guards import (
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
    }
    for name, value in positive.items():
      if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    if self.settling_time_s < 0.0 or self.settling_time_s >= self.duration_s:
      raise ValueError("settling_time_s must be in [0,duration_s)")
    if self.pose_step_time_s <= self.settling_time_s or self.pose_step_time_s >= self.duration_s:
      raise ValueError("pose_step_time_s must be inside the evaluation interval")
    if not 0.0 < self.force_filter_alpha <= 1.0:
      raise ValueError("force_filter_alpha must be in (0,1]")
    for name in (
      "lateral_primary_amplitude_m",
      "lateral_secondary_amplitude_m",
      "pose_step_m",
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
  command = current_q + gain * delta + 0.01 * (Q_NOMINAL[nominal_indices] - current_q)
  # A resolved-rate target can jump when the hfield normal changes rapidly.
  # Rate-limit the commanded joint displacement before actuator/joint clipping
  # so a narrow ridge cannot become a one-frame impact command.
  command = np.clip(command, current_q - 0.012, current_q + 0.012)
  lower = handles.hand_joint_ranges_rad[nominal_indices, 0] + 0.05
  upper = handles.hand_joint_ranges_rad[nominal_indices, 1] - 0.05
  return np.clip(command, lower, upper)


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
  finger_q = Q_NOMINAL + rng.normal(0.0, config.initial_joint_noise_std_rad, 16)
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


def run_e05_mcc(config: E05MCCConfig = E05MCCConfig()) -> tuple[E05MCCTrace, dict[str, Any]]:
  handles = build_full_robot(
    FullRobotModelConfig(
      surface=config.surface,
      timestep_s=config.dt_s,
      gravity_m_s2=0.0,
      arm_kp=1800.0,
      arm_damping_ratio=0.9,
    )
  )
  handles.model.geom_friction[handles.object_geom_id, 0] = config.friction_coefficient
  handles.model.geom_friction[handles.tip_geom_ids, 0] = config.friction_coefficient
  data = _initialize_data(handles, config)
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
      max_abs_velocity=(0.04, 0.04, 0.04, 0.3, 0.3, 0.3),
      max_abs_acceleration=(0.8, 0.8, 0.8, 5.0, 5.0, 5.0),
    )
  )
  finger_controllers = tuple(
    FingertipMCC(
      MCCConfig(
        virtual_mass=0.08,
        damping=14.0,
        stiffness=25.0,
        dt_s=config.dt_s,
        max_offset_m=0.015,
        max_velocity_m_s=0.08,
        max_acceleration_m_s2=30.0,
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

  for step, timestamp_s in enumerate(time_s):
    tic = perf_counter()
    planned_pose = _planned_palm_pose(initial_pose, config, float(timestamp_s))
    active = contact_active.copy()
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

    estimate = estimator.estimate(data)
    filtered_wrench = (
      (1.0 - config.force_filter_alpha) * filtered_wrench
      + config.force_filter_alpha * estimate.wrench_world
    )
    internal_error = np.zeros(4)
    active_positions = measured_positions.copy()
    active_positions[~active] = surface_points[~active]
    if np.any(active):
      coordinated = coordinator.step(
        active_positions,
        normals,
        np.full(4, config.desired_force_n),
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

    if config.mode == "E05-H-MCC" and step % config.wrist_update_period_steps == 0:
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
    elif config.mode == "E05-F-MCC":
      current_wrist_command = planned_pose.copy()
    arm_control = pose_ik.solve(data, current_wrist_command)
    data.ctrl[handles.arm_actuator_ids] = arm_control

    noisy_forces = np.maximum(
      0.0,
      measured_forces + rng.normal(0.0, config.force_noise_std_n, 4),
    )
    for index, controller in enumerate(finger_controllers):
      full_error = config.desired_force_n - noisy_forces[index]
      if config.mode == "E05-H-MCC" and active[index]:
        force_error = internal_error[index]
      else:
        # Initial MAKE/recovery remains a local finger responsibility until
        # actual force confirms the contact and admits it to H_A.
        force_error = full_error
      command = controller.step_force_error(
        touching_centers[index],
        -normals[index],
        float(force_error),
      )
      joint_command = _finger_ik(
        handles,
        data,
        index,
        command.position_command,
        -normals[index],
      )
      actuator_indices = np.array(
        [int(name) for name in FINGERS[index].joint_names],
        dtype=np.int32,
      )
      data.ctrl[handles.hand_actuator_ids[actuator_indices]] = joint_command

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
    commanded_palm[step] = current_wrist_command
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
    disturbance[step] = timestamp_s >= config.pose_step_time_s
    guard_reason[step] = decision.reason.value
    non_tip[step] = non_tip_count
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
  return trace, trace_metrics(trace, config, handles)


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
  step_window = (trace.time_s >= config.pose_step_time_s) & (
    trace.time_s < config.pose_step_time_s + 1.0
  )
  after_step = trace.time_s >= config.pose_step_time_s
  force_settled = np.all(
    np.abs(trace.fingertip_forces_n - config.desired_force_n) <= 0.5,
    axis=1,
  ) & np.all(trace.actual_contacts, axis=1)
  force_settling_latency = _recovery_latency(
    trace.time_s,
    force_settled,
    config.pose_step_time_s,
  )
  guard_values, guard_counts = np.unique(trace.guard_reason[mask], return_counts=True)
  guard_histogram = {
    str(name): int(count) for name, count in zip(guard_values, guard_counts)
  }
  return {
    "cell": config.mode,
    "controller": "MCC",
    "dp_evaluated": False,
    "surface": config.surface,
    "seed": config.seed,
    "duration_s": config.duration_s,
    "contact_continuity_probability": float(np.mean(any_contact)),
    "average_contact_count": float(np.mean(contact_count)),
    "minimum_contact_count": int(np.min(contact_count)),
    "per_finger_contact_probability": np.mean(contacts, axis=0).tolist(),
    "zero_contact_time_s": float(np.sum(~any_contact) * config.dt_s),
    "contact_loss_events": losses,
    "force_rmse_n": float(np.sqrt(np.mean(force_error**2))),
    "force_mae_n": float(np.mean(np.abs(force_error))),
    "force_p95_n": float(np.percentile(forces, 95.0)),
    "max_tip_force_n": float(np.max(forces)),
    "force_violation_probability": float(np.mean(forces > config.force_limit_n)),
    "force_violation_time_s": float(np.sum(np.any(forces > config.force_limit_n, axis=1)) * config.dt_s),
    "pose_step_peak_force_n": float(np.max(trace.fingertip_forces_n[step_window])),
    "any_contact_recovery_s": _recovery_latency(
      trace.time_s,
      np.any(trace.actual_contacts, axis=1),
      config.pose_step_time_s,
    ),
    "four_contact_recovery_s": _recovery_latency(
      trace.time_s,
      np.all(trace.actual_contacts, axis=1),
      config.pose_step_time_s,
    ),
    "force_settling_s": force_settling_latency,
    "actual_palm_path_length_m": float(np.sum(np.linalg.norm(actual_delta, axis=1))),
    "planned_palm_path_length_m": float(np.sum(np.linalg.norm(planned_delta, axis=1))),
    "traversal_y_m": float(trace.palm_pose_world[mask][-1, 1] - trace.palm_pose_world[mask][0, 1]),
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
  }
