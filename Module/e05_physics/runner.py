"""MuJoCo runner connecting fingertip MCC commands to Leap Hand Jacobian IK."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np
from numpy.typing import NDArray

from Module.e05_physics.extreme_surface import query_surface as query_extreme_surface
from Module.e05_physics.scene import (
  FINGERS,
  PAD_HALF_SIZE_M,
  Q_NOMINAL,
  SceneHandles,
  build_scene,
)
from Module.module_2_fingertip_mcc import FingertipMCC, MCCConfig


@dataclass(frozen=True, slots=True)
class PhysicsConfig:
  scenario: str = "maintenance_translation"
  duration_s: float = 3.0
  settling_time_s: float = 0.75
  dt_s: float = 0.002
  desired_force_n: float = 2.0
  contact_threshold_n: float = 0.20
  force_limit_n: float = 8.0
  break_clearance_m: float = 0.012
  ik_damping: float = 0.01
  ik_gain: float = 0.20
  orientation_weight_m_per_rad: float = 0.012
  friction_coefficient: float = 0.90
  initial_joint_noise_std_rad: float = 0.0
  force_noise_std_n: float = 0.0
  surface_bias_m: float = 0.0
  wrist_error_amplitude_m: float = 0.0
  motion_scale: float = 1.0
  pose_step_time_s: float = 9.0
  pose_step_m: float = 0.004
  seed: int = 7

  def __post_init__(self) -> None:
    valid_scenarios = {
      "maintenance_translation",
      "maintenance_rotation",
      "maintenance_curved",
      "extreme_surface",
      "handover",
    }
    if self.scenario not in valid_scenarios:
      raise ValueError(f"unknown scenario: {self.scenario}")
    positive = {
      "duration_s": self.duration_s,
      "dt_s": self.dt_s,
      "desired_force_n": self.desired_force_n,
      "contact_threshold_n": self.contact_threshold_n,
      "force_limit_n": self.force_limit_n,
      "break_clearance_m": self.break_clearance_m,
      "ik_damping": self.ik_damping,
      "ik_gain": self.ik_gain,
      "orientation_weight_m_per_rad": self.orientation_weight_m_per_rad,
      "friction_coefficient": self.friction_coefficient,
      "motion_scale": self.motion_scale,
    }
    for name, value in positive.items():
      if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    non_negative = {
      "settling_time_s": self.settling_time_s,
      "initial_joint_noise_std_rad": self.initial_joint_noise_std_rad,
      "force_noise_std_n": self.force_noise_std_n,
      "wrist_error_amplitude_m": self.wrist_error_amplitude_m,
      "pose_step_time_s": self.pose_step_time_s,
      "pose_step_m": self.pose_step_m,
    }
    for name, value in non_negative.items():
      if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    if not np.isfinite(self.surface_bias_m):
      raise ValueError("surface_bias_m must be finite")
    if self.settling_time_s >= self.duration_s:
      raise ValueError("settling_time_s must be shorter than duration_s")
    if self.scenario == "extreme_surface" and self.pose_step_time_s >= self.duration_s:
      raise ValueError("pose_step_time_s must occur before the extreme-surface episode ends")


@dataclass(frozen=True, slots=True)
class PhysicsTrace:
  time_s: NDArray[np.float64]
  desired_contacts: NDArray[np.bool_]
  actual_contacts: NDArray[np.bool_]
  fingertip_forces_n: NDArray[np.float64]
  controller_forces_n: NDArray[np.float64]
  fingertip_positions_m: NDArray[np.float64]
  pad_normals_world: NDArray[np.float64]
  contact_positions_m: NDArray[np.float64]
  contact_head_clearances_m: NDArray[np.float64]
  joint_positions_rad: NDArray[np.float64]
  joint_commands_rad: NDArray[np.float64]
  object_positions_m: NDArray[np.float64]
  object_quaternions: NDArray[np.float64]
  surface_curvatures_inv_m: NDArray[np.float64]
  disturbance_active: NDArray[np.bool_]
  non_tip_contact_count: NDArray[np.int32]


def _quaternion_from_axis_angle(axis: NDArray[np.float64], angle: float) -> NDArray[np.float64]:
  unit_axis = np.asarray(axis, dtype=np.float64)
  unit_axis /= np.linalg.norm(unit_axis)
  half = 0.5 * angle
  return np.array([np.cos(half), *(np.sin(half) * unit_axis)], dtype=np.float64)


def _smoothstep(value: float) -> float:
  clipped = float(np.clip(value, 0.0, 1.0))
  return clipped * clipped * (3.0 - 2.0 * clipped)


def _object_pose(
  handles: SceneHandles,
  config: PhysicsConfig,
  timestamp_s: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
  position = handles.object_spec.initial_position.copy()
  quaternion = handles.object_spec.initial_quaternion.copy()
  motion_time = max(0.0, timestamp_s - config.settling_time_s)
  motion_duration = max(config.duration_s - config.settling_time_s, config.dt_s)
  progress = _smoothstep(motion_time / motion_duration)
  if config.scenario == "maintenance_translation":
    position[0] -= config.motion_scale * 0.04 * progress
  elif config.scenario == "maintenance_rotation":
    angle = config.motion_scale * np.deg2rad(5.0) * np.sin(np.pi * progress)
    quaternion = _quaternion_from_axis_angle(np.array([0.0, 1.0, 0.0]), float(angle))
  elif config.scenario == "maintenance_curved":
    position[0] -= config.motion_scale * 0.025 * progress
  elif config.scenario == "extreme_surface":
    # A long S-shaped scan forces all four physical pads through different
    # two-dimensional surface regions instead of replaying one short profile.
    position[0] += config.motion_scale * (
      0.045 * np.sin(2.0 * np.pi * progress)
      + 0.018 * np.sin(6.0 * np.pi * progress)
    )
    position[1] -= config.motion_scale * 0.480 * progress
    if timestamp_s >= config.pose_step_time_s:
      position[2] -= config.pose_step_m
  elif config.scenario == "handover":
    position[0] -= config.motion_scale * 0.015 * progress
  else:
    raise ValueError(f"unknown scenario: {config.scenario}")
  if motion_time > 0.0 and config.wrist_error_amplitude_m > 0.0:
    phase = 0.37 * float(config.seed % 17)
    position[2] += config.wrist_error_amplitude_m * np.sin(
      2.0 * np.pi * 1.1 * motion_time + phase
    )
  return position, quaternion


def _desired_contact_mask(config: PhysicsConfig, timestamp_s: float) -> NDArray[np.bool_]:
  if config.scenario != "handover":
    return np.ones(4, dtype=np.bool_)
  if timestamp_s < config.settling_time_s + 1.0:
    return np.array([True, True, True, False])
  if timestamp_s < config.settling_time_s + 1.25:
    return np.array([True, True, False, False])
  return np.array([True, True, False, True])


def _rotation_matrix(quaternion: NDArray[np.float64]) -> NDArray[np.float64]:
  matrix = np.zeros(9, dtype=np.float64)
  mujoco.mju_quat2Mat(matrix, quaternion)
  return matrix.reshape(3, 3)


def _surface_reference(
  handles: SceneHandles,
  object_position: NDArray[np.float64],
  object_quaternion: NDArray[np.float64],
  fingertip_position: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
  if handles.object_spec.shape == "plane":
    rotation = _rotation_matrix(object_quaternion)
    normal = rotation[:, 2]
    top_center = object_position + handles.object_spec.size[2] * normal
    distance = float(np.dot(fingertip_position - top_center, normal))
    surface_point = fingertip_position - distance * normal
    curvature = 0.0
  elif handles.object_spec.shape == "sphere":
    radial = fingertip_position - object_position
    radial_norm = float(np.linalg.norm(radial))
    normal = radial / radial_norm if radial_norm > 1e-12 else np.array([0.0, 0.0, 1.0])
    surface_point = object_position + handles.object_spec.size[0] * normal
    curvature = 1.0 / float(handles.object_spec.size[0])
  else:
    rotation = _rotation_matrix(object_quaternion)
    local_tip = rotation.T @ (fingertip_position - object_position)
    height, local_normal, curvature = query_extreme_surface(
      float(local_tip[0]),
      float(local_tip[1]),
    )
    local_surface = np.array([local_tip[0], local_tip[1], height], dtype=np.float64)
    surface_point = object_position + rotation @ local_surface
    normal = rotation @ local_normal
  return surface_point, normal, curvature


def _pad_support_radius(
  handles: SceneHandles,
  data: mujoco.MjData,
  finger_index: int,
  direction: NDArray[np.float64],
) -> float:
  geom_id = int(handles.tip_geom_ids[finger_index])
  geom_rotation = data.geom_xmat[geom_id].reshape(3, 3)
  local_direction = geom_rotation.T @ direction
  return float(np.linalg.norm(PAD_HALF_SIZE_M * local_direction))


def _contact_forces(
  handles: SceneHandles,
  data: mujoco.MjData,
) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
  forces = np.zeros(4, dtype=np.float64)
  weighted_positions = np.zeros((4, 3), dtype=np.float64)
  tip_lookup = {
    int(geom_id): index
    for index, geom_id in enumerate(handles.tip_geom_ids)
  }
  contact_force = np.zeros(6, dtype=np.float64)
  non_tip_contacts = 0
  for contact_index in range(data.ncon):
    contact = data.contact[contact_index]
    geom_1 = int(contact.geom1)
    geom_2 = int(contact.geom2)
    if handles.object_geom_id not in (geom_1, geom_2):
      continue
    other = geom_2 if geom_1 == handles.object_geom_id else geom_1
    finger_index = tip_lookup.get(other)
    if finger_index is None:
      non_tip_contacts += 1
      continue
    mujoco.mj_contactForce(handles.model, data, contact_index, contact_force)
    normal_force = abs(float(contact_force[0]))
    forces[finger_index] += normal_force
    weighted_positions[finger_index] += normal_force * np.asarray(contact.pos)
  positions = np.full((4, 3), np.nan, dtype=np.float64)
  valid = forces > 0.0
  positions[valid] = weighted_positions[valid] / forces[valid, None]
  return forces, positions, non_tip_contacts


def _initialize_data(
  handles: SceneHandles,
  config: PhysicsConfig,
  rng: np.random.Generator,
) -> mujoco.MjData:
  data = mujoco.MjData(handles.model)
  initial_q = Q_NOMINAL + rng.normal(
    0.0,
    config.initial_joint_noise_std_rad,
    size=Q_NOMINAL.shape,
  )
  initial_q = np.clip(
    initial_q,
    handles.joint_ranges_rad[:, 0] + 0.02,
    handles.joint_ranges_rad[:, 1] - 0.02,
  )
  data.qpos[handles.joint_qpos_adrs] = initial_q
  data.ctrl[:] = initial_q
  data.mocap_pos[handles.object_mocap_id] = handles.object_spec.initial_position
  data.mocap_quat[handles.object_mocap_id] = handles.object_spec.initial_quaternion
  mujoco.mj_forward(handles.model, data)
  return data


def _ik_joint_command(
  handles: SceneHandles,
  data: mujoco.MjData,
  finger_index: int,
  target_position: NDArray[np.float64],
  target_pad_normal: NDArray[np.float64],
  config: PhysicsConfig,
) -> NDArray[np.float64]:
  jacobian_position = np.zeros((3, handles.model.nv), dtype=np.float64)
  jacobian_rotation = np.zeros((3, handles.model.nv), dtype=np.float64)
  mujoco.mj_jacSite(
    handles.model,
    data,
    jacobian_position,
    jacobian_rotation,
    int(handles.tip_site_ids[finger_index]),
  )
  dof_adrs = handles.finger_dof_adrs[finger_index]
  qpos_adrs = handles.finger_qpos_adrs[finger_index]
  position_jacobian = jacobian_position[:, dof_adrs]
  rotation_jacobian = jacobian_rotation[:, dof_adrs]
  current_position = data.site_xpos[int(handles.tip_site_ids[finger_index])]
  current_pad_normal = data.site_xmat[
    int(handles.tip_site_ids[finger_index])
  ].reshape(3, 3)[:, 2]
  normal_cross = np.cross(current_pad_normal, target_pad_normal)
  cross_norm = float(np.linalg.norm(normal_cross))
  if cross_norm > 1e-9:
    orientation_axis = normal_cross / cross_norm
    orientation_angle = float(
      np.arctan2(
        cross_norm,
        np.clip(np.dot(current_pad_normal, target_pad_normal), -1.0, 1.0),
      )
    )
  else:
    orientation_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    orientation_angle = 0.0
  weight = config.orientation_weight_m_per_rad
  orientation_jacobian = orientation_axis @ rotation_jacobian
  jacobian = np.vstack((position_jacobian, weight * orientation_jacobian))
  error = np.concatenate(
    (
      target_position - current_position,
      np.array([weight * orientation_angle], dtype=np.float64),
    )
  )
  regularized = (
    jacobian @ jacobian.T
    + config.ik_damping**2 * np.eye(jacobian.shape[0])
  )
  delta_q = jacobian.T @ np.linalg.solve(regularized, error)
  current_q = data.qpos[qpos_adrs]
  nominal_q = Q_NOMINAL[np.array([int(name) for name in FINGERS[finger_index].joint_names])]
  posture_bias = 0.015 * (nominal_q - current_q)
  command = current_q + config.ik_gain * delta_q + posture_bias
  return command


def _metrics(
  trace: PhysicsTrace,
  config: PhysicsConfig,
  handles: SceneHandles,
) -> dict[str, Any]:
  evaluation_mask = trace.time_s >= config.settling_time_s
  contacts = trace.actual_contacts[evaluation_mask]
  desired = trace.desired_contacts[evaluation_mask]
  forces = trace.fingertip_forces_n[evaluation_mask]
  active_forces = forces[desired]
  any_contact = np.any(contacts, axis=1)
  contact_loss_events = 0
  unexpected_contact_loss_events = 0
  for finger_index in range(4):
    finger_contact = contacts[:, finger_index]
    contact_loss_events += int(np.sum(finger_contact[:-1] & ~finger_contact[1:]))
    unexpected_contact_loss_events += int(
      np.sum(
        finger_contact[:-1]
        & ~finger_contact[1:]
        & desired[:-1, finger_index]
        & desired[1:, finger_index]
      )
    )
  joint_positions = trace.joint_positions_rad[evaluation_mask]
  lower_margin = joint_positions - handles.joint_ranges_rad[:, 0]
  upper_margin = handles.joint_ranges_rad[:, 1] - joint_positions
  joint_margin = np.minimum(lower_margin, upper_margin)
  contact_positions = trace.contact_positions_m[evaluation_mask]
  pad_centers = trace.fingertip_positions_m[evaluation_mask]
  pad_normals = trace.pad_normals_world[evaluation_mask]
  contact_vectors = contact_positions - pad_centers
  contact_vector_norms = np.linalg.norm(contact_vectors, axis=2)
  valid_contact_geometry = np.isfinite(contact_positions).all(axis=2) & (
    contact_vector_norms > 1e-9
  )
  contact_alignment = np.sum(contact_vectors * pad_normals, axis=2)
  contact_alignment[valid_contact_geometry] /= contact_vector_norms[
    valid_contact_geometry
  ]
  head_clearances = trace.contact_head_clearances_m[evaluation_mask]
  valid_head_clearances = np.isfinite(head_clearances)
  per_finger_head_clearance = []
  for finger_index in range(4):
    finger_values = head_clearances[:, finger_index]
    finger_values = finger_values[np.isfinite(finger_values)]
    per_finger_head_clearance.append(
      float(np.min(finger_values)) if finger_values.size else None
    )
  return {
    "contact_continuity_probability": float(np.mean(any_contact)),
    "average_contact_count": float(np.mean(np.sum(contacts, axis=1))),
    "per_finger_contact_retention": [float(value) for value in np.mean(contacts, axis=0)],
    "contact_loss_events": contact_loss_events,
    "unexpected_contact_loss_events": unexpected_contact_loss_events,
    "force_rmse_n": float(
      np.sqrt(np.mean((active_forces - config.desired_force_n) ** 2))
    ),
    "force_violation_probability": float(np.mean(active_forces > config.force_limit_n)),
    "max_tip_force_n": float(np.max(forces)),
    "zero_contact_time_s": float(np.sum(~any_contact) * config.dt_s),
    "non_tip_contact_count": int(np.sum(trace.non_tip_contact_count[evaluation_mask])),
    "max_abs_joint_command_error_rad": float(
      np.max(np.abs(trace.joint_commands_rad[evaluation_mask] - joint_positions))
    ),
    "minimum_joint_margin_rad": float(np.min(joint_margin)),
    "joint_limit_probability": float(np.mean(joint_margin <= 0.0)),
    "pad_contact_alignment_mean": float(
      np.mean(contact_alignment[valid_contact_geometry])
    ),
    "pad_contact_alignment_min": float(
      np.min(contact_alignment[valid_contact_geometry])
    ),
    "contact_distal_head_clearance_min_m": float(
      np.min(head_clearances[valid_head_clearances])
    ),
    "per_finger_contact_distal_head_clearance_min_m": per_finger_head_clearance,
    "thumb_contact_probability": float(np.mean(contacts[:, 3])),
  }


def run_scenario(config: PhysicsConfig = PhysicsConfig()) -> tuple[PhysicsTrace, dict[str, Any]]:
  if config.scenario == "maintenance_curved":
    shape = "sphere"
  elif config.scenario == "extreme_surface":
    shape = "extreme"
  else:
    shape = "plane"
  handles = build_scene(shape, timestep_s=config.dt_s)
  rng = np.random.default_rng(config.seed)
  handles.model.geom_friction[handles.object_geom_id, 0] = config.friction_coefficient
  handles.model.geom_friction[handles.tip_geom_ids, 0] = config.friction_coefficient
  data = _initialize_data(handles, config, rng)
  controllers = [
    FingertipMCC(MCCConfig(dt_s=config.dt_s, max_offset_m=0.015))
    for _ in range(4)
  ]
  num_steps = int(round(config.duration_s / config.dt_s))
  time_s = np.arange(num_steps, dtype=np.float64) * config.dt_s
  desired_contacts = np.zeros((num_steps, 4), dtype=np.bool_)
  actual_contacts = np.zeros_like(desired_contacts)
  forces_log = np.zeros((num_steps, 4), dtype=np.float64)
  controller_forces = np.zeros_like(forces_log)
  tip_positions = np.zeros((num_steps, 4, 3), dtype=np.float64)
  pad_normals = np.zeros((num_steps, 4, 3), dtype=np.float64)
  contact_positions = np.full((num_steps, 4, 3), np.nan, dtype=np.float64)
  contact_head_clearances = np.full((num_steps, 4), np.nan, dtype=np.float64)
  q_log = np.zeros((num_steps, 16), dtype=np.float64)
  command_log = np.zeros_like(q_log)
  object_positions = np.zeros((num_steps, 3), dtype=np.float64)
  object_quaternions = np.zeros((num_steps, 4), dtype=np.float64)
  surface_curvatures = np.zeros((num_steps, 4), dtype=np.float64)
  disturbance_active = np.zeros(num_steps, dtype=np.bool_)
  non_tip_contacts = np.zeros(num_steps, dtype=np.int32)
  measured_forces = np.zeros(4, dtype=np.float64)

  for step, timestamp_s in enumerate(time_s):
    object_position, object_quaternion = _object_pose(
      handles,
      config,
      float(timestamp_s),
    )
    data.mocap_pos[handles.object_mocap_id] = object_position
    data.mocap_quat[handles.object_mocap_id] = object_quaternion
    desired_mask = _desired_contact_mask(config, float(timestamp_s))
    desired_contacts[step] = desired_mask

    for finger_index, controller in enumerate(controllers):
      current_tip = data.site_xpos[int(handles.tip_site_ids[finger_index])].copy()
      surface_point, normal, curvature = _surface_reference(
        handles,
        object_position,
        object_quaternion,
        current_tip,
      )
      surface_curvatures[step, finger_index] = curvature
      touching_center = surface_point + _pad_support_radius(
        handles,
        data,
        finger_index,
        normal,
      ) * normal
      touching_center = touching_center + config.surface_bias_m * normal
      if not desired_mask[finger_index]:
        touching_center = touching_center + config.break_clearance_m * normal
      target_force = config.desired_force_n if desired_mask[finger_index] else 0.0
      noisy_force = max(
        0.0,
        float(measured_forces[finger_index])
        + float(rng.normal(0.0, config.force_noise_std_n)),
      )
      controller_forces[step, finger_index] = noisy_force
      command = controller.step(
        touching_center,
        -normal,
        target_force,
        noisy_force,
      )
      joint_command = _ik_joint_command(
        handles,
        data,
        finger_index,
        command.position_command,
        -normal,
        config,
      )
      joint_indices = np.array(
        [int(name) for name in FINGERS[finger_index].joint_names],
        dtype=np.int32,
      )
      data.ctrl[joint_indices] = np.clip(
        joint_command,
        np.maximum(
          handles.model.actuator_ctrlrange[joint_indices, 0],
          handles.joint_ranges_rad[joint_indices, 0] + 0.08,
        ),
        np.minimum(
          handles.model.actuator_ctrlrange[joint_indices, 1],
          handles.joint_ranges_rad[joint_indices, 1] - 0.08,
        ),
      )

    mujoco.mj_step(handles.model, data)
    measured_forces, measured_contact_positions, non_tip_count = _contact_forces(
      handles,
      data,
    )
    actual_contacts[step] = measured_forces >= config.contact_threshold_n
    forces_log[step] = measured_forces
    tip_positions[step] = data.site_xpos[handles.tip_site_ids]
    pad_normals[step] = np.stack(
      [
        data.geom_xmat[int(geom_id)].reshape(3, 3)[:, 2]
        for geom_id in handles.tip_geom_ids
      ]
    )
    contact_positions[step] = measured_contact_positions
    for finger_index, finger in enumerate(FINGERS):
      if not np.isfinite(measured_contact_positions[finger_index]).all():
        continue
      body_id = int(handles.tip_body_ids[finger_index])
      body_rotation = data.xmat[body_id].reshape(3, 3)
      contact_local = body_rotation.T @ (
        measured_contact_positions[finger_index] - data.xpos[body_id]
      )
      contact_head_clearances[step, finger_index] = (
        float(contact_local[1]) - finger.distal_head_y_m
      )
    q_log[step] = data.qpos[handles.joint_qpos_adrs]
    command_log[step] = data.ctrl.copy()
    object_positions[step] = object_position
    object_quaternions[step] = object_quaternion
    disturbance_active[step] = bool(
      config.scenario == "extreme_surface"
      and timestamp_s >= config.pose_step_time_s
    )
    non_tip_contacts[step] = non_tip_count

  trace = PhysicsTrace(
    time_s=time_s,
    desired_contacts=desired_contacts,
    actual_contacts=actual_contacts,
    fingertip_forces_n=forces_log,
    controller_forces_n=controller_forces,
    fingertip_positions_m=tip_positions,
    pad_normals_world=pad_normals,
    contact_positions_m=contact_positions,
    contact_head_clearances_m=contact_head_clearances,
    joint_positions_rad=q_log,
    joint_commands_rad=command_log,
    object_positions_m=object_positions,
    object_quaternions=object_quaternions,
    surface_curvatures_inv_m=surface_curvatures,
    disturbance_active=disturbance_active,
    non_tip_contact_count=non_tip_contacts,
  )
  return trace, _metrics(trace, config, handles)
