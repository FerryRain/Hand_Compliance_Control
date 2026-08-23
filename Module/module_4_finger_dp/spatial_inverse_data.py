"""Physical forward -> spatial inverse -> physical replay data pipeline.

The module deliberately keeps three ideas separate:

* spatial role inversion changes the hand/object reference frame;
* temporal reversal is a different experiment and is not used here;
* fingertip force/contact channels are measured again during replay.

The forward controller is present only in the data-generation environment.  A
replay receives the recorded forward finger command in the original time order
and has no IK, force-repair, or MCC correction on the finger command.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h5py
import mujoco
import numpy as np
from numpy.typing import ArrayLike, NDArray

from Module.e05_physics.scene import FINGERS, SceneHandles, build_scene
from Module.fr3_leap import (
  ARM_HOME_Q,
  HAND_NATURAL_Q,
  FullRobotModelConfig,
  build_full_robot,
)
from Module.fr3_leap.model import SOURCE_OBJECT_POSITIONS_M
from Module.module_2_fingertip_mcc import FingertipMCC, MCCConfig
from Module.module_4_finger_dp.inverse_replay import (
  pose_to_matrix,
  relative_pose,
  spatial_inverse_replay_proposal,
)
from Module.module_4_whole_hand_mcc.robot_control import (
  PalmPoseIK,
  PalmPoseIKConfig,
)
from Module.module_4_whole_hand_mcc.runner import (
  _finger_ik,
  _pad_support_radius,
  _surface_reference,
)


SPATIAL_INVERSE_PAIR_SCHEMA_VERSION = "fr3-leap-spatial-inverse-pair.v1"
NUM_FINGERS = 4
NUM_FINGER_JOINTS = 16


def _smoothstep(value: float) -> float:
  clipped = float(np.clip(value, 0.0, 1.0))
  return clipped * clipped * (3.0 - 2.0 * clipped)


def _pose_from_site(data: mujoco.MjData, site_id: int) -> NDArray[np.float64]:
  quaternion = np.zeros(4, dtype=np.float64)
  mujoco.mju_mat2Quat(quaternion, data.site_xmat[site_id])
  return np.concatenate((np.array(data.site_xpos[site_id], copy=True), quaternion))


def _pose_from_body(data: mujoco.MjData, body_id: int) -> NDArray[np.float64]:
  return np.concatenate(
    (
      np.array(data.xpos[body_id], copy=True),
      np.array(data.xquat[body_id], copy=True),
    )
  )


def _readonly(value: ArrayLike, dtype: Any = np.float64) -> NDArray[Any]:
  result = np.array(value, dtype=dtype, copy=True)
  result.setflags(write=False)
  return result


@dataclass(frozen=True, slots=True)
class SpatialInverseConfig:
  """Configuration for one small, auditable physical pair."""

  duration_s: float = 3.0
  dt_s: float = 0.002
  maximum_initialization_s: float = 5.0
  stable_initial_contact_steps: int = 100
  minimum_initial_contacts: int = 3
  desired_force_n: float = 1.2
  contact_threshold_n: float = 0.20
  force_limit_n: float = 8.0
  object_traversal_y_m: float = -0.010
  object_lateral_x_m: float = 0.0015
  surface: str = "extreme"
  seed: int = 31

  def __post_init__(self) -> None:
    if self.surface != "extreme":
      raise ValueError("the v1 physical-pair audit is frozen to the extreme surface")
    for name in (
      "duration_s",
      "dt_s",
      "maximum_initialization_s",
      "desired_force_n",
      "contact_threshold_n",
      "force_limit_n",
    ):
      value = float(getattr(self, name))
      if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    if self.duration_s < 2.0:
      raise ValueError("the minimum physical-pair episode is 2 seconds")
    if not 1 <= self.minimum_initial_contacts <= NUM_FINGERS:
      raise ValueError("minimum_initial_contacts must be in [1,4]")
    if self.stable_initial_contact_steps < 1:
      raise ValueError("stable_initial_contact_steps must be positive")
    for name in ("object_traversal_y_m", "object_lateral_x_m"):
      if not np.isfinite(getattr(self, name)):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True, slots=True)
class PhysicalInteractionTrace:
  """Causal physical samples from either forward collection or replay."""

  phase: str
  controller_source: str
  time_s: ArrayLike
  palm_pose_plan_world: ArrayLike
  palm_pose_real_world: ArrayLike
  object_pose_world: ArrayLike
  object_pose_in_palm: ArrayLike
  arm_q_meas_rad: ArrayLike
  arm_dq_meas_rad_s: ArrayLike
  arm_command_rad: ArrayLike
  q_f_meas_rad: ArrayLike
  dq_f_meas_rad_s: ArrayLike
  q_f_command_rad: ArrayLike
  fingertip_position_world_m: ArrayLike
  contact_force_n: ArrayLike
  contact_mask: ArrayLike
  contact_position_world_m: ArrayLike
  contact_normal_world: ArrayLike
  surface_distance_m: ArrayLike
  non_tip_contact_count: ArrayLike

  def __post_init__(self) -> None:
    if self.phase not in {"FORWARD_PHYSICAL", "SPATIAL_REPLAY_PHYSICAL"}:
      raise ValueError("unsupported physical interaction phase")
    if not self.controller_source:
      raise ValueError("controller_source must be non-empty")
    time = np.asarray(self.time_s, dtype=np.float64)
    if time.ndim != 1 or len(time) < 2 or np.any(np.diff(time) <= 0.0):
      raise ValueError("time_s must be a strictly increasing vector")
    length = len(time)
    shapes = {
      "time_s": (length,),
      "palm_pose_plan_world": (length, 7),
      "palm_pose_real_world": (length, 7),
      "object_pose_world": (length, 7),
      "object_pose_in_palm": (length, 7),
      "arm_q_meas_rad": (length, 7),
      "arm_dq_meas_rad_s": (length, 7),
      "arm_command_rad": (length, 7),
      "q_f_meas_rad": (length, NUM_FINGER_JOINTS),
      "dq_f_meas_rad_s": (length, NUM_FINGER_JOINTS),
      "q_f_command_rad": (length, NUM_FINGER_JOINTS),
      "fingertip_position_world_m": (length, NUM_FINGERS, 3),
      "contact_force_n": (length, NUM_FINGERS),
      "contact_position_world_m": (length, NUM_FINGERS, 3),
      "contact_normal_world": (length, NUM_FINGERS, 3),
      "surface_distance_m": (length, NUM_FINGERS),
      "non_tip_contact_count": (length,),
    }
    for name, shape in shapes.items():
      value = np.asarray(getattr(self, name), dtype=np.float64)
      if value.shape != shape or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be finite with shape {shape}, got {value.shape}")
      object.__setattr__(self, name, _readonly(value))
    mask = np.asarray(self.contact_mask, dtype=np.bool_)
    if mask.shape != (length, NUM_FINGERS):
      raise ValueError("contact_mask has the wrong shape")
    object.__setattr__(self, "contact_mask", _readonly(mask, np.bool_))
    if np.any(self.contact_force_n < 0.0):
      raise ValueError("contact forces must be non-negative")
    valid_normals = self.contact_mask
    lengths = np.linalg.norm(self.contact_normal_world, axis=2)
    if np.any(np.abs(lengths[valid_normals] - 1.0) > 1e-5):
      raise ValueError("measured contact normals must be unit length")
    if np.any(self.non_tip_contact_count < 0.0):
      raise ValueError("non-tip counts must be non-negative")

  @property
  def length(self) -> int:
    return len(self.time_s)


@dataclass(frozen=True, slots=True)
class SpatialInversePhysicalPair:
  forward: PhysicalInteractionTrace
  replay: PhysicalInteractionTrace
  inversion_mode: str
  time_mapping: str
  finger_command_mapping: str
  forward_provenance: str
  replay_repair_policy: str
  maximum_se3_residual: float
  maximum_finger_command_mapping_residual_rad: float

  def __post_init__(self) -> None:
    if self.inversion_mode != "SPATIAL_ONLY":
      raise ValueError("v1 pair must use SPATIAL_ONLY inversion")
    if self.time_mapping != "SAME_T_FORWARD_ORDER":
      raise ValueError("v1 pair must preserve forward time order")
    if self.finger_command_mapping != "IDENTITY_Q_CMD_FORWARD_TO_REPLAY":
      raise ValueError("v1 pair must reuse the recorded command without reversal")
    if self.replay_repair_policy != "NONE":
      raise ValueError("raw replay cannot contain IK/MCC/force repair")
    if self.forward.length != self.replay.length:
      raise ValueError("forward and replay traces must have equal length")
    if not np.allclose(self.forward.time_s, self.replay.time_s, atol=0.0, rtol=0.0):
      raise ValueError("forward and replay timestamps must be identical")
    if not np.isfinite(self.maximum_se3_residual) or self.maximum_se3_residual < 0.0:
      raise ValueError("invalid SE(3) residual")
    if (
      not np.isfinite(self.maximum_finger_command_mapping_residual_rad)
      or self.maximum_finger_command_mapping_residual_rad < 0.0
    ):
      raise ValueError("invalid finger command residual")


@dataclass(frozen=True, slots=True)
class SpatialInverseAuditConfig:
  minimum_contact_continuity: float = 0.995
  maximum_zero_contact_gap_s: float = 0.05
  maximum_force_n: float = 8.0
  maximum_non_tip_contact_frames: int = 0
  maximum_se3_residual: float = 1e-8
  maximum_finger_command_residual_rad: float = 1e-12
  maximum_palm_tracking_error_m: float = 0.004


@dataclass(frozen=True, slots=True)
class SpatialInverseAudit:
  accepted: bool
  reasons: tuple[str, ...]
  forward_contact_continuity: float
  replay_contact_continuity: float
  forward_average_contact_count: float
  replay_average_contact_count: float
  forward_contact_retention_probability: float
  contact_mask_entry_agreement: float
  replay_per_finger_contact_probability: tuple[float, float, float, float]
  replay_longest_zero_contact_gap_s: float
  forward_maximum_force_n: float
  replay_maximum_force_n: float
  replay_non_tip_contact_frames: int
  maximum_se3_residual: float
  maximum_finger_command_mapping_residual_rad: float
  palm_tracking_rmse_m: float
  palm_tracking_maximum_m: float
  force_profile_rmse_n: float


def _contact_snapshot(
  handles: Any,
  data: mujoco.MjData,
  *,
  surface: str,
  object_position_m: NDArray[np.float64],
  previous_mask: NDArray[np.bool_],
  contact_threshold_n: float,
) -> tuple[
  NDArray[np.float64],
  NDArray[np.bool_],
  NDArray[np.float64],
  NDArray[np.float64],
  NDArray[np.float64],
  NDArray[np.float64],
  int,
]:
  forces = np.zeros(NUM_FINGERS, dtype=np.float64)
  weighted_positions = np.zeros((NUM_FINGERS, 3), dtype=np.float64)
  lookup = {int(geom): index for index, geom in enumerate(handles.tip_geom_ids)}
  contact_force = np.zeros(6, dtype=np.float64)
  non_tip = 0
  for contact_index in range(data.ncon):
    contact = data.contact[contact_index]
    geom_1 = int(contact.geom1)
    geom_2 = int(contact.geom2)
    if int(handles.object_geom_id) not in (geom_1, geom_2):
      continue
    other = geom_2 if geom_1 == int(handles.object_geom_id) else geom_1
    finger_index = lookup.get(other)
    if finger_index is None:
      non_tip += 1
      continue
    mujoco.mj_contactForce(handles.model, data, contact_index, contact_force)
    normal_force = abs(float(contact_force[0]))
    forces[finger_index] += normal_force
    weighted_positions[finger_index] += normal_force * np.asarray(contact.pos)

  mask = np.where(
    previous_mask,
    forces >= 0.5 * contact_threshold_n,
    forces >= contact_threshold_n,
  )
  tips = np.array(
    [data.site_xpos[int(site)] for site in handles.tip_site_ids],
    dtype=np.float64,
  )
  positions = np.zeros((NUM_FINGERS, 3), dtype=np.float64)
  normals = np.zeros((NUM_FINGERS, 3), dtype=np.float64)
  distances = np.zeros(NUM_FINGERS, dtype=np.float64)
  for index in range(NUM_FINGERS):
    surface_point, normal, _ = _surface_reference(
      surface,
      object_position_m,
      tips[index],
    )
    support = _pad_support_radius(handles, data, index, normal)
    distances[index] = float(np.dot(tips[index] - surface_point, normal) - support)
    if forces[index] > 0.0:
      positions[index] = weighted_positions[index] / forces[index]
      _, normals[index], _ = _surface_reference(
        surface,
        object_position_m,
        positions[index],
      )
    elif mask[index]:
      # Hysteresis can retain a contact for one low-force tick.  Use the
      # current oracle geometry but never fabricate a force sample.
      positions[index] = surface_point
      normals[index] = normal
  return forces, mask, positions, normals, distances, tips, non_tip


def _forward_adapter(handles: SceneHandles) -> SimpleNamespace:
  return SimpleNamespace(
    model=handles.model,
    object_geom_id=handles.object_geom_id,
    tip_geom_ids=handles.tip_geom_ids,
    tip_site_ids=handles.tip_site_ids,
    finger_dof_adrs=handles.finger_dof_adrs,
    finger_qpos_adrs=handles.finger_qpos_adrs,
    hand_joint_ranges_rad=handles.joint_ranges_rad,
  )


def _forward_finger_command(
  handles: SceneHandles,
  data: mujoco.MjData,
  controllers: tuple[FingertipMCC, ...],
  measured_force_n: NDArray[np.float64],
  config: SpatialInverseConfig,
) -> NDArray[np.float64]:
  adapter = _forward_adapter(handles)
  object_position = np.array(data.xpos[handles.object_body_id], copy=True)
  command = np.array(data.ctrl, dtype=np.float64, copy=True)
  for index, controller in enumerate(controllers):
    tip = np.array(data.site_xpos[int(handles.tip_site_ids[index])], copy=True)
    surface_point, normal, _ = _surface_reference(config.surface, object_position, tip)
    touching_center = surface_point + _pad_support_radius(
      adapter,
      data,
      index,
      normal,
    ) * normal
    cartesian = controller.step(
      touching_center,
      -normal,
      config.desired_force_n,
      float(measured_force_n[index]),
    )
    local = _finger_ik(
      adapter,
      data,
      index,
      cartesian.position_command,
      -normal,
    )
    actuator_indices = np.asarray(
      [int(name) for name in FINGERS[index].joint_names],
      dtype=np.int32,
    )
    command[actuator_indices] = local
  return command


def _make_trace(
  phase: str,
  controller_source: str,
  logs: dict[str, list[NDArray[Any] | float | int]],
) -> PhysicalInteractionTrace:
  return PhysicalInteractionTrace(
    phase=phase,
    controller_source=controller_source,
    **{name: np.asarray(value) for name, value in logs.items()},
  )


def collect_forward_physical_episode(
  config: SpatialInverseConfig = SpatialInverseConfig(),
) -> PhysicalInteractionTrace:
  """Collect a real moving-object interaction; no trajectory is fabricated."""

  handles = build_scene(config.surface, timestep_s=config.dt_s)
  adapter = _forward_adapter(handles)
  handles.model.geom_friction[handles.object_geom_id, 0] = 0.9
  handles.model.geom_friction[handles.tip_geom_ids, 0] = 0.9
  data = mujoco.MjData(handles.model)
  data.qpos[handles.joint_qpos_adrs] = HAND_NATURAL_Q
  data.ctrl[:] = HAND_NATURAL_Q
  initial_object_position = SOURCE_OBJECT_POSITIONS_M[config.surface].copy()
  data.mocap_pos[handles.object_mocap_id] = initial_object_position
  data.mocap_quat[handles.object_mocap_id] = np.array([1.0, 0.0, 0.0, 0.0])
  mujoco.mj_forward(handles.model, data)
  palm_site_id = mujoco.mj_name2id(
    handles.model,
    mujoco.mjtObj.mjOBJ_SITE,
    "palm_center",
  )
  if palm_site_id < 0:
    raise RuntimeError("source hand is missing palm_center")
  controllers = tuple(
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
    for _ in range(NUM_FINGERS)
  )
  mask = np.zeros(NUM_FINGERS, dtype=np.bool_)
  measured_force = np.zeros(NUM_FINGERS, dtype=np.float64)
  stable = 0
  initialization_steps = int(round(config.maximum_initialization_s / config.dt_s))
  for _ in range(initialization_steps):
    command = _forward_finger_command(
      handles,
      data,
      controllers,
      measured_force,
      config,
    )
    data.ctrl[:] = command
    mujoco.mj_step(handles.model, data)
    measured_force, mask, *_ = _contact_snapshot(
      adapter,
      data,
      surface=config.surface,
      object_position_m=np.array(data.xpos[handles.object_body_id], copy=True),
      previous_mask=mask,
      contact_threshold_n=config.contact_threshold_n,
    )
    stable = stable + 1 if int(np.count_nonzero(mask)) >= config.minimum_initial_contacts else 0
    if stable >= config.stable_initial_contact_steps:
      break
  else:
    raise RuntimeError(
      "forward physical initialization did not establish the required contacts"
    )

  count = int(round(config.duration_s / config.dt_s))
  logs: dict[str, list[Any]] = {
    "time_s": [],
    "palm_pose_plan_world": [],
    "palm_pose_real_world": [],
    "object_pose_world": [],
    "object_pose_in_palm": [],
    "arm_q_meas_rad": [],
    "arm_dq_meas_rad_s": [],
    "arm_command_rad": [],
    "q_f_meas_rad": [],
    "dq_f_meas_rad_s": [],
    "q_f_command_rad": [],
    "fingertip_position_world_m": [],
    "contact_force_n": [],
    "contact_mask": [],
    "contact_position_world_m": [],
    "contact_normal_world": [],
    "surface_distance_m": [],
    "non_tip_contact_count": [],
  }
  for step in range(count):
    time_s = step * config.dt_s
    progress = _smoothstep(time_s / max(config.duration_s - config.dt_s, config.dt_s))
    data.mocap_pos[handles.object_mocap_id] = initial_object_position + np.array(
      [
        config.object_lateral_x_m * np.sin(2.0 * np.pi * progress),
        config.object_traversal_y_m * progress,
        0.0,
      ]
    )
    mujoco.mj_forward(handles.model, data)
    measured_force, mask, positions, normals, distances, tips, non_tip = (
      _contact_snapshot(
        adapter,
        data,
        surface=config.surface,
        object_position_m=np.array(data.xpos[handles.object_body_id], copy=True),
        previous_mask=mask,
        contact_threshold_n=config.contact_threshold_n,
      )
    )
    command = _forward_finger_command(
      handles,
      data,
      controllers,
      measured_force,
      config,
    )
    palm_pose = _pose_from_site(data, palm_site_id)
    logs["time_s"].append(time_s)
    logs["palm_pose_plan_world"].append(palm_pose)
    logs["palm_pose_real_world"].append(palm_pose)
    object_pose = _pose_from_body(data, handles.object_body_id)
    logs["object_pose_world"].append(object_pose)
    logs["object_pose_in_palm"].append(relative_pose(palm_pose, object_pose))
    logs["arm_q_meas_rad"].append(np.zeros(7))
    logs["arm_dq_meas_rad_s"].append(np.zeros(7))
    logs["arm_command_rad"].append(np.zeros(7))
    logs["q_f_meas_rad"].append(np.array(data.qpos[handles.joint_qpos_adrs], copy=True))
    logs["dq_f_meas_rad_s"].append(np.array(data.qvel[handles.joint_dof_adrs], copy=True))
    logs["q_f_command_rad"].append(command)
    logs["fingertip_position_world_m"].append(tips)
    logs["contact_force_n"].append(measured_force)
    logs["contact_mask"].append(mask)
    logs["contact_position_world_m"].append(positions)
    logs["contact_normal_world"].append(normals)
    logs["surface_distance_m"].append(distances)
    logs["non_tip_contact_count"].append(non_tip)
    data.ctrl[:] = command
    mujoco.mj_step(handles.model, data)
  return _make_trace(
    "FORWARD_PHYSICAL",
    "SIM_PRIVILEGED_FINGERTIP_MCC_FORWARD_COLLECTION",
    logs,
  )


def replay_spatial_inverse(
  forward: PhysicalInteractionTrace,
  config: SpatialInverseConfig = SpatialInverseConfig(),
) -> tuple[PhysicalInteractionTrace, float, float]:
  """Replay the recorded command over a fixed object without finger repair."""

  if forward.phase != "FORWARD_PHYSICAL":
    raise ValueError("forward must be a FORWARD_PHYSICAL trace")
  if forward.length != int(round(config.duration_s / config.dt_s)):
    raise ValueError("forward length does not match config")
  handles = build_full_robot(
    FullRobotModelConfig(
      surface=config.surface,
      timestep_s=config.dt_s,
      gravity_m_s2=0.0,
      arm_kp=8000.0,
      arm_damping_ratio=0.95,
    )
  )
  handles.model.geom_friction[handles.object_geom_id, 0] = 0.9
  handles.model.geom_friction[handles.tip_geom_ids, 0] = 0.9
  data = mujoco.MjData(handles.model)
  data.qpos[handles.arm_qpos_adrs] = ARM_HOME_Q
  data.qpos[handles.hand_qpos_adrs] = forward.q_f_meas_rad[0]
  data.ctrl[handles.arm_actuator_ids] = ARM_HOME_Q
  data.ctrl[handles.hand_actuator_ids] = forward.q_f_command_rad[0]
  mujoco.mj_forward(handles.model, data)
  fixed_object_pose = _pose_from_body(data, handles.object_body_id)
  proposal = spatial_inverse_replay_proposal(
    forward.palm_pose_real_world,
    forward.object_pose_world,
    fixed_object_pose,
  )
  current_palm = _pose_from_site(data, handles.palm_site_id)
  initial_position_error = float(
    np.linalg.norm(proposal.wrist_pose_world[0, :3] - current_palm[:3])
  )
  initial_orientation_dot = abs(
    float(np.dot(proposal.wrist_pose_world[0, 3:], current_palm[3:]))
  )
  if initial_position_error > 1e-6 or 1.0 - initial_orientation_dot > 1e-8:
    raise RuntimeError("source and FR3+Leap palm frames are not aligned")
  pose_ik = PalmPoseIK(
    handles,
    PalmPoseIKConfig(
      gain=0.55,
      damping=0.012,
      posture_gain=0.002,
      max_joint_step_rad=0.015,
    ),
  )
  mask = np.zeros(NUM_FINGERS, dtype=np.bool_)
  logs: dict[str, list[Any]] = {
    "time_s": [],
    "palm_pose_plan_world": [],
    "palm_pose_real_world": [],
    "object_pose_world": [],
    "object_pose_in_palm": [],
    "arm_q_meas_rad": [],
    "arm_dq_meas_rad_s": [],
    "arm_command_rad": [],
    "q_f_meas_rad": [],
    "dq_f_meas_rad_s": [],
    "q_f_command_rad": [],
    "fingertip_position_world_m": [],
    "contact_force_n": [],
    "contact_mask": [],
    "contact_position_world_m": [],
    "contact_normal_world": [],
    "surface_distance_m": [],
    "non_tip_contact_count": [],
  }
  for step in range(forward.length):
    object_position = np.array(data.xpos[handles.object_body_id], copy=True)
    forces, mask, positions, normals, distances, tips, non_tip = _contact_snapshot(
      handles,
      data,
      surface=config.surface,
      object_position_m=object_position,
      previous_mask=mask,
      contact_threshold_n=config.contact_threshold_n,
    )
    arm_command = pose_ik.solve(data, proposal.wrist_pose_world[step])
    # Frozen spatial mapping: same t, same numeric finger command.  No
    # current-q hold, IK, MCC, or force-based repair is allowed below.
    finger_command = np.array(forward.q_f_command_rad[step], copy=True)
    logs["time_s"].append(float(forward.time_s[step]))
    palm_pose = _pose_from_site(data, handles.palm_site_id)
    object_pose = _pose_from_body(data, handles.object_body_id)
    logs["palm_pose_plan_world"].append(proposal.wrist_pose_world[step])
    logs["palm_pose_real_world"].append(palm_pose)
    logs["object_pose_world"].append(object_pose)
    logs["object_pose_in_palm"].append(relative_pose(palm_pose, object_pose))
    logs["arm_q_meas_rad"].append(np.array(data.qpos[handles.arm_qpos_adrs], copy=True))
    logs["arm_dq_meas_rad_s"].append(np.array(data.qvel[handles.arm_dof_adrs], copy=True))
    logs["arm_command_rad"].append(arm_command)
    logs["q_f_meas_rad"].append(np.array(data.qpos[handles.hand_qpos_adrs], copy=True))
    logs["dq_f_meas_rad_s"].append(np.array(data.qvel[handles.hand_dof_adrs], copy=True))
    logs["q_f_command_rad"].append(finger_command)
    logs["fingertip_position_world_m"].append(tips)
    logs["contact_force_n"].append(forces)
    logs["contact_mask"].append(mask)
    logs["contact_position_world_m"].append(positions)
    logs["contact_normal_world"].append(normals)
    logs["surface_distance_m"].append(distances)
    logs["non_tip_contact_count"].append(non_tip)
    data.ctrl[handles.arm_actuator_ids] = arm_command
    data.ctrl[handles.hand_actuator_ids] = finger_command
    mujoco.mj_step(handles.model, data)
  replay = _make_trace(
    "SPATIAL_REPLAY_PHYSICAL",
    "RECORDED_FORWARD_Q_CMD_RAW_NO_FINGER_REPAIR",
    logs,
  )
  command_residual = float(
    np.max(np.abs(replay.q_f_command_rad - forward.q_f_command_rad))
  )
  return replay, proposal.maximum_relative_transform_residual, command_residual


def run_spatial_inverse_physical_pair(
  config: SpatialInverseConfig = SpatialInverseConfig(),
) -> SpatialInversePhysicalPair:
  forward = collect_forward_physical_episode(config)
  replay, se3_residual, command_residual = replay_spatial_inverse(forward, config)
  return SpatialInversePhysicalPair(
    forward=forward,
    replay=replay,
    inversion_mode="SPATIAL_ONLY",
    time_mapping="SAME_T_FORWARD_ORDER",
    finger_command_mapping="IDENTITY_Q_CMD_FORWARD_TO_REPLAY",
    forward_provenance=forward.controller_source,
    replay_repair_policy="NONE",
    maximum_se3_residual=se3_residual,
    maximum_finger_command_mapping_residual_rad=command_residual,
  )


def _longest_false_gap(mask: NDArray[np.bool_], dt_s: float) -> float:
  zero = ~mask
  starts = np.flatnonzero(zero & np.r_[True, ~zero[:-1]])
  ends = np.flatnonzero(zero & np.r_[~zero[1:], True])
  return max(
    ((int(end) - int(start) + 1) * dt_s for start, end in zip(starts, ends)),
    default=0.0,
  )


def audit_spatial_inverse_pair(
  pair: SpatialInversePhysicalPair,
  config: SpatialInverseAuditConfig = SpatialInverseAuditConfig(),
) -> SpatialInverseAudit:
  forward_any = np.any(pair.forward.contact_mask, axis=1)
  replay_any = np.any(pair.replay.contact_mask, axis=1)
  dt_s = float(np.median(np.diff(pair.replay.time_s)))
  forward_continuity = float(np.mean(forward_any))
  replay_continuity = float(np.mean(replay_any))
  forward_average_contacts = float(np.mean(np.sum(pair.forward.contact_mask, axis=1)))
  replay_average_contacts = float(np.mean(np.sum(pair.replay.contact_mask, axis=1)))
  forward_contact_count = int(np.count_nonzero(pair.forward.contact_mask))
  retained_forward_contacts = int(
    np.count_nonzero(pair.forward.contact_mask & pair.replay.contact_mask)
  )
  forward_retention = (
    retained_forward_contacts / forward_contact_count if forward_contact_count else 1.0
  )
  contact_agreement = float(
    np.mean(pair.forward.contact_mask == pair.replay.contact_mask)
  )
  replay_per_finger = tuple(
    float(value) for value in np.mean(pair.replay.contact_mask, axis=0)
  )
  zero_gap = _longest_false_gap(replay_any, dt_s)
  forward_max_force = float(np.max(pair.forward.contact_force_n))
  replay_max_force = float(np.max(pair.replay.contact_force_n))
  non_tip_frames = int(np.count_nonzero(pair.replay.non_tip_contact_count > 0))
  palm_error = np.linalg.norm(
    pair.replay.palm_pose_real_world[:, :3]
    - pair.replay.palm_pose_plan_world[:, :3],
    axis=1,
  )
  palm_rmse = float(np.sqrt(np.mean(palm_error**2)))
  palm_max = float(np.max(palm_error))
  force_rmse = float(
    np.sqrt(np.mean((pair.replay.contact_force_n - pair.forward.contact_force_n) ** 2))
  )
  reasons: list[str] = []
  if forward_continuity < config.minimum_contact_continuity:
    reasons.append("FORWARD_CONTACT_CONTINUITY")
  if replay_continuity < config.minimum_contact_continuity:
    reasons.append("REPLAY_CONTACT_CONTINUITY")
  if zero_gap > config.maximum_zero_contact_gap_s:
    reasons.append("REPLAY_ZERO_CONTACT_GAP")
  if forward_max_force > config.maximum_force_n:
    reasons.append("FORWARD_TIP_OVERFORCE")
  if replay_max_force > config.maximum_force_n:
    reasons.append("REPLAY_TIP_OVERFORCE")
  if non_tip_frames > config.maximum_non_tip_contact_frames:
    reasons.append("REPLAY_NON_TIP_CONTACT")
  if pair.maximum_se3_residual > config.maximum_se3_residual:
    reasons.append("SE3_RESIDUAL")
  if (
    pair.maximum_finger_command_mapping_residual_rad
    > config.maximum_finger_command_residual_rad
  ):
    reasons.append("FINGER_COMMAND_MAPPING")
  if palm_max > config.maximum_palm_tracking_error_m:
    reasons.append("PALM_TRACKING")
  return SpatialInverseAudit(
    accepted=not reasons,
    reasons=tuple(reasons),
    forward_contact_continuity=forward_continuity,
    replay_contact_continuity=replay_continuity,
    forward_average_contact_count=forward_average_contacts,
    replay_average_contact_count=replay_average_contacts,
    forward_contact_retention_probability=float(forward_retention),
    contact_mask_entry_agreement=contact_agreement,
    replay_per_finger_contact_probability=replay_per_finger,
    replay_longest_zero_contact_gap_s=zero_gap,
    forward_maximum_force_n=forward_max_force,
    replay_maximum_force_n=replay_max_force,
    replay_non_tip_contact_frames=non_tip_frames,
    maximum_se3_residual=pair.maximum_se3_residual,
    maximum_finger_command_mapping_residual_rad=(
      pair.maximum_finger_command_mapping_residual_rad
    ),
    palm_tracking_rmse_m=palm_rmse,
    palm_tracking_maximum_m=palm_max,
    force_profile_rmse_n=force_rmse,
  )


def save_spatial_inverse_pair(
  path: str | Path,
  pair: SpatialInversePhysicalPair,
) -> Path:
  destination = Path(path)
  destination.parent.mkdir(parents=True, exist_ok=True)
  with h5py.File(destination, "w") as stream:
    stream.attrs["schema_version"] = SPATIAL_INVERSE_PAIR_SCHEMA_VERSION
    stream.attrs["inversion_mode"] = pair.inversion_mode
    stream.attrs["time_mapping"] = pair.time_mapping
    stream.attrs["finger_command_mapping"] = pair.finger_command_mapping
    stream.attrs["forward_provenance"] = pair.forward_provenance
    stream.attrs["replay_repair_policy"] = pair.replay_repair_policy
    stream.attrs["maximum_se3_residual"] = pair.maximum_se3_residual
    stream.attrs["maximum_finger_command_mapping_residual_rad"] = (
      pair.maximum_finger_command_mapping_residual_rad
    )
    for name, trace in (("forward", pair.forward), ("replay", pair.replay)):
      group = stream.create_group(name)
      group.attrs["phase"] = trace.phase
      group.attrs["controller_source"] = trace.controller_source
      for definition in fields(trace):
        field_name = definition.name
        if field_name in {"phase", "controller_source"}:
          continue
        group.create_dataset(
          field_name,
          data=getattr(trace, field_name),
          compression="gzip",
          shuffle=True,
        )
  return destination


def load_spatial_inverse_pair(path: str | Path) -> SpatialInversePhysicalPair:
  with h5py.File(Path(path), "r") as stream:
    if stream.attrs.get("schema_version") != SPATIAL_INVERSE_PAIR_SCHEMA_VERSION:
      raise ValueError("unsupported spatial inverse pair schema")
    traces: dict[str, PhysicalInteractionTrace] = {}
    for name in ("forward", "replay"):
      group = stream[name]
      values = {dataset_name: dataset[...] for dataset_name, dataset in group.items()}
      traces[name] = PhysicalInteractionTrace(
        phase=str(group.attrs["phase"]),
        controller_source=str(group.attrs["controller_source"]),
        **values,
      )
    return SpatialInversePhysicalPair(
      forward=traces["forward"],
      replay=traces["replay"],
      inversion_mode=str(stream.attrs["inversion_mode"]),
      time_mapping=str(stream.attrs["time_mapping"]),
      finger_command_mapping=str(stream.attrs["finger_command_mapping"]),
      forward_provenance=str(stream.attrs["forward_provenance"]),
      replay_repair_policy=str(stream.attrs["replay_repair_policy"]),
      maximum_se3_residual=float(stream.attrs["maximum_se3_residual"]),
      maximum_finger_command_mapping_residual_rad=float(
        stream.attrs["maximum_finger_command_mapping_residual_rad"]
      ),
    )


def palm_frame_contact_geometry(
  trace: PhysicalInteractionTrace,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
  """Transform freshly measured replay contact geometry to the palm frame."""

  positions = np.zeros_like(trace.contact_position_world_m)
  normals = np.zeros_like(trace.contact_normal_world)
  for step in range(trace.length):
    transform = pose_to_matrix(trace.palm_pose_real_world[step])
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    positions[step] = (trace.contact_position_world_m[step] - translation) @ rotation
    normals[step] = trace.contact_normal_world[step] @ rotation
  positions.setflags(write=False)
  normals.setflags(write=False)
  return positions, normals
