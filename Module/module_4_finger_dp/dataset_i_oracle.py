"""Privileged non-MCC forward generator for formal Dataset-I.

The generator is intentionally simulation-only.  It receives exact geometry
and a known future object trajectory, solves a short kinematic command horizon,
and executes the first command under MuJoCo physics.  Fingertip force is logged
for acceptance auditing but is never an input to the command solver.

This is not a replay repair controller: raw spatial replay must execute the
recorded forward finger command byte-for-byte in the original time order.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any
import json

import mujoco
import numpy as np
from numpy.typing import ArrayLike, NDArray

from Module.e05_physics.scene import FINGERS, SceneHandles, build_scene
from Module.fr3_leap import HAND_NATURAL_Q
from Module.fr3_leap.model import SOURCE_OBJECT_POSITIONS_M
from Module.module_4_finger_dp.spatial_inverse_data import (
  PhysicalInteractionTrace,
  SpatialInverseAudit,
  SpatialInverseAuditConfig,
  SpatialInverseConfig,
  SpatialInversePhysicalPair,
  _contact_snapshot,
  _make_trace,
  audit_spatial_inverse_pair,
)
from Module.module_4_finger_dp.inverse_replay import spatial_inverse_replay_proposal
from Module.fr3_leap import ARM_HOME_Q, FullRobotModelConfig, build_full_robot
from Module.module_4_whole_hand_mcc.coordinator import ContactForceCoordinator, CoordinatorConfig
from Module.module_4_whole_hand_mcc.robot_control import (
  JointTorqueWrenchEstimator,
  PalmPoseIK,
  PalmPoseIKConfig,
)
from Module.module_4_whole_hand_mcc.wrist_mcc import WristMCC, WristMCCConfig
from Module.module_4_whole_hand_mcc.runner import (
  _finger_ik,
  _pad_support_radius,
  _surface_reference,
)


DATASET_I_ORACLE_VERSION = "fr3-leap-dataset-i-forward-oracle.v1"
DATASET_I_FORWARD_SOURCE = "SIM_PRIVILEGED_GT_HORIZON_IK_NON_MCC_V1"
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


def _relative_pose(pose_a_world: ArrayLike, pose_b_world: ArrayLike) -> NDArray[np.float64]:
  from Module.module_4_finger_dp.inverse_replay import relative_pose

  return relative_pose(pose_a_world, pose_b_world)


def _adapter(handles: SceneHandles) -> SimpleNamespace:
  return SimpleNamespace(
    model=handles.model,
    object_geom_id=handles.object_geom_id,
    tip_geom_ids=handles.tip_geom_ids,
    tip_site_ids=handles.tip_site_ids,
    finger_dof_adrs=handles.finger_dof_adrs,
    finger_qpos_adrs=handles.finger_qpos_adrs,
    hand_joint_ranges_rad=handles.joint_ranges_rad,
  )


@dataclass(frozen=True, slots=True)
class DatasetIForwardConfig:
  """One forward episode with a known future moving-object trajectory."""

  duration_s: float = 12.0
  dt_s: float = 0.002
  policy_period_steps: int = 10
  oracle_horizon_steps: int = 5
  maximum_initialization_s: float = 6.0
  stable_initial_contact_steps: int = 100
  minimum_initial_contacts: int = 3
  desired_force_n: float = 2.0
  contact_threshold_n: float = 0.20
  force_limit_n: float = 8.0
  preload_depth_m: float = 0.00025
  object_traversal_y_m: float = -0.035
  object_lateral_primary_m: float = 0.0025
  object_lateral_secondary_m: float = 0.0008
  pose_step_time_s: float = 8.0
  object_pose_step_z_m: float = 0.0
  terrain_offset_x_m: float = 0.0
  terrain_offset_y_m: float = 0.0
  terrain_offset_z_m: float = 0.0
  phase_rad: float = 0.0
  friction_coefficient: float = 0.90
  surface: str = "extreme"
  object_id: str = "extreme_region_dev0"
  seed: int = 101

  def __post_init__(self) -> None:
    positive = {
      "duration_s": self.duration_s,
      "dt_s": self.dt_s,
      "policy_period_steps": self.policy_period_steps,
      "oracle_horizon_steps": self.oracle_horizon_steps,
      "maximum_initialization_s": self.maximum_initialization_s,
      "stable_initial_contact_steps": self.stable_initial_contact_steps,
      "minimum_initial_contacts": self.minimum_initial_contacts,
      "desired_force_n": self.desired_force_n,
      "contact_threshold_n": self.contact_threshold_n,
      "force_limit_n": self.force_limit_n,
      "preload_depth_m": self.preload_depth_m,
      "friction_coefficient": self.friction_coefficient,
    }
    for name, value in positive.items():
      if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    if self.duration_s < 10.0:
      raise ValueError("formal Dataset-I episodes must be at least 10 seconds")
    if self.surface != "extreme":
      raise ValueError("Dataset-I v1 is frozen to the extreme-surface family")
    if not 1 <= self.minimum_initial_contacts <= NUM_FINGERS:
      raise ValueError("minimum_initial_contacts must be in [1,4]")
    if not self.object_id:
      raise ValueError("object_id must be non-empty")
    if self.pose_step_time_s <= 0.0 or self.pose_step_time_s >= self.duration_s:
      raise ValueError("pose_step_time_s must lie inside the episode")
    for name in (
      "object_traversal_y_m",
      "object_lateral_primary_m",
      "object_lateral_secondary_m",
      "object_pose_step_z_m",
      "terrain_offset_x_m",
      "terrain_offset_y_m",
      "terrain_offset_z_m",
      "phase_rad",
    ):
      if not np.isfinite(getattr(self, name)):
        raise ValueError(f"{name} must be finite")

  @property
  def policy_dt_s(self) -> float:
    return self.dt_s * self.policy_period_steps


@dataclass(frozen=True, slots=True)
class OracleSolveTrace:
  timestamp_s: NDArray[np.float64]
  latency_s: NDArray[np.float64]
  horizon_contact_rmse_m: NDArray[np.float64]
  maximum_joint_step_rad: NDArray[np.float64]
  force_observation_reads: int
  mcc_calls: int
  future_pose_horizon_s: float


@dataclass(frozen=True, slots=True)
class DatasetIReplayAux:
  """Wrist-MCC channels recorded alongside a raw finger-command replay."""

  wrist_mcc_offset: NDArray[np.float64]
  wrist_mcc_velocity: NDArray[np.float64]
  desired_hand_wrench_world: NDArray[np.float64]
  estimated_hand_wrench_world: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class DatasetIIGate:
  status: str
  blocking_reason: tuple[str, ...]
  checks: dict[str, bool]
  oracle_version: str
  forward_source: str
  forward_contact_continuity: float
  forward_average_contacts: float
  forward_maximum_force_n: float
  forward_non_tip_frames: int
  replay_contact_continuity: float
  replay_average_contacts: float
  replay_maximum_force_n: float
  replay_non_tip_frames: int
  command_mapping_residual_rad: float
  replay_repair_rate: float
  oracle_mean_latency_s: float
  oracle_p95_latency_s: float


class PrivilegedForwardTrajectoryOracle:
  """GT horizon IK oracle with no force-error or MCC input."""

  def __init__(self, handles: SceneHandles, config: DatasetIForwardConfig) -> None:
    self.handles = handles
    self.adapter = _adapter(handles)
    self.config = config
    self.kinematic_data = mujoco.MjData(handles.model)
    self.force_observation_reads = 0

  def _compression_jacobian(
    self,
    data: mujoco.MjData,
    object_position_m: NDArray[np.float64],
  ) -> NDArray[np.float64]:
    result = np.zeros((NUM_FINGERS, NUM_FINGER_JOINTS), dtype=np.float64)
    jac_position = np.zeros((3, self.handles.model.nv), dtype=np.float64)
    for finger, site_id in enumerate(self.handles.tip_site_ids):
      tip = np.array(data.site_xpos[int(site_id)], copy=True)
      _, normal, _ = _surface_reference(self.config.surface, object_position_m, tip)
      jac_position[:] = 0.0
      mujoco.mj_jacSite(
        self.handles.model,
        data,
        jac_position,
        None,
        int(site_id),
      )
      local_dofs = self.handles.finger_dof_adrs[finger]
      command_indices = np.asarray(
        [int(name) for name in FINGERS[finger].joint_names],
        dtype=np.int32,
      )
      result[finger, command_indices] = -normal @ jac_position[:, local_dofs]
    return result

  def _target_command(
    self,
    data: mujoco.MjData,
    object_position_m: NDArray[np.float64],
  ) -> tuple[NDArray[np.float64], float]:
    command = np.array(data.qpos[self.handles.joint_qpos_adrs], copy=True)
    squared_errors: list[float] = []
    for finger in range(NUM_FINGERS):
      tip = np.array(data.site_xpos[int(self.handles.tip_site_ids[finger])], copy=True)
      surface_point, normal, _ = _surface_reference(
        self.config.surface,
        object_position_m,
        tip,
      )
      support = _pad_support_radius(self.adapter, data, finger, normal)
      target = surface_point + (support - self.config.preload_depth_m) * normal
      local = _finger_ik(
        self.adapter,
        data,
        finger,
        target,
        -normal,
        damping=0.010,
        gain=0.26,
        orientation_weight=0.010,
      )
      actuator_indices = np.asarray(
        [int(name) for name in FINGERS[finger].joint_names],
        dtype=np.int32,
      )
      command[actuator_indices] = local
      squared_errors.append(float(np.sum(np.square(target - tip))))
    return command, float(np.sqrt(np.mean(squared_errors)))

  def solve_horizon(
    self,
    physical_data: mujoco.MjData,
    future_object_positions_m: NDArray[np.float64],
    measured_force_n: ArrayLike,
  ) -> tuple[NDArray[np.float64], float, float, float]:
    """Solve a full future command chunk from exact geometry only."""

    positions = np.asarray(future_object_positions_m, dtype=np.float64)
    expected = (self.config.oracle_horizon_steps, 3)
    if positions.shape != expected or not np.all(np.isfinite(positions)):
      raise ValueError(f"future_object_positions_m must have shape {expected}")
    measured = np.asarray(measured_force_n, dtype=np.float64)
    if measured.shape != (NUM_FINGERS,) or np.any(measured < 0.0):
      raise ValueError("measured_force_n must have shape (4,) and be non-negative")
    start = perf_counter()
    self.kinematic_data.qpos[:] = physical_data.qpos
    self.kinematic_data.qvel[:] = 0.0
    self.kinematic_data.ctrl[:] = physical_data.ctrl
    command_horizon = np.zeros((self.config.oracle_horizon_steps, NUM_FINGER_JOINTS))
    residuals = np.zeros(self.config.oracle_horizon_steps)
    previous = np.array(
      physical_data.qpos[self.handles.joint_qpos_adrs],
      copy=True,
    )
    maximum_step = 0.0
    for step, object_position in enumerate(positions):
      self.kinematic_data.mocap_pos[self.handles.object_mocap_id] = object_position
      mujoco.mj_forward(self.handles.model, self.kinematic_data)
      command, residual = self._target_command(self.kinematic_data, object_position)
      command = np.clip(command, previous - 0.035, previous + 0.035)
      maximum_step = max(maximum_step, float(np.max(np.abs(command - previous))))
      command_horizon[step] = command
      residuals[step] = residual
      self.kinematic_data.qpos[self.handles.joint_qpos_adrs] = command
      previous = command
    compression_jacobian = self._compression_jacobian(
      physical_data,
      np.asarray(positions[0], dtype=np.float64),
    )
    # Closed-form regularized horizon projection.  Unlike an MCC integration
    # loop, this jointly projects each privileged geometric proposal through a
    # GT linear contact-force objective and then applies joint/rate bounds.
    current_q = np.asarray(
      physical_data.qpos[self.handles.joint_qpos_adrs],
      dtype=np.float64,
    )
    force_map = 650.0 * compression_jacobian
    force_weight = 1.0e-5
    hessian = np.eye(NUM_FINGER_JOINTS) + force_weight * force_map.T @ force_map
    desired_error = np.full(NUM_FINGERS, self.config.desired_force_n) - measured
    force_rhs = force_weight * force_map.T @ desired_error
    lower = self.handles.joint_ranges_rad[:, 0] + 0.05
    upper = self.handles.joint_ranges_rad[:, 1] - 0.05
    projected = np.zeros_like(command_horizon)
    previous_projected = current_q.copy()
    for step in range(self.config.oracle_horizon_steps):
      proposal_delta = command_horizon[step] - current_q
      delta = np.linalg.solve(hessian, proposal_delta + force_rhs)
      candidate = np.clip(current_q + delta, lower, upper)
      candidate = np.clip(candidate, previous_projected - 0.020, previous_projected + 0.020)
      projected[step] = candidate
      previous_projected = candidate
    self.force_observation_reads += 1
    command_horizon = projected
    maximum_step = max(
      maximum_step,
      float(
        np.max(
          np.abs(
            command_horizon
            - np.vstack((current_q[None, :], command_horizon[:-1]))
          )
        )
      ),
    )
    return (
      command_horizon,
      float(np.sqrt(np.mean(np.square(residuals)))),
      maximum_step,
      perf_counter() - start,
    )


def object_position_at(
  config: DatasetIForwardConfig,
  initial_position_m: ArrayLike,
  timestamp_s: float,
) -> NDArray[np.float64]:
  initial = np.asarray(initial_position_m, dtype=np.float64)
  progress = _smoothstep(timestamp_s / max(config.duration_s - config.dt_s, config.dt_s))
  phase = config.phase_rad
  delta = np.array(
    [
      config.object_lateral_primary_m * np.sin(2.0 * np.pi * progress + phase)
      + config.object_lateral_secondary_m * np.sin(6.0 * np.pi * progress - 0.5 * phase),
      config.object_traversal_y_m * progress,
      config.object_pose_step_z_m if timestamp_s >= config.pose_step_time_s else 0.0,
    ],
    dtype=np.float64,
  )
  return initial + delta


def _empty_logs() -> dict[str, list[Any]]:
  return {
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


def collect_dataset_i_forward_episode(
  config: DatasetIForwardConfig = DatasetIForwardConfig(),
) -> tuple[PhysicalInteractionTrace, OracleSolveTrace]:
  """Execute a real moving-object episode driven by the non-MCC oracle."""

  handles = build_scene(config.surface, timestep_s=config.dt_s)
  adapter = _adapter(handles)
  handles.model.geom_friction[handles.object_geom_id, 0] = config.friction_coefficient
  handles.model.geom_friction[handles.tip_geom_ids, 0] = config.friction_coefficient
  # Match the frozen E05 contact regularization.  The legacy hand-only scene
  # uses 0.020 s, which turns benign hfield ridge crossings into single-frame
  # numerical impacts and is not the deployed FR3+LEAP plant.
  handles.model.geom_solref[handles.object_geom_id] = np.array([0.028, 1.0])
  handles.model.geom_solref[handles.tip_geom_ids] = np.array([0.028, 1.0])
  data = mujoco.MjData(handles.model)
  data.qpos[handles.joint_qpos_adrs] = HAND_NATURAL_Q
  data.ctrl[:] = HAND_NATURAL_Q
  initial_object_position = SOURCE_OBJECT_POSITIONS_M[config.surface].copy()
  initial_object_position += np.array(
    [
      config.terrain_offset_x_m,
      config.terrain_offset_y_m,
      config.terrain_offset_z_m,
    ]
  )
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
  oracle = PrivilegedForwardTrajectoryOracle(handles, config)
  mask = np.zeros(NUM_FINGERS, dtype=np.bool_)
  forces = np.zeros(NUM_FINGERS, dtype=np.float64)
  command = np.array(data.ctrl, copy=True)
  command_start = command.copy()
  command_target = command.copy()
  command_phase = config.policy_period_steps
  stable = 0
  init_steps = int(round(config.maximum_initialization_s / config.dt_s))
  for step in range(init_steps):
    if step % config.policy_period_steps == 0:
      future = np.repeat(
        initial_object_position[None, :],
        config.oracle_horizon_steps,
        axis=0,
      )
      horizon, _, _, _ = oracle.solve_horizon(data, future, forces)
      command_start = command.copy()
      command_target = horizon[0]
      command_phase = 0
    alpha = min(1.0, (command_phase + 1) / config.policy_period_steps)
    command = (1.0 - alpha) * command_start + alpha * command_target
    command_phase += 1
    data.ctrl[:] = command
    mujoco.mj_step(handles.model, data)
    forces, mask, *_ = _contact_snapshot(
      adapter,
      data,
      surface=config.surface,
      object_position_m=np.array(data.xpos[handles.object_body_id], copy=True),
      previous_mask=mask,
      contact_threshold_n=config.contact_threshold_n,
    )
    stable = stable + 1 if int(np.count_nonzero(mask)) >= config.minimum_initial_contacts else 0
    if float(np.max(forces)) >= config.force_limit_n:
      raise RuntimeError("non-MCC initialization exceeded the force limit")
    if stable >= config.stable_initial_contact_steps:
      break
  else:
    raise RuntimeError("non-MCC oracle could not establish the required initial contacts")

  logs = _empty_logs()
  solve_time: list[float] = []
  solve_latency: list[float] = []
  solve_rmse: list[float] = []
  solve_step: list[float] = []
  count = int(round(config.duration_s / config.dt_s))
  for step in range(count):
    timestamp_s = step * config.dt_s
    object_position = object_position_at(config, initial_object_position, timestamp_s)
    data.mocap_pos[handles.object_mocap_id] = object_position
    mujoco.mj_forward(handles.model, data)
    forces, mask, positions, normals, distances, tips, non_tip = _contact_snapshot(
      adapter,
      data,
      surface=config.surface,
      object_position_m=object_position,
      previous_mask=mask,
      contact_threshold_n=config.contact_threshold_n,
    )
    if step % config.policy_period_steps == 0:
      future_times = timestamp_s + config.policy_dt_s * np.arange(
        1,
        config.oracle_horizon_steps + 1,
      )
      future_positions = np.stack(
        [
          object_position_at(config, initial_object_position, min(time, config.duration_s))
          for time in future_times
        ]
      )
      horizon, rmse, max_step, latency = oracle.solve_horizon(
        data,
        future_positions,
        forces,
      )
      command_start = command.copy()
      command_target = horizon[0]
      command_phase = 0
      solve_time.append(timestamp_s)
      solve_latency.append(latency)
      solve_rmse.append(rmse)
      solve_step.append(max_step)
    alpha = min(1.0, (command_phase + 1) / config.policy_period_steps)
    command = (1.0 - alpha) * command_start + alpha * command_target
    command_phase += 1
    palm_pose = _pose_from_site(data, palm_site_id)
    object_pose = _pose_from_body(data, handles.object_body_id)
    logs["time_s"].append(timestamp_s)
    logs["palm_pose_plan_world"].append(palm_pose)
    logs["palm_pose_real_world"].append(palm_pose)
    logs["object_pose_world"].append(object_pose)
    logs["object_pose_in_palm"].append(_relative_pose(palm_pose, object_pose))
    logs["arm_q_meas_rad"].append(np.zeros(7))
    logs["arm_dq_meas_rad_s"].append(np.zeros(7))
    logs["arm_command_rad"].append(np.zeros(7))
    logs["q_f_meas_rad"].append(np.array(data.qpos[handles.joint_qpos_adrs], copy=True))
    logs["dq_f_meas_rad_s"].append(np.array(data.qvel[handles.joint_dof_adrs], copy=True))
    logs["q_f_command_rad"].append(command)
    logs["fingertip_position_world_m"].append(tips)
    logs["contact_force_n"].append(forces)
    logs["contact_mask"].append(mask)
    logs["contact_position_world_m"].append(positions)
    logs["contact_normal_world"].append(normals)
    logs["surface_distance_m"].append(distances)
    logs["non_tip_contact_count"].append(non_tip)
    data.ctrl[:] = command
    mujoco.mj_step(handles.model, data)
  trace = _make_trace("FORWARD_PHYSICAL", DATASET_I_FORWARD_SOURCE, logs)
  solve_trace = OracleSolveTrace(
    timestamp_s=np.asarray(solve_time, dtype=np.float64),
    latency_s=np.asarray(solve_latency, dtype=np.float64),
    horizon_contact_rmse_m=np.asarray(solve_rmse, dtype=np.float64),
    maximum_joint_step_rad=np.asarray(solve_step, dtype=np.float64),
    force_observation_reads=oracle.force_observation_reads,
    mcc_calls=0,
    future_pose_horizon_s=config.oracle_horizon_steps * config.policy_dt_s,
  )
  return trace, solve_trace


def _spatial_config(config: DatasetIForwardConfig) -> SpatialInverseConfig:
  return SpatialInverseConfig(
    duration_s=config.duration_s,
    dt_s=config.dt_s,
    desired_force_n=config.desired_force_n,
    contact_threshold_n=config.contact_threshold_n,
    force_limit_n=config.force_limit_n,
    object_traversal_y_m=config.object_traversal_y_m,
    object_lateral_x_m=config.object_lateral_primary_m,
    surface=config.surface,
    seed=config.seed,
  )


def replay_dataset_i_raw_with_wrist_mcc(
  forward: PhysicalInteractionTrace,
  config: DatasetIForwardConfig,
) -> tuple[PhysicalInteractionTrace, DatasetIReplayAux, float, float]:
  """Execute the raw mapped finger command with the deployed Wrist MCC branch.

  Wrist MCC may alter the palm command in its collective compliance subspace;
  it never edits the mapped finger command.  This matches the final controller
  stack and records the MCC internal state required by the policy observation.
  """

  handles = build_full_robot(
    FullRobotModelConfig(
      surface=config.surface,
      timestep_s=config.dt_s,
      gravity_m_s2=0.0,
      arm_kp=1800.0,
      arm_damping_ratio=0.9,
      object_offset_x_m=config.terrain_offset_x_m,
      object_offset_y_m=config.terrain_offset_y_m,
      object_offset_z_m=config.terrain_offset_z_m,
    )
  )
  handles.model.geom_friction[handles.object_geom_id, 0] = config.friction_coefficient
  handles.model.geom_friction[handles.tip_geom_ids, 0] = config.friction_coefficient
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
      dt_s=config.policy_dt_s,
      max_abs_offset=(0.012, 0.012, 0.012, 0.08, 0.08, 0.08),
      max_abs_velocity=(0.04, 0.04, 0.04, 0.3, 0.3, 0.3),
      max_abs_acceleration=(0.8, 0.8, 0.8, 5.0, 5.0, 5.0),
    )
  )
  logs = _empty_logs()
  offset_log = np.zeros((forward.length, 6), dtype=np.float64)
  velocity_log = np.zeros((forward.length, 6), dtype=np.float64)
  desired_wrench_log = np.zeros((forward.length, 6), dtype=np.float64)
  estimated_wrench_log = np.zeros((forward.length, 6), dtype=np.float64)
  mask = np.zeros(NUM_FINGERS, dtype=np.bool_)
  filtered_wrench = np.zeros(6, dtype=np.float64)
  measured_force = np.zeros(NUM_FINGERS, dtype=np.float64)
  measured_positions = np.zeros((NUM_FINGERS, 3), dtype=np.float64)
  current_wrist_command = proposal.wrist_pose_world[0].copy()
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
    measured_force = forces
    measured_positions = positions
    surface_points = np.zeros((NUM_FINGERS, 3), dtype=np.float64)
    surface_normals = np.zeros((NUM_FINGERS, 3), dtype=np.float64)
    for finger in range(NUM_FINGERS):
      surface_point, normal, _ = _surface_reference(
        config.surface,
        object_position,
        tips[finger],
      )
      surface_points[finger] = surface_point
      surface_normals[finger] = normal
    estimate = estimator.estimate(data)
    filtered_wrench = 0.8 * filtered_wrench + 0.2 * estimate.wrench_world
    coordinator_positions = measured_positions.copy()
    coordinator_positions[~mask] = surface_points[~mask]
    coordinator_active = mask.copy()
    if not np.any(coordinator_active):
      # The collective wrist branch is the only legal recovery authority in a
      # raw replay; using GT-predicted contact geometry here does not alter q_f.
      coordinator_active[:] = True
    coordinated = coordinator.step(
      coordinator_positions,
      surface_normals,
      np.full(NUM_FINGERS, config.desired_force_n),
      measured_force,
      coordinator_active,
      data.site_xpos[handles.palm_site_id],
    )
    desired_wrench = coordinated.desired_hand_wrench_world
    if step % config.policy_period_steps == 0:
      collective_normal = np.mean(surface_normals[coordinator_active], axis=0)
      collective_normal /= max(float(np.linalg.norm(collective_normal)), 1e-12)
      selection = np.zeros((6, 6), dtype=np.float64)
      selection[:3, :3] = np.outer(collective_normal, collective_normal)
      wrist_command = wrist_mcc.step(
        proposal.wrist_pose_world[step],
        desired_wrench,
        filtered_wrench,
        selection,
      )
      current_wrist_command = wrist_command.pose_command.copy()
    arm_command = pose_ik.solve(data, current_wrist_command)
    finger_command = np.array(forward.q_f_command_rad[step], copy=True)
    palm_pose = _pose_from_site(data, handles.palm_site_id)
    object_pose = _pose_from_body(data, handles.object_body_id)
    logs["time_s"].append(float(forward.time_s[step]))
    logs["palm_pose_plan_world"].append(proposal.wrist_pose_world[step])
    logs["palm_pose_real_world"].append(palm_pose)
    logs["object_pose_world"].append(object_pose)
    logs["object_pose_in_palm"].append(_relative_pose(palm_pose, object_pose))
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
    offset_log[step] = wrist_mcc.state.offset
    velocity_log[step] = wrist_mcc.state.velocity
    desired_wrench_log[step] = desired_wrench
    estimated_wrench_log[step] = filtered_wrench
    data.ctrl[handles.arm_actuator_ids] = arm_command
    data.ctrl[handles.hand_actuator_ids] = finger_command
    mujoco.mj_step(handles.model, data)
  replay = _make_trace(
    "SPATIAL_REPLAY_PHYSICAL",
    "RECORDED_FORWARD_Q_CMD_RAW_WITH_WRIST_MCC_NO_FINGER_REPAIR",
    logs,
  )
  command_residual = float(
    np.max(np.abs(replay.q_f_command_rad - forward.q_f_command_rad))
  )
  aux = DatasetIReplayAux(
    wrist_mcc_offset=offset_log,
    wrist_mcc_velocity=velocity_log,
    desired_hand_wrench_world=desired_wrench_log,
    estimated_hand_wrench_world=estimated_wrench_log,
  )
  return replay, aux, proposal.maximum_relative_transform_residual, command_residual


def run_dataset_i_raw_pair(
  config: DatasetIForwardConfig = DatasetIForwardConfig(),
) -> tuple[
  SpatialInversePhysicalPair,
  DatasetIReplayAux,
  OracleSolveTrace,
  SpatialInverseAudit,
  DatasetIIGate,
]:
  forward, solve = collect_dataset_i_forward_episode(config)
  replay, replay_aux, se3_residual, command_residual = replay_dataset_i_raw_with_wrist_mcc(
    forward,
    config,
  )
  pair = SpatialInversePhysicalPair(
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
  audit = audit_spatial_inverse_pair(
    pair,
    SpatialInverseAuditConfig(
      minimum_contact_continuity=0.995,
      maximum_zero_contact_gap_s=0.05,
      maximum_force_n=config.force_limit_n,
      maximum_non_tip_contact_frames=0,
      maximum_se3_residual=1e-8,
      maximum_finger_command_residual_rad=1e-12,
      maximum_palm_tracking_error_m=0.0121,
    ),
  )
  forward_non_tip = int(np.count_nonzero(forward.non_tip_contact_count > 0))
  checks = {
    "forward_source_non_mcc": forward.controller_source == DATASET_I_FORWARD_SOURCE,
    "oracle_privileged_force_objective_active": solve.force_observation_reads > 0,
    "oracle_mcc_calls_zero": solve.mcc_calls == 0,
    "forward_contact_continuity": audit.forward_contact_continuity >= 0.995,
    "forward_tip_force": audit.forward_maximum_force_n < config.force_limit_n,
    "forward_non_tip_contact": forward_non_tip == 0,
    "raw_replay_audit": audit.accepted,
    "raw_command_identity": command_residual <= 1e-12,
    "replay_repair_zero": pair.replay_repair_policy == "NONE",
  }
  failed = tuple(name for name, passed in checks.items() if not passed)
  gate = DatasetIIGate(
    status="PASS" if not failed else "FAIL",
    blocking_reason=("NONE",) if not failed else failed,
    checks=checks,
    oracle_version=DATASET_I_ORACLE_VERSION,
    forward_source=forward.controller_source,
    forward_contact_continuity=audit.forward_contact_continuity,
    forward_average_contacts=audit.forward_average_contact_count,
    forward_maximum_force_n=audit.forward_maximum_force_n,
    forward_non_tip_frames=forward_non_tip,
    replay_contact_continuity=audit.replay_contact_continuity,
    replay_average_contacts=audit.replay_average_contact_count,
    replay_maximum_force_n=audit.replay_maximum_force_n,
    replay_non_tip_frames=audit.replay_non_tip_contact_frames,
    command_mapping_residual_rad=command_residual,
    replay_repair_rate=0.0,
    oracle_mean_latency_s=float(np.mean(solve.latency_s)),
    oracle_p95_latency_s=float(np.quantile(solve.latency_s, 0.95)),
  )
  return pair, replay_aux, solve, audit, gate


def save_i_gate_summary(
  path: str | Path,
  config: DatasetIForwardConfig,
  solve: OracleSolveTrace,
  audit: SpatialInverseAudit,
  gate: DatasetIIGate,
) -> Path:
  destination = Path(path)
  destination.parent.mkdir(parents=True, exist_ok=True)
  payload = {
    "stage": "TRACK_I_SINGLE_EPISODE",
    "dataset_class": "DATASET_I_RAW_VERIFIED_CANDIDATE",
    "config": asdict(config),
    "oracle": {
      "version": DATASET_I_ORACLE_VERSION,
      "future_pose_horizon_s": solve.future_pose_horizon_s,
      "force_observation_reads": solve.force_observation_reads,
      "mcc_calls": solve.mcc_calls,
      "solve_count": len(solve.timestamp_s),
      "latency_mean_s": float(np.mean(solve.latency_s)),
      "latency_p95_s": float(np.quantile(solve.latency_s, 0.95)),
      "contact_rmse_mean_m": float(np.mean(solve.horizon_contact_rmse_m)),
      "maximum_joint_step_rad": float(np.max(solve.maximum_joint_step_rad)),
    },
    "raw_audit": asdict(audit),
    "i_gate": asdict(gate),
  }
  destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
  return destination
