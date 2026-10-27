"""Closed-loop physical imitation for the Dataset-D learning smoke test.

The first second is an explicitly labelled teacher-command warm-up used to
recover the demonstrated physical initial state.  From the activation boundary
onward, Finger MCC is absent: Finger DP, its action authority filter and the
runtime guard are the only finger-command authorities.  Wrist MCC remains the
same collective controller used by E05-H-MCC.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import mujoco
import numpy as np
from numpy.typing import ArrayLike, NDArray
import torch

from Module.fr3_leap import FullRobotModelConfig, build_full_robot
from Module.module_4_finger_dp.authority_filter import (
  AuthorityFilterConfig,
  DPActionAuthorityFilter,
  opposition_metrics,
)
from Module.module_4_finger_dp.contact_hysteresis import (
  ContactHysteresisConfig,
  MeasuredContactHysteresis,
)
from Module.module_4_finger_dp.force_history import CausalForcePreprocessor
from Module.module_4_finger_dp.guard_state_machine import (
  DPGuardConfig,
  DPGuardState,
  DPRuntimeGuardExecutor,
)
from Module.module_4_finger_dp.gpu_runtime import require_cuda, synchronize_cuda
from Module.module_4_finger_dp.policy import FingerDiffusionPolicy
from Module.module_4_finger_dp.track_d_dataset import (
  TRACK_D_INPUT_NAMES,
  finger_state_geometry_features,
  load_e05_h_teacher_trace,
  pose_twist_series,
  relative_plan_twists,
)
from Module.module_4_finger_dp.track_d_train import load_track_d_policy
from Module.module_4_whole_hand_mcc.coordinator import (
  ContactForceCoordinator,
  CoordinatorConfig,
)
from Module.module_4_whole_hand_mcc.robot_control import (
  JointTorqueWrenchEstimator,
  PalmPoseIK,
  PalmPoseIKConfig,
)
from Module.module_4_whole_hand_mcc.runner import (
  E05MCCConfig,
  E05MCCTrace,
  _contact_state,
  _initialize_data,
  _pad_support_radius,
  _quaternion_from_matrix,
  _surface_reference,
)
from Module.module_4_whole_hand_mcc.wrist_mcc import WristMCC, WristMCCConfig


TRACK_D_CLOSED_LOOP_SCHEMA_VERSION = "fr3-leap-track-d-closed-loop.v1"


@dataclass(frozen=True, slots=True)
class TrackDClosedLoopConfig:
  duration_s: float = 4.0
  dp_activation_s: float = 1.0
  dt_s: float = 0.002
  policy_period_steps: int = 10
  desired_force_n: float = 2.0
  contact_threshold_n: float = 0.20
  force_limit_n: float = 8.0
  soft_force_n: float = 6.0
  recover_force_n: float = 2.5
  seed: int = 7
  deterministic_zero_latent: bool = True
  torch_device: str = "cuda:0"
  surface: str = "extreme"
  friction_coefficient: float = 0.90
  force_filter_alpha: float = 0.20
  force_noise_std_n: float = 0.0
  initial_joint_noise_std_rad: float = 0.0
  object_offset_x_m: float = 0.0
  object_offset_y_m: float = 0.0
  object_offset_z_m: float = 0.0
  # The extreme surface is traversed at roughly 5--6 mm/s and can locally
  # have slope > 1.  Ten mm/s permits curvature-following finger motion while
  # remaining one quarter of the 40 mm/s Wrist-MCC translational authority.
  collective_normal_velocity_limit_m_s: float = 0.010
  maximum_opposition_energy: float = 1e-5
  rebase_wrist_plan_at_dp_activation: bool = True
  minimum_contact_continuity: float = 0.995
  maximum_zero_contact_gap_s: float = 0.05
  maximum_non_tip_contact_frames: int = 0
  maximum_open_loop_first_command_rmse_rad: float = 0.010

  def __post_init__(self) -> None:
    if self.duration_s <= self.dp_activation_s or self.dp_activation_s < 0.2:
      raise ValueError("Track-D duration/activation window is invalid")
    if self.dt_s <= 0.0 or self.policy_period_steps < 1:
      raise ValueError("Track-D rates must be positive")
    if not self.recover_force_n < self.soft_force_n < self.force_limit_n:
      raise ValueError("force thresholds must satisfy recover < soft < hard")
    if not self.torch_device.startswith("cuda"):
      raise ValueError("Finger DP inference is CUDA-only")
    if self.surface != "extreme":
      raise ValueError("the v1 DP controller is frozen to the extreme surface")
    if self.friction_coefficient <= 0.0:
      raise ValueError("friction_coefficient must be positive")
    if not 0.0 < self.force_filter_alpha <= 1.0:
      raise ValueError("force_filter_alpha must be in (0,1]")
    if self.force_noise_std_n < 0.0 or not np.isfinite(self.force_noise_std_n):
      raise ValueError("force_noise_std_n must be finite and non-negative")
    if self.collective_normal_velocity_limit_m_s <= 0.0:
      raise ValueError("collective normal velocity limit must be positive")
    if self.maximum_opposition_energy <= 0.0:
      raise ValueError("maximum opposition energy must be positive")
    if not all(
      np.isfinite(value)
      for value in (
        self.object_offset_x_m,
        self.object_offset_y_m,
        self.object_offset_z_m,
      )
    ):
      raise ValueError("object offsets must be finite")

  @property
  def policy_dt_s(self) -> float:
    return self.dt_s * self.policy_period_steps


@dataclass(frozen=True, slots=True)
class TrackDClosedLoopTrace:
  time_s: NDArray[np.float64]
  arm_q_rad: NDArray[np.float64]
  arm_dq_rad_s: NDArray[np.float64]
  arm_command_rad: NDArray[np.float64]
  finger_q_rad: NDArray[np.float64]
  finger_dq_rad_s: NDArray[np.float64]
  finger_command_rad: NDArray[np.float64]
  teacher_reference_command_rad: NDArray[np.float64]
  palm_pose_world: NDArray[np.float64]
  planned_palm_pose_world: NDArray[np.float64]
  commanded_palm_pose_world: NDArray[np.float64]
  wrist_mcc_offset: NDArray[np.float64]
  fingertip_positions_world_m: NDArray[np.float64]
  contact_positions_world_m: NDArray[np.float64]
  contact_normals_world: NDArray[np.float64]
  fingertip_forces_n: NDArray[np.float64]
  actual_contacts: NDArray[np.bool_]
  non_tip_contact_count: NDArray[np.int32]
  command_owner: NDArray[np.str_]
  guard_state: NDArray[np.str_]
  guard_reason: NDArray[np.str_]
  authority_solver_success: NDArray[np.bool_]
  authority_solver_status: NDArray[np.str_]
  authority_intervention_norm_rad: NDArray[np.float64]
  authority_maximum_constraint_violation: NDArray[np.float64]
  authority_latency_s: NDArray[np.float64]
  policy_replan: NDArray[np.bool_]
  policy_latency_s: NDArray[np.float64]
  predicted_first_offset_rad: NDArray[np.float64]
  wrist_contact_normal_velocity_m_s: NDArray[np.float64]
  finger_collective_normal_velocity_m_s: NDArray[np.float64]
  wrist_plan_recenter: NDArray[np.bool_]
  authority_contact_transition: NDArray[np.bool_]
  desired_hand_wrench_world: NDArray[np.float64]
  estimated_hand_wrench_world: NDArray[np.float64]
  arm_external_torque_nm: NDArray[np.float64]
  physics_step_latency_s: NDArray[np.float64]
  loop_latency_s: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class TrackDClosedLoopMetrics:
  evaluation_duration_s: float
  contact_continuity: float
  zero_contact_time_s: float
  longest_zero_contact_gap_s: float
  average_contact_count: float
  minimum_contact_count: int
  per_finger_contact_probability: tuple[float, float, float, float]
  contact_loss_events: int
  maximum_force_n: float
  force_p95_n: float
  force_rmse_n: float
  soft_force_exposure_s: float
  non_tip_contact_frames: int
  teacher_command_rmse_rad: float
  teacher_command_maximum_error_rad: float
  policy_replan_count: int
  policy_latency_mean_s: float
  policy_latency_p95_s: float
  authority_intervention_probability: float
  authority_intervention_mean_rad: float
  authority_solver_failure_frames: int
  authority_maximum_constraint_violation: float
  hard_guard_frames: int
  soft_recovery_frames: int
  dp_active_probability: float
  opposition_rate: float
  opposition_energy: float
  opposition_valid_frames: int
  opposition_conflict_frames: int
  finger_collective_normal_velocity_p95_m_s: float
  finger_collective_normal_max_abs_velocity_m_s: float
  wrist_collective_normal_velocity_p95_m_s: float
  wrist_plan_recenter_count: int
  authority_contact_transition_count: int


@dataclass(frozen=True, slots=True)
class TrackDGateVerdict:
  status: str
  blocking_reason: tuple[str, ...]
  checks: Mapping[str, bool]


def _normal_jacobians(
  handles: Any,
  data: mujoco.MjData,
  outward_normals_world: NDArray[np.float64],
) -> NDArray[np.float64]:
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
    result[finger] = (
      outward_normals_world[finger]
      @ jacobian_position[:, handles.hand_dof_adrs]
    )
  return result


def _palm_geometry(
  palm_pose_world: NDArray[np.float64],
  positions_world: NDArray[np.float64],
  normals_world: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
  rotation = np.zeros(9, dtype=np.float64)
  mujoco.mju_quat2Mat(rotation, palm_pose_world[3:])
  rotation = rotation.reshape(3, 3)
  return (
    (positions_world - palm_pose_world[:3]) @ rotation,
    normals_world @ rotation,
  )


def _longest_false_gap(mask: NDArray[np.bool_], dt_s: float) -> float:
  zero = ~mask
  starts = np.flatnonzero(zero & np.r_[True, ~zero[:-1]])
  ends = np.flatnonzero(zero & np.r_[~zero[1:], True])
  return max(
    ((int(end) - int(start) + 1) * dt_s for start, end in zip(starts, ends)),
    default=0.0,
  )


def _policy_inputs(
  *,
  model: FingerDiffusionPolicy,
  force_processor: CausalForcePreprocessor,
  real_twist_history: deque[NDArray[np.float64]],
  wrist_offset_history: deque[NDArray[np.float64]],
  wrist_velocity_history: deque[NDArray[np.float64]],
  finger_state_geometry: NDArray[np.float32],
  current_palm_pose_world: NDArray[np.float64],
  future_plan_poses_world: NDArray[np.float64],
  previous_command_rad: NDArray[np.float64],
  policy_dt_s: float,
) -> dict[str, torch.Tensor]:
  if not force_processor.ready or len(real_twist_history) < 20:
    raise RuntimeError("Track-D policy observation history is not ready")
  horizons = policy_dt_s * np.arange(1, 21, dtype=np.float64)
  rotation = np.zeros(9, dtype=np.float64)
  mujoco.mju_quat2Mat(rotation, current_palm_pose_world[3:])
  rotation = rotation.reshape(3, 3)
  offsets = np.asarray(wrist_offset_history, dtype=np.float64).copy()
  velocities = np.asarray(wrist_velocity_history, dtype=np.float64).copy()
  offsets[:, :3] = offsets[:, :3] @ rotation
  offsets[:, 3:] = offsets[:, 3:] @ rotation
  velocities[:, :3] = velocities[:, :3] @ rotation
  velocities[:, 3:] = velocities[:, 3:] @ rotation
  values: dict[str, NDArray[np.float32]] = {
    "force_history": force_processor.window().encoder_input(),
    "finger_state_geometry": finger_state_geometry,
    "wrist_real_twist_history": np.asarray(real_twist_history, dtype=np.float32),
    "wrist_mcc_offset_history": offsets.astype(np.float32),
    "wrist_mcc_velocity_history": velocities.astype(np.float32),
    "future_wrist_plan_twist": relative_plan_twists(
      current_palm_pose_world,
      future_plan_poses_world,
      horizons,
    ).astype(np.float32),
    "previous_executed_command": previous_command_rad.astype(np.float32),
  }
  if set(values) != set(TRACK_D_INPUT_NAMES):
    raise AssertionError("online Track-D condition contract drifted")
  device = next(model.parameters()).device
  if device.type != "cuda":
    raise RuntimeError("Finger DP inference refuses a non-CUDA model")
  return {
    name: torch.from_numpy(np.array(value, copy=True)).unsqueeze(0).to(
      device=device,
      non_blocking=True,
    )
    for name, value in values.items()
  }


def _plant_config(config: TrackDClosedLoopConfig) -> E05MCCConfig:
  """Create the physical initialization contract; the plan comes from trace."""

  duration_s = max(config.duration_s + 0.5, 2.0)
  return E05MCCConfig(
    mode="E05-H-MCC",
    surface=config.surface,
    duration_s=duration_s,
    settling_time_s=min(1.0, 0.25 * duration_s),
    desired_force_n=config.desired_force_n,
    pose_step_time_s=0.5 * duration_s,
    pose_step_m=0.0,
    friction_coefficient=config.friction_coefficient,
    force_filter_alpha=config.force_filter_alpha,
    initial_joint_noise_std_rad=config.initial_joint_noise_std_rad,
    seed=config.seed,
  )


def run_track_d_closed_loop(
  checkpoint_path: str | Path,
  teacher_trace_path: str | Path,
  config: TrackDClosedLoopConfig = TrackDClosedLoopConfig(),
  *,
  checkpoint_kind: str = "TRACK_D_DIAGNOSTIC",
) -> tuple[TrackDClosedLoopTrace, TrackDClosedLoopMetrics]:
  device, _ = require_cuda(config.torch_device)
  if checkpoint_kind == "TRACK_D_DIAGNOSTIC":
    policy = load_track_d_policy(checkpoint_path, device=device)
  elif checkpoint_kind == "FORMAL_DATASET_I":
    from Module.module_4_finger_dp.formal_train import load_formal_dp_policy

    policy = load_formal_dp_policy(checkpoint_path, device=device)
  else:
    raise ValueError("unsupported checkpoint_kind")
  policy.eval()
  teacher = load_e05_h_teacher_trace(teacher_trace_path)
  base = _plant_config(config)
  if config.dt_s != base.dt_s:
    raise ValueError("DP closed loop must match the physical teacher rate")
  count = int(round(config.duration_s / config.dt_s))
  if count > len(teacher.time_s):
    raise ValueError("teacher plan is shorter than the requested run")

  handles = build_full_robot(
    FullRobotModelConfig(
      surface=base.surface,
      timestep_s=config.dt_s,
      gravity_m_s2=0.0,
      arm_kp=1800.0,
      arm_damping_ratio=0.9,
      object_offset_x_m=config.object_offset_x_m,
      object_offset_y_m=config.object_offset_y_m,
      object_offset_z_m=config.object_offset_z_m,
    )
  )
  handles.model.geom_friction[handles.object_geom_id, 0] = base.friction_coefficient
  handles.model.geom_friction[handles.tip_geom_ids, 0] = base.friction_coefficient
  data = _initialize_data(handles, base)
  observation_rng = np.random.default_rng(config.seed + 103)
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
      dt_s=config.policy_dt_s,
      max_abs_offset=(0.012, 0.012, 0.012, 0.08, 0.08, 0.08),
      max_abs_velocity=(0.04, 0.04, 0.04, 0.3, 0.3, 0.3),
      max_abs_acceleration=(0.8, 0.8, 0.8, 5.0, 5.0, 5.0),
    )
  )
  authority = DPActionAuthorityFilter(
    AuthorityFilterConfig(
      joint_lower_rad=handles.hand_joint_ranges_rad[:, 0],
      joint_upper_rad=handles.hand_joint_ranges_rad[:, 1],
      # The 500 Hz command is already linearly interpolated from a 50 Hz
      # chunk.  These limits remain bounded but permit the measured-state
      # authority transition to brake without making the collective QP
      # artificially infeasible.
      max_abs_delta_rad=0.05,
      max_velocity_rad_s=5.0,
      max_acceleration_rad_s2=500.0,
      max_seam_rad=0.02,
      collective_limit_m=(
        config.collective_normal_velocity_limit_m_s * config.dt_s
      ),
    )
  )
  guard = DPRuntimeGuardExecutor(
    DPGuardConfig(
      joint_lower_rad=handles.hand_joint_ranges_rad[:, 0],
      joint_upper_rad=handles.hand_joint_ranges_rad[:, 1],
      dt_s=config.dt_s,
      soft_force_n=config.soft_force_n,
      hard_force_n=config.force_limit_n,
      recover_force_n=config.recover_force_n,
    )
  )
  force_processor = CausalForcePreprocessor()
  contact_hysteresis = MeasuredContactHysteresis(
    ContactHysteresisConfig(
      enter_force_n=config.contact_threshold_n,
      exit_force_n=0.5 * config.contact_threshold_n,
      confirm_steps=5,
      release_steps=5,
    )
  )
  real_twist_history: deque[NDArray[np.float64]] = deque(maxlen=20)
  wrist_offset_history: deque[NDArray[np.float64]] = deque(maxlen=20)
  wrist_velocity_history: deque[NDArray[np.float64]] = deque(maxlen=20)

  time_s = np.arange(count, dtype=np.float64) * config.dt_s
  arrays: dict[str, NDArray[Any]] = {
    "arm_q_rad": np.zeros((count, 7)),
    "arm_dq_rad_s": np.zeros((count, 7)),
    "arm_command_rad": np.zeros((count, 7)),
    "finger_q_rad": np.zeros((count, 16)),
    "finger_dq_rad_s": np.zeros((count, 16)),
    "finger_command_rad": np.zeros((count, 16)),
    "teacher_reference_command_rad": np.array(teacher.finger_command_rad[:count], copy=True),
    "palm_pose_world": np.zeros((count, 7)),
    "planned_palm_pose_world": np.zeros((count, 7)),
    "commanded_palm_pose_world": np.zeros((count, 7)),
    "wrist_mcc_offset": np.zeros((count, 6)),
    "fingertip_positions_world_m": np.zeros((count, 4, 3)),
    "contact_positions_world_m": np.zeros((count, 4, 3)),
    "contact_normals_world": np.zeros((count, 4, 3)),
    "fingertip_forces_n": np.zeros((count, 4)),
    "actual_contacts": np.zeros((count, 4), dtype=np.bool_),
    "non_tip_contact_count": np.zeros(count, dtype=np.int32),
    "command_owner": np.full(count, "TEACHER_WARMUP", dtype="U40"),
    "guard_state": np.full(count, DPGuardState.INITIALIZE.value, dtype="U40"),
    "guard_reason": np.full(count, "NONE", dtype="U64"),
    "authority_solver_success": np.ones(count, dtype=np.bool_),
    "authority_solver_status": np.full(count, "NOT_APPLICABLE", dtype="U64"),
    "authority_intervention_norm_rad": np.zeros(count),
    "authority_maximum_constraint_violation": np.zeros(count),
    "authority_latency_s": np.zeros(count),
    "policy_replan": np.zeros(count, dtype=np.bool_),
    "policy_latency_s": np.zeros(count),
    "predicted_first_offset_rad": np.zeros((count, 16)),
    "wrist_contact_normal_velocity_m_s": np.zeros((count, 4)),
    "finger_collective_normal_velocity_m_s": np.zeros((count, 4)),
    "wrist_plan_recenter": np.zeros(count, dtype=np.bool_),
    "authority_contact_transition": np.zeros(count, dtype=np.bool_),
    "desired_hand_wrench_world": np.zeros((count, 6)),
    "estimated_hand_wrench_world": np.zeros((count, 6)),
    "arm_external_torque_nm": np.zeros((count, 7)),
    "physics_step_latency_s": np.zeros(count),
    "loop_latency_s": np.zeros(count),
  }

  measured_forces = np.zeros(4)
  measured_positions = np.zeros((4, 3))
  contact_active = np.zeros(4, dtype=np.bool_)
  filtered_wrench = np.zeros(6)
  desired_wrench = np.zeros(6)
  current_wrist_command = initial_pose.copy()
  previous_palm_pose = initial_pose.copy()
  previous_executed_command = np.array(data.ctrl[handles.hand_actuator_ids], copy=True)
  previous_executed_velocity = np.zeros(16)
  chunk_start_command = previous_executed_command.copy()
  chunk_target_command = previous_executed_command.copy()
  chunk_phase = config.policy_period_steps
  dp_authority_initialized = False
  wrist_plan_translation_rebase = np.zeros(3, dtype=np.float64)
  wrist_plan_recentered = False
  authority_active_mask = np.zeros(4, dtype=np.bool_)

  for step, timestamp_s in enumerate(time_s):
    loop_begin = perf_counter()
    planned_pose = np.array(teacher.planned_palm_pose_world[step], copy=True)
    if (
      config.rebase_wrist_plan_at_dp_activation
      and not wrist_plan_recentered
      and timestamp_s >= config.dp_activation_s
    ):
      # A committed prefix begins from the real contact snapshot.  Absorb the
      # initialization MCC displacement into that new nominal wrist plan so
      # the compliance state regains symmetric authority without changing the
      # commanded pose at the ownership transition.
      wrist_plan_translation_rebase[:] = wrist_mcc.state.offset[:3]
      wrist_mcc.reset()
      current_wrist_command = planned_pose.copy()
      current_wrist_command[:3] += wrist_plan_translation_rebase
      wrist_plan_recentered = True
      arrays["wrist_plan_recenter"][step] = True
    planned_pose[:3] += wrist_plan_translation_rebase
    active = contact_active.copy()
    surface_points = np.zeros((4, 3))
    normals = np.zeros((4, 3))
    touching_centers = np.zeros((4, 3))
    for finger in range(4):
      current_tip = np.array(data.site_xpos[int(handles.tip_site_ids[finger])], copy=True)
      surface_point, normal, _ = _surface_reference(
        base.surface,
        handles.object_position_m,
        current_tip,
      )
      surface_points[finger] = surface_point
      normals[finger] = normal
      touching_centers[finger] = surface_point + _pad_support_radius(
        handles,
        data,
        finger,
        normal,
      ) * normal

    estimate = estimator.estimate(data)
    filtered_wrench = (
      (1.0 - base.force_filter_alpha) * filtered_wrench
      + base.force_filter_alpha * estimate.wrench_world
    )
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
    else:
      desired_wrench[:] = 0.0
    if step % config.policy_period_steps == 0:
      selection = np.zeros((6, 6), dtype=np.float64)
      if np.any(active):
        collective_normal = np.mean(normals[active], axis=0)
        collective_normal /= np.linalg.norm(collective_normal)
        selection[:3, :3] = np.outer(collective_normal, collective_normal)
      wrist_command = wrist_mcc.step(
        planned_pose,
        desired_wrench,
        filtered_wrench,
        selection,
      )
      current_wrist_command = wrist_command.pose_command.copy()
    current_palm_pose = np.concatenate(
      (
        data.site_xpos[handles.palm_site_id].copy(),
        _quaternion_from_matrix(data.site_xmat[handles.palm_site_id]),
      )
    )
    current_twist = pose_twist_series(
      np.stack((previous_palm_pose, current_palm_pose)),
      config.dt_s,
    )[1]
    previous_palm_pose = current_palm_pose.copy()
    observed_forces = np.maximum(
      0.0,
      measured_forces
      + observation_rng.normal(0.0, config.force_noise_std_n, 4),
    )
    emitted = force_processor.push(
      observed_forces,
      contact_active,
      np.ones(4, dtype=np.bool_),
    )
    if emitted:
      real_twist_history.append(current_twist)
      wrist_offset_history.append(wrist_mcc.state.offset.copy())
      wrist_velocity_history.append(wrist_mcc.state.velocity.copy())

    jacobian_outward = _normal_jacobians(handles, data, normals)
    signed_compression = -jacobian_outward
    guard_output = guard.step(
      fingertip_force_n=measured_forces,
      force_valid_mask=np.ones(4, dtype=np.bool_),
      history_ready=(force_processor.ready and len(real_twist_history) == 20),
      current_q_rad=data.qpos[handles.hand_qpos_adrs],
      signed_compression_jacobian=signed_compression,
    )
    if guard_output.reset_history:
      force_processor.reset()
      real_twist_history.clear()
      wrist_offset_history.clear()
      wrist_velocity_history.clear()

    wrist_scale = guard_output.wrist_velocity_scale
    guarded_wrist_command = current_palm_pose.copy()
    guarded_wrist_command[:3] += wrist_scale * (
      current_wrist_command[:3] - current_palm_pose[:3]
    )
    target_quaternion = current_wrist_command[3:].copy()
    if np.dot(current_palm_pose[3:], target_quaternion) < 0.0:
      target_quaternion *= -1.0
    guarded_quaternion = (
      (1.0 - wrist_scale) * current_palm_pose[3:]
      + wrist_scale * target_quaternion
    )
    guarded_wrist_command[3:] = guarded_quaternion / np.linalg.norm(
      guarded_quaternion
    )
    arm_control = pose_ik.solve(data, guarded_wrist_command)
    data.ctrl[handles.arm_actuator_ids] = arm_control

    if timestamp_s < config.dp_activation_s:
      finger_command = np.array(teacher.finger_command_rad[step], copy=True)
      owner = "TEACHER_WARMUP"
      authority_result = None
    else:
      if not dp_authority_initialized:
        # A teacher command may contain a collective component that is outside
        # the DP envelope.  Authority transfer is therefore re-anchored at the
        # measured state instead of asking the QP to inherit an incompatible
        # teacher velocity/acceleration constraint.
        previous_executed_command = np.array(
          data.qpos[handles.hand_qpos_adrs],
          copy=True,
        )
        previous_executed_velocity[:] = 0.0
        chunk_start_command = previous_executed_command.copy()
        chunk_target_command = previous_executed_command.copy()
        chunk_phase = config.policy_period_steps
        dp_authority_initialized = True
        authority_active_mask = active.copy()
      elif not np.array_equal(active, authority_active_mask):
        # With incremental affine authority constraints, an unchanged command
        # has zero collective motion in either the old or new P_C.  Record the
        # transition for audit but preserve preload and chunk continuity.
        authority_active_mask = active.copy()
        arrays["authority_contact_transition"][step] = True
      owner = "FINGER_DP"
      if (
        step % config.policy_period_steps == 0
        and guard_output.dp_authority_scale > 0.0
        and force_processor.ready
        and len(real_twist_history) == 20
      ):
        future_indices = np.minimum(
          step + config.policy_period_steps * np.arange(1, 21),
          len(teacher.time_s) - 1,
        )
        future_plan = np.array(
          teacher.planned_palm_pose_world[future_indices],
          copy=True,
        )
        future_plan[:, :3] += wrist_plan_translation_rebase
        state_geometry = finger_state_geometry_features(
          finger_q_rad=data.qpos[handles.hand_qpos_adrs],
          finger_dq_rad_s=data.qvel[handles.hand_dof_adrs],
          fingertip_positions_world_m=data.site_xpos[handles.tip_site_ids],
          measured_contact_positions_world_m=measured_positions,
          actual_contact_mask=contact_active,
          palm_pose_world=current_palm_pose,
          object_position_world_m=handles.object_position_m,
          desired_force_n=config.desired_force_n,
        )
        condition = _policy_inputs(
          model=policy,
          force_processor=force_processor,
          real_twist_history=real_twist_history,
          wrist_offset_history=wrist_offset_history,
          wrist_velocity_history=wrist_velocity_history,
          finger_state_geometry=state_geometry,
          current_palm_pose_world=current_palm_pose,
          future_plan_poses_world=future_plan,
          previous_command_rad=previous_executed_command,
          policy_dt_s=config.policy_dt_s,
        )
        noise = torch.zeros(
          (1, policy.config.action_horizon_steps, 16),
          dtype=torch.float32,
          device=device,
        )
        synchronize_cuda(device)
        policy_begin = perf_counter()
        with torch.inference_mode():
          predicted = policy.sample(
            condition,
            initial_noise=noise,
            deterministic=config.deterministic_zero_latent,
          )[0]
        synchronize_cuda(device)
        arrays["policy_latency_s"][step] = perf_counter() - policy_begin
        predicted = predicted.cpu().numpy()
        arrays["policy_replan"][step] = True
        arrays["predicted_first_offset_rad"][step] = predicted[0]
        chunk_start_command = previous_executed_command.copy()
        chunk_target_command = (
          np.array(data.qpos[handles.hand_qpos_adrs], copy=True)
          + guard_output.dp_authority_scale * predicted[0]
        )
        chunk_phase = 0

      if guard_output.override_delta_rad is not None:
        nominal_command = (
          np.array(data.qpos[handles.hand_qpos_adrs], copy=True)
          + guard_output.override_delta_rad
        )
        owner = "HARD_GUARD_RELEASE"
      elif guard_output.dp_authority_scale <= 0.0:
        nominal_command = previous_executed_command.copy()
        owner = "GUARD_HOLD"
      else:
        alpha = min(1.0, (chunk_phase + 1) / config.policy_period_steps)
        nominal_command = (
          (1.0 - alpha) * chunk_start_command + alpha * chunk_target_command
        )
        chunk_phase += 1

      if guard_output.dp_authority_scale <= 0.0:
        # M03 has higher execution authority than the learned-action QP.  A
        # deterministic release/hold must not be projected back toward the
        # unsafe DP command space.
        finger_command = np.clip(
          nominal_command,
          handles.hand_joint_ranges_rad[:, 0] + 0.02,
          handles.hand_joint_ranges_rad[:, 1] - 0.02,
        )
        arrays["authority_solver_status"][step] = "BYPASSED_BY_M03_FORCE_SAFETY"
      else:
        palm_positions, palm_normals = _palm_geometry(
          current_palm_pose,
          active_positions[active],
          normals[active],
        )
        active_jacobian = jacobian_outward[active]
        palm_rotation = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(palm_rotation, current_palm_pose[3:])
        palm_rotation = palm_rotation.reshape(3, 3)
        wrist_selection = np.zeros((6, 0), dtype=np.float64)
        if np.any(active):
          collective_normal_palm = np.mean(normals[active] @ palm_rotation, axis=0)
          collective_normal_palm /= np.linalg.norm(collective_normal_palm)
          wrist_selection = np.zeros((6, 1), dtype=np.float64)
          wrist_selection[:3, 0] = collective_normal_palm
        authority_result = authority.step(
          current_q_rad=data.qpos[handles.hand_qpos_adrs],
          nominal_delta_rad=(nominal_command - data.qpos[handles.hand_qpos_adrs]),
          previous_executed_command_rad=previous_executed_command,
          previous_executed_velocity_rad_s=previous_executed_velocity,
          finger_normal_jacobian_m_per_rad=active_jacobian,
          active_contact_positions_palm_m=palm_positions,
          active_outward_normals_palm=palm_normals,
          wrist_compliance_selection=wrist_selection,
          dt_s=config.dt_s,
        )
        finger_command = authority_result.safe_command_rad.copy()
        arrays["authority_solver_success"][step] = authority_result.solver_success
        arrays["authority_solver_status"][step] = authority_result.solver_status
        arrays["authority_intervention_norm_rad"][step] = authority_result.intervention_norm_rad
        arrays["authority_maximum_constraint_violation"][step] = (
          authority_result.maximum_constraint_violation
        )
        arrays["authority_latency_s"][step] = authority_result.latency_s
        active_indices = np.flatnonzero(active)
        wrist_velocity_world = wrist_mcc.state.velocity
        wrist_velocity_palm = np.concatenate(
          (
            wrist_velocity_world[:3] @ palm_rotation,
            wrist_velocity_world[3:] @ palm_rotation,
          )
        )
        arrays["wrist_contact_normal_velocity_m_s"][step, active_indices] = (
          authority_result.wrist_contact_normal_map @ wrist_velocity_palm
        )
        arrays["finger_collective_normal_velocity_m_s"][step, active_indices] = (
          authority_result.safe_collective_motion_m / config.dt_s
        )

    data.ctrl[handles.hand_actuator_ids] = finger_command
    current_velocity = (finger_command - previous_executed_command) / config.dt_s
    previous_executed_velocity = current_velocity
    previous_executed_command = finger_command.copy()
    physics_begin = perf_counter()
    mujoco.mj_step(handles.model, data)
    arrays["physics_step_latency_s"][step] = perf_counter() - physics_begin
    measured_forces, _, measured_positions, non_tip = _contact_state(handles, data)
    contact_active = contact_hysteresis.update(
      measured_forces,
    ).actual_contact_mask

    logged_palm = np.concatenate(
      (
        data.site_xpos[handles.palm_site_id],
        _quaternion_from_matrix(data.site_xmat[handles.palm_site_id]),
      )
    )
    arrays["arm_q_rad"][step] = data.qpos[handles.arm_qpos_adrs]
    arrays["arm_dq_rad_s"][step] = data.qvel[handles.arm_dof_adrs]
    arrays["arm_command_rad"][step] = arm_control
    arrays["finger_q_rad"][step] = data.qpos[handles.hand_qpos_adrs]
    arrays["finger_dq_rad_s"][step] = data.qvel[handles.hand_dof_adrs]
    arrays["finger_command_rad"][step] = finger_command
    arrays["palm_pose_world"][step] = logged_palm
    arrays["planned_palm_pose_world"][step] = planned_pose
    arrays["commanded_palm_pose_world"][step] = guarded_wrist_command
    arrays["wrist_mcc_offset"][step] = wrist_mcc.state.offset
    arrays["fingertip_positions_world_m"][step] = data.site_xpos[handles.tip_site_ids]
    arrays["contact_positions_world_m"][step] = measured_positions
    arrays["contact_normals_world"][step] = normals
    arrays["fingertip_forces_n"][step] = measured_forces
    arrays["actual_contacts"][step] = contact_active
    arrays["non_tip_contact_count"][step] = non_tip
    arrays["command_owner"][step] = owner
    arrays["guard_state"][step] = guard_output.state.value
    arrays["guard_reason"][step] = guard_output.reason
    arrays["desired_hand_wrench_world"][step] = desired_wrench
    arrays["estimated_hand_wrench_world"][step] = filtered_wrench
    arrays["arm_external_torque_nm"][step] = estimate.joint_external_torque_nm
    arrays["loop_latency_s"][step] = perf_counter() - loop_begin

  trace = TrackDClosedLoopTrace(time_s=time_s, **arrays)
  metrics = track_d_closed_loop_metrics(trace, config)
  return trace, metrics


def track_d_closed_loop_metrics(
  trace: TrackDClosedLoopTrace,
  config: TrackDClosedLoopConfig,
) -> TrackDClosedLoopMetrics:
  mask = trace.time_s >= config.dp_activation_s
  contacts = trace.actual_contacts[mask]
  forces = trace.fingertip_forces_n[mask]
  any_contact = np.any(contacts, axis=1)
  contact_count = np.sum(contacts, axis=1)
  losses = 0
  for finger in range(4):
    signal = contacts[:, finger]
    losses += int(np.count_nonzero(signal[:-1] & ~signal[1:]))
  replans = trace.policy_replan[mask]
  policy_latency = trace.policy_latency_s[mask][replans]
  intervention = trace.authority_intervention_norm_rad[mask]
  hard = np.isin(
    trace.guard_state[mask],
    [DPGuardState.HARD_RELEASE.value, DPGuardState.SAFE_HOLD.value, DPGuardState.BUFFER_RESET.value],
  )
  soft = trace.guard_state[mask] == DPGuardState.SOFT_RECOVERY.value
  command_error = (
    trace.finger_command_rad[mask] - trace.teacher_reference_command_rad[mask]
  )
  opposition = opposition_metrics(
    trace.finger_collective_normal_velocity_m_s[mask],
    trace.wrist_contact_normal_velocity_m_s[mask],
    dt_s=config.dt_s,
    finger_norm_threshold=1e-5,
    wrist_norm_threshold=1e-5,
  )
  finger_collective = trace.finger_collective_normal_velocity_m_s[mask]
  wrist_collective = trace.wrist_contact_normal_velocity_m_s[mask]
  return TrackDClosedLoopMetrics(
    evaluation_duration_s=float(np.count_nonzero(mask) * config.dt_s),
    contact_continuity=float(np.mean(any_contact)),
    zero_contact_time_s=float(np.count_nonzero(~any_contact) * config.dt_s),
    longest_zero_contact_gap_s=_longest_false_gap(any_contact, config.dt_s),
    average_contact_count=float(np.mean(contact_count)),
    minimum_contact_count=int(np.min(contact_count)),
    per_finger_contact_probability=tuple(float(value) for value in np.mean(contacts, axis=0)),
    contact_loss_events=losses,
    maximum_force_n=float(np.max(forces)),
    force_p95_n=float(np.quantile(forces, 0.95)),
    force_rmse_n=float(np.sqrt(np.mean((forces - config.desired_force_n) ** 2))),
    soft_force_exposure_s=float(
      np.count_nonzero(np.any(forces >= config.soft_force_n, axis=1)) * config.dt_s
    ),
    non_tip_contact_frames=int(np.count_nonzero(trace.non_tip_contact_count[mask] > 0)),
    teacher_command_rmse_rad=float(np.sqrt(np.mean(command_error**2))),
    teacher_command_maximum_error_rad=float(np.max(np.abs(command_error))),
    policy_replan_count=int(np.count_nonzero(replans)),
    policy_latency_mean_s=float(np.mean(policy_latency)) if len(policy_latency) else 0.0,
    policy_latency_p95_s=float(np.quantile(policy_latency, 0.95)) if len(policy_latency) else 0.0,
    authority_intervention_probability=float(np.mean(intervention > 1e-10)),
    authority_intervention_mean_rad=float(np.mean(intervention)),
    authority_solver_failure_frames=int(
      np.count_nonzero(~trace.authority_solver_success[mask])
    ),
    authority_maximum_constraint_violation=float(
      np.max(trace.authority_maximum_constraint_violation[mask])
    ),
    hard_guard_frames=int(np.count_nonzero(hard)),
    soft_recovery_frames=int(np.count_nonzero(soft)),
    dp_active_probability=float(
      np.mean(trace.command_owner[mask] == "FINGER_DP")
    ),
    opposition_rate=opposition.opposition_rate,
    opposition_energy=opposition.opposition_energy,
    opposition_valid_frames=opposition.valid_frame_count,
    opposition_conflict_frames=opposition.conflict_frame_count,
    finger_collective_normal_velocity_p95_m_s=float(
      np.quantile(np.linalg.norm(finger_collective, axis=1), 0.95)
    ),
    finger_collective_normal_max_abs_velocity_m_s=float(
      np.max(np.abs(finger_collective))
    ),
    wrist_collective_normal_velocity_p95_m_s=float(
      np.quantile(np.linalg.norm(wrist_collective, axis=1), 0.95)
    ),
    wrist_plan_recenter_count=int(np.count_nonzero(trace.wrist_plan_recenter)),
    authority_contact_transition_count=int(
      np.count_nonzero(trace.authority_contact_transition)
    ),
  )


def d_gate_verdict(
  *,
  causal_audit_passed: bool,
  open_loop_first_command_rmse_rad: float,
  closed_loop: TrackDClosedLoopMetrics,
  config: TrackDClosedLoopConfig,
) -> TrackDGateVerdict:
  checks = {
    "causal_data_audit": bool(causal_audit_passed),
    "open_loop_first_command_rmse": (
      open_loop_first_command_rmse_rad
      <= config.maximum_open_loop_first_command_rmse_rad
    ),
    "closed_loop_contact_continuity": (
      closed_loop.contact_continuity >= config.minimum_contact_continuity
    ),
    "closed_loop_zero_contact_gap": (
      closed_loop.longest_zero_contact_gap_s <= config.maximum_zero_contact_gap_s
    ),
    "closed_loop_tip_force": closed_loop.maximum_force_n < config.force_limit_n,
    "closed_loop_non_tip_contact": (
      closed_loop.non_tip_contact_frames <= config.maximum_non_tip_contact_frames
    ),
    "authority_solver": closed_loop.authority_solver_failure_frames == 0,
    "authority_collective_velocity": (
      closed_loop.finger_collective_normal_max_abs_velocity_m_s
      <= config.collective_normal_velocity_limit_m_s + 1e-4
    ),
    "hard_guard_takeover": closed_loop.hard_guard_frames == 0,
    "dp_executed": closed_loop.policy_replan_count > 0,
  }
  failed = tuple(name for name, passed in checks.items() if not passed)
  return TrackDGateVerdict(
    status="PASS" if not failed else "FAIL",
    blocking_reason=("NONE",) if not failed else failed,
    checks=checks,
  )


def save_track_d_closed_loop(
  output_directory: str | Path,
  trace: TrackDClosedLoopTrace,
  metrics: TrackDClosedLoopMetrics,
  verdict: TrackDGateVerdict,
  config: TrackDClosedLoopConfig,
) -> tuple[Path, Path]:
  _, cuda_info = require_cuda(config.torch_device)
  output = Path(output_directory)
  output.mkdir(parents=True, exist_ok=True)
  trace_path = output / "closed_loop_trace.npz"
  np.savez_compressed(
    trace_path,
    **{definition.name: getattr(trace, definition.name) for definition in fields(trace)},
  )
  summary_path = output / "d_gate_summary.json"
  summary_path.write_text(
    json.dumps(
      {
        "stage": "D2_CLOSED_LOOP_PHYSICAL_IMITATION",
        "dataset_class": "DATASET_D_DIAGNOSTIC",
        "generalization_evaluated": False,
        "formal_dataset_i_training": False,
        "cuda_only": True,
        "cuda_runtime": cuda_info.to_dict(),
        "finger_mcc_after_dp_activation": False,
        "wrist_mcc_enabled": True,
        "deterministic_zero_latent": config.deterministic_zero_latent,
        "config": asdict(config),
        "metrics": asdict(metrics),
        "d_gate": asdict(verdict),
      },
      indent=2,
      sort_keys=True,
    ),
    encoding="utf-8",
  )
  return trace_path, summary_path


def load_track_d_closed_loop(path: str | Path) -> TrackDClosedLoopTrace:
  with np.load(Path(path), allow_pickle=False) as archive:
    length = len(archive["time_s"])
    values: dict[str, NDArray[Any]] = {}
    for definition in fields(TrackDClosedLoopTrace):
      if definition.name in archive:
        values[definition.name] = archive[definition.name]
      elif definition.name in {
        "wrist_contact_normal_velocity_m_s",
        "finger_collective_normal_velocity_m_s",
      }:
        values[definition.name] = np.zeros((length, 4), dtype=np.float64)
      elif definition.name == "wrist_plan_recenter":
        values[definition.name] = np.zeros(length, dtype=np.bool_)
      elif definition.name == "authority_contact_transition":
        values[definition.name] = np.zeros(length, dtype=np.bool_)
      elif definition.name in {
        "arm_dq_rad_s",
        "arm_command_rad",
        "arm_external_torque_nm",
      }:
        values[definition.name] = np.zeros((length, 7), dtype=np.float64)
      elif definition.name in {
        "desired_hand_wrench_world",
        "estimated_hand_wrench_world",
      }:
        values[definition.name] = np.zeros((length, 6), dtype=np.float64)
      elif definition.name in {"physics_step_latency_s", "loop_latency_s"}:
        values[definition.name] = np.zeros(length, dtype=np.float64)
      else:
        raise ValueError(f"closed-loop trace is missing {definition.name}")
    return TrackDClosedLoopTrace(
      **values
    )
