"""Build the causal Track-D diagnostic dataset from an E05-H-MCC trace.

Track D intentionally distils a known controller only to test whether the
learning and execution pipeline works.  The resulting samples are never
labelled Dataset-I and never support a generalization claim.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
from typing import Any, Mapping

import mujoco
import numpy as np
from numpy.typing import ArrayLike, NDArray

from Module.fr3_leap import FullRobotModelConfig, build_full_robot
from Module.module_4_finger_dp.contracts import (
  ACTION_HORIZON_STEPS,
  FORCE_HISTORY_STEPS,
  NUM_FINGERS,
  NUM_FINGER_JOINTS,
  WRIST_HISTORY_STEPS,
)
from Module.module_4_finger_dp.force_history import CausalForcePreprocessor
from Module.module_4_whole_hand_mcc.runner import (
  E05MCCTrace,
  _surface_reference,
)


TRACK_D_SAMPLE_SCHEMA_VERSION = "fr3-leap-track-d-samples.v1"
TRACK_D_INPUT_NAMES = (
  "force_history",
  "finger_state_geometry",
  "wrist_real_twist_history",
  "wrist_mcc_offset_history",
  "wrist_mcc_velocity_history",
  "future_wrist_plan_twist",
  "previous_executed_command",
)


def _readonly(value: ArrayLike, dtype: Any = np.float32) -> NDArray[Any]:
  result = np.asarray(value, dtype=dtype)
  if not np.all(np.isfinite(result)):
    raise ValueError("Track-D arrays must be finite")
  result = np.array(result, dtype=dtype, copy=True)
  result.setflags(write=False)
  return result


def _quaternion_matrix(quaternion: ArrayLike) -> NDArray[np.float64]:
  value = np.asarray(quaternion, dtype=np.float64)
  if value.shape != (4,) or not np.isclose(np.linalg.norm(value), 1.0, atol=1e-5):
    raise ValueError("quaternion must be unit length with shape (4,)")
  matrix = np.zeros(9, dtype=np.float64)
  mujoco.mju_quat2Mat(matrix, value)
  return matrix.reshape(3, 3)


def _rotation_log(rotation: NDArray[np.float64]) -> NDArray[np.float64]:
  cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
  angle = float(np.arccos(cosine))
  skew = np.array(
    [
      rotation[2, 1] - rotation[1, 2],
      rotation[0, 2] - rotation[2, 0],
      rotation[1, 0] - rotation[0, 1],
    ],
    dtype=np.float64,
  )
  if angle < 1e-8:
    return 0.5 * skew
  sine = float(np.sin(angle))
  if abs(sine) < 1e-8:
    eigenvalues, eigenvectors = np.linalg.eig(rotation)
    axis = np.real(eigenvectors[:, np.argmin(np.abs(eigenvalues - 1.0))])
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    return angle * axis
  return angle * skew / (2.0 * sine)


def pose_twist_series(
  poses_world: ArrayLike,
  dt_s: float,
) -> NDArray[np.float64]:
  """Return causal backward-difference twists in each current palm frame."""

  poses = np.asarray(poses_world, dtype=np.float64)
  if poses.ndim != 2 or poses.shape[1] != 7 or len(poses) < 2:
    raise ValueError("poses_world must have shape (T,7)")
  result = np.zeros((len(poses), 6), dtype=np.float64)
  for index in range(1, len(poses)):
    current_rotation = _quaternion_matrix(poses[index, 3:])
    previous_rotation = _quaternion_matrix(poses[index - 1, 3:])
    result[index, :3] = (
      current_rotation.T @ (poses[index, :3] - poses[index - 1, :3]) / dt_s
    )
    relative_rotation = previous_rotation.T @ current_rotation
    angular_previous = _rotation_log(relative_rotation) / dt_s
    result[index, 3:] = current_rotation.T @ previous_rotation @ angular_previous
  result[0] = result[1]
  return result


def relative_plan_twists(
  current_pose_world: ArrayLike,
  future_plan_poses_world: ArrayLike,
  future_dt_s: ArrayLike,
) -> NDArray[np.float64]:
  """Express future wrist-plan displacement rates in the current palm frame."""

  current = np.asarray(current_pose_world, dtype=np.float64)
  future = np.asarray(future_plan_poses_world, dtype=np.float64)
  horizons = np.asarray(future_dt_s, dtype=np.float64)
  if current.shape != (7,) or future.ndim != 2 or future.shape[1] != 7:
    raise ValueError("current/future poses have invalid shapes")
  if horizons.shape != (len(future),) or np.any(horizons <= 0.0):
    raise ValueError("future_dt_s must be positive with shape (H,)")
  current_rotation = _quaternion_matrix(current[3:])
  result = np.zeros((len(future), 6), dtype=np.float64)
  for index, pose in enumerate(future):
    target_rotation = _quaternion_matrix(pose[3:])
    result[index, :3] = (
      current_rotation.T @ (pose[:3] - current[:3]) / horizons[index]
    )
    result[index, 3:] = _rotation_log(current_rotation.T @ target_rotation) / horizons[index]
  return result


def load_e05_h_teacher_trace(path: str | Path) -> E05MCCTrace:
  """Load a current H-MCC Dataset-D trace without accepting old DP artefacts."""

  source = Path(path)
  with np.load(source, allow_pickle=False) as archive:
    values: dict[str, NDArray[Any]] = {}
    for definition in fields(E05MCCTrace):
      prefixed = f"H__{definition.name}"
      key = prefixed if prefixed in archive else definition.name
      if key not in archive:
        raise ValueError(
          f"teacher trace is missing {prefixed} or {definition.name}"
        )
      values[definition.name] = archive[key]
  return E05MCCTrace(**values)


@dataclass(frozen=True, slots=True)
class TrackDSampleConfig:
  physics_dt_s: float = 0.002
  policy_period_steps: int = 10
  force_history_period_steps: int = 5
  action_horizon_steps: int = ACTION_HORIZON_STEPS
  start_time_s: float = 0.40
  stop_time_s: float = 5.00
  desired_force_n: float = 2.0
  surface: str = "extreme"
  teacher_source: str = "E05_H_MCC_DATASET_D_DIAGNOSTIC"

  def __post_init__(self) -> None:
    if self.physics_dt_s <= 0.0:
      raise ValueError("physics_dt_s must be positive")
    if self.policy_period_steps < 1 or self.force_history_period_steps < 1:
      raise ValueError("sampling periods must be positive")
    if self.policy_period_steps % self.force_history_period_steps:
      raise ValueError("policy period must align with the force-history rate")
    if self.action_horizon_steps != ACTION_HORIZON_STEPS:
      raise ValueError("Track D uses the frozen v1 action horizon")
    if self.start_time_s < 0.2 or self.stop_time_s <= self.start_time_s:
      raise ValueError("invalid Track-D time window")
    if self.surface != "extreme":
      raise ValueError("Track D is frozen to the E05 extreme surface")

  @property
  def policy_dt_s(self) -> float:
    return self.physics_dt_s * self.policy_period_steps


@dataclass(frozen=True, slots=True)
class TrackDCausalAudit:
  passed: bool
  reasons: tuple[str, ...]
  sample_count: int
  teacher_source: str
  source_start_time_s: float
  source_stop_time_s: float
  physics_rate_hz: float
  force_history_rate_hz: float
  policy_rate_hz: float
  force_history_duration_s: float
  action_horizon_duration_s: float
  maximum_history_timestamp_minus_anchor_s: float
  minimum_target_timestamp_minus_anchor_s: float
  maximum_target_timestamp_minus_anchor_s: float
  future_leakage_count: int
  nonfinite_value_count: int
  maximum_force_n: float
  contact_continuity: float
  teacher_command_vs_measured_rmse_rad: float
  maximum_anchor_construction_residual_rad: float


@dataclass(frozen=True, slots=True)
class TrackDSamples:
  inputs: Mapping[str, NDArray[np.float32]]
  target_action_offsets_rad: NDArray[np.float32]
  anchor_q_meas_rad: NDArray[np.float32]
  future_teacher_command_rad: NDArray[np.float32]
  source_raw_index: NDArray[np.int64]
  timestamp_s: NDArray[np.float64]
  config: TrackDSampleConfig
  audit: TrackDCausalAudit

  def __post_init__(self) -> None:
    if set(self.inputs) != set(TRACK_D_INPUT_NAMES):
      raise ValueError("Track-D condition inputs do not match the frozen policy contract")
    count = len(self.timestamp_s)
    expected = {
      "force_history": (count, NUM_FINGERS, FORCE_HISTORY_STEPS, 3),
      "finger_state_geometry": (count, NUM_FINGERS, 20),
      "wrist_real_twist_history": (count, WRIST_HISTORY_STEPS, 6),
      "wrist_mcc_offset_history": (count, WRIST_HISTORY_STEPS, 6),
      "wrist_mcc_velocity_history": (count, WRIST_HISTORY_STEPS, 6),
      "future_wrist_plan_twist": (count, ACTION_HORIZON_STEPS, 6),
      "previous_executed_command": (count, NUM_FINGER_JOINTS),
    }
    frozen_inputs: dict[str, NDArray[np.float32]] = {}
    for name, shape in expected.items():
      value = _readonly(self.inputs[name], np.float32)
      if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
      frozen_inputs[name] = value
    object.__setattr__(self, "inputs", frozen_inputs)
    target = _readonly(self.target_action_offsets_rad, np.float32)
    anchor = _readonly(self.anchor_q_meas_rad, np.float32)
    future = _readonly(self.future_teacher_command_rad, np.float32)
    indices = np.asarray(self.source_raw_index, dtype=np.int64)
    timestamps = np.asarray(self.timestamp_s, dtype=np.float64)
    if target.shape != (count, ACTION_HORIZON_STEPS, NUM_FINGER_JOINTS):
      raise ValueError("target_action_offsets_rad has the wrong shape")
    if anchor.shape != (count, NUM_FINGER_JOINTS):
      raise ValueError("anchor_q_meas_rad has the wrong shape")
    if future.shape != target.shape:
      raise ValueError("future teacher command has the wrong shape")
    if indices.shape != (count,) or timestamps.shape != (count,):
      raise ValueError("sample metadata has the wrong shape")
    indices = np.array(indices, copy=True)
    timestamps = np.array(timestamps, copy=True)
    indices.setflags(write=False)
    timestamps.setflags(write=False)
    object.__setattr__(self, "target_action_offsets_rad", target)
    object.__setattr__(self, "anchor_q_meas_rad", anchor)
    object.__setattr__(self, "future_teacher_command_rad", future)
    object.__setattr__(self, "source_raw_index", indices)
    object.__setattr__(self, "timestamp_s", timestamps)

  @property
  def count(self) -> int:
    return len(self.timestamp_s)


def finger_state_geometry_features(
  *,
  finger_q_rad: ArrayLike,
  finger_dq_rad_s: ArrayLike,
  fingertip_positions_world_m: ArrayLike,
  measured_contact_positions_world_m: ArrayLike,
  actual_contact_mask: ArrayLike,
  palm_pose_world: ArrayLike,
  object_position_world_m: ArrayLike,
  desired_force_n: float,
) -> NDArray[np.float32]:
  """Build the shared causal state/geometry feature online or offline."""

  palm_pose = np.asarray(palm_pose_world, dtype=np.float64)
  tips = np.asarray(fingertip_positions_world_m, dtype=np.float64)
  measured_positions = np.asarray(measured_contact_positions_world_m, dtype=np.float64)
  contact = np.asarray(actual_contact_mask, dtype=np.bool_)
  q_flat = np.asarray(finger_q_rad, dtype=np.float64)
  dq_flat = np.asarray(finger_dq_rad_s, dtype=np.float64)
  object_position = np.asarray(object_position_world_m, dtype=np.float64)
  if palm_pose.shape != (7,) or tips.shape != (NUM_FINGERS, 3):
    raise ValueError("online palm/tip geometry has the wrong shape")
  if measured_positions.shape != (NUM_FINGERS, 3) or contact.shape != (NUM_FINGERS,):
    raise ValueError("online contact geometry has the wrong shape")
  if q_flat.shape != (NUM_FINGER_JOINTS,) or dq_flat.shape != (NUM_FINGER_JOINTS,):
    raise ValueError("online finger state has the wrong shape")
  rotation = _quaternion_matrix(palm_pose[3:])
  translation = palm_pose[:3]
  positions_world = np.zeros((NUM_FINGERS, 3), dtype=np.float64)
  normals_world = np.zeros((NUM_FINGERS, 3), dtype=np.float64)
  distances = np.zeros(NUM_FINGERS, dtype=np.float64)
  for finger in range(NUM_FINGERS):
    tip = tips[finger]
    surface_point, predicted_normal, _ = _surface_reference(
      "extreme",
      object_position,
      tip,
    )
    if contact[finger] and np.linalg.norm(measured_positions[finger]) > 1e-12:
      positions_world[finger] = measured_positions[finger]
      # Dataset-D uses the Oracle surface normal even at a measured contact;
      # the position is physical, while the normal is never fabricated from
      # a force vector or a future state.
      normals_world[finger] = predicted_normal
    else:
      positions_world[finger] = surface_point
      normals_world[finger] = predicted_normal
    normals_world[finger] /= max(float(np.linalg.norm(normals_world[finger])), 1e-12)
    distances[finger] = float(np.dot(tip - surface_point, predicted_normal))

  positions_palm = (positions_world - translation) @ rotation
  normals_palm = normals_world @ rotation
  q = q_flat.reshape(NUM_FINGERS, 4)
  dq = dq_flat.reshape(NUM_FINGERS, 4)
  state = np.concatenate(
    (
      q,
      dq,
      positions_palm,
      normals_palm,
      distances[:, None],
      np.zeros((NUM_FINGERS, 1), dtype=np.float64),
      contact[:, None].astype(np.float64),
      np.ones((NUM_FINGERS, 1), dtype=np.float64),
      np.full((NUM_FINGERS, 1), desired_force_n, dtype=np.float64),
      np.ones((NUM_FINGERS, 1), dtype=np.float64),
    ),
    axis=1,
  )
  if state.shape != (NUM_FINGERS, 20):
    raise AssertionError("Track-D finger state feature dimension drifted")
  return state.astype(np.float32)


def _finger_geometry_features(
  trace: E05MCCTrace,
  raw_index: int,
  object_position_world_m: NDArray[np.float64],
  desired_force_n: float,
) -> NDArray[np.float32]:
  return finger_state_geometry_features(
    finger_q_rad=trace.finger_q_rad[raw_index],
    finger_dq_rad_s=trace.finger_dq_rad_s[raw_index],
    fingertip_positions_world_m=trace.fingertip_positions_world_m[raw_index],
    measured_contact_positions_world_m=trace.contact_positions_world_m[raw_index],
    actual_contact_mask=trace.actual_contacts[raw_index],
    palm_pose_world=trace.palm_pose_world[raw_index],
    object_position_world_m=object_position_world_m,
    desired_force_n=desired_force_n,
  )


def build_track_d_samples(
  trace: E05MCCTrace,
  config: TrackDSampleConfig = TrackDSampleConfig(),
  *,
  object_position_series_world_m: ArrayLike | None = None,
  wrist_mcc_velocity_override: ArrayLike | None = None,
) -> TrackDSamples:
  """Create causal 50 Hz DP samples from a 500 Hz physical teacher trace."""

  if len(trace.time_s) < 2:
    raise ValueError("teacher trace is empty")
  dt_s = float(np.median(np.diff(trace.time_s)))
  if not np.isclose(dt_s, config.physics_dt_s, atol=1e-12):
    raise ValueError("teacher physics rate differs from Track-D config")
  handles = build_full_robot(
    FullRobotModelConfig(surface=config.surface, timestep_s=config.physics_dt_s)
  )
  if object_position_series_world_m is None:
    object_position = np.repeat(
      handles.object_position_m[None, :],
      len(trace.time_s),
      axis=0,
    )
  else:
    object_position = np.asarray(object_position_series_world_m, dtype=np.float64)
    if object_position.shape != (len(trace.time_s), 3) or not np.all(
      np.isfinite(object_position)
    ):
      raise ValueError("object_position_series_world_m must have shape (T,3)")
  real_twist = pose_twist_series(trace.palm_pose_world, dt_s)
  wrist_velocity = np.zeros_like(trace.wrist_compliance_offset)
  # The saved Wrist MCC state updates at the same 50 Hz boundary as the DP.
  # A single-tick derivative would be an impulse that the 100 Hz observation
  # can miss, so reconstruct the held controller velocity over one policy
  # period using a causal backward difference.
  period = config.policy_period_steps
  if wrist_mcc_velocity_override is None:
    wrist_velocity[period:] = (
      trace.wrist_compliance_offset[period:]
      - trace.wrist_compliance_offset[:-period]
    ) / config.policy_dt_s
    wrist_velocity[:period] = wrist_velocity[period]
  else:
    override = np.asarray(wrist_mcc_velocity_override, dtype=np.float64)
    if override.shape != wrist_velocity.shape or not np.all(np.isfinite(override)):
      raise ValueError("wrist_mcc_velocity_override must have shape (T,6)")
    wrist_velocity[:] = override

  force_processor = CausalForcePreprocessor()
  force_windows: dict[int, NDArray[np.float32]] = {}
  for raw_index in range(len(trace.time_s)):
    emitted = force_processor.push(
      trace.fingertip_forces_n[raw_index],
      trace.actual_contacts[raw_index],
      np.ones(NUM_FINGERS, dtype=np.bool_),
    )
    if emitted and force_processor.ready:
      force_windows[raw_index] = force_processor.window().encoder_input()

  inputs: dict[str, list[NDArray[Any]]] = {name: [] for name in TRACK_D_INPUT_NAMES}
  targets: list[NDArray[np.float64]] = []
  anchors: list[NDArray[np.float64]] = []
  future_commands: list[NDArray[np.float64]] = []
  raw_indices: list[int] = []
  timestamps: list[float] = []
  latest_history_offsets: list[float] = []
  first_target_offsets: list[float] = []
  last_target_offsets: list[float] = []
  future_leakage_count = 0
  anchor_residual = 0.0

  history_raw_offsets = config.force_history_period_steps * np.arange(
    WRIST_HISTORY_STEPS - 1,
    -1,
    -1,
    dtype=np.int64,
  )
  future_raw_offsets = config.policy_period_steps * np.arange(
    1,
    ACTION_HORIZON_STEPS + 1,
    dtype=np.int64,
  )
  future_horizons_s = future_raw_offsets.astype(np.float64) * dt_s
  for raw_index, timestamp_s in enumerate(trace.time_s):
    if (raw_index + 1) % config.policy_period_steps:
      continue
    if timestamp_s < config.start_time_s or timestamp_s > config.stop_time_s:
      continue
    history_indices = raw_index - history_raw_offsets
    target_indices = raw_index + future_raw_offsets
    if history_indices[0] < 0 or target_indices[-1] >= len(trace.time_s):
      continue
    if raw_index not in force_windows:
      continue
    future_leakage_count += int(np.count_nonzero(history_indices > raw_index))
    latest_history_offsets.append(float(trace.time_s[history_indices[-1]] - timestamp_s))
    first_target_offsets.append(float(trace.time_s[target_indices[0]] - timestamp_s))
    last_target_offsets.append(float(trace.time_s[target_indices[-1]] - timestamp_s))

    palm_rotation = _quaternion_matrix(trace.palm_pose_world[raw_index, 3:])
    history_real = real_twist[history_indices]
    history_offset = np.array(trace.wrist_compliance_offset[history_indices], copy=True)
    history_velocity = np.array(wrist_velocity[history_indices], copy=True)
    history_offset[:, :3] = history_offset[:, :3] @ palm_rotation
    history_offset[:, 3:] = history_offset[:, 3:] @ palm_rotation
    history_velocity[:, :3] = history_velocity[:, :3] @ palm_rotation
    history_velocity[:, 3:] = history_velocity[:, 3:] @ palm_rotation
    future_plan = relative_plan_twists(
      trace.palm_pose_world[raw_index],
      trace.planned_palm_pose_world[target_indices],
      future_horizons_s,
    )
    anchor = np.array(trace.finger_q_rad[raw_index], copy=True)
    future_command = np.array(trace.finger_command_rad[target_indices], copy=True)
    target = future_command - anchor[None, :]
    anchor_residual = max(
      anchor_residual,
      float(np.max(np.abs((target + anchor[None, :]) - future_command))),
    )

    inputs["force_history"].append(force_windows[raw_index])
    inputs["finger_state_geometry"].append(
      _finger_geometry_features(
        trace,
        raw_index,
        object_position[raw_index],
        config.desired_force_n,
      )
    )
    inputs["wrist_real_twist_history"].append(history_real)
    inputs["wrist_mcc_offset_history"].append(history_offset)
    inputs["wrist_mcc_velocity_history"].append(history_velocity)
    inputs["future_wrist_plan_twist"].append(future_plan)
    inputs["previous_executed_command"].append(
      trace.finger_command_rad[max(raw_index - 1, 0)]
    )
    targets.append(target)
    anchors.append(anchor)
    future_commands.append(future_command)
    raw_indices.append(raw_index)
    timestamps.append(float(timestamp_s))

  if not targets:
    raise RuntimeError("Track-D sampling produced no training examples")
  input_arrays = {
    name: np.asarray(value, dtype=np.float32) for name, value in inputs.items()
  }
  target_array = np.asarray(targets, dtype=np.float32)
  anchor_array = np.asarray(anchors, dtype=np.float32)
  future_array = np.asarray(future_commands, dtype=np.float32)
  all_arrays = [*input_arrays.values(), target_array, anchor_array, future_array]
  nonfinite_count = sum(int(np.count_nonzero(~np.isfinite(value))) for value in all_arrays)
  selected = np.asarray(raw_indices, dtype=np.int64)
  selected_contact = np.any(trace.actual_contacts[selected], axis=1)
  reasons: list[str] = []
  if future_leakage_count:
    reasons.append("HISTORY_FUTURE_LEAKAGE")
  if nonfinite_count:
    reasons.append("NONFINITE_SAMPLE")
  if not np.isclose(max(latest_history_offsets), 0.0, atol=1e-12):
    reasons.append("HISTORY_NOT_ANCHORED_AT_CURRENT_TICK")
  if not np.isclose(min(first_target_offsets), config.policy_dt_s, atol=1e-12):
    reasons.append("FIRST_ACTION_TIMESTAMP_MISALIGNED")
  if anchor_residual > 1e-7:
    reasons.append("MEASURED_ANCHOR_RESIDUAL")
  audit = TrackDCausalAudit(
    passed=not reasons,
    reasons=tuple(reasons),
    sample_count=len(targets),
    teacher_source=config.teacher_source,
    source_start_time_s=float(timestamps[0]),
    source_stop_time_s=float(timestamps[-1]),
    physics_rate_hz=1.0 / dt_s,
    force_history_rate_hz=1.0 / (dt_s * config.force_history_period_steps),
    policy_rate_hz=1.0 / config.policy_dt_s,
    force_history_duration_s=FORCE_HISTORY_STEPS * dt_s * config.force_history_period_steps,
    action_horizon_duration_s=ACTION_HORIZON_STEPS * config.policy_dt_s,
    maximum_history_timestamp_minus_anchor_s=max(latest_history_offsets),
    minimum_target_timestamp_minus_anchor_s=min(first_target_offsets),
    maximum_target_timestamp_minus_anchor_s=max(last_target_offsets),
    future_leakage_count=future_leakage_count,
    nonfinite_value_count=nonfinite_count,
    maximum_force_n=float(np.max(trace.fingertip_forces_n[selected])),
    contact_continuity=float(np.mean(selected_contact)),
    teacher_command_vs_measured_rmse_rad=float(
      np.sqrt(
        np.mean(
          (
            trace.finger_command_rad[selected]
            - trace.finger_q_rad[selected]
          )
          ** 2
        )
      )
    ),
    maximum_anchor_construction_residual_rad=anchor_residual,
  )
  return TrackDSamples(
    inputs=input_arrays,
    target_action_offsets_rad=target_array,
    anchor_q_meas_rad=anchor_array,
    future_teacher_command_rad=future_array,
    source_raw_index=selected,
    timestamp_s=np.asarray(timestamps, dtype=np.float64),
    config=config,
    audit=audit,
  )


def save_track_d_samples(path: str | Path, samples: TrackDSamples) -> Path:
  destination = Path(path)
  destination.parent.mkdir(parents=True, exist_ok=True)
  values: dict[str, NDArray[Any]] = {
    **samples.inputs,
    "target_action_offsets_rad": samples.target_action_offsets_rad,
    "anchor_q_meas_rad": samples.anchor_q_meas_rad,
    "future_teacher_command_rad": samples.future_teacher_command_rad,
    "source_raw_index": samples.source_raw_index,
    "timestamp_s": samples.timestamp_s,
  }
  np.savez_compressed(destination, **values)
  metadata = {
    "schema_version": TRACK_D_SAMPLE_SCHEMA_VERSION,
    "dataset_class": "DATASET_D_DIAGNOSTIC",
    "training_authorization": "D_GATE_ONLY",
    "generalization_claim_allowed": False,
    "formal_dataset_i_ready": False,
    "config": asdict(samples.config),
    "causal_audit": asdict(samples.audit),
  }
  destination.with_suffix(".json").write_text(
    json.dumps(metadata, indent=2, sort_keys=True),
    encoding="utf-8",
  )
  return destination


def load_track_d_samples(path: str | Path) -> TrackDSamples:
  source = Path(path)
  metadata = json.loads(source.with_suffix(".json").read_text(encoding="utf-8"))
  if metadata.get("schema_version") != TRACK_D_SAMPLE_SCHEMA_VERSION:
    raise ValueError("unsupported Track-D sample schema")
  with np.load(source, allow_pickle=False) as archive:
    inputs = {name: archive[name] for name in TRACK_D_INPUT_NAMES}
    return TrackDSamples(
      inputs=inputs,
      target_action_offsets_rad=archive["target_action_offsets_rad"],
      anchor_q_meas_rad=archive["anchor_q_meas_rad"],
      future_teacher_command_rad=archive["future_teacher_command_rad"],
      source_raw_index=archive["source_raw_index"],
      timestamp_s=archive["timestamp_s"],
      config=TrackDSampleConfig(**metadata["config"]),
      audit=TrackDCausalAudit(**metadata["causal_audit"]),
    )
