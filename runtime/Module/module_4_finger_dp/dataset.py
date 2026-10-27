"""Versioned physical-replay dataset contract for Finger DP v1."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from numpy.typing import ArrayLike, NDArray

from Module.module_4_finger_dp.action_chunk import (
  TeacherCommandChunk,
  build_teacher_command_chunks,
)
from Module.module_4_finger_dp.contracts import NUM_FINGERS, NUM_FINGER_JOINTS


DP_DATASET_SCHEMA_VERSION = "fr3-leap-finger-dp-dataset.v1"


def _array(value: ArrayLike, name: str, shape: tuple[int, ...]) -> NDArray[np.float64]:
  result = np.asarray(value, dtype=np.float64)
  if result.shape != shape or not np.all(np.isfinite(result)):
    raise ValueError(f"{name} must be finite with shape {shape}, got {result.shape}")
  result = np.array(result, dtype=np.float64, copy=True)
  result.setflags(write=False)
  return result


def _bool_array(value: ArrayLike, name: str, shape: tuple[int, ...]) -> NDArray[np.bool_]:
  result = np.asarray(value, dtype=np.bool_)
  if result.shape != shape:
    raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
  result = np.array(result, dtype=np.bool_, copy=True)
  result.setflags(write=False)
  return result


def _string_array(value: ArrayLike, name: str, length: int) -> NDArray[np.str_]:
  result = np.asarray(value, dtype=np.str_)
  if result.shape != (length,) or np.any(result == ""):
    raise ValueError(f"{name} must contain {length} non-empty strings")
  result = np.array(result, dtype=np.str_, copy=True)
  result.setflags(write=False)
  return result


@dataclass(frozen=True, slots=True)
class DPDatasetEpisode:
  episode_id: str
  seed: int
  surface_model_version: str
  termination_reason: str
  time_s: ArrayLike
  arm_q_meas_rad: ArrayLike
  arm_dq_meas_rad_s: ArrayLike
  arm_command_rad: ArrayLike
  q_f_meas_rad: ArrayLike
  dq_f_meas_rad_s: ArrayLike
  q_f_teacher_nominal_cmd_rad: ArrayLike
  q_f_teacher_executed_cmd_rad: ArrayLike
  force_raw_n: ArrayLike
  force_filtered_n: ArrayLike
  desired_force_n: ArrayLike
  contact_mask: ArrayLike
  force_valid_mask: ArrayLike
  contact_position_palm_m: ArrayLike
  contact_normal_palm: ArrayLike
  surface_distance_m: ArrayLike
  surface_uncertainty_m: ArrayLike
  geometry_from_contact: ArrayLike
  surface_geometry_valid: ArrayLike
  palm_pose_plan_world: ArrayLike
  wrist_mcc_offset: ArrayLike
  palm_pose_command_world: ArrayLike
  palm_pose_real_world: ArrayLike
  wrist_mcc_velocity: ArrayLike
  collision_distance_m: ArrayLike
  non_tip_contact_count: ArrayLike
  guard_state: ArrayLike
  guard_reason: ArrayLike
  authority_owner: ArrayLike
  teacher_source: ArrayLike
  repair_mask: ArrayLike
  authority_transition_reset_mask: ArrayLike
  authority_filter_solver_success: ArrayLike
  authority_filter_solver_status: ArrayLike
  authority_filter_solver_iterations: ArrayLike
  authority_filter_intervention_norm_rad: ArrayLike
  authority_filter_maximum_constraint_violation: ArrayLike
  authority_filter_latency_s: ArrayLike
  physics_dt_s: float = 0.002

  def __post_init__(self) -> None:
    for name in (
      "episode_id",
      "surface_model_version",
      "termination_reason",
    ):
      if not str(getattr(self, name)):
        raise ValueError(f"{name} must be non-empty")
    if not np.isfinite(self.physics_dt_s) or self.physics_dt_s <= 0.0:
      raise ValueError("physics_dt_s must be finite and positive")
    time = np.asarray(self.time_s, dtype=np.float64)
    if time.ndim != 1 or len(time) < 2 or not np.all(np.isfinite(time)):
      raise ValueError("time_s must be a finite vector with at least two samples")
    if np.any(np.diff(time) <= 0.0):
      raise ValueError("time_s must be strictly increasing")
    length = len(time)
    object.__setattr__(self, "time_s", _array(time, "time_s", (length,)))

    shapes = {
      "arm_q_meas_rad": (length, 7),
      "arm_dq_meas_rad_s": (length, 7),
      "arm_command_rad": (length, 7),
      "q_f_meas_rad": (length, NUM_FINGER_JOINTS),
      "dq_f_meas_rad_s": (length, NUM_FINGER_JOINTS),
      "q_f_teacher_nominal_cmd_rad": (length, NUM_FINGER_JOINTS),
      "q_f_teacher_executed_cmd_rad": (length, NUM_FINGER_JOINTS),
      "force_raw_n": (length, NUM_FINGERS),
      "force_filtered_n": (length, NUM_FINGERS),
      "desired_force_n": (length, NUM_FINGERS),
      "contact_position_palm_m": (length, NUM_FINGERS, 3),
      "contact_normal_palm": (length, NUM_FINGERS, 3),
      "surface_distance_m": (length, NUM_FINGERS),
      "surface_uncertainty_m": (length, NUM_FINGERS),
      "palm_pose_plan_world": (length, 7),
      "wrist_mcc_offset": (length, 6),
      "palm_pose_command_world": (length, 7),
      "palm_pose_real_world": (length, 7),
      "wrist_mcc_velocity": (length, 6),
      "collision_distance_m": (length,),
      "non_tip_contact_count": (length,),
      "authority_filter_intervention_norm_rad": (length,),
      "authority_filter_solver_iterations": (length,),
      "authority_filter_maximum_constraint_violation": (length,),
      "authority_filter_latency_s": (length,),
    }
    bool_shapes = {
      "contact_mask": (length, NUM_FINGERS),
      "force_valid_mask": (length, NUM_FINGERS),
      "geometry_from_contact": (length, NUM_FINGERS),
      "surface_geometry_valid": (length, NUM_FINGERS),
      "repair_mask": (length,),
      "authority_transition_reset_mask": (length, NUM_FINGERS),
      "authority_filter_solver_success": (length,),
    }
    for name, shape in shapes.items():
      object.__setattr__(self, name, _array(getattr(self, name), name, shape))
    for name, shape in bool_shapes.items():
      object.__setattr__(self, name, _bool_array(getattr(self, name), name, shape))
    for name in (
      "guard_state",
      "guard_reason",
      "authority_owner",
      "teacher_source",
      "authority_filter_solver_status",
    ):
      object.__setattr__(self, name, _string_array(getattr(self, name), name, length))

    if (
      np.any(self.force_raw_n < 0.0)
      or np.any(self.force_filtered_n < 0.0)
      or np.any(self.desired_force_n < 0.0)
    ):
      raise ValueError("force magnitudes must be non-negative")
    if np.any(self.surface_uncertainty_m < 0.0):
      raise ValueError("surface uncertainty must be non-negative")
    if np.any(self.non_tip_contact_count < 0.0):
      raise ValueError("non_tip_contact_count must be non-negative")
    if np.any(self.authority_filter_intervention_norm_rad < 0.0):
      raise ValueError("authority filter intervention norm must be non-negative")
    if np.any(self.authority_filter_maximum_constraint_violation < 0.0):
      raise ValueError("authority constraint violation must be non-negative")
    if np.any(self.authority_filter_latency_s < 0.0):
      raise ValueError("authority filter latency must be non-negative")
    if np.any(self.authority_filter_solver_iterations < 0.0):
      raise ValueError("authority filter iterations must be non-negative")
    if np.any(self.geometry_from_contact & ~self.surface_geometry_valid):
      raise ValueError("measured contact geometry must be valid")
    valid_normals = self.surface_geometry_valid
    normal_lengths = np.linalg.norm(self.contact_normal_palm, axis=2)
    if np.any(np.abs(normal_lengths[valid_normals] - 1.0) > 1e-5):
      raise ValueError("valid geometry normals must be unit length")

  @property
  def length(self) -> int:
    return len(self.time_s)

  @property
  def duration_s(self) -> float:
    return float(self.time_s[-1] - self.time_s[0] + self.physics_dt_s)

  def usable_command_mask(self) -> NDArray[np.bool_]:
    """Exclude guard-owned and invalid-sensor steps from imitation labels."""

    owner_ok = np.isin(self.authority_owner, ["TEACHER", "REPAIR_ORACLE"])
    guard_ok = np.isin(self.guard_state, ["INITIALIZE", "BUFFER_FILL", "DP_ACTIVE"])
    result = owner_ok & guard_ok & np.all(self.force_valid_mask, axis=1)
    result.setflags(write=False)
    return result

  def teacher_chunks(
    self,
    *,
    horizon: int,
    stride: int = 1,
    use_executed_command: bool = True,
  ) -> tuple[TeacherCommandChunk, ...]:
    """Build command-imitation labels, never future-state labels.

    The default target is the post-authority, pre-plant command that was
    actually issued by the successful teacher tick.  The privileged nominal
    is retained for intervention audits and ablations, but guard-owned frames
    and their crossing chunks are always excluded.
    """

    command = (
      self.q_f_teacher_executed_cmd_rad
      if use_executed_command
      else self.q_f_teacher_nominal_cmd_rad
    )
    return build_teacher_command_chunks(
      self.q_f_meas_rad,
      command,
      horizon=horizon,
      stride=stride,
      usable_mask=self.usable_command_mask(),
      teacher_source="MIXED_VERIFIED_INVERSE_AND_REPAIR",
      repair_mask=self.repair_mask,
    )


def save_dataset_episode(path: str | Path, episode: DPDatasetEpisode) -> Path:
  destination = Path(path)
  destination.parent.mkdir(parents=True, exist_ok=True)
  with h5py.File(destination, "w") as stream:
    stream.attrs["schema_version"] = DP_DATASET_SCHEMA_VERSION
    stream.attrs["episode_id"] = episode.episode_id
    stream.attrs["seed"] = episode.seed
    stream.attrs["surface_model_version"] = episode.surface_model_version
    stream.attrs["termination_reason"] = episode.termination_reason
    stream.attrs["physics_dt_s"] = episode.physics_dt_s
    string_dtype = h5py.string_dtype(encoding="utf-8")
    metadata_names = {
      "episode_id",
      "seed",
      "surface_model_version",
      "termination_reason",
      "physics_dt_s",
    }
    for definition in fields(episode):
      name = definition.name
      if name in metadata_names:
        continue
      value = getattr(episode, name)
      if value.dtype.kind in {"U", "S", "O"}:
        stream.create_dataset(name, data=np.asarray(value, dtype=object), dtype=string_dtype)
      else:
        stream.create_dataset(name, data=value, compression="gzip", shuffle=True)
  return destination


def load_dataset_episode(path: str | Path) -> DPDatasetEpisode:
  source = Path(path)
  with h5py.File(source, "r") as stream:
    if stream.attrs.get("schema_version") != DP_DATASET_SCHEMA_VERSION:
      raise ValueError("unsupported DP dataset schema")
    values: dict[str, Any] = {
      "episode_id": str(stream.attrs["episode_id"]),
      "seed": int(stream.attrs["seed"]),
      "surface_model_version": str(stream.attrs["surface_model_version"]),
      "termination_reason": str(stream.attrs["termination_reason"]),
      "physics_dt_s": float(stream.attrs["physics_dt_s"]),
    }
    for name, dataset in stream.items():
      value = dataset[...]
      if h5py.check_string_dtype(dataset.dtype) is not None:
        value = value.astype(str)
      values[name] = value
  return DPDatasetEpisode(**values)


@dataclass(frozen=True, slots=True)
class ReplayAcceptanceConfig:
  minimum_duration_s: float = 15.0
  minimum_contact_continuity: float = 0.995
  minimum_initial_contacts: int = 3
  initial_contact_stable_steps: int = 25
  maximum_contact_establishment_s: float = 3.5
  maximum_zero_contact_gap_s: float = 0.05
  maximum_force_n: float = 8.0
  minimum_collision_distance_m: float = 0.0
  maximum_non_tip_contacts: int = 0
  allow_guard_takeover: bool = False
  maximum_repair_probability: float = 0.50
  allow_authority_solver_failure: bool = False
  maximum_authority_constraint_violation: float = 1e-6
  maximum_authority_filter_p95_latency_s: float = 0.002

  def __post_init__(self) -> None:
    if not 0.0 <= self.maximum_repair_probability <= 1.0:
      raise ValueError("maximum_repair_probability must be in [0,1]")
    if not 1 <= self.minimum_initial_contacts <= NUM_FINGERS:
      raise ValueError("minimum_initial_contacts must be in [1,4]")
    if self.initial_contact_stable_steps < 1:
      raise ValueError("initial_contact_stable_steps must be positive")
    for name in (
      "minimum_duration_s",
      "maximum_contact_establishment_s",
      "maximum_zero_contact_gap_s",
      "maximum_force_n",
      "maximum_authority_constraint_violation",
      "maximum_authority_filter_p95_latency_s",
    ):
      value = float(getattr(self, name))
      if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    if not 0.0 <= self.minimum_contact_continuity <= 1.0:
      raise ValueError("minimum_contact_continuity must be in [0,1]")


@dataclass(frozen=True, slots=True)
class ReplayAudit:
  accepted: bool
  reasons: tuple[str, ...]
  duration_s: float
  overall_contact_continuity: float
  contact_continuity: float
  contact_establishment_s: float
  longest_zero_contact_gap_s: float
  maximum_force_n: float
  minimum_collision_distance_m: float
  non_tip_contact_frames: int
  guard_takeover_frames: int
  authority_solver_failure_frames: int
  maximum_authority_constraint_violation: float
  authority_filter_latency_p95_s: float
  repair_probability: float
  usable_chunk_probability: float


def audit_physical_replay(
  episode: DPDatasetEpisode,
  config: ReplayAcceptanceConfig | None = None,
) -> ReplayAudit:
  c = config or ReplayAcceptanceConfig()
  any_contact = np.any(episode.contact_mask, axis=1)
  overall_contact_continuity = float(np.mean(any_contact))
  enough_initial_contacts = (
    np.sum(episode.contact_mask, axis=1) >= c.minimum_initial_contacts
  )
  if len(enough_initial_contacts) >= c.initial_contact_stable_steps:
    stable_kernel = np.ones(c.initial_contact_stable_steps, dtype=np.int32)
    stable_windows = np.flatnonzero(
      np.convolve(enough_initial_contacts.astype(np.int32), stable_kernel, mode="valid")
      == c.initial_contact_stable_steps
    )
  else:
    stable_windows = np.zeros(0, dtype=np.int64)
  establishment_index = int(stable_windows[0]) if len(stable_windows) else -1
  contact_establishment_s = (
    float(episode.time_s[establishment_index]) if establishment_index >= 0 else -1.0
  )
  evaluation_contact = (
    any_contact[establishment_index:] if establishment_index >= 0 else any_contact
  )
  contact_continuity = float(np.mean(evaluation_contact))
  zero_contact = ~evaluation_contact
  zero_starts = np.flatnonzero(zero_contact & np.r_[True, ~zero_contact[:-1]])
  zero_ends = np.flatnonzero(zero_contact & np.r_[~zero_contact[1:], True])
  longest_zero_gap = max(
    (
      (int(end) - int(start) + 1) * episode.physics_dt_s
      for start, end in zip(zero_starts, zero_ends)
    ),
    default=0.0,
  )
  maximum_force = float(np.max(episode.force_raw_n))
  minimum_distance = float(np.min(episode.collision_distance_m))
  non_tip_frames = int(np.count_nonzero(episode.non_tip_contact_count > c.maximum_non_tip_contacts))
  guard_frames = int(
    np.count_nonzero(
      np.isin(
        episode.guard_state,
        ["SOFT_RECOVERY", "HARD_RELEASE", "SAFE_HOLD", "BUFFER_RESET", "ABORTED"],
      )
    )
  )
  solver_failure_frames = int(np.count_nonzero(~episode.authority_filter_solver_success))
  maximum_authority_violation = float(
    np.max(episode.authority_filter_maximum_constraint_violation)
  )
  authority_latency_p95 = float(np.quantile(episode.authority_filter_latency_s, 0.95))
  repair_probability = float(np.mean(episode.repair_mask))
  usable_probability = float(np.mean(episode.usable_command_mask()))
  reasons: list[str] = []
  if episode.duration_s < c.minimum_duration_s:
    reasons.append("DURATION_TOO_SHORT")
  if establishment_index < 0:
    reasons.append("CONTACT_NOT_ESTABLISHED")
  elif contact_establishment_s > c.maximum_contact_establishment_s:
    reasons.append("CONTACT_ESTABLISHMENT_LATE")
  if contact_continuity < c.minimum_contact_continuity:
    reasons.append("CONTACT_CONTINUITY")
  if longest_zero_gap > c.maximum_zero_contact_gap_s:
    reasons.append("ZERO_CONTACT_GAP")
  if maximum_force > c.maximum_force_n:
    reasons.append("TIP_OVERFORCE")
  if minimum_distance < c.minimum_collision_distance_m:
    reasons.append("COLLISION_DISTANCE")
  if non_tip_frames:
    reasons.append("NON_TIP_CONTACT")
  if guard_frames and not c.allow_guard_takeover:
    reasons.append("GUARD_TAKEOVER")
  if solver_failure_frames and not c.allow_authority_solver_failure:
    reasons.append("AUTHORITY_FILTER_FAILURE")
  if maximum_authority_violation > c.maximum_authority_constraint_violation:
    reasons.append("AUTHORITY_CONSTRAINT_VIOLATION")
  if authority_latency_p95 > c.maximum_authority_filter_p95_latency_s:
    reasons.append("AUTHORITY_FILTER_DEADLINE")
  if repair_probability > c.maximum_repair_probability:
    reasons.append("REPAIR_DOMINATED")
  if not np.all(episode.force_valid_mask):
    reasons.append("FORCE_SENSOR_INVALID")
  return ReplayAudit(
    accepted=not reasons,
    reasons=tuple(reasons),
    duration_s=episode.duration_s,
    overall_contact_continuity=overall_contact_continuity,
    contact_continuity=contact_continuity,
    contact_establishment_s=contact_establishment_s,
    longest_zero_contact_gap_s=float(longest_zero_gap),
    maximum_force_n=maximum_force,
    minimum_collision_distance_m=minimum_distance,
    non_tip_contact_frames=non_tip_frames,
    guard_takeover_frames=guard_frames,
    authority_solver_failure_frames=solver_failure_frames,
    maximum_authority_constraint_violation=maximum_authority_violation,
    authority_filter_latency_p95_s=authority_latency_p95,
    repair_probability=repair_probability,
    usable_chunk_probability=usable_probability,
  )
