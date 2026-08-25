"""Long, fail-closed Dataset-D collection for GPU Finger-DP diagnostics.

Only complete trajectories that establish contact and then preserve at least
one real fingertip contact for ten seconds or longer enter the training pool.
Rejected episodes remain available as diagnostics but contribute zero samples.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from Module.module_4_finger_dp.track_d_dataset import (
  TrackDCausalAudit,
  TrackDSampleConfig,
  TrackDSamples,
  build_track_d_samples,
  save_track_d_samples,
)
from Module.module_4_whole_hand_mcc.runner import (
  E05MCCConfig,
  E05MCCTrace,
  run_e05_mcc,
)


LONG_DATASET_SCHEMA_VERSION = "fr3-leap-long-dataset-d.v2"


@dataclass(frozen=True, slots=True)
class LongTrajectorySpec:
  episode_id: str
  split: str
  seed: int
  duration_s: float = 12.0
  desired_force_n: float = 2.0
  traversal_y_m: float = 0.060
  lateral_primary_amplitude_m: float = 0.008
  lateral_secondary_amplitude_m: float = 0.003
  friction_coefficient: float = 0.90
  force_noise_std_n: float = 0.0
  initial_joint_noise_std_rad: float = 0.0
  wrist_surface_following: bool = True

  def __post_init__(self) -> None:
    if not self.episode_id:
      raise ValueError("episode_id must be non-empty")
    if self.split not in {"TRAIN", "EVAL"}:
      raise ValueError("split must be TRAIN or EVAL")
    if self.duration_s < 10.0:
      raise ValueError("every long trajectory must be at least 10 seconds")
    if not self.wrist_surface_following:
      raise ValueError("long v2 trajectories require surface-following wrist plans")

  def mcc_config(self) -> E05MCCConfig:
    return E05MCCConfig(
      mode="E05-H-MCC",
      surface="extreme",
      duration_s=self.duration_s,
      settling_time_s=1.0,
      desired_force_n=self.desired_force_n,
      traversal_y_m=self.traversal_y_m,
      lateral_primary_amplitude_m=self.lateral_primary_amplitude_m,
      lateral_secondary_amplitude_m=self.lateral_secondary_amplitude_m,
      # No artificial pose step is used for training collection.  Long-term
      # contact must be provided by the physical controller, not repaired by
      # selecting a short interval around a deliberately injected failure.
      pose_step_time_s=0.5 * self.duration_s,
      pose_step_m=0.0,
      friction_coefficient=self.friction_coefficient,
      force_noise_std_n=self.force_noise_std_n,
      initial_joint_noise_std_rad=self.initial_joint_noise_std_rad,
      wrist_surface_following=self.wrist_surface_following,
      seed=self.seed,
    )


@dataclass(frozen=True, slots=True)
class LongTrajectoryGateConfig:
  minimum_episode_duration_s: float = 10.0
  minimum_training_contact_duration_s: float = 10.0
  minimum_initial_contacts: int = 3
  contact_confirmation_steps: int = 25
  maximum_force_n: float = 8.0
  maximum_non_tip_contact_frames: int = 0
  allowed_guard_reason: str = "NONE"


@dataclass(frozen=True, slots=True)
class LongTrajectoryAudit:
  episode_id: str
  split: str
  accepted_for_training: bool
  classification: str
  reasons: tuple[str, ...]
  episode_duration_s: float
  contact_establishment_s: float
  training_start_s: float
  training_stop_s: float
  training_contact_duration_s: float
  training_contact_continuity: float
  training_zero_contact_time_s: float
  average_training_contact_count: float
  minimum_training_contact_count: int
  maximum_force_n: float
  force_p95_n: float
  non_tip_contact_frames: int
  guard_violation_frames: int


@dataclass(frozen=True, slots=True)
class LongTrajectoryRecord:
  spec: LongTrajectorySpec
  trace_path: Path
  audit: LongTrajectoryAudit


def _first_stable_contact_start(
  contact_count: NDArray[np.int64],
  minimum_contacts: int,
  confirmation_steps: int,
) -> int:
  enough = contact_count >= minimum_contacts
  if len(enough) < confirmation_steps:
    return -1
  stable = np.convolve(
    enough.astype(np.int32),
    np.ones(confirmation_steps, dtype=np.int32),
    mode="valid",
  )
  candidates = np.flatnonzero(stable == confirmation_steps)
  return int(candidates[0]) if len(candidates) else -1


def audit_long_trajectory(
  trace: E05MCCTrace,
  spec: LongTrajectorySpec,
  config: LongTrajectoryGateConfig = LongTrajectoryGateConfig(),
) -> LongTrajectoryAudit:
  dt_s = float(np.median(np.diff(trace.time_s)))
  duration_s = float(len(trace.time_s) * dt_s)
  contact_count = np.sum(trace.actual_contacts, axis=1).astype(np.int64)
  start = _first_stable_contact_start(
    contact_count,
    config.minimum_initial_contacts,
    config.contact_confirmation_steps,
  )
  if start >= 0:
    training_contact = contact_count[start:]
    training_forces = trace.fingertip_forces_n[start:]
    training_duration_s = float(len(training_contact) * dt_s)
    continuity = float(np.mean(training_contact >= 1))
    zero_time_s = float(np.count_nonzero(training_contact == 0) * dt_s)
    average_contacts = float(np.mean(training_contact))
    minimum_contacts = int(np.min(training_contact))
    start_s = float(trace.time_s[start])
  else:
    training_contact = np.zeros(0, dtype=np.int64)
    training_forces = np.zeros((0, 4), dtype=np.float64)
    training_duration_s = 0.0
    continuity = 0.0
    zero_time_s = duration_s
    average_contacts = 0.0
    minimum_contacts = 0
    start_s = -1.0
  non_tip_frames = int(np.count_nonzero(trace.non_tip_contact_count > 0))
  guard_frames = int(
    np.count_nonzero(trace.guard_reason != config.allowed_guard_reason)
  )
  maximum_force = float(np.max(trace.fingertip_forces_n))
  force_p95 = (
    float(np.quantile(training_forces, 0.95))
    if len(training_forces)
    else 0.0
  )
  reasons: list[str] = []
  if duration_s < config.minimum_episode_duration_s:
    reasons.append("EPISODE_SHORTER_THAN_10S")
  if start < 0:
    reasons.append("CONTACT_NOT_ESTABLISHED")
  if training_duration_s < config.minimum_training_contact_duration_s:
    reasons.append("CONTACT_MAINTAINED_FOR_LESS_THAN_10S")
  if continuity < 1.0:
    reasons.append("ANY_FINGERTIP_CONTACT_LOSS")
  if maximum_force >= config.maximum_force_n:
    reasons.append("TIP_OVERFORCE")
  if non_tip_frames > config.maximum_non_tip_contact_frames:
    reasons.append("NON_TIP_CONTACT")
  if guard_frames:
    reasons.append("GUARD_VIOLATION")
  accepted = not reasons
  return LongTrajectoryAudit(
    episode_id=spec.episode_id,
    split=spec.split,
    accepted_for_training=accepted and spec.split == "TRAIN",
    classification=(
      "ACCEPTED_TRAIN"
      if accepted and spec.split == "TRAIN"
      else "ACCEPTED_EVAL"
      if accepted
      else "REJECTED_DIAGNOSTIC"
    ),
    reasons=tuple(reasons) if reasons else ("NONE",),
    episode_duration_s=duration_s,
    contact_establishment_s=start_s,
    training_start_s=start_s,
    training_stop_s=float(trace.time_s[-1]),
    training_contact_duration_s=training_duration_s,
    training_contact_continuity=continuity,
    training_zero_contact_time_s=zero_time_s,
    average_training_contact_count=average_contacts,
    minimum_training_contact_count=minimum_contacts,
    maximum_force_n=maximum_force,
    force_p95_n=force_p95,
    non_tip_contact_frames=non_tip_frames,
    guard_violation_frames=guard_frames,
  )


def save_long_trace(path: str | Path, trace: E05MCCTrace) -> Path:
  destination = Path(path)
  destination.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(
    destination,
    **{definition.name: getattr(trace, definition.name) for definition in fields(trace)},
  )
  return destination


def load_long_trace(path: str | Path) -> E05MCCTrace:
  with np.load(Path(path), allow_pickle=False) as archive:
    return E05MCCTrace(
      **{definition.name: archive[definition.name] for definition in fields(E05MCCTrace)}
    )


def collect_long_trajectory(
  spec: LongTrajectorySpec,
  output_directory: str | Path,
  gate: LongTrajectoryGateConfig = LongTrajectoryGateConfig(),
) -> LongTrajectoryRecord:
  output = Path(output_directory)
  output.mkdir(parents=True, exist_ok=True)
  trace, _ = run_e05_mcc(spec.mcc_config())
  trace_path = save_long_trace(output / f"{spec.episode_id}.npz", trace)
  audit = audit_long_trajectory(trace, spec, gate)
  (output / f"{spec.episode_id}.json").write_text(
    json.dumps(
      {
        "schema_version": LONG_DATASET_SCHEMA_VERSION,
        "dataset_class": "DATASET_D_DIAGNOSTIC",
        "formal_dataset_i": False,
        "source_controller": "E05_H_MCC",
        "spec": asdict(spec),
        "audit": asdict(audit),
      },
      indent=2,
      sort_keys=True,
    ),
    encoding="utf-8",
  )
  return LongTrajectoryRecord(spec=spec, trace_path=trace_path, audit=audit)


def _combine_samples(samples: list[TrackDSamples]) -> TrackDSamples:
  if not samples:
    raise RuntimeError("no accepted long trajectories are available for training")
  inputs = {
    name: np.concatenate([sample.inputs[name] for sample in samples], axis=0)
    for name in samples[0].inputs
  }
  counts = [sample.count for sample in samples]
  source_indices = np.concatenate(
    [sample.source_raw_index + 10_000_000 * index for index, sample in enumerate(samples)]
  )
  timestamps = np.concatenate(
    [sample.timestamp_s + 100.0 * index for index, sample in enumerate(samples)]
  )
  audits = [sample.audit for sample in samples]
  audit = TrackDCausalAudit(
    passed=all(value.passed for value in audits),
    reasons=tuple(
      sorted({reason for value in audits for reason in value.reasons})
    ),
    sample_count=sum(counts),
    teacher_source="COMPLIANT_LONG_E05_H_MCC_DATASET_D",
    source_start_time_s=min(value.source_start_time_s for value in audits),
    source_stop_time_s=max(value.source_stop_time_s for value in audits),
    physics_rate_hz=audits[0].physics_rate_hz,
    force_history_rate_hz=audits[0].force_history_rate_hz,
    policy_rate_hz=audits[0].policy_rate_hz,
    force_history_duration_s=audits[0].force_history_duration_s,
    action_horizon_duration_s=audits[0].action_horizon_duration_s,
    maximum_history_timestamp_minus_anchor_s=max(
      value.maximum_history_timestamp_minus_anchor_s for value in audits
    ),
    minimum_target_timestamp_minus_anchor_s=min(
      value.minimum_target_timestamp_minus_anchor_s for value in audits
    ),
    maximum_target_timestamp_minus_anchor_s=max(
      value.maximum_target_timestamp_minus_anchor_s for value in audits
    ),
    future_leakage_count=sum(value.future_leakage_count for value in audits),
    nonfinite_value_count=sum(value.nonfinite_value_count for value in audits),
    maximum_force_n=max(value.maximum_force_n for value in audits),
    contact_continuity=float(
      np.average(
        [value.contact_continuity for value in audits],
        weights=counts,
      )
    ),
    teacher_command_vs_measured_rmse_rad=float(
      np.average(
        [value.teacher_command_vs_measured_rmse_rad for value in audits],
        weights=counts,
      )
    ),
    maximum_anchor_construction_residual_rad=max(
      value.maximum_anchor_construction_residual_rad for value in audits
    ),
  )
  return TrackDSamples(
    inputs=inputs,
    target_action_offsets_rad=np.concatenate(
      [sample.target_action_offsets_rad for sample in samples], axis=0
    ),
    anchor_q_meas_rad=np.concatenate(
      [sample.anchor_q_meas_rad for sample in samples], axis=0
    ),
    future_teacher_command_rad=np.concatenate(
      [sample.future_teacher_command_rad for sample in samples], axis=0
    ),
    source_raw_index=source_indices,
    timestamp_s=timestamps,
    config=replace(
      samples[0].config,
      teacher_source="COMPLIANT_LONG_E05_H_MCC_DATASET_D",
    ),
    audit=audit,
  )


def build_compliant_long_training_set(
  records: Iterable[LongTrajectoryRecord],
  output_path: str | Path,
) -> TrackDSamples:
  record_list = list(records)
  accepted = [
    record for record in record_list if record.audit.accepted_for_training
  ]
  rejected = [
    record for record in record_list if not record.audit.accepted_for_training
  ]
  samples: list[TrackDSamples] = []
  ranges: list[dict[str, object]] = []
  cursor = 0
  for record in accepted:
    trace = load_long_trace(record.trace_path)
    # Use only the post-establishment, fully compliant part.  The final 0.4 s
    # remains available as the future action horizon and is not an input leak.
    start_s = max(0.40, record.audit.training_start_s + 0.20)
    stop_s = record.audit.training_stop_s - 0.40
    episode_samples = build_track_d_samples(
      trace,
      TrackDSampleConfig(
        start_time_s=start_s,
        stop_time_s=stop_s,
        desired_force_n=record.spec.desired_force_n,
        teacher_source=f"{record.spec.episode_id}:E05_H_MCC_DATASET_D",
      ),
    )
    samples.append(episode_samples)
    ranges.append(
      {
        "episode_id": record.spec.episode_id,
        "sample_start": cursor,
        "sample_stop_exclusive": cursor + episode_samples.count,
        "sample_count": episode_samples.count,
      }
    )
    cursor += episode_samples.count
  combined = _combine_samples(samples)
  destination = save_track_d_samples(output_path, combined)
  manifest = {
    "schema_version": LONG_DATASET_SCHEMA_VERSION,
    "dataset_class": "DATASET_D_DIAGNOSTIC",
    "formal_dataset_i": False,
    "gpu_training_authorized": True,
    "minimum_trajectory_duration_s": 10.0,
    "accepted_episode_count": len(accepted),
    "rejected_episode_count": len(rejected),
    "accepted_sample_count": combined.count,
    "accepted_episode_ranges": ranges,
    "accepted_episode_ids": [record.spec.episode_id for record in accepted],
    "rejected_episode_ids": [record.spec.episode_id for record in rejected],
    "episode_audits": [asdict(record.audit) for record in record_list],
  }
  destination.with_name("long_dataset_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True),
    encoding="utf-8",
  )
  return combined
