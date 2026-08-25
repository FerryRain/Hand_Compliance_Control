"""Formal Dataset-I generation, classification, split and causal sampling."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from Module.module_4_finger_dp.dataset_i_oracle import (
  DATASET_I_FORWARD_SOURCE,
  DATASET_I_ORACLE_VERSION,
  DatasetIForwardConfig,
  DatasetIIGate,
  DatasetIReplayAux,
  OracleSolveTrace,
  run_dataset_i_raw_pair,
  save_i_gate_summary,
)
from Module.module_4_finger_dp.spatial_inverse_data import (
  SpatialInverseAudit,
  SpatialInversePhysicalPair,
  save_spatial_inverse_pair,
)
from Module.module_4_finger_dp.track_d_dataset import (
  TRACK_D_INPUT_NAMES,
  TrackDCausalAudit,
  TrackDSampleConfig,
  TrackDSamples,
  build_track_d_samples,
)
from Module.module_4_whole_hand_mcc.runner import E05MCCTrace


DATASET_I_EPISODE_VERSION = "fr3-leap-dataset-i-episode.v1"
DATASET_I_SAMPLE_VERSION = "fr3-leap-dataset-i-samples.v1"
DATASET_I_SPLIT_VERSION = "fr3-leap-dataset-i-object-split.v1"


@dataclass(frozen=True, slots=True)
class DatasetIObjectRegion:
  object_id: str
  split: str
  offset_x_m: float
  offset_y_m: float
  offset_z_m: float


# These sets are frozen before formal collection.  They are translated crops
# of the same 0.60 x 0.84 m extreme hfield, not claims about distinct YCB IDs.
OBJECT_REGIONS_V1 = (
  DatasetIObjectRegion("dev_crop_0", "development", 0.000, 0.000, 0.0000000),
  DatasetIObjectRegion("dev_crop_1", "development", 0.026, 0.018, 0.0017589),
  DatasetIObjectRegion("dev_crop_2", "development", -0.028, 0.020, 0.0021404),
  DatasetIObjectRegion("dev_crop_3", "development", 0.015, 0.030, 0.0007454),
  DatasetIObjectRegion("dev_crop_4", "development", -0.030, 0.035, 0.0027811),
  DatasetIObjectRegion("train_crop_0", "train", 0.010, 0.050, 0.0024355),
  DatasetIObjectRegion("train_crop_1", "train", -0.025, 0.030, 0.0033674),
  DatasetIObjectRegion("train_crop_2", "train", 0.018, 0.050, 0.0020947),
  DatasetIObjectRegion("validation_crop_0", "validation", 0.020, 0.055, 0.0029006),
  DatasetIObjectRegion("test_crop_0", "test", -0.040, 0.025, 0.0013680),
)


@dataclass(frozen=True, slots=True)
class DatasetIEpisodeResult:
  episode_id: str
  object_id: str
  split: str
  classification: str
  episode_directory: Path
  pair_path: Path | None
  samples_path: Path | None
  gate: DatasetIIGate | None
  audit: SpatialInverseAudit | None
  sample_count: int


@dataclass(frozen=True, slots=True)
class DatasetISampleBundle:
  samples: TrackDSamples
  episode_id: NDArray[np.str_]
  object_id: NDArray[np.str_]
  split: NDArray[np.str_]

  def __post_init__(self) -> None:
    count = self.samples.count
    for name in ("episode_id", "object_id", "split"):
      value = np.asarray(getattr(self, name), dtype=np.str_)
      if value.shape != (count,):
        raise ValueError(f"{name} must have shape ({count},)")
      copied = np.array(value, copy=True)
      copied.setflags(write=False)
      object.__setattr__(self, name, copied)


def _as_e05_trace(
  pair: SpatialInversePhysicalPair,
  aux: DatasetIReplayAux,
  config: DatasetIForwardConfig,
) -> E05MCCTrace:
  replay = pair.replay
  count = replay.length
  normals = np.array(replay.contact_normal_world, copy=True)
  # Free-finger rows are filled by the causal OracleSurfaceModel inside the
  # sample builder; these placeholders are not deployment observations.
  return E05MCCTrace(
    time_s=np.array(replay.time_s, copy=True),
    arm_q_rad=np.array(replay.arm_q_meas_rad, copy=True),
    arm_dq_rad_s=np.array(replay.arm_dq_meas_rad_s, copy=True),
    arm_command_rad=np.array(replay.arm_command_rad, copy=True),
    finger_q_rad=np.array(replay.q_f_meas_rad, copy=True),
    finger_dq_rad_s=np.array(replay.dq_f_meas_rad_s, copy=True),
    finger_command_rad=np.array(replay.q_f_command_rad, copy=True),
    palm_pose_world=np.array(replay.palm_pose_real_world, copy=True),
    planned_palm_pose_world=np.array(replay.palm_pose_plan_world, copy=True),
    commanded_palm_pose_world=np.array(replay.palm_pose_plan_world, copy=True),
    fingertip_positions_world_m=np.array(replay.fingertip_position_world_m, copy=True),
    pad_normals_world=normals,
    contact_positions_world_m=np.array(replay.contact_position_world_m, copy=True),
    contact_normals_world=normals,
    fingertip_forces_n=np.array(replay.contact_force_n, copy=True),
    actual_contacts=np.array(replay.contact_mask, copy=True),
    desired_hand_wrench_world=np.array(aux.desired_hand_wrench_world, copy=True),
    estimated_hand_wrench_world=np.array(aux.estimated_hand_wrench_world, copy=True),
    contact_hand_wrench_world=np.zeros((count, 6)),
    arm_external_torque_nm=np.zeros((count, 7)),
    wrist_compliance_offset=np.array(aux.wrist_mcc_offset, copy=True),
    finger_compliance_offsets_m=np.zeros((count, 4)),
    coordinator_rank=np.zeros(count, dtype=np.int32),
    coordinator_condition=np.zeros(count),
    coordinator_internal_leakage_n=np.zeros(count),
    surface_curvature_inv_m=np.zeros((count, 4)),
    disturbance_active=np.asarray(
      replay.time_s >= config.pose_step_time_s,
      dtype=np.bool_,
    ) & bool(abs(config.object_pose_step_z_m) > 0.0),
    controller_latency_s=np.zeros(count),
    physics_step_latency_s=np.zeros(count),
    loop_latency_s=np.zeros(count),
    guard_reason=np.full(count, "NONE", dtype="U40"),
    non_tip_contact_count=np.asarray(replay.non_tip_contact_count, dtype=np.int32),
  )


def build_dataset_i_episode_samples(
  pair: SpatialInversePhysicalPair,
  aux: DatasetIReplayAux,
  config: DatasetIForwardConfig,
) -> TrackDSamples:
  if pair.forward_provenance != DATASET_I_FORWARD_SOURCE:
    raise ValueError("formal Dataset-I refuses non-oracle forward provenance")
  if pair.replay_repair_policy != "NONE":
    raise ValueError("formal RAW_VERIFIED samples require zero replay repair")
  trace = _as_e05_trace(pair, aux, config)
  return build_track_d_samples(
    trace,
    TrackDSampleConfig(
      physics_dt_s=config.dt_s,
      policy_period_steps=config.policy_period_steps,
      force_history_period_steps=5,
      start_time_s=0.40,
      stop_time_s=config.duration_s - 0.42,
      desired_force_n=config.desired_force_n,
      surface=config.surface,
      teacher_source=DATASET_I_FORWARD_SOURCE,
    ),
    object_position_series_world_m=pair.replay.object_pose_world[:, :3],
    wrist_mcc_velocity_override=aux.wrist_mcc_velocity,
  )


def save_dataset_i_samples(
  path: str | Path,
  bundle: DatasetISampleBundle,
  *,
  metadata: Mapping[str, Any] | None = None,
) -> Path:
  destination = Path(path)
  destination.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(
    destination,
    **bundle.samples.inputs,
    target_action_offsets_rad=bundle.samples.target_action_offsets_rad,
    anchor_q_meas_rad=bundle.samples.anchor_q_meas_rad,
    future_teacher_command_rad=bundle.samples.future_teacher_command_rad,
    source_raw_index=bundle.samples.source_raw_index,
    timestamp_s=bundle.samples.timestamp_s,
    episode_id=bundle.episode_id,
    object_id=bundle.object_id,
    split=bundle.split,
  )
  payload = {
    "schema_version": DATASET_I_SAMPLE_VERSION,
    "dataset_class": "DATASET_I_RAW_VERIFIED",
    "forward_oracle_version": DATASET_I_ORACLE_VERSION,
    "forward_provenance": DATASET_I_FORWARD_SOURCE,
    "replay_repair_policy": "NONE",
    "formal_training_authorized": True,
    "sample_config": asdict(bundle.samples.config),
    "causal_audit": asdict(bundle.samples.audit),
    "sample_count": bundle.samples.count,
    "episode_count": len(np.unique(bundle.episode_id)),
    "object_ids": sorted(np.unique(bundle.object_id).tolist()),
    "splits": sorted(np.unique(bundle.split).tolist()),
    **dict(metadata or {}),
  }
  destination.with_suffix(".json").write_text(
    json.dumps(payload, indent=2, sort_keys=True),
    encoding="utf-8",
  )
  return destination


def load_dataset_i_samples(path: str | Path) -> DatasetISampleBundle:
  source = Path(path)
  metadata = json.loads(source.with_suffix(".json").read_text(encoding="utf-8"))
  if metadata.get("schema_version") != DATASET_I_SAMPLE_VERSION:
    raise ValueError("unsupported Dataset-I sample schema")
  if metadata.get("dataset_class") != "DATASET_I_RAW_VERIFIED":
    raise ValueError("formal loader refuses ambiguous dataset provenance")
  if metadata.get("replay_repair_policy") != "NONE":
    raise ValueError("formal loader refuses replay-repaired samples")
  with np.load(source, allow_pickle=False) as archive:
    samples = TrackDSamples(
      inputs={name: archive[name] for name in TRACK_D_INPUT_NAMES},
      target_action_offsets_rad=archive["target_action_offsets_rad"],
      anchor_q_meas_rad=archive["anchor_q_meas_rad"],
      future_teacher_command_rad=archive["future_teacher_command_rad"],
      source_raw_index=archive["source_raw_index"],
      timestamp_s=archive["timestamp_s"],
      config=TrackDSampleConfig(**metadata["sample_config"]),
      audit=TrackDCausalAudit(**metadata["causal_audit"]),
    )
    return DatasetISampleBundle(
      samples=samples,
      episode_id=archive["episode_id"],
      object_id=archive["object_id"],
      split=archive["split"],
    )


def _single_bundle(
  samples: TrackDSamples,
  episode_id: str,
  object_id: str,
  split: str,
) -> DatasetISampleBundle:
  return DatasetISampleBundle(
    samples=samples,
    episode_id=np.full(samples.count, episode_id),
    object_id=np.full(samples.count, object_id),
    split=np.full(samples.count, split),
  )


def concatenate_dataset_i_bundles(
  bundles: Sequence[DatasetISampleBundle],
  *,
  split: str,
) -> DatasetISampleBundle:
  selected = [bundle for bundle in bundles if np.all(bundle.split == split)]
  if not selected:
    raise ValueError(f"no Dataset-I bundles for split {split}")
  inputs = {
    name: np.concatenate([bundle.samples.inputs[name] for bundle in selected], axis=0)
    for name in TRACK_D_INPUT_NAMES
  }
  target = np.concatenate(
    [bundle.samples.target_action_offsets_rad for bundle in selected], axis=0
  )
  anchor = np.concatenate([bundle.samples.anchor_q_meas_rad for bundle in selected], axis=0)
  future = np.concatenate(
    [bundle.samples.future_teacher_command_rad for bundle in selected], axis=0
  )
  timestamps = np.concatenate([bundle.samples.timestamp_s for bundle in selected])
  sample_count = len(timestamps)
  audits = [bundle.samples.audit for bundle in selected]
  config = TrackDSampleConfig(
    **{
      **asdict(selected[0].samples.config),
      "teacher_source": DATASET_I_FORWARD_SOURCE,
    }
  )
  audit = TrackDCausalAudit(
    passed=all(value.passed for value in audits),
    reasons=tuple(sorted({reason for value in audits for reason in value.reasons})),
    sample_count=sample_count,
    teacher_source=DATASET_I_FORWARD_SOURCE,
    source_start_time_s=float(min(value.source_start_time_s for value in audits)),
    source_stop_time_s=float(max(value.source_stop_time_s for value in audits)),
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
        weights=[value.sample_count for value in audits],
      )
    ),
    teacher_command_vs_measured_rmse_rad=float(
      np.average(
        [value.teacher_command_vs_measured_rmse_rad for value in audits],
        weights=[value.sample_count for value in audits],
      )
    ),
    maximum_anchor_construction_residual_rad=max(
      value.maximum_anchor_construction_residual_rad for value in audits
    ),
  )
  samples = TrackDSamples(
    inputs=inputs,
    target_action_offsets_rad=target,
    anchor_q_meas_rad=anchor,
    future_teacher_command_rad=future,
    source_raw_index=np.arange(sample_count, dtype=np.int64),
    timestamp_s=timestamps,
    config=config,
    audit=audit,
  )
  return DatasetISampleBundle(
    samples=samples,
    episode_id=np.concatenate([bundle.episode_id for bundle in selected]),
    object_id=np.concatenate([bundle.object_id for bundle in selected]),
    split=np.concatenate([bundle.split for bundle in selected]),
  )


def save_object_split_manifest(path: str | Path) -> Path:
  destination = Path(path)
  destination.parent.mkdir(parents=True, exist_ok=True)
  payload = {
    "schema_version": DATASET_I_SPLIT_VERSION,
    "frozen_before_formal_generation": True,
    "surface_family": "extreme_hfield_translated_crops",
    "object_region_definition": [asdict(value) for value in OBJECT_REGIONS_V1],
    "frame_split_forbidden": True,
    "same_object_across_splits_forbidden": True,
  }
  destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
  return destination


def pilot_configs() -> tuple[tuple[str, str, DatasetIForwardConfig], ...]:
  configs: list[tuple[str, str, DatasetIForwardConfig]] = []
  dev = [value for value in OBJECT_REGIONS_V1 if value.split == "development"]
  for object_index, region in enumerate(dev):
    for motion_index in range(4):
      seed = 1000 + 17 * object_index + motion_index
      configs.append(
        (
          f"pilot_{region.object_id}_motion{motion_index}",
          region.split,
          DatasetIForwardConfig(
            object_id=region.object_id,
            terrain_offset_x_m=region.offset_x_m,
            terrain_offset_y_m=region.offset_y_m,
            terrain_offset_z_m=region.offset_z_m,
            seed=seed,
            phase_rad=0.45 * motion_index,
            object_traversal_y_m=-(0.030 + 0.002 * motion_index),
            object_lateral_primary_m=0.0022 + 0.0002 * motion_index,
            object_lateral_secondary_m=0.0006 + 0.0001 * motion_index,
            friction_coefficient=(0.82, 0.90, 1.00, 0.90)[motion_index],
          ),
        )
      )
  return tuple(configs)


def formal_configs(
  *,
  train_episodes_per_object: int = 7,
  validation_episodes_per_object: int = 4,
  test_episodes_per_object: int = 4,
) -> tuple[tuple[str, str, DatasetIForwardConfig], ...]:
  counts = {
    "train": train_episodes_per_object,
    "validation": validation_episodes_per_object,
    "test": test_episodes_per_object,
  }
  configs: list[tuple[str, str, DatasetIForwardConfig]] = []
  for object_index, region in enumerate(OBJECT_REGIONS_V1):
    count = counts.get(region.split, 0)
    for motion_index in range(count):
      seed = 5000 + 101 * object_index + motion_index
      fraction = motion_index / max(count - 1, 1)
      configs.append(
        (
          f"formal_{region.object_id}_motion{motion_index:03d}",
          region.split,
          DatasetIForwardConfig(
            object_id=region.object_id,
            terrain_offset_x_m=region.offset_x_m,
            terrain_offset_y_m=region.offset_y_m,
            terrain_offset_z_m=region.offset_z_m,
            seed=seed,
            phase_rad=2.0 * np.pi * fraction,
            object_traversal_y_m=-(0.030 + 0.004 * fraction),
            object_lateral_primary_m=0.0021 + 0.0006 * fraction,
            object_lateral_secondary_m=0.0006 + 0.0005 * (1.0 - fraction),
            friction_coefficient=0.80 + 0.20 * fraction,
          ),
        )
      )
  return tuple(configs)


def scaling_train_configs(
  episodes_per_object: int = 50,
) -> tuple[tuple[str, str, DatasetIForwardConfig], ...]:
  """Additional train-only candidates for the nested D100 scaling pool.

  The generator/controller is unchanged.  Motion diversity adds small abrupt
  normal disturbances on two fifths of candidates so accepted demonstrations
  contain physical contact recovery without exposing validation/test crops.
  """

  configs: list[tuple[str, str, DatasetIForwardConfig]] = []
  train_regions = [value for value in OBJECT_REGIONS_V1 if value.split == "train"]
  for object_index, region in enumerate(train_regions):
    for motion_index in range(episodes_per_object):
      phase_fraction = ((17 * motion_index + 5 * object_index) % 53) / 53.0
      cycle = motion_index % 5
      pose_step = (-0.0005, -0.0010, 0.0, 0.0, 0.0)[cycle]
      configs.append(
        (
          f"scale_{region.object_id}_motion{motion_index:03d}",
          "train",
          DatasetIForwardConfig(
            object_id=region.object_id,
            terrain_offset_x_m=region.offset_x_m,
            terrain_offset_y_m=region.offset_y_m,
            terrain_offset_z_m=region.offset_z_m,
            seed=9000 + 211 * object_index + motion_index,
            phase_rad=2.0 * np.pi * phase_fraction,
            object_traversal_y_m=-(0.028 + 0.008 * ((motion_index % 9) / 8.0)),
            object_lateral_primary_m=0.0018 + 0.0012 * ((motion_index % 7) / 6.0),
            object_lateral_secondary_m=0.0005 + 0.0006 * ((motion_index % 6) / 5.0),
            pose_step_time_s=7.0 + 1.5 * ((motion_index % 4) / 3.0),
            object_pose_step_z_m=pose_step,
            friction_coefficient=0.75 + 0.30 * ((motion_index % 8) / 7.0),
          ),
        )
      )
  return tuple(configs)


def _save_episode_aux(path: Path, solve: OracleSolveTrace, aux: DatasetIReplayAux) -> None:
  np.savez_compressed(
    path,
    oracle_timestamp_s=solve.timestamp_s,
    oracle_latency_s=solve.latency_s,
    oracle_horizon_contact_rmse_m=solve.horizon_contact_rmse_m,
    oracle_maximum_joint_step_rad=solve.maximum_joint_step_rad,
    wrist_mcc_offset=aux.wrist_mcc_offset,
    wrist_mcc_velocity=aux.wrist_mcc_velocity,
    desired_hand_wrench_world=aux.desired_hand_wrench_world,
    estimated_hand_wrench_world=aux.estimated_hand_wrench_world,
  )


def generate_dataset_i_episode(
  output_root: str | Path,
  episode_id: str,
  split: str,
  config: DatasetIForwardConfig,
  *,
  retain_pair: bool = True,
) -> DatasetIEpisodeResult:
  episode_dir = Path(output_root) / episode_id
  episode_dir.mkdir(parents=True, exist_ok=True)
  try:
    pair, aux, solve, audit, gate = run_dataset_i_raw_pair(config)
  except Exception as error:
    failure = {
      "schema_version": DATASET_I_EPISODE_VERSION,
      "episode_id": episode_id,
      "object_id": config.object_id,
      "split": split,
      "classification": "REJECTED_DIAGNOSTIC",
      "replay_repair_rate": 0.0,
      "config": asdict(config),
      "generation_exception": type(error).__name__,
      "blocking_reason": str(error),
      "sample_count": 0,
      "retained_full_pair": False,
    }
    (episode_dir / "manifest.json").write_text(
      json.dumps(failure, indent=2, sort_keys=True),
      encoding="utf-8",
    )
    return DatasetIEpisodeResult(
      episode_id=episode_id,
      object_id=config.object_id,
      split=split,
      classification="REJECTED_DIAGNOSTIC",
      episode_directory=episode_dir,
      pair_path=None,
      samples_path=None,
      gate=None,
      audit=None,
      sample_count=0,
    )
  classification = "RAW_VERIFIED" if gate.status == "PASS" else "REJECTED_DIAGNOSTIC"
  save_i_gate_summary(episode_dir / "episode_summary.json", config, solve, audit, gate)
  _save_episode_aux(episode_dir / "oracle_replay_aux.npz", solve, aux)
  pair_path: Path | None = None
  samples_path: Path | None = None
  sample_count = 0
  if retain_pair or gate.status != "PASS":
    pair_path = save_spatial_inverse_pair(episode_dir / "forward_replay_pair.h5", pair)
  if gate.status == "PASS":
    samples = build_dataset_i_episode_samples(pair, aux, config)
    bundle = _single_bundle(samples, episode_id, config.object_id, split)
    samples_path = save_dataset_i_samples(
      episode_dir / "causal_samples.npz",
      bundle,
      metadata={
        "episode_id": episode_id,
        "object_id": config.object_id,
        "split": split,
        "classification": classification,
      },
    )
    sample_count = samples.count
  manifest = {
    "schema_version": DATASET_I_EPISODE_VERSION,
    "episode_id": episode_id,
    "object_id": config.object_id,
    "split": split,
    "classification": classification,
    "replay_repair_rate": 0.0,
    "forward_provenance": pair.forward_provenance,
    "config": asdict(config),
    "gate": asdict(gate),
    "sample_count": sample_count,
    "retained_full_pair": pair_path is not None,
  }
  (episode_dir / "manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True),
    encoding="utf-8",
  )
  return DatasetIEpisodeResult(
    episode_id=episode_id,
    object_id=config.object_id,
    split=split,
    classification=classification,
    episode_directory=episode_dir,
    pair_path=pair_path,
    samples_path=samples_path,
    gate=gate,
    audit=audit,
    sample_count=sample_count,
  )


def _worker(argument: tuple[str, str, str, DatasetIForwardConfig, bool]) -> DatasetIEpisodeResult:
  output_root, episode_id, split, config, retain_pair = argument
  return generate_dataset_i_episode(
    output_root,
    episode_id,
    split,
    config,
    retain_pair=retain_pair,
  )


def generate_dataset_i_batch(
  output_root: str | Path,
  configs: Iterable[tuple[str, str, DatasetIForwardConfig]],
  *,
  workers: int = 1,
  retain_pairs: bool = True,
) -> tuple[DatasetIEpisodeResult, ...]:
  output = Path(output_root)
  output.mkdir(parents=True, exist_ok=True)
  arguments = [
    (str(output), episode_id, split, config, retain_pairs)
    for episode_id, split, config in configs
  ]
  results: list[DatasetIEpisodeResult] = []
  if workers <= 1:
    results = [_worker(argument) for argument in arguments]
  else:
    with ProcessPoolExecutor(max_workers=workers) as pool:
      futures = {pool.submit(_worker, argument): argument[1] for argument in arguments}
      for future in as_completed(futures):
        results.append(future.result())
  results.sort(key=lambda value: value.episode_id)
  classifications = {
    name: sum(value.classification == name for value in results)
    for name in ("RAW_VERIFIED", "REPAIRED", "REJECTED_DIAGNOSTIC")
  }
  batch_manifest = {
    "schema_version": DATASET_I_EPISODE_VERSION,
    "episode_count": len(results),
    "classifications": classifications,
    "raw_acceptance_rate": classifications["RAW_VERIFIED"] / max(len(results), 1),
    "replay_repair_rate": 0.0,
    "episodes": [
      {
        "episode_id": value.episode_id,
        "object_id": value.object_id,
        "split": value.split,
        "classification": value.classification,
        "sample_count": value.sample_count,
        "gate": asdict(value.gate) if value.gate is not None else None,
      }
      for value in results
    ],
  }
  (output / "batch_manifest.json").write_text(
    json.dumps(batch_manifest, indent=2, sort_keys=True),
    encoding="utf-8",
  )
  return tuple(results)


def merge_batch_samples(
  results: Sequence[DatasetIEpisodeResult],
  output_directory: str | Path,
) -> dict[str, Path]:
  bundles = [
    load_dataset_i_samples(value.samples_path)
    for value in results
    if value.classification == "RAW_VERIFIED" and value.samples_path is not None
  ]
  output = Path(output_directory)
  output.mkdir(parents=True, exist_ok=True)
  paths: dict[str, Path] = {}
  for split in sorted({str(bundle.split[0]) for bundle in bundles}):
    combined = concatenate_dataset_i_bundles(bundles, split=split)
    paths[split] = save_dataset_i_samples(
      output / f"dataset_i_{split}.npz",
      combined,
      metadata={"merged": True},
    )
  return paths
