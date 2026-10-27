"""Relabel verified Dataset-I episodes for the DPRef + role architecture.

The raw physical episodes are reused.  This module changes only the supervised
contract: the continuous target becomes a measured-q-anchored nominal command
chunk, while intentional contact transitions are inferred with temporal
confirmation and kinematic/force evidence.  Ambiguous contact flicker is
masked out of the categorical loss instead of being mislabeled as BREAK/MAKE.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from numpy.typing import NDArray

from Module.fr3_leap import ARM_HOME_Q, FullRobotModelConfig, build_full_robot
from Module.module_4_whole_hand_mcc.reference_interpreter import ContactRole


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / "Module/generated/finger_dp_formal_v1/scaling"
DEFAULT_OUTPUT = REPO_ROOT / "Module/generated/dpref_v1/relabelled_dataset_i"
SCHEMA_VERSION = "fr3-leap-dpref-dataset.v1"

INPUT_KEYS = (
  "force_history",
  "finger_state_geometry",
  "wrist_real_twist_history",
  "wrist_mcc_offset_history",
  "wrist_mcc_velocity_history",
  "future_wrist_plan_twist",
)


@dataclass(frozen=True, slots=True)
class DPRefLabelConfig:
  policy_dt_s: float = 0.020
  confirmation_time_s: float = 0.100
  transition_lookahead_s: float = 0.300
  minimum_normal_motion_m: float = 0.00005
  minimum_force_change_n: float = 0.05
  force_normalization_n: float = 2.0

  def __post_init__(self) -> None:
    for name in (
      "policy_dt_s",
      "confirmation_time_s",
      "transition_lookahead_s",
      "minimum_normal_motion_m",
      "minimum_force_change_n",
      "force_normalization_n",
    ):
      if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    if self.transition_lookahead_s <= self.confirmation_time_s:
      raise ValueError("transition lookahead must exceed confirmation time")

  @property
  def confirmation_steps(self) -> int:
    return int(round(self.confirmation_time_s / self.policy_dt_s))

  @property
  def lookahead_steps(self) -> int:
    return int(round(self.transition_lookahead_s / self.policy_dt_s))


class _FingerKinematics:
  """Palm-local fingertip displacement for evidence-only relabeling."""

  def __init__(self) -> None:
    self.handles = build_full_robot(
      FullRobotModelConfig(surface="extreme", gravity_m_s2=0.0)
    )
    self.data = mujoco.MjData(self.handles.model)

  def local_tips(self, finger_q_rad: NDArray[np.float64]) -> NDArray[np.float64]:
    self.data.qpos[self.handles.arm_qpos_adrs] = ARM_HOME_Q
    self.data.qpos[self.handles.hand_qpos_adrs] = finger_q_rad
    mujoco.mj_forward(self.handles.model, self.data)
    palm = self.data.site_xpos[self.handles.palm_site_id]
    rotation = self.data.site_xmat[self.handles.palm_site_id].reshape(3, 3)
    return (
      self.data.site_xpos[self.handles.tip_site_ids] - palm[None, :]
    ) @ rotation


def _stable(signal: NDArray[np.bool_], end: int, steps: int, value: bool) -> bool:
  if end < steps - 1:
    return False
  window = signal[end - steps + 1 : end + 1]
  return bool(np.all(window if value else ~window))


def _first_confirmed_future(
  signal: NDArray[np.bool_],
  start: int,
  *,
  confirmation_steps: int,
  lookahead_steps: int,
  value: bool,
) -> int | None:
  first_end = start + confirmation_steps
  last_end = min(len(signal) - 1, start + lookahead_steps)
  for end in range(first_end, last_end + 1):
    if _stable(signal, end, confirmation_steps, value):
      return end
  return None


def _normal_displacement_to_horizon(
  *,
  kinematics: _FingerKinematics,
  q_current: NDArray[np.float64],
  q_future: NDArray[np.float64],
  finger: int,
  contact_position_palm_m: NDArray[np.float64],
  normal_palm: NDArray[np.float64],
  future_wrist_twist: NDArray[np.float64],
  horizon_s: float,
) -> float:
  current_tip = kinematics.local_tips(q_current)[finger]
  future_tip = kinematics.local_tips(q_future)[finger]
  finger_motion = future_tip - current_tip
  wrist_translation = future_wrist_twist[:3] * horizon_s
  wrist_rotation = np.cross(
    future_wrist_twist[3:] * horizon_s,
    contact_position_palm_m,
  )
  return float(np.dot(finger_motion + wrist_translation + wrist_rotation, normal_palm))


def _previous_nominal_commands(
  future_teacher_command: NDArray[np.float32],
  anchor_q: NDArray[np.float32],
  episode_id: NDArray[np.str_],
) -> NDArray[np.float32]:
  previous = np.array(anchor_q, copy=True)
  for episode in np.unique(episode_id):
    indices = np.flatnonzero(episode_id == episode)
    if len(indices) > 1:
      previous[indices[1:]] = future_teacher_command[indices[:-1], 0]
  return previous


def _role_labels(
  arrays: dict[str, NDArray[Any]],
  config: DPRefLabelConfig,
) -> tuple[NDArray[np.int64], NDArray[np.bool_], dict[str, Any]]:
  force_history = np.asarray(arrays["force_history"], dtype=np.float64)
  contact = force_history[:, :, -1, 1] >= 0.5
  filtered_force = config.force_normalization_n * np.sinh(force_history[:, :, -1, 0])
  geometry = np.asarray(arrays["finger_state_geometry"], dtype=np.float64)
  q_current = np.asarray(arrays["anchor_q_meas_rad"], dtype=np.float64)
  q_future = np.asarray(arrays["future_teacher_command_rad"], dtype=np.float64)
  wrist_plan = np.asarray(arrays["future_wrist_plan_twist"], dtype=np.float64)
  episode_id = np.asarray(arrays["episode_id"], dtype=np.str_)
  labels = np.full(contact.shape, int(ContactRole.KEEP), dtype=np.int64)
  valid = np.zeros(contact.shape, dtype=np.bool_)
  evidence_normal_m = np.full(contact.shape, np.nan, dtype=np.float64)
  evidence_force_n = np.full(contact.shape, np.nan, dtype=np.float64)
  rejection_reasons: dict[str, int] = {
    "UNCONFIRMED_CURRENT_STATE": 0,
    "AMBIGUOUS_RELEASE": 0,
    "AMBIGUOUS_MAKE": 0,
  }
  kinematics = _FingerKinematics()
  confirmation = config.confirmation_steps
  lookahead = config.lookahead_steps

  for episode in np.unique(episode_id):
    indices = np.flatnonzero(episode_id == episode)
    episode_contact = contact[indices]
    for local_index, global_index in enumerate(indices):
      for finger in range(4):
        signal = episode_contact[:, finger]
        stable_contact = _stable(signal, local_index, confirmation, True)
        stable_clear = _stable(signal, local_index, confirmation, False)
        if not stable_contact and not stable_clear:
          rejection_reasons["UNCONFIRMED_CURRENT_STATE"] += 1
          continue
        opposite_end = _first_confirmed_future(
          signal,
          local_index,
          confirmation_steps=confirmation,
          lookahead_steps=lookahead,
          value=not stable_contact,
        )
        if opposite_end is None:
          labels[global_index, finger] = int(
            ContactRole.KEEP if stable_contact else ContactRole.FREE
          )
          valid[global_index, finger] = True
          continue

        event_index = int(indices[opposite_end])
        horizon_steps = event_index - global_index
        if horizon_steps < 1:
          raise AssertionError("future role evidence must be causal in the label direction")
        action_index = min(horizon_steps, q_future.shape[1]) - 1
        horizon_s = (action_index + 1) * config.policy_dt_s
        normal_motion = _normal_displacement_to_horizon(
          kinematics=kinematics,
          q_current=q_current[global_index],
          q_future=q_future[global_index, action_index],
          finger=finger,
          contact_position_palm_m=geometry[global_index, finger, 8:11],
          normal_palm=geometry[global_index, finger, 11:14],
          future_wrist_twist=wrist_plan[global_index, action_index],
          horizon_s=horizon_s,
        )
        force_change = float(
          filtered_force[event_index, finger] - filtered_force[global_index, finger]
        )
        evidence_normal_m[global_index, finger] = normal_motion
        evidence_force_n[global_index, finger] = force_change
        if stable_contact:
          intentional = (
            normal_motion >= config.minimum_normal_motion_m
            and force_change <= -config.minimum_force_change_n
          )
          if intentional:
            labels[global_index, finger] = int(ContactRole.RELEASE)
            valid[global_index, finger] = True
          else:
            rejection_reasons["AMBIGUOUS_RELEASE"] += 1
        else:
          intentional = (
            normal_motion <= -config.minimum_normal_motion_m
            and force_change >= config.minimum_force_change_n
          )
          if intentional:
            labels[global_index, finger] = int(ContactRole.MAKE)
            valid[global_index, finger] = True
          else:
            rejection_reasons["AMBIGUOUS_MAKE"] += 1

  counts = {
    role.name: int(np.count_nonzero(valid & (labels == int(role))))
    for role in ContactRole
  }
  audit = {
    "label_counts": counts,
    "valid_role_labels": int(np.count_nonzero(valid)),
    "invalid_role_labels": int(np.count_nonzero(~valid)),
    "valid_role_fraction": float(np.mean(valid)),
    "rejection_reasons": rejection_reasons,
    "transition_evidence": {
      "finite_normal_motion_count": int(np.count_nonzero(np.isfinite(evidence_normal_m))),
      "normal_motion_min_m": float(np.nanmin(evidence_normal_m)),
      "normal_motion_max_m": float(np.nanmax(evidence_normal_m)),
      "force_change_min_n": float(np.nanmin(evidence_force_n)),
      "force_change_max_n": float(np.nanmax(evidence_force_n)),
    },
  }
  return labels, valid, audit


def relabel_split(
  source_path: str | Path,
  output_path: str | Path,
  config: DPRefLabelConfig = DPRefLabelConfig(),
) -> dict[str, Any]:
  source = Path(source_path)
  destination = Path(output_path)
  with np.load(source, allow_pickle=False) as archive:
    arrays = {name: archive[name] for name in archive.files}
  missing = set((*INPUT_KEYS, "target_action_offsets_rad", "anchor_q_meas_rad", "future_teacher_command_rad", "episode_id", "object_id", "split", "timestamp_s")) - set(arrays)
  if missing:
    raise ValueError(f"Dataset-I source is missing {sorted(missing)}")
  role, role_valid, role_audit = _role_labels(arrays, config)
  previous_nominal = _previous_nominal_commands(
    arrays["future_teacher_command_rad"],
    arrays["anchor_q_meas_rad"],
    arrays["episode_id"],
  )
  # Verified inverse episodes did not contain Finger MCC.  Zero is the only
  # truthful previous-MCC state; deployment-in-the-loop data can later provide
  # nonzero corrections without corrupting the raw inverse provenance.
  previous_mcc_correction = np.zeros_like(arrays["anchor_q_meas_rad"], dtype=np.float32)
  destination.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(
    destination,
    **{key: arrays[key] for key in INPUT_KEYS},
    q_meas_rad=arrays["anchor_q_meas_rad"],
    previous_nominal_command_rad=previous_nominal,
    previous_mcc_correction_rad=previous_mcc_correction,
    target_nominal_offsets_rad=arrays["target_action_offsets_rad"],
    future_nominal_command_rad=arrays["future_teacher_command_rad"],
    target_role=role,
    role_label_valid=role_valid,
    source_raw_index=arrays.get("source_raw_index", np.arange(len(role))),
    timestamp_s=arrays["timestamp_s"],
    episode_id=arrays["episode_id"],
    object_id=arrays["object_id"],
    split=arrays["split"],
  )
  metadata = {
    "schema_version": SCHEMA_VERSION,
    "source_dataset": str(source),
    "source_dataset_class": "DATASET_I_RAW_VERIFIED",
    "replay_repair_policy": "NONE",
    "continuous_label": "q_nom_future_command_minus_current_q_meas",
    "role_label": "time_confirmed_intentional_transition_with_force_and_kinematic_evidence",
    "old_checkpoint_reusable": False,
    "finger_mcc_present_in_source": False,
    "previous_mcc_correction_semantics": "truthful_zero_for_non_mcc_source",
    "sample_count": int(len(role)),
    "episode_count": int(len(np.unique(arrays["episode_id"]))),
    "object_ids": sorted(np.unique(arrays["object_id"]).tolist()),
    "config": asdict(config),
    "role_audit": role_audit,
  }
  destination.with_suffix(".json").write_text(
    json.dumps(metadata, indent=2, sort_keys=True),
    encoding="utf-8",
  )
  return metadata


def relabel_dataset_i(
  source_root: str | Path = DEFAULT_SOURCE,
  output_root: str | Path = DEFAULT_OUTPUT,
  config: DPRefLabelConfig = DPRefLabelConfig(),
) -> dict[str, Any]:
  source = Path(source_root)
  output = Path(output_root)
  output.mkdir(parents=True, exist_ok=True)
  source_mapping = {
    "i20_train": "dataset_i_d20_train.npz",
    "i100_train": "dataset_i_d100_train.npz",
    "validation": "dataset_i_validation.npz",
    "test": "dataset_i_test.npz",
  }
  splits: dict[str, Any] = {}
  for split, filename in source_mapping.items():
    splits[split] = relabel_split(
      source / filename,
      output / f"dpref_{split}.npz",
      config,
    )
  summary = {
    "schema_version": SCHEMA_VERSION,
    "status": "RELABELLED_AND_AUDITED",
    "config": asdict(config),
    "splits": {
      split: {
        "sample_count": value["sample_count"],
        "episode_count": value["episode_count"],
        **value["role_audit"],
      }
      for split, value in splits.items()
    },
  }
  (output / "summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True),
    encoding="utf-8",
  )
  lines = [
    "# DPRef Dataset-I relabel audit",
    "",
    "The raw `RAW_VERIFIED` inverse episodes are unchanged.  Only the supervised",
    "contract is rebuilt.  Ambiguous contact flicker is masked from role CE loss.",
    "",
    "| Pool | Episodes | Samples | KEEP | RELEASE | FREE | MAKE | Valid role labels |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
  ]
  for split, value in summary["splits"].items():
    counts = value["label_counts"]
    lines.append(
      f"| {split} | {value['episode_count']} | {value['sample_count']} | "
      f"{counts['KEEP']} | {counts['RELEASE']} | {counts['FREE']} | "
      f"{counts['MAKE']} | {value['valid_role_fraction']:.3%} |"
    )
  lines.extend(
    (
      "",
      "Reproduce:",
      "",
      "```bash",
      "/home/ferry/data/Anaconda/envs/handcomp/bin/python -m "
      "Module.module_4_finger_dp.dpref_dataset",
      "```",
    )
  )
  (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
  return summary


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
  parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
  args = parser.parse_args()
  summary = relabel_dataset_i(args.source, args.output)
  print(json.dumps({"status": summary["status"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
  main()
