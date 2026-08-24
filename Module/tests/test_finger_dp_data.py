from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from Module.module_4_finger_dp import (
  DPDatasetEpisode,
  PrivilegedContactRepairOracle,
  PrivilegedRepairConfig,
  ReplayAcceptanceConfig,
  SpatialInverseConfig,
  audit_physical_replay,
  audit_spatial_inverse_pair,
  inverse_replay_wrist_proposal,
  load_spatial_inverse_pair,
  load_dataset_episode,
  run_spatial_inverse_physical_pair,
  save_dataset_episode,
  save_spatial_inverse_pair,
  spatial_inverse_replay_proposal,
)


def identity_pose(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> np.ndarray:
  return np.array([x, y, z, 1.0, 0.0, 0.0, 0.0])


def dataset_episode(length: int = 40, force_n: float = 2.0) -> DPDatasetEpisode:
  pose = np.tile(identity_pose(), (length, 1))
  normals = np.tile([0.0, 0.0, 1.0], (length, 4, 1))
  return DPDatasetEpisode(
    episode_id="unit-episode",
    seed=7,
    surface_model_version="oracle-v1",
    termination_reason="HORIZON_COMPLETE",
    time_s=np.arange(length) * 0.002,
    arm_q_meas_rad=np.zeros((length, 7)),
    arm_dq_meas_rad_s=np.zeros((length, 7)),
    arm_command_rad=np.zeros((length, 7)),
    q_f_meas_rad=np.zeros((length, 16)),
    dq_f_meas_rad_s=np.zeros((length, 16)),
    q_f_teacher_nominal_cmd_rad=np.full((length, 16), 0.01),
    q_f_teacher_executed_cmd_rad=np.full((length, 16), 0.008),
    force_raw_n=np.full((length, 4), force_n),
    force_filtered_n=np.full((length, 4), min(force_n, 6.0)),
    desired_force_n=np.full((length, 4), 2.0),
    contact_mask=np.ones((length, 4), dtype=bool),
    force_valid_mask=np.ones((length, 4), dtype=bool),
    contact_position_palm_m=np.zeros((length, 4, 3)),
    contact_normal_palm=normals,
    surface_distance_m=np.zeros((length, 4)),
    surface_uncertainty_m=np.zeros((length, 4)),
    geometry_from_contact=np.ones((length, 4), dtype=bool),
    surface_geometry_valid=np.ones((length, 4), dtype=bool),
    palm_pose_plan_world=pose,
    wrist_mcc_offset=np.zeros((length, 6)),
    palm_pose_command_world=pose,
    palm_pose_real_world=pose,
    wrist_mcc_velocity=np.zeros((length, 6)),
    collision_distance_m=np.full(length, 0.01),
    non_tip_contact_count=np.zeros(length),
    guard_state=np.full(length, "DP_ACTIVE"),
    guard_reason=np.full(length, "DP_AUTHORITY_ACTIVE"),
    authority_owner=np.full(length, "TEACHER"),
    teacher_source=np.full(length, "VERIFIED_INVERSE"),
    repair_mask=np.zeros(length, dtype=bool),
    authority_transition_reset_mask=np.zeros((length, 4), dtype=bool),
    authority_filter_solver_success=np.ones(length, dtype=bool),
    authority_filter_solver_status=np.full(length, "NOMINAL_FEASIBLE"),
    authority_filter_solver_iterations=np.zeros(length),
    authority_filter_intervention_norm_rad=np.zeros(length),
    authority_filter_maximum_constraint_violation=np.zeros(length),
    authority_filter_latency_s=np.zeros(length),
  )


class InverseReplayTest(unittest.TestCase):
  def test_v1_named_spatial_inverse_preserves_time_order(self) -> None:
    hand = np.tile(identity_pose(), (5, 1))
    object_world = np.stack([identity_pose(0.01 * index) for index in range(5)])
    proposal = spatial_inverse_replay_proposal(
      hand,
      object_world,
      identity_pose(),
    )
    self.assertFalse(proposal.reverse_time)
    self.assertTrue(np.allclose(proposal.object_pose_in_hand[:, 0], [0.0, 0.01, 0.02, 0.03, 0.04]))
    self.assertTrue(np.allclose(proposal.wrist_pose_world[:, 0], [0.0, -0.01, -0.02, -0.03, -0.04]))

  def test_generic_inverse_defaults_to_spatial_not_temporal(self) -> None:
    hand = np.tile(identity_pose(), (3, 1))
    object_world = np.stack([identity_pose(0.01 * index) for index in range(3)])
    proposal = inverse_replay_wrist_proposal(hand, object_world, identity_pose())
    self.assertFalse(proposal.reverse_time)
    self.assertTrue(np.allclose(proposal.object_pose_in_hand[:, 0], [0.0, 0.01, 0.02]))

  def test_fixed_object_proposal_preserves_relative_transform(self) -> None:
    hand = np.tile(identity_pose(), (5, 1))
    object_world = np.stack([identity_pose(0.01 * index) for index in range(5)])
    proposal = inverse_replay_wrist_proposal(
      hand,
      object_world,
      identity_pose(),
      reverse_time=True,
    )
    self.assertLess(proposal.maximum_relative_transform_residual, 1e-12)
    self.assertTrue(np.allclose(proposal.object_pose_in_hand[:, 0], [0.04, 0.03, 0.02, 0.01, 0.0]))
    self.assertTrue(np.allclose(proposal.wrist_pose_world[:, 0], [-0.04, -0.03, -0.02, -0.01, 0.0]))


class DatasetContractTest(unittest.TestCase):
  def test_hdf5_round_trip_and_command_chunks(self) -> None:
    episode = dataset_episode()
    generated = Path("Module/generated")
    generated.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=generated) as directory:
      path = Path(directory) / "episode.h5"
      save_dataset_episode(path, episode)
      loaded = load_dataset_episode(path)
      self.assertEqual(loaded.episode_id, episode.episode_id)
      self.assertTrue(np.allclose(loaded.q_f_teacher_executed_cmd_rad, 0.008))
      chunks = loaded.teacher_chunks(horizon=5, stride=5)
      self.assertGreater(len(chunks), 0)
      self.assertTrue(np.allclose(chunks[0].target_offsets_rad, 0.008))

  def test_physical_replay_audit_uses_execution_not_inverse_consistency(self) -> None:
    accepted = audit_physical_replay(
      dataset_episode(),
      ReplayAcceptanceConfig(minimum_duration_s=0.05),
    )
    self.assertTrue(accepted.accepted)
    rejected = audit_physical_replay(
      dataset_episode(force_n=9.0),
      ReplayAcceptanceConfig(minimum_duration_s=0.05),
    )
    self.assertFalse(rejected.accepted)
    self.assertIn("TIP_OVERFORCE", rejected.reasons)

  def test_repair_dominated_episode_is_not_silently_accepted(self) -> None:
    base = dataset_episode()
    episode = replace(
      base,
      repair_mask=np.ones(base.length, dtype=bool),
    )
    audit = audit_physical_replay(
      episode,
      ReplayAcceptanceConfig(minimum_duration_s=0.05),
    )
    self.assertFalse(audit.accepted)
    self.assertIn("REPAIR_DOMINATED", audit.reasons)

  def test_initial_buffer_fill_is_not_misreported_as_guard_takeover(self) -> None:
    base = dataset_episode()
    states = np.array(base.guard_state, copy=True)
    owners = np.array(base.authority_owner, copy=True)
    states[:5] = "BUFFER_FILL"
    owners[:5] = "INITIALIZATION"
    episode = replace(base, guard_state=states, authority_owner=owners)
    audit = audit_physical_replay(
      episode,
      ReplayAcceptanceConfig(minimum_duration_s=0.05),
    )
    self.assertTrue(audit.accepted)
    self.assertEqual(audit.guard_takeover_frames, 0)

  def test_minimum_real_spatial_pair_has_identity_command_mapping(self) -> None:
    config = SpatialInverseConfig(
      duration_s=2.0,
      object_traversal_y_m=-0.004,
      object_lateral_x_m=0.0005,
    )
    pair = run_spatial_inverse_physical_pair(config)
    audit = audit_spatial_inverse_pair(pair)
    self.assertTrue(audit.accepted, audit.reasons)
    self.assertEqual(pair.inversion_mode, "SPATIAL_ONLY")
    self.assertEqual(pair.time_mapping, "SAME_T_FORWARD_ORDER")
    self.assertEqual(pair.replay_repair_policy, "NONE")
    self.assertEqual(pair.maximum_finger_command_mapping_residual_rad, 0.0)
    self.assertGreater(audit.replay_average_contact_count, 3.0)
    self.assertGreater(audit.forward_contact_retention_probability, 0.80)
    self.assertTrue(
      np.array_equal(pair.forward.q_f_command_rad, pair.replay.q_f_command_rad)
    )
    self.assertFalse(
      np.array_equal(pair.forward.contact_force_n, pair.replay.contact_force_n)
    )
    generated = Path("Module/generated")
    generated.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=generated) as directory:
      path = Path(directory) / "pair.h5"
      save_spatial_inverse_pair(path, pair)
      loaded = load_spatial_inverse_pair(path)
      self.assertEqual(loaded.forward_provenance, pair.forward_provenance)
      self.assertTrue(
        np.array_equal(loaded.replay.contact_force_n, pair.replay.contact_force_n)
      )


class PrivilegedRepairOracleTest(unittest.TestCase):
  def test_non_mcc_horizon_optimization_reduces_force_error(self) -> None:
    oracle = PrivilegedContactRepairOracle(
      PrivilegedRepairConfig(
        joint_lower_rad=-np.ones(16),
        joint_upper_rad=np.ones(16),
        contact_stiffness_n_per_m=1000.0,
        max_joint_step_rad=0.06,
      )
    )
    jacobian = np.zeros((1, 16))
    jacobian[0, 0] = 0.01
    result = oracle.repair(
      current_q_rad=np.zeros(16),
      proposal_command_rad=np.zeros((4, 16)),
      signed_compression_jacobian_m_per_rad=jacobian,
      measured_normal_force_n=[0.0],
      desired_normal_force_n=[2.0],
    )
    self.assertTrue(result.success)
    self.assertLess(result.force_error_after_n, result.force_error_before_n)
    steps = np.diff(np.vstack((np.zeros((1, 16)), result.repaired_command_rad)), axis=0)
    self.assertLessEqual(float(np.max(np.abs(steps))), 0.060001)
    self.assertGreater(result.repaired_command_rad[-1, 0], 0.0)


if __name__ == "__main__":
  unittest.main()
