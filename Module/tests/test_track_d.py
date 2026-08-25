from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from Module.module_4_finger_dp.track_d_closed_loop import (
  TrackDClosedLoopConfig,
  TrackDClosedLoopMetrics,
  d_gate_verdict,
)
from Module.module_4_finger_dp.gpu_runtime import require_cuda
from Module.module_4_finger_dp.long_trajectory_dataset import (
  LongTrajectoryRecord,
  LongTrajectorySpec,
  audit_long_trajectory,
  build_compliant_long_training_set,
)
from Module.module_4_finger_dp.track_d_dataset import (
  build_track_d_samples,
  load_track_d_samples,
  save_track_d_samples,
)
from Module.module_4_whole_hand_mcc.runner import E05MCCTrace


def synthetic_teacher_trace(length: int = 2800) -> E05MCCTrace:
  time_s = np.arange(length, dtype=np.float64) * 0.002
  palm = np.tile([0.38, 0.10, 0.50, 1.0, 0.0, 0.0, 0.0], (length, 1))
  planned = palm.copy()
  planned[:, 1] += 0.002 * time_s
  q = np.zeros((length, 16), dtype=np.float64)
  q[:, 1::4] = 0.4
  q[:, 2::4] = 0.7
  q[:, 3::4] = 0.5
  q += 0.002 * np.sin(time_s[:, None] * 2.0)
  dq = np.gradient(q, 0.002, axis=0)
  command = q + 0.003
  fingertips = np.tile(
    np.array(
      [[0.35, 0.06, 0.46], [0.37, 0.08, 0.46], [0.39, 0.10, 0.46], [0.41, 0.12, 0.46]],
      dtype=np.float64,
    ),
    (length, 1, 1),
  )
  normals = np.tile([0.0, 0.0, 1.0], (length, 4, 1))
  zeros6 = np.zeros((length, 6), dtype=np.float64)
  zeros4 = np.zeros((length, 4), dtype=np.float64)
  zeros7 = np.zeros((length, 7), dtype=np.float64)
  return E05MCCTrace(
    time_s=time_s,
    arm_q_rad=zeros7,
    arm_dq_rad_s=zeros7,
    arm_command_rad=zeros7,
    finger_q_rad=q,
    finger_dq_rad_s=dq,
    finger_command_rad=command,
    palm_pose_world=palm,
    planned_palm_pose_world=planned,
    commanded_palm_pose_world=palm,
    fingertip_positions_world_m=fingertips,
    pad_normals_world=normals,
    contact_positions_world_m=fingertips,
    contact_normals_world=normals,
    fingertip_forces_n=np.full((length, 4), 2.0),
    actual_contacts=np.ones((length, 4), dtype=bool),
    desired_hand_wrench_world=zeros6,
    estimated_hand_wrench_world=zeros6,
    contact_hand_wrench_world=zeros6,
    arm_external_torque_nm=zeros7,
    wrist_compliance_offset=zeros6,
    finger_compliance_offsets_m=zeros4,
    coordinator_rank=np.zeros(length, dtype=np.int32),
    coordinator_condition=np.zeros(length),
    coordinator_internal_leakage_n=np.zeros(length),
    surface_curvature_inv_m=np.zeros((length, 4)),
    disturbance_active=np.zeros(length, dtype=bool),
    controller_latency_s=np.zeros(length),
    physics_step_latency_s=np.zeros(length),
    loop_latency_s=np.zeros(length),
    guard_reason=np.full(length, "NONE"),
    non_tip_contact_count=np.zeros(length, dtype=np.int32),
  )


def passing_metrics() -> TrackDClosedLoopMetrics:
  return TrackDClosedLoopMetrics(
    evaluation_duration_s=3.0,
    contact_continuity=1.0,
    zero_contact_time_s=0.0,
    longest_zero_contact_gap_s=0.0,
    average_contact_count=4.0,
    minimum_contact_count=3,
    per_finger_contact_probability=(1.0, 1.0, 1.0, 1.0),
    contact_loss_events=0,
    maximum_force_n=3.0,
    force_p95_n=2.5,
    force_rmse_n=0.3,
    soft_force_exposure_s=0.0,
    non_tip_contact_frames=0,
    teacher_command_rmse_rad=0.002,
    teacher_command_maximum_error_rad=0.01,
    policy_replan_count=100,
    policy_latency_mean_s=0.03,
    policy_latency_p95_s=0.05,
    authority_intervention_probability=0.1,
    authority_intervention_mean_rad=0.001,
    authority_solver_failure_frames=0,
    authority_maximum_constraint_violation=0.0,
    hard_guard_frames=0,
    soft_recovery_frames=0,
    dp_active_probability=1.0,
    opposition_rate=0.0,
    opposition_energy=0.0,
    opposition_valid_frames=0,
    opposition_conflict_frames=0,
    finger_collective_normal_velocity_p95_m_s=0.001,
    finger_collective_normal_max_abs_velocity_m_s=0.002,
    wrist_collective_normal_velocity_p95_m_s=0.004,
    wrist_plan_recenter_count=1,
    authority_contact_transition_count=0,
  )


class TrackDDatasetTest(unittest.TestCase):
  def test_causal_sample_builder_has_no_future_observation_leakage(self) -> None:
    samples = build_track_d_samples(synthetic_teacher_trace())
    self.assertTrue(samples.audit.passed, samples.audit.reasons)
    self.assertEqual(samples.audit.future_leakage_count, 0)
    self.assertAlmostEqual(samples.audit.force_history_duration_s, 0.2)
    self.assertAlmostEqual(samples.audit.minimum_target_timestamp_minus_anchor_s, 0.02)
    self.assertEqual(samples.target_action_offsets_rad.shape[1:], (20, 16))
    generated = Path("Module/generated")
    generated.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=generated) as directory:
      path = Path(directory) / "track_d_samples.npz"
      save_track_d_samples(path, samples)
      loaded = load_track_d_samples(path)
      self.assertEqual(loaded.count, samples.count)
      self.assertTrue(np.array_equal(loaded.source_raw_index, samples.source_raw_index))


class TrackDGateTest(unittest.TestCase):
  def test_gate_reports_specific_blocking_reason(self) -> None:
    config = TrackDClosedLoopConfig()
    passed = d_gate_verdict(
      causal_audit_passed=True,
      open_loop_first_command_rmse_rad=0.003,
      closed_loop=passing_metrics(),
      config=config,
    )
    self.assertEqual(passed.status, "PASS")
    self.assertEqual(passed.blocking_reason, ("NONE",))
    failed = d_gate_verdict(
      causal_audit_passed=True,
      open_loop_first_command_rmse_rad=0.003,
      closed_loop=replace(passing_metrics(), maximum_force_n=9.0),
      config=config,
    )
    self.assertEqual(failed.status, "FAIL")
    self.assertIn("closed_loop_tip_force", failed.blocking_reason)


class LongTrajectoryGateTest(unittest.TestCase):
  def test_accepts_only_a_full_ten_second_post_contact_segment(self) -> None:
    trace = synthetic_teacher_trace(length=6100)
    trace.actual_contacts[:50] = False
    spec = LongTrajectorySpec(
      episode_id="synthetic_train",
      split="TRAIN",
      seed=1,
      duration_s=12.2,
    )
    audit = audit_long_trajectory(trace, spec)
    self.assertTrue(audit.accepted_for_training, audit.reasons)
    self.assertGreaterEqual(audit.training_contact_duration_s, 10.0)
    self.assertEqual(audit.training_zero_contact_time_s, 0.0)

  def test_rejects_one_whole_hand_contact_dropout(self) -> None:
    trace = synthetic_teacher_trace(length=6100)
    trace.actual_contacts[:50] = False
    trace.actual_contacts[3000] = False
    spec = LongTrajectorySpec(
      episode_id="synthetic_dropout",
      split="TRAIN",
      seed=2,
      duration_s=12.2,
    )
    audit = audit_long_trajectory(trace, spec)
    self.assertFalse(audit.accepted_for_training)
    self.assertIn("ANY_FINGERTIP_CONTACT_LOSS", audit.reasons)

  def test_rejects_overforce_even_when_contact_is_continuous(self) -> None:
    trace = synthetic_teacher_trace(length=6100)
    trace.actual_contacts[:50] = False
    trace.fingertip_forces_n[3000, 1] = 8.0
    spec = LongTrajectorySpec(
      episode_id="synthetic_overforce",
      split="TRAIN",
      seed=3,
      duration_s=12.2,
    )
    audit = audit_long_trajectory(trace, spec)
    self.assertFalse(audit.accepted_for_training)
    self.assertIn("TIP_OVERFORCE", audit.reasons)

  def test_rejected_episode_contributes_no_training_samples(self) -> None:
    trace = synthetic_teacher_trace(length=6100)
    trace.actual_contacts[:50] = False
    trace.actual_contacts[3000] = False
    spec = LongTrajectorySpec(
      episode_id="synthetic_rejected",
      split="TRAIN",
      seed=4,
      duration_s=12.2,
    )
    record = LongTrajectoryRecord(
      spec=spec,
      trace_path=Path("this_rejected_trace_must_not_be_loaded.npz"),
      audit=audit_long_trajectory(trace, spec),
    )
    generated = Path("Module/generated")
    generated.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=generated) as directory:
      with self.assertRaisesRegex(RuntimeError, "no accepted long trajectories"):
        build_compliant_long_training_set(
          [record],
          Path(directory) / "must_not_exist.npz",
        )


class GPURuntimeTest(unittest.TestCase):
  def test_cpu_fallback_is_refused(self) -> None:
    with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
      require_cuda("cpu")


if __name__ == "__main__":
  unittest.main()
