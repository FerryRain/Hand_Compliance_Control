from __future__ import annotations

import unittest

import numpy as np

from Module.module_4_whole_hand_mcc.benchmark import evaluate_episode_thresholds
from Module.module_4_whole_hand_mcc.runner import E05MCCConfig, run_e05_mcc


class E05MCCFullRobotSmokeTest(unittest.TestCase):
  def test_both_mcc_cells_execute_on_the_same_full_robot_contract(self) -> None:
    for mode in ("E05-F-MCC", "E05-H-MCC"):
      with self.subTest(mode=mode):
        trace, metrics = run_e05_mcc(
          E05MCCConfig(
            mode=mode,
            duration_s=1.5,
            settling_time_s=0.3,
            pose_step_time_s=1.0,
            traversal_y_m=0.012,
            lateral_primary_amplitude_m=0.002,
            lateral_secondary_amplitude_m=0.001,
          )
        )
        self.assertEqual(trace.arm_q_rad.shape[1], 7)
        self.assertEqual(trace.finger_q_rad.shape[1], 16)
        # This deliberately short smoke includes the initial MAKE transient;
        # the 0.995 formal threshold applies only to the frozen 15 s protocol.
        self.assertGreater(metrics["contact_continuity_probability"], 0.90)
        self.assertGreater(metrics["traversal_y_m"], 0.005)
        self.assertFalse(metrics["dp_evaluated"])

  def test_performance_threshold_does_not_change_execution_semantics(self) -> None:
    _, metrics = run_e05_mcc(
      E05MCCConfig(
        duration_s=1.2,
        settling_time_s=0.25,
        pose_step_time_s=0.8,
        traversal_y_m=0.01,
        lateral_primary_amplitude_m=0.001,
        lateral_secondary_amplitude_m=0.0005,
      )
    )
    metrics["max_tip_force_n"] = 100.0
    metrics["traversal_y_m"] = 0.18
    thresholds = evaluate_episode_thresholds(metrics)
    self.assertFalse(thresholds["performance_met"])
    self.assertFalse(thresholds["checks"]["max_tip_force_n"]["met"])

  def test_shared_safety_has_bounded_command_transitions(self) -> None:
    trace, _ = run_e05_mcc(
      E05MCCConfig(
        mode="E05-H-MCC",
        duration_s=1.5,
        settling_time_s=0.3,
        pose_step_time_s=1.0,
        traversal_y_m=0.012,
        lateral_primary_amplitude_m=0.002,
        lateral_secondary_amplitude_m=0.001,
        enforce_shared_force_safety=True,
      )
    )
    wrist_delta = np.linalg.norm(
      np.diff(trace.commanded_palm_pose_world[:, :3], axis=0),
      axis=1,
    )
    finger_delta = np.max(np.abs(np.diff(trace.finger_command_rad, axis=0)), axis=1)
    self.assertLessEqual(float(np.max(wrist_delta)), 0.000081)
    self.assertLessEqual(float(np.max(finger_delta)), 0.001000001)


if __name__ == "__main__":
  unittest.main()
