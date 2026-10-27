from __future__ import annotations

import unittest

import numpy as np

from Module.module_2_fingertip_mcc import (
  FingertipMCC,
  FullRobotFingertipMCC,
  MCCConfig,
)
from Module.module_3_runtime_guards import (
  FullRobotGuardConfig,
  FullRobotGuardObservation,
  FullRobotGuardReason,
  FullRobotRuntimeGuards,
  HoldScope,
)


class FullRobotM02Test(unittest.TestCase):
  def test_signed_coordinated_error_and_contact_transition_reset(self) -> None:
    config = MCCConfig(dt_s=0.002)
    controller = FullRobotFingertipMCC(
      tuple(FingertipMCC(config) for _ in range(4))
    )
    positions = np.zeros((4, 3))
    directions = np.tile([0.0, 0.0, -1.0], (4, 1))
    result = controller.step(positions, directions, [1.0, -1.0, 0.0, 0.0], [1, 1, 0, 0])
    self.assertGreater(result.commands[0].offset_m, 0.0)
    self.assertLess(result.commands[1].offset_m, 0.0)
    controller.step(positions, directions, np.ones(4), [1, 1, 0, 0])
    released = controller.step(positions, directions, np.ones(4), [0, 1, 0, 0])
    self.assertEqual(released.commands[0].force_error_n, 0.0)
    self.assertEqual(released.commands[0].offset_m, 0.0)


def guard_observation(**updates: object) -> FullRobotGuardObservation:
  values: dict[str, object] = {
    "arm_q_rad": np.zeros(7),
    "arm_qd_command_rad_s": np.zeros(7),
    "arm_qd_actual_rad_s": np.zeros(7),
    "finger_q_rad": np.zeros(16),
    "finger_qd_command_rad_s": np.zeros(16),
    "finger_qd_actual_rad_s": np.zeros(16),
    "fingertip_forces_n": np.zeros(4),
    "wrist_wrench": np.zeros(6),
    "arm_external_torque_nm": np.zeros(7),
    "wrist_compliance_offset": np.zeros(6),
    "finger_compliance_offsets_m": np.zeros(4),
    "sensor_validity": {"joint_state": True, "tip_force": True, "wrist_wrench": True},
  }
  values.update(updates)
  return FullRobotGuardObservation(**values)


class FullRobotM03Test(unittest.TestCase):
  def setUp(self) -> None:
    self.config = FullRobotGuardConfig(
      arm_joint_lower_rad=-np.ones(7),
      arm_joint_upper_rad=np.ones(7),
      finger_joint_lower_rad=-np.ones(16),
      finger_joint_upper_rad=np.ones(16),
      dt_s=0.01,
    )

  def test_finger_stall_is_not_masked_by_arm_progress(self) -> None:
    guards = FullRobotRuntimeGuards(self.config)
    arm_command = np.full(7, 0.1)
    finger_command = np.zeros(16)
    finger_command[4:8] = 0.1
    decision = None
    for _ in range(16):
      decision = guards.evaluate(
        guard_observation(
          arm_qd_command_rad_s=arm_command,
          arm_qd_actual_rad_s=arm_command,
          finger_qd_command_rad_s=finger_command,
        )
      )
    assert decision is not None
    self.assertEqual(decision.reason, FullRobotGuardReason.SUSPECTED_FINGER_BLOCKAGE)
    self.assertEqual(decision.hold_scope, HoldScope.FINGER_LOCAL)
    self.assertEqual(decision.affected_indices, (1,))

  def test_wrist_force_sensor_invalid_and_arm_limit_are_global(self) -> None:
    guards = FullRobotRuntimeGuards(self.config)
    wrench = guards.evaluate(guard_observation(wrist_wrench=[0, 0, 81, 0, 0, 0]))
    self.assertEqual(wrench.reason, FullRobotGuardReason.WRIST_WRENCH_LIMIT)
    self.assertEqual(wrench.hold_scope, HoldScope.GLOBAL_SAFE_HOLD)
    invalid = guards.evaluate(
      guard_observation(sensor_validity={"joint_state": True, "wrist_wrench": False})
    )
    self.assertEqual(invalid.reason, FullRobotGuardReason.SENSOR_INVALID)
    limit_q = np.zeros(7)
    limit_q[0] = 0.99
    command = np.zeros(7)
    command[0] = 0.1
    limit = guards.evaluate(
      guard_observation(arm_q_rad=limit_q, arm_qd_command_rad_s=command)
    )
    self.assertEqual(limit.reason, FullRobotGuardReason.ARM_JOINT_LIMIT)


if __name__ == "__main__":
  unittest.main()
