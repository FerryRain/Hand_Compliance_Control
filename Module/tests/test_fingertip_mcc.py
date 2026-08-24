from __future__ import annotations

import unittest

import numpy as np

from Module.module_2_fingertip_mcc import FingertipMCC, MCCConfig
from Module.module_2_fingertip_mcc.benchmarks import (
  DEFAULT_CONFIG,
  run_curved_surface,
  run_static_contact,
  run_tangential_sliding,
)


class FingertipMCCTest(unittest.TestCase):
  def test_controller_modifies_only_compliance_direction(self) -> None:
    controller = FingertipMCC()
    plan = np.array([0.04, -0.03, 0.0])
    direction = np.array([0.0, 0.0, -1.0])
    command = controller.step(plan, direction, 2.0, 0.0)
    displacement = command.position_command - plan

    self.assertAlmostEqual(float(displacement[0]), 0.0, places=15)
    self.assertAlmostEqual(float(displacement[1]), 0.0, places=15)
    self.assertGreater(float(np.dot(displacement, direction)), 0.0)

  def test_controller_enforces_command_limits(self) -> None:
    config = MCCConfig(
      max_offset_m=0.001,
      max_velocity_m_s=0.002,
      max_acceleration_m_s2=0.01,
    )
    controller = FingertipMCC(config)
    observed_saturation = False
    for _ in range(1000):
      command = controller.step([0.0, 0.0, 0.0], [0.0, 0.0, -1.0], 100.0, 0.0)
      observed_saturation |= bool(command.saturated_limits)
      self.assertLessEqual(abs(command.offset_m), config.max_offset_m)
      self.assertLessEqual(abs(command.velocity_m_s), config.max_velocity_m_s)
      self.assertLessEqual(abs(command.acceleration_m_s2), config.max_acceleration_m_s2)
    self.assertTrue(observed_saturation)

  def test_rejects_non_unit_direction(self) -> None:
    controller = FingertipMCC()
    with self.assertRaisesRegex(ValueError, "unit length"):
      controller.step([0.0, 0.0, 0.0], [0.0, 0.0, -2.0], 1.0, 0.0)

  def test_static_contact_protocol(self) -> None:
    for desired_force in (1.0, 2.0, 3.0):
      with self.subTest(desired_force=desired_force):
        metrics = run_static_contact(desired_force)
        self.assertLessEqual(metrics.force_rmse_n, 0.05)
        self.assertLessEqual(metrics.overshoot_n, 0.20)
        self.assertEqual(metrics.force_violation_probability, 0.0)
        self.assertLessEqual(metrics.max_abs_offset_m, DEFAULT_CONFIG.max_offset_m)
        self.assertLessEqual(metrics.max_abs_velocity_m_s, DEFAULT_CONFIG.max_velocity_m_s)
        self.assertLessEqual(
          metrics.max_abs_acceleration_m_s2,
          DEFAULT_CONFIG.max_acceleration_m_s2,
        )

  def test_tangential_sliding_protocol(self) -> None:
    metrics = run_tangential_sliding()
    self.assertLessEqual(metrics.force_rmse_n, 0.05)
    self.assertLessEqual(metrics.max_tangential_error_m, 1e-9)
    self.assertEqual(metrics.contact_loss_count_after_settling, 0)

  def test_curved_surface_protocol(self) -> None:
    for surface in ("cylinder", "sphere"):
      with self.subTest(surface=surface):
        metrics = run_curved_surface(surface)
        self.assertLessEqual(metrics.force_rmse_n, 0.06)
        self.assertLessEqual(metrics.max_tangential_error_m, 1e-6)
        self.assertEqual(metrics.contact_loss_count_after_settling, 0)


if __name__ == "__main__":
  unittest.main()
