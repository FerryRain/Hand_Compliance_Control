from __future__ import annotations

import unittest

import numpy as np

from Module.i01_bunny_physics.surface import canonical_bunny_heightfield
from Module.i02_i03_bunny_physics.core import (
  BunnyPadCenterShape,
  I02I03BunnyConfig,
  planned_cumulative_distance_m,
  planned_path_coordinate_m,
)
from Module.i02_i03_bunny_physics.runner import run_i02_i03_bunny


class I02I03ProtocolTest(unittest.TestCase):
  def test_frozen_path_has_110_mm_cumulative_distance(self) -> None:
    expected_coordinate = {
      0.0: 0.0,
      3.0: 0.0,
      4.0: 0.010,
      7.0: 0.010,
      12.0: 0.060,
      17.0: 0.010,
      20.0: 0.010,
    }
    for timestamp, coordinate in expected_coordinate.items():
      self.assertAlmostEqual(planned_path_coordinate_m(timestamp), coordinate)
    self.assertAlmostEqual(planned_cumulative_distance_m(20.0), 0.110)
    times = np.arange(0.0, 20.0, 0.002)
    path = np.asarray([planned_path_coordinate_m(time) for time in times])
    self.assertTrue(np.all(np.isfinite(path)))
    self.assertGreaterEqual(float(np.min(path)), 0.0)
    self.assertLessEqual(float(np.max(path)), 0.060 + 1e-12)

  def test_config_and_bunny_pad_center_contract(self) -> None:
    with self.assertRaises(ValueError):
      I02I03BunnyConfig(cell="unknown")
    with self.assertRaises(ValueError):
      I02I03BunnyConfig(cell="i02_short", duration_s=19.0)
    bunny = canonical_bunny_heightfield()
    shape = BunnyPadCenterShape(bunny, [0.0, 0.0, 0.0])
    center = np.array([0.0, 0.0, bunny.query(0.0, 0.0)[0]])
    projection = shape.query(center)
    self.assertAlmostEqual(float(np.linalg.norm(projection.normal)), 1.0)
    self.assertAlmostEqual(
      float(np.dot(projection.point - center, projection.normal)),
      0.002,
      places=9,
    )


class I02I03PhysicalSmokeTest(unittest.TestCase):
  def test_i02_short_seed7_has_three_fresh_reposition_barriers(self) -> None:
    _, metrics, _ = run_i02_i03_bunny(
      I02I03BunnyConfig(cell="i02_short", seed=7)
    )
    self.assertTrue(metrics["common_task_pass"])
    self.assertTrue(metrics["mechanism_pass"])
    self.assertEqual(metrics["reposition_certificate_count"], 3)
    self.assertEqual(metrics["reposition_barrier_count"], 3)
    self.assertTrue(metrics["fresh_measured_root_evidence"])
    self.assertTrue(metrics["all_certificates_authentic"])
    self.assertEqual(metrics["prediction_suffix_command_count"], 0)

  def test_i03_shadow_filters_the_seed7_dead_end(self) -> None:
    _, beam, _ = run_i02_i03_bunny(
      I02I03BunnyConfig(cell="i03_beam", seed=7)
    )
    _, shadow, _ = run_i02_i03_bunny(
      I02I03BunnyConfig(cell="i03_shadow", seed=7)
    )
    self.assertEqual(beam["selected_sequence"], ["SLIDE(3)"])
    self.assertEqual(beam["actual_terminal_viability"], "NONVIABLE")
    self.assertEqual(beam["dead_end_count"], 1)
    self.assertEqual(shadow["selected_sequence"], ["SLIDE(1)"])
    self.assertEqual(shadow["actual_terminal_viability"], "VIABLE")
    self.assertEqual(shadow["dead_end_count"], 0)
    self.assertTrue(shadow["common_task_pass"])
    self.assertTrue(shadow["mechanism_pass"])
    self.assertGreaterEqual(shadow["actual_terminal_joint_margin_rad"], 0.025)
    self.assertTrue(shadow["actual_successor_fingers"])
    self.assertEqual(shadow["shadow_execution_authority_count"], 0)


if __name__ == "__main__":
  unittest.main()
