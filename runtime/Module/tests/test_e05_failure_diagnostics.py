import unittest

import numpy as np

from Module.module_4_finger_dp.e05_failure_diagnostics import (
  causal_filtered_force,
  contiguous_segments,
  first_persistent_true,
)


class E05FailureDiagnosticTest(unittest.TestCase):
  def test_segments_and_persistent_event(self) -> None:
    mask = np.array([False, True, True, False, True, True, True])
    self.assertEqual(contiguous_segments(mask), [(1, 3), (4, 7)])
    self.assertEqual(first_persistent_true(mask, 3), 4)
    self.assertIsNone(first_persistent_true(mask, 4))

  def test_causal_force_filter_has_no_future_response(self) -> None:
    force = np.zeros((100, 4))
    force[50:, 2] = 10.0
    filtered = causal_filtered_force(force)
    self.assertEqual(filtered.shape, force.shape)
    self.assertTrue(np.allclose(filtered[:50, 2], 0.0, atol=1e-12))
    self.assertGreater(filtered[-1, 2], 9.0)


if __name__ == "__main__":
  unittest.main()
