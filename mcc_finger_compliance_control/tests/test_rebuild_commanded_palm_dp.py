from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from rebuild_commanded_palm_dp import commanded_delta, plan_indices  # noqa: E402


class CommandedPalmFeatureTest(unittest.TestCase):
    def test_endpoint_indexing_keeps_physical_twenty_frame_horizon(self) -> None:
        schedule=np.array([0,1,1478,1479,1499])
        np.testing.assert_array_equal(plan_indices(schedule,20,1500,1500,True),[20,21,1498,1499,1499])
    def test_plan_indices_match_collector_time_mapping(self) -> None:
        schedule = np.array([0, 1000, 2479, 2480, 2499], dtype=np.int64)
        actual = plan_indices(schedule, 20, 2500, 2500)
        expected = np.rint(np.minimum(schedule + 20, 2500) / 2500 * 2499).astype(int)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual[-1], 2499)

    def test_commanded_delta_uses_current_measured_palm_frame(self) -> None:
        current_rotation = Rotation.from_euler("z", 90, degrees=True)
        future_rotation = current_rotation * Rotation.from_euler("x", 30, degrees=True)
        current = np.r_[1.0, 2.0, 3.0, current_rotation.as_quat()[[3, 0, 1, 2]]]
        future = np.r_[1.0, 3.0, 3.0, future_rotation.as_quat()[[3, 0, 1, 2]]]
        result = commanded_delta(current[None], future[None])[0]
        np.testing.assert_allclose(result[:3], [1.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(result[3:], [np.pi / 6, 0.0, 0.0], atol=1e-6)

    def test_invalid_plan_parameters_fail(self) -> None:
        with self.assertRaises(ValueError):
            plan_indices(np.array([0]), 20, 0, 2500)


if __name__ == "__main__":
    unittest.main()
