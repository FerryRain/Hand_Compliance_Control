from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from palm_planner_features import future_palm_delta_pose_palm  # noqa: E402


class PalmPlannerFeaturesTest(unittest.TestCase):
    def test_stationary_plan_is_zero(self) -> None:
        pose = np.zeros((5, 7), dtype=np.float64)
        pose[:, 3] = 1.0
        value = future_palm_delta_pose_palm(
            pose, np.zeros(5), waypoint_count=2, step_frames=1
        )
        np.testing.assert_allclose(value, 0.0, atol=1.0e-7)

    def test_translation_and_rotation_are_in_current_palm_frame(self) -> None:
        pose = np.zeros((3, 7), dtype=np.float64)
        half = np.pi / 4.0
        pose[:, 3:7] = (np.cos(half), 0.0, 0.0, np.sin(half))
        pose[:, 0] = (0.0, 1.0, 2.0)
        value = future_palm_delta_pose_palm(
            pose, np.zeros(3), waypoint_count=1, step_frames=1
        )
        # Object +X is palm -Y after a +90 degree palm rotation around Z.
        np.testing.assert_allclose(value[:2, 0, :3], ((0, -1, 0), (0, -1, 0)), atol=1e-6)
        np.testing.assert_allclose(value[..., 3:], 0.0, atol=1e-6)

    def test_relative_rotation_vector_and_episode_boundary(self) -> None:
        pose = np.zeros((4, 7), dtype=np.float64)
        pose[:, 3] = 1.0
        angle = np.pi / 6.0
        pose[1, 3:7] = (np.cos(angle / 2), 0, 0, np.sin(angle / 2))
        pose[2:, 0] = (10.0, 11.0)
        episode = np.asarray((0, 0, 1, 1))
        value = future_palm_delta_pose_palm(
            pose, episode, waypoint_count=1, step_frames=1
        )
        self.assertAlmostEqual(float(value[0, 0, 5]), angle, places=6)
        # The first episode must clamp at index 1, not jump to episode 1.
        np.testing.assert_allclose(value[1], 0.0, atol=1e-7)
        self.assertAlmostEqual(float(value[2, 0, 0]), 1.0, places=6)
        np.testing.assert_allclose(value[3], 0.0, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
