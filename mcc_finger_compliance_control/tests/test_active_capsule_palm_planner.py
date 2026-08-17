from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from active_capsule_palm_planner import (  # noqa: E402
    ActiveCapsulePalmPlanner,
    ActiveCapsulePalmPlannerConfig,
)


class ActiveCapsulePalmPlannerTest(unittest.TestCase):
    def setUp(self) -> None:
        # Palm origin 5 cm outside the middle of a Z-axis capsule.
        self.pose = np.asarray((0.20, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0))
        self.config = ActiveCapsulePalmPlannerConfig(
            surface_speed_m_s=0.01,
            travel_m=0.02,
            max_surface_acceleration_m_s2=1.0,
        )

    def test_contact_gates_surface_progress(self) -> None:
        planner = ActiveCapsulePalmPlanner(self.pose, self.config)
        for _ in range(20):
            planner.step(2, enabled=True)
        self.assertAlmostEqual(planner.progress_m, 0.0, places=9)
        self.assertGreater(planner.pause_steps, 0)
        for _ in range(20):
            planner.step(3, enabled=True)
        self.assertGreater(planner.progress_m, 0.0)

    def test_transports_standoff_and_emits_local_command(self) -> None:
        planner = ActiveCapsulePalmPlanner(self.pose, self.config)
        initial_radius = np.linalg.norm(planner.pose_object[:2])
        for _ in range(20):
            planner.step(4, enabled=True)
        self.assertAlmostEqual(
            np.linalg.norm(planner.pose_object[:2]), initial_radius, places=5
        )
        feature = planner.planner_feature(waypoint_count=1, step_frames=20)
        self.assertEqual(feature.shape, (6,))
        self.assertGreater(np.linalg.norm(feature), 0.0)
        self.assertAlmostEqual(
            np.linalg.norm(planner.pose_object[3:]), 1.0, places=6
        )


if __name__ == "__main__":
    unittest.main()
