"""Physical residence, orientation and connected routing regressions."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "PARK-3"))
from frontier_planner import Residence, global_frontier, surface_cost

CFG = SimpleNamespace(**json.loads((HERE / "frontier.json").read_bytes()))


class FrontierTest(unittest.TestCase):
    def test_dwell_and_failed_target_are_independent_and_decay(self):
        memory = Residence(CFG)
        up = np.array([0., 0., 1.])
        wrist = np.zeros(3)
        first = np.zeros(3)
        unreachable = np.array([.2, 0., 0.])
        target = dict(point=unreachable, normal=up)
        touch = SimpleNamespace(points=np.tile(first, (4, 1)), normals=np.tile(up, (4, 1)),
                                loads=np.ones(4), forces=np.ones(4), measured_heads=np.ones(4, bool))
        for tick in range(0, 1001, CFG.sample_ticks):
            memory.observe(touch, wrist, target, tick)
        snapshot = memory.snapshot(1000)
        surface = SimpleNamespace(points=np.array([first, first, unreachable]), normals=np.array([up, -up, up]))
        overlap, seconds, saturation = surface_cost(surface, first[None], up[None], snapshot, CFG)
        np.testing.assert_array_equal(overlap, [1, 0, 0])
        self.assertTrue(9 < seconds[0, 0] < 10)  # Four coincident fingers share one wall-clock duration.
        self.assertGreater(CFG.dwell_weight * saturation[0, 0], 2.3)
        self.assertGreater(CFG.failed_weight * saturation[2, 1], 1.7)
        self.assertLess(saturation[0, 1], 1e-8)
        self.assertLess(saturation[2, 0], 1e-8)
        np.testing.assert_array_equal(saturation[1], [0, 0])
        half_life_ticks = round(CFG.memory_half_life_s * 100)
        np.testing.assert_allclose(memory.snapshot(1000 + half_life_ticks)[:, 6:8], .5 * snapshot[:, 6:8])
        touch.measured_heads[:] = False
        memory.observe(touch, unreachable, None, 1010)
        self.assertEqual(memory.seen, {memory.key(first, up)})
        self.assertTrue(np.all(memory.snapshot(1010)[:, 6] <= snapshot[:, 6]))

    def test_global_route_rewards_reachable_intermediate(self):
        xyz = np.column_stack((np.array([0, .05, .10, .15, .20, .60]), np.zeros((6, 2))))
        surface = SimpleNamespace(points=xyz, normals=np.tile([0., 0., 1.], (len(xyz), 1)))
        indices, goal, progress, distance = global_frontier(surface, np.arange(len(xyz)), xyz[0], CFG)
        self.assertEqual(goal, 4)
        self.assertAlmostEqual(distance, .2)
        by_point = dict(zip(indices, progress))
        self.assertGreater(by_point[1], by_point[0])
        self.assertLess(by_point[5], by_point[0])


if __name__ == "__main__":
    unittest.main()
