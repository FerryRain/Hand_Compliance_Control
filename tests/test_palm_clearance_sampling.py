"""Per-trajectory margins must be reproducible without changing orientation draws."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'mcc_finger_compliance_control/scripts'))
from generate_manifold_palm_plan import sample_palm_clearance


class PalmClearanceSamplingTests(unittest.TestCase):
    def test_reproducible_in_any_worker_order(self):
        seeds = list(range(221))
        forward = {s: sample_palm_clearance((.03, .04), s) for s in seeds}
        reverse = {s: sample_palm_clearance((.03, .04), s) for s in reversed(seeds)}
        self.assertEqual(forward, reverse)
        self.assertTrue(all(.03 <= x <= .04 for x in forward.values()))
        self.assertGreater(len(set(forward.values())), 200)

    def test_does_not_consume_object_random_stream(self):
        rng = np.random.default_rng(800012)
        reference = rng.normal(size=4)
        rng = np.random.default_rng(800012)
        sample_palm_clearance((.03, .04), 800012)
        np.testing.assert_array_equal(reference, rng.normal(size=4))

    def test_fixed_margin_and_invalid_ranges(self):
        self.assertEqual(sample_palm_clearance((.04, .04), 12), .04)
        for bounds in [(.04, .03), (0., .04), (float('nan'), .04), (.03, float('inf'))]:
            with self.assertRaises(ValueError):
                sample_palm_clearance(bounds, 12)


if __name__ == '__main__':
    unittest.main()
