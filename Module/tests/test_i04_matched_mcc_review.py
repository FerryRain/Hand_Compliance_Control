"""Recovery censorship and common-prefix denominators affect the conclusion."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from Module.i04_geometric_traversal.matched_mcc_review import contact_summary, recovery_metrics


class MatchedReviewChecks(unittest.TestCase):
    def test_initial_acquisition_and_terminal_censorship_are_not_successful_recovery(self):
        safe = np.zeros((10, 4), dtype=bool)
        safe[[2, 3, 6], :2] = True
        data = dict(time_s=np.arange(10)*.002, safe_belly=safe,
                    forces_n=safe.astype(float), coverage=np.arange(10)/100,
                    visited_vertex_count=np.arange(10), phase=np.full(10, 'TEST'))
        with tempfile.TemporaryDirectory() as directory:
            case = Path(directory)
            # Last observation is terminal and has no applied 2ms interval.
            (case/'summary.json').write_text(json.dumps(dict(executed_s=.018)))
            with patch('Module.i04_geometric_traversal.matched_mcc_review.native_arrays', return_value=data):
                result = recovery_metrics(case)
        self.assertEqual(result['whole_hand_loss_episodes'], 2)
        self.assertEqual(result['recovered_episodes'], 1)
        self.assertEqual(result['terminal_censored_episodes'], 1)
        self.assertTrue(result['episodes'][0]['initial_acquisition'])
        self.assertAlmostEqual(result['episodes'][-1]['gap_s'], .004)
        self.assertIsNone(result['episodes'][-1]['first_recontact_s'])
        self.assertIsNone(result['episodes'][1]['next_stable_support_s'])

    def test_full_timeline_contact_denominator_includes_empty_and_partial_contact(self):
        safe = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1]], dtype=bool)
        data = dict(safe_belly=safe, visited_vertex_count=np.arange(4),
                    coverage=np.arange(4)/10, forces_n=safe.astype(float))
        result = contact_summary(data)
        self.assertEqual(result['any_fraction'], .75)
        self.assertEqual(result['two_fraction'], .5)
        self.assertEqual(result['three_fraction'], .25)
        self.assertEqual(result['four_fraction'], .25)


if __name__ == '__main__':
    unittest.main()
