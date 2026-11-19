"""Protect pairing and variable-length scoring in the regional DP experiment."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] /
                       'mcc_finger_compliance_control/scripts'))
from evaluate_regional_dp import contact_statistics, pad_episode, summarize


class RegionalEvaluationTest(unittest.TestCase):
    def test_padding_preserves_original_time_samples(self):
        source = np.array([[0., 1.], [2., 3.], [4., 5.]])
        padded = pad_episode(source, 6)
        np.testing.assert_array_equal(padded[:3], source)
        np.testing.assert_array_equal(padded[3:], np.repeat(source[-1:], 3, axis=0))
        with self.assertRaises(ValueError):
            pad_episode(source, 2)

    def test_true_tip_loss_is_not_hidden_by_found(self):
        strict = np.ones((20, 4), dtype=bool)
        strict[2:8, 0] = False
        result = contact_statistics(strict, np.ones_like(strict))
        self.assertAlmostEqual(result['strict_healthy_four_rate'], .7)
        self.assertEqual(result['found_four_rate'], 1.)
        self.assertEqual(result['longest_four_loss_frames'], 6)
        self.assertEqual(result['first_loss_over5_frame'], 2)

    @staticmethod
    def report(strict, noise=None):
        return dict(lanes=[dict(episode_id=1,branch=0,parent='parentA',complete=True,
            post_bootstrap=dict(strict_healthy_four_rate=strict,found_four_rate=1.,
                strict_healthy_per_finger=[strict]*4,longest_four_loss_frames=0,
                over100N_frames=0))],noise=noise or [])

    def test_failed_oracle_is_flagged_without_dropping_dp_result(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)
            for arm, score in [('oracle',.7),('baseline19k',.6)]:
                (output/f'batch_00_{arm}.json').write_text(json.dumps(self.report(score)))
            result=summarize(output,dict(batches=[dict(index=0)]))
            self.assertFalse(result['oracle_fidelity_by_episode']['1'])
            self.assertEqual(len(result['arms']['baseline19k']['lanes']),1)

    def test_different_diffusion_draws_reject_paired_conclusion(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)
            (output/'batch_00_baseline19k.json').write_text(json.dumps(self.report(.9,[dict(frame=75,before='a')])))
            (output/'batch_00_continued_best.json').write_text(json.dumps(self.report(.95,[dict(frame=75,before='b')])))
            with self.assertRaisesRegex(ValueError,'noise schedule differs'):
                summarize(output,dict(batches=[dict(index=0)]))


if __name__=='__main__':
    unittest.main()
