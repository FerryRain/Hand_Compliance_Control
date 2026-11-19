import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from bounded_residual import BoundedResidualPolicy, RecordedResidualActor


class BoundedResidualTests(unittest.TestCase):
    def test_actual_slew_and_decision_timing(self):
        actor = RecordedResidualActor(torch.tensor([[[2.] * 16], [[-2.] * 16]]))
        policy = BoundedResidualPolicy(actor, torch.full((1, 16), .1), torch.full((1, 16), .02))
        offsets = [policy(None, None, None) for _ in range(10)]
        self.assertEqual(actor.index, 2)
        self.assertEqual(policy.actor_decisions, 2)
        self.assertTrue(torch.allclose(offsets[4], torch.full((1, 16), .02)))
        self.assertTrue(torch.allclose(offsets[9], torch.zeros(1, 16), atol=1e-8))
        self.assertLessEqual(float(torch.stack(offsets).diff(dim=0).abs().max()), .004 + 1e-8)

    def test_zero_and_reset_preserve_no_motion(self):
        actor = RecordedResidualActor(torch.zeros(4, 2, 16))
        policy = BoundedResidualPolicy(actor, torch.ones(2, 16), torch.ones(2, 16))
        for _ in range(11):
            self.assertTrue(torch.equal(policy(None, None, None), torch.zeros(2, 16)))
        policy.reset()
        self.assertEqual(actor.index, 0)
        self.assertEqual(policy.actor_decisions, 0)

    def test_nonfinite_action_fails_without_corrupting_offset(self):
        actor = RecordedResidualActor(torch.full((1, 1, 16), float('nan')))
        policy = BoundedResidualPolicy(actor, torch.ones(1, 16), torch.ones(1, 16))
        with self.assertRaises(ValueError):
            policy(None, None, None)
        self.assertTrue(torch.equal(policy.offset, torch.zeros(1, 16)))


if __name__ == "__main__":
    unittest.main()
