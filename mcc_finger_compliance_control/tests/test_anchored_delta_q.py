from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from dp_dataset import (  # noqa: E402
    FingertipDiffusionDataset,
    compute_normalization,
)


class AnchoredDeltaQTest(unittest.TestCase):
    def setUp(self) -> None:
        time = np.arange(60, dtype=np.float32)
        q = np.stack(
            [0.25 * np.sin(time * (0.015 + 0.002 * j)) for j in range(16)],
            axis=-1,
        )
        state = np.zeros((len(time), 46), dtype=np.float32)
        state[:, :16] = q
        self.episodes = {0: (state, q.copy())}
        self.obs_horizon = 4
        self.pred_horizon = 8

    def test_waypoint_scales_and_reconstruction(self) -> None:
        norm = compute_normalization(
            self.episodes, [0], "anchored_delta_q", "force_normal",
            self.obs_horizon, self.pred_horizon,
        )
        self.assertEqual(norm.action_mean.shape, (8, 16))
        self.assertEqual(norm.action_std.shape, (8, 16))
        self.assertTrue(np.all(norm.action_mean == 0.0))
        dataset = FingertipDiffusionDataset(
            self.episodes, [0], norm, self.obs_horizon, self.pred_horizon,
            40, "force_normal", 0.0, 3, "anchored_delta_q", 0.05, 1.0,
        )
        self.assertLess(float(norm.action_std[0].mean()), float(norm.action_std[-1].mean()))
        for index in range(len(dataset)):
            normalized = dataset[index]["action"].numpy()
            self.assertLessEqual(float(np.abs(normalized).max()), 1.000001)
            rebuilt = dataset.action_base(index) + normalized * norm.action_std
            np.testing.assert_allclose(rebuilt, dataset.future_q(index), atol=1.0e-6)


if __name__ == "__main__":
    unittest.main()
