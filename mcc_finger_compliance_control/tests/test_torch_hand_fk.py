from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from surface_mcc_finger import (  # noqa: E402
    FullHandMCCFingerController,
    SurfaceMCCFingerController,
)
from torch_hand_fk import TorchHandFK  # noqa: E402


class TorchHandFKTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fk = TorchHandFK()
        cls.fullhand = FullHandMCCFingerController()
        cls.jacobian_controller = SurfaceMCCFingerController()

    def test_positions_match_mujoco(self) -> None:
        q = np.random.default_rng(2).uniform(
            self.fullhand.lower, self.fullhand.upper, size=(20, 16)
        )
        result = self.fk(torch.tensor(q, dtype=torch.float32)).detach().numpy()
        expected = np.stack([self.fullhand.tip_positions_palm(row) for row in q])
        self.assertLess(float(np.max(np.abs(result - expected))), 1.0e-6)

    def test_gradient_matches_mujoco_site_jacobian(self) -> None:
        q = np.random.default_rng(3).uniform(
            self.fullhand.lower, self.fullhand.upper
        )
        tensor = torch.tensor(q, dtype=torch.float32, requires_grad=True)
        result = torch.autograd.functional.jacobian(self.fk, tensor).detach().numpy()
        expected = np.stack(
            self.jacobian_controller._tip_jacobians_action_order(q)
        )
        self.assertLess(float(np.max(np.abs(result - expected))), 1.0e-6)


if __name__ == "__main__":
    unittest.main()
