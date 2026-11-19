from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from dp_fk_aux_loss import diffusion_with_fk_loss  # noqa: E402
from torch_hand_fk import TorchHandFK  # noqa: E402
from train_dp import build_policy  # noqa: E402


class DPFKAuxLossTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        args = SimpleNamespace(
            robot_state_dim=16,
            environment_state_dim=6,
            action_dim=16,
            obs_horizon=2,
            pred_horizon=4,
            down_dims=(16, 32),
            kernel_size=3,
            n_groups=8,
            diffusion_step_embed_dim=32,
            noise_scheduler="DDPM",
            diffusion_steps=10,
            inference_steps=5,
        )
        cls.policy = build_policy(args, torch.device("cpu"))
        cls.fk = TorchHandFK()
        cls.batch = {
            "observation.state": torch.randn(2, 2, 16),
            "observation.environment_state": torch.randn(2, 2, 6),
            "action": torch.randn(2, 4, 16) * 0.2,
            "action_is_pad": torch.zeros(2, 4, dtype=torch.bool),
        }
        cls.mean = torch.zeros(16)
        cls.scale = torch.ones(16)

    def test_zero_weight_matches_original_loss(self) -> None:
        torch.manual_seed(17)
        expected = self.policy.diffusion.compute_loss(self.batch)
        torch.manual_seed(17)
        actual, noise, auxiliary = diffusion_with_fk_loss(
            self.policy,
            self.batch,
            fk=None,
            action_mean=self.mean,
            action_scale=self.scale,
            fk_weight=0.0,
            fk_max_noise_timestep=9,
        )
        self.assertTrue(torch.equal(expected, actual))
        self.assertTrue(torch.equal(expected, noise))
        self.assertEqual(auxiliary.item(), 0.0)

    def test_positive_fk_loss_has_gradients(self) -> None:
        torch.manual_seed(7)
        total, noise, auxiliary = diffusion_with_fk_loss(
            self.policy,
            self.batch,
            fk=self.fk,
            action_mean=self.mean,
            action_scale=self.scale,
            fk_weight=1.0e-4,
            fk_max_noise_timestep=9,
        )
        self.assertTrue(torch.isfinite(total))
        self.assertGreater(auxiliary.item(), 0.0)
        self.assertAlmostEqual(
            total.item(), (noise + 1.0e-4 * auxiliary).item(), places=6
        )
        self.policy.zero_grad()
        total.backward()
        gradients = [
            parameter.grad
            for parameter in self.policy.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(value).all() for value in gradients))


if __name__ == "__main__":
    unittest.main()
