from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from residual_reward import (  # noqa: E402
    HealthComponents, HealthWeights, RewardConfig, RewardTransition, compute_reward,
    FullRewardConfig, FullResidualReward,
)
from b_residual_reward_features import BRewardGeometry, HeldAngleCost, HeldContactMarginCost  # noqa: E402


class ResidualRewardTest(unittest.TestCase):
    def test_recovery_requires_continuous_safe_force(self) -> None:
        cfg = FullRewardConfig(force_warn_n=20)
        engine = FullResidualReward(1, "cpu", cfg)
        engine.reset(torch.zeros(1))
        engine.lost.fill_(True)
        def observe(force):
            return engine.step(strict=torch.ones(1, 4, dtype=torch.bool),
                force_n=torch.full((1, 4), float(force)), geometry_cost=torch.zeros(1, 4),
                q_prior_cost=torch.zeros(1), reference_valid=torch.zeros(1, dtype=torch.bool),
                normalized_residual=torch.zeros(1, 16), completed=torch.zeros(1, dtype=torch.bool),
                failure=torch.zeros(1, dtype=torch.bool), decision_boundary=True)
        for _ in range(5):
            unsafe = observe(99)
            self.assertEqual(float(unsafe['recovery']), 0)
            self.assertLess(float(unsafe['total']), 0)
        for _ in range(4):
            self.assertEqual(float(observe(10)['recovery']), 0)
        self.assertEqual(float(observe(30)['recovery']), 0)  # breaks confirmation
        for _ in range(4):
            self.assertEqual(float(observe(10)['recovery']), 0)
        self.assertEqual(float(observe(10)['recovery']), 4)

    def setUp(self) -> None:
        self.config = RewardConfig(
            gamma_decision=0.9, decision_frames=2, contact_weight=0,
            potential_weight=1, force_weight=0, residual_weight=0,
            force_soft_n=12, force_scale_n=12, safety_failure_penalty=0,
        )
        self.health_weights = HealthWeights(q=1, angle=0)

    @staticmethod
    def transition(before: float, after: float, *, terminal: bool = False,
                   failure: bool = False) -> RewardTransition:
        zeros = torch.zeros(2, 1, 4)
        return RewardTransition(
            contact_quality=zeros, force_n=zeros, frame_mask=torch.ones(2, 1, dtype=torch.bool),
            health_before=HealthComponents(torch.tensor([before]), torch.zeros(1)),
            health_after=HealthComponents(torch.tensor([after]), torch.zeros(1)),
            normalized_residual=torch.zeros(1, 16),
            terminated=torch.tensor([terminal]), safety_failure=torch.tensor([failure]),
        )

    def test_potential_telescopes_across_decisions(self) -> None:
        first = compute_reward(self.transition(0.2, 0.8), self.config, self.health_weights)
        second = compute_reward(self.transition(0.8, 0.4, terminal=True), self.config, self.health_weights)
        discounted_total = first.total + self.config.gamma_decision * second.total
        self.assertAlmostEqual(float(discounted_total.item()), 0.2, places=6)

    def test_safety_termination_has_no_exit_reward(self) -> None:
        config = RewardConfig(
            gamma_decision=0.9, decision_frames=2, contact_weight=0,
            potential_weight=1, force_weight=0, residual_weight=0,
            force_soft_n=12, force_scale_n=12, safety_failure_penalty=10,
        )
        result = compute_reward(self.transition(0.2, 4.0, terminal=True, failure=True),
                                config, self.health_weights)
        self.assertAlmostEqual(float(result.potential.item()), 0.2, places=6)
        self.assertAlmostEqual(float(result.total.item()), -9.8, places=6)
        self.assertEqual(float(result.bootstrap_discount.item()), 0.0)

    def test_batched_partial_terminal_and_contact_force_separation(self) -> None:
        config = RewardConfig(
            gamma_decision=0.9, decision_frames=2, contact_weight=1,
            potential_weight=0, force_weight=1, residual_weight=0,
            force_soft_n=12, force_scale_n=12, safety_failure_penalty=0,
        )
        quality = torch.ones(2, 2, 4)
        quality[0, 0, 3] = 0.5
        force = torch.zeros_like(quality)
        force[0, 0, 0] = 24
        mask = torch.tensor([[True, True], [False, True]])
        zeros = HealthComponents(torch.zeros(2), torch.zeros(2))
        result = compute_reward(RewardTransition(
            contact_quality=quality, force_n=force, frame_mask=mask,
            health_before=zeros, health_after=zeros,
            normalized_residual=torch.zeros(2, 16),
            terminated=torch.tensor([True, False]), safety_failure=torch.tensor([False, False]),
        ), config, HealthWeights(q=1, angle=1))
        self.assertAlmostEqual(float(result.contact[0]), 0.5, places=6)
        self.assertAlmostEqual(float(result.force[0]), 0.25, places=6)
        self.assertAlmostEqual(float(result.contact[1]), 1 + config.gamma_frame, places=6)
        self.assertEqual(float(result.bootstrap_discount[0]), 0.0)
        self.assertAlmostEqual(float(result.bootstrap_discount[1]), 0.9, places=6)

    def test_losing_contact_to_earn_recovery_is_unprofitable(self) -> None:
        cfg = FullRewardConfig(force_warn_n=20, contact_weight=0)
        engine = FullResidualReward(1, "cpu", cfg)
        engine.reset(torch.zeros(1))
        rewards, losses, recoveries = [], 0, 0
        for i in range(cfg.loss_frames + cfg.recovery_frames):
            result = engine.step(
                strict=torch.full((1, 4), i >= cfg.loss_frames), force_n=torch.zeros(1, 4),
                geometry_cost=torch.zeros(1, 4), q_prior_cost=torch.zeros(1),
                reference_valid=torch.ones(1, dtype=torch.bool), normalized_residual=torch.zeros(1, 16),
                completed=torch.zeros(1, dtype=torch.bool), failure=torch.zeros(1, dtype=torch.bool),
                decision_boundary=(i % 5 == 0))
            rewards.append(float(result["total"]) * cfg.gamma_frame ** i)
            losses += int(result["loss_events"])
            recoveries += int(result["recovery_events"])
        self.assertEqual(losses, 4)
        self.assertEqual(recoveries, 4)
        self.assertLess(sum(rewards), 0)

    def test_missing_normal_does_not_reset_angle_badness(self) -> None:
        memory = HeldAngleCost(1, torch.full((4,), 60.), torch.full((4,), 10.))
        before = memory.update(torch.full((1, 4), 70.), torch.ones(1, 4, dtype=torch.bool))
        after = memory.update(torch.zeros(1, 4), torch.zeros(1, 4, dtype=torch.bool))
        torch.testing.assert_close(before, after)

    def test_healthy_q_divergence_has_zero_prior_gate(self) -> None:
        engine = FullResidualReward(1, "cpu", FullRewardConfig(force_warn_n=20))
        phi, gate, _ = engine.potential(torch.ones(1, 4, dtype=torch.bool), torch.ones(1, 4),
                                        torch.zeros(1, 4), torch.ones(1), torch.ones(1, dtype=torch.bool))
        self.assertEqual(float(phi), 0)
        self.assertEqual(float(gate), 0)

    def test_nonfinite_lane_does_not_poison_parallel_lane(self) -> None:
        cfg = FullRewardConfig(force_warn_n=20)
        engine = FullResidualReward(2, "cpu", cfg)
        engine.reset(torch.zeros(2))
        force = torch.zeros(2, 4)
        force[0, 0] = float("nan")
        result = engine.step(strict=torch.ones(2, 4, dtype=torch.bool), force_n=force,
                             geometry_cost=torch.zeros(2, 4), q_prior_cost=torch.zeros(2),
                             reference_valid=torch.ones(2, dtype=torch.bool), normalized_residual=torch.zeros(2, 16),
                             completed=torch.zeros(2, dtype=torch.bool), failure=torch.zeros(2, dtype=torch.bool),
                             decision_boundary=True)
        self.assertTrue(result["done"][0])
        self.assertFalse(result["done"][1])
        self.assertTrue(torch.isfinite(result["total"]).all())

    def test_geometry_alone_does_not_enable_oracle_prior(self) -> None:
        engine = FullResidualReward(1, "cpu", FullRewardConfig(force_warn_n=20))
        phi, gate, _ = engine.potential(torch.ones(1, 4, dtype=torch.bool), torch.ones(1, 4),
                                        torch.ones(1, 4), torch.ones(1), torch.ones(1, dtype=torch.bool))
        self.assertEqual(float(gate), 0)
        self.assertAlmostEqual(float(phi), -0.3, places=6)

    def test_full_reward_potential_telescopes(self) -> None:
        cfg = FullRewardConfig(force_warn_n=20, contact_weight=0)
        engine = FullResidualReward(1, "cpu", cfg)
        engine.reset(torch.tensor([-.15]))
        total = 0
        for i, angle in enumerate((.8, .1, .9, .4)):
            result = engine.step(strict=torch.ones(1, 4, dtype=torch.bool), force_n=torch.zeros(1, 4),
                                 geometry_cost=torch.full((1, 4), angle), q_prior_cost=torch.ones(1),
                                 reference_valid=torch.ones(1, dtype=torch.bool), normalized_residual=torch.zeros(1, 16),
                                 completed=torch.tensor([i == 3]), failure=torch.zeros(1, dtype=torch.bool),
                                 decision_boundary=(i == 0))
            total += float(result["potential"]) * cfg.gamma_frame ** i
        self.assertAlmostEqual(total, .15, places=6)

    def test_delaying_real_contact_loss_improves_return(self) -> None:
        cfg = FullRewardConfig(force_warn_n=20)
        engine = FullResidualReward(2, "cpu", cfg)
        engine.reset(torch.zeros(2))
        total = torch.zeros(2)
        for i in range(300):
            strict = torch.ones(2, 4, dtype=torch.bool)
            strict[0, 0] = i < 100
            strict[1, 0] = i < 200
            result = engine.step(strict=strict, force_n=torch.zeros(2, 4), geometry_cost=torch.zeros(2, 4),
                                 q_prior_cost=torch.zeros(2), reference_valid=torch.zeros(2, dtype=torch.bool),
                                 normalized_residual=torch.zeros(2, 16), completed=torch.full((2,), i == 299),
                                 failure=torch.zeros(2, dtype=torch.bool), decision_boundary=(i % 5 == 0))
            total += result["total"] * cfg.gamma_frame ** i
        self.assertGreater(float(total[1]), float(total[0]))

    def test_unsafe_exit_is_worse_than_contactless_continuation(self) -> None:
        cfg = FullRewardConfig(force_warn_n=20)
        engine = FullResidualReward(2, "cpu", cfg)
        engine.reset(torch.zeros(2))
        total = torch.zeros(2)
        for i in range(30):
            failure = torch.tensor([i == 15, False])
            result = engine.step(strict=torch.zeros(2, 4, dtype=torch.bool), force_n=torch.zeros(2, 4),
                                 geometry_cost=torch.zeros(2, 4), q_prior_cost=torch.zeros(2),
                                 reference_valid=torch.zeros(2, dtype=torch.bool), normalized_residual=torch.zeros(2, 16),
                                 completed=torch.full((2,), i == 29), failure=failure, decision_boundary=(i % 5 == 0))
            total += result["total"] * cfg.gamma_frame ** i
        self.assertLess(float(total[0]), float(total[1]))

    def test_smoothness_compares_consecutive_decisions_after_slew(self) -> None:
        cfg = FullRewardConfig(force_warn_n=20)
        engine = FullResidualReward(1, "cpu", cfg)
        engine.reset(torch.zeros(1))
        for value, boundary in ((.2, True), (.4, False), (.8, True)):
            result = engine.step(strict=torch.ones(1, 4, dtype=torch.bool), force_n=torch.zeros(1, 4),
                                 geometry_cost=torch.zeros(1, 4), q_prior_cost=torch.zeros(1),
                                 reference_valid=torch.zeros(1, dtype=torch.bool),
                                 normalized_residual=torch.full((1, 16), value),
                                 completed=torch.zeros(1, dtype=torch.bool), failure=torch.zeros(1, dtype=torch.bool),
                                 decision_boundary=boundary)
        self.assertAlmostEqual(float(result["smooth"]), -.01 * .6 ** 2, places=6)

    def test_contact_margin_warns_before_binary_contact_failure(self) -> None:
        calibration = {"pad_radius_margin_m": [.025] * 4, "pad_x_margin_m": [.002] * 4,
                       "loaded_force_lower_margin_n": [3.] * 4}
        feature = HeldContactMarginCost(1, calibration, "cpu")
        risk = feature.update(torch.full((1, 4), .003), torch.full((1, 4), .04),
                              torch.ones(1, 4, dtype=torch.bool), torch.full((1, 4), .2))
        self.assertTrue((risk > .8).all())
        local = torch.tensor([.003, (.04 ** 2 - .003 ** 2) ** .5, 0.]).expand(1, 4, 3)
        strict = BRewardGeometry("cpu").strict(local, torch.ones(1, 4, dtype=torch.bool), torch.full((1, 4), .2))
        self.assertTrue(strict.all())

    def test_unloaded_contact_placeholder_does_not_erase_margin_cost(self) -> None:
        calibration = {"pad_radius_margin_m": [.025] * 4, "pad_x_margin_m": [.002] * 4,
                       "loaded_force_lower_margin_n": [3.] * 4}
        feature = HeldContactMarginCost(1, calibration, "cpu")
        feature.update(torch.full((1, 4), .003), torch.full((1, 4), .04),
                       torch.ones(1, 4, dtype=torch.bool), torch.full((1, 4), 10.))
        position_before = feature.position_cost.clone()
        after = feature.update(torch.full((1, 4), -100.), torch.full((1, 4), 100.),
                               torch.ones(1, 4, dtype=torch.bool), torch.zeros(1, 4))
        torch.testing.assert_close(feature.position_cost, position_before)
        self.assertTrue((after >= position_before).all())

    def test_four_finger_task_retains_partial_progress_reward(self) -> None:
        values = []
        for four_weight in (0., .5):
            cfg = FullRewardConfig(force_warn_n=20, contact_four_fraction=four_weight)
            engine = FullResidualReward(1, "cpu", cfg)
            engine.reset(torch.zeros(1))
            result = engine.step(strict=torch.tensor([[True, True, True, False]]), force_n=torch.zeros(1, 4),
                                 geometry_cost=torch.zeros(1, 4), q_prior_cost=torch.zeros(1),
                                 reference_valid=torch.zeros(1, dtype=torch.bool), normalized_residual=torch.zeros(1, 16),
                                 completed=torch.zeros(1, dtype=torch.bool), failure=torch.zeros(1, dtype=torch.bool),
                                 decision_boundary=True)
            values.append(float(result["contact"]))
        self.assertGreater(values[1], 0)
        self.assertLess(values[1], values[0])


if __name__ == "__main__":
    unittest.main()
