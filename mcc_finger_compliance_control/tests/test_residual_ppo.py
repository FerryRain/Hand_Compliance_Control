import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from residual_ppo import PPO, advantages, PPO_OBS_DIM, execution_feedback
from residual_training_adapter import OBS_DIM
from residual_reward import FullRewardConfig, FullResidualReward

class PPOTest(unittest.TestCase):
    def test_feedback_is_actual_target_error_and_limiter_gap(self):
        q=torch.zeros(2,16); command=torch.full_like(q,.03)
        result=execution_feedback(q,command,torch.full_like(q,.02),torch.full_like(q,.005),torch.full_like(q,.03))
        self.assertTrue(torch.allclose(result[:,:16],torch.full_like(q,-.06)))
        self.assertTrue(torch.allclose(result[:,16:],torch.full_like(q,.5)))

    def test_persistent_loss_is_state_cost_not_repeated_event(self):
        reward=FullResidualReward(1,'cpu',FullRewardConfig(force_warn_n=20,
            loss_weight=0,recovery_weight=0,persistent_loss_weight=.08,q_prior_weight=0))
        reward.reset(torch.zeros(1))
        strict=torch.tensor([[False,True,True,True]])
        losses=[]
        for _ in range(13):
            row=reward.step(strict=strict,force_n=torch.ones(1,4),geometry_cost=torch.zeros(1,4),
                q_prior_cost=torch.ones(1),reference_valid=torch.ones(1,dtype=torch.bool),
                normalized_residual=torch.zeros(1,16),completed=torch.zeros(1,dtype=torch.bool),
                failure=torch.zeros(1,dtype=torch.bool),decision_boundary=False)
            losses.append(float(row['loss']))
        self.assertEqual(losses[:10],[0.]*10)
        for value in losses[10:]:self.assertAlmostEqual(value,-.004)

    def test_lane_gae_terminal_and_partial_discount(self):
        rows=[]
        for lanes,rewards,discount in [([0,1],[1.,2.],[.9,0.]),([0],[3.],[0.])]:
            n=len(lanes)
            rows.append(dict(lane=torch.tensor(lanes), reward=torch.tensor(rewards),
                discount=torch.tensor(discount),lam=torch.ones(n),value=torch.zeros(n),next_value=torch.zeros(n)))
        self.assertTrue(torch.allclose(advantages(rows),torch.tensor([3.7,2.,3.])))

    def test_raw_ratio_and_real_update(self):
        cfg=dict(history=1,hidden=16,actor_lr=1e-4,value_lr=3e-4,seed=7,
                 gae_lambda=.97,epochs=2,minibatch=16,clip=.1,target_kl=.01)
        agent=PPO(cfg,dict(amplitude_rad=[.02]*16,decision_slew_rad=[.002]*16),'cpu')
        obs=torch.randn(32,PPO_OBS_DIM); critic=torch.cat([obs,torch.zeros(32,5)],-1)
        action=agent.act(obs)
        logp=agent.actor.distribution(obs).log_prob(agent.last_z).sum(-1)
        self.assertTrue(torch.equal(logp,agent.last_logp))
        self.assertTrue(torch.equal(action,agent.last_z.tanh()))
        before=[p.detach().clone() for p in agent.actor.parameters()]
        agent.rows=[dict(lane=torch.arange(32),obs=obs,critic=critic,z=agent.last_z,
            logp=agent.last_logp,reward=torch.linspace(-1,1,32),discount=torch.zeros(32),
            lam=torch.ones(32),value=torch.zeros(32),next_value=torch.zeros(32))]
        agent.finish_rollout()
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before,agent.actor.parameters())))
        self.assertFalse(agent.rows)
        self.assertGreaterEqual(agent.last_losses['actor_updates_this_rollout'],1)
        expected_lr=cfg['actor_lr']*(.5 if agent.last_losses['kl_stopped'] else 1.)
        self.assertEqual(agent.last_losses['next_actor_lr'],expected_lr)
        with self.assertRaises(RuntimeError):agent.update()

if __name__=='__main__':unittest.main()
