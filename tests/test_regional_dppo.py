"""DP decision credit, native end boundaries and record-only overforce."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'mcc_finger_compliance_control/scripts'))
from regional_dppo import RegionalDPPO
from dppo_control_env import DecisionCollector
from regional_dppo_protocol import training_reward_adjustments,coverage_batches


class RegionalDPPOTest(unittest.TestCase):
    def test_training_reward_remains_sensitive_after_force_saturation(self):
        force=torch.tensor([10.,100.,160.,200.,1000.,8000.])
        risk=((force-40)/60).clamp(0,2).square()
        values=dict(contact=torch.full_like(force,.04),force=-.8*risk,posture=torch.zeros_like(force))
        values['total']=sum(values.values())
        adjusted=training_reward_adjustments(values,force,torch.zeros_like(force),
            dict(reward_version='deployable_health_v2',force_warn_n=40.))
        self.assertTrue(torch.isfinite(adjusted['total']).all())
        self.assertTrue((adjusted['total'][1:]<adjusted['total'][:-1]).all())
        self.assertEqual(adjusted['force_tail'][0],0)
        self.assertIs(training_reward_adjustments(values,force,force,{}),values)

    def test_all_training_episodes_covered_without_dev_leakage(self):
        rows=[dict(episode_id=i,frames=800+i*37,split='train',related_plan_group_id=str(i//3)) for i in range(31)]
        batches=coverage_batches(rows,8,47,0)
        self.assertEqual({r['episode_id'] for b in batches for r in b},set(range(31)))
        self.assertTrue(all(len(b)==8 for b in batches))
        self.assertEqual(batches,coverage_batches(rows,8,47,0))
        with self.assertRaises(ValueError):coverage_batches([dict(rows[0],split='val')],8,47,0)

    def test_privileged_critic_values_do_not_change_actor_sample(self):
        c=RegionalDPPO.__new__(RegionalDPPO)
        sampler=SimpleNamespace(sample=lambda n,cond,generator,**kw:torch.randn(n,16,16,generator=generator)+cond[:,0,None,None])
        agent=SimpleNamespace(device=torch.device('cpu'),sampler=sampler)
        observer=SimpleNamespace(latest=dict(pad_force=torch.ones(2,4,3)))
        DecisionCollector.__init__(c,agent,{},2,observer)
        c.training=False;c.probe=False;c.lengths=torch.tensor([800,800]);c.last_parts=torch.zeros(2,4);c.checks=dict(plans=0)
        diffusion=SimpleNamespace(conditional_sample=lambda *a,**kw:None)
        chain=SimpleNamespace(runtime=SimpleNamespace(policy=SimpleNamespace(diffusion=diffusion)),context=dict(frame=100))
        c.attach(chain);condition=torch.zeros(2,1536)
        a=diffusion.conditional_sample(2,condition,torch.Generator().manual_seed(42))
        before=c.critic_input(condition,chain.context)
        c.last_parts[:,1:]=1000.
        after=c.critic_input(condition,chain.context)
        b=diffusion.conditional_sample(2,condition,torch.Generator().manual_seed(42))
        self.assertFalse(torch.equal(before,after));self.assertTrue(torch.equal(a,b))

    def test_variable_lengths_no_bootstrap_or_padding_rewards(self):
        c=RegionalDPPO.__new__(RegionalDPPO)
        cfg=dict(force_warn_n=40.,gamma_frame=.997**.2,seam_warn_m=.01)
        sampler=SimpleNamespace(last=None)
        def sample(n,cond,generator,**kw):
            sampler.last=dict(condition=cond.clone())
            return torch.zeros(n,16,16)
        sampler.sample=sample
        agent=SimpleNamespace(device=torch.device('cpu'),sampler=sampler,
                              critic=lambda x:torch.zeros(len(x),1))
        observer=SimpleNamespace(latest=dict(pad_force=torch.ones(2,4,3),
            connector_found=torch.zeros(2,4,dtype=torch.bool),connector_force=torch.zeros(2,4,3)))
        DecisionCollector.__init__(c,agent,cfg,2,observer)
        c.training=True;c.probe=False;c.lengths=torch.tensor([80,96]);c.finished=torch.zeros(2,dtype=torch.bool)
        c.unsafe_seen=c.finished.clone();c.last_parts=torch.zeros(2,4);c.valid_frames=torch.zeros(2,dtype=torch.long)
        c.checks=dict(plans=0)
        diffusion=SimpleNamespace(conditional_sample=lambda *a,**k:None)
        chain=SimpleNamespace(runtime=SimpleNamespace(policy=SimpleNamespace(diffusion=diffusion)),bootstrap_end=75)
        c.attach(chain)
        geometry=lambda *a:(torch.ones(2,4,dtype=torch.bool),torch.zeros(2,4),torch.zeros(2))
        with patch('regional_dppo.RegionalResidual.geometry_costs',side_effect=geometry):
            for frame in range(75,105):
                chain.context=dict(frame=frame)
                if frame in (75,85,95):diffusion.conditional_sample(2,torch.zeros(2,1536),torch.Generator())
                parts=torch.tensor([[10.,0.,0.,0.],[10.,0.,0.,0.]])
                if frame==78:parts[1,2]=110.
                c.post_physics(chain,frame,torch.zeros(2,16),torch.zeros(2,7),parts)
        self.assertEqual(c.valid_frames.tolist(),[4,20])
        self.assertTrue(c.finished.all());self.assertFalse(c.pending.any())
        self.assertEqual(c.unsafe_seen.tolist(),[False,True])
        self.assertEqual(float(c.components['unsafe_event']),-100.)
        self.assertEqual(float(c.components['terminal']),20.)
        durations=[(int(l),int(d)) for row in c.rows for l,d in zip(row['lane'],row['duration'])]
        self.assertEqual(durations,[(0,4),(1,9),(1,10),(1,1)])
        self.assertEqual(sum(int((r['discount']==0).sum()) for r in c.rows),2)


if __name__=='__main__':unittest.main()
