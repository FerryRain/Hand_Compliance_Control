"""Transfer-clock regression tests; native physics is audited separately."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from Module.i04_geometric_traversal.whole_hand_recovery import RecoveryWholeHandTraversal


class TransferClockTests(unittest.TestCase):
    def task(self,force_seconds):
        t=object.__new__(RecoveryWholeHandTraversal)
        t.last=np.zeros(23);t.data=SimpleNamespace(time=0.,qvel=np.zeros(23));t.duration=100.
        t.h=SimpleNamespace(model=None,tip_geom_ids=np.arange(4));t.scratch=None
        t.planner=SimpleNamespace(data=None,order=np.arange(23),lower=np.full(23,-1.),upper=np.ones(23))
        t.rejections=0;t.events=[];t.refreshed=False
        t.measure=lambda:dict(f=np.array([3. if t.data.time<force_seconds else 0.,0.,0.,0.]),
            a=np.array([True,False,False,False]),n=np.tile([1.,0.,0.],(4,1)),p=np.zeros((4,3)))
        def tick(command,phase,obs):
            self.assertLessEqual(np.max(abs(command-t.last)),.0012+1e-10)
            t.last=command.copy();t.data.time+=.002
        t.tick=tick
        def refresh():t.refreshed=True
        t.refresh_surface=refresh
        return t

    def drive(self,t):
        target=np.full(23,.1)
        path=SimpleNamespace(duration_s=1.,waypoints=np.array([np.zeros(23),target]),
            sample=lambda progress:target*min(1.,progress))
        module='Module.i04_geometric_traversal.whole_hand_recovery.'
        with patch(module+'set_kinematics'),patch(module+'joint_state',return_value=np.zeros(23)), \
             patch(module+'belly_ik_terms',return_value=(np.zeros(6),np.zeros((6,23)),{})), \
             patch(module+'bounded_belly_step',return_value=(np.zeros(23),{'feasible':True})):
            return t.air_move(path,'NECESSARY_REORIENTATION')

    def test_force_pause_longer_than_old_eight_second_budget_finishes(self):
        t=self.task(11.018)
        self.assertTrue(self.drive(t));self.assertFalse(t.refreshed)
        self.assertGreaterEqual(t.data.time,12.018-1e-6);self.assertLess(t.data.time,12.03)
        self.assertLess(np.max(abs(t.last-.1)),.002)
        self.assertGreater(t.events[-1]['force_pause_s'],11.)

    def test_sustained_pause_is_bounded_and_refreshes_stale_targets(self):
        t=self.task(100.)
        self.assertFalse(self.drive(t));self.assertTrue(t.refreshed)
        self.assertGreaterEqual(t.data.time,40.-1e-6);self.assertLess(t.data.time,40.01)
        self.assertEqual(t.events[-1]['kind'],'TRANSFER_SEGMENT_REPLAN')


if __name__=='__main__':unittest.main()
