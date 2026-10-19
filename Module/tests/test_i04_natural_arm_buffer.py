"""Native regression of the observed arm-buffer ratchet under servo lag."""
import json
from pathlib import Path
import unittest

import numpy as np

from Module.i04_geometric_traversal.belly_bounded_step import bounded_belly_step
from Module.i04_geometric_traversal.contracts import ExperimentConfig
from Module.i04_geometric_traversal.natural_belly_traversal import build_scene, NaturalBellyPlanner
from Module.i04_geometric_traversal.sliding_safety import SlidingSafetyPlanner


class NativeArmBufferRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory=Path(__file__).resolve().parents[1]/'generated/test_natural_arm_buffer'
        directory.mkdir(parents=True,exist_ok=True)
        cls.h,graph,_,_=build_scene(directory)
        cls.planner=NaturalBellyPlanner(cls.h,graph)
        cls.original=SlidingSafetyPlanner(cls.h,ExperimentConfig())
        source=Path(__file__).parent/'fixtures/i04_arm_buffer_ratchet.json'
        cls.fixture=json.loads(source.read_text())

    def test_real_servo_lag_can_pass_measured_guard_while_reference_enters_buffer(self):
        f=self.fixture
        measured=np.array(f['measured_q_rad']);prior=np.array(f['previous_reference_rad']);command=np.array(f['proposed_command_rad'])
        self.assertIsNone(self.original.command_collision_reason(measured,command))
        self.assertEqual(self.original.command_collision_reason(prior,command),'ARM_SELF_CLEARANCE')
        self.assertGreater(self.planner.minimum_arm_clearance(prior),.001)
        self.assertLess(self.planner.minimum_arm_clearance(command),.001)

    def test_soft_buffer_step_increases_real_geometry_clearance_within_original_guards(self):
        q=np.array(self.fixture['previous_reference_rad']);actual=np.array(self.fixture['measured_q_rad'])
        J,error=self.planner.arm_buffer_terms(q,np.arange(23))
        self.assertGreater(len(error),0)
        delta,diagnostic=bounded_belly_step(J,error,q,self.planner.lower,self.planner.upper,maximum_joint_step=.0004)
        self.assertTrue(diagnostic['feasible'])
        after=q+delta
        self.assertGreater(self.planner.minimum_arm_clearance(after),self.planner.minimum_arm_clearance(q))
        self.assertIsNone(self.original.command_collision_reason(actual,after))
        self.assertIsNone(self.original.command_collision_reason(q,after))


if __name__=='__main__':unittest.main()
