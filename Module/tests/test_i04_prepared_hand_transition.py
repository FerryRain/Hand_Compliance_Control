"""Known safe endpoints never bypass hand-edge or plant-state integrity."""
import unittest
from unittest.mock import patch
import mujoco
import numpy as np

from Module.fr3_leap import ARM_HOME_Q,HAND_NATURAL_Q,FullRobotModelConfig,build_full_robot
from Module.i04_geometric_traversal.contracts import ExperimentConfig,WholeHandTarget,WristTarget
from Module.i04_geometric_traversal.full_surface import JointPath
from Module.i04_geometric_traversal.physics import set_joint_state,pose,tip_positions
from Module.i04_geometric_traversal.sliding_safety import SlidingSafetyPlanner
from Module.i04_geometric_traversal.prepared_hand_transition import plan_prepared_hand_transition


class PreparedHandTransitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.h=build_full_robot(FullRobotModelConfig(surface='plane',gravity_m_s2=0.))

    def setUp(self):
        self.p=SlidingSafetyPlanner(self.h,ExperimentConfig())
        self.q=np.clip(np.r_[ARM_HOME_Q,HAND_NATURAL_Q],self.p.lower,self.p.upper)
        self.data=mujoco.MjData(self.h.model);set_joint_state(self.h,self.data,self.q)
        self.target=WholeHandTarget(WristTarget(pose(self.h,self.data)),tip_positions(self.h,self.data))
        self.assertIsNone(self.p.collision_reason(self.q))

    def test_nonfinite_zero_or_wrong_shape_outward_rejected(self):
        for value in ([np.nan,0.,1.],[0.,np.inf,0.],[0.,0.,0.],[0.,1.]):
            with self.subTest(value=value),self.assertRaises(ValueError):
                plan_prepared_hand_transition(self.p,self.data,self.target,self.q,outward=value)

    def test_witness_must_match_full_cartesian_goal_before_any_path_search(self):
        moved=self.target.fingertip_positions_world_m.copy();moved[2,0]+=.01
        wrong=WholeHandTarget(self.target.wrist,moved)
        with patch.object(self.p,'plan') as plan:
            path,diag=plan_prepared_hand_transition(self.p,self.data,wrong,self.q,outward=[0,0,1])
        self.assertIsNone(path);self.assertEqual(diag['reason'],'WITNESS_TARGET_MISMATCH')
        self.assertGreater(diag['residuals'][2],.009);plan.assert_not_called()

    def test_blocked_hand_edge_cannot_use_safe_endpoints_and_native_state_unchanged(self):
        state={key:np.asarray(getattr(self.data,key)).copy() for key in ('qpos','qvel','ctrl','qacc_warmstart','time')}
        margins=self.h.model.geom_margin.copy()
        retreat=JointPath(np.vstack([self.q,self.q]),np.array([.1]),self.target.wrist.pose_world,self.target.wrist.pose_world,0.)
        with patch.object(self.p,'plan',return_value=(retreat,None)) as plan,patch.object(self.p,'edge_free',return_value=False):
            path,diag=plan_prepared_hand_transition(self.p,self.data,self.target,self.q,outward=[0,0,1])
        self.assertIsNone(path)
        self.assertEqual(len(diag['attempts']),35)
        self.assertTrue(all(row['reason']=='HAND_PREPARATION_EDGE_BLOCKED' for row in diag['attempts']))
        self.assertTrue(all('allow_rrt' not in call.kwargs for call in plan.call_args_list))
        distances=[np.linalg.norm(call.args[1].wrist.pose_world[:3]-self.target.wrist.pose_world[:3]) for call in plan.call_args_list]
        self.assertAlmostEqual(max(distances),.04)
        self.assertEqual(tuple(diag['retreat_candidate_distances_m']),(.004,.008,.012,.025,.04))
        for key,value in state.items():np.testing.assert_array_equal(getattr(self.data,key),value)
        np.testing.assert_array_equal(self.h.model.geom_margin,margins)

    def test_extended_retreat_is_explicit_and_recorded(self):
        with patch.object(self.p,'plan',return_value=(None,{'reason':'TEST_BLOCKED'})):
            _,diag=plan_prepared_hand_transition(self.p,self.data,self.target,self.q,outward=[0,0,1],distances=(.04,.06,.08))
        self.assertEqual(tuple(diag['retreat_candidate_distances_m']),(.04,.06,.08))
        self.assertEqual(max(row['distance_m'] for row in diag['attempts']),.08)
        for value in ((),(float('nan'),),(.081,)):
            with self.subTest(value=value),self.assertRaises(ValueError):
                plan_prepared_hand_transition(self.p,self.data,self.target,self.q,outward=[0,0,1],distances=value)


if __name__=='__main__':unittest.main()
