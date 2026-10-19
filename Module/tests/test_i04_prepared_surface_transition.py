"""The fallback cannot bypass the original policy or award geometric coverage."""
from types import SimpleNamespace
from unittest.mock import MagicMock,patch
import unittest
import mujoco
import numpy as np

from Module.fr3_leap import ARM_HOME_Q,HAND_NATURAL_Q,FullRobotModelConfig,build_full_robot
from Module.i04_geometric_traversal.contracts import ExperimentConfig
from Module.i04_geometric_traversal.full_surface import JointPath
from Module.i04_geometric_traversal.physics import set_joint_state,pose,tip_positions,rotation
from Module.i04_geometric_traversal.sliding_safety import SlidingSafetyPlanner
from Module.i04_geometric_traversal.prepared_surface_transition import plan_surface_transition_with_prepared

PREFIX='Module.i04_geometric_traversal.prepared_surface_transition.'


class PreparedSurfaceTransitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.h=build_full_robot(FullRobotModelConfig(surface='plane',gravity_m_s2=0.))

    def setUp(self):
        self.p=SlidingSafetyPlanner(self.h,ExperimentConfig())
        self.q=np.clip(np.r_[ARM_HOME_Q,HAND_NATURAL_Q],self.p.lower,self.p.upper)
        self.data=mujoco.MjData(self.h.model);set_joint_state(self.h,self.data,self.q)
        self.witness=dict(q=self.q.copy(),pose=pose(self.h,self.data),tips=tip_positions(self.h,self.data))
        self.path=JointPath(np.vstack([self.q,self.q]),np.array([.25]),self.witness['pose'],self.witness['pose'],0.)
        graph=SimpleNamespace(required_vertices=np.array([0]),vertices_m=np.array([[0.,0.,0.],[.01,0.,0.]]))
        self.sp=SimpleNamespace(h=self.h,graph=graph,planner=self.p,cell='B-MCC')
        self.coverage=SimpleNamespace(visited_vertices=np.zeros(2,dtype=bool),visited_goals=np.zeros(1,dtype=bool))
        self.attempts=np.array([999.,0.])
        self.oracle=SimpleNamespace(find_pose=MagicMock(return_value=(self.witness,{})),
            initial_rotation=rotation(self.witness['pose'][3:]),initial_normal=np.array([0.,0.,1.]))

    def call(self,**kwargs):
        return plan_surface_transition_with_prepared(self.sp,self.oracle,self.data,self.coverage,self.attempts,[0,0,1],**kwargs)

    def test_original_success_is_returned_before_any_extra_information_or_search(self):
        expected=([('NEAR_TRANSFER',self.path)],np.array([0.,0.,1.]),{'vertex':1})
        with patch(PREFIX+'plan_surface_transition',return_value=expected),patch(PREFIX+'plan_prepared_hand_transition') as fallback:
            actual=self.call()
        self.assertIs(actual,expected);fallback.assert_not_called();self.oracle.find_pose.assert_not_called()
        self.assertFalse(self.sp.last_prepared_transition_diagnostics['fallback_invoked'])

    def test_prepared_success_keeps_entire_path_air_and_does_not_award_contact(self):
        before={key:np.asarray(getattr(self.data,key)).copy() for key in ('qpos','qvel','ctrl','qacc_warmstart','time')}
        margins=self.h.model.geom_margin.copy()
        with patch(PREFIX+'plan_surface_transition',return_value=None),patch.object(self.p,'plan',return_value=(None,{'reason':'IK'})), \
             patch(PREFIX+'plan_prepared_hand_transition',return_value=(self.path,{'reason':None,'segment_phases':['LIFT','HAND_PREPARATION','TRANSFER'],'attempts':[]})), \
             patch(PREFIX+'np.random.default_rng') as random_unused,patch('mujoco.mj_step') as step:
            result=self.call()
        self.assertEqual(result[0][0][0],'TRANSFER');self.assertIs(result[0][0][1],self.path)
        self.assertFalse(result[2]['physical_contact_or_coverage_claim'])
        self.assertEqual(result[2]['internal_segment_phases'],['LIFT','HAND_PREPARATION','TRANSFER'])
        self.assertFalse(self.coverage.visited_vertices.any());self.assertFalse(self.coverage.visited_goals.any())
        self.assertEqual(self.oracle.find_pose.call_args.kwargs['variant'],0)
        for key,value in before.items():np.testing.assert_array_equal(getattr(self.data,key),value)
        np.testing.assert_array_equal(self.h.model.geom_margin,margins);step.assert_not_called();random_unused.assert_not_called()

    def test_all_failed_finite_variants_remain_none_and_always_include_zero(self):
        self.oracle.find_pose.return_value=(None,{'IK':8})
        with patch(PREFIX+'plan_surface_transition',return_value=None),patch(PREFIX+'plan_prepared_hand_transition') as prepared:
            result=self.call(variants=(3,2))
        self.assertIsNone(result);prepared.assert_not_called()
        self.assertEqual([c.kwargs['variant'] for c in self.oracle.find_pose.call_args_list],[0,3,2])
        self.assertEqual(self.sp.last_prepared_transition_diagnostics['reason'],'NO_SAFE_PREPARED_ROUTE_IN_FINITE_CANDIDATES')
        self.assertFalse(self.coverage.visited_vertices.any());self.assertFalse(self.coverage.visited_goals.any())

    def test_mismatched_cached_target_cannot_enter_either_path_planner(self):
        wrong=self.witness['tips'].copy();wrong[1,0]+=.02
        cache=dict(vertex=np.array([0]),q=self.q[None],pose=self.witness['pose'][None],tips=wrong[None])
        self.oracle.find_pose.return_value=(None,{})
        with patch(PREFIX+'plan_surface_transition',return_value=None),patch.object(self.p,'plan') as ordinary, \
             patch(PREFIX+'plan_prepared_hand_transition') as prepared:
            result=self.call(witnesses=cache)
        self.assertIsNone(result);ordinary.assert_not_called();prepared.assert_not_called()
        self.assertEqual(self.sp.last_prepared_transition_diagnostics['candidates'][0]['reason'],'WITNESS_TARGET_MISMATCH')

    def test_b_only_fallback_does_not_expand_a_information_boundary(self):
        self.sp.cell='A-MCC'
        with patch(PREFIX+'plan_surface_transition',return_value=None),patch(PREFIX+'plan_prepared_hand_transition') as prepared:
            self.assertIsNone(self.call())
        prepared.assert_not_called();self.oracle.find_pose.assert_not_called()


if __name__=='__main__':unittest.main()
