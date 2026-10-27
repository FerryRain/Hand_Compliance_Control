"""Motion reserve must respect redundancy, bounds, and nonpenetration."""
from pathlib import Path
import json
import sys
import unittest

import numpy as np
from scipy.optimize import linprog

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'mcc_finger_compliance_control/scripts'))
from healthy_grasp_planner import directional_capacity,HealthyGraspPlanner,HealthConfig,StableSplitHullGeometry
from object_catalog import load_object_config
from scipy.spatial.transform import Rotation


class HealthyGraspTests(unittest.TestCase):
    def test_rigid_origin_recheck_recovers_close_pair_and_keeps_witness_frame(self):
        fixture=json.loads((Path(__file__).parent/'fixtures/healthy_grasp_distance_regressions.json').read_text())
        geometry=StableSplitHullGeometry(load_object_config(fixture['object_id']))
        for row in fixture['cases']:
            pose=np.asarray(row['palm_pose_object'])
            value=geometry.evaluate(row['q'],pose,non_tip_clearance=True)
            np.testing.assert_allclose(value['distance'],row['expected_gap_m'],atol=2e-8)
            rotation=Rotation.from_quat(pose[[4,5,6,3]]).as_matrix()
            hand_object=value['hand_points_palm'] @ rotation.T+pose[:3]
            witness_distance=np.linalg.norm(hand_object-value['object_points'],axis=-1)
            np.testing.assert_allclose(witness_distance,value['distance'],atol=1e-7)
            self.assertTrue((value['distance']>0).all())
            self.assertTrue((value['non_tip_distance']>.005).all())

    def test_redundancy_preserves_capacity_when_one_joint_has_little_room(self):
        j=np.array([[1.,0.,0.,1.],[0.,1.,0.,0.],[0.,0.,1.,0.]])
        actual=directional_capacity(j,[[1.,0.,0.]],[-.1]*4,[.005,.1,.1,.1])
        self.assertAlmostEqual(float(actual[0]),.105,places=10)

    def test_batched_active_bounds_match_independent_linear_program(self):
        rng=np.random.default_rng(20261016)
        for _ in range(30):
            j=rng.normal(size=(3,4))*.03
            directions=rng.normal(size=(5,3));directions/=np.linalg.norm(directions,axis=-1,keepdims=True)
            lo=-rng.uniform(.001,.1,4);hi=rng.uniform(.001,.1,4)
            actual=directional_capacity(j,directions,lo,hi)
            expected=[]
            for d in directions:
                lp=linprog([0,0,0,0,-1],A_eq=np.column_stack((j,-d)),b_eq=[0,0,0],
                    bounds=list(zip(lo,hi))+[(0,None)],method='highs')
                self.assertTrue(lp.success);expected.append(lp.x[-1])
            np.testing.assert_allclose(actual,expected,atol=1e-9,rtol=1e-7)

    def test_geometric_penetration_is_not_a_planning_tolerance(self):
        p=HealthyGraspPlanner.__new__(HealthyGraspPlanner)
        p.config=HealthConfig();p.health_floor=np.array([[.01]*4,[.1]*4,[.001]*4])
        state=dict(geometry=dict(distance=np.full(4,.00015),non_tip_distance=np.full(17,.003),
            self_penetration=0.,cut_cap=np.zeros(4,bool)),
            health=np.array([[.1]*4,[.5]*4,[.02]*4]),joint_margin_rad=np.full(16,.1))
        self.assertTrue(p.valid(state))
        state['geometry']['distance'][0]=-.0001
        self.assertFalse(p.valid(state))
        state['geometry']['distance'][0]=.00015
        state['joint_margin_rad'][3]=-.01
        self.assertFalse(p.valid(state))


if __name__=='__main__':unittest.main()
