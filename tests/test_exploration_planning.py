"""New exploration paths preserve geometry and use explicit speed semantics."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'mcc_finger_compliance_control/scripts'))
from generate_manifold_palm_plan import sweep_curve
from plan_exploration_dataset import candidate, time_by_speed, repair_frames, collection_admission,attempt


class ExplorationPlanningTests(unittest.TestCase):
    def test_rejected_healthy_reference_never_falls_back_to_preload_admission(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=argparse.Namespace(output=Path(tmp),seed=20261015,speeds_m_s=[.03,.05],
                palm_clearance_range_m=[.03,.04],palm_source=None,max_wrist_speed_deg_s=60.,
                stage='teacher',hand_planner='healthy',reference_bank=Path(tmp)/'bank.npz',
                knot_stride=25,repair_rounds=2,max_repair_knots=24)
            commands=[]
            def run(command,**kwargs):
                commands.append(command)
                if command[1].endswith('replan_healthy_reference.py'):
                    out=Path(command[command.index('--output')+1]);out.mkdir()
                    (out/'summary.json').write_text(json.dumps(dict(collection_admissible=False,
                        output=str(out/'reference_rejected.h5'),rounds=[dict(valid_fraction=.999)])))
                return argparse.Namespace(returncode=0)
            def retime(source,output,*args):
                with h5py.File(output,'w'):pass
                return dict(frames=100)
            with patch('plan_exploration_dataset.subprocess.run',side_effect=run), \
                 patch('plan_exploration_dataset.time_by_speed',side_effect=retime):
                result=attempt(candidate(0,args),args)
            self.assertEqual(result['status'],'healthy_reference_rejected')
            self.assertFalse(result['passed'])
            self.assertFalse(result['collection_candidate'])
            self.assertNotIn('collection_plan',result)
            self.assertEqual(len(commands),2)
            self.assertTrue(commands[-1][1].endswith('replan_healthy_reference.py'))

    def test_spiral_tangent_matches_position_derivative(self):
        axis=np.array([0.,0.,1.]);center=np.array([.1,.2,.3]);arc=2.3
        def evaluate(u):
            theta=.4-arc*u
            position=np.stack([2*np.cos(theta),np.sin(theta),.2*np.sin(theta)],axis=-1)+center
            tangent=-np.stack([-2*np.sin(theta),np.cos(theta),.2*np.cos(theta)],axis=-1)
            return sweep_curve(position,tangent,center,axis,u,arc,370.,35.)
        u=np.linspace(.01,.99,17);eps=1e-7
        _,tangent=evaluate(u)
        numerical=(evaluate(u+eps)[0]-evaluate(u-eps)[0])/(2*eps)
        np.testing.assert_allclose(tangent,numerical,atol=2e-7)

    def test_speed_pairs_share_geometry(self):
        args=argparse.Namespace(seed=20261015,speeds_m_s=[.03,.05])
        for group in range(4):
            slow=candidate(group*2,args);fast=candidate(group*2+1,args)
            self.assertEqual({k:v for k,v in slow.items()if k not in ['candidate_id','target_speed_m_s']},
                             {k:v for k,v in fast.items()if k not in ['candidate_id','target_speed_m_s']})

    def test_speed_changes_duration_without_padding_or_shortening(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'raw.h5'
            with h5py.File(source,'w') as f:
                pose=np.zeros((101,1,7));pose[:,0,0]=np.linspace(0,.3,101);pose[:,:,3]=1
                f['palm_pose_object']=pose;f['q_hand']=np.zeros((101,1,16));f['qvel']=np.zeros((101,1,16))
                f['episode_id']=np.zeros((101,1),int)
                f.attrs.update(object_id='ycb_mustard',control_dt=.01,planner_palm_outline_clearance_m=.035)
            with patch('plan_exploration_dataset.load_object_config',return_value=None), \
                 patch('plan_exploration_dataset.MeshNormalOracle.from_config',return_value=None), \
                 patch('plan_exploration_dataset.palm_outline_clearance_stats',return_value=(.05,.04,None)):
                slow=time_by_speed(source,Path(tmp)/'slow.h5',.03)
                fast=time_by_speed(source,Path(tmp)/'fast.h5',.06)
            self.assertAlmostEqual(slow['duration_s'],2*fast['duration_s'],places=2)
            self.assertAlmostEqual(slow['realized_mean_speed_m_s'],.03,places=4)
            for name in ['slow.h5','fast.h5']:
                with h5py.File(Path(tmp)/name) as f:
                    p=f['palm_pose_object'][:,0,0]
                    self.assertAlmostEqual(float(p[0]),0.,places=6)
                    self.assertAlmostEqual(float(p[-1]),.3,places=6)
                    self.assertTrue((np.diff(p)>=0).all())
                    # The zero-speed endpoint and float32 rounding may repeat
                    # the last sample, but must not create a padded interval.
                    self.assertGreater(float((np.diff(p)>0).mean()),.99)

    def test_repair_selects_violation_interior_not_existing_knots(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'geometry.npz';valid=np.ones(20,bool);valid[4:15]=False
            non=np.full((20,17),.003);non[10]=-0.02
            np.savez(p,valid=valid,non_tip_distance=non,pad_distance=np.full((20,4),-.003))
            chosen=repair_frames(p,{0,5,19},count=3)
            self.assertEqual(chosen[0],10)
            self.assertNotIn(5,chosen)
            self.assertTrue(all(not valid[i]for i in chosen))

    def test_reserve_dip_is_not_confused_with_training_acceptance_or_collision(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'geometry.npz'
            def save(margin,cap=False):
                np.savez(p,non_tip_distance=np.full((2,17),margin),pad_distance=np.full((2,4),-.003),
                    cross_finger_penetration=np.zeros(2),cut_cap=np.full((2,4),cap))
            save(.0019)
            self.assertTrue(collection_admission(p))
            save(-.0001)
            self.assertFalse(collection_admission(p))
            save(.003,cap=True)
            self.assertFalse(collection_admission(p))


if __name__=='__main__':unittest.main()
