"""Replanning preserves source intent and rejects unsafe interpolation."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'mcc_finger_compliance_control/scripts'))
from replan_grasp_221 import commands, continuous_geometry, read_source


class ReplanGrasp221Tests(unittest.TestCase):
    def test_source_motion_and_seed_are_kept_in_regeneration(self):
        args = SimpleNamespace(seed=20261014, palm_clearance_range_m=(.03,.04),
            frames=1500, knot_stride=100, reference_bank=Path('/tmp/seeds.npz'))
        manifest = ROOT / 'mcc_finger_compliance_control/data/true_tip_v2_20261009/rescreen221_final/summary.json'
        episode = json.loads(manifest.read_text())['episodes'][100]
        row = read_source(episode, args)
        generate, optimize = commands(row,args,Path('/tmp/raw.h5'),Path('/tmp/new.h5'))
        value = lambda flag: generate[generate.index(flag)+1]
        self.assertEqual(int(value('--direction')), row['candidate']['direction'])
        self.assertEqual(int(value('--seed')), row['candidate']['object_seed'])
        self.assertEqual(float(value('--angle-deg')), row['candidate']['angle_deg'])
        self.assertEqual(int(value('--palm-clearance-seed')), args.seed+row['episode'])
        self.assertIn('--no-extract-feasible-segment', optimize)
        self.assertNotIn('--reference-trajectory', optimize)

    def test_safe_knots_do_not_hide_unsafe_midpoint(self):
        class Geometry:
            def __init__(self,*args): pass
            def evaluate(self,q,pose,**kw):
                bad = abs(q[0]-.5)<1e-8
                return dict(distance=np.full(4,-.001),connector_distance=np.full(4,.005),
                    cut_cap=np.zeros(4,bool),non_tip_distance=np.full(17,-.001 if bad else .003),
                    self_penetration=0.)
        with tempfile.TemporaryDirectory() as temporary:
            source=Path(temporary)/'plan.h5'
            with h5py.File(source,'w') as f:
                pose=np.zeros((3,1,7));pose[...,3]=1
                f['palm_pose_object']=pose
                f['grasp_keyframe_frame_index']=[0,2]
                f['grasp_keyframe_q']=np.array([np.zeros(16),np.ones(16)])
                f['grasp_keyframe_valid']=[True,True]
                f.attrs['object_id']='ycb_mustard'
            with patch('true_tip_waypoint_optimizer.SplitHullGeometry',Geometry):
                result=continuous_geometry(source,Path(temporary)/'geometry.npz')
            self.assertFalse(result['passed'])
            self.assertEqual(result['checked_frames'],3)
            self.assertEqual(result['first_invalid_frame'],1)
            with h5py.File(source,'r+') as f:
                f.attrs['non_tip_clearance_m']=.005
            with patch('true_tip_waypoint_optimizer.SplitHullGeometry',Geometry):
                stricter=continuous_geometry(source,Path(temporary)/'stricter.npz')
            self.assertEqual(stricter['first_invalid_frame'],0)
            self.assertEqual(stricter['valid_fraction'],0.)


if __name__=='__main__': unittest.main()
