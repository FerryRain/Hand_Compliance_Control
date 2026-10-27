"""Acceptance checks the executed interpolation and honest exported diagnostics."""
from pathlib import Path
import json
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest

import h5py
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'mcc_finger_compliance_control/scripts'))
from healthy_grasp_planner import HealthConfig
from replan_healthy_reference import certify_dense,export_reference,TemporalHealthyPlanner,plan_continuous
from bundle_manifold_plans import bundle


def state(q,penetrating=False):
    return dict(q=np.asarray(q),valid=True,geometric_health_valid=True,
        geometry=dict(distance=np.full(4,-.0001 if penetrating else .00015),
            non_tip_distance=np.full(2,.005),self_penetration=0.,
            cut_cap=np.zeros(4,bool),
            object_points=np.zeros((4,3)),pad_surface_normals=np.tile([1.,0.,0.],(4,1)),
            top_fraction=np.full(4,.8)),
        normal_object=np.tile([0.,1.,0.],(4,1)),
        health=np.tile(np.array([.05,.6,.02])[:,None],(1,4)),
        joint_margin_rad=np.full(16,.2),capacity_m=np.full((4,5),.003))


class HealthyReferenceTests(unittest.TestCase):
    def test_failed_continuation_cannot_be_exported_as_a_shorter_path(self):
        q=np.zeros(16);initial=state(q);failed=state(q,penetrating=True);failed['valid']=False
        calls=[]
        planner=SimpleNamespace(solve=lambda *args,**kwargs:(q,initial),
            advance=lambda *args:(calls.append(1) or q,failed))
        reference=SimpleNamespace(poses=np.zeros((3,7)),dt=.01,sample=lambda frame:(None,q))
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);args=SimpleNamespace(output=folder,input=folder/'source.h5',
                max_reference_velocity_rad_s=2.,continue_after_planning_failure=False)
            plan_continuous(args,reference,planner,time.monotonic())
            report=json.loads((folder/'summary.json').read_text())
            self.assertFalse(report['collection_admissible'])
            self.assertFalse(report['full_reference_completed'])
            self.assertEqual(report['failed_frame'],1)
            self.assertEqual(report['total_reference_frames'],3)
            self.assertEqual(len(calls),1)
            self.assertFalse((folder/'reference.h5').exists())
            self.assertFalse((folder/'reference_rejected.h5').exists())

    def test_continuation_opens_a_gap_without_exceeding_joint_velocity(self):
        planner=TemporalHealthyPlanner.__new__(TemporalHealthyPlanner)
        planner.config=HealthConfig();planner.lower=np.full(16,-1.);planner.upper=np.ones(16)
        planner.health_floor=np.array([[.01]*4,[.1]*4,[.001]*4])
        planner.health_target=np.array([[.05]*4,[.6]*4,[.02]*4])
        planner.oracle=SimpleNamespace(normals_at_world=lambda points:np.tile([0.,0.,1.],(4,1)))
        def plane_state(q,pose,capacity=False):
            result=state(q);result['geometry']['distance']=.00003+np.asarray(q)[[2,6,10,14]]
            result['jacobians']=np.tile(np.eye(3,4),(4,1,1))
            return result
        planner.state=plane_state
        q,checked=planner.advance(np.array([0.,0.,0.,1.,0.,0.,0.]),np.zeros(16),.0001)
        self.assertTrue(checked['valid'])
        self.assertLessEqual(float(abs(q).max()),.0001+1e-10)
        self.assertTrue((checked['geometry']['distance']>=HealthConfig().min_gap_m).all())
        np.testing.assert_allclose(q[[2,6,10,14]],.0001,atol=1e-10)

    def test_safe_knots_do_not_hide_penetrating_interpolation(self):
        class Planner:
            config=HealthConfig()
            temporal_limits=None
            def state(self,q,pose,capacity):
                return state(q,penetrating=.2<float(q[0])<.8)
            def constraints(self,s):return s['geometry']['distance']-.00005
            def valid(self,s,include_capacity):return bool((self.constraints(s)>=0).all())
        reference=SimpleNamespace(poses=np.zeros((3,7)),dt=.01)
        with tempfile.TemporaryDirectory() as tmp:
            dense,valid,severity,report=certify_dense(Planner(),reference,[0,2],
                np.array([np.zeros(16),np.ones(16)]),100.,Path(tmp)/'continuous.npz')
        np.testing.assert_array_equal(valid,[True,False,True])
        self.assertFalse(report['passed'])
        self.assertEqual(report['first_invalid_frame'],1)
        self.assertGreater(severity[1],severity[0])
        json.dumps(report,allow_nan=False)

    def test_export_preserves_palm_and_replaces_stale_acceptance_and_normals(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'source.h5';output=Path(tmp)/'new.h5'
            poses=np.zeros((3,1,7));poses[:,:,3]=1.
            with h5py.File(source,'w') as h:
                h['palm_pose_object']=poses;h['palm_twist_object']=np.ones((3,1,6))
                h['fingertip_pose_object']=np.ones((3,1,4,7))
                h.attrs.update(object_id='ycb_mustard',finger_contact_all_feasible=True,grasp_keyframe_valid_ratio=1.)
            q=np.array([np.zeros(16),np.ones(16)*.1]);states=[state(x) for x in q]
            states[1]['valid']=False
            reference=SimpleNamespace(dt=.01)
            calibration=dict(config=vars(HealthConfig()),non_tip_geom_names=['connector','palm'])
            dense=np.array([q[0],(q[0]+q[1])/2,q[1]])
            export_reference(source,output,reference,[0,2],q,states,dense,False,calibration)
            with h5py.File(output) as h:
                np.testing.assert_array_equal(h['palm_pose_object'][:],poses)
                np.testing.assert_allclose(h['grasp_keyframe_pad_normal_error'][:],np.pi/2)
                np.testing.assert_array_equal(h['grasp_keyframe_valid'][:,0],[True,False])
                np.testing.assert_array_equal(h['non_tip_geom_name'].asstr()[:],['connector','palm'])
                self.assertFalse(h.attrs['collection_admissible'])
                self.assertFalse(h.attrs['finger_contact_all_feasible'])
                self.assertEqual(h.attrs['grasp_keyframe_valid_ratio'],.5)
                self.assertNotIn('fingertip_pose_object',h)
            # A single-env bundle must not acquire an extra environment axis
            # when the collection coordinator bundles it again.
            bundled=Path(tmp)/'bundle.h5'
            with self.assertRaisesRegex(ValueError,'Rejected healthy reference'):
                bundle([output],bundled)
            states[1]['valid']=True
            export_reference(source,output,reference,[0,2],q,states,dense,True,calibration)
            bundle([output,output],bundled)
            with h5py.File(bundled) as h:
                self.assertTrue(h.attrs['healthy_reference_v1'])
                self.assertTrue(h.attrs['healthy_reference_admissible'])
                self.assertEqual(h.attrs['finger_contact_preload_m'],0.)
                self.assertEqual(h['palm_pose_object'].shape,(3,2,7))
                self.assertEqual(h['grasp_keyframe_q'].shape,(2,2,16))
                self.assertEqual(h['grasp_keyframe_local_motion_capacity_m'].shape,(2,2,4,5))
                np.testing.assert_array_equal(h['grasp_keyframe_valid'][:],np.ones((2,2),bool))


if __name__=='__main__':unittest.main()
