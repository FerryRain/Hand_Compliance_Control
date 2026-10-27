"""Reference scrubbing preserves waypoint indices, q order, and palm frames."""
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'mcc_finger_compliance_control/scripts'))
from plan_waypoint_viewer import PlanReference,ReferenceScene,WaypointBrowser


class PlanWaypointTests(unittest.TestCase):
    def fixture(self,path):
        pose=np.zeros((31,1,7));pose[:,0,0]=np.linspace(.1,.2,31);pose[:,0,1]=.15
        rotation=Rotation.from_euler('z',np.linspace(0,.6,31)).as_quat()
        pose[:,0,3:]=rotation[:,[3,0,1,2]]
        q=np.array([np.linspace(.1,.4,16),np.linspace(.2,.5,16),np.linspace(.3,.6,16)])
        with h5py.File(path,'w') as h:
            h.attrs.update(object_id='ycb_mustard',control_dt=.01)
            h['palm_pose_object']=pose
            h['q_hand']=np.zeros((31,1,16))  # Must not replace the actual planned knots.
            h['grasp_keyframe_frame_index']=np.array([[0],[15],[30]])
            h['grasp_keyframe_q']=q[:,None,:]
            h['grasp_keyframe_valid']=np.array([[False],[True],[True]])
        return pose[:,0],q

    def test_scrubbing_hits_planned_waypoints_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'plan.h5';pose,q=self.fixture(path)
            ref=PlanReference.load(path)
            for i,frame in enumerate([0,15,30]):
                actual_pose,actual_q,*_=ref.sample(frame)
                np.testing.assert_allclose(actual_pose,pose[frame],atol=1e-12)
                np.testing.assert_array_equal(actual_q,q[i])
            # Uses the same smoothstep between the knot times, not dense q_hand.
            u=5/15;u=u*u*(3-2*u)
            np.testing.assert_allclose(ref.sample(5)[1],(1-u)*q[0]+u*q[1])

    def test_valid_filter_does_not_shift_the_palm_timeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'plan.h5';pose,q=self.fixture(path)
            ref=PlanReference.load(path)
            actual_pose,actual_q,left,right,_=ref.sample(0,valid_only=True)
            np.testing.assert_array_equal(actual_pose,pose[0])
            np.testing.assert_array_equal(actual_q,q[1])
            self.assertEqual((left,right),(1,2))
            self.assertFalse(ref.valid[0])

    def test_native_hand_matches_full_planned_pose_and_joint_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'plan.h5';pose,q=self.fixture(path)
            ref=PlanReference.load(path)
            scene=ReferenceScene(ref,ghost_count=2)
            scene.seek(15)
            palm=scene.model.body('hand_0/palm_lower').id
            np.testing.assert_allclose(scene.data.xpos[palm],pose[15,:3],atol=1e-12)
            np.testing.assert_allclose(scene.data.xquat[palm],pose[15,3:],atol=1e-12)
            np.testing.assert_array_equal(scene.data.qpos[scene.addresses[0]],q[1])
            for i,frame in enumerate([0,30],1):
                body=scene.model.body(f'hand_{i}/palm_lower').id
                np.testing.assert_allclose(scene.data.xpos[body],pose[frame,:3],atol=1e-12)
            self.assertEqual(scene.check()['max_palm_position_error_m'],0.)

    def test_reset_camera_keeps_object_look_at_after_position_change(self):
        class Camera:
            def __init__(self):
                self._position=np.array([3.,3.,3.]);self.look_at=np.zeros(3)
            @property
            def position(self):return self._position
            @position.setter
            def position(self,value):
                # Viser's camera API translates look_at with position.
                self.look_at=self.look_at+value-self._position
                self._position=np.asarray(value)
        camera=Camera()
        browser=WaypointBrowser.__new__(WaypointBrowser)
        browser.reference=SimpleNamespace(control=np.array([[0.,.1,.2],[.2,.3,.4]]))
        browser.server=SimpleNamespace(initial_camera=SimpleNamespace(),
            get_clients=lambda:{0:SimpleNamespace(camera=camera)})
        browser.reset_camera()
        np.testing.assert_allclose(camera.look_at,[.1,.2,.3])


if __name__=='__main__':unittest.main()
