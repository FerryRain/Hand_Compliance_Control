import sys
from pathlib import Path
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'mcc_finger_compliance_control/scripts'))
from collect_substep_object_probe import substep_poses


class SubstepMotionTests(unittest.TestCase):
    def test_endpoints_and_displacement(self):
        start=np.array([.1,.2,.3]);end=start+np.array([.001,-.002,.003])
        poses=substep_poses(start,[1,0,0,0],end,[1,0,0,0],10)
        np.testing.assert_allclose(poses[-1,:3],end)
        np.testing.assert_allclose(np.diff(np.vstack([start,poses[:,:3]]),axis=0),
                                   np.tile((end-start)/10,(10,1)),atol=1e-16)

    def test_rotation_short_arc_and_antipodal(self):
        q=Rotation.from_euler('z',20,degrees=True).as_quat()[[3,0,1,2]]
        poses=substep_poses([0,0,0],[1,0,0,0],[0,0,0],-q,10)
        angles=Rotation.from_quat(poses[:,[4,5,6,3]]).as_rotvec()[:,2]
        np.testing.assert_allclose(angles,np.radians(np.arange(2,21,2)),atol=1e-15)


if __name__=='__main__':unittest.main()
