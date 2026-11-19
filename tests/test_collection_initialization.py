"""Initialization geometry and force feedback must share frames and signs."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'mcc_finger_compliance_control/scripts'))
from collection_initialization import bounded_normal_step, bounded_cartesian_normal_step, relative_palm_pose
from object_catalog import load_object_config
from surface_mcc_finger import HAND_JOINT_NAMES
from true_tip_waypoint_optimizer import SplitHullGeometry


class InitializationTests(unittest.TestCase):
    def test_relative_pose_invariant_to_world_rigid_transform(self):
        obj_r = Rotation.from_rotvec([.2, -.3, .1])
        palm_r = Rotation.from_rotvec([-.4, .1, .2])
        center = np.array([.3, -.1, .4])
        palm = np.array([.4, .2, .6])
        def pose(p, r):
            return np.r_[p, r.as_quat()[[3, 0, 1, 2]]]
        oracle = SimpleNamespace(center_world=center, rotation_world_from_object=obj_r.as_matrix())
        initial = relative_palm_pose(pose(palm, palm_r), oracle)
        world_r = Rotation.from_rotvec([.7, .2, -.2])
        shift = np.array([1., 2., -.5])
        transformed = SimpleNamespace(center_world=world_r.apply(center)+shift,
            rotation_world_from_object=(world_r*obj_r).as_matrix())
        actual = relative_palm_pose(pose(world_r.apply(palm)+shift, world_r*palm_r), transformed)
        np.testing.assert_allclose(initial[:3], actual[:3], atol=1e-12)
        self.assertAlmostEqual(abs(initial[3:] @ actual[3:]), 1., places=12)

    def test_normal_search_and_retreat_have_opposite_direction_after_limit(self):
        row = np.array([.03, -.01, .02, .005])
        for distance in [-.00015, .0002, -.05, .05]:
            delta = bounded_normal_step(row, distance)
            self.assertLessEqual(np.abs(delta).max(), .012+1e-12)
            self.assertGreater(float(row @ delta)*distance, 0.)
        np.testing.assert_array_equal(bounded_normal_step(np.zeros(4), .001), np.zeros(4))

    def test_actual_hull_witness_jacobian_matches_native_finite_difference(self):
        g = SplitHullGeometry(load_object_config('ycb_mustard'))
        q = np.full(16, .6)
        pose = np.array([.03, .08, .07, 1., 0., 0., 0.])
        result = g.evaluate(q, pose)
        mujoco.mj_comPos(g.model, g.data)
        finger = 0
        body = int(g.model.geom_bodyid[g.pads[finger]])
        point = result['hand_points_palm'][finger]
        local = g.data.xmat[body].reshape(3, 3).T @ (point-g.data.xpos[body])
        jac = np.zeros((3, g.model.nv))
        mujoco.mj_jac(g.model, g.data, jac, None, point, body)
        for j in range(4):
            joint = mujoco.mj_name2id(g.model, mujoco.mjtObj.mjOBJ_JOINT, HAND_JOINT_NAMES[j])
            dq = np.zeros(16); dq[j] = 1e-6
            g.position(q+dq, pose)
            plus = g.data.xpos[body]+g.data.xmat[body].reshape(3, 3) @ local
            g.position(q-dq, pose)
            minus = g.data.xpos[body]+g.data.xmat[body].reshape(3, 3) @ local
            np.testing.assert_allclose((plus-minus)/2e-6, jac[:, g.model.jnt_dofadr[joint]], atol=1e-8)

    def test_normal_feedback_does_not_command_tangential_scraping(self):
        jac = np.array([[.01, .03, -.02, .01], [.02, -.01, .03, .01], [.03, .02, .01, -.02]])
        normal = np.array([1., 2., -1.]); normal /= np.linalg.norm(normal)
        for distance in [-.0001, .0002, -.02, .02, 0.]:
            delta = bounded_cartesian_normal_step(jac, normal, distance)
            actual = jac @ delta
            np.testing.assert_allclose(actual-normal*(normal @ actual), 0., atol=1e-12)
            self.assertLessEqual(np.abs(delta).max(), .012+1e-12)
            if distance:
                self.assertGreater(float(normal @ actual)*distance, 0.)
        # An inaccessible normal direction must hold, rather than slide.
        singular = np.array([[.1, 0., 0., 0.], [0., .1, 0., 0.], [0., 0., 0., 0.]])
        np.testing.assert_array_equal(bounded_cartesian_normal_step(singular, [0, 0, 1], .0001), np.zeros(4))


if __name__ == '__main__':
    unittest.main()
