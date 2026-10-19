import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from Module.i04_geometric_traversal.finger_reconfiguration_planner import (
    ellipsoid_support_derivative, FingerReconfigurationConfig, FingerReconfigurationPath,
)


class FingerReconfigurationTests(unittest.TestCase):
    def test_orientation_dependent_support_jacobian_matches_finite_difference(self):
        frame = Rotation.from_rotvec([.4, -.7, .3]).as_matrix()
        sizes = np.array([.012, .008, .002])
        normal = np.array([.2, .5, .8]); normal /= np.linalg.norm(normal)
        axes = np.array([[1., .2, -.3, .8], [.3, 1., .4, -.2], [.2, -.4, 1., .1]])
        analytic = ellipsoid_support_derivative(frame, sizes, normal, axes)
        def support(rotation):
            local = rotation.T @ normal
            return rotation @ (sizes**2 * local / np.linalg.norm(sizes*local))
        eps = 1e-6
        finite = np.column_stack([(support(Rotation.from_rotvec(eps*axis).as_matrix()@frame)
                                   -support(Rotation.from_rotvec(-eps*axis).as_matrix()@frame))/(2*eps)
                                  for axis in axes.T])
        np.testing.assert_allclose(analytic, finite, atol=2e-11, rtol=1e-7)
        assert np.max(abs(analytic)) > .001  # The orientation term is materially nonzero.

    def test_sphere_orientation_has_zero_support_derivative(self):
        result = ellipsoid_support_derivative(Rotation.from_euler('xyz', [.1,.2,.3]).as_matrix(),
                                               np.ones(3)*.01, np.array([0.,0.,1.]), np.eye(3))
        np.testing.assert_allclose(result, 0., atol=1e-16)

    def test_slow_path_changes_only_selected_finger(self):
        root = np.arange(23)*.01
        goal = root.copy(); goal[11:15] += [.1, 1.5, -.2, .3]
        limit = .15
        duration = 1.875*np.max(abs(goal-root))/limit
        path = FingerReconfigurationPath(np.stack([root,goal]), np.array([duration]),
                                          np.zeros(7),np.zeros(7),0.,2,1)
        samples = np.array([path.sample(t) for t in np.linspace(0,duration,501)])
        fixed = np.r_[np.arange(11),np.arange(15,23)]
        np.testing.assert_array_equal(samples[:,fixed],np.tile(root[fixed],(len(samples),1)))
        assert np.max(abs(np.diff(samples,axis=0)))/(duration/500) <= limit+1e-8

    def test_rejects_excessive_inward_target(self):
        with self.assertRaises(ValueError):
            FingerReconfigurationConfig(target_clearance_m=-.005)


if __name__ == '__main__':
    unittest.main()
