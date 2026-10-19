"""Explicit targets must be points on their claimed original mesh triangles."""
import unittest

import numpy as np
import trimesh

from Module.i04_geometric_traversal.explicit_surface_finger_planner import validate_mesh_face_point


class OriginalMeshTargetValidationTests(unittest.TestCase):
    def setUp(self):
        self.mesh = trimesh.Trimesh(vertices=[[0., 0., 0.], [.1, 0., 0.], [0., .1, 0.]],
                                   faces=[[0, 1, 2]], process=False)

    def test_real_face_point_and_outward_normal_are_preserved(self):
        point = np.array([.02, .03, 0.])
        actual, normal, barycentric = validate_mesh_face_point(self.mesh, point, 0)
        np.testing.assert_array_equal(actual, point)
        np.testing.assert_array_equal(normal, [0., 0., 1.])
        np.testing.assert_allclose(barycentric, [.5, .2, .3])
        actual[:] = 1.
        np.testing.assert_array_equal(point, [.02, .03, 0.])

    def test_off_surface_or_outside_triangle_cannot_be_labeled_mesh_target(self):
        for point in ([.02, .03, .001], [.09, .09, 0.], [-.001, .03, 0.], [float('nan'), 0., 0.]):
            with self.assertRaises(ValueError): validate_mesh_face_point(self.mesh, point, 0)
        with self.assertRaises(ValueError): validate_mesh_face_point(self.mesh, [.02, .03, 0.], 1)


if __name__ == '__main__': unittest.main()
