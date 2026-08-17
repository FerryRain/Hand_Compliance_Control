from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from object_catalog import list_object_ids, load_object_config, object_local_aabb  # noqa: E402
from surface_mcc_finger import GeometrySurfaceOracle  # noqa: E402


class GeometrySurfaceOracleTest(unittest.TestCase):
    def test_all_catalog_objects_return_finite_unit_normals(self) -> None:
        for object_id in list_object_ids():
            with self.subTest(object_id=object_id):
                config = load_object_config(object_id)
                lower, upper = object_local_aabb(config)
                extent = upper - lower
                radius = float(np.linalg.norm(extent)) + 0.1
                query = radius * np.asarray(
                    (
                        (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0),
                        (0.0, 1.0, 0.0), (0.0, -1.0, 0.0),
                        (0.0, 0.0, 1.0), (0.0, 0.0, -1.0),
                    )
                )
                oracle = GeometrySurfaceOracle(config)
                result = oracle.observe(query)
                self.assertTrue(np.all(np.isfinite(result.points_world)))
                self.assertTrue(np.all(np.isfinite(result.normals_world)))
                self.assertTrue(np.all(result.signed_distance > 0.0))
                np.testing.assert_allclose(
                    np.linalg.norm(result.normals_world, axis=-1), 1.0, atol=1e-6
                )
                # The outward normal must point generally toward an exterior
                # query, not into the object or an internal compound seam.
                direction = query - result.points_world
                self.assertTrue(
                    np.all(np.einsum("ij,ij->i", direction, result.normals_world) > 0.0)
                )

    def test_pose_update_rotates_points_and_normals(self) -> None:
        oracle = GeometrySurfaceOracle(load_object_config("ellipsoid_medium"))
        query_local = np.asarray(((0.30, 0.0, 0.0),))
        base = oracle.observe(query_local)
        half_angle = np.pi / 4.0
        quaternion = np.asarray((np.cos(half_angle), 0.0, 0.0, np.sin(half_angle)))
        center = np.asarray((0.4, -0.2, 0.7))
        oracle.set_pose(center, quaternion)
        query_world = center + np.asarray(((0.0, 0.30, 0.0),))
        moved = oracle.observe(query_world)
        expected_point = center + np.asarray(((-base.points_world[0, 1], base.points_world[0, 0], 0.0)))
        expected_normal = np.asarray(((-base.normals_world[0, 1], base.normals_world[0, 0], 0.0)))
        np.testing.assert_allclose(moved.points_world[0], expected_point, atol=2e-5)
        np.testing.assert_allclose(moved.normals_world[0], expected_normal, atol=2e-5)

    def test_inside_queries_project_to_real_boundaries(self) -> None:
        capsule = GeometrySurfaceOracle(load_object_config("capsule_medium"))
        capsule_result = capsule.observe(np.zeros((1, 3)))
        self.assertLess(float(capsule_result.signed_distance[0]), 0.0)
        self.assertGreater(float(np.linalg.norm(capsule_result.points_world[0])), 0.1)

        rounded = GeometrySurfaceOracle(load_object_config("rounded_box_medium"))
        rounded_result = rounded.observe(np.zeros((1, 3)))
        self.assertLess(float(rounded_result.signed_distance[0]), 0.0)
        self.assertGreater(float(np.linalg.norm(rounded_result.points_world[0])), 0.08)


if __name__ == "__main__":
    unittest.main()
