from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from surface_manifold_gp import (  # noqa: E402
    GPManifoldConfig,
    equal_area_disk_queries,
    local_gp_point_features,
)


class SurfaceManifoldGPTest(unittest.TestCase):
    def test_equal_area_queries_are_bounded_and_deterministic(self) -> None:
        first = equal_area_disk_queries(8, 0.01)
        second = equal_area_disk_queries(8, 0.01)
        np.testing.assert_array_equal(first, second)
        self.assertLessEqual(float(np.linalg.norm(first, axis=1).max()), 0.01)
        np.testing.assert_array_equal(first[0], 0.0)

    def test_plane_reconstruction(self) -> None:
        x = np.linspace(-0.004, 0.004, 16)
        positions = np.stack((x, np.zeros_like(x), np.zeros_like(x)), axis=-1)
        normals = np.tile((0.0, 0.0, 1.0), (len(x), 1))
        features = local_gp_point_features(
            positions,
            normals,
            np.ones(len(x), dtype=bool),
            GPManifoldConfig(query_count=8, query_radius=0.003),
        )
        self.assertEqual(features.shape, (8, 10))
        np.testing.assert_allclose(features[:, 2], 0.0, atol=1.0e-7)
        np.testing.assert_allclose(features[:, 3:5], 0.0, atol=1.0e-7)
        np.testing.assert_allclose(features[:, 5], 1.0, atol=1.0e-7)
        np.testing.assert_array_equal(features[:, 9], 1.0)

    def test_no_contact_produces_invalid_zero_points(self) -> None:
        features = local_gp_point_features(
            np.zeros((16, 3)), np.zeros((16, 3)), np.zeros(16, dtype=bool)
        )
        np.testing.assert_array_equal(features, 0.0)


if __name__ == "__main__":
    unittest.main()
