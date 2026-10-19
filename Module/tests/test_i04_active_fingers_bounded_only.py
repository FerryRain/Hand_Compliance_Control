"""The solver probe adapter retains the baseline's lifecycle semantics."""
import unittest
from unittest.mock import patch

import numpy as np

from Module.tests import test_i04_active_fingers as native_fixture
from Module.i04_geometric_traversal.active_fingers import ActiveFingerSearch
from Module.i04_geometric_traversal.active_fingers_bounded_only import BoundedOnlyFingerSearch


class BoundedOnlyLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): native_fixture.ActiveFingerNativeGeometryTests.setUpClass()

    def test_all_non_solver_phases_and_lifecycle_match_original(self):
        fixture = native_fixture.ActiveFingerNativeGeometryTests(); fixture.setUp()
        h, data, q = fixture.h, fixture.data, fixture.q
        original, candidate = ActiveFingerSearch(h, fixture.graph), BoundedOnlyFingerSearch(h, fixture.graph)
        forces, normals, active = np.zeros(4), np.zeros((4, 3)), np.zeros(4, dtype=bool)
        for obj in (original, candidate):
            obj.rebase(q)
            obj.bias[:] = .001
            obj.search_distance[:] = .004
            obj.loss_age[:] = .4
        for enabled, measured, load in ((True, True, 1.), (True, False, 8.), (False, False, 0.), (True, False, 0.)):
            active[:] = measured; forces[:] = load
            outputs = [obj.update(data, q, forces, normals, active, enabled=enabled) for obj in (original, candidate)]
            np.testing.assert_array_equal(outputs[0][0], outputs[1][0])
            for key in outputs[0][1]: self.assertEqual(outputs[0][1][key], outputs[1][1][key], key)
        original.rebase(q); candidate.rebase(q)
        for key in ('bias', 'search_distance', 'loss_age', 'was_active', 'target_valid'):
            np.testing.assert_array_equal(getattr(original, key), getattr(candidate, key))

    def test_native_selected_step_has_original_bounds_and_sensor_authority(self):
        fixture = native_fixture.ActiveFingerNativeGeometryTests(); fixture.setUp()
        search = BoundedOnlyFingerSearch(fixture.h, fixture.graph)
        active = np.array([False, True, True, True])
        forces = active.astype(float); normals = np.tile([0., 0., 1.], (4, 1))
        before = fixture.data.qpos.copy()
        for _ in range(50):
            result, diag = search.update(fixture.data, fixture.q, forces, normals, active)
        self.assertTrue(diag['seeking'][0])
        self.assertFalse(any(diag['seeking'][1:]))
        np.testing.assert_array_equal(result[:7], fixture.q[:7])
        np.testing.assert_array_equal(result[11:], fixture.q[11:])
        np.testing.assert_array_equal(fixture.data.qpos, before)
        self.assertEqual(diag['measured_active'], active.tolist())
        self.assertLessEqual(np.max(abs(np.asarray(diag['search_delta_rad']))), .001+1e-12)
        self.assertLessEqual(diag['bounded_solver'][0]['cartesian_step_m'], 8e-6+1e-12)


if __name__ == '__main__': unittest.main()
