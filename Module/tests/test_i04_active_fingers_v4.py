"""Native support derivatives, dual speed limits, and unchanged V3 episodes."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from Module.tests import test_i04_active_fingers as native_fixture
from Module.tests import test_i04_active_fingers_v3 as lifecycle_fixture
from Module.i04_geometric_traversal.active_fingers import ActiveFingerSearchConfig
from Module.i04_geometric_traversal.active_fingers_v4 import (
    ActiveFingerSearchV4, bounded_support_finger_step, support_point_jacobians,
)
from Module.i04_geometric_traversal.full_surface import pad_support_vector
from Module.i04_geometric_traversal.physics import set_joint_state


class V4InheritedLifecycleTests(lifecycle_fixture.ActiveFingerV3Tests):
    """The same native V3 behavioral contract must hold for the new geometry."""

    def setUp(self):
        super().setUp()
        self.search = ActiveFingerSearchV4(self.h, self.graph)

    def test_flicker_and_stroke_rebase_cannot_refresh_exhausted_search(self):
        self.search = ActiveFingerSearchV4(self.h, self.graph, config=replace(
            ActiveFingerSearchConfig(), maximum_search_time_s=.12))
        result, _ = self.prime(count=55)
        for _ in range(8): _, diag = self.tick()
        self.assertEqual(diag['state'], ['SEARCH_LIMIT']*4)
        self.active[:] = True; self.forces[:] = 1.; self.normals[:] = [0., 0., 1.]
        for _ in range(10): self.tick()
        self.active[:] = False; self.forces[:] = 0.
        self.search.rebase(result)
        _, diag = self.tick(nominal=result)
        self.assertEqual(diag['state'], ['SEARCH_LIMIT']*4)


class SupportNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        native_fixture.ActiveFingerNativeGeometryTests.setUpClass()

    def setUp(self):
        fixture = native_fixture.ActiveFingerNativeGeometryTests()
        fixture.setUp()
        self.h, self.data, self.graph, self.q = fixture.h, fixture.data, fixture.graph, fixture.q.copy()
        self.search = ActiveFingerSearchV4(self.h, self.graph)

    def support(self, data, finger, normal):
        geom = int(self.h.tip_geom_ids[finger])
        return data.geom_xpos[geom]-pad_support_vector(self.h.model, geom,
            data.geom_xmat[geom].reshape(3, 3), normal)

    def test_true_native_fk_central_difference_matches_all_finger_support_jacobians(self):
        scratch = mujoco.MjData(self.h.model)
        h = 1e-6
        maximum_missing_orientation_term = 0.
        for finger, columns in enumerate(self.search.columns):
            normal = np.array([-.9, .2+finger*.1, .17]); normal /= np.linalg.norm(normal)
            center, analytic = support_point_jacobians(self.h, self.data, finger, columns, normal)
            finite = np.zeros((3, 4))
            for j, column in enumerate(columns):
                plus, minus = self.q.copy(), self.q.copy()
                plus[column+7] += h; minus[column+7] -= h
                set_joint_state(self.h, scratch, plus); a = self.support(scratch, finger, normal)
                set_joint_state(self.h, scratch, minus); b = self.support(scratch, finger, normal)
                finite[:, j] = (a-b)/(2*h)
            np.testing.assert_allclose(analytic, finite, atol=2e-9, rtol=1e-6)
            maximum_missing_orientation_term = max(maximum_missing_orientation_term,
                                                    float(np.max(abs(center-finite))))
        self.assertGreater(maximum_missing_orientation_term, .005)

    def test_each_tick_refreshes_support_against_same_stored_mesh_point(self):
        self.search.rebase(self.q)
        self.search._refresh_targets(self.data)
        previous_points = self.search.surface_points.copy()
        previous_normal = self.search.target_normals.copy()
        previous_target = self.search.targets.copy()
        refresh_count = self.search.refresh_count
        changed = self.q.copy(); changed[9] += .03
        set_joint_state(self.h, self.data, changed)
        with patch.object(self.search, '_refresh_targets', side_effect=AssertionError('Unexpected mesh refresh')):
            _, diag = self.search.update(self.data, self.q, np.zeros(4), np.zeros((4, 3)),
                                         np.zeros(4, dtype=bool), search_mask=[True, False, False, False])
        np.testing.assert_array_equal(self.search.surface_points, previous_points)
        np.testing.assert_array_equal(self.search.target_normals, previous_normal)
        self.assertEqual(self.search.refresh_count, refresh_count)
        self.assertGreater(np.linalg.norm(self.search.targets[0]-previous_target[0]), 1e-5)
        n = previous_normal[0]
        error = self.search.targets[0]-self.data.geom_xpos[int(self.h.tip_geom_ids[0])]
        np.testing.assert_allclose(error, previous_points[0]+self.search.config.target_clearance_m*n
                                   - self.support(self.data, 0, n), atol=1e-15)
        self.assertFalse(diag['nonlinear_contact_progress_guaranteed'])

    def test_real_recorded_center_step_can_decrease_center_error_but_open_support_gap(self):
        # Fresh v3.6 middle sequence 4 at t=429.000 s. A static kinematics
        # counterexample, not simulated movement or a new contact assertion.
        q = np.array([2.311652975116931, -1.7151299490270497, .46570467369243745,
            -2.820337687929566, -.861968703166224, .5784168297858816, -2.892670101108328,
            .4700410601894369, 2.0898038285586855, -.04210429690436389, -.3510000768854459,
            .38614178071486654, 2.0357865270180286, -.042137296518542415, -.35100208539767275,
            .1338871008905255, 2.214044272003067, .06747567665457894, -.34585100644924976,
            2.018603496454428, -.018346479796810446, -.21778371449875045, -.3903230015051981])
        normal = np.array([-.9887402704751557, .1466393366780286, .02982586929737444])
        target = np.array([.15877442020409327, -.012423131183151316, .5157464484909778])
        delta = np.array([-1.3188714312918465e-5, 6.679816175199666e-5, -2.0798356156092785e-6, 0.])
        set_joint_state(self.h, self.data, q)
        geom = int(self.h.tip_geom_ids[1])
        center_before, support_before = self.data.geom_xpos[geom].copy(), self.support(self.data, 1, normal)
        center_jac, support_jac = support_point_jacobians(self.h, self.data, 1, self.search.columns[1], normal)
        moved = q.copy(); moved[11:15] += delta
        scratch = mujoco.MjData(self.h.model); set_joint_state(self.h, scratch, moved)
        support_delta = self.support(scratch, 1, normal)-support_before
        self.assertLess(np.linalg.norm(target-scratch.geom_xpos[geom])-np.linalg.norm(target-center_before), -5e-6)
        self.assertGreater(normal@support_delta, 8e-7)
        np.testing.assert_allclose(support_jac@delta, support_delta, atol=1e-9)
        self.assertGreater(np.linalg.norm((center_jac-support_jac)@delta), 5e-7)

    def test_search_charge_bounds_both_native_jacobians_without_changing_plant(self):
        self.search.loss_age[:] = 1.
        before = (self.data.qpos.copy(), self.data.qvel.copy(), self.data.ctrl.copy(), self.data.time)
        previous = np.zeros(4)
        for _ in range(60):
            _, diag = self.search.update(self.data, self.q, np.zeros(4), np.zeros((4, 3)),
                                         np.zeros(4, dtype=bool))
            delta = np.asarray(diag['search_delta_rad'])
            charged = np.asarray(diag['search_distance_m'])-previous
            for finger, columns in enumerate(self.search.columns):
                center, support = self.search._support_jacobians(self.data, finger)
                center_step = np.linalg.norm(center@delta[columns])
                support_step = np.linalg.norm(support@delta[columns])
                self.assertLessEqual(max(center_step, support_step), 8e-6+1e-12)
                self.assertAlmostEqual(charged[finger], max(center_step, support_step), delta=1e-14)
            previous = np.asarray(diag['search_distance_m'])
        for actual, original in zip((self.data.qpos, self.data.qvel, self.data.ctrl, self.data.time), before):
            np.testing.assert_array_equal(actual, original)


class DualBoundStepTests(unittest.TestCase):
    def solve(self, center, support, *, nominal=None, bias=None, lower=None, upper=None):
        return bounded_support_finger_step(center, support, np.array([8e-6, 0., 0.]),
            np.zeros(4) if nominal is None else nominal, np.zeros(4) if bias is None else bias,
            np.full(4, -1.) if lower is None else lower, np.ones(4) if upper is None else upper)

    def test_rotational_cancellation_cannot_relax_original_center_speed_cap(self):
        support, center = np.eye(3, 4)*.001, np.eye(3, 4)*.1
        delta, diag = self.solve(center, support)
        self.assertTrue(diag['feasible'])
        self.assertEqual(diag['reason'], 'SUPPORT_DLS_CENTER_SCALED')
        self.assertLessEqual(np.linalg.norm(center@delta), 8e-6+1e-12)
        self.assertLessEqual(np.linalg.norm(support@delta), 8e-6+1e-12)
        self.assertAlmostEqual(diag['charged_step_m'], np.linalg.norm(center@delta), delta=1e-14)
        self.assertFalse(diag['optimality_confirmed'])

    def test_nonzero_repair_cannot_be_invalidated_by_second_speed_bound(self):
        lower = np.array([.0005, -1., -1., -1.])
        delta, diag = self.solve(np.eye(3, 4), np.eye(3, 4)*.001, lower=lower)
        self.assertFalse(diag['feasible'])
        self.assertEqual(diag['reason'], 'CENTER_BOUND_SCALE_VIOLATES_REQUIRED_REPAIR')
        np.testing.assert_array_equal(delta, np.zeros(4))

    def test_feasible_nonzero_repair_preserves_joint_bias_and_step_bounds(self):
        lower = np.array([.0005, -1., -1., -1.])
        delta, diag = self.solve(np.eye(3, 4)*.002, np.eye(3, 4)*.001, lower=lower)
        self.assertTrue(diag['feasible'])
        self.assertGreaterEqual(delta[0], .0005)
        self.assertLessEqual(np.max(abs(delta)), .001+1e-12)
        self.assertLessEqual(diag['charged_step_m'], 8e-6+1e-12)

    def test_nonfinite_jacobian_is_explicit_infeasible_hold(self):
        delta, diag = self.solve(np.full((3, 4), np.nan), np.eye(3, 4))
        self.assertFalse(diag['feasible'])
        np.testing.assert_array_equal(delta, np.zeros(4))


if __name__ == '__main__': unittest.main()
