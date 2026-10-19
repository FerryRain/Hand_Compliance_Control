"""Primary normal progress, original constraints, and inherited episode rules."""
from types import SimpleNamespace
from dataclasses import replace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np
import trimesh

from Module.tests import test_i04_active_fingers_v3 as lifecycle_fixture
from Module.i04_geometric_traversal.active_fingers import ActiveFingerSearchConfig
from Module.i04_geometric_traversal.active_fingers_v4 import bounded_support_finger_step, support_point_jacobians
from Module.i04_geometric_traversal.active_fingers_v5 import (
    ActiveFingerSearchV5, bounded_normal_finger_step, closest_finger_pair,
)


class V5LifecycleTests(lifecycle_fixture.ActiveFingerV3Tests):
    """Same V3 lifecycle with the fixture's synthetic nearby geometry only."""

    def setUp(self):
        super().setUp()
        self.search = ActiveFingerSearchV5(self.h, self.graph)
        def nearby_fixture_pair(h, data, finger):
            # The lifecycle fixture uses a nearby synthetic surface. This
            # mock changes geometric proposals only, never the plant/sensors.
            geom = int(h.tip_geom_ids[finger])
            points, _, faces = trimesh.proximity.closest_point(self.graph.mesh,
                (data.geom_xpos[geom]-h.object_position_m)[None])
            normal = np.asarray(self.graph.mesh.face_normals)[int(faces[0])]
            return dict(gap_m=.006, object_point_world_m=points[0]+h.object_position_m,
                pad_point_world_m=data.geom_xpos[geom].copy(), normal_world=normal, object_geom_id=0)
        mock = patch('Module.i04_geometric_traversal.active_fingers_v5.closest_finger_pair', nearby_fixture_pair)
        mock.start(); self.addCleanup(mock.stop)

    def test_flicker_and_stroke_rebase_cannot_refresh_exhausted_search(self):
        self.search = ActiveFingerSearchV5(self.h, self.graph, config=replace(
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


class NormalBoundTests(unittest.TestCase):
    def solve(self, center, support, normal, *, nominal=None, bias=None, lower=None, upper=None):
        return bounded_normal_finger_step(center, support, normal,
            np.zeros(4) if nominal is None else nominal, np.zeros(4) if bias is None else bias,
            np.full(4, -1.) if lower is None else lower, np.ones(4) if upper is None else upper)

    def test_tangential_vector_error_cannot_outvote_normal_gap(self):
        # A nearly singular support map can fit the large tangent error by
        # opening its smaller normal error, even with the correct Jacobian.
        support = np.array([[.01, 0., 0., 0.], [.1, 0., 0., 0.], [0., .05, 0., 0.]])
        normal = np.array([1., 0., 0.])
        error = np.array([-.0006, .003, 0.])
        old, _ = bounded_support_finger_step(support, support, error*8e-6/np.linalg.norm(error),
            np.zeros(4), np.zeros(4), np.full(4, -1.), np.ones(4))
        new, diag = self.solve(support, support, normal)
        self.assertGreater(normal@(support@old), 0.)
        self.assertTrue(diag['feasible'])
        self.assertLess(normal@(support@new), -5e-7)
        self.assertLessEqual(np.linalg.norm(support@new), 8e-6+1e-12)

    def test_blocked_inward_direction_explicitly_holds_without_claiming_repair(self):
        delta, diag = self.solve(np.eye(3, 4), np.eye(3, 4), [1., 0., 0.], lower=np.zeros(4))
        np.testing.assert_array_equal(delta, np.zeros(4))
        self.assertTrue(diag['feasible'])
        self.assertEqual(diag['reason'], 'NO_LEGAL_INWARD_STEP_FOUND')
        self.assertEqual(diag['charged_step_m'], 0.)

    def test_nonzero_outward_repair_is_not_disguised_as_legal_zero_hold(self):
        delta, diag = self.solve(np.eye(3, 4)*.001, np.eye(3, 4)*.001, [1., 0., 0.],
                                lower=np.array([.0005, -1., -1., -1.]))
        self.assertFalse(diag['feasible'])
        self.assertEqual(diag['reason'], 'NO_FEASIBLE_INWARD_LIMIT_REPAIR')
        np.testing.assert_array_equal(delta, np.zeros(4))

    def test_dual_cartesian_joint_bias_limits_all_remain_active(self):
        center = np.array([[.08, .02, 0., 0.], [0., .10, .01, 0.], [0., 0., .08, .03]])
        support = center.copy(); support[:, 1] *= .2
        bias = np.array([-.4998, .4999, 0., 0.])
        delta, diag = self.solve(center, support, [1., .2, .1], bias=bias)
        self.assertTrue(diag['feasible'])
        self.assertLessEqual(np.linalg.norm(center@delta), 8e-6+1e-12)
        self.assertLessEqual(np.linalg.norm(support@delta), 8e-6+1e-12)
        self.assertLessEqual(np.max(abs(delta)), .001+1e-12)
        self.assertLessEqual(np.max(abs(bias+delta)), .5+1e-12)
        self.assertLess(diag['predicted_gap_change_m'], 0.)
        self.assertAlmostEqual(diag['charged_step_m'], max(np.linalg.norm(center@delta),
                                                          np.linalg.norm(support@delta)), delta=1e-14)

    def test_invalid_input_cannot_generate_motion(self):
        delta, diag = self.solve(np.eye(3, 4), np.eye(3, 4), [np.nan, 0., 1.])
        self.assertFalse(diag['feasible']); np.testing.assert_array_equal(delta, np.zeros(4))

    def test_all_joint_directions_fixed_is_a_finite_explicit_hold(self):
        delta, diag = self.solve(np.eye(3, 4), np.eye(3, 4), [1., 0., 0.],
                                lower=np.zeros(4), upper=np.zeros(4))
        self.assertTrue(diag['feasible'])
        self.assertFalse(diag['optimality_confirmed'])
        np.testing.assert_array_equal(delta, np.zeros(4))


class NativeGapGradientTests(unittest.TestCase):
    def setUp(self):
        xml = '''<mujoco><option gravity="0 0 0"/><worldbody>
          <geom name="object" type="plane" size="1 1 .01" pos="0 0 .02"/>
          <body pos="0 0 .06"><joint axis="0 1 0"/>
           <inertial pos="0 0 0" mass=".01" diaginertia=".001 .001 .001"/>
           <body pos=".02 0 .01"><joint axis="1 0 0"/>
            <inertial pos="0 0 0" mass=".01" diaginertia=".001 .001 .001"/>
            <body pos=".02 .01 .01"><joint axis="0 1 0"/>
             <inertial pos="0 0 0" mass=".01" diaginertia=".001 .001 .001"/>
             <body pos=".01 0 .01"><joint axis="1 0 0"/>
              <geom name="pad" type="ellipsoid" size=".014 .022 .010" pos=".01 .01 .01"/>
             </body></body></body></body></worldbody></mujoco>'''
        model = mujoco.MjModel.from_xml_string(xml)
        self.h = SimpleNamespace(model=model, hand_dof_adrs=np.arange(4),
            tip_geom_ids=np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'pad')]),
            object_geom_ids=np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'object')]))
        self.data = mujoco.MjData(model); self.data.qpos[:] = [.25, -.35, .2, .15]
        mujoco.mj_forward(model, self.data)

    def test_closest_pair_gap_gradient_matches_native_finite_difference(self):
        pair = closest_finger_pair(self.h, self.data, 0)
        self.assertGreater(pair['gap_m'], 0.)
        np.testing.assert_allclose(pair['normal_world'], [0., 0., 1.], atol=1e-10)
        center, support = support_point_jacobians(self.h, self.data, 0, np.arange(4), pair['normal_world'])
        gradient = pair['normal_world']@support
        q = self.data.qpos.copy(); numeric = np.zeros(4)
        for j in range(4):
            self.data.qpos[:] = q; self.data.qpos[j] += 1e-6; mujoco.mj_forward(self.h.model, self.data)
            plus = closest_finger_pair(self.h, self.data, 0)['gap_m']
            self.data.qpos[:] = q; self.data.qpos[j] -= 1e-6; mujoco.mj_forward(self.h.model, self.data)
            minus = closest_finger_pair(self.h, self.data, 0)['gap_m']
            numeric[j] = (plus-minus)/2e-6
        np.testing.assert_allclose(gradient, numeric, atol=2e-9, rtol=1e-6)
        self.assertGreater(np.max(abs(pair['normal_world']@center-numeric)), .0005)

    def test_bounded_inward_step_reduces_actual_geometry_gap_without_native_step(self):
        before = closest_finger_pair(self.h, self.data, 0)
        center, support = support_point_jacobians(self.h, self.data, 0, np.arange(4), before['normal_world'])
        q = self.data.qpos.copy()
        delta, diag = bounded_normal_finger_step(center, support, before['normal_world'],
            q, np.zeros(4), np.full(4, -2.), np.full(4, 2.))
        self.assertTrue(diag['feasible'])
        self.data.qpos[:] = q+delta; mujoco.mj_forward(self.h.model, self.data)
        after = closest_finger_pair(self.h, self.data, 0)
        self.assertLess(after['gap_m'], before['gap_m'])
        self.assertEqual(self.data.time, 0.)  # Only geometry, never contact evidence.

    def test_shallow_overlap_keeps_outward_normal_sign(self):
        pair = closest_finger_pair(self.h, self.data, 0)
        obj = int(self.h.object_geom_ids[0])
        self.h.model.geom_pos[obj, 2] += pair['gap_m']+.0001
        mujoco.mj_forward(self.h.model, self.data)
        overlap = closest_finger_pair(self.h, self.data, 0)
        self.assertLess(overlap['gap_m'], 0.)
        np.testing.assert_allclose(overlap['normal_world'], [0., 0., 1.], atol=1e-10)
        self.assertAlmostEqual(overlap['gap_m'], -.0001, delta=1e-10)


if __name__ == '__main__': unittest.main()
