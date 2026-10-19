"""Native broad-phase semantics and unchanged V7 controller method bodies."""
import ast
import inspect
import textwrap
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from Module.i04_geometric_traversal.active_fingers_v5 import ActiveFingerSearchV5, closest_finger_pair
from Module.i04_geometric_traversal.active_fingers_v6 import ActiveFingerSearchV6
from Module.i04_geometric_traversal.active_fingers_v7 import ActiveFingerSearchV7
from Module.i04_geometric_traversal.native_closest_pair import closest_finger_pair_pruned


class NativeClosestPairTests(unittest.TestCase):
    def scene(self, objects, *, pad_type='ellipsoid', pad_size='.01 .012 .008'):
        geoms = ''.join(objects)
        model = mujoco.MjModel.from_xml_string(f'<mujoco><worldbody>'
            f'<geom name="pad" type="{pad_type}" size="{pad_size}"/>{geoms}</worldbody></mujoco>')
        data = mujoco.MjData(model); mujoco.mj_forward(model, data)
        return SimpleNamespace(model=model, tip_geom_ids=np.zeros(4, dtype=int),
            object_geom_ids=np.arange(1, model.ngeom)), data

    def equal_pair(self, expected, actual):
        if expected is None:
            self.assertIsNone(actual); return
        self.assertEqual(set(expected), set(actual))
        for key in expected: np.testing.assert_array_equal(expected[key], actual[key])

    def test_finite_native_candidates_prune_far_geoms_without_changing_any_result(self):
        objects = ['<geom type="sphere" size=".01" pos=".035 0 0"/>']
        objects += [f'<geom type="sphere" size=".01" pos="{.07+.01*i} 0 0"/>' for i in range(30)]
        h, data = self.scene(objects)
        original_q, original_pos, original_radii = data.qpos.copy(), data.geom_xpos.copy(), h.model.geom_rbound.copy()
        expected = closest_finger_pair(h, data, 0)
        native = mujoco.mj_geomDistance
        with patch.object(mujoco, 'mj_geomDistance', wraps=native) as query:
            actual = closest_finger_pair_pruned(h, data, 0)
        self.equal_pair(expected, actual)
        self.assertEqual(query.call_count, 1)
        np.testing.assert_array_equal(data.qpos, original_q)
        np.testing.assert_array_equal(data.geom_xpos, original_pos)
        np.testing.assert_array_equal(h.model.geom_rbound, original_radii)

    def test_equal_gap_retains_first_geom_in_original_order(self):
        h, data = self.scene(['<geom type="sphere" size=".01" pos=".04 0 0"/>']*2,
                             pad_type='sphere', pad_size='.01')
        for order in ([1, 2], [2, 1], [2, 1, 2]):
            h.object_geom_ids = np.asarray(order)
            native = mujoco.mj_geomDistance
            with patch.object(mujoco, 'mj_geomDistance', wraps=native) as query:
                actual = closest_finger_pair_pruned(h, data, 0)
            self.assertEqual(query.call_count, len(order))
            self.assertEqual(actual['object_geom_id'], order[0])
            self.equal_pair(closest_finger_pair(h, data, 0), actual)

    def test_signed_overlap_and_outward_normal_are_preserved(self):
        h, data = self.scene(['<geom type="sphere" size=".01" pos=".019 0 0"/>',
                              '<geom type="sphere" size=".01" pos=".018 0 0"/>',
                              '<geom type="sphere" size=".01" pos=".10 0 0"/>'],
                             pad_type='sphere', pad_size='.01')
        actual = closest_finger_pair_pruned(h, data, 0)
        self.equal_pair(closest_finger_pair(h, data, 0), actual)
        self.assertLess(actual['gap_m'], 0.)
        self.assertEqual(actual['object_geom_id'], 2)
        self.assertGreater(actual['normal_world']@(data.geom_xpos[0]-actual['object_point_world_m']), 0.)

    def test_distance_max_no_result_still_queries_all_candidates(self):
        h, data = self.scene(['<geom type="sphere" size=".01" pos="1 0 0"/>',
                              '<geom type="sphere" size=".01" pos="2 0 0"/>'])
        native = mujoco.mj_geomDistance
        with patch.object(mujoco, 'mj_geomDistance', wraps=native) as query:
            actual = closest_finger_pair_pruned(h, data, 0)
        self.equal_pair(closest_finger_pair(h, data, 0), actual)
        self.assertIsNone(actual)
        self.assertEqual(query.call_count, 2)
        self.assertTrue(all(call.args[4] == .2 for call in query.call_args_list))

    def test_disabled_and_empty_geometry_preserve_none(self):
        h, data = self.scene(['<geom type="sphere" size=".01" pos=".02 0 0" contype="0"/>'])
        self.equal_pair(closest_finger_pair(h, data, 0), closest_finger_pair_pruned(h, data, 0))
        h.object_geom_ids = np.array([], dtype=int)
        self.assertIsNone(closest_finger_pair_pruned(h, data, 0))

    def test_unbounded_plane_is_not_rejected_by_zero_radius(self):
        h, data = self.scene(['<geom type="sphere" size=".01" pos=".05 0 0"/>',
                              '<geom type="plane" size="1 1 .01" pos="0 0 -.02"/>'])
        native = mujoco.mj_geomDistance
        with patch.object(mujoco, 'mj_geomDistance', wraps=native) as query:
            actual = closest_finger_pair_pruned(h, data, 0)
        self.equal_pair(closest_finger_pair(h, data, 0), actual)
        self.assertEqual(query.call_count, 2)
        self.assertEqual(actual['object_geom_id'], 2)

    def test_current_pose_recomputes_bounds_instead_of_reusing_old_query(self):
        h, data = self.scene(['<geom type="sphere" size=".01" pos=".04 0 0"/>',
                              '<geom type="sphere" size=".01" pos=".10 0 0"/>'])
        self.assertEqual(closest_finger_pair_pruned(h, data, 0)['object_geom_id'], 1)
        h.model.geom_pos[2, 0] = .025; mujoco.mj_forward(h.model, data)
        actual = closest_finger_pair_pruned(h, data, 0)
        self.equal_pair(closest_finger_pair(h, data, 0), actual)
        self.assertEqual(actual['object_geom_id'], 2)

    def test_mesh_rounding_cushion_cannot_prune_a_subnanometre_better_result(self):
        h, data = self.scene(['<geom type="sphere" size=".01" pos=".04 0 0"/>',
                              '<geom type="sphere" size=".01" pos=".0399999999 0 0"/>'],
                             pad_type='sphere', pad_size='.01')
        # Mimic the <1 nm radius underestimate observed in compiled mustard
        # float32 mesh vertices. A double-precision ulp guard alone misses it.
        h.model.geom_rbound[2] -= 8e-10
        actual = closest_finger_pair_pruned(h, data, 0)
        self.equal_pair(closest_finger_pair(h, data, 0), actual)
        self.assertEqual(actual['object_geom_id'], 2)

    def test_invalid_or_unbounded_radius_falls_back_to_native_query(self):
        h, data = self.scene(['<geom type="sphere" size=".01" pos=".04 0 0"/>',
                              '<geom type="sphere" size=".01" pos=".10 0 0"/>'])
        native = mujoco.mj_geomDistance
        for radius in (np.nan, np.inf, 0., -1.):
            h.model.geom_rbound[2] = radius
            with self.subTest(radius=radius), np.errstate(invalid='ignore'), \
                    patch.object(mujoco, 'mj_geomDistance', wraps=native) as query:
                actual = closest_finger_pair_pruned(h, data, 0)
            self.equal_pair(closest_finger_pair(h, data, 0), actual)
            self.assertEqual(query.call_count, 2)

    def test_only_query_binding_changes_strategy_methods_and_update_is_inherited(self):
        self.assertIs(ActiveFingerSearchV7.update, ActiveFingerSearchV6.update)
        for owner, name in ((ActiveFingerSearchV5, '_update_orientation_targets'),
                            (ActiveFingerSearchV6, '_normal_search_step')):
            original = ast.parse(textwrap.dedent(inspect.getsource(getattr(owner, name))))
            candidate = ast.parse(textwrap.dedent(inspect.getsource(getattr(ActiveFingerSearchV7, name))))
            self.assertEqual(ast.dump(original), ast.dump(candidate))


if __name__ == '__main__': unittest.main()
