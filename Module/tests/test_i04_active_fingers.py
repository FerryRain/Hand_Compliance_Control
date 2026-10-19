"""Bounded independent finger search; sensor evidence remains authoritative."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np
import trimesh

from Module.fr3_leap import ARM_HOME_Q, HAND_NATURAL_Q, FullRobotModelConfig, build_full_robot
from Module.i04_geometric_traversal.active_fingers import ActiveFingerSearch, ActiveFingerSearchConfig
from Module.i04_geometric_traversal.full_surface import pad_support_vector
from Module.i04_geometric_traversal.physics import set_joint_state


class ActiveFingerNativeGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.h = build_full_robot(FullRobotModelConfig(surface='plane', gravity_m_s2=0.))
        cls.q = np.r_[ARM_HOME_Q, HAND_NATURAL_Q]

    def setUp(self):
        self.data = mujoco.MjData(self.h.model)
        set_joint_state(self.h, self.data, self.q)
        centers = self.data.geom_xpos[self.h.tip_geom_ids].copy()
        normal = np.array([0., 0., 1.])
        radius = [pad_support_vector(self.h.model, int(g),
                    self.data.geom_xmat[int(g)].reshape(3, 3), normal)[2]
                  for g in self.h.tip_geom_ids]
        self.plane_z = float(np.min(centers[:, 2] - radius) - .006)
        mesh = trimesh.creation.box(extents=[2., 2., .02])
        mesh.apply_translation([centers[:, 0].mean() - self.h.object_position_m[0],
                                centers[:, 1].mean() - self.h.object_position_m[1],
                                self.plane_z - self.h.object_position_m[2] - .01])
        self.graph = SimpleNamespace(mesh=mesh)
        self.search = ActiveFingerSearch(self.h, self.graph)
        self.forces, self.normals, self.active = np.zeros(4), np.zeros((4, 3)), np.zeros(4, dtype=bool)

    def tick(self, search=None, nominal=None, enabled=True):
        return (search or self.search).update(self.data, self.q if nominal is None else nominal,
                                              self.forces, self.normals, self.active, enabled=enabled)

    def prime(self, count=80):
        for _ in range(count):
            result, diag = self.tick()
        self.assertTrue(any(diag['seeking']))
        return result, diag

    def test_only_lost_finger_moves_after_debounce_and_sensor_arrays_stay_real(self):
        self.active[:] = True; self.active[0] = False
        self.forces[:] = 1.; self.forces[0] = 0.
        self.normals[1:] = [0., 0., 1.]
        initial = (self.data.qpos.copy(), self.data.qvel.copy(), self.data.ctrl.copy(), self.data.time)
        sensors = [self.forces.copy(), self.normals.copy(), self.active.copy()]
        for _ in range(20):
            result, diag = self.tick()
            np.testing.assert_array_equal(result, self.q)
            self.assertFalse(any(diag['seeking']))
        for _ in range(20):
            result, diag = self.tick()
        self.assertTrue(diag['seeking'][0])
        self.assertFalse(any(diag['seeking'][1:]))
        np.testing.assert_array_equal(result[:7], self.q[:7])
        np.testing.assert_array_equal(result[11:], self.q[11:])
        self.assertGreater(np.linalg.norm(result[7:11] - self.q[7:11]), 0.)
        for actual, expected in zip((self.data.qpos, self.data.qvel, self.data.ctrl, self.data.time), initial):
            np.testing.assert_array_equal(actual, expected)
        for actual, expected in zip((self.forces, self.normals, self.active), sensors):
            np.testing.assert_array_equal(actual, expected)
        self.assertEqual(diag['measured_active'], self.active.tolist())

    def test_new_contact_keeps_flexed_shape_and_rebase_does_not_double_it(self):
        result, diag = self.prime()
        bias = np.array(diag['joint_bias_rad'])
        self.active[:] = True; self.forces[:] = 1.; self.normals[:] = [0., 0., 1.]
        held, contact = self.tick()
        np.testing.assert_array_equal(held, result)
        np.testing.assert_array_equal(contact['joint_bias_rad'], bias)
        self.assertEqual(contact['state'], ['CONTACT_HOLD'] * 4)
        self.assertFalse(any(contact['seeking']))
        self.assertTrue(any(contact['recontacted']))
        # A new trajectory is rooted at the attained hand reference.
        self.search.rebase(held)
        rerooted, fresh = self.tick(nominal=held)
        np.testing.assert_array_equal(rerooted, held)
        np.testing.assert_array_equal(fresh['joint_bias_rad'], np.zeros(16))

    def test_joint_bias_follows_moving_nominal_without_restarting_search_from_measured_q(self):
        result, diag = self.prime()
        bias = np.array(diag['joint_bias_rad'])
        self.active[:] = True; self.forces[:] = 1.; self.normals[:] = [0., 0., 1.]
        moved_nominal = self.q.copy(); moved_nominal[7:] += .003
        moved, _ = self.tick(nominal=moved_nominal)
        np.testing.assert_allclose(moved[7:], moved_nominal[7:] + bias, atol=1e-14)
        np.testing.assert_array_equal(moved[:7], moved_nominal[:7])
        # The measured plant was intentionally held still throughout.
        np.testing.assert_array_equal(self.data.qpos[self.h.hand_qpos_adrs], self.q[7:])

    def test_search_obeys_cartesian_joint_and_total_exploration_limits(self):
        config = replace(ActiveFingerSearchConfig(), maximum_search_distance_m=.0003)
        search = ActiveFingerSearch(self.h, self.graph, config=config)
        previous_distance = np.zeros(4)
        for _ in range(180):
            result, diag = self.tick(search=search)
            distance = np.array(diag['search_distance_m'])
            self.assertTrue(np.all(distance <= config.maximum_search_distance_m + 1e-12))
            self.assertTrue(np.all(distance - previous_distance <= config.speed_m_s * config.dt_s + 1e-12))
            self.assertLessEqual(np.max(abs(np.array(diag['search_delta_rad']))),
                                 config.maximum_search_joint_speed_rad_s * config.dt_s + 1e-12)
            self.assertLessEqual(np.max(abs(np.array(diag['joint_bias_rad']))), config.maximum_joint_bias_rad + 1e-12)
            self.assertTrue(np.all(result[7:] >= search.lower))
            self.assertTrue(np.all(result[7:] <= search.upper))
            previous_distance = distance
        self.assertIn('SEARCH_LIMIT', diag['state'])
        self.assertGreater(float(distance.max()), .00029)

    def test_force_pause_stops_all_search_and_air_move_clears_bias(self):
        result, before = self.prime()
        self.forces[0] = 8.; self.active[0] = True; self.normals[0] = [0., 0., 1.]
        paused, diag = self.tick()
        np.testing.assert_array_equal(paused, result)
        self.assertTrue(diag['force_pause'])
        self.assertFalse(any(diag['seeking']))
        np.testing.assert_array_equal(diag['search_delta_rad'], np.zeros(16))
        np.testing.assert_array_equal(diag['search_distance_m'], before['search_distance_m'])
        with patch.object(self.search, '_refresh_targets', side_effect=AssertionError('Air move queried surface')):
            disabled, diag = self.tick(enabled=False)
        np.testing.assert_array_equal(disabled, self.q)
        self.assertEqual(diag['state'], ['DISABLED'] * 4)
        np.testing.assert_array_equal(diag['joint_bias_rad'], np.zeros(16))
        self.assertFalse(any(diag['target_valid']))

    def test_targets_refresh_with_measured_wrist_and_actual_pad_orientation(self):
        _, initial = self.tick()
        old = np.array(initial['target_positions_world_m'])
        q = self.q.copy(); q[:7] += [.06, -.02, .03, -.01, .02, 0., -.01]
        set_joint_state(self.h, self.data, q)
        for _ in range(26):
            _, diag = self.tick(nominal=q)
        self.assertGreater(diag['target_refresh_count'], initial['target_refresh_count'])
        targets = np.array(diag['target_positions_world_m'])
        self.assertGreater(np.linalg.norm(targets - old), .001)
        for finger, geom in enumerate(self.h.tip_geom_ids):
            if not diag['target_valid'][finger]:
                continue
            normal = np.array(diag['target_normals_world'][finger])
            self.assertAlmostEqual(np.linalg.norm(normal), 1.)
            frame = self.data.geom_xmat[int(geom)].reshape(3, 3)
            support = pad_support_vector(self.h.model, int(geom), frame, normal)
            point = targets[finger] - support - self.search.config.target_clearance_m * normal
            self.assertAlmostEqual(point[2], self.plane_z, places=10)

    def test_far_surface_cannot_trigger_unbounded_search(self):
        self.graph.mesh.apply_translation([0., 0., -1.])
        for _ in range(40):
            result, diag = self.tick()
        self.assertFalse(any(diag['target_valid']))
        self.assertFalse(any(diag['seeking']))
        self.assertEqual(diag['state'], ['OUT_OF_REACH'] * 4)
        np.testing.assert_array_equal(result, self.q)


if __name__ == '__main__':
    unittest.main()
