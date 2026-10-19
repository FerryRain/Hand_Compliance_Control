"""Episode budgets, measured-contact handoff and infeasible command behavior."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from Module.tests import test_i04_active_fingers as native_fixture
from Module.i04_geometric_traversal.active_fingers import ActiveFingerSearchConfig
from Module.i04_geometric_traversal.active_fingers_v3 import ActiveFingerSearchV3
from Module.i04_geometric_traversal.physics import set_control


class ActiveFingerV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        native_fixture.ActiveFingerNativeGeometryTests.setUpClass()

    def setUp(self):
        fixture = native_fixture.ActiveFingerNativeGeometryTests()
        fixture.setUp()
        self.h, self.data, self.graph, self.q = fixture.h, fixture.data, fixture.graph, fixture.q.copy()
        self.forces, self.normals, self.active = fixture.forces, fixture.normals, fixture.active
        self.search = ActiveFingerSearchV3(self.h, self.graph)
        set_control(self.h, self.data, self.q)

    def tick(self, nominal=None, **kwargs):
        return self.search.update(self.data, self.q if nominal is None else nominal,
                                  self.forces, self.normals, self.active, **kwargs)

    def prime(self, count=55):
        for _ in range(count): result, diag = self.tick()
        self.assertTrue(any(diag['seeking']))
        return result, diag

    def test_rebase_does_not_duplicate_bias_or_reset_loss_budget(self):
        result, old = self.prime()
        self.search.rebase(result)
        np.testing.assert_array_equal(self.search.bias, np.zeros(16))
        np.testing.assert_array_equal(self.search.loss_age, old['loss_age_s'])
        np.testing.assert_array_equal(self.search.search_elapsed, old['search_elapsed_s'])
        np.testing.assert_array_equal(self.search.search_distance, old['search_distance_m'])
        held, new = self.tick(nominal=result, search_mask=np.zeros(4, dtype=bool))
        np.testing.assert_array_equal(held, result)
        self.assertTrue(np.all(np.array(new['loss_age_s']) > old['loss_age_s']))

    def test_first_contact_stops_push_but_only_150ms_stability_resets_budget(self):
        result, old = self.prime()
        self.active[:] = True; self.forces[:] = 1.; self.normals[:] = [0., 0., 1.]
        held, diag = self.tick()
        np.testing.assert_array_equal(held, result)
        self.assertFalse(any(diag['seeking']))
        np.testing.assert_array_equal(diag['search_delta_rad'], np.zeros(16))
        np.testing.assert_array_equal(diag['search_distance_m'], old['search_distance_m'])
        for _ in range(73): _, diag = self.tick()
        self.assertTrue(all(np.array(diag['loss_age_s']) > 0.))
        self.assertFalse(any(diag['stable_recontacted']))
        _, diag = self.tick()
        self.assertTrue(all(diag['stable_recontacted']))
        np.testing.assert_array_equal(diag['loss_age_s'], np.zeros(4))
        np.testing.assert_array_equal(diag['search_distance_m'], np.zeros(4))

    def test_flicker_and_stroke_rebase_cannot_refresh_exhausted_search(self):
        self.search = ActiveFingerSearchV3(self.h, self.graph, config=replace(
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

    def test_explicit_posture_attempt_renews_only_local_budget_and_has_episode_cap(self):
        _, old = self.prime()
        original_age = self.search.loss_age.copy()
        for number in (1, 2, 3):
            result = self.search.begin_posture_attempt(0)
            self.assertTrue(result['accepted'])
            self.assertEqual(result['attempt_id'], number)
            np.testing.assert_array_equal(self.search.loss_age, original_age)
            self.assertEqual(self.search.search_elapsed[0], 0.)
            self.assertEqual(self.search.search_distance[0], 0.)
            self.assertGreater(self.search.search_distance[1], 0.)
        self.assertFalse(self.search.begin_posture_attempt(0)['accepted'])
        self.search.rebase(self.q)
        self.assertFalse(self.search.begin_posture_attempt(0)['accepted'])

    def test_air_clears_bias_without_erasing_actual_loss_or_search_budget(self):
        _, old = self.prime()
        with patch.object(self.search, '_refresh_targets', side_effect=AssertionError('Air queried mesh')):
            for _ in range(20): result, diag = self.tick(enabled=False, intentional_air=True)
        np.testing.assert_array_equal(result, self.q)
        np.testing.assert_array_equal(diag['joint_bias_rad'], np.zeros(16))
        np.testing.assert_array_equal(diag['search_elapsed_s'], old['search_elapsed_s'])
        np.testing.assert_array_equal(diag['search_distance_m'], old['search_distance_m'])
        self.assertTrue(np.all(np.array(diag['loss_age_s']) > old['loss_age_s']))
        result, diag = self.tick(search_mask=np.zeros(4, dtype=bool))
        np.testing.assert_array_equal(result, self.q)

    def test_measured_context_reopens_quota_but_preserves_true_loss_and_command(self):
        self.prime()
        for _ in range(3): self.search.begin_posture_attempt(0)
        self.tick()
        previous = {name: getattr(self.search, name).copy() for name in
                    ('loss_age', 'bias', 'targets', 'posture_attempts_this_episode')}
        grant = dict(kind='MEASURED_RECOVERY_CONTEXT_GRANTED', accepted=True,
                     finger=0, context_id=1)
        self.assertTrue(self.search.apply_recovery_context(grant)['accepted'])
        for name, value in previous.items():
            np.testing.assert_array_equal(getattr(self.search, name), value)
        self.assertEqual(self.search.search_elapsed[0], 0.)
        self.assertEqual(self.search.search_distance[0], 0.)
        self.assertTrue(self.search.begin_posture_attempt(0)['accepted'])
        self.assertEqual(self.search.posture_attempts_this_episode[0], 4)
        self.assertEqual(self.search.posture_attempts_this_context[0], 1)
        self.search.search_elapsed[0] = 12.
        self.assertFalse(self.search.apply_recovery_context(grant)['accepted'])
        self.assertEqual(self.search.search_elapsed[0], 12.)

    def test_context_cannot_reopen_current_contact_or_malformed_token(self):
        self.prime()
        self.active[0], self.forces[0] = True, 1.
        self.tick()
        grant = dict(kind='MEASURED_RECOVERY_CONTEXT_GRANTED', accepted=True,
                     finger=0, context_id=1)
        self.assertFalse(self.search.apply_recovery_context(grant)['accepted'])
        self.assertEqual(self.search.recovery_context_id[0], 0)
        with self.assertRaises(ValueError): self.search.apply_recovery_context({})

    def test_search_mask_preserves_existing_other_finger_contact_bias(self):
        result, old = self.prime()
        self.active[1:] = True; self.forces[1:] = 1.; self.normals[1:] = [0., 0., 1.]
        new, diag = self.tick(search_mask=[True, False, False, False])
        np.testing.assert_array_equal(new[11:], result[11:])
        self.assertFalse(any(diag['seeking'][1:]))
        self.assertTrue(diag['seeking'][0])

    def test_selected_reference_handoff_does_not_erase_other_contact_biases(self):
        result, old = self.prime()
        columns = self.search.columns[0]
        nominal = self.q.copy(); nominal[columns+7] = result[columns+7]
        self.search.rebase(nominal, finger_mask=[True, False, False, False])
        self.active[:] = True; self.forces[:] = 1.; self.normals[:] = [0., 0., 1.]
        held, diag = self.tick(nominal=nominal)
        np.testing.assert_array_equal(held, result)
        np.testing.assert_array_equal(np.asarray(diag['joint_bias_rad'])[columns], np.zeros(4))
        np.testing.assert_array_equal(np.asarray(diag['joint_bias_rad'])[4:], np.asarray(old['joint_bias_rad'])[4:])

    def test_infeasible_repair_holds_actual_control_and_requests_measured_rebase(self):
        self.prime()
        nominal = self.q.copy()
        columns = self.search.columns[0]
        nominal[columns+7] = self.search.lower[columns]-.1
        applied = self.q.copy(); applied[columns+7] += .002
        set_control(self.h, self.data, applied)
        result, diag = self.tick(nominal=nominal, search_mask=[True, False, False, False])
        self.assertTrue(diag['requires_rebase'][0])
        self.assertFalse(diag['step_feasible'][0])
        np.testing.assert_array_equal(result[columns+7], applied[columns+7])
        np.testing.assert_array_equal(np.asarray(diag['search_delta_rad'])[columns], np.zeros(4))
        np.testing.assert_array_equal(result[:7], nominal[:7])

    def test_joint_and_cartesian_limits_remain_original_bounds(self):
        previous_distance = np.zeros(4)
        for _ in range(80):
            result, diag = self.tick()
            distance = np.array(diag['search_distance_m'])
            self.assertTrue(np.all(distance-previous_distance <= 8e-6+1e-12))
            self.assertLessEqual(np.max(abs(np.asarray(diag['search_delta_rad']))), .001+1e-12)
            self.assertLessEqual(np.max(abs(np.asarray(diag['joint_bias_rad']))), .5+1e-12)
            self.assertTrue(np.all(result[7:] >= self.search.lower-1e-12))
            self.assertTrue(np.all(result[7:] <= self.search.upper+1e-12))
            previous_distance = distance


if __name__ == '__main__': unittest.main()
