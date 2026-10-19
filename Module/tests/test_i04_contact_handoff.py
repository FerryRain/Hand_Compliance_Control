"""Native Jacobians/contact forces back reference-only preload handoff checks."""
from dataclasses import asdict
import unittest
import mujoco
import numpy as np

from Module.e05_physics.scene import FINGERS
from Module.fr3_leap import ARM_HOME_Q, HAND_NATURAL_Q, FullRobotModelConfig, build_full_robot
from Module.i04_geometric_traversal.contact_handoff import ContactPreservingMCC
from Module.i04_geometric_traversal.full_surface import pad_support_vector
from Module.i04_geometric_traversal.physics import set_joint_state, set_control, contact_measurement
from Module.i04_geometric_traversal.surface_slide import GraceMCC


class ContactHandoffNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.h = build_full_robot(FullRobotModelConfig(surface='plane', gravity_m_s2=0.))
        cls.q = np.r_[ARM_HOME_Q, HAND_NATURAL_Q]
        # Construct a physical fixture once: a horizontal surface indents the
        # lowest real ellipsoid pad by 0.8 mm. No test controller moves it.
        data = mujoco.MjData(cls.h.model)
        set_joint_state(cls.h, data, cls.q)
        centers = data.geom_xpos[cls.h.tip_geom_ids].copy()
        normal = np.array([0., 0., 1.])
        support = np.array([pad_support_vector(cls.h.model, int(g),
            data.geom_xmat[int(g)].reshape(3, 3), normal)[2] for g in cls.h.tip_geom_ids])
        cls.h.model.body_pos[cls.h.object_body_id] = [centers[:, 0].mean(),
            centers[:, 1].mean(), np.min(centers[:, 2]-support)-.01+.0008]
        cls.h.model.body_quat[cls.h.object_body_id] = [1., 0., 0., 0.]

    def setUp(self):
        self.data = mujoco.MjData(self.h.model)
        self.nominal = self.q.copy()
        self.previous = self.q.copy()
        self.previous[:7] += .001
        self.previous[7:] += np.tile([.006, -.004, .008, .003], 4)
        set_joint_state(self.h, self.data, self.q)
        set_control(self.h, self.data, self.previous)
        mujoco.mj_forward(self.h.model, self.data)
        self.force, self.points, self.normals, self.active, unsafe, _ = contact_measurement(
            self.h, self.data, .2)
        self.assertFalse(unsafe)
        self.assertTrue(np.any(self.active))
        self.assertTrue(np.all(self.force[self.active] < 8.))
        self.contact = int(np.flatnonzero(self.active)[0])
        self.columns = np.asarray([int(x) for x in FINGERS[self.contact].joint_names]) + 7

    def controller(self):
        return ContactPreservingMCC(self.h, .002, self.nominal, self.previous,
                                    self.active, self.force)

    def command(self, controller, nominal=None, **kwargs):
        return controller.command(self.data, self.nominal if nominal is None else nominal,
            self.force, self.normals, self.active, 2., **kwargs)

    def test_first_native_contact_reference_retains_preload_plus_original_mcc_increment(self):
        controller = self.controller()
        baseline = self.command(GraceMCC(self.h, .002))
        result = self.command(controller)
        correction = baseline-self.nominal
        np.testing.assert_allclose(result[self.columns],
            self.previous[self.columns]+correction[self.columns], atol=2e-16)
        remaining = np.setdiff1d(np.arange(23), self.columns)
        np.testing.assert_array_equal(result[remaining], baseline[remaining])
        self.assertGreater(np.max(abs(self.previous[self.columns]-self.nominal[self.columns])), .005)
        self.assertLess(np.max(abs(correction[self.columns])), .001)
        np.testing.assert_array_equal(result[:7], self.nominal[:7])
        np.testing.assert_array_equal(controller.seeded_fingers, self.active)

    def test_only_actual_safe_contact_can_seed_reference_bias(self):
        controller = ContactPreservingMCC(self.h, .002)
        result = controller.seed_handoff(self.nominal, self.previous,
            [True, True, True, False], [.2, .199, 8., 2.])
        np.testing.assert_array_equal(controller.seeded_fingers, [True, False, False, False])
        np.testing.assert_array_equal(controller.held_bias_rad[:4],
                                      self.previous[7:11]-self.nominal[7:11])
        np.testing.assert_array_equal(controller.held_bias_rad[4:], np.zeros(12))
        self.assertFalse(result['measured_contact_claim'])
        self.assertFalse(result['coverage_credit'])

    def test_repeated_seed_replaces_bias_and_resets_old_compliance(self):
        controller = self.controller()
        first = self.command(controller)
        for _ in range(25):
            self.command(controller)
        controller.seed_handoff(self.nominal, self.previous, self.active, self.force)
        np.testing.assert_array_equal(self.command(controller), first)
        controller.seed_handoff(self.nominal, self.previous, self.active, self.force)
        np.testing.assert_array_equal(self.command(controller), first)

    def test_new_nominal_already_containing_command_does_not_double_old_preload(self):
        controller = self.controller()
        previous_applied = self.command(controller)
        new_nominal = previous_applied.copy()
        controller.seed_handoff(new_nominal, previous_applied, self.active, self.force)
        np.testing.assert_array_equal(controller.held_bias_rad, np.zeros(16))
        expected = self.command(GraceMCC(self.h, .002), nominal=new_nominal)
        np.testing.assert_array_equal(self.command(controller, nominal=new_nominal), expected)

    def test_explicit_reset_clears_preload_and_grace_without_changing_plant(self):
        controller = self.controller()
        self.command(controller)
        controller.reset()
        np.testing.assert_array_equal(controller.held_bias_rad, np.zeros(16))
        np.testing.assert_array_equal(controller.seeded_fingers, np.zeros(4, bool))
        self.assertTrue(np.isinf(controller.age).all())
        np.testing.assert_array_equal(self.command(controller), self.command(GraceMCC(self.h, .002)))

    def test_intentional_air_discards_preload_and_cannot_continue_grace_push(self):
        controller = self.controller()
        self.command(controller)
        result = self.command(controller, intentional_air=True)
        np.testing.assert_array_equal(result, self.nominal)
        np.testing.assert_array_equal(controller.held_bias_rad, np.zeros(16))
        result = controller.command(self.data, self.nominal, np.zeros(4), np.zeros((4, 3)),
                                    np.zeros(4, bool), 2.)
        np.testing.assert_array_equal(result, self.nominal)
        controller.seed_handoff(self.nominal, self.previous, self.active, self.force,
                                intentional_air=True)
        np.testing.assert_array_equal(controller.held_bias_rad, np.zeros(16))

    def test_force_feedback_and_every_original_mcc_limit_are_unchanged(self):
        controller = self.controller()
        reference = self.nominal.copy()
        reference[7:] += controller.held_bias_rad
        baseline = GraceMCC(self.h, .002)
        for current, original in zip(controller.controllers, baseline.controllers):
            self.assertEqual(asdict(current.config), asdict(original.config))
        # Deliberately exercise both force-error signs and saturation behavior
        # using native Jacobians. Neither controller applies these references.
        for force_value in (0., 7.9, 12.):
            forces = np.zeros(4); forces[self.contact] = force_value
            for _ in range(150):
                actual = controller.command(self.data, self.nominal, forces, self.normals, self.active, 2.)
                expected = baseline.command(self.data, reference, forces, self.normals, self.active, 2.)
                np.testing.assert_array_equal(actual, expected)
            for current, original in zip(controller.controllers, baseline.controllers):
                self.assertEqual(current.state, original.state)

    def test_original_brief_loss_grace_is_preserved_without_new_contact_claims(self):
        controller = self.controller()
        baseline = GraceMCC(self.h, .002)
        reference = self.nominal.copy(); reference[7:] += controller.held_bias_rad
        self.command(controller)
        baseline.command(self.data, reference, self.force, self.normals, self.active, 2.)
        for _ in range(60):
            result = controller.command(self.data, self.nominal, np.zeros(4), np.zeros((4, 3)),
                                        np.zeros(4, bool), 2.)
            expected = baseline.command(self.data, reference, np.zeros(4), np.zeros((4, 3)),
                                         np.zeros(4, bool), 2.)
            np.testing.assert_array_equal(result, expected)
        # After >100 ms no compliance continuation remains. The explicit
        # command-reference preload is still a reference, never contact data.
        np.testing.assert_array_equal(result, reference)

    def test_module_does_not_write_native_state_model_or_input_arrays(self):
        model_fields = ('body_pos', 'body_quat', 'geom_margin', 'geom_solref',
                        'geom_solimp', 'dof_damping', 'actuator_gainprm', 'actuator_ctrlrange')
        data_fields = ('qpos', 'qvel', 'ctrl', 'qacc', 'qacc_warmstart', 'geom_xpos', 'site_xpos')
        before_model = {key: getattr(self.h.model, key).copy() for key in model_fields}
        before_data = {key: getattr(self.data, key).copy() for key in data_fields}
        native_time = self.data.time
        inputs = [x.copy() for x in (self.nominal, self.previous, self.active, self.force, self.normals)]
        controller = self.controller()
        for _ in range(10):
            self.command(controller)
        controller.reset()
        for key, value in before_model.items():
            np.testing.assert_array_equal(getattr(self.h.model, key), value)
        for key, value in before_data.items():
            np.testing.assert_array_equal(getattr(self.data, key), value)
        self.assertEqual(self.data.time, native_time)
        for actual, expected in zip((self.nominal, self.previous, self.active, self.force, self.normals), inputs):
            np.testing.assert_array_equal(actual, expected)

    def test_recorded_bias_is_a_copy_and_limits_remain_callers_responsibility(self):
        controller = self.controller()
        original = controller.held_bias_rad
        exposed = controller.held_bias_rad; exposed[:] = 99.
        mask = controller.seeded_fingers; mask[:] = False
        np.testing.assert_array_equal(controller.held_bias_rad, original)
        np.testing.assert_array_equal(controller.seeded_fingers, self.active)
        result = self.command(controller)
        # No hidden clipping of the servo preload replaces the caller's final
        # full-command slew/joint/collision/force guards.
        expected = self.nominal.copy(); expected[7:] += original
        expected = GraceMCC(self.h, .002).command(self.data, expected, self.force,
                                                self.normals, self.active, 2.)
        np.testing.assert_array_equal(result, expected)

    def test_invalid_handoff_is_rejected_without_erasing_valid_existing_bias(self):
        controller = self.controller()
        original = controller.held_bias_rad
        for bad in (np.zeros(22), np.full(23, np.nan)):
            with self.assertRaises(ValueError):
                controller.seed_handoff(bad, self.previous, self.active, self.force)
            np.testing.assert_array_equal(controller.held_bias_rad, original)
        with self.assertRaises(ValueError):
            ContactPreservingMCC(self.h, .002, self.nominal)
        with self.assertRaises(ValueError):
            ContactPreservingMCC(self.h, 0.)


if __name__ == '__main__':
    unittest.main()
