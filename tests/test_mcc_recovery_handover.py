"""Physical controller regressions for contact reacquisition and force control."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"mcc_finger_compliance_control/scripts"))
from surface_mcc_finger import FullHandMCCFingerConfig, FullHandMCCFingerController
from collect_leap_force_matched import matched_force_config


class RecoveryHandoverTests(unittest.TestCase):
    def test_auxiliary_initialization_controller_does_not_receive_handover(self):
        config=FullHandMCCFingerConfig(enable_loss_state_machine=False)
        result=matched_force_config(config,(10.,)*4,True)
        self.assertFalse(result.enable_recovery_handover)
        self.assertEqual(result.desired_force_per_finger,(10.,)*4)
        active=matched_force_config(replace(config,enable_loss_state_machine=True,
                                           use_direct_force_servo=True),(10.,)*4,True)
        self.assertTrue(active.enable_recovery_handover)

    def controller(self, enabled=True):
        return FullHandMCCFingerController(FullHandMCCFingerConfig(
            enable_loss_state_machine=True, enable_recovery_handover=enabled,
            use_direct_force_servo=True, desired_force_per_finger=(10.,)*4,
            nominal_surface_preload=.003, normal_synergy_control=False,
            qp_max_joint_velocity=.8, back_contact_x_limit_m=(1.,)*4))

    def physical_inputs(self, c):
        q = c.clamp_joint_positions(np.full(16, .5)).astype(float)
        points = c.tip_positions_palm(q)
        normals = np.tile([0., 0., 1.], (4, 1))
        return dict(q_live=q, palm_pose_world=np.array([0.,0.,0.,1.,0.,0.,0.]),
                    surface_points_world=points, surface_normals_world=normals,
                    contact_points_world=points, nominal_posture_q=q.copy(),
                    force_world=10.*normals, found=np.ones(4, bool),
                    force_magnitude_only=True, use_contact_point_jacobian=True)

    def enter_recovery(self, c):
        for _ in range(c.config.transient_loss_frames+1):
            c._update_contact_recovery_state(np.zeros(4, bool))
        np.testing.assert_array_equal(c.contact_phase, [2]*4)

    def test_grazing_contact_cannot_restart_transient_delay(self):
        c = self.controller(); self.enter_recovery(c)
        c._update_contact_recovery_state(np.ones(4, bool))
        np.testing.assert_array_equal(c.contact_phase, [3]*4)
        c._update_contact_recovery_state(np.zeros(4, bool))
        np.testing.assert_array_equal(c.contact_phase, [2]*4)
        for _ in range(c.config.recovery_contact_confirm_frames):
            c._update_contact_recovery_state(np.ones(4, bool))
        np.testing.assert_array_equal(c.contact_phase, [0]*4)
        self.assertFalse(c.recovery_owned.any())

    def test_recontact_discards_stale_integral_and_rebases_command(self):
        c = self.controller(); x = self.physical_inputs(c)
        self.enter_recovery(c)
        c.previous_command = x['q_live']+.005
        expected = c.previous_command.copy()
        c.force_servo_offset[:] = -.016
        c.transient_search_offset[:] = .001
        x['nominal_posture_q'] += .15
        original_nominal = x['nominal_posture_q'].copy()
        result, debug = c.update(**x)
        np.testing.assert_allclose(debug['handover_reference_q'], expected, atol=1.e-7)
        np.testing.assert_allclose(c.force_servo_offset, -.003, atol=1.e-9)
        np.testing.assert_allclose(debug['normal_offset'], 0., atol=1.e-9)
        self.assertLessEqual(abs(result-expected).max(), .008001)
        np.testing.assert_array_equal(original_nominal, x['nominal_posture_q'])

    def test_healthy_contact_does_not_change_existing_controller(self):
        a, b = self.controller(False), self.controller(True)
        x = self.physical_inputs(a)
        for force in (10., 7., 13., 10.):
            x['force_world'][:] = force*x['surface_normals_world']
            qa, _ = a.update(**x); qb, _ = b.update(**x)
            np.testing.assert_array_equal(qa, qb)

    def test_persistent_recovery_preserves_existing_operator_and_authority(self):
        c = self.controller(); x = self.physical_inputs(c)
        target = x['surface_points_world']+.02
        inputs=dict(q_live=x['q_live'], base_command_q=x['q_live'],
                    palm_pose_world=x['palm_pose_world'], surface_points_world=target,
                    surface_normals_world=x['surface_normals_world'], persistent_loss=np.ones(4,bool),
                    surface_preload_m=.004)
        q, _ = c.recover_surface_contacts(**inputs)
        original, _ = self.controller(False).recover_surface_contacts(**inputs)
        np.testing.assert_array_equal(q, original)
        self.assertLessEqual(abs(q-x['q_live']).max(), .040001)
        self.assertGreater(abs(q-x['q_live']).max(), .008001)

    def test_overforce_safety_takes_priority_over_handover(self):
        c = self.controller(); x = self.physical_inputs(c)
        self.enter_recovery(c); c.previous_command = x['q_live']+.04
        x['force_world'] *= 10.
        result, debug = c.update(**x)
        self.assertTrue(debug['collision_safety_active'].all())
        np.testing.assert_allclose(result, x['q_live'], atol=1.e-7)


if __name__ == '__main__':
    unittest.main()
