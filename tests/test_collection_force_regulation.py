"""Force direction, asymmetric authority and final connector arbitration."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'mcc_finger_compliance_control/scripts'))
from surface_mcc_finger import FullHandMCCFingerConfig, FullHandMCCFingerController
from collection_force_regulation import (CollectionForceObserver, collection_force_config,
                                         protect_collection_contacts)


class CollectionForceTests(unittest.TestCase):
    def make_controller(self):
        cfg = collection_force_config(FullHandMCCFingerConfig(force_servo_integral_gain=.003), 'regulated')
        c = FullHandMCCFingerController(cfg)
        q = c.clamp_joint_positions(np.full(16, .5)).astype(float)
        pose = np.array([0., 0., 0., 1., 0., 0., 0.])
        points = c.tip_positions_palm(q)
        return c, q, pose, points

    def step(self, c, q, pose, points, force):
        normals = np.tile([0., 0., 1.], (4, 1))
        return c.update(q_live=q, palm_pose_world=pose,
            force_world=force*normals, found=np.ones(4, bool),
            surface_points_world=points, surface_normals_world=normals,
            nominal_posture_q=q, force_magnitude_only=True,
            contact_points_world=points, use_contact_point_jacobian=True)

    def test_weak_force_presses_and_strong_force_releases(self):
        c, q, pose, points = self.make_controller()
        self.step(c, q, pose, points, 1.)
        self.assertTrue(np.all(c.force_servo_offset > 0.))
        c.reset()
        self.step(c, q, pose, points, 30.)
        self.assertTrue(np.all(c.force_servo_offset < 0.))
        self.assertTrue(np.all(c.overforce_active))

    def test_larger_retreat_does_not_increase_press_limit(self):
        c, q, pose, points = self.make_controller()
        c.force_servo_offset[:] = .010
        self.step(c, q, pose, points, 1.)
        np.testing.assert_allclose(c.force_servo_offset[:3], .003)
        c.reset(); c.force_servo_offset[:] = -.006
        self.step(c, q, pose, points, 10.)
        np.testing.assert_allclose(c.force_servo_offset, -.006)

    def test_emergency_release_has_no_offset_jump(self):
        c, q, pose, points = self.make_controller()
        c.force_servo_offset[:] = -.012
        self.step(c, q, pose, points, 1.)
        self.assertTrue(np.all(c.force_servo_offset < -.0119))

    def test_connector_escape_preserves_pad_anchor(self):
        c, q, pose, points = self.make_controller()
        c.update_contact_point_anchors(q, pose, points, np.ones(4, bool))
        anchors = c.contact_point_body_local.copy()
        c._set_q(c.data, q)
        jac = c._finger_control_point_jacobian(0)
        normal = jac[:, 0] / np.linalg.norm(jac[:, 0])
        jn = normal @ jac
        candidate = q.copy(); candidate[:4] -= .01*jn/np.linalg.norm(jn)
        x = {}
        for role in ('pad', 'connector'):
            x[role+'_physical_found'] = np.zeros((1, 4), bool)
            x[role+'_physical_force'] = np.zeros((1, 4, 3))
            x[role+'_physical_points'] = points[None].copy()
            x[role+'_physical_inward'] = np.zeros((1, 4, 3))
        x['connector_physical_found'][0, 0] = True
        x['connector_physical_force'][0, 0] = -8.*normal
        x['connector_physical_inward'][0, 0] = -normal
        result, debug = protect_collection_contacts(c, q, candidate, pose, x, 0)
        self.assertTrue(debug['hard_active'][0])
        self.assertGreater(float(jn @ (result[:4]-q[:4])), .00009)
        np.testing.assert_array_equal(anchors, c.contact_point_body_local)
        np.testing.assert_allclose(result[4:], candidate[4:], atol=1.e-7)

    def test_legacy_config_is_unchanged(self):
        cfg = FullHandMCCFingerConfig()
        self.assertIs(collection_force_config(cfg, 'legacy'), cfg)

    def test_collision_guard_uses_matched_physical_slots(self):
        from mjlab.sensor.contact_sensor import ContactMatch, ContactSensorCfg
        sensors = tuple(ContactSensorCfg(name=site+'_contact',
            primary=ContactMatch(mode='geom', pattern=site), reduce='netforce')
            for site in ('if_tip', 'mf_tip', 'rf_tip', 'th_tip'))
        cfg = SimpleNamespace(scene=SimpleNamespace(sensors=sensors))
        observer = CollectionForceObserver(controller_scope='pad')
        with patch('split_fingertip_collision.configure_split_collision'):
            observer.configure(cfg)
        guards = [s for s in cfg.scene.sensors if s.name.endswith('_force_guard')]
        self.assertEqual(len(guards), 8)
        for s in guards:
            self.assertEqual(s.reduce, 'maxforce')
            self.assertTrue({'pos', 'normal', 'force'} <= set(s.fields))
            self.assertEqual(s.num_slots, 1)


if __name__ == '__main__':
    unittest.main()
