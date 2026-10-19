"""The larger candidate rotation does not widen physical safety boundaries."""
from dataclasses import FrozenInstanceError, asdict, replace
import unittest

from Module.i04_geometric_traversal.contact_recovery_planner import ContactRecoveryConfig
from Module.i04_geometric_traversal.extended_contact_recovery import ExtendedContactRecoveryConfig


class ExtendedContactRecoveryConfigTests(unittest.TestCase):
    def test_only_angle_and_center_defaults_differ_and_original_stays_small(self):
        original, extended = ContactRecoveryConfig(), ExtendedContactRecoveryConfig()
        self.assertIsInstance(extended, ContactRecoveryConfig)
        old, new = asdict(original), asdict(extended)
        self.assertEqual({key for key in old if old[key] != new[key]},
                         {'angles_deg', 'anchor_center_drift_m'})
        self.assertEqual(extended.angles_deg, (3., 6., 12.))
        self.assertEqual(extended.anchor_center_drift_m, .003)
        with self.assertRaises(ValueError): ContactRecoveryConfig(angles_deg=(12.,))
        with self.assertRaises(FrozenInstanceError): extended.anchor_support_drift_m = .01

    def test_invalid_angle_center_and_nonfinite_values_are_rejected(self):
        for angles in ((), (0.,), (-3.,), (12.01,), (float('inf'),), (float('nan'),)):
            with self.subTest(angles=angles), self.assertRaises(ValueError):
                ExtendedContactRecoveryConfig(angles_deg=angles)
        for value in (0., -.001, .003001, float('nan'), float('inf')):
            with self.subTest(center=value), self.assertRaises(ValueError):
                ExtendedContactRecoveryConfig(anchor_center_drift_m=value)
        for name in asdict(ContactRecoveryConfig()):
            if name == 'angles_deg': continue
            with self.subTest(field=name), self.assertRaises(ValueError):
                ExtendedContactRecoveryConfig(**{name: float('nan')})

    def test_each_original_safety_limit_cannot_be_widened(self):
        original = ContactRecoveryConfig()
        caps = ('angular_waypoint_deg', 'anchor_support_drift_m', 'secondary_anchor_drift_m',
                'maximum_anchor_inward_m', 'wrist_speed_m_s', 'rotation_speed_rad_s',
                'joint_speed_rad_s', 'force_pause_n', 'position_tolerance_m', 'rotation_tolerance_rad')
        for name in caps:
            with self.subTest(field=name), self.assertRaises(ValueError):
                ExtendedContactRecoveryConfig(**{name: getattr(original, name)*1.01})
        with self.assertRaises(ValueError):
            ExtendedContactRecoveryConfig(minimum_gap_improvement_m=.0009)
        # Stricter limits and a restricted angle set are supported explicitly.
        strict = replace(ExtendedContactRecoveryConfig(), angles_deg=(12.,),
                         maximum_anchor_inward_m=.0001, joint_speed_rad_s=.10,
                         anchor_center_drift_m=.002)
        self.assertEqual(strict.maximum_anchor_inward_m, .0001)


if __name__ == '__main__': unittest.main()
