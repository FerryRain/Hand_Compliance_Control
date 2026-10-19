"""Measured support gates and native non-mutation of wrist recovery planning."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np

from Module.i04_geometric_traversal.contact_recovery_planner import (
    ContactRecoveryConfig, ContactRecoveryPlanner,
)
from Module.i04_geometric_traversal.full_surface import acquire, build_scene
from Module.i04_geometric_traversal.physics import contact_measurement, joint_state
from Module.tests.test_i04_sliding_safety import native_pair_fixture


class MeasuredSupportGateTests(unittest.TestCase):
    def setUp(self):
        self.h = native_pair_fixture()
        self.data = mujoco.MjData(self.h.model)
        mujoco.mj_forward(self.h.model, self.data)
        self.planner = ContactRecoveryPlanner(self.h, None)
        self.normals = np.tile([1., 0., 0.], (4, 1))

    def test_force_without_measured_active_cannot_become_anchor(self):
        path, diag = self.planner.plan(self.data, 0, [False]*4, [0., 2., 0., 0.], self.normals)
        self.assertIsNone(path)
        self.assertEqual(diag['reason'], 'NO_MEASURED_SAFE_ANCHOR')

    def test_loaded_or_already_contacting_fingers_do_not_trigger_pivot(self):
        path, diag = self.planner.plan(self.data, 0, [False, True, False, False], [0., 8., 0., 0.], self.normals)
        self.assertIsNone(path)
        self.assertEqual(diag['reason'], 'MEASURED_FORCE_OR_SELF_CONTACT')
        path, diag = self.planner.plan(self.data, 0, [True, True, False, False], [2., 2., 0., 0.], self.normals)
        self.assertIsNone(path)
        self.assertEqual(diag['reason'], 'SELECTED_FINGER_ALREADY_ACTIVE')

    def test_active_flag_without_a_usable_measured_normal_is_insufficient(self):
        path, diag = self.planner.plan(self.data, 0, [False, True, False, False], [0., 2., 0., 0.], np.zeros((4, 3)))
        self.assertIsNone(path)
        self.assertEqual(diag['reason'], 'NO_MEASURED_SAFE_ANCHOR')

    def test_angle_configuration_stays_small(self):
        with self.assertRaises(ValueError): ContactRecoveryConfig(angles_deg=(12.,))
        with self.assertRaises(ValueError): ContactRecoveryConfig(angles_deg=(float('nan'),))


class NativeRecoveryPlanningTests(unittest.TestCase):
    def test_acquired_missing_finger_proposal_preserves_plant_and_model(self):
        parent = Path(__file__).resolve().parents[2]/'.runtime_tmp'
        parent.mkdir(exist_ok=True)
        with TemporaryDirectory(dir=parent, prefix='contact_recovery_test_') as directory:
            # TemporaryDirectory cleanup is confined to this checked workspace root.
            self.assertEqual(Path(directory).resolve().parent, parent.resolve())
            h, graph, initial, _ = build_scene(directory)
            data, _ = acquire(h, graph, initial, seed=7)
            forces, _, normals, active, _, _ = contact_measurement(h, data, .2)
            missing = np.flatnonzero(~active)
            self.assertGreater(len(missing), 0, 'Fixture should exercise an actually missing finger')
            native_before = {key:getattr(data, key).copy() for key in ('qpos', 'qvel', 'ctrl', 'qacc_warmstart')}
            model_before = {key:getattr(h.model, key).copy() for key in ('geom_margin', 'geom_solref', 'geom_solimp', 'dof_damping')}
            planner = ContactRecoveryPlanner(h, graph)
            path, diag = planner.plan(data, int(missing[0]), active, forces, normals)
            self.assertTrue(diag['candidates'])
            for key, expected in native_before.items(): np.testing.assert_array_equal(getattr(data, key), expected)
            for key, expected in model_before.items(): np.testing.assert_array_equal(getattr(h.model, key), expected)
            self.assertFalse(diag['physical_recontact_tested'])
            if path is not None:
                root = joint_state(h, data)
                self.assertTrue(active[path.anchor_finger])
                np.testing.assert_array_equal(path.root_q, root)
                samples = np.array([path.sample(t) for t in np.linspace(0., path.duration_s, 101)])
                np.testing.assert_array_equal(samples[:, 7:], np.tile(root[7:], (len(samples), 1)))
                for a, b in zip(samples[:-1], samples[1:]): self.assertTrue(planner.planner.edge_free(a, b))
                speed = np.max(1.875*abs(np.diff(path.waypoints, axis=0))/path.segment_durations[:, None])
                self.assertLessEqual(speed, planner.config.joint_speed_rad_s+1e-12)


if __name__ == '__main__': unittest.main()
