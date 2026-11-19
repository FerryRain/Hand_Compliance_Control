"""Regression: previously invisible rigid palm/object overlaps are real contacts."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from Module.i04_geometric_traversal.palm_collision import (
    PALM_NAMES, PalmMonitor, enable_palm_collisions,
)


def fixture():
    root = ET.Element('mujoco')
    ET.SubElement(root, 'compiler', angle='radian')
    ET.SubElement(root, 'option', gravity='0 0 0')
    world = ET.SubElement(root, 'worldbody')
    palm = ET.SubElement(world, 'body', name='palm_lower')
    ET.SubElement(palm, 'freejoint')
    ET.SubElement(palm, 'inertial', pos='0 0 0', mass='1', diaginertia='.01 .01 .01')
    for index, name in enumerate(PALM_NAMES):
        ET.SubElement(palm, 'geom', name=name, type='sphere', size='.01',
                      pos=f'0 {index * .1} 0', contype='0', conaffinity='0')
    obj = ET.SubElement(world, 'body', name='fr3_e05_object', pos='.034 0 0')
    ET.SubElement(obj, 'geom', name='object', type='sphere', size='.02', contype='2', conaffinity='5')
    return ET.tostring(root, encoding='unicode')


class TestPalmCollision(unittest.TestCase):
    def test_original_overlap_is_missed_and_fixed_overlap_generates_force(self):
        xml = fixture()
        old = mujoco.MjModel.from_xml_string(xml)
        fixed = mujoco.MjModel.from_xml_string(enable_palm_collisions(xml))
        for model, expected in ((old, False), (fixed, True)):
            data = mujoco.MjData(model)
            data.qpos[0] = .014  # Two spheres overlap by 10 mm.
            mujoco.mj_forward(model, data)
            if expected:
                self.assertLess(PalmMonitor(model).observe(data)['clearance_m'], -.009)
                wrench = np.zeros(6)
                mujoco.mj_contactForce(model, data, 0, wrench)
                self.assertGreater(wrench[0], 0.)
            else:
                self.assertEqual(data.ncon, 0)

    def test_positive_query_buffer_applies_no_early_force(self):
        model = mujoco.MjModel.from_xml_string(enable_palm_collisions(fixture()))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        self.assertGreater(data.ncon, 0)
        self.assertAlmostEqual(PalmMonitor(model).observe(data)['clearance_m'], .004, places=12)
        for i in range(data.ncon):
            self.assertEqual(data.contact[i].efc_address, -1)
            wrench = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, wrench)
            np.testing.assert_array_equal(wrench, 0.)

    def test_geometry_and_inertial_properties_preserved(self):
        xml = fixture()
        old = mujoco.MjModel.from_xml_string(xml)
        fixed = mujoco.MjModel.from_xml_string(enable_palm_collisions(xml))
        for key in ('geom_type', 'geom_pos', 'geom_size', 'geom_quat', 'body_mass',
                    'body_inertia', 'body_parentid', 'jnt_qposadr'):
            np.testing.assert_array_equal(getattr(old, key), getattr(fixed, key))
        palm = mujoco.mj_name2id(fixed, mujoco.mjtObj.mjOBJ_BODY, 'palm_lower')
        self.assertTrue(fixed.body_contype[palm] & 8)
        self.assertTrue(fixed.body_conaffinity[palm] & 2)

    def test_reapplying_patch_rejected(self):
        with self.assertRaisesRegex(ValueError, 'pre-existing palm mask'):
            enable_palm_collisions(enable_palm_collisions(fixture()))

    def test_rejected_command_does_not_add_an_uncommitted_palm_row(self):
        from Module.i04_geometric_traversal.palm_safe_mcc import PalmSafeTraversal, FullBTraversal
        runtime = object.__new__(PalmSafeTraversal)
        obs = dict(palm=dict(clearance_m=.006, nearest_geom=''))
        for committed in (False, True):
            runtime.logs = {}
            def rejected_parent(self, command, phase, observation):
                if committed:
                    self.logs['time_s'] = [0.]
                raise RuntimeError('REJECTED_TEST_COMMAND')
            with patch.object(FullBTraversal, 'tick', rejected_parent):
                with self.assertRaisesRegex(RuntimeError, 'REJECTED_TEST_COMMAND'):
                    runtime.tick(np.zeros(23), 'TEST', obs)
            for key in ('palm_clearance_m', 'nearest_palm_geom'):
                self.assertEqual(len(runtime.logs.get(key, [])), int(committed))
            if not committed:
                self.assertEqual(runtime.logs, {})

    def test_predictive_retreat_uses_actual_closing_velocity(self):
        from Module.i04_geometric_traversal.palm_safe_recovery import PalmRecoveryTraversal
        xml = fixture().replace('.034', '.035')
        model = mujoco.MjModel.from_xml_string(enable_palm_collisions(xml))
        data = mujoco.MjData(model)
        runtime = object.__new__(PalmRecoveryTraversal)
        runtime.h = SimpleNamespace(model=model)
        runtime.data = data
        runtime.palm_monitor = PalmMonitor(model)
        for velocity, expected in ((.02, True), (-.02, False)):
            data.qvel[0] = velocity
            mujoco.mj_forward(model, data)
            obs = dict(palm=runtime.palm_monitor.observe(data))
            risk, projected = runtime.transfer_risk(obs)
            self.assertEqual(risk, expected)
            self.assertAlmostEqual(projected, .0026 if expected else .005, places=12)


if __name__ == '__main__':
    unittest.main()
