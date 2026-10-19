"""Anatomical pad-face classification and native FK orientation constraints."""
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from Module.e05_physics.scene import PAD_HALF_SIZE_M, PAD_LOCAL_ROTATION
from Module.fr3_leap import ARM_HOME_Q, HAND_NATURAL_Q, FullRobotModelConfig, build_full_robot
from Module.i04_geometric_traversal.belly_contact import (
    BellyContactConfig, belly_alignment_terms, belly_contact_measurement,
    belly_ik_terms, classify_belly_patch, measured_belly_contacts, registered_belly_frame,
)


class BellyPatchTests(unittest.TestCase):
    def classify(self, local_point, normal=(0., 0., -1.), frame=None):
        frame = np.eye(3) if frame is None else frame
        center = np.array([.21, -.11, .42])
        return classify_belly_patch(center, frame, PAD_HALF_SIZE_M,
                                   center+frame@np.asarray(local_point), frame@np.asarray(normal))

    def test_central_anatomical_front_passes_but_back_and_edge_do_not(self):
        front = self.classify([0., 0., .002])
        self.assertTrue(front['geometry_qualified'])
        self.assertEqual(front['authority'], 'GEOMETRY_ONLY_NOT_CONTACT_OR_COVERAGE')
        back = self.classify([0., 0., -.002], (0., 0., 1.))
        self.assertFalse(back['geometry_qualified'])
        self.assertIn('NOT_PALMAR_HEMISPHERE', back['rejection_reasons'])
        edge = self.classify([-.012, 0., 0.])
        self.assertFalse(edge['geometry_qualified'])
        self.assertIn('OUTSIDE_CENTRAL_BELLY_PATCH', edge['rejection_reasons'])

    def test_good_axis_cannot_hide_rim_point_and_good_point_cannot_hide_wrong_axis(self):
        rim = self.classify([-.012*.95, 0., .002*np.sqrt(1.-.95**2)])
        self.assertEqual(rim['alignment_angle_deg'], 0.)
        self.assertFalse(rim['geometry_qualified'])
        wrong = self.classify([0., 0., .002], (0., 0., 1.))
        self.assertEqual(wrong['alignment_angle_deg'], 180.)
        self.assertFalse(wrong['geometry_qualified'])
        residual, _, _ = belly_alignment_terms(np.eye(3), [0., 0., 1.])
        self.assertEqual(np.linalg.norm(residual), 2., 'Antipodal orientation must not have zero IK residual')

    def test_thin_ellipsoid_requires_patch_test_in_addition_to_30_degree_test(self):
        # A 20-degree normal passes axis tolerance but its support is near the rim.
        normal = np.array([np.sin(np.deg2rad(20.)), 0., -np.cos(np.deg2rad(20.))])
        support = -(PAD_HALF_SIZE_M**2)*normal/np.linalg.norm(PAD_HALF_SIZE_M*normal)
        result = self.classify(support, normal)
        self.assertLess(result['alignment_angle_deg'], 30.)
        self.assertGreater(result['projected_patch_radius'], .8)
        self.assertFalse(result['geometry_qualified'])

    def test_projection_does_not_turn_interior_or_remote_points_into_belly_contact(self):
        self.assertFalse(self.classify([0., 0., 0.])['geometry_qualified'])
        self.assertFalse(self.classify([0., 0., .02])['geometry_qualified'])
        result = self.classify([0., 0., .0002])
        self.assertIn('POINT_NOT_ON_PAD_SURFACE', result['rejection_reasons'])

    def test_shell_tolerance_boundary_and_inconsistent_witness_normal_are_explicit(self):
        self.assertTrue(self.classify([0., 0., .0019])['geometry_qualified'])
        outside = self.classify([0., 0., .0018999])
        self.assertIn('POINT_NOT_ON_PAD_SURFACE', outside['rejection_reasons'])
        inconsistent = self.classify([0., 0., .002], [0., 0., 1.])
        self.assertEqual(inconsistent['contact_normal_cosine'], -1.)
        self.assertIn('CONTACT_NORMAL_OPPOSES_PAD_SURFACE_NORMAL', inconsistent['rejection_reasons'])

    def test_classification_is_rigid_transform_invariant(self):
        point = np.array([.004, -.002, .002*np.sqrt(1.-(.004/.012)**2-(.002/.008)**2)])
        original = self.classify(point)
        rotated = self.classify(point, frame=Rotation.from_rotvec([.4, -.7, .2]).as_matrix())
        self.assertEqual(original['geometry_qualified'], rotated['geometry_qualified'])
        for key in ('projected_patch_radius', 'radial_shell_error_m', 'alignment_cosine'):
            self.assertAlmostEqual(original[key], rotated[key], places=12)

    def test_invalid_geometry_cannot_silently_pass(self):
        for kwargs in ({'maximum_alignment_angle_deg':90.}, {'maximum_patch_radius':1.},
                       {'surface_tolerance_m':float('nan')}):
            with self.assertRaises(ValueError): BellyContactConfig(**kwargs)
        with self.assertRaises(ValueError): self.classify([0., 0., .002], (0., 0., 0.))
        with self.assertRaises(ValueError): self.classify([0., 0., .002], frame=np.diag([1., 1., -1.]))


class NativeBellyIKTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.h = build_full_robot(FullRobotModelConfig(surface='plane', gravity_m_s2=0.))

    def setUp(self):
        self.data = mujoco.MjData(self.h.model)
        self.data.qpos[self.h.arm_qpos_adrs] = ARM_HOME_Q
        self.data.qpos[self.h.hand_qpos_adrs] = HAND_NATURAL_Q
        mujoco.mj_forward(self.h.model, self.data)

    def test_native_registered_z_is_body_minus_x_for_all_four_fingers(self):
        np.testing.assert_array_equal(PAD_LOCAL_ROTATION[:, 2], [-1., 0., 0.])
        for geom in self.h.tip_geom_ids:
            frame = registered_belly_frame(self.h.model, self.data, int(geom))
            body = int(self.h.model.geom_bodyid[geom])
            anatomical = self.data.xmat[body].reshape(3, 3) @ np.array([-1., 0., 0.])
            np.testing.assert_allclose(frame[:, 2], anatomical, atol=1e-12, rtol=0.)
            np.testing.assert_allclose(self.h.model.geom_size[geom], PAD_HALF_SIZE_M, atol=0., rtol=0.)

    def test_full_six_row_jacobian_matches_native_finite_differences_all_fingers_and_arm(self):
        model = self.h.model
        original = self.data.qpos.copy()
        maximum_orientation_term = 0.
        for finger, geom in enumerate(self.h.tip_geom_ids):
            normal = np.array([.23, -.61, .76]); normal /= np.linalg.norm(normal)
            target = self.data.geom_xpos[geom]+[.001, -.002, .003]
            columns = np.r_[self.h.arm_dof_adrs, self.h.hand_dof_adrs[4*finger:4*finger+4]]
            residual, jacobian, diag = belly_ik_terms(model, self.data, int(geom), target, normal,
                clearance_m=-.0003, orientation_weight_m=.025, dof_indices=columns)
            self.assertEqual(residual.shape, (6,))
            finite = np.zeros_like(jacobian)
            eps = 1e-6
            for j, column in enumerate(columns):
                delta = np.zeros(model.nv); delta[column] = 1.
                self.data.qpos[:] = original
                mujoco.mj_integratePos(model, self.data.qpos, delta, eps)
                mujoco.mj_forward(model, self.data)
                plus = belly_ik_terms(model, self.data, int(geom), target, normal,
                                     clearance_m=-.0003, orientation_weight_m=.025)[0]
                self.data.qpos[:] = original
                mujoco.mj_integratePos(model, self.data.qpos, delta, -eps)
                mujoco.mj_forward(model, self.data)
                minus = belly_ik_terms(model, self.data, int(geom), target, normal,
                                      clearance_m=-.0003, orientation_weight_m=.025)[0]
                finite[:, j] = (plus-minus)/(2*eps)
            self.data.qpos[:] = original; mujoco.mj_forward(model, self.data)
            np.testing.assert_allclose(jacobian, finite, atol=2e-9, rtol=2e-6)
            maximum_orientation_term = max(maximum_orientation_term, float(np.max(abs(jacobian[3:]))))
        self.assertGreater(maximum_orientation_term, .02, 'The regression must exercise nonzero orientation derivatives')

    def test_zero_position_error_at_back_support_still_fails_belly_constraint(self):
        geom = int(self.h.tip_geom_ids[0])
        frame = registered_belly_frame(self.h.model, self.data, geom)
        normal = frame[:, 2]
        back = self.data.geom_xpos[geom]-.002*normal
        residual, _, diag = belly_ik_terms(self.h.model, self.data, geom, back, normal)
        self.assertLess(np.linalg.norm(residual[:3]), 1e-12)
        self.assertAlmostEqual(np.linalg.norm(residual[3:]), .02)
        self.assertFalse(diag['geometry_qualified'])

    def test_helpers_leave_native_state_and_physical_model_unchanged(self):
        state = {key:getattr(self.data, key).copy() for key in ('qpos', 'qvel', 'ctrl', 'qacc_warmstart')}
        model = {key:getattr(self.h.model, key).copy() for key in ('geom_size', 'geom_quat', 'geom_margin', 'geom_solref')}
        geom = int(self.h.tip_geom_ids[0])
        belly_ik_terms(self.h.model, self.data, geom, [0., 0., 0.], [0., 0., 1.])
        belly_contact_measurement(self.h, self.data)
        for key, value in state.items(): np.testing.assert_array_equal(getattr(self.data, key), value)
        for key, value in model.items(): np.testing.assert_array_equal(getattr(self.h.model, key), value)


class NativeContactFilteringTests(unittest.TestCase):
    @staticmethod
    def fixture(pad_first=False, back=False, rim=False):
        # Actual native contact forces, not mocked sensor values. The physical pad
        # has the same registered body frame and half sizes as the experiment.
        pad_rotation = Rotation.from_matrix(PAD_LOCAL_ROTATION).as_quat()
        pad_quaternion = ' '.join(map(str, np.r_[pad_rotation[3], pad_rotation[:3]]))
        # belly +Z points down; back +Z points up, both physically touch plane.
        orient = Rotation.from_euler('x', 180. if not back else 0., degrees=True).as_matrix()
        if rim: orient = Rotation.from_euler('y', 90., degrees=True).as_matrix()
        body_rotation = orient @ PAD_LOCAL_ROTATION.T
        q = Rotation.from_matrix(body_rotation).as_quat(); body_q = ' '.join(map(str, np.r_[q[3], q[:3]]))
        z = .0118 if rim else .0018
        floor = '<geom name="object" type="plane" size="1 1 .1"/>'
        body = f'<body pos="0 0 {z}" quat="{body_q}"><freejoint/><geom name="pad" type="ellipsoid" size=".012 .008 .002" quat="{pad_quaternion}" mass=".02"/></body>'
        xml = '<mujoco><option gravity="0 0 -9.81"/><worldbody>'+ (body+floor if pad_first else floor+body) +'</worldbody></mujoco>'
        model = mujoco.MjModel.from_xml_string(xml); data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'pad')
        obj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'object')
        return SimpleNamespace(model=model, tip_geom_ids=np.array([geom]), object_geom_id=obj), data

    def test_native_front_force_is_kept_and_reverse_side_force_remains_raw_only(self):
        for pad_first in (False, True):
            h, data = self.fixture(pad_first=pad_first)
            force, _, normal, active, diag = belly_contact_measurement(h, data, threshold=.01)
            self.assertGreater(force[0], .01)
            self.assertTrue(active[0])
            np.testing.assert_allclose(normal[0], [0., 0., 1.], atol=1e-12)
            np.testing.assert_allclose(diag['records'][0]['pad_witness_world_m'][2], -.0002, atol=1e-12)
            np.testing.assert_allclose(diag['raw_contact_force_n'], force, atol=0., rtol=0.)
        for kwargs in ({'back':True}, {'rim':True}):
            h, data = self.fixture(**kwargs)
            force, _, _, active, diag = belly_contact_measurement(h, data, threshold=.01)
            self.assertGreater(diag['raw_contact_force_n'][0], .01)
            self.assertEqual(force[0], 0.)
            self.assertFalse(active[0])
            self.assertGreater(diag['rejected_contact_force_n'][0], .01)

    def test_off_object_geometric_proximity_creates_no_contact_evidence(self):
        h, data = self.fixture()
        data.qpos[2] += .1; mujoco.mj_forward(h.model, data)
        self.assertEqual(measured_belly_contacts(h, data), [])
        force, _, _, active, _ = belly_contact_measurement(h, data)
        np.testing.assert_array_equal(force, [0.])
        np.testing.assert_array_equal(active, [False])


if __name__ == '__main__': unittest.main()
