"""Regressions for the hand-only plant, DP grasp gate and joint patch objective."""
import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np

from Module.i04_geometric_traversal.fingertip_hand import (
    HEAD_FRACTION, FingertipHand, build_model, reference_grasp,
)
from Module.i04_geometric_traversal.fingertip_continuous import ContinuousTraversal
from Module.i04_geometric_traversal.fingertip_plan import JointPatchPlanner
from Module.i04_geometric_traversal.fingertip_replay import audit_case
from Module.i04_geometric_traversal.fingertip_traversal import Traversal, surface_graph


class FingertipTraversalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hand = FingertipHand(build_model())
        cls.graph = surface_graph()

    def test_reference_and_incidental_palm_contact(self):
        h = self.hand
        p, r, q = reference_grasp()
        data = h.initialize(p, r, q)
        actual = h.measure(data)
        self.assertEqual((h.model.nv, h.model.nu), (22, 16))
        self.assertTrue(actual.eligible.all())
        self.assertTrue((actual.min_fraction >= HEAD_FRACTION).all())
        self.assertGreater(actual.posture_metrics[3], 0.)

        # Translating the same hand shape into a palm contact must be sensed
        # without turning the contact itself into an illegal grasp.
        data = h.initialize(p - np.array([0., 0., .105]), r, q)
        bad = h.measure(data)
        self.assertIn('palm_lower_collision', bad.incidental)
        self.assertTrue(bad.legal_posture)

        far = h.initialize(p + np.array([0., 0., .4]), r, q + .24)
        self.assertFalse(h.measure(far).legal_posture)

    def test_only_head_contacts_count_in_both_geom_orders(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        h = self.hand
        d = h.initialize(*reference_grasp())
        for f in (0, 3):
            body = h.tip_bodies[f]
            for fraction in (.4, .92):
                local = h.head_origins[f] + fraction * h.head_spans[f] * h.head_axes[f]
                point = d.xpos[body] + d.xmat[body].reshape(3, 3) @ local
                for a, b in ((h.pads[f], min(h.objects)), (min(h.objects), h.pads[f])):
                    c = SimpleNamespace(geom1=a, geom2=b, dist=-.0001, pos=point, frame=np.eye(3).ravel())
                    mock = SimpleNamespace(contact=[c], xmat=d.xmat, xpos=d.xpos, qpos=d.qpos,
                                           geom_xmat=d.geom_xmat, geom_xpos=d.geom_xpos)
                    def loaded(model, data, index, wrench):
                        wrench[0] = 1.
                    with patch.object(mujoco, 'mj_contactForce', loaded):
                        result = h.measure(mock)
                    self.assertEqual(bool(result.eligible[f]), fraction >= HEAD_FRACTION)
                    self.assertEqual(bool(result.incidental), fraction < HEAD_FRACTION)

    def test_simultaneous_incidental_load_does_not_erase_head_witness(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        h = self.hand
        d = h.initialize(*reference_grasp())
        f, obj = 0, min(h.objects)
        body = h.tip_bodies[f]
        contacts = []
        for geom, fraction in ((h.pads[f], .94), (h.pads[f], .5),
                               (h.model.geom('palm_lower_collision').id, 0.)):
            local = h.head_origins[f] + fraction * h.head_spans[f] * h.head_axes[f]
            point = d.xpos[body] + d.xmat[body].reshape(3, 3) @ local
            contacts.append(SimpleNamespace(geom1=obj, geom2=geom, dist=-.0001, pos=point,
                                            frame=np.eye(3).ravel()))
        sample = SimpleNamespace(contact=contacts, xmat=d.xmat, xpos=d.xpos, qpos=d.qpos,
                                 geom_xmat=d.geom_xmat, geom_xpos=d.geom_xpos)
        def loaded(model, data, index, wrench):
            wrench[0] = 1.
        with patch.object(mujoco, 'mj_contactForce', loaded):
            result = h.measure(sample)
        self.assertTrue(result.eligible[f])
        self.assertEqual(result.incidental, {'index_proximal_pad', 'palm_lower_collision'})
        self.assertEqual(result.forces[f], 1.)
        np.testing.assert_equal(result.loads, [2., 0., 0., 0., 1.])
        np.testing.assert_allclose(result.points[f], contacts[0].pos)

    def test_wrist_and_fingers_adapt_in_the_same_patch(self):
        planner = JointPatchPlanner(self.hand, self.graph)
        root = planner.seed
        tip = self.hand.measure(planner.data).points[1]
        vertex = self.graph.nearest_vertices([tip + [0., .03, 0.]])[0][0]
        patch = next(planner.candidates(vertex, root, 0))
        self.assertEqual(patch.points.shape, (4, 3))
        self.assertEqual(patch.normals.shape, (4, 3))
        np.testing.assert_allclose(patch.points[patch.anchor], self.graph.vertices_m[vertex])
        wrist_seed = self.graph.vertices_m[vertex] - patch.posture.orientation.apply(
            planner.reference_points[patch.anchor])
        self.assertGreater(np.linalg.norm(patch.posture.position - wrist_seed), .0005)
        self.assertTrue((abs(patch.posture.joints - root.joints).reshape(4, 4).max(axis=1) > .005).all())
        planner.set_pose(patch.posture)
        self.assertTrue(planner.buffered())

    def test_native_replay_of_short_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            case = Path(directory) / 'case'
            result = Traversal(case, .4).run()
            self.assertEqual(result['states'], 200)
            self.assertGreater(result['planned_patches'], 0)
            self.assertFalse(result['full_surface_complete'])
            self.assertEqual(result['stop_reason'], 'SIMULATION_BUDGET')
            replay = audit_case(case)
            self.assertTrue(replay['native_replay_matched'])
            self.assertEqual(replay['incidental_contact_steps'], result['incidental_contact_steps'])
            self.assertEqual(replay['illegal_grasp_states'], 0)

    def test_surface_stroke_visits_multiple_waypoints_with_contact(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            case = Path(directory) / 'case'
            summary = ContinuousTraversal(case, 8.).run()
            events = json.loads((case / 'planned_patches.json').read_bytes())
            self.assertGreaterEqual(len(events), 2)
            self.assertTrue(all(event['mode'] == 'surface_stroke' for event in events))
            self.assertTrue(any(len(event['route_goals']) > 1 for event in events))
            self.assertTrue(all(len(event['waypoints']) >= 3 for event in events))
            self.assertEqual(summary['lifts'], 0)
            self.assertEqual(summary['illegal_grasp_states'], 0)
            self.assertLess(summary['simultaneous_qualified_steps_0_to_4'][0], .05 * summary['states'])
            timing = summary['timing']
            self.assertGreaterEqual(timing['per_route_request_including_preview']['calls'], len(events))
            self.assertGreater(timing['per_joint_patch_solve']['calls'], len(events))
            self.assertTrue((case / 'planning_timing.npz').is_file())
            audit = audit_case(case)
            self.assertTrue(audit['native_replay_matched'])
            with np.load(case / 'native_head_contacts.npz') as contacts:
                self.assertEqual(len(contacts['point_m']), audit['head_observations'])
                self.assertTrue(np.all(contacts['normal_force_n'] >= .2))
                self.assertEqual(contacts['step'][-1], summary['states'])


if __name__ == '__main__':
    unittest.main()
