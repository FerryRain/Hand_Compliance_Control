"""The full-surface wrist planner may use rigid hand geometry but only issues wrist poses."""
import unittest

import numpy as np

from Module.i04_geometric_traversal.fingertip_traversal import surface_graph
from Module.i04_geometric_traversal.i01_a_contact_wrist import fit_contact_guide, policy_reference
from Module.i04_geometric_traversal.i01_a_full_wrist import EVALUATION, RigidFootprint, fixture, routing_adjacency


class ContactWristTests(unittest.TestCase):
    def test_two_static_hand_models_share_the_guide_and_keep_the_measured_start(self):
        surface, hand, grasp, _ = fixture('mustard')
        graph = surface_graph(surface)
        adjacency, _ = routing_adjacency(graph)
        seed = RigidFootprint(hand, grasp)
        touch = hand.measure(seed.data)
        typical, sources = policy_reference(EVALUATION)
        self.assertEqual({r['object'] for r in sources}, {'ycb_002_master_chef_can', 'ycb_021_bleach_cleanser'})
        operating = RigidFootprint(hand, (grasp[0], grasp[1], typical))
        start, _ = graph.nearest_vertex(touch.points[1])
        neighbor = adjacency.indices[adjacency.indptr[start]]
        guide = graph.vertices_m[[start, neighbor]]

        xyz, rotations, actual_guide, _, counts, meta = fit_contact_guide(
            graph, adjacency, seed, grasp, touch, guide, .008, supports=3, operating_footprint=operating)

        np.testing.assert_array_equal(actual_guide[meta['guide_prefix_knots']:], guide)
        np.testing.assert_array_equal(seed.data.qpos[hand.q], grasp[2])
        np.testing.assert_array_equal(operating.data.qpos[hand.q], typical)
        np.testing.assert_array_equal(xyz[0], grasp[0])
        np.testing.assert_allclose(rotations[0].as_matrix(), grasp[1].as_matrix())
        self.assertTrue(np.isfinite(xyz).all())
        self.assertEqual(counts[0], 4)
        self.assertEqual(meta['variables_optimized_per_knot'], 3)
        self.assertTrue((np.diff(xyz, axis=0) != 0.).any())


if __name__ == '__main__':
    unittest.main()
