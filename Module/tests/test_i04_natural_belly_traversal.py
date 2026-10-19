"""Safety/coverage contracts for natural whole-object exploration."""
from types import SimpleNamespace
import unittest
import numpy as np

from Module.i04_geometric_traversal.natural_belly_traversal import (
    hand_reference, posture_bounds, raster_goals, route_endpoint_reached,
)


class NaturalTraversalContracts(unittest.TestCase):
    def test_segment_handoff_tolerates_short_contact_bouts_at_observed_endpoint(self):
        endpoint=np.zeros(3)
        self.assertTrue(route_endpoint_reached(.095,.1,True,[.0027,0,0],endpoint))
        self.assertFalse(route_endpoint_reached(.095,.1,False,[.0027,0,0],endpoint))
        self.assertFalse(route_endpoint_reached(.05,.1,True,[.0027,0,0],endpoint))
        self.assertFalse(route_endpoint_reached(.095,.1,True,[.006,0,0],endpoint))

    def test_pivot_cannot_finish_merely_because_its_contact_point_is_stationary(self):
        self.assertFalse(route_endpoint_reached(.095,.1,True,np.zeros(3),np.zeros(3),pivot=True))
        self.assertTrue(route_endpoint_reached(.1,.1,True,np.zeros(3),np.zeros(3),pivot=True))

    def test_all_lead_choices_keep_long_finger_splay_small_and_distals_nonnegative(self):
        lo,hi=posture_bounds(np.full(23,-3.),np.full(23,3.))
        for lead in range(4):
            for broad in (False,True):
                q=np.r_[np.zeros(7),hand_reference(lead,broad)]
                self.assertTrue(np.all((q>=lo)&(q<=hi)))
                self.assertTrue(np.all(q[7:19:4]==0))
                self.assertTrue(np.all(q[[9,10,13,14,17,18]]>=0))
            curled=np.r_[np.zeros(7),hand_reference(lead,curved=True)]
            self.assertTrue(np.all((curled>=lo)&(curled<=hi)))

    def test_task_posture_never_expands_physical_ranges(self):
        physical_lo=np.linspace(-.2,-.1,23);physical_hi=np.linspace(.3,.4,23)
        lo,hi=posture_bounds(physical_lo,physical_hi)
        self.assertTrue(np.all(lo>=physical_lo));self.assertTrue(np.all(hi<=physical_hi))

    def test_old_splayed_reverse_flexion_is_rejected(self):
        lo,hi=posture_bounds(np.full(23,-3.),np.full(23,3.))
        old=np.r_[np.zeros(7),hand_reference()]
        old[7]=np.radians(59.);old[9]=np.radians(-13.)
        self.assertFalse(np.all((old>=lo)&(old<=hi)))

    def test_raster_retains_every_original_goal_including_both_ends(self):
        normals=np.vstack([np.tile(s*np.eye(3)[a],(5,1)) for a in range(3) for s in (-1,1)])
        rng=np.random.default_rng(19);points=rng.uniform(-.25,.25,(len(normals),3))
        required=np.arange(len(points))[::-1]
        graph=SimpleNamespace(vertices_m=points,normals=normals,required_vertices=required)
        before=required.copy();ordered=raster_goals(graph)
        np.testing.assert_array_equal(np.sort(ordered),np.arange(len(points)))
        np.testing.assert_array_equal(graph.required_vertices,before)
        self.assertEqual(len(set(ordered)),len(points))


if __name__=='__main__':unittest.main()
