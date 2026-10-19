"""Contact transitions must preserve the commanded physical support target."""
import unittest
import numpy as np
from Module.i04_geometric_traversal.whole_hand_belly import contact_clearance_handoff, cooperative_window_ready


class CooperativeContactHandoff(unittest.TestCase):
    def test_early_contact_does_not_jump_inward_by_remaining_hover_clearance(self):
        clearance, offset = contact_clearance_handoff(.0015, .0002)
        self.assertAlmostEqual(clearance-offset, .0013)
        self.assertEqual(clearance, 0.)
        self.assertLess(offset, 0.)

    def test_mcc_saturation_preserves_clearance_instead_of_clipping_target(self):
        clearance, offset = contact_clearance_handoff(.006, -.0035)
        self.assertAlmostEqual(offset, -.004)
        self.assertGreater(clearance, 0.)
        self.assertAlmostEqual(clearance-offset, .0095)

    def test_every_finger_handoff_leaves_shared_target_positions_unchanged(self):
        rng=np.random.default_rng(42)
        for _ in range(100):
            clearance=rng.uniform(0,.006,4);offset=rng.uniform(-.004,.004,4)
            after=np.array([contact_clearance_handoff(c,o) for c,o in zip(clearance,offset)])
            np.testing.assert_allclose(after[:,0]-after[:,1],clearance-offset,atol=1e-12,rtol=0)
            self.assertTrue(np.all(abs(after[:,1])<=.004+1e-12))

    def test_three_contacts_with_a_passive_fourth_is_not_whole_hand_ready(self):
        safe=np.ones((150,4),bool);safe[:-1,3]=False
        self.assertFalse(cooperative_window_ready(safe))

    def test_brief_individual_gaps_do_not_require_all_fingers_to_restart(self):
        safe=np.ones((150,4),bool)
        for k in range(149):safe[k,k%4]=False
        self.assertTrue(cooperative_window_ready(safe))

    def test_one_or_two_finger_contact_cannot_pass_the_shared_handoff(self):
        safe=np.ones((150,4),bool);safe[:30,:2]=False
        self.assertFalse(cooperative_window_ready(safe))


if __name__=='__main__':unittest.main()
