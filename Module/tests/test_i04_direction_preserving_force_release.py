import unittest
import numpy as np
from Module.i04_geometric_traversal.direction_preserving_force_release import scale_step_to_joint_box


class DirectionPreservingReliefTests(unittest.TestCase):
    def test_real_terminal_slew_counterexample_preserves_direction(self):
        raw=np.array([-.00022667456892300208,.0030097661917580787,-.0006015078005091185,-.012096455131403616])
        step,diag=scale_step_to_joint_box(np.zeros(4),raw,-np.ones(4),np.ones(4))
        np.testing.assert_allclose(step,raw*diag['alpha'],atol=0,rtol=0)
        self.assertAlmostEqual(diag['alpha'],.19840523309753433)
        self.assertLessEqual(abs(step).max(),.0024)
        self.assertGreater(step[1],0.)
        self.assertLess(step[1],.0006)

    def test_one_blocked_joint_holds_entire_direction(self):
        q=np.array([0.,0.,0.,-.351]);d=np.array([-.00023,.00105,.00146,-.00952])
        step,diag=scale_step_to_joint_box(q,d,np.array([-1.,-1.,-1.,-.351]),np.ones(4))
        np.testing.assert_array_equal(step,np.zeros(4))
        self.assertEqual(diag['reason'],'DIRECTION_BLOCKED_HOLD')

    def test_joint_limit_can_be_tighter_than_slew(self):
        q=np.array([.999,0.,0.,0.]);raw=np.array([.002,.001,-.004,.0002])
        step,diag=scale_step_to_joint_box(q,raw,-np.ones(4),np.ones(4))
        self.assertAlmostEqual(diag['alpha'],.5)
        self.assertTrue(np.all(q+step<=1))
        np.testing.assert_array_equal(step,raw*diag['alpha'])

    def test_illegal_previous_is_explicitly_infeasible_not_component_repaired(self):
        step,diag=scale_step_to_joint_box(np.array([1.1,0,0,0]),np.ones(4)*-.001,-np.ones(4),np.ones(4))
        self.assertFalse(diag['feasible'])
        np.testing.assert_array_equal(step,np.zeros(4))

    def test_zero_and_nonfinite(self):
        step,diag=scale_step_to_joint_box(np.zeros(4),np.zeros(4),-np.ones(4),np.ones(4))
        self.assertEqual(diag['alpha'],1.)
        np.testing.assert_array_equal(step,np.zeros(4))
        with self.assertRaises(ValueError):
            scale_step_to_joint_box(np.zeros(4),np.array([0.,0.,np.nan,0.]),-np.ones(4),np.ones(4))


if __name__=='__main__':unittest.main()
