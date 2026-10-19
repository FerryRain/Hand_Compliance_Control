"""Regression for cancellation of an already applied retreat reference."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from Module.i04_geometric_traversal.reference_force_release import applied_reference_force_release


class AppliedReferenceReleaseTest(unittest.TestCase):
    def setUp(self):
        self.handles=SimpleNamespace(tip_site_ids=np.arange(4))
        self.config=SimpleNamespace(dt_s=.002,force_relief_speed_m_s=.030)
        self.jac=np.zeros((3,23));self.jac[0,7]=.1;self.jac[1,8]=.1;self.jac[2,9]=.1
        self.forces=np.array([9.,0.,0.,0.])
        self.normals=np.zeros((4,3));self.normals[0]=[1.,0.,0.]
        self.previous=np.zeros(23);self.previous[7]=.04
        self.proposed=np.arange(23)*.001

    def call(self, previous=None, forces=None, normals=None):
        with patch('Module.i04_geometric_traversal.reference_force_release.position_jacobian',
                   return_value=(self.jac,np.zeros_like(self.jac))):
            return applied_reference_force_release(self.handles,None,self.proposed,
                self.previous if previous is None else previous,
                self.forces if forces is None else forces,
                self.normals if normals is None else normals,self.config)

    def test_preserves_retreat_instead_of_reanchoring_below_previous_reference(self):
        # Measured q=0 would give ~0.0006 rad, retreating the command inward
        # from 0.04 rad. The actual applied reference must continue outward.
        result=self.call()
        self.assertGreater(result[7],self.previous[7])
        self.assertLess(result[7]-self.previous[7],.000601)
        np.testing.assert_array_equal(result[:7],self.proposed[:7])
        np.testing.assert_array_equal(result[11:],self.proposed[11:])

    def test_next_tick_integrates_from_the_new_applied_reference(self):
        first=self.call();second=self.call(previous=first)
        self.assertAlmostEqual(second[7]-first[7],first[7]-self.previous[7],places=14)

    def test_unloaded_or_missing_normal_preserves_existing_proposal(self):
        np.testing.assert_array_equal(self.call(forces=np.zeros(4)),self.proposed)
        np.testing.assert_array_equal(self.call(normals=np.zeros((4,3))),self.proposed)

    def test_inputs_not_modified(self):
        original=(self.previous.copy(),self.proposed.copy(),self.forces.copy(),self.normals.copy())
        self.call()
        for expected,actual in zip(original,(self.previous,self.proposed,self.forces,self.normals)):
            np.testing.assert_array_equal(actual,expected)

    def test_missing_or_nonfinite_applied_reference_rejected(self):
        for bad in (np.zeros(16),np.full(23,np.nan)):
            with self.assertRaises(ValueError):self.call(previous=bad)


if __name__=='__main__':unittest.main()
