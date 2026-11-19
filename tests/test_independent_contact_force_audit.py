"""Known contact-force coordinates and net-force stop semantics."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'mcc_finger_compliance_control/scripts'))
from independent_contact_force_audit import NativeContactDecoder, stop_load


class NativeForceTests(unittest.TestCase):
    def test_native_pyramid_decoder_with_tangential_load(self):
        decoder = NativeContactDecoder()
        f = decoder.force(0, 3, [0], [.5,.25,0.,0.,0.], np.array([1.,2.,3.,4.]))
        np.testing.assert_allclose(f, [10.,-.5,-.25,0.,0.,0.])

    def test_elliptic_noncontiguous_addresses(self):
        f=NativeContactDecoder().force(1,3,[3,1,5],[.5]*5,np.arange(7.))
        np.testing.assert_array_equal(f,[3.,1.,5.,0.,0.,0.])

    def test_opposing_contacts_do_not_sum_magnitudes(self):
        p=np.zeros((2,4,3));c=p.copy();p[0,0,0]=80;c[0,0,0]=-90
        p[1,2,0]=60;c[1,2,0]=50
        peak,components=stop_load(p,c)
        np.testing.assert_array_equal(peak,[90.,110.])
        np.testing.assert_array_equal(components[0,0],[80.,90.,10.])

if __name__=='__main__':unittest.main()
