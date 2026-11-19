import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'mcc_finger_compliance_control/scripts'))
from collect_trajectories import collection_abort_mask


class ForceResponseTests(unittest.TestCase):
    def setUp(self):
        self.q=torch.zeros(3,16)
        self.limits=torch.tensor([-1.,1.]).expand(3,16,2)

    def test_force_event_is_not_a_termination(self):
        force=torch.tensor([99.,101.,900.])
        self.assertEqual(collection_abort_mask(force,self.q,self.limits,100,'record').tolist(),[False]*3)
        self.assertEqual(collection_abort_mask(force,self.q,self.limits,100,'stop').tolist(),[False,True,True])

    def test_finite_joint_excursion_is_retained_for_diagnosis(self):
        self.q[1,2]=1.02
        self.assertFalse(collection_abort_mask(torch.zeros(3),self.q,self.limits,100,'record').any())
        self.assertEqual(collection_abort_mask(torch.zeros(3),self.q,self.limits,100,'stop').tolist(),[False,True,False])

    def test_numerical_invalidity_is_not_reported_as_full_execution(self):
        self.q[2,1]=torch.nan
        result=collection_abort_mask(torch.tensor([torch.inf,0.,0.]),self.q,self.limits,100,'record')
        self.assertEqual(result.tolist(),[True,False,True])


if __name__=='__main__':unittest.main()
