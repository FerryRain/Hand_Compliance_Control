import unittest
import numpy as np
from Module.i04_geometric_traversal.active_sliding_video import control_intent_label


def record(phase='FINGER_RECOVERY'):
    return dict(phase=np.array([phase]),active=np.array([[True,False,False,False]]),
        forces_n=np.array([[2.,0.,0.,0.]]),finger_search_state=np.full((1,4),'DISABLED',dtype='U32'),
        coordinated_recovery_finger=np.array([3]),coordinated_recovery_sequence=np.array([1]),
        coordinated_recovery_stage=np.array([0]))


class ControlLabelTests(unittest.TestCase):
    def test_nonselected_safe_contact_is_mcc_hold_not_disabled_search(self):
        self.assertEqual(control_intent_label(record(),0,0,{}),'MCC CONTACT HOLD')
        self.assertEqual(control_intent_label(record(),0,1,{}),'WAITING DURING RECOVERY')

    def test_selected_path_and_stabilization_are_distinct(self):
        self.assertEqual(control_intent_label(record(),0,3,{}),'PLANNED FINGER MOTION')
        self.assertEqual(control_intent_label(record(),0,3,{1:'WRIST_PIVOT'}),'PLANNED WRIST MOTION')
        self.assertEqual(control_intent_label(record('FINGER_STABILIZE'),0,3,{}),'CONTACT STABILIZE')

    def test_force_alone_cannot_claim_hold_in_air_unknown_or_without_actual_contact(self):
        self.assertEqual(control_intent_label(record('LIFT'),0,0,{}),'AIR / REPOSITION')
        self.assertEqual(control_intent_label(record('UNKNOWN'),0,0,{}),'LOCAL SEARCH OFF')
        r=record();r['active'][0,0]=False
        self.assertEqual(control_intent_label(r,0,0,{}),'WAITING DURING RECOVERY')
        r['active'][0,0]=True;r['forces_n'][0,0]=8.
        self.assertNotEqual(control_intent_label(r,0,0,{}),'MCC CONTACT HOLD')

    def test_lost_finger_local_search_is_retained_during_slide(self):
        r=record('SLIDE');r['finger_search_state'][0,1]='SEEK'
        self.assertEqual(control_intent_label(r,0,1,{}),'SEEK')


if __name__=='__main__':unittest.main()
