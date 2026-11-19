"""Guard child-reference timing and the complete-rollout selection boundary."""
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from run_variable_length_collection import (
    balanced_ranges, execution_contract, raw_palm_slice, rejection_reasons)


class VariableCollectionTests(unittest.TestCase):
    def test_balanced_children_cover_the_source_without_tiny_tail(self):
        ranges = balanced_ranges(327, 4765, 2000, 800)
        self.assertEqual(ranges[0][0], 327)
        self.assertEqual(ranges[-1][1], 4765)
        self.assertTrue(all(800 <= b-a <= 2000 for a,b in ranges))
        self.assertTrue(all(a[1] == b[0] for a,b in zip(ranges,ranges[1:])))

    def test_raw_slice_keeps_palm_timing_but_drops_old_joint_supervision(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'source.h5'; output=Path(directory)/'child.h5'
            pose=np.arange(20*7,dtype=np.float32).reshape(20,1,7)
            q=np.arange(20*16,dtype=np.float32).reshape(20,1,16)
            with h5py.File(source,'w') as h:
                h['palm_pose_object']=pose; h['q_hand']=q
                h['qvel']=np.ones_like(q);h['record_step']=np.arange(20)[:,None]
                h['episode_id']=np.zeros((20,1),int)
                h['grasp_keyframe_q']=q;h['fingertip_pose_object']=np.zeros((20,1,4,7))
                h.attrs.update(healthy_reference_admissible=True,finger_contact_plan=True,control_dt=.01)
            raw_palm_slice(source,output,5,15,'parentA')
            with h5py.File(output) as h:
                np.testing.assert_array_equal(h['palm_pose_object'][:],pose[5:15])
                np.testing.assert_array_equal(h['q_hand'][:],np.repeat(q[5:6],10,axis=0))
                np.testing.assert_array_equal(h['record_step'][:,0],np.arange(10))
                self.assertNotIn('grasp_keyframe_q',h)
                self.assertNotIn('fingertip_pose_object',h)
                self.assertFalse(h.attrs['healthy_reference_admissible'])
                self.assertEqual(h.attrs['related_plan_group_id'],'parentA')
                self.assertEqual(h.attrs['control_dt'],.01)

    def test_full_acceptance_never_hides_high_force_or_long_loss(self):
        row=dict(status='completed',stable_start=True,strict_healthy_four_rate_full=.95,
            strict_per_finger_full=[.95]*4,longest_observed_four_loss_frames=10,
            execution_contract=dict(passed=True),offline_motion_statistics=dict(over100_ms=0),
            peak_all_parts_n=100.)
        self.assertEqual(rejection_reasons(row),[])
        self.assertIn('over100N_during_motion',rejection_reasons(dict(row,
            offline_motion_statistics=dict(over100_ms=1),peak_all_parts_n=101.)))
        self.assertIn('persistent_contact_loss',rejection_reasons(dict(row,
            longest_observed_four_loss_frames=11)))
        self.assertIn('incomplete_execution',rejection_reasons(dict(row,status='initialization_error')))

    def test_substep_hold_and_first_motion_tick_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            (folder/'summary.json').write_text(json.dumps(dict(actual_motion_start_step=100)))
            np.savez(folder/'trace.npz',step=[101,102],force_setpoint=np.full((2,1,4),10.))
            physics=dict(step=np.repeat([100,101],10),substep=np.tile(np.arange(1,11),2),
                ctrl=np.zeros((20,1,16)),actuator_force=np.full((20,1,16),.8))
            np.savez(folder/'physics_trace.npz',**physics)
            with h5py.File(folder/'collected.h5','w') as h:
                h['episode_step']=np.array([[100],[101]])
                for name in ('q_hand','q_cmd','qvel'):h[name]=np.zeros((2,1,16))
                h['palm_command_pose_object']=np.zeros((2,7))
            row=dict(status='completed',requested_frames=2,force_audit=dict(passed=True))
            self.assertTrue(execution_contract(folder,row)['passed'])
            physics['ctrl'][0,0,0]=.01
            np.savez(folder/'physics_trace.npz',**physics)
            self.assertFalse(execution_contract(folder,row)['passed'])


if __name__=='__main__':unittest.main()
