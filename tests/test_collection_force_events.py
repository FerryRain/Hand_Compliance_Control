import sys
from pathlib import Path
import unittest
import json
import tempfile
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'mcc_finger_compliance_control/scripts'))
from collection_force_events import segments, summarize_force_events


class ForceEventTests(unittest.TestCase):
    def test_contiguous_duration_and_cluster_elapsed_are_distinct(self):
        mask=[False,True,True,False,False,True,False]
        self.assertEqual(segments(mask),[(1,2,2),(5,5,1)])
        self.assertEqual(segments(mask,gap=2),[(1,5,3)])
        self.assertEqual(segments([False]*5),[])

    def test_delayed_initialization_is_excluded_and_events_are_zero_based(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)
            (d/'summary.json').write_text(json.dumps(dict(actual_motion_start_step=1600)))
            pad=np.zeros((4,1,4,3));pad[...,0]=3.
            # A large startup transient must not contaminate motion peaks.
            pad[0,0,0,0]=900.;pad[2,0,0,0]=110.
            parts=np.zeros((4,1,4));parts[0,0,0]=900.;parts[2,0,0]=110.
            np.savez(d/'physics_trace.npz',step=[1599,1600,1601,1602],substep=[1]*4,
                pad_force=pad,connector_force=np.zeros_like(pad),
                pad_found=np.ones((4,1,4),bool),connector_found=np.zeros((4,1,4),bool),
                peak_force=parts.max(-1),part_net=parts,part_single=parts,
                part_geom=np.zeros((4,1,4),int),geom_names=['tip'],
                qpos=np.zeros((4,1,23)),qvel=np.zeros((4,1,22)),ctrl=np.zeros((4,1,16)))
            report=summarize_force_events(d)
            self.assertEqual(report['samples'],2)
            self.assertEqual(report['excluded_initialization_samples'],2)
            self.assertEqual(report['total_overforce_ms'],1)
            self.assertEqual(report['peak_n'],110.)
            event=json.loads((d/'force_events.json').read_text())['events'][0]
            self.assertEqual(event['start_task_frame'],0)
            self.assertEqual(event['peak_task_frame'],0)
            (d/'summary.json').write_text(json.dumps(dict(actual_motion_start_step=None)))
            report=summarize_force_events(d)
            self.assertEqual(report['samples'],0)
            self.assertEqual(report['total_overforce_ms'],0)
            self.assertIsNone(report['peak_n'])


if __name__=='__main__':unittest.main()
