"""Failed initialization must remain in the audit denominator."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'mcc_finger_compliance_control/scripts'))
from analyze_collection_initialization import startup_metrics, assess_startup


class StartupAuditTests(unittest.TestCase):
    def fixture(self, directory, start, ready):
        summary=dict(actual_motion_start_step=start,
            error=None if ready else 'Initialization failed before motion',
            last_initialization_debug=[dict(ready=ready)],
            diagnostic_prefix_completed=ready)
        (directory/'summary.json').write_text(json.dumps(summary))
        steps=np.arange(1151,1201)
        force=np.zeros((50,1,4,3));force[...,0]=3.
        np.savez(directory/'trace.npz',step=steps,
            strict=np.ones((50,1,4),bool),found=np.ones((50,1,4),bool),
            teacher_calibrated=np.full((50,1),ready,bool),q=np.zeros((50,1,16)),
            pad_force=force,connector_force=np.zeros_like(force),force=np.full((50,1),3.))

    def test_timeout_is_measured_and_rejected_even_with_last_frame_contacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);self.fixture(path,None,False)
            row=assess_startup(startup_metrics(path))
            self.assertEqual(row['initial_strict_four_rate'],1.)
            self.assertEqual(row['initial_found_four_rate'],1.)
            self.assertEqual(row['measurement_window_frames'],50)
            self.assertEqual(row['prefix_observed_frames'],0)
            self.assertFalse(row['startup_pass'])

    def test_clean_start_keeps_both_metrics_and_separate_stationarity_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);self.fixture(path,1200,True)
            row=assess_startup(startup_metrics(path))
            self.assertTrue(row['startup_pass'])
            self.assertTrue(row['posture_stationarity_check'])
            self.assertEqual(row['measurement_window'],'pre_motion')
            self.assertEqual(row['initial_found_four_rate'],1.)


if __name__=='__main__':unittest.main()
