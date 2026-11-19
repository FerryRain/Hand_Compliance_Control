"""Opt-in common flat/single provider, leaving the frozen recovery/MCC intact."""
import json
import runpy
import sys
from pathlib import Path
import single_finger_dp
from batched_b_control_chain import BatchedBControlChain
from controlled_policy_ablation import ControlledRuntime


if __name__=='__main__':
    request=Path(sys.argv[1]);task=json.loads(request.read_text())
    assert not task['matched_progress_noise'] # this provider owns the joint-space noise bank
    previous=single_finger_dp.SingleFingerRuntime
    step=BatchedBControlChain.step
    def attach(self,**kwargs):
        self.single_finger_runtime.supervisor=self.command_supervisor
        return step(self,**kwargs)
    single_finger_dp.SingleFingerRuntime=ControlledRuntime
    BatchedBControlChain.step=attach
    try:
        runpy.run_path(str(Path(__file__).with_name('evaluate_adaptive_regrasp.py')),run_name='__main__')
    finally:
        single_finger_dp.SingleFingerRuntime=previous
        BatchedBControlChain.step=step
