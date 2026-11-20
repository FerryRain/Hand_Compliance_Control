"""Opt-in common flat/single provider, leaving the frozen recovery/MCC intact."""
import json
import runpy
import sys
from pathlib import Path

# Git on Windows can check out the script symlinks as plain text.
shared_scripts = Path(__file__).resolve().parents[2] / "mcc_finger_compliance_control" / "scripts"
sys.path.insert(0, str(shared_scripts))

import single_finger_dp
from batched_b_control_chain import BatchedBControlChain
from controlled_policy_ablation import ControlledRuntime


if __name__ == "__main__":
    if sys.platform == "win32":
        # Concurrent NVRTC jobs on Windows can delete each other's temporary .pch files.
        import warp
        warp.config.use_precompiled_headers = False

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
        runpy.run_module('evaluate_adaptive_regrasp', run_name='__main__')
    finally:
        single_finger_dp.SingleFingerRuntime=previous
        BatchedBControlChain.step=step
