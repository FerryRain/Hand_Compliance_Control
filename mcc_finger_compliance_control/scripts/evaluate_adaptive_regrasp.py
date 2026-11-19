"""Standalone entry for opt-in outer supervisor v2; DP and MCC unchanged."""
import json
import sys
from pathlib import Path
import smooth_regrasp_reference as outer
from adaptive_regrasp import adaptive_class
import torch
from batched_b_control_chain import BatchedBControlChain
from palm_posture_reset import RELOAD, DP_WARMUP

if __name__=='__main__':
    task=json.loads(Path(sys.argv[1]).read_text())
    original=outer.regrasp_class
    original_orchestrate=BatchedBControlChain._orchestrate
    def orchestrate(chain,**kwargs):
        supervisor=getattr(chain,'command_supervisor',None)
        if supervisor is None:return original_orchestrate(chain,**kwargs)
        hold=(supervisor.phase==RELOAD)|(supervisor.phase==DP_WARMUP)
        if not bool(hold.any()):return original_orchestrate(chain,**kwargs)
        nominal=kwargs['nominal']
        result=nominal if bool(hold.all()) else original_orchestrate(chain,**kwargs)
        chain.runtime_recovery_offset[hold]=0
        chain.recovery_active[hold]=False
        chain.loss_streak[hold]=0
        chain.recovery_confirm_streak[hold]=0
        return torch.where(hold[:,None],nominal,result)
    BatchedBControlChain._orchestrate=orchestrate
    outer.regrasp_class=lambda:adaptive_class(task['adaptive_regrasp'])
    try:
        if task.get('matched_progress_noise',True):
            import evaluate_matched_regrasp
            evaluate_matched_regrasp.main(sys.argv[1])
        else:
            import evaluate_smooth_regrasp
            evaluate_smooth_regrasp.main(sys.argv[1])
    finally:
        outer.regrasp_class=original
        BatchedBControlChain._orchestrate=original_orchestrate
