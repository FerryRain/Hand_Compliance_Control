"""Isolated evaluation entry: frozen DP/MCC, optional pre-MCC reference filter.

All monkeypatches are process-local. Existing evaluation entry stays unchanged.
"""
import json
import sys
from pathlib import Path
import numpy as np


def main(request):
    task = json.loads(Path(request).read_text())
    import deploy_b_gpu_serial_parity as deployment
    import evaluate_regional_dp as evaluation
    from batched_b_control_chain import BatchedBControlChain
    from smooth_regrasp_reference import ReferenceMotionFilter, regrasp_class
    if task.get('regrasp_initial_healthy'):
        if not task.get('local_contact_search') or not task.get('local_recovery'):
            raise ValueError('Regrasp requires the bounded recovery/search transaction')
        import local_contact_search
        local_contact_search.LocalContactSearch = regrasp_class()
    capacity = task.get('simulation_constraint_capacity')
    original_cfg = deployment.replay_env_cfg
    if capacity:
        def configured(**kwargs):
            cfg = original_cfg(**kwargs)
            cfg.sim.njmax = max(cfg.sim.njmax or 0, int(capacity))
            return cfg
        deployment.replay_env_cfg = configured
    original = BatchedBControlChain._orchestrate
    records = []
    settings = task.get('reference_motion_filter')
    def orchestrate(self, **kwargs):
        target = original(self, **kwargs)
        if settings is None:
            return target
        if not hasattr(self, '_reference_motion_filter'):
            self._reference_motion_filter = ReferenceMotionFilter(self.mcc.lower, self.mcc.upper, **settings)
        target = self._reference_motion_filter(target, self.previous_nominal)
        records.append(dict(frame=self.control_frame, **{
            k:v.detach().cpu().numpy() for k,v in self._reference_motion_filter.diagnostic.items()}))
        return target
    BatchedBControlChain._orchestrate = orchestrate
    try:
        evaluation.worker(Path(request))
    finally:
        if records:
            np.savez_compressed(Path(task['output']).with_suffix('.reference_filter.npz'),
                **{k:np.stack([r[k] for r in records]) for k in records[0]})
        BatchedBControlChain._orchestrate = original
        deployment.replay_env_cfg = original_cfg


if __name__ == '__main__':
    main(sys.argv[1])
