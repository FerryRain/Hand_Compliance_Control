"""Collect a certified reference with nonpenetrating measured-palm IK.

This process-local adapter preserves the frozen collector/MCC and replaces
only its initial geometric solve. Contact load is built by the existing
normal force servo, not by a geometric preload target.
"""
from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path
import sys

import h5py
import numpy as np


def install_initializer(object_id,bank):
    import collection_initialization as initialization
    from healthy_grasp_planner import HealthyGraspPlanner,state_report
    original=initialization.StableContactInitializer

    class HealthyInitializer(original):
        def __init__(self,controller,object_config,scale,seed_q):
            super().__init__(controller,object_config,scale,seed_q)
            self.healthy_planner=HealthyGraspPlanner(object_id,scale,Path(bank))
            self.geometry=self.healthy_planner.geometry

        def _solve(self,pose):
            q,state=self.healthy_planner.solve(pose,self.nominal_q,multistart=3)
            if not state['valid']:
                # Never load an invalid geometric initialization. The caller
                # reports this candidate as failed, without starting motion.
                raise RuntimeError('No healthy nonpenetrating measured-palm reference found: '
                    +json.dumps(state_report(state)))
            self.nominal_q=np.asarray(q,float)
            self.last_solve_pose=pose.copy()
            self.solved_geometry_valid=True
            self.solve_health=dict(isotropic_reachability=state['health'][0].tolist(),
                manipulability=state['health'][1].tolist(),sigma_min=state['health'][2].tolist(),
                planned_gap_m=state['geometry']['distance'].tolist(),
                geometry_preload_m=0.,local_capacity_m=state['capacity_m'].tolist())
            self.solve_count+=1
    initialization.StableContactInitializer=HealthyInitializer


def main():
    if '--help' in sys.argv or '-h' in sys.argv:
        from collect_long_oracle_probe import main as collect
        collect()
        return
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--plan',type=Path,required=True)
    known,_=parser.parse_known_args()
    with h5py.File(known.plan) as h:
        if not h.attrs.get('healthy_reference_admissible',False):
            raise ValueError('A certified healthy reference is required')
        calibration=json.loads(h.attrs['healthy_reference_calibration_json'])
        object_id=str(h.attrs['object_id'])
    from healthy_grasp_planner import DEFAULT_BANK
    install_initializer(object_id,calibration.get('bank_path',str(DEFAULT_BANK)))
    if '--stable-initialization' not in sys.argv:sys.argv.append('--stable-initialization')
    from collect_long_oracle_probe import main as collect
    try:
        collect()
    finally:
        if '--output' in sys.argv:
            report=Path(sys.argv[sys.argv.index('--output')+1])/'summary.json'
            if report.exists():
                value=json.loads(report.read_text())
                value.update(healthy_initialization_adapter=True,initialization_geometry_preload_m=0.,
                    initialization_planner_sha256=hashlib.sha256(Path(__file__).with_name('healthy_grasp_planner.py').read_bytes()).hexdigest(),
                    initialization_adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
                report.write_text(json.dumps(value,indent=2)+'\n')


if __name__=='__main__':main()
