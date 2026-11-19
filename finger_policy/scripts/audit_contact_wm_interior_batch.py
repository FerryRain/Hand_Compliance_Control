"""Inventory complete fixed-grasp collection attempts, including failures.

Contact/force gates use the frozen evaluator. Motion coverage is reported
separately. This inventory is not blanket admission to WM training.
"""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from audit_contact_wm_interior import audit
from audit_contact_wm_reference import audit as teacher_gate


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def inspect(folder):
    if not (folder/'rollout.h5').exists():
        return dict(folder=str(folder),status='no_motion_rollout',training_admissible=False)
    authority=audit(folder);teacher=teacher_gate(folder)
    with h5py.File(folder/'rollout.h5') as h:
        meta=json.loads(h.attrs['metadata']);raw=h['raw_100hz'];q=raw['post_mcc_q'][:]
        active=raw['active_finger'][0].astype(bool)
        enabled=(raw['mcc_final_enabled'][1:]>0)&active[None]
        signed=raw['mcc_final_direction'][1:]*raw['mcc_final_normal_after'][1:]
        limits=meta['mcc_config'];bound=limits['action_rate_limit']*limits['command_ema_alpha']
        command_check=dict(enabled_frames=int(enabled.sum()),
            signed_direction_violations=int(((signed < -1.e-8)&enabled).sum()),
            maximum_step_rad=float(np.abs(np.diff(q,axis=0)).max()),
            rate_bound_rad=bound,
            rate_bound_violation_rad=float(np.maximum(np.abs(np.diff(q,axis=0))-bound,0).max()))
        starts=h['view_20hz/state_index'][:];ends=h['view_20hz/next_state_index'][:]
        assert np.array_equal(ends,starts+5)
        assert np.all(np.abs(raw['dt'][1:]-.01)<1.e-7)
        lane_count=q.shape[1]
        transitions=len(starts)*lane_count
        # The five actual actuator commands of each 50ms interval are kept.
        assert np.array_equal(h['view_20hz/actuator_q_sequence'][:],
            np.stack([raw['actuator_target_q'][k+1:k+6] for k in starts]))
        assert np.array_equal(h['view_20hz/pre_mcc_q_sequence'][:],
            np.stack([raw['pre_mcc_q'][k+1:k+6] for k in starts]))
    for row,gate in zip(authority['rows'],teacher['rows']):
        row['teacher_contact_force_gate']=gate['teacher_contact_force_gate']
        span=row['loaded_contact_pca_span_mm']
        row['motion_coverage_gate']=bool(max(row['measured_q_range_deg'])>=10 and span is not None
                                       and span[0]>=100 and span[1]>=30)
        row['force_p95_substep_n']=gate['force_p95_n']
        row['longest_over25n_s']=gate['longest_force_above25n_s']
    expected_frames = None
    if (folder/'object_path_plan.npz').exists():
        with np.load(folder/'object_path_plan.npz') as plan:
            expected_frames = len(plan['commanded'])
    return dict(folder=str(folder),object=meta['object_id'],split=meta['split'],
        status='complete' if expected_frames is not None and q.shape[0]-1==expected_frames and meta['terminal_reason'] is None else 'partial',
        recorded_frames=q.shape[0]-1,expected_frames=expected_frames,
        lane_count=lane_count, active_finger_row_count=len(authority['rows']),
        controller=meta['mcc_candidate_overrides'],rows=authority['rows'],physics=authority['physics'],
        commands=command_check,model_sha256=sha(folder/'physics_model.mjb'),
        rollout_sha256=sha(folder/'rollout.h5'),transitions_20hz=transitions,
        physical_motion_seconds=float((q.shape[0]-1)*.01*q.shape[1]),
        contact_force_pass=teacher['teacher_contact_force_pass'],
        coverage_pass=sum(r['motion_coverage_gate'] for r in authority['rows']),
        coverage_unit='active finger within lane; not an independent trajectory',
        correct_five_command_transition_alignment=True,training_admissible=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folders',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();reports=[]
    for folder in a.folders:
        result=inspect(folder);reports.append(result)
        print(folder.parent.name,folder.name,result.get('contact_force_pass'),
              result.get('coverage_pass'),result.get('transitions_20hz'),flush=True)
        summary=dict(reports=reports,total_lanes=sum(r.get('lane_count',0) for r in reports),
            total_active_finger_rows=sum(r.get('active_finger_row_count',0) for r in reports),
            transitions_20hz=sum(r.get('transitions_20hz',0) for r in reports),
            contact_force_pass=sum(r.get('contact_force_pass',0) for r in reports),
            coverage_pass=sum(r.get('coverage_pass',0) for r in reports),training_admissible=False,
            scope='Single- or multi-finger collection attempts. Lanes count trajectories; per-finger coverage rows share a lane. Force-paired paths are related trials, not independent samples. No WM model-performance claim.')
        a.output.write_text(json.dumps(summary,indent=2))


if __name__=='__main__':main()
