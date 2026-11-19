"""Audit measured-motion imitation candidates without deleting WM failures.

Use the existing contact/force eligibility, then separately reject incomplete
records, joint-limit violations and previously flagged fast measured steps.
These are conservative DP label checks, not WM data or motion-coverage gates.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from prepare_wm_finger_candidates import index


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit-dirs',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    rows={}
    for folder in a.audit_dirs:
        for r in index(folder)['rows']:
            key=(r['folder'],r['lane'],r['finger'])
            if key in rows and rows[key]['rollout_sha256']!=r['rollout_sha256']:
                raise ValueError('Conflicting source hash')
            if key not in rows or r['candidate']:rows[key]=r
    folders=sorted({r['folder'] for r in rows.values()})
    for folder in folders:
        selected=[r for r in rows.values() if r['folder']==folder]
        with h5py.File(Path(folder)/'rollout.h5') as h:
            raw=h['raw_100hz'];q=raw['q'][:];meta=json.loads(h.attrs['metadata'])
            with np.load(Path(folder)/'object_path_plan.npz') as plan:
                expected=len(plan['commanded'])
            for r in selected:
                lane,f=r['lane'],r['finger']
                values=q[:,lane,4*f:4*f+4]
                steps=np.abs(np.diff(values,axis=0)).max(axis=1)
                r.update(measured_max_step_rad=float(steps.max()),
                    fast_step_frames=(np.flatnonzero(steps>.05)+1).tolist(),
                    complete_motion=len(q)-1==expected and meta['terminal_reason'] is None)
                report=json.loads(Path(r['source_audit']).read_text())
                source=next(t for t in report['reports'] if str(Path(t['folder']).resolve())==folder)
                diagnostic=next(t for t in source['rows'] if t['lane']==lane and t['finger']==f)
                r['minimum_joint_margin_rad']=diagnostic.get('minimum_joint_margin_rad')
                if not r['complete_motion']:r['reasons'].append('incomplete_motion')
                if len(r['fast_step_frames']):r['reasons'].append('fast_measured_step_pending_review')
                if r['minimum_joint_margin_rad'] is None or r['minimum_joint_margin_rad']<0:
                    r['reasons'].append('joint_margin_not_verified_nonnegative')
                r['candidate']=not r['reasons']
        print(Path(folder).parent.name,len(selected),'reviewed',flush=True)
    out=dict(complete=True,rows=list(rows.values()),unique_finger_trajectories=len(rows),
        candidates=sum(r['candidate'] for r in rows.values()),
        protocol=dict(max_measured_step_rad_per_10ms=.05,min_joint_margin_rad=0.,
            contact_force='unchanged original per-finger teacher gate',
            motion_coverage_threshold=False,raw_WM_records_preserved=True,
            labels='actual measured future q only',further_causal_export_validation_required=True))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(out,indent=2))
    print({k:v for k,v in out.items() if k not in ('rows','protocol')})


if __name__=='__main__':main()
