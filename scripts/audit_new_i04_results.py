"""Independent result integrity audit and matched-time diagnostics."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    output = args.directory.resolve()
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    result = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    mismatches = [p for p, digest in manifest['code_hashes'].items()
                  if sha256((ROOT / p).read_bytes()).hexdigest() != digest]
    if mismatches:
        raise RuntimeError(f'Runtime source changed: {mismatches}')
    rows = result['episodes']
    if len(rows) != len(manifest['seeds']) * len(manifest['cells']):
        raise RuntimeError('Missing episodes')
    for row in rows:
        folder = output / f"{row['cell']}_seed{row['seed']}"
        if sha256((folder / 'trace.npz').read_bytes()).hexdigest() != row['trace_sha256']:
            raise RuntimeError('Trace hash mismatch')
        with np.load(folder / 'trace.npz') as trace:
            for key in ('q_rad', 'command_rad', 'palm_pose_world', 'forces_n', 'coverage'):
                if not np.all(np.isfinite(trace[key])):
                    raise RuntimeError(f'Nonfinite {key}')
            if np.any(np.diff(trace['time_s']) <= 0):
                raise RuntimeError('Nonmonotonic trace')
            if np.any(np.diff(trace['coverage']) < -1e-10):
                raise RuntimeError('Coverage ledger regressed')
            if not 0 <= row['contact_continuity'] <= 1:
                raise RuntimeError('Contact ratio outside [0,1]')
            if not np.all((np.array(row['per_finger_contact_fraction']) >= 0)
                          & (np.array(row['per_finger_contact_fraction']) <= 1)):
                raise RuntimeError('Per-finger ratios outside [0,1]')
        events = json.loads((folder / 'events.json').read_text(encoding='utf-8'))
        expected = ['wrist_pose'] if row['cell'] == 'A-MCC' else ['wrist_pose', 'tip_positions']
        commits = [e for e in events if e['kind'] == 'PLAN_COMMIT']
        if not all(e['exposed_fields'] == expected for e in commits):
            raise RuntimeError('Method information boundary violated')
        if row['task_contact_veto_count'] != 0:
            raise RuntimeError('Task-level contact veto detected')
        if row['required_goal_count'] != manifest['geometry']['required_goals']:
            raise RuntimeError('Surface denominator changed')
    matched = []
    for seed in manifest['seeds']:
        group = [r for r in rows if r['seed'] == seed]
        if len({r['initial_state']['state_sha256'] for r in group}) != 1:
            raise RuntimeError('Paired cells did not share initialization')
        endpoint = min(r['executed_post_acquisition_s'] for r in group)
        pair = {'seed': seed, 'common_duration_s': endpoint,
                'sampling_s': manifest['config']['dt_s'] * manifest['config']['log_stride'],
                'scope': 'Post-hoc diagnostics from saved 10 ms samples; not a new acceptance gate',
                'cells': {}}
        for row in group:
            with np.load(output / f"{row['cell']}_seed{seed}" / 'trace.npz') as trace:
                mask = trace['time_s'] <= endpoint
                count = trace['active'][mask].sum(axis=1)
                pair['cells'][row['cell']] = {
                    'last_sample_s': float(trace['time_s'][mask][-1]),
                    'contact_continuity': float(np.mean(count > 0)),
                    'mean_contacts': float(np.mean(count)),
                    'four_contact_fraction': float(np.mean(count == 4)),
                    'covered_area_fraction': float(trace['coverage'][mask][-1]),
                    'sampled_peak_force_n': float(np.max(trace['forces_n'][mask]))}
        matched.append(pair)
    audit = {'status': 'VERIFIED', 'episode_count': len(rows), 'runtime_source_hashes_match': True,
             'trace_hashes_match': True, 'paired_initial_states_match': True,
             'all_method_boundaries_match': True, 'all_traces_finite': True,
             'all_ratios_bounded': True, 'no_task_contact_veto': True,
             'complete_surface_claims': sum(r['full_surface_complete'] for r in rows)}
    (output / 'integrity_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    (output / 'matched_time_diagnostics.json').write_text(json.dumps(matched, indent=2), encoding='utf-8')
    print(json.dumps(audit, indent=2))
    print(json.dumps(matched, indent=2))


if __name__ == '__main__':
    main()
