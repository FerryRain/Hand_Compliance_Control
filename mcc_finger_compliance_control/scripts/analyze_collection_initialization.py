"""Compare startup quality without treating a short prefix as a full rollout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def startup_metrics(directory):
    summary = json.loads((directory/'summary.json').read_text())
    z = np.load(directory/'trace.npz')
    start = summary.get('actual_motion_start_step', 1200)
    end = start if start is not None else int(z['step'][-1]) if len(z['step']) else 0
    window = (z['step'] > max(end-50, 799)) & (z['step'] <= end)
    task = z['step'] > start if start is not None else np.zeros(len(z['step']), bool)
    debug = summary.get('last_initialization_debug', [{}])[0]
    def fraction(key, mask):
        return float(z[key][mask, 0].all(-1).mean()) if mask.any() else None
    return dict(
        directory=str(directory), error=summary['error'], motion_start_step=start,
        measurement_window_end_step=end,
        measurement_window=('pre_motion' if start is not None else 'end_of_failed_initialization'),
        measurement_window_frames=int(window.sum()),
        initial_strict_four_rate=fraction('strict', window),
        initial_found_four_rate=fraction('found', window),
        calibrated_start=(bool(z['teacher_calibrated'][window, 0].all())
                          if window.any() else False),
        final_50_frame_q_span_max_rad=(float(np.ptp(z['q'][window, 0], axis=0).max())
                                     if window.any() else None),
        initial_force_range_per_finger_n=(np.stack((np.linalg.norm(z['pad_force'][window, 0], axis=-1).min(0),
            np.linalg.norm(z['pad_force'][window, 0], axis=-1).max(0)), axis=-1).tolist()
            if window.any() else None),
        initial_connector_force_max_n=(float(np.linalg.norm(z['connector_force'][window, 0], axis=-1).max())
                                       if window.any() else None),
        prefix_observed_frames=int(task.sum()),
        prefix_strict_four_rate=fraction('strict', task),
        prefix_found_four_rate=fraction('found', task),
        prefix_peak_force_n=float(z['force'][task, 0].max()) if task.any() else None,
        prefix_complete=bool(summary.get('diagnostic_prefix_completed', False)),
        last_initialization_debug=debug,
        frame0_overrides=summary.get('initialization_overrides', []))


def assess_startup(row):
    """Startup gate is unchanged; stationarity is reported independently."""
    row['startup_pass'] = bool(row['motion_start_step'] is not None
        and row['measurement_window_frames'] == 50
        and row['initial_strict_four_rate'] == 1.
        and row['initial_found_four_rate'] == 1. and row['calibrated_start']
        and row['initial_connector_force_max_n'] < .05
        and row['last_initialization_debug'].get('ready', False))
    span = row['final_50_frame_q_span_max_rad']
    row['posture_stationarity_check'] = bool(span is not None and span <= .005)
    row['posture_stationarity_span_limit_rad'] = .005
    # This check observes stationary joint configuration; it does not claim
    # the material contact point has zero physical slip during acquisition.
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--ids', nargs='+', type=int, required=True)
    args = parser.parse_args()
    rows = []
    for eid in args.ids:
        directory = args.root/f'candidate_{eid:04d}_v2'
        if not (directory/'summary.json').exists():
            directory = args.root/f'candidate_{eid:04d}_v2_retry'
        if not (directory/'summary.json').exists():
            continue
        original = json.loads((args.source/'results'/f'candidate_{eid:04d}.json').read_text())
        row = startup_metrics(directory)
        row.update(candidate_id=eid, family=original['family'],
                   original_stable_start=original['stable_start'])
        assess_startup(row)
        rows.append(row)
    result = dict(
        scope='Measured startup + first 200 original-speed motion frames; not full trajectory certification',
        control='Only initialization changed; tube disabled for this startup diagnostic',
        metrics='Strict means true split fingertip contact >=0.05N; found includes true fingertip/connector',
        grasp_hold='After first load, no intentional tangent or null-space motion; normal force correction only',
        attribution='Reference old startup status; not a paired seed/tube full-rollout comparison',
        ready_frames=50, rows=rows, attempted=len(rows), passed=sum(r['startup_pass'] for r in rows))
    (args.root/'initialization_summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('scope', 'attempted', 'passed')}, indent=2))
    for row in rows:
        print(row['candidate_id'], row['startup_pass'],
              'strict/found', row['initial_strict_four_rate'], row['initial_found_four_rate'],
              'q span', row['final_50_frame_q_span_max_rad'],
              'prefix strict/found', row['prefix_strict_four_rate'], row['prefix_found_four_rate'])


if __name__ == '__main__':
    main()
