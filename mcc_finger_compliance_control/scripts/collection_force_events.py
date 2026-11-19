"""Offline force episodes and before/after state windows from 1ms traces."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np


def segments(mask, gap=0):
    indices=np.flatnonzero(mask)
    if not len(indices):return []
    cut=np.flatnonzero(np.diff(indices)>gap+1)+1
    return [(int(x[0]),int(x[-1]),len(x)) for x in np.split(indices,cut)]


def summarize_force_events(directory):
    directory=Path(directory);p=directory/'physics_trace.npz'
    if not p.exists():return None
    archive=np.load(p);threshold=100.
    summary_path=directory/'summary.json'
    summary=json.loads(summary_path.read_text()) if summary_path.exists() else {}
    motion_start=summary.get('actual_motion_start_step',1200)
    task=(archive['step']>motion_start if motion_start is not None
          else np.zeros(len(archive['step']),bool))
    # Initial acquisition transients are reported separately by the startup
    # audit. They must not become motion overforce events or shifted labels.
    sample_keys=('step','substep','pad_force','connector_force','pad_found',
        'connector_found','part_net','part_single','peak_force','part_geom',
        'qpos','ctrl','qvel')
    z={key:archive[key][task] for key in sample_keys}
    z['geom_names']=archive['geom_names']
    pad=np.linalg.norm(z['pad_force'][:,0],axis=-1)
    connector=np.linalg.norm(z['connector_force'][:,0],axis=-1)
    combined=np.linalg.norm(z['pad_force'][:,0]+z['connector_force'][:,0],axis=-1)
    comp=np.stack([pad,connector,combined],axis=-1)
    parts=np.maximum(z['part_net'][:,0],z['part_single'][:,0])
    sensor=z['peak_force'][:,0];peak=np.maximum(sensor,parts.max(-1))
    finite=np.isfinite(peak);above=finite&(peak>threshold)
    strict=(z['pad_found'][:,0] & np.isfinite(pad) & (pad>=.05)).all(-1)
    found=(z['pad_found'][:,0]|z['connector_found'][:,0]).all(-1)
    healthy=strict&finite&(peak<=threshold)
    good_run=np.convolve(healthy.astype(np.int32),np.ones(100,dtype=np.int32),mode='valid')==100 if len(healthy)>=100 else np.empty(0,bool)
    good_indices=np.flatnonzero(good_run)
    raw=segments(above);clusters=segments(above,gap=20)
    rows=[];names=z['geom_names'];labels=('true_fingertip','connector','other_link','palm')
    for start,end,n in clusters:
        k=start+int(np.argmax(peak[start:end+1]));region=int(parts[k].argmax())
        gid=int(z['part_geom'][k,0,region]);finger,component=np.unravel_index(comp[k].argmax(),comp[k].shape)
        left=max(0,start-100);right=min(len(peak),end+201)
        recovery=good_indices[good_indices>end]
        rows.append(dict(start_task_frame=int(z['step'][start]-motion_start-1),start_substep=int(z['substep'][start]),
            end_task_frame=int(z['step'][end]-motion_start-1),end_substep=int(z['substep'][end]),
            peak_task_frame=int(z['step'][k]-motion_start-1),peak_substep=int(z['substep'][k]),
            threshold_exceeded_ms=int(n),cluster_elapsed_ms=end-start+1,
            peak_force_n=float(peak[k]),peak_sensor_net_n=float(sensor[k]),
            peak_region=labels[region],peak_region_geom=str(names[gid]) if gid>=0 else None,
            sensor_peak_finger=('index','middle','ring','thumb')[finger],
            sensor_peak_component=('pad','connector','combined')[component],
            pad_forces_at_peak_n=pad[k].tolist(),connector_forces_at_peak_n=connector[k].tolist(),
            single_point_peaks_at_peak_n=z['part_single'][k,0].tolist(),
            strict_100ms_before=float(strict[left:start].mean()) if start>left else None,
            found_100ms_before=float(found[left:start].mean()) if start>left else None,
            strict_200ms_after=float(strict[end+1:right].mean()) if right>end+1 else None,
            found_200ms_after=float(found[end+1:right].mean()) if right>end+1 else None,
            subsequent_peak_200ms_n=float(peak[end+1:right].max()) if right>end+1 else None,
            first_100ms_strict_and_under100_reacquisition_delay_ms=(int(recovery[0]-end) if len(recovery) else None),
            q_before=z['qpos'][left,0,-16:].tolist(),q_at_peak=z['qpos'][k,0,-16:].tolist(),q_after=z['qpos'][right-1,0,-16:].tolist(),
            actual_target_at_peak=z['ctrl'][k,0,-16:].tolist(),
            actual_target_previous_substep=z['ctrl'][max(0,k-1),0,-16:].tolist(),
            velocity_at_peak=z['qvel'][k,0,-16:].tolist(),
            trace_index_start=start,trace_index_end=end,trace_index_peak=k))
    output=dict(threshold_n=100,physics_dt_ms=1,samples=len(peak),
        monitored_force='max of per-finger pad/connector/combined net force AND per-geom net/single contact on all hand regions',
        total_overforce_ms=int(above.sum()),overforce_fraction=float(above.mean()) if len(above) else 0.,
        raw_contiguous_events=len(raw),longest_contiguous_overforce_ms=max((e-s+1 for s,e,_ in raw),default=0),
        clusters_merge_gap_ms=20,clusters=len(clusters),peak_n=float(peak[finite].max()) if finite.any() else None,
        nonfinite_samples=int((~finite).sum()),events=rows,
        actual_motion_start_step=motion_start,task_frame_indexing='zero based; step - motion_start - 1',
        excluded_initialization_samples=int((~task).sum()),
        timing='Forces are from the solve, q/qvel after its integration. Windows are100ms before/200ms after. Cluster elapsed includes gaps; exceeded_ms counts actual samples above100N. Full trace retains every 1ms sample.')
    (directory/'force_events.json').write_text(json.dumps(output,indent=2)+'\n')
    return {k:v for k,v in output.items() if k not in ('events','timing')}
