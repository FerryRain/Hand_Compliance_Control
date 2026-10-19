"""Read-only same-prefix evidence for a running coordinated mustard scan.

This command never stops a run, changes its files, or credits a command as
contact. It snapshots completed trace bytes, validates a continuous saved grid,
and waits for the requested horizon. Coverage comes from that exact trace row,
never a later rolling checkpoint. Output is a separate offline JSON artifact.
Native event success, sampled same-finger confirmation and audit status remain
separate; a 100 Hz prefix cannot prove absence of hidden 2 ms contact gaps.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import zipfile

import numpy as np

from .active_sliding_report import KEYS, _loss_metrics
from .sliding_metrics import analyze_trace
from .sliding_report import _initial_pair


GENERATED = Path(__file__).resolve().parents[1]/'generated'
DEFAULT_CASE = GENERATED/'mustard_coordinated_prepared_run01_20261016/B-MCC_seed7'
HISTORICAL_CASES = (
    ('v2.2', 'mustard_active_sliding_run04_20261016'),
    ('v3.2', 'mustard_coordinated_run02_20261016'),
    ('v3.3-priority', 'mustard_coordinated_slide_priority_20261016'),
    ('v3.4-context', 'mustard_coordinated_context_20261016'),
    ('v3.5-boundary', 'mustard_coordinated_boundary_20261016'),
)


def _json_snapshot(path):
    path = Path(path)
    if not path.exists():
        return None, None
    raw = path.read_bytes()
    return json.loads(raw), dict(path=str(path.resolve()), sha256=sha256(raw).hexdigest(), bytes=len(raw))


def read_completed_prefix(directory, end_s=300.):
    """Return one frozen prefix or WAITING; do not bridge missing trace rows.

    A partially written final chunk is ignored. A malformed earlier chunk,
    nonuniform grid, duplicate timestamps, missing metric fields, or a prefix
    beginning after zero is an error, not a shortened comparison.
    """
    directory, end_s = Path(directory), float(end_s)
    if not np.isfinite(end_s) or end_s <= 0:
        raise ValueError('end_s must be positive and finite')
    full = directory/'trace.npz'
    paths = [full] if full.exists() else sorted((directory/'trace_chunks').glob('*.npz'))
    chunks, sources = [], []
    for index, path in enumerate(paths):
        raw = path.read_bytes()
        try:
            with np.load(BytesIO(raw), allow_pickle=False) as saved:
                times = np.array(saved['time_s'], copy=True)
                if not len(times):
                    continue
                count = int(np.searchsorted(times, end_s+1e-9, side='right'))
                chunk = {key: np.array(saved[key][:count], copy=True) for key in KEYS if key in saved.files}
        except (EOFError, zipfile.BadZipFile, OSError, ValueError):
            if path != full and index == len(paths)-1:
                break
            raise
        sources.append(dict(path=str(path.resolve()), sha256=sha256(raw).hexdigest(), bytes=len(raw),
                            saved_start_s=float(times[0]), saved_end_s=float(times[-1])))
        if len(chunk['time_s']):
            chunks.append(chunk)
        if times[-1] >= end_s-1e-9:
            break
    if not chunks:
        return dict(status='WAITING_FOR_COMPLETED_PREFIX', requested_end_s=end_s,
                    available_end_s=None, sources=sources, trace=None)
    keys = set(chunks[0])
    required = {'time_s','active','forces_n','phase','q_rad','coverage','covered_goals',
                'contact_positions_world_m','tip_positions_world_m','normals_world'}
    if not required.issubset(keys) or any(set(c) != keys for c in chunks):
        raise ValueError('Incomplete or inconsistent trace metric schema')
    trace = {key: np.concatenate([c[key] for c in chunks]) for key in keys}
    t = trace['time_s']
    if abs(t[0]) > 1e-9 or len(t) < 2 or np.any(np.diff(t) <= 0):
        raise ValueError('A strictly increasing prefix beginning at zero is required')
    dt = np.diff(t)
    if not np.allclose(dt, np.median(dt), atol=1e-9, rtol=0):
        raise ValueError('Trace has a missing row or changed sampling interval')
    ready = abs(t[-1]-end_s) <= 1e-9
    return dict(status='READY' if ready else 'WAITING_FOR_COMPLETED_PREFIX', requested_end_s=end_s,
                available_end_s=float(t[-1]), sources=sources, trace=trace if ready else None)


def phase_gap_attribution(trace):
    """Attribute real whole-hand absence to saved phases without judging lifts."""
    t, active, forces, phase = (trace[k] for k in ('time_s','active','forces_n','phase'))
    contact = np.any(active & np.isfinite(forces) & (forces >= .2), axis=1)
    dt = np.diff(t)
    phases = {}
    for label in np.unique(phase):
        mask = phase[:-1] == label
        absent = mask & ~contact[:-1]
        longest = current = 0.
        for flag, duration in zip(absent, dt):
            current = current+duration if flag else 0.
            longest = max(current, longest)
        phases[str(label)] = dict(duration_s=float(dt[mask].sum()),
            observed_absence_s=float(dt[absent].sum()), longest_absence_within_phase_s=float(longest))
    absent = ~contact
    starts = np.flatnonzero(absent & np.r_[True,~absent[:-1]])
    ends = np.flatnonzero(absent & np.r_[~absent[1:],True])
    intervals = []
    for start, last in zip(starts, ends):
        stop = min(last+1,len(t)-1)
        by_phase = {str(p):float(dt[start:stop][phase[start:stop]==p].sum())
                    for p in np.unique(phase[start:stop])}
        intervals.append(dict(start_s=float(t[start]), end_s=float(t[stop]),
            duration_s=float(t[stop]-t[start]), phase_durations_s=by_phase,
            touches_prefix_end=bool(last==len(t)-1)))
    return dict(phases=phases, longest_whole_hand_absence_intervals=sorted(intervals,key=lambda x:-x['duration_s'])[:20],
                interpretation='Descriptive phase attribution; intentional air is retained and is not automatically failure')


def _window_contact(t, active, forces):
    if len(t) < 2:
        return None
    dt = np.diff(t)
    actual = active & np.isfinite(forces) & (forces >= .2)
    safe = actual & (forces < 8.)
    longest_gap = longest_safe = gap = stable = 0.
    for i, d in enumerate(dt):
        gap = gap+d if not actual[i] else 0.
        stable = stable+d if safe[i] else 0.
        longest_gap, longest_safe = max(longest_gap,gap), max(longest_safe,stable)
    return dict(start_s=float(t[0]),end_s=float(t[-1]),duration_s=float(t[-1]-t[0]),
                contact_fraction=float(np.sum(dt*actual[:-1])/sum(dt)),
                safe_contact_fraction=float(np.sum(dt*safe[:-1])/sum(dt)),
                longest_absence_s=float(longest_gap),longest_safe_contact_s=float(longest_safe))


def recovery_evidence(trace, events, *, source_run_audit_verified=False, followup_s=30.):
    """Expose measured motion and event/sample agreement, never command success."""
    t, q = trace['time_s'], trace['q_rad']
    horizon = float(t[-1])
    events = [e for e in events if float(e.get('time_s',np.inf)) <= horizon+1e-9]
    episodes = []
    for begin in (e for e in events if e.get('kind')=='COORDINATED_RECOVERY_BEGIN'):
        sequence, finger = begin['sequence'], int(begin['selected_finger'])
        end = next((e for e in events if e.get('kind')=='COORDINATED_RECOVERY_END'
                    and e.get('sequence')==sequence),None)
        first = next((e for e in events if e.get('kind')=='COORDINATED_RECOVERY_FIRST_CONTACT'
                      and e.get('sequence')==sequence),None)
        stop = float(end['time_s']) if end else horizon
        selected = (t >= begin['time_s']) & (t <= stop)
        joint = q[selected,7+4*finger:11+4*finger]
        measured = _window_contact(t[selected],trace['active'][selected,finger],trace['forces_n'][selected,finger])
        # Include the sample just before the native terminal event to diagnose
        # agreement, not to replace the executor's 2 ms stability evidence.
        terminal = (t >= stop-.16) & (t <= stop)
        terminal_measured = _window_contact(t[terminal],trace['active'][terminal,finger],trace['forces_n'][terminal,finger])
        stable_claim = bool(end and end.get('measured_recovery_success'))
        followup = (t >= stop) & (t <= min(horizon,stop+followup_s))
        after_terminal = (_window_contact(t[followup],trace['active'][followup,finger],trace['forces_n'][followup,finger])
                          if end else None)
        episodes.append(dict(sequence=sequence, finger=finger, begin_s=begin['time_s'],end_s=stop,
            complete_within_prefix=bool(end), terminal_status=end.get('status') if end else 'IN_PROGRESS_AT_PREFIX_END',
            terminal_reason=end.get('reason') if end else None,
            dispatch_mode=begin.get('dispatch_mode'),path_duration_s=begin.get('duration_s'),
            actual_phase_duration_s=stop-float(begin['time_s']),
            first_native_contact_event_s=first['time_s'] if first else None,
            native_stable_event_claim=stable_claim,
            native_stable_event_source_audit_verified=bool(stable_claim and source_run_audit_verified),
            sampled_terminal_contact=terminal_measured, actual_contact_during_episode=measured,
            sampled_terminal_safe_span_at_least_150ms=bool(terminal_measured and terminal_measured['longest_safe_contact_s'] >= .15-1e-9),
            actual_joint_range_deg=np.rad2deg(np.ptp(joint,axis=0)).tolist() if len(joint) else None,
            actual_joint_net_change_deg=np.rad2deg(joint[-1]-joint[0]).tolist() if len(joint) else None,
            after_terminal_contact=after_terminal,
            followup_contact=after_terminal if stable_claim else None))
    return dict(episodes=episodes, actual_recovery_phase_duration_s=sum(e['actual_phase_duration_s'] for e in episodes),
        native_stable_event_count=sum(e['native_stable_event_claim'] for e in episodes),
        audited_native_stable_event_count=sum(e['native_stable_event_source_audit_verified'] for e in episodes),
        limitation='Native events are provisional until their source run audit matches; 100Hz samples cannot certify uninterrupted 2ms safety')


def _case_evidence(directory, frozen):
    directory, trace = Path(directory), frozen['trace']
    horizon = float(trace['time_s'][-1])
    events, event_source = _json_snapshot(directory/'events.json')
    events = [e for e in (events or []) if e['time_s'] <= horizon+1e-9]
    summary, summary_source = _json_snapshot(directory/'summary.json')
    manifest, manifest_source = _json_snapshot(directory/'manifest.json')
    audit_path = next((directory/name for name in ('audit_summary.json','native_audit.json','audit.json')
                       if (directory/name).exists()), directory/'audit.json')
    audit, audit_source = _json_snapshot(audit_path)
    trace_source = next((s for s in frozen['sources'] if Path(s['path']).name=='trace.npz'),None)
    verified = bool(summary and audit and trace_source and audit.get('status',audit.get('result'))=='PASSED'
        and summary.get('trace_sha256')==audit.get('trace_sha256')==trace_source['sha256'])
    return dict(directory=str(directory.resolve()), horizon_s=horizon,
        protocol=(manifest or {}).get('protocol'), status='FINISHED' if summary else 'RUNNING_PREFIX',
        final_run_stop_reason=(summary or {}).get('stop_reason'), source_run_audit_verified=verified,
        full_surface_claim=False, coverage=float(trace['coverage'][-1]),covered_goals=int(trace['covered_goals'][-1]),
        sampled_peak_force_n=float(np.max(trace['forces_n'])),
        native_whole_run_peak_force_n=(summary or {}).get('peak_force_n'),
        force_peak_scope='Prefix peak is sampled at100Hz; final whole-run peak is separate and may include time after this prefix',
        metrics=analyze_trace(trace,events=events), loss=_loss_metrics(trace),
        phase_gap_attribution=phase_gap_attribution(trace),
        recovery=recovery_evidence(trace,events,source_run_audit_verified=verified),
        trace_sources=frozen['sources'], evidence=dict(events=event_source,summary=summary_source,
            manifest=manifest_source,audit=audit_source), source_hashes=(manifest or {}).get('source_hashes',{}))


def observe(directory=DEFAULT_CASE, end_s=300., comparisons=None):
    """One nonblocking observation; callers decide when to invoke again."""
    directory = Path(directory)
    current = read_completed_prefix(directory,end_s)
    result = dict(schema='i04.coordinated_prefix_observer.v1',
        created_utc=datetime.now(timezone.utc).isoformat(),current_case=str(directory.resolve()),
        status=current['status'],requested_horizon_s=float(end_s),available_end_s=current['available_end_s'],
        full_surface_claim=False,source_scope='READ_ONLY_FROZEN_PREFIX',
        method='Identical sliding_metrics.v1.2 and saved time grid for every case',
        causal_scope='Descriptive combined-strategy comparison; not a single-component ablation',
        side_effects='Writes only caller-selected offline output; no stop request, controller or source modifications')
    if current['status'] != 'READY':
        return result
    result['current'] = _case_evidence(directory,current)
    result['comparisons'], result['unavailable_comparisons'] = {}, {}
    comparisons = ({label:GENERATED/f'{name}/B-MCC_seed7' for label,name in HISTORICAL_CASES}
                   if comparisons is None else comparisons)
    for label, case in comparisons.items():
        case = Path(case)
        saved = read_completed_prefix(case,end_s)
        if saved['status'] != 'READY':
            result['unavailable_comparisons'][label] = dict(status=saved['status'],available_end_s=saved['available_end_s'])
            continue
        initial = _initial_pair(directory,case)
        if not initial['exactly_equal']:
            raise ValueError(f'{label}: initial qpos/qvel/ctrl mismatch')
        if not np.array_equal(current['trace']['time_s'],saved['trace']['time_s']):
            raise ValueError(f'{label}: saved time grid mismatch')
        result['comparisons'][label] = dict(initial_state=initial,**_case_evidence(case,saved))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case',type=Path,nargs='?',default=DEFAULT_CASE)
    parser.add_argument('--end',type=float,default=300.)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    result = observe(args.case,args.end)
    output = args.output or GENERATED/'mustard_coordinated_review_20261016'/f'prepared_prefix_{args.end:g}s_comparison.json'
    # Never use an experiment path as the default output, including while waiting.
    if args.case.resolve() in output.resolve().parents:
        raise ValueError('Output must be outside the observed experiment directory')
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary = output.with_suffix(output.suffix+'.tmp')
    temporary.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(output)
    print(json.dumps(dict(status=result['status'],available_end_s=result['available_end_s'],output=str(output.resolve()),
        full_surface_claim=False),ensure_ascii=False))


if __name__=='__main__':
    main()
