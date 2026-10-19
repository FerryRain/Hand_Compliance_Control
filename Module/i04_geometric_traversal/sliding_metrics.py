"""Measure sampled, same-finger contact migration without rewarding a static hold.

``analyze_trace`` accepts the native I04 trace mapping (or an eagerly loaded NPZ).
All fractions are time weighted over ``time_s[-1] - time_s[0]``. An interval
counts only when both endpoint samples satisfy its condition; no unobserved
contact or motion is credited. Movement requires at least 1 mm/s net contact
displacement over a complete 0.5 s window inside ONE uninterrupted finger run.
When fingertip positions are available, their displacement must corroborate
movement too. This deliberately excludes the first 0.5 s after each reacquisition.
If ``normals_world`` is recorded, all motion and distance measurements are
projected into the tangent plane of the averaged endpoint surface normals.
Pure normal indentation then contributes no moving contact. Invalid normals
break motion evidence; antipodal endpoint normals conservatively give zero
projected displacement rather than an arbitrary tangent plane.

Paths use 0.1 s samples and a 0.5 mm spatial deadband rather than summing noisy
10 ms contact positions. A sliding bout requires uninterrupted moving contact
for 2 s and at least 10 mm displacement from its first point. Short gaps may be
debounced for a separate recovery description, but NEVER contribute to strict
sliding bouts, motion windows, or path lengths. Fingers are never joined through
a hand-wide contact centroid. Optional phase/stroke summaries use these same
rules, not planned distance as evidence of measured motion.

These are conservative sampled kinematic metrics on a fixed object. They do
not prove frictional slip (as opposed to rolling), continuous intersample
contact, or object-relative motion if the object moves. A changing simulator
contact manifold can also move its representative point; fingertip corroboration
and the bout displacement threshold reduce, but cannot eliminate, that ambiguity.

Per-finger participation uses physical force samples. Optional measured
``finger_q_rad[N,F,J]`` or canonical I04 ``q_rad[N,23]`` adds actual joint
movement; command arrays never count as movement. Explicit ``finger_search_active``
or FINGER_SEARCH_BEGIN/END events delimit attempted searches. A search outcome
requires the SAME finger's measured safe contact to persist for 150 ms, and
actual joint movement is reported separately. This temporal association does
not establish whether finger motion, wrist motion, or both caused reacquisition.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path

import numpy as np


def _runs(mask):
    """Return inclusive sample-index runs, without joining fingers or gaps."""
    edges = np.diff(np.r_[False, np.asarray(mask, dtype=bool), False].astype(np.int8))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1)


def _duration(mask, dt):
    mask = np.asarray(mask, dtype=bool)
    return float(np.sum(dt[mask[:-1] & mask[1:]]))


def _tangent_displacement(displacement, first_normal, last_normal):
    first_norm = np.linalg.norm(first_normal, axis=-1, keepdims=True)
    last_norm = np.linalg.norm(last_normal, axis=-1, keepdims=True)
    first_unit = np.divide(first_normal, first_norm, out=np.zeros_like(first_normal), where=first_norm > 1e-10)
    last_unit = np.divide(last_normal, last_norm, out=np.zeros_like(last_normal), where=last_norm > 1e-10)
    average = first_unit + last_unit
    norm = np.linalg.norm(average, axis=-1, keepdims=True)
    unit = np.divide(average, norm, out=np.zeros_like(average), where=norm > 1e-10)
    tangent = displacement - np.sum(displacement * unit, axis=-1, keepdims=True) * unit
    valid = (first_norm > 1e-10) & (last_norm > 1e-10) & (norm > 1e-10)
    return np.where(valid, tangent, 0.)


def _path_stats(times, points, sample_s, deadband_m, normals=None):
    if len(times) < 2:
        return dict(path_length_m=0., net_displacement_m=0., span_from_start_m=0.)
    sampled_t = np.arange(times[0], times[-1], sample_s)
    sampled_t = np.r_[sampled_t, times[-1]]
    sampled = np.column_stack([np.interp(sampled_t, times, points[:, j]) for j in range(3)])
    sampled_normals = (None if normals is None else
        np.column_stack([np.interp(sampled_t, times, normals[:, j]) for j in range(3)]))
    anchor_index = 0
    length = 0.
    for index, point in enumerate(sampled[1:], start=1):
        displacement = point - sampled[anchor_index]
        if sampled_normals is not None:
            displacement = _tangent_displacement(displacement, sampled_normals[anchor_index], sampled_normals[index])
        distance = float(np.linalg.norm(displacement))
        if distance >= deadband_m:
            length += distance
            anchor_index = index
    from_start = sampled - sampled[0]
    if sampled_normals is not None:
        from_start = _tangent_displacement(from_start, sampled_normals[0], sampled_normals)
    return dict(path_length_m=length,
                net_displacement_m=float(np.linalg.norm(from_start[-1])),
                span_from_start_m=float(np.max(np.linalg.norm(from_start, axis=1))))


def _gaps(times, mask):
    """Sampled-absence durations; leading/trailing absences are explicit."""
    gaps = []
    for first, last in _runs(~mask):
        recovered = bool(last + 1 < len(times))
        end = times[last + 1] if recovered else times[last]
        gaps.append(dict(start_s=float(times[first]), end_s=float(end),
                         duration_s=float(end - times[first]),
                         leading=bool(first == 0), recovered=recovered))
    return gaps


def _measured_finger_joints(trace, count, finger_count, columns):
    if 'finger_q_rad' in trace:
        q = np.asarray(trace['finger_q_rad'], dtype=float)
        if q.ndim != 3 or q.shape[:2] != (count, finger_count) or q.shape[2] < 1:
            raise ValueError('finger_q_rad must have shape [N,F,J]')
        if not np.all(np.isfinite(q)):
            raise ValueError('Measured finger joint positions must be finite')
        return [q[:, finger] for finger in range(finger_count)], 'finger_q_rad', None
    if 'q_rad' not in trace:
        if columns is not None:
            raise ValueError('finger_joint_columns requires measured q_rad')
        return None, 'UNAVAILABLE', None
    q = np.asarray(trace['q_rad'], dtype=float)
    if q.ndim != 2 or q.shape[0] != count or not np.all(np.isfinite(q)):
        raise ValueError('q_rad must be finite with shape [N,J]')
    if columns is None:
        if q.shape[1] != 23 or finger_count != 4:
            return None, 'UNAVAILABLE_MAPPING_REQUIRED', None
        columns = [list(range(7 + 4 * finger, 11 + 4 * finger)) for finger in range(4)]
        source = 'q_rad; canonical I04 7 arm + 4x4 finger joint order'
    else:
        source = 'q_rad; explicit finger_joint_columns'
    if len(columns) != finger_count:
        raise ValueError('finger_joint_columns must contain one nonempty group per finger')
    selected, seen = [], set()
    for group in columns:
        if not len(group) or any(not isinstance(column, (int, np.integer)) or column < 0 or column >= q.shape[1] for column in group):
            raise ValueError('Finger joint columns must be valid integer indices')
        if any(int(column) in seen for column in group) or len(set(group)) != len(group):
            raise ValueError('Finger joint column groups must not overlap')
        seen.update(int(column) for column in group)
        selected.append(q[:, group])
    return selected, source, [list(map(int, group)) for group in columns]


def _joint_motion(times, q, window_s=.5, speed_min_rad_s=.01, sample_s=.1, deadband_rad=.001):
    sampled_t = np.r_[np.arange(times[0], times[-1], sample_s), times[-1]]
    sampled = np.column_stack([np.interp(sampled_t, times, q[:, joint]) for joint in range(q.shape[1])])
    anchor = sampled[0].copy()
    path = np.zeros(q.shape[1])
    for row in sampled[1:]:
        distance = abs(row - anchor)
        moved = distance >= deadband_rad
        path[moved] += distance[moved]
        anchor[moved] = row[moved]
    indices = np.flatnonzero(times - times[0] >= window_s - 1e-12)
    motion = np.zeros(len(times), dtype=bool)
    if len(indices):
        past = np.column_stack([np.interp(times[indices] - window_s, times, q[:, joint]) for joint in range(q.shape[1])])
        motion[indices] = np.linalg.norm(q[indices] - past, axis=1) / window_s >= speed_min_rad_s
    return motion, dict(range_rad_by_joint=np.ptp(q, axis=0).tolist(),
                        net_change_rad_by_joint=(q[-1] - q[0]).tolist(),
                        filtered_path_rad_by_joint=path.tolist(),
                        max_displacement_from_start_l2_rad=float(np.linalg.norm(q - q[0], axis=1).max()),
                        motion_window_s=window_s, motion_speed_threshold_l2_rad_s=speed_min_rad_s,
                        path_sample_s=sample_s, path_deadband_rad=deadband_rad)


def _search_windows(trace, events, times, finger_count):
    """Declared intent only: no event type can itself establish success."""
    if 'finger_search_active' in trace:
        values = np.asarray(trace['finger_search_active'])
        if values.shape != (len(times), finger_count) or not np.all(np.isin(values, [0, 1])):
            raise ValueError('finger_search_active must contain boolean [N,F] samples')
        windows = []
        for finger in range(finger_count):
            for first, last in _runs(values[:, finger]):
                end = times[last + 1] if last + 1 < len(times) else times[last]
                windows.append(dict(finger=finger, start_s=float(times[first]), end_s=float(end),
                                    right_censored=bool(last + 1 == len(times)), source='finger_search_active'))
        return windows, 'RECORDED_TRACE_MASK'
    starts = {'FINGER_SEARCH_BEGIN', 'FINGER_SEARCH_START'}
    stops = {'FINGER_SEARCH_END', 'FINGER_SEARCH_STOP', 'FINGER_SEARCH_TIMEOUT',
             'FINGER_SEARCH_ABORT', 'FINGER_SEARCH_CANCEL', 'FINGER_SEARCH_DONE'}
    relevant = [event for event in (events or []) if event.get('kind') in starts | stops]
    if not relevant:
        return [], 'NOT_RECORDED'
    windows, ongoing = [], {}
    for event in sorted(relevant, key=lambda event: event['time_s']):
        finger = event.get('finger', event.get('finger_index'))
        if not isinstance(finger, (int, np.integer)) or not 0 <= finger < finger_count:
            raise ValueError('Finger-search events require a valid zero-based finger index')
        time = float(event['time_s'])
        if not np.isfinite(time):
            raise ValueError('Finger-search event time_s must be finite')
        if event['kind'] in starts:
            if finger in ongoing:
                begin = ongoing.pop(finger)
                windows.append(dict(finger=int(finger), start_s=max(float(times[0]), begin),
                                    end_s=min(float(times[-1]), time), right_censored=False,
                                    source='events', end_event='RESTARTED'))
            ongoing[finger] = time
        elif finger in ongoing:
            begin = ongoing.pop(finger)
            windows.append(dict(finger=int(finger), start_s=max(float(times[0]), begin),
                                end_s=min(float(times[-1]), time), right_censored=time > times[-1],
                                source='events', end_event=event['kind']))
    for finger, begin in ongoing.items():
        windows.append(dict(finger=int(finger), start_s=max(float(times[0]), begin), end_s=float(times[-1]),
                            right_censored=True, source='events', end_event='TRACE_END'))
    return [window for window in windows if window['end_s'] >= window['start_s']], 'RECORDED_EVENTS'


def _per_finger_evidence(trace, times, observed, safe, moving, events, joint_columns,
                         stable_recovery_s=.15, search_motion_min_rad=.002):
    duration, dt = float(times[-1] - times[0]), np.diff(times)
    count = observed.shape[1]
    names = ('index', 'middle', 'ring', 'thumb') if count == 4 else tuple(f'finger_{index}' for index in range(count))
    joints, joint_source, resolved_columns = _measured_finger_joints(trace, len(times), count, joint_columns)
    windows, search_status = _search_windows(trace, events, times, count)
    per_finger, attempts = [], []
    phase = np.asarray(trace['phase']) if 'phase' in trace else None
    for finger in range(count):
        contact, safe_contact = observed[:, finger], safe[:, finger]
        stable_runs = [(first, last) for first, last in _runs(safe_contact)
                       if times[last] - times[first] >= stable_recovery_s - 1e-12]
        stable_onsets = {first for first, _ in stable_runs}
        gaps = _gaps(times, contact)
        for gap in gaps:
            row = int(np.searchsorted(times, gap['end_s'] - 1e-10))
            gap['stable_safe_contact_after_return'] = bool(gap['recovered'] and row in stable_onsets)
        off = ~contact
        item = dict(finger=finger, name=names[finger],
                    contact_duration_s=_duration(contact, dt), contact_fraction=_duration(contact, dt) / duration,
                    safe_contact_duration_s=_duration(safe_contact, dt), safe_contact_fraction=_duration(safe_contact, dt) / duration,
                    moving_contact_duration_s=_duration(moving[:, finger], dt),
                    moving_contact_fraction=_duration(moving[:, finger], dt) / duration,
                    safe_moving_contact_fraction=_duration(moving[:, finger] & safe_contact, dt) / duration,
                    observed_off_contact_duration_s=_duration(off, dt),
                    first_measured_contact_s=float(times[np.flatnonzero(contact)[0]]) if contact.any() else None,
                    longest_same_finger_contact_s=max((float(times[last] - times[first]) for first, last in _runs(contact)), default=0.),
                    longest_absence_s=max((gap['duration_s'] for gap in gaps), default=0.),
                    loss_count=sum(not gap['leading'] for gap in gaps),
                    measured_return_count=sum(gap['recovered'] and not gap['leading'] for gap in gaps),
                    stable_safe_return_count=sum(gap['stable_safe_contact_after_return'] and not gap['leading'] for gap in gaps),
                    contact_gaps=gaps)
        q = None if joints is None else joints[finger]
        motion = np.zeros(len(times), dtype=bool)
        if q is not None:
            motion, stats = _joint_motion(times, q)
            off_motion = _duration(motion & off, dt)
            stats.update(observed_motion_duration_s=_duration(motion, dt),
                         observed_motion_fraction=_duration(motion, dt) / duration,
                         observed_motion_while_off_contact_s=off_motion,
                         off_contact_motion_fraction=off_motion / item['observed_off_contact_duration_s']
                            if item['observed_off_contact_duration_s'] else 0.,
                         source=joint_source, command_arrays_used=False)
            item['measured_joint_motion'] = stats
        else:
            item['measured_joint_motion'] = dict(source=joint_source, available=False, command_arrays_used=False)
        if phase is not None:
            phase_stats = {}
            for value in np.unique(phase):
                mask = phase == value
                phase_duration = _duration(mask, dt)
                phase_stats[str(value)] = dict(duration_s=phase_duration,
                    contact_duration_s=_duration(mask & contact, dt),
                    safe_contact_duration_s=_duration(mask & safe_contact, dt),
                    actual_joint_motion_duration_s=_duration(mask & motion, dt) if q is not None else None,
                    actual_joint_motion_while_off_contact_s=_duration(mask & motion & off, dt) if q is not None else None)
            item['phase_summary'] = phase_stats
        for window in (window for window in windows if window['finger'] == finger):
            first = min(int(np.searchsorted(times, window['start_s'] - 1e-10)), len(times) - 1)
            last = min(int(np.searchsorted(times, window['end_s'] + 1e-10, side='right')) - 1, len(times) - 1)
            if first > last:
                continue
            previous = max(0, int(np.searchsorted(times, window['start_s'] + 1e-10, side='right')) - 1)
            already_safe = bool(safe_contact[previous])
            returns = np.flatnonzero(contact[first:last + 1])
            first_return = int(first + returns[0]) if len(returns) else None
            stable = next(((start, end) for start, end in stable_runs
                           if not already_safe and first <= start <= last), None)
            stable_start = stable[0] if stable is not None else None
            measurement_end = stable_start if stable_start is not None else last
            displacement = None if q is None else float(np.linalg.norm(q[first:measurement_end + 1] - q[first], axis=1).max())
            actual_motion = displacement is not None and displacement >= search_motion_min_rad
            attempts.append(dict(**window, name=names[finger], already_in_safe_contact=already_safe,
                first_measured_contact_s=float(times[first_return]) if first_return is not None else None,
                stable_safe_contact_onset_s=float(times[stable_start]) if stable_start is not None else None,
                stable_safe_contact_confirmed_s=float(times[min(stable[1], np.searchsorted(times, times[stable_start] + stable_recovery_s - 1e-10))]) if stable is not None else None,
                measured_recovery_latency_s=float(times[stable_start] - window['start_s']) if stable_start is not None else None,
                measured_stable_safe_recovery=stable is not None,
                actual_joint_max_displacement_l2_rad=displacement,
                actual_joint_motion_observed=actual_motion,
                measured_recovery_with_actual_joint_motion=stable is not None and actual_motion,
                recovery_inferred_from_command=False))
        per_finger.append(item)
    search = dict(evidence_status=search_status, stable_safe_contact_required_s=stable_recovery_s,
                  actual_joint_motion_min_displacement_l2_rad=search_motion_min_rad,
                  attempt_count=len(attempts), measured_stable_recovery_count=sum(a['measured_stable_safe_recovery'] for a in attempts),
                  measured_recovery_with_actual_joint_motion_count=sum(a['measured_recovery_with_actual_joint_motion'] for a in attempts),
                  attempts=attempts, causal_attribution='Temporal association only; wrist-vs-finger contribution unresolved')
    return per_finger, search, dict(source=joint_source, finger_joint_columns=resolved_columns)


def analyze_trace(trace: Mapping, *, active_force_min_n=.2, safe_force_max_n=8.,
                  gap_tolerance_s=.1, movement_window_s=.5,
                  movement_speed_min_m_s=.001, path_sample_s=.1,
                  path_deadband_m=.0005, sliding_bout_min_duration_s=2.,
                  sliding_bout_min_span_m=.01, events=None, finger_joint_columns=None):
    """Return JSON-compatible contact, recovery, and genuine-motion descriptors.

    Required arrays: ``time_s[N]``, ``active[N,F]``, ``forces_n[N,F]``, and
    ``contact_positions_world_m[N,F,3]``. Optional arrays:
    ``tip_positions_world_m[N,F,3]``, ``normals_world[N,F,3]``, ``phase[N]``, ``stroke_id[N]`` and
    ``nominal_path_distance_m[N]``. ``phase`` and ``stroke_id`` are reporting
    labels only. Planned cumulative distance is reported separately from measured
    contact distance and never changes the motion/contact classification.
    Actual joint evidence optionally uses ``finger_q_rad[N,F,J]`` or
    ``q_rad[N,23]`` (7 arm joints then four groups of four finger joints).
    For other measured q layouts, pass ``finger_joint_columns`` explicitly.
    Search intent optionally uses ``finger_search_active[N,F]`` or ``events``
    with FINGER_SEARCH_BEGIN/END/TIMEOUT, zero-based finger and time_s. Command
    arrays and event outcome labels cannot establish physical recovery.
    """
    settings = dict(active_force_min_n=active_force_min_n, safe_force_max_n=safe_force_max_n,
                    gap_tolerance_s=gap_tolerance_s, movement_window_s=movement_window_s,
                    movement_speed_min_m_s=movement_speed_min_m_s, path_sample_s=path_sample_s,
                    path_deadband_m=path_deadband_m,
                    sliding_bout_min_duration_s=sliding_bout_min_duration_s,
                    sliding_bout_min_span_m=sliding_bout_min_span_m)
    if any(not np.isfinite(value) or value < 0 for value in settings.values()):
        raise ValueError('Metric thresholds must be finite and nonnegative')
    if movement_window_s <= 0 or path_sample_s <= 0 or safe_force_max_n < active_force_min_n:
        raise ValueError('Window/sample periods must be positive and force bounds ordered')
    times = np.asarray(trace['time_s'], dtype=float)
    active = np.asarray(trace['active'], dtype=bool)
    forces = np.asarray(trace['forces_n'], dtype=float)
    positions = np.asarray(trace['contact_positions_world_m'], dtype=float)
    if times.ndim != 1 or len(times) < 2 or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError('time_s must contain at least two finite, strictly increasing samples')
    if active.ndim != 2 or active.shape[0] != len(times) or active.shape[1] == 0:
        raise ValueError('active must have shape [N, F] with at least one finger')
    if forces.shape != active.shape or positions.shape != (*active.shape, 3):
        raise ValueError('forces/contact position shapes must match active [N,F] / [N,F,3]')
    tips = None
    if 'tip_positions_world_m' in trace:
        tips = np.asarray(trace['tip_positions_world_m'], dtype=float)
        if tips.shape != positions.shape:
            raise ValueError('tip_positions_world_m must match contact_positions_world_m')
    normals = None
    if 'normals_world' in trace:
        normals = np.asarray(trace['normals_world'], dtype=float)
        if normals.shape != positions.shape:
            raise ValueError('normals_world must match contact_positions_world_m')
    observed = active & np.isfinite(forces) & (forces >= active_force_min_n)
    valid_positions = np.all(np.isfinite(positions), axis=2)
    safe = observed & (forces <= safe_force_max_n)
    dt = np.diff(times)
    duration = float(times[-1] - times[0])
    moving = np.zeros_like(observed)
    contact_migration = np.zeros_like(observed)
    strict_runs, gaps, debounced_runs = [], [], []
    for finger in range(active.shape[1]):
        mask = observed[:, finger]
        finger_gaps = _gaps(times, mask)
        gaps.extend(dict(finger=finger, **gap) for gap in finger_gaps)
        debounced = mask.copy()
        for first, last in _runs(~mask):
            if first > 0 and last + 1 < len(times) and times[last + 1] - times[first] <= gap_tolerance_s + 1e-12:
                debounced[first:last + 1] = True
        for first, last in _runs(debounced):
            bridged = [gap for gap in finger_gaps if not gap['leading'] and gap['recovered']
                       and times[first] <= gap['start_s'] and gap['end_s'] <= times[last]]
            debounced_runs.append(dict(finger=finger, start_s=float(times[first]), end_s=float(times[last]),
                                       duration_s=float(times[last] - times[first]),
                                       bridged_gap_count=len(bridged),
                                       bridged_gap_duration_s=float(sum(gap['duration_s'] for gap in bridged))))
        for first, last in _runs(mask):
            strict_runs.append(dict(finger=finger, start_s=float(times[first]), end_s=float(times[last]),
                                    duration_s=float(times[last] - times[first]), sample_count=int(last - first + 1)))
        # Missing coordinates break motion evidence even if the force sensor is active.
        coordinate_mask = mask & valid_positions[:, finger]
        if tips is not None:
            coordinate_mask &= np.all(np.isfinite(tips[:, finger]), axis=1)
        if normals is not None:
            coordinate_mask &= np.all(np.isfinite(normals[:, finger]), axis=1) & (np.linalg.norm(normals[:, finger], axis=1) > 1e-10)
        for first, last in _runs(coordinate_mask):
            indices = np.arange(first, last + 1)
            eligible = indices[times[indices] - times[first] >= movement_window_s - 1e-12]
            if not len(eligible):
                continue
            past_t = times[eligible] - movement_window_s
            past = np.column_stack([np.interp(past_t, times[indices], positions[indices, finger, j]) for j in range(3)])
            displacement = positions[eligible, finger] - past
            if normals is not None:
                past_normals = np.column_stack([np.interp(past_t, times[indices], normals[indices, finger, j]) for j in range(3)])
                displacement = _tangent_displacement(displacement, past_normals, normals[eligible, finger])
            migration = np.linalg.norm(displacement, axis=1) / movement_window_s >= movement_speed_min_m_s
            contact_migration[eligible, finger] = migration
            if tips is not None:
                past_tip = np.column_stack([np.interp(past_t, times[indices], tips[indices, finger, j]) for j in range(3)])
                tip_displacement = tips[eligible, finger] - past_tip
                if normals is not None:
                    tip_displacement = _tangent_displacement(tip_displacement, past_normals, normals[eligible, finger])
                migration &= np.linalg.norm(tip_displacement, axis=1) / movement_window_s >= movement_speed_min_m_s
            moving[eligible, finger] = migration
    bouts, moving_runs = [], []
    for finger in range(active.shape[1]):
        for first, last in _runs(moving[:, finger]):
            stats = _path_stats(times[first:last + 1], positions[first:last + 1, finger], path_sample_s, path_deadband_m,
                                None if normals is None else normals[first:last + 1, finger])
            item = dict(finger=finger, start_s=float(times[first]), end_s=float(times[last]),
                        duration_s=float(times[last] - times[first]),
                        safe_duration_s=_duration(safe[first:last + 1, finger], dt[first:last]), **stats)
            moving_runs.append(item)
            if item['duration_s'] + 1e-12 >= sliding_bout_min_duration_s and stats['span_from_start_m'] + 1e-12 >= sliding_bout_min_span_m:
                bouts.append(item)
    any_contact, any_safe = observed.any(axis=1), safe.any(axis=1)
    # A moving interval requires the SAME finger at both ends, unlike an any-finger
    # mask which could mistakenly join two alternating fingers.
    moving_intervals = (moving[:-1] & moving[1:]).any(axis=1)
    safe_moving = moving & safe
    safe_moving_intervals = (safe_moving[:-1] & safe_moving[1:]).any(axis=1)
    moving_s = float(np.sum(dt[moving_intervals]))
    safe_moving_s = float(np.sum(dt[safe_moving_intervals]))
    result = dict(schema='i04.sliding_metrics.v1.2', parameters=settings,
                  sample_count=len(times), finger_count=active.shape[1], duration_s=duration,
                  fingertip_motion_corroborated=tips is not None,
                  tangent_projection_applied=normals is not None,
                  motion_space='averaged_surface_tangent_plane' if normals is not None else 'world_displacement',
                  contact_duration_s=_duration(any_contact, dt),
                  contact_continuity_fraction=_duration(any_contact, dt) / duration,
                  safe_contact_duration_s=_duration(any_safe, dt),
                  safe_contact_continuity_fraction=_duration(any_safe, dt) / duration,
                  moving_contact_duration_s=moving_s, moving_contact_fraction=moving_s / duration,
                  safe_moving_contact_duration_s=safe_moving_s,
                  safe_moving_contact_fraction=safe_moving_s / duration,
                  contact_migration_fraction=float(np.sum(dt[(contact_migration[:-1] & contact_migration[1:]).any(axis=1)])) / duration,
                  strict_same_finger_runs=strict_runs,
                  longest_strict_same_finger_contact_s=max((run['duration_s'] for run in strict_runs), default=0.),
                  debounced_same_finger_runs=debounced_runs,
                  same_finger_gaps=gaps, any_contact_gaps=_gaps(times, any_contact),
                  moving_runs=moving_runs, sliding_bouts=bouts, sliding_bout_count=len(bouts),
                  longest_sliding_bout_s=max((bout['duration_s'] for bout in bouts), default=0.),
                  longest_sliding_bout_span_m=max((bout['span_from_start_m'] for bout in bouts), default=0.),
                  measured_moving_path_sum_over_fingers_m=float(sum(run['path_length_m'] for run in moving_runs)))
    for label in ('phase', 'stroke_id'):
        if label not in trace:
            continue
        values = np.asarray(trace[label])
        if values.shape != times.shape:
            raise ValueError(f'{label} must have shape [N]')
        summaries = {}
        for value in np.unique(values):
            intervals = (values[:-1] == value) & (values[1:] == value)
            total = float(np.sum(dt[intervals]))
            summaries[str(value)] = dict(duration_s=total,
                moving_contact_duration_s=float(np.sum(dt[intervals & moving_intervals])),
                safe_moving_contact_duration_s=float(np.sum(dt[intervals & safe_moving_intervals])),
                moving_contact_fraction=float(np.sum(dt[intervals & moving_intervals])) / total if total else 0.)
        result[label + '_summary'] = summaries
    if 'nominal_path_distance_m' in trace:
        distance = np.asarray(trace['nominal_path_distance_m'], dtype=float)
        if distance.shape != times.shape or not np.all(np.isfinite(distance)):
            raise ValueError('nominal_path_distance_m must be finite with shape [N]')
        result['planned_path_distance_increment_m'] = float(np.sum(np.maximum(0., np.diff(distance))))
    per_finger, searches, joint_source = _per_finger_evidence(trace, times, observed, safe, moving,
                                                            events, finger_joint_columns)
    result['per_finger'] = per_finger
    result['finger_search_recovery'] = searches
    result['measured_finger_joint_mapping'] = joint_source
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path, help='Saved native trace.npz')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--events', type=Path, help='Optional explicit finger-search event ledger')
    args = parser.parse_args()
    with np.load(args.trace, allow_pickle=False) as archive:
        trace = {key: archive[key] for key in archive.files}
    events = json.loads(args.events.read_text(encoding='utf-8')) if args.events else None
    result = analyze_trace(trace, events=events)
    encoded = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + '\n', encoding='utf-8')
    else:
        print(encoded)


if __name__ == '__main__':
    main()
