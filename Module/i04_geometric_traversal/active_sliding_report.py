"""B-only active-finger sliding evidence, separate from the legacy A/B report.

Compares the same acquired state and same saved time prefix with validation04.
Search proposals, actual joint motion and stable measured recontact are shown
separately. Neither a proposal, geometric target, video nor standalone probe
can complete the main run's surface ledger.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import html
from io import BytesIO
import json
import os
from pathlib import Path
import zipfile

import numpy as np

from .sliding_metrics import analyze_trace, _gaps
from .sliding_report import _initial_pair, _prefix_pair


DEFAULT_BASELINE = Path(__file__).resolve().parents[1] / 'generated/mustard_sliding_validation04_20261016/B-MCC_seed7'
KEYS = ('time_s', 'active', 'forces_n', 'contact_positions_world_m', 'tip_positions_world_m',
        'normals_world', 'q_rad', 'phase', 'stroke_id', 'nominal_path_distance_m', 'coverage',
        'covered_goals',
        'finger_search_active', 'finger_search_enabled', 'finger_search_bias_rad',
        'finger_search_distance_m', 'finger_search_delta_rad', 'finger_search_state')


def _json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def _read_case(path):
    with np.load(path / 'trace.npz', allow_pickle=False) as archive:
        trace = {key: archive[key] for key in KEYS if key in archive.files}
        masks = {key: archive[key] for key in ('covered_vertices', 'required_goal_visited') if key in archive.files}
    return trace, masks


def compare_checkpoint_prefix(directory, baseline, end_s=None):
    """Read an immutable set of completed chunks; return evidence, write nothing.

    This helper never reads a future coverage checkpoint or treats a running
    experiment as completed. Callers may save the returned explicitly partial
    evidence separately from the final report. Events beyond the saved prefix
    are excluded. Hashes describe the exact bytes used for each chunk.
    """
    case, baseline = Path(directory).resolve(), Path(baseline).resolve()
    if not (case / 'manifest.json').exists():
        case /= 'B-MCC_seed7'
    if not (baseline / 'manifest.json').exists():
        baseline /= 'B-MCC_seed7'
    files = sorted((case / 'trace_chunks').glob('*.npz'))
    chunks, sources = [], []
    for path in files:
        raw = path.read_bytes()
        try:
            with np.load(BytesIO(raw), allow_pickle=False) as archive:
                chunk = {key: archive[key] for key in KEYS if key in archive.files}
        except (EOFError, zipfile.BadZipFile):
            if path == files[-1]:
                break
            raise
        if not len(chunk.get('time_s', [])):
            continue
        chunks.append(chunk)
        sources.append(dict(file=str(path), sha256=sha256(raw).hexdigest(),
                            start_s=float(chunk['time_s'][0]), end_s=float(chunk['time_s'][-1])))
        if end_s is not None and chunk['time_s'][-1] >= end_s:
            break
    if not chunks:
        raise ValueError('No completed checkpoint chunks')
    keys = set(chunks[0])
    if any(set(chunk) != keys for chunk in chunks):
        raise ValueError('Checkpoint schemas differ')
    trace = {key: np.concatenate([chunk[key] for chunk in chunks]) for key in keys}
    times = trace['time_s']
    if len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError('A positive, strictly increasing completed prefix is required')
    dt = float(np.median(np.diff(times)))
    if not np.allclose(np.diff(times), dt, atol=1e-9, rtol=0):
        raise ValueError('Completed checkpoints have a timestamp gap')
    initial = _initial_pair(case, baseline)
    if not initial['exactly_equal']:
        raise ValueError('Checkpoint and baseline initial qpos/qvel/ctrl differ')
    old, _ = _read_case(baseline)
    requested_end = min(float(times[-1]), float(end_s) if end_s is not None else float(times[-1]))
    horizon, new_prefix, old_prefix = _prefix_pair(trace, old, requested_end, float(old['time_s'][-1]))
    events = _json(case / 'events.json')
    old_events = _json(baseline / 'events.json')
    events = [event for event in events if event['time_s'] <= horizon]
    old_events = [event for event in old_events if event['time_s'] <= horizon]
    return dict(schema='i04.active_sliding_checkpoint_comparison.v1',
                source_scope='DEVELOPMENT PREFIX', full_surface_claim=False,
                current_case=str(case), baseline_case=str(baseline), sources=sources,
                available_completed_end_s=float(times[-1]), horizon_s=horizon,
                initial_state=initial, new_metrics=analyze_trace(new_prefix, events=events),
                old_metrics=analyze_trace(old_prefix, events=old_events),
                new_loss=_loss_metrics(new_prefix), old_loss=_loss_metrics(old_prefix),
                new_coverage=float(new_prefix['coverage'][-1]), old_coverage=float(old_prefix['coverage'][-1]),
                new_progress=_progress_metrics(new_prefix, events), old_progress=_progress_metrics(old_prefix, old_events),
                new_transitions=_transition_metrics(new_prefix, events),
                old_transitions=_transition_metrics(old_prefix, old_events),
                new_sampled_peak_force_n=float(np.max(new_prefix['forces_n'])),
                force_peak_scope='Saved 100 Hz samples only; not the complete native-step peak',
                causal_scope='Combined strategy version comparison; not a single-component ablation')


def _loss_metrics(trace):
    times, force = trace['time_s'], trace['forces_n']
    contact = (trace['active'] & np.isfinite(force) & (force >= .2)).any(axis=1)
    safe = (trace['active'] & np.isfinite(force) & (force >= .2) & (force <= 8.)).any(axis=1)
    dt = np.diff(times)
    gap = _gaps(times, contact)
    safe_gap = _gaps(times, safe)
    return dict(longest_any_contact_gap_s=max((item['duration_s'] for item in gap), default=0.),
                longest_any_safe_contact_gap_s=max((item['duration_s'] for item in safe_gap), default=0.),
                observed_no_contact_duration_s=float(dt[~contact[:-1] & ~contact[1:]].sum()),
                contact_gaps=gap, safe_contact_gaps=safe_gap)


def surface_normal_regions(graph_path, covered_vertices):
    """Partition the unchanged full ledger by dominant saved outward normal.

    Areas use one third of each incident triangle's area, matching runtime
    coverage weights. A covered region still means the existing measured
    contact geodesic-neighborhood ledger, not direct contact at every vertex.
    This diagnostic never removes vertices or changes the global denominator.
    """
    graph_path = Path(graph_path)
    raw = graph_path.read_bytes()
    with np.load(BytesIO(raw), allow_pickle=False) as graph:
        vertices, faces, normals = graph['vertices'], graph['faces'], graph['normals']
        required = graph['required_vertices']
    covered = np.asarray(covered_vertices, dtype=bool)
    if covered.shape != (len(vertices),) or not np.isfinite(normals).all() or np.any(np.linalg.norm(normals, axis=1) < 1e-12):
        raise ValueError('Full coverage mask and valid saved outward normals required')
    tri = vertices[faces]
    face_area = .5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    vertex_area = np.zeros(len(vertices))
    np.add.at(vertex_area, faces.ravel(), np.repeat(face_area / 3., 3))
    dominant = np.argmax(np.abs(normals), axis=1)
    sign = normals[np.arange(len(normals)), dominant] >= 0
    whole_area = float(vertex_area.sum())
    regions = []
    for axis, axis_name in enumerate('XYZ'):
        for positive in (True, False):
            mask = (dominant == axis) & (sign == positive)
            area = float(vertex_area[mask].sum())
            covered_area = float(vertex_area[mask & covered].sum())
            goals = mask[required]
            regions.append(dict(region=('+' if positive else '-') + axis_name,
                                area_m2=area, area_fraction_of_full_surface=area / whole_area,
                                covered_area_m2=covered_area,
                                covered_fraction_within_region=covered_area / area if area else 0.,
                                covered_fraction_of_full_surface=covered_area / whole_area,
                                vertex_count=int(mask.sum()), covered_vertex_count=int((mask & covered).sum()),
                                required_goal_count=int(goals.sum()),
                                covered_required_goals=int((covered[required] & goals).sum())))
    assert sum(item['vertex_count'] for item in regions) == len(vertices)
    return dict(definition='Largest absolute component of saved outward task-surface normal; ties X, then Y, then Z',
                coordinate_frame='Saved surface_graph task frame; no axes reorientation',
                vertex_area_rule='Incident triangle area divided equally among its three vertices',
                surface_graph_sha256=sha256(raw).hexdigest(), full_area_m2=whole_area,
                full_vertex_count=len(vertices), full_required_goal_count=len(required),
                measured_coverage_fraction=float(vertex_area @ covered / whole_area),
                coverage_meaning='Existing measured-contact 25 mm geodesic neighborhood; not direct touch at every vertex',
                denominator_unchanged=True, regions=regions)


def _finger_search(metrics, finger):
    windows = [item for item in metrics['finger_search_recovery']['attempts'] if item['finger'] == finger]
    return dict(proposal_windows=len(windows),
                measured_stable_recoveries=sum(item['measured_stable_safe_recovery'] for item in windows),
                recoveries_with_actual_joint_motion=sum(item['measured_recovery_with_actual_joint_motion'] for item in windows),
                right_censored_windows=sum(item['right_censored'] for item in windows))


def _progress_metrics(trace, events):
    times, coverage = trace['time_s'], trace['coverage']
    increases = np.flatnonzero(np.diff(coverage) > 1e-12)
    last = int(increases[-1] + 1) if len(increases) else 0
    start = max(float(times[0]), float(times[-1]) - 10.)
    row = int(np.searchsorted(times, start))
    return dict(last_measured_coverage_increase_s=float(times[last]),
                terminal_no_new_coverage_s=float(times[-1] - times[last]),
                terminal_strokes_started=int(sum(event.get('kind') == 'SURFACE_STROKE'
                    and times[last] < event['time_s'] <= times[-1] for event in events)),
                final_10s_start_s=float(times[row]),
                final_10s_start_coverage=float(coverage[row]),
                final_10s_end_coverage=float(coverage[-1]),
                final_10s_added_area_fraction=float(coverage[-1] - coverage[row]))


def _transition_metrics(trace, events):
    """Event command spans are timing context, never sensor success evidence."""
    times, phases = trace['time_s'], trace['phase']
    dt = np.diff(times)
    durations = {str(phase): float(dt[phases[:-1] == phase].sum()) for phase in np.unique(phases)}
    observed = [event for event in events if times[0] <= event['time_s'] <= times[-1]]
    cycles = []
    for i, event in enumerate(observed):
        if event.get('kind') != 'LIFT_COMMIT':
            continue
        following = next((item for item in observed[i + 1:]
                          if item.get('kind') in ('LIFT_RECONTACT', 'LIFT_COMMIT')), None)
        end = following['time_s'] if following and following['kind'] == 'LIFT_RECONTACT' else float(times[-1])
        cycles.append(dict(start_s=event['time_s'], end_s=end,
                           duration_s=float(end - event['time_s']),
                           ended_with_recontact_event=bool(following and following['kind'] == 'LIFT_RECONTACT'),
                           planned_duration_s=event.get('duration_s'), pose_policy=event.get('pose_policy')))
    return dict(sampled_phase_duration_s=durations, lift_cycles=cycles,
                near_bridge_commands=[{key: event[key] for key in ('time_s', 'duration_s', 'pose_policy') if key in event}
                                      for event in observed if event.get('kind') == 'NEAR_BRIDGE_COMMIT'],
                event_success_not_used_for_contact_metrics=True)


def _plots(output, current, metrics, comparison, old_prefix=None, new_prefix=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), layout='constrained')
    paired = comparison['status'] == 'PAIRED'
    ax = axes[0, 0]
    if paired:
        old, new = comparison['old_metrics'], comparison['new_metrics']
        x = np.arange(2)
        keys = ('safe_contact_continuity_fraction', 'safe_moving_contact_fraction')
        ax.bar(x - .18, [100 * old[key] for key in keys], .36, label='Previous strategy', color='#8995a7')
        ax.bar(x + .18, [100 * new[key] for key in keys], .36, label='Current strategy', color='#13867a')
        ax.set_xticks(x, ['Safe contact', 'Safe tangent motion'])
        ax.legend(fontsize=8)
    ax.set(title='Same initial state, same observed prefix', ylabel='Observed time (%)', ylim=(0, 100))
    ax.grid(axis='y', alpha=.2)
    ax = axes[0, 1]
    names = ['Index', 'Middle', 'Ring', 'Thumb']
    if paired:
        x = np.arange(4)
        ax.bar(x - .18, [100 * item['safe_contact_fraction'] for item in old['per_finger']], .36, color='#8995a7', label='Previous')
        ax.bar(x + .18, [100 * item['safe_contact_fraction'] for item in new['per_finger']], .36, color='#13867a', label='Current')
        ax.set_xticks(x, names)
        ax.legend(fontsize=8)
    ax.set(title='Each finger: actual safe contact, same prefix', ylabel='Observed time (%)', ylim=(0, 100))
    ax.grid(axis='y', alpha=.2)
    ax = axes[1, 0]
    times = current['time_s']
    for finger in range(4):
        contact = current['active'][:, finger] & (current['forces_n'][:, finger] >= .2) & (current['forces_n'][:, finger] <= 8.)
        ax.fill_between(times, finger, finger + .65, where=contact, step='post', color=plt.get_cmap('tab10')(finger))
    ax.set_yticks(np.arange(4) + .325, names)
    ax.set(title='Current complete recorded run: measured safe contact', xlabel='Simulation time (s)', xlim=(times[0], times[-1]))
    ax.grid(axis='x', alpha=.2)
    ax = axes[1, 1]
    for finger, item in enumerate(metrics['per_finger']):
        # One norm per physical finger from actual q, never command/bias arrays.
        if 'q_rad' in current and current['q_rad'].shape[1] == 23:
            q = current['q_rad'][:, 7 + 4 * finger:11 + 4 * finger]
            ax.plot(times, np.rad2deg(np.linalg.norm(q - q[0], axis=1)), lw=.8, label=names[finger])
    ax.set(title='Actual finger joint displacement from initial posture', xlabel='Simulation time (s)', ylabel='Four-joint L2 displacement (deg)', xlim=(times[0], times[-1]))
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    ax = axes[2, 0]
    ax.plot(times, current['forces_n'].max(axis=1), color='#13867a', lw=.8, label='Actual maximum pad force')
    ax.axhline(8., ls='--', color='#ad7020', lw=.8, label='8 N relief threshold')
    ax.set(title='Measured force; brief spikes remain visible', xlabel='Simulation time (s)', ylabel='Force (N)', xlim=(times[0], times[-1]))
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    ax = axes[2, 1]
    ax.plot(times, current['coverage'] * 100, color='#13867a', label='Current complete recorded run')
    if paired:
        ax.plot(old_prefix['time_s'], old_prefix['coverage'] * 100, color='#8995a7', label='Previous, common prefix only')
    ax.set(title='Actual contact geodesic coverage, full denominator', xlabel='Simulation time (s)', ylabel='Surface area (%)', xlim=(times[0], times[-1]), ylim=(0, 100))
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    fig.suptitle('Mustard | B-only strategy comparison | search intent is not measured recovery', fontsize=14)
    path = output / 'active_sliding_evidence.png'
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path.name


def _video_html(case, output):
    sections, metadata = [], []
    for filename in ('sliding_closeup.mp4', 'full_traversal.mp4', 'active_fingers_closeup.mp4'):
        video, meta = case / filename, case / (Path(filename).stem + '_render.json')
        if not video.exists() or not meta.exists():
            continue  # A render without its completed metadata may still be in progress.
        values = _json(meta)
        required = ('start_s', 'end_s', 'playback_speed', 'uncut')
        if any(key not in values for key in required):
            continue
        reference = Path(os.path.relpath(video, output)).as_posix()
        title = '连续贴面与手指实际运动近景' if 'closeup' in filename else '完整保存状态回放'
        caption = (f'源仿真时间 {values["start_s"]:.2f}–{values["end_s"]:.2f} s，'
                   f'{values["playback_speed"]:g}× 回放；' + ('区间内无剪接。' if values['uncut'] else '存在剪接。'))
        if values.get('source_scope'):
            caption += f'来源范围：{values["source_scope"]}；只读保存关节状态的 FK 回放。'
        sections.append(f'<section><h2>{title}</h2><video controls preload="metadata" src="{html.escape(reference)}"></video>'
                        f'<p>{html.escape(caption)}接触点来自原始物理传感器；录像不会补充覆盖或替代恢复判定。</p></section>')
        metadata.append(dict(file=reference, metadata=values))
    return ''.join(sections), metadata


def _read_probe(path):
    path = Path(path).resolve()
    paired_audit = path / 'paired_probe_audit.json'
    if path.is_dir() and paired_audit.exists():
        audit = _json(paired_audit)
        cases = {}
        for label in ('ON', 'OFF'):
            case = path / label
            summary = _json(case / 'summary.json')
            independent = _json(case / 'probe_audit.json')
            cases[label] = dict(summary=summary, audit=independent)
        verified = (audit.get('status') == 'PASSED' and audit.get('same_source_hashes')
                    and audit.get('identical_fresh_acquisition')
                    and audit.get('sole_changed_parameter') == 'recovery.active_search_enabled'
                    and all(value['audit'].get('status') == 'PASSED'
                            and value['audit'].get('trace_sha256') == value['summary'].get('trace_sha256')
                            and value['audit'].get('wrist_command_fixed')
                            and value['audit'].get('only_selected_finger_search_applied')
                            and not value['audit'].get('coverage_counted', True) for value in cases.values()))
        videos = []
        for metadata_path in sorted((path / 'ON').glob('*_render.json')):
            video = metadata_path.with_name(metadata_path.name.removesuffix('_render.json') + '.mp4')
            metadata = _json(metadata_path)
            if video.exists() and metadata.get('trace_sha256') == cases['ON']['summary'].get('trace_sha256'):
                videos.append(dict(path=str(video), metadata=metadata))
        return dict(path=str(paired_audit), kind='PAIRED_SELECTED_FINGER_RECOVERY',
                    paired_independent_audit_verified=bool(verified), audit=audit, cases=cases,
                    videos=videos, included_in_main_metrics=False)
    summary_path = path / 'summary.json' if path.is_dir() else path
    return dict(path=str(summary_path), kind='STANDALONE_DIAGNOSTIC', evidence=_json(summary_path),
                included_in_main_metrics=False)


def _probe_section(probe, output):
    if probe is None:
        return ''
    source = html.escape(Path(os.path.relpath(probe['path'], output)).as_posix())
    if probe['kind'] != 'PAIRED_SELECTED_FINGER_RECOVERY':
        return f'<section><h2>独立诊断 probe</h2><p>此项用于定位局部问题，与主运行分开保留，不增加主运行覆盖、接触时长或成功次数。</p><p><a href="{source}">查看独立 probe 原始结果</a></p></section>'
    on, off = probe['cases']['ON'], probe['cases']['OFF']
    names = ('食指', '中指', '无名指', '拇指')
    finger = names[int(on['summary']['selected_finger'])]
    withdrawal = 1000 * on['audit']['independently_measured_withdrawal_m']
    rows = []
    for label, case in (('ON', on), ('OFF', off)):
        summary = case['summary']
        first = summary.get('first_recontact_delay_s')
        confirmation = summary.get('search_to_stable_confirmation_s')
        window = summary['recovery_observation_window_s']
        first_text = f'{first:.3f} s' if first is not None else f'{window:g} s 内未重接触'
        stable_text = f'{confirmation:.3f} s' if confirmation is not None else f'{window:g} s 内未确认'
        rows.append(f'<tr><td>{label}</td><td>{first_text}</td><td>{stable_text}</td>'
                    f'<td>{summary["final_selected_force_n"]:.3f} N</td><td>{case["audit"]["status"]}</td></tr>')
    videos = []
    for video in probe['videos']:
        metadata = video['metadata']
        if not all(key in metadata for key in ('simulation_start_s', 'simulation_end_s', 'speed')):
            continue
        reference = html.escape(Path(os.path.relpath(video['path'], output)).as_posix())
        # This dedicated renderer states uncut playback in its source metadata;
        # ordinary run videos instead carry an explicit uncut boolean.
        uncut = bool(metadata.get('uncut')) or str(metadata.get('source', '')).lower().startswith('uncut fk playback')
        caption = (f'ON：源仿真时间 {metadata["simulation_start_s"]:.3f}–{metadata["simulation_end_s"]:.3f} s，'
                   f'{metadata["speed"]:g}×，' + ('全程无剪接。' if uncut else '连续性以来源元数据为准。'))
        videos.append(f'<video controls preload="metadata" src="{reference}"></video><p>{html.escape(caption)}显示角度来自实际关节状态。</p>')
    status = '配对及两侧独立审计通过' if probe['paired_independent_audit_verified'] else '配对审计仍待完整核实'
    return (f'<section><h2>独立恢复 ON/OFF 配对实验 · 仅{finger}</h2><p>{status}。相同新采集初态、相同源码，'
            f'恢复前 {probe["audit"].get("identical_pre_recovery_samples",0)} 个样本一致；唯一切换参数为主动寻触开关。'
            f'受控撤离后的实际位移为 {withdrawal:.3f} mm，腕部参考保持固定，仅允许选定{finger}执行搜索。</p>'
            '<div class="scroll"><table><thead><tr><th>主动寻触</th><th>开始恢复至首次实际接触</th><th>至150 ms安全稳定确认</th><th>末端实测力</th><th>独立审计</th></tr></thead><tbody>'
            + ''.join(rows) + '</tbody></table></div><p>OFF 的结果仅表示该固定5秒观测窗内未恢复。'
            '这是一个受控单指恢复案例，与主运行的连续滑动、四指表现和整瓶覆盖分开；不并入主运行成功次数或覆盖。</p>'
            + ''.join(videos) + f'<p><a href="{source}">配对原始审计及完整原生重放误差</a></p></section>')


def _probe_sections(probes, output):
    if len(probes) < 2:
        return _probe_section(probes[0] if probes else None, output)
    paired = [probe for probe in probes if probe['kind'] == 'PAIRED_SELECTED_FINGER_RECOVERY']
    standalone = [probe for probe in probes if probe['kind'] != 'PAIRED_SELECTED_FINGER_RECOVERY']
    if not paired:
        return ''.join(_probe_section(probe, output) for probe in standalone)
    names = ('食指', '中指', '无名指', '拇指')
    rows, references, videos = [], [], []
    for probe in paired:
        on = probe['cases']['ON']
        name = names[int(on['summary']['selected_finger'])]
        withdrawal = 1000 * on['audit']['independently_measured_withdrawal_m']
        for label in ('ON', 'OFF'):
            case = probe['cases'][label]
            summary = case['summary']
            first, stable = summary.get('first_recontact_delay_s'), summary.get('search_to_stable_confirmation_s')
            window = summary['recovery_observation_window_s']
            first_text = f'{first:.3f} s' if first is not None else f'{window:g} s 内未重接触'
            stable_text = f'{stable:.3f} s' if stable is not None else f'{window:g} s 内未确认'
            rows.append(f'<tr><td>{name}</td><td>{label}</td><td>{withdrawal:.3f} mm</td><td>{first_text}</td>'
                        f'<td>{stable_text}</td><td>{summary["final_selected_force_n"]:.3f} N</td>'
                        f'<td>{case["audit"]["status"]}</td></tr>')
        reference = html.escape(Path(os.path.relpath(probe['path'], output)).as_posix())
        state = '配对核验通过' if probe['paired_independent_audit_verified'] else '配对仍待完整核验'
        references.append(f'<a href="{reference}">{name}：{state}，恢复前 '
                          f'{probe["audit"].get("identical_pre_recovery_samples", 0)} 个相同样本</a>')
        for video in probe['videos']:
            metadata = video['metadata']
            if not all(key in metadata for key in ('simulation_start_s', 'simulation_end_s', 'speed')):
                continue
            reference = html.escape(Path(os.path.relpath(video['path'], output)).as_posix())
            uncut = bool(metadata.get('uncut')) or str(metadata.get('source', '')).lower().startswith('uncut fk playback')
            caption = (f'{name} ON：源仿真时间 {metadata["simulation_start_s"]:.3f}–{metadata["simulation_end_s"]:.3f} s，'
                       f'{metadata["speed"]:g}×，' + ('全程无剪接。' if uncut else '连续性以来源元数据为准。'))
            videos.append(f'<video controls preload="metadata" src="{reference}"></video><p>{html.escape(caption)}'
                          '只读实际关节状态回放，角度来自测量。</p>')
    return ('<section><h2>独立单指恢复 ON/OFF 配对实验</h2><p>每一对均重新采集相同初态、使用相同源码；'
            '只撤离本行选定手指，腕部参考保持不变，恢复阶段仅允许该手指搜索，唯一切换参数为主动寻触开关。'
            '各指分别运行，不是同时撤离三指的实验。</p><div class="scroll"><table><thead><tr>'
            '<th>选定手指</th><th>主动寻触</th><th>实际撤离</th><th>恢复开始至首次实测接触</th>'
            '<th>至150 ms安全稳定确认</th><th>末端实测力</th><th>独立审计</th></tr></thead><tbody>'
            + ''.join(rows) + '</tbody></table></div><p>' + ' · '.join(references) + '</p>'
            '<p>OFF 只表示固定5秒观测窗内未恢复。以上受控局部结果与主运行四指动作、连续滑动和整瓶覆盖分开；'
            '不并入主运行成功次数或覆盖，也不据此宣称其他初态、拇指或整瓶均已解决。</p>'
            + ''.join(videos) + '</section>' + ''.join(_probe_section(probe, output) for probe in standalone))


def report(directory, baseline=DEFAULT_BASELINE, probe=None):
    output = Path(directory).resolve()
    case = output if (output / 'summary.json').exists() else output / 'B-MCC_seed7'
    baseline = Path(baseline).resolve()
    if not (baseline / 'summary.json').exists() and (baseline / 'B-MCC_seed7').is_dir():
        baseline /= 'B-MCC_seed7'
    if output == baseline or baseline in output.parents or output in baseline.parents:
        raise ValueError('Current output and baseline must be separate, non-nested directories')
    summary, manifest = _json(case / 'summary.json'), _json(case / 'manifest.json')
    if summary['cell'] != 'B-MCC':
        raise ValueError('Geometry-guided active-finger report is explicitly B-only')
    audit = _json(case / 'audit.json') if (case / 'audit.json').exists() else dict(status='MISSING')
    events = _json(case / 'events.json')
    trace, masks = _read_case(case)
    if 'finger_search_active' not in trace:
        raise ValueError('Active-finger report requires physical trace plus search-intent telemetry')
    metrics = analyze_trace(trace, events=events)
    loss = _loss_metrics(trace)
    progress = _progress_metrics(trace, events)
    transitions = _transition_metrics(trace, events)
    normal_regions = surface_normal_regions(case / 'surface_graph.npz', masks['covered_vertices'])
    if abs(normal_regions['measured_coverage_fraction'] - summary['final_covered_area_fraction']) > 1e-10:
        raise ValueError('Normal-region area weights do not reproduce the full recorded coverage')
    expected_masks = (len(masks.get('covered_vertices', [])) == 8192 and len(masks.get('required_goal_visited', [])) == 388)
    all_measured = bool(expected_masks and masks['covered_vertices'].all() and masks['required_goal_visited'].all())
    full_confirmed = bool(summary.get('full_surface_complete') and all_measured and audit.get('status') == 'PASSED'
                          and audit.get('completion_hold_verified') and summary.get('stop_reason') == 'FULL_MEASURED_SURFACE')
    initial = _initial_pair(case, baseline)
    comparison = dict(status='EXCLUDED_INITIAL_STATE_MISMATCH', initial_state=initial)
    old_prefix = new_prefix = None
    if initial['exactly_equal']:
        old, _ = _read_case(baseline)
        old_summary = _json(baseline / 'summary.json')
        horizon, new_prefix, old_prefix = _prefix_pair(trace, old, summary['executed_s'], old_summary['executed_s'])
        comparison = dict(status='PAIRED', initial_state=initial, start_s=0., horizon_s=horizon,
                          old_metrics=analyze_trace(old_prefix), new_metrics=analyze_trace(new_prefix),
                          old_coverage=float(old_prefix['coverage'][-1]), new_coverage=float(new_prefix['coverage'][-1]),
                          old_loss=_loss_metrics(old_prefix), new_loss=_loss_metrics(new_prefix))
    figure = _plots(output, trace, metrics, comparison, old_prefix, new_prefix)
    video_html, videos = _video_html(case, output)
    probe_paths = [] if probe is None else ([probe] if isinstance(probe, (str, Path)) else list(probe))
    probe_records = [_read_probe(path) for path in probe_paths]
    evidence = dict(schema='i04.active_sliding_report.v1.1', cell_scope='B_ONLY_GEOMETRY_GUIDED_STRATEGY',
                    current_case=str(case), baseline_case=str(baseline), protocol=manifest.get('protocol'),
                    current_summary={key: value for key, value in summary.items() if key != 'sliding_metrics'},
                    source_hashes=manifest.get('source_hashes', {}), audit_status=audit.get('status'),
                    full_surface_confirmed=full_confirmed, all_8192_vertices_and_388_goals_measured=all_measured,
                    metrics_source_sha256=sha256(Path(__file__).with_name('sliding_metrics.py').read_bytes()).hexdigest(),
                    metrics=metrics, loss=loss, progress=progress, transitions=transitions, normal_regions=normal_regions,
                    comparison=comparison, videos=videos, probes=probe_records,
                    probe=probe_records[0] if probe_records else None,
                    causal_scope='Combined planning/control/active-finger strategy; not a single-component ablation')
    (output / 'active_comparison.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    relative_case = Path(os.path.relpath(case, output)).as_posix()
    def case_link(filename):
        return html.escape((Path(relative_case) / filename).as_posix())
    finger_names = ('食指', '中指', '无名指', '拇指')
    rows = []
    for item in metrics['per_finger']:
        finger = item['finger']
        joint = item['measured_joint_motion']
        ranges = ' / '.join(f'{value:.1f}°' for value in np.rad2deg(joint.get('range_rad_by_joint', []))) or '无实测关节数据'
        search = _finger_search(metrics, finger)
        rows.append(f'<tr><td>{finger_names[finger]}</td><td>{item["safe_contact_duration_s"]:.2f} s / {100*item["safe_contact_fraction"]:.2f}%</td>'
                    f'<td>{item["longest_absence_s"]:.2f} s</td><td>{ranges}</td>'
                    f'<td>{joint.get("observed_motion_while_off_contact_s", 0.):.2f} s</td>'
                    f'<td>{search["proposal_windows"]}</td><td>{search["measured_stable_recoveries"]}</td>'
                    f'<td>{search["recoveries_with_actual_joint_motion"]}</td></tr>')
    comparison_html = '<p class="warning">新旧初态不完全相同，已排除配对比较。</p>'
    if comparison['status'] == 'PAIRED':
        old, new = comparison['old_metrics'], comparison['new_metrics']
        comparison_html = f'''<p>初始化 qpos、qvel 和 ctrl 逐元素完全一致；只比较双方共同的 0–{comparison['horizon_s']:.2f} s 保存前缀。</p>
<div class="scroll"><table><thead><tr><th>指标</th><th>此前方案</th><th>本轮方案</th></tr></thead><tbody>
<tr><td>安全接触占比</td><td>{100*old['safe_contact_continuity_fraction']:.2f}%</td><td>{100*new['safe_contact_continuity_fraction']:.2f}%</td></tr>
<tr><td>安全切向移动占比</td><td>{100*old['safe_moving_contact_fraction']:.2f}%</td><td>{100*new['safe_moving_contact_fraction']:.2f}%</td></tr>
<tr><td>最长实际连续滑动</td><td>{old['longest_sliding_bout_s']:.2f} s</td><td>{new['longest_sliding_bout_s']:.2f} s</td></tr>
<tr><td>实测面积覆盖</td><td>{100*comparison['old_coverage']:.2f}%</td><td>{100*comparison['new_coverage']:.2f}%</td></tr>
<tr><td>最长全部手指失联</td><td>{comparison['old_loss']['longest_any_contact_gap_s']:.2f} s</td><td>{comparison['new_loss']['longest_any_contact_gap_s']:.2f} s</td></tr>
</tbody></table></div>'''
    scope = '整瓶实测覆盖与末段安全接触均已核验' if full_confirmed else '开发验证 · 尚未完成整瓶覆盖'
    never = [finger_names[item['finger']] for item in metrics['per_finger'] if item['contact_duration_s'] == 0.]
    absence = ('<p class="warning">本次' + '、'.join(never) + '没有实测接触。即使关节已运动或几何目标距离很小，也不能记作寻触成功。</p>') if never else ''
    development_stop = ('<p class="warning">本次由开发者在检查问题后主动终止（DEVELOPMENT_STOP_REQUESTED），'
                        '属于开发验证记录，未达到整瓶覆盖终止条件。</p>'
                        if summary['stop_reason'] == 'DEVELOPMENT_STOP_REQUESTED' else '')
    phase_text = '；'.join(f'{name} {duration:.2f} s' for name, duration in transitions['sampled_phase_duration_s'].items())
    cycle_text = '；'.join(f'{item["start_s"]:.3f}–{item["end_s"]:.3f} s，共 {item["duration_s"]:.3f} s'
                          + ('（至重触事件）' if item['ended_with_recontact_event'] else '（记录末仍未结束）')
                          for item in transitions['lift_cycles']) or '没有抬起切换事件'
    region_rows = ''.join(f'<tr><td>{region["region"]}</td><td>{100*region["area_fraction_of_full_surface"]:.2f}%</td>'
                          f'<td>{100*region["covered_fraction_within_region"]:.2f}%</td>'
                          f'<td>{region["covered_vertex_count"]}/{region["vertex_count"]}</td>'
                          f'<td>{region["covered_required_goals"]}/{region["required_goal_count"]}</td></tr>'
                          for region in normal_regions['regions'])
    probe_html = _probe_sections(probe_records, output)
    audit_link = f'<a href="{case_link("audit.json")}">{html.escape(str(audit.get("status")))}</a>' if (case / 'audit.json').exists() else '尚未审计'
    page = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mustard · 连续贴滑与各指主动寻触</title><style>body{{font-family:system-ui,"Microsoft YaHei",sans-serif;background:#f4f7f8;color:#172d36;max-width:1380px;margin:36px auto;padding:0 24px}}h1{{font-size:34px}}p{{line-height:1.8}}section{{background:white;border:1px solid #d7e2e5;border-radius:14px;padding:24px;margin:22px 0}}.badge{{display:inline-block;background:#dff2ec;padding:7px 12px;border-radius:7px;margin:5px 7px 0 0}}.warning{{color:#9b5718}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:11px 9px;border-bottom:1px solid #dfe7eb;text-align:left;white-space:nowrap}}th{{background:#f1f5f6}}a{{color:#196bc0}}video,img{{width:100%;border-radius:8px}}code{{font-size:12px}}</style></head><body>
<h1>Mustard · 连续贴滑与各指主动寻触</h1>
<p>在保持可用接触的同时沿瓶面推进；失联手指独立寻触，必要时抬起调整方位。所有效果由实际关节状态、触点和力验证。</p>
<span class="badge">{scope}</span><span class="badge">B-MCC 单案 · seed {summary['seed']}</span><span class="badge">固定物体 MuJoCo · DP 未运行</span>
<section><h2>本次完整保存记录</h2><p>连续执行 {summary['executed_s']:.2f} s；停止原因 <code>{html.escape(summary['stop_reason'])}</code>。
实测面积覆盖 <b>{100*summary['final_covered_area_fraction']:.2f}%</b>，表面点 <b>{summary['covered_vertex_count']}/8192</b>，必访点 <b>{summary['visited_required_goals']}/388</b>。
整瓶完成标志 <b>{str(full_confirmed).lower()}</b>；未因不可达而删除表面区域。</p>
<p>安全接触 {100*metrics['safe_contact_continuity_fraction']:.2f}%，安全切向移动 {100*metrics['safe_moving_contact_fraction']:.2f}%；最长实际连续滑动 {metrics['longest_sliding_bout_s']:.2f} s，最大滑动跨度 {1000*metrics['longest_sliding_bout_span_m']:.1f} mm（两项可能来自不同片段）。
抬起启动 {summary.get('lift_count',0)} 次，最长全部手指离面 {loss['longest_any_contact_gap_s']:.2f} s，实际无接触累计 {loss['observed_no_contact_duration_s']:.2f} s，瞬时峰值力 {summary['peak_force_n']:.2f} N。</p>
<p>证据审计：{audit_link}。审计通过说明记录和物理证据一致，不等于整瓶覆盖或四指均成功恢复。</p>{development_stop}{absence}</section>
{video_html}
<section><h2>整瓶各朝向区域是否开始覆盖</h2><p>按保存表面外法向的最大绝对分量划分 ±X/±Y/±Z；这是诊断分区，
全表面 8192 点与 388 必访点分母保持不变。面积沿用每个三角形的面积均分至三个顶点，与原覆盖账本一致。</p>
<div class="scroll"><table><thead><tr><th>主导外法向</th><th>该区占整瓶面积</th><th>该区内实测覆盖</th><th>已覆盖点/区域点</th><th>已覆盖必访点/区域必访点</th></tr></thead><tbody>{region_rows}</tbody></table></div>
<p>坐标沿用 surface_graph 的任务坐标系。覆盖仍指实际触点的 25 mm 测地邻域，不表示该区每个顶点都被直接触到；各区合计复现本次整段实测覆盖。</p></section>
<section><h2>换面耗时与尚未解决的覆盖停滞</h2><p>保存样本的阶段占用：{html.escape(phase_text)}。抬起启动至重触事件：{html.escape(cycle_text)}。
阶段和事件只用于解释调度耗时；是否接触、离面多久仍采用实测力，不以重触事件代替传感器。</p>
<p>最后一次实测覆盖增加在 {progress['last_measured_coverage_increase_s']:.2f} s，此后 {progress['terminal_no_new_coverage_s']:.2f} s 没有新增覆盖，期间仍启动 {progress['terminal_strokes_started']} 条表面轨迹。
末段 {progress['final_10s_start_s']:.2f}–{trace['time_s'][-1]:.2f} s 的覆盖为 {100*progress['final_10s_start_coverage']:.4f}% → {100*progress['final_10s_end_coverage']:.4f}%。
持续接触和重复滑动并不自动等于遍历新表面；此项单独保留为后续规划需要解决的问题。</p></section>
<section><h2>各指到底动了多少、是否重新接触</h2><div class="scroll"><table><thead><tr><th>手指</th><th>实际安全接触</th><th>最长离面（含初始未接触）</th><th>四关节实际角度范围</th><th>离面且实际关节运动时间</th><th>寻触增量窗口</th><th>同指实测稳定恢复</th><th>恢复且有关节实测动作</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p>寻触窗口来自规划层的增量提议标记，可能被下游 MCC、限速或几何保护覆盖；它不是独立成功次数。实际关节角来自 q_rad，指令、偏置和几何目标均不冒充实际运动。
恢复要求同一手指实测 0.2–8 N 连续至少 150 ms；最后一列还要求该窗口内实际关节组位移至少 0.002 rad。腕部运动也可能参与恢复，因此这里只报告时间关联，不作单独因果归因。</p></section>
<section><h2>与此前方案的相同时间前缀比较</h2>{comparison_html}
<p>本轮同时改变表面规划、接触手指选择、姿态调整及失联手指搜索，是 B-only 方案级比较，不是纯主动手指模块的单因素消融。不能把局部较好的片段比例当成整段比例，也不能把旧 A 腕部输入边界实验作为同信息条件对照。</p></section>
<section><h2>原始测量图</h2><img src="{figure}" alt="同前缀接触比较、各指实际角度和力以及全表面覆盖"><p>关节图使用实际角度；接触图显示四指各自的实测接触，未把不同手指拼成一条连续滑动段。</p></section>
{probe_html}
<section><h2>判定边界与来源</h2><p>切向移动要求同一手指完整 0.5 s 位移窗达到 1 mm/s，并由指尖位移共同支持；法向压入不计入。严格滑动段至少 2 s、10 mm 切向跨度，短断触不会拼接。实际接触移动仍不能单凭这些数据区分摩擦滑移与滚动。</p>
<p>覆盖使用实际触点的 25 mm 表面测地邻域；全部 8192 点和 388 必访点均覆盖后，还须通过末段稳定安全接触审计，才确认整瓶完成。仿真秒数不是实机实时运行验证。协议 <code>{html.escape(manifest.get('protocol',''))}</code>。</p>
<p><a href="active_comparison.json">完整指标、各指搜索窗口与配对来源</a> · <a href="{case_link('summary.json')}">运行摘要</a> · <a href="{case_link('manifest.json')}">冻结来源与参数</a></p></section>
</body></html>'''
    (output / 'index.html').write_text(page, encoding='utf-8')
    print('ACTIVE_SLIDING_REPORT', output / 'index.html', flush=True)
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--baseline', type=Path, default=DEFAULT_BASELINE)
    parser.add_argument('--probe', type=Path, action='append',
                        help='Independent probe summary/directory; repeat for separate selected-finger pairs')
    args = parser.parse_args()
    report(args.directory, args.baseline, args.probe)


if __name__ == '__main__':
    main()
