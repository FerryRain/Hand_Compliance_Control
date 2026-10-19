"""Chinese development report with verified, equal-horizon sliding comparisons.

Reads each case's native trace, summary, audit and initial state. Only writes
comparison.json, sliding_comparison.png and index.html beneath the NEW output
root. The old full-surface experiment is always read-only. A failed initial-state
pair check is reported and excluded from the paired comparison.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import html
import json
from pathlib import Path
import re

import numpy as np

from .sliding_metrics import analyze_trace, _gaps


DEFAULT_BASELINE = Path(__file__).resolve().parents[1] / 'generated/new_i04_mustard_full_surface_v12_formal_20261013'
TRACE_KEYS = ('time_s', 'active', 'forces_n', 'contact_positions_world_m',
              'tip_positions_world_m', 'normals_world', 'phase', 'stroke_id',
              'nominal_path_distance_m', 'coverage')
METRIC_KEYS = ('schema', 'parameters', 'duration_s', 'sample_count',
               'fingertip_motion_corroborated', 'tangent_projection_applied', 'motion_space',
               'contact_continuity_fraction', 'safe_contact_continuity_fraction',
               'moving_contact_fraction', 'safe_moving_contact_fraction',
               'moving_contact_duration_s', 'safe_moving_contact_duration_s',
               'longest_strict_same_finger_contact_s', 'sliding_bout_count',
               'longest_sliding_bout_s', 'longest_sliding_bout_span_m',
               'measured_moving_path_sum_over_fingers_m', 'phase_summary')


def _json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def _trace(path):
    # Loading only the sensor ledger avoids repeatedly decompressing large q/qvel arrays.
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in TRACE_KEYS if name in archive.files}


def _compact(metrics, trace=None):
    result = {key: metrics[key] for key in METRIC_KEYS if key in metrics}
    if trace is not None:
        times = trace['time_s']
        contact = (trace['active'] & (trace['forces_n'] >= .2)).any(axis=1)
        safe = (trace['active'] & (trace['forces_n'] >= .2) & (trace['forces_n'] <= 8.)).any(axis=1)
        dt = np.diff(times)
        contact_gaps, safe_gaps = _gaps(times, contact), _gaps(times, safe)
        no_contact = ~contact[:-1] & ~contact[1:]
        reposition = np.isin(trace.get('phase', np.full(len(times), 'UNKNOWN')),
                               ['LIFT', 'TRANSFER', 'APPROACH', 'REACQUIRE', 'RETREAT', 'OUTSIDE'])
        reposition_intervals = reposition[:-1] & reposition[1:]
        result['contact_loss_and_reposition'] = dict(
            max_any_contact_gap_s=max((gap['duration_s'] for gap in contact_gaps), default=0.),
            max_any_safe_contact_gap_s=max((gap['duration_s'] for gap in safe_gaps), default=0.),
            contact_gap_count=len([gap for gap in contact_gaps if not gap['leading']]),
            recovered_contact_gap_count=len([gap for gap in contact_gaps if not gap['leading'] and gap['recovered']]),
            total_observed_no_contact_s=float(np.sum(dt[no_contact])),
            reposition_phase_time_s=float(np.sum(dt[reposition_intervals])),
            reposition_observed_airtime_s=float(np.sum(dt[reposition_intervals & no_contact])),
            gap_duration_definition='first absent sample to next present sample; trailing gaps end at last saved sample',
            reposition_phase_definitions_differ_between_versions=True)
    return result


def _initial_pair(new_case, old_case):
    checks = {}
    with np.load(new_case / 'initial.npz', allow_pickle=False) as new, np.load(old_case / 'initial.npz', allow_pickle=False) as old:
        for key in ('qpos', 'qvel', 'ctrl'):
            checks[key] = key in new.files and key in old.files and np.array_equal(new[key], old[key])
    return dict(exactly_equal=all(checks.values()), fields=checks)


def _prefix_pair(new_trace, old_trace, new_executed, old_executed):
    new_t, old_t = new_trace['time_s'], old_trace['time_s']
    if abs(float(new_t[0])) > 1e-9 or abs(float(old_t[0])) > 1e-9:
        raise ValueError('Paired traces must both begin at the acquisition endpoint, time_s=0')
    bound = min(float(new_t[-1]), float(old_t[-1]), float(new_executed), float(old_executed))
    # Select a shared observed timestamp. No interpolated force/contact samples
    # and no comparison of different-length windows at a non-aligned endpoint.
    shared = np.intersect1d(np.round(new_t[new_t <= bound + 1e-9], 9),
                           np.round(old_t[old_t <= bound + 1e-9], 9))
    if len(shared) < 2:
        raise ValueError('Fewer than two common saved timestamps in the available horizon')
    horizon = float(shared[-1])
    new_prefix = {key: value[new_t <= horizon + 1e-9] for key, value in new_trace.items()}
    old_prefix = {key: value[old_t <= horizon + 1e-9] for key, value in old_trace.items()}
    if abs(float(new_prefix['time_s'][-1]) - float(old_prefix['time_s'][-1])) > 1e-9:
        raise ValueError('Common timestamp selection did not align both trace endpoints')
    return horizon, new_prefix, old_prefix


def _moving_samples(trace, metrics):
    moving = np.zeros(len(trace['time_s']), dtype=bool)
    for run in metrics['moving_runs']:
        moving |= (trace['time_s'] >= run['start_s'] - 1e-10) & (trace['time_s'] <= run['end_s'] + 1e-10)
    return moving


def _plot(directory, cases, selected):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return dict(created=False, reason='matplotlib unavailable')
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout='constrained')
    ax = axes[0, 0]
    paired = [case for case in cases if case.get('comparison', {}).get('status') == 'PAIRED']
    x = np.arange(len(paired))
    if paired:
        ax.bar(x - .18, [100 * case['comparison']['old']['safe_moving_contact_fraction'] for case in paired],
               .36, label='Old route, same horizon', color='#8995a7')
        ax.bar(x + .18, [100 * case['comparison']['new']['safe_moving_contact_fraction'] for case in paired],
               .36, label='Contact-first sliding', color='#13867a')
        ax.set_xticks(x, [case['name'].replace('_seed', '\ns') for case in paired], fontsize=9)
        ax.legend(fontsize=9)
    ax.set(title='Safe moving contact: paired equal horizons', ylabel='Observed time (%)', ylim=(0, 100))
    ax.grid(axis='y', alpha=.2)
    if selected is not None:
        new, old, new_m, old_m, horizon = selected
        ax = axes[0, 1]
        for label, trace, color in [('Old route', old, '#8995a7'), ('Contact-first', new, '#13867a')]:
            ax.plot(trace['time_s'], trace['forces_n'].max(axis=1), lw=.8, label=label, color=color)
        ax.axhline(8., ls='--', color='#ad7020', lw=.8, label='8 N relief threshold')
        ax.set(title='B-MCC seed 7: maximum measured fingertip force', xlabel='Simulation time (s)', ylabel='Force (N)', xlim=(0, horizon))
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
        ax = axes[1, 0]
        for trace, metrics, base, color in [(old, old_m, 2., '#8995a7'), (new, new_m, 0., '#13867a')]:
            safe = (trace['active'] & (trace['forces_n'] >= .2) & (trace['forces_n'] <= 8.)).any(axis=1)
            moving = _moving_samples(trace, metrics)
            for mask, offset in [(safe, 1.), (moving, 0.)]:
                ax.fill_between(trace['time_s'], base + offset, base + offset + .7,
                                where=mask, step='post', color=color, alpha=.85)
        ax.set_yticks([.35, 1.35, 2.35, 3.35], ['New moving', 'New safe contact', 'Old moving', 'Old safe contact'])
        ax.set(title='Contact and same-finger tangential movement', xlabel='Simulation time (s)', xlim=(0, horizon))
        ax.grid(axis='x', alpha=.2)
        ax = axes[1, 1]
        active = new['active'] & (new['forces_n'] >= .2) & (new['forces_n'] <= 8.)
        rows, fingers = np.nonzero(active)
        stride = max(1, len(rows) // 15000)
        rows, fingers = rows[::stride], fingers[::stride]
        if len(rows):
            points = new['contact_positions_world_m'][rows, fingers]
            scatter = ax.scatter(points[:, 0], points[:, 1], c=new['time_s'][rows], s=3, cmap='viridis', vmin=0, vmax=horizon)
            fig.colorbar(scatter, ax=ax, label='Simulation time (s)')
        ax.set(title='New B-MCC seed 7: actual safe contact points (XY)', xlabel='World X (m)', ylabel='World Y (m)', aspect='equal')
        ax.grid(alpha=.2)
    else:
        for ax in (axes[0, 1], axes[1, 0], axes[1, 1]):
            ax.text(.5, .5, 'B-MCC seed 7 paired evidence unavailable', ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
    fig.suptitle('Mustard fixed-object MuJoCo | partial development validation | measured evidence only', fontsize=13)
    path = directory / 'sliding_comparison.png'
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return dict(created=True, file=path.name, selected_case='B-MCC_seed7' if selected is not None else None)


def _video_sections(directory):
    case = directory / 'B-MCC_seed7'
    sections, records = [], []
    for filename, title in [('full_traversal.mp4', 'B-MCC · seed 7 · 整段原生状态回放'),
                            ('sliding_closeup.mp4', 'B-MCC · seed 7 · 连续滑动近景')]:
        if not (case / filename).exists():
            continue
        metadata_path = case / (Path(filename).stem + '_render.json')
        record = _json(metadata_path) if metadata_path.exists() else {}
        records.append(dict(file=f'B-MCC_seed7/{filename}', metadata=record))
        period = [record['start_s'], record['end_s']] if 'start_s' in record and 'end_s' in record else None
        speed = record.get('playback_speed')
        if period is not None and len(period) == 2 and speed is not None:
            caption = f'仿真时间 {float(period[0]):.2f}–{float(period[1]):.2f} s；{float(speed):g}× 回放。'
            caption += '该时间区间连续保留，无中途剪接。' if record.get('uncut') else '连续性请以视频来源记录为准。'
        else:
            caption = '本地保存状态回放；精确倍速和源时间区间尚无配套元数据，不用于定量接触判定。'
        sections.append(f'<section><h2>{title}</h2><video controls preload="metadata" src="B-MCC_seed7/{filename}"></video>'
                        f'<p>{html.escape(caption)}录像由保存的仿真状态生成；接触与覆盖由原始物理日志核验。</p></section>')
    return ''.join(sections), records


def _development_history(paths):
    batches = []
    for path in paths:
        path = Path(path).resolve()
        failures = []
        for summary_path in sorted(path.glob('[AB]-MCC_seed*/summary.json')):
            summary = _json(summary_path)
            if summary.get('stop_reason') != 'SUSTAINED_DANGEROUS_FORCE':
                continue
            manifest = _json(summary_path.parent / 'manifest.json')
            failures.append(dict(case=summary_path.parent.name, stop_reason=summary['stop_reason'],
                                 executed_s=summary['executed_s'], peak_force_n=summary['peak_force_n'],
                                 source_hashes=manifest.get('source_hashes', {})))
        batches.append(dict(directory=str(path), name=path.name,
                            included_in_current_metrics=False, sustained_force_stops=failures))
    return batches


def report(directory, baseline=DEFAULT_BASELINE, history=()):
    directory = Path(directory).resolve()
    baseline = Path(baseline).resolve()
    if directory == baseline or baseline in directory.parents or directory in baseline.parents:
        raise ValueError('New output root and baseline must be separate directories, not nested')
    paths = sorted((path for path in directory.iterdir() if path.is_dir() and re.fullmatch(r'[AB]-MCC_seed\d+', path.name)),
                   key=lambda path: (path.name.split('_')[0], int(path.name.split('seed')[1])))
    if not paths:
        raise ValueError(f'No A/B MCC case directories in {directory}')
    cases, selected, issues = [], None, []
    for case_path in paths:
        name = case_path.name
        required = ('summary.json', 'trace.npz', 'initial.npz', 'manifest.json')
        missing = [filename for filename in required if not (case_path / filename).exists()]
        if missing:
            issues.append(dict(case=name, issue='INCOMPLETE_CASE', missing=missing))
            continue
        summary, manifest = _json(case_path / 'summary.json'), _json(case_path / 'manifest.json')
        audit = _json(case_path / 'audit.json') if (case_path / 'audit.json').exists() else {'status': 'MISSING'}
        new = _trace(case_path / 'trace.npz')
        metrics = analyze_trace(new)
        compact_summary = {key: value for key, value in summary.items() if key != 'sliding_metrics'}
        item = dict(name=name, summary=compact_summary, audit_status=audit.get('status', 'UNKNOWN'),
                    audit_scope=audit.get('scope', 'Unavailable'), new_run=_compact(metrics, new),
                    protocol=manifest.get('protocol'), source_hashes=manifest.get('source_hashes', {}),
                    comparison=dict(status='BASELINE_UNAVAILABLE'))
        old_path = baseline / name
        if all((old_path / filename).exists() for filename in ('summary.json', 'trace.npz', 'initial.npz')):
            pair = _initial_pair(case_path, old_path)
            item['paired_initial_state'] = pair
            if not pair['exactly_equal']:
                item['comparison'] = dict(status='EXCLUDED_INITIAL_STATE_MISMATCH')
                issues.append(dict(case=name, issue='INITIAL_STATE_MISMATCH', checks=pair['fields']))
            else:
                old = _trace(old_path / 'trace.npz')
                old_summary = _json(old_path / 'summary.json')
                try:
                    horizon, new_prefix, old_prefix = _prefix_pair(new, old, summary['executed_s'], old_summary['executed_s'])
                    new_m, old_m = analyze_trace(new_prefix), analyze_trace(old_prefix)
                    item['comparison'] = dict(status='PAIRED', start_s=0., horizon_s=horizon,
                        new=_compact(new_m, new_prefix), old=_compact(old_m, old_prefix),
                        new_covered_area_fraction=float(new_prefix['coverage'][-1]),
                        old_covered_area_fraction=float(old_prefix['coverage'][-1]),
                        new_logged_peak_force_n=float(new_prefix['forces_n'].max()),
                        old_logged_peak_force_n=float(old_prefix['forces_n'].max()))
                    if name == 'B-MCC_seed7':
                        selected = (new_prefix, old_prefix, new_m, old_m, horizon)
                except ValueError as error:
                    item['comparison'] = dict(status='EXCLUDED_TIME_ALIGNMENT', reason=str(error))
                    issues.append(dict(case=name, issue='TIME_ALIGNMENT', reason=str(error)))
        cases.append(item)
    if not cases:
        raise ValueError('No complete new traces to report')
    plot = _plot(directory, cases, selected)
    videos, video_records = _video_sections(directory)
    prior_batches = _development_history(history)
    metrics_source = Path(__file__).with_name('sliding_metrics.py')
    result = dict(schema='i04.sliding_report.v1', scope='PARTIAL_DEVELOPMENT_VALIDATION',
                  full_surface_claim=False, baseline_directory=str(baseline),
                  metrics_source_sha256=sha256(metrics_source.read_bytes()).hexdigest(),
                  initial_pair_policy='qpos, qvel and ctrl must match exactly before comparison',
                  comparison_policy='From common acquired time 0 to latest shared saved timestamp bounded by both executed times and trace ends',
                  cases=cases, issues=issues, plot=plot, videos=video_records,
                  development_history=prior_batches)
    paired = [case for case in cases if case['comparison']['status'] == 'PAIRED']
    aggregates = {}
    for group in ('ALL', 'A-MCC', 'B-MCC'):
        group_cases = [case for case in paired if group == 'ALL' or case['name'].startswith(group)]
        if not group_cases:
            continue
        aggregates[group] = dict(case_count=len(group_cases))
        for version in ('old', 'new'):
            aggregates[group][version] = {
                metric: float(np.mean([case['comparison'][version][metric] for case in group_cases]))
                for metric in ('safe_contact_continuity_fraction', 'safe_moving_contact_fraction',
                               'longest_sliding_bout_s', 'longest_sliding_bout_span_m')}
            aggregates[group][version]['mean_covered_area_fraction'] = float(np.mean([
                case['comparison'][version + '_covered_area_fraction'] for case in group_cases]))
    result['paired_case_means'] = aggregates
    (directory / 'comparison.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    rows, paired_rows = [], []
    for case in cases:
        summary, m, name = case['summary'], case['new_run'], case['name']
        audit_link = f'<a href="{name}/audit.json">{html.escape(case["audit_status"])}</a>' if (directory / name / 'audit.json').exists() else '尚未审计'
        rows.append(f'<tr><td><a href="{name}/summary.json">{name}</a></td><td>{summary["executed_s"]:.2f}</td>'
                    f'<td>{100*summary["final_covered_area_fraction"]:.2f}%</td><td>{100*m["safe_contact_continuity_fraction"]:.2f}%</td>'
                    f'<td>{100*m["safe_moving_contact_fraction"]:.2f}%</td><td>{m["longest_sliding_bout_s"]:.2f} s</td>'
                    f'<td>{1000*m["longest_sliding_bout_span_m"]:.1f} mm</td><td>{summary.get("lift_count", 0)}</td>'
                    f'<td>{summary["peak_force_n"]:.2f} N</td><td><code>{html.escape(summary["stop_reason"])}</code></td><td>{audit_link}</td></tr>')
        comparison = case['comparison']
        if comparison['status'] == 'PAIRED':
            old, new = comparison['old'], comparison['new']
            paired_rows.append(f'<tr><td>{name}</td><td>{comparison["horizon_s"]:.2f} s</td>'
                f'<td>{100*old["safe_contact_continuity_fraction"]:.2f}% → {100*new["safe_contact_continuity_fraction"]:.2f}%</td>'
                f'<td>{100*old["safe_moving_contact_fraction"]:.2f}% → <b>{100*new["safe_moving_contact_fraction"]:.2f}%</b></td>'
                f'<td>{old["longest_sliding_bout_s"]:.2f} → {new["longest_sliding_bout_s"]:.2f} s</td>'
                f'<td>{100*comparison["old_covered_area_fraction"]:.2f}% → {100*comparison["new_covered_area_fraction"]:.2f}%</td>'
                '<td>qpos / qvel / ctrl 完全相同</td></tr>')
        else:
            paired_rows.append(f'<tr><td>{name}</td><td colspan="6">未纳入配对比较：{html.escape(comparison["status"])}</td></tr>')
    audit_passed = sum(case['audit_status'] == 'PASSED' for case in cases)
    source_hashes = {case['source_hashes'].get('Module/i04_geometric_traversal/surface_slide.py') for case in cases}
    frozen = len(source_hashes) == 1 and None not in source_hashes
    figure = '<section><h2>实测接触、切向移动和力</h2><img src="sliding_comparison.png" alt="配对滑动指标与实测时间序列"><p>柱状图使用各组相同前缀；B-MCC seed 7 时间序列也截取相同时间区间。右下图只显示真实安全触点的 XY 投影，不连接失联前后的触点。</p></section>' if plot['created'] else ''
    issues_html = ('<p class="warning">部分证据未纳入比较：' + html.escape('; '.join(f'{entry["case"]}: {entry["issue"]}' for entry in issues)) + '。详细原因见 comparison.json。</p>') if issues else ''
    history_rows = []
    for batch in prior_batches:
        for failure in batch['sustained_force_stops']:
            history_rows.append(f'<tr><td>{html.escape(batch["name"])}</td><td>{failure["case"]}</td>'
                                f'<td>{failure["executed_s"]:.3f} s</td><td>{failure["peak_force_n"]:.2f} N</td>'
                                '<td>持续过力保护停止</td></tr>')
    history_html = ('<section><h2>保留的开发失败记录</h2><p>此前版本出现持续过力停止。后续增加主动腕部卸力，并在卸力恢复后从实测状态重规划，避免重新压回原路径。下面的失败记录独立保留，不并入本轮结果，不与本轮源码版本混合。</p>'
                    '<div class="scroll"><table><thead><tr><th>开发批次</th><th>组 / seed</th><th>停止时间</th><th>峰值力</th><th>原因</th></tr></thead><tbody>'
                    + ''.join(history_rows) + '</tbody></table></div><p>本轮重新运行和审计，效果以本页当前结果为准；历史失败并不因修改后重跑而消失。</p></section>') if history_rows else ''
    b7 = next((case for case in paired if case['name'] == 'B-MCC_seed7'), None)
    tradeoff = ''
    if b7 is not None:
        pair = b7['comparison']
        tradeoff = (f'<p>以 B-MCC seed 7 的相同前 {pair["horizon_s"]:.0f} s 为例，安全移动接触占比 '
                    f'{100*pair["old"]["safe_moving_contact_fraction"]:.2f}% → {100*pair["new"]["safe_moving_contact_fraction"]:.2f}%；'
                    f'覆盖率 {100*pair["old_covered_area_fraction"]:.2f}% → {100*pair["new_covered_area_fraction"]:.2f}%。'
                    '连续滑动与覆盖推进速度存在取舍，整段接触比例仍受换面空中转移影响；不能用局部贴面片段的高比例代表整个过程。</p>')
    page = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mustard · 优先连续贴面滑动</title><style>
body{{font-family:system-ui,"Microsoft YaHei",sans-serif;background:#f4f7f8;color:#172d36;max-width:1380px;margin:36px auto;padding:0 24px}}h1{{font-size:34px;margin-bottom:8px}}p,li{{line-height:1.8}}section{{background:white;border:1px solid #d7e2e5;border-radius:14px;padding:24px;margin:22px 0}}h2{{font-size:23px}}.badge{{display:inline-block;background:#dff2ec;padding:7px 12px;border-radius:7px;margin:5px 7px 0 0}}.warning{{color:#9b5718}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:11px 9px;border-bottom:1px solid #dfe7eb;text-align:left;white-space:nowrap}}th{{background:#f1f5f6}}a{{color:#196bc0}}video,img{{width:100%;border-radius:8px}}code{{font-size:12px}}details{{line-height:1.8}}summary{{cursor:pointer}}</style></head><body>
<h1>Mustard · 优先连续贴面滑动</h1>
<p>先沿已接触表面连续推进，并在规划中保持相邻点的姿态与法向连续；局部滑动无法继续时，再抬起、调整方位、重新接触。</p>
<span class="badge">局部开发验证 · {len(cases)} 次运行</span><span class="badge">固定物体 MuJoCo 仿真</span><span class="badge">MCC · DP 未运行</span>
<section><h2>这次优先验证的效果</h2><p>重点是手指在物体上保持接触并产生实际切向位移。静止按住可以增加接触时间，但不会被计为连续滑动。短暂失联可恢复；严格连续滑动段在失联时仍然断开，不把前后两段拼接。</p>
<p class="warning">本页是局部连续滑动的开发验证，不宣称完成整瓶遍历。表中保留实际终止原因和部分覆盖结果；全表面覆盖、长时间换面稳定性与实机性能仍须继续验证。</p></section>
<section><h2>本轮全部运行</h2><div class="scroll"><table><thead><tr><th>组 / seed</th><th>执行秒数</th><th>实测面积覆盖</th><th>安全接触占比</th><th>安全移动接触占比</th><th>最长滑动时长</th><th>最大滑动跨度</th><th>抬起启动数</th><th>瞬时峰值力</th><th>停止原因</th><th>证据审计</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p>时长最大值和跨度最大值可能来自不同滑动段；抬起启动数来自实际 LIFT_COMMIT 事件。证据审计 {audit_passed}/{len(cases)} 通过，只表示记录、传感器和覆盖账本一致，不等于行为目标全部达成。运行实现哈希{'一致' if frozen else '不完全一致，请查看各组来源'}。</p></section>
<section><h2>与旧方案比较 · 相同初态、相同时长</h2><p>每一行先逐元素核对初始化 qpos、qvel 和 ctrl，再截取双方共同的仿真时间前缀。表格均为“旧方案 → 本轮方案”，不会把本轮较短片段与旧方案数千秒的整段平均值比较。</p>
<div class="scroll"><table><thead><tr><th>组 / seed</th><th>共同时间区间</th><th>安全接触占比</th><th>安全移动接触占比</th><th>最长连续滑动</th><th>实测面积覆盖</th><th>初态核对</th></tr></thead><tbody>{''.join(paired_rows)}</tbody></table></div>{tradeoff}
<p>本轮表面弧长目标速度为 6 mm/s；旧方案腕部平移上限为 25 mm/s，两者不是同一种速度指标。规划与控制均有调整，因此这是方案级比较，不能归因于某一个组件。SLIDE 与旧 SURFACE 的阶段定义也不同，阶段内比例只用于各自诊断。最长实际连续滑动段和连续视频共同展示局部效果。</p>{issues_html}</section>
{videos}{figure}
{history_html}
<section><h2>指标怎样计算</h2><p>接触必须由实际力传感器支持；安全接触阈值为 0.2–8 N。移动判定要求同一手指在完整 0.5 s 窗口内，触点与指尖的切向净位移都达到 1 mm/s。法向压入、静止按压、不同手指切换和小幅抖动不作为有效滑动。</p>
<p>严格滑动段必须持续至少 2 s，且切向跨度至少 10 mm。路径以 0.1 s 采样和 0.5 mm 空间死区抑制高频抖动；≤0.1 s 的短断触可单列恢复统计，但不连接严格滑动段。覆盖来自真实触点周围 25 mm 表面测地邻域，与规划目标和录像无关。</p>
<p>以上占比按已记录的仿真时间加权，运动窗口不跨越断触。采样之间的连续性未经直接观测；“切向接触移动”仍不能区分摩擦滑移与滚动。此处仿真秒数不是实机实时运行证明，视频倍速也不改变统计。</p></section>
<section><details><summary>来源和可复核记录</summary><p>A-MCC 只接收 wrist 位姿；B-MCC 额外接收四个 tip XYZ。各组 summary、audit、trace 和 source_snapshot 保留原始证据，旧实验目录只读。</p>
<p><a href="comparison.json">本轮指标、相同时长比较与来源哈希</a> · <a href="B-MCC_seed7/manifest.json">B-MCC seed 7 配置</a></p>
<p>当前指标版本 <code>{html.escape(cases[0]['new_run']['schema'])}</code>；滑动统计源码 SHA256 <code>{result['metrics_source_sha256']}</code>。</p></details></section>
</body></html>'''
    (directory / 'index.html').write_text(page, encoding='utf-8')
    print('SLIDING_REPORT', directory / 'index.html', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path, help='New root containing A/B-MCC_seedN cases')
    parser.add_argument('--baseline', type=Path, default=DEFAULT_BASELINE)
    parser.add_argument('--history', action='append', type=Path, default=[],
                        help='Separate earlier development batch to disclose; never mixed into metrics')
    args = parser.parse_args()
    report(args.directory, args.baseline, args.history)


if __name__ == '__main__':
    main()
