"""Offline evidence report for actual finger motion and coordinated recovery.

Reads saved sources only; writes a separate report directory. Native audits
establish recording integrity, while same-finger measured contact establishes
the behavioral outcome. An audited failure remains a failure. The frozen v2.2
whole-surface checkpoint is never credited to the new integrated controller.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from html import escape
from io import BytesIO
import json
import os
from pathlib import Path
from urllib.parse import quote

import numpy as np

from .active_sliding_report import surface_normal_regions, KEYS, _loss_metrics
from .sliding_metrics import analyze_trace
from .sliding_report import _initial_pair, _prefix_pair


GENERATED = Path(__file__).resolve().parents[1] / 'generated'
FINGER_NAMES = ('食指', '中指', '无名指', '拇指')


def read_json(path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def evidence_file(path):
    path = Path(path)
    if not path.is_file():
        return None
    raw = path.read_bytes()
    return dict(path=str(path.resolve()), sha256=sha256(raw).hexdigest(), bytes=len(raw))


def probe_pair(directory, label):
    """Require saved summaries and matching native audits before claiming success."""
    directory = Path(directory)
    paired_path = next((directory/name for name in ('paired_audit.json', 'paired_probe_audit.json')
                        if (directory/name).exists()), directory/'paired_audit.json')
    paired = read_json(paired_path)
    cases = {}
    for name in ('ON', 'OFF'):
        case = directory/name
        summary = read_json(case/'summary.json')
        audit_path = next((case/item for item in ('native_probe_audit.json', 'probe_audit.json')
                           if (case/item).exists()), case/'native_probe_audit.json')
        audit = read_json(audit_path)
        trace = evidence_file(case/'trace.npz') if summary and audit else None
        integrity = bool(summary and audit and trace and audit.get('status') == 'PASSED'
            and summary.get('trace_sha256') == trace['sha256'] == audit.get('trace_sha256'))
        stable = (summary.get('stable_confirmation_delay_s', summary.get('search_to_stable_confirmation_s'))
                  if summary else None)
        cases[name] = dict(summary=summary, audit=audit, audit_verified=integrity,
            first_contact_delay_s=summary.get('first_recontact_delay_s') if summary else None,
            stable_confirmation_delay_s=stable,
            measured_stable_recovery=bool(integrity and summary.get('success') and stable is not None
                and audit.get('same_finger_stable_recontact_observed')),
            evidence=dict(summary=evidence_file(case/'summary.json'), audit=evidence_file(audit_path), trace=trace))
    verified = bool(paired and paired.get('status') == 'PASSED'
                    and all(item['audit_verified'] for item in cases.values()))
    return dict(label=label, directory=str(directory.resolve()), paired_audit=paired,
                paired_audit_file=evidence_file(paired_path), paired_verified=verified, cases=cases,
                success=bool(verified and cases['ON']['measured_stable_recovery']),
                scope='Local paired recovery only; no whole-surface coverage')


def completed_run_evidence(directory, summary):
    directory = Path(directory)
    audit_path = next((directory/name for name in ('audit_summary.json', 'native_audit.json', 'audit.json')
                       if (directory/name).exists()), directory/'audit.json')
    audit = read_json(audit_path)
    trace = evidence_file(directory/'trace.npz') if summary and audit else None
    verified = bool(summary and audit and audit.get('status', audit.get('result')) == 'PASSED' and trace
        and audit.get('trace_sha256') == trace['sha256'] == summary.get('trace_sha256'))
    full = bool(verified and summary.get('full_surface_complete'))
    status = ('RUNNING_WITHOUT_FINAL_SUMMARY' if summary is None else
              'FULL_MEASURED_AUDIT_VERIFIED' if full else
              'FULL_MEASURED_PENDING_AUDIT' if summary.get('full_surface_complete') else 'FINISHED_INCOMPLETE')
    return dict(audit=audit, audit_verified=verified, full_surface_success=full, status=status,
                audit_file=evidence_file(audit_path), trace_file=trace)


def old_checkpoint(directory, snapshot_path=None, *, version_scope='Frozen v2.2 only; not evidence for v3.2 integration'):
    directory = Path(directory)
    checkpoint = directory/'coverage_checkpoint.npz'
    result = dict(directory=str(directory.resolve()), manifest=read_json(directory/'manifest.json'),
        progress=read_json(directory/'progress.json'), summary=read_json(directory/'summary.json'),
        version_scope=version_scope, checkpoint=None)
    if checkpoint.exists():
        # One read fixes the coverage mask and timestamp atomically together.
        raw = checkpoint.read_bytes()
        if snapshot_path is not None:
            Path(snapshot_path).write_bytes(raw)
        with np.load(BytesIO(raw), allow_pickle=False) as saved:
            regions = surface_normal_regions(directory/'surface_graph.npz', saved['covered_vertices'])
            result['checkpoint'] = dict(time_s=float(saved['elapsed']),
                source_sha256=sha256(raw).hexdigest(), covered_vertices=int(saved['covered_vertices'].sum()),
                visited_required_goals=int(saved['required_goal_visited'].sum()), **regions)
    result['evidence'] = {name: evidence_file(directory/name) for name in
        ('manifest.json', 'progress.json', 'summary.json', 'surface_graph.npz')}
    result.update(completed_run_evidence(directory, result['summary']))
    manifest = result['manifest'] or {}
    source = manifest.get('runtime_entrypoint', 'Module/i04_geometric_traversal/surface_slide.py')
    frozen_hash = manifest.get('source_hashes', {}).get(source)
    workspace = evidence_file(Path(__file__).resolve().parents[2]/source)
    snapshot = evidence_file(directory/'source_snapshot'/source)
    result['runtime_source'] = dict(entrypoint=source, frozen_manifest_sha256=frozen_hash,
        saved_snapshot_sha256=snapshot['sha256'] if snapshot else None,
        current_workspace_sha256=workspace['sha256'] if workspace else None,
        frozen_snapshot_matches=bool(snapshot and snapshot['sha256'] == frozen_hash),
        current_workspace_matches=bool(workspace and workspace['sha256'] == frozen_hash))
    return result


def same_prefix_comparison(new_run, old_run, end_s=120.):
    """Same saved grid/method for both versions, retaining unfavorable outcomes."""
    new_run, old_run = Path(new_run), Path(old_run)
    initial = _initial_pair(new_run, old_run)
    if not initial['exactly_equal']:
        raise ValueError('Version comparison requires identical initial qpos/qvel/ctrl')
    def prefix(directory):
        paths = ([directory/'trace.npz'] if (directory/'trace.npz').exists()
                 else sorted((directory/'trace_chunks').glob('*.npz')))
        chunks, sources = [], []
        for path in paths:
            raw = path.read_bytes()
            with np.load(BytesIO(raw), allow_pickle=False) as saved:
                chunk = {key: saved[key].copy() for key in KEYS if key in saved.files}
            sources.append(dict(file=str(path.resolve()), sha256=sha256(raw).hexdigest()))
            chunks.append(chunk)
            if chunk['time_s'][-1] >= end_s:
                break
        if not chunks:
            raise ValueError('No saved comparison trace')
        keys = set(chunks[0])
        trace = {key: np.concatenate([chunk[key] for chunk in chunks]) for key in keys}
        if trace['time_s'][-1] < end_s:
            raise ValueError('Requested comparison prefix is not complete')
        return trace, sources
    new, new_sources = prefix(new_run)
    old, old_sources = prefix(old_run)
    horizon, new, old = _prefix_pair(new, old, end_s, end_s)
    if not np.array_equal(new['time_s'], old['time_s']):
        raise ValueError('Both versions must use the same saved sampling grid')
    result = dict(horizon_s=horizon, initial_state=initial, sources=dict(new=new_sources, old=old_sources),
        method='Identical sliding_metrics.v1.2 on saved 100 Hz trace; stable return counts are sampled observations',
        causal_scope='Combined planning/control strategy comparison, not a single-factor ablation')
    for name, directory, trace in (('new', new_run, new), ('old', old_run, old)):
        events = [item for item in (read_json(directory/'events.json') or []) if item['time_s'] <= horizon]
        result[name] = dict(metrics=analyze_trace(trace, events=events), loss=_loss_metrics(trace),
            coverage=float(trace['coverage'][-1]), covered_goals=int(trace['covered_goals'][-1]))
    return result


def integrated_run(directory):
    directory = Path(directory)
    summary, manifest = read_json(directory/'summary.json'), read_json(directory/'manifest.json')
    completed = completed_run_evidence(directory, summary)
    events = read_json(directory/'events.json') or []
    recovery_ends = [event for event in events if event.get('kind') == 'COORDINATED_RECOVERY_END']
    return dict(directory=str(directory.resolve()), manifest=manifest, summary=summary,
        progress=read_json(directory/'progress.json'), **completed,
        recovery_ends=recovery_ends,
        actual_stable_recovery_count=sum(bool(event.get('measured_recovery_success')) for event in recovery_ends),
        evidence={name: evidence_file(directory/name) for name in
            ('manifest.json', 'summary.json', 'audit.json', 'progress.json', 'events.json')})


def actual_recovery_episodes(directory, followup_s=30.):
    """Pair native recovery events with actual q and sampled follow-up contact."""
    directory = Path(directory)
    if not (directory/'trace.npz').exists() or not (directory/'summary.json').exists():
        return None
    events = read_json(directory/'events.json') or []
    with np.load(directory/'trace.npz', allow_pickle=False) as saved:
        time, q, active, forces, phase = (saved[key] for key in ('time_s','q_rad','active','forces_n','phase'))
    result = []
    for begin in (e for e in events if e.get('kind')=='COORDINATED_RECOVERY_BEGIN'):
        sequence, finger = begin['sequence'], begin['selected_finger']
        end = next((e for e in events if e.get('kind')=='COORDINATED_RECOVERY_END'
                    and e.get('sequence')==sequence), None)
        first = next((e for e in events if e.get('kind')=='COORDINATED_RECOVERY_FIRST_CONTACT'
                      and e.get('sequence')==sequence), None)
        stop = end['time_s'] if end else float(time[-1])
        window = (time >= begin['time_s']) & (time <= stop)
        joints = q[window, 7+4*finger:11+4*finger]
        item = dict(sequence=sequence, finger=finger, begin_s=begin['time_s'], end_s=stop,
            path_duration_s=begin['duration_s'], dispatch_mode=begin.get('dispatch_mode'),
            actual_recovery_phase_duration_s=stop-begin['time_s'],
            first_measured_contact_s=first['time_s'] if first else None,
            native_measured_stable=bool(end and end.get('measured_recovery_success')),
            terminal_status=end.get('status') if end else 'NO_TERMINAL_EVENT',
            actual_joint_change_deg=(np.rad2deg(joints[-1]-joints[0]).tolist() if len(joints) else None),
            actual_joint_range_deg=(np.rad2deg(np.ptp(joints,axis=0)).tolist() if len(joints) else None),
            followup=None)
        if item['native_measured_stable']:
            selected = (time >= stop) & (time <= stop+followup_s)
            t = time[selected]
            measured = active[selected, finger] & (forces[selected,finger]>=.2)
            safe = measured & (forces[selected,finger]<8.)
            if len(t)>1:
                dt = np.diff(t)
                longest_gap = current_gap = longest_safe = current_safe = 0.
                for index, duration in enumerate(dt):
                    current_gap = current_gap+duration if not measured[index] else 0.
                    current_safe = current_safe+duration if safe[index] else 0.
                    longest_gap, longest_safe = max(longest_gap,current_gap), max(longest_safe,current_safe)
                item['followup'] = dict(start_s=float(t[0]), end_s=float(t[-1]), duration_s=float(t[-1]-t[0]),
                    safe_contact_fraction=float(np.sum(dt*safe[:-1])/(t[-1]-t[0])),
                    actual_contact_fraction=float(np.sum(dt*measured[:-1])/(t[-1]-t[0])),
                    longest_sampled_absence_s=float(longest_gap), longest_sampled_safe_contact_s=float(longest_safe))
        result.append(item)
    absent = ~np.any(active & (forces>=.2),axis=1)
    starts = np.flatnonzero(absent & np.r_[True,~absent[:-1]])
    ends = np.flatnonzero(absent & np.r_[~absent[1:],True])
    gaps = [dict(start_s=float(time[a]), end_s=float(time[min(b+1,len(time)-1)]),
                 duration_s=float(time[min(b+1,len(time)-1)]-time[a]),
                 phases=np.unique(phase[a:b+1]).tolist()) for a,b in zip(starts,ends)]
    dt = np.diff(time)
    slide_gap = longest_slide_gap = 0.
    for i,duration in enumerate(dt):
        slide_gap = slide_gap+duration if absent[i] and phase[i]=='SLIDE' else 0.
        longest_slide_gap = max(longest_slide_gap,slide_gap)
    return dict(scope='Native first/stable events plus 100Hz actual joint/contact follow-up, no commanded-motion credit',
                followup_requested_s=followup_s, episodes=result,
                total_recovery_phase_duration_s=sum(e['actual_recovery_phase_duration_s'] for e in result),
                phase_durations_s={str(p):float(dt[phase[:-1]==p].sum()) for p in np.unique(phase)},
                longest_whole_hand_absence_intervals=sorted(gaps,key=lambda x:-x['duration_s'])[:10],
                longest_whole_hand_absence_within_slide_phase_s=float(longest_slide_gap),
                source_trace=evidence_file(directory/'trace.npz'), source_events=evidence_file(directory/'events.json'))


def terminal_chain_evidence(directory, old_record):
    """Keep a separately audited continuation distinct from both fresh runs."""
    directory = Path(directory)
    audit_path, evidence_path = directory/'independent_chain_audit.json', directory/'complete_chain_evidence.json'
    audit, evidence = read_json(audit_path), read_json(evidence_path)
    if audit is None or evidence is None:
        return None
    segments = []
    for segment in audit.get('segments', []):
        case = Path(segment['directory'])
        trace = evidence_file(case/'trace.npz')
        summary = read_json(case/'summary.json')
        matches = bool(trace and trace['sha256'] == segment.get('trace_sha256')
                       and summary and summary.get('trace_sha256') == trace['sha256'])
        segments.append(dict(**segment, actual_trace=trace, summary=summary, trace_matches=matches,
                             directory_normalized=str(case.resolve())))
    verified = bool(audit.get('status') == 'PASSED' and len(segments) == 6
        and all(segment['trace_matches'] for segment in segments)
        and old_record['audit_verified'] and old_record['trace_file']
        and audit.get('original_trace_sha256') == old_record['trace_file']['sha256']
        and audit.get('all_parent_endpoints_exact') is True
        and audit.get('maximum_native_replay_error') == 0.
        and audit.get('combined_vertices') == audit.get('total_vertices') == 8192
        and audit.get('combined_required_goals') == audit.get('total_required_goals') == 388
        and audit.get('final_full_surface_safe_hold_s', 0.) >= .150)
    return dict(directory=str(directory.resolve()), audit=audit, evidence=evidence, segments=segments,
                audit_verified=verified, audit_file=evidence_file(audit_path), evidence_file=evidence_file(evidence_path),
                scope='OLD_STOPPED_V2_2_PLUS_SEPARATE_NATIVE_CONTINUATION_CHAIN',
                fresh_whole_traversal_success=False, new_v3_coverage_credit=False)


def terminal_chain_html(chain, output):
    if chain is None:
        return ''
    if not chain['audit_verified']:
        return '<section><h2>旧终态的追加遍历</h2><p>追加链记录存在，完整审计与当前文件哈希尚未全部匹配，暂不声明完成。</p></section>'
    a = chain['audit']
    rows = ''.join(f'<tr><td>{i+1}</td><td>{segment["executed_s"]:.3f} s</td>'
        f'<td>{", ".join(str(v) for v in segment["added_vertices"])}</td>'
        f'<td>{segment["peak_force_n"]:.3f} N</td>'
        f'<td>{link(Path(segment["directory_normalized"])/"summary.json", "本段实测 summary",output)}</td></tr>'
        for i,segment in enumerate(chain['segments']))
    figure_dir = Path(chain['directory'])
    figure = figure_dir/'old_v22_continuation_contact_three_views.png'
    figure_html = ''
    if figure.exists():
        figure_html = (f'<img class="figure" src="{relative_url(figure,output)}" alt="旧 v2.2 与六段终态续接的实测接触轨迹，三个原始网格视角"/>'
            '<p class="caption">蓝线是旧运行中的实际接触；彩色短线与点是六段追加接触，红圈保留旧停止时尚未覆盖的 11 点。'
            '同一手指失联、接触点跳变和续接段之间断线。覆盖采用实测接触的 25 mm 测地邻域，不代表逐一直接触碰全部顶点。'
            f'{link(figure_dir/"old_v22_continuation_contact_three_views.pdf","下载图 PDF",output)} · '
            f'{link(figure_dir/"contact_figure_provenance.json","绘图来源与规则",output)} · '
            f'{link(figure_dir/"plot_contact_chain.py","重绘脚本",output)}</p>')
    return ('<section><h2>旧终态继续完成：独立的六段物理连续链</h2>'
        '<p>旧 v2.2 原运行在 99.78008% 停止，原文件和结果保留。从它的最终实际状态继续，'
        f'六段额外执行 <b>{a["continuation_duration_s"]:.3f} s</b>，补齐剩余 11 个表面点。'
        '<b>原运行＋追加链合计达到 8192/8192 表面点与 388/388 必访点。</b></p>'
        f'<p>各段起点与上一段真实末态逐值相同；{a["native_states_replayed_including_shared_endpoints"]:,} 个 '
        f'2 ms 原生状态逐帧重放最大误差为 0。最后完整覆盖后保持安全接触 '
        f'{a["final_full_surface_safe_hold_s"]:.3f} s；追加过程峰值力 {a["continuation_peak_force_n"]:.3f} N，腕自碰撞力为 0。</p>'
        '<p class="warning">这是从旧终态出发的追加验证。旧独立运行仍为 8181/8192、99.78008%；'
        '它不是新版 v3 从初态完成整瓶的结果。追加过程包含离面转移，全部覆盖也不等于全程连续贴滑。</p>'
        +figure_html+
        '<details><summary>查看六段覆盖及审计证据</summary><table><tr><th>段</th><th>执行时间</th>'
        '<th>新增原表面点 ID</th><th>峰值力</th><th>证据</th></tr>'+rows+'</table></details>'
        f'<p>{link(chain["audit_file"]["path"],"连续链独立原生审计",output)} · '
        f'{link(chain["evidence_file"]["path"],"追加链来源与限制",output)}</p></section>')


def boundary_run_html(record, output, recovery=None, comparisons=None):
    directory = Path(record['directory'])
    summary = record['summary']
    text = ('<section><h2>当前开发验证：滑动段结束后再短时调指</h2>'
        '<p>滑动途中记录失联手指的待办；在真实段终点，才尝试最长 8 s 的固定腕单指路径。'
        '端点无解不升级为腕调整；长调形和腕调整仍留给没有可行滑动续段时的恢复阶段。</p>')
    text += completion_html(record)
    if summary:
        text += (f'<p>本轮协议 <code>{escape(str((record["manifest"] or {}).get("protocol", "未知")))}</code>；'
            f'原生峰值力 {summary.get("peak_force_n", 0):.3f} N。'
            '调形会暂时降低表面移动速度，必须结合实际恢复与停滑时间评价，不能仅靠待办数量评价。</p>')
    else:
        progress = record['progress'] or {}
        text += f'<p>最近进度 {number(progress.get("time_s"),1)} s，面积 {100*progress.get("coverage",0):.2f}%；待完整窗口与审计。</p>'
    if comparisons and all(label in comparisons['cases'] for label in ('滑动优先＋新上下文 v3.4','自然端点短调形 v3.5')):
        previous = comparisons['cases']['滑动优先＋新上下文 v3.4']['metrics']
        current = comparisons['cases']['自然端点短调形 v3.5']['metrics']
        text += ('<p class="warning">同一 300 s 内，相比加入新上下文的上一版，拇指实际接触率 '
            f'{100*previous["per_finger"][3]["contact_fraction"]:.2f}% → {100*current["per_finger"][3]["contact_fraction"]:.2f}%，明显改善；'
            f'食指 {100*previous["per_finger"][0]["contact_fraction"]:.2f}% → {100*current["per_finger"][0]["contact_fraction"]:.2f}%，'
            f'中指 {100*previous["per_finger"][1]["contact_fraction"]:.2f}% → {100*current["per_finger"][1]["contact_fraction"]:.2f}%，均退步。'
            f'全程安全移动接触率 {100*previous["safe_moving_contact_fraction"]:.2f}% → {100*current["safe_moving_contact_fraction"]:.2f}%。'
            '本版证明了自然扫描中的主动调形回触，还没有同时改善四指参与和整体连续滑动。</p>')
    filename = ('boundary_active_recovery_clear_1x.mp4' if (directory/'boundary_active_recovery_clear_1x_render.json').exists()
                else 'boundary_active_recovery_1x.mp4')
    text += video(directory, filename, output, record['audit_verified'])
    text += ('<p class="caption">录像展示保存的实际关节 q 与实测接触。CONTROL INTENT 区分局部微寻触、完整关节调形与接触稳定阶段；'
        '它只表示控制意图。实际运动和回触以关节角度、Δ 和实测接触力为准。</p>'
        '<p>'+ ' · '.join(link(directory/name, name, output) for name in ('summary.json','audit.json','events.json'))+'</p></section>')
    if recovery and record['audit_verified']:
        rows = []
        for episode in recovery['episodes']:
            f = episode['followup']
            actual_range = ' / '.join(f'{value:.1f}' for value in episode['actual_joint_range_deg'] or [])
            retained = (f'{100*f["safe_contact_fraction"]:.1f}%（{f["duration_s"]:.2f}s窗口） / 最大失联 {f["longest_sampled_absence_s"]:.2f}s'
                        if f else '未达到稳定恢复')
            rows.append(f'<tr><td>{FINGER_NAMES[episode["finger"]]}</td>'
                f'<td>{episode["begin_s"]:.3f}–{episode["end_s"]:.3f}s</td>'
                f'<td>{actual_range}</td><td>{escape(episode["terminal_status"])}</td><td>{retained}</td></tr>')
        table = ('<p>这段自然扫描中的实际调形及恢复后参与：</p><table><tr><th>手指</th><th>实际恢复阶段</th>'
            '<th>实际四关节变化范围 / °</th><th>原生终态</th><th>后续约30s安全接触 / 最长失联</th></tr>'+''.join(rows)+'</table>'
            f'<p class="caption">全部恢复阶段合计 {recovery["total_recovery_phase_duration_s"]:.3f} s。'
            '表中恢复后参与来自 100 Hz 采样，不替代原生 2 ms 稳定判定，可能遗漏短于10ms的间断。'
            f'{link(output/"boundary_actual_recovery_followup.json","实际角度与恢复后窗口证据",output)}</p>')
        longest = (recovery['longest_whole_hand_absence_intervals'] or [dict(duration_s=0.,start_s=0.,end_s=0.,phases=[])])[0]
        table += (f'<p>本轮记录 {summary.get("lift_count")} 次明确抬离换区。最长全手失联 '
            f'{longest["duration_s"]:.2f} s 位于 {longest["start_s"]:.2f}–{longest["end_s"]:.2f} s，'
            f'记录阶段为 {escape(" / ".join(longest["phases"]))}。'
            f'SLIDE 阶段内，最长全手无接触采样间隔为 {recovery["longest_whole_hand_absence_within_slide_phase_s"]:.2f} s。'
            '允许短暂失联和必要的换区；这些间隔保留在结果中，不自动判定失败。</p>')
        text = text.replace('</section>',table+'</section>')
    return text


def number(value, digits=3):
    return '未观测' if value is None else f'{value:.{digits}f}'


def relative_url(path, output):
    return quote(Path(os.path.relpath(Path(path).resolve(), output.resolve())).as_posix(), safe='/')


def link(path, label, output):
    path = Path(path)
    return (f'<a href="{relative_url(path, output)}">{escape(label)}</a>' if path.exists()
            else f'<span class="muted">{escape(label)}（待生成）</span>')


def video(case, filename, output, verified):
    case = Path(case)
    source, metadata_path = case/filename, case/(Path(filename).stem+'_render.json')
    meta = read_json(metadata_path)
    if not verified or not source.exists() or not meta:
        return '<p class="muted">该片段等待完整审计或渲染元数据，暂不展示。</p>'
    trace = evidence_file(case/'trace.npz')
    matching_trace = (meta.get('trace_sha256') == trace['sha256'] or
        any(source.get('sha256') == trace['sha256'] for source in meta.get('sources', [])))
    if not matching_trace and meta.get('development_prefix') is True:
        from .active_sliding_video import KEYS as render_keys
        checks = []
        with np.load(case/'trace.npz', allow_pickle=False) as final:
            final_time = final['time_s']
            for item in meta.get('sources', []):
                source_path = Path(item['file'])
                saved_source = evidence_file(source_path)
                if (source_path.resolve().parent != (case/'trace_chunks').resolve() or not saved_source
                        or saved_source['sha256'] != item['sha256']):
                    checks.append(dict(source=str(source_path), match=False, reason='SOURCE_HASH_OR_SCOPE_MISMATCH'))
                    continue
                with np.load(source_path, allow_pickle=False) as chunk:
                    indices = np.searchsorted(final_time, chunk['time_s'])
                    fields = {}
                    if np.any(indices >= len(final_time)):
                        checks.append(dict(source=str(source_path), match=False, reason='OUTSIDE_FINAL_TRACE'))
                        continue
                    for key in render_keys:
                        if key not in chunk.files:
                            continue
                        a, b = chunk[key], final[key][indices]
                        fields[key] = bool(np.array_equal(a, b, equal_nan=True) if a.dtype.kind in 'fc'
                                           else np.array_equal(a, b))
                    checks.append(dict(source=str(source_path), samples=len(indices), fields=fields,
                                       match=bool(fields and all(fields.values()))))
        matching_trace = bool(checks and all(item['match'] for item in checks))
        prefix_check = dict(video=filename, final_trace_sha256=trace['sha256'],
                            matching_final_trace=matching_trace, sources=checks)
        (output/(Path(filename).stem+'_prefix_to_final_audit.json')).write_text(
            json.dumps(prefix_check, ensure_ascii=False, indent=2), encoding='utf-8')
    if not matching_trace:
        return '<p class="warning">录像源哈希未匹配，暂不展示。</p>'
    if meta.get('video_sha256') and meta['video_sha256'] != evidence_file(source)['sha256']:
        return '<p class="warning">录像文件哈希未匹配，暂不展示。</p>'
    speed = meta.get('playback_speed', meta.get('speed'))
    start = meta.get('start_s', meta.get('simulation_start_s'))
    end = meta.get('end_s', meta.get('simulation_end_s'))
    uncut = meta.get('uncut') is True or 'uncut' in str(meta.get('source', '')).lower()
    description = f'{number(start)}–{number(end)} s · {number(speed, 1)}× · '+('连续未剪辑' if uncut else '剪辑状态未声明')
    return (f'<video controls preload="metadata" src="{relative_url(source, output)}"></video>'
            f'<p class="caption">{description}。由保存的实际 q 做 FK 回放，力与接触读取实测记录；'
            f'视频帧率低于 2 ms 物理采样。{link(metadata_path, "录像元数据", output)}</p>')


def middle_figure(pair, output):
    if not pair['paired_verified']:
        return None
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    case = Path(pair['directory'])/'ON'
    with np.load(case/'trace.npz', allow_pickle=False) as saved:
        t, q, force = saved['time_s'], saved['q_rad'], saved['forces_n']
    finger = pair['cases']['ON']['summary']['selected_finger']
    with np.load(Path(pair['directory'])/'OFF/trace.npz', allow_pickle=False) as saved:
        off_t, off_force = saved['time_s'], saved['forces_n'][:, finger]
    fig, axes = plt.subplots(2, 1, figsize=(12, 5.8), sharex=True, constrained_layout=True)
    for joint in range(4):
        col = 7+4*finger+joint
        axes[0].plot(t, np.rad2deg(q[:, col]-q[0, col]), label=f'Middle joint {joint+1}')
    axes[0].set(ylabel='Measured change (deg)', title='Actual middle-finger motion from the saved t=205 s state')
    axes[0].legend(ncol=4, loc='upper left', fontsize=9)
    axes[1].plot(t, force[:, finger], label='ON: middle measured force', color='#167a67')
    axes[1].plot(off_t, off_force, label='OFF: middle measured force', color='#c16941', linestyle='--')
    axes[1].axhline(.2, color='#777', lw=1, linestyle=':', label='Contact threshold: 0.2 N')
    for axis in axes:
        axis.axvline(pair['cases']['ON']['first_contact_delay_s'], color='#334f80', lw=1, linestyle='--')
        axis.grid(alpha=.18); axis.set_xlim(float(t[0]), float(t[-1]))
    axes[1].set(ylabel='Actual force (N)', xlabel='Local probe time (s)')
    axes[1].legend(loc='upper left', fontsize=9)
    path = output/'middle_measured_motion.png'
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


def probe_table(pair, output):
    rows = []
    for name, record in pair['cases'].items():
        summary = record['summary'] or {}
        outcome = ('稳定恢复' if pair['paired_verified'] and record['measured_stable_recovery'] else
                   '发生重触，未稳定恢复' if record['first_contact_delay_s'] is not None else '窗口内未重触')
        if not pair['paired_verified']:
            outcome = '记录待完整审计 · '+outcome
        case = Path(pair['directory'])/name
        rows.append(f'<tr><td>{name}</td><td>{number(record["first_contact_delay_s"])} s</td>'
            f'<td>{number(record["stable_confirmation_delay_s"])} s</td><td>{number(summary.get("peak_force_n"))} N</td>'
            f'<td>{outcome}</td><td>{link(case/"summary.json", "summary", output)}</td></tr>')
    audit = pair['paired_audit_file']
    audit_link = link(audit['path'], '配对原生重放审计', output) if audit else '审计待生成'
    return ('<table><thead><tr><th>条件</th><th>首次实测重触</th><th>150 ms 稳定确认</th>'
            '<th>峰值力</th><th>行为结果</th><th>证据</th></tr></thead><tbody>'+''.join(rows)+
            f'</tbody></table><p class="caption">{audit_link}。PASSED 表示证据可重放，行为结果以同指实测接触为准。</p>')


def completion_html(record):
    summary = record['summary']
    if summary is None:
        return '<p><b>状态：运行中，尚无最终 summary。</b> 当前覆盖不能作为整瓶完成结论。</p>'
    state = ('整瓶测量完成，终态 summary 与 PASSED 审计均匹配实际 trace 哈希'
             if record['full_surface_success'] else
             'summary 声明完成，完整审计或 trace 哈希匹配仍待确认'
             if summary.get('full_surface_complete') else '运行已结束，整瓶未完成')
    return (f'<p><b>状态：{state}。</b>停止原因 <code>{escape(str(summary.get("stop_reason")))}</code>；'
        f'最终执行 {number(summary.get("executed_s"), 2)} s；面积 '
        f'{100*summary.get("final_covered_area_fraction", 0):.4f}%；表面点 '
        f'{summary.get("covered_vertex_count")}/{summary.get("vertex_count")}，必访点 '
        f'{summary.get("visited_required_goals")}/{summary.get("required_goal_count")}。</p>')


def long_run_html(record, snapshot_path, output, prefix_audit_path):
    checkpoint, progress = record['checkpoint'], record['progress'] or {}
    source, directory = record['runtime_source'], Path(record['directory'])
    text = ('<section><h2>指定新版长程运行：独立覆盖账本</h2>'
        f'<p>协议 <code>{escape(str((record["manifest"] or {}).get("protocol", "未知")))}</code>。</p>'+completion_html(record))
    if checkpoint:
        text += (f'<p>同一 checkpoint 快照 <b>{checkpoint["time_s"]:.2f} s</b>：面积 '
            f'<b>{100*checkpoint["measured_coverage_fraction"]:.4f}%</b>，表面点 '
            f'<b>{checkpoint["covered_vertices"]}/{checkpoint["full_vertex_count"]}</b>，必访点 '
            f'<b>{checkpoint["visited_required_goals"]}/{checkpoint["full_required_goal_count"]}</b>。</p>')
    else:
        text += '<p>尚无完整落盘的 checkpoint。</p>'
    text += (f'<p>另列最后进度日志：{number(progress.get("time_s"), 1)} s，'
        f'{100*progress.get("coverage", 0):.4f}% 面积，阶段 <code>{escape(str(progress.get("phase", "未知")))}</code>。'
        '该日志不用于补写上一个 checkpoint 的点数或面积。</p>'
        f'<p class="caption">冻结入口：<code>{escape(source["entrypoint"])}</code><br>'
        f'本次运行 manifest SHA-256：<code>{escape(str(source["frozen_manifest_sha256"]))}</code><br>'
        f'当前工作区入口 SHA-256：<code>{escape(str(source["current_workspace_sha256"]))}</code><br>'
        f'运行快照匹配：{source["frozen_snapshot_matches"]}；当前工作区匹配：{source["current_workspace_matches"]}。'
        '运行来源以自己的冻结 manifest 与快照为准，后续工作区修改不回写历史。</p>'
        '<p class="warning">本区只读取新版长程运行目录，未合并旧 run04 或专项 probe 的任何覆盖。'
        '只有最终 summary 明确 full_surface_complete=true，且 PASSED 审计、summary 与实际 trace 哈希一致，才称新版整瓶通过。</p>'
        '<p>'+ ' · '.join(link(path, label, output) for path,label in (
            (snapshot_path, '本页新版 checkpoint 快照'), (directory/'manifest.json', '新版 manifest'),
            (directory/'progress.json', '新版进度日志'), (directory/'summary.json', '新版最终 summary'),
            (directory/'audit.json', '新版审计'), (prefix_audit_path, '前 120 s 复现审计')) )+'</p></section>')
    return text


def comparison_300_html(comparison, output):
    if comparison is None:
        return '<section><h2>同初态 0–300 s 对照</h2><p>两边完整的 300 s 前缀落盘后才计算。</p></section>'
    old, new = comparison['old'], comparison['new']
    rows = []
    for name, a, b, unit in (
        ('面积覆盖', 100*old['coverage'], 100*new['coverage'], '%'),
        ('全手安全接触率', 100*old['metrics']['safe_contact_continuity_fraction'], 100*new['metrics']['safe_contact_continuity_fraction'], '%'),
        ('全手安全移动接触率', 100*old['metrics']['safe_moving_contact_fraction'], 100*new['metrics']['safe_moving_contact_fraction'], '%'),
        ('最长全手无接触间隔', old['loss']['longest_any_contact_gap_s'], new['loss']['longest_any_contact_gap_s'], 's')):
        rows.append(f'<tr><td>{name}</td><td>{a:.3f} {unit}</td><td>{b:.3f} {unit}</td></tr>')
    finger_rows, motion_rows = [], []
    for i, (a, b) in enumerate(zip(old['metrics']['per_finger'], new['metrics']['per_finger'])):
        finger_rows.append(f'<tr><td>{FINGER_NAMES[i]}</td><td>{100*a["contact_fraction"]:.2f}% → {100*b["contact_fraction"]:.2f}%</td>'
            f'<td>{100*a["safe_moving_contact_fraction"]:.2f}% → {100*b["safe_moving_contact_fraction"]:.2f}%</td>'
            f'<td>{a["longest_absence_s"]:.2f} → {b["longest_absence_s"]:.2f} s</td>'
            f'<td>{a["stable_safe_return_count"]} → {b["stable_safe_return_count"]}</td></tr>')
        qa, qb = a.get('measured_joint_motion', {}), b.get('measured_joint_motion', {})
        ranges = [' / '.join(f'{angle:.1f}' for angle in np.rad2deg(q.get('range_rad_by_joint', []))) for q in (qa,qb)]
        motion_rows.append(f'<tr><td>{FINGER_NAMES[i]}</td><td>{ranges[0]}</td><td>{ranges[1]}</td>'
            f'<td>{qa.get("observed_motion_while_off_contact_s", 0):.2f} → {qb.get("observed_motion_while_off_contact_s", 0):.2f} s</td></tr>')
    return ('<section><h2>同初态 0–300 s：实际接触与实际手指运动</h2>'
        '<p>初始 qpos、qvel、ctrl 逐元素完全一致；同一 300 s 窗口、同一 100 Hz 网格、同一指标算法。'
        '这里只比较冻结 run04 与本页指定新版长程目录，不使用更晚 checkpoint 的覆盖数值。</p>'
        '<table><tr><th>指标</th><th>旧 run04 / v2.2</th><th>指定新版长程运行</th></tr>'+''.join(rows)+'</table>'
        '<table><tr><th>手指</th><th>实际接触率（旧→新）</th><th>安全移动接触率（旧→新）</th>'
        '<th>最长失联（旧→新）</th><th>采样稳定返回次数（旧→新）</th></tr>'+''.join(finger_rows)+'</table>'
        '<table><tr><th>手指</th><th>旧实际四关节范围 / °</th><th>新实际四关节范围 / °</th>'
        '<th>离面期间实际运动时长（旧→新）</th></tr>'+''.join(motion_rows)+'</table>'
        '<p class="caption">关节范围是实际 q 的 max−min，包含随腕换姿和调形，不是主动搜索的因果证据。'
        '离面运动只说明失联时手指实际在动，不能当作接触恢复。移动接触来自实测接触点的切向运动并由实际指尖位移佐证。'
        '稳定返回计数排除起始失联后的首次接触，且受 10 ms 采样限制；不与原生执行器首次恢复计数混用。'
        '两版的规划与控制均不同，所有正负变化都保留，不归因于单个组件。</p>'
        f'<p>{link(output/"comparison_300s.json", "完整 300 s 指标、接触间隔及来源哈希", output)}</p></section>')


def schedule_comparison_html(comparison, output):
    if not comparison:
        return ''
    labels = list(comparison['cases'])
    cases = [comparison['cases'][label] for label in labels]
    header = '<tr><th>实测指标</th>'+''.join(f'<th>{escape(label)}</th>' for label in labels)+'</tr>'
    rows = []
    for name, getter, unit in (
        ('面积覆盖', lambda c: 100*c['coverage'], '%'),
        ('全手安全接触', lambda c: 100*c['metrics']['safe_contact_continuity_fraction'], '%'),
        ('安全移动接触', lambda c: 100*c['metrics']['safe_moving_contact_fraction'], '%'),
        ('最长全手失联', lambda c: c['loss']['longest_any_contact_gap_s'], 's'),
        ('最长同指连续滑动段', lambda c: c['metrics']['longest_sliding_bout_s'], 's')):
        rows.append('<tr><td>'+name+'</td>'+''.join(f'<td>{getter(c):.2f} {unit}</td>' for c in cases)+'</tr>')
    fingers = []
    for finger, name in enumerate(FINGER_NAMES):
        for metric, key, scale, unit in (
                ('实际接触', 'contact_fraction', 100., '%'),
                ('安全移动接触', 'safe_moving_contact_fraction', 100., '%'),
                ('最长失联', 'longest_absence_s', 1., 's'),
                ('离面时实际关节运动', 'observed_motion_while_off_contact_s', 1., 's')):
            values = []
            for c in cases:
                item = c['metrics']['per_finger'][finger]
                value = item['measured_joint_motion'][key] if key=='observed_motion_while_off_contact_s' else item[key]
                values.append(f'<td>{scale*value:.2f} {unit}</td>')
            fingers.append(f'<tr><td>{name} · {metric}</td>'+''.join(values)+'</tr>')
    return ('<section><h2>连续滑动优先：同初态、同 0–300 s 调度比较</h2>'
        '<p>每案都核对相同初始 qpos、qvel、ctrl，并用同一 100 Hz 轨迹及同一指标计算。'
        '“滑动优先”保留正在安全前进的滑动段，完整调形仅在局部搜索阶段请求；有界微寻触仍可随滑动进行。</p>'
        '<table>'+header+''.join(rows)+'</table>'
        '<p class="warning">滑动优先版的全手连续性明显改善，但面积覆盖略慢；各指仍存在长失联。'
        '其 300 s 内完整恢复请求为 0，不能把它解释为每根手指都已能主动找回接触。'
        '全手持续滑动与失联手指恢复必须分别验证。</p>'
        '<details><summary>查看四指实际接触、移动和最长失联</summary><table>'+header+''.join(fingers)+'</table></details>'
        '<p class="caption">关节运动来自实际 q；离面运动不能当作恢复成功。安全移动接触用接触点切向运动并结合实际指尖位移，'
        '仍不能区分滚动与滑移。300 s 是开发窗口，没有任何一版因此获得整瓶完成结论。'
        '新上下文版只在完整 300 s 记录与最终 summary 落盘后加入本表，原失联历史和累计次数保留。</p>'
        f'<p>{link(output/"comparison_schedule_300s.json", "全部逐指指标、相同初态核验与源哈希", output)}</p></section>')


def final_remaining_surface(record, output):
    """Freeze audited final coverage; the rolling checkpoint is never substituted."""
    directory, output = Path(record['directory']), Path(output)
    if not record['summary'] or not record['audit_verified'] or not (directory/'coverage_independent.npz').exists():
        return None
    frozen = output/'old_v22_final_snapshot'; frozen.mkdir(exist_ok=True)
    sources = {}
    for name in ('coverage_independent.npz', 'surface_graph.npz', 'final_state.npz', 'summary.json', 'audit.json', 'events.json'):
        raw = (directory/name).read_bytes(); (frozen/name).write_bytes(raw)
        sources[name] = dict(source=str((directory/name).resolve()), sha256=sha256(raw).hexdigest())
    with np.load(frozen/'surface_graph.npz', allow_pickle=False) as saved:
        vertices, faces, required = saved['vertices'], saved['faces'], saved['required_vertices']
    with np.load(frozen/'coverage_independent.npz', allow_pickle=False) as saved:
        covered = saved['covered_vertices']
    with np.load(directory/'trace.npz', allow_pickle=False) as saved:
        last_trace = float(saved['time_s'][-1]); recorded_mask = saved['covered_vertices']
    with np.load(frozen/'final_state.npz', allow_pickle=False) as saved:
        native_final = float(saved['time'])
    summary = record['summary']
    if (not np.array_equal(covered, recorded_mask) or int(covered.sum()) != summary['covered_vertex_count']
            or int(covered[required].sum()) != summary['visited_required_goals']):
        raise ValueError('Independent final coverage, final trace and summary disagree')
    ids = np.flatnonzero(~covered); remaining_goals = required[~covered[required]]
    triangles = vertices[faces]
    areas = .5*np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)
    vertex_area = np.zeros(len(vertices)); np.add.at(vertex_area, faces.ravel(), np.repeat(areas/3.,3))
    adjacency = {int(i): set() for i in ids}
    for face in faces:
        members = [int(i) for i in face if int(i) in adjacency]
        for i in members: adjacency[i].update(j for j in members if j != i)
    components, unseen = [], set(adjacency)
    while unseen:
        stack, component = [min(unseen)], []
        while stack:
            i=stack.pop()
            if i not in unseen: continue
            unseen.remove(i); component.append(i); stack.extend(adjacency[i] & unseen)
        components.append(sorted(component))
    result = dict(scope='Audited final v2.2 coverage only; no coverage credited to v3.2', sources=sources,
        trace=record['trace_file'], final_relative_time_s=summary['executed_s'], final_native_time_s=native_final,
        last_coverage_trace_time_s=last_trace, terminal_after_last_trace_s=summary['executed_s']-last_trace,
        final_checkpoint_used=False, remaining_vertex_ids=ids.tolist(), remaining_goal_vertex_ids=remaining_goals.tolist(),
        remaining_positions_m=vertices[ids].tolist(), original_full_vertices=len(vertices), original_full_goals=len(required),
        remaining_area_m2=float(vertex_area[ids].sum()), remaining_area_fraction=float(vertex_area[ids].sum()/vertex_area.sum()),
        remaining_induced_mesh_components=components,
        independent_mask_equals_final_trace=True, counts_equal_final_summary=True)
    (output/'old_v22_final_remaining.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig = plt.figure(figsize=(13,6), constrained_layout=True)
    for panel, angle in enumerate((35.,215.)):
        ax=fig.add_subplot(1,2,panel+1,projection='3d')
        ax.plot_trisurf(vertices[:,0],vertices[:,1],vertices[:,2],triangles=faces,
            color='#c3cbd0',alpha=.26,linewidth=0,antialiased=False,shade=False,rasterized=True)
        ax.scatter(*vertices[ids].T,c='#d32f2f',s=42,depthshade=False,label=f'Unvisited vertices: {len(ids)}')
        ax.scatter(*vertices[remaining_goals].T,c='#7027a0',s=130,marker='*',depthshade=False,
                   label=f'Unvisited required goals: {len(remaining_goals)}')
        ax.set_box_aspect(np.ptp(vertices,axis=0)); ax.view_init(elev=24,azim=angle)
        ax.set(xlabel='Task X (m)',ylabel='Task Y (m)',zlabel='Task Z (m)',title=f'Original mesh, view {panel+1}')
        ax.legend(loc='upper left',fontsize=8)
    fig.suptitle(f'Frozen v2.2 final: {summary["covered_vertex_count"]}/{len(vertices)} vertices, '
        f'{summary["visited_required_goals"]}/{len(required)} goals; full surface incomplete',fontsize=14)
    fig.savefig(output/'old_v22_final_remaining.png',dpi=160); plt.close(fig)
    return result


def generate(output, *, old_run=None, new_run=None, middle_pair=None, index_pair=None, combo_pair=None, smoke=None,
             priority_run=None, context_run=None, boundary_run=None, terminal_chain=None):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    old_run = old_run or GENERATED/'mustard_active_sliding_run04_20261016/B-MCC_seed7'
    new_run = new_run or GENERATED/'mustard_coordinated_run02_20261016/B-MCC_seed7'
    middle_pair = middle_pair or GENERATED/'i04_posture_recovery_v2_seed7_20261016/middle_t205_reconfiguration'
    index_pair = index_pair or GENERATED/'i04_index_recovery_paired_seed7_20261016'
    combo_pair = combo_pair or GENERATED/'i04_index_combo_v4_seed7_20261016'
    smoke = smoke or GENERATED/'mustard_coordinated_smoke03_20261016/B-MCC_seed7'
    priority_run = priority_run or GENERATED/'mustard_coordinated_slide_priority_20261016/B-MCC_seed7'
    context_run = context_run or GENERATED/'mustard_coordinated_context_20261016/B-MCC_seed7'
    boundary_run = boundary_run or GENERATED/'mustard_coordinated_boundary_20261016/B-MCC_seed7'
    terminal_chain = terminal_chain or GENERATED/'run04_terminal_completion_chain_20261016'
    old = old_checkpoint(old_run, output/'frozen_v22_checkpoint_snapshot.npz')
    terminal_completion = terminal_chain_evidence(terminal_chain, old)
    boundary = integrated_run(boundary_run)
    boundary_recovery = actual_recovery_episodes(boundary_run)
    if boundary_recovery:
        (output/'boundary_actual_recovery_followup.json').write_text(json.dumps(boundary_recovery,ensure_ascii=False,indent=2),encoding='utf-8')
    new_long = old_checkpoint(new_run, output/'frozen_new_long_run_checkpoint_snapshot.npz',
                             version_scope='Selected new long run only; independent coverage and frozen sources')
    old_remaining = final_remaining_surface(old, output)
    old_remaining_html = ''
    if old_remaining:
        old_remaining_html = (f'<img class="figure" src="{relative_url(output/"old_v22_final_remaining.png",output)}" alt="旧v2.2最终剩余表面点和必访点"/>'
            f'<p>最终仍有 {len(old_remaining["remaining_vertex_ids"])} 个表面点、'
            f'{len(old_remaining["remaining_goal_vertex_ids"])} 个必访点未覆盖；对应面积 '
            f'{100*old_remaining["remaining_area_fraction"]:.4f}%。图用审计重建的最终覆盖，未使用更早的滚动 checkpoint。</p>'
            f'<p class="caption">最后覆盖采样 {old_remaining["last_coverage_trace_time_s"]:.3f} s，实际终态 '
            f'{old_remaining["final_relative_time_s"]:.3f} s，相差 '
            f'{1000*old_remaining["terminal_after_last_trace_s"]:.1f} ms；独立重建 mask、末 trace mask 与 summary 计数一致。'
            f'{link(output/"old_v22_final_remaining.json", "剩余点坐标、连通分量与冻结源哈希",output)}</p>')
    middle = probe_pair(middle_pair, '中指：远离表面后的大幅调形')
    index = probe_pair(index_pair, '食指：约 2.4 mm 撤离后的局部寻触')
    combo = probe_pair(combo_pair, '食指：腕调整后再调形') if combo_pair else None
    integrated = integrated_run(smoke) if smoke else None
    comparison = (same_prefix_comparison(smoke, old_run) if integrated and integrated['summary']
                  and integrated['summary'].get('executed_s', 0.) >= 120. else None)
    prefix_audit_path = Path(new_run)/'reference_prefix_audit.json'
    prefix_audit = read_json(prefix_audit_path)
    if comparison:
        (output/'comparison_120s.json').write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding='utf-8')
    comparison300 = None
    if all((record['checkpoint'] or {}).get('time_s', 0.) >= 300. or
           (record['summary'] or {}).get('executed_s', 0.) >= 300. for record in (new_long, old)):
        comparison300 = same_prefix_comparison(new_run, old_run, 300.)
        (output/'comparison_300s.json').write_text(json.dumps(comparison300, ensure_ascii=False, indent=2), encoding='utf-8')
    schedule_comparison = None
    v32_run = GENERATED/'mustard_coordinated_run02_20261016/B-MCC_seed7'
    schedule_base = (comparison300 if Path(new_run).resolve()==v32_run.resolve() else
                     same_prefix_comparison(v32_run, old_run, 300.) if (v32_run/'summary.json').exists() else None)
    if schedule_base:
        schedule_comparison = dict(horizon_s=300., method=schedule_base['method'],
            cases={'旧 v2.2': schedule_base['old'], '恢复整合 v3.2': schedule_base['new']},
            initial_checks={'恢复整合 v3.2': schedule_base['initial_state']},
            source_hashes={'恢复整合 v3.2': schedule_base['sources']}, audits={})
        for label, directory, filename in (
                ('滑动优先 v3.3', priority_run, 'comparison_priority_300s.json'),
                ('滑动优先＋新上下文 v3.4', context_run, 'comparison_context_300s.json'),
                ('自然端点短调形 v3.5', boundary_run, 'comparison_boundary_300s.json')):
            summary = read_json(Path(directory)/'summary.json')
            if not summary or summary.get('executed_s', 0.) < 300.:
                continue
            paired = same_prefix_comparison(directory, old_run, 300.)
            (output/filename).write_text(json.dumps(paired, ensure_ascii=False, indent=2), encoding='utf-8')
            schedule_comparison['cases'][label] = paired['new']
            schedule_comparison['initial_checks'][label] = paired['initial_state']
            schedule_comparison['source_hashes'][label] = paired['sources']
            schedule_comparison['audits'][label] = completed_run_evidence(directory, summary)
        (output/'comparison_schedule_300s.json').write_text(json.dumps(schedule_comparison, ensure_ascii=False, indent=2), encoding='utf-8')
    analysis = read_json(Path(middle_pair)/'middle_independent_analysis.json') or {}
    combo_analysis = read_json(Path(combo_pair)/'ON/stable_contact_analysis.json') if combo_pair else None
    figure = middle_figure(middle, output)
    checkpoint = old['checkpoint']
    graph_stats = (f'<b>{100*checkpoint["measured_coverage_fraction"]:.2f}%</b> 面积；'
        f'<b>{checkpoint["covered_vertices"]}/{checkpoint["full_vertex_count"]}</b> 表面点；'
        f'<b>{checkpoint["visited_required_goals"]}/{checkpoint["full_required_goal_count"]}</b> 必访点'
        if checkpoint else '尚无完整 checkpoint')
    checkpoint_time = number(checkpoint['time_s'], 1) if checkpoint else '未知'
    progress = old['progress'] or {}
    protocol = (old['manifest'] or {}).get('protocol', '未记录')
    changed = analysis.get('selected_actual_joint_change_initial_to_first_deg')
    angles = ' / '.join(f'{v:.1f}°' for v in changed) if changed else '待分析'
    figures_html = (f'<img class="figure" src="{relative_url(figure, output)}" alt="中指四关节实测角度和实测接触力"/>'
                    if figure else '')
    combo_html = '<p class="muted">新食指组合实验可独立接入；未提供完整 summary 与审计前不声明成功。</p>'
    if combo:
        combo_html = (f'<p>本记录协议：<code>{escape(str((combo["cases"]["ON"]["summary"] or {}).get("protocol", "进行中")))}</code>。'
            '从 run04 的 t=240 s 状态单独复现；腕调整后从实际末态重规划手指。该局部复现不增加整瓶覆盖。</p>'
            +probe_table(combo, output))
        if combo['paired_verified'] and not combo['success']:
            combo_html += '<p class="warning">这一案仍未达到连续 150 ms 的同指安全接触。首次碰到物体不能作为恢复成功。</p>'
        if combo['success']:
            combo_html += ('<p>本案同时使用持续失联预算的 ActiveFingerSearchV3 与保持接触预载的 ContactPreservingMCC，'
                '是完整恢复策略的验证，不能把成功单独归于更换求解器。</p>'
                +video(Path(combo_pair)/'ON', 'index_combo_1x_uncut.mp4', output, combo['paired_verified'])
                +'<p class="caption">录像中的 Path progress 对应第一阶段腕路径；随后实际手指调形和寻触继续显示在关节角度、力与画面中。</p>')
            if combo_analysis:
                a = combo_analysis['after_15_11_s']
                combo_html += (f'<p>15.110–35 s 的同指安全接触占比为 <b>{100*a["safe_sample_fraction"]:.2f}%</b>，'
                    f'最长连续安全接触 <b>{a["longest_continuous_safe_state_span_s"]:.3f} s</b>，其后仍有周期性短间断。'
                    '本案达到稳定恢复判据，但仍应继续改善接触连续性；无名指锚点在这 35 s 全部原生采样中保持安全接触。</p>'
                    f'<p>{link(Path(combo_pair)/"ON/stable_contact_analysis.json", "恢复后短间断与实际关节变化", output)}</p>')
    integrated_html = '<p class="muted">新整合运行尚未指定。</p>'
    if integrated:
        s, p, m = integrated['summary'], integrated['progress'] or {}, integrated['manifest'] or {}
        integrated_html = f'<p>协议 <code>{escape(str(m.get("protocol", "未知")))}</code>。</p>'
        if s:
            integrated_html += (f'<p>已执行 {number(s.get("executed_s"), 1)} s；停止原因 '
                f'<code>{escape(str(s.get("stop_reason")))}</code>；实测面积 {100*s.get("final_covered_area_fraction", 0):.2f}%；'
                f'表面点 {s.get("covered_vertex_count")}/{s.get("vertex_count")}，必访点 '
                f'{s.get("visited_required_goals")}/{s.get("required_goal_count")}；抬离 {s.get("lift_count")} 次；'
                f'原生峰值力 {number(s.get("peak_force_n"))} N。日志记录稳定恢复 {integrated["actual_stable_recovery_count"]} 次。'
                +('运行证据审计与 trace 哈希已匹配。' if integrated['audit_verified'] else '运行证据审计或 trace 哈希匹配仍待确认。')+'</p>')
            if integrated['audit_verified']:
                audit = integrated['audit']
                integrated_html += (f'<p class="caption">该运行审计检查了 {audit.get("fk_checked_states", "—")} 个 FK 状态、'
                    f'{audit.get("sensor_reconstructed_states", "—")} 个传感状态，以及 '
                    f'{audit.get("native_force_event_states_checked", "—")} 个原生峰值事件；'
                    '这与局部配对试验逐 2 ms 全程重放的审计范围不同。</p>')
        else:
            integrated_html += (f'<p>进行中：最近落盘进度 {number(p.get("time_s"), 1)} s，'
                f'面积 {100*p.get("coverage", 0):.2f}%，阶段 {escape(str(p.get("phase", "未知")))}。'
                '尚无最终 summary，不能评价完整运行成功。</p>')
        integrated_html += '<p>'+ ' · '.join(link(Path(smoke)/name, name, output) for name in
            ('manifest.json', 'summary.json', 'audit.json', 'force_event_audit.json', 'events.json'))+'</p>'
        scan_video = ('coordinated_scan_0_40s_clear_1x.mp4'
            if (Path(smoke)/'coordinated_scan_0_40s_clear_1x_render.json').exists()
            else 'coordinated_scan_0_40s_1x.mp4')
        integrated_html += video(Path(smoke), scan_video, output, integrated['audit_verified'])
        integrated_html += '<p class="caption">这是新整合扫描自身的 0–40 s 连续片段；其中自然失联后的恢复与专项源状态复现分开记录。该录像来源为 10 ms 保存轨迹。</p>'
        integrated_html += ('<p class="caption">FINGER_RECOVERY 阶段显示局部搜索 DISABLED 时，仍可能正在执行完整的手指关节调形路径。'
            'DISABLED 只指局部微寻触层；是否实际运动请看画面、实测关节角度和 Δ 角度，不用 SEEK 标记替代实际 q。</p>')
        if integrated['recovery_ends']:
            rows = ''.join(f'<tr><td>{number(e.get("time_s"))}</td><td>{e.get("sequence", "—")}</td>'
                f'<td>{FINGER_NAMES[e["selected_finger"]] if "selected_finger" in e else "—"}</td>'
                f'<td>{escape(str(e.get("status")))}</td><td>{escape(str(e.get("reason")))}</td></tr>'
                for e in integrated['recovery_ends'])
            integrated_html += '<table><tr><th>时间/s</th><th>尝试</th><th>手指</th><th>状态</th><th>结束原因</th></tr>'+rows+'</table>'
    comparison_html = ''
    if comparison:
        old_c, new_c = comparison['old'], comparison['new']
        metrics_rows = [
            ('实测面积覆盖', 100*old_c['coverage'], 100*new_c['coverage'], '%'),
            ('全手安全接触率', 100*old_c['metrics']['safe_contact_continuity_fraction'], 100*new_c['metrics']['safe_contact_continuity_fraction'], '%'),
            ('安全移动接触率', 100*old_c['metrics']['safe_moving_contact_fraction'], 100*new_c['metrics']['safe_moving_contact_fraction'], '%'),
            ('最长全手无接触间隔', old_c['loss']['longest_any_contact_gap_s'], new_c['loss']['longest_any_contact_gap_s'], 's')]
        rows = ''.join(f'<tr><td>{name}</td><td>{a:.2f} {unit}</td><td>{b:.2f} {unit}</td></tr>' for name,a,b,unit in metrics_rows)
        finger_rows = ''.join(f'<tr><td>{FINGER_NAMES[i]}</td><td>{100*a["contact_fraction"]:.2f}% → {100*b["contact_fraction"]:.2f}%</td>'
            f'<td>{a["longest_absence_s"]:.2f} → {b["longest_absence_s"]:.2f} s</td>'
            f'<td>{a["stable_safe_return_count"]} → {b["stable_safe_return_count"]}</td></tr>'
            for i,(a,b) in enumerate(zip(old_c['metrics']['per_finger'], new_c['metrics']['per_finger'])))
        comparison_html = ('<section><h2>同初态、同 0–120 s：改善与退步</h2>'
            '<p>initial 中 qpos、qvel、ctrl 已逐元素完全一致。两方都使用同一 100 Hz 记录和同一指标算法，全部结果保留。</p>'
            '<table><tr><th>指标</th><th>冻结 run04 / v2.2</th><th>新 smoke03 / v3.2</th></tr>'+rows+'</table>'
            '<table><tr><th>手指</th><th>实际接触率（旧 → 新）</th><th>最长失联（旧 → 新）</th><th>采样稳定返回次数（旧 → 新）</th></tr>'+finger_rows+'</table>'
            '<p class="warning">全手存在接触不能代替每指参与。这个前缀中，食指和中指参与改善，'
            '无名指与拇指参与退步，整体覆盖和安全移动接触比例也下降，仍需要继续解决。</p>'
            '<p class="caption">稳定返回次数：离开后再次出现至少 150 ms 的同指安全接触采样段，排除起始失联段；'
            '100 Hz 记录可能遗漏更短间断，这个数不替代 2 ms 原生恢复判定。移动接触结合表面切向位移与实际指尖运动，'
            '仍不能区分滑移与滚动。上述变化涉及规划、控制和恢复策略，不能作单因素因果解释。</p>'
            '<p>具体到拇指：执行器在 15.584 s 确认了这次初始失联后的首次稳定接触；'
            '采样中首次接触约为 15.43 s，但约 17.85 s 再次失联后直到 120 s 没有回来。'
            '表里的“0 次返回”排除了起始失联的首次触达，不表示执行器那次恢复没有发生；问题是恢复后未能持续参与。</p>'
            f'<p>{link(output/"comparison_120s.json", "同前缀完整指标、每指间隔与初态核验", output)} · '
            f'{link(prefix_audit_path, "新长程运行前 120 s 精确复现审计", output)}</p></section>')
        if prefix_audit and prefix_audit.get('status') == 'PASSED' and not prefix_audit.get('mismatches'):
            comparison_html = comparison_html.replace('</section>',
                f'<p>另一次新长程运行的前 {prefix_audit["duration_s"]:.0f} s，在 '
                f'{prefix_audit["sampled_native_states_compared"]} 行、{len(prefix_audit["checked_fields"])} 个保存字段上精确复现 smoke03，'
                '并核对了相同源哈希。这只证明相同前缀可复现，不代表后续整瓶完成。</p></section>')
    generated_at = datetime.now(timezone.utc).isoformat()
    payload = dict(schema='i04.coordinated_recovery_review.v2', generated_at_utc=generated_at,
        old_frozen_run=old, old_final_remaining=old_remaining, new_long_run=new_long,
        middle_pair=middle, index_pair=index, combo_pair=combo, integrated_run=integrated,
        middle_analysis=analysis, combo_analysis=combo_analysis, comparison_120s=comparison, comparison_300s=comparison300,
        schedule_comparison_300s=schedule_comparison,
        boundary_run=boundary, boundary_actual_recovery=boundary_recovery, old_terminal_continuation=terminal_completion,
        new_run_reference_prefix_audit=prefix_audit, new_run_reference_prefix_audit_file=evidence_file(prefix_audit_path),
        coverage_credit_from_probes=False,
        new_controller_full_surface_claim=new_long['full_surface_success'])
    (output/'evidence.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    html = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mustard：实际手指运动与主动恢复</title><style>
:root{{--ink:#182c3b;--muted:#536575;--line:#d7e0e6;--green:#167a67;--bg:#f3f6f7}}*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);font:17px/1.7 "Microsoft YaHei",system-ui,sans-serif}}
main{{max-width:1230px;margin:auto;padding:36px 26px 70px}}h1{{font-size:36px;line-height:1.35;margin:10px 0 20px}}
h2{{font-size:25px;line-height:1.45;margin:0 0 16px}}p{{margin:12px 0}}section{{background:white;padding:26px;margin-top:24px;border:1px solid var(--line);border-radius:14px}}
.eyebrow{{color:var(--green);font-size:14px;font-weight:700;letter-spacing:1px}}.lead{{font-size:21px;max-width:1050px}}
.muted,.caption{{color:var(--muted)}}.caption{{font-size:14px}}.warning{{padding:12px 16px;border-left:4px solid #bf7a36;background:#fff6e9}}
video,.figure{{display:block;width:100%;max-height:760px;background:#10151b;border-radius:8px}}.figure{{background:white}}
table{{border-collapse:collapse;width:100%;font-size:15px;margin:18px 0}}th,td{{padding:11px 10px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}}th{{background:#edf3f4}}
code{{font-family:Consolas,monospace;font-size:13px;overflow-wrap:anywhere}}a{{color:#176d9b;text-underline-offset:3px}}b{{color:#105f52}}
.steps{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}.step{{padding:14px;background:#edf5f3;border-radius:8px;font-size:15px}}
@media(max-width:760px){{main{{padding:20px 12px}}h1{{font-size:28px}}section{{padding:18px}}.steps{{grid-template-columns:1fr 1fr}}table{{font-size:13px}}th,td{{padding:8px 5px}}}}
</style><main><div class="eyebrow">MCC ONLY · MUSTARD · 实测恢复证据</div>
<h1>手指如何主动动起来，失联后如何找回接触</h1>
<p class="lead">已经观察到两种实际恢复：小距离撤离后主动寻触，以及中指大幅弯曲调形后重新接触。先看实际关节运动和接触力，再看新整合运行与仍未解决的失败案。</p>
<p class="caption">生成时间 {escape(generated_at)} · {link(output/'evidence.json', '本页数据与源文件哈希', output)}</p>
{boundary_run_html(boundary, output, boundary_recovery, schedule_comparison)}
<section><h2>1. 中指不再只随腕移动：四个关节实际调形</h2>
<p>从冻结 run04 的 <b>t=205 s</b> 状态单独复现。首次重触前，中指四关节实测变化为 <b>{angles}</b>。
腕参考保持源状态；其它手指保留接触预载。该实验只验证一次局部恢复，时轴是重新启动的 0–25 s。</p>
{video(Path(middle_pair)/'ON', 'posture_recovery_1x_uncut.mp4', output, middle['paired_verified'])}
{probe_table(middle, output)}
<p>实际路径在 <b>{number(analysis.get('path_complete_s'))} s</b> 走完，之后仍有
<b>{number(analysis.get('path_to_first_contact_s'))} s</b> 末端局部寻触才出现实测接触。
首触立即停下原路径，再交给 MCC；稳定确认需要同指连续 150 ms 安全接触。</p>
<p class="warning">拇指支撑存在反复短间断，最长按区间计约 44 ms；它不是连续稳定锚点。
中指重触后则连续保持到 25 s 末。视频采样不能展示每一次 2 ms 接触间断。</p>
{figures_html}<p class="caption">图中角度来自物理关节 q，实线力来自接触传感；规划目标、搜索指令和 FK 到达都不能替代实测接触。
{link(Path(middle_pair)/'middle_independent_analysis.json', '路径结束、首触与预载交接的独立分析', output)}</p></section>
<section><h2>2. 小范围失联：只撤离食指，主动搜索再贴回</h2>
<p>只对选定食指施加可控撤离；搜索开始时实际离开约 2.4 mm，恢复期间腕参考不变。
ON/OFF 使用相同初态；OFF 在相同 5 s 恢复观察窗口没有重触。</p>
{video(Path(index_pair)/'ON', 'index_recovery_1x_uncut.mp4', output, index['paired_verified'])}
{probe_table(index, output)}<p class="caption">表中延迟从恢复阶段开始计时；录像还包含前面的撤离阶段。</p></section>
<section><h2>3. 新整合策略：先局部寻触，必要时调形和调整腕部</h2>
<div class="steps"><div class="step"><b>① 真实失联计时</b><br>保留每指失联历史，不因更换滑动段而归零。</div>
<div class="step"><b>② 有界局部搜索</b><br>按实际关节与指腹位置评估进展，遇到停滞再请求调形。</div>
<div class="step"><b>③ 安全调形 / 腕调整</b><br>先固定腕调指；必要时用安全锚点小幅调整腕，再从实际末态求解。</div>
<div class="step"><b>④ 实测重触交接</b><br>首触停路径并保持预载，连续 150 ms 后才算恢复。</div></div>
<p>腕调整后必须实际到达规划末端并保持安全锚指接触，才允许第二阶段求解。
一次恢复最多一个腕阶段和一个指阶段，共用原失联预算；超过 60 s 结束尝试，恢复时间也计入 45 s 无新增覆盖预算。</p>
{integrated_html}<p class="caption">此处是新整合控制器的独立记录。小试验成功与整瓶滑动覆盖属于不同验证问题；尚未完成的整合运行不声明整瓶成功。</p></section>
{long_run_html(new_long, output/'frozen_new_long_run_checkpoint_snapshot.npz', output, prefix_audit_path)}
{comparison_html}
{comparison_300_html(comparison300, output)}
{schedule_comparison_html(schedule_comparison, output)}
<section><h2>4. 食指组合恢复与失败记录</h2>{combo_html}
<p>固定腕最近表面点无解时，候选规划器检查同一原网格 40 mm 邻域的有限分散点，仍保留动态指腹支撑、原关节限位和严格路径碰撞检查。
静态有解只表示找到候选路径；它不证明物理执行后会产生稳定接触。</p>
<p>{link(GENERATED/'multi_target_finger_validation_20261016/validation.json', '固定腕与邻域目标静态验证', output)} ·
{link(GENERATED/'i04_index_combo_seed7_20261016/ON/summary.json', '早期组合案：发生重触但未稳定', output)} ·
{link(GENERATED/'i04_index_combo_seed7_20261016/paired_audit.json', '早期失败案原生审计', output)}</p></section>
<section><h2>5. 原整瓶实验状态：冻结 v2.2，单独记账</h2>
{completion_html(old)}
<p>同一落盘 checkpoint：<b>{checkpoint_time} s</b>，{graph_stats}。</p>
<p>最新进度日志另记为 {number(progress.get('time_s'), 1)} s、面积 {100*progress.get('coverage', 0):.2f}%。
日志与 checkpoint 的时间可能不同，上面的点数和面积只使用同一 checkpoint，未混合未来记录。</p>
<p class="warning">这是旧版本 <code>{escape(str(protocol))}</code> 的独立记录，不属于新 v3.2 的验证结果。
完整分母保留全部 8192 个表面点和 388 个必访点，没有删去难以到达的区域。</p>
<p class="caption">面积按原三角网格权重计算。覆盖沿用实测接触周围 25 mm 测地邻域账本，并不意味着指腹直接触碰了每个顶点；全手有接触也不意味着四指都在参与。
{link(Path(old_run)/'manifest.json', '旧运行 manifest', output)} · {link(Path(old_run)/'progress.json', '旧运行进度', output)} ·
{link(Path(old_run)/'summary.json', '旧运行最终 summary', output)} · {link(Path(old_run)/'audit.json', '旧运行审计', output)} ·
{link(output/'frozen_v22_checkpoint_snapshot.npz', '本页读取的旧覆盖 checkpoint 快照', output)}</p></section>
{('<section><h2>旧 v2.2 最终剩余表面</h2>'+old_remaining_html+'</section>') if old_remaining_html else ''}
{terminal_chain_html(terminal_completion, output)}
<section><h2>证据边界</h2><p>报告展示实际关节运动和实测接触；SEEK 表示提议的搜索动作，可能受限速、力反馈或几何保护影响。
短时失联允许恢复，只有持续稳定的实测重触才计恢复成功。接触点在表面移动仍不能单独区分纯滑移与滚动。</p>
<p>恢复、规划和控制同时发生变化，整合运行不构成单一因素消融。已有 ON/OFF 小试验的因果结论仅适用于各自明确的干预。
邻域求解曾需要约 20–23 s CPU 时间；当前仿真规划会暂停虚拟时间，因此不声明实时运行能力。</p></section>
</main></html>'''
    (output/'index.html').write_text(html, encoding='utf-8')
    return payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=GENERATED/'mustard_coordinated_review_20261016')
    parser.add_argument('--old-run', type=Path)
    parser.add_argument('--new-run', type=Path, default=GENERATED/'mustard_coordinated_run02_20261016/B-MCC_seed7')
    parser.add_argument('--middle-pair', type=Path)
    parser.add_argument('--index-pair', type=Path)
    parser.add_argument('--combo-pair', type=Path)
    parser.add_argument('--smoke', type=Path)
    parser.add_argument('--priority-run', type=Path)
    parser.add_argument('--context-run', type=Path)
    parser.add_argument('--boundary-run', type=Path)
    parser.add_argument('--terminal-chain', type=Path)
    args = parser.parse_args()
    evidence = generate(args.output, old_run=args.old_run, new_run=args.new_run, middle_pair=args.middle_pair,
        index_pair=args.index_pair, combo_pair=args.combo_pair, smoke=args.smoke,
        priority_run=args.priority_run, context_run=args.context_run,
        boundary_run=args.boundary_run, terminal_chain=args.terminal_chain)
    print(json.dumps(dict(report=str((args.output/'index.html').resolve()),
        middle_verified=evidence['middle_pair']['paired_verified'],
        index_verified=evidence['index_pair']['paired_verified'],
        new_controller_full_surface_claim=evidence['new_controller_full_surface_claim']), ensure_ascii=False))
