"""Build the audited A / immutable B comparison and planning ledgers."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from hashlib import sha256
import html
import json
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET

import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .a_frozen_b_mcc import BASELINE
from .ab_frozen_b_cost_review import event_rows, save_tables
from .ab_frozen_b_planning import plain
from .b_full_portable import verify_download, register
from .b_full_video import digest, read
from .matched_mcc_review import native_arrays, contact_summary
from .natural_belly_report import coverage_figure, contact_path_figure
from .published_whole_hand import ROOT, install_asset_adapter, asset_map
from .whole_hand_surface_review import load_verified


def write(path, value):
    Path(path).write_text(json.dumps(plain(value), ensure_ascii=False, indent=2), encoding='utf-8')


def stats(values):
    values = np.asarray(values, dtype=float)
    return dict(n=len(values), median=float(np.median(values)), mean=float(values.mean()),
                p95=float(np.percentile(values, 95)), maximum=float(values.max()), total=float(values.sum())) if len(values) else dict(n=0)


def original_b_costs(b, out):
    rows = event_rows(b)
    selected = [row for row in rows if row['selected_ranking_cost'] is not None]
    figure, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    ax.scatter([row['plan_index'] for row in selected], [row['planned_motion_s'] for row in selected],
               label='Planned transfer motion (s)', s=18)
    ax.scatter([row['plan_index'] for row in selected], [row['selected_ranking_cost'] for row in selected],
               label='Selected transfer ranking cost', marker='x', s=24)
    ax.set_xlabel('Recorded B plan index'); ax.set_ylabel('Motion seconds / ranking score')
    ax.set_title('Original B evidence: these values are NOT computation time')
    ax.legend(); ax.grid(alpha=.2)
    figure.savefig(out/'b_original_selected_costs.png', dpi=150); plt.close(figure)
    events = read(b/'events.json')
    failures = [dict(event_index=i, **row) for i, row in enumerate(events)
                if row['kind'] in ('NO_COOPERATIVE_TRANSFER', 'COMPLETION_SEARCH_RETRY')]
    result = dict(recorded_plan_actions=len(rows), selected_transfer_scores=len(selected),
                  selected_transfer_score=stats([row['selected_ranking_cost'] for row in selected]),
                  planned_transfer_motion_s=stats([row['planned_motion_s'] for row in selected]),
                  logged_search_exhaustion_events=failures,
                  unlogged_rejected_candidates='Unknown; do not infer zero',
                  original_per_call_wall_s=None, original_per_call_optimizer_cost=None)
    write(out/'b_original_cost_summary.json', result)
    return result


def transfer_timing(a, b, out):
    rows = []
    result = {}
    for method, case in (('A', a), ('B', b)):
        selected = None
        events = read(case/'events.json')
        method_rows = []
        for index, event in enumerate(events):
            if event['kind'] in ('NECESSARY_COOPERATIVE_REORIENTATION', 'COMPLETION_COOPERATIVE_REORIENTATION'):
                selected = (index, event)
            if event['kind'] not in ('TRANSFER_SEGMENT_FINISHED', 'TRANSFER_SEGMENT_REPLAN'):
                continue
            assert selected is not None
            row = dict(method=method, event_index=index, selected_plan_event=selected[0],
                goal=selected[1]['goal'], selected_plan_motion_s=selected[1]['duration_s'],
                segment_kind=event['kind'], phase=event['phase'], end_s=event['time_s'],
                actual_segment_elapsed_s=event['elapsed_s'], actual_force_pause_s=event['force_pause_s'],
                planned_segment_s=event.get('planned_s'), achieved_path_parameter_s=event.get('progress_s'))
            rows.append(row); method_rows.append(row)
        elapsed = sum(row['actual_segment_elapsed_s'] for row in method_rows)
        pause = sum(row['actual_force_pause_s'] for row in method_rows)
        result[method] = dict(recorded_segments=len(method_rows), recorded_elapsed_s=elapsed,
            recorded_force_pause_s=pause, recorded_pause_fraction=pause/elapsed if elapsed else None,
            completed_segments=sum(row['segment_kind'] == 'TRANSFER_SEGMENT_FINISHED' for row in method_rows),
            replan_segments=sum(row['segment_kind'] == 'TRANSFER_SEGMENT_REPLAN' for row in method_rows),
            force_pause_s=stats([row['actual_force_pause_s'] for row in method_rows]),
            scope='Logged completed or replan transfer segments only; excludes approach/recontact and any terminal segment without a result event')
    with (out/'transfer_segment_times.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write(out/'transfer_time_summary.json', result)
    return result


def verify_alignment(a, b):
    am, bm = read(a/'manifest.json'), read(b/'manifest.json')
    fields = ('geometry_quality', 'belly_contact', 'coverage_radius_m', 'required_vertices', 'required_goals',
              'posture_lower_rad', 'posture_upper_rad', 'dt_s', 'proactive_release_n', 'release_resume_n',
              'force_relief_n', 'hard_force_n', 'hard_force_duration_s', 'requested_duration_s',
              'desired_force_per_finger_n', 'controlled_fingers', 'sliding_speed_m_s', 'final_stable_dwell_s',
              'initial_handoff', 'subsequent_handoff', 'scheduler', 'force_stop_policy', 'coverage_force_exclusion_n')
    for key in fields:
        assert am[key] == bm[key], key
    with np.load(a/'initial.npz') as aa, np.load(b/'initial.npz') as bb:
        errors = {key: float(np.max(np.abs(aa[key]-bb[key]))) for key in aa.files}
        assert all(value == 0 for value in errors.values())
    for name, expected in bm['source_hashes'].items():
        assert digest(ROOT/name) == expected
        assert digest(a/'source_snapshot'/name) == digest(b/'source_snapshot'/name) == expected
    register(a)
    assert (b/'portable_assets.json').is_file(), 'Frozen B catalog must already exist'
    register(b)
    mapping = asset_map()
    def normalized_model(case):
        model = ET.fromstring((case/'model.xml').read_text(encoding='utf-8'))
        for item in model.findall('./asset/*[@file]'):
            item.set('file', 'sha256:'+digest(mapping[item.get('file')]))
        return ET.tostring(model)
    ax, bx = normalized_model(a), normalized_model(b)
    assert ax == bx, 'Normalized model XML mismatch'
    return dict(status='PASSED_FROZEN_B_COMMON_CONDITIONS', fields=list(fields), initial_errors=errors,
                common_frozen_sources=len(bm['source_hashes']), normalized_model_xml_identical=True,
                user_selected_supervisor_metadata=True, strict_prediction_free_legacy_A=False,
                same_realized_wrist_path=False, same_supervisor_algorithms=True,
                contrast='B surface finger targets and joint nominal planning versus A robot-FK targets and fixed nominal hand; shared native MCC can move all fingers')


def verify_a_reference_boundary(a):
    summary = read(a/'summary.json')
    path = a/'nominal_wrist_reference.jsonl'
    assert digest(path) == summary['nominal_wrist_reference_sha256']
    allowed = {'index', 'requested_at_s', 'phase', 'poses_world', 'path_parameter_m', 'segment_durations_s'}
    count = 0
    for line in path.read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        assert set(row) <= allowed
        poses = np.asarray(row['poses_world'])
        assert poses.ndim == 2 and poses.shape[1] == 7 and np.isfinite(poses).all()
        np.testing.assert_allclose(np.linalg.norm(poses[:, 3:], axis=1), 1., atol=1e-7)
        count += len(poses)
    sources = set()
    for chunk in summary['trace_chunks']:
        with np.load(a/chunk['file']) as archive:
            assert archive['reference_is_robot_nominal'].all()
            assert archive['controlled_fingers'].all()
            sources.update(map(str, np.unique(archive['input_normal_source'])))
    assert sources <= {'A_ROBOT_NOMINAL', 'MEASURED_RAW', 'MEASURED_BELLY'}
    return dict(status='PASSED_A_RECORDED_REFERENCE_BOUNDARY', wrist_poses=count, normal_sources=sorted(sources),
                supervisor_prediction_metadata_is_explicitly_allowed=True,
                scope='Reference-constructor schema and native fields; same-process source boundary, not OS isolation',
                air_normal_note='Native source tag is last surface-selector context during air motion; shared air force relief reads current measured normals')


def online_planning(a, out):
    calls = [json.loads(line) for line in (a/'planning_calls.jsonl').read_text().splitlines()]
    # The frozen observer checks the returned tuple for find_patch, although
    # (None, attempts) is also a non-None tuple. Preserve the raw flag and
    # recover patch acceptance from its archived solution metadata. This
    # corrects reporting only, never the executing experiment or raw log.
    for row in calls:
        row['recorded_success'] = row.get('success')
        if row['name'] == 'private_supervisor.find_patch':
            row['success'] = ('solution_count' in row and row.get('reason') is None
                              and row['status'] == 'RETURNED')
    by_id = {row['id']: row for row in calls}
    assert len(by_id) == len(calls)
    fields = sorted(set().union(*(row.keys() for row in calls)))
    with (out/'a_all_planning_calls.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in sorted(calls, key=lambda item: item['id']):
            writer.writerow({key: json.dumps(value) if isinstance(value, (dict, list)) else value for key, value in row.items()})
    groups = {}
    for name in sorted({row['name'] for row in calls}):
        selected = [row for row in calls if row['name'] == name]
        groups[name] = dict(wall_s=stats([row['wall_s'] for row in selected]),
                            cpu_s=stats([row['cpu_s'] for row in selected]),
                            returned=sum(row['status'] == 'RETURNED' for row in selected),
                            failed=sum(row.get('success') is False or row['status'] != 'RETURNED' for row in selected))
    top = [row for row in calls if row['parent_id'] is None]
    solvers = [row for row in calls if row['name'] == 'least_squares']
    top_groups = {}
    for name in sorted({row['name'] for row in top}):
        selected = [row for row in top if row['name'] == name]
        top_groups[name] = dict(wall_s=stats([row['wall_s'] for row in selected]),
            cpu_s=stats([row['cpu_s'] for row in selected]),
            explicitly_rejected=sum(row.get('success') is False for row in selected),
            raised=sum(row['status'] != 'RETURNED' for row in selected))
    optimizer_groups = {}
    for objective in sorted({row['objective'] for row in solvers}):
        selected = [row for row in solvers if row['objective'] == objective]
        optimizer_groups[objective] = dict(
            variables=sorted({row['variables'] for row in selected}),
            wall_s=stats([row['wall_s'] for row in selected]),
            own_cost=stats([row['cost'] for row in selected]),
            nfev=stats([row['nfev'] for row in selected]),
            optimizer_success=sum(row['solver_success'] for row in selected),
            caveat='Optimizer convergence is not geometric acceptance; costs across objectives are not comparable')
    same_time = {}
    for row in top:
        instant = same_time.setdefault(row['simulation_time_s'], dict(
            simulation_time_s=row['simulation_time_s'], observed_top_level_wall_s=0., calls=0))
        instant['observed_top_level_wall_s'] += row['wall_s']
        instant['calls'] += 1
    instants = sorted(same_time.values(), key=lambda row: row['simulation_time_s'])
    with (out/'a_planning_by_simulation_instant.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(instants[0])); writer.writeheader(); writer.writerows(instants)
    descendants = {row['id']: [] for row in top}
    for solver in solvers:
        ancestor = solver
        while ancestor['parent_id'] is not None:
            ancestor = by_id[ancestor['parent_id']]
        descendants[ancestor['id']].append(solver)
    requests = []
    for row in sorted(top, key=lambda item: item['id']):
        request = {key: row.get(key) for key in ('id', 'name', 'simulation_time_s', 'wall_s', 'cpu_s',
                                                'status', 'success', 'recorded_success', 'error', 'planned_motion_s', 'route_length_m')}
        for variables, name in ((7, 'wrist'), (23, 'joint_oracle')):
            selected = sorted([item for item in descendants[row['id']] if item['variables'] == variables], key=lambda item: item['id'])
            request[name+'_solve_count'] = len(selected)
            request[name+'_last_optimizer_cost'] = selected[-1]['cost'] if selected else None
            request[name+'_all_optimizer_costs'] = json.dumps([item['cost'] for item in selected])
        requests.append(request)
    with (out/'a_planning_requests.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(requests[0])); writer.writeheader(); writer.writerows(requests)
    figure, axes = plt.subplots(2, 1, figsize=(13, 7), constrained_layout=True)
    for name in sorted({row['name'] for row in top}):
        rows = [row for row in top if row['name'] == name]
        axes[0].scatter([row['simulation_time_s'] for row in rows], [max(1e-6, row['wall_s']) for row in rows], s=12, label=name)
    axes[0].set_yscale('log'); axes[0].set_ylabel('Online top-level wall time (s)')
    axes[0].legend(fontsize=6, ncols=2)
    for objective in sorted({row['objective'] for row in solvers}):
        rows = [row for row in solvers if row['objective'] == objective]
        axes[1].scatter([row['simulation_time_s'] for row in rows], [max(1e-20, row['cost']) for row in rows], s=6,
                        label=f'{objective.split(":", 1)[0]} ({rows[0]["variables"]} variables)')
    axes[1].set_yscale('log'); axes[1].set_ylabel('Own optimizer cost (different objectives)')
    axes[1].set_xlabel('Simulation request time (s)'); axes[1].legend()
    figure.savefig(out/'a_online_planning.png', dpi=150); plt.close(figure)
    result = dict(calls=len(calls), top_level_calls=len(top), optimizer_calls=len(solvers), by_name=groups,
                  nonoverlapping_top_level_by_name=top_groups, optimizer_by_objective=optimizer_groups,
                  slowest_simulation_instants=sorted(instants, key=lambda row: row['observed_top_level_wall_s'], reverse=True)[:10],
                  nonoverlapping_top_level_wall_s=sum(row['wall_s'] for row in top),
                  nonoverlapping_top_level_cpu_s=sum(row['cpu_s'] for row in top),
                  nested_times_must_not_be_summed=True, setup_and_native_control_loop_excluded=True,
                  uninstrumented_overhead='Outer candidate sorting, scheduler orchestration and root log flush are excluded; this is observed-call time, not all planning time',
                  original_B_matching_call_ledger='NOT_RECORDED; cannot recover exact historical computation time')
    result['observer_success_flag_correction'] = dict(
        corrected_calls=sum(row.get('success') != row['recorded_success'] for row in calls
                            if row['name'] == 'private_supervisor.find_patch'),
        scope='find_patch raw success tests the return tuple, including (None, attempts); report success uses saved solution_count and reason. Raw JSONL and recorded_success are preserved.')
    write(out/'a_online_planning_summary.json', result)
    return result


def benchmark_summary(folder, out):
    metadata = read(folder/'benchmark_metadata.json')
    rows = [json.loads(line) for line in (folder/'paired_static_ik.jsonl').read_text().splitlines()]
    calls = [json.loads(line) for line in (folder/'benchmark_calls.jsonl').read_text().splitlines()]
    by_id = {row['id']: row for row in calls}
    own_costs = {}
    for row in calls:
        if row['name'] != 'least_squares':
            continue
        parent = by_id[row['parent_id']]
        own_costs.setdefault((parent['plan_index'], parent['name'][-1]), []).append(row)
    flat = []
    for row in rows:
        item = {key: row[key] for key in ('plan_index', 'event_index', 'kind', 'source_start_s', 'source_end_s')}
        for method in ('A', 'B'):
            item.update({method+'_'+key: value for key, value in row['results'][method].items()})
            solves = own_costs[(row['plan_index'], method)]
            item[method+'_last_own_optimizer_cost'] = solves[-1]['cost']
            item[method+'_least_squares_calls'] = len(solves)
            item[method+'_optimizer_nfev_total'] = sum(call['nfev'] for call in solves)
        flat.append(item)
    with (out/'per_plan_paired_static_ik.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
    result = dict(metadata=metadata, methods={})
    for method in ('A', 'B'):
        result['methods'][method] = {key: stats([row['results'][method][key] for row in rows]) for key in
            ('wall_s', 'cpu_s', 'common_four_pad_position_rmse_m', 'common_four_pad_angle_mean_deg', 'wrist_position_error_m')}
        result['methods'][method]['own_acceptance_failures'] = sum(row['results'][method]['reason'] is not None for row in rows)
        result['methods'][method]['last_own_optimizer_cost'] = stats([row[method+'_last_own_optimizer_cost'] for row in flat])
    result['B_over_A_paired_wall_ratio'] = stats([row['results']['B']['wall_s']/max(row['results']['A']['wall_s'], 1e-9) for row in rows])
    figure, axes = plt.subplots(3, 1, figsize=(13, 9), constrained_layout=True)
    for method, color in (('A', '#dc7c32'), ('B', '#247bb0')):
        axes[0].plot([row['results'][method]['wall_s'] for row in rows], label=method, color=color, linewidth=.8)
        axes[1].plot([max(1e-20, row[method+'_last_own_optimizer_cost']) for row in flat], label=method, color=color, linewidth=.8)
        axes[2].plot([1000*row['results'][method]['common_four_pad_position_rmse_m'] for row in rows], label=method, color=color, linewidth=.8)
    axes[0].set_ylabel('Offline static solve wall time (s)'); axes[0].set_yscale('log')
    axes[1].set_ylabel('Last own cost (NOT common objective)'); axes[1].set_yscale('log')
    axes[2].set_ylabel('Common four-pad position RMSE (mm)'); axes[2].set_xlabel('Recorded B plan index')
    for ax in axes:
        ax.legend(); ax.grid(alpha=.2)
    figure.savefig(out/'paired_planning_cost_time.png', dpi=150); plt.close(figure)
    write(out/'paired_static_ik_summary.json', result)
    return result


def verify_videos(case):
    manifest = read(case/'complete_videos/video_manifest.json')
    summary = read(case/'summary.json')
    assert manifest['trace_chunks'] == summary['trace_chunks']
    assert manifest['summary_sha256'] == digest(case/'summary.json')
    checks = []
    for video in manifest['videos']:
        path = case/'complete_videos'/video['file']
        assert digest(path) == video['sha256']
        reader = imageio.get_reader(path)
        info = reader.get_meta_data(); count = reader.count_frames(); reader.close()
        assert count == video['frames'] and info['fps'] == 20
        assert video['time_range_s'][0] == 0 and video['final_native_observation_included']
        assert abs(video['time_range_s'][1]-(summary['executed_s']-.002)) < .0021
        checks.append(dict(file=video['file'], decoded_frames=count, size=info['size'], sha256=video['sha256']))
    return dict(status='PASSED_COMPLETE_VIDEO_PROVENANCE_AND_DECODE', videos=checks)


def report(a, output):
    install_asset_adapter()
    a, b, out = Path(a).resolve(), BASELINE.resolve(), Path(output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    before = verify_download(b)
    alignment = verify_alignment(a, b)
    boundary = verify_a_reference_boundary(a)
    write(out/'a_reference_boundary.json', boundary)
    # load_verified writes derived metrics, so call it ONLY for new A. B stays read-only.
    _, aa, sa, da, ma = load_verified(a)
    sb, ab, mb = read(b/'summary.json'), read(b/'independent_native_audit/audit.json'), read(b/'whole_hand_metrics.json')
    for name, expected in ab['source_files'].items():
        assert digest(b/name) == expected
    va = verify_videos(a)
    write(a/'complete_videos/verification.json', va)
    vb = read(b/'complete_videos/verification.json')
    prefix = min(sa['executed_s'], sb['executed_s'])
    ap = contact_summary(native_arrays(a, prefix)); bp = contact_summary(native_arrays(b, prefix))
    full_a, full_b = native_arrays(a), native_arrays(b)
    af, bf = contact_summary(full_a), contact_summary(full_b)
    figure, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
    for method, data, color in (('A', full_a, '#dc7c32'), ('B', full_b, '#247bb0')):
        axes[0].plot(data['time_s'][::500], 100*data['coverage'][::500], label=method, color=color)
        bins = np.arange(0, data['time_s'][-1]+10., 10.)
        count = data['safe_belly'].sum(axis=1)
        rate = [float(np.mean(count[(data['time_s']>=left)&(data['time_s']<right)]>=3)) for left,right in zip(bins[:-1],bins[1:])]
        axes[1].plot((bins[:-1]+bins[1:])/2, 100*np.asarray(rate), label=method, color=color)
    axes[0].set_ylabel('Strict geodesic area coverage (%)'); axes[1].set_ylabel('>=3 fingers (% in 10 s bins)')
    axes[1].set_xlabel('Simulation time (s)')
    for ax in axes:
        ax.axvline(prefix, color='grey', linestyle=':', label='Common elapsed prefix'); ax.legend(); ax.grid(alpha=.2)
    figure.savefig(out/'coverage_contact_comparison.png', dpi=160); plt.close(figure)
    coverage_figure(a); contact_path_figure(a)
    shutil.copy2(a/'coverage_verified.png', out/'a_coverage.png')
    shutil.copy2(a/'contact_paths_verified.png', out/'a_contact_paths.png')
    planning = online_planning(a, out)
    bplans = event_rows(b); write(out/'b_original_plan_events.json', bplans)
    bcost = original_b_costs(b, out)
    transfers = transfer_timing(a, b, out)
    event_counts = {method: dict(Counter(event['kind'] for event in read(case/'events.json')))
                    for method, case in (('A', a), ('B', b))}
    adir = out/'a_original_events'; adir.mkdir(exist_ok=True); save_tables(adir, event_rows(a))
    bench = benchmark_summary(out/'b_offline_ik', out)
    assert bench['metadata']['plans'] == len(bplans), 'Incomplete offline per-event benchmark'
    assert bench['metadata']['source_summary_sha256'] == digest(b/'summary.json')
    assert bench['metadata']['source_events_sha256'] == digest(b/'events.json')
    result = dict(protocol='i04.ab.frozen-b-supervisor.comparison.v1',
        baseline_unchanged=before, common_conditions=alignment, common_elapsed_s=prefix,
        A_reference_boundary=boundary,
        common_prefix=dict(A=ap, B=bp), full=dict(A=af, B=bf),
        A=dict(summary=sa, metrics=ma, audit_status=aa['status']),
        B=dict(summary=sb, metrics=mb, audit_status=ab['status']),
        online_A_planning=planning, offline_static_ik=bench,
        original_B_selected_costs=bcost,
        actual_transfer_times=transfers,
        recorded_event_counts=event_counts,
        limitations=['One seed and fixed object; no statistical population claim',
                     'B historical per-call costs/times and rejected candidates were not saved',
                     'Offline paired IK is a newly constructed benchmark, not original B search latency',
                     'Same supervisor policy, not the same realized wrist path',
                     'Supervisor feasibility is predicted with a joint hand candidate that A does not execute; this is a frozen-supervisor ablation, not an independently optimized wrist-only planner',
                     'Coverage is a 25 mm graph-geodesic contact neighborhood, not physical skin pressure footprint'])
    write(out/'comparison.json', result)
    def link(case, filename):
        return '../'+case.name+'/'+filename
    rows = ''
    for label, values in (
        ('仿真执行时长', [f'{s["executed_s"]:.3f} s' for s in (sa, sb)]),
        ('实际运行墙钟时长（含初始化、规划、执行和写盘）', [f'{s["wall_elapsed_s"]/60:.2f} min' for s in (sa, sb)]),
        ('严格面积覆盖', [f'{100*s["coverage_fraction"]:.4f}%' for s in (sa, sb)]),
        ('顶点 / 目标', [f'{s["visited_vertices"]}/8192 · {s["visited_goals"]}/388' for s in (sa, sb)]),
        ('≥3指同时接触', [f'{100*v["three_fraction"]:.2f}%' for v in (af, bf)]),
        ('四指同时接触', [f'{100*v["four_fraction"]:.2f}%' for v in (af, bf)]),
        ('最长全手有效断触', [f'{m["longest_safe_gap_s"][-1]:.3f} s' for m in (ma, mb)]),
        ('原始力峰值', [f'{s["peak_force_n"]:.3f} N' for s in (sa, sb)]),
        ('原生卸力阶段时长', [f'{audit["phase_duration_s"].get("FORCE_RELEASE", 0):.3f} s' for audit in (aa, ab)]),
        ('几何守护回退次数', [str(s['geometry_backtracks']) for s in (sa, sb)]),
        ('换面计划 / 贴面路线与转向尝试', [f'{s["necessary_lifts"]} / {s["routes"]}' for s in (sa, sb)]),
        ('未找到可执行换面的搜索轮次（扫面 / 补扫）',
         [f'{event_counts[m].get("NO_COOPERATIVE_TRANSFER", 0)} / {event_counts[m].get("COMPLETION_SEARCH_RETRY", 0)}' for m in ('A', 'B')]),
        ('停止原因', [s['stop_reason'] for s in (sa, sb)]),
    ):
        rows += '<tr><th>'+html.escape(label)+'</th>'+''.join('<td>'+html.escape(v)+'</td>' for v in values)+'</tr>'
    videos = ''
    for method, case in (('A', a), ('B', b)):
        for speed in (12, 1):
            path = link(case, f'complete_videos/{method.lower()}_complete_{speed}x.mp4')
            videos += f'<section><h2>{method} · {speed}倍速完整时间轴</h2><video controls preload="metadata" src="{path}"></video><p><a href="{path}">打开完整视频</a>；包含所有离面、卸力、失败尝试和恢复。</p></section>'
    bmeth = bench['methods']
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>I04 A/B 冻结B调度对照</title>
<style>body{{font:16px/1.65 system-ui,"Microsoft YaHei",sans-serif;background:#eef2f6;color:#173044;margin:0}}main{{max-width:1180px;margin:auto;padding:28px}}section{{background:white;padding:24px;margin:20px 0;border-radius:12px}}h1,h2{{line-height:1.3}}video,img{{width:100%;border-radius:7px}}table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid #d5dfe7;padding:11px;text-align:left}}code{{overflow-wrap:anywhere}}a{{color:#12668d}}.note{{border-left:4px solid #cf8733;padding:14px;background:#fff7e9}}</style><main>
<h1>I04 A-MCC / B-MCC：冻结 B 调度的手指规划对照</h1><p>2026-10-21 · 原 B 完整成果保持不变 · A 从相同初态、零覆盖重新运行</p>
<section><h2>实验目的和边界</h2><p>比较在同一外层调度、MCC、物理模型和保护条件下，增加四指表面目标与联合手指规划的作用。按用户选择，A 的外层调度保留 B 的预计接触人数和参考手指；A 的名义参考构造只接收腕位姿，四指实际动作仍由公共 MCC 产生。</p><p class="note">这是“冻结 B 调度”的消融对照，区别于旧协议中连外层预测信息也禁用的 A。两组使用相同调度算法，闭环状态不同会使实际目标、腕轨迹和结束时间不同。</p><p>共同预算9000秒；2ms原生步长；超过12N仅记录，超过40N连续200ms才停止；≥8N状态仍排除覆盖。旧 B 的599份发布文件及50份运行源码保持原哈希。</p></section>
<section><h2>完整结果</h2><table><tr><th>指标</th><th>A-MCC</th><th>B-MCC</th></tr>{rows}</table><p>共同前 {prefix:.3f} 秒：A/B 覆盖 {100*ap['area_fraction']:.4f}% / {100*bp['area_fraction']:.4f}%；≥3指同时接触 {100*ap['three_fraction']:.2f}% / {100*bp['three_fraction']:.2f}%。每组完整视频播放到其真实停止点，不延长或截短以伪装时长相等。</p><img src="coverage_contact_comparison.png"><p>覆盖使用原25mm测地邻域并集；100%不等于物理皮肤压力足迹扫过每一毫米。单次固定Mustard、单种子结果不代表多种子统计结论。</p></section>
{videos}
<section><h2>每次规划：cost 和时间</h2><p>必须区分“求解计算耗时”和“计划动作时长”。B 原日志没有逐次计算计时，也没有保存各次最小二乘 cost 或被拒绝候选；不能从仿真时间反推。原始B表完整列出 {len(bplans)} 次已记录路线／转向／换面计划，可重建的换面评分为动作秒数，补扫时再加 3×缺少的预计接触手指数；贴面路线采用首个可行解，没有统一的路线标量cost。</p><p>A 实时记录 {planning['calls']} 次嵌套调用，其中顶层 {planning['top_level_calls']} 次、不重叠总耗时 {planning['nonoverlapping_top_level_wall_s']:.3f} 秒；初始化和原生控制循环不在该合计内。嵌套子调用不能重复加总。</p><img src="a_online_planning.png"><p><a href="a_all_planning_calls.csv">A逐次调用、原始优化cost与耗时CSV</a> · <a href="b_offline_ik/original_plan_events.csv">B全部原始规划事件CSV（缺失时间明确留空）</a></p>
<h2>补充：逐规划事件的配对静态IK微基准</h2><p>在B每个已记录规划动作的保存状态上，分别执行A腕IK和B四指联合IK，交替求解顺序。输入是保存的命令、触点参考和滤波法向，不能精确重建历史候选搜索。因此以下是新测量的静态求解性能，绝不是B当时的完整规划耗时。</p><p>A/B 中位求解时间 {1000*bmeth['A']['wall_s']['median']:.2f} / {1000*bmeth['B']['wall_s']['median']:.2f} ms；P95 {1000*bmeth['A']['wall_s']['p95']:.2f} / {1000*bmeth['B']['wall_s']['p95']:.2f} ms。共同四指位置RMSE中位数 {1000*bmeth['A']['common_four_pad_position_rmse_m']['median']:.2f} / {1000*bmeth['B']['common_four_pad_position_rmse_m']['median']:.2f} mm。</p><p>A的优化器cost衡量腕位置、腕朝向和臂间隙；B的cost衡量四指目标、姿态先验和间隙，二者维度、残差项不同，不能用cost较小说某方法更好。另列同一四指位置误差，作为共同后验评价。</p><img src="paired_planning_cost_time.png"><p><a href="per_plan_paired_static_ik.csv">逐事件A/B求解cost、wall/CPU时间与共同误差CSV</a> · <a href="paired_static_ik_summary.json">统计及范围声明</a> · <a href="b_offline_ik/benchmark_calls.jsonl">全部原始求解日志</a></p></section>
<section><h2>A实际覆盖与接触轨迹</h2><img src="a_coverage.png"><img src="a_contact_paths.png"></section>
<section><h2>核验与复现</h2><p>两组均从原始初态连续重放；状态、接触力、覆盖及每个已执行命令的两次几何守护分别核验。B沿用已完成的原生核验与原视频；本次没有重新运行或修改其动态实验。</p><p><a href="comparison.json">完整机器可读报告</a> · <a href="common_conditions.json">共同条件核验</a> · <a href="{link(a,'independent_native_audit/audit.json')}">A独立原生核验</a> · <a href="{link(b,'independent_native_audit/audit.json')}">B原独立核验</a> · <a href="../../I04_FROZEN_B_AB_RUNBOOK.md">如何运行</a></p></section></main></html>'''
    marker = '<h2>补充：逐规划事件的配对静态IK微基准</h2>'
    assert page.count(marker) == 1
    page = page.replace(marker, '<p>A调用图中的23关节求解属于共同的私有外层几何服务；A方法自身只做7关节腕IK。两者均计入实际开销，但不会将外层四指几何参考传入A的名义手指控制。</p><img src="b_original_selected_costs.png">'+marker)
    marker = '<a href="a_all_planning_calls.csv">'
    page = page.replace(marker, '<a href="a_planning_requests.csv">A逐次顶层规划汇总CSV</a> · '+marker)
    timing_rows = ''
    for name, group in sorted(planning['nonoverlapping_top_level_by_name'].items(),
                              key=lambda item: item[1]['wall_s']['total'], reverse=True):
        timing = group['wall_s']
        timing_rows += (f'<tr><td><code>{html.escape(name)}</code></td><td>{timing["n"]}</td>'
            f'<td>{timing["total"]:.3f}</td><td>{1000*timing["median"]:.3f}</td>'
            f'<td>{1000*timing["p95"]:.3f}</td><td>{group["explicitly_rejected"]} / {group["raised"]}</td></tr>')
    marker = '<img src="a_online_planning.png">'
    page = page.replace(marker, '<table><tr><th>A实测顶层请求</th><th>次数</th><th>总wall / s</th>'
        '<th>中位 / ms</th><th>P95 / ms</th><th>明确拒绝 / 异常</th></tr>'+timing_rows+'</table>'
        '<p>此表各行互不嵌套，可以相加；“明确拒绝”只统计调用显式返回失败，未提供成功字段的返回值不视为成功。'
        '每次调用、同一仿真时刻累计搜索时间及求解器收敛状态均保留。'
        '<a href="a_planning_by_simulation_instant.csv">按仿真时刻汇总的观察到的搜索耗时</a> · '
        '<a href="a_online_planning_summary.json">分类统计JSON</a></p>'
        '<p>计时日志中find_patch的原success字段把返回元组也算作成功；报告依据已记录的解是否存在还原其接受/拒绝状态。'
        'CSV保留recorded_success原值，success为修正统计；原始JSONL、运行控制和所有耗时均未更改。</p>'+marker)
    dominant_name, dominant = max(planning['nonoverlapping_top_level_by_name'].items(),
                                  key=lambda item: item[1]['wall_s']['total'])
    marker = '<h2>补充：逐规划事件的配对静态IK微基准</h2>'
    page = page.replace(marker, f'<p>A已观测顶层调用中，累计开销最多的是<code>{html.escape(dominant_name)}</code>，'
        f'共{dominant["wall_s"]["total"]:.3f}秒，占该调用合计的'
        f'{100*dominant["wall_s"]["total"]/max(planning["nonoverlapping_top_level_wall_s"], 1e-12):.2f}%。'
        '单次腕IK的耗时需要结合候选搜索、换面重试、接触结果与最终覆盖率一起评价。'
        '下方静态微基准只比较单次参考求解，不能替代上述整任务开销。</p>'+marker)
    page = page.replace(marker, marker+'<p>静态输入分布来自B记录：共用保存的手臂起点，A使用公共初态的名义手形，B使用保存状态的手形种子。'
        '每个事件每种方法测一次，未绑核、未重复计时；环境版本和线程设置见微基准metadata。它不是两组各自全部在线请求的同分布统计。</p>')
    marker = '<img src="paired_planning_cost_time.png">'
    page = page.replace(marker, f'<p>各自优化器cost中位数：A为{bmeth["A"]["last_own_optimizer_cost"]["median"]:.6g}，'
        f'B为{bmeth["B"]["last_own_optimizer_cost"]["median"]:.6g}；因目标函数不同，这两个数不能直接排名。'
        f'按各自判据拒绝的请求数：A {bmeth["A"]["own_acceptance_failures"]}，B {bmeth["B"]["own_acceptance_failures"]}，'
        f'每组总计{bench["metadata"]["plans"]}个请求。统计包含全部请求，未剔除失败项；两者的接受判据也不等价。</p>'+marker)
    marker = '<img src="a_online_planning.png">'
    page = page.replace(marker, '<p>WRIST表示7变量腕目标函数，既可能来自A参考构造，也可能是共同外层服务内部的腕种子求解；'
        '归属需要看父调用ID，不能只按变量数划分A与B的开销。cost对数图显示下限为1e−20，原始数值在CSV/JSON中完整保留。</p>'+marker)
    marker = '<h2>完整结果</h2>'
    page = page.replace(marker, marker+f'<p>本轮A的主要问题表现在多指支撑和恢复：至少三指合格接触仅占'
        f'{100*af["three_fraction"]:.4f}%，原生卸力阶段占{100*aa["phase_duration_s"].get("FORCE_RELEASE", 0)/sa["executed_s"]:.2f}%。'
        f'A经历{event_counts["A"].get("NO_COOPERATIVE_TRANSFER", 0)}轮扫面无可执行换面搜索和'
        f'{event_counts["A"].get("COMPLETION_SEARCH_RETRY", 0)}轮补扫重试后停止，未完成整瓶覆盖。'
        '这描述的是本次冻结B外层流程的结果；同一时间前缀的比较也显示了覆盖差异。'
        '是否存在更适合A的独立腕规划策略，需要另外实验。</p>')
    marker = '<h2>每次规划：cost 和时间</h2>'
    page = page.replace(marker, marker+f'<p>已记录换面段中，A实际耗时{transfers["A"]["recorded_elapsed_s"]:.3f}秒，其中卸力暂停{transfers["A"]["recorded_force_pause_s"]:.3f}秒；B分别为{transfers["B"]["recorded_elapsed_s"]:.3f}秒和{transfers["B"]["recorded_force_pause_s"]:.3f}秒。这里是物理执行时间，排除接近/重触阶段及未记录结果的终端片段。选中评分使用计划动作时长，实际卸力暂停会增加执行耗时。<a href="transfer_segment_times.csv">逐段计划与实际时长CSV</a></p>')
    marker = '<h2>每次规划：cost 和时间</h2>'
    page = page.replace(marker, marker+'<p>规划期间仿真时钟不推进。表中的完整运行墙钟时长包含初始化、规划、原生执行和写盘，两次运行均有并行审计与视频任务；不能把它等同于纯规划时间。A调用合计覆盖已包围的几何搜索、路径与腕IK函数，尚未包含外层候选排序、调度代码及顶层日志写入。</p>')
    marker = '<h2>实验目的和边界</h2>'
    page = page.replace(marker, marker+'<p>本轮检验的是去掉手指参考后能否沿用既有流程完成任务。外层预计可接触的手指数来自联合手形候选，A不执行该手形，所以预测可行不保证A实际可行。结果不代表为A重新设计过的最优腕规划器性能。</p>')
    (out/'review.html').write_text(page, encoding='utf-8')
    write(out/'common_conditions.json', alignment)
    references = re.findall(r'(?:src|href)="([^"]+)"', page)
    for reference in references:
        assert (out/reference).resolve().is_file(), reference
    write(out/'report_validation.json', dict(status='PASSED_STATIC_REPORT_REFERENCES', references=references,
          A_video_verification=va, B_video_verification=vb, browser_click_test=False,
          browser_limit='Local file browser automation denied by URL policy; verified static links, decoded video and inspected PNGs'))
    (out/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    write(out/'b_immutable_after.json', verify_download(b))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--a', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report(args.a, args.output)
