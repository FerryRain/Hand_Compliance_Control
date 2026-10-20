"""Verified report for the full B simulation with explicit force tolerance."""
from __future__ import annotations

import argparse
from functools import lru_cache
import html
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.csgraph import dijkstra

from .b_full_mcc import PROTOCOL
from .b_full_video import digest, read
from .published_whole_hand import install_asset_adapter
from .whole_hand_surface_review import load_verified, region_samples
from .natural_belly_report import coverage_figure, contact_path_figure
from .fixtures import SurfaceGraph
from .full_review import task_mesh


def force_tolerant_coverage(case, audit, summary):
    """Secondary metric: remove only the original 8N exclusion, retain geometry.

    The base ledger and every contact field have already been independently
    replayed. Add only geometry-qualified observations excluded by raw force.
    This does not change the controller, strict coverage, or completion flag.
    """
    assert audit['status'] == 'PASSED_NATIVE_EVIDENCE_AUDIT'
    for name in ('final_state.npz', 'model.xml', 'summary.json'):
        assert digest(case / name) == audit['source_files'][name]
    with np.load(case / 'final_state.npz') as final:
        covered = final['visited_vertices'].copy()
    result = dict(strict_area_fraction=audit['area_fraction'],
                  area_fraction=audit['area_fraction'], covered_vertices=audit['covered_vertices'],
                  covered_goals=audit['covered_goals'], added_vertices=0,
                  central_belly_high_force_observations=0, mesh_qualified_high_force_contacts=0,
                  scope='Secondary force-tolerant coverage only; original geometry, normals, 25mm radius and full denominator retained',
                  does_not_change_strict_completion=True)
    if max(audit['raw_force_peak_n']) < 8.:
        return result
    graph = SurfaceGraph(task_mesh(), coverage_radius_m=.025)
    assert len(graph.vertices_m) == 8192 and len(graph.required_vertices) == 388
    area = np.zeros(8192)
    np.add.at(area, graph.faces.ravel(), np.repeat(graph.mesh.area_faces / 3, 3))
    assert abs(float(area @ covered / area.sum()) - audit['area_fraction']) < 1e-12
    body = ET.fromstring((case / 'model.xml').read_text(encoding='utf-8')).find(".//body[@name='fr3_e05_object']")
    origin = np.fromstring(body.get('pos'), sep=' ')

    @lru_cache(maxsize=8192)
    def neighborhood(vertex):
        return np.flatnonzero(np.isfinite(dijkstra(
            graph.adjacency, directed=False, indices=int(vertex), limit=.025)))

    observations = contacts = 0
    for chunk in summary['trace_chunks']:
        path = case / chunk['file']
        assert digest(path) == chunk['sha256']
        with np.load(path) as z:
            end = len(z['time_s']) - int(chunk.get('skip_last', 0))
            high = np.flatnonzero(z['forces_n'][:end].max(axis=1) >= 8.)
            if not len(high):
                continue
            live = z['belly_active'][high] & (z['rejected_contact_force_n'][high] < .2)
            live[(z['unsafe_count'][high] != 0) | (z['arm_self_force_n'][high] > .5)] = False
            observations += int(live.any(axis=1).sum())
            points = z['belly_contact_positions_world_m'][high]
            normals = z['belly_normals_world'][high]
            for row, finger in zip(*np.nonzero(live)):
                distances, indices = graph._tree.query(points[row, finger] - origin, k=24)
                cosines = graph.normals[indices] @ normals[row, finger]
                eligible = distances <= distances[0] + .0015
                col = int(np.argmax(np.where(eligible, cosines, -np.inf)))
                if distances[col] <= .005 and cosines[col] >= .5:
                    covered[neighborhood(int(indices[col]))] = True
                    contacts += 1
    result.update(area_fraction=float(area @ covered / area.sum()),
                  covered_vertices=int(covered.sum()),
                  covered_goals=int(covered[graph.required_vertices].sum()),
                  added_vertices=int(covered.sum()) - audit['covered_vertices'],
                  central_belly_high_force_observations=observations,
                  mesh_qualified_high_force_contacts=contacts)
    assert result['area_fraction'] >= audit['area_fraction'] - 1e-12
    return result


def measured_statistics(case, audit, summary):
    width = 50
    bins = (audit['native_states'] + width - 1) // width
    force = np.zeros(bins)
    coverage = np.zeros(bins)
    contact = np.zeros((bins, 4))
    counts = np.zeros(bins)
    physical_contact_count = np.zeros(5, np.int64)
    physical_finger_count = np.zeros(4, np.int64)
    high = {key: dict(observations=0, applied_samples=0, episodes=0, longest_samples=0, tail=0)
            for key in ('at_least_8n', 'above_12n', 'above_40n')}
    stages = {}
    total = 0
    peak, peak_time = -1., None

    def runs(mask, record):
        edges = np.diff(np.r_[False, mask, False].astype(int))
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        record['episodes'] += len(starts) - int(bool(len(starts) and starts[0] == 0 and record['tail']))
        if len(starts):
            lengths = ends - starts
            if starts[0] == 0:
                lengths[0] += record['tail']
            record['longest_samples'] = max(record['longest_samples'], int(lengths.max()))
            record['tail'] = int(lengths[-1]) if ends[-1] == len(mask) else 0
        else:
            record['tail'] = 0

    physical_gap_record = dict(episodes=0, longest_samples=0, tail=0)
    for chunk in summary['trace_chunks']:
        path = case / chunk['file']
        assert digest(path) == chunk['sha256']
        with np.load(path) as z:
            end = len(z['time_s']) - int(chunk.get('skip_last', 0))
            times = z['time_s'][:end]
            raw = z['forces_n'][:end].max(axis=1)
            if raw.max() > peak:
                k = int(raw.argmax())
                peak, peak_time = float(raw[k]), float(times[k])
            index = (total + np.arange(end)) // width
            total += end
            np.maximum.at(force, index, raw)
            np.maximum.at(coverage, index, z['coverage'][:end])
            np.add.at(counts, index, 1)
            for finger in range(4):
                np.add.at(contact[:, finger], index, z['safe_belly'][:end, finger])
            physical = z['belly_active'][:end]
            physical_contact_count += np.bincount(physical.sum(axis=1), minlength=5)
            physical_finger_count += physical.sum(axis=0)
            runs(~physical.any(axis=1), physical_gap_record)
            applied = times < summary['executed_s'] - .001
            for key, mask in (('at_least_8n', raw >= 8.), ('above_12n', raw > 12.),
                              ('above_40n', raw > 40.)):
                record = high[key]
                record['observations'] += int(mask.sum())
                live = mask & applied
                record['applied_samples'] += int(live.sum())
                runs(live, record)
            assert not z['safe_belly'][:end][raw >= 8.].any()
            assert z['controlled_fingers'][:end].all()
            for name in np.unique(z['traversal_stage'][:end]):
                ids = np.flatnonzero(z['traversal_stage'][:end] == name)
                record = stages.setdefault(str(name), dict(observations=0, first_s=float(times[ids[0]])))
                record['observations'] += len(ids)
                record['last_s'] = float(times[ids[-1]])
                record['last_coverage'] = float(z['coverage'][ids[-1]])
    assert total == audit['native_states']
    assert abs(peak - max(audit['raw_force_peak_n'])) < 1e-9
    assert high['above_12n']['observations'] == summary['force_warning_observations']
    for record in high.values():
        record['total_applied_s'] = record.pop('applied_samples') * .002
        record['longest_applied_s'] = record.pop('longest_samples') * .002
        record.pop('tail')
    result = dict(
        native_states=total, raw_peak_n=peak, raw_peak_time_s=peak_time,
        force_thresholds=high, stages=stages,
        physical_central_belly_contact_count_fraction=(physical_contact_count / total).tolist(),
        physical_central_belly_finger_contact_fraction=(physical_finger_count / total).tolist(),
        physical_central_belly_longest_whole_hand_gap_s=physical_gap_record['longest_samples'] * .002,
        physical_contact_definition='Measured central palmar contact using the original 0.2N detection threshold, including high-force observations; not coverage qualification',
        force_policy=read(case / 'manifest.json')['force_stop_policy'],
        original_8n_coverage_exclusion_unchanged=True,
    )
    (case / 'full_b_statistics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    t = np.minimum((np.arange(bins) + .5) * .1, summary['executed_s'])
    contact /= counts[:, None]
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True,
                             gridspec_kw={'height_ratios': [1, 1, 1.2]})
    axes[0].plot(t, coverage * 100, color='#18835e')
    axes[0].set_ylabel('Measured coverage (%)')
    axes[0].set_ylim(0, 102)
    axes[1].plot(t, force, color='#344f85', lw=.8, label='Raw maximum in each 100 ms bin')
    axes[1].axhline(8, color='#ba8b13', ls='--', label='8 N: excluded from coverage')
    axes[1].axhline(12, color='#b95555', ls=':', label='12 N: recorded warning only')
    axes[1].set_ylim(0, max(14., peak * 1.15))
    axes[1].set_ylabel('Raw force (N)')
    axes[1].legend(loc='upper left', fontsize=8)
    image = axes[2].imshow(contact.T, aspect='auto', origin='lower', interpolation='nearest',
                           cmap='Blues', vmin=0, vmax=1,
                           extent=(0, summary['executed_s'], -.5, 3.5))
    axes[2].set_yticks(range(4), ['Index', 'Middle', 'Ring', 'Thumb'])
    axes[2].set_ylabel('Qualified belly contact')
    axes[2].set_xlabel('Elapsed physical simulation time (s)')
    for ax in axes:
        ax.set_xlim(0, summary['executed_s'])
        ax.grid(axis='x', alpha=.15)
    color_axis = axes[2].inset_axes([1.01, 0, .012, 1])
    fig.colorbar(image, cax=color_axis, label='Qualified fraction per 100 ms bin')
    fig.suptitle('COMPLETE REPLAY-VERIFIED B TRAJECTORY\n'
                 'Simulation force stop: >40 N for 200 ms. All measured peaks and contact gaps retained.')
    fig.subplots_adjust(left=.09, right=.92, top=.88, bottom=.08, hspace=.2)
    fig.savefig(case / 'timeline_verified.png', dpi=150)
    plt.close(fig)
    return result


def report(case):
    install_asset_adapter()
    case, audit, summary, data, metrics = load_verified(case)
    manifest = read(case / 'manifest.json')
    assert manifest['protocol'] == PROTOCOL
    assert (case / 'run_complete.json').exists()
    assert summary['fresh_single_version'] and not summary['coverage_inherited']
    regions = coverage_figure(case)
    contact_path_figure(case)
    measured = measured_statistics(case, audit, summary)
    relaxed = force_tolerant_coverage(case, audit, summary)
    (case / 'force_tolerant_coverage.json').write_text(json.dumps(relaxed, indent=2), encoding='utf-8')
    body = ET.fromstring((case / 'model.xml').read_text(encoding='utf-8')).find(".//body[@name='fr3_e05_object']")
    chapters = region_samples(data, np.fromstring(body.get('pos'), sep=' '))
    (case / 'video_chapters.json').write_text(json.dumps(
        dict(scope='Cooperative example locations in the complete 1x video; full trajectory is retained',
             regions=chapters), ensure_ascii=False, indent=2), encoding='utf-8')
    chapter_links = ' · '.join(
        (f'<a href="#normal-video" onclick="const v=document.getElementById(\'normal-video\');'
         f'v.currentTime={item["time_s"]:.6f};v.scrollIntoView({{block:\'center\'}});return false;">'
         f'{html.escape(item["region"])} {item["time_s"]:.1f}s</a>')
        if item['available'] else f'<span class="note">{html.escape(item["region"])}（未选出示例）</span>'
        for item in chapters)
    metadata = read(case / 'complete_videos/video_manifest.json')
    assert metadata['manifest_sha256'] == audit['source_files']['manifest.json']
    assert metadata['summary_sha256'] == audit['source_files']['summary.json']
    assert metadata['trace_chunks'] == summary['trace_chunks']
    assert metadata['native_observations'] == audit['native_states']
    video_blocks, video_checks = [], []
    for video in sorted(metadata['videos'], key=lambda item: -item['speed']):
        path = case / 'complete_videos' / video['file']
        assert digest(path) == video['sha256']
        with imageio.get_reader(path) as reader:
            frames = reader.count_frames()
            info = reader.get_meta_data()
        assert frames == video['frames']
        assert info['size'] == (2400, 700) and abs(info['fps'] - 20.) < 1e-9
        assert video['full_time_axis'] and video['final_native_observation_included']
        assert video['time_range_s'][0] == 0.
        assert summary['executed_s'] - .0021 <= video['time_range_s'][1] <= summary['executed_s']
        video_checks.append(dict(file=video['file'], decoded_frames=frames, size=info['size'],
                                 fps=info['fps'], sha256=video['sha256']))
        label = '正常速度完整视频' if video['speed'] == 1 else '完整时间轴总览'
        identity = ' id="normal-video"' if video['speed'] == 1 else ''
        video_blocks.append(
            f'<section><h2>{label} · {video["speed"]}倍速</h2>'
            f'<video{identity} controls preload="metadata" src="complete_videos/{video["file"]}"></video>'
            f'<p>视频时长{video["duration_s"]:.2f}秒；从仿真0秒到{video["time_range_s"][1]:.3f}秒，'
            '保留全部离面、卸力、恢复及结束状态。</p></section>')
    (case / 'complete_videos/verification.json').write_text(json.dumps(
        dict(status='PASSED_COMPLETE_VIDEO_PROVENANCE_AND_DECODE',
             independent_audit_sha256=digest(case / 'independent_native_audit/audit.json'),
             videos=video_checks), indent=2), encoding='utf-8')
    region_rows = ''.join(
        f'<tr><td>{row["region"]}</td><td>{row["covered"]}/{row["vertices"]}</td>'
        f'<td>{100 * row["area_fraction"]:.2f}%</td></tr>' for row in regions['regions'])
    finger_rows = ''.join(
        f'<tr><td>{name}</td><td>{100 * measured["physical_central_belly_finger_contact_fraction"][i]:.2f}%</td>'
        f'<td>{100 * metrics["all_time_finger_contact_fraction"][i]:.2f}%</td>'
        f'<td>{metrics["actual_tangential_material_motion_m"][i]:.3f}m</td>'
        f'<td>{metrics["actual_joint_min_deg"][i][2]:.1f}–{metrics["actual_joint_max_deg"][i][2]:.1f}° / '
        f'{metrics["actual_joint_min_deg"][i][3]:.1f}–{metrics["actual_joint_max_deg"][i][3]:.1f}°</td></tr>'
        for i, name in enumerate(('食指', '中指', '无名指', '拇指')))
    stage_rows = ''.join(
        f'<tr><td>{"整瓶贴面遍历" if name == "surface_sweep" else "遗漏区域补扫"}</td>'
        f'<td>{row["first_s"]:.3f}–{row["last_s"]:.3f}s</td>'
        f'<td>{100 * row["last_coverage"]:.3f}%</td></tr>' for name, row in measured['stages'].items())
    complete = bool(audit['full_surface_complete'])
    status = '整瓶遍历完成' if complete else '本次尚未完成整瓶遍历'
    above12 = measured['force_thresholds']['above_12n']
    above8 = measured['force_thresholds']['at_least_8n']
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>I04B · Mustard 完整测试</title>
<style>body{{margin:0;background:#eef2f6;color:#17283b;font:17px/1.7 system-ui,"Microsoft YaHei",sans-serif}}
main{{max-width:1440px;margin:auto;padding:28px}}section{{background:white;border-radius:14px;padding:24px;margin:22px 0}}
video,img{{width:100%;border-radius:8px}}table{{width:100%;border-collapse:collapse}}th,td{{text-align:left;padding:10px;border-bottom:1px solid #dbe3ec}}
a{{color:#1768a8}}.note{{color:#526779;font-size:15px}}code{{font-size:14px}}@media(max-width:700px){{main{{padding:12px}}section{{padding:12px}}}}</style>
<main><h1>I04B · Mustard 完整测试</h1>
<p><b>{status}</b>：{100 * audit['area_fraction']:.3f}%，{audit['covered_vertices']}/8192顶点、
{audit['covered_goals']}/388目标；实际仿真{summary['executed_s']:.3f}秒。</p>
<p>手掌与四指共同规划，主动指腹贴合、自然屈指、贴面优先换面。单一冻结版本从共同悬停初态、零覆盖开始。</p>
<p>按用户要求容忍仿真高力：超过12N只记录，超过40N持续200ms才停止。原始力完整保留；覆盖仍使用原8N排除规则。</p>
<section><h2>六方向定位</h2><p>{chapter_links}</p>
<p class="note">点击跳到正常速度完整视频中的接触示例，再按播放。全部离面、恢复和未成功的局部尝试仍在完整时间轴中。</p></section>
{''.join(video_blocks)}
<section><h2>整瓶是否摸全</h2><img src="coverage_verified.png" alt="六方向独立核验的覆盖图">
<table><tr><th>区域</th><th>已覆盖顶点</th><th>面积指标</th></tr>{region_rows}</table>
<p class="note">绿色为合格实际指腹接触的25mm网格边测地邻域；灰色为未覆盖。保留整个原始网格分母。
该指标不是皮肤压力足迹面积，规划目标和历史成功轨迹不增加本次覆盖。</p>
<p>单独忽略8N力排除后的几何合格指腹覆盖为{100 * relaxed['area_fraction']:.3f}%：
{relaxed['covered_vertices']}/8192顶点、{relaxed['covered_goals']}/388目标，比原严格指标增加{relaxed['added_vertices']}个顶点。
此项只用于查看仿真力阈值对计分的影响，控制器和上面的严格完成判据保持原样。</p>
<img src="contact_paths_verified.png" alt="六方向实际指腹接触位置"></section>
<section><h2>四指是否共同参与</h2><table><tr><th>手指</th><th>实际指腹接触率（含高力）</th>
<th>原严格合格接触率</th><th>合格接触时切向材料运动</th><th>末两关节实际角度范围</th></tr>{finger_rows}</table>
<p>全程至少2指合格接触{100 * sum(metrics['all_time_contact_count_fraction'][2:]):.2f}%，
至少3指{100 * sum(metrics['all_time_contact_count_fraction'][3:]):.2f}%，
四指{100 * metrics['all_time_contact_count_fraction'][4]:.2f}%。统计包含接近、离面和恢复，不要求始终四指同时接触。</p>
<p class="note">实测中央指腹接触沿用原0.2N检测门槛；“含高力”与放宽力统计只移除上限排除，保留接触位置和朝向判定。</p>
<p>最长完全没有实测中央指腹接触{measured['physical_central_belly_longest_whole_hand_gap_s']:.3f}s；
按原严格判据，最长无合格指腹接触{metrics['longest_safe_gap_s'][-1]:.3f}s。覆盖完成与接触连续性分别评价。</p></section>
<section><h2>遍历、补扫与原始力</h2><table><tr><th>阶段</th><th>仿真时段</th><th>阶段结束覆盖</th></tr>{stage_rows}</table>
<img src="timeline_verified.png" alt="完整时间轴覆盖、原始力峰及四指接触">
<p>原始力峰{measured['raw_peak_n']:.4f}N，发生于{measured['raw_peak_time_s']:.3f}s；
超过12N累计{above12['total_applied_s']:.3f}s，最长连续{above12['longest_applied_s']:.3f}s；
达到8N的状态共{above8['total_applied_s']:.3f}s，均不增加覆盖。所有高力观测仍保留。</p>
<p>换姿计划{summary['necessary_lifts']}次，贴面路径/转向尝试{summary['routes']}次；
终止原因<code>{html.escape(summary['stop_reason'])}</code>。</p></section>
<section><h2>怎么实现与核验</h2><p>恢复原B起步/推进窗口、逐指法向限速和参考手指更新；
目标排序不再突然重投影当前触点。先整瓶遍历，再用同一物理状态补扫。MCC、主动卸力、自然关节范围及两类几何守护保持原实现。</p>
<p>{audit['native_states']:,}个2ms原生状态从t=0连续独立重放，分块不重置；
所有实际命令的实态与上一参考几何检查通过。力、接触和覆盖重新测量，非指腹/机械臂危险接触记录{audit['unsafe_states']}个。</p>
<p class="note">仅验证固定Mustard、零重力、MCC。DP继续暂缓。本版B尚未与采用完全相同非对照条件的新A配对，
不以它和历史A的差异作单因素性能结论。</p>
<p><a href="../../I04_B_FULL_RUNBOOK.md">运行说明</a> · <a href="manifest.json">冻结配置与源码哈希</a> ·
<a href="summary.json">完整轨迹分块清单</a> · <a href="full_b_statistics.json">原始力与接触统计</a> ·
<a href="force_tolerant_coverage.json">忽略仿真力阈值的覆盖统计</a> ·
<a href="whole_hand_metrics.json">整手指标</a> · <a href="independent_native_audit/audit.json">独立核验</a> ·
<a href="complete_videos/video_manifest.json">视频来源和帧数</a> ·
<a href="complete_videos/verification.json">视频解码核验</a></p></section></main></html>'''
    (case / 'review.html').write_text(page, encoding='utf-8')
    source = case / 'visualization_source'
    source.mkdir(exist_ok=True)
    (source / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    return dict(status=status, report=str(case / 'review.html'), complete=complete)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(report(args.case), ensure_ascii=False), flush=True)
