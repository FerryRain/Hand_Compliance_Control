"""Independent audit, truthful A/B videos and matched-prefix MCC reports."""
from __future__ import annotations

import argparse
from hashlib import sha256
import html
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .published_whole_hand import CASE, install_asset_adapter, portable_guard_initializer
from .whole_hand_surface_review import load_verified, region_samples
from .sliding_audit import _native_context
from .pad_contact_visual import camera_from_direction, marker
from .natural_belly_report import coverage_figure, contact_path_figure, timeline_figure, verified_force_excursions


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def audit(case, workers=4):
    case = Path(case).resolve()
    if case == CASE or (case/'independent_native_audit').exists():
        raise FileExistsError('Audit only into a new, non-archival output directory')
    install_asset_adapter()
    from . import whole_hand_parallel_audit as auditor
    auditor._published_original_initializer = auditor.initialize_guard_worker
    auditor.initialize_guard_worker = portable_guard_initializer
    return auditor.audit(case, workers=workers, output_name='independent_native_audit')


def native_arrays(case, through_s=None):
    summary = read(case/'summary.json')
    arrays = {}
    keys = ('time_s', 'safe_belly', 'forces_n', 'coverage', 'visited_vertex_count', 'phase')
    for chunk in summary['trace_chunks']:
        path = case/chunk['file']
        assert sha256(path.read_bytes()).hexdigest() == chunk['sha256']
        with np.load(path) as z:
            end = len(z['time_s'])-int(chunk.get('skip_last', 0))
            mask = np.ones(end, dtype=bool) if through_s is None else z['time_s'][:end] < through_s-1e-9
            for key in keys:
                arrays.setdefault(key, []).append(z[key][:end][mask].copy())
    return {key: np.concatenate(parts) for key, parts in arrays.items()}


def contact_summary(data):
    safe = data['safe_belly']; count = safe.sum(axis=1)
    return dict(observations=len(count), covered_vertices=int(data['visited_vertex_count'][-1]),
                area_fraction=float(data['coverage'][-1]), raw_peak_n=float(data['forces_n'].max()),
                any_fraction=float(np.mean(count >= 1)), two_fraction=float(np.mean(count >= 2)),
                three_fraction=float(np.mean(count >= 3)), four_fraction=float(np.mean(count == 4)),
                finger_fraction=safe.mean(axis=0).tolist())


def input_boundary_record(case):
    """Check the recorded public schema and provenance; not a claim of OS isolation."""
    manifest = read(case/'manifest.json'); summary = read(case/'summary.json')
    reference = case/'nominal_wrist_reference.jsonl'
    assert sha256(reference.read_bytes()).hexdigest() == manifest['nominal_wrist_reference_sha256']
    allowed = {'index', 'requested_at_s', 'phase', 'poses_world', 'path_parameter_m', 'segment_durations_s'}
    references = 0
    for line in reference.read_text(encoding='utf-8').splitlines():
        entry = json.loads(line)
        assert set(entry) <= allowed
        poses = np.asarray(entry['poses_world'], dtype=float)
        assert poses.ndim == 2 and poses.shape[1] == 7 and np.isfinite(poses).all()
        np.testing.assert_allclose(np.linalg.norm(poses[:, 3:], axis=1), 1., atol=1e-7)
        references += len(poses)
    sources = set(); states = 0
    for chunk in summary['trace_chunks']:
        with np.load(case/chunk['file']) as z:
            end = len(z['time_s'])-int(chunk.get('skip_last', 0))
            assert z['controlled_fingers'][:end].all()
            assert np.all(z['reference_is_robot_nominal'][:end] == (manifest['method'] == 'A-MCC'))
            sources.update(map(str, np.unique(z['input_normal_source'][:end])))
            states += end
    if manifest['method'] == 'A-MCC':
        assert sources <= {'A_ROBOT_NOMINAL', 'MEASURED_RAW', 'MEASURED_BELLY'}
    else:
        assert sources <= {'B_SURFACE_PLANNING', 'MEASURED_RAW', 'MEASURED_BELLY'}
    result = dict(status='PASSED_RECORDED_INPUT_SCHEMA_CHECK', method=manifest['method'],
                  native_states=states, public_wrist_poses=references, normal_sources=sorted(sources),
                  scope='Recorded schema/provenance plus source boundary tests; private oracle and public method run in the same process',
                  air_reference_note='JSONL preserves nominal transfer pose sequences and durations; per-tick public_wrist_pose_world is last surface request during air transfer',
                  normal_source_field_scope='Surface MCC selector only; during air moves this is the last surface selector context. Air force relief reads current raw measured normals directly, as in the shared source.')
    write(case/'input_boundary_record.json', result)
    return result


def recovery_metrics(case):
    data = native_arrays(case); n = len(data['time_s']); dt = .002
    executed_s = read(case/'summary.json')['executed_s']
    live = data['safe_belly'].any(axis=1)
    counts = data['safe_belly'].sum(axis=1)
    cumulative = np.r_[0, np.cumsum(counts >= 2)]
    stable = np.zeros(n, dtype=bool)
    if n >= 150:
        stable[149:] = ((cumulative[150:]-cumulative[:-150]) >= 120) & (counts[149:] >= 2)
    stable_ids = np.flatnonzero(stable)
    edges = np.diff(np.r_[False, ~live, False].astype(int))
    starts = np.flatnonzero(edges == 1); ends = np.flatnonzero(edges == -1)
    episodes = []
    for start, end in zip(starts, ends):
        later = stable_ids[stable_ids >= end]
        recovered = end < n
        episodes.append(dict(
            start_s=float(data['time_s'][start]),
            first_recontact_s=float(data['time_s'][end]) if recovered else None,
            gap_s=float(max(0., min(executed_s, end*dt)-data['time_s'][start])), initial_acquisition=bool(start == 0),
            endpoint_censored=bool(not recovered),
            first_recontact_count=int(counts[end]) if recovered else None,
            next_stable_support_s=float(data['time_s'][later[0]]) if len(later) else None,
            added_vertices_within_1s_of_recontact=(int(data['visited_vertex_count'][min(n-1, end+500)]
                                                     -data['visited_vertex_count'][max(0, start-1)]) if recovered else None),
        ))
    lost = [item for item in episodes if not item['initial_acquisition']]
    result = dict(definition='Whole hand: zero safe central palmar contacts; initial acquisition separate; no debounce removes gaps',
                  stable_support='Current >=2, >=2 for80% of preceding300ms',
                  whole_hand_loss_episodes=len(lost), recovered_episodes=sum(not item['endpoint_censored'] for item in lost),
                  terminal_censored_episodes=sum(item['endpoint_censored'] for item in lost),
                  longest_whole_hand_gap_s=max((item['gap_s'] for item in episodes), default=0.),
                  whole_time=contact_summary(data), episodes=episodes)
    write(case/'recovery_metrics.json', result)
    return result


def render(case, speed=12., start_s=None, end_s=None):
    install_asset_adapter()
    case, checked, summary, data, metrics = load_verified(case)
    method = read(case/'manifest.json')['method']
    c = _native_context(case); m, d = c.model, c.data; pads = c.handles.tip_geom_ids
    options = mujoco.MjvOption(); options.geomgroup[3:] = 0; options.sitegroup[:] = 0
    camera = mujoco.MjvCamera(); camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    renderer = mujoco.Renderer(m, height=540, width=800)
    m.vis.headlight.ambient[:] = [.25, .25, .25]; m.vis.headlight.diffuse[:] = [.45, .45, .45]
    rgba = m.geom_rgba.copy(); matid = m.geom_matid.copy()
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, geom) or '' for geom in range(m.ngeom)]
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 20)
    small = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 18)
    title = ImageFont.truetype('C:/Windows/Fonts/msyhbd.ttc', 24)
    full = start_s is None and end_s is None
    start = 0. if start_s is None else max(0., start_s)
    end = summary['executed_s'] if end_s is None else min(summary['executed_s'], end_s)
    if speed <= 0 or end <= start:
        raise ValueError('Positive speed and a nonempty time interval are required')
    selected = np.clip(np.searchsorted(data['time_s'], np.arange(start, end, speed/20)),
                       0, len(data['time_s'])-1).tolist()
    last = min(np.searchsorted(data['time_s'], end, side='right')-1, len(data['time_s'])-1)
    if selected[-1] != last:
        selected.append(last)
    scope_tag = 'complete' if full else f'segment_{start:g}-{end:g}s'
    stem = f'i04_{method.lower()}_{scope_tag}_{speed:g}x'
    video = case/(stem+'.mp4')
    writer = imageio.get_writer(video, fps=20, codec='libx264', quality=7, macro_block_size=2,
                                output_params=['-threads', '2'])
    captures = []
    labels = {'COOPERATIVE_APPROACH': '接近物体', 'COOPERATIVE_SLIDE': '沿表面推进',
              'FORCE_RELEASE': '主动卸力', 'COOPERATIVE_WITHDRAWAL': '撤离接触区',
              'NECESSARY_REORIENTATION': '离面调姿', 'COOPERATIVE_RECONTACT': '主动重接触',
              'CLEARANCE_EXIT': '间隙保护', 'FINAL_SAFE_HOLD': '完成后的安全保持'}
    try:
        for index, k in enumerate(selected):
            d.qpos[c.qpos] = data['q_rad'][k]; mujoco.mj_forward(m, d)
            centers = d.geom_xpos[pads].copy(); frames = d.geom_xmat[pads].reshape(4, 3, 3)
            normal = frames[:3, :, 2].mean(axis=0); normal /= np.linalg.norm(normal)
            canvas = Image.new('RGB', (2400, 700), '#122238')
            for panel in range(3):
                m.geom_rgba[:] = rgba; m.geom_matid[:] = matid
                for geom, name in enumerate(names):
                    if 'normal_marker' in name:
                        m.geom_rgba[geom, 3] = 0.
                    if name == 'mustard_visual':
                        m.geom_matid[geom] = -1
                        m.geom_rgba[geom] = [.96, .72, .08, .12 if panel else 1.]
                if panel == 2:
                    camera_from_direction(camera, frames[3, :, 2]+.2*frames[3, :, 1], centers[3], .27)
                elif panel:
                    camera_from_direction(camera, normal+.65*frames[1, :, 1], centers.mean(axis=0), .44)
                else:
                    camera.lookat[:] = [.3, 0., .49]; camera.distance = 1.05
                    camera.azimuth = 130.; camera.elevation = -25.
                renderer.update_scene(d, camera=camera, scene_option=options)
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = not bool(panel)
                for finger in np.flatnonzero(data['safe_belly'][k]):
                    marker(renderer.scene, data['belly_contact_positions_world_m'][k, finger], .002, [1., .1, .55, 1.])
                picture = Image.fromarray(renderer.render()); draw = ImageDraw.Draw(picture)
                draw.rectangle((0, 0, 800, 70), fill='#122238')
                heading = (f'I04{method} · Mustard 实际轨迹', '三根长指 · 仅物体透明', '拇指 · 实际关节状态')[panel]
                draw.text((12, 7), heading, font=title, fill='white')
                draw.text((12, 42), f't={data["time_s"][k]:.2f}s · 安全指腹 {int(data["safe_belly"][k].sum())}/4',
                          font=font, fill='#a7dbf0')
                canvas.paste(picture, (800*panel, 0))
            draw = ImageDraw.Draw(canvas)
            phase = labels.get(str(data['phase'][k]), str(data['phase'][k]))
            timeline_label = '完整仿真时间轴' if full else f'区间节选 {start:g}–{end:g}s'
            draw.text((15, 548), f'{phase} | {speed:g}×{timeline_label} | 实测测地覆盖 {100*data["coverage"][k]:.2f}% | 独立原生重放通过',
                      font=font, fill='white')
            for i, name in enumerate(('食指', '中指', '无名指', '拇指')):
                q = np.degrees(data['q_rad'][k, 7+4*i:11+4*i]); x = 15+1190*(i%2); y = 582+32*(i//2)
                text = f'{name} 关节° [{q[0]:5.1f}, {q[1]:5.1f}, {q[2]:5.1f}, {q[3]:5.1f}]  {data["forces_n"][k,i]:.2f} N'
                text += '  合格指腹接触' if data['safe_belly'][k, i] else '  当前无合格指腹接触'
                draw.text((x, y), text, font=small, fill='#a7edba' if data['safe_belly'][k, i] else '#d7e1ea')
            scope = ('A：仅腕参考＋固定名义手形＋四指MCC。' if method == 'A-MCC'
                     else 'B：腕与逐指表面参考＋四指MCC。')
            scope += ('包含全部离面、卸力、恢复；粉点仅为实际合格接触。' if full else '本片段无时间跳转，区间之外未展示；完整轨迹另附。')
            draw.text((15, 660), scope, font=small, fill='#f3d38d')
            writer.append_data(np.asarray(canvas))
            if index in {0, len(selected)//3, 2*len(selected)//3, len(selected)-1}:
                filename = f'{stem}_{index:05d}.png'; canvas.save(case/filename)
                captures.append(dict(file=filename, time_s=float(data['time_s'][k])))
            if index%250 == 0:
                print('RENDER', method, index, len(selected), flush=True)
    finally:
        writer.close(); renderer.close()
    result = dict(video=video.name, speed=speed, fps=20, frames=len(selected), full_time_axis=full,
                  time_range_s=[float(data['time_s'][selected[0]]), float(data['time_s'][selected[-1]])],
                  actual_saved_q_only=True, no_physics_step=True, captures=captures,
                  video_sha256=sha256(video.read_bytes()).hexdigest(),
                  source_manifest_sha256=checked['source_files']['manifest.json'])
    write(case/('video_manifest.json' if full else f'video_{stem}.json'), result)
    source = case/'visualization_source'; source.mkdir(exist_ok=True)
    (source/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    return result


STYLE = '''body{margin:0;background:#edf2f6;color:#17283b;font:17px/1.7 system-ui,"Microsoft YaHei",sans-serif}
main{max-width:1440px;margin:auto;padding:32px}section{background:white;border-radius:14px;padding:24px;margin:22px 0}
video,img{width:100%;border-radius:8px}table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:10px;border-bottom:1px solid #dbe3ec}
.note{color:#526779;font-size:15px}a{color:#1768a8}code{font-size:14px}'''


def stop_text(reason):
    return {'COMPLETION_SEARCH_EXHAUSTED': '连续16轮没有新增覆盖，搜索停止',
            'SUSTAINED_DANGEROUS_FORCE': '接触力持续超过12N达到20ms，触发高力保护停止',
            'SIMULATION_BUDGET': '到达仿真时间上限',
            'DEVELOPMENT_STOP_REQUESTED': '外部停止请求（需查阅对应记录）',
            'FULL_MEASURED_BELLY_SURFACE': '整瓶覆盖和最终安全保持完成'}.get(reason, reason)


def report(case):
    case, checked, summary, data, metrics = load_verified(case)
    manifest = read(case/'manifest.json'); method = manifest['method']
    input_boundary_record(case)
    covered = coverage_figure(case); contact_path_figure(case); timeline_figure(case, checked, summary)
    forces = verified_force_excursions(case, checked, summary); recovery = recovery_metrics(case)
    rows = ''.join(f'<tr><td>{r["region"]}</td><td>{r["covered"]}/{r["vertices"]}</td><td>{100*r["area_fraction"]:.2f}%</td></tr>'
                   for r in covered['regions'])
    fingers = ''.join(f'<tr><td>{name}</td><td>{100*metrics["all_time_finger_contact_fraction"][i]:.2f}%</td>'
                      f'<td>{metrics["actual_tangential_material_motion_m"][i]:.3f}m</td></tr>'
                      for i, name in enumerate(('食指', '中指', '无名指', '拇指')))
    videos = ''
    if (case/'video_manifest.json').exists():
        video = read(case/'video_manifest.json')
        assert sha256((case/video['video']).read_bytes()).hexdigest() == video['video_sha256']
        videos = f'<section><h2>完整实际时间轴 · {video["speed"]:g}倍速</h2><video controls preload="metadata" src="{video["video"]}"></video></section>'
    for path in sorted(case.glob('video_i04_*_segment_*.json')):
        video = read(path)
        assert sha256((case/video['video']).read_bytes()).hexdigest() == video['video_sha256']
        begin, end = video['time_range_s']
        videos += f'<section><h2>区间节选 {begin:.2f}–{end:.2f}s · {video["speed"]:g}倍速</h2><video controls preload="metadata" src="{video["video"]}"></video><p class="note">本片段内部无时间跳转；完整轨迹另附。</p></section>'
    method_text = ('A只接收手掌位姿。手指名义姿态固定为公共初态手形，四指继续接受实测MCC、正常屈指先验和共同安全保护。'
                   if method == 'A-MCC' else 'B接收同源手掌目标和各指表面位置、法向，联合规划手掌与手指，其后使用同一MCC和保护。')
    status = '已完成整瓶覆盖' if checked['full_surface_complete'] else '未完成整瓶覆盖'
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>I04{method} · Mustard 对照运行</title><style>{STYLE}</style><main><h1>I04{method} · Mustard 对照运行</h1>
<p><b>{status}</b>：面积指标{100*checked['area_fraction']:.3f}%，{checked['covered_vertices']}/8192顶点，{checked['covered_goals']}/388目标。</p>
<p>{method_text}</p><p>从零覆盖、单一控制版本运行{summary['executed_s']:.3f}秒；预算{manifest['requested_duration_s']:g}秒。终止：<b>{html.escape(stop_text(summary['stop_reason']))}</b>。</p>
<p class="note">物体固定、零重力、Mustard 2.8倍缩放；MCC，不含DP。短时失联允许恢复。物理核验通过与遍历完成是两项独立结论。
这是离线规划仿真：求解时暂停仿真计时。运行墙钟耗时{summary['wall_elapsed_s']:.1f}秒（不含后续独立核验/渲染），不是实时控制性能。</p>{videos}
<section><h2>覆盖与剩余区域</h2><img src="coverage_verified.png"><table><tr><th>区域</th><th>顶点</th><th>面积指标</th></tr>{rows}</table>
<p class="note">25mm网格边测地邻域，仅合格的实际中央指腹接触计分；分母和规则与B一致。不是实际皮肤压痕面积。</p><img src="contact_paths_verified.png"></section>
<section><h2>各指参与</h2><table><tr><th>手指</th><th>全程合格接触率</th><th>实际切向材料运动</th></tr>{fingers}</table>
<p>至少两指同时接触{100*recovery['whole_time']['two_fraction']:.2f}%；至少三指{100*recovery['whole_time']['three_fraction']:.2f}%；四指{100*recovery['whole_time']['four_fraction']:.2f}%。</p>
<p class="note">全程口径包含接近、换面和恢复。四指控制开启不代表四指始终接触。</p></section>
<section><h2>离面、恢复和安全</h2><img src="timeline_verified.png"><p>换面计划{summary['necessary_lifts']}次；全手失联{recovery['whole_hand_loss_episodes']}段，重触{recovery['recovered_episodes']}段，终点未恢复{recovery['terminal_censored_episodes']}段。
最长全手缺口（含初次接近）{recovery['longest_whole_hand_gap_s']:.3f}秒。原始力峰{forces['peak_n']:.4f}N，≥8N共{forces['total_s']:.3f}秒，均不计覆盖。</p>
<p>{checked['native_states']}个原生状态从唯一初态连续独立重放通过；全部已执行命令完成实态与上一参考双几何检查。</p></section>
<section><h2>可复核文件</h2><p><a href="manifest.json">配置和输入边界</a> · <a href="summary.json">终态</a> · <a href="nominal_wrist_reference.jsonl">手掌名义参考</a> · <a href="trace_chunks.json">完整2ms轨迹清单</a> · <a href="independent_native_audit/audit.json">独立核验</a> · <a href="recovery_metrics.json">每段失联与恢复</a> · <a href="whole_hand_metrics.json">接触和姿态指标</a> · <a href="events.json">规划与换面事件</a></p></section></main></html>'''
    (case/'review.html').write_text(page, encoding='utf-8')
    return str(case/'review.html')


def compare(a, b, output):
    a, b, output = Path(a).resolve(), Path(b).resolve(), Path(output).resolve()
    ma, mb = read(a/'manifest.json'), read(b/'manifest.json')
    for key in ('protocol', 'shared_config_sha256', 'initial_state_sha256', 'shared_source_hashes'):
        assert ma[key] == mb[key], key
    normalized_models = [(case/'model.xml').read_text(encoding='utf-8').replace(str(case), 'CASE_ROOT')
                         for case in (a, b)]
    assert normalized_models[0] == normalized_models[1], 'native model mismatch'
    assert ma['method'] == 'A-MCC' and mb['method'] == 'B-MCC'
    audits = [read(case/'independent_native_audit/audit.json') for case in (a, b)]
    assert all(item['status'] == 'PASSED_NATIVE_EVIDENCE_AUDIT' and item['fresh_single_version'] for item in audits)
    for case, checked in zip((a, b), audits):
        for name, digest in checked['source_files'].items():
            assert sha256((case/name).read_bytes()).hexdigest() == digest
    summaries = [read(case/'summary.json') for case in (a, b)]
    common = min(item['executed_s'] for item in summaries)
    values = [contact_summary(native_arrays(case, through_s=common)) for case in (a, b)]
    result = dict(protocol=ma['protocol'], comparison='One-seed matched runtime trial, not a multi-seed conclusion',
                  common_prefix_s=common, common_budget_s=ma['requested_duration_s'],
                  same_initial_state=True, same_shared_config=True, same_shared_source=True, same_native_model=True,
                  A=dict(case=str(a), terminal=summaries[0], common_prefix=values[0]),
                  B=dict(case=str(b), terminal=summaries[1], common_prefix=values[1]))
    output.mkdir(parents=True, exist_ok=True); write(output/'comparison.json', result)
    rows = ''
    for label, key, scale, suffix in [('覆盖面积指标', 'area_fraction', 100, '%'), ('覆盖顶点', 'covered_vertices', 1, ''),
                                     ('至少一指', 'any_fraction', 100, '%'), ('至少两指', 'two_fraction', 100, '%'),
                                     ('至少三指', 'three_fraction', 100, '%'), ('四指', 'four_fraction', 100, '%'),
                                     ('原始力峰', 'raw_peak_n', 1, 'N')]:
        rows += f'<tr><td>{label}</td>'+''.join(f'<td>{item[key]*scale:.3f}{suffix}</td>' for item in values)+'</tr>'
    import os
    links = [Path(os.path.relpath(case/'review.html', output)).as_posix() for case in (a, b)]
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>I04A/B MCC 对照</title><style>{STYLE}</style>
<main><h1>I04A/B · MCC 首次新协议对照</h1><p>同一物理初态、公共配置、执行代码与7000秒预算；各自从零覆盖。A仅腕参考，B增加逐指几何规划。DP暂缓。</p>
<p class="note">本页为单种子实现试验，不代表三种子统计结论。旧run10不进入此表。两组采用相同目标选择规则，但实际接触与覆盖不同后，目标序列允许不同。</p>
<section><h2>共同时间前缀：0–{common:.3f}秒</h2><table><tr><th>指标</th><th>A-MCC</th><th>B-MCC</th></tr>{rows}</table>
<p class="note">接触比例使用全部原生观测，含初次接近、离面和恢复；统计不剪掉失败片段。</p></section>
<section><h2>各自完整运行</h2><p><a href="{links[0]}">A：完整视频、覆盖、恢复、原生轨迹</a> · <a href="{links[1]}">B：完整视频、覆盖、恢复、原生轨迹</a></p>
<p>A：{summaries[0]['executed_s']:.3f}s，{100*summaries[0]['coverage_fraction']:.3f}%，{html.escape(stop_text(summaries[0]['stop_reason']))}。</p>
<p>B：{summaries[1]['executed_s']:.3f}s，{100*summaries[1]['coverage_fraction']:.3f}%，{html.escape(stop_text(summaries[1]['stop_reason']))}。</p>
<p>覆盖增长、力与安全停止需要一起看，不能只凭覆盖率判断整体方法成功。完整物体是否完成以各案核验结果为准。</p>
<p class="note">规划求解暂停仿真计时；图表使用仿真时间，非实时墙钟性能。</p><p><a href="comparison.json">机器可读完整对照</a></p></section></main></html>'''
    (output/'review.html').write_text(page, encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command', required=True)
    for command in ('audit', 'render', 'report'):
        p = sub.add_parser(command); p.add_argument('--case', type=Path, required=True)
        if command == 'audit': p.add_argument('--workers', type=int, default=4)
        if command == 'render':
            p.add_argument('--speed', type=float, default=12.)
            p.add_argument('--start', type=float); p.add_argument('--end', type=float)
    p = sub.add_parser('compare'); p.add_argument('--a', required=True); p.add_argument('--b', required=True); p.add_argument('--output', required=True)
    args = parser.parse_args(); install_asset_adapter()
    if args.command == 'audit': result = audit(args.case, args.workers)
    elif args.command == 'render': result = render(args.case, args.speed, args.start, args.end)
    elif args.command == 'report': result = report(args.case)
    else: result = compare(args.a, args.b, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
