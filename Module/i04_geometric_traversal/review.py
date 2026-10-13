"""Figures, transparent episode tables, and replay of recorded MCC traces."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS = {'A-MCC': '#2563eb', 'B-MCC': '#ea580c'}


def build_review(directory):
    directory = Path(directory)
    result = json.loads((directory / 'summary.json').read_text(encoding='utf-8'))
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    rows = result['episodes']
    recovery_run = 'recovery' in manifest
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    for row in rows:
        trace = np.load(directory / f"{row['cell']}_seed{row['seed']}" / 'trace.npz')
        cell = row['cell']
        color = COLORS[cell]
        label = f"{cell}, seed {row['seed']}"
        time = trace['time_s']
        axes[0, 0].plot(time, 100 * trace['coverage'], color=color, alpha=0.7, label=label)
        axes[0, 1].plot(time, trace['active'].sum(axis=1), color=color, alpha=0.45)
        axes[1, 0].plot(time, trace['forces_n'].max(axis=1), color=color, alpha=0.55)
        axes[1, 1].plot(time, 1000 * trace['wrist_tracking_m'], color=color, alpha=0.55)
        axes[0, 2].plot(time, np.cumsum(np.r_[0., np.linalg.norm(
            np.diff(trace['palm_pose_world'][:, :3], axis=0), axis=1)]), color=color, alpha=.55)
        if 'recovery_mode' in trace:
            modes = np.array([{'NORMAL': 0, 'RECOVER': 1, 'FORCE_RELEASE': 2}[m]
                              for m in trace['recovery_mode']])
            axes[1, 2].step(time, modes, color=color, alpha=.5, where='post')
    axes[0, 0].set(ylabel='Measured surface area coverage (%)', xlabel='Time after acquisition (s)')
    axes[0, 0].legend(fontsize=8, ncol=2)
    axes[0, 1].set(ylabel='Actual pad contact count', xlabel='Time after acquisition (s)', ylim=(-0.1, 4.2))
    axes[1, 0].set(ylabel='Largest physical pad force (N)', xlabel='Time after acquisition (s)')
    axes[1, 1].set(ylabel='Wrist nominal tracking error (mm)', xlabel='Time after acquisition (s)')
    axes[0, 2].set(ylabel='Actual wrist path length (m)', xlabel='Time after acquisition (s)')
    axes[1, 2].set(ylabel='Shared recovery state', xlabel='Time after acquisition (s)',
                      yticks=[0, 1, 2], yticklabels=['Normal', 'Recover', 'Force relief'])
    axes[1, 0].axhline(manifest['config']['hard_force_n'], color='#991b1b', ls='--', lw=1)
    for ax in axes.ravel():
        ax.grid(alpha=0.18)
        ax.spines[['top', 'right']].set_visible(False)
    object_name = manifest['config'].get('object_name', 'bunny')
    task = manifest['config'].get('task', 'traverse')
    policy = 'ON' if manifest.get('recovery', {}).get('enabled', False) else 'OFF'
    fig.suptitle(f"New I04 — {object_name} / {task} / MCC / recovery {policy}", fontsize=16)
    fig.savefig(directory / 'comparison.png', dpi=160)
    plt.close(fig)
    aggregate = {}
    for cell in manifest['cells']:
        group = [r for r in rows if r['cell'] == cell]
        aggregate[cell] = {key: {'mean': float(np.mean([r[key] for r in group])),
                                'min': float(min(r[key] for r in group)),
                                'max': float(max(r[key] for r in group))}
                            for key in ('contact_continuity', 'mean_contacts', 'four_contact_fraction',
                                        'final_covered_area_fraction', 'maximum_all_contact_gap_s',
                                        'peak_force_n', 'executed_post_acquisition_s')}
        aggregate[cell]['full_budget_episode_count'] = sum(r['full_requested_budget'] for r in group)
        aggregate[cell]['episode_count'] = len(group)
        if recovery_run:
            losses = [e for r in group for e in r['contact_loss_episodes']]
            recovered = [e for e in losses if e['recovered']]
            aggregate[cell]['recovery'] = {
                'safe_contact_gap_count': len(losses), 'stable_recovered_gap_count': len(recovered),
                'unrecovered_gap_count': len(losses) - len(recovered),
                'recovery_time_mean_s': float(np.mean([e['recovery_time_s'] for e in recovered])) if recovered else None,
                'recovery_time_max_s': max((e['recovery_time_s'] for e in recovered), default=None),
                'policy_attempt_count': sum(r['recovery_attempt_count'] for r in group),
                'policy_success_count': sum(r['recovery_policy_success_count'] for r in group),
                'policy_timeout_count': sum(r['recovery_timeout_count'] for r in group),
                'policy_in_progress_count': sum(r['recovery_in_progress_count'] for r in group),
                'contexts': {context: {'gaps': sum(e['context'] == context for e in losses),
                                      'stable_recoveries': sum(e['context'] == context for e in recovered)}
                             for context in ('natural_motion', 'physical_withdrawal')}}
    (directory / 'aggregate.json').write_text(json.dumps(aggregate, indent=2), encoding='utf-8')
    header = ['Cell', 'Seed', 'Executed (s)', 'Contact continuity', 'Mean contacts', '4-contact',
              'Area covered', 'Goals', 'Max zero-contact (s)', 'Peak force (N)', 'Stop']
    body = []
    for r in rows:
        body.append([r['cell'], str(r['seed']), f"{r['executed_post_acquisition_s']:.2f}",
                     f"{100*r['contact_continuity']:.2f}%", f"{r['mean_contacts']:.3f}",
                     f"{100*r['four_contact_fraction']:.2f}%", f"{100*r['final_covered_area_fraction']:.2f}%",
                     f"{r['visited_required_goals']}/{r['required_goal_count']}",
                     f"{r['maximum_all_contact_gap_s']:.3f}", f"{r['peak_force_n']:.3f}", r['stop_reason']])
    table = '<thead><tr>' + ''.join('<th>'+html.escape(x)+'</th>' for x in header) + '</tr></thead><tbody>'
    table += ''.join('<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in row)+'</tr>' for row in body)+'</tbody>'
    recovery_card = ''
    if recovery_run:
        gap_rows = []
        for r in rows:
            for e in r['contact_loss_episodes']:
                gap_rows.append([r['cell'], str(r['seed']), e['context'], f"{e['lost_at_s']:.3f}",
                    f"{e['first_touch_at_s']:.3f}" if e['first_touch_at_s'] is not None else '—',
                    f"{e['recovery_time_s']:.3f}" if e['recovered'] else f"未恢复 / 截尾 {e['observed_gap_s']:.3f}",
                    str(e.get('contact_count_after_recovery', '—')),
                    f"{100*e['additional_covered_area_fraction']:.3f}%" if e['recovered'] else '—',
                    f"{1000*e['wrist_path_after_recovery_m']:.1f}" if e['recovered'] else '—'])
        headings = ['组', 'Seed', '失联情境', '失联时刻 s', '首次触碰时刻 s', '稳定恢复耗时 s',
                    '恢复时接触数', '此后新增覆盖', '此后 wrist 路程 mm']
        gap_table = '<tr>' + ''.join('<th>'+h+'</th>' for h in headings) + '</tr>'
        gap_table += ''.join('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in r)+'</tr>' for r in gap_rows)
        if not gap_rows:
            gap_table += '<tr><td colspan="9">未观察到全指安全接触丢失。</td></tr>'
        recovery_card = f'''<div class="card"><b>失联与恢复（不是失败判决）</b>
        <p>正常接触力范围为 {manifest['config']['contact_threshold_n']}–{manifest['config']['force_relief_n']} N；
        至少一指连续接触 {manifest['config']['recovery_confirmation_s']} s 才记稳定恢复。
        单指丢失与重新接触另存于 summary。超时记录后继续运行。</p>
        <div style="overflow-x:auto"><table>{gap_table}</table></div>
        <p>“此后”指标覆盖从该次恢复到 episode 终点的整个后缀，可能包含再次失联或恢复，不能相加。
        短促触碰不当作稳定恢复；physical_withdrawal 只标记撤离期间开始的真实失联。</p></div>'''
    incomplete = any(not r['full_requested_budget'] for r in rows)
    note = ('部分 episode 被硬件保护提前停止，执行时间不足共同预算。指标只对应已执行的时间，不能作等时长优势结论。'
            if incomplete else '所有 episode 均运行到共同时间预算；时间预算完成不表示整只物体覆盖完成。')
    text = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>新 I04 MCC 对照实验</title>
    <style>body{{font-family:system-ui,"Microsoft YaHei",sans-serif;background:#f6f8fb;color:#182235;max-width:1240px;margin:36px auto;padding:0 22px}}
    h1{{font-size:30px}}p{{line-height:1.65}}.card{{background:white;padding:22px;border:1px solid #dde3ec;border-radius:14px;margin:18px 0}}
    table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:10px;text-align:left;border-bottom:1px solid #e5e9f0}}th{{background:#f0f4f9}}img{{max-width:100%}}a{{color:#2563eb}}code{{font-size:12px}}video{{width:100%}}</style>
    <h1>新 I04 · {object_name} 接触与恢复</h1><p>A-MCC：只给手腕 SE(3) 目标。B-MCC：额外给四指几何目标。
    两组使用相同 MCC，恢复层 {policy}。DPRef 按用户要求延后。</p>
    <div class="card"><b>结果范围 · {task}</b><p>{note}</p><p>固定已知 {object_name}，gravity-off，23 DoF；
    真实 pad contact 的 geodesic 面积并集，完整 {manifest['geometry']['required_goals']} 个必访点保留。
    未调用 legacy contact-mode search 或 last-contact veto。</p>
    <p>{'hold 是恢复层隔离验证：不使用 Oracle，撤离后保持撤离位置，只有恢复控制器可以主动回退；不作 A/B 遍历比较。' if task == 'hold' else 'traverse 使用相同几何规划框架。恢复层依据真实传感历史返回安全接触附近，沿实测法向有限探触；稳定恢复后重新规划继续探索。'}</p>
    <p>物理撤离挑战：{manifest.get('physical_loss_challenge', {}).get('enabled', False)}。
    即时丢失接触不会判失败；持续危险力及非指尖碰撞仍独立停止。规划是同步离线运行，不宣称硬件实时性。</p></div>
    <div class="card"><img src="comparison.png" alt="contact and coverage comparison"></div>
    <div class="card" style="overflow-x:auto"><table>{table}</table></div>
    {recovery_card}
    <div class="card"><b>输入与复现</b><p>Seeds：{manifest['seeds']}。总预算：{manifest['config']['duration_s']} s，其中共用 acquisition {manifest['config']['acquisition_s']} s。</p>
    <p><a href="manifest.json">环境、配置和源码 hashes</a> · <a href="summary.json">逐集 summary</a> · <a href="aggregate.json">组内均值与范围</a></p>
    <p>本报告描述实际仿真结果。未完成完整表面覆盖时，不将局部接触或轨迹执行记作全任务成功。</p></div></html>'''
    (directory / 'index.html').write_text(text, encoding='utf-8')
    print(json.dumps(aggregate, indent=2), flush=True)
    return aggregate


def render_pair(directory, seed=7, speed=5.0, fps=24, control_directory=None):
    """FK-only replay of saved q; never recompute acceptance metrics."""
    import mujoco
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont
    from Module.fr3_leap.model import compile_model_xml
    directory = Path(directory)
    from Module.fr3_leap.model import ARM_JOINT_NAMES, HAND_JOINT_NAMES
    model = compile_model_xml((directory / 'model.xml').read_text(encoding='utf-8'))
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
                 for n in (*ARM_JOINT_NAMES, *HAND_JOINT_NAMES)]
    addresses = model.jnt_qposadr[joint_ids]
    object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'fr3_e05_object')
    renderer = mujoco.Renderer(model, height=480, width=640)
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = model.body_pos[object_id] + np.array([0, 0.03, 0.10])
    camera.distance, camera.azimuth, camera.elevation = 0.90, 130, -28
    scenes = [(directory, 'A-MCC', 'A-MCC'), (directory, 'B-MCC', 'B-MCC')]
    if control_directory is not None:
        scenes = [(directory, 'A-MCC', 'Recovery ON'), (Path(control_directory), 'A-MCC', 'Recovery OFF')]
    traces = [np.load(path / f'{cell}_seed{seed}/trace.npz') for path, cell, _ in scenes]
    durations = [float(t['time_s'][-1]) for t in traces]
    duration = max(durations)
    video = directory / f"{'recovery_control' if control_directory else 'paired'}_seed{seed}.mp4"
    writer = imageio.get_writer(video, fps=fps, codec='libx264', quality=7, macro_block_size=16)
    font_path = Path('C:/Windows/Fonts/arial.ttf')
    font = ImageFont.truetype(str(font_path), 18) if font_path.exists() else ImageFont.load_default()
    for time in np.arange(0, duration + 1e-9, speed / fps):
        panels = []
        for (_, cell, label), trace in zip(scenes, traces):
            i = min(int(np.searchsorted(trace['time_s'], time)), len(trace['time_s']) - 1)
            data.qpos[addresses] = trace['q_rad'][i]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            panel = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(panel)
            draw.rectangle((0, 0, 640, 80), fill=(15, 23, 42))
            suffix = '  [stopped]' if time > trace['time_s'][-1] else ''
            draw.text((14, 6), f'{label} | seed {seed} | actual t={trace["time_s"][i]:.2f}s{suffix}', font=font, fill='white')
            draw.text((14, 30), f'contacts={trace["active"][i].sum()}  measured coverage={100*trace["coverage"][i]:.2f}%', font=font, fill='white')
            mode = trace['recovery_mode'][i] if 'recovery_mode' in trace else 'NORMAL'
            if 'perturbation_active' in trace and trace['perturbation_active'][i]:
                mode += ' / PHYSICAL WITHDRAWAL'
            draw.text((14, 54), f'state={mode}', font=font, fill='#bce7ff')
            panels.append(np.asarray(panel))
        frame = np.concatenate(panels, axis=1)
        writer.append_data(frame)
        if abs(time - 10) < speed / fps / 2:
            Image.fromarray(frame).save(directory / 'replay_preview.png')
    writer.close()
    from OpenGL import GL
    backend = {key: GL.glGetString(value).decode() for key, value in (
        ('vendor', GL.GL_VENDOR), ('renderer', GL.GL_RENDERER), ('version', GL.GL_VERSION))}
    (directory / 'render_backend.json').write_text(json.dumps(backend, indent=2), encoding='utf-8')
    renderer.close()
    path = directory / 'index.html'
    text = path.read_text(encoding='utf-8')
    text = text.replace('</html>', f'<div class="card"><b>同一 seed 的记录回放（{speed:g}×）</b><video controls src="{video.name}"></video><p>使用保存的关节状态进行 FK 回放；提前停止的组保持最后一帧并标注 stopped。</p></div></html>')
    path.write_text(text, encoding='utf-8')
    print(f'Replay: {video}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--control', type=Path, help='MCC recovery-OFF hold run for a paired recovery replay')
    args = parser.parse_args()
    build_review(args.directory)
    if args.video:
        render_pair(args.directory, args.seed, control_directory=args.control)


if __name__ == '__main__':
    main()
