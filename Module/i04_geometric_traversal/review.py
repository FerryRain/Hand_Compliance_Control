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
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
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
    axes[0, 0].set(ylabel='Measured surface area coverage (%)', xlabel='Time after acquisition (s)')
    axes[0, 0].legend(fontsize=8, ncol=2)
    axes[0, 1].set(ylabel='Actual pad contact count', xlabel='Time after acquisition (s)', ylim=(-0.1, 4.2))
    axes[1, 0].set(ylabel='Largest physical pad force (N)', xlabel='Time after acquisition (s)')
    axes[1, 1].set(ylabel='Wrist nominal tracking error (mm)', xlabel='Time after acquisition (s)')
    axes[1, 0].axhline(manifest['config']['hard_force_n'], color='#991b1b', ls='--', lw=1)
    for ax in axes.ravel():
        ax.grid(alpha=0.18)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle(f"New I04 — MCC-only paired experiment ({result['profile']})", fontsize=16)
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
    incomplete = any(not r['full_requested_budget'] for r in rows)
    note = ('部分 episode 被硬件保护提前停止，执行时间不足共同预算。指标只对应已执行的时间，不能作等时长优势结论。'
            if incomplete else '所有 episode 均运行到共同时间预算；时间预算完成不表示整只物体覆盖完成。')
    text = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>新 I04 MCC 对照实验</title>
    <style>body{{font-family:system-ui,"Microsoft YaHei",sans-serif;background:#f6f8fb;color:#182235;max-width:1240px;margin:36px auto;padding:0 22px}}
    h1{{font-size:30px}}p{{line-height:1.65}}.card{{background:white;padding:22px;border:1px solid #dde3ec;border-radius:14px;margin:18px 0}}
    table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:10px;text-align:left;border-bottom:1px solid #e5e9f0}}th{{background:#f0f4f9}}img{{max-width:100%}}a{{color:#2563eb}}code{{font-size:12px}}video{{width:100%}}</style>
    <h1>新 I04 · MCC 两组对照</h1><p>A-MCC：只给手腕 SE(3) 目标，手指被动法向柔顺。B-MCC：额外给四指几何目标，传统整手规划＋相同 MCC。DPRef 按用户要求延后。</p>
    <div class="card"><b>结果范围</b><p>{note}</p><p>固定已知 Bunny，gravity-off，23 DoF；真实 pad contact 的 geodesic 面积并集。未调用 contact-mode search 或 last-contact veto。</p></div>
    <div class="card"><img src="comparison.png" alt="contact and coverage comparison"></div>
    <div class="card" style="overflow-x:auto"><table>{table}</table></div>
    <div class="card"><b>输入与复现</b><p>Seeds：{manifest['seeds']}。总预算：{manifest['config']['duration_s']} s，其中共用 acquisition {manifest['config']['acquisition_s']} s。</p>
    <p><a href="manifest.json">环境、配置和源码 hashes</a> · <a href="summary.json">逐集 summary</a> · <a href="aggregate.json">组内均值与范围</a></p>
    <p>本报告描述实际仿真结果。未完成完整表面覆盖时，不将局部接触或轨迹执行记作全任务成功。</p></div></html>'''
    (directory / 'index.html').write_text(text, encoding='utf-8')
    print(json.dumps(aggregate, indent=2), flush=True)
    return aggregate


def render_pair(directory, seed=7, speed=5.0, fps=24):
    """FK-only replay of saved q; never recompute acceptance metrics."""
    import mujoco
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont
    from Module.fr3_leap.model import compile_model_xml
    from .physics import set_joint_state
    directory = Path(directory)
    from .physics import make_plant
    h = make_plant(directory / 'canonical_bunny.obj', 0.002)
    renderer = mujoco.Renderer(h.model, height=480, width=640)
    data = mujoco.MjData(h.model)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = h.object_position_m + np.array([0, 0, 0.08])
    camera.distance, camera.azimuth, camera.elevation = 0.72, 130, -28
    traces = [np.load(directory / f'{cell}_seed{seed}/trace.npz') for cell in ('A-MCC', 'B-MCC')]
    durations = [float(t['time_s'][-1]) for t in traces]
    duration = max(durations)
    video = directory / f'paired_seed{seed}.mp4'
    writer = imageio.get_writer(video, fps=fps, codec='libx264', quality=7, macro_block_size=16)
    font_path = Path('C:/Windows/Fonts/arial.ttf')
    font = ImageFont.truetype(str(font_path), 18) if font_path.exists() else ImageFont.load_default()
    for time in np.arange(0, duration + 1e-9, speed / fps):
        panels = []
        for cell, trace in zip(('A-MCC', 'B-MCC'), traces):
            i = min(int(np.searchsorted(trace['time_s'], time)), len(trace['time_s']) - 1)
            set_joint_state(h, data, trace['q_rad'][i])
            renderer.update_scene(data, camera=camera)
            panel = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(panel)
            draw.rectangle((0, 0, 640, 56), fill=(15, 23, 42))
            suffix = '  [stopped]' if time > trace['time_s'][-1] else ''
            draw.text((14, 6), f'{cell} | seed {seed} | actual t={trace["time_s"][i]:.2f}s{suffix}', font=font, fill='white')
            draw.text((14, 30), f'contacts={trace["active"][i].sum()}  measured coverage={100*trace["coverage"][i]:.2f}%', font=font, fill='white')
            panels.append(np.asarray(panel))
        writer.append_data(np.concatenate(panels, axis=1))
    writer.close()
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
    args = parser.parse_args()
    build_review(args.directory)
    if args.video:
        render_pair(args.directory, args.seed)


if __name__ == '__main__':
    main()
