"""Combine frozen mustard runs without recomputing or rewriting raw metrics."""
from __future__ import annotations

import argparse
import csv
import html
import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    for name in ('natural', 'challenge', 'hold_on', 'hold_off', 'output'):
        parser.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    runs = {}
    for name in ('natural', 'challenge', 'hold_on', 'hold_off'):
        directory = getattr(args, name).resolve()
        runs[name] = {'directory': str(directory), **{
            key: json.loads((directory / filename).read_text(encoding='utf-8'))
            for key, filename in (('manifest', 'manifest.json'), ('results', 'summary.json'),
                                  ('audit', 'integrity_audit.json'), ('aggregate', 'aggregate.json'))}}
        if runs[name]['audit']['status'] != 'VERIFIED':
            raise RuntimeError('Evidence needs verified runs')

    paired = []
    for seed in runs['hold_on']['manifest']['seeds']:
        on = next(r for r in runs['hold_on']['results']['episodes'] if r['seed'] == seed)
        off = next(r for r in runs['hold_off']['results']['episodes'] if r['seed'] == seed)
        if on['initial_state']['state_sha256'] != off['initial_state']['state_sha256']:
            raise RuntimeError('Hold ablation initial states differ')
        traces = [np.load(Path(runs[name]['directory']) / f'A-MCC_seed{seed}/trace.npz')
                  for name in ('hold_on', 'hold_off')]
        # Before withdrawal the treatment must be identical; check every
        # saved measurement, not just a matched initial-state digest.
        end = runs['hold_on']['manifest']['config']['withdrawal_at_s']
        mask = traces[0]['time_s'] < end
        for key in ('time_s', 'q_rad', 'command_rad', 'forces_n', 'active'):
            if not np.array_equal(traces[0][key][mask], traces[1][key][mask]):
                raise RuntimeError(f'Pre-withdrawal history differs: {key}, seed {seed}')
        paired.append({'seed': seed, 'initial_state_match': True, 'pre_withdrawal_history_match': True,
            'recovery_on_stable_count': on['stable_contact_recovery_count'],
            'recovery_off_stable_count': off['stable_contact_recovery_count'],
            'on_gap_episodes': on['contact_loss_episodes'], 'off_gap_episodes': off['contact_loss_episodes']})

    challenges = []
    for row in runs['challenge']['results']['episodes']:
        folder = Path(runs['challenge']['directory']) / f"{row['cell']}_seed{row['seed']}"
        events = json.loads((folder / 'events.json').read_text(encoding='utf-8'))
        start = next(e for e in events if e['kind'] == 'PHYSICAL_WITHDRAWAL_START')
        end = next(e for e in events if e['kind'] == 'PHYSICAL_WITHDRAWAL_END')
        episode = next(e for e in row['contact_loss_episodes'] if e['context'] == 'physical_withdrawal'
                       and e['recovered'] and e['stable_recovered_at_s'] > end['time_s'])
        trace = np.load(folder / 'trace.npz')
        indices = [int(np.argmin(abs(trace['time_s'] - e['time_s']))) for e in (start, end)]
        displacement = trace['palm_pose_world'][indices[1], :3] - trace['palm_pose_world'][indices[0], :3]
        projection = float(displacement @ np.asarray(start['direction_world']))
        if start['safe_contact_count_at_start'] < 1 or projection < .020:
            raise RuntimeError('Challenge did not start in contact or did not physically withdraw 20 mm')
        challenges.append({'cell': row['cell'], 'seed': row['seed'],
            'safe_contact_count_at_withdrawal_start': start['safe_contact_count_at_start'],
            'measured_outward_displacement_m': projection,
            'first_stable_after_release_at_s': episode['stable_recovered_at_s'],
            'stable_delay_after_release_s': episode['stable_recovered_at_s'] - end['time_s'],
            'final_persistent_gap_recovery_s': episode['recovery_time_s'],
            'additional_covered_area_fraction': episode['additional_covered_area_fraction']})

    rows = []
    for name, run in runs.items():
        for row in run['results']['episodes']:
            rows.append({'run': name, **row})
    columns = ['run', 'cell', 'seed', 'executed_post_acquisition_s', 'contact_continuity',
               'mean_contacts', 'final_covered_area_fraction', 'contact_loss_episode_count',
               'stable_contact_recovery_count', 'recovery_attempt_count', 'recovery_policy_success_count',
               'recovery_timeout_count', 'peak_force_n', 'stop_reason']
    with (output / 'episodes.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row[key] for key in columns} for row in rows)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    colors = ['#2563eb', '#ea580c']
    for panel, key, scale, ylabel in (
        (0, 'contact_continuity', 100, 'Measured contact continuity (%)'),
        (1, 'final_covered_area_fraction', 100, 'Full surface area covered (%)')):
        for group, name in enumerate(('natural', 'challenge')):
            for i, cell in enumerate(('A-MCC', 'B-MCC')):
                values = [scale*r[key] for r in runs[name]['results']['episodes'] if r['cell'] == cell]
                x = group * 3 + i
                axes[panel].bar(x, np.mean(values), color=colors[i], alpha=.8)
                axes[panel].scatter(np.full(len(values), x), values, color='#111827', s=22, zorder=3)
        axes[panel].set(xticks=[0, 1, 3, 4], xticklabels=['A\nNatural', 'B\nNatural', 'A\nWithdrawal', 'B\nWithdrawal'], ylabel=ylabel)
    for x, name in enumerate(('hold_on', 'hold_off')):
        successes = sum(any(e['recovered'] and e['context'] == 'physical_withdrawal'
                            for e in r['contact_loss_episodes']) for r in runs[name]['results']['episodes'])
        axes[2].bar(x, successes, color=['#16a34a', '#64748b'][x])
        axes[2].text(x, successes + .06, f'{successes}/3', ha='center')
    axes[2].set(xticks=[0, 1], xticklabels=['Recovery ON', 'Recovery OFF'],
                ylabel='Hold: stable recovery after real withdrawal', ylim=(0, 3.6), yticks=[0, 1, 2, 3])
    for ax in axes:
        ax.grid(axis='y', alpha=.15)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Mustard / MCC only / fixed paired seeds 7, 11, 19', fontsize=14)
    fig.savefig(output / 'overview.png', dpi=170)
    plt.close(fig)

    table = '| 运行 | 组 | Seed | treatment s | 接触率 | 覆盖 | 稳定恢复/失联 | 保护停止或预算 |\n'
    table += '| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |\n'
    body = []
    for r in rows:
        values = [r['run'], r['cell'], str(r['seed']), f"{r['executed_post_acquisition_s']:.2f}",
                  f"{100*r['contact_continuity']:.2f}%", f"{100*r['final_covered_area_fraction']:.3f}%",
                  f"{r['stable_contact_recovery_count']}/{r['contact_loss_episode_count']}", r['stop_reason']]
        table += '| ' + ' | '.join(values) + ' |\n'
        body.append('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in values)+'</tr>')
    links = ' · '.join(f'<a href="../{Path(run["directory"]).name}/index.html">{name}</a>' for name, run in runs.items())
    full_budget = sum(r['full_requested_budget'] for r in rows)
    recover_on = sum(p['recovery_on_stable_count'] > 0 for p in paired)
    recover_off = sum(p['recovery_off_stable_count'] > 0 for p in paired)
    hold_times = [e['recovery_time_s'] for p in paired for e in p['on_gap_episodes'] if e['recovered']]
    scope = (f'完成 {len(rows)} 次真实 native MuJoCo 运行，其中 {full_budget} 次达到预算。'
             f'隔离撤离实验：恢复层开启 {recover_on}/3 次稳定恢复，关闭 {recover_off}/3；'
             f'开启时稳定恢复耗时 {min(hold_times):.3f}–{max(hold_times):.3f} s。'
             '两侧初态及撤离前完整保存历史逐元素一致。')
    findings = []
    for name in ('natural', 'challenge'):
        for cell in ('A-MCC', 'B-MCC'):
            ag = runs[name]['aggregate'][cell]
            rec = ag['recovery']
            findings.append(f"{name} / {cell}：平均接触率 {100*ag['contact_continuity']['mean']:.2f}%，"
                f"覆盖 {100*ag['final_covered_area_fraction']['mean']:.3f}%，"
                f"安全接触失联 {rec['safe_contact_gap_count']} 次、稳定恢复 {rec['stable_recovered_gap_count']} 次，"
                f"保护提前停止 {ag['episode_count']-ag['full_budget_episode_count']} 次。")
    for cell in ('A-MCC', 'B-MCC'):
        group = [r for r in challenges if r['cell'] == cell]
        values = [r['final_persistent_gap_recovery_s'] for r in group]
        findings.append(f"撤离后持续失联段 / {cell}：稳定恢复 {min(values):.3f}–{max(values):.3f} s，"
            f"恢复后到终点新增覆盖 {100*min(r['additional_covered_area_fraction'] for r in group):.3f}–"
            f"{100*max(r['additional_covered_area_fraction'] for r in group):.3f} 个百分点。")
    findings.append('A 当前以最近两指稳定接触的历史作为恢复参考；后段仅一指接触时，该参考会较旧。'
                    '恢复速度和更多手指的参与仍需改进，四指同时接触比例在本轮均为 0。')
    limitations = ('失联次数包括短暂 gap；稳定恢复要求至少一指连续正常接触 150 ms。'
        '恢复后运动和新增覆盖在各次运行报告逐事件列出；其后缀可能重叠，不能相加。'
        'wrist 路程包含细微接触振荡，探索推进以新增实测覆盖为据。'
        '撤离挑战若开始时已无接触，不能算作撤离造成失联。'
        '未完成完整 mustard 覆盖；3 seeds、固定物体、gravity-off、GT 离线规划的结果不外推到实机。')
    video_path = Path(runs['hold_on']['directory']) / 'recovery_control_seed7.mp4'
    video_card = (f'<div class="card"><b>恢复开关对照回放 · seed 7 · 5×</b><p>同一初态、相同真实撤离、无自动返回；画面来自保存的实际关节状态。mustard 表面统一黄色便于识别。</p><video style="width:100%" controls src="../{video_path.parent.name}/{video_path.name}"></video></div>'
                  if video_path.exists() else '')
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>mustard MCC 接触恢复</title>
    <style>body{{font-family:system-ui,"Microsoft YaHei",sans-serif;max-width:1300px;margin:30px auto;padding:0 20px;background:#f6f8fb;color:#182235}}p{{line-height:1.7}}.card{{background:white;border:1px solid #dde3ec;border-radius:12px;padding:22px;margin:20px 0}}img{{max-width:100%}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:9px;border-bottom:1px solid #e5e9f0;text-align:left}}a{{color:#2563eb}}</style>
    <h1>mustard · MCC 接触丢失与恢复</h1><p>{scope}</p><p>DPRef 未运行。自然探索、撤离挑战和固定位置的恢复层消融分别报告。</p>
    <div class="card"><img src="overview.png" alt="mustard MCC and recovery results"><p>{'<br>'.join(findings)}</p></div>
    <div class="card"><b>接触可以中断，恢复需要证据</b><p>全指失联 100 ms 后，由共享恢复层返回最近安全接触附近，保留指令预载，再沿实测法向作最多 3 mm 局部探触。
    稳定接触恢复后重新规划；8 s 超时继续实验。≥8 N 局部卸力；持续 >12 N/20 ms 及真实非指尖碰撞等仍独立保护。</p><p>{limitations}</p></div>
    <div class="card" style="overflow:auto"><table><tr>{''.join('<th>'+v+'</th>' for v in ['运行','组','Seed','treatment s','接触率','覆盖','稳定恢复/失联','停止原因'])}</tr>{''.join(body)}</table></div>
    {video_card}
    <div class="card"><b>原始证据与回放</b><p>{links}</p><p><a href="episodes.csv">全部 episode 数据</a> · <a href="hold_ablation_audit.json">恢复层隔离比较审计</a> · <a href="evidence.json">配置、结果和审计汇总</a></p></div></html>'''
    (output / 'index.html').write_text(page, encoding='utf-8')
    payload = {'status': 'EVALUATED / FULL_COVERAGE_INCOMPLETE', 'runs': runs,
               'hold_ablation_pairs': paired, 'withdrawal_diagnostics': challenges, 'limitations': limitations,
               'analysis_code_hashes': {name: sha256((ROOT / name).read_bytes()).hexdigest() for name in (
                   'scripts/record_mustard_recovery_evidence.py', 'Module/i04_geometric_traversal/review.py')}}
    (output / 'evidence.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    (output / 'hold_ablation_audit.json').write_text(json.dumps(paired, indent=2), encoding='utf-8')
    (output / 'withdrawal_diagnostics.json').write_text(json.dumps(challenges, indent=2), encoding='utf-8')
    evidence = ROOT / 'Module/evidence'
    md = f'''# 新 I04 mustard 接触恢复记录

日期：2026-10-13。用户指定 MCC-only、允许短暂失联、从 mustard 开始。
状态：`EVALUATED / FULL_COVERAGE_INCOMPLETE`，DPRef：`DEFERRED_BY_USER`。

{scope}

{chr(10).join(findings)}

{table}

## 方法和局限

两组共享 Basic MCC + SharedContactRecovery。A 仅获取 wrist SE(3)，B 额外获取四指几何位置。
恢复层只读取实测位置、实际下发 command、接触和力的因果历史；不读取 Oracle 或撤离时刻。
每个 seed 共用 5 s acquisition 的 q/qvel/ctrl snapshot。mustard scale 2.8 已烘焙，runtime=1；
使用既有 256-part V-HACD，完整 source mesh 的三个连通分量、388 必访点全部保留。
总表面面积 0.356718412 m²；覆盖只来自真实 pad contacts 的 geodesic union。

全指安全接触缺失不终止运行。100 ms 去抖、150 ms 稳定确认、8 s 恢复 timeout 后继续。
≥8 N force relief 在消融两侧仍保留，持续 >12 N/20 ms 等独立保护不放宽。
traversal 的自然和 t=8 s 撤离各运行 60 s 总预算；hold 隔离实验 25 s、t=4 s 撤离，
撤离后保持撤离位置、无自动返回、无 Oracle、不作遍历比较。

{limitations}

运行前 28 项接口、恢复语义、指令推进、覆盖、FR3/MCC 回归检查通过。
各运行 source/manifest/model/256 parts/trace hashes、paired snapshots、因果恢复参考、稳定实测接触、
恢复事件记账与 A/B 信息边界审计均为 VERIFIED。hold 开/关撤离前完整保存历史逐元素相同。
MuJoCo native CPU physics；GPU OpenGL 仅用于记录回放；离线同步规划的 wall latency 单独保存。

冻结协议：[`../I04_MUSTARD_RECOVERY_PROTOCOL.md`](../I04_MUSTARD_RECOVERY_PROTOCOL.md)。
本机汇总报告：[`../generated/{output.name}/index.html`](../generated/{output.name}/index.html)。
运行 commit：`{runs['natural']['manifest']['source_commit']}`；每次 manifest 保存配置、环境和源码 hashes。
旧 Bunny v1 证据保留在 `2026-10-13_NEW_I04_MCC.md`，本轮不覆盖历史记录。
'''
    (evidence / '2026-10-13_NEW_I04_MUSTARD_RECOVERY.md').write_text(md, encoding='utf-8')
    (evidence / '2026-10-13_NEW_I04_MUSTARD_RECOVERY.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(scope)
    print('\n'.join(findings))
    print('Review:', output / 'index.html')


if __name__ == '__main__':
    main()
