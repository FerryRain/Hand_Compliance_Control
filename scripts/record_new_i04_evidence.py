"""Save reviewable evidence from the frozen run without changing its metrics."""
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    output = args.directory.resolve()
    result = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    audit = json.loads((output / 'integrity_audit.json').read_text(encoding='utf-8'))
    matched = json.loads((output / 'matched_time_diagnostics.json').read_text(encoding='utf-8'))
    aggregate = json.loads((output / 'aggregate.json').read_text(encoding='utf-8'))
    rows = result['episodes']
    columns = ['cell', 'seed', 'executed_post_acquisition_s', 'contact_continuity', 'mean_contacts',
               'four_contact_fraction', 'final_covered_area_fraction', 'visited_required_goals',
               'maximum_all_contact_gap_s', 'peak_force_n', 'stop_reason', 'full_requested_budget']
    with (output / 'episodes.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row[key] for key in columns} for row in rows)
    table = '| Cell | Seed | 实际 treatment s | 接触连续率 | 平均接触数 | 覆盖面积 | Goals | 最大零接触 s | 峰值 N |\n'
    table += '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n'
    for r in rows:
        table += (f"| {r['cell']} | {r['seed']} | {r['executed_post_acquisition_s']:.3f} | "
                  f"{100*r['contact_continuity']:.2f}% | {r['mean_contacts']:.3f} | "
                  f"{100*r['final_covered_area_fraction']:.3f}% | {r['visited_required_goals']}/{r['required_goal_count']} | "
                  f"{r['maximum_all_contact_gap_s']:.3f} | {r['peak_force_n']:.3f} |\n")
    common_mean = {cell: {key: float(np.mean([pair['cells'][cell][key] for pair in matched]))
                         for key in ('mean_contacts', 'four_contact_fraction', 'covered_area_fraction')}
                   for cell in ('A-MCC', 'B-MCC')}
    a, b = common_mean['A-MCC'], common_mean['B-MCC']
    report = f'''# 新 I04 MCC-only 首轮实验记录

日期：2026-10-13（Asia/Shanghai）。状态：`EVALUATED / FULL_COVERAGE_INCOMPLETE`。
用户授权新 I04，随后明确只运行 MCC。DPRef：`DEFERRED_BY_USER`。

## 结果与解释

完成 seeds 7/11/19 × A-MCC/B-MCC 共 6 次实际 MuJoCo 运行；6/6 均因
`SUSTAINED_DANGEROUS_FORCE` 提前停止。完整 60 s 总预算未完成，完整 Bunny coverage 未完成。
两组由同一 3 s acquisition 的 q/qvel/ctrl snapshot 启动，三个 seed 均为四指实际接触。

{table}

A-MCC 平均末段覆盖 {100*aggregate['A-MCC']['final_covered_area_fraction']['mean']:.3f}%；
B-MCC 为 {100*aggregate['B-MCC']['final_covered_area_fraction']['mean']:.3f}%。
这些末段覆盖来自不同执行时长，不能直接解释为等时长因果优势。

等时长前缀按每个 seed 的较早停止时间（5.418–5.998 s）截断，使用保存的 10 ms
samples 作独立描述性诊断：平均 contact 数 A/B 为 {a['mean_contacts']:.3f}/{b['mean_contacts']:.3f}；
四指比例 {100*a['four_contact_fraction']:.2f}%/{100*b['four_contact_fraction']:.2f}%；
面积覆盖 {100*a['covered_area_fraction']:.3f}%/{100*b['covered_area_fraction']:.3f}%。
这支持 B 的早期接触较丰富；其后仍存在长时间零接触及高力终止，不支持完整连续覆盖成功。

结论仅限当前已知 GT、固定物体、gravity-off、3 seeds 和冻结 controller/planner 配置。
不宣称 MCC 普遍不可行、DPRef 已有效、硬件可运行或未知物体能力。

## 方法边界与验证

- A 仅暴露 wrist SE(3)；B 增加四个 tip position。两组同一几何规划框架和 MCC 参数。
- 不 import M07/M11/M12 active search；无 desired contact occupancy、last-contact veto 或 contact task constraint。
- 只有真实 solver pad contact 增加 mesh-edge geodesic area union，274 required goals 分母完整保留。
- 规划同步离线运行，latency 按 wall clock 单独记录，不宣称实机实时频率。
- 21 项接口/coverage authority/MCC force-sign/FR3 与 MCC 回归检查通过。
- 独立审计：`{audit['status']}`；runtime hashes、trace hashes、paired snapshots、finite traces、
  [0,1] 接触比例及 A/B exposed fields 全部正确；完整覆盖成功数量为 0。

## 来源与产物

运行源码 commit：`{manifest['source_commit']}`；环境/配置/源码/物体/initial-state hashes 在 manifest。
Python 为 Windows Conda handcomp；MuJoCo 3.6.0 native CPU physics，GPU OpenGL 回放。
Windows 中文路径通过 VFS 读取二进制 meshes，不修改原机器人网格。

冻结协议：[`../I04_GEOMETRIC_MCC_PROTOCOL.md`](../I04_GEOMETRIC_MCC_PROTOCOL.md)。
本机报告：[`../generated/{output.name}/index.html`](../generated/{output.name}/index.html)。
同次运行保存每组 NPZ trace、events JSON、summary、CSV、aggregate、matched-time diagnostics、
integrity audit、comparison PNG 和 seed7 并排 recorded FK replay。
二进制仍遵守仓库 generated ignore；本证据 JSON 保存非二进制结果、输入和代码 provenance。

## 复现

`scripts/run_new_i04_mcc.ps1 -ExperimentArgs @('--profile','formal','--output','Module/generated/<fresh-directory>')`

`python -m Module.i04_geometric_traversal.review <same-directory> --video`

`python scripts/audit_new_i04_results.py <same-directory>`

`python -m unittest Module.tests.test_i04_geometric_traversal Module.tests.test_fr3_leap_model Module.tests.test_fingertip_mcc -v`

原 simulation_repo 的用户改动保留。本轮代码位于独立 checkout `i04_experiment_repo`，branch
`codex/new-i04-geometric-experiment`；结果未推送或伪装成已通过完整系统任务。
'''
    evidence = ROOT / 'Module/evidence'
    (evidence / '2026-10-13_NEW_I04_MCC.md').write_text(report, encoding='utf-8')
    payload = {'manifest': manifest, 'results': result, 'audit': audit,
               'matched_time_diagnostics': matched, 'aggregate': aggregate}
    (evidence / '2026-10-13_NEW_I04_MCC.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    index = output / 'index.html'
    text = index.read_text(encoding='utf-8')
    card = f'''<div class="card"><b>等时长前缀诊断</b><p>每个 seed 截断到两组共同已执行时间（5.418–5.998 s），基于 10 ms 保存样本；仅作描述性诊断。</p>
    <table><tr><th>组</th><th>平均接触数</th><th>四指接触率</th><th>覆盖面积</th></tr>
    <tr><td>A-MCC</td><td>{a['mean_contacts']:.3f}</td><td>{100*a['four_contact_fraction']:.2f}%</td><td>{100*a['covered_area_fraction']:.3f}%</td></tr>
    <tr><td>B-MCC</td><td>{b['mean_contacts']:.3f}</td><td>{100*b['four_contact_fraction']:.2f}%</td><td>{100*b['covered_area_fraction']:.3f}%</td></tr></table>
    <p>B 早期接触较丰富；长期仍有失联和高力停止。末段覆盖对应不同执行时长，不作为等时长优势结论。</p>
    <p><a href="matched_time_diagnostics.json">逐 seed 等时长诊断</a> · <a href="integrity_audit.json">完整性审计</a> · <a href="episodes.csv">逐集 CSV</a></p></div>'''
    text = text.replace('</html>', card + '</html>')
    index.write_text(text, encoding='utf-8')
    print('Evidence saved:', evidence / '2026-10-13_NEW_I04_MCC.md')
    print('Common-window means:', json.dumps(common_mean))


if __name__ == '__main__':
    main()
