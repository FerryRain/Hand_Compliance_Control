"""Save compact, source-bound evidence for the completed MCC evaluation."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


def evidence(directory):
    directory = Path(directory).resolve()
    result = json.loads((directory/'benchmark_summary.json').read_text(encoding='utf-8'))
    manifest = json.loads((directory/'benchmark_manifest.json').read_text(encoding='utf-8'))
    assert result['audited_cases'] == result['requested_cases'] == 6
    assert not result['errors']
    assert all(pair['initial_q_qvel_ctrl_exactly_equal'] for pair in result['paired_initial_states'])
    cells = {}
    records, table = [], []
    for case in result['results']:
        s, a = case['summary'], case['audit']
        assert a['status'] == 'PASSED' and a['continuous_from_acquisition']
        assert a['runtime_source_hashes_verified']
        record = {key: s[key] for key in ('cell', 'seed', 'stop_reason', 'full_surface_complete',
            'final_covered_area_fraction', 'visited_required_goals', 'required_goal_count',
            'covered_vertex_count', 'vertex_count', 'executed_s', 'wall_s', 'contact_continuity',
            'mean_contacts', 'peak_force_n', 'force_rmse_n', 'force_relief_count',
            'endpoint_local_recovery_count', 'oracle_geometric_refinement_count')}
        gaps = s['contact_loss_episodes']
        record['safe_contact_gap_count'] = len(gaps)
        record['stable_safe_recovered_gap_count'] = sum(bool(g.get('recovered')) for g in gaps)
        record['maximum_safe_contact_gap_s'] = max((g.get('recovery_time_s', 0.) for g in gaps), default=0.)
        record['actual_contact_samples'] = a['actual_contact_samples']
        record['accepted_contact_samples'] = a['accepted_contact_samples']
        record['accepted_contact_anchor_vertices'] = a['accepted_contact_anchor_vertices']
        record['audit'] = a
        record['directory'] = case['directory']
        records.append(record)
        table.append(f"| {s['cell']} | {s['seed']} | {'FULL' if s['full_surface_complete'] else s['stop_reason']} | "
            f"{100*s['final_covered_area_fraction']:.4f}% | {s['visited_required_goals']}/{s['required_goal_count']} | "
            f"{s['covered_vertex_count']}/{s['vertex_count']} | {s['executed_s']:.3f} | {s['peak_force_n']:.3f} |")
    for cell in manifest['cells']:
        group = [r for r in records if r['cell'] == cell]
        cells[cell] = dict(full_completed=sum(r['full_surface_complete'] for r in group), runs=len(group),
            mean_coverage=float(np.mean([r['final_covered_area_fraction'] for r in group])),
            mean_execution_s=float(np.mean([r['executed_s'] for r in group])),
            mean_contact_continuity=float(np.mean([r['contact_continuity'] for r in group])),
            mean_force_rmse_n=float(np.mean([r['force_rmse_n'] for r in group])),
            maximum_safe_contact_gap_s=max(r['maximum_safe_contact_gap_s'] for r in group),
            maximum_instantaneous_force_n=max(r['peak_force_n'] for r in group))
    first_manifest = json.loads((Path(records[0]['directory'])/'manifest.json').read_text(encoding='utf-8'))
    quality = first_manifest['geometry_quality']
    compact = dict(module='NEW_I04_MUSTARD_FULL_TRAVERSAL', protocol=manifest['protocol'],
        source_commit=manifest['source_commit'], runtime_sha256=manifest['runtime_sha256'],
        status='FULL_TRAVERSAL_VERIFIED' if result['full_completed_cases'] == 6 else 'EVALUATED_FULL_SCOPE_PARTIAL',
        dp='DEFERRED_BY_USER', source_directory=str(directory),
        object_position_m=first_manifest['object_position_m'], geometry_quality=quality,
        coverage=first_manifest['coverage'], cells=cells, runs=records,
        paired_initial_states=result['paired_initial_states'], regression_tests_passed=31,
        benchmark_summary_sha256=sha256((directory/'benchmark_summary.json').read_bytes()).hexdigest(),
        scope='Native MuJoCo fixed-object mustard simulation; existing 2.8 baked scale, 256-part collision; not hardware',
        attribution='Observed recontact is reported; this full traversal has no recovery-OFF ablation, so all gaps are not causally attributed to the endpoint layer')
    root = Path(__file__).resolve().parents[2]
    destination = root/'Module/evidence'; destination.mkdir(exist_ok=True)
    base = destination/'2026-10-14_NEW_I04_MUSTARD_FULL_TRAVERSAL'
    base.with_suffix('.json').write_text(json.dumps(compact, indent=2, ensure_ascii=False), encoding='utf-8')
    lines = f'''# 新 I04 · mustard 全物体实际触摸遍历

协议冻结于 2026-10-13，配对评测与产物于 2026-10-14 完成。

状态：`{compact['status']}`。协议 `{manifest['protocol']}`，冻结实现 `{manifest['source_commit']}`。
MCC-only，DP 依用户要求延后。完整同版配对评测：A/B × seeds 7/11/19，共 6 次。

此前 60 s 恢复实验只完成局部接触和失联恢复，不能作为整物体遍历。本次逐面执行整个闭合瓶体的
几何触摸任务，换面途中可以离开，抵达新区域后局部重新接触；不回到旧区域代替继续推进。
全部 vertices 和 required points 的分母在规划前冻结，只有实测接触的测地覆盖并集能增加覆盖。

## 同一冻结版本的连续运行结果

| 组 | seed | 完成/停止 | 实测面积覆盖 | 必访点 | 表面 vertices | treatment 时间 s | 瞬时峰值力 N |
| --- | --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(table)}

各记录都从共同 5 s acquisition 开始，未继承 development 覆盖，未在运行中设置机器人 qpos。
三组 initial q/qvel/ctrl snapshots 逐元素一致。完成还要求末段至少一指安全接触连续 150 ms、全部 pad force ≤8 N。
完成时间允许不同；配对等时长前缀覆盖另保存在 JSON，不能用不同执行时长的接触率推断方法优劣。

## 覆盖定义和固定场景

原有 mustard 尺寸 2.8 已烘焙，runtime scale=1；固定位置 `[0.30,0,0.40] m`，gravity-off，23 DoF，dt=2 ms。
物理仍使用全部 256 个原始 V-HACD parts。GT task surface 为正体积闭合主体：
8192 vertices /16380 faces /{quality['exterior_area_m2']:.12f} m²，必访点 388。
两个附加零体积重复/退化分量共 4 faces，移除面积比例 {quality['removed_area_fraction']:.12g}；
raw source 和物理碰撞 parts 保留。未因可达性或覆盖结果删去任何闭合瓶体区域。

覆盖半径 25 mm mesh-edge geodesic；actual pad force ≥0.2 N，contact-to-mesh vertex residual ≤5 mm，normal cosine ≥0.5。
这是 25 mm 扫描分辨率的整表面覆盖，不表示每一微小面片都直接承受了指尖压力。
最终覆盖、所有 vertices 和所有必访点均由独立接触账本重建，目标、IK witness 和录像不提供覆盖信用。

## 共享控制和恢复

A 仅接收 wrist SE(3)，B 另接收四个 tip Cartesian positions；无 joint witness、contact labels、role intentions。
两侧共用 bounded IK、几何路径、真实接触 MCC、接近速度和安全阈值。pad support 使用实际 ellipsoid vector，
含斜法向下的切向偏移；实际 support cache 有版本和 SHA256 检查。

到达后失联/当前区域尚未覆盖超过 150 ms，沿已暴露 wrist frame 和初始实测 normal history 做局部接近：
1.5 mm/s、最多 8 mm、7 s 局部超时后继续其他区域。已有触点而区域未覆盖记录为 geometric refinement；
无触点记录为 endpoint local recovery。全表面完成不以一次失联、接触率或时间预算为替代。

外侧 offset 不可达时检查 surface target 的安全 joint-space 路径。已有正间隙进入规划缓冲时，
仅允许间隙不减、pad penetration 不增地退出；退出后不得重入缓冲，几何数值容差 1 μm。
真实碰撞和力保护不变。≥8 N 暂停路径并卸力，低于 4 N 后继续；>12 N 连续 20 ms、
实际 non-pad/object 或 self collision、关节越界、非有限状态均独立停止。瞬时力峰仍存在，不能宣称冲击已消除。

本次未做全遍历 recovery-OFF 消融；重接触有几何运动和自然恢复成分，不能把每次恢复都归因于显式层。
历史 hold 开关消融仍独立保留在 mustard recovery v2 证据中。
完整覆盖已成立，但计划转移仍耗时：平均真实接触率 A/B 为 {100*cells['A-MCC']['mean_contact_continuity']:.3f}% /
{100*cells['B-MCC']['mean_contact_continuity']:.3f}%，最长安全接触 gap 为
{max(cells[c]['maximum_safe_contact_gap_s'] for c in cells):.3f} s。允许失联不等于转移已经足够短；不能宣称连续贴面扫描。

## 核验与产物

31 项 I04 回归检查通过，包括实际正间隙停住状态的退出回归。
六份独立 audit 核对 source snapshot/hash、完整 mesh、实测覆盖曲线/终态、201 个保存状态的 FK、
command rate、A/B 信息边界和配对初态。子进程在初始化前验证冻结 runtime SHA256。
开发续跑、v1/v1.1 中断及不同版本初步记录均保留，但不混入这 6 次冻结评测。

报告与同次数据回放：`{directory.name}/index.html`，位于 `Module/generated/`。
各 case 含 `trace.npz`、`final_state.npz`、120 s 分段检查点、`events.json`、`manifest.json`、`audit.json`。
seed 7 A/B 含完整双视角录像、六面覆盖图及各处首次实际覆盖时间图；视频仅回放保存状态，不重仿真补覆盖。
这是原生 MuJoCo CPU 固定物体扫描；NVIDIA OpenGL 用于渲染，未做实机、物体运动或真实抓持验证。

复现入口与数字配置：`../I04_MUSTARD_FULL_TRAVERSAL_PROTOCOL.md`。
'''
    base.with_suffix('.md').write_text(lines, encoding='utf-8')
    print('EVIDENCE', base, compact['status'], json.dumps(cells), flush=True)
    return compact


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('directory', type=Path)
    evidence(parser.parse_args().directory)


if __name__ == '__main__': main()
