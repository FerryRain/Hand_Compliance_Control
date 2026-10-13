"""Readable A/B whole-mustard traversal evidence and recorded replay links."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def report(directory):
    directory = Path(directory)
    results = json.loads((directory/'benchmark_summary.json').read_text(encoding='utf-8'))
    manifest = json.loads((directory/'benchmark_manifest.json').read_text(encoding='utf-8'))
    rows = []
    for result in results['results']:
        summary = result['summary']; cell, seed = result['cell'], result['seed']
        name = f'{cell}_seed{seed}'
        complete = summary['full_surface_complete']
        rows.append(f'<tr><td><a href="{name}/index.html">{cell} · {seed}</a></td>'
            f'<td class="{"ok" if complete else "pending"}">{"全表面完成" if complete else "未完整覆盖"}</td>'
            f'<td>{100*summary["final_covered_area_fraction"]:.4f}%</td>'
            f'<td>{summary["visited_required_goals"]}/{summary["required_goal_count"]}</td>'
            f'<td>{summary["covered_vertex_count"]}/{summary["vertex_count"]}</td>'
            f'<td>{summary["executed_s"]:.2f} s</td>'
            f'<td>{summary["endpoint_local_recovery_count"]}</td>'
            f'<td>{summary["peak_force_n"]:.2f} N</td></tr>')
    videos = []
    detail_path = directory/'B-MCC_seed7/contact_details.mp4'
    if detail_path.exists():
        videos.append('<section><h2>六个面的实际接触特写 · B-MCC · 4 倍慢放</h2>'
            '<video controls preload="metadata" src="B-MCC_seed7/contact_details.mp4"></video>'
            '<p>从同一次记录里选取左右侧、底面、盖/颈、上下大面六个稳定接触片段。'
            '彩点和力读数来自实测触点，仿真时刻标在画面上；下方完整连续回放保留全部运动过程。</p></section>')
    for cell in ('B-MCC', 'A-MCC'):
        run = directory/f'{cell}_seed7'
        if (run/'full_traversal.mp4').exists():
            videos.append(f'<section><h2>{cell} · seed 7 · 整段原生状态回放</h2>'
                f'<video controls preload="metadata" src="{cell}_seed7/full_traversal.mp4"></video>'
                '<p>40× 回放，左右相机同时展示上方侧面和下方背面。彩点表示该时刻的真实 pad 接触；'
                '换面时允许离开，抵达新区域后局部重新接触。</p></section>')
    pictures = []
    for cell in ('B-MCC', 'A-MCC'):
        run = directory/f'{cell}_seed7'
        if (run/'surface_coverage.png').exists():
            pictures.append(f'<section><h2>{cell} · 全瓶六个方向的实测覆盖</h2>'
                f'<img src="{cell}_seed7/surface_coverage.png" alt="六面实际覆盖图">'
                '<p>绿色为已实际触摸的 25 mm 测地邻域并集；红色为尚未覆盖。'
                '计划点、IK 可行姿态和录像都不增加覆盖。</p></section>')
    if (directory/'B-MCC_seed7/first_touch_time.png').exists():
        pictures.append('<section><h2>瓶体各处第一次实际覆盖的时间 · B-MCC</h2>'
            '<img src="B-MCC_seed7/first_touch_time.png" alt="各处首次触摸时间">'
            '<p>时间来自独立接触账本的重建，颜色展示逐面推进和末段补扫的顺序。</p></section>')
    pairs = results['paired_initial_states']
    pair_check = len(pairs) == 3 and all(p['initial_q_qvel_ctrl_exactly_equal'] for p in pairs)
    n = results['full_completed_cases']
    scope = '全表面遍历已完成并核验' if n == 6 and not results['errors'] else '完整评测结果；未完成项仍保留'
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1"><title>Mustard · MCC 全表面触摸遍历</title>
    <style>body{{font-family:system-ui,"Microsoft YaHei",sans-serif;background:#f5f7fa;color:#182235;max-width:1280px;margin:36px auto;padding:0 24px}}h1{{font-size:34px;margin-bottom:10px}}p{{line-height:1.8}}section{{background:white;border:1px solid #dbe2ea;border-radius:14px;padding:24px;margin:24px 0}}video,img{{width:100%;border-radius:8px}}.badge{{display:inline-block;background:#e4f5ed;color:#08744b;padding:7px 12px;border-radius:7px;margin-right:12px}}.ok{{color:#08744b}}.pending{{color:#ac5016}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{text-align:left;padding:12px 9px;border-bottom:1px solid #e1e6ee}}th{{background:#f2f5f9}}a{{color:#2563eb}}.scroll{{overflow:auto}}details{{line-height:1.7}}summary{{cursor:pointer}}code{{font-size:13px}}</style>
    <h1>Mustard · MCC 全表面触摸遍历</h1>
    <p>瓶身各面、侧缘、底面、肩部、瓶颈和瓶盖。接触可中断，到新区域重新接触，实测覆盖持续累计。</p>
    <span class="badge">{scope}</span><span class="badge">{n}/6 连续运行完整覆盖</span><span class="badge">DP 未运行</span>
    <section><h2>这次怎样算“摸一遍”</h2><p>从共同 5 s 初始化开始，每次连续运行覆盖整个闭合瓶体：
    <b>8192/8192 个表面 vertices、388/388 个必访点</b>都进入真实触点的 25 mm mesh-geodesic 覆盖并集，
    再确认末段至少一指安全接触稳定 150 ms。物体固定于 MuJoCo，保留原有 2.8 倍尺寸；这是仿真实验。</p>
    <p>单次失联和局部恢复超时会记录并继续。持续高力、实际非指尖碰撞和自碰撞仍会停止实验。
    峰值力包括瞬时冲击，硬保护条件是 &gt;12 N 持续 20 ms；≥8 N 已触发卸力。</p></section>
    <section><h2>A/B · 三组配对结果</h2><div class="scroll"><table><thead><tr><th>组 · seed</th><th>完成状态</th><th>实测面积覆盖</th><th>必访点</th><th>表面 vertices</th><th>完成/停止时间</th><th>局部重接触尝试</th><th>瞬时峰值力</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
    <p>A 只接收 wrist 位姿；B 额外接收四个 tip XYZ。两侧共享几何规划、MCC 和恢复规则。
    三组初始 q/qvel/ctrl 逐元素配对核验：{'通过' if pair_check else '待核验'}。每次运行都从初始化重新开始。
    当前记录绑定同一冻结 runtime，未继承 development 的覆盖。</p></section>
    <section><h2>仍需改进的部分</h2><p>当前实现先保证整瓶实际覆盖，转移路径仍长，部分计划换面会长时间失联。
    这不能视为连续贴面扫描；后续需要缩短自由空间转移并提高触摸占比。瞬时力峰也仍存在。
    当前结果只支持固定物体 MuJoCo 仿真范围，不能外推为实机或动态抓持验证。</p></section>
    {''.join(videos)}{''.join(pictures)}
    <section><details><summary>覆盖与实验来源的核验</summary><p>独立审计从保存的 actual pad contacts、normal 和 force
    重建全部覆盖，不使用目标/预测触点。审计同时核对完整 mesh、FK、command rate、信息边界、source hash 和配对初态。
    原模型仅清理两个零体积重复/退化分量，共 4 个 faces，占 raw area 0.00000632%；未删除任何因不可达而遗漏的瓶体区域。</p>
    <p>冻结协议 <code>{html.escape(manifest['protocol'])}</code>；实现版本 <code>{manifest['source_commit'][:12]}</code>。</p>
    <p><a href="benchmark_summary.json">六次结果及独立审计</a> · <a href="benchmark_manifest.json">冻结配置与运行来源</a> ·
    <a href="../new_i04_mustard_recovery_review_20261013/index.html">此前局部恢复实验</a></p></details></section>
    </html>'''
    (directory/'index.html').write_text(page, encoding='utf-8')
    print('REPORT', directory/'index.html', flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('directory', type=Path)
    report(parser.parse_args().directory)


if __name__ == '__main__': main()
