"""Verify, score, and render the smooth-shape shared-wrist benchmark."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
from time import sleep

import numpy as np

from .fingertip_benchmark import read
from .fingertip_hand import DT
from .fingertip_traversal import write_json
from .i01_a_fair import METHODS, locations
from .i01_a_fair_report import CONTROLLERS, concatenate_chapters, render_group, score_group
from .i01_a_smooth import DEFAULT_OUTPUT, SHAPES, STARTS, groups, verify_locked
from .i01_a_wrist import STARTUP_S


METRICS = ('two_scan', 'low_two_scan', 'low_three_scan', 'sustained_three_scan',
           'longest_light_three_s', 'over_40_s', 'coverage_10mm', 'coverage_25mm', 'fscore_5mm')


def runs(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return edges[::2], edges[1::2]


def analyze_pairs(root, rows, protocol):
    plan = read(root / rows[0]['plan'] / 'wrist_plan.json')
    steps = np.arange(rows[0]['states'])
    scan = (steps * DT >= STARTUP_S) & (steps < plan['scan_end_step'])
    count_scan = int(scan.sum())
    minimum = round(protocol['primary_minimum_bout_s'] / DT)
    classes = {}
    for row in rows:
        with np.load(root / row['trial'] / 'moving_contact.npz', allow_pickle=False) as data:
            count, load = data['head_count'], data['region_load_n']
        threshold = protocol['low_region_load_n']
        lower = scan & (load < threshold)
        two = scan & (count >= 2)
        light_three = lower & (count >= 3)
        begin, end = runs(light_three)
        lengths = end - begin
        held = lengths >= minimum
        events = np.zeros(len(scan) + 1, dtype=np.int8)
        events[begin[held]], events[end[held]] = 1, -1
        continuous = np.cumsum(events[:-1]) > 0
        classes[row['method']] = dict(two_only=lower & (count == 2), three=light_three, continuous=continuous)
        row.update(two_scan=float(two.sum() / count_scan), low_two_scan=float((two & lower).sum() / count_scan),
                   low_three_scan=float(light_three.sum() / count_scan),
                   sustained_three_scan=float(lengths[held].sum() / count_scan),
                   longest_light_three_s=float(lengths.max(initial=0) * DT),
                   sustained_bouts=int(held.sum()), scan_steps=count_scan)
    dp, mcc = classes['fixed-dp'], classes['fixed-mcc']
    eligible = mcc['two_only']
    opportunities = int(eligible.sum())
    gained = eligible & dp['three']
    gained_sustained = eligible & dp['continuous']
    return dict(
        mcc_exact_two_light_s=opportunities * DT,
        dp_three_light_when_mcc_two_s=float(gained.sum() * DT),
        dp_sustained_three_when_mcc_two_s=float(gained_sustained.sum() * DT),
        dp_three_light_when_mcc_two=float(gained.sum() / opportunities) if opportunities else None,
        dp_sustained_three_when_mcc_two=float(gained_sustained.sum() / opportunities) if opportunities else None,
        dp_only_three_light=float((dp['three'] & ~mcc['three']).sum() / count_scan),
        mcc_only_three_light=float((mcc['three'] & ~dp['three']).sum() / count_scan))


def header(path, obj, tag, regime, rows):
    from PIL import Image, ImageDraw, ImageFont

    background = Image.new('RGB', (1024, 90), '#162331')
    draw = ImageDraw.Draw(background)
    title = ImageFont.truetype('C:/Windows/Fonts/msyhbd.ttc', 22)
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 16)
    draw.text((14, 4), f"平滑曲面同腕轨对照  ·  {SHAPES[obj]['name']}  ·  {STARTS[tag]['name']}",
              fill='#eff5fa', font=title)
    draw.text((14, 35), f"共同曲面路线 {rows[0]['route_length_m']:.2f} m  ·  "
              f"仿真扫描 {rows[0]['scan_duration_s']:.1f} s", fill='#c8d5e0', font=font)
    draw.text((640, 35), '绿色为有效指尖接触（≥0.2 N）', fill='#92edbb', font=font)
    values = '    '.join(f"{CONTROLLERS[row['method']]} {row['sustained_three_scan']:.1%}" for row in rows)
    draw.text((14, 63), f'持续至少 1 s 的轻载三指占比    {values}', fill='#e1eaf2', font=font)
    background.save(path)


def chapter_path(root, obj, start):
    return root / 'chapters' / f'{obj}_{start}'


def chapter(root, obj, start):
    directory = chapter_path(root, obj, start)
    directory.mkdir(parents=True, exist_ok=True)
    print(f'Score and render {obj}/{start}', flush=True)
    rows = score_group(root, obj, start, 'shared')
    paired = analyze_pairs(root, rows, read(root / 'protocol.json'))
    video = render_group(root, obj, start, 'shared', directory, rows, make_header=header)
    write_json(directory / 'chapter.json', dict(object=obj, start=start, trials=rows, paired=paired,
                                                video=video.relative_to(root).as_posix()))
    print(f'Completed video {obj}/{start}', flush=True)


def verify_trials(root, ordered):
    verify_locked(root)
    checkpoint = read(root / 'protocol.json')['checkpoint_sha256']
    force_settings, policy_settings = set(), set()
    for obj in SHAPES:
        obj_groups = [group for group in ordered if group['object'] == obj]
        if len({group['trials'][0]['initial_sha256'] for group in obj_groups}) != len(STARTS):
            raise ValueError(f'Expected three distinct starting grasps for {obj}')
        if len({group['trials'][0]['surface_sha256'] for group in obj_groups}) != 1:
            raise ValueError(f'Test object geometry changes across grasps for {obj}')
    for group in ordered:
        for row in group['trials']:
            case = root / row['trial']
            manifest = read(case / 'manifest.json')
            controller = manifest['controller']
            expected = row['method'] != 'fixed-grasp'
            if controller['mcc_feedback_enabled'] != expected:
                raise ValueError(f'Unexpected force feedback in {case}')
            if expected:
                force_settings.add(tuple(sorted(controller['parameters'].items())))
            if row['method'] == 'fixed-dp':
                p = manifest['policy']
                if p['checkpoint_sha256'] != checkpoint:
                    raise ValueError(f'Finger Policy weights changed in {case}')
                policy_settings.add((p['denoise_steps'], p['schedule'], p['control_s'], p['waypoint_s']))
    if len(force_settings) != 1 or len(policy_settings) != 1:
        raise ValueError('Control settings vary among the 27 compared runs')


def aggregation(ordered):
    results = {}
    for obj in (*SHAPES, 'overall'):
        groups_for_object = [group for group in ordered if obj == 'overall' or group['object'] == obj]
        results[obj] = {}
        for method in METHODS:
            selected = [row for group in groups_for_object for row in group['trials'] if row['method'] == method]
            results[obj][method] = {key: float(np.mean([row[key] for row in selected])) for key in METRICS}
            results[obj][method].update(n=len(selected), max_peak_n=max(row['peak_load_n'] for row in selected))
    return results


def percent(value):
    return '—' if value is None else f'{value:.2%}'


def paired_summary(ordered):
    comparisons = {}
    for baseline in METHODS[1:]:
        differences = np.array([group['trials'][0]['sustained_three_scan'] -
                                next(row['sustained_three_scan'] for row in group['trials']
                                     if row['method'] == baseline) for group in ordered])
        comparisons[baseline] = dict(mean_difference=float(differences.mean()),
                                     higher=int((differences > 0.).sum()), equal=int((differences == 0.).sum()),
                                     lower=int((differences < 0.).sum()))
    opportunity = sum(group['paired']['mcc_exact_two_light_s'] for group in ordered)
    raw_gain = sum(group['paired']['dp_three_light_when_mcc_two_s'] for group in ordered)
    sustained_gain = sum(group['paired']['dp_sustained_three_when_mcc_two_s'] for group in ordered)
    return dict(versus=comparisons, opportunity_s=opportunity,
                dp_three_light_when_mcc_two=raw_gain / opportunity if opportunity else None,
                dp_sustained_three_when_mcc_two=sustained_gain / opportunity if opportunity else None)


def report(root, ordered, means, videos, paired):
    report_lines = [
        '# i01-a 平滑曲面同腕轨复测',
        '',
        '本轮在扁椭球、不对称卵形体和超椭球的三个新建仿真网格上各测三个起始抓姿。'
        '起始抓姿按正面、反面和侧面方向从几何求解，均以四指轻载接触开始。'
        '每对照中的三种手指控制逐帧接收相同腕部指令，使用同一初态和同一仿真步长。',
        '',
        '腕轨在 27 轮仿真前冻结。先按几何生成遍及物体各面的螺旋引导线，再用初始手型和同一三指几何目标'
        '拟合腕距及两个倾角。螺旋采样点在网格上注册后，覆盖半径不超过 25 mm。',
        '',
        '主指标按共同扫描阶段的全部步数计分。每根有效指尖的法向力至少 0.2 N，'
        '低载荷要求四指与掌部五个分区均低于 8 N。至少三指轻载的连续区间达到 1.0 s，'
        '才把这个区间计入持续三指接触率。两指率和普通低载荷三指率使用同一分母。'
        '首尾的准备及稳定阶段单独记录。',
        '',
        '## 汇总指标',
        '',
        '先算每轮的比例，再给三个起始抓姿等权。总平均给三个物体等权。',
        '',
        '| 物体 | 控制 | 至少两指 | 低载荷两指 | 低载荷三指 | 持续低载荷三指 ≥1 s | 25 mm 覆盖 | F-score@5 mm | 平均过力时间 >40 N | 最大分区峰值 |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for obj, methods in means.items():
        title = '总平均' if obj == 'overall' else SHAPES[obj]['name']
        for method, row in methods.items():
            fields = [percent(row[m]) for m in ('two_scan', 'low_two_scan', 'low_three_scan', 'sustained_three_scan',
                                               'coverage_25mm', 'fscore_5mm')]
            report_lines.append(f"| {title} | {CONTROLLERS[method]} | {' | '.join(fields)} | "
                                f"{row['over_40_s']:.2f} s | {row['max_peak_n']:.1f} N |")
    report_lines += [
        '', '## 持续三指接触的配对比较', '',
        '| 对照 | single_all 平均差（百分点） | 高于对照 | 持平 | 低于对照 |',
        '| --- | ---: | ---: | ---: | ---: |',
    ]
    for baseline, row in paired['versus'].items():
        report_lines.append(f"| {CONTROLLERS[baseline]} | {100*row['mean_difference']:+.2f} | "
                            f"{row['higher']} | {row['equal']} | {row['lower']} |")
    report_lines += [
        '', '## 逐帧配对的第三指', '',
        '比较相同扫描时刻。在纯 MCC 恰有两指轻载接触的时段，'
        '统计 single_all＋MCC 的三指轻载接触，以及落在 ≥1 s 连续三指轻载区间内的比例。',
        '',
        '| 物体 | 起始抓姿 | 纯 MCC 恰好两指的时长 | single_all 三指 | single_all 持续三指 |',
        '| --- | --- | ---: | ---: | ---: |',
    ]
    for group in ordered:
        pair = group['paired']
        report_lines.append(f"| {SHAPES[group['object']]['name']} | {STARTS[group['start']]['name']} | "
                            f"{pair['mcc_exact_two_light_s']:.2f} s | {percent(pair['dp_three_light_when_mcc_two'])} | "
                            f"{percent(pair['dp_sustained_three_when_mcc_two'])} |")
    report_lines.append(f"| 按时长汇总 | 全部九个起点 | {paired['opportunity_s']:.2f} s | "
                        f"{percent(paired['dp_three_light_when_mcc_two'])} | "
                        f"{percent(paired['dp_sustained_three_when_mcc_two'])} |")
    report_lines += ['', '## 各轮记录与重建', '',
                      '| 物体 | 起始抓姿 | 控制 | 两指 | 低载荷三指 | ≥1 s 持续三指 | 最长三指轻载 | 重建 |',
                      '| --- | --- | --- | ---: | ---: | ---: | ---: | --- |']
    for group in ordered:
        for row in group['trials']:
            mesh = (f"[PLY 网格]({row['trial']}/r_gpis.ply)"
                    if row['reconstruction_status'] == 'written' else row['reconstruction_status'])
            report_lines.append(f"| {SHAPES[group['object']]['name']} | {STARTS[group['start']]['name']} | "
                                f"{CONTROLLERS[row['method']]} | {percent(row['two_scan'])} | "
                                f"{percent(row['low_three_scan'])} | {percent(row['sustained_three_scan'])} | "
                                f"{row['longest_light_three_s']:.2f} s | {mesh} |")
    report_lines += ['', '## 同步视频', '']
    for obj, video in videos.items():
        report_lines.append(f"- [{SHAPES[obj]['name']}的三个起始抓姿]({video})")
    report_lines += [
        '', '[完整逐轮指标](metrics.json) · [冻结的协议](protocol.json) · [冻结腕轨校验](locked_wrist_plans.json)',
        '', '## 复现', '', '从实验仓库根目录运行，并使用新的输出目录。', '',
        '```powershell',
        ".\\scripts\\run_i01_a.ps1 smooth all --root Module/generated/i01_a_smooth_repeat",
        ".\\scripts\\run_i01_a.ps1 smooth-report all --root Module/generated/i01_a_smooth_repeat",
        '```', '',
    ]
    (root / 'REPORT.md').write_text('\n'.join(report_lines), encoding='utf-8')


def final_report(root):
    ordered = [read(chapter_path(root, obj, start) / 'chapter.json') for obj, start in groups()]
    verify_trials(root, ordered)
    means = aggregation(ordered)
    paired = paired_summary(ordered)
    videos = {}
    for obj in SHAPES:
        members = [group for group in ordered if group['object'] == obj]
        videos[obj] = concatenate_chapters(root, members, root / f'{obj}_comparison.mp4')
    write_json(root / 'metrics.json', dict(means=means, groups=ordered, paired=paired, videos=videos))
    report(root, ordered, means, videos, paired)
    print(root / 'REPORT.md', flush=True)


def render_worker(command):
    result = subprocess.run(command)
    # On Windows the OpenGL replay subprocess can exit before the first frame.
    if result.returncode & 0xffffffff == 0xc0000008:
        print('Retrying the recorded replay after a Windows handle error', flush=True)
        subprocess.run(command, check=True)
    else:
        result.check_returncode()


def wait_for_chapters(root, workers):
    pending = groups()
    active = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while pending or active:
            for group in tuple(active):
                if active[group].done():
                    active.pop(group).result()
            for obj, start in pending[:]:
                group = (obj, start)
                if len(active) >= workers:
                    break
                if (chapter_path(root, obj, start) / 'chapter.json').exists():
                    pending.remove(group)
                    continue
                if all((locations(root, obj, start, 'shared', method)[2] / 'stopwatch.json').is_file()
                       for method in METHODS):
                    command = [sys.executable, '-u', '-m', __spec__.name, 'chapter', '--root', str(root),
                               '--group', obj, start]
                    active[group] = pool.submit(render_worker, command)
                    pending.remove(group)
            if pending or active:
                sleep(3.)
    final_report(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('chapter', 'all', 'final'))
    parser.add_argument('--root', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--group', nargs=2)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == 'chapter':
        if tuple(args.group or ()) not in groups():
            parser.error('Select a declared object and start')
        chapter(root, *args.group)
    elif args.action == 'all':
        wait_for_chapters(root, args.workers)
    else:
        final_report(root)


if __name__ == '__main__':
    main()
