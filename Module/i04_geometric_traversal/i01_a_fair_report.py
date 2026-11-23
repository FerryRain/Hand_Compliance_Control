"""Score, verify, and render the three paired i01-a experiments."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import sleep

import numpy as np

from .fingertip_benchmark import digest, read
from .fingertip_hand import DT
from .fingertip_replay import render_case
from .fingertip_traversal import write_json
from .i01_a import STARTUP_S
from .i01_a_fair import DEFAULT_OUTPUT, METHODS, OBJECTS, REGIMES, STARTS, locations
from .i01_a_report import score


NAMES = {'meat-can': '午餐肉罐', 'ellipsoid': '椭球'}
CONTROLLERS = {'fixed-dp': 'single_all＋MCC', 'fixed-mcc': '纯 MCC', 'fixed-grasp': '固定手型'}
TAGS = {'start-01': '起始抓姿 A', 'start-02': '起始抓姿 B'}
WRISTS = {'shared': '共同腕轨', 'optimized': '各自优化腕轨'}
CHAPTER_FRAMES = 320
FPS = 20


def groups():
    return [(obj, tag, regime) for tag, _ in STARTS for regime in REGIMES for obj in OBJECTS]


def group_id(obj, tag, regime):
    return f'{obj}_{tag}_{regime}'


def metric_row(root, plan_path, case, record):
    plan = read(plan_path / 'wrist_plan.json')
    actual_plan = read(case / 'wrist_plan.json')
    if actual_plan['nominal_wrist_sha256'] != plan['nominal_wrist_sha256']:
        raise ValueError(f'Trial received another planned trajectory in {case}')
    if record['stop_reason'] != 'WRIST_TRAJECTORY_COMPLETE' or record['states'] != plan['nominal_wrist_steps']:
        raise ValueError(f'Incomplete wrist execution in {case}')
    if record['wrist_sha256'] != plan['nominal_wrist_sha256']:
        raise ValueError(f'Executed wrist reference differs from the locked plan in {case}')
    with np.load(case / 'moving_contact.npz', allow_pickle=False) as data:
        head_count, load = data['head_count'], data['region_load_n']
        progress = data['progress_m']
        delta = np.diff(progress, prepend=0.)
        if np.any(delta < 0.) or sha256(np.ascontiguousarray(progress, dtype='<f8').tobytes()).hexdigest() != plan['common_progress_sha256']:
            raise ValueError(f'Unmatched surface distance schedule in {case}')
        scan_steps = np.arange(len(head_count))
        scanning = (scan_steps * DT >= STARTUP_S) & (scan_steps < plan['scan_end_step'])
        over_40 = float(np.count_nonzero(scanning & (load > 40.)) * DT)
        two_distance = float(delta[head_count >= 2].sum() / progress[-1])
        low_three_distance = float(delta[(head_count >= 3) & (load < 8.)].sum() / progress[-1])
    return dict(
        method=record['branch'], route_length_m=plan['path_length_m'], scan_duration_s=plan['scan_time_s'],
        states=record['states'], seed=record['seed'], total_simulated_s=record['executed_s'],
        two=record['motion']['moving_at_least_two_heads'], three=record['motion']['moving_at_least_three_heads'],
        low_two=record['motion']['moving_two_heads_below_8n'],
        low_three=record['motion']['moving_three_heads_below_8n'],
        two_by_distance=two_distance, low_three_by_distance=low_three_distance,
        moving_s=record['motion']['moving_s'],
        longest_no_head_s=record['motion']['moving_no_head_longest_s'], over_40_s=over_40,
        peak_load_n=record['peak_load_n'], longest_continuous_over_40_s=record['longest_over_40n_s'],
        coverage_10mm=record['quality']['coverage_10mm'], coverage_25mm=record['quality']['coverage_25mm'],
        coverage_25mm_below_8n=record['quality']['coverage_25mm_below_8n'],
        fscore_5mm=record['quality']['fscore_5mm'], reconstruction_status=record['reconstruction_status'],
        geometry_planning_s=plan['geometry_planning_s'],
        trial=str(case.relative_to(root).as_posix()), plan=str(plan_path.relative_to(root).as_posix()),
        initial_sha256=digest(case / 'initial.npz'), wrist_sha256=record['wrist_sha256'],
        progress_sha256=plan['common_progress_sha256'], model_sha256=plan['model_sha256'],
        surface_sha256=plan['surface_sha256'], calibration_sha256=plan['calibration_sha256'])


def score_group(root, obj, tag, regime):
    rows = []
    _, shared, _ = locations(root, obj, tag, 'shared', METHODS[0])
    shared_rows = {}
    if regime == 'shared':
        result, _, _ = score(shared)
        if len(result['fixed_triplets']) != 1:
            raise ValueError(f'Expected one matched trial triplet for {obj}/{tag}')
        shared_rows = {record['branch']: record for record in result['trials']}
    for method in METHODS:
        plan, run, case = locations(root, obj, tag, regime, method)
        record = shared_rows[method] if regime == 'shared' else score(run)[0]['trials'][0]
        row = metric_row(root, plan, case, record)
        rows.append(row)
    for field in ('initial_sha256', 'progress_sha256', 'states', 'seed', 'model_sha256', 'surface_sha256', 'calibration_sha256'):
        if len({row[field] for row in rows}) != 1:
            raise ValueError(f'Unmatched {field} for {obj}/{tag}/{regime}')
    if regime == 'shared' and len({row['wrist_sha256'] for row in rows}) != 1:
        raise ValueError(f'Unmatched shared wrist commands for {obj}/{tag}')
    return rows


def experiment(tag, regime):
    if tag == 'start-02':
        return 'E3'
    return 'E1' if regime == 'shared' else 'E2'


def header(path, obj, tag, regime, rows):
    from PIL import Image, ImageDraw, ImageFont

    background = Image.new('RGB', (1024, 66), '#162331')
    draw = ImageDraw.Draw(background)
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 17)
    bold = ImageFont.truetype('C:/Windows/Fonts/msyhbd.ttc', 22)
    draw.text((14, 2), f'{experiment(tag, regime)}  {WRISTS[regime]}  ·  {NAMES[obj]}  ·  {TAGS[tag]}',
              fill='#f1f5f9', font=bold)
    draw.text((14, 36),
              f"三组同步  ·  路径 {rows[0]['route_length_m']:.2f} m  ·  实际仿真扫描 {rows[0]['scan_duration_s']:.1f} s",
              fill='#c8d5e0', font=font)
    draw.text((640, 36), '绿色为有效指尖接触（≥0.2 N）', fill='#92edbb', font=font)
    background.save(path)


def render_group(root, obj, tag, regime, directory, rows, *, make_header=header):
    from imageio_ffmpeg import count_frames_and_secs, get_ffmpeg_exe

    output = directory / 'comparison.mp4'
    if output.is_file() and count_frames_and_secs(str(output))[0] == CHAPTER_FRAMES:
        return output
    # Every branch has the same clock, so sampling frame times also pairs the guide progress.
    speed = rows[0]['total_simulated_s'] * FPS / (CHAPTER_FRAMES - .25)
    with TemporaryDirectory(dir=directory) as temp:
        temp = Path(temp)
        videos = []
        for row in rows:
            case = root / row['trial']
            piece = temp / (row['method'] + '.mp4')
            media = render_case(case, piece, speed, surface_evidence=case / 'measured_area.npz',
                                recorded=True, show_moving_contact=True, show_low_load=True)
            if media['frames'] != CHAPTER_FRAMES or media['native_states'] != row['states']:
                raise ValueError(f'Incorrect replay length in {case}')
            videos.append(piece)
        frame = temp / 'header.png'
        make_header(frame, obj, tag, regime, rows)
        pending = temp / output.name
        concatenate = '[0:v]setsar=1[h];' + ''.join(
            f'[{i}:v]scale=1024:-2,setsar=1[v{i}];' for i in range(1, 4))
        concatenate += '[h][v1][v2][v3]vstack=inputs=4:shortest=1'
        inputs = [part for path in videos for part in ('-i', str(path))]
        subprocess.run([get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error',
                        '-loop', '1', '-framerate', str(FPS), '-i', str(frame), *inputs,
                        '-filter_complex', concatenate, '-frames:v', str(CHAPTER_FRAMES),
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', '-preset', 'veryfast',
                        '-movflags', '+faststart', str(pending)], check=True)
        if count_frames_and_secs(str(pending))[0] != CHAPTER_FRAMES:
            raise ValueError(f'Invalid comparison video length for {obj}/{tag}/{regime}')
        pending.replace(output)
    return output


def chapter(root, obj, tag, regime):
    directory = root / 'chapters' / group_id(obj, tag, regime)
    directory.mkdir(parents=True, exist_ok=True)
    print(f'Score and render {obj}/{tag}/{regime}', flush=True)
    rows = score_group(root, obj, tag, regime)
    output = render_group(root, obj, tag, regime, directory, rows)
    write_json(directory / 'chapter.json', dict(experiment=experiment(tag, regime), object=obj, start=tag,
               regime=regime, trials=rows, video=output.relative_to(root).as_posix()))
    print(f'Video complete for {obj}/{tag}/{regime}', flush=True)


def verify_pairs(root, ordered):
    paired = {(g['object'], g['start'], g['regime']): g['trials'] for g in ordered}
    for obj in OBJECTS:
        initials = set()
        for tag, _ in STARTS:
            rows = paired[obj, tag, 'shared'] + paired[obj, tag, 'optimized']
            for field in ('initial_sha256', 'progress_sha256', 'states', 'seed', 'model_sha256'):
                if len({row[field] for row in rows}) != 1:
                    raise ValueError(f'Unmatched {field} across regimes for {obj}/{tag}')
            initials.add(rows[0]['initial_sha256'])
        if len(initials) != len(STARTS):
            raise ValueError(f'Repeat did not use a different starting grasp for {obj}')
    parameters, policy_settings = set(), set()
    policy_fields = ('checkpoint_sha256', 'denoise_steps', 'schedule', 'control_s', 'action_points', 'waypoint_s')
    for group in ordered:
        for row in group['trials']:
            manifest = read(root / row['trial'] / 'manifest.json')
            plan = manifest['planning']
            if plan['target_supports'] != 3 or plan['startup_supports'] != 3:
                raise ValueError(f"The three-tip objective differs in {row['trial']}")
            controller = manifest['controller']
            feedback = row['method'] != 'fixed-grasp'
            if controller['mcc_feedback_enabled'] != feedback:
                raise ValueError(f"Unexpected force feedback setting in {row['trial']}")
            if feedback:
                parameters.add(tuple(sorted(controller['parameters'].items())))
            if row['method'] == 'fixed-dp':
                policy_settings.add(tuple(manifest['policy'][k] for k in policy_fields))
    if len(parameters) != 1 or len(policy_settings) != 1:
        raise ValueError('MCC tuning or Finger Policy settings vary among the matched trials')


AVERAGES = ('two', 'three', 'low_two', 'low_three', 'two_by_distance', 'low_three_by_distance',
            'coverage_10mm', 'coverage_25mm', 'coverage_25mm_below_8n', 'over_40_s',
            'longest_no_head_s', 'fscore_5mm')


def report_groups(ordered):
    groupings = (
        ('E1 同腕轨', 'start-01', 'shared'),
        ('E2 各自优化', 'start-01', 'optimized'),
        ('E3 同腕轨复测', 'start-02', 'shared'),
        ('E3 各自优化复测', 'start-02', 'optimized'),
    )
    averages = {}
    for title, tag, regime in groupings:
        samples = [g['trials'] for g in ordered if g['start'] == tag and g['regime'] == regime]
        averages[title] = {}
        for method in METHODS:
            rows = [row for group in samples for row in group if row['method'] == method]
            stat = {key: float(np.mean([row[key] for row in rows])) for key in AVERAGES}
            stat.update(max_peak_n=max(row['peak_load_n'] for row in rows), trials=len(rows))
            averages[title][method] = stat
    return averages


def concatenate_chapters(root, members, output):
    from imageio_ffmpeg import count_frames_and_secs, get_ffmpeg_exe

    with TemporaryDirectory(dir=root) as temp:
        manifest = Path(temp) / 'clips.txt'
        manifest.write_text(''.join(f"file '../{g['video']}'\n" for g in members), encoding='utf-8')
        video = Path(temp) / output.name
        subprocess.run([get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error', '-f', 'concat', '-safe', '0',
                        '-i', str(manifest), '-c', 'copy', '-movflags', '+faststart', str(video)], check=True)
        if count_frames_and_secs(str(video))[0] != len(members) * CHAPTER_FRAMES:
            raise ValueError(f'Invalid chapter concatenation for {output.name}')
        video.replace(output)
    return output.name


def concatenate_videos(root, ordered):
    videos = {}
    for name in ('E1', 'E2', 'E3'):
        members = [g for g in ordered if g['experiment'] == name]
        videos[name] = concatenate_chapters(root, members, root / f'{name}_comparison.mp4')
    return videos


def report(root, ordered, averages, videos):
    pct = lambda x: f'{x:.2%}'
    body = [
        '# i01-a 三项接触对照',
        '',
        '对照采用午餐肉罐的放大凸碰撞模型和一件新建的椭球。每件物体有两个独立几何求解的起始抓姿，'
        '全部从四指轻载接触开始。E1 给三种控制相同腕轨。E2 给各控制按工作手型拟合的腕轨。'
        'E3 从另一面重新起步，重复这两种腕轨条件。三个实验分别保存 6、6 和 12 轮记录。',
        '',
        '午餐肉罐沿用此前 YCB 实验的 2.8 倍比例。椭球半轴为 105、137 和 93 mm。'
        '表面路线经过 25 mm 测地间距的覆盖点，实际触点的覆盖率从仿真接触记录计算。',
        '',
        '全部腕轨以三指尖靠近表面为几何目标，各节点使用相同迭代上限及腕距和倾角约束。'
        '各自优化的工作手型由芥末瓶和饼干盒上的旧配对记录得到，每种控制、每个开发物体取 256 个合格帧。'
        '将实测手型与起始手型的中位偏移乘以 0.75，再叠加到新场景的起始手型。'
        '一个起始抓姿内的六轮共享表面引导线，并使用四条腕轨逐点可共同满足的推进速度。',
        '',
        '接触以指尖头部法向力至少 0.2 N 为准。低载荷要求四指与掌部五个分区均低于 8 N。'
        '时间接触率在手腕实际运动时统计。过力时间是扫描中任一分区载荷超过 40 N 的累计时间。'
        'R-GPIS 用实测指尖接触点及法向重建，F-score 对比完整参考表面。各百分比先按整轮计算，再对两个物体等权平均。',
        '',
        '| 实验 | 控制 | 两指 | 低载荷两指 | 低载荷三指 | 25 mm 覆盖 | F-score@5 mm | 平均过力时间 | 最大分区峰值 |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for title, methods in averages.items():
        for method, row in methods.items():
            body.append(f"| {title} | {CONTROLLERS[method]} | {pct(row['two'])} | {pct(row['low_two'])} | "
                        f"{pct(row['low_three'])} | {pct(row['coverage_25mm'])} | {pct(row['fscore_5mm'])} | "
                        f"{row['over_40_s']:.2f} s | {row['max_peak_n']:.1f} N |")
    body += ['', '## 配对差值', '',
             '差值为 single_all＋MCC 减去对照，单位为百分点。每个条件包含两个物体，两件物体上的观测为本次确定的对照样本。',
             '', '| 实验 | 对照 | 两指差值均值 | 低载荷三指差值均值 |', '| --- | --- | ---: | ---: |']
    for title, methods in averages.items():
        for baseline in METHODS[1:]:
            dp, control = methods[METHODS[0]], methods[baseline]
            body.append(f"| {title} | {CONTROLLERS[baseline]} | {100*(dp['two'] - control['two']):+.2f} | "
                        f"{100*(dp['low_three'] - control['low_three']):+.2f} |")
    body += ['', '## 视频', '']
    for name, video in videos.items():
        body.append(f'- [{name} 三组同步视频]({video})。')
    body += ['', '## 逐轮结果与重建', '',
             '| 物体 | 起点 | 腕轨 | 控制 | 两指 | 低载荷三指 | 25 mm 覆盖 | 分区峰值 | 重建 |',
             '| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |']
    for group in ordered:
        for row in group['trials']:
            mesh = ('[PLY 网格](' + row['trial'] + '/r_gpis.ply)'
                    if row['reconstruction_status'] == 'written' else row['reconstruction_status'])
            body.append(f"| {NAMES[group['object']]} | {TAGS[group['start']]} | {WRISTS[group['regime']]} | "
                        f"{CONTROLLERS[row['method']]} | {pct(row['two'])} | {pct(row['low_three'])} | "
                        f"{pct(row['coverage_25mm'])} | {row['peak_load_n']:.1f} N | {mesh} |")
    body += ['', '[完整逐轮指标](metrics.json) · [冻结的试验参数](protocol.json)', '',
             '## 复现', '', '从实验仓库根目录运行，使用一个新的输出目录。', '',
             '```powershell',
             ".\\scripts\\run_i01_a.ps1 fair all --root Module/generated/i01_a_fair_repeat",
             ".\\scripts\\run_i01_a.ps1 fair-report all --root Module/generated/i01_a_fair_repeat",
             '```', '']
    (root / 'REPORT.md').write_text('\n'.join(body), encoding='utf-8')


def final_report(root):
    ordered = [read(root / 'chapters' / group_id(*group) / 'chapter.json') for group in groups()]
    verify_pairs(root, ordered)
    averages = report_groups(ordered)
    videos = concatenate_videos(root, ordered)
    write_json(root / 'metrics.json', dict(group_means=averages, groups=ordered, videos=videos))
    report(root, ordered, averages, videos)
    print(root / 'REPORT.md', flush=True)


def wait_for_chapters(root, workers):
    waiting = list(groups())
    active = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while waiting or active:
            for job in tuple(active):
                if active[job].done():
                    active.pop(job).result()
            for job in waiting[:]:
                if len(active) >= workers:
                    break
                if (root / 'chapters' / group_id(*job) / 'chapter.json').exists():
                    waiting.remove(job)
                    continue
                if all((locations(root, *job, method)[2] / 'stopwatch.json').is_file() for method in METHODS):
                    command = [sys.executable, '-u', '-m', __spec__.name, 'chapter', '--root', str(root),
                               '--group', *job]
                    active[job] = pool.submit(subprocess.run, command, check=True)
                    waiting.remove(job)
            if waiting or active:
                sleep(3.)
    final_report(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('chapter', 'all', 'final'))
    parser.add_argument('--root', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--group', nargs=3)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == 'chapter':
        if tuple(args.group or ()) not in groups():
            parser.error('Select a declared object, start, and wrist condition')
        chapter(root, *args.group)
    elif args.action == 'all':
        wait_for_chapters(root, args.workers)
    else:
        final_report(root)


if __name__ == '__main__':
    main()
