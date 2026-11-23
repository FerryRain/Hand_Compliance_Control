"""Pair native Active-02 replay frames with signed GPIS reconstructions from their touch prefixes."""

import argparse
from bisect import bisect_right
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from math import ceil
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import trimesh

from .active_timing import A_ARCHIVE, B_ARCHIVE
from .experiment import ARMS, CONTROL_DT, digest, read_json, write_json
from .full_signed import signed_gpis
from ..i04_geometric_traversal import fingertip_r_gpis as rgpis


SNAPSHOT_TIMES_S = (0, 15, 40, 75, 120, 180, 240, 300, 360, 480, 600, 720, 840, 960, 1080)
SECTION_TITLES = {
    'active-02-a': '未知形状 · 在线触觉 GPIS',
    'active-02-b': '真值几何辅助 · LEAP 配对',
}
ARM_TITLES = ('Aude + MCC', 'Aude + single_all + MCC')
COLORS = ('#6ed3bf', '#f1b77e')
SIZE = (1280, 720)
PANEL_X = (26, 650)
PANEL_WIDTH = 604
FONT = Path('C:/Windows/Fonts/msyh.ttc')
BOLD = Path('C:/Windows/Fonts/msyhbd.ttc')


def preparations(label, root, output):
    video = read_json(root / 'video.json')
    results = read_json(root / 'results.json')
    signed = read_json(root / 'signed-results.json')
    if digest(root / video['video']) != video['sha256']:
        raise ValueError(f'{label} native replay hash mismatch')
    if signed['coverage_results_sha256'] != digest(root / 'results.json'):
        raise ValueError(f'{label} reconstruction and results came from different runs')
    summaries = []
    for arm, scored in zip(ARMS, signed['runs']):
        case = root / arm
        run = next(r for r in results['runs'] if r['arm'] == arm)
        if arm != scored['arm'] or digest(case / 'trace.npz') != video['trace_hashes'][arm]:
            raise ValueError(f'{label} {arm} native trace mismatch')
        if scored['trace_sha256'] != video['trace_hashes'][arm]:
            raise ValueError(f'{label} {arm} reconstruction trace mismatch')
        last = round(run['seconds'] / CONTROL_DT)
        archived = {round(s['time_s'] / CONTROL_DT): case / s['mesh'] for s in scored['snapshots'] if s['mesh']}
        points = sorted({last, *(round(s / CONTROL_DT) for s in SNAPSHOT_TIMES_S if s < run['seconds'])})
        checkpoints = [(tick, archived.get(tick), output / 'checkpoints' / f'{label}-{arm}-{tick:06d}.png')
                       for tick in points]
        summaries.append(dict(case=case, arm=arm, last=last, checkpoints=checkpoints, mesh=case / scored['final']['mesh']))
    return dict(label=label, root=root, video=video, arms=summaries,
                maximum_load=read_json(root / 'protocol.json')['qualified_surface_region_load_n'])


@lru_cache(maxsize=2)
def measured_contacts(path, maximum_load):
    with np.load(path, allow_pickle=False) as trace:
        valid = trace['pad_valid'] & (trace['region_load_n'].max(axis=1) <= maximum_load)[:, None]
        tick, fingers = np.nonzero(valid)
        return tick, trace['points_m'][tick, fingers], trace['normals_outward'][tick, fingers]


def reconstruct_checkpoint(job):
    case, maximum_load, color, bounds, tick, archived, image_path = job
    if archived is not None:
        mesh = trimesh.load(archived, process=False, force='mesh')
    else:
        ticks, points, normals = measured_contacts(case / 'trace.npz', maximum_load)
        end = int(np.searchsorted(ticks, tick, side='right'))
        if end < 4:
            Image.new('RGBA', (604, 267)).save(image_path)
            return
        points, normals = points[:end], normals[:end]
        selected = rgpis.voxel_representatives(points)
        selected = selected[rgpis.farthest_indices(points[selected], rgpis.CONTACT_BUDGET)]
        if len(selected) < 4:
            Image.new('RGBA', (604, 267)).save(image_path)
            return
        mesh, _ = signed_gpis(points[selected], normals[selected])
    save_mesh_image(mesh, bounds, color, image_path)


def save_mesh_image(mesh, bounds, color, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb

    fig = plt.figure(figsize=(6.04, 2.67), dpi=100)
    ax = fig.add_axes((0, 0, 1, 1), projection='3d')
    faces = ax.plot_trisurf(*mesh.vertices.T, triangles=mesh.faces, linewidth=0, antialiased=True, shade=False)
    light = np.array([.37, .35, .86])
    shades = .49 + .51 * np.clip(mesh.face_normals @ light, 0., 1.)
    faces.set_facecolors(shades[:, None] * np.array(to_rgb(color)))
    ax.set(xlim=bounds[:, 0], ylim=bounds[:, 1], zlim=bounds[:, 2])
    ax.set_box_aspect(np.ptp(bounds, axis=0), zoom=1.46)
    ax.set_proj_type('ortho')
    ax.view_init(elev=23, azim=46)
    ax.set_axis_off()
    fig.savefig(path, transparent=True, dpi=100)
    plt.close(fig)


def draw_frame(scene, label, t, rate, duration, films, fonts):
    canvas = Image.new('RGB', SIZE, '#101a20')
    draw = ImageDraw.Draw(canvas)
    small, regular, heading, bold = fonts
    draw.text((26, 19), label, font=bold, fill='#eaf3ef')
    draw.text((238, 25), SECTION_TITLES[label], font=heading, fill='#c8d7d3')
    draw.text((1254, 27), f'仿真 {t:06.1f} s / {duration:.1f} s   ·   {rate}×', font=heading,
              fill='#eaf3ef', anchor='ra')
    for i, (x, color, film) in enumerate(zip(PANEL_X, COLORS, films)):
        last = film['last']
        tick = min(round(t / CONTROL_DT), last)
        snap = bisect_right(film['ticks'], tick) - 1
        mesh_tick, mesh = film['ticks'][snap], film['images'][snap]
        thumb_box = (x, 106, x + PANEL_WIDTH - 1, 332)
        draw.rounded_rectangle(thumb_box, radius=11, fill='#070b0e', outline='#31454b', width=1)
        draw.text((x, 73), ARM_TITLES[i], font=heading, fill=color)
        draw.text((x + PANEL_WIDTH, 77), '主动触摸回放', font=small, fill='#b3c4c1', anchor='ra')
        thumbnail = ImageOps.contain(scene.crop((640 * i, 44, 640 * (i + 1), 391)), (572, 223))
        canvas.paste(thumbnail, (x + (PANEL_WIDTH - thumbnail.width) // 2, 108))
        contacts = film['pad_valid'][tick].sum()
        coverage = film['coverage'][tick]
        current = '探索结束' if t > last * CONTROL_DT + CONTROL_DT else f'接触 {int(contacts)} / 4'
        draw.text((x + 4, 342), f'{current}   ·   10 mm 实触覆盖 {coverage:.1%}', font=regular, fill='#dfebe7')
        draw.line((x, 380, x + PANEL_WIDTH, 380), fill='#364a4f', width=1)
        draw.text((x + 4, 392), '累计触觉重建', font=heading, fill=color)
        draw.text((x + PANEL_WIDTH, 402), f'使用截至 {mesh_tick * CONTROL_DT:.1f} s 的触点',
                  font=small, fill='#c6d4d2', anchor='ra')
        draw.rounded_rectangle((x, 428, x + PANEL_WIDTH - 1, 676), radius=11, fill='#142329', outline='#31454b')
        canvas.paste(mesh, (x + (PANEL_WIDTH - mesh.width) // 2, 430), mesh)
    draw.rounded_rectangle((26, 697, 1254, 704), radius=4, fill='#394b50')
    if t > 0.:
        draw.rounded_rectangle((26, 697, round(26 + 1228 * t / duration), 704), radius=4, fill='#8ccdc0')
    return canvas


def compose(job, output):
    films = []
    label, root, video = job['label'], job['root'], job['video']
    for arm in job['arms']:
        with np.load(arm['case'] / 'trace.npz', allow_pickle=False) as trace, np.load(
                arm['case'] / 'coverage.npz', allow_pickle=False) as coverage:
            films.append(dict(last=arm['last'], pad_valid=trace['pad_valid'], coverage=coverage['coverage_10_mm'],
                              ticks=[tick for tick, _, _ in arm['checkpoints']],
                              images=[ImageOps.contain(Image.open(file).convert('RGBA'), (570, 243))
                                      for _, _, file in arm['checkpoints']]))
    small = ImageFont.truetype(str(FONT), 17)
    regular = ImageFont.truetype(str(FONT), 20)
    heading = ImageFont.truetype(str(BOLD), 24)
    fonts = (small, regular, heading, ImageFont.truetype(str(BOLD), 34))
    fps, speed = video['fps'], video['playback_speed']
    duration, opening = video['complete_time_span_sim_s'], video['realtime_opening_s']
    first_frames = int(round(opening * fps))
    frames = first_frames + ceil((duration - opening) * fps / speed) + 1
    finish_frames = 2 * fps
    destination = output / f'{label}-timeline.mp4'
    poster = output / f'{label}-poster.webp'
    settings = ['-crf', '21', '-maxrate', '660k', '-bufsize', '1320k', '-movflags', '+faststart']
    with imageio.get_reader(root / video['video']) as source, imageio.get_writer(
            destination, fps=fps, codec='libx264', quality=None, macro_block_size=2, ffmpeg_params=settings) as writer:
        for frame_index in range(frames):
            rate = 1 if frame_index < first_frames else speed
            t = frame_index / fps if rate == 1 else min(duration, opening + (frame_index - first_frames) * speed / fps)
            frame = draw_frame(Image.fromarray(source.get_next_data()), label, t, rate, duration, films, fonts)
            writer.append_data(np.asarray(frame))
            if frame_index == first_frames + round((300. - opening) * fps / speed):
                frame.save(poster, quality=85, method=6)
        if not poster.is_file():
            frame.save(poster, quality=85, method=6)
        for _ in range(finish_frames):
            writer.append_data(np.asarray(frame))
    return dict(video=destination.name, poster=poster.name, fps=fps, frames=frames + finish_frames,
                length_s=(frames + finish_frames) / fps, native_replay_sha256=video['sha256'],
                sha256=digest(destination), simulation_duration_s=duration, native_source_frames=frames,
                trace_hashes=video['trace_hashes'], checkpoint_times_s={
                    arm['arm']: [tick * CONTROL_DT for tick, _, _ in arm['checkpoints']] for arm in job['arms']})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--a', type=Path, default=A_ARCHIVE)
    parser.add_argument('--b', type=Path, default=B_ARCHIVE)
    parser.add_argument('--workers', type=int, choices=range(1, 9), default=6)
    args = parser.parse_args()
    output = args.output.resolve()
    (output / 'checkpoints').mkdir(parents=True, exist_ok=False)
    jobs = [preparations(label, root.resolve(), output) for label, root in
            (('active-02-a', args.a), ('active-02-b', args.b))]
    bounds = np.stack([trimesh.load(arm['mesh'], process=False, force='mesh').bounds
                       for job in jobs for arm in job['arms']])
    bounds = np.vstack((bounds[:, 0].min(axis=0) - .009, bounds[:, 1].max(axis=0) + .009))
    snapshots = [(arm['case'], job['maximum_load'], color, bounds, tick, archived, path)
                 for job in jobs for arm, color in zip(job['arms'], COLORS)
                 for tick, archived, path in arm['checkpoints']]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, _ in enumerate(pool.map(reconstruct_checkpoint, snapshots), 1):
            if index % 10 == 0 or index == len(snapshots):
                print(f'{index}/{len(snapshots)} causal reconstruction frames ready', flush=True)
    with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
        futures = [pool.submit(compose, job, output) for job in jobs]
        summaries = [task.result() for task in futures]
    write_json(output / 'site-video.json', dict(size_px=list(SIZE), clips=summaries,
                                               contact_input='each run, force-qualified measurements through checkpoint'))
    for entry in summaries:
        print(f'{entry["video"]}: {entry["frames"]} frames at {entry["fps"]} fps', flush=True)


if __name__ == '__main__':
    main()
