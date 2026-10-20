"""Stream complete B videos from immutable native trace chunks.

The simulation and audit can run concurrently. Rendering uses saved q only,
never advances physics, and makes no audit claim before the final report.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import time

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .published_whole_hand import install_asset_adapter
from .sliding_audit import _native_context
from .pad_contact_visual import camera_from_direction, marker


PHASES = {
    'COOPERATIVE_APPROACH': '四指主动接近',
    'COOPERATIVE_SLIDE': '整手连续贴面滑动',
    'FORCE_RELEASE': '主动卸力',
    'COOPERATIVE_CONTACT_PIVOT': '保持接触调整方位',
    'COOPERATIVE_WITHDRAWAL': '必要撤离：四指协调让位',
    'NECESSARY_REORIENTATION': '必要离面调姿',
    'COOPERATIVE_RECONTACT': '整手主动重接触',
    'CLEARANCE_EXIT': '调整姿态保持间隙',
    'FINAL_SAFE_HOLD': '覆盖完成后的实际接触保持',
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    h = sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def render(case, output_name='complete_videos', follow=False):
    install_asset_adapter()
    case = Path(case).resolve()
    if Path(output_name).name != output_name or output_name in ('.', '..'):
        raise ValueError('Use a new single-directory output name')
    out = case / output_name
    out.mkdir(exist_ok=False)
    manifest_hash = digest(case / 'manifest.json')
    context = _native_context(case)
    model, data = context.model, context.data
    pads = context.handles.tip_geom_ids
    options = mujoco.MjvOption()
    options.geomgroup[3:] = 0
    options.sitegroup[:] = 0
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    renderer = mujoco.Renderer(model, height=540, width=800)
    model.vis.headlight.ambient[:] = [.25, .25, .25]
    model.vis.headlight.diffuse[:] = [.45, .45, .45]
    rgba, matid = model.geom_rgba.copy(), model.geom_matid.copy()
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ''
             for g in range(model.ngeom)]
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 20)
    small = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 18)
    title = ImageFont.truetype('C:/Windows/Fonts/msyhbd.ttc', 24)
    speeds = (1, 12)
    writers = {speed: imageio.get_writer(
        out / f'b_complete_{speed}x.mp4', fps=20, codec='libx264',
        quality=7, macro_block_size=2, output_params=['-threads', '1'])
        for speed in speeds}
    frames = dict.fromkeys(speeds, 0)
    first_time, last_time, last_row = {}, {}, {}

    def draw_row(row, global_index, selected_speeds):
        data.qpos[context.qpos] = row['q_rad']
        mujoco.mj_forward(model, data)
        centers = data.geom_xpos[pads].copy()
        rotations = data.geom_xmat[pads].reshape(4, 3, 3)
        normal = rotations[:3, :, 2].mean(axis=0)
        normal /= np.linalg.norm(normal)
        canvas = Image.new('RGB', (2400, 700), '#122238')
        for panel in range(3):
            model.geom_rgba[:] = rgba
            model.geom_matid[:] = matid
            for geom, name in enumerate(names):
                if 'normal_marker' in name:
                    model.geom_rgba[geom, 3] = 0.
                if name == 'mustard_visual':
                    model.geom_matid[geom] = -1
                    model.geom_rgba[geom] = [.96, .72, .08, .12 if panel else 1.]
            if panel == 2:
                camera_from_direction(camera, rotations[3, :, 2] + .2 * rotations[3, :, 1],
                                      centers[3], .27)
            elif panel:
                camera_from_direction(camera, normal + .65 * rotations[1, :, 1],
                                      centers.mean(axis=0), .44)
            else:
                camera.lookat[:] = [.3, 0., .49]
                camera.distance = 1.05
                camera.azimuth = 130.
                camera.elevation = -25.
            renderer.update_scene(data, camera=camera, scene_option=options)
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = not bool(panel)
            for finger in np.flatnonzero(row['safe_belly']):
                marker(renderer.scene, row['belly_contact_positions_world_m'][finger],
                       .002, [1., .1, .55, 1.])
            picture = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(picture)
            draw.rectangle((0, 0, 800, 70), fill='#122238')
            heading = ('I04B · Mustard 完整原生轨迹', '三根长指 · 仅物体透明',
                       '拇指 · 实际关节状态')[panel]
            draw.text((12, 7), heading, font=title, fill='white')
            draw.text((12, 42),
                      f't={float(row["time_s"]):.2f}s · 合格指腹 {int(row["safe_belly"].sum())}/4',
                      font=font, fill='#a7dbf0')
            canvas.paste(picture, (800 * panel, 0))
        draw = ImageDraw.Draw(canvas)
        for i, name in enumerate(('食指', '中指', '无名指', '拇指')):
            q = np.degrees(row['q_rad'][7 + 4 * i:11 + 4 * i])
            text = (f'{name} 关节° [{q[0]:5.1f}, {q[1]:5.1f}, {q[2]:5.1f}, {q[3]:5.1f}]'
                    f'  {row["forces_n"][i]:.2f} N')
            text += '  合格指腹接触' if row['safe_belly'][i] else '  当前无合格指腹接触'
            draw.text((15 + 1190 * (i % 2), 582 + 32 * (i // 2)), text, font=small,
                      fill='#a7edba' if row['safe_belly'][i] else '#d7e1ea')
        draw.text((15, 660),
                  '包含全部离面、卸力、恢复；粉点为实际合格接触。原始仿真力保留，最终独立核验见报告。',
                  font=small, fill='#f3d38d')
        phase = PHASES.get(str(row['phase']), str(row['phase']))
        for speed in selected_speeds:
            draw.rectangle((0, 540, 2400, 578), fill='#122238')
            draw.text((15, 548),
                      f'{phase} | {speed}×完整仿真时间轴 | 实测测地覆盖 {100 * float(row["coverage"]):.2f}%',
                      font=font, fill='white')
            writers[speed].append_data(np.asarray(canvas))
            frames[speed] += 1
            first_time.setdefault(speed, float(row['time_s']))
            last_time[speed] = float(row['time_s'])
            last_row[speed] = global_index
        return canvas

    total = 0
    consumed = []
    last_observation = None
    started = time.perf_counter()
    keys = ('time_s', 'q_rad', 'forces_n', 'safe_belly',
            'belly_contact_positions_world_m', 'phase', 'coverage')
    try:
        while True:
            try:
                finished = (case / 'run_complete.json').exists() or (
                    not follow and (case / 'summary.json').exists())
                catalog = read(case / ('summary.json' if finished else 'trace_chunks.json'))
                if finished:
                    catalog = catalog['trace_chunks']
            except (FileNotFoundError, json.JSONDecodeError):
                if not follow:
                    raise
                time.sleep(1.)
                continue
            for chunk in catalog[len(consumed):]:
                path = case / chunk['file']
                assert digest(path) == chunk['sha256']
                with np.load(path) as archive:
                    values = {key: archive[key] for key in keys}
                count = len(values['time_s']) - int(chunk.get('skip_last', 0))
                for k in range((-total) % 25, count, 25):
                    index = total + k
                    row = {key: value[k] for key, value in values.items()}
                    selected = [speed for speed in speeds if index % (25 * speed) == 0]
                    canvas = draw_row(row, index, selected)
                    if index == 0:
                        canvas.save(out / 'first_frame.png')
                last_observation = {key: value[count - 1] for key, value in values.items()}
                total += count
                consumed.append(chunk)
                canvas.save(out / 'latest_frame.png')
                progress = dict(native_states_read=total, through_s=float(last_observation['time_s']),
                                frames=frames, wall_s=time.perf_counter() - started,
                                status='RENDERING_NATIVE_TRAJECTORY_AUDIT_PENDING')
                (out / 'progress.json').write_text(json.dumps(progress, indent=2))
                print('VIDEO_CHUNK', json.dumps(progress), flush=True)
            if finished:
                break
            if not follow:
                raise ValueError('Run is still active; use --follow')
            time.sleep(1.)
        assert last_observation is not None
        selected = [speed for speed in speeds if last_row[speed] != total - 1]
        if selected:
            canvas = draw_row(last_observation, total - 1, selected)
        canvas.save(out / 'final_frame.png')
    finally:
        for writer in writers.values():
            writer.close()
        renderer.close()
    assert digest(case / 'manifest.json') == manifest_hash
    summary = read(case / 'summary.json')
    assert consumed == summary['trace_chunks']
    videos = []
    for speed in speeds:
        path = out / f'b_complete_{speed}x.mp4'
        videos.append(dict(file=path.name, speed=speed, fps=20, frames=frames[speed],
                           duration_s=frames[speed] / 20.,
                           time_range_s=[first_time[speed], last_time[speed]],
                           sha256=digest(path), bytes=path.stat().st_size,
                           full_time_axis=True,
                           native_sampling_stride=25 * speed, final_native_observation_included=True))
    result = dict(status='RENDERED_NATIVE_TRAJECTORY_FINAL_AUDIT_RECORDED_SEPARATELY',
                  videos=videos, native_observations=total, trace_chunks=consumed,
                  manifest_sha256=manifest_hash, summary_sha256=digest(case / 'summary.json'),
                  actual_saved_q_only=True, no_physics_step=True,
                  wall_s=time.perf_counter() - started)
    (out / 'video_manifest.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    (out / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--output-name', default='complete_videos')
    parser.add_argument('--follow', action='store_true')
    args = parser.parse_args()
    render(args.case, args.output_name, args.follow)
