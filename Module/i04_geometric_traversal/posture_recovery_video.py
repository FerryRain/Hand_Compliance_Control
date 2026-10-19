"""Uncut 1x measured-state playback of one local posture-recovery probe.

This is FK visualization of a completed native record, never re-simulation or
whole-bottle traversal evidence. Rendering uses a new model/data instance and
does not modify the probe, its commands, sensors, or frozen source snapshot.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from Module.fr3_leap import ARM_JOINT_NAMES, HAND_JOINT_NAMES
from Module.fr3_leap.model import compile_model_xml
from .active_sliding_video import COLORS, NAMES, _sphere


def render(directory, filename='posture_recovery_1x_uncut.mp4'):
    directory = Path(directory).resolve()
    if Path(filename).name != filename or not filename.endswith('.mp4'):
        raise ValueError('Expected a local MP4 filename')
    manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    summary = json.loads((directory/'summary.json').read_text(encoding='utf-8'))
    trace_bytes = (directory/'trace.npz').read_bytes()
    from io import BytesIO
    with np.load(BytesIO(trace_bytes)) as saved:
        trace = {key: saved[key].copy() for key in saved.files}
    times = trace['time_s']
    if len(times) < 2 or not np.allclose(np.diff(times), .002, atol=1e-10, rtol=0.):
        raise ValueError('A continuous native 2 ms record is required')
    if sha256(trace_bytes).hexdigest() != summary['trace_sha256']:
        raise ValueError('Trace differs from completed probe summary')
    selected, anchor = int(manifest['selected_finger']), int(manifest['anchor_finger'])
    source_elapsed = float(manifest['source_state']['source_trace']['relative_time_s'])
    model_bytes = (directory/'model.xml').read_bytes()
    model = compile_model_xml(model_bytes.decode('utf-8'))
    visual = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'mustard_visual')
    if visual >= 0:
        model.geom_matid[visual] = -1
        model.geom_rgba[visual] = [1., .72, .08, 1.]
    joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
              for name in (*ARM_JOINT_NAMES, *HAND_JOINT_NAMES)]
    addresses = model.jnt_qposadr[joints]
    object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'fr3_e05_object')
    object_center = model.body_pos[object_id] + np.array([0., 0., .095])
    palm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'fr3_palm_control_site')
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=540, width=720)
    camera = mujoco.MjvCamera(); camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    options = mujoco.MjvOption(); options.geomgroup[3:] = 0
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 19)
    small = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 17)
    bold = ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf', 21)
    big = ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf', 27)
    output = directory/filename
    temporary = output.with_name(output.stem+'.rendering.mp4')
    frame_times = np.r_[np.arange(float(times[0]), float(times[-1]), .05), times[-1]]
    snapshot_frames = {0, int(round(13.75/.05)), int(round(14.2/.05)), len(frame_times)-1}
    snapshots = []
    columns = slice(7+4*selected, 11+4*selected)
    initial_q = trace['q_rad'][0, columns]
    force_max = max(3., float(trace['forces_n'][:, selected].max())*1.1)
    with imageio.get_writer(temporary, fps=20, codec='libx264', quality=8,
                            macro_block_size=2, output_params=['-threads', '2']) as writer:
        for frame, time in enumerate(frame_times):
            row = min(int(np.searchsorted(times, time)), len(times)-1)
            actual_time = float(times[row])
            data.qpos[addresses] = trace['q_rad'][row]
            mujoco.mj_forward(model, data)
            panels = []
            for close in (False, True):
                camera.azimuth, camera.elevation = (-50., 24.) if close else (130., -30.)
                camera.distance = .36 if close else .66
                camera.lookat[:] = (.75*trace['tip_positions_world_m'][row, selected]
                                    + .25*data.site_xpos[palm_id] if close else
                                    .5*object_center+.5*trace['tip_positions_world_m'][row].mean(axis=0))
                renderer.update_scene(data, camera=camera, scene_option=options)
                for finger in np.flatnonzero(trace['active'][row]):
                    _sphere(renderer.scene, trace['contact_positions_world_m'][row, finger],
                            .003, COLORS[finger])
                panel = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(panel)
                draw.rectangle((0, 0, 720, 79), fill=(15, 23, 42))
                draw.text((12, 6), 'LOCAL RECOVERY PROBE | '+('FINGER CLOSE-UP' if close else 'WHOLE HAND'),
                          font=bold, fill='white')
                draw.text((12, 33), f'Source: run04 at {source_elapsed:.3f} s | uncut 1x', font=font, fill='#bce7ff')
                draw.text((12, 56), f'Probe t={actual_time:6.3f} s | {trace["phase"][row]}',
                          font=small, fill='#f6ce74')
                panels.append(panel)
            image = Image.new('RGB', (1440, 824), (15, 23, 42))
            image.paste(panels[0], (0, 0)); image.paste(panels[1], (720, 0))
            draw = ImageDraw.Draw(image)
            current_force = float(trace['forces_n'][row, selected])
            active = bool(trace['active'][row, selected])
            draw.text((12, 549), f'{NAMES[selected]} actual force: {current_force:.3f} N | '
                      + ('CONTACT' if active else 'NO CONTACT'), font=big,
                      fill='#ef64dc' if active else 'white')
            q = np.rad2deg(trace['q_rad'][row, columns])
            delta = np.rad2deg(trace['q_rad'][row, columns]-initial_q)
            draw.text((750, 551), 'Actual selected joints (deg): '+'  '.join(f'{x:.1f}' for x in q),
                      font=bold, fill='white')
            draw.text((750, 584), 'Change from start: '+'  '.join(f'{x:+.1f}' for x in delta),
                      font=bold, fill='#bce7ff')
            for finger in range(4):
                color = tuple(int(x*255) for x in COLORS[finger][:3])
                state = 'CONTACT' if trace['active'][row, finger] else 'NO CONTACT'
                draw.text((750, 619+27*finger),
                          f'{NAMES[finger]:6s}  {trace["forces_n"][row, finger]:5.2f} N  {state}',
                          font=font, fill=color)
            x0, y0, width, height = 20, 615, 675, 99
            draw.rectangle((x0, y0, x0+width, y0+height), outline='#8290a7')
            plot_rows = np.arange(0, row+1, 5)
            if row not in plot_rows:
                plot_rows = np.r_[plot_rows, row]
            curve = [(x0+width*float(times[i]/times[-1]),
                      y0+height-height*float(trace['forces_n'][i, selected]/force_max)) for i in plot_rows]
            if len(curve) > 1:
                draw.line(curve, fill='#ef64dc', width=2)
            cursor = x0+width*actual_time/times[-1]
            draw.line((cursor, y0, cursor, y0+height), fill='white', width=1)
            draw.text((x0, y0-24), f'Native selected force history | 0 to {force_max:.1f} N', font=small, fill='#bcc7d5')
            draw.text((x0, y0+103), f'0 s                                         {times[-1]:.1f} s', font=small, fill='#bcc7d5')
            elapsed = float(trace['posture_path_elapsed_s'][row])
            seek = bool(trace['finger_search_active'][row, selected])
            draw.text((12, 756), f'Path progress {elapsed:.3f}/{summary["planned_path_duration_s"]:.3f} s | '
                      f'local SEEK proposal: {"YES" if seek else "NO"} | colored dots = measured contacts',
                      font=font, fill='#bce7ff')
            draw.text((12, 791), 'Local source-state experiment; no whole-bottle coverage. Saved actual q + sensor replay; no added physics or contact.',
                      font=small, fill='#bce7ff')
            writer.append_data(np.asarray(image))
            if frame in snapshot_frames:
                final = directory/f'{output.stem}_frame_{frame:04d}.png'
                temp = final.with_name(final.stem+'.rendering.png')
                image.save(temp); snapshots.append((temp, final))
            if frame % 100 == 0:
                print(f'POSTURE_VIDEO {frame}/{len(frame_times)} t={actual_time:.3f}', flush=True)
    from OpenGL import GL
    gl_renderer = GL.glGetString(GL.GL_RENDERER).decode()
    renderer.close()
    reader = imageio.get_reader(temporary)
    try:
        if reader.count_frames() != len(frame_times) or reader.get_data(len(frame_times)-1).shape != (824, 1440, 3):
            raise RuntimeError('Video failed frame count/last-frame decode; previous output preserved')
    finally:
        reader.close()
    audits = {}
    for audit in (directory/'native_probe_audit.json', directory.parent/'paired_audit.json'):
        if audit.exists():
            raw = audit.read_bytes()
            audits[audit.name] = dict(sha256=sha256(raw).hexdigest(), result=json.loads(raw))
    meta = dict(source='Completed native 2 ms q and measured force/contact record; FK playback only',
        source_scope='LOCAL SOURCE-STATE RECOVERY PROBE; NOT WHOLE TRAVERSAL',
        source_state_run04_elapsed_s=source_elapsed, source_manifest=manifest['source_state'],
        trace_sha256=sha256(trace_bytes).hexdigest(), model_sha256=sha256(model_bytes).hexdigest(),
        video_sha256=sha256(temporary.read_bytes()).hexdigest(), start_s=float(times[0]), end_s=float(times[-1]),
        playback_speed=1., fps=20, frames=len(frame_times), encoded_duration_s=len(frame_times)/20.,
        uncut=True, selected_finger=selected, anchor_finger=anchor,
        original_native_samples=len(times), native_sample_interval_s=.002,
        sampling_note='Video samples native record at 20 fps; 2 ms contact flicker remains in the native trace.',
        path_then_search='Path endpoint is not contact success; overlay distinguishes subsequent bounded search.',
        right_camera='0.36 m; 75% actual selected tip + 25% actual palm site focus',
        full_surface_claim=False, coverage_counted=False, renderer=gl_renderer, audits=audits)
    metadata_path = output.with_name(output.stem+'_render.json')
    temp_meta = metadata_path.with_name(metadata_path.stem+'.rendering.json')
    temp_meta.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    temporary.replace(output); temp_meta.replace(metadata_path)
    for temp, final in snapshots:
        temp.replace(final)
    print('POSTURE_VIDEO_DONE', output, flush=True)
    return meta


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--filename', default='posture_recovery_1x_uncut.mp4')
    args = parser.parse_args()
    render(args.directory, filename=args.filename)
